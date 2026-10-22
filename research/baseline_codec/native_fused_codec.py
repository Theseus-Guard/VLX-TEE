# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hakan Damar and Theseus Guard, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Developed by Theseus Guard Research (https://www.theseusguard.com)
# Repository: https://github.com/Theseus-Guard/ZTCP
# Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
"""Native C++17 Single-Pass Fused Cold-Page Codec Wrapper.

Developed by Theseus Guard, Inc. Compiles and loads `csrc/fused_cold_page_cpu.cpp`
on-the-fly via `torch.utils.cpp_extension` with `-O3 -march=armv8.5-a+fp16` (on Apple
Silicon) or `-O3 -march=native` so all quantization, bit-packing, RFC 8439 ChaCha20 encryption,
and 128-bit SipHash-2-4 authentication run in a single native C++ pass with zero intermediate
tensor allocations.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from .cold_page_spec import HEADER_BYTES, ColdPageAAD, ColdPageGeometry, hash32
from .fused_quant_crypto import (
    CrossTenantContaminationError,
    EphemeralSessionKeyRing,
    KVIntegrityError,
    NonceReuseError,
    SessionShreddedError,
    StalePageReplayError,
)

_NATIVE_MOD: Any = None


def get_native_fused_module() -> Any:
    global _NATIVE_MOD
    if _NATIVE_MOD is not None:
        return _NATIVE_MOD

    csrc_path = Path(__file__).resolve().parent / "csrc" / "fused_cold_page_cpu.cpp"
    build_dir = Path(__file__).resolve().parent / "csrc" / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the running Python venv's bin directory (containing ninja) is on PATH
    venv_bin = str(Path(sys.executable).parent)
    current_path = os.environ.get("PATH", "")
    if venv_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    _NATIVE_MOD = load(
        name="ztcp_fused_cold_page_cpp",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17", "-march=armv8.5-a+fp16", "-ffast-math"],
        verbose=False,
    )
    return _NATIVE_MOD


class NativeCppFusedColdPageCodec:
    """High-speed C++17 Single-Pass Fused Quantizer + ARX Stream Cipher + 128-bit SipMAC."""

    def __init__(
        self,
        geometry: ColdPageGeometry,
        keyring: EphemeralSessionKeyRing,
    ) -> None:
        self.geometry = geometry
        self.keyring = keyring
        self._ext = get_native_fused_module()

    def _validate_single_shapes(
        self, k_page_cpu: torch.Tensor, v_page_cpu: torch.Tensor | None
    ) -> None:
        exp = (
            self.geometry.num_tokens,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
        )
        if tuple(k_page_cpu.shape) != exp:
            raise RuntimeError(
                f"k_page_cpu shape {tuple(k_page_cpu.shape)} does not match geometry {exp}"
            )
        if not self.geometry.is_mla_single_buffer:
            if v_page_cpu is None or tuple(v_page_cpu.shape) != exp:
                raise RuntimeError(
                    f"v_page_cpu shape {None if v_page_cpu is None else tuple(v_page_cpu.shape)} does not match geometry {exp}"
                )

    @staticmethod
    def _validate_token_hashes_tensor(
        page_token_hashes_i64: torch.Tensor,
    ) -> list[int]:
        if not isinstance(page_token_hashes_i64, torch.Tensor):
            raise RuntimeError("page_token_hashes_i64 must be a torch.Tensor")
        if (
            not page_token_hashes_i64.device.type == "cpu"
            or not page_token_hashes_i64.is_contiguous()
            or page_token_hashes_i64.dtype != torch.int64
            or page_token_hashes_i64.dim() != 1
            or page_token_hashes_i64.numel() < 1
        ):
            raise RuntimeError(
                f"page_token_hashes_i64 must be a non-empty 1D contiguous CPU int64 tensor, got shape={tuple(page_token_hashes_i64.shape)} dtype={page_token_hashes_i64.dtype} device={page_token_hashes_i64.device}"
            )
        return page_token_hashes_i64.tolist()

    @staticmethod
    def _tensors_overlap(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return False
        a_start = a.data_ptr()
        a_end = a_start + a.numel() * a.element_size()
        b_start = b.data_ptr()
        b_end = b_start + b.numel() * b.element_size()
        return (a_start < b_end) and (b_start < a_end)

    def _validate_batch_shapes(
        self,
        k_pages_cpu: torch.Tensor,
        v_pages_cpu: torch.Tensor | None,
        n_pages: int,
    ) -> None:
        exp = (
            n_pages,
            self.geometry.num_tokens,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
        )
        if tuple(k_pages_cpu.shape) != exp:
            raise RuntimeError(
                f"k_pages_cpu shape {tuple(k_pages_cpu.shape)} does not match batch geometry {exp}"
            )
        if not self.geometry.is_mla_single_buffer:
            if v_pages_cpu is None or tuple(v_pages_cpu.shape) != exp:
                raise RuntimeError(
                    f"v_pages_cpu shape {None if v_pages_cpu is None else tuple(v_pages_cpu.shape)} does not match batch geometry {exp}"
                )

    def encode_cold_page_tensor(
        self,
        k_page_cpu: torch.Tensor,
        v_page_cpu: torch.Tensor | None,
        tenant_id: str,
        session_id: str,
        aad: ColdPageAAD,
    ) -> torch.Tensor:
        """Returns a contiguous 1D `uint8` tensor (ready for zero-copy RDMA put)."""
        staged_nonces, staged_slots, last_seq = self.keyring.preflight_validate_egress(
            tenant_id=tenant_id,
            session_id=session_id,
            model_id_hash=aad.model_id_hash,
            layer_id=aad.layer_id,
            head_group_id=aad.head_group_id,
            base_sequence_nonce=aad.sequence_nonce,
            token_hashes=[aad.page_token_hash],
        )
        try:
            enc_key, mac_key = self.keyring.begin_egress_execution(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )
            self._validate_single_shapes(k_page_cpu, v_page_cpu)
            k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
            mk0, mk1 = struct.unpack("<QQ", mac_key[:16])
            expected_payload = self.geometry.payload_bytes(aad.preset, aad.rope_mode)

            blob = self._ext.fused_encode_cold_page_cpu(
                k_page_cpu,
                v_page_cpu,
                int(aad.preset),
                int(aad.rope_mode),
                self.geometry.rope_dim,
                expected_payload,
                aad.tenant_id_hash,
                aad.model_id_hash,
                aad.layer_id,
                aad.head_group_id,
                aad.page_token_hash & 0xFFFFFFFFFFFFFFFF,
                aad.sequence_nonce & 0xFFFFFFFFFFFFFFFF,
                k0,
                k1,
                mk0,
                mk1,
                k2,
                k3,
            )
            try:
                self.keyring.commit_egress(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    staged_nonces=staged_nonces,
                    staged_slots=staged_slots,
                    last_seq=last_seq,
                )
            except BaseException:
                blob.zero_()
                raise
            return blob
        finally:
            self.keyring.release_egress_reservation(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )

    def decode_cold_page_tensor(
        self,
        cold_blob_u8: torch.Tensor,
        expected_tenant_id: str,
        session_id: str,
        expected_model_id: str,
        expected_layer_id: int,
        expected_token_hash: int,
        min_sequence_nonce: int | None = None,
        expected_sequence_nonce: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Verifies AAD + 128-bit SipMAC + Freshness, decrypts, and dequantizes into FP16 KV tensors."""
        gen, enc_key, mac_key = self.keyring.begin_ingress(
            expected_tenant_id, session_id
        )
        k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
        mk0, mk1 = struct.unpack("<QQ", mac_key[:16])

        try:
            k_out, v_out = self._ext.fused_decode_cold_page_cpu(
                cold_blob_u8,
                self.geometry.num_tokens,
                self.geometry.num_kv_heads,
                self.geometry.head_dim,
                self.geometry.rope_dim,
                self.geometry.is_mla_single_buffer,
                hash32(expected_tenant_id),
                hash32(expected_model_id),
                expected_layer_id,
                expected_token_hash & 0xFFFFFFFFFFFFFFFF,
                k0,
                k1,
                mk0,
                mk1,
                k2,
                k3,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "CROSS_TENANT_CONTAMINATION" in msg:
                raise CrossTenantContaminationError(msg) from exc
            if "KV_INTEGRITY_ERROR" in msg:
                raise KVIntegrityError(msg) from exc
            raise

        hdr_bytes = cold_blob_u8[:HEADER_BYTES].numpy().tobytes()
        aad = ColdPageAAD.unpack_header(hdr_bytes)
        try:
            self.keyring.verify_and_update_freshness(
                tenant_id=expected_tenant_id,
                session_id=session_id,
                model_id_hash=aad.model_id_hash,
                layer_id=aad.layer_id,
                head_group_id=aad.head_group_id,
                page_token_hash=aad.page_token_hash,
                sequence_nonce=aad.sequence_nonce,
                min_sequence_nonce=min_sequence_nonce,
                expected_sequence_nonce=expected_sequence_nonce,
                expected_generation=gen,
            )
            self.keyring.finish_ingress(expected_tenant_id, session_id, gen)
        except BaseException:
            k_out.zero_()
            if v_out is not None:
                v_out.zero_()
            raise
        return k_out, v_out

    def decode_cold_pages_batch_tensor(
        self,
        cold_blobs_u8: torch.Tensor,
        expected_token_hashes_i64: torch.Tensor,
        expected_tenant_id: str,
        session_id: str,
        expected_model_id: str,
        expected_layer_id: int,
        min_sequence_nonce: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Multi-threaded (GIL-free) batch ingress for [B, blob_bytes] -> ([B, T, H, D], [B, T, H, D] | None)."""
        gen, enc_key, mac_key = self.keyring.begin_ingress(
            expected_tenant_id, session_id
        )
        token_hashes_list = self._validate_token_hashes_tensor(
            expected_token_hashes_i64
        )
        if cold_blobs_u8.dim() != 2 or cold_blobs_u8.size(0) != len(token_hashes_list):
            raise RuntimeError(
                f"cold_blobs_u8 must be 2D [batch, blob_bytes] matching expected_token_hashes_i64 length {len(token_hashes_list)}, got shape={tuple(cold_blobs_u8.shape)}"
            )
        k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
        mk0, mk1 = struct.unpack("<QQ", mac_key[:16])

        try:
            k_out, v_out = self._ext.fused_decode_cold_pages_batch_cpu(
                cold_blobs_u8,
                expected_token_hashes_i64,
                self.geometry.num_tokens,
                self.geometry.num_kv_heads,
                self.geometry.head_dim,
                self.geometry.rope_dim,
                self.geometry.is_mla_single_buffer,
                hash32(expected_tenant_id),
                hash32(expected_model_id),
                expected_layer_id,
                k0,
                k1,
                mk0,
                mk1,
                k2,
                k3,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "CROSS_TENANT_CONTAMINATION" in msg:
                raise CrossTenantContaminationError(msg) from exc
            if "KV_INTEGRITY_ERROR" in msg:
                raise KVIntegrityError(msg) from exc
            raise

        hdrs_bytes = cold_blobs_u8[:, :HEADER_BYTES].contiguous().numpy().tobytes()
        n_b = cold_blobs_u8.size(0)
        try:
            records = []
            for b_idx in range(n_b):
                _, _, _, _, m_hash, l_id, g_id, tok_u64, seq_u64 = struct.unpack_from(
                    "<HBBIIHHQQ", hdrs_bytes, b_idx * HEADER_BYTES
                )
                records.append((m_hash, l_id, g_id, tok_u64, seq_u64))
            self.keyring.verify_and_update_freshness_batch(
                tenant_id=expected_tenant_id,
                session_id=session_id,
                records=records,
                min_sequence_nonce=min_sequence_nonce,
                expected_generation=gen,
            )
            self.keyring.finish_ingress(expected_tenant_id, session_id, gen)
        except BaseException:
            k_out.zero_()
            if v_out is not None:
                v_out.zero_()
            raise
        return k_out, v_out

    def decode_cold_pages_peak_into(
        self,
        cold_blobs_u8: torch.Tensor,
        expected_token_hashes_i64: torch.Tensor,
        out_k_pages_cpu: torch.Tensor,
        out_v_pages_cpu: torch.Tensor | None,
        expected_tenant_id: str,
        session_id: str,
        expected_model_id: str,
        expected_layer_id: int,
        min_sequence_nonce: int | None = None,
    ) -> None:
        """Zero-allocation GCD multi-core peak batch ingress directly into pre-allocated FP16 KV tensors."""
        gen, enc_key, mac_key = self.keyring.begin_ingress(
            expected_tenant_id, session_id
        )
        token_hashes_list = self._validate_token_hashes_tensor(
            expected_token_hashes_i64
        )
        n_b = len(token_hashes_list)
        if cold_blobs_u8.dim() != 2 or int(cold_blobs_u8.size(0)) != n_b:
            raise RuntimeError(
                f"cold_blobs_u8 must be 2D [batch, blob_bytes] with batch={n_b}, got shape={tuple(cold_blobs_u8.shape)}"
            )
        self._validate_batch_shapes(out_k_pages_cpu, out_v_pages_cpu, n_b)
        if (
            self._tensors_overlap(out_k_pages_cpu, out_v_pages_cpu)
            or self._tensors_overlap(out_k_pages_cpu, cold_blobs_u8)
            or self._tensors_overlap(out_k_pages_cpu, expected_token_hashes_i64)
            or self._tensors_overlap(out_v_pages_cpu, cold_blobs_u8)
            or self._tensors_overlap(out_v_pages_cpu, expected_token_hashes_i64)
        ):
            raise RuntimeError(
                "Peak ingress output tensors must not alias or overlap each other or input tensors"
            )
        k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
        mk0, mk1 = struct.unpack("<QQ", mac_key[:16])

        try:
            try:
                self._ext.fused_decode_cold_pages_peak_into_cpu(
                    cold_blobs_u8,
                    expected_token_hashes_i64,
                    out_k_pages_cpu,
                    out_v_pages_cpu,
                    self.geometry.rope_dim,
                    self.geometry.is_mla_single_buffer,
                    hash32(expected_tenant_id),
                    hash32(expected_model_id),
                    expected_layer_id,
                    k0,
                    k1,
                    mk0,
                    mk1,
                    k2,
                    k3,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "CROSS_TENANT_CONTAMINATION" in msg:
                    raise CrossTenantContaminationError(msg) from exc
                if "KV_INTEGRITY_ERROR" in msg:
                    raise KVIntegrityError(msg) from exc
                raise

            hdrs_bytes = cold_blobs_u8[:, :HEADER_BYTES].contiguous().numpy().tobytes()
            records = []
            for b_idx in range(n_b):
                _, _, _, _, m_hash, l_id, g_id, tok_u64, seq_u64 = struct.unpack_from(
                    "<HBBIIHHQQ", hdrs_bytes, b_idx * HEADER_BYTES
                )
                records.append((m_hash, l_id, g_id, tok_u64, seq_u64))
            self.keyring.verify_and_update_freshness_batch(
                tenant_id=expected_tenant_id,
                session_id=session_id,
                records=records,
                min_sequence_nonce=min_sequence_nonce,
                expected_generation=gen,
            )
            self.keyring.finish_ingress(expected_tenant_id, session_id, gen)
        except BaseException:
            if isinstance(out_k_pages_cpu, torch.Tensor):
                out_k_pages_cpu.zero_()
            if isinstance(out_v_pages_cpu, torch.Tensor):
                out_v_pages_cpu.zero_()
            raise

    def encode_cold_pages_batch_tensor(
        self,
        k_pages_cpu: torch.Tensor,
        v_pages_cpu: torch.Tensor | None,
        page_token_hashes_i64: torch.Tensor,
        tenant_id: str,
        session_id: str,
        aad_template: ColdPageAAD,
    ) -> torch.Tensor:
        """Multi-threaded (GIL-free) batch egress for [B, T, H, D] pages -> [B, blob_bytes]."""
        if not isinstance(page_token_hashes_i64, torch.Tensor) or page_token_hashes_i64.numel() < 1:
            raise RuntimeError("page_token_hashes_i64 must be a non-empty torch.Tensor")
        token_hashes_list = page_token_hashes_i64.reshape(-1).tolist()
        staged_nonces, staged_slots, last_seq = self.keyring.preflight_validate_egress(
            tenant_id=tenant_id,
            session_id=session_id,
            model_id_hash=aad_template.model_id_hash,
            layer_id=aad_template.layer_id,
            head_group_id=aad_template.head_group_id,
            base_sequence_nonce=aad_template.sequence_nonce,
            token_hashes=token_hashes_list,
        )
        try:
            enc_key, mac_key = self.keyring.begin_egress_execution(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )
            self._validate_token_hashes_tensor(page_token_hashes_i64)
            self._validate_batch_shapes(
                k_pages_cpu, v_pages_cpu, len(token_hashes_list)
            )
            k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
            mk0, mk1 = struct.unpack("<QQ", mac_key[:16])
            expected_payload = self.geometry.payload_bytes(
                aad_template.preset, aad_template.rope_mode
            )
            out_blobs = self._ext.fused_encode_cold_pages_batch_cpu(
                k_pages_cpu,
                v_pages_cpu,
                page_token_hashes_i64,
                int(aad_template.preset),
                int(aad_template.rope_mode),
                self.geometry.rope_dim,
                expected_payload,
                aad_template.tenant_id_hash,
                aad_template.model_id_hash,
                aad_template.layer_id,
                aad_template.head_group_id,
                aad_template.sequence_nonce & 0xFFFFFFFFFFFFFFFF,
                k0,
                k1,
                mk0,
                mk1,
                k2,
                k3,
            )
            try:
                self.keyring.commit_egress(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    staged_nonces=staged_nonces,
                    staged_slots=staged_slots,
                    last_seq=last_seq,
                )
            except BaseException:
                out_blobs.zero_()
                raise
            return out_blobs
        finally:
            self.keyring.release_egress_reservation(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )

    def encode_cold_pages_peak_into(
        self,
        k_pages_cpu: torch.Tensor,
        v_pages_cpu: torch.Tensor | None,
        page_token_hashes_i64: torch.Tensor,
        out_blobs_u8: torch.Tensor,
        tenant_id: str,
        session_id: str,
        aad_template: ColdPageAAD,
    ) -> None:
        """Zero-allocation GCD multi-core peak batch egress directly into pre-registered RDMA buffer."""
        if not isinstance(page_token_hashes_i64, torch.Tensor) or page_token_hashes_i64.numel() < 1:
            raise RuntimeError("page_token_hashes_i64 must be a non-empty torch.Tensor")
        token_hashes_list = page_token_hashes_i64.reshape(-1).tolist()
        staged_nonces, staged_slots, last_seq = self.keyring.preflight_validate_egress(
            tenant_id=tenant_id,
            session_id=session_id,
            model_id_hash=aad_template.model_id_hash,
            layer_id=aad_template.layer_id,
            head_group_id=aad_template.head_group_id,
            base_sequence_nonce=aad_template.sequence_nonce,
            token_hashes=token_hashes_list,
        )
        try:
            if (
                self._tensors_overlap(out_blobs_u8, k_pages_cpu)
                or self._tensors_overlap(out_blobs_u8, v_pages_cpu)
                or self._tensors_overlap(out_blobs_u8, page_token_hashes_i64)
            ):
                raise RuntimeError(
                    "out_blobs_u8 must not alias or overlap input tensors"
                )
            try:
                enc_key, mac_key = self.keyring.begin_egress_execution(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    staged_nonces=staged_nonces,
                )
                self._validate_token_hashes_tensor(page_token_hashes_i64)
                self._validate_batch_shapes(
                    k_pages_cpu, v_pages_cpu, len(token_hashes_list)
                )
                k0, k1, k2, k3 = struct.unpack("<QQQQ", enc_key[:32])
                mk0, mk1 = struct.unpack("<QQ", mac_key[:16])
                expected_payload = self.geometry.payload_bytes(
                    aad_template.preset, aad_template.rope_mode
                )
                self._ext.fused_encode_cold_pages_peak_into_cpu(
                    k_pages_cpu,
                    v_pages_cpu,
                    page_token_hashes_i64,
                    out_blobs_u8,
                    int(aad_template.preset),
                    int(aad_template.rope_mode),
                    self.geometry.rope_dim,
                    expected_payload,
                    aad_template.tenant_id_hash,
                    aad_template.model_id_hash,
                    aad_template.layer_id,
                    aad_template.head_group_id,
                    aad_template.sequence_nonce & 0xFFFFFFFFFFFFFFFF,
                    k0,
                    k1,
                    mk0,
                    mk1,
                    k2,
                    k3,
                )
                self.keyring.commit_egress(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    staged_nonces=staged_nonces,
                    staged_slots=staged_slots,
                    last_seq=last_seq,
                )
            except BaseException:
                if isinstance(out_blobs_u8, torch.Tensor):
                    out_blobs_u8.zero_()
                raise
        finally:
            self.keyring.release_egress_reservation(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )
