# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hakan Damar and Theseus Guard, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Developed by Theseus Guard Research (https://www.theseusguard.com)
# Repository: https://github.com/Theseus-Guard/ZTCP
# Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
"""End-to-End Validation & Benchmark Suite for Zero-Trust Cold Pages (ZTCP).

Developed by Theseus Guard, Inc. Evaluates:
1. Compression Ratio & RDMA Wire Footprint across GQA (Qwen3.6-35B-A3B/Llama) and MLA (DeepSeek-V4.1-Flash) geometries.
2. Attention Logit / Cosine Reconstruction Fidelity (comparing RoPE-quantized vs. RoPE-preserved modes).
3. Security Invariants:
   - Cross-Tenant / Cross-Prefix KV Contamination rejection (AAD mismatch).
   - Remote RDMA Bit-Flip / KV Poisoning detection (128-bit MAC failure).
   - Instant Crypto-Shredding verification (session key destruction).
4. Effective RDMA Transfer Latency Model across:
   - RoCEv2 / InfiniBand 100 Gbps (12.5 GB/s line rate, 11.2 GB/s effective goodput)
   - RoCEv2 / InfiniBand 200 Gbps (25.0 GB/s line rate, 22.5 GB/s effective goodput)
   - RoCEv2 / InfiniBand 400 Gbps (50.0 GB/s line rate, 45.0 GB/s effective goodput)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.baseline_codec import (
    ColdPageAAD,
    ColdPageGeometry,
    CrossTenantContaminationError,
    FusedColdPageCodec,
    KVIntegrityError,
    NativeCppFusedColdPageCodec,
    NonceReuseError,
    QuantPreset,
    RoPEMode,
    SessionShreddedError,
    StalePageReplayError,
)
from research.baseline_codec.cold_page_spec import hash32, hash64_tokens
from research.baseline_codec.fused_quant_crypto import (
    EphemeralSessionKeyRing,
)


def generate_realistic_kv_page(
    geom: ColdPageGeometry, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Synthesizes KV tensors with realistic heavy-tailed RoPE channel outliers."""
    torch.manual_seed(42)
    shape = (geom.num_tokens, geom.num_kv_heads, geom.head_dim)
    k = torch.randn(shape, dtype=torch.float32, device=device) * 0.4
    # RoPE channels in modern LLMs (Qwen/DeepSeek) exhibit high-magnitude structured frequencies
    if geom.rope_dim > 0:
        rope_outliers = torch.linspace(
            -4.5, 4.5, geom.rope_dim, device=device
        ).reshape(1, 1, geom.rope_dim)
        k[..., : geom.rope_dim] += rope_outliers * torch.sin(
            torch.arange(geom.num_tokens, device=device, dtype=torch.float32).reshape(
                -1, 1, 1
            )
        )
    k = k.to(torch.float16)

    if geom.is_mla_single_buffer:
        return k, None

    v = (torch.randn(shape, dtype=torch.float32, device=device) * 0.25).to(
        torch.float16
    )
    return k, v


def evaluate_attention_fidelity(
    k_orig: torch.Tensor,
    v_orig: torch.Tensor | None,
    k_rec: torch.Tensor,
    v_rec: torch.Tensor | None,
) -> dict[str, float]:
    """Measures K cosine/MSE and simulated single-page attention fidelity.

    When V is present (GQA), computes attention-weighted Value output cosine
    similarity; when V is None (single-buffer DeepSeek-MLA), computes softmax
    attention-weights cosine similarity. Note: query q is sampled with seed=99
    on k_ref.device.
    """
    k_ref = k_orig.to(torch.float32)
    k_hat = k_rec.to(torch.float32)

    k_cos = F.cosine_similarity(k_ref.reshape(-1), k_hat.reshape(-1), dim=0).item()
    k_mse = F.mse_loss(k_hat, k_ref).item()

    # Simulate query vector attending over the 64-token page
    torch.manual_seed(99)
    q = torch.randn(
        (1, k_ref.shape[1], k_ref.shape[2]), dtype=torch.float32, device=k_ref.device
    )
    scale = k_ref.shape[2] ** -0.5

    # Scores: [num_heads, num_tokens]
    logits_ref = torch.einsum("qhd,thd->ht", q, k_ref) * scale
    logits_hat = torch.einsum("qhd,thd->ht", q, k_hat) * scale
    attn_ref = F.softmax(logits_ref, dim=-1)
    attn_hat = F.softmax(logits_hat, dim=-1)

    if v_orig is not None and v_rec is not None:
        v_ref = v_orig.to(torch.float32)
        v_hat = v_rec.to(torch.float32)
        out_ref = torch.einsum("ht,thd->hd", attn_ref, v_ref)
        out_hat = torch.einsum("ht,thd->hd", attn_hat, v_hat)
        attn_out_cos = F.cosine_similarity(
            out_ref.reshape(-1), out_hat.reshape(-1), dim=0
        ).item()
    else:
        attn_out_cos = F.cosine_similarity(
            attn_ref.reshape(-1), attn_hat.reshape(-1), dim=0
        ).item()

    return {
        "k_cos": k_cos,
        "k_mse": k_mse,
        "attn_out_cos": attn_out_cos,
    }


def run_security_verification(codec: FusedColdPageCodec, geom: ColdPageGeometry) -> None:
    print("\n" + "=" * 78)
    print("3. ZERO-TRUST SECURITY & INTEGRITY VERIFICATION")
    print("=" * 78)

    k_page, v_page = generate_realistic_kv_page(geom, codec.device)
    tokens = tuple(range(1000, 1000 + geom.num_tokens))
    token_hash = hash64_tokens(tokens)

    aad = ColdPageAAD.from_context(
        tenant_id="tenant-alpha-bank",
        model_id="Qwen/Qwen3.6-35B-A3B",
        layer_id=14,
        token_ids=tokens,
        sequence_nonce=1,
        preset=QuantPreset.K8V4_ASYM,
        rope_mode=RoPEMode.PRESERVE_FP8,
    )

    blob = codec.encode_cold_page(
        k_page, v_page, tenant_id="tenant-alpha-bank", session_id="sess-001", aad=aad
    )

    # Test 3.1: Valid round-trip
    k_dec, v_dec = codec.decode_cold_page(
        blob,
        expected_tenant_id="tenant-alpha-bank",
        session_id="sess-001",
        expected_model_id="Qwen/Qwen3.6-35B-A3B",
        expected_layer_id=14,
        expected_token_hash=token_hash,
    )
    assert k_dec.shape == k_page.shape
    print("[PASS] 3.1 Valid Tenant + Session + AAD round-trip authenticated & decoded.")

    # Test 3.2: Cross-Tenant Contamination (LMCache Issue #4385 scenario)
    # Tenant Beta has a valid active session, but Mooncake/L2 accidentally returns Tenant Alpha's blob
    codec.keyring.derive_or_get("tenant-beta-defense", "sess-beta-009")
    try:
        codec.decode_cold_page(
            blob,
            expected_tenant_id="tenant-beta-defense",
            session_id="sess-beta-009",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Cross-tenant access should have failed!")
    except CrossTenantContaminationError as exc:
        print(f"[PASS] 3.2 Cross-Tenant KV Contamination blocked -> {exc}")

    # Test 3.3: Simulated RDMA Memory Poisoning (Local Single Bit-Flip in Ciphertext)
    tampered = bytearray(blob)
    tampered[len(tampered) // 2] ^= 0x01
    try:
        codec.decode_cold_page(
            bytes(tampered),
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-001",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Tampered ciphertext should have failed MAC verification!")
    except KVIntegrityError as exc:
        print(f"[PASS] 3.3 Simulated RDMA KV Poisoning (local 1-bit flip) blocked -> {exc}")

    # Test 3.4: Instant Crypto-Shredding + Re-Derivation Non-Recoverability (Python Codec)
    old_enc_ref, old_mac_ref, _, _ = codec.keyring._active_keys[
        ("tenant-alpha-bank", "sess-001")
    ]
    old_enc_snapshot = bytes(old_enc_ref)
    old_mac_snapshot = bytes(old_mac_ref)
    shredded = codec.keyring.crypto_shred_session("tenant-alpha-bank", "sess-001")
    assert shredded is True
    assert all(b == 0 for b in old_enc_ref) and all(b == 0 for b in old_mac_ref)
    try:
        codec.decode_cold_page(
            blob,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-001",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Shredded session should have failed!")
    except SessionShreddedError as exc:
        print(f"[PASS] 3.4a Instant Crypto-Shredding verified -> {exc}")

    # Re-deriving for the same (tenant_id, session_id) after shred must produce an independent key
    # and must NOT make the archived blob decodable (B5).
    new_enc_key, new_mac_key, _, _ = codec.keyring.derive_or_get(
        "tenant-alpha-bank", "sess-001"
    )
    assert bytes(new_enc_key) != old_enc_snapshot and bytes(new_mac_key) != old_mac_snapshot
    try:
        codec.decode_cold_page(
            blob,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-001",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError(
            "Archived blob from shredded session must remain undecipherable after re-derivation!"
        )
    except KVIntegrityError as exc:
        print(
            f"[PASS] 3.4b Post-shred re-derivation produces independent key; archived blob rejected -> {exc}"
        )

    # 3.5 - 3.11: Native C++ Peak Path End-to-End Security, Tamper, Freshness & Shredding Suite (B4/B5/B6)
    cpu_dev = torch.device("cpu")
    k_cpu = k_page.to(cpu_dev).contiguous()
    v_cpu = v_page.to(cpu_dev).contiguous() if v_page is not None else None
    cpp_codec = NativeCppFusedColdPageCodec(geom, codec.keyring)
    token_hash_i64 = struct.unpack("<q", struct.pack("<Q", token_hash))[0]
    th_tensor = torch.tensor([token_hash_i64], dtype=torch.int64, device=cpu_dev)

    aad_peak_v1 = ColdPageAAD.from_context(
        tenant_id="tenant-alpha-bank",
        model_id="Qwen/Qwen3.6-35B-A3B",
        layer_id=14,
        token_ids=tokens,
        sequence_nonce=10,
        preset=QuantPreset.INT4_SYM,
        rope_mode=RoPEMode.PRESERVE_FP8,
    )
    peak_blob_bytes = geom.cold_blob_bytes(
        aad_peak_v1.preset, aad_peak_v1.rope_mode
    )
    peak_out_v1 = torch.empty(
        (1, peak_blob_bytes), dtype=torch.uint8, device=cpu_dev
    )
    cpp_codec.encode_cold_pages_peak_into(
        k_cpu.unsqueeze(0),
        v_cpu.unsqueeze(0) if v_cpu is not None else None,
        th_tensor,
        peak_out_v1,
        tenant_id="tenant-alpha-bank",
        session_id="sess-peak-sec",
        aad_template=aad_peak_v1,
    )

    # Test 3.5: Native Peak valid round-trip + legitimate cache re-read (B4 & B6)
    k_peak_dec1, v_peak_dec1 = cpp_codec.decode_cold_page_tensor(
        peak_out_v1[0],
        expected_tenant_id="tenant-alpha-bank",
        session_id="sess-peak-sec",
        expected_model_id="Qwen/Qwen3.6-35B-A3B",
        expected_layer_id=14,
        expected_token_hash=token_hash,
    )
    k_peak_dec2, _ = cpp_codec.decode_cold_page_tensor(
        peak_out_v1[0],
        expected_tenant_id="tenant-alpha-bank",
        session_id="sess-peak-sec",
        expected_model_id="Qwen/Qwen3.6-35B-A3B",
        expected_layer_id=14,
        expected_token_hash=token_hash,
    )
    assert k_peak_dec1.shape == k_cpu.shape and torch.equal(k_peak_dec1, k_peak_dec2)
    peak_sec_fid = evaluate_attention_fidelity(k_cpu, v_cpu, k_peak_dec1, v_peak_dec1)
    assert peak_sec_fid["attn_out_cos"] > 0.995
    print(
        f"[PASS] 3.5 Native Peak round-trip & legitimate cache re-read verified (Attn CosSim={peak_sec_fid['attn_out_cos']:.6f})"
    )

    # Offsets within payload for GQA INT4_SYM + PRESERVE_FP8:
    vecs = geom.vectors_per_buffer
    gs = geom.group_size
    r_dim = geom.rope_dim
    n_dim = geom.nope_dim
    s_rope_start = 32 + vecs * r_dim
    q_nope_start = s_rope_start + vecs * (r_dim // gs) * 2
    q_v_start = q_nope_start + vecs * (n_dim // 2) + vecs * (n_dim // gs) * 2

    # Test 3.6: Native Peak Scale Tamper Rejection (B2 & B4)
    peak_scale_tampered = peak_out_v1[0].clone()
    peak_scale_tampered[s_rope_start] ^= 0x01
    try:
        cpp_codec.decode_cold_page_tensor(
            peak_scale_tampered,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Native Peak scale tamper should have failed!")
    except KVIntegrityError as exc:
        print(f"[PASS] 3.6 Native Peak scale tamper blocked -> {exc}")

    # Test 3.7: Native Peak 16B NoPE Block-Swap Rejection (B2 & B4)
    peak_block_swapped = peak_out_v1[0].clone()
    blk0 = peak_block_swapped[q_nope_start : q_nope_start + 16].clone()
    blk1 = peak_block_swapped[q_nope_start + 16 : q_nope_start + 32].clone()
    peak_block_swapped[q_nope_start : q_nope_start + 16] = blk1
    peak_block_swapped[q_nope_start + 16 : q_nope_start + 32] = blk0
    try:
        cpp_codec.decode_cold_page_tensor(
            peak_block_swapped,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Native Peak NoPE block swap should have failed!")
    except KVIntegrityError as exc:
        print(f"[PASS] 3.7 Native Peak NoPE 16B block swap blocked -> {exc}")

    # Test 3.8: Native Peak Paired V-Block Bit-Flip Rejection (B2 & B4)
    peak_v_paired = peak_out_v1[0].clone()
    peak_v_paired[q_v_start] ^= 0x01
    peak_v_paired[q_v_start + 16] ^= 0x01
    try:
        cpp_codec.decode_cold_page_tensor(
            peak_v_paired,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Native Peak paired V bit-flips should have failed!")
    except KVIntegrityError as exc:
        print(f"[PASS] 3.8 Native Peak paired V-block bit-flips blocked -> {exc}")

    # Test 3.9: Native Peak AAD Contamination Rejection across Tenant/Model/Layer/TokenHash (B4)
    for bad_tenant, bad_model, bad_layer, bad_tok in (
        ("tenant-beta-defense", "Qwen/Qwen3.6-35B-A3B", 14, token_hash),
        ("tenant-alpha-bank", "Qwen/Other-Model", 14, token_hash),
        ("tenant-alpha-bank", "Qwen/Qwen3.6-35B-A3B", 15, token_hash),
        ("tenant-alpha-bank", "Qwen/Qwen3.6-35B-A3B", 14, token_hash ^ 1),
    ):
        codec.keyring.derive_or_get(bad_tenant, "sess-peak-sec")
        try:
            cpp_codec.decode_cold_page_tensor(
                peak_out_v1[0],
                expected_tenant_id=bad_tenant,
                session_id="sess-peak-sec",
                expected_model_id=bad_model,
                expected_layer_id=bad_layer,
                expected_token_hash=bad_tok,
            )
            raise AssertionError("Native Peak AAD mismatch should have failed!")
        except CrossTenantContaminationError:
            pass
    print(
        "[PASS] 3.9 Native Peak AAD mismatch (tenant/model/layer/token_hash) blocked."
    )

    # Test 3.10: Freshness Contract (Stale-Version Replay Blocked) & Final ChaCha20 Nonce Uniqueness (B6)
    # 3.10a: Reusing the same sequence_nonce on encode_cold_pages_peak_into under the same session must raise NonceReuseError
    try:
        cpp_codec.encode_cold_pages_peak_into(
            k_cpu.unsqueeze(0),
            v_cpu.unsqueeze(0) if v_cpu is not None else None,
            th_tensor,
            peak_out_v1,
            tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            aad_template=aad_peak_v1,
        )
        raise AssertionError("Duplicate ChaCha20 nonce under same session must fail!")
    except NonceReuseError as exc:
        print(f"[PASS] 3.10a Session ChaCha20 nonce reuse blocked -> {exc}")

    # 3.10b: Bijective 96-bit ChaCha20 nonce packing prevents XOR cancellation across distinct sequence_nonce values
    aad_xor_a = dataclasses.replace(
        aad_peak_v1, sequence_nonce=(1 << 32) | 7, page_token_hash=1
    )
    aad_xor_b = dataclasses.replace(
        aad_peak_v1, sequence_nonce=(2 << 32) | 7, page_token_hash=2
    )
    assert FusedColdPageCodec._derive_chacha20_nonce(
        aad_xor_a
    ) != FusedColdPageCodec._derive_chacha20_nonce(aad_xor_b)

    # 3.10c: Encode v2 (sequence_nonce=11) for the same page slot; v2 decodes (and re-reads), while stale v1 (nonce=10) is rejected
    aad_peak_v2 = dataclasses.replace(aad_peak_v1, sequence_nonce=11)
    peak_out_v2 = torch.empty(
        (1, peak_blob_bytes), dtype=torch.uint8, device=cpu_dev
    )
    cpp_codec.encode_cold_pages_peak_into(
        k_cpu.unsqueeze(0),
        v_cpu.unsqueeze(0) if v_cpu is not None else None,
        th_tensor,
        peak_out_v2,
        tenant_id="tenant-alpha-bank",
        session_id="sess-peak-sec",
        aad_template=aad_peak_v2,
    )
    _ = cpp_codec.decode_cold_page_tensor(
        peak_out_v2[0],
        expected_tenant_id="tenant-alpha-bank",
        session_id="sess-peak-sec",
        expected_model_id="Qwen/Qwen3.6-35B-A3B",
        expected_layer_id=14,
        expected_token_hash=token_hash,
    )
    try:
        cpp_codec.decode_cold_page_tensor(
            peak_out_v1[0],
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Stale cold-page version v1 (nonce=10 < 11) must fail!")
    except StalePageReplayError as exc:
        print(f"[PASS] 3.10b Stale cold-page version replay blocked -> {exc}")

    # Test 3.11: Native Peak Irreversible Crypto-Shredding (B5)
    assert codec.keyring.crypto_shred_session("tenant-alpha-bank", "sess-peak-sec")
    codec.keyring.derive_or_get("tenant-alpha-bank", "sess-peak-sec")
    try:
        cpp_codec.decode_cold_page_tensor(
            peak_out_v2[0],
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-peak-sec",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError(
            "Archived native Peak blob must remain undecipherable after shred + re-derive!"
        )
    except KVIntegrityError as exc:
        print(
            f"[PASS] 3.11 Native Peak post-shred re-derivation cannot decrypt archived blob -> {exc}"
        )

    # Test 3.12: Transactional Session State, Concurrency & Monotonic Watermark (CODE-003)
    codec.keyring.derive_or_get("tenant-alpha-bank", "sess-code003")
    # 3.12a: Concurrent same-nonce race on Native Peak -> exactly 1 succeeds, 1 raises NonceReuseError
    race_aad_a = dataclasses.replace(
        aad_peak_v1, sequence_nonce=20, page_token_hash=token_hash
    )
    race_aad_b = dataclasses.replace(
        aad_peak_v1, sequence_nonce=20, page_token_hash=token_hash ^ 3
    )
    race_barrier = threading.Barrier(2)

    def _peak_race_worker(aad_w: ColdPageAAD, th_w: int) -> torch.Tensor:
        out_w = torch.empty((1, peak_blob_bytes), dtype=torch.uint8, device=cpu_dev)
        th_w_i64 = struct.unpack("<q", struct.pack("<Q", th_w))[0]
        race_barrier.wait(timeout=10)
        cpp_codec.encode_cold_pages_peak_into(
            k_cpu.unsqueeze(0),
            v_cpu.unsqueeze(0) if v_cpu is not None else None,
            torch.tensor([th_w_i64], dtype=torch.int64, device=cpu_dev),
            out_w,
            tenant_id="tenant-alpha-bank",
            session_id="sess-code003",
            aad_template=aad_w,
        )
        return out_w

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_peak_race_worker, race_aad_a, token_hash)
        fut_b = pool.submit(_peak_race_worker, race_aad_b, token_hash ^ 3)
        outcomes = []
        for fut in (fut_a, fut_b):
            try:
                outcomes.append(("ok", fut.result(timeout=15)))
            except NonceReuseError as exc:
                outcomes.append(("nonce_reuse", exc))
    assert [o[0] for o in outcomes].count("ok") == 1
    assert [o[0] for o in outcomes].count("nonce_reuse") == 1

    # 3.12b: Failed encode after preflight still burns the reserved nonce
    fail_aad = dataclasses.replace(aad_peak_v1, sequence_nonce=25)
    try:
        cpp_codec.encode_cold_page_tensor(
            k_cpu[:1],  # invalid token count -> fails in C++ after preflight reservation
            v_cpu,
            tenant_id="tenant-alpha-bank",
            session_id="sess-code003",
            aad=fail_aad,
        )
        raise AssertionError("Mismatched k/v shape should have failed!")
    except RuntimeError:
        pass
    try:
        cpp_codec.encode_cold_page_tensor(
            k_cpu,
            v_cpu,
            tenant_id="tenant-alpha-bank",
            session_id="sess-code003",
            aad=fail_aad,
        )
        raise AssertionError("Nonce from failed encode must remain burned!")
    except NonceReuseError:
        pass

    # 3.12c: Out-of-order commit does NOT roll back slot watermark (Native Single)
    orig_preflight = codec.keyring.preflight_validate_egress
    low_ready_ev, high_done_ev = threading.Event(), threading.Event()

    def _ordered_preflight(*args, **kwargs):
        res = orig_preflight(*args, **kwargs)
        if (
            kwargs.get("session_id") == "sess-code003"
            and kwargs.get("base_sequence_nonce") == 30
        ):
            low_ready_ev.set()
            assert high_done_ev.wait(timeout=10)
        return res

    codec.keyring.preflight_validate_egress = _ordered_preflight
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_low = pool.submit(
                cpp_codec.encode_cold_page_tensor,
                k_cpu,
                v_cpu,
                "tenant-alpha-bank",
                "sess-code003",
                dataclasses.replace(aad_peak_v1, sequence_nonce=30),
            )
            assert low_ready_ev.wait(timeout=10)
            blob_high31 = cpp_codec.encode_cold_page_tensor(
                k_cpu,
                v_cpu,
                "tenant-alpha-bank",
                "sess-code003",
                dataclasses.replace(aad_peak_v1, sequence_nonce=31),
            )
            high_done_ev.set()
            blob_low30 = f_low.result(timeout=15)
    finally:
        codec.keyring.preflight_validate_egress = orig_preflight

    try:
        cpp_codec.decode_cold_page_tensor(
            blob_low30,
            expected_tenant_id="tenant-alpha-bank",
            session_id="sess-code003",
            expected_model_id="Qwen/Qwen3.6-35B-A3B",
            expected_layer_id=14,
            expected_token_hash=token_hash,
        )
        raise AssertionError("Late-committed seq=30 must be rejected as stale (< 31)!")
    except StalePageReplayError:
        pass
    _ = cpp_codec.decode_cold_page_tensor(
        blob_high31,
        expected_tenant_id="tenant-alpha-bank",
        session_id="sess-code003",
        expected_model_id="Qwen/Qwen3.6-35B-A3B",
        expected_layer_id=14,
        expected_token_hash=token_hash,
    )
    print(
        "[PASS] 3.12 Transactional session state (atomic nonce burn, monotonic watermark & concurrency) verified."
    )


def _stat_summary(vals: list[float], ndigits: int = 2) -> dict[str, float]:
    mean_v = statistics.mean(vals)
    std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
    med_v = statistics.median(vals)
    return {
        "mean": round(mean_v, ndigits),
        "std": round(std_v, ndigits),
        "median": round(med_v, ndigits),
        "min": round(min(vals), ndigits),
        "max": round(max(vals), ndigits),
    }


def run_single_process_benchmark(run_index: int = 1) -> dict:
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    cpu_device = torch.device("cpu")
    print(
        f"ZTCP Research Benchmark Suite (Run #{run_index}, PID {os.getpid()}) | "
        f"PyTorch {torch.__version__} | Device: {device}"
    )

    pepper = os.urandom(32)
    keyring = EphemeralSessionKeyRing(deployment_pepper=pepper)

    geometries = [
        (
            "GQA-Standard (64 tok, 8 heads, dim 128, RoPE 64)",
            ColdPageGeometry(
                num_tokens=64,
                num_kv_heads=8,
                head_dim=128,
                rope_dim=64,
                group_size=16,
                hot_element_bytes=2,
            ),
        ),
        (
            "DeepSeek-MLA (128 tok, 1 latent head, dim 576, RoPE 64)",
            ColdPageGeometry(
                num_tokens=128,
                num_kv_heads=1,
                head_dim=576,
                rope_dim=64,
                group_size=16,
                hot_element_bytes=2,
                is_mla_single_buffer=True,
            ),
        ),
    ]

    configs = [
        ("INT4_SYM + Quantize All", QuantPreset.INT4_SYM, RoPEMode.QUANTIZE_ALL),
        ("INT4_SYM + Preserve RoPE (FP8)", QuantPreset.INT4_SYM, RoPEMode.PRESERVE_FP8),
        ("INT4_SYM + Preserve RoPE (Raw)", QuantPreset.INT4_SYM, RoPEMode.PRESERVE_RAW),
        ("K8V4_ASYM + Preserve RoPE (FP8)", QuantPreset.K8V4_ASYM, RoPEMode.PRESERVE_FP8),
        ("FP8_ONLY (2x Baseline)", QuantPreset.FP8_ONLY, RoPEMode.QUANTIZE_ALL),
    ]

    print("\n" + "=" * 96)
    print(
        "1. SYNTHETIC ROPE-OUTLIER KV FIDELITY & WIRE COMPRESSION (GQA=Attn Output Cos; MLA=Attn Weights Cos)"
    )
    print("=" * 96)

    torch.manual_seed(42)
    fidelity_summary: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for geom_idx, (geom_name, geom) in enumerate(geometries):
        attn_col_py = "Py Out Cos" if not geom.is_mla_single_buffer else "Py Wt Cos "
        attn_col_cpp = (
            "C++ Out Cos" if not geom.is_mla_single_buffer else "C++ Wt Cos "
        )
        metric_kind = (
            "attention_output_cosine (softmax(QK^T/sqrt(d))V)"
            if not geom.is_mla_single_buffer
            else "attention_weights_cosine (softmax(QK^T/sqrt(d)); V is None in single-buffer MLA, stored in *_attn_out_cos fields for schema uniformity)"
        )
        print(
            f"\n--- Geometry: {geom_name} | Hot Page: {geom.hot_page_bytes:,} B "
            f"({geom.hot_page_bytes / 1024.0:.1f} KiB / {geom.hot_page_bytes / 1e6:.3f} MB) ---"
        )
        print(
            f"{'Configuration':<34} | {'Cold Blob':>10} | {'Ratio':>6} | "
            f"{'Py K Cos':>8} | {attn_col_py:>11} | {attn_col_cpp:>12}"
        )
        print("-" * 96)

        codec = FusedColdPageCodec(geom, keyring, device=device)
        cpp_geom_codec = NativeCppFusedColdPageCodec(geom, keyring)
        k_page, v_page = generate_realistic_kv_page(geom, device)
        k_page_cpu = k_page.to(cpu_device).contiguous()
        v_page_cpu = (
            v_page.to(cpu_device).contiguous() if v_page is not None else None
        )
        tokens = tuple(range(geom.num_tokens))
        t_hash = hash64_tokens(tokens)
        geom_key = "gqa_standard" if geom_idx == 0 else "deepseek_mla"
        fidelity_summary[geom_key] = {}

        for cfg_idx, (label, preset, rope_mode) in enumerate(configs):
            aad = ColdPageAAD.from_context(
                tenant_id="bench-tenant",
                model_id="bench-model",
                layer_id=geom_idx,
                token_ids=tokens,
                sequence_nonce=100 + geom_idx * 10 + cfg_idx,
                preset=preset,
                rope_mode=rope_mode,
            )
            blob = codec.encode_cold_page(
                k_page, v_page, "bench-tenant", "bench-sess-py", aad
            )
            k_rec, v_rec = codec.decode_cold_page(
                blob, "bench-tenant", "bench-sess-py", "bench-model", geom_idx, t_hash
            )
            metrics_py = evaluate_attention_fidelity(k_page, v_page, k_rec, v_rec)

            blob_cpp_cfg = cpp_geom_codec.encode_cold_page_tensor(
                k_page_cpu, v_page_cpu, "bench-tenant", "bench-sess-cpp", aad
            )
            k_rec_cpp, v_rec_cpp = cpp_geom_codec.decode_cold_page_tensor(
                blob_cpp_cfg,
                "bench-tenant",
                "bench-sess-cpp",
                "bench-model",
                geom_idx,
                t_hash,
            )
            metrics_cpp = evaluate_attention_fidelity(
                k_page_cpu, v_page_cpu, k_rec_cpp, v_rec_cpp
            )
            ratio = geom.compression_ratio(preset, rope_mode)
            print(
                f"{label:<34} | {len(blob):>8,} B | {ratio:>5.2f}x | "
                f"{metrics_py['k_cos']:>8.5f} | {metrics_py['attn_out_cos']:>11.6f} | "
                f"{metrics_cpp['attn_out_cos']:>12.6f}"
            )
            fidelity_summary[geom_key][label] = {
                "attn_metric_kind": metric_kind,
                "cold_blob_bytes": len(blob),
                "compression_ratio": round(ratio, 4),
                "py_k_cos": round(metrics_py["k_cos"], 6),
                "py_attn_out_cos": round(metrics_py["attn_out_cos"], 6),
                "cpp_k_cos": round(metrics_cpp["k_cos"], 6),
                "cpp_attn_out_cos": round(metrics_cpp["attn_out_cos"], 6),
            }

    gqa_geom = geometries[0][1]

    # 3. Security verification
    run_security_verification(
        FusedColdPageCodec(gqa_geom, keyring, device=device), gqa_geom
    )

    # 4. Real Microbenchmark: Multi-Op PyTorch vs. C++17 Single-Pass Fused Kernel
    print("\n" + "=" * 96)
    print(
        "4. REAL KERNEL MICROBENCHMARK: Multi-Op Python/PyTorch vs. C++17 Fused (RFC 8439 ChaCha20 + SipHash-2-4-128)"
    )
    print("=" * 96)
    py_codec = FusedColdPageCodec(gqa_geom, keyring, device=cpu_device)
    cpp_codec = NativeCppFusedColdPageCodec(gqa_geom, keyring)

    # Note: generate_realistic_kv_page sets torch.manual_seed(42) on cpu_device
    # (whereas Section 1 synthesized seed=42 on device='mps' before copying to CPU).
    k_cpu, v_cpu = generate_realistic_kv_page(gqa_geom, cpu_device)
    tokens = tuple(range(gqa_geom.num_tokens))
    t_hash = hash64_tokens(tokens)
    aad = ColdPageAAD.from_context(
        tenant_id="perf-tenant",
        model_id="Qwen/Qwen3.6-35B-A3B",
        layer_id=7,
        token_ids=tokens,
        sequence_nonce=555,
        preset=QuantPreset.INT4_SYM,
        rope_mode=RoPEMode.PRESERVE_FP8,
    )

    # Warmup & Fidelity Check on C++ Native Kernel
    blob_py = py_codec.encode_cold_page(k_cpu, v_cpu, "perf-tenant", "perf-sess", aad)
    blob_cpp = cpp_codec.encode_cold_page_tensor(
        k_cpu,
        v_cpu,
        "perf-tenant",
        "perf-sess",
        dataclasses.replace(aad, sequence_nonce=556),
    )
    k_cpp_dec, v_cpp_dec = cpp_codec.decode_cold_page_tensor(
        blob_cpp, "perf-tenant", "perf-sess", "Qwen/Qwen3.6-35B-A3B", 7, t_hash
    )
    cpp_fid = evaluate_attention_fidelity(k_cpu, v_cpu, k_cpp_dec, v_cpp_dec)

    iters = 200
    py_aads = [dataclasses.replace(aad, sequence_nonce=1000 + i) for i in range(iters)]
    t0 = time.perf_counter()
    for i in range(iters):
        b = py_codec.encode_cold_page(
            k_cpu, v_cpu, "perf-tenant", "perf-sess", py_aads[i]
        )
    py_enc_us = ((time.perf_counter() - t0) / iters) * 1e6

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = py_codec.decode_cold_page(
            b, "perf-tenant", "perf-sess", "Qwen/Qwen3.6-35B-A3B", 7, t_hash
        )
    py_dec_us = ((time.perf_counter() - t0) / iters) * 1e6

    cpp_aads = [dataclasses.replace(aad, sequence_nonce=2000 + i) for i in range(iters)]
    t0 = time.perf_counter()
    for i in range(iters):
        b_t = cpp_codec.encode_cold_page_tensor(
            k_cpu, v_cpu, "perf-tenant", "perf-sess", cpp_aads[i]
        )
    cpp_enc_us = ((time.perf_counter() - t0) / iters) * 1e6

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cpp_codec.decode_cold_page_tensor(
            b_t, "perf-tenant", "perf-sess", "Qwen/Qwen3.6-35B-A3B", 7, t_hash
        )
    cpp_dec_us = ((time.perf_counter() - t0) / iters) * 1e6

    cold_blob_bytes = int(blob_cpp.numel())
    page_kib = gqa_geom.hot_page_bytes / 1024.0
    page_mb_si = gqa_geom.hot_page_bytes / 1e6
    py_enc_gbs = (gqa_geom.hot_page_bytes / 1e9) / (py_enc_us * 1e-6)
    py_enc_wire_gbs = (cold_blob_bytes / 1e9) / (py_enc_us * 1e-6)
    cpp_enc_gbs = (gqa_geom.hot_page_bytes / 1e9) / (cpp_enc_us * 1e-6)
    cpp_enc_wire_gbs = (cold_blob_bytes / 1e9) / (cpp_enc_us * 1e-6)
    cpp_dec_gbs = (gqa_geom.hot_page_bytes / 1e9) / (cpp_dec_us * 1e-6)
    cpp_1c_egress_speedup = py_enc_us / cpp_enc_us
    cpp_1c_roundtrip_speedup = (py_enc_us + py_dec_us) / (cpp_enc_us + cpp_dec_us)

    print(
        f"Page Size: {gqa_geom.hot_page_bytes:,} B ({page_kib:.1f} KiB / {page_mb_si:.3f} MB) -> "
        f"Cold Blob: {cold_blob_bytes:,} B | C++ Attn CosSim: {cpp_fid['attn_out_cos']:.6f}"
    )
    print("-" * 96)
    print(
        f"{'Implementation (Single CPU Core)':<36} | {'Egress (Enc+MAC)':>16} | {'Ingress (MAC+Dec)':>17} | {'Egress Speedup':>14}"
    )
    print("-" * 96)
    print(
        f"{'Multi-Op Python / PyTorch':<36} | {py_enc_us:>13.1f} us | {py_dec_us:>14.1f} us | {'1.00x':>14}"
    )
    print(
        f"{'C++17 Single-Pass Fused (-O3 SIMD)':<36} | {cpp_enc_us:>13.1f} us | {cpp_dec_us:>14.1f} us | "
        f"{cpp_1c_egress_speedup:>13.2f}x"
    )
    print(
        f"Single-Core C++ Fused Throughput     : Egress = {cpp_enc_gbs:.2f} GB/s hot ({cpp_enc_wire_gbs:.2f} GB/s wire) | "
        f"Ingress = {cpp_dec_gbs:.2f} GB/s hot"
    )

    # 5. Multi-Core Batch Egress Benchmark (64 Pages = 16.78 MB / 16.0 MiB Layer Batch via at::parallel_for)
    batch_pages = 64
    k_batch = k_cpu.unsqueeze(0).repeat(batch_pages, 1, 1, 1).contiguous()
    v_batch = v_cpu.unsqueeze(0).repeat(batch_pages, 1, 1, 1).contiguous()
    token_hashes_i64 = torch.arange(
        10000, 10000 + batch_pages, dtype=torch.int64, device=cpu_device
    )

    # Warmup batch
    _ = cpp_codec.encode_cold_pages_batch_tensor(
        k_batch,
        v_batch,
        token_hashes_i64,
        "perf-tenant",
        "perf-sess",
        dataclasses.replace(aad, sequence_nonce=10000),
    )
    batch_iters = 50
    batch_aads = [
        dataclasses.replace(aad, sequence_nonce=11000 + i * batch_pages)
        for i in range(batch_iters)
    ]
    t0 = time.perf_counter()
    for i in range(batch_iters):
        batch_blobs = cpp_codec.encode_cold_pages_batch_tensor(
            k_batch, v_batch, token_hashes_i64, "perf-tenant", "perf-sess", batch_aads[i]
        )
    batch_total_s = (time.perf_counter() - t0) / batch_iters
    batch_ms = batch_total_s * 1000.0
    batch_per_page_us = (batch_total_s / batch_pages) * 1e6
    batch_raw_bytes = batch_pages * gqa_geom.hot_page_bytes
    batch_wire_bytes = batch_pages * cold_blob_bytes
    batch_gbs = (batch_raw_bytes / 1e9) / batch_total_s
    batch_wire_gbs = (batch_wire_bytes / 1e9) / batch_total_s

    # Multi-Core Batch-64 Ingress (Decode) Benchmark
    _ = cpp_codec.decode_cold_pages_batch_tensor(
        batch_blobs, token_hashes_i64, "perf-tenant", "perf-sess", "Qwen/Qwen3.6-35B-A3B", 7
    )
    t0 = time.perf_counter()
    for _ in range(batch_iters):
        _ = cpp_codec.decode_cold_pages_batch_tensor(
            batch_blobs,
            token_hashes_i64,
            "perf-tenant",
            "perf-sess",
            "Qwen/Qwen3.6-35B-A3B",
            7,
        )
    batch_dec_total_s = (time.perf_counter() - t0) / batch_iters
    batch_dec_per_page_us = (batch_dec_total_s / batch_pages) * 1e6
    batch_dec_gbs = (batch_raw_bytes / 1e9) / batch_dec_total_s

    print(
        f"Multi-Core Batch (64 Pages = 16.78 MB / 16.0 MiB): Egress = {batch_ms:.2f} ms ({batch_per_page_us:.2f} us/page, {batch_gbs:.2f} GB/s hot, {batch_wire_gbs:.2f} GB/s wire) | "
        f"Ingress = {batch_dec_per_page_us:.2f} us/page ({batch_dec_gbs:.2f} GB/s hot)"
    )

    # 5b. Zero-Allocation Preallocated CPU Buffer Peak Benchmark (256 Pages = 67.11 MB / 64.0 MiB Batch)
    peak_pages = 256
    k_peak = k_cpu.unsqueeze(0).repeat(peak_pages, 1, 1, 1).contiguous()
    v_peak = v_cpu.unsqueeze(0).repeat(peak_pages, 1, 1, 1).contiguous()
    peak_hashes = torch.arange(
        20000, 20000 + peak_pages, dtype=torch.int64, device=cpu_device
    )
    blob_bytes = gqa_geom.cold_blob_bytes(QuantPreset.INT4_SYM, RoPEMode.PRESERVE_FP8)
    out_pinned_ring = torch.empty(
        (peak_pages, blob_bytes), dtype=torch.uint8, device=cpu_device
    )

    # Warmup GCD worker pool & page tables
    for w_idx in range(5):
        cpp_codec.encode_cold_pages_peak_into(
            k_peak,
            v_peak,
            peak_hashes,
            out_pinned_ring,
            "perf-tenant",
            "perf-sess",
            dataclasses.replace(aad, sequence_nonce=50000 + w_idx * peak_pages),
        )
    # Full B4 verification on the actual Peak buffer: round-trip decode, fidelity, and MAC tamper rejection
    assert int(out_pinned_ring[0, 0]) == 0x54 and int(out_pinned_ring[0, 1]) == 0x5A
    k_pk0, v_pk0 = cpp_codec.decode_cold_page_tensor(
        out_pinned_ring[0],
        "perf-tenant",
        "perf-sess",
        "Qwen/Qwen3.6-35B-A3B",
        7,
        int(peak_hashes[0].item()),
    )
    k_pk_last, v_pk_last = cpp_codec.decode_cold_page_tensor(
        out_pinned_ring[-1],
        "perf-tenant",
        "perf-sess",
        "Qwen/Qwen3.6-35B-A3B",
        7,
        int(peak_hashes[-1].item()),
    )
    assert (
        evaluate_attention_fidelity(k_cpu, v_cpu, k_pk0, v_pk0)["attn_out_cos"] > 0.995
    )
    assert (
        evaluate_attention_fidelity(k_cpu, v_cpu, k_pk_last, v_pk_last)["attn_out_cos"]
        > 0.995
    )
    tampered_peak = out_pinned_ring[0].clone()
    tampered_peak[64] ^= 0x01
    try:
        cpp_codec.decode_cold_page_tensor(
            tampered_peak,
            "perf-tenant",
            "perf-sess",
            "Qwen/Qwen3.6-35B-A3B",
            7,
            int(peak_hashes[0].item()),
        )
        raise AssertionError("Tampered Peak ring page must fail MAC verification!")
    except KVIntegrityError:
        pass

    peak_iters = 50
    peak_aads = [
        dataclasses.replace(aad, sequence_nonce=100000 + i * peak_pages)
        for i in range(peak_iters)
    ]
    t0 = time.perf_counter()
    for i in range(peak_iters):
        cpp_codec.encode_cold_pages_peak_into(
            k_peak,
            v_peak,
            peak_hashes,
            out_pinned_ring,
            "perf-tenant",
            "perf-sess",
            peak_aads[i],
        )
    peak_total_s = (time.perf_counter() - t0) / peak_iters
    peak_ms = peak_total_s * 1000.0
    peak_per_page_us = (peak_total_s / peak_pages) * 1e6
    peak_raw_bytes = peak_pages * gqa_geom.hot_page_bytes
    peak_wire_bytes = peak_pages * blob_bytes
    peak_gbs = (peak_raw_bytes / 1e9) / peak_total_s
    peak_wire_gbs = (peak_wire_bytes / 1e9) / peak_total_s

    # Zero-Allocation Preallocated CPU Buffer Peak Ingress (256 Pages)
    out_k_peak_dec = torch.empty_like(k_peak)
    out_v_peak_dec = torch.empty_like(v_peak)
    for _ in range(3):
        cpp_codec.decode_cold_pages_peak_into(
            out_pinned_ring,
            peak_hashes,
            out_k_peak_dec,
            out_v_peak_dec,
            "perf-tenant",
            "perf-sess",
            "Qwen/Qwen3.6-35B-A3B",
            7,
        )
    t0 = time.perf_counter()
    for _ in range(peak_iters):
        cpp_codec.decode_cold_pages_peak_into(
            out_pinned_ring,
            peak_hashes,
            out_k_peak_dec,
            out_v_peak_dec,
            "perf-tenant",
            "perf-sess",
            "Qwen/Qwen3.6-35B-A3B",
            7,
        )
    peak_dec_total_s = (time.perf_counter() - t0) / peak_iters
    peak_dec_per_page_us = (peak_dec_total_s / peak_pages) * 1e6
    peak_dec_gbs = (peak_raw_bytes / 1e9) / peak_dec_total_s

    # Also measure Peak 256-page throughput for K8V4_ASYM + Preserve RoPE (FP8) so the K8V4 RDMA model uses measured CPU speed
    k8v4_blob_b = gqa_geom.cold_blob_bytes(
        QuantPreset.K8V4_ASYM, RoPEMode.PRESERVE_FP8
    )
    out_pinned_k8v4 = torch.empty(
        (peak_pages, k8v4_blob_b), dtype=torch.uint8, device=cpu_device
    )
    aad_k8v4_base = dataclasses.replace(aad, preset=QuantPreset.K8V4_ASYM)
    for w_idx in range(3):
        cpp_codec.encode_cold_pages_peak_into(
            k_peak,
            v_peak,
            peak_hashes,
            out_pinned_k8v4,
            "perf-tenant",
            "perf-sess",
            dataclasses.replace(aad_k8v4_base, sequence_nonce=300000 + w_idx * peak_pages),
        )
    k8v4_aads = [
        dataclasses.replace(aad_k8v4_base, sequence_nonce=310000 + i * peak_pages)
        for i in range(peak_iters)
    ]
    t0 = time.perf_counter()
    for i in range(peak_iters):
        cpp_codec.encode_cold_pages_peak_into(
            k_peak,
            v_peak,
            peak_hashes,
            out_pinned_k8v4,
            "perf-tenant",
            "perf-sess",
            k8v4_aads[i],
        )
    peak_k8v4_total_s = (time.perf_counter() - t0) / peak_iters
    peak_k8v4_ms = peak_k8v4_total_s * 1000.0
    peak_k8v4_per_page_us = (peak_k8v4_total_s / peak_pages) * 1e6
    peak_k8v4_gbs = (peak_raw_bytes / 1e9) / peak_k8v4_total_s
    peak_k8v4_wire_gbs = ((peak_pages * k8v4_blob_b) / 1e9) / peak_k8v4_total_s

    print(
        f"M3 Max Peak Prealloc CPU INT4 (256p) : Egress = {peak_ms:.2f} ms ({peak_per_page_us:.2f} us/page, {peak_gbs:.2f} GB/s hot, {peak_wire_gbs:.2f} GB/s wire) | "
        f"Ingress = {peak_dec_per_page_us:.2f} us/page ({peak_dec_gbs:.2f} GB/s hot)"
    )
    print(
        f"M3 Max Peak Prealloc CPU K8V4 (256p) : {peak_k8v4_ms:.2f} ms total ({peak_k8v4_per_page_us:.2f} us/page) | "
        f"Throughput = {peak_k8v4_gbs:.2f} GB/s hot ({peak_k8v4_wire_gbs:.2f} GB/s wire)"
    )

    # 2. Analytical Serial-Egress Distributed RDMA Transfer Model (64 Layers, 4K Token Prefix = 4,096 Pages)
    # Separates Empirically Measured CPU Fused Kernel from Hypothetical GPU SRAM (950 GB/s) Projection,
    # and uses consistent SI units (1 MB = 10^6 B, 1 GB/s = 10^9 B/s) alongside binary MiB.
    print("\n" + "=" * 106)
    print(
        "5. ANALYTICAL SERIAL-EGRESS RDMA MODEL (4,096-Token Prefix across 64 Layers; 100GbE @ 11.2 GB/s Goodput)"
    )
    print("=" * 106)
    num_layers = 64
    pages_per_layer = 64  # 64 pages * 64 tokens = 4,096 tokens
    total_pages = num_layers * pages_per_layer
    int4_blob_b = blob_bytes

    raw_total_bytes = total_pages * gqa_geom.hot_page_bytes
    fp8_raw_total_bytes = total_pages * (gqa_geom.hot_page_bytes // 2)
    k8v4_total_bytes = total_pages * k8v4_blob_b
    int4_total_bytes = total_pages * int4_blob_b

    raw_total_mb_si = raw_total_bytes / 1e6
    raw_total_mib = raw_total_bytes / (1024.0 * 1024.0)
    fp8_total_mb_si = fp8_raw_total_bytes / 1e6
    fp8_total_mib = fp8_raw_total_bytes / (1024.0 * 1024.0)
    k8v4_total_mb_si = k8v4_total_bytes / 1e6
    k8v4_total_mib = k8v4_total_bytes / (1024.0 * 1024.0)
    int4_total_mb_si = int4_total_bytes / 1e6
    int4_total_mib = int4_total_bytes / (1024.0 * 1024.0)

    rdma_bw_gbs = 11.2  # 11.2 * 10^9 B/s effective RoCEv2 goodput (89.6% of 12.5 GB/s line rate)
    host_copy_bw_gbs = 28.0  # Assumed host temp-buffer copy + CPU AES-GCM rate (28.0 * 10^9 B/s)
    gpu_fused_bw_gbs = 950.0  # Hypothetical GPU SRAM single-pass fused throughput (950.0 * 10^9 B/s)

    t_raw_ms = (raw_total_bytes / (rdma_bw_gbs * 1e9)) * 1000.0
    t_fp8_raw_ms = (fp8_raw_total_bytes / (rdma_bw_gbs * 1e9)) * 1000.0
    t_k8v4_wire_ms = (k8v4_total_bytes / (rdma_bw_gbs * 1e9)) * 1000.0
    t_int4_wire_ms = (int4_total_bytes / (rdma_bw_gbs * 1e9)) * 1000.0
    t_host_stage_ms = (raw_total_bytes / (host_copy_bw_gbs * 1e9)) * 1000.0
    t_discrete_ms = t_host_stage_ms + t_k8v4_wire_ms

    # Empirically Measured M3 Max CPU Peak Fused Compute + Analytical 100GbE Wire Transfer (Serial Non-Overlapped)
    t_cpu_k8v4_compute_ms = (raw_total_bytes / (peak_k8v4_gbs * 1e9)) * 1000.0
    t_cpu_int4_compute_ms = (raw_total_bytes / (peak_gbs * 1e9)) * 1000.0
    t_ztcp_k8v4_cpu_ms = t_cpu_k8v4_compute_ms + t_k8v4_wire_ms
    t_ztcp_int4_cpu_ms = t_cpu_int4_compute_ms + t_int4_wire_ms

    # Hypothetical GPU SRAM (950 GB/s) Compute + Analytical 100GbE Wire Transfer (Serial Non-Overlapped)
    t_gpu_compute_ms = (raw_total_bytes / (gpu_fused_bw_gbs * 1e9)) * 1000.0
    t_ztcp_k8v4_gpu_ms = t_gpu_compute_ms + t_k8v4_wire_ms
    t_ztcp_int4_gpu_ms = t_gpu_compute_ms + t_int4_wire_ms

    print(
        f"{'Pipeline Path (100GbE RoCEv2 @ 11.2 GB/s)':<48} | {'Wire (SI / Binary)':>22} | {'Compute + Wire':>16} | {'Speedup':>10}"
    )
    print("-" * 106)
    print(
        f"{'1. Raw BF16 Baseline (Unencrypted)':<48} | {raw_total_mb_si:>7.2f} MB ({raw_total_mib:>6.1f} MiB) | {t_raw_ms:>13.2f} ms | {'1.00x':>10}"
    )
    print(
        f"{'2. Discrete Host Copy (28 GB/s) + K8V4 Wire':<48} | {k8v4_total_mb_si:>7.2f} MB ({k8v4_total_mib:>6.1f} MiB) | {t_discrete_ms:>13.2f} ms | {t_raw_ms / t_discrete_ms:>9.2f}x"
    )
    print(
        f"{'3. Raw FP8 Baseline (Unencrypted, Hot Loss)':<48} | {fp8_total_mb_si:>7.2f} MB ({fp8_total_mib:>6.1f} MiB) | {t_fp8_raw_ms:>13.2f} ms | {t_raw_ms / t_fp8_raw_ms:>9.2f}x"
    )
    print(
        f"{'4a. ZTCP K8V4 (Measured M3 Max CPU + 100GbE)':<48} | {k8v4_total_mb_si:>7.2f} MB ({k8v4_total_mib:>6.1f} MiB) | {t_ztcp_k8v4_cpu_ms:>13.2f} ms | {t_raw_ms / t_ztcp_k8v4_cpu_ms:>9.2f}x"
    )
    print(
        f"{'5a. ZTCP INT4 (Measured M3 Max CPU + 100GbE)':<48} | {int4_total_mb_si:>7.2f} MB ({int4_total_mib:>6.1f} MiB) | {t_ztcp_int4_cpu_ms:>13.2f} ms | {t_raw_ms / t_ztcp_int4_cpu_ms:>9.2f}x"
    )
    print(
        f"{'4b. ZTCP K8V4 (Projected GPU 950 GB/s + 100GbE)':<48} | {k8v4_total_mb_si:>7.2f} MB ({k8v4_total_mib:>6.1f} MiB) | {t_ztcp_k8v4_gpu_ms:>13.2f} ms | {t_raw_ms / t_ztcp_k8v4_gpu_ms:>9.2f}x"
    )
    print(
        f"{'5b. ZTCP INT4 (Projected GPU 950 GB/s + 100GbE)':<48} | {int4_total_mb_si:>7.2f} MB ({int4_total_mib:>6.1f} MiB) | {t_ztcp_int4_gpu_ms:>13.2f} ms | {t_raw_ms / t_ztcp_int4_gpu_ms:>9.2f}x"
    )

    return {
        "run_index": run_index,
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "synthetic_fidelity_summary": fidelity_summary,
        "microbenchmark_gqa_256kb_page": {
            "hot_page_bytes": gqa_geom.hot_page_bytes,
            "cold_blob_bytes_int4_rope_fp8": cold_blob_bytes,
            "cold_blob_bytes_k8v4_rope_fp8": k8v4_blob_b,
            "compression_ratio": round(
                gqa_geom.compression_ratio(QuantPreset.INT4_SYM, RoPEMode.PRESERVE_FP8),
                4,
            ),
            "compression_ratio_k8v4": round(
                gqa_geom.compression_ratio(QuantPreset.K8V4_ASYM, RoPEMode.PRESERVE_FP8),
                4,
            ),
            "py_reference_attention_cosine_similarity": fidelity_summary[
                "gqa_standard"
            ]["INT4_SYM + Preserve RoPE (FP8)"]["py_attn_out_cos"],
            "cpp_attention_cosine_similarity": round(cpp_fid["attn_out_cos"], 6),
            "python_multi_op_egress_us": round(py_enc_us, 2),
            "python_multi_op_ingress_us": round(py_dec_us, 2),
            "python_multi_op_egress_gbs": round(py_enc_gbs, 2),
            "python_multi_op_egress_wire_gbs": round(py_enc_wire_gbs, 2),
            "cpp_single_core_egress_us": round(cpp_enc_us, 2),
            "cpp_single_core_ingress_us": round(cpp_dec_us, 2),
            "cpp_single_core_egress_speedup_vs_python": round(cpp_1c_egress_speedup, 2),
            "cpp_single_core_speedup": round(cpp_1c_roundtrip_speedup, 2),
            "cpp_single_core_egress_gbs": round(cpp_enc_gbs, 2),
            "cpp_single_core_egress_wire_gbs": round(cpp_enc_wire_gbs, 2),
            "cpp_single_core_ingress_gbs": round(cpp_dec_gbs, 2),
            "cpp_multicore_batch64_total_ms": round(batch_ms, 3),
            "cpp_multicore_batch64_per_page_us": round(batch_per_page_us, 2),
            "cpp_multicore_batch64_egress_gbs": round(batch_gbs, 2),
            "cpp_multicore_batch64_egress_wire_gbs": round(batch_wire_gbs, 2),
            "cpp_multicore_batch64_ingress_per_page_us": round(batch_dec_per_page_us, 2),
            "cpp_multicore_batch64_ingress_gbs": round(batch_dec_gbs, 2),
            "cpp_multicore_speedup_vs_python": round(
                py_enc_us / batch_per_page_us, 2
            ),
            "cpp_multicore_speedup_vs_1core_cpp": round(
                cpp_enc_us / batch_per_page_us, 2
            ),
            "cpp_peak_zero_alloc_batch256_total_ms": round(peak_ms, 3),
            "cpp_peak_zero_alloc_per_page_us": round(peak_per_page_us, 2),
            "cpp_peak_zero_alloc_egress_gbs": round(peak_gbs, 2),
            "cpp_peak_zero_alloc_egress_wire_gbs": round(peak_wire_gbs, 2),
            "cpp_peak_zero_alloc_ingress_per_page_us": round(peak_dec_per_page_us, 2),
            "cpp_peak_zero_alloc_ingress_gbs": round(peak_dec_gbs, 2),
            "cpp_peak_zero_alloc_speedup_vs_python": round(
                py_enc_us / peak_per_page_us, 2
            ),
            "cpp_peak_zero_alloc_speedup_vs_1core_cpp": round(
                cpp_enc_us / peak_per_page_us, 2
            ),
            "cpp_peak_zero_alloc_k8v4_batch256_total_ms": round(peak_k8v4_ms, 3),
            "cpp_peak_zero_alloc_k8v4_per_page_us": round(peak_k8v4_per_page_us, 2),
            "cpp_peak_zero_alloc_k8v4_egress_gbs": round(peak_k8v4_gbs, 2),
            "cpp_peak_zero_alloc_k8v4_egress_wire_gbs": round(
                peak_k8v4_wire_gbs, 2
            ),
        },
        "rdma_100gbe_4k_prefix_64layers_model": {
            "rdma_effective_goodput_gbs_si": rdma_bw_gbs,
            "assumed_host_copy_crypto_gbs_si": host_copy_bw_gbs,
            "projected_gpu_sram_fused_gbs_si": gpu_fused_bw_gbs,
            "raw_bf16_bytes": raw_total_bytes,
            "raw_bf16_mb_si": round(raw_total_mb_si, 2),
            "raw_bf16_mib": round(raw_total_mib, 2),
            "raw_bf16_ms": round(t_raw_ms, 2),
            "raw_fp8_bytes": fp8_raw_total_bytes,
            "raw_fp8_mb_si": round(fp8_total_mb_si, 2),
            "raw_fp8_mib": round(fp8_total_mib, 2),
            "raw_fp8_ms": round(t_fp8_raw_ms, 2),
            "discrete_host_copy_ms": round(t_discrete_ms, 2),
            "ztcp_fused_k8v4_bytes": k8v4_total_bytes,
            "ztcp_fused_k8v4_mb_si": round(k8v4_total_mb_si, 2),
            "ztcp_fused_k8v4_mib": round(k8v4_total_mib, 2),
            "ztcp_fused_k8v4_wire_only_ms": round(t_k8v4_wire_ms, 2),
            "ztcp_fused_k8v4_cpu_compute_ms": round(t_cpu_k8v4_compute_ms, 2),
            "ztcp_fused_k8v4_cpu_measured_total_ms": round(t_ztcp_k8v4_cpu_ms, 2),
            "ztcp_fused_k8v4_cpu_speedup_vs_raw_bf16": round(
                t_raw_ms / t_ztcp_k8v4_cpu_ms, 2
            ),
            "ztcp_fused_k8v4_gpu_projected_total_ms": round(t_ztcp_k8v4_gpu_ms, 2),
            "ztcp_fused_k8v4_gpu_speedup_vs_raw_bf16": round(
                t_raw_ms / t_ztcp_k8v4_gpu_ms, 2
            ),
            "ztcp_fused_int4_bytes": int4_total_bytes,
            "ztcp_fused_int4_mb_si": round(int4_total_mb_si, 2),
            "ztcp_fused_int4_mib": round(int4_total_mib, 2),
            "ztcp_fused_int4_wire_only_ms": round(t_int4_wire_ms, 2),
            "ztcp_fused_int4_cpu_compute_ms": round(t_cpu_int4_compute_ms, 2),
            "ztcp_fused_int4_cpu_measured_total_ms": round(t_ztcp_int4_cpu_ms, 2),
            "ztcp_fused_int4_cpu_speedup_vs_raw_bf16": round(
                t_raw_ms / t_ztcp_int4_cpu_ms, 2
            ),
            "ztcp_fused_int4_gpu_projected_total_ms": round(t_ztcp_int4_gpu_ms, 2),
            "ztcp_fused_int4_gpu_speedup_vs_raw_bf16": round(
                t_raw_ms / t_ztcp_int4_gpu_ms, 2
            ),
        },
        "security_tests": {
            "valid_aad_roundtrip": "PASS",
            "cross_tenant_contamination_blocked": "PASS",
            "remote_rdma_1bit_poisoning_blocked": "PASS",
            "instant_crypto_shredding_and_rederive_verified": "PASS",
            "native_peak_roundtrip_and_reread_verified": "PASS",
            "native_peak_scale_tamper_blocked": "PASS",
            "native_peak_nope_block_swap_blocked": "PASS",
            "native_peak_paired_v_bitflip_blocked": "PASS",
            "native_peak_aad_contamination_blocked": "PASS",
            "bijective_nonce_and_stale_replay_blocked": "PASS",
            "transactional_concurrency_code003_verified": "PASS",
        },
    }


def aggregate_independent_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("runs list must be non-empty")
    first = runs[0]
    mb_keys = [
        ("python_multi_op_egress_us", 2),
        ("python_multi_op_ingress_us", 2),
        ("python_multi_op_egress_gbs", 2),
        ("python_multi_op_egress_wire_gbs", 2),
        ("cpp_single_core_egress_us", 2),
        ("cpp_single_core_ingress_us", 2),
        ("cpp_single_core_egress_speedup_vs_python", 2),
        ("cpp_single_core_speedup", 2),
        ("cpp_single_core_egress_gbs", 2),
        ("cpp_single_core_egress_wire_gbs", 2),
        ("cpp_single_core_ingress_gbs", 2),
        ("cpp_multicore_batch64_total_ms", 3),
        ("cpp_multicore_batch64_per_page_us", 2),
        ("cpp_multicore_batch64_egress_gbs", 2),
        ("cpp_multicore_batch64_egress_wire_gbs", 2),
        ("cpp_multicore_batch64_ingress_per_page_us", 2),
        ("cpp_multicore_batch64_ingress_gbs", 2),
        ("cpp_multicore_speedup_vs_python", 2),
        ("cpp_multicore_speedup_vs_1core_cpp", 2),
        ("cpp_peak_zero_alloc_batch256_total_ms", 3),
        ("cpp_peak_zero_alloc_per_page_us", 2),
        ("cpp_peak_zero_alloc_egress_gbs", 2),
        ("cpp_peak_zero_alloc_egress_wire_gbs", 2),
        ("cpp_peak_zero_alloc_ingress_per_page_us", 2),
        ("cpp_peak_zero_alloc_ingress_gbs", 2),
        ("cpp_peak_zero_alloc_speedup_vs_python", 2),
        ("cpp_peak_zero_alloc_speedup_vs_1core_cpp", 2),
        ("cpp_peak_zero_alloc_k8v4_batch256_total_ms", 3),
        ("cpp_peak_zero_alloc_k8v4_per_page_us", 2),
        ("cpp_peak_zero_alloc_k8v4_egress_gbs", 2),
        ("cpp_peak_zero_alloc_k8v4_egress_wire_gbs", 2),
    ]
    agg_stats: dict[str, dict[str, float]] = {}
    mb_summary = dict(first["microbenchmark_gqa_256kb_page"])
    for key, nd in mb_keys:
        vals = [float(r["microbenchmark_gqa_256kb_page"][key]) for r in runs]
        st = _stat_summary(vals, nd)
        agg_stats[key] = st
        mb_summary[key] = st["mean"]
        mb_summary[f"{key}_std"] = st["std"]
        mb_summary[f"{key}_median"] = st["median"]
        mb_summary[f"{key}_min"] = st["min"]
        mb_summary[f"{key}_max"] = st["max"]

    rdma_keys = [
        ("ztcp_fused_k8v4_cpu_compute_ms", 2),
        ("ztcp_fused_k8v4_cpu_measured_total_ms", 2),
        ("ztcp_fused_k8v4_cpu_speedup_vs_raw_bf16", 2),
        ("ztcp_fused_int4_cpu_compute_ms", 2),
        ("ztcp_fused_int4_cpu_measured_total_ms", 2),
        ("ztcp_fused_int4_cpu_speedup_vs_raw_bf16", 2),
    ]
    rdma_summary = dict(first["rdma_100gbe_4k_prefix_64layers_model"])
    for key, nd in rdma_keys:
        vals = [float(r["rdma_100gbe_4k_prefix_64layers_model"][key]) for r in runs]
        st = _stat_summary(vals, nd)
        agg_stats[key] = st
        rdma_summary[key] = st["mean"]
        rdma_summary[f"{key}_std"] = st["std"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": first["platform"],
        "processor": first["processor"],
        "python_version": first["python_version"],
        "torch_version": first["torch_version"],
        "num_threads": first["num_threads"],
        "num_independent_runs": len(runs),
        "metric_and_methodology_notes": {
            "gqa_vs_mla_attention_metric": (
                "In synthetic_fidelity_summary, gqa_standard (where V is present) computes "
                "simulated attention-weighted Value output cosine similarity "
                "cos(softmax(QK^T/sqrt(d))V, softmax(QK_hat^T/sqrt(d))V_hat), whereas "
                "single-buffer deepseek_mla (where V is None) computes softmax attention-weights "
                "cosine similarity cos(softmax(QK^T/sqrt(d)), softmax(QK_hat^T/sqrt(d))). Both are "
                "stored under py_attn_out_cos / cpp_attn_out_cos for schema uniformity and tagged "
                "via attn_metric_kind."
            ),
            "kv_and_query_rng_seeds": (
                "generate_realistic_kv_page always resets torch.manual_seed(42) on its target "
                "device, and evaluate_attention_fidelity always resets torch.manual_seed(99) on "
                "k_ref.device when sampling query q. Section 1 synthesizes the seed=42 KV page on "
                "device='mps' (when available) before copying to CPU (cpp_attn_out_cos=0.996103), "
                "whereas Section 4 synthesizes the seed=42 KV page directly on device='cpu' "
                "(cpp_attention_cosine_similarity=0.995925); there is no second independent KV seed=99 sample."
            ),
            "quantizer_precision_and_device_query_note": (
                "FusedColdPageCodec (Python) is mixed-precision (FP16 input/output tensors and FP16 "
                "stored group scales with FP32 intermediate group division). NativeCppFusedColdPageCodec "
                "(C++17 ARMv8.5-A NEON, csrc/fused_cold_page_cpu.cpp) computes FP16 group "
                "abs-max (vmaxvq_f16), widens to FP32 for scale and exact IEEE-754 reciprocal division "
                "(vectorized vdivq_f32 in 4-group kernels; scalar 1.0f / static_cast<float>(sc) in 2-group "
                "fallback), narrows the reciprocal to FP16 (vcvt_f16_f32 / vdupq_n_f16), and executes "
                "vectorized FP16 multiplication and nearest-even rounding (vmulq_f16, vcvtnq_s16_f16). "
                "Because Section 1 evaluates Python on MPS and C++ on CPU, query q (seed=99) is sampled "
                "from device-specific RNG streams."
            ),
            "tier_progression_and_buffer_scope": (
                f"Speedup ratios ({agg_stats['cpp_single_core_egress_speedup_vs_python']['mean']:.2f}x, "
                f"{agg_stats['cpp_multicore_speedup_vs_1core_cpp']['mean']:.2f}x, "
                f"{agg_stats['cpp_peak_zero_alloc_egress_gbs']['mean'] / agg_stats['cpp_multicore_batch64_egress_gbs']['mean']:.2f}x, "
                f"{agg_stats['cpp_peak_zero_alloc_speedup_vs_1core_cpp']['mean']:.2f}x, "
                f"{agg_stats['cpp_peak_zero_alloc_speedup_vs_python']['mean']:.2f}x) are descriptive incremental "
                "implementation-tier comparisons across compounded changes (language/runtime, quantizer "
                "rounding, BLAKE2b vs. SipHash-2-4-128 MAC backend, batch size 1->64->256, threading "
                "runtime, and per-call allocation vs. caller-preallocated CPU uint8 tensor buffer), not "
                "controlled single-factor causal ablations isolating register fusion or NIC RDMA registration."
            ),
            "int4_cpu_rdma_speedup_estimator_note": (
                f"rdma_100gbe_4k_prefix_64layers_model.ztcp_fused_int4_cpu_speedup_vs_raw_bf16="
                f"{agg_stats['ztcp_fused_int4_cpu_speedup_vs_raw_bf16']['mean']:.2f} "
                f"(std={agg_stats['ztcp_fused_int4_cpu_speedup_vs_raw_bf16']['std']:.2f}) is the mean of the 5 per-run "
                f"2-decimal rounded ratios {[r['rdma_100gbe_4k_prefix_64layers_model']['ztcp_fused_int4_cpu_speedup_vs_raw_bf16'] for r in runs]}, "
                f"whereas the ratio of the 5-run mean latencies ({rdma_summary['raw_bf16_ms']:.2f} ms / "
                f"{rdma_summary['ztcp_fused_int4_cpu_measured_total_ms']:.2f} ms = "
                f"{rdma_summary['raw_bf16_ms'] / rdma_summary['ztcp_fused_int4_cpu_measured_total_ms']:.4f}x) rounds to "
                f"{rdma_summary['raw_bf16_ms'] / rdma_summary['ztcp_fused_int4_cpu_measured_total_ms']:.2f}x."
            ),
            "rdma_and_security_test_scope": (
                "Section 5 is an analytical serial-egress latency model combining measured M3 Max CPU "
                "compute time (or hypothetical 950 GB/s GPU SRAM compute time) with analytical 100GbE "
                "wire time at 11.2 GB/s. In the bundled benchmark (run_security_verification), tests "
                "3.1-3.4 run on the Python reference codec (with 3.3 simulating a 1-bit flip in local "
                "memory), 3.5-3.9 and 3.11 run on the native C++17 Peak path + EphemeralSessionKeyRing, "
                "3.10a-c exercises Native Peak duplicate-nonce and stale-replay checks plus the Python "
                "pack_chacha20_nonce helper, and 3.12 exercises Native Peak same-nonce race + Native Single "
                "failed-encode nonce burn and monotonic watermark checks. In-flight egress and ingress shred "
                "checks across all four codec paths (py_single, cpp_single, cpp_batch, cpp_peak) are verified "
                "in verify_package2.py, CODE-003; overlapping-batch checks are verified in B6-2."
            ),
        },
        "synthetic_fidelity_summary": first["synthetic_fidelity_summary"],
        "microbenchmark_gqa_256kb_page": mb_summary,
        "rdma_100gbe_4k_prefix_64layers_model": rdma_summary,
        "aggregated_5run_stats": agg_stats,
        "security_tests": first["security_tests"],
        "independent_runs": runs,
    }


def print_multi_run_summary(payload: dict) -> None:
    runs = payload["independent_runs"]
    stats = payload["aggregated_5run_stats"]
    print("\n" + "=" * 112)
    print(
        f"MULTI-RUN STATISTICAL SUMMARY (N={len(runs)} Independent OS Process Runs x 50 Batch/Peak Iterations)"
    )
    print("=" * 112)
    print(
        f"{'Run':<10} | {'PID':>7} | {'1-Core Enc (us)':>15} | {'1-Core (GB/s)':>13} | "
        f"{'Batch 64p (GB/s)':>16} | {'Peak 256p (ms)':>14} | {'Peak Hot (GB/s)':>15} | {'Peak Wire (GB/s)':>16}"
    )
    print("-" * 112)
    for r in runs:
        mb = r["microbenchmark_gqa_256kb_page"]
        print(
            f"{'Run ' + str(r['run_index']):<10} | {r['pid']:>7} | "
            f"{mb['cpp_single_core_egress_us']:>12.2f} us | "
            f"{mb['cpp_single_core_egress_gbs']:>10.2f} GB/s | "
            f"{mb['cpp_multicore_batch64_egress_gbs']:>13.2f} GB/s | "
            f"{mb['cpp_peak_zero_alloc_batch256_total_ms']:>11.3f} ms | "
            f"{mb['cpp_peak_zero_alloc_egress_gbs']:>12.2f} GB/s | "
            f"{mb['cpp_peak_zero_alloc_egress_wire_gbs']:>13.2f} GB/s"
        )
    print("-" * 112)
    s_1c_us = stats["cpp_single_core_egress_us"]
    s_1c_gbs = stats["cpp_single_core_egress_gbs"]
    s_b64_gbs = stats["cpp_multicore_batch64_egress_gbs"]
    s_pk_ms = stats["cpp_peak_zero_alloc_batch256_total_ms"]
    s_pk_hot = stats["cpp_peak_zero_alloc_egress_gbs"]
    s_pk_wire = stats["cpp_peak_zero_alloc_egress_wire_gbs"]
    print(
        f"{'Mean±Std':<10} | {'-':>7} | "
        f"{s_1c_us['mean']:>5.2f}±{s_1c_us['std']:<5.2f} us | "
        f"{s_1c_gbs['mean']:>4.2f}±{s_1c_gbs['std']:<4.2f} GB/s | "
        f"{s_b64_gbs['mean']:>6.2f}±{s_b64_gbs['std']:<5.2f} GB/s | "
        f"{s_pk_ms['mean']:>5.3f}±{s_pk_ms['std']:<5.3f} ms | "
        f"{s_pk_hot['mean']:>5.2f}±{s_pk_hot['std']:<5.2f} GB/s | "
        f"{s_pk_wire['mean']:>6.2f}±{s_pk_wire['std']:<5.2f} GB/s"
    )
    print(
        f"{'Median':<10} | {'-':>7} | "
        f"{s_1c_us['median']:>12.2f} us | "
        f"{s_1c_gbs['median']:>10.2f} GB/s | "
        f"{s_b64_gbs['median']:>13.2f} GB/s | "
        f"{s_pk_ms['median']:>11.3f} ms | "
        f"{s_pk_hot['median']:>12.2f} GB/s | "
        f"{s_pk_wire['median']:>13.2f} GB/s"
    )
    print(
        f"{'Min-Max':<10} | {'-':>7} | "
        f"{s_1c_us['min']:.1f}-{s_1c_us['max']:.1f} us | "
        f"{s_1c_gbs['min']:.2f}-{s_1c_gbs['max']:.2f} | "
        f"{s_b64_gbs['min']:.2f}-{s_b64_gbs['max']:.2f} | "
        f"{s_pk_ms['min']:.3f}-{s_pk_ms['max']:.3f} | "
        f"{s_pk_hot['min']:.2f}-{s_pk_hot['max']:.2f} | "
        f"{s_pk_wire['min']:.2f}-{s_pk_wire['max']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZTCP Reproducible Multi-Run Benchmark & Security Suite"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent OS process runs to execute and aggregate (default: 5)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Execute a single benchmark run in the current process",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="Run index label when invoked with --single-run",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON path for --single-run or aggregated output",
    )
    parser.add_argument(
        "--reaggregate-only",
        action="store_true",
        help="Recompute aggregated statistics directly from independent_runs in latest_results.json",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    default_json_path = bench_dir / "latest_results.json"
    out_path = Path(args.output_json) if args.output_json else default_json_path

    if args.reaggregate_only:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        runs = existing.get("independent_runs", [])
        payload = aggregate_independent_runs(runs)
        print_multi_run_summary(payload)
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[RE-AGGREGATED] Reproducible {len(runs)}-run statistics written to: {out_path}")
        return

    if args.single_run or args.runs == 1:
        single_payload = run_single_process_benchmark(run_index=args.run_index)
        if args.single_run and args.output_json:
            out_path.write_text(
                json.dumps(single_payload, indent=2) + "\n", encoding="utf-8"
            )
            return
        agg_payload = aggregate_independent_runs([single_payload])
        out_path.write_text(
            json.dumps(agg_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[SAVED] Structured benchmark results written to: {out_path}")
        return

    # Ensure C++17 extension is compiled before spawning N independent child processes
    from research.baseline_codec.native_fused_codec import get_native_fused_module

    _ = get_native_fused_module()

    runs_list: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ztcp_bench_runs_") as tmp_dir:
        for r_idx in range(1, args.runs + 1):
            run_file = Path(tmp_dir) / f"run_{r_idx}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-run",
                "--run-index",
                str(r_idx),
                "--output-json",
                str(run_file),
            ]
            subprocess.run(cmd, check=True)
            run_data = json.loads(run_file.read_text(encoding="utf-8"))
            runs_list.append(run_data)

    results_payload = aggregate_independent_runs(runs_list)
    print_multi_run_summary(results_payload)
    out_path.write_text(
        json.dumps(results_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[SAVED] Structured {args.runs}-run paper benchmark results written to: {out_path}")


if __name__ == "__main__":
    main()
