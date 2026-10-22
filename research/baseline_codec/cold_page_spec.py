# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hakan Damar and Theseus Guard, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Developed by Theseus Guard Research (https://www.theseusguard.com)
# Repository: https://github.com/Theseus-Guard/ZTCP
# Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
"""Compact Cold-Page Blob Layout and Geometry Specification.

Developed by Theseus Guard, Inc. as part of the Zero-Trust Cold Pages (ZTCP)
research architecture. Defines the 64-byte-aligned wire/RDMA format for compressed
and authenticated KV pages migrating between active GPU HBM and untrusted Host DRAM,
NVMe Disk, or cross-node RDMA pools (Mooncake, LMCache, SGLang HiCache).

Wire Layout:
    +-------------------------------------------------------------------+
    | 1. HEADER / AAD (32 bytes, little-endian, authenticated)          |
    |    0..2   : magic (0x5A54 = "ZT")                                 |
    |    2..3   : version (uint8 = 1)                                   |
    |    3..4   : flags (bits 0..3: QuantPreset, bits 4..5: RoPEMode)   |
    |    4..8   : tenant_id_hash (uint32)                               |
    |    8..12  : model_id_hash (uint32)                                |
    |    12..14 : layer_id (uint16)                                     |
    |    14..16 : reserved / head_group_id (uint16)                     |
    |    16..24 : page_token_hash (uint64)                              |
    |    24..32 : sequence_nonce (uint64, monotonic per session)        |
    +-------------------------------------------------------------------+
    | 2. ENCRYPTED COMPACT PAYLOAD (variable bytes, padded to 4B word)  |
    |    - [Optional] Preserved RoPE slice (FP8 E4M3 or BF16)           |
    |    - Packed quantized K/V payload (INT4 / K8V4)                   |
    |    - Per-group E4M3/FP8 scales                                    |
    +-------------------------------------------------------------------+
    | 3. AUTHENTICATION TAG / MAC (16 bytes, 128-bit)                   |
    |    - Keyed MAC over (32B Header AAD || Encrypted Payload)         |
    +-------------------------------------------------------------------+
"""

from __future__ import annotations

import enum
import hashlib
import struct
from dataclasses import dataclass

MAGIC_ZTCP = 0x5A54  # "ZT"
CODEC_VERSION = 1
HEADER_BYTES = 32
TAG_BYTES = 16
HEADER_STRUCT = struct.Struct("<HBBIIHHQQ")


class QuantPreset(enum.IntEnum):
    """Cold-page quantization formats."""

    INT4_SYM = 1  # Both K and V symmetric 4-bit + FP8/FP16 group scales
    K8V4_ASYM = 2  # Key in 8-bit (FP8/INT8), Value in 4-bit + group scales
    FP8_ONLY = 3  # Both K and V in 8-bit (2x baseline compression)


class RoPEMode(enum.IntEnum):
    """Handling for rotary position embedding (RoPE) dimensions in K cache."""

    QUANTIZE_ALL = 0  # Quantize RoPE dimensions along with NoPE
    PRESERVE_FP8 = 1  # Keep RoPE sub-vector in 8-bit while NoPE uses 4-bit
    PRESERVE_RAW = 2  # Keep RoPE sub-vector in original hot dtype (BF16/FP16)


def hash32(text: str) -> int:
    """Deterministic 32-bit domain identifier hash."""
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def hash64_tokens(token_ids: list[int] | tuple[int, ...]) -> int:
    """Deterministic 64-bit hash of token IDs in a KV page (prefix identity)."""
    buf = struct.pack(f"<{len(token_ids)}i", *token_ids)
    digest = hashlib.blake2s(buf, digest_size=8).digest()
    return int.from_bytes(digest, "little")


@dataclass(frozen=True)
class ColdPageAAD:
    """Associated Data (AAD) bound to every cold KV page's cryptographic tag."""

    tenant_id_hash: int
    model_id_hash: int
    layer_id: int
    page_token_hash: int
    sequence_nonce: int
    preset: QuantPreset = QuantPreset.K8V4_ASYM
    rope_mode: RoPEMode = RoPEMode.PRESERVE_FP8
    head_group_id: int = 0

    @classmethod
    def from_context(
        cls,
        tenant_id: str,
        model_id: str,
        layer_id: int,
        token_ids: list[int] | tuple[int, ...],
        sequence_nonce: int,
        preset: QuantPreset = QuantPreset.K8V4_ASYM,
        rope_mode: RoPEMode = RoPEMode.PRESERVE_FP8,
    ) -> ColdPageAAD:
        return cls(
            tenant_id_hash=hash32(tenant_id),
            model_id_hash=hash32(model_id),
            layer_id=layer_id,
            page_token_hash=hash64_tokens(token_ids),
            sequence_nonce=sequence_nonce,
            preset=preset,
            rope_mode=rope_mode,
        )

    def pack_header(self) -> bytes:
        flags = (int(self.preset) & 0x0F) | ((int(self.rope_mode) & 0x0F) << 4)
        return HEADER_STRUCT.pack(
            MAGIC_ZTCP,
            CODEC_VERSION,
            flags,
            self.tenant_id_hash & 0xFFFFFFFF,
            self.model_id_hash & 0xFFFFFFFF,
            self.layer_id & 0xFFFF,
            self.head_group_id & 0xFFFF,
            self.page_token_hash & 0xFFFFFFFFFFFFFFFF,
            self.sequence_nonce & 0xFFFFFFFFFFFFFFFF,
        )

    @classmethod
    def unpack_header(cls, header_bytes: bytes) -> ColdPageAAD:
        if len(header_bytes) < HEADER_BYTES:
            raise ValueError(f"Header too short: {len(header_bytes)} < {HEADER_BYTES}")
        (
            magic,
            version,
            flags,
            tenant_hash,
            model_hash,
            layer_id,
            head_group_id,
            page_token_hash,
            sequence_nonce,
        ) = HEADER_STRUCT.unpack(header_bytes[:HEADER_BYTES])
        if magic != MAGIC_ZTCP or version != CODEC_VERSION:
            raise ValueError(
                f"Invalid cold-page header magic/version: {hex(magic)} v{version}"
            )
        preset = QuantPreset(flags & 0x0F)
        rope_mode = RoPEMode((flags >> 4) & 0x0F)
        return cls(
            tenant_id_hash=tenant_hash,
            model_id_hash=model_hash,
            layer_id=layer_id,
            page_token_hash=page_token_hash,
            sequence_nonce=sequence_nonce,
            preset=preset,
            rope_mode=rope_mode,
            head_group_id=head_group_id,
        )


@dataclass(frozen=True)
class ColdPageGeometry:
    """Describes the shape and byte footprint of one KV page (hot vs. cold)."""

    num_tokens: int = 64  # Tokens per page (block size)
    num_kv_heads: int = 8  # GQA KV heads per TP rank
    head_dim: int = 128  # Total head dimension (e.g., 128 for GQA, 576 for MLA)
    rope_dim: int = 64  # Leading/trailing RoPE dimension inside K head_dim
    group_size: int = 16  # Elements per quantization scale block (NVFP4 standard = 16)
    hot_element_bytes: int = 2  # 2 for BF16/FP16, 1 for FP8
    is_mla_single_buffer: bool = False  # True for DeepSeek MLA latent KV

    def __post_init__(self) -> None:
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")
        if self.rope_dim < 0 or self.rope_dim > self.head_dim:
            raise ValueError("Invalid rope_dim")
        if self.rope_dim % self.group_size != 0:
            raise ValueError("rope_dim must be divisible by group_size")

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def vectors_per_buffer(self) -> int:
        return self.num_tokens * self.num_kv_heads

    @property
    def hot_page_bytes(self) -> int:
        buffers = 1 if self.is_mla_single_buffer else 2
        return (
            buffers
            * self.num_tokens
            * self.num_kv_heads
            * self.head_dim
            * self.hot_element_bytes
        )

    def payload_bytes(self, preset: QuantPreset, rope_mode: RoPEMode) -> int:
        """Compute exact compressed payload bytes (excluding 32B header and 16B tag)."""
        vecs = self.vectors_per_buffer
        gs = self.group_size

        # 1. Key buffer footprint
        if rope_mode == RoPEMode.PRESERVE_RAW and self.rope_dim > 0:
            k_rope_bytes = vecs * self.rope_dim * 2  # 2B (FP16/BF16)
            k_quant_dim = self.nope_dim
        elif rope_mode == RoPEMode.PRESERVE_FP8 and self.rope_dim > 0:
            k_rope_bytes = vecs * self.rope_dim * 1  # 1B (FP8/INT8) + 1B scale per group
            k_rope_Scale_bytes = vecs * (self.rope_dim // gs) * 2
            k_rope_bytes += k_rope_Scale_bytes
            k_quant_dim = self.nope_dim
        else:
            k_rope_bytes = 0
            k_quant_dim = self.head_dim

        if preset == QuantPreset.INT4_SYM:
            k_nope_data = vecs * (k_quant_dim // 2)
            k_nope_scales = vecs * (k_quant_dim // gs) * 2  # FP16/BF16 scale per group
        elif preset in (QuantPreset.K8V4_ASYM, QuantPreset.FP8_ONLY):
            k_nope_data = vecs * k_quant_dim
            k_nope_scales = vecs * (k_quant_dim // gs) * 2
        else:
            raise ValueError(f"Unsupported preset: {preset}")

        k_total = k_rope_bytes + k_nope_data + k_nope_scales

        if self.is_mla_single_buffer:
            raw_payload = k_total
        else:
            # 2. Value buffer footprint
            if preset in (QuantPreset.INT4_SYM, QuantPreset.K8V4_ASYM):
                v_data = vecs * (self.head_dim // 2)
                v_scales = vecs * (self.head_dim // gs) * 2
            else:
                v_data = vecs * self.head_dim
                v_scales = vecs * (self.head_dim // gs) * 2
            raw_payload = k_total + v_data + v_scales

        # Pad to 4-byte boundary for 32-bit word-aligned ARX stream encryption
        rem = raw_payload % 4
        return raw_payload if rem == 0 else raw_payload + (4 - rem)

    def cold_blob_bytes(self, preset: QuantPreset, rope_mode: RoPEMode) -> int:
        """Total RDMA wire bytes = 32B Header + Encrypted Payload + 16B Auth Tag."""
        return HEADER_BYTES + self.payload_bytes(preset, rope_mode) + TAG_BYTES

    def compression_ratio(self, preset: QuantPreset, rope_mode: RoPEMode) -> float:
        return self.hot_page_bytes / self.cold_blob_bytes(preset, rope_mode)
