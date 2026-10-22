# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hakan Damar and Theseus Guard, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Developed by Theseus Guard Research (https://www.theseusguard.com)
# Repository: https://github.com/Theseus-Guard/ZTCP
# Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
"""Single-Pass Fused Cold-Page Quantization + Authenticated Encryption Engine.

Developed by Theseus Guard, Inc. Implements the Zero-Trust Cold Pages (ZTCP)
research baseline:
1. Session-key derivation (HKDF-SHA256) once per (deployment_pepper, tenant_id, session_id)
   mixed with a per-session 256-bit CSPRNG ephemeral secret (`os.urandom(32)`) and monotonic
   generation counter, keeping the hot path free of per-page KDF overhead.
2. Asymmetric RoPE-preserving group quantization (INT4 / K8V4 / FP8) fused with vectorized
   256-bit IETF RFC 8439 ChaCha20 encryption (with collision-free 96-bit nonce packing and
   session-wide nonce uniqueness enforcement) and 128-bit keyed MAC authentication over the
   32-byte AAD header (`Tenant_ID | Model_ID | Layer_ID | Page_Token_Hash | Nonce`).
3. Versioned Page Freshness Contract: tracks the highest `sequence_nonce` per logical KV page
   slot `(model_id_hash, layer_id, head_group_id, page_token_hash)`, permitting legitimate
   repeated cache reads of the current version while rejecting stale-version replays with
   `StalePageReplayError`.
4. In-Process Crypto-Shredding Scope & Boundaries: zeroing the mutable session key buffers
   and discarding the per-session 256-bit CSPRNG secret ensures that re-deriving for the same
   `(tenant_id, session_id)` produces an independent key that cannot decrypt archived cold
   blobs. In this public user-space reference codec, key buffers reside in standard process
   RAM (not hardware TEE enclave memory or `mlock` pages); protection against OS swap/core-dump
   extraction and distributed multi-node key revocation requires the enterprise TEE enclave
   and control-plane integration.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import threading
from dataclasses import dataclass, field

import torch

from .cold_page_spec import (
    HEADER_BYTES,
    TAG_BYTES,
    ColdPageAAD,
    ColdPageGeometry,
    QuantPreset,
    RoPEMode,
    hash32,
)


class KVIntegrityError(RuntimeError):
    """Raised when a cold KV page fails MAC authentication (tampering / bit-flip)."""


class CrossTenantContaminationError(KVIntegrityError):
    """Raised when a cold KV page's AAD does not match the requesting tenant/model/layer."""


class StalePageReplayError(KVIntegrityError):
    """Raised when a cold KV page's sequence_nonce is older than the latest slot version."""


class NonceReuseError(ValueError):
    """Raised when an encode call attempts to reuse a final ChaCha20 nonce under the same session key."""


class SessionShreddedError(RuntimeError):
    """Raised when attempting to use a session whose ephemeral key was crypto-shredded."""


class _StagedNonceReservation(set):
    """Set of reserved 96-bit ChaCha20 nonces bound to a specific session key generation."""

    def __init__(
        self,
        nonces: set[bytes] | None,
        generation: int,
        enc_buf: bytearray,
        mac_buf: bytearray,
        first_nonce: bytes | None = None,
    ) -> None:
        super().__init__(nonces) if nonces is not None else super().__init__()
        self.generation: int = generation
        self.enc_buf: bytearray = enc_buf
        self.mac_buf: bytearray = mac_buf
        self.revoked: bool = False
        self.first_nonce: bytes | None = first_nonce

    def zeroize(self) -> None:
        self.enc_buf[:] = b"\x00" * len(self.enc_buf)
        self.mac_buf[:] = b"\x00" * len(self.mac_buf)


@dataclass
class EphemeralSessionKeyRing:
    """Manages per-tenant/per-session ephemeral keys, nonce uniqueness, freshness, and crypto-shredding.

    Guarantee Scope & Boundaries:
    - Each session derivation mixes `deployment_pepper`, `(tenant_id, session_id)`, a monotonic
      generation counter, and a fresh 256-bit CSPRNG secret (`os.urandom(32)`) that is zeroized
      immediately after HKDF extraction.
    - All session state transitions (`derive_or_get`, `next_sequence_nonce`,
      `preflight_validate_egress`, `begin_egress_execution`, `commit_egress`,
      `verify_and_update_freshness`, `crypto_shred_session`, `crypto_shred_tenant`) are
      synchronized via a re-entrant mutex (`_lock`) and bound to the session key generation.
    - Nonces are atomically reserved (burned into `_used_chacha_nonces`) inside
      `preflight_validate_egress` before encryption begins; if an operation fails mid-flight,
      its reserved nonces remain burned and are never reused under that session key.
    - Slot freshness watermarks (`_slot_latest_nonce`) advance strictly monotonically (`max`)
      so an older in-flight operation completing out-of-order cannot roll back the watermark.
    - `crypto_shred_session` / `crypto_shred_tenant` overwrites stored mutable `bytearray`
      key buffers and any in-flight reservation buffers in-place with zeros, marks in-flight
      reservations for that generation as revoked, and drops session state. In-flight operations
      bound to a shredded generation abort with `SessionShreddedError` without encrypting with
      zeroed keys, without publishing output, and without committing into a newly derived
      generation.
    - Because this public PoC runs in normal Python/C++ user-space memory (without hardware TEE
      sealing or `mlock`), caller-held immutable `bytes` references prior to GC and OS-level memory
      dumps are outside the local zeroization boundary.
    """

    deployment_pepper: bytes
    _active_keys: dict[tuple[str, str], tuple[bytearray, bytearray, int, int]] = field(
        default_factory=dict
    )
    _active_key_buffers: dict[tuple[str, str], tuple[bytearray, bytearray]] = field(
        default_factory=dict
    )
    _session_generations: dict[tuple[str, str], int] = field(default_factory=dict)
    _shredded_generations: dict[tuple[str, str], int] = field(default_factory=dict)
    _used_chacha_nonces: dict[tuple[str, str], set[bytes]] = field(
        default_factory=dict
    )
    _slot_latest_nonce: dict[tuple[str, str], dict[tuple[int, int, int, int], int]] = (
        field(default_factory=dict)
    )
    _session_max_nonce: dict[tuple[str, str], int] = field(default_factory=dict)
    _inflight_reservations: dict[
        tuple[tuple[str, str], int], list[_StagedNonceReservation]
    ] = field(default_factory=dict)
    _nonce_generations: dict[tuple[tuple[str, str], bytes], int] = field(
        default_factory=dict
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.deployment_pepper) < 32:
            raise ValueError("deployment_pepper must be at least 32 bytes")

    @staticmethod
    def pack_chacha20_nonce(
        sequence_nonce: int, layer_id: int, head_group_id: int
    ) -> bytes:
        """Bijective 96-bit ChaCha20 nonce packing: sequence_nonce (64b) || layer_id (16b) || head_group_id (16b)."""
        seq_u64 = sequence_nonce & 0xFFFFFFFFFFFFFFFF
        seq_lo = seq_u64 & 0xFFFFFFFF
        seq_hi = (seq_u64 >> 32) & 0xFFFFFFFF
        loc_word = ((layer_id & 0xFFFF) << 16) | (head_group_id & 0xFFFF)
        return struct.pack("<3I", seq_lo, seq_hi, loc_word)

    def _ensure_active_generation_locked(self, cache_key: tuple[str, str]) -> int:
        """Ensures active session metadata is consistent under `_lock` and returns its generation."""
        if cache_key not in self._active_keys:
            raise SessionShreddedError(
                f"Session {cache_key} key is missing or crypto-shredded"
            )
        enc_ref, mac_ref, k0, k1 = self._active_keys[cache_key]
        if cache_key not in self._active_key_buffers:
            enc_buf = (
                enc_ref if isinstance(enc_ref, bytearray) else bytearray(enc_ref)
            )
            mac_buf = (
                mac_ref if isinstance(mac_ref, bytearray) else bytearray(mac_ref)
            )
            self._active_key_buffers[cache_key] = (enc_buf, mac_buf)
            self._active_keys[cache_key] = (enc_buf, mac_buf, k0, k1)
        else:
            enc_buf, mac_buf = self._active_key_buffers[cache_key]

        if (
            len(enc_buf) != 32
            or len(mac_buf) != 32
            or not any(enc_buf)
            or not any(mac_buf)
        ):
            raise SessionShreddedError(
                f"Session {cache_key} key buffer was zeroized or crypto-shredded"
            )

        if cache_key not in self._session_generations:
            gen = self._shredded_generations.get(cache_key, 0) + 1
            self._session_generations[cache_key] = gen
        return self._session_generations[cache_key]

    def derive_or_get(
        self, tenant_id: str, session_id: str
    ) -> tuple[bytearray, bytearray, int, int]:
        """Returns (enc_key_32b, mac_key_32b, k0_u64, k1_u64) for the session."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key in self._active_keys:
                self._ensure_active_generation_locked(cache_key)
                return self._active_keys[cache_key]

            gen = (
                max(
                    self._session_generations.get(cache_key, 0),
                    self._shredded_generations.get(cache_key, 0),
                )
                + 1
            )
            self._session_generations[cache_key] = gen
            ephemeral_secret = bytearray(os.urandom(32))
            try:
                prk = hmac.new(
                    self.deployment_pepper,
                    f"{tenant_id}:{session_id}:gen={gen}:".encode("utf-8")
                    + bytes(ephemeral_secret),
                    hashlib.sha256,
                ).digest()
            finally:
                ephemeral_secret[:] = b"\x00" * len(ephemeral_secret)

            enc_buf = bytearray(
                hmac.new(prk, b"ztcp-cold-page-enc-v1\x01", hashlib.sha256).digest()
            )
            mac_buf = bytearray(
                hmac.new(prk, b"ztcp-cold-page-mac-v1\x02", hashlib.sha256).digest()
            )
            self._active_key_buffers[cache_key] = (enc_buf, mac_buf)
            self._used_chacha_nonces[cache_key] = set()
            self._slot_latest_nonce[cache_key] = {}
            self._session_max_nonce[cache_key] = 0
            self._inflight_reservations[(cache_key, gen)] = []

            k0, k1 = struct.unpack("<QQ", enc_buf[:16])
            entry = (enc_buf, mac_buf, k0, k1)
            self._active_keys[cache_key] = entry
            return entry

    def next_sequence_nonce(
        self, tenant_id: str, session_id: str, count: int = 1
    ) -> int:
        """Allocates a fresh monotonic sequence_nonce range [base, base + count) for the session."""
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key not in self._active_keys:
                self.derive_or_get(tenant_id, session_id)
            self._ensure_active_generation_locked(cache_key)
            base = self._session_max_nonce.get(cache_key, 0) + 1
            self._session_max_nonce[cache_key] = base + count - 1
            return base

    def preflight_validate_egress(
        self,
        tenant_id: str,
        session_id: str,
        model_id_hash: int,
        layer_id: int,
        head_group_id: int,
        base_sequence_nonce: int,
        token_hashes: list[int],
    ) -> tuple[_StagedNonceReservation, dict[tuple[int, int, int, int], int], int]:
        """Atomically validates and reserves (burns) ChaCha20 96-bit nonces before encryption."""
        if not token_hashes:
            raise ValueError("token_hashes must be non-empty")
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key not in self._active_keys:
                self.derive_or_get(tenant_id, session_id)
            gen = self._ensure_active_generation_locked(cache_key)
            enc_buf, mac_buf = self._active_key_buffers[cache_key]

            used_nonces = self._used_chacha_nonces.setdefault(cache_key, set())
            slot_watermarks = self._slot_latest_nonce.setdefault(cache_key, {})
            m_hash = model_id_hash & 0xFFFFFFFF
            l_id = layer_id & 0xFFFF
            g_id = head_group_id & 0xFFFF
            loc_suffix = struct.pack("<I", (l_id << 16) | g_id)
            base_seq = base_sequence_nonce & 0xFFFFFFFFFFFFFFFF
            n_pages = len(token_hashes)

            staged_set: set[bytes] = set()
            staged_slots: dict[tuple[int, int, int, int], int] = {}
            sw_get = slot_watermarks.get
            ss_get = staged_slots.get
            set_add = staged_set.add

            for idx, th in enumerate(token_hashes):
                seq_b = (base_seq + idx) & 0xFFFFFFFFFFFFFFFF
                tok_u64 = th & 0xFFFFFFFFFFFFFFFF
                slot_key = (m_hash, l_id, g_id, tok_u64)
                prev_watermark = ss_get(slot_key)
                if prev_watermark is None:
                    prev_watermark = sw_get(slot_key, 0)
                if seq_b < prev_watermark:
                    raise StalePageReplayError(
                        f"STALE_PAGE_REPLAY: Cannot encode cold page with stale sequence_nonce={seq_b} < latest={prev_watermark}"
                    )

                nonce_12b = seq_b.to_bytes(8, "little") + loc_suffix
                if nonce_12b in used_nonces or nonce_12b in staged_set:
                    raise NonceReuseError(
                        f"NONCE_REUSE_DETECTED: Final 96-bit ChaCha20 nonce {nonce_12b.hex()} already used under session {cache_key}"
                    )
                set_add(nonce_12b)
                staged_slots[slot_key] = seq_b

            # Only after all stale/duplicate checks succeed, create the key-bearing reservation,
            # register it in _inflight_reservations, and atomically burn reserved nonces BEFORE encryption.
            first_n = (base_seq & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little") + loc_suffix
            reservation = _StagedNonceReservation(
                staged_set,
                generation=gen,
                enc_buf=bytearray(enc_buf),
                mac_buf=bytearray(mac_buf),
                first_nonce=first_n,
            )
            try:
                self._inflight_reservations.setdefault((cache_key, gen), []).append(
                    reservation
                )
                used_nonces.update(staged_set)
                self._nonce_generations[(cache_key, first_n)] = gen

                last_seq = (base_seq + max(0, n_pages - 1)) & 0xFFFFFFFFFFFFFFFF
                if last_seq > self._session_max_nonce.get(cache_key, 0):
                    self._session_max_nonce[cache_key] = last_seq
            except BaseException:
                reservation.zeroize()
                raise

            return reservation, staged_slots, last_seq

    def _resolve_reservation_generation_locked(
        self, cache_key: tuple[str, str], staged_nonces: set[bytes]
    ) -> tuple[int, bool]:
        if isinstance(staged_nonces, _StagedNonceReservation):
            return staged_nonces.generation, staged_nonces.revoked
        first_n = next(iter(staged_nonces), None)
        if first_n is None:
            return -1, True
        gen = self._nonce_generations.get((cache_key, first_n), -1)
        revoked = self._shredded_generations.get(cache_key, 0) >= gen or gen <= 0
        return gen, revoked

    def begin_egress_execution(
        self,
        tenant_id: str,
        session_id: str,
        staged_nonces: set[bytes],
    ) -> tuple[bytes, bytes]:
        """Verifies the reserved generation is still active after preflight and returns immutable key snapshots."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            res_gen, is_revoked = self._resolve_reservation_generation_locked(
                cache_key, staged_nonces
            )
            if (
                is_revoked
                or cache_key not in self._active_keys
                or self._session_generations.get(cache_key) != res_gen
                or self._shredded_generations.get(cache_key, 0) >= res_gen
            ):
                if isinstance(staged_nonces, _StagedNonceReservation):
                    staged_nonces.zeroize()
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) generation {res_gen} was crypto-shredded before or during egress"
                )

            if isinstance(staged_nonces, _StagedNonceReservation):
                if (
                    len(staged_nonces.enc_buf) != 32
                    or len(staged_nonces.mac_buf) != 32
                    or not any(staged_nonces.enc_buf)
                    or not any(staged_nonces.mac_buf)
                ):
                    staged_nonces.zeroize()
                    raise SessionShreddedError(
                        f"Session ({tenant_id}, {session_id}) generation {res_gen} key buffer was zeroized"
                    )
                return bytes(staged_nonces.enc_buf), bytes(staged_nonces.mac_buf)

            self._ensure_active_generation_locked(cache_key)
            enc_buf, mac_buf = self._active_key_buffers[cache_key]
            return bytes(enc_buf), bytes(mac_buf)

    def commit_egress(
        self,
        tenant_id: str,
        session_id: str,
        staged_nonces: set[bytes],
        staged_slots: dict[tuple[int, int, int, int], int],
        last_seq: int,
    ) -> None:
        """Commits slot watermarks monotonically after verifying the key generation was not shredded."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            res_gen, is_revoked = self._resolve_reservation_generation_locked(
                cache_key, staged_nonces
            )
            if (
                is_revoked
                or cache_key not in self._active_keys
                or self._session_generations.get(cache_key) != res_gen
                or self._shredded_generations.get(cache_key, 0) >= res_gen
            ):
                if isinstance(staged_nonces, _StagedNonceReservation):
                    staged_nonces.zeroize()
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) generation {res_gen} was crypto-shredded before commit"
                )

            if not isinstance(staged_nonces, _StagedNonceReservation):
                self._used_chacha_nonces.setdefault(cache_key, set()).update(
                    staged_nonces
                )
            slot_watermarks = self._slot_latest_nonce.setdefault(cache_key, {})
            sw_get = slot_watermarks.get
            for slot_key, seq_b in staged_slots.items():
                if seq_b > sw_get(slot_key, 0):
                    slot_watermarks[slot_key] = seq_b

            if last_seq > self._session_max_nonce.get(cache_key, 0):
                self._session_max_nonce[cache_key] = last_seq

    def release_egress_reservation(
        self,
        tenant_id: str,
        session_id: str,
        staged_nonces: set[bytes],
    ) -> None:
        """Zeroizes per-operation key buffers and unregisters the in-flight reservation."""
        with self._lock:
            if isinstance(staged_nonces, _StagedNonceReservation):
                staged_nonces.zeroize()
                if staged_nonces.first_nonce is not None:
                    self._nonce_generations.pop(
                        ((tenant_id, session_id), staged_nonces.first_nonce), None
                    )
                key = ((tenant_id, session_id), staged_nonces.generation)
                inflight = self._inflight_reservations.get(key)
                if inflight is not None:
                    try:
                        inflight.remove(staged_nonces)
                    except ValueError:
                        pass

    def begin_ingress(
        self, tenant_id: str, session_id: str
    ) -> tuple[int, bytes, bytes]:
        """Captures the active session generation and immutable key snapshots for ingress decode."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key not in self._active_keys:
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) key is missing or crypto-shredded"
                )
            gen = self._ensure_active_generation_locked(cache_key)
            enc_buf, mac_buf = self._active_key_buffers[cache_key]
            return gen, bytes(enc_buf), bytes(mac_buf)

    def verify_and_update_freshness(
        self,
        tenant_id: str,
        session_id: str,
        model_id_hash: int,
        layer_id: int,
        head_group_id: int,
        page_token_hash: int,
        sequence_nonce: int,
        min_sequence_nonce: int | None = None,
        expected_sequence_nonce: int | None = None,
        expected_generation: int | None = None,
    ) -> None:
        """Enforces the cold-page freshness contract AFTER MAC authentication has succeeded.

        - Verifies the session key generation has not been shredded while ingress was in flight.
        - Legitimate cache re-reads of the current version (`sequence_nonce == latest_slot_nonce`)
          are accepted.
        - Replays of an older superseded version (`sequence_nonce < latest_slot_nonce`) or pages
          violating caller-supplied `min_sequence_nonce` / `expected_sequence_nonce` raise
          `StalePageReplayError`.
        """
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key not in self._active_keys:
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) key is missing or crypto-shredded"
                )
            cur_gen = self._ensure_active_generation_locked(cache_key)
            if expected_generation is not None and (
                cur_gen != expected_generation
                or self._shredded_generations.get(cache_key, 0) >= expected_generation
            ):
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) generation {expected_generation} was crypto-shredded during ingress"
                )

            seq_u64 = sequence_nonce & 0xFFFFFFFFFFFFFFFF

            if expected_sequence_nonce is not None:
                exp_u64 = expected_sequence_nonce & 0xFFFFFFFFFFFFFFFF
                if seq_u64 != exp_u64:
                    raise StalePageReplayError(
                        f"STALE_PAGE_REPLAY: Cold-page sequence_nonce={seq_u64} != expected_sequence_nonce={exp_u64}"
                    )

            if min_sequence_nonce is not None:
                min_u64 = min_sequence_nonce & 0xFFFFFFFFFFFFFFFF
                if seq_u64 < min_u64:
                    raise StalePageReplayError(
                        f"STALE_PAGE_REPLAY: Cold-page sequence_nonce={seq_u64} < min_sequence_nonce={min_u64}"
                    )

            slot_watermarks = self._slot_latest_nonce.setdefault(cache_key, {})
            slot_key = (
                model_id_hash & 0xFFFFFFFF,
                layer_id & 0xFFFF,
                head_group_id & 0xFFFF,
                page_token_hash & 0xFFFFFFFFFFFFFFFF,
            )
            latest_nonce = slot_watermarks.get(slot_key, 0)
            if seq_u64 < latest_nonce:
                raise StalePageReplayError(
                    f"STALE_PAGE_REPLAY: Cold-page sequence_nonce={seq_u64} is older than latest slot version={latest_nonce}"
                )

            if seq_u64 > latest_nonce:
                slot_watermarks[slot_key] = seq_u64
            if seq_u64 > self._session_max_nonce.get(cache_key, 0):
                self._session_max_nonce[cache_key] = seq_u64

    def verify_and_update_freshness_batch(
        self,
        tenant_id: str,
        session_id: str,
        records: list[tuple[int, int, int, int, int]],
        min_sequence_nonce: int | None = None,
        expected_generation: int | None = None,
    ) -> None:
        """Atomically verifies and updates freshness watermarks for a batch of decoded headers and finishes ingress."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            if cache_key not in self._active_keys:
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) key is missing or crypto-shredded"
                )
            cur_gen = self._ensure_active_generation_locked(cache_key)
            if expected_generation is not None and (
                cur_gen != expected_generation
                or self._shredded_generations.get(cache_key, 0) >= expected_generation
            ):
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) generation {expected_generation} was crypto-shredded during ingress"
                )

            slot_watermarks = self._slot_latest_nonce.setdefault(cache_key, {})
            sw_get = slot_watermarks.get
            max_s = self._session_max_nonce.get(cache_key, 0)
            min_u64 = (
                (min_sequence_nonce & 0xFFFFFFFFFFFFFFFF)
                if min_sequence_nonce is not None
                else None
            )

            staged_updates: dict[tuple[int, int, int, int], int] = {}
            su_get = staged_updates.get
            for m_hash, l_id, g_id, tok_u64, seq_u64 in records:
                seq_u64 &= 0xFFFFFFFFFFFFFFFF
                if min_u64 is not None and seq_u64 < min_u64:
                    raise StalePageReplayError(
                        f"STALE_PAGE_REPLAY: Cold-page sequence_nonce={seq_u64} < min_sequence_nonce={min_u64}"
                    )
                slot_key = (
                    m_hash & 0xFFFFFFFF,
                    l_id & 0xFFFF,
                    g_id & 0xFFFF,
                    tok_u64 & 0xFFFFFFFFFFFFFFFF,
                )
                latest_nonce = su_get(slot_key)
                if latest_nonce is None:
                    latest_nonce = sw_get(slot_key, 0)
                if seq_u64 < latest_nonce:
                    raise StalePageReplayError(
                        f"STALE_PAGE_REPLAY: Cold-page sequence_nonce={seq_u64} is older than latest slot version={latest_nonce}"
                    )
                if seq_u64 > latest_nonce:
                    staged_updates[slot_key] = seq_u64
                if seq_u64 > max_s:
                    max_s = seq_u64

            slot_watermarks.update(staged_updates)
            self._session_max_nonce[cache_key] = max_s

    def finish_ingress(
        self, tenant_id: str, session_id: str, expected_generation: int
    ) -> None:
        """Verifies the session generation remains active right before returning decoded tensors."""
        with self._lock:
            cache_key = (tenant_id, session_id)
            if (
                cache_key not in self._active_keys
                or self._session_generations.get(cache_key) != expected_generation
                or self._shredded_generations.get(cache_key, 0) >= expected_generation
            ):
                raise SessionShreddedError(
                    f"Session ({tenant_id}, {session_id}) generation {expected_generation} was crypto-shredded during ingress"
                )

    def _zeroize_and_remove_session(self, cache_key: tuple[str, str]) -> bool:
        with self._lock:
            gen = self._session_generations.get(cache_key, 1)
            bufs = self._active_key_buffers.pop(cache_key, None)
            if bufs is not None:
                enc_buf, mac_buf = bufs
                enc_buf[:] = b"\x00" * len(enc_buf)
                mac_buf[:] = b"\x00" * len(mac_buf)
            active_entry = self._active_keys.pop(cache_key, None)
            existed = active_entry is not None
            if active_entry is not None:
                enc_ref, mac_ref, _, _ = active_entry
                if isinstance(enc_ref, bytearray):
                    enc_ref[:] = b"\x00" * len(enc_ref)
                if isinstance(mac_ref, bytearray):
                    mac_ref[:] = b"\x00" * len(mac_ref)
            for res in self._inflight_reservations.pop((cache_key, gen), []):
                res.revoked = True
                res.zeroize()
            self._used_chacha_nonces.pop(cache_key, None)
            self._slot_latest_nonce.pop(cache_key, None)
            self._session_max_nonce.pop(cache_key, None)
            if existed:
                self._shredded_generations[cache_key] = max(
                    self._shredded_generations.get(cache_key, 0), gen
                )
            return existed

    def crypto_shred_session(self, tenant_id: str, session_id: str) -> bool:
        """Instantly zeroes the session's mutable key buffers and revokes in-flight operations."""
        with self._lock:
            return self._zeroize_and_remove_session((tenant_id, session_id))

    def crypto_shred_tenant(self, tenant_id: str) -> int:
        """Destroys and zeroes all active session keys belonging to a tenant."""
        with self._lock:
            to_remove = [k for k in self._active_keys if k[0] == tenant_id]
            for k in to_remove:
                self._zeroize_and_remove_session(k)
            return len(to_remove)


def chacha20_rfc8439_block(key_32b: bytes, counter: int, nonce_12b: bytes) -> bytes:
    """Reference IETF RFC 8439 Section 2.3 ChaCha20 64-byte block function."""
    if len(key_32b) != 32:
        raise ValueError("ChaCha20 requires a 32-byte (256-bit) key")
    if len(nonce_12b) != 12:
        raise ValueError("ChaCha20 requires a 12-byte (96-bit) nonce")

    def _rotl32(v: int, c: int) -> int:
        return ((v << c) & 0xFFFFFFFF) | (v >> (32 - c))

    def _qr(x: list[int], a: int, b: int, c: int, d: int) -> None:
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF
        x[d] = _rotl32(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF
        x[b] = _rotl32(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF
        x[d] = _rotl32(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF
        x[b] = _rotl32(x[b] ^ x[c], 7)

    constants = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
    key_words = struct.unpack("<8I", key_32b)
    nonce_words = struct.unpack("<3I", nonce_12b)
    state = list(constants) + list(key_words) + [counter & 0xFFFFFFFF] + list(nonce_words)
    working = list(state)
    for _ in range(10):
        _qr(working, 0, 4, 8, 12)
        _qr(working, 1, 5, 9, 13)
        _qr(working, 2, 6, 10, 14)
        _qr(working, 3, 7, 11, 15)
        _qr(working, 0, 5, 10, 15)
        _qr(working, 1, 6, 11, 12)
        _qr(working, 2, 7, 8, 13)
        _qr(working, 3, 4, 9, 14)
    out = [(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)]
    return struct.pack("<16I", *out)


def siphash24_128(key_16b: bytes, data: bytes) -> bytes:
    """Reference 128-bit SipHash-2-4 (Aumasson & Bernstein specification)."""
    if len(key_16b) < 16:
        raise ValueError("SipHash-2-4-128 requires at least 16 bytes of key")
    k0, k1 = struct.unpack("<QQ", key_16b[:16])
    v0 = 0x736F6D6570736575 ^ k0
    v1 = 0x646F72616E646F6D ^ k1 ^ 0xEE
    v2 = 0x6C7967656E657261 ^ k0
    v3 = 0x7465646279746573 ^ k1

    def _rotl64(x: int, b: int) -> int:
        return ((x << b) | (x >> (64 - b))) & 0xFFFFFFFFFFFFFFFF

    def _sipround() -> None:
        nonlocal v0, v1, v2, v3
        v0 = (v0 + v1) & 0xFFFFFFFFFFFFFFFF
        v1 = _rotl64(v1, 13) ^ v0
        v0 = _rotl64(v0, 32)
        v2 = (v2 + v3) & 0xFFFFFFFFFFFFFFFF
        v3 = _rotl64(v3, 16) ^ v2
        v0 = (v0 + v3) & 0xFFFFFFFFFFFFFFFF
        v3 = _rotl64(v3, 21) ^ v0
        v2 = (v2 + v1) & 0xFFFFFFFFFFFFFFFF
        v1 = _rotl64(v1, 17) ^ v2
        v2 = _rotl64(v2, 32)

    n_words = len(data) // 8
    for i in range(n_words):
        (m,) = struct.unpack_from("<Q", data, i * 8)
        v3 ^= m
        _sipround()
        _sipround()
        v0 ^= m

    rem = len(data) & 7
    last = (len(data) & 0xFF) << 56
    tail = data[n_words * 8 :]
    for b in range(rem):
        last |= tail[b] << (8 * b)
    v3 ^= last
    _sipround()
    _sipround()
    v0 ^= last

    v2 ^= 0xEE
    for _ in range(4):
        _sipround()
    tag_lo = v0 ^ v1 ^ v2 ^ v3
    v1 ^= 0xDD
    for _ in range(4):
        _sipround()
    tag_hi = v0 ^ v1 ^ v2 ^ v3
    return struct.pack("<QQ", tag_lo, tag_hi)


class FusedColdPageCodec:
    """Reference GPU/Vectorized Fused Quantization + AEAD Codec for Cold KV Pages."""

    def __init__(
        self,
        geometry: ColdPageGeometry,
        keyring: EphemeralSessionKeyRing,
        device: str | torch.device = "cpu",
    ) -> None:
        self.geometry = geometry
        self.keyring = keyring
        self.device = torch.device(device)
        self._counterCache: dict[int, torch.Tensor] = {}

    def _get_counters(self, num_words64: int) -> torch.Tensor:
        cached = self._counterCache.get(num_words64)
        if cached is None or cached.device != self.device:
            cached = torch.arange(
                num_words64, dtype=torch.int64, device=self.device
            )
            self._counterCache[num_words64] = cached
        return cached

    @staticmethod
    def _quantize_int8_grouped(
        x: torch.Tensor, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric 8-bit group quantization -> (uint8_bytes, fp16_scales_bytes)."""
        orig_shape = x.shape
        grouped = x.reshape(-1, group_size).to(torch.float32)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (amax / 127.0).to(torch.float16)
        q = torch.clamp(torch.round(grouped / scale.to(torch.float32)), -127, 127).to(
            torch.int8
        )
        return q.view(torch.uint8).reshape(-1), scale.view(torch.uint8).reshape(-1)

    @staticmethod
    def _dequantize_int8_grouped(
        q_u8: torch.Tensor,
        scale_u8: torch.Tensor,
        target_shape: tuple[int, ...],
        group_size: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        q = q_u8.view(torch.int8).reshape(-1, group_size).to(torch.float32)
        scale = scale_u8.view(torch.float16).reshape(-1, 1).to(torch.float32)
        return (q * scale).to(out_dtype).reshape(target_shape)

    @staticmethod
    def _quantize_int4_grouped(
        x: torch.Tensor, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric 4-bit group quantization packed 2 values per uint8 + fp16 scale."""
        grouped = x.reshape(-1, group_size).to(torch.float32)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (amax / 7.0).to(torch.float16)
        # Map [-7, +7] -> [1, 15] unsigned nibble (offset +8)
        q = torch.clamp(
            torch.round(grouped / scale.to(torch.float32)) + 8.0, 0.0, 15.0
        ).to(torch.uint8)
        low_nibble = q[:, 0::2] & 0x0F
        high_nibble = (q[:, 1::2] & 0x0F) << 4
        packed = (low_nibble | high_nibble).reshape(-1)
        return packed, scale.view(torch.uint8).reshape(-1)

    @staticmethod
    def _dequantize_int4_grouped(
        packed_u8: torch.Tensor,
        scale_u8: torch.Tensor,
        target_shape: tuple[int, ...],
        group_size: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        packed = packed_u8.reshape(-1, group_size // 2)
        low = (packed & 0x0F).to(torch.float32) - 8.0
        high = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
        unpacked = torch.empty(
            (packed.shape[0], group_size), dtype=torch.float32, device=packed.device
        )
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high
        scale = scale_u8.view(torch.float16).reshape(-1, 1).to(torch.float32)
        return (unpacked * scale).to(out_dtype).reshape(target_shape)

    @staticmethod
    def _derive_chacha20_nonce(aad: ColdPageAAD) -> bytes:
        return EphemeralSessionKeyRing.pack_chacha20_nonce(
            aad.sequence_nonce, aad.layer_id, aad.head_group_id
        )

    def _apply_arx_keystream_inplace(
        self, payload_u8: torch.Tensor, enc_key: bytes, aad: ColdPageAAD
    ) -> torch.Tensor:
        """Vectorized IETF RFC 8439 ChaCha20 (256-bit key, 96-bit AAD nonce, counter=1)."""
        if len(enc_key) != 32:
            raise ValueError("ChaCha20 requires a 32-byte (256-bit) encryption key")
        n_bytes = payload_u8.numel()
        n_blocks = (n_bytes + 63) // 64
        dev = payload_u8.device
        nonce_12b = self._derive_chacha20_nonce(aad)

        c0, c1, c2, c3 = 0x61707865, 0x3320646E, 0x79622D32, 0x6B206574
        k = struct.unpack("<8I", enc_key)
        n = struct.unpack("<3I", nonce_12b)
        counters = self._get_counters(n_blocks) + 1

        s = [
            torch.full((n_blocks,), c0, dtype=torch.int64, device=dev),
            torch.full((n_blocks,), c1, dtype=torch.int64, device=dev),
            torch.full((n_blocks,), c2, dtype=torch.int64, device=dev),
            torch.full((n_blocks,), c3, dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[0], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[1], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[2], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[3], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[4], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[5], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[6], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), k[7], dtype=torch.int64, device=dev),
            counters,
            torch.full((n_blocks,), n[0], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), n[1], dtype=torch.int64, device=dev),
            torch.full((n_blocks,), n[2], dtype=torch.int64, device=dev),
        ]
        r0 = torch.stack(s[0:4], dim=1)
        r1 = torch.stack(s[4:8], dim=1)
        r2 = torch.stack(s[8:12], dim=1)
        r3 = torch.stack(s[12:16], dim=1)
        init0, init1, init2, init3 = r0.clone(), r1.clone(), r2.clone(), r3.clone()
        mask = 0xFFFFFFFF

        for _ in range(10):
            for diag in (False, True):
                if diag:
                    r1 = torch.roll(r1, shifts=-1, dims=1)
                    r2 = torch.roll(r2, shifts=-2, dims=1)
                    r3 = torch.roll(r3, shifts=-3, dims=1)
                r0 = (r0 + r1) & mask
                r3 = r3 ^ r0
                r3 = ((r3 << 16) | (r3 >> 16)) & mask
                r2 = (r2 + r3) & mask
                r1 = r1 ^ r2
                r1 = ((r1 << 12) | (r1 >> 20)) & mask
                r0 = (r0 + r1) & mask
                r3 = r3 ^ r0
                r3 = ((r3 << 8) | (r3 >> 24)) & mask
                r2 = (r2 + r3) & mask
                r1 = r1 ^ r2
                r1 = ((r1 << 7) | (r1 >> 25)) & mask
                if diag:
                    r1 = torch.roll(r1, shifts=1, dims=1)
                    r2 = torch.roll(r2, shifts=2, dims=1)
                    r3 = torch.roll(r3, shifts=3, dims=1)

        ks_u8 = (
            torch.cat(
                [
                    (r0 + init0) & mask,
                    (r1 + init1) & mask,
                    (r2 + init2) & mask,
                    (r3 + init3) & mask,
                ],
                dim=1,
            )
            .to(torch.uint32)
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)[:n_bytes]
        )
        return payload_u8 ^ ks_u8

    @staticmethod
    def _compute_mac_tag(mac_key: bytes, header_bytes: bytes, ciphertext_bytes: bytes) -> bytes:
        """Computes 128-bit keyed MAC binding the 32B AAD Header and Ciphertext."""
        h = hashlib.blake2b(key=mac_key, digest_size=TAG_BYTES)
        h.update(header_bytes)
        h.update(ciphertext_bytes)
        return h.digest()

    def encode_cold_page(
        self,
        k_page: torch.Tensor,
        v_page: torch.Tensor | None,
        tenant_id: str,
        session_id: str,
        aad: ColdPageAAD,
    ) -> bytes:
        """Single-pass Egress: Quantize (RoPE-aware) + Encrypt + Authenticate into Compact Blob."""
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

            expected_shape = (
                self.geometry.num_tokens,
                self.geometry.num_kv_heads,
                self.geometry.head_dim,
            )
            if tuple(k_page.shape) != expected_shape:
                raise ValueError(
                    f"k_page shape {tuple(k_page.shape)} does not match geometry {expected_shape}"
                )

            gs = self.geometry.group_size
            r_dim = self.geometry.rope_dim
            chunks: list[torch.Tensor] = []

            # 1. RoPE / NoPE split on K page: shape [num_tokens, num_kv_heads, head_dim]
            if aad.rope_mode == RoPEMode.PRESERVE_RAW and r_dim > 0:
                k_rope = k_page[..., :r_dim].contiguous().to(torch.float16)
                chunks.append(k_rope.view(torch.uint8).reshape(-1))
                k_nope = k_page[..., r_dim:].contiguous()
            elif aad.rope_mode == RoPEMode.PRESERVE_FP8 and r_dim > 0:
                k_rope = k_page[..., :r_dim].contiguous()
                q_rope, s_rope = self._quantize_int8_grouped(k_rope, gs)
                chunks.extend([q_rope, s_rope])
                k_nope = k_page[..., r_dim:].contiguous()
            else:
                k_nope = k_page.contiguous()

            # 2. Quantize K NoPE region
            if k_nope.numel() > 0:
                if aad.preset == QuantPreset.INT4_SYM:
                    qk, sk = self._quantize_int4_grouped(k_nope, gs)
                else:
                    qk, sk = self._quantize_int8_grouped(k_nope, gs)
                chunks.extend([qk, sk])

            # 3. Quantize V buffer (unless single-buffer MLA)
            if not self.geometry.is_mla_single_buffer:
                if v_page is None:
                    raise ValueError("v_page is required for non-MLA geometry")
                if tuple(v_page.shape) != expected_shape:
                    raise ValueError(
                        f"v_page shape {tuple(v_page.shape)} does not match geometry {expected_shape}"
                    )
                v_contig = v_page.contiguous()
                if aad.preset in (QuantPreset.INT4_SYM, QuantPreset.K8V4_ASYM):
                    qv, sv = self._quantize_int4_grouped(v_contig, gs)
                else:
                    qv, sv = self._quantize_int8_grouped(v_contig, gs)
                chunks.extend([qv, sv])

            compact_u8 = torch.cat(chunks, dim=0)
            expected_payload = self.geometry.payload_bytes(aad.preset, aad.rope_mode)
            if compact_u8.numel() < expected_payload:
                pad = torch.zeros(
                    expected_payload - compact_u8.numel(),
                    dtype=torch.uint8,
                    device=compact_u8.device,
                )
                compact_u8 = torch.cat([compact_u8, pad], dim=0)

            # 4. In-register / device-side RFC 8439 ChaCha20 (256-bit key) stream encryption
            cipher_u8 = self._apply_arx_keystream_inplace(compact_u8, enc_key, aad)
            cipher_bytes = cipher_u8.cpu().numpy().tobytes()

            # 5. Pack 32B AAD Header & compute 16B MAC Footer
            header_bytes = aad.pack_header()
            tag_bytes = self._compute_mac_tag(mac_key, header_bytes, cipher_bytes)
            self.keyring.commit_egress(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
                staged_slots=staged_slots,
                last_seq=last_seq,
            )
            return header_bytes + cipher_bytes + tag_bytes
        finally:
            self.keyring.release_egress_reservation(
                tenant_id=tenant_id,
                session_id=session_id,
                staged_nonces=staged_nonces,
            )

    def decode_cold_page(
        self,
        cold_blob: bytes,
        expected_tenant_id: str,
        session_id: str,
        expected_model_id: str,
        expected_layer_id: int,
        expected_token_hash: int,
        out_dtype: torch.dtype = torch.float16,
        min_sequence_nonce: int | None = None,
        expected_sequence_nonce: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-pass Ingress: Verify AAD & MAC & Freshness -> Decrypt -> Dequantize to Hot KV Dtype."""
        gen, enc_key, mac_key = self.keyring.begin_ingress(
            expected_tenant_id, session_id
        )

        header_bytes = cold_blob[:HEADER_BYTES]
        tag_bytes = cold_blob[-TAG_BYTES:]
        cipher_bytes = cold_blob[HEADER_BYTES:-TAG_BYTES]

        try:
            aad = ColdPageAAD.unpack_header(header_bytes)
        except ValueError as exc:
            raise KVIntegrityError(
                f"KV_INTEGRITY_ERROR: Invalid cold-page header magic, version, or flags: {exc}"
            ) from exc

        # 1. Verify AAD context (Cross-Tenant / Cross-Layer / Cross-Prefix isolation)
        exp_tenant_hash = hash32(expected_tenant_id)
        exp_model_hash = hash32(expected_model_id)
        exp_tok_u64 = expected_token_hash & 0xFFFFFFFFFFFFFFFF
        if (
            aad.tenant_id_hash != exp_tenant_hash
            or aad.model_id_hash != exp_model_hash
            or aad.layer_id != expected_layer_id
            or aad.page_token_hash != exp_tok_u64
        ):
            raise CrossTenantContaminationError(
                f"Cold-page AAD mismatch! Expected tenant={hex(exp_tenant_hash)} "
                f"layer={expected_layer_id} token_hash={hex(exp_tok_u64)}, "
                f"got tenant={hex(aad.tenant_id_hash)} layer={aad.layer_id} "
                f"token_hash={hex(aad.page_token_hash)}"
            )

        # 2. Constant-time MAC verification before touching Attention memory or freshness state
        computed_tag = self._compute_mac_tag(mac_key, header_bytes, cipher_bytes)
        if not hmac.compare_digest(computed_tag, tag_bytes):
            raise KVIntegrityError(
                "Cold-page MAC verification failed: remote RDMA KV poisoning or bit-flip detected!"
            )

        # 3. Verify versioned page freshness contract (permits legitimate re-reads, blocks stale replay)
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

        # 4. Decrypt payload via RFC 8439 ChaCha20 (256-bit key)
        cipher_u8 = torch.frombuffer(
            bytearray(cipher_bytes), dtype=torch.uint8
        ).to(self.device)
        plain_u8 = self._apply_arx_keystream_inplace(cipher_u8, enc_key, aad)

        # 5. Unpack & Dequantize K and V back to hot GPU representation
        gs = self.geometry.group_size
        vecs = self.geometry.vectors_per_buffer
        t_len = self.geometry.num_tokens
        n_heads = self.geometry.num_kv_heads
        h_dim = self.geometry.head_dim
        r_dim = self.geometry.rope_dim
        n_dim = self.geometry.nope_dim

        offset = 0
        if aad.rope_mode == RoPEMode.PRESERVE_RAW and r_dim > 0:
            n_rope_b = vecs * r_dim * 2
            k_rope = (
                plain_u8[offset : offset + n_rope_b]
                .view(torch.float16)
                .to(out_dtype)
                .reshape(t_len, n_heads, r_dim)
            )
            offset += n_rope_b
            k_quant_dim = n_dim
        elif aad.rope_mode == RoPEMode.PRESERVE_FP8 and r_dim > 0:
            n_q = vecs * r_dim
            n_s = vecs * (r_dim // gs) * 2
            k_rope = self._dequantize_int8_grouped(
                plain_u8[offset : offset + n_q],
                plain_u8[offset + n_q : offset + n_q + n_s],
                (t_len, n_heads, r_dim),
                gs,
                out_dtype,
            )
            offset += n_q + n_s
            k_quant_dim = n_dim
        else:
            k_rope = None
            k_quant_dim = h_dim

        if k_quant_dim > 0:
            if aad.preset == QuantPreset.INT4_SYM:
                n_qk = vecs * (k_quant_dim // 2)
                n_sk = vecs * (k_quant_dim // gs) * 2
                k_nope = self._dequantize_int4_grouped(
                    plain_u8[offset : offset + n_qk],
                    plain_u8[offset + n_qk : offset + n_qk + n_sk],
                    (t_len, n_heads, k_quant_dim),
                    gs,
                    out_dtype,
                )
            else:
                n_qk = vecs * k_quant_dim
                n_sk = vecs * (k_quant_dim // gs) * 2
                k_nope = self._dequantize_int8_grouped(
                    plain_u8[offset : offset + n_qk],
                    plain_u8[offset + n_qk : offset + n_qk + n_sk],
                    (t_len, n_heads, k_quant_dim),
                    gs,
                    out_dtype,
                )
            offset += n_qk + n_sk
            k_page = (
                torch.cat([k_rope, k_nope], dim=-1) if k_rope is not None else k_nope
            )
        else:
            assert k_rope is not None
            k_page = k_rope

        if self.geometry.is_mla_single_buffer:
            try:
                self.keyring.finish_ingress(expected_tenant_id, session_id, gen)
            except BaseException:
                k_page.zero_()
                raise
            return k_page, None

        if aad.preset in (QuantPreset.INT4_SYM, QuantPreset.K8V4_ASYM):
            n_qv = vecs * (h_dim // 2)
            n_sv = vecs * (h_dim // gs) * 2
            v_page = self._dequantize_int4_grouped(
                plain_u8[offset : offset + n_qv],
                plain_u8[offset + n_qv : offset + n_qv + n_sv],
                (t_len, n_heads, h_dim),
                gs,
                out_dtype,
            )
        else:
            n_qv = vecs * h_dim
            n_sv = vecs * (h_dim // gs) * 2
            v_page = self._dequantize_int8_grouped(
                plain_u8[offset : offset + n_qv],
                plain_u8[offset + n_qv : offset + n_qv + n_sv],
                (t_len, n_heads, h_dim),
                gs,
                out_dtype,
            )

        try:
            self.keyring.finish_ingress(expected_tenant_id, session_id, gen)
        except BaseException:
            k_page.zero_()
            v_page.zero_()
            raise
        return k_page, v_page
