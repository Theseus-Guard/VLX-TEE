// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hakan Damar and Theseus Guard, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Developed by Theseus Guard Research (https://www.theseusguard.com)
// Repository: https://github.com/Theseus-Guard/ZTCP
// Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
//
// Zero-Trust Cold Pages (ZTCP): Register-Fused Cold-Page Quantization
// + IETF RFC 8439 ChaCha20 (256-bit Key) Encryption + Standard 128-Bit SipHash-2-4 MAC Kernel.
//
// Micro-Architectural & Cryptographic Design:
// 1. Unified Encoder/Decoder Contract: Single-page (`fused_encode_cold_page_cpu`),
//    multi-threaded allocated batch (`fused_encode_cold_pages_batch_cpu`), and
//    zero-allocation GCD batch (`fused_encode_cold_pages_peak_into_cpu`) all execute the
//    exact same per-page quantization + RFC 8439 ChaCha20 + SipHash-2-4-128 pipeline and
//    produce bit-for-bit identical cold-page blobs that verify and round-trip through
//    `fused_decode_cold_page_cpu`.
// 2. Standard 256-Bit Counter-Mode ARX Stream Cipher (RFC 8439 ChaCha20):
//    Uses all 256 bits (32 bytes = k0, k1, k2, k3) of the session encryption key, a 96-bit
//    per-page nonce bound to the AAD header, and 4-block transposed ARM NEON vectorization
//    (`chacha20_4blocks_rfc8439`). Immune to known-plaintext key recovery.
// 3. Complete Ordered Payload + Scale + AAD Protection (Standard SipHash-2-4-128):
//    Encrypt-then-MAC over the 32-byte AAD header followed by every byte of the ordered
//    ciphertext payload (quantized RoPE, RoPE FP16 scales, quantized NoPE, NoPE FP16 scales,
//    quantized V, V FP16 scales, and alignment tail padding).
// 4. Execution Pass Transparency (Single-Pass Quantized Vectors + Section-Staged Scales):
//    Quantized values (`q_rope`, `q_nope`, `q_v`, 83.3% of payload bytes on GQA INT4) are
//    quantized in ARMv8.5-A FP16 NEON registers, XORed in-register with the ChaCha20 keystream,
//    stored once to the output buffer, and absorbed in-register into sequential SipHash-2-4-128.
//    Because the wire format places each section's 2-byte FP16 scale array contiguously after
//    its quantized array (`[q_section][s_section]`), the per-group FP16 scales (16.7% of payload,
//    4-8 KB per section) are staged in a stack-resident L1 buffer during each section's loop and
//    encrypted + SipHash-absorbed immediately at the end of that section.

#include <torch/extension.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace {

constexpr uint16_t MAGIC_ZTCP = 0x5A54;
constexpr uint8_t CODEC_VERSION = 1;
constexpr size_t HEADER_BYTES = 32;
constexpr size_t TAG_BYTES = 16;

#pragma pack(push, 1)
struct ColdPageHeaderRaw {
    uint16_t magic;
    uint8_t version;
    uint8_t flags;
    uint32_t tenant_id_hash;
    uint32_t model_id_hash;
    uint16_t layer_id;
    uint16_t head_group_id;
    uint64_t page_token_hash;
    uint64_t sequence_nonce;
};
#pragma pack(pop)

static_assert(sizeof(ColdPageHeaderRaw) == HEADER_BYTES, "Header must be 32 bytes");

inline uint32_t rotl32(uint32_t x, int b) noexcept {
    return (x << b) | (x >> (32 - b));
}

inline uint64_t rotl64(uint64_t x, int b) noexcept {
    return (x << b) | (x >> (64 - b));
}

inline void unpack_key256_words(
    uint64_t k0,
    uint64_t k1,
    uint64_t k2,
    uint64_t k3,
    uint32_t out_key[8]
) noexcept {
    out_key[0] = static_cast<uint32_t>(k0 & 0xFFFFFFFFULL);
    out_key[1] = static_cast<uint32_t>(k0 >> 32);
    out_key[2] = static_cast<uint32_t>(k1 & 0xFFFFFFFFULL);
    out_key[3] = static_cast<uint32_t>(k1 >> 32);
    out_key[4] = static_cast<uint32_t>(k2 & 0xFFFFFFFFULL);
    out_key[5] = static_cast<uint32_t>(k2 >> 32);
    out_key[6] = static_cast<uint32_t>(k3 & 0xFFFFFFFFULL);
    out_key[7] = static_cast<uint32_t>(k3 >> 32);
}

inline void derive_chacha20_nonce_words(
    const ColdPageHeaderRaw& hdr,
    uint32_t out_nonce[3]
) noexcept {
    out_nonce[0] = static_cast<uint32_t>(hdr.sequence_nonce & 0xFFFFFFFFULL);
    out_nonce[1] = static_cast<uint32_t>(hdr.sequence_nonce >> 32);
    out_nonce[2] = (static_cast<uint32_t>(hdr.layer_id) << 16) |
                   static_cast<uint32_t>(hdr.head_group_id);
}

// Standard IETF RFC 8439 Section 2.3 ChaCha20 64-byte block function (256-bit key, 32-bit counter, 96-bit nonce)
inline void chacha20_block_rfc8439(
    const uint32_t key[8],
    uint32_t counter,
    const uint32_t nonce[3],
    uint8_t out_block[64]
) noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    uint32x4_t r0 = {0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U};
    uint32x4_t r1 = vld1q_u32(key);
    uint32x4_t r2 = vld1q_u32(key + 4);
    uint32x4_t r3 = {counter, nonce[0], nonce[1], nonce[2]};
    const uint32x4_t init0 = r0;
    const uint32x4_t init1 = r1;
    const uint32x4_t init2 = r2;
    const uint32x4_t init3 = r3;

    for (int i = 0; i < 10; ++i) {
        // Column round
        r0 = vaddq_u32(r0, r1); r3 = veorq_u32(r3, r0); r3 = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(r3)));
        r2 = vaddq_u32(r2, r3); r1 = veorq_u32(r1, r2); r1 = vsriq_n_u32(vshlq_n_u32(r1, 12), r1, 20);
        r0 = vaddq_u32(r0, r1); r3 = veorq_u32(r3, r0); r3 = vsriq_n_u32(vshlq_n_u32(r3, 8), r3, 24);
        r2 = vaddq_u32(r2, r3); r1 = veorq_u32(r1, r2); r1 = vsriq_n_u32(vshlq_n_u32(r1, 7), r1, 25);

        // Diagonal shift
        r1 = vextq_u32(r1, r1, 1);
        r2 = vextq_u32(r2, r2, 2);
        r3 = vextq_u32(r3, r3, 3);

        // Diagonal round
        r0 = vaddq_u32(r0, r1); r3 = veorq_u32(r3, r0); r3 = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(r3)));
        r2 = vaddq_u32(r2, r3); r1 = veorq_u32(r1, r2); r1 = vsriq_n_u32(vshlq_n_u32(r1, 12), r1, 20);
        r0 = vaddq_u32(r0, r1); r3 = veorq_u32(r3, r0); r3 = vsriq_n_u32(vshlq_n_u32(r3, 8), r3, 24);
        r2 = vaddq_u32(r2, r3); r1 = veorq_u32(r1, r2); r1 = vsriq_n_u32(vshlq_n_u32(r1, 7), r1, 25);

        // Undo diagonal shift
        r1 = vextq_u32(r1, r1, 3);
        r2 = vextq_u32(r2, r2, 2);
        r3 = vextq_u32(r3, r3, 1);
    }

    vst1q_u32(reinterpret_cast<uint32_t*>(out_block), vaddq_u32(r0, init0));
    vst1q_u32(reinterpret_cast<uint32_t*>(out_block + 16), vaddq_u32(r1, init1));
    vst1q_u32(reinterpret_cast<uint32_t*>(out_block + 32), vaddq_u32(r2, init2));
    vst1q_u32(reinterpret_cast<uint32_t*>(out_block + 48), vaddq_u32(r3, init3));
#else
    uint32_t state[16] = {
        0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        counter, nonce[0], nonce[1], nonce[2]
    };
    uint32_t x[16];
    std::memcpy(x, state, sizeof(state));

    auto qr = [](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) noexcept {
        a += b; d = rotl32(d ^ a, 16);
        c += d; b = rotl32(b ^ c, 12);
        a += b; d = rotl32(d ^ a, 8);
        c += d; b = rotl32(b ^ c, 7);
    };

    for (int i = 0; i < 10; ++i) {
        qr(x[0], x[4], x[8], x[12]);
        qr(x[1], x[5], x[9], x[13]);
        qr(x[2], x[6], x[10], x[14]);
        qr(x[3], x[7], x[11], x[15]);
        qr(x[0], x[5], x[10], x[15]);
        qr(x[1], x[6], x[11], x[12]);
        qr(x[2], x[7], x[8], x[13]);
        qr(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; ++i) {
        x[i] += state[i];
    }
    std::memcpy(out_block, x, 64);
#endif
}

// 4-block transposed ARM NEON RFC 8439 ChaCha20 generator (256 bytes per call)
inline void chacha20_4blocks_rfc8439(
    const uint32_t key[8],
    uint32_t counter,
    const uint32_t nonce[3],
    uint8_t out_blocks[256]
) noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint32x4_t c0 = vdupq_n_u32(0x61707865U);
    const uint32x4_t c1 = vdupq_n_u32(0x3320646eU);
    const uint32x4_t c2 = vdupq_n_u32(0x79622d32U);
    const uint32x4_t c3 = vdupq_n_u32(0x6b206574U);
    const uint32x4_t k0 = vdupq_n_u32(key[0]);
    const uint32x4_t k1 = vdupq_n_u32(key[1]);
    const uint32x4_t k2 = vdupq_n_u32(key[2]);
    const uint32x4_t k3 = vdupq_n_u32(key[3]);
    const uint32x4_t k4 = vdupq_n_u32(key[4]);
    const uint32x4_t k5 = vdupq_n_u32(key[5]);
    const uint32x4_t k6 = vdupq_n_u32(key[6]);
    const uint32x4_t k7 = vdupq_n_u32(key[7]);
    const uint32x4_t ctr = {counter, counter + 1, counter + 2, counter + 3};
    const uint32x4_t n0 = vdupq_n_u32(nonce[0]);
    const uint32x4_t n1 = vdupq_n_u32(nonce[1]);
    const uint32x4_t n2 = vdupq_n_u32(nonce[2]);

    uint32x4_t s0 = c0, s1 = c1, s2 = c2, s3 = c3;
    uint32x4_t s4 = k0, s5 = k1, s6 = k2, s7 = k3;
    uint32x4_t s8 = k4, s9 = k5, s10 = k6, s11 = k7;
    uint32x4_t s12 = ctr, s13 = n0, s14 = n1, s15 = n2;

    #define ZTCP_CHACHA_QR4(a, b, c, d) do { \
        a = vaddq_u32(a, b); d = veorq_u32(d, a); d = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(d))); \
        c = vaddq_u32(c, d); b = veorq_u32(b, c); b = vsriq_n_u32(vshlq_n_u32(b, 12), b, 20); \
        a = vaddq_u32(a, b); d = veorq_u32(d, a); d = vsriq_n_u32(vshlq_n_u32(d, 8), d, 24); \
        c = vaddq_u32(c, d); b = veorq_u32(b, c); b = vsriq_n_u32(vshlq_n_u32(b, 7), b, 25); \
    } while (0)

    for (int i = 0; i < 10; ++i) {
        ZTCP_CHACHA_QR4(s0, s4, s8, s12);
        ZTCP_CHACHA_QR4(s1, s5, s9, s13);
        ZTCP_CHACHA_QR4(s2, s6, s10, s14);
        ZTCP_CHACHA_QR4(s3, s7, s11, s15);
        ZTCP_CHACHA_QR4(s0, s5, s10, s15);
        ZTCP_CHACHA_QR4(s1, s6, s11, s12);
        ZTCP_CHACHA_QR4(s2, s7, s8, s13);
        ZTCP_CHACHA_QR4(s3, s4, s9, s14);
    }
    #undef ZTCP_CHACHA_QR4

    s0 = vaddq_u32(s0, c0); s1 = vaddq_u32(s1, c1); s2 = vaddq_u32(s2, c2); s3 = vaddq_u32(s3, c3);
    s4 = vaddq_u32(s4, k0); s5 = vaddq_u32(s5, k1); s6 = vaddq_u32(s6, k2); s7 = vaddq_u32(s7, k3);
    s8 = vaddq_u32(s8, k4); s9 = vaddq_u32(s9, k5); s10 = vaddq_u32(s10, k6); s11 = vaddq_u32(s11, k7);
    s12 = vaddq_u32(s12, ctr); s13 = vaddq_u32(s13, n0); s14 = vaddq_u32(s14, n1); s15 = vaddq_u32(s15, n2);

    #define ZTCP_TRANSPOSE_STORE_4X4(a0, a1, a2, a3, word_offset) do { \
        uint32x4x2_t t01 = vtrnq_u32(a0, a1); \
        uint32x4x2_t t23 = vtrnq_u32(a2, a3); \
        uint32x4_t b0 = vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0])); \
        uint32x4_t b1 = vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1])); \
        uint32x4_t b2 = vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0])); \
        uint32x4_t b3 = vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1])); \
        vst1q_u32(reinterpret_cast<uint32_t*>(out_blocks + 0 * 64 + word_offset), b0); \
        vst1q_u32(reinterpret_cast<uint32_t*>(out_blocks + 1 * 64 + word_offset), b1); \
        vst1q_u32(reinterpret_cast<uint32_t*>(out_blocks + 2 * 64 + word_offset), b2); \
        vst1q_u32(reinterpret_cast<uint32_t*>(out_blocks + 3 * 64 + word_offset), b3); \
    } while (0)

    ZTCP_TRANSPOSE_STORE_4X4(s0, s1, s2, s3, 0);
    ZTCP_TRANSPOSE_STORE_4X4(s4, s5, s6, s7, 16);
    ZTCP_TRANSPOSE_STORE_4X4(s8, s9, s10, s11, 32);
    ZTCP_TRANSPOSE_STORE_4X4(s12, s13, s14, s15, 48);
    #undef ZTCP_TRANSPOSE_STORE_4X4
#else
    for (uint32_t b = 0; b < 4; ++b) {
        chacha20_block_rfc8439(key, counter + b, nonce, out_blocks + b * 64);
    }
#endif
}

// Direct in-register 4-block ChaCha20 XOR (avoids 256B ks_buf stack store/load round-trip)
inline void chacha20_4blocks_xor_direct_rfc8439(
    const uint32_t key[8],
    uint32_t counter,
    const uint32_t nonce[3],
    const uint8_t* __restrict__ src256,
    uint8_t* __restrict__ dst256
) noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint32x4_t c0 = vdupq_n_u32(0x61707865U);
    const uint32x4_t c1 = vdupq_n_u32(0x3320646eU);
    const uint32x4_t c2 = vdupq_n_u32(0x79622d32U);
    const uint32x4_t c3 = vdupq_n_u32(0x6b206574U);
    const uint32x4_t k0 = vdupq_n_u32(key[0]);
    const uint32x4_t k1 = vdupq_n_u32(key[1]);
    const uint32x4_t k2 = vdupq_n_u32(key[2]);
    const uint32x4_t k3 = vdupq_n_u32(key[3]);
    const uint32x4_t k4 = vdupq_n_u32(key[4]);
    const uint32x4_t k5 = vdupq_n_u32(key[5]);
    const uint32x4_t k6 = vdupq_n_u32(key[6]);
    const uint32x4_t k7 = vdupq_n_u32(key[7]);
    const uint32x4_t ctr = {counter, counter + 1, counter + 2, counter + 3};
    const uint32x4_t n0 = vdupq_n_u32(nonce[0]);
    const uint32x4_t n1 = vdupq_n_u32(nonce[1]);
    const uint32x4_t n2 = vdupq_n_u32(nonce[2]);

    uint32x4_t s0 = c0, s1 = c1, s2 = c2, s3 = c3;
    uint32x4_t s4 = k0, s5 = k1, s6 = k2, s7 = k3;
    uint32x4_t s8 = k4, s9 = k5, s10 = k6, s11 = k7;
    uint32x4_t s12 = ctr, s13 = n0, s14 = n1, s15 = n2;

    #define ZTCP_CHACHA_QR4(a, b, c, d) do { \
        a = vaddq_u32(a, b); d = veorq_u32(d, a); d = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(d))); \
        c = vaddq_u32(c, d); b = veorq_u32(b, c); b = vsriq_n_u32(vshlq_n_u32(b, 12), b, 20); \
        a = vaddq_u32(a, b); d = veorq_u32(d, a); d = vsriq_n_u32(vshlq_n_u32(d, 8), d, 24); \
        c = vaddq_u32(c, d); b = veorq_u32(b, c); b = vsriq_n_u32(vshlq_n_u32(b, 7), b, 25); \
    } while (0)

    for (int i = 0; i < 10; ++i) {
        ZTCP_CHACHA_QR4(s0, s4, s8, s12);
        ZTCP_CHACHA_QR4(s1, s5, s9, s13);
        ZTCP_CHACHA_QR4(s2, s6, s10, s14);
        ZTCP_CHACHA_QR4(s3, s7, s11, s15);
        ZTCP_CHACHA_QR4(s0, s5, s10, s15);
        ZTCP_CHACHA_QR4(s1, s6, s11, s12);
        ZTCP_CHACHA_QR4(s2, s7, s8, s13);
        ZTCP_CHACHA_QR4(s3, s4, s9, s14);
    }
    #undef ZTCP_CHACHA_QR4

    s0 = vaddq_u32(s0, c0); s1 = vaddq_u32(s1, c1); s2 = vaddq_u32(s2, c2); s3 = vaddq_u32(s3, c3);
    s4 = vaddq_u32(s4, k0); s5 = vaddq_u32(s5, k1); s6 = vaddq_u32(s6, k2); s7 = vaddq_u32(s7, k3);
    s8 = vaddq_u32(s8, k4); s9 = vaddq_u32(s9, k5); s10 = vaddq_u32(s10, k6); s11 = vaddq_u32(s11, k7);
    s12 = vaddq_u32(s12, ctr); s13 = vaddq_u32(s13, n0); s14 = vaddq_u32(s14, n1); s15 = vaddq_u32(s15, n2);

    #define ZTCP_TRANSPOSE_XOR_STORE_4X4(a0, a1, a2, a3, word_offset) do { \
        uint32x4x2_t t01 = vtrnq_u32(a0, a1); \
        uint32x4x2_t t23 = vtrnq_u32(a2, a3); \
        uint32x4_t b0 = vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0])); \
        uint32x4_t b1 = vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1])); \
        uint32x4_t b2 = vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0])); \
        uint32x4_t b3 = vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1])); \
        uint32x4_t p0 = vld1q_u32(reinterpret_cast<const uint32_t*>(src256 + 0 * 64 + word_offset)); \
        uint32x4_t p1 = vld1q_u32(reinterpret_cast<const uint32_t*>(src256 + 1 * 64 + word_offset)); \
        uint32x4_t p2 = vld1q_u32(reinterpret_cast<const uint32_t*>(src256 + 2 * 64 + word_offset)); \
        uint32x4_t p3 = vld1q_u32(reinterpret_cast<const uint32_t*>(src256 + 3 * 64 + word_offset)); \
        vst1q_u32(reinterpret_cast<uint32_t*>(dst256 + 0 * 64 + word_offset), veorq_u32(p0, b0)); \
        vst1q_u32(reinterpret_cast<uint32_t*>(dst256 + 1 * 64 + word_offset), veorq_u32(p1, b1)); \
        vst1q_u32(reinterpret_cast<uint32_t*>(dst256 + 2 * 64 + word_offset), veorq_u32(p2, b2)); \
        vst1q_u32(reinterpret_cast<uint32_t*>(dst256 + 3 * 64 + word_offset), veorq_u32(p3, b3)); \
    } while (0)

    ZTCP_TRANSPOSE_XOR_STORE_4X4(s0, s1, s2, s3, 0);
    ZTCP_TRANSPOSE_XOR_STORE_4X4(s4, s5, s6, s7, 16);
    ZTCP_TRANSPOSE_XOR_STORE_4X4(s8, s9, s10, s11, 32);
    ZTCP_TRANSPOSE_XOR_STORE_4X4(s12, s13, s14, s15, 48);
    #undef ZTCP_TRANSPOSE_XOR_STORE_4X4
#else
    alignas(16) uint8_t ks_buf[256];
    chacha20_4blocks_rfc8439(key, counter, nonce, ks_buf);
    for (size_t j = 0; j < 256; ++j) {
        dst256[j] = src256[j] ^ ks_buf[j];
    }
#endif
}

inline void chacha20_xor_bytes_rfc8439(
    const uint8_t* src,
    uint8_t* dst,
    size_t len,
    const uint32_t key[8],
    uint32_t initial_counter,
    const uint32_t nonce[3]
) noexcept {
    alignas(16) uint8_t ks_buf[256];
    uint32_t counter = initial_counter;
    size_t offset = 0;

    while (len - offset >= 256) {
        chacha20_4blocks_xor_direct_rfc8439(key, counter, nonce, src + offset, dst + offset);
        counter += 4;
        offset += 256;
    }

    while (len - offset >= 64) {
        chacha20_block_rfc8439(key, counter, nonce, ks_buf);
        counter += 1;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (size_t j = 0; j < 64; j += 16) {
            uint8x16_t p = vld1q_u8(src + offset + j);
            uint8x16_t k = vld1q_u8(ks_buf + j);
            vst1q_u8(dst + offset + j, veorq_u8(p, k));
        }
#else
        for (size_t j = 0; j < 64; ++j) {
            dst[offset + j] = src[offset + j] ^ ks_buf[j];
        }
#endif
        offset += 64;
    }

    if (offset < len) {
        chacha20_block_rfc8439(key, counter, nonce, ks_buf);
        size_t rem = len - offset;
        for (size_t j = 0; j < rem; ++j) {
            dst[offset + j] = src[offset + j] ^ ks_buf[j];
        }
    }
}

// Standard 128-bit SipHash-2-4 (Aumasson & Bernstein reference specification)
struct SipMac128State {
    uint64_t v0, v1, v2, v3;

    inline void init(uint64_t mk0, uint64_t mk1) noexcept {
        v0 = 0x736f6d6570736575ULL ^ mk0;
        v1 = 0x646f72616e646f6dULL ^ mk1 ^ 0xeeULL;
        v2 = 0x6c7967656e657261ULL ^ mk0;
        v3 = 0x7465646279746573ULL ^ mk1;
    }

    inline void sipround() noexcept {
        v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
        v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
        v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
        v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
    }

    inline void absorb_word(uint64_t m) noexcept {
        v3 ^= m;
        sipround();
        sipround();
        v0 ^= m;
    }

    inline void absorb_bytes(const uint8_t* data, size_t len) noexcept {
        size_t n_words = len / 8;
        const uint64_t* w_ptr = reinterpret_cast<const uint64_t*>(data);
        size_t i = 0;
        for (; i + 3 < n_words; i += 4) {
            absorb_word(w_ptr[i]);
            absorb_word(w_ptr[i + 1]);
            absorb_word(w_ptr[i + 2]);
            absorb_word(w_ptr[i + 3]);
        }
        for (; i < n_words; ++i) {
            absorb_word(w_ptr[i]);
        }
        size_t rem = len & 7;
        uint64_t last = static_cast<uint64_t>(len & 0xff) << 56;
        const uint8_t* tail = data + n_words * 8;
        for (size_t b = 0; b < rem; ++b) {
            last |= static_cast<uint64_t>(tail[b]) << (8 * b);
        }
        absorb_word(last);
    }

    inline void finalize(uint8_t out_tag[16]) noexcept {
        v2 ^= 0xeeULL;
        sipround(); sipround(); sipround(); sipround();
        uint64_t tag_lo = v0 ^ v1 ^ v2 ^ v3;
        v1 ^= 0xddULL;
        sipround(); sipround(); sipround(); sipround();
        uint64_t tag_hi = v0 ^ v1 ^ v2 ^ v3;
        std::memcpy(out_tag, &tag_lo, 8);
        std::memcpy(out_tag + 8, &tag_hi, 8);
    }

    inline void finalize_with_tail(
        size_t total_len,
        const uint8_t* tail,
        size_t rem,
        uint8_t out_tag[16]
    ) noexcept {
        uint64_t last = static_cast<uint64_t>(total_len & 0xff) << 56;
        for (size_t b = 0; b < rem; ++b) {
            last |= static_cast<uint64_t>(tail[b]) << (8 * b);
        }
        absorb_word(last);
        finalize(out_tag);
    }
};

// Streaming Encrypt-then-MAC cursor combining 4-block RFC 8439 ChaCha20 with sequential SipHash-2-4-128
struct ChaCha20SipMacStream {
    uint32_t key_words[8];
    uint32_t nonce_words[3];
    uint32_t next_counter;
    alignas(16) uint8_t ks_buf[256];
    size_t ks_pos;
    SipMac128State mac;
    uint8_t mac_tail[8];
    size_t mac_tail_len;
    size_t total_mac_bytes;

    inline void init(
        uint64_t k0,
        uint64_t k1,
        uint64_t k2,
        uint64_t k3,
        uint64_t mk0,
        uint64_t mk1,
        const ColdPageHeaderRaw& hdr,
        const uint8_t* hdr_bytes
    ) noexcept {
        unpack_key256_words(k0, k1, k2, k3, key_words);
        derive_chacha20_nonce_words(hdr, nonce_words);
        next_counter = 1U;
        ks_pos = 256;
        mac.init(mk0, mk1);
        for (int i = 0; i < 4; ++i) {
            uint64_t w;
            std::memcpy(&w, hdr_bytes + i * 8, 8);
            mac.absorb_word(w);
        }
        mac_tail_len = 0;
        total_mac_bytes = HEADER_BYTES;
    }

    inline void refill_ks() noexcept {
        chacha20_4blocks_rfc8439(key_words, next_counter, nonce_words, ks_buf);
        next_counter += 4U;
        ks_pos = 0;
    }

    inline void encrypt_store_and_absorb_bytes_scalar(
        const uint8_t* src,
        uint8_t* dst,
        size_t len
    ) noexcept {
        for (size_t idx = 0; idx < len; ++idx) {
            if (ks_pos >= 256) {
                refill_ks();
            }
            uint8_t ct_b = src[idx] ^ ks_buf[ks_pos++];
            dst[idx] = ct_b;
            mac_tail[mac_tail_len++] = ct_b;
            total_mac_bytes += 1;
            if (mac_tail_len == 8) {
                uint64_t w;
                std::memcpy(&w, mac_tail, 8);
                mac.absorb_word(w);
                mac_tail_len = 0;
            }
        }
    }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // Fast path for 16-byte register-resident quantized vectors when aligned (ks_pos % 16 == 0, 256 - ks_pos >= 16, mac_tail_len == 0).
    // Falls back to the byte-exact path whenever a partial MAC word is pending, keystream cursor is unaligned, or <16B remain in ks_buf.
    inline void encrypt_store_and_absorb_16B(
        uint64x2_t plain128,
        uint8_t* __restrict__ dst
    ) noexcept {
        if (__builtin_expect(ks_pos >= 256, 0)) {
            refill_ks();
        }
        if (__builtin_expect(mac_tail_len != 0 || (ks_pos & 15) != 0 || (256 - ks_pos) < 16, 0)) {
            alignas(16) uint8_t plain_bytes[16];
            vst1q_u8(plain_bytes, vreinterpretq_u8_u64(plain128));
            encrypt_store_and_absorb_bytes_scalar(plain_bytes, dst, 16);
            return;
        }
        uint64x2_t ks = vld1q_u64(reinterpret_cast<const uint64_t*>(ks_buf + ks_pos));
        ks_pos += 16;
        uint64x2_t ct = veorq_u64(plain128, ks);
        vst1q_u8(dst, vreinterpretq_u8_u64(ct));
        mac.absorb_word(vgetq_lane_u64(ct, 0));
        mac.absorb_word(vgetq_lane_u64(ct, 1));
        total_mac_bytes += 16;
    }

    // Branchless-assuming 16B absorber when caller guarantees mac_tail_len == 0 && (ks_pos & 15) == 0 && ks_pos + 16 <= 256
    inline void absorb_16B_unchecked(
        uint64x2_t plain128,
        uint8_t* __restrict__ dst
    ) noexcept {
        uint64x2_t ks = vld1q_u64(reinterpret_cast<const uint64_t*>(ks_buf + ks_pos));
        ks_pos += 16;
        uint64x2_t ct = veorq_u64(plain128, ks);
        vst1q_u8(dst, vreinterpretq_u8_u64(ct));
        mac.absorb_word(vgetq_lane_u64(ct, 0));
        mac.absorb_word(vgetq_lane_u64(ct, 1));
        total_mac_bytes += 16;
    }

    // Fast path for 32-byte (two 16B vectors) when aligned (mac_tail_len == 0 && (ks_pos & 31) == 0)
    inline void encrypt_store_and_absorb_32B(
        uint64x2_t plain0,
        uint64x2_t plain1,
        uint8_t* __restrict__ dst
    ) noexcept {
        if (__builtin_expect(ks_pos >= 256, 0)) {
            refill_ks();
        }
        if (__builtin_expect(mac_tail_len != 0 || (ks_pos & 31) != 0, 0)) {
            encrypt_store_and_absorb_16B(plain0, dst);
            encrypt_store_and_absorb_16B(plain1, dst + 16);
            return;
        }
        const uint64_t* ks_ptr = reinterpret_cast<const uint64_t*>(ks_buf + ks_pos);
        uint64x2_t ks0 = vld1q_u64(ks_ptr);
        uint64x2_t ks1 = vld1q_u64(ks_ptr + 2);
        ks_pos += 32;
        uint64x2_t ct0 = veorq_u64(plain0, ks0);
        uint64x2_t ct1 = veorq_u64(plain1, ks1);
        vst1q_u8(dst, vreinterpretq_u8_u64(ct0));
        vst1q_u8(dst + 16, vreinterpretq_u8_u64(ct1));
        mac.absorb_word(vgetq_lane_u64(ct0, 0));
        mac.absorb_word(vgetq_lane_u64(ct0, 1));
        mac.absorb_word(vgetq_lane_u64(ct1, 0));
        mac.absorb_word(vgetq_lane_u64(ct1, 1));
        total_mac_bytes += 32;
    }

    // Fast path for 64-byte (four 16B vectors) when aligned (mac_tail_len == 0 && (ks_pos & 63) == 0)
    inline void encrypt_store_and_absorb_64B(
        uint64x2_t p0,
        uint64x2_t p1,
        uint64x2_t p2,
        uint64x2_t p3,
        uint8_t* __restrict__ dst
    ) noexcept {
        if (__builtin_expect(ks_pos >= 256, 0)) {
            refill_ks();
        }
        if (__builtin_expect(mac_tail_len != 0 || (ks_pos & 63) != 0, 0)) {
            encrypt_store_and_absorb_32B(p0, p1, dst);
            encrypt_store_and_absorb_32B(p2, p3, dst + 32);
            return;
        }
        const uint64_t* ks_ptr = reinterpret_cast<const uint64_t*>(ks_buf + ks_pos);
        uint64x2_t ks0 = vld1q_u64(ks_ptr);
        uint64x2_t ks1 = vld1q_u64(ks_ptr + 2);
        uint64x2_t ks2 = vld1q_u64(ks_ptr + 4);
        uint64x2_t ks3 = vld1q_u64(ks_ptr + 6);
        ks_pos += 64;
        uint64x2_t ct0 = veorq_u64(p0, ks0);
        uint64x2_t ct1 = veorq_u64(p1, ks1);
        uint64x2_t ct2 = veorq_u64(p2, ks2);
        uint64x2_t ct3 = veorq_u64(p3, ks3);
        vst1q_u8(dst, vreinterpretq_u8_u64(ct0));
        vst1q_u8(dst + 16, vreinterpretq_u8_u64(ct1));
        vst1q_u8(dst + 32, vreinterpretq_u8_u64(ct2));
        vst1q_u8(dst + 48, vreinterpretq_u8_u64(ct3));
        mac.absorb_word(vgetq_lane_u64(ct0, 0));
        mac.absorb_word(vgetq_lane_u64(ct0, 1));
        mac.absorb_word(vgetq_lane_u64(ct1, 0));
        mac.absorb_word(vgetq_lane_u64(ct1, 1));
        mac.absorb_word(vgetq_lane_u64(ct2, 0));
        mac.absorb_word(vgetq_lane_u64(ct2, 1));
        mac.absorb_word(vgetq_lane_u64(ct3, 0));
        mac.absorb_word(vgetq_lane_u64(ct3, 1));
        total_mac_bytes += 64;
    }
#endif

    inline void encrypt_store_and_absorb_bytes(
        const uint8_t* src,
        uint8_t* dst,
        size_t len
    ) noexcept {
        size_t idx = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (__builtin_expect(mac_tail_len == 0 && (ks_pos & 15) == 0, 1)) {
            while (ks_pos < 256 && (len - idx) >= 16) {
                uint64x2_t p = vreinterpretq_u64_u8(vld1q_u8(src + idx));
                uint64x2_t ks = vld1q_u64(reinterpret_cast<const uint64_t*>(ks_buf + ks_pos));
                ks_pos += 16;
                uint64x2_t ct = veorq_u64(p, ks);
                vst1q_u8(dst + idx, vreinterpretq_u8_u64(ct));
                mac.absorb_word(vgetq_lane_u64(ct, 0));
                mac.absorb_word(vgetq_lane_u64(ct, 1));
                total_mac_bytes += 16;
                idx += 16;
            }
            while ((len - idx) >= 256) {
                chacha20_4blocks_rfc8439(key_words, next_counter, nonce_words, ks_buf);
                next_counter += 4U;
                const uint8_t* s_ptr = src + idx;
                uint8_t* d_ptr = dst + idx;
                for (size_t j = 0; j < 256; j += 32) {
                    uint64x2_t p0 = vreinterpretq_u64_u8(vld1q_u8(s_ptr + j));
                    uint64x2_t p1 = vreinterpretq_u64_u8(vld1q_u8(s_ptr + j + 16));
                    uint64x2_t ks0 = vld1q_u64(reinterpret_cast<const uint64_t*>(ks_buf + j));
                    uint64x2_t ks1 = vld1q_u64(reinterpret_cast<const uint64_t*>(ks_buf + j + 16));
                    uint64x2_t ct0 = veorq_u64(p0, ks0);
                    uint64x2_t ct1 = veorq_u64(p1, ks1);
                    vst1q_u8(d_ptr + j, vreinterpretq_u8_u64(ct0));
                    vst1q_u8(d_ptr + j + 16, vreinterpretq_u8_u64(ct1));
                    mac.absorb_word(vgetq_lane_u64(ct0, 0));
                    mac.absorb_word(vgetq_lane_u64(ct0, 1));
                    mac.absorb_word(vgetq_lane_u64(ct1, 0));
                    mac.absorb_word(vgetq_lane_u64(ct1, 1));
                }
                ks_pos = 256;
                total_mac_bytes += 256;
                idx += 256;
            }
        }
        while (len - idx >= 16) {
            if (ks_pos >= 256) {
                refill_ks();
            }
            if (mac_tail_len != 0 || (ks_pos & 15) != 0 || (256 - ks_pos) < 16) {
                break;
            }
            uint64x2_t p = vreinterpretq_u64_u8(vld1q_u8(src + idx));
            encrypt_store_and_absorb_16B(p, dst + idx);
            idx += 16;
        }
#endif
        if (idx < len) {
            encrypt_store_and_absorb_bytes_scalar(src + idx, dst + idx, len - idx);
        }
    }

    inline void finalize(uint8_t out_tag[16]) noexcept {
        mac.finalize_with_tail(total_mac_bytes, mac_tail, mac_tail_len, out_tag);
    }
};

// Quantize a single 16-element FP16 group into 16 INT8 bytes + 2B FP16 scale
inline void quantize_group16_int8(
    const at::Half* __restrict__ src,
    uint8_t* __restrict__ dst_q,
    at::Half* __restrict__ dst_scale
) noexcept {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const float16x8_t v0 = vld1q_f16(reinterpret_cast<const __fp16*>(src));
    const float16x8_t v1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 8));
    const float16x8_t abs_max_vec = vmaxq_f16(vabsq_f16(v0), vabsq_f16(v1));
    __fp16 amax_h = vmaxvq_f16(abs_max_vec);
    float amax = std::max(static_cast<float>(amax_h), 1e-6f);
    __fp16 scale_fp16 = static_cast<__fp16>(amax * (1.0f / 127.0f));
    std::memcpy(dst_scale, &scale_fp16, sizeof(__fp16));
    __fp16 inv_s_h = static_cast<__fp16>(1.0f / static_cast<float>(scale_fp16));

    float16x8_t inv_vec = vdupq_n_f16(inv_s_h);
    int16x8_t s01 = vcvtnq_s16_f16(vmulq_f16(v0, inv_vec));
    int16x8_t s23 = vcvtnq_s16_f16(vmulq_f16(v1, inv_vec));
    int8x16_t q8 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
    q8 = vmaxq_s8(q8, vdupq_n_s8(-127));
    vst1q_u8(dst_q, vreinterpretq_u8_s8(q8));
#else
    float amax = 1e-6f;
    float vals[16];
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(src[i]);
        vals[i] = v;
        float av = std::fabs(v);
        if (av > amax) amax = av;
    }
    at::Half scale_h = static_cast<at::Half>(amax / 127.0f);
    *dst_scale = scale_h;
    float inv_scale = 1.0f / static_cast<float>(scale_h);

    for (int i = 0; i < 16; ++i) {
        float qf = std::nearbyintf(vals[i] * inv_scale);
        if (qf < -127.0f) qf = -127.0f;
        if (qf > 127.0f) qf = 127.0f;
        int8_t qi = static_cast<int8_t>(qf);
        dst_q[i] = static_cast<uint8_t>(qi);
    }
#endif
}

inline void dequantize_group16_int8(
    const uint8_t* __restrict__ src_q,
    at::Half scale_h,
    at::Half* __restrict__ dst
) noexcept {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 scale_fp16;
    std::memcpy(&scale_fp16, &scale_h, sizeof(__fp16));
    float16x8_t s_vec = vdupq_n_f16(scale_fp16);
    int8x16_t q8 = vreinterpretq_s8_u8(vld1q_u8(src_q));
    int16x8_t lo16 = vmovl_s8(vget_low_s8(q8));
    int16x8_t hi16 = vmovl_s8(vget_high_s8(q8));
    vst1q_f16(reinterpret_cast<__fp16*>(dst), vmulq_f16(vcvtq_f16_s16(lo16), s_vec));
    vst1q_f16(reinterpret_cast<__fp16*>(dst + 8), vmulq_f16(vcvtq_f16_s16(hi16), s_vec));
#else
    float scale = static_cast<float>(scale_h);
    for (int i = 0; i < 16; ++i) {
        int8_t qi = static_cast<int8_t>(src_q[i]);
        dst[i] = static_cast<at::Half>(static_cast<float>(qi) * scale);
    }
#endif
}

// Quantize a single 16-element FP16 group into 8 packed INT4 bytes + 2B FP16 scale
inline void quantize_group16_int4(
    const at::Half* __restrict__ src,
    uint8_t* __restrict__ dst_packed,
    at::Half* __restrict__ dst_scale
) noexcept {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const float16x8_t v0 = vld1q_f16(reinterpret_cast<const __fp16*>(src));
    const float16x8_t v1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 8));
    const float16x8_t abs_max_vec = vmaxq_f16(vabsq_f16(v0), vabsq_f16(v1));
    __fp16 amax_h = vmaxvq_f16(abs_max_vec);
    float amax = std::max(static_cast<float>(amax_h), 1e-6f);
    __fp16 scale_fp16 = static_cast<__fp16>(amax * (1.0f / 7.0f));
    std::memcpy(dst_scale, &scale_fp16, sizeof(__fp16));
    __fp16 inv_s_h = static_cast<__fp16>(1.0f / static_cast<float>(scale_fp16));

    float16x8_t inv_vec = vdupq_n_f16(inv_s_h);
    int16x8_t s01 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(v0, inv_vec)), vdupq_n_s16(8));
    int16x8_t s23 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(v1, inv_vec)), vdupq_n_s16(8));
    uint8x16_t u8 = vcombine_u8(vqmovun_s16(s01), vqmovun_s16(s23));
    u8 = vminq_u8(u8, vdupq_n_u8(15));

    uint8x8x2_t unzipped = vuzp_u8(vget_low_u8(u8), vget_high_u8(u8));
    uint8x8_t packed = vorr_u8(unzipped.val[0], vshl_n_u8(unzipped.val[1], 4));
    vst1_u8(dst_packed, packed);
#else
    float amax = 1e-6f;
    float vals[16];
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(src[i]);
        vals[i] = v;
        float av = std::fabs(v);
        if (av > amax) amax = av;
    }
    at::Half scale_h = static_cast<at::Half>(amax / 7.0f);
    *dst_scale = scale_h;
    float inv_scale = 1.0f / static_cast<float>(scale_h);

    for (int i = 0; i < 8; ++i) {
        float q0 = std::nearbyintf(vals[2 * i] * inv_scale) + 8.0f;
        float q1 = std::nearbyintf(vals[2 * i + 1] * inv_scale) + 8.0f;
        int u0 = std::clamp(static_cast<int>(q0), 0, 15);
        int u1 = std::clamp(static_cast<int>(q1), 0, 15);
        dst_packed[i] = static_cast<uint8_t>((u0 & 0x0F) | ((u1 & 0x0F) << 4));
    }
#endif
}

inline void dequantize_group16_int4(
    const uint8_t* __restrict__ src_packed,
    at::Half scale_h,
    at::Half* __restrict__ dst
) noexcept {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 scale_fp16;
    std::memcpy(&scale_fp16, &scale_h, sizeof(__fp16));
    float16x8_t s_vec = vdupq_n_f16(scale_fp16);
    uint8x8_t packed = vld1_u8(src_packed);
    uint8x8_t lo_nib = vand_u8(packed, vdup_n_u8(0x0F));
    uint8x8_t hi_nib = vshr_n_u8(packed, 4);
    uint8x8x2_t zipped = vzip_u8(lo_nib, hi_nib);
    int16x8_t s_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.val[0])), vdupq_n_s16(8));
    int16x8_t s_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.val[1])), vdupq_n_s16(8));
    vst1q_f16(reinterpret_cast<__fp16*>(dst), vmulq_f16(vcvtq_f16_s16(s_lo), s_vec));
    vst1q_f16(reinterpret_cast<__fp16*>(dst + 8), vmulq_f16(vcvtq_f16_s16(s_hi), s_vec));
#else
    float scale = static_cast<float>(scale_h);
    for (int i = 0; i < 8; ++i) {
        uint8_t b = src_packed[i];
        float v0 = static_cast<float>(b & 0x0F) - 8.0f;
        float v1 = static_cast<float>((b >> 4) & 0x0F) - 8.0f;
        dst[2 * i] = static_cast<at::Half>(v0 * scale);
        dst[2 * i + 1] = static_cast<at::Half>(v1 * scale);
    }
#endif
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void dequantize_2groups32_int4_neon(
    const uint8_t* __restrict__ src_packed16,
    const at::Half* __restrict__ src_scales2,
    at::Half* __restrict__ dst32
) noexcept {
    const __fp16* sc_ptr = reinterpret_cast<const __fp16*>(src_scales2);
    float16x8_t s0_vec = vdupq_n_f16(sc_ptr[0]);
    float16x8_t s1_vec = vdupq_n_f16(sc_ptr[1]);

    uint8x16_t packed = vld1q_u8(src_packed16);
    uint8x16_t lo_nib = vandq_u8(packed, vdupq_n_u8(0x0F));
    uint8x16_t hi_nib = vshrq_n_u8(packed, 4);
    uint8x16x2_t zipped = vzipq_u8(lo_nib, hi_nib);
    const int16x8_t eight16 = vdupq_n_s16(8);

    int16x8_t g0_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(zipped.val[0]))), eight16);
    int16x8_t g0_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(zipped.val[0]))), eight16);
    int16x8_t g1_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(zipped.val[1]))), eight16);
    int16x8_t g1_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(zipped.val[1]))), eight16);

    __fp16* d_ptr = reinterpret_cast<__fp16*>(dst32);
    vst1q_f16(d_ptr, vmulq_f16(vcvtq_f16_s16(g0_0), s0_vec));
    vst1q_f16(d_ptr + 8, vmulq_f16(vcvtq_f16_s16(g0_1), s0_vec));
    vst1q_f16(d_ptr + 16, vmulq_f16(vcvtq_f16_s16(g1_0), s1_vec));
    vst1q_f16(d_ptr + 24, vmulq_f16(vcvtq_f16_s16(g1_1), s1_vec));
}

inline void dequantize_2groups32_int8_neon(
    const uint8_t* __restrict__ src_q32,
    const at::Half* __restrict__ src_scales2,
    at::Half* __restrict__ dst32
) noexcept {
    const __fp16* sc_ptr = reinterpret_cast<const __fp16*>(src_scales2);
    float16x8_t s0_vec = vdupq_n_f16(sc_ptr[0]);
    float16x8_t s1_vec = vdupq_n_f16(sc_ptr[1]);

    int8x16_t q0 = vreinterpretq_s8_u8(vld1q_u8(src_q32));
    int8x16_t q1 = vreinterpretq_s8_u8(vld1q_u8(src_q32 + 16));

    __fp16* d_ptr = reinterpret_cast<__fp16*>(dst32);
    vst1q_f16(d_ptr, vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q0))), s0_vec));
    vst1q_f16(d_ptr + 8, vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(q0))), s0_vec));
    vst1q_f16(d_ptr + 16, vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q1))), s1_vec));
    vst1q_f16(d_ptr + 24, vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(q1))), s1_vec));
}
#endif

// Dual-Group (32 FP16 elements -> 16 packed INT4 bytes in ONE 128-bit NEON register + two FP16 scales)
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline uint64x2_t quantize_2groups32_int4_neon(
    const at::Half* __restrict__ src,
    at::Half* __restrict__ dst_scales2
) noexcept {
    const float16x8_t g0_0 = vld1q_f16(reinterpret_cast<const __fp16*>(src));
    const float16x8_t g0_1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 8));
    const float16x8_t g1_0 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 16));
    const float16x8_t g1_1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 24));

    __fp16 amax0 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g0_0), vabsq_f16(g0_1)));
    __fp16 amax1 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g1_0), vabsq_f16(g1_1)));
    float fmax0 = std::max(static_cast<float>(amax0), 1e-6f);
    float fmax1 = std::max(static_cast<float>(amax1), 1e-6f);

    __fp16 sc0 = static_cast<__fp16>(fmax0 * (1.0f / 7.0f));
    __fp16 sc1 = static_cast<__fp16>(fmax1 * (1.0f / 7.0f));
    reinterpret_cast<__fp16*>(dst_scales2)[0] = sc0;
    reinterpret_cast<__fp16*>(dst_scales2)[1] = sc1;

    float16x8_t inv0 = vdupq_n_f16(static_cast<__fp16>(1.0f / static_cast<float>(sc0)));
    float16x8_t inv1 = vdupq_n_f16(static_cast<__fp16>(1.0f / static_cast<float>(sc1)));
    const int16x8_t eight16 = vdupq_n_s16(8);
    const uint8x16_t fifteen8 = vdupq_n_u8(15);

    int16x8_t q0_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g0_0, inv0)), eight16);
    int16x8_t q0_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g0_1, inv0)), eight16);
    int16x8_t q1_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g1_0, inv1)), eight16);
    int16x8_t q1_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g1_1, inv1)), eight16);

    uint8x16_t u0 = vminq_u8(vcombine_u8(vqmovun_s16(q0_0), vqmovun_s16(q0_1)), fifteen8);
    uint8x16_t u1 = vminq_u8(vcombine_u8(vqmovun_s16(q1_0), vqmovun_s16(q1_1)), fifteen8);

    uint8x16x2_t unzipped = vuzpq_u8(u0, u1);
    uint8x16_t packed16 = vorrq_u8(unzipped.val[0], vshlq_n_u8(unzipped.val[1], 4));
    return vreinterpretq_u64_u8(packed16);
}

// Dual-Group (32 FP16 elements -> 32 INT8 bytes in TWO 128-bit NEON registers + two FP16 scales)
inline void quantize_2groups32_int8_neon(
    const at::Half* __restrict__ src,
    uint64x2_t& out0,
    uint64x2_t& out1,
    at::Half* __restrict__ dst_scales2
) noexcept {
    const float16x8_t g0_0 = vld1q_f16(reinterpret_cast<const __fp16*>(src));
    const float16x8_t g0_1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 8));
    const float16x8_t g1_0 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 16));
    const float16x8_t g1_1 = vld1q_f16(reinterpret_cast<const __fp16*>(src + 24));

    __fp16 amax0 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g0_0), vabsq_f16(g0_1)));
    __fp16 amax1 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g1_0), vabsq_f16(g1_1)));
    float fmax0 = std::max(static_cast<float>(amax0), 1e-6f);
    float fmax1 = std::max(static_cast<float>(amax1), 1e-6f);

    __fp16 sc0 = static_cast<__fp16>(fmax0 * (1.0f / 127.0f));
    __fp16 sc1 = static_cast<__fp16>(fmax1 * (1.0f / 127.0f));
    reinterpret_cast<__fp16*>(dst_scales2)[0] = sc0;
    reinterpret_cast<__fp16*>(dst_scales2)[1] = sc1;

    float16x8_t inv0 = vdupq_n_f16(static_cast<__fp16>(1.0f / static_cast<float>(sc0)));
    float16x8_t inv1 = vdupq_n_f16(static_cast<__fp16>(1.0f / static_cast<float>(sc1)));
    const int8x16_t min127 = vdupq_n_s8(-127);

    int16x8_t s0_0 = vcvtnq_s16_f16(vmulq_f16(g0_0, inv0));
    int16x8_t s0_1 = vcvtnq_s16_f16(vmulq_f16(g0_1, inv0));
    int16x8_t s1_0 = vcvtnq_s16_f16(vmulq_f16(g1_0, inv1));
    int16x8_t s1_1 = vcvtnq_s16_f16(vmulq_f16(g1_1, inv1));

    out0 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s0_0), vqmovn_s16(s0_1)), min127));
    out1 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s1_0), vqmovn_s16(s1_1)), min127));
}

// Quad-Group (64 FP16 elements -> 32 packed INT4 bytes in TWO 128-bit NEON registers + four FP16 scales)
inline void quantize_4groups64_int4_neon(
    const at::Half* __restrict__ src,
    uint64x2_t& out_p0,
    uint64x2_t& out_p1,
    at::Half* __restrict__ dst_scales4
) noexcept {
    const __fp16* s_ptr = reinterpret_cast<const __fp16*>(src);
    const float16x8_t g0_0 = vld1q_f16(s_ptr);
    const float16x8_t g0_1 = vld1q_f16(s_ptr + 8);
    const float16x8_t g1_0 = vld1q_f16(s_ptr + 16);
    const float16x8_t g1_1 = vld1q_f16(s_ptr + 24);
    const float16x8_t g2_0 = vld1q_f16(s_ptr + 32);
    const float16x8_t g2_1 = vld1q_f16(s_ptr + 40);
    const float16x8_t g3_0 = vld1q_f16(s_ptr + 48);
    const float16x8_t g3_1 = vld1q_f16(s_ptr + 56);

    __fp16 amax0 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g0_0), vabsq_f16(g0_1)));
    __fp16 amax1 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g1_0), vabsq_f16(g1_1)));
    __fp16 amax2 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g2_0), vabsq_f16(g2_1)));
    __fp16 amax3 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g3_0), vabsq_f16(g3_1)));

    float32x4_t fmax_v = {
        std::max(static_cast<float>(amax0), 1e-6f),
        std::max(static_cast<float>(amax1), 1e-6f),
        std::max(static_cast<float>(amax2), 1e-6f),
        std::max(static_cast<float>(amax3), 1e-6f)
    };
    float32x4_t sc_f32 = vmulq_n_f32(fmax_v, 1.0f / 7.0f);
    float16x4_t sc_f16 = vcvt_f16_f32(sc_f32);
    vst1_f16(reinterpret_cast<__fp16*>(dst_scales4), sc_f16);

    float32x4_t sc_round_f32 = vcvt_f32_f16(sc_f16);
    float32x4_t inv_f32 = vdivq_f32(vdupq_n_f32(1.0f), sc_round_f32);
    float16x4_t inv_f16 = vcvt_f16_f32(inv_f32);

    float16x8_t inv0 = vdupq_lane_f16(inv_f16, 0);
    float16x8_t inv1 = vdupq_lane_f16(inv_f16, 1);
    float16x8_t inv2 = vdupq_lane_f16(inv_f16, 2);
    float16x8_t inv3 = vdupq_lane_f16(inv_f16, 3);

    const int16x8_t eight16 = vdupq_n_s16(8);
    const uint8x16_t fifteen8 = vdupq_n_u8(15);

    int16x8_t q0_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g0_0, inv0)), eight16);
    int16x8_t q0_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g0_1, inv0)), eight16);
    int16x8_t q1_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g1_0, inv1)), eight16);
    int16x8_t q1_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g1_1, inv1)), eight16);
    uint8x16_t u0 = vminq_u8(vcombine_u8(vqmovun_s16(q0_0), vqmovun_s16(q0_1)), fifteen8);
    uint8x16_t u1 = vminq_u8(vcombine_u8(vqmovun_s16(q1_0), vqmovun_s16(q1_1)), fifteen8);
    uint8x16x2_t uz01 = vuzpq_u8(u0, u1);
    out_p0 = vreinterpretq_u64_u8(vorrq_u8(uz01.val[0], vshlq_n_u8(uz01.val[1], 4)));

    int16x8_t q2_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g2_0, inv2)), eight16);
    int16x8_t q2_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g2_1, inv2)), eight16);
    int16x8_t q3_0 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g3_0, inv3)), eight16);
    int16x8_t q3_1 = vaddq_s16(vcvtnq_s16_f16(vmulq_f16(g3_1, inv3)), eight16);
    uint8x16_t u2 = vminq_u8(vcombine_u8(vqmovun_s16(q2_0), vqmovun_s16(q2_1)), fifteen8);
    uint8x16_t u3 = vminq_u8(vcombine_u8(vqmovun_s16(q3_0), vqmovun_s16(q3_1)), fifteen8);
    uint8x16x2_t uz23 = vuzpq_u8(u2, u3);
    out_p1 = vreinterpretq_u64_u8(vorrq_u8(uz23.val[0], vshlq_n_u8(uz23.val[1], 4)));
}

// Quad-Group (64 FP16 elements -> 64 INT8 bytes in FOUR 128-bit NEON registers + four FP16 scales)
inline void quantize_4groups64_int8_neon(
    const at::Half* __restrict__ src,
    uint64x2_t& out0,
    uint64x2_t& out1,
    uint64x2_t& out2,
    uint64x2_t& out3,
    at::Half* __restrict__ dst_scales4
) noexcept {
    const __fp16* s_ptr = reinterpret_cast<const __fp16*>(src);
    const float16x8_t g0_0 = vld1q_f16(s_ptr);
    const float16x8_t g0_1 = vld1q_f16(s_ptr + 8);
    const float16x8_t g1_0 = vld1q_f16(s_ptr + 16);
    const float16x8_t g1_1 = vld1q_f16(s_ptr + 24);
    const float16x8_t g2_0 = vld1q_f16(s_ptr + 32);
    const float16x8_t g2_1 = vld1q_f16(s_ptr + 40);
    const float16x8_t g3_0 = vld1q_f16(s_ptr + 48);
    const float16x8_t g3_1 = vld1q_f16(s_ptr + 56);

    __fp16 amax0 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g0_0), vabsq_f16(g0_1)));
    __fp16 amax1 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g1_0), vabsq_f16(g1_1)));
    __fp16 amax2 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g2_0), vabsq_f16(g2_1)));
    __fp16 amax3 = vmaxvq_f16(vmaxq_f16(vabsq_f16(g3_0), vabsq_f16(g3_1)));

    float32x4_t fmax_v = {
        std::max(static_cast<float>(amax0), 1e-6f),
        std::max(static_cast<float>(amax1), 1e-6f),
        std::max(static_cast<float>(amax2), 1e-6f),
        std::max(static_cast<float>(amax3), 1e-6f)
    };
    float32x4_t sc_f32 = vmulq_n_f32(fmax_v, 1.0f / 127.0f);
    float16x4_t sc_f16 = vcvt_f16_f32(sc_f32);
    vst1_f16(reinterpret_cast<__fp16*>(dst_scales4), sc_f16);

    float32x4_t sc_round_f32 = vcvt_f32_f16(sc_f16);
    float32x4_t inv_f32 = vdivq_f32(vdupq_n_f32(1.0f), sc_round_f32);
    float16x4_t inv_f16 = vcvt_f16_f32(inv_f32);

    float16x8_t inv0 = vdupq_lane_f16(inv_f16, 0);
    float16x8_t inv1 = vdupq_lane_f16(inv_f16, 1);
    float16x8_t inv2 = vdupq_lane_f16(inv_f16, 2);
    float16x8_t inv3 = vdupq_lane_f16(inv_f16, 3);
    const int8x16_t min127 = vdupq_n_s8(-127);

    int16x8_t s0_0 = vcvtnq_s16_f16(vmulq_f16(g0_0, inv0));
    int16x8_t s0_1 = vcvtnq_s16_f16(vmulq_f16(g0_1, inv0));
    int16x8_t s1_0 = vcvtnq_s16_f16(vmulq_f16(g1_0, inv1));
    int16x8_t s1_1 = vcvtnq_s16_f16(vmulq_f16(g1_1, inv1));
    out0 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s0_0), vqmovn_s16(s0_1)), min127));
    out1 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s1_0), vqmovn_s16(s1_1)), min127));

    int16x8_t s2_0 = vcvtnq_s16_f16(vmulq_f16(g2_0, inv2));
    int16x8_t s2_1 = vcvtnq_s16_f16(vmulq_f16(g2_1, inv2));
    int16x8_t s3_0 = vcvtnq_s16_f16(vmulq_f16(g3_0, inv3));
    int16x8_t s3_1 = vcvtnq_s16_f16(vmulq_f16(g3_1, inv3));
    out2 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s2_0), vqmovn_s16(s2_1)), min127));
    out3 = vreinterpretq_u64_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(s3_0), vqmovn_s16(s3_1)), min127));
}
#endif

// Unified per-page encoder used by single-page, batch, and zero-alloc peak paths.
inline void encode_one_cold_page_into_cpu(
    const at::Half* __restrict__ k_ptr,
    const at::Half* __restrict__ v_ptr,
    uint8_t* __restrict__ blob_ptr,
    int64_t vecs,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t preset,
    int64_t rope_mode,
    int64_t expected_payload_bytes,
    uint32_t tenant_id_hash,
    uint32_t model_id_hash,
    uint16_t layer_id,
    uint16_t head_group_id,
    uint64_t page_token_hash,
    uint64_t sequence_nonce,
    uint64_t k0,
    uint64_t k1,
    uint64_t k2,
    uint64_t k3,
    uint64_t mk0,
    uint64_t mk1
) noexcept {
    constexpr int64_t gs = 16;
    const int64_t nope_dim = head_dim - rope_dim;

    // 1. Write 32-byte AAD Header
    ColdPageHeaderRaw hdr{};
    hdr.magic = MAGIC_ZTCP;
    hdr.version = CODEC_VERSION;
    hdr.flags = static_cast<uint8_t>((preset & 0x0F) | ((rope_mode & 0x0F) << 4));
    hdr.tenant_id_hash = tenant_id_hash;
    hdr.model_id_hash = model_id_hash;
    hdr.layer_id = layer_id;
    hdr.head_group_id = head_group_id;
    hdr.page_token_hash = page_token_hash;
    hdr.sequence_nonce = sequence_nonce;
    std::memcpy(blob_ptr, &hdr, HEADER_BYTES);

    uint8_t* __restrict__ payload_ptr = blob_ptr + HEADER_BYTES;
    ChaCha20SipMacStream stream;
    stream.init(k0, k1, k2, k3, mk0, mk1, hdr, blob_ptr);

    // L1 stack-resident scale staging buffer (4-8 KB for standard GQA/MLA pages; zero heap alloc)
    constexpr size_t STACK_SCALE_CAP = 8192;
    alignas(16) std::array<at::Half, STACK_SCALE_CAP> stack_scales;
    std::vector<at::Half> heap_scales;
    const size_t max_section_groups = static_cast<size_t>(vecs * (head_dim / gs));
    at::Half* __restrict__ s_scratch = stack_scales.data();
    if (max_section_groups > STACK_SCALE_CAP) {
        heap_scales.resize(max_section_groups);
        s_scratch = heap_scales.data();
    }

    auto quantize_encrypt_int8_section = [&](
        const at::Half* section_src,
        int64_t row_stride,
        int64_t col_offset,
        int64_t quant_dim,
        size_t& offset_io
    ) noexcept {
        const int64_t groups = quant_dim / gs;
        uint8_t* __restrict__ q_base = payload_ptr + offset_io;
        const size_t q_bytes = static_cast<size_t>(vecs * quant_dim);
        uint8_t* __restrict__ s_base = q_base + q_bytes;
        const size_t s_bytes = static_cast<size_t>(vecs * groups) * sizeof(at::Half);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if ((groups & 1) == 0) {
            if (col_offset == 0 && quant_dim == row_stride) {
                const int64_t total_groups = vecs * groups;
                int64_t tg = 0;
                for (; tg + 3 < total_groups; tg += 4) {
                    uint64x2_t r0, r1, r2, r3;
                    quantize_4groups64_int8_neon(
                        section_src + tg * gs,
                        r0, r1, r2, r3,
                        s_scratch + tg
                    );
                    stream.encrypt_store_and_absorb_64B(r0, r1, r2, r3, q_base + tg * 16);
                }
                for (; tg < total_groups; tg += 2) {
                    uint64x2_t r0, r1;
                    quantize_2groups32_int8_neon(
                        section_src + tg * gs,
                        r0, r1,
                        s_scratch + tg
                    );
                    stream.encrypt_store_and_absorb_32B(r0, r1, q_base + tg * 16);
                }
            } else {
                for (int64_t v = 0; v < vecs; ++v) {
                    const at::Half* row = section_src + v * row_stride + col_offset;
                    int64_t g = 0;
                    for (; g + 3 < groups; g += 4) {
                        uint64x2_t r0, r1, r2, r3;
                        quantize_4groups64_int8_neon(
                            row + g * gs,
                            r0,
                            r1,
                            r2,
                            r3,
                            s_scratch + (v * groups + g)
                        );
                        uint8_t* dst = q_base + (v * groups + g) * 16;
                        stream.encrypt_store_and_absorb_64B(r0, r1, r2, r3, dst);
                    }
                    for (; g < groups; g += 2) {
                        uint64x2_t r0, r1;
                        quantize_2groups32_int8_neon(
                            row + g * gs,
                            r0,
                            r1,
                            s_scratch + (v * groups + g)
                        );
                        uint8_t* dst = q_base + (v * groups + g) * 16;
                        stream.encrypt_store_and_absorb_32B(r0, r1, dst);
                    }
                }
            }
            stream.encrypt_store_and_absorb_bytes(
                reinterpret_cast<const uint8_t*>(s_scratch),
                s_base,
                s_bytes
            );
            offset_io += q_bytes + s_bytes;
            return;
        }
#endif
        alignas(16) uint8_t tmp_q[16];
        for (int64_t v = 0; v < vecs; ++v) {
            const at::Half* row = section_src + v * row_stride + col_offset;
            for (int64_t g = 0; g < groups; ++g) {
                quantize_group16_int8(
                    row + g * gs,
                    tmp_q,
                    s_scratch + (v * groups + g)
                );
                stream.encrypt_store_and_absorb_bytes(
                    tmp_q,
                    q_base + (v * groups + g) * 16,
                    16
                );
            }
        }
        stream.encrypt_store_and_absorb_bytes(
            reinterpret_cast<const uint8_t*>(s_scratch),
            s_base,
            s_bytes
        );
        offset_io += q_bytes + s_bytes;
    };

    auto quantize_encrypt_int4_section = [&](
        const at::Half* section_src,
        int64_t row_stride,
        int64_t col_offset,
        int64_t quant_dim,
        size_t& offset_io
    ) noexcept {
        const int64_t groups = quant_dim / gs;
        uint8_t* __restrict__ q_base = payload_ptr + offset_io;
        const size_t q_bytes = static_cast<size_t>(vecs * (quant_dim / 2));
        uint8_t* __restrict__ s_base = q_base + q_bytes;
        const size_t s_bytes = static_cast<size_t>(vecs * groups) * sizeof(at::Half);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if ((groups & 1) == 0) {
            if (col_offset == 0 && quant_dim == row_stride) {
                const int64_t total_groups = vecs * groups;
                int64_t tg = 0;
                for (; tg + 7 < total_groups; tg += 8) {
                    uint64x2_t p0, p1, p2, p3;
                    quantize_4groups64_int4_neon(
                        section_src + tg * gs,
                        p0, p1,
                        s_scratch + tg
                    );
                    quantize_4groups64_int4_neon(
                        section_src + (tg + 4) * gs,
                        p2, p3,
                        s_scratch + tg + 4
                    );
                    stream.encrypt_store_and_absorb_64B(p0, p1, p2, p3, q_base + tg * 8);
                }
                for (; tg + 3 < total_groups; tg += 4) {
                    uint64x2_t p0, p1;
                    quantize_4groups64_int4_neon(
                        section_src + tg * gs,
                        p0, p1,
                        s_scratch + tg
                    );
                    stream.encrypt_store_and_absorb_32B(p0, p1, q_base + tg * 8);
                }
                for (; tg < total_groups; tg += 2) {
                    uint64x2_t p128 = quantize_2groups32_int4_neon(
                        section_src + tg * gs,
                        s_scratch + tg
                    );
                    stream.encrypt_store_and_absorb_16B(p128, q_base + tg * 8);
                }
            } else {
                for (int64_t v = 0; v < vecs; ++v) {
                    const at::Half* row = section_src + v * row_stride + col_offset;
                    int64_t g = 0;
                    for (; g + 7 < groups; g += 8) {
                        uint64x2_t p0, p1, p2, p3;
                        quantize_4groups64_int4_neon(
                            row + g * gs,
                            p0, p1,
                            s_scratch + (v * groups + g)
                        );
                        quantize_4groups64_int4_neon(
                            row + (g + 4) * gs,
                            p2, p3,
                            s_scratch + (v * groups + g + 4)
                        );
                        stream.encrypt_store_and_absorb_64B(p0, p1, p2, p3, q_base + (v * groups + g) * 8);
                    }
                    for (; g + 3 < groups; g += 4) {
                        uint64x2_t p0, p1;
                        quantize_4groups64_int4_neon(
                            row + g * gs,
                            p0,
                            p1,
                            s_scratch + (v * groups + g)
                        );
                        uint8_t* dst = q_base + (v * groups + g) * 8;
                        stream.encrypt_store_and_absorb_32B(p0, p1, dst);
                    }
                    for (; g < groups; g += 2) {
                        uint64x2_t p128 = quantize_2groups32_int4_neon(
                            row + g * gs,
                            s_scratch + (v * groups + g)
                        );
                        uint8_t* dst = q_base + (v * groups + g) * 8;
                        stream.encrypt_store_and_absorb_16B(p128, dst);
                    }
                }
            }
            stream.encrypt_store_and_absorb_bytes(
                reinterpret_cast<const uint8_t*>(s_scratch),
                s_base,
                s_bytes
            );
            offset_io += q_bytes + s_bytes;
            return;
        }
#endif
        alignas(8) uint8_t tmp_q[8];
        for (int64_t v = 0; v < vecs; ++v) {
            const at::Half* row = section_src + v * row_stride + col_offset;
            for (int64_t g = 0; g < groups; ++g) {
                quantize_group16_int4(
                    row + g * gs,
                    tmp_q,
                    s_scratch + (v * groups + g)
                );
                stream.encrypt_store_and_absorb_bytes(
                    tmp_q,
                    q_base + (v * groups + g) * 8,
                    8
                );
            }
        }
        stream.encrypt_store_and_absorb_bytes(
            reinterpret_cast<const uint8_t*>(s_scratch),
            s_base,
            s_bytes
        );
        offset_io += q_bytes + s_bytes;
    };

    size_t offset = 0;

    // 2. Section 1: K RoPE
    if (rope_mode == 2 && rope_dim > 0) {
        const size_t row_rope_bytes = static_cast<size_t>(rope_dim) * sizeof(at::Half);
        for (int64_t v = 0; v < vecs; ++v) {
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(k_ptr + v * head_dim);
            uint8_t* dst_row = payload_ptr + offset + static_cast<size_t>(v) * row_rope_bytes;
            stream.encrypt_store_and_absorb_bytes(src_row, dst_row, row_rope_bytes);
        }
        offset += static_cast<size_t>(vecs) * row_rope_bytes;
    } else if (rope_mode == 1 && rope_dim > 0) {
        quantize_encrypt_int8_section(k_ptr, head_dim, 0, rope_dim, offset);
    }

    // 3. Section 2: K NoPE (or full K when rope_mode == 0)
    const int64_t k_quant_dim = (rope_mode == 0) ? head_dim : nope_dim;
    const int64_t k_col_start = (rope_mode == 0) ? 0 : rope_dim;
    if (k_quant_dim > 0) {
        if (preset == 1) {
            quantize_encrypt_int4_section(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
        } else {
            quantize_encrypt_int8_section(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
        }
    }

    // 4. Section 3: V buffer (when present)
    if (v_ptr != nullptr) {
        if (preset == 1 || preset == 2) {
            quantize_encrypt_int4_section(v_ptr, head_dim, 0, head_dim, offset);
        } else {
            quantize_encrypt_int8_section(v_ptr, head_dim, 0, head_dim, offset);
        }
    }

    // 5. Section 4: Deterministic zero tail padding (encrypted & authenticated in order)
    const size_t expected_len = static_cast<size_t>(expected_payload_bytes);
    if (offset < expected_len) {
        alignas(16) uint8_t zero_pad[64] = {0};
        size_t rem_pad = expected_len - offset;
        while (rem_pad > 0) {
            size_t chunk = std::min<size_t>(rem_pad, sizeof(zero_pad));
            stream.encrypt_store_and_absorb_bytes(zero_pad, payload_ptr + offset, chunk);
            offset += chunk;
            rem_pad -= chunk;
        }
    }

    // 6. Finalize standard 128-bit SipHash-2-4 tag
    stream.finalize(blob_ptr + HEADER_BYTES + expected_len);
}

inline bool tensors_overlap_memory(const torch::Tensor& a, const torch::Tensor& b) noexcept {
    if (!a.defined() || !b.defined() || a.numel() == 0 || b.numel() == 0) {
        return false;
    }
    const uintptr_t a_start = reinterpret_cast<uintptr_t>(a.data_ptr());
    const uintptr_t a_end = a_start + static_cast<uintptr_t>(a.numel()) * static_cast<uintptr_t>(a.element_size());
    const uintptr_t b_start = reinterpret_cast<uintptr_t>(b.data_ptr());
    const uintptr_t b_end = b_start + static_cast<uintptr_t>(b.numel()) * static_cast<uintptr_t>(b.element_size());
    return (a_start < b_end) && (b_start < a_end);
}

inline size_t required_payload_bytes_for_flags(
    int64_t preset,
    int64_t rope_mode,
    int64_t vecs,
    int64_t head_dim,
    int64_t rope_dim,
    bool has_v
) {
    if (preset < 1 || preset > 3 || rope_mode < 0 || rope_mode > 2) {
        throw std::runtime_error("KV_INTEGRITY_ERROR: Invalid cold-page header flags (preset or rope_mode)");
    }
    constexpr int64_t gs = 16;
    const int64_t nope_dim = head_dim - rope_dim;
    size_t req = 0;
    if (rope_mode == 2 && rope_dim > 0) {
        req += static_cast<size_t>(vecs) * static_cast<size_t>(rope_dim) * sizeof(at::Half);
    } else if (rope_mode == 1 && rope_dim > 0) {
        int64_t r_groups = rope_dim / gs;
        req += static_cast<size_t>(vecs) * static_cast<size_t>(rope_dim) +
               static_cast<size_t>(vecs) * static_cast<size_t>(r_groups) * sizeof(at::Half);
    }
    const int64_t k_quant_dim = (rope_mode == 0) ? head_dim : nope_dim;
    if (k_quant_dim > 0) {
        int64_t k_groups = k_quant_dim / gs;
        size_t q_b = (preset == 1) ? static_cast<size_t>(vecs * (k_quant_dim / 2))
                                   : static_cast<size_t>(vecs * k_quant_dim);
        req += q_b + static_cast<size_t>(vecs) * static_cast<size_t>(k_groups) * sizeof(at::Half);
    }
    if (has_v) {
        int64_t v_groups = head_dim / gs;
        size_t q_b = (preset == 1 || preset == 2) ? static_cast<size_t>(vecs * (head_dim / 2))
                                                  : static_cast<size_t>(vecs * head_dim);
        req += q_b + static_cast<size_t>(vecs) * static_cast<size_t>(v_groups) * sizeof(at::Half);
    }
    return req;
}

}  // namespace

torch::Tensor fused_encode_cold_page_cpu(
    torch::Tensor k_page,
    c10::optional<torch::Tensor> v_page_opt,
    int64_t preset,
    int64_t rope_mode,
    int64_t rope_dim,
    int64_t expected_payload_bytes,
    uint32_t tenant_id_hash,
    uint32_t model_id_hash,
    uint16_t layer_id,
    uint16_t head_group_id,
    uint64_t page_token_hash,
    uint64_t sequence_nonce,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(k_page.device().is_cpu(), "k_page must be a CPU tensor for native CPU kernel");
    TORCH_CHECK(k_page.scalar_type() == torch::kFloat16, "k_page must be float16");
    TORCH_CHECK(k_page.is_contiguous(), "k_page must be contiguous");
    TORCH_CHECK(k_page.dim() == 3, "k_page must have shape [num_tokens, num_heads, head_dim]");

    const int64_t num_tokens = k_page.size(0);
    const int64_t num_heads = k_page.size(1);
    const int64_t head_dim = k_page.size(2);
    const int64_t vecs = num_tokens * num_heads;
    TORCH_CHECK(num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid k_page dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");

    const at::Half* k_ptr = k_page.data_ptr<at::Half>();
    const at::Half* v_ptr = nullptr;
    if (v_page_opt.has_value() && v_page_opt.value().defined()) {
        const auto& v_page = v_page_opt.value();
        TORCH_CHECK(v_page.device().is_cpu(), "v_page must be a CPU tensor");
        TORCH_CHECK(v_page.scalar_type() == torch::kFloat16, "v_page must be float16");
        TORCH_CHECK(v_page.is_contiguous(), "v_page must be contiguous");
        TORCH_CHECK(v_page.sizes() == k_page.sizes(), "v_page shape must match k_page shape");
        v_ptr = v_page.data_ptr<at::Half>();
    }
    TORCH_CHECK(preset >= 1 && preset <= 3, "Invalid preset (must be 1..3)");
    TORCH_CHECK(rope_mode >= 0 && rope_mode <= 2, "Invalid rope_mode (must be 0..2)");
    const size_t min_payload = required_payload_bytes_for_flags(
        preset, rope_mode, vecs, head_dim, rope_dim, v_ptr != nullptr
    );
    TORCH_CHECK(expected_payload_bytes >= static_cast<int64_t>(min_payload), "expected_payload_bytes smaller than required payload");

    const int64_t total_blob_bytes = HEADER_BYTES + expected_payload_bytes + TAG_BYTES;
    auto out_blob = torch::empty({total_blob_bytes}, torch::TensorOptions().dtype(torch::kUInt8));
    uint8_t* blob_ptr = out_blob.data_ptr<uint8_t>();

    encode_one_cold_page_into_cpu(
        k_ptr,
        v_ptr,
        blob_ptr,
        vecs,
        head_dim,
        rope_dim,
        preset,
        rope_mode,
        expected_payload_bytes,
        tenant_id_hash,
        model_id_hash,
        layer_id,
        head_group_id,
        page_token_hash,
        sequence_nonce,
        k0,
        k1,
        k2,
        k3,
        mk0,
        mk1
    );
    return out_blob;
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> fused_decode_cold_page_cpu(
    torch::Tensor cold_blob,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rope_dim,
    bool is_mla_single_buffer,
    uint32_t expected_tenant_hash,
    uint32_t expected_model_hash,
    uint16_t expected_layer_id,
    uint64_t expected_token_hash,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(cold_blob.device().is_cpu(), "cold_blob must be on CPU");
    TORCH_CHECK(cold_blob.scalar_type() == torch::kUInt8, "cold_blob must be uint8");
    TORCH_CHECK(cold_blob.is_contiguous(), "cold_blob must be contiguous");
    TORCH_CHECK(cold_blob.dim() == 1, "cold_blob must be a 1D tensor");
    TORCH_CHECK(cold_blob.numel() > static_cast<int64_t>(HEADER_BYTES + TAG_BYTES), "cold_blob too small");
    TORCH_CHECK(num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid KV geometry dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");

    const uint8_t* blob_ptr = cold_blob.data_ptr<uint8_t>();
    const size_t total_len = static_cast<size_t>(cold_blob.numel());
    const size_t payload_len = total_len - HEADER_BYTES - TAG_BYTES;

    ColdPageHeaderRaw hdr{};
    std::memcpy(&hdr, blob_ptr, HEADER_BYTES);
    if (hdr.magic != MAGIC_ZTCP || hdr.version != CODEC_VERSION) {
        throw std::runtime_error("KV_INTEGRITY_ERROR: Invalid cold-page header magic or version");
    }

    // 1. Verify AAD context before any memory write
    if (hdr.tenant_id_hash != expected_tenant_hash ||
        hdr.model_id_hash != expected_model_hash ||
        hdr.layer_id != expected_layer_id ||
        hdr.page_token_hash != expected_token_hash) {
        throw std::runtime_error("CROSS_TENANT_CONTAMINATION: Cold-page AAD mismatch");
    }

    // 2. Verify standard 128-bit SipHash-2-4 over Header + Ciphertext before decrypting
    SipMac128State mac;
    mac.init(mk0, mk1);
    mac.absorb_bytes(blob_ptr, HEADER_BYTES + payload_len);

    uint8_t computed_tag[16];
    mac.finalize(computed_tag);
    const uint8_t* expected_tag = blob_ptr + HEADER_BYTES + payload_len;
    uint8_t diff = 0;
    for (int i = 0; i < 16; ++i) {
        diff |= (computed_tag[i] ^ expected_tag[i]);
    }
    if (diff != 0) {
        throw std::runtime_error("KV_INTEGRITY_ERROR: Cold-page MAC verification failed");
    }

    int preset = hdr.flags & 0x0F;
    int rope_mode = (hdr.flags >> 4) & 0x0F;
    const int64_t vecs = num_tokens * num_heads;
    const size_t min_payload = required_payload_bytes_for_flags(
        preset, rope_mode, vecs, head_dim, rope_dim, !is_mla_single_buffer
    );
    if (payload_len < min_payload) {
        throw std::runtime_error("KV_INTEGRITY_ERROR: Cold-page payload truncated for header flags");
    }

    // 3. Decrypt via RFC 8439 ChaCha20 (256-bit key) & dequantize to K and V
    static thread_local std::vector<uint8_t> tls_plain_buf;
    if (tls_plain_buf.size() < payload_len) {
        tls_plain_buf.resize(payload_len);
    }
    uint8_t* __restrict__ plain_ptr = tls_plain_buf.data();

    uint32_t key_words[8];
    uint32_t nonce_words[3];
    unpack_key256_words(k0, k1, k2, k3, key_words);
    derive_chacha20_nonce_words(hdr, nonce_words);

    const uint8_t* cipher_ptr = blob_ptr + HEADER_BYTES;
    chacha20_xor_bytes_rfc8439(cipher_ptr, plain_ptr, payload_len, key_words, 1U, nonce_words);

    const int64_t nope_dim = head_dim - rope_dim;
    constexpr int64_t gs = 16;

    auto k_out = torch::empty({num_tokens, num_heads, head_dim}, torch::TensorOptions().dtype(torch::kFloat16));
    at::Half* k_ptr = k_out.data_ptr<at::Half>();
    const uint8_t* p_ptr = plain_ptr;
    size_t offset = 0;

    auto dequantize_int8_section = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
        const int64_t groups = quant_dim / gs;
        const uint8_t* q_base = p_ptr + off;
        const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * quant_dim);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if ((groups & 1) == 0) {
            if (col_offset == 0 && quant_dim == row_stride) {
                const int64_t total_groups = vecs * groups;
                for (int64_t tg = 0; tg < total_groups; tg += 2) {
                    dequantize_2groups32_int8_neon(
                        q_base + tg * gs,
                        s_base + tg,
                        dst_base + tg * gs
                    );
                }
            } else {
                for (int64_t v = 0; v < vecs; ++v) {
                    at::Half* row = dst_base + v * row_stride + col_offset;
                    for (int64_t g = 0; g < groups; g += 2) {
                        dequantize_2groups32_int8_neon(
                            q_base + (v * groups + g) * gs,
                            s_base + (v * groups + g),
                            row + g * gs
                        );
                    }
                }
            }
            off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
            return;
        }
#endif
        for (int64_t v = 0; v < vecs; ++v) {
            at::Half* row = dst_base + v * row_stride + col_offset;
            for (int64_t g = 0; g < groups; ++g) {
                dequantize_group16_int8(
                    q_base + (v * groups + g) * gs,
                    s_base[v * groups + g],
                    row + g * gs
                );
            }
        }
        off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
    };

    auto dequantize_int4_section = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
        const int64_t groups = quant_dim / gs;
        const uint8_t* q_base = p_ptr + off;
        const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * (quant_dim / 2));
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if ((groups & 1) == 0) {
            if (col_offset == 0 && quant_dim == row_stride) {
                const int64_t total_groups = vecs * groups;
                for (int64_t tg = 0; tg < total_groups; tg += 2) {
                    dequantize_2groups32_int4_neon(
                        q_base + tg * (gs / 2),
                        s_base + tg,
                        dst_base + tg * gs
                    );
                }
            } else {
                for (int64_t v = 0; v < vecs; ++v) {
                    at::Half* row = dst_base + v * row_stride + col_offset;
                    for (int64_t g = 0; g < groups; g += 2) {
                        dequantize_2groups32_int4_neon(
                            q_base + (v * groups + g) * (gs / 2),
                            s_base + (v * groups + g),
                            row + g * gs
                        );
                    }
                }
            }
            off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
            return;
        }
#endif
        for (int64_t v = 0; v < vecs; ++v) {
            at::Half* row = dst_base + v * row_stride + col_offset;
            for (int64_t g = 0; g < groups; ++g) {
                dequantize_group16_int4(
                    q_base + (v * groups + g) * (gs / 2),
                    s_base[v * groups + g],
                    row + g * gs
                );
            }
        }
        off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
    };

    if (rope_mode == 2 && rope_dim > 0) {
        const at::Half* src_rope = reinterpret_cast<const at::Half*>(p_ptr + offset);
        for (int64_t v = 0; v < vecs; ++v) {
            std::memcpy(k_ptr + v * head_dim, src_rope + v * rope_dim, rope_dim * sizeof(at::Half));
        }
        offset += vecs * rope_dim * sizeof(at::Half);
    } else if (rope_mode == 1 && rope_dim > 0) {
        dequantize_int8_section(k_ptr, head_dim, 0, rope_dim, offset);
    }

    int64_t k_quant_dim = (rope_mode == 0) ? head_dim : nope_dim;
    int64_t k_col_start = (rope_mode == 0) ? 0 : rope_dim;
    if (k_quant_dim > 0) {
        if (preset == 1) {
            dequantize_int4_section(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
        } else {
            dequantize_int8_section(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
        }
    }

    if (is_mla_single_buffer) {
        std::memset(plain_ptr, 0, payload_len);
        return std::make_tuple(k_out, c10::nullopt);
    }

    auto v_out = torch::empty({num_tokens, num_heads, head_dim}, torch::TensorOptions().dtype(torch::kFloat16));
    at::Half* v_ptr = v_out.data_ptr<at::Half>();
    if (preset == 1 || preset == 2) {
        dequantize_int4_section(v_ptr, head_dim, 0, head_dim, offset);
    } else {
        dequantize_int8_section(v_ptr, head_dim, 0, head_dim, offset);
    }

    std::memset(plain_ptr, 0, payload_len);
    return std::make_tuple(k_out, v_out);
}

torch::Tensor fused_encode_cold_pages_batch_cpu(
    torch::Tensor k_pages,
    c10::optional<torch::Tensor> v_pages_opt,
    torch::Tensor page_token_hashes,
    int64_t preset,
    int64_t rope_mode,
    int64_t rope_dim,
    int64_t expected_payload_bytes,
    uint32_t tenant_id_hash,
    uint32_t model_id_hash,
    uint16_t layer_id,
    uint16_t head_group_id,
    uint64_t base_sequence_nonce,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(k_pages.device().is_cpu() && k_pages.is_contiguous() && k_pages.scalar_type() == torch::kFloat16, "k_pages must be contiguous CPU float16 tensor");
    TORCH_CHECK(k_pages.dim() == 4, "k_pages must have shape [batch, num_tokens, num_heads, head_dim]");
    const int64_t batch_size = k_pages.size(0);
    const int64_t num_tokens = k_pages.size(1);
    const int64_t num_heads = k_pages.size(2);
    const int64_t head_dim = k_pages.size(3);
    const int64_t vecs = num_tokens * num_heads;
    TORCH_CHECK(batch_size > 0 && num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid k_pages dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");
    TORCH_CHECK(page_token_hashes.device().is_cpu() && page_token_hashes.is_contiguous() && page_token_hashes.scalar_type() == torch::kInt64, "page_token_hashes must be contiguous CPU int64");
    TORCH_CHECK(page_token_hashes.dim() == 1 && page_token_hashes.size(0) == batch_size, "page_token_hashes must be 1D with length equal to batch_size");

    const bool has_v = v_pages_opt.has_value() && v_pages_opt.value().defined();
    const at::Half* v_all_ptr = nullptr;
    if (has_v) {
        const auto& v_pages = v_pages_opt.value();
        TORCH_CHECK(v_pages.device().is_cpu() && v_pages.is_contiguous() && v_pages.scalar_type() == torch::kFloat16, "v_pages must be contiguous CPU float16 tensor");
        TORCH_CHECK(v_pages.sizes() == k_pages.sizes(), "v_pages shape must match k_pages shape");
        v_all_ptr = v_pages.data_ptr<at::Half>();
    }
    TORCH_CHECK(preset >= 1 && preset <= 3, "Invalid preset (must be 1..3)");
    TORCH_CHECK(rope_mode >= 0 && rope_mode <= 2, "Invalid rope_mode (must be 0..2)");
    const size_t min_payload = required_payload_bytes_for_flags(
        preset, rope_mode, vecs, head_dim, rope_dim, has_v
    );
    TORCH_CHECK(expected_payload_bytes >= static_cast<int64_t>(min_payload), "expected_payload_bytes smaller than required payload");

    const int64_t total_blob_bytes = HEADER_BYTES + expected_payload_bytes + TAG_BYTES;
    auto out_blobs = torch::empty({batch_size, total_blob_bytes}, torch::TensorOptions().dtype(torch::kUInt8));

    const at::Half* k_all_ptr = k_pages.data_ptr<at::Half>();
    const int64_t* token_hash_ptr = page_token_hashes.data_ptr<int64_t>();
    uint8_t* out_all_ptr = out_blobs.data_ptr<uint8_t>();
    const int64_t page_elems = vecs * head_dim;

    auto worker_fn = [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            encode_one_cold_page_into_cpu(
                k_all_ptr + b * page_elems,
                has_v ? (v_all_ptr + b * page_elems) : nullptr,
                out_all_ptr + b * total_blob_bytes,
                vecs,
                head_dim,
                rope_dim,
                preset,
                rope_mode,
                expected_payload_bytes,
                tenant_id_hash,
                model_id_hash,
                layer_id,
                head_group_id,
                static_cast<uint64_t>(token_hash_ptr[b]),
                base_sequence_nonce + static_cast<uint64_t>(b),
                k0,
                k1,
                k2,
                k3,
                mk0,
                mk1
            );
        }
    };

    const int64_t hw_threads = std::min<int64_t>(
        batch_size,
        std::max<int64_t>(1, static_cast<int64_t>(std::thread::hardware_concurrency()))
    );
    if (hw_threads <= 1) {
        worker_fn(0, batch_size);
    } else {
#if defined(__APPLE__)
        int64_t chunk = (batch_size + hw_threads - 1) / hw_threads;
        int64_t num_tasks = (batch_size + chunk - 1) / chunk;
        auto dispatch_chunk = [&](size_t t_idx) {
            int64_t start = static_cast<int64_t>(t_idx) * chunk;
            int64_t stop = std::min<int64_t>(batch_size, start + chunk);
            if (start < stop) {
                worker_fn(start, stop);
            }
        };
        auto fn_ptr = &dispatch_chunk;
        dispatch_apply_f(
            static_cast<size_t>(num_tasks),
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
            fn_ptr,
            [](void* ctx, size_t idx) {
                (*static_cast<decltype(fn_ptr)>(ctx))(idx);
            }
        );
#else
        std::vector<std::thread> pool;
        pool.reserve(hw_threads);
        int64_t chunk = (batch_size + hw_threads - 1) / hw_threads;
        for (int64_t t = 0; t < hw_threads; ++t) {
            int64_t start = t * chunk;
            int64_t stop = std::min<int64_t>(batch_size, start + chunk);
            if (start < stop) {
                pool.emplace_back(worker_fn, start, stop);
            }
        }
        for (auto& th : pool) {
            th.join();
        }
#endif
    }

    return out_blobs;
}

void fused_encode_cold_pages_peak_into_cpu(
    torch::Tensor k_pages,
    c10::optional<torch::Tensor> v_pages_opt,
    torch::Tensor page_token_hashes,
    torch::Tensor out_blobs,
    int64_t preset,
    int64_t rope_mode,
    int64_t rope_dim,
    int64_t expected_payload_bytes,
    uint32_t tenant_id_hash,
    uint32_t model_id_hash,
    uint16_t layer_id,
    uint16_t head_group_id,
    uint64_t base_sequence_nonce,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(k_pages.device().is_cpu() && k_pages.is_contiguous() && k_pages.scalar_type() == torch::kFloat16, "k_pages must be contiguous CPU float16 tensor");
    TORCH_CHECK(k_pages.dim() == 4, "k_pages must have shape [batch, num_tokens, num_heads, head_dim]");
    TORCH_CHECK(out_blobs.device().is_cpu() && out_blobs.is_contiguous() && out_blobs.scalar_type() == torch::kUInt8, "out_blobs must be contiguous CPU uint8 tensor");
    TORCH_CHECK(out_blobs.dim() == 2, "out_blobs must have shape [batch, total_blob_bytes]");
    const int64_t batch_size = k_pages.size(0);
    const int64_t num_tokens = k_pages.size(1);
    const int64_t num_heads = k_pages.size(2);
    const int64_t head_dim = k_pages.size(3);
    const int64_t vecs = num_tokens * num_heads;
    const int64_t total_blob_bytes = HEADER_BYTES + expected_payload_bytes + TAG_BYTES;
    TORCH_CHECK(batch_size > 0 && num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid k_pages dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");
    TORCH_CHECK(out_blobs.size(0) == batch_size && out_blobs.size(1) == total_blob_bytes, "out_blobs shape must be [batch_size, total_blob_bytes]");
    TORCH_CHECK(page_token_hashes.device().is_cpu() && page_token_hashes.is_contiguous() && page_token_hashes.scalar_type() == torch::kInt64, "page_token_hashes must be contiguous CPU int64");
    TORCH_CHECK(page_token_hashes.dim() == 1 && page_token_hashes.size(0) == batch_size, "page_token_hashes must be 1D with length equal to batch_size");
    TORCH_CHECK(!tensors_overlap_memory(out_blobs, k_pages) && !tensors_overlap_memory(out_blobs, page_token_hashes), "out_blobs must not alias or overlap input tensors");

    const at::Half* k_all_ptr = k_pages.data_ptr<at::Half>();
    const bool has_v = v_pages_opt.has_value() && v_pages_opt.value().defined();
    const at::Half* v_all_ptr = nullptr;
    if (has_v) {
        const auto& v_pages = v_pages_opt.value();
        TORCH_CHECK(v_pages.device().is_cpu() && v_pages.is_contiguous() && v_pages.scalar_type() == torch::kFloat16, "v_pages must be contiguous CPU float16 tensor");
        TORCH_CHECK(v_pages.sizes() == k_pages.sizes(), "v_pages shape must match k_pages shape");
        TORCH_CHECK(!tensors_overlap_memory(out_blobs, v_pages), "out_blobs must not alias or overlap v_pages");
        v_all_ptr = v_pages.data_ptr<at::Half>();
    }
    TORCH_CHECK(preset >= 1 && preset <= 3, "Invalid preset (must be 1..3)");
    TORCH_CHECK(rope_mode >= 0 && rope_mode <= 2, "Invalid rope_mode (must be 0..2)");
    const size_t min_payload = required_payload_bytes_for_flags(
        preset, rope_mode, vecs, head_dim, rope_dim, has_v
    );
    TORCH_CHECK(expected_payload_bytes >= static_cast<int64_t>(min_payload), "expected_payload_bytes smaller than required payload");

    const int64_t* token_hash_ptr = page_token_hashes.data_ptr<int64_t>();
    uint8_t* out_all_ptr = out_blobs.data_ptr<uint8_t>();
    const int64_t page_elems = vecs * head_dim;

    auto encode_one_page = [&](size_t b) {
        encode_one_cold_page_into_cpu(
            k_all_ptr + static_cast<int64_t>(b) * page_elems,
            has_v ? (v_all_ptr + static_cast<int64_t>(b) * page_elems) : nullptr,
            out_all_ptr + static_cast<int64_t>(b) * total_blob_bytes,
            vecs,
            head_dim,
            rope_dim,
            preset,
            rope_mode,
            expected_payload_bytes,
            tenant_id_hash,
            model_id_hash,
            layer_id,
            head_group_id,
            static_cast<uint64_t>(token_hash_ptr[b]),
            base_sequence_nonce + static_cast<uint64_t>(b),
            k0,
            k1,
            k2,
            k3,
            mk0,
            mk1
        );
    };

#if defined(__APPLE__)
    auto fn_ptr = &encode_one_page;
    dispatch_apply_f(
        static_cast<size_t>(batch_size),
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        fn_ptr,
        [](void* ctx, size_t idx) {
            (*static_cast<decltype(fn_ptr)>(ctx))(idx);
        }
    );
#else
    for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
        encode_one_page(b);
    }
#endif
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> fused_decode_cold_pages_batch_cpu(
    torch::Tensor cold_blobs,
    torch::Tensor expected_token_hashes,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rope_dim,
    bool is_mla_single_buffer,
    uint32_t expected_tenant_hash,
    uint32_t expected_model_hash,
    uint16_t expected_layer_id,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(cold_blobs.device().is_cpu() && cold_blobs.is_contiguous() && cold_blobs.scalar_type() == torch::kUInt8, "cold_blobs must be contiguous CPU uint8");
    TORCH_CHECK(cold_blobs.dim() == 2, "cold_blobs must have shape [batch, total_blob_bytes]");
    TORCH_CHECK(expected_token_hashes.device().is_cpu() && expected_token_hashes.is_contiguous() && expected_token_hashes.scalar_type() == torch::kInt64, "expected_token_hashes must be contiguous CPU int64");

    const int64_t batch_size = cold_blobs.size(0);
    const int64_t total_len = cold_blobs.size(1);
    TORCH_CHECK(batch_size > 0 && num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid KV geometry dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");
    TORCH_CHECK(expected_token_hashes.dim() == 1 && expected_token_hashes.size(0) == batch_size, "expected_token_hashes must be 1D with length equal to batch_size");
    TORCH_CHECK(total_len > static_cast<int64_t>(HEADER_BYTES + TAG_BYTES), "cold_blob too small");
    const size_t payload_len = static_cast<size_t>(total_len) - HEADER_BYTES - TAG_BYTES;

    auto k_out = torch::empty({batch_size, num_tokens, num_heads, head_dim}, torch::TensorOptions().dtype(torch::kFloat16));
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt;
    at::Half* v_all_ptr = nullptr;
    if (!is_mla_single_buffer) {
        auto v_out = torch::empty({batch_size, num_tokens, num_heads, head_dim}, torch::TensorOptions().dtype(torch::kFloat16));
        v_all_ptr = v_out.data_ptr<at::Half>();
        v_out_opt = v_out;
    }

    const uint8_t* blobs_all_ptr = cold_blobs.data_ptr<uint8_t>();
    const int64_t* token_hash_ptr = expected_token_hashes.data_ptr<int64_t>();
    at::Half* k_all_ptr = k_out.data_ptr<at::Half>();
    const int64_t vecs = num_tokens * num_heads;
    const int64_t page_elems = vecs * head_dim;
    const int64_t nope_dim = head_dim - rope_dim;
    constexpr int64_t gs = 16;

    uint32_t key_words[8];
    unpack_key256_words(k0, k1, k2, k3, key_words);

    // Status codes per page: 0 = OK, 1 = bad magic/version, 2 = AAD contamination, 3 = MAC integrity error
    std::vector<int> status_codes(static_cast<size_t>(batch_size), 0);

    auto decode_one_page = [&](size_t b) noexcept {
        const uint8_t* blob_ptr = blobs_all_ptr + static_cast<size_t>(b) * static_cast<size_t>(total_len);
        ColdPageHeaderRaw hdr{};
        std::memcpy(&hdr, blob_ptr, HEADER_BYTES);
        if (hdr.magic != MAGIC_ZTCP || hdr.version != CODEC_VERSION) {
            status_codes[b] = 1;
            return;
        }
        if (hdr.tenant_id_hash != expected_tenant_hash ||
            hdr.model_id_hash != expected_model_hash ||
            hdr.layer_id != expected_layer_id ||
            hdr.page_token_hash != static_cast<uint64_t>(token_hash_ptr[b])) {
            status_codes[b] = 2;
            return;
        }

        SipMac128State mac;
        mac.init(mk0, mk1);
        mac.absorb_bytes(blob_ptr, HEADER_BYTES + payload_len);
        uint8_t computed_tag[16];
        mac.finalize(computed_tag);
        const uint8_t* expected_tag = blob_ptr + HEADER_BYTES + payload_len;
        uint8_t diff = 0;
        for (int i = 0; i < 16; ++i) {
            diff |= (computed_tag[i] ^ expected_tag[i]);
        }
        if (diff != 0) {
            status_codes[b] = 3;
            return;
        }

        int preset = hdr.flags & 0x0F;
        int rope_mode = (hdr.flags >> 4) & 0x0F;
        if (preset < 1 || preset > 3 || rope_mode < 0 || rope_mode > 2) {
            status_codes[b] = 1;
            return;
        }
        try {
            size_t min_p = required_payload_bytes_for_flags(preset, rope_mode, vecs, head_dim, rope_dim, !is_mla_single_buffer);
            if (payload_len < min_p) {
                status_codes[b] = 1;
                return;
            }
        } catch (...) {
            status_codes[b] = 1;
            return;
        }

        static thread_local std::vector<uint8_t> tls_plain;
        if (tls_plain.size() < payload_len) {
            tls_plain.resize(payload_len);
        }
        uint8_t* __restrict__ p_ptr = tls_plain.data();

        uint32_t nonce_words[3];
        derive_chacha20_nonce_words(hdr, nonce_words);
        chacha20_xor_bytes_rfc8439(blob_ptr + HEADER_BYTES, p_ptr, payload_len, key_words, 1U, nonce_words);

        at::Half* k_ptr = k_all_ptr + static_cast<int64_t>(b) * page_elems;
        at::Half* v_ptr = v_all_ptr ? (v_all_ptr + static_cast<int64_t>(b) * page_elems) : nullptr;
        size_t offset = 0;

        auto deq_int8_sec = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
            const int64_t groups = quant_dim / gs;
            const uint8_t* q_base = p_ptr + off;
            const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * quant_dim);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if ((groups & 1) == 0) {
                if (col_offset == 0 && quant_dim == row_stride) {
                    const int64_t total_groups = vecs * groups;
                    for (int64_t tg = 0; tg < total_groups; tg += 2) {
                        dequantize_2groups32_int8_neon(q_base + tg * gs, s_base + tg, dst_base + tg * gs);
                    }
                } else {
                    for (int64_t v = 0; v < vecs; ++v) {
                        at::Half* row = dst_base + v * row_stride + col_offset;
                        for (int64_t g = 0; g < groups; g += 2) {
                            dequantize_2groups32_int8_neon(q_base + (v * groups + g) * gs, s_base + (v * groups + g), row + g * gs);
                        }
                    }
                }
                off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
                return;
            }
#endif
            for (int64_t v = 0; v < vecs; ++v) {
                at::Half* row = dst_base + v * row_stride + col_offset;
                for (int64_t g = 0; g < groups; ++g) {
                    dequantize_group16_int8(q_base + (v * groups + g) * gs, s_base[v * groups + g], row + g * gs);
                }
            }
            off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
        };

        auto deq_int4_sec = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
            const int64_t groups = quant_dim / gs;
            const uint8_t* q_base = p_ptr + off;
            const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * (quant_dim / 2));
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if ((groups & 1) == 0) {
                if (col_offset == 0 && quant_dim == row_stride) {
                    const int64_t total_groups = vecs * groups;
                    for (int64_t tg = 0; tg < total_groups; tg += 2) {
                        dequantize_2groups32_int4_neon(q_base + tg * (gs / 2), s_base + tg, dst_base + tg * gs);
                    }
                } else {
                    for (int64_t v = 0; v < vecs; ++v) {
                        at::Half* row = dst_base + v * row_stride + col_offset;
                        for (int64_t g = 0; g < groups; g += 2) {
                            dequantize_2groups32_int4_neon(q_base + (v * groups + g) * (gs / 2), s_base + (v * groups + g), row + g * gs);
                        }
                    }
                }
                off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
                return;
            }
#endif
            for (int64_t v = 0; v < vecs; ++v) {
                at::Half* row = dst_base + v * row_stride + col_offset;
                for (int64_t g = 0; g < groups; ++g) {
                    dequantize_group16_int4(q_base + (v * groups + g) * (gs / 2), s_base[v * groups + g], row + g * gs);
                }
            }
            off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
        };

        if (rope_mode == 2 && rope_dim > 0) {
            const at::Half* src_rope = reinterpret_cast<const at::Half*>(p_ptr + offset);
            for (int64_t v = 0; v < vecs; ++v) {
                std::memcpy(k_ptr + v * head_dim, src_rope + v * rope_dim, rope_dim * sizeof(at::Half));
            }
            offset += vecs * rope_dim * sizeof(at::Half);
        } else if (rope_mode == 1 && rope_dim > 0) {
            deq_int8_sec(k_ptr, head_dim, 0, rope_dim, offset);
        }

        int64_t k_quant_dim = (rope_mode == 0) ? head_dim : nope_dim;
        int64_t k_col_start = (rope_mode == 0) ? 0 : rope_dim;
        if (k_quant_dim > 0) {
            if (preset == 1) {
                deq_int4_sec(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
            } else {
                deq_int8_sec(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
            }
        }

        if (v_ptr != nullptr) {
            if (preset == 1 || preset == 2) {
                deq_int4_sec(v_ptr, head_dim, 0, head_dim, offset);
            } else {
                deq_int8_sec(v_ptr, head_dim, 0, head_dim, offset);
            }
        }
        std::memset(p_ptr, 0, payload_len);
    };

#if defined(__APPLE__)
    auto fn_ptr = &decode_one_page;
    dispatch_apply_f(
        static_cast<size_t>(batch_size),
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        fn_ptr,
        [](void* ctx, size_t idx) {
            (*static_cast<decltype(fn_ptr)>(ctx))(idx);
        }
    );
#else
    for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
        decode_one_page(b);
    }
#endif

    for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
        if (status_codes[b] != 0) {
            k_out.zero_();
            if (v_out_opt.has_value()) {
                v_out_opt.value().zero_();
            }
            if (status_codes[b] == 1) {
                throw std::runtime_error("KV_INTEGRITY_ERROR: Invalid cold-page header magic, version, or flags");
            } else if (status_codes[b] == 2) {
                throw std::runtime_error("CROSS_TENANT_CONTAMINATION: Cold-page AAD mismatch");
            } else {
                throw std::runtime_error("KV_INTEGRITY_ERROR: Cold-page MAC verification failed");
            }
        }
    }

    return std::make_tuple(k_out, v_out_opt);
}

void fused_decode_cold_pages_peak_into_cpu(
    torch::Tensor cold_blobs,
    torch::Tensor expected_token_hashes,
    torch::Tensor out_k_pages,
    c10::optional<torch::Tensor> out_v_pages_opt,
    int64_t rope_dim,
    bool is_mla_single_buffer,
    uint32_t expected_tenant_hash,
    uint32_t expected_model_hash,
    uint16_t expected_layer_id,
    uint64_t k0,
    uint64_t k1,
    uint64_t mk0,
    uint64_t mk1,
    uint64_t k2,
    uint64_t k3
) {
    TORCH_CHECK(cold_blobs.device().is_cpu() && cold_blobs.is_contiguous() && cold_blobs.scalar_type() == torch::kUInt8, "cold_blobs must be contiguous CPU uint8");
    TORCH_CHECK(cold_blobs.dim() == 2, "cold_blobs must have shape [batch, total_blob_bytes]");
    TORCH_CHECK(out_k_pages.device().is_cpu() && out_k_pages.is_contiguous() && out_k_pages.scalar_type() == torch::kFloat16, "out_k_pages must be contiguous CPU float16");
    TORCH_CHECK(out_k_pages.dim() == 4, "out_k_pages must have shape [batch, num_tokens, num_heads, head_dim]");
    TORCH_CHECK(expected_token_hashes.device().is_cpu() && expected_token_hashes.is_contiguous() && expected_token_hashes.scalar_type() == torch::kInt64, "expected_token_hashes must be contiguous CPU int64");

    const int64_t batch_size = out_k_pages.size(0);
    const int64_t num_tokens = out_k_pages.size(1);
    const int64_t num_heads = out_k_pages.size(2);
    const int64_t head_dim = out_k_pages.size(3);
    const int64_t total_len = cold_blobs.size(1);
    TORCH_CHECK(batch_size > 0 && num_tokens > 0 && num_heads > 0 && head_dim >= 16 && (head_dim % 16) == 0, "Invalid out_k_pages dimensions");
    TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && (rope_dim % 16) == 0, "Invalid rope_dim");
    TORCH_CHECK(cold_blobs.size(0) == batch_size, "cold_blobs batch size must match out_k_pages batch size");
    TORCH_CHECK(expected_token_hashes.dim() == 1 && expected_token_hashes.size(0) == batch_size, "expected_token_hashes must be 1D with length equal to batch_size");
    TORCH_CHECK(total_len > static_cast<int64_t>(HEADER_BYTES + TAG_BYTES), "cold_blob too small");
    TORCH_CHECK(!tensors_overlap_memory(out_k_pages, cold_blobs) && !tensors_overlap_memory(out_k_pages, expected_token_hashes), "out_k_pages must not alias or overlap input tensors");
    const size_t payload_len = static_cast<size_t>(total_len) - HEADER_BYTES - TAG_BYTES;

    at::Half* k_all_ptr = out_k_pages.data_ptr<at::Half>();
    at::Half* v_all_ptr = nullptr;
    if (!is_mla_single_buffer) {
        TORCH_CHECK(out_v_pages_opt.has_value() && out_v_pages_opt.value().defined(), "out_v_pages is required for non-MLA geometry");
        const auto& out_v_pages = out_v_pages_opt.value();
        TORCH_CHECK(out_v_pages.device().is_cpu() && out_v_pages.is_contiguous() && out_v_pages.scalar_type() == torch::kFloat16, "out_v_pages must be contiguous CPU float16");
        TORCH_CHECK(out_v_pages.sizes() == out_k_pages.sizes(), "out_v_pages shape must match out_k_pages shape");
        TORCH_CHECK(!tensors_overlap_memory(out_k_pages, out_v_pages), "out_k_pages and out_v_pages must not alias or overlap in memory");
        TORCH_CHECK(!tensors_overlap_memory(out_v_pages, cold_blobs) && !tensors_overlap_memory(out_v_pages, expected_token_hashes), "out_v_pages must not alias or overlap input tensors");
        v_all_ptr = out_v_pages.data_ptr<at::Half>();
    }

    const uint8_t* blobs_all_ptr = cold_blobs.data_ptr<uint8_t>();
    const int64_t* token_hash_ptr = expected_token_hashes.data_ptr<int64_t>();
    const int64_t vecs = num_tokens * num_heads;
    const int64_t page_elems = vecs * head_dim;
    const int64_t nope_dim = head_dim - rope_dim;
    constexpr int64_t gs = 16;

    uint32_t key_words[8];
    unpack_key256_words(k0, k1, k2, k3, key_words);
    std::vector<int> status_codes(static_cast<size_t>(batch_size), 0);

    auto decode_one_page = [&](size_t b) noexcept {
        const uint8_t* blob_ptr = blobs_all_ptr + static_cast<size_t>(b) * static_cast<size_t>(total_len);
        ColdPageHeaderRaw hdr{};
        std::memcpy(&hdr, blob_ptr, HEADER_BYTES);
        if (hdr.magic != MAGIC_ZTCP || hdr.version != CODEC_VERSION) {
            status_codes[b] = 1;
            return;
        }
        if (hdr.tenant_id_hash != expected_tenant_hash ||
            hdr.model_id_hash != expected_model_hash ||
            hdr.layer_id != expected_layer_id ||
            hdr.page_token_hash != static_cast<uint64_t>(token_hash_ptr[b])) {
            status_codes[b] = 2;
            return;
        }

        SipMac128State mac;
        mac.init(mk0, mk1);
        mac.absorb_bytes(blob_ptr, HEADER_BYTES + payload_len);
        uint8_t computed_tag[16];
        mac.finalize(computed_tag);
        const uint8_t* expected_tag = blob_ptr + HEADER_BYTES + payload_len;
        uint8_t diff = 0;
        for (int i = 0; i < 16; ++i) {
            diff |= (computed_tag[i] ^ expected_tag[i]);
        }
        if (diff != 0) {
            status_codes[b] = 3;
            return;
        }

        int preset = hdr.flags & 0x0F;
        int rope_mode = (hdr.flags >> 4) & 0x0F;
        if (preset < 1 || preset > 3 || rope_mode < 0 || rope_mode > 2) {
            status_codes[b] = 1;
            return;
        }
        try {
            size_t min_p = required_payload_bytes_for_flags(preset, rope_mode, vecs, head_dim, rope_dim, !is_mla_single_buffer);
            if (payload_len < min_p) {
                status_codes[b] = 1;
                return;
            }
        } catch (...) {
            status_codes[b] = 1;
            return;
        }

        static thread_local std::vector<uint8_t> tls_plain;
        if (tls_plain.size() < payload_len) {
            tls_plain.resize(payload_len);
        }
        uint8_t* __restrict__ p_ptr = tls_plain.data();

        uint32_t nonce_words[3];
        derive_chacha20_nonce_words(hdr, nonce_words);
        chacha20_xor_bytes_rfc8439(blob_ptr + HEADER_BYTES, p_ptr, payload_len, key_words, 1U, nonce_words);

        at::Half* k_ptr = k_all_ptr + static_cast<int64_t>(b) * page_elems;
        at::Half* v_ptr = v_all_ptr ? (v_all_ptr + static_cast<int64_t>(b) * page_elems) : nullptr;
        size_t offset = 0;

        auto deq_int8_sec = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
            const int64_t groups = quant_dim / gs;
            const uint8_t* q_base = p_ptr + off;
            const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * quant_dim);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if ((groups & 1) == 0) {
                if (col_offset == 0 && quant_dim == row_stride) {
                    const int64_t total_groups = vecs * groups;
                    for (int64_t tg = 0; tg < total_groups; tg += 2) {
                        dequantize_2groups32_int8_neon(q_base + tg * gs, s_base + tg, dst_base + tg * gs);
                    }
                } else {
                    for (int64_t v = 0; v < vecs; ++v) {
                        at::Half* row = dst_base + v * row_stride + col_offset;
                        for (int64_t g = 0; g < groups; g += 2) {
                            dequantize_2groups32_int8_neon(q_base + (v * groups + g) * gs, s_base + (v * groups + g), row + g * gs);
                        }
                    }
                }
                off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
                return;
            }
#endif
            for (int64_t v = 0; v < vecs; ++v) {
                at::Half* row = dst_base + v * row_stride + col_offset;
                for (int64_t g = 0; g < groups; ++g) {
                    dequantize_group16_int8(q_base + (v * groups + g) * gs, s_base[v * groups + g], row + g * gs);
                }
            }
            off += vecs * quant_dim + vecs * groups * sizeof(at::Half);
        };

        auto deq_int4_sec = [&](at::Half* dst_base, int64_t row_stride, int64_t col_offset, int64_t quant_dim, size_t& off) noexcept {
            const int64_t groups = quant_dim / gs;
            const uint8_t* q_base = p_ptr + off;
            const at::Half* s_base = reinterpret_cast<const at::Half*>(p_ptr + off + vecs * (quant_dim / 2));
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if ((groups & 1) == 0) {
                if (col_offset == 0 && quant_dim == row_stride) {
                    const int64_t total_groups = vecs * groups;
                    for (int64_t tg = 0; tg < total_groups; tg += 2) {
                        dequantize_2groups32_int4_neon(q_base + tg * (gs / 2), s_base + tg, dst_base + tg * gs);
                    }
                } else {
                    for (int64_t v = 0; v < vecs; ++v) {
                        at::Half* row = dst_base + v * row_stride + col_offset;
                        for (int64_t g = 0; g < groups; g += 2) {
                            dequantize_2groups32_int4_neon(q_base + (v * groups + g) * (gs / 2), s_base + (v * groups + g), row + g * gs);
                        }
                    }
                }
                off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
                return;
            }
#endif
            for (int64_t v = 0; v < vecs; ++v) {
                at::Half* row = dst_base + v * row_stride + col_offset;
                for (int64_t g = 0; g < groups; ++g) {
                    dequantize_group16_int4(q_base + (v * groups + g) * (gs / 2), s_base[v * groups + g], row + g * gs);
                }
            }
            off += vecs * (quant_dim / 2) + vecs * groups * sizeof(at::Half);
        };

        if (rope_mode == 2 && rope_dim > 0) {
            const at::Half* src_rope = reinterpret_cast<const at::Half*>(p_ptr + offset);
            for (int64_t v = 0; v < vecs; ++v) {
                std::memcpy(k_ptr + v * head_dim, src_rope + v * rope_dim, rope_dim * sizeof(at::Half));
            }
            offset += vecs * rope_dim * sizeof(at::Half);
        } else if (rope_mode == 1 && rope_dim > 0) {
            deq_int8_sec(k_ptr, head_dim, 0, rope_dim, offset);
        }

        int64_t k_quant_dim = (rope_mode == 0) ? head_dim : nope_dim;
        int64_t k_col_start = (rope_mode == 0) ? 0 : rope_dim;
        if (k_quant_dim > 0) {
            if (preset == 1) {
                deq_int4_sec(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
            } else {
                deq_int8_sec(k_ptr, head_dim, k_col_start, k_quant_dim, offset);
            }
        }

        if (v_ptr != nullptr) {
            if (preset == 1 || preset == 2) {
                deq_int4_sec(v_ptr, head_dim, 0, head_dim, offset);
            } else {
                deq_int8_sec(v_ptr, head_dim, 0, head_dim, offset);
            }
        }
        std::memset(p_ptr, 0, payload_len);
    };

#if defined(__APPLE__)
    auto fn_ptr = &decode_one_page;
    dispatch_apply_f(
        static_cast<size_t>(batch_size),
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        fn_ptr,
        [](void* ctx, size_t idx) {
            (*static_cast<decltype(fn_ptr)>(ctx))(idx);
        }
    );
#else
    for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
        decode_one_page(b);
    }
#endif

    for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
        if (status_codes[b] != 0) {
            out_k_pages.zero_();
            if (out_v_pages_opt.has_value() && out_v_pages_opt.value().defined()) {
                out_v_pages_opt.value().zero_();
            }
            if (status_codes[b] == 1) {
                throw std::runtime_error("KV_INTEGRITY_ERROR: Invalid cold-page header magic, version, or flags");
            } else if (status_codes[b] == 2) {
                throw std::runtime_error("CROSS_TENANT_CONTAMINATION: Cold-page AAD mismatch");
            } else {
                throw std::runtime_error("KV_INTEGRITY_ERROR: Cold-page MAC verification failed");
            }
        }
    }
}

// Standalone test-vector verification helpers for RFC 8439 ChaCha20 & Aumasson-Bernstein SipHash-2-4-128
torch::Tensor chacha20_rfc8439_xor_cpu(
    torch::Tensor data_u8,
    torch::Tensor key_u8,
    uint32_t initial_counter,
    torch::Tensor nonce_u8
) {
    TORCH_CHECK(data_u8.device().is_cpu() && data_u8.is_contiguous() && data_u8.scalar_type() == torch::kUInt8, "data_u8 must be contiguous CPU uint8");
    TORCH_CHECK(key_u8.device().is_cpu() && key_u8.is_contiguous() && key_u8.numel() == 32, "key_u8 must be 32-byte CPU uint8");
    TORCH_CHECK(nonce_u8.device().is_cpu() && nonce_u8.is_contiguous() && nonce_u8.numel() == 12, "nonce_u8 must be 12-byte CPU uint8");

    uint32_t key_words[8];
    uint32_t nonce_words[3];
    std::memcpy(key_words, key_u8.data_ptr<uint8_t>(), 32);
    std::memcpy(nonce_words, nonce_u8.data_ptr<uint8_t>(), 12);

    auto out = torch::empty_like(data_u8);
    chacha20_xor_bytes_rfc8439(
        data_u8.data_ptr<uint8_t>(),
        out.data_ptr<uint8_t>(),
        static_cast<size_t>(data_u8.numel()),
        key_words,
        initial_counter,
        nonce_words
    );
    return out;
}

torch::Tensor siphash24_128_cpu(
    torch::Tensor data_u8,
    uint64_t mk0,
    uint64_t mk1
) {
    TORCH_CHECK(data_u8.device().is_cpu() && data_u8.is_contiguous() && data_u8.scalar_type() == torch::kUInt8, "data_u8 must be contiguous CPU uint8");
    SipMac128State mac;
    mac.init(mk0, mk1);
    mac.absorb_bytes(data_u8.data_ptr<uint8_t>(), static_cast<size_t>(data_u8.numel()));
    auto out = torch::empty({16}, torch::TensorOptions().dtype(torch::kUInt8));
    mac.finalize(out.data_ptr<uint8_t>());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_encode_cold_page_cpu",
        &fused_encode_cold_page_cpu,
        "Single-pass C++ Fused Cold-Page Egress (RFC 8439 ChaCha20 + SipHash-2-4-128)",
        py::arg("k_page"),
        py::arg("v_page_opt"),
        py::arg("preset"),
        py::arg("rope_mode"),
        py::arg("rope_dim"),
        py::arg("expected_payload_bytes"),
        py::arg("tenant_id_hash"),
        py::arg("model_id_hash"),
        py::arg("layer_id"),
        py::arg("head_group_id"),
        py::arg("page_token_hash"),
        py::arg("sequence_nonce"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "fused_decode_cold_page_cpu",
        &fused_decode_cold_page_cpu,
        "Single-pass C++ Fused Cold-Page Ingress (SipHash-2-4-128 + RFC 8439 ChaCha20)",
        py::arg("cold_blob"),
        py::arg("num_tokens"),
        py::arg("num_heads"),
        py::arg("head_dim"),
        py::arg("rope_dim"),
        py::arg("is_mla_single_buffer"),
        py::arg("expected_tenant_hash"),
        py::arg("expected_model_hash"),
        py::arg("expected_layer_id"),
        py::arg("expected_token_hash"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "fused_decode_cold_pages_batch_cpu",
        &fused_decode_cold_pages_batch_cpu,
        "Multi-threaded GCD Batch Cold-Page Ingress (SipHash-2-4-128 + RFC 8439 ChaCha20)",
        py::arg("cold_blobs"),
        py::arg("expected_token_hashes"),
        py::arg("num_tokens"),
        py::arg("num_heads"),
        py::arg("head_dim"),
        py::arg("rope_dim"),
        py::arg("is_mla_single_buffer"),
        py::arg("expected_tenant_hash"),
        py::arg("expected_model_hash"),
        py::arg("expected_layer_id"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "fused_decode_cold_pages_peak_into_cpu",
        &fused_decode_cold_pages_peak_into_cpu,
        "Zero-Alloc GCD Peak Batch Cold-Page Ingress (SipHash-2-4-128 + RFC 8439 ChaCha20)",
        py::arg("cold_blobs"),
        py::arg("expected_token_hashes"),
        py::arg("out_k_pages"),
        py::arg("out_v_pages_opt"),
        py::arg("rope_dim"),
        py::arg("is_mla_single_buffer"),
        py::arg("expected_tenant_hash"),
        py::arg("expected_model_hash"),
        py::arg("expected_layer_id"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "fused_encode_cold_pages_batch_cpu",
        &fused_encode_cold_pages_batch_cpu,
        "Multi-threaded Batch Cold-Page Egress (RFC 8439 ChaCha20 + SipHash-2-4-128)",
        py::arg("k_pages"),
        py::arg("v_pages_opt"),
        py::arg("page_token_hashes"),
        py::arg("preset"),
        py::arg("rope_mode"),
        py::arg("rope_dim"),
        py::arg("expected_payload_bytes"),
        py::arg("tenant_id_hash"),
        py::arg("model_id_hash"),
        py::arg("layer_id"),
        py::arg("head_group_id"),
        py::arg("base_sequence_nonce"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "fused_encode_cold_pages_peak_into_cpu",
        &fused_encode_cold_pages_peak_into_cpu,
        "Zero-Alloc GCD Peak Batch Cold-Page Egress (RFC 8439 ChaCha20 + SipHash-2-4-128)",
        py::arg("k_pages"),
        py::arg("v_pages_opt"),
        py::arg("page_token_hashes"),
        py::arg("out_blobs"),
        py::arg("preset"),
        py::arg("rope_mode"),
        py::arg("rope_dim"),
        py::arg("expected_payload_bytes"),
        py::arg("tenant_id_hash"),
        py::arg("model_id_hash"),
        py::arg("layer_id"),
        py::arg("head_group_id"),
        py::arg("base_sequence_nonce"),
        py::arg("k0"),
        py::arg("k1"),
        py::arg("mk0"),
        py::arg("mk1"),
        py::arg("k2") = 0ULL,
        py::arg("k3") = 0ULL
    );
    m.def(
        "chacha20_rfc8439_xor_cpu",
        &chacha20_rfc8439_xor_cpu,
        "RFC 8439 ChaCha20 stream XOR verification entrypoint"
    );
    m.def(
        "siphash24_128_cpu",
        &siphash24_128_cpu,
        "Standard SipHash-2-4-128 MAC verification entrypoint"
    );
}
