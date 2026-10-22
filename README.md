# Zero-Trust Cold Pages (`ZTCP`)

**Zero-Trust Cold Pages (`ZTCP`)** — Fused RoPE-preserving KV-cache quantization & authenticated encryption (**256-bit IETF RFC 8439 ChaCha20 + 128-bit SipHash-2-4-128 Encrypt-then-MAC**, **26.65 ± 0.46 GB/s hot / 10.00 ± 0.17 GB/s wire egress**, **25.84 ± 0.28 GB/s hot ingress SIMD**) for distributed RDMA LLM serving (**SGLang HiCache**, **vLLM / LMCache**, **Mooncake**).

[![DOI](https://zenodo.org/badge/1427930948.svg)](https://doi.org/10.5281/zenodo.23527047)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper PDF](https://img.shields.io/badge/Paper-ZTCP__Paper.pdf-b31b1b.svg)](research/paper/ZTCP_Paper.pdf)
[![Benchmark Results](https://img.shields.io/badge/Benchmarks-26.65_GB%2Fs_Hot_(10.00_GB%2Fs_Wire)-10b981.svg)](research/benchmarks/RESULTS_LOG.md)
[![Theseus Guard](https://img.shields.io/badge/Theseus_Guard-DREP_%7C_ZT--ADP_%7C_TG--Sentinel:G1-1e3a8a.svg)](https://theseusguard.com)

- **Paper:** [Zero-Trust Cold Pages: Accelerating Distributed RDMA KV-Cache Offloading via Fused Asymmetric Quantization and Authenticated Encryption (`research/paper/ZTCP_Paper.pdf`)](research/paper/ZTCP_Paper.pdf)
- **DOI:** [`10.5281/zenodo.23527048`](https://doi.org/10.5281/zenodo.23527048) (Concept DOI: [`10.5281/zenodo.23527047`](https://doi.org/10.5281/zenodo.23527047))
- **Author:** Hakan Damar · Founder, Theseus Guard, Inc. (`hakan.damar@theseusguard.com`)
- **Full 5-Run Experimental Log & Raw Per-Process Artifact:** [`research/benchmarks/RESULTS_LOG.md`](research/benchmarks/RESULTS_LOG.md) · [`research/benchmarks/latest_results.json`](research/benchmarks/latest_results.json)

---

## Executive Summary

Disaggregated Large Language Model (LLM) serving and cross-node Key-Value (KV) cache pooling over RDMA (**Mooncake**, **LMCache**, **SGLang HiCache**, **NVIDIA NIXL**) suffer from a dual bottleneck:
1. **Bandwidth Bloat:** Evicted KV pages traverse PCIe and RDMA fabrics in their uncompressed runtime precision (`BF16`/`FP16`/`FP8`) or incur costly host-side temporary buffer copies (`L1Manager.reserve_write(temp_keys)`) during serialization.
2. **Plaintext Exposure & Cross-Tenant Contamination:** Sensitive activation vectors leave the compute node unencrypted or under a single shared fleet key, leaving multi-tenant pools vulnerable to prompt-inversion recovery, cross-request cache mixing, or remote memory poisoning.

**Zero-Trust Cold Pages (`ZTCP`)** addresses both problems by combining **RoPE-preserving asymmetric cold-page quantization** with **256-bit IETF RFC 8439 ChaCha20 stream encryption and 128-bit SipHash-2-4-128 Encrypt-then-MAC authentication** (distinct from ChaCha20-Poly1305) over the 32-byte AAD header and ordered ciphertext payload:
* **Hot KV stays untouched:** Active GPU Attention KV remains in its native runtime precision (`BF16`/`FP16`/`FP8`). Compression and authenticated encryption occur strictly at the cold-page egress/ingress boundary.
* **26.65 ± 0.46 GB/s Hot-Byte (`10.00 ± 0.17 GB/s` Wire) Egress & `25.84 ± 0.28 GB/s` Hot-Byte Ingress C++17 NEON Throughput ($N=5$ Independent Process Runs):** Across $N=5$ independent OS process runs ($\times 50$ batch iterations per run) on an Apple M3 Max CPU (`Qwen/Qwen3.6-35B-A3B` GQA page profile), our C++17 ARMv8.5-A FP16 NEON implementation achieves **`2.35 ± 0.01 GB/s` hot (`0.88 ± 0.01 GB/s` wire)** single-core egress (**`20.85x`** vs. multi-operator Python/PyTorch) and **`2.32 ± 0.01 GB/s` hot** single-core ingress, **`17.47 ± 0.43 GB/s` hot (`6.55 ± 0.16 GB/s` wire) egress** and **`23.02 ± 0.38 GB/s` hot ingress** on allocated 64-page batches, and **`26.65 ± 0.46 GB/s` hot (`10.00 ± 0.17 GB/s` wire) egress** and **`25.84 ± 0.28 GB/s` hot ingress** on 256-page batches into **preallocated CPU buffers** (**`236.05x` descriptive tier ratio** over multi-operator Python and **`11.32x`** over single-core C++; note that these ratios represent incremental implementation-tier comparisons across compounded runtime, MAC backend, batch size, threading, and buffer-allocation changes rather than controlled single-factor causal ablations isolating register fusion alone).
* **Exceeds 100GbE RoCEv2 Break-Even on CPU (`1.26 ± 0.01x` in Analytical Serial-Egress Model; `2.22x–2.58x` Projected on GPU SRAM):** By shrinking cold pages by **`2.67x–2.94x`** *before* encryption, ZTCP reduces cryptographic work by >62%. Under our consistent SI **analytical serial-egress 100GbE RoCEv2 model** (`11.2 GB/s = 11.2 × 10^9 B/s` effective goodput) for a 4,096-token prefix across 64 layers (`1,073.74 MB` SI / `1,024.0 MiB`):
  * **Analytical Serial Latency Using Measured Apple M3 Max CPU Compute + 100GbE Wire:** Totals **`76.28 ± 0.70 ms`** (`40.31 ± 0.70 ms` measured CPU + `35.97 ms` modeled wire) for `INT4_SYM` versus **`95.87 ms`** for unencrypted raw `BF16` (**`1.26 ± 0.01x`** speedup, consistent across both the ratio of 5-run mean latencies `95.87 / 76.28 = 1.2568x` and the mean of 2-decimal rounded per-run ratios `[1.25, 1.27, 1.27, 1.24, 1.26]` in `latest_results.json`), and **`87.15 ± 0.53 ms`** (**`1.10 ± 0.01x`** speedup) for `K8V4_ASYM`.
  * **Analytical Serial Latency Using Hypothetical GPU SRAM (`950 GB/s`) + 100GbE Wire:** Projects **`37.10 ms`** (`2.58x` vs. raw `BF16`, `1.29x` vs. raw `FP8`) for `INT4_SYM` and **`43.09 ms`** (`2.22x` vs. raw `BF16`) for `K8V4_ASYM`.
* **Synthetic RoPE-Outlier Single-Page Fidelity (`KV seed=42`, Device-Specific `Query seed=99`):** On synthetic KV pages with structured RoPE outlier injection, preserving the 64-element RoPE sub-vector in 8-bit (`PRESERVE_FP8`) while quantizing NoPE and Value channels to 4-bit achieves **`0.996196` (Python mixed-FP32/FP16) / `0.996103` (C++17 NEON on `mps`-synthesized `seed=42` KV; `0.995925` on `cpu`-synthesized `seed=42` KV) simulated Attention Output Cosine Similarity** on GQA (`Qwen/Qwen3.6-35B-A3B` profile, `V` present) and **`0.999473` (Python) / `0.999520` (C++17 NEON) Softmax Attention Weights Cosine Similarity** on single-buffer DeepSeek-MLA (`DeepSeek-V4.1-Flash` profile, `V is None`).

---

## Visual Benchmark Highlights

### 1. Kernel Throughput Tiers & 100GbE Break-Even vs. Wire Saturation
![Figure 1: Kernel Throughput Progression](research/paper/figures/fig1_throughput_progression.svg?v=6)

### 2. Analytical Serial-Egress 100GbE Prefix Transfer Model (Measured CPU vs. Projected GPU)
![Figure 2: RDMA Transfer Latency Comparison](research/paper/figures/fig2_rdma_latency_comparison.svg?v=6)

### 3. Synthetic RoPE-Outlier Fidelity vs. Wire Compression Ratio (GQA Output Cos vs. MLA Weights Cos)
![Figure 3: RoPE Fidelity Pareto](research/paper/figures/fig3_rope_fidelity_pareto.svg?v=6)

---

## Break-Even Bound vs. Link Wire-Saturation Bound

Let $B_{\text{hot}}$ be the byte size of a hot KV page, $r = B_{\text{hot}} / B_{\text{cold}} > 1$ be the net cold-page compression ratio (including the 32B AAD header and 16B MAC tag), $BW_{\text{net}}$ be the effective RDMA network goodput, $BW_{\text{fused,hot}} = B_{\text{hot}} / t_{\text{page}}$ be the hot-byte processing throughput, and $BW_{\text{fused,wire}} = B_{\text{cold}} / t_{\text{page}} = BW_{\text{fused,hot}} / r$ be the encoded wire output throughput (all in SI $\text{GB/s} = 10^9\text{ B/s}$):

1. **Serial Compress-then-Encrypt Break-Even Bound (Theorem 1):** Authenticated, encrypted serial transfer is **strictly faster than unencrypted raw transfer** ($T_{\text{ztcp}} < T_{\text{raw}}$) if and only if:
   $$BW_{\text{fused,hot}} > \left(\frac{r}{r - 1}\right) BW_{\text{net}} \iff BW_{\text{fused,wire}} > \left(\frac{1}{r - 1}\right) BW_{\text{net}}$$
2. **Full Link Wire-Saturation Bound (Corollary 1):** Keeping the RDMA link 100% saturated with compressed cold blobs requires:
   $$BW_{\text{fused,wire}} \ge BW_{\text{net}} \iff BW_{\text{fused,hot}} \ge r \cdot BW_{\text{net}}$$

For GQA cold pages with `INT4_SYM + Preserve RoPE (FP8)` ($r = 262,144 / 98,352 = 2.6654$, $\frac{r}{r-1} = 1.6005$):
* **100GbE RoCEv2 ($BW_{\text{net}} = 11.2\text{ GB/s}$):**
  * **Break-even** requires $BW_{\text{fused,hot}} > \mathbf{17.93\text{ GB/s}}$ (`6.73 GB/s` wire). Our measured M3 Max CPU kernel (**`26.65 ± 0.46 GB/s` hot / `10.00 ± 0.17 GB/s` wire**) exceeds 100GbE break-even by **48.6%**, yielding a **`1.26 ± 0.01x` analytical serial speedup** (`76.28 ms` vs. `95.87 ms`), while `K8V4_ASYM` (`23.76 ± 0.27 GB/s` hot / `10.40 ± 0.12 GB/s` wire) also exceeds its `19.92 GB/s` hot break-even bound by **19.3%** (`1.10 ± 0.01x` speedup, `87.15 ms` vs. `95.87 ms`).
  * **Full wire saturation** requires $BW_{\text{fused,hot}} \ge \mathbf{29.85\text{ GB/s}}$ (`11.20 GB/s` wire); in the analytical link model ($BW_{\text{fused,wire}} / BW_{\text{net}}$, not a live NIC measurement), the single-socket CPU kernel corresponds to **89.3% (`INT4_SYM`) to 92.9% (`K8V4_ASYM`) of modeled 100GbE wire capacity**.
* **400GbE NDR InfiniBand ($BW_{\text{net}} = 45.0\text{ GB/s}$):**
  * **Break-even** requires $BW_{\text{fused,hot}} > \mathbf{72.02\text{ GB/s}}$ (`27.02 GB/s` wire) and **full wire saturation** requires $BW_{\text{fused,hot}} \ge \mathbf{119.94\text{ GB/s}}$ (`45.00 GB/s` wire), motivating GPU SRAM execution.

---

## Repository Structure

```text
ZTCP/
└── research/
    ├── baseline_codec/
    │   ├── cold_page_spec.py           # 32B AAD Header + Wire Format Specification
    │   ├── fused_quant_crypto.py       # Transactional KeyRing + Reference RFC 8439 ChaCha20 Codec
    │   ├── native_fused_codec.py       # JIT Loader & Generation-Bound Wrapper for C++17 Extension
    │   └── csrc/
    │       └── fused_cold_page_cpu.cpp # Unified C++17 ARMv8.5-A FP16 NEON + ChaCha20 + SipHash-2-4-128 Kernel
    ├── benchmarks/
    │   ├── bench_cold_page_pipeline.py # Reproducible 5-Run Process Benchmark & 12-Part Security Suite
    │   ├── RESULTS_LOG.md              # Verified 5-Run Experimental Tables, Tier Comparisons & Caveats
    │   └── latest_results.json         # Machine-Readable 5-Run Artifact (with Raw Per-Process Runs)
    └── paper/
        ├── ZTCP_Paper.pdf              # Compiled Two-Column Academic Preprint
        ├── main.tex                    # LaTeX Manuscript Source
        ├── references.bib              # BibTeX Bibliography
        └── figures/                    # Pre-Rendered Vector SVG Figures Plotted from latest_results.json
```

---

## Quick Start: Reproduce All 5 Independent Process Runs (`PKG-006`)

### Verified Platform & Prerequisites
* **Verified Hardware / Toolchain:** Apple Silicon (`arm64` macOS) with Apple `clang++` supporting `-march=armv8.5-a+fp16` (note: `research/baseline_codec/native_fused_codec.py` passes `-march=armv8.5-a+fp16` unconditionally on POSIX builds; building on `x86_64` requires modifying `extra_cflags`).
* **Python Environment:** Python 3.10+ with `torch >= 2.2`, `numpy`, and `ninja` (for JIT C++17 native extension compilation).

### Run the 5-Run Benchmark Suite & Non-Destructive Aggregation Check
```bash
# 1. Spawn 5 independent OS child processes (--runs 5), save each raw run in
#    latest_results.json["independent_runs"], and write aggregated stats (overwrites latest_results.json):
python3 research/benchmarks/bench_cold_page_pipeline.py --runs 5

# 2. Verify deterministic re-aggregation from saved independent_runs WITHOUT modifying
#    the tracked latest_results.json (note: --reaggregate-only writes to its target path
#    and updates timestamp_utc, so verify against a temporary copy):
cp research/benchmarks/latest_results.json /tmp/ztcp_reagg_check.json
python3 research/benchmarks/bench_cold_page_pipeline.py --reaggregate-only --output-json /tmp/ztcp_reagg_check.json
```

---

## Implementation Scope, Protocol Contracts & Caveats (`PUB-001`–`PUB-005`)

1. **Execution Pass, Quantizer Arithmetic & L1 Scale Staging (`B3`, `DS-P3-04`):**
   * In `csrc/fused_cold_page_cpu.cpp`, ARMv8.5-A NEON quantization computes `FP16` group abs-max (`vmaxvq_f16`), widens to `FP32` to compute the group scale and exact IEEE-754 `FP32` reciprocal division (vectorized `vdivq_f32` in 4-group kernels; scalar `1.0f / static_cast<float>(sc)` in 2-group fallback), narrows that reciprocal to `FP16` (`vcvt_f16_f32` / `vdupq_n_f16`), and performs vectorized `FP16` multiplication and round-to-nearest-even (`vmulq_f16`, `vcvtnq_s16_f16`). The Python reference codec (`FusedColdPageCodec`) is mixed-precision (`FP16` tensors/scales with `FP32` intermediate division).
   * Quantized vectors (`q_rope`, `q_nope`, `q_v`, **83.3%** of GQA `INT4_SYM` payload) are quantized, RFC 8439 ChaCha20-encrypted, and SipHash-2-4-128 absorbed in registers before a single store. To preserve the contiguous `[q_section][s_section]` wire layout without reading back destination memory, per-group `FP16` scales (`s_rope`, `s_nope`, `s_v`, **16.7%** of payload, `2–4 KB` per section on GQA) are staged in an L1 stack array (`stack_scales[8192]`, with `std::vector` heap fallback only for non-standard geometries exceeding 8,192 groups/section) and encrypted + MAC-absorbed immediately at the end of each section.
2. **In-Process Keyring Concurrency, Linearization & Buffer Ownership Contracts (`B5`, `B6`, `CODE-003`, `PUB-001`, `PUB-004`):**
   * **Domain-Separated HMAC-SHA256 Derivation:** `EphemeralSessionKeyRing.derive_or_get` derives per-generation keys via `prk = HMAC-SHA256(deployment_pepper, tenant:session:gen || os.urandom(32))` followed by domain-separated `HMAC-SHA256(prk, "ztcp-cold-page-enc-v1\x01")` and `HMAC-SHA256(prk, "ztcp-cold-page-mac-v1\x02")` (distinct from multi-block RFC 5869 HKDF-Expand chaining).
   * **Single-Process Lock & Atomic Nonce Burn:** All session state transitions are synchronized under `threading.RLock` within a single Python process. Nonces are atomically burned in `preflight_validate_egress` before encryption (`1 success + 1 NonceReuseError` on concurrent duplicate-nonce races; failed encodes keep their nonces burned), and per-slot freshness watermarks advance strictly monotonically (`max()`).
   * **Pre-Shred Linearization vs. Post-Revocation Abort:** If an operation completes its locked `commit_egress` or `finish_ingress` check *before* `crypto_shred_session` / `crypto_shred_tenant` acquires the lock, that operation is linearized prior to the shred and may return normally after the concurrent shred call returns. If `crypto_shred_session` acquires the lock *before* `begin_egress_execution`, `commit_egress`, or `finish_ingress`, the operation zeroes outputs populated inside its execution block (`out_blobs_u8.zero_()` / `k_out.zero_()`) and raises `SessionShreddedError`.
   * **Caller Exclusive Peak Buffer Ownership & Exception Invalidation:** The caller of `encode_cold_pages_peak_into` (and `decode_cold_pages_peak_into`) must exclusively own the preallocated CPU tensor buffer, read or publish it only after a normal return, and discard it as invalid on *every* raised exception; failures during `preflight_validate_egress` or input/output buffer-overlap checks in `native_fused_codec.py` occur prior to the `try ... except` buffer-zeroing handler and do not clear previous buffer contents.
   * **User-Space Zeroization Boundary:** While active and staged mutable `bytearray(32)` key buffers are overwritten in-place on shred and per-call mutable snapshots are wiped in `finally` blocks, immutable Python `bytes` objects, unpacked integer/stack key words (`k0..k3`, `mk0..mk1`) inside native C++ stack frames, CPython GC temporaries, and OS swap/core-dumps remain outside any strong hardware memory-erasure guarantee.
3. **Wire Format Version, MAC Backend Non-Interoperability & Binding Defaults:**
   * `FORMAT_VERSION = 1` is retained in the 32B header across this research PoC.
   * The Python reference codec (`FusedColdPageCodec`) uses 128-bit `BLAKE2b` whereas the native C++17 codec (`NativeCppFusedColdPageCodec`) uses 128-bit `SipHash-2-4-128`; each backend authenticates and decodes its own blobs.
   * Raw pybind11 entrypoints keep default `k2=0, k3=0` arguments for legacy signature compatibility, while `NativeCppFusedColdPageCodec` always unpacks and passes all 256 bits (`k0..k3`) of the 32-byte encryption key.
4. **Public PoC vs. Target Enterprise Hardware & Regression Scope (`B8`, `PUB-005`, `PUB-005-R2`):** This public repository empirically benchmarks the CPU reference and C++17 ARMv8.5-A FP16 NEON codec, in-process session keyring, Python/CPU upstream hook ABIs, in-memory simulated bit-flip/tamper tests, and analytical serial-egress RDMA link models. In the bundled `bench_cold_page_pipeline.py` security suite, tests `3.1–3.4` run on the Python reference codec, `3.5–3.9` and `3.11` run on the native C++17 `Peak` path, `3.10a–c` exercises the Python `pack_chacha20_nonce` helper plus Native `Peak` duplicate-nonce and stale-replay checks, and `3.12` exercises Native `Peak` same-nonce race + Native `Single` failed-encode nonce burn and monotonic watermark checks. In-flight egress and ingress shred checks across all four codec paths (`py_single`, `cpp_single`, `cpp_batch`, `cpp_peak`) are verified in `verify_package2.py`, `CODE-003`; overlapping-batch checks are verified in `B6-2` (separate local regression suite). Hardware TEE enclave attestation (NVIDIA H100/B200 Confidential Computing, Intel TDX, AMD SEV-SNP), GPU SM90/SM100 CUDA/Triton kernels, live RoCEv2/InfiniBand NIC transfers, and NIST FIPS 203 (`ML-KEM`) / FIPS 204 (`ML-DSA`) control-plane handshakes are target enterprise/production components described for architectural context.

For enterprise licensing or architecture inquiries, contact **TG-Team** at [contact@theseusguard.com](mailto:contact@theseusguard.com) or [https://www.theseusguard.com/contact](https://www.theseusguard.com/contact).

---

## Citation

```bibtex
@techreport{damar2026ztcp,
  author      = {Damar, Hakan},
  title       = {Zero-Trust Cold Pages: Accelerating Distributed RDMA KV-Cache Offloading via Fused Asymmetric Quantization and Authenticated Encryption},
  institution = {Theseus Guard Research},
  year        = {2026},
  month       = oct,
  doi         = {10.5281/zenodo.23527048},
  url         = {https://doi.org/10.5281/zenodo.23527048}
}
```
