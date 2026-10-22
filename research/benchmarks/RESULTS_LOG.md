<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Hakan Damar and Theseus Guard, Inc.
Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
Developed by Theseus Guard Research (https://www.theseusguard.com)
Repository: https://github.com/Theseus-Guard/ZTCP
Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
-->

# Zero-Trust Cold Pages (ZTCP) — Experimental Benchmark & Verification Log (`RESULTS_LOG.md`)

- **Author & Organization:** Hakan Damar · Theseus Guard, Inc. (`https://www.theseusguard.com`)
- **Last Updated:** October 22, 2026
- **Verified Hardware & Compiler Environment (`PKG-006`):** Apple M3 Max (`arm64-apple-darwin`, 12 Performance Cores + 4 Efficiency Cores, 64 GB Unified Memory), Apple Clang (`-O3 -std=c++17 -march=armv8.5-a+fp16 -ffast-math` + ARMv8.5-A FP16 NEON SIMD), Python 3.12.11, PyTorch 2.10.0. *(Note: `native_fused_codec.py` passes `-march=armv8.5-a+fp16` unconditionally on POSIX builds; reproduction is verified on Apple Silicon `arm64`.)*
- **Cryptographic & Key-Derivation Construction (`B7`, `PUB-004`):** 256-bit IETF RFC 8439 ChaCha20 stream encryption (`counter=1`, bijective 96-bit nonce `sequence_nonce (64b) || layer_id (16b) || head_group_id (16b)`) + Encrypt-then-MAC (`128-bit SipHash-2-4-128` in C++17 native kernel; `128-bit keyed BLAKE2b` in Python reference codec) over `32B AAD Header || Ordered Encrypted Payload` (covering all quantized sub-vectors, encrypted FP16 scales, and 64B-alignment tail padding), paired with **HMAC-SHA256-based, domain-separated per-generation key derivation** (`prk = HMAC-SHA256(deployment_pepper, tenant:session:gen || csprng_32b)`, `enc = HMAC-SHA256(prk, "ztcp-cold-page-enc-v1\x01")`, `mac = HMAC-SHA256(prk, "ztcp-cold-page-mac-v1\x02")`). *(Note: This construction is RFC 8439 ChaCha20 + SipHash-2-4-128 Encrypt-then-MAC with domain-separated HMAC-SHA256 key derivation, not ChaCha20-Poly1305 or multi-block RFC 5869 HKDF-Expand chaining.)*
- **Machine-Readable 5-Run Artifact (incl. Raw Per-Process Runs):** [`research/benchmarks/latest_results.json`](latest_results.json)
- **Reproducible 5-Run Execution & Non-Destructive Re-Aggregation Commands (`PKG-006`):**
```bash
# 1. Execute 5 fresh independent OS child-process runs (overwrites latest_results.json):
python3 research/benchmarks/bench_cold_page_pipeline.py --runs 5

# 2. Verify aggregation from saved independent_runs WITHOUT modifying the tracked artifact
#    (note: --reaggregate-only writes to its target path and updates timestamp_utc):
cp research/benchmarks/latest_results.json /tmp/ztcp_reagg_check.json
python3 research/benchmarks/bench_cold_page_pipeline.py --reaggregate-only --output-json /tmp/ztcp_reagg_check.json
```

---

## Experiment 1: Wire Compression Ratio & Synthetic RoPE-Outlier Fidelity (`KV seed=42`, `Query seed=99`)

> **Methodology, Metric Semantics, Precision & RNG Scope (`B11`, `DS-P3-01 / PUB-002`, `DS-P3-03`, `DS-P3-04`):**
> 1. **Synthetic Single-Page Proxy vs. Real LLM Quality (`B11`):** The fidelity metrics below are evaluated on **synthetic KV pages with structured sinusoidal RoPE outlier injection** (`generate_realistic_kv_page` with leading 64 RoPE dimensions scaled by $\pm 4.5 \sin(t)$ vs. $\mathcal{N}(0, 0.4^2)$ NoPE and $\mathcal{N}(0, 0.25^2)$ Value channels, modeling `Qwen/Qwen3.6-35B-A3B` GQA and `DeepSeek-V4.1-Flash` MLA page geometries) and a single-page simulated attention projection (`evaluate_attention_fidelity`). They measure **local tensor and single-page attention proxy fidelity**, not end-to-end LLM perplexity or downstream task accuracy on full pretrained model weights.
> 2. **GQA Attention Output Cosine vs. Single-Buffer MLA Attention Weights Cosine (`DS-P3-01 / PUB-002`):** In `bench_cold_page_pipeline.py:129–140`, when a separate Value tensor is present (**GQA-Standard**, Table 1.1), `evaluate_attention_fidelity` computes **simulated Attention Output Cosine Similarity** $\cos(\text{softmax}(QK^\top/\sqrt{d})V,\, \text{softmax}(Q\hat{K}^\top/\sqrt{d})\hat{V})$. When `v_orig is None` (**single-buffer DeepSeek-MLA**, Table 1.2), it computes **Softmax Attention Weights Cosine Similarity** $\cos(\text{softmax}(QK^\top/\sqrt{d}),\, \text{softmax}(Q\hat{K}^\top/\sqrt{d}))$ (stored under the shared `py_attn_out_cos` / `cpp_attn_out_cos` keys in `latest_results.json` for schema uniformity and labeled via `attn_metric_kind`).
> 3. **Quantizer Arithmetic & Mixed Precision (`DS-P3-04`):** The Python reference codec (`FusedColdPageCodec`) is **mixed-precision**: inputs, outputs, and stored group scales are `FP16`, while intermediate group abs-max division and rounding are computed in `FP32`. The native C++17 ARMv8.5-A NEON codec (`csrc/fused_cold_page_cpu.cpp`) computes `FP16` group abs-max via `vmaxvq_f16`, widens to `FP32` to compute the scale (`fmax * (1.0f / 7.0f)` or `1.0f / 127.0f`, stored as `FP16`) and exact IEEE-754 **`FP32` reciprocal division** (vectorized `vdivq_f32` in 4-group kernels; scalar `1.0f / static_cast<float>(sc)` in 2-group fallback), narrows that reciprocal to `FP16` (`vcvt_f16_f32` / `vdupq_n_f16`), and performs vectorized `FP16` multiplication and round-to-nearest-even (`vmulq_f16`, `vcvtnq_s16_f16`). It does **not** invoke `vrecpeq_f16` / `vrecpsq_f16` reciprocal-approximation intrinsics.
> 4. **KV & Query RNG Seeds Across Devices (`DS-P3-03`, `DS-P3-04`):** `generate_realistic_kv_page` always resets `torch.manual_seed(42)` on its target `device`, and `evaluate_attention_fidelity` always resets `torch.manual_seed(99)` on `k_ref.device` when sampling the random query vector `q`. In Experiment 1, when `mps` is available, the `seed=42` KV page is synthesized on `mps` and copied to `cpu` for C++ evaluation; therefore, `evaluate_attention_fidelity` samples `q` (`seed=99`) from the `mps` RNG for Python and from the `cpu` RNG for C++. Consequently, differences between the Python and C++ cosine columns reflect **both** quantizer rounding differences and device-specific query RNG streams. Similarly, the C++ GQA cosine value in Experiment 1 (`0.996103`, `mps`-generated `seed=42` KV copied to `cpu`) and the C++ GQA warmup cosine in Experiment 2 (`0.995925`, `cpu`-generated `seed=42` KV) are both `seed=42` KV pages with `seed=99` CPU queries, differing only by the device RNG used during KV synthesis (not a second independent `seed=99` KV sample).

### 1.1 GQA-Standard Geometry (`Qwen/Qwen3.6-35B-A3B` Profile: `64 tokens/page`, `8 KV heads`, `head_dim=128`, `rope_dim=64`, Hot Page = `262,144 B` / `256.0 KiB` / `0.262 MB`) — *Metric: Simulated Attention Output Cosine (`V` Present)*
| Configuration | Cold Blob Size (incl. 32B AAD + 16B MAC) | Net Compression Ratio | Py Key CosSim (`MPS`) | Py Mixed-FP32/FP16 Attn Output CosSim (`MPS q`) | Native C++17 NEON Attn Output CosSim (`CPU q`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `INT4_SYM + Quantize All` (No RoPE Preservation) | `81,968 B` | **3.20x** (`3.1981`) | `0.99820` | `0.992378` | `0.993649` |
| **`INT4_SYM + Preserve RoPE (FP8)`** | **`98,352 B`** | **2.67x** (`2.6654`) | **`0.99984`** | **`0.996196`** | **`0.996103`** (`0.995925` on CPU-synthesized `seed=42` KV) |
| `INT4_SYM + Preserve RoPE (Raw FP16)` | `127,024 B` | **2.06x** (`2.0637`) | `0.99984` | `0.996194` | `0.996088` |
| **`K8V4_ASYM + Preserve RoPE (FP8)`** | **`114,736 B`** | **2.28x** (`2.2848`) | **`0.99999`** | **`0.996316`** | **`0.996354`** |
| `FP8_ONLY (2x Baseline)` | `147,504 B` | **1.78x** (`1.7772`) | `0.99999` | `0.999974` | `0.999974` |

### 1.2 Single-Buffer DeepSeek-MLA Geometry (`DeepSeek-V4.1-Flash` Profile: `128 tokens/page`, `1 latent head`, `head_dim=576`, `rope_dim=64`, Hot Page = `147,456 B` / `144.0 KiB` / `0.147 MB`) — *Metric: Softmax Attention Weights Cosine (`V is None`)*
| Configuration | Cold Blob Size (incl. 32B AAD + 16B MAC) | Net Compression Ratio | Py Latent K CosSim (`MPS`) | Py Mixed-FP32/FP16 Attn Weights CosSim (`MPS q`) | Native C++17 NEON Attn Weights CosSim (`CPU q`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `INT4_SYM + Quantize All` (No RoPE Preservation) | `46,128 B` | **3.20x** (`3.1967`) | `0.99781` | `0.998846` | `0.998615` |
| **`INT4_SYM + Preserve RoPE (FP8)`** | **`50,224 B`** | **2.94x** (`2.9360`) | **`0.99905`** | **`0.999473`** | **`0.999520`** |
| `INT4_SYM + Preserve RoPE (Raw FP16)` | `57,392 B` | **2.57x** (`2.5693`) | `0.99906` | `0.999517` | `0.999521` |
| **`K8V4_ASYM + Preserve RoPE (FP8)`** | **`82,992 B`** | **1.78x** (`1.7767`) | **`0.99999`** | **`0.999994`** | **`0.999995`** |
| `FP8_ONLY (2x Baseline)` | `82,992 B` | **1.78x** (`1.7767`) | `0.99999` | `0.999994` | `0.999995` |

---

## Experiment 2: Fused Kernel Micro-Benchmark & Descriptive Implementation-Tier Comparisons ($N=5$ Independent OS Process Runs)

**Workload:** Encode a `262,144 B` (`256.0 KiB` / `0.262144 MB`) hot GQA KV page (`Qwen/Qwen3.6-35B-A3B` geometry) into `RoPE-FP8 + NoPE-INT4 + V-INT4` (`98,352 B` cold blob, compression ratio $r = 2.6654\times$), encrypt all quantized sections, FP16 scales, and padding with 256-bit IETF RFC 8439 ChaCha20, and compute a 128-bit MAC (`BLAKE2b` in Tier 1 Python; `SipHash-2-4-128` in Tiers 2–4 C++17) over the 32-byte AAD header + ordered ciphertext.

> **Execution Pass & Scale Staging Architecture (`B3`):** Quantized vectors (`q_rope`, `q_nope`, `q_v`, accounting for **83.3%** of the GQA `INT4_SYM + Preserve RoPE FP8` payload) are quantized in 4-group/8-group ARMv8.5-A NEON passes, ChaCha20-encrypted, and SipHash-2-4-128 absorbed directly in 32B/64B aligned NEON register super-blocks prior to a single memory store. To preserve the contiguous `[q_section][s_section]` wire layout without reading back destination memory, per-group `FP16` scales (`s_rope`, `s_nope`, `s_v`, accounting for **16.7%** of payload, `2–4 KB` per section on GQA) are staged in a fixed L1 stack array (`stack_scales`, up to `8,192` halfs = `16 KB`, with `std::vector` heap fallback only for oversized non-standard geometries) and encrypted + MAC-absorbed immediately at the end of each section.

> **Throughput Metrics & Incremental Implementation-Tier Interpretation (`B10`, `B12`, `DS-P3-02 / PUB-003`):**
> * **Hot-Byte Throughput (`GB/s hot`):** Hot KV bytes processed per second ($B_{\text{hot}} / t$, where $1\text{ GB} = 10^9\text{ B}$).
> * **Encoded Wire Throughput (`GB/s wire`):** Compressed + encrypted cold blob bytes emitted per second ($B_{\text{cold}} / t = \text{Hot GB/s} / r$).
> * **Descriptive Implementation-Tier Comparisons (Not Controlled Single-Factor Causal Ablations):** The cumulative **`236.05 ± 4.76x`** ratio from Tier 1 to Tier 4a and the intermediate tier ratios (**`20.85x`**, **`7.42x`**, **`1.53x`** / **`11.32x`**) are **incremental implementation-tier comparisons across compounded system changes**, not controlled single-factor causal ablations isolating register fusion alone:
>   1. **Tier 1 $\rightarrow$ Tier 2 (`20.85 ± 0.29x` vs. Multi-Op Python, 1 Core):** Simultaneously changes the execution runtime (multi-operator Python/PyTorch $\rightarrow$ compiled C++17 `-O3` with ARMv8.5-A FP16 NEON), quantizer scale arithmetic (FP32 group division $\rightarrow$ vectorized `vdivq_f32` / scalar FP32 reciprocal narrowed to FP16 + `vmulq_f16`/`vcvtnq_s16_f16`), MAC algorithm (`BLAKE2b-128` $\rightarrow$ `SipHash-2-4-128`), and memory pass structure (discrete intermediate tensors $\rightarrow$ in-register 32B/64B vector encryption/MAC absorption with L1 stack scale staging), reducing single-page egress latency from `2,322.36 ± 36.21 µs` to `111.40 ± 0.56 µs` (`2.35 ± 0.01 GB/s` hot / `0.88 ± 0.01 GB/s` wire).
>   2. **Tier 2 $\rightarrow$ Tier 3 (`7.42 ± 0.20x` over 1-Core C++17; `154.77 ± 5.51x` vs. Python):** Evaluates a 64-page batch (`16.78 MB` / `16.0 MiB`) parallelized via GCD chunked worker dispatch (`dispatch_apply_f`), including per-call output tensor allocation (`torch::empty({64, 98352})`) and transactional keyring nonce reservation (`17.47 ± 0.43 GB/s` hot / `6.55 ± 0.16 GB/s` wire egress; `23.02 ± 0.38 GB/s` hot ingress).
>   3. **Tier 3 $\rightarrow$ Tier 4a/4b (`1.53x` over Tier 3; `11.32 ± 0.23x` over 1-Core C++17; `236.05 ± 4.76x` vs. Python):** Simultaneously increases batch size (`64` $\rightarrow$ `256` pages = `67.11 MB` / `64.0 MiB`) and writes directly into a **caller-preallocated CPU `uint8` tensor buffer** (`out_pinned_ring`, avoiding per-call output tensor allocation in the timing loop; note that this buffer is a standard preallocated CPU tensor and is not registered with an RDMA NIC via `ibv_reg_mr`). This tier reaches **`26.65 ± 0.46 GB/s` hot (`10.00 ± 0.17 GB/s` wire) egress** and **`25.84 ± 0.28 GB/s` hot (`10.15 ± 0.11 µs/page`) ingress** for `INT4_SYM`, and **`23.76 ± 0.27 GB/s` hot (`10.40 ± 0.12 GB/s` wire) egress** for `K8V4_ASYM`.

| Measured Implementation Tier (Apple M3 Max, $N=5$ Process Runs) | Egress (`µs/page`) | Ingress (`µs/page`) | Hot-Byte Throughput (`GB/s`) | Encoded Wire Throughput (`GB/s`) | Tier Ratio vs. Python | Tier Ratio vs. 1c C++ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. Multi-Op Python / PyTorch (1-Core, RFC 8439 ChaCha20 + BLAKE2b)** | `2,322.36 ± 36.21 µs` | `1,838.33 ± 40.77 µs` | `0.11 ± 0.00 GB/s` | `0.04 ± 0.00 GB/s` | `1.00x (Base)` | `0.05x` |
| **2. C++17 Single-Core Fused (`-O3` FP16 NEON + ChaCha20 + SipHash-2-4-128)** | **`111.40 ± 0.56 µs`** | **`113.07 ± 0.44 µs`** | **`2.35 ± 0.01 GB/s`** (`2.32 ± 0.01` dec) | **`0.88 ± 0.01 GB/s`** | **`20.85 ± 0.29x`** | **`1.00x`** |
| **3. C++17 Multi-Core Batch (64 Pages = `16.78 MB` / `16.0 MiB`, w/ Output Alloc)** | **`15.02 ± 0.38 µs`** (`0.961 ms`) | **`11.39 ± 0.19 µs`** | **`17.47 ± 0.43 GB/s`** (`23.02 ± 0.38` dec) | **`6.55 ± 0.16 GB/s`** | **`154.77 ± 5.51x`** | **`7.42 ± 0.20x`** |
| **4a. C++17 Peak Preallocated CPU Buffer `INT4_SYM` (256 Pages = `67.11 MB` / `64.0 MiB`)** | **`9.84 ± 0.17 µs`** (`2.519 ms`) | **`10.15 ± 0.11 µs`** | **`26.65 ± 0.46 GB/s`** (`25.84 ± 0.28` dec) | **`10.00 ± 0.17 GB/s`** (`80.0 Gbps`) | **`236.05 ± 4.76x`** | **`11.32 ± 0.23x`** |
| **4b. C++17 Peak Preallocated CPU Buffer `K8V4_ASYM` (256 Pages = `67.11 MB` / `64.0 MiB`)** | **`11.04 ± 0.13 µs`** (`2.825 ms`) | — | **`23.76 ± 0.27 GB/s`** (`190.1 Gbps`) | **`10.40 ± 0.12 GB/s`** (`83.2 Gbps`) | **`210.36x`** | **`10.09x`** |

### Multi-Run Statistical Verification ($N=5$ Independent OS Process Runs × 50 Iterations)
| Process Run | OS PID | 1-Core Egress (`µs`) | 1-Core Hot (`GB/s`) | 1-Core Ingress (`GB/s`) | Batch 64p Hot (`GB/s`) | Peak 256p Total (`ms`) | Peak 256p (`µs/page`) | Peak 256p Hot (`GB/s`) | Peak 256p Wire (`GB/s`) | Peak 256p Ingress (`GB/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Run 1** | `19786` | `111.38 µs` | `2.35 GB/s` | `2.32 GB/s` | `16.76 GB/s` | `2.541 ms` | `9.93 µs` | **`26.41 GB/s`** | **`9.91 GB/s`** | **`25.86 GB/s`** |
| **Run 2** | `19796` | `112.37 µs` | `2.33 GB/s` | `2.32 GB/s` | `17.93 GB/s` | `2.477 ms` | `9.68 µs` | **`27.09 GB/s`** | **`10.16 GB/s`** | **`25.47 GB/s`** |
| **Run 3** | `19802` | `111.04 µs` | `2.36 GB/s` | `2.33 GB/s` | `17.55 GB/s` | `2.480 ms` | `9.69 µs` | **`27.06 GB/s`** | **`10.15 GB/s`** | **`26.23 GB/s`** |
| **Run 4** | `19808` | `111.07 µs` | `2.36 GB/s` | `2.32 GB/s` | `17.57 GB/s` | `2.581 ms` | `10.08 µs` | **`26.00 GB/s`** | **`9.75 GB/s`** | **`25.71 GB/s`** |
| **Run 5** | `19827` | `111.14 µs` | `2.36 GB/s` | `2.31 GB/s` | `17.52 GB/s` | `2.516 ms` | `9.83 µs` | **`26.67 GB/s`** | **`10.01 GB/s`** | **`25.95 GB/s`** |
| **Mean ± Std** | — | **`111.40 ± 0.56 µs`** | **`2.35 ± 0.01 GB/s`** | **`2.32 ± 0.01 GB/s`** | **`17.47 ± 0.43 GB/s`** | **`2.519 ± 0.044 ms`** | **`9.84 ± 0.17 µs`** | **`26.65 ± 0.46 GB/s`** | **`10.00 ± 0.17 GB/s`** | **`25.84 ± 0.28 GB/s`** |
| **Median** | — | **`111.14 µs`** | **`2.36 GB/s`** | **`2.32 GB/s`** | **`17.55 GB/s`** | **`2.516 ms`** | **`9.83 µs`** | **`26.67 GB/s`** | **`10.01 GB/s`** | **`25.86 GB/s`** |
| **Min – Max** | — | `111.04 – 112.37 µs` | `2.33 – 2.36 GB/s` | `2.31 – 2.33 GB/s` | `16.76 – 17.93 GB/s` | `2.477 – 2.581 ms` | `9.68 – 10.08 µs` | **`26.00 – 27.09 GB/s`** | **`9.75 – 10.16 GB/s`** | **`25.47 – 26.23 GB/s`** |

### Break-Even Threshold vs. Full Link Wire-Saturation Threshold (`B10`)
For `INT4_SYM + Preserve RoPE (FP8)` ($r = 262,144 / 98,352 = 2.6654$):
* **Theorem 1 Serial Break-Even ($T_{\text{ztcp}} < T_{\text{raw}}$):** Requires $BW_{\text{fused,hot}} > \frac{r}{r-1} BW_{\text{net}} = 1.6005 \times BW_{\text{net}}$ (or $BW_{\text{fused,wire}} > \frac{1}{r-1} BW_{\text{net}} = 0.6005 \times BW_{\text{net}}$).
  * **100GbE RoCEv2 ($BW_{\text{net}} = 11.2\text{ GB/s}$):** Break-even requires **`17.93 GB/s` hot** (`6.73 GB/s` wire). Our measured M3 Max CPU kernel (**`26.65 ± 0.46 GB/s` hot / `10.00 ± 0.17 GB/s` wire**) **exceeds the 100GbE break-even threshold by 48.6%** (and `K8V4_ASYM` at **`23.76 ± 0.27 GB/s` hot / `10.40 ± 0.12 GB/s` wire** also exceeds its `19.92 GB/s` hot break-even threshold by **19.3%**).
  * **400GbE NDR InfiniBand ($BW_{\text{net}} = 45.0\text{ GB/s}$):** Break-even requires **`72.02 GB/s` hot** (`27.02 GB/s` wire), which exceeds single-socket CPU throughput (`26.65 GB/s` hot) and requires GPU SRAM execution.
* **Full Link Wire-Saturation ($BW_{\text{fused,wire}} \ge BW_{\text{net}}$):** Requires emitting compressed wire bytes at full link goodput ($BW_{\text{fused,hot}} \ge r \cdot BW_{\text{net}}$):
  * **100GbE RoCEv2 ($11.2\text{ GB/s}$ wire):** Full wire saturation requires **`11.20 GB/s` wire** (**`29.85 GB/s` hot**). In the analytical link model ($BW_{\text{fused,wire}} / BW_{\text{net}}$, not a live NIC measurement), the M3 Max CPU kernel alone (`10.00 GB/s` wire / `26.65 GB/s` hot on `INT4_SYM`, and `10.40 GB/s` wire on `K8V4_ASYM`) corresponds to **89.3% (`INT4_SYM`) to 92.9% (`K8V4_ASYM`) of modeled 100GbE goodput** from a single CPU socket.
  * **400GbE NDR InfiniBand ($45.0\text{ GB/s}$ wire):** Full wire saturation requires **`45.00 GB/s` wire** (**`119.94 GB/s` hot**).

---

## Experiment 3: Analytical Serial-Egress 100GbE Prefix Transfer Model (4,096-Token Prefix, 64 Layers @ `11.2 GB/s`)

**Analytical Model Scope & Unit Convention (`B9`, `B10`, `PUB-005`, `DS-P3-05`):** This experiment evaluates an **analytical non-overlapped serial-egress model** ($T_{\text{total}} = T_{\text{compute}} + T_{\text{wire}}$, not a live NIC-to-NIC RDMA measurement) for offloading a 4,096-token cached prefix across 64 layers (`4,096 pages = 1,073,741,824 B = 1,073.74 MB SI = 1,024.00 MiB binary` raw `BF16`). All latency calculations use consistent decimal SI units ($T = \text{Bytes} / (BW_{\text{GB/s}} \times 10^9) \times 1000\text{ ms}$) at an assumed effective 100GbE RoCEv2 goodput of $BW_{\text{net}} = 11.2\text{ GB/s}$ ($11.2 \times 10^9\text{ B/s}$). We explicitly separate **analytical latency using Empirically Measured Apple M3 Max CPU Fused Compute (`4a`, `5a`)** from the **Hypothetical GPU SRAM (`950.0 GB/s`) Analytical Projection (`4b`, `5b`)**.

> **Estimator Disclosure (`DS-P3-05`, `1.26x` Consistency):** For Row `5a` (`ZTCP INT4` with measured M3 Max CPU compute), both the ratio of 5-run mean latencies ($95.87\text{ ms} / 76.28\text{ ms} = 1.2568\times \rightarrow \mathbf{1.26\times}$) and the mean of the 5 per-run 2-decimal rounded ratios in `latest_results.json` (`mean([1.25, 1.27, 1.27, 1.24, 1.26]) = 1.258` $\rightarrow$ **`1.26 ± 0.01x`**) round to **`1.26x`**.

| Analytical Serial-Egress Pipeline Path (Compute + 100GbE Wire @ `11.2 GB/s`) | Wire Volume (SI / Binary) | Compute Time (`ms`) | Analytical Wire Time (`ms`) | Total Serial Latency (`ms`) | Speedup vs. Raw BF16 | Security Posture |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **1. Unencrypted Raw `BF16` Baseline** | `1,073.74 MB` (`1,024.0 MiB`) | `0.00 ms` | `95.87 ms` | **`95.87 ms`** | `1.00x` | **Plaintext (Insecure)** |
| **2. Discrete Host Temp-Copy (`28 GB/s` assumed) + `K8V4` Wire** | `469.96 MB` (`448.2 MiB`) | `38.35 ms` | `41.96 ms` | **`80.31 ms`** | `1.19x` | Encrypted (Host-Bottlenecked) |
| **3. Unencrypted Raw `FP8` (Hot KV Quantized)** | `536.87 MB` (`512.0 MiB`) | `0.00 ms` | `47.93 ms` | **`47.93 ms`** | `2.00x` | **Plaintext (Hot Attention Loss)** |
| **4a. ZTCP `K8V4 + RoPE-FP8 + AEAD` (Measured M3 Max CPU Compute)** | `469.96 MB` (`448.2 MiB`) | `45.19 ± 0.53 ms` | `41.96 ms` | **`87.15 ± 0.53 ms`** | **`1.10 ± 0.01x`** | **Zero-Trust ChaCha20 + SipHash-2-4-128** |
| **5a. ZTCP `INT4 + RoPE-FP8 + AEAD` (Measured M3 Max CPU Compute)** | `402.85 MB` (`384.2 MiB`) | `40.31 ± 0.70 ms` | `35.97 ms` | **`76.28 ± 0.70 ms`** | **`1.26 ± 0.01x`** | **Zero-Trust ChaCha20 + SipHash-2-4-128** |
| **4b. ZTCP `K8V4 + RoPE-FP8 + AEAD` (Hypothetical GPU @ `950 GB/s`)** | `469.96 MB` (`448.2 MiB`) | `1.13 ms` *(proj.)* | `41.96 ms` | **`43.09 ms`** *(proj.)* | **`2.22x`** *(proj.)* | **Zero-Trust AEAD (Analytical Projection)** |
| **5b. ZTCP `INT4 + RoPE-FP8 + AEAD` (Hypothetical GPU @ `950 GB/s`)** | `402.85 MB` (`384.2 MiB`) | `1.13 ms` *(proj.)* | `35.97 ms` | **`37.10 ms`** *(proj.)* | **`2.58x`** *(proj.)* | **Zero-Trust AEAD (Analytical Projection)** |

---

## Experiment 4: Zero-Trust Security, Freshness, Crypto-Shredding & Concurrency Verification (`B1`–`B6`, `CODE-002`, `CODE-003`, `PUB-005`)

The automated verification suite in `bench_cold_page_pipeline.py:149–603` executes 12 test groups. To avoid overclaiming backend coverage (`PUB-005`), the table below explicitly specifies the **Exercised Backend / Code Path** for each test group and labels local in-memory byte/bit mutations as **simulated RDMA tampering**:

| Test ID | Exercised Backend / Code Path | Verified Threat / Invariant Scenario | Verified Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **3.1** | Python Reference (`FusedColdPageCodec`, `BLAKE2b-128`) | Valid `Tenant + Session + Model + Layer + Prefix Hash` (AAD) Egress $\rightarrow$ Ingress | Authenticated round-trip recovery | **PASS** |
| **3.2** | Python Reference (`FusedColdPageCodec`, `BLAKE2b-128`) | **Cross-Tenant KV Contamination** (wrong expected `tenant_id`) | Rejected via 32B AAD `tenant_id_hash` check (`CrossTenantContaminationError`) | **PASS** |
| **3.3** | Python Reference (`FusedColdPageCodec`, `BLAKE2b-128`) | **Simulated RDMA 1-Bit Poisoning** (local in-memory 1-bit flip in ciphertext) | Rejected via 128-bit MAC verification (`KVIntegrityError`) | **PASS** |
| **3.4a/b** | Python Reference (`FusedColdPageCodec`) + `EphemeralSessionKeyRing` | **In-Process Crypto-Shredding & Post-Shred Re-Derivation** (`crypto_shred_session` + `derive_or_get`) | Mutable `bytearray(32)` key buffers zeroed in-place; CSPRNG-salted `gen+1` re-derivation rejects archived blobs (`KVIntegrityError`) | **PASS** |
| **3.5** | Native C++17 Peak (`encode_cold_pages_peak_into` $\rightarrow$ `decode_cold_page_tensor`, `SipHash-2-4-128`) | **Native Peak Round-Trip & Legitimate Cache Re-Read** ($\times 2$ decode of current version) | Lossless authentication (`Attn Output CosSim = 0.996103`) and repeated current-version cache reads accepted | **PASS** |
| **3.6** | Native C++17 Peak (`NativeCppFusedColdPageCodec`, `SipHash-2-4-128`) | **Simulated Scale Tampering on Native Peak Blob** (local 1-bit flip in encrypted `s_rope` region) | Rejected prior to dequantization (`KVIntegrityError`) | **PASS** |
| **3.7** | Native C++17 Peak (`NativeCppFusedColdPageCodec`, `SipHash-2-4-128`) | **Simulated 16B NoPE Block Swap on Native Peak Blob** (local swap of two 16B quantized NoPE blocks) | Rejected via sequential SipHash-2-4-128 MAC (`KVIntegrityError`) | **PASS** |
| **3.8** | Native C++17 Peak (`NativeCppFusedColdPageCodec`, `SipHash-2-4-128`) | **Simulated Paired V-Block Bit-Flips on Native Peak Blob** (matching bit flips across two 16B V blocks) | Do not cancel; rejected via SipHash-2-4-128 (`KVIntegrityError`) | **PASS** |
| **3.9** | Native C++17 Peak (`NativeCppFusedColdPageCodec`, `SipHash-2-4-128`) | **Native Peak AAD Contamination** (`tenant_id`, `model_id`, `layer_id`, `page_token_hash` mismatches) | All 4 mismatch vectors rejected (`CrossTenantContaminationError`) | **PASS** |
| **3.10a/b/c** | Native C++17 Peak + Python `KeyRing.pack_chacha20_nonce` | **Bijective 96-Bit Nonce Helper, Native Peak Duplicate Nonce & Stale Replay** | Distinct packed 96b nonces (`3.10a`); duplicate Peak session nonces rejected (`NonceReuseError`, `3.10b`); stale Peak cold-page versions (`seq < latest`) rejected (`StalePageReplayError`, `3.10c`) | **PASS** |
| **3.11** | Native C++17 Peak (`NativeCppFusedColdPageCodec`) + `EphemeralSessionKeyRing` | **Native Peak Irreversible Post-Shred Re-Derivation** | Archived Peak blob rejected after session shred + re-derive (`KVIntegrityError`) | **PASS** |
| **3.12** | Native C++17 (`Peak` & `Single`) + `EphemeralSessionKeyRing` | **Transactional Session State & Concurrency (`CODE-003`)** | Native Peak same-nonce race (`1 success + 1 NonceReuseError`) + Native Single failed-encode nonce burn and monotonic `max()` watermark checks | **PASS** |

> **Separate Local Regression Suite Scope (`PUB-005-R2`):** The table above reflects the exact tests bundled in `research/benchmarks/bench_cold_page_pipeline.py` (`run_security_verification`). **In-flight egress and ingress shred checks across all four codec paths (`py_single`, `cpp_single`, `cpp_batch`, `cpp_peak`) are verified in `verify_package2.py`, `CODE-003`; overlapping-batch checks are verified in `B6-2`** (separate local regression suite, distinct from the bundled benchmark script).

---

## Implementation Scope, Protocol Contracts & Open Caveats (`B8`, `PUB-001`, `PUB-004`)

1. **Public PoC vs. Hardware TEE / PQC Scope (`B8`, `PUB-005`):** This public repository benchmarks the CPU reference and C++17 ARMv8.5-A FP16 NEON cold-page codec, in-process `EphemeralSessionKeyRing`, zero-copy Python/CPU hook ABIs (`sglang`, `lmcache`, `mooncake`), and analytical RDMA link models. Hardware TEE enclave attestation (NVIDIA H100/B200 Confidential Computing, Intel TDX, AMD SEV-SNP), GPU SM90/SM100 CUDA/Triton kernels, live RoCEv2/InfiniBand NIC transfers, and NIST FIPS 203 (`ML-KEM`) / FIPS 204 (`ML-DSA`) control-plane key exchange are target enterprise/production deployment components and are not claimed as empirically measured in this public benchmark suite.
2. **In-Process Keyring Concurrency, Linearization & Buffer Ownership Contracts (`B5`, `CODE-003`, `PUB-001`):**
   * **Single-Process Lock Scope:** `EphemeralSessionKeyRing` enforces thread-safe nonce reservation, monotonic watermarks, and generation binding in-memory within a single Python process via `threading.RLock` (validated by deterministic multi-threaded regression tests, not unbounded fuzzing or cross-process shared memory).
   * **Pre-Shred Linearization:** If an in-flight egress or ingress operation completes its locked `commit_egress` or `finish_ingress` (final generation check) *before* a concurrent `crypto_shred_session` / `crypto_shred_tenant` call acquires the keyring lock, that operation is linearized prior to the shred and may legitimately return success to its caller after the shred call returns; not every call overlapping a shred in wall-clock time is required to abort.
   * **Post-Revocation Abort & Output Zeroing:** If `crypto_shred_session` acquires the lock *before* `begin_egress_execution`, `commit_egress`, or `finish_ingress`, the operation detects the revoked generation, zeroes any output tensor populated inside the execution `try` block (`out_blobs_u8.zero_()` / `k_out.zero_()`), refuses to commit into any newer generation, and raises `SessionShreddedError`.
   * **Caller Exclusive Ownership & Exception Invalidation of Preallocated Peak Buffers:** The caller of `encode_cold_pages_peak_into` (and `decode_cold_pages_peak_into`) must exclusively own the preallocated output tensor buffer and must only read or publish its contents after the call returns normally without exception. Callers must treat preallocated output buffers as invalid and discard them on *every* raised exception; in particular, failures during `preflight_validate_egress` or input/output buffer-overlap checks (prior to `begin_egress_execution` in `native_fused_codec.py`) occur outside the `try ... except` buffer-zeroing handler and therefore leave previous buffer contents untouched.
   * **User-Space Zeroization Boundary:** While active and staged mutable `bytearray(32)` key buffers are zeroed in-place on shred and per-call mutable key snapshots are wiped in `finally` blocks, immutable Python `bytes` objects, unpacked integer/stack key words (`k0..k3`, `mk0..mk1`) in native C++ stack frames, CPython GC temporaries, and OS swap/core-dumps remain outside any strong hardware memory-erasure guarantee.
3. **Wire Format Version, MAC Backend Non-Interoperability, Binding Defaults & Scale Staging:**
   * **`FORMAT_VERSION = 1`:** Retained as version `1` in the 32B header across this research PoC despite upgrading the stream cipher and MAC to 256-bit RFC 8439 ChaCha20 + SipHash-2-4-128.
   * **Python vs. C++ MAC Backend:** `FusedColdPageCodec` (Python reference) computes 128-bit `BLAKE2b` tags whereas `NativeCppFusedColdPageCodec` (C++17 native) computes 128-bit `SipHash-2-4-128` tags; each codec authenticates and decodes blobs produced by its own backend.
   * **Low-Level C++ Binding Defaults (`k2=0, k3=0`):** The raw pybind11 entrypoints keep default `k2=0, k3=0` parameters for backward signature compatibility, whereas `NativeCppFusedColdPageCodec` always unpacks and passes all 256 bits (`k0, k1, k2, k3`) of the 32-byte session encryption key.
   * **Geometry-Conditional Stack vs. Heap Scale Staging:** `encode_one_cold_page_into_cpu` stages per-group `FP16` scales in a fixed 16 KB stack array (`std::array<at::Half, 8192> stack_scales`) for all geometries with $\le 8,192$ groups per section, and falls back to a `std::vector<at::Half>` heap allocation only if a non-standard geometry exceeds 8,192 groups per section.
