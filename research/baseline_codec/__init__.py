# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hakan Damar and Theseus Guard, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Developed by Theseus Guard Research (https://www.theseusguard.com)
# Repository: https://github.com/Theseus-Guard/ZTCP
# Contact: TG-Team <contact@theseusguard.com> | https://www.theseusguard.com/contact
"""Zero-Trust Cold Pages (ZTCP) Research Baseline Codec Package.

Developed by Theseus Guard, Inc. to extend the Zero-Trust AI Data Plane (ZT-ADP)
and DREP (Detect-Route-Enforce-Prove) governance lifecycle to distributed RDMA
KV-cache pools (SGLang HiCache, vLLM/LMCache, Mooncake).

Provides the public specification and reference implementation for single-pass
Asymmetric Cold-Page KV Quantization + Authenticated Encryption (AEAD) with
Associated Data (AAD) binding.
"""

from .cold_page_spec import (
    HEADER_BYTES,
    TAG_BYTES,
    ColdPageAAD,
    ColdPageGeometry,
    QuantPreset,
    RoPEMode,
)
from .fused_quant_crypto import (
    CrossTenantContaminationError,
    EphemeralSessionKeyRing,
    FusedColdPageCodec,
    KVIntegrityError,
    NonceReuseError,
    SessionShreddedError,
    StalePageReplayError,
)
from .native_fused_codec import NativeCppFusedColdPageCodec

__all__ = [
    "HEADER_BYTES",
    "TAG_BYTES",
    "ColdPageAAD",
    "ColdPageGeometry",
    "QuantPreset",
    "RoPEMode",
    "CrossTenantContaminationError",
    "EphemeralSessionKeyRing",
    "FusedColdPageCodec",
    "KVIntegrityError",
    "NonceReuseError",
    "SessionShreddedError",
    "StalePageReplayError",
    "NativeCppFusedColdPageCodec",
]
