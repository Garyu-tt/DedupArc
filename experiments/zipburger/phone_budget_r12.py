"""R12 conservative whole-GGUF memory feasibility bound.

Uses the pinned real GPT-OSS-20B GGUF file size, not a hypothetical count
of 20 billion independent 4-bit weights. This is an OPTIMISTIC LOWER bound
on disk/model storage, not measured CPU RSS, phone runtime or agent parity.
In the impossible all-bytes-are-MXFP4-experts scenario, a 17->13 byte/block
code cannot save more than 4/17 of the original file. In actual GPT-OSS the
protected attention/router/embeddings and headers reduce savings further.
"""
from __future__ import annotations

import argparse
import json

from .real_mxfp4_ranges_r04 import EXPECTED_FILE_BYTES, FILE, REVISION

MXFP4_SOURCE_BYTES = 17
R08_CANDIDATE_BYTES = 13


def bound(*, model_bytes=EXPECTED_FILE_BYTES, target_bytes=8_000_000_000,
          expert_fraction=1.0, extra_runtime_bytes=0):
    if (type(model_bytes) is not int or model_bytes < 1 or
            type(target_bytes) is not int or target_bytes < 1 or
            type(extra_runtime_bytes) is not int or extra_runtime_bytes < 0 or
            not isinstance(expert_fraction, (float, int)) or
            not 0 <= expert_fraction <= 1):
        raise ValueError("invalid model/target/fraction/overhead")
    # Optimistic bound: assumes every expert byte qualifies, no per-tile
    # header or correction cost, and lossy quality is magically adequate.
    upper_saving = int(model_bytes * expert_fraction *
                       (MXFP4_SOURCE_BYTES - R08_CANDIDATE_BYTES) /
                       MXFP4_SOURCE_BYTES)
    optimistic_min = model_bytes - upper_saving
    required_saving = max(0, model_bytes + extra_runtime_bytes - target_bytes)
    return {
        "pinned_gguf_file": FILE,
        "pinned_revision": REVISION,
        "original_model_bytes": model_bytes,
        "target_total_bytes": target_bytes,
        "assumed_compressible_expert_fraction_of_original_file": expert_fraction,
        "assumed_extra_runtime_bytes": extra_runtime_bytes,
        "hypothetical_max_R08_3bit_saving_bytes": upper_saving,
        "hypothetical_min_model_bytes": optimistic_min,
        "hypothetical_min_total_bytes": optimistic_min + extra_runtime_bytes,
        "required_reduction_bytes": required_saving,
        "required_reduction_pct_of_original_model": round(100 * required_saving / model_bytes, 5),
        "optimistic_absolute_max_reduction_pct": round(
            100 * upper_saving / model_bytes, 5),
        "r08_3bit_physically_can_fit_under_assumptions": bool(
            optimistic_min + extra_runtime_bytes <= target_bytes),
        "lossy_R08_validation_passes_on_real_20b_fragments": 0,
        "lossy_R08_validation_samples": 18,
        "no_real_model_or_phone_runtime_measured": True,
        "protected_non_expert_weights_unmodified": True,
        "kv_cache_and_runtime_will_add_overhead": True,
        "agentic_equivalence_proven": False,
        "bound_type": "optimistic_necessary_condition_not_sufficient",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-fraction", type=float, default=1.0)
    ap.add_argument("--extra-runtime-bytes", type=int, default=0)
    ap.add_argument("--target-gb", type=float, default=8.0,
                    help="decimal gigabytes, 1 GB = 1e9 bytes")
    args = ap.parse_args(argv)
    if not 0 < args.target_gb <= 64:
        ap.error("target-gb must be in (0,64]")
    result = bound(target_bytes=int(args.target_gb * 1_000_000_000),
                   expert_fraction=args.expert_fraction,
                   extra_runtime_bytes=args.extra_runtime_bytes)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
