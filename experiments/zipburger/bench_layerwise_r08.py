"""R08: bounded GPT-OSS-20B expert-tile test on frozen public GGUF ranges.

No full model download, no agent inference, no model writes. Validation
activations are synthetic Gaussian placeholders until genuine captures exist.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .palette3_r08 import evaluate_expert


def benchmark_one(name, blocks, scales, *, row_start=0, real=False,
                  calibration=None, validation=None):
    from .mxfp4_palette_r04 import pack as lossless_pack
    seed = 2108 + row_start + sum(ord(ch) for ch in name)
    rng = np.random.default_rng(seed)
    width = scales.shape[1] * 32
    real_teacher_inputs = calibration is not None and validation is not None
    if (calibration is None) != (validation is None):
        raise ValueError("both calibration and held-out vectors are required")
    if not real_teacher_inputs:
        calibration = rng.standard_normal((32, width))
        validation = rng.standard_normal((16, width))
    started = time.perf_counter()
    selected, result = evaluate_expert(
        name, blocks, scales, calibration, validation,
        max_relative_l2=0.04, min_cosine=0.999, fallback_factory=lossless_pack)
    from .residual_adapter_r08 import measure as measure_residual
    # Compare a tiny learned encoder/decoder correction without assuming its
    # per-tile gains transfer to the 20B model or real prompt activations.
    _, r1 = measure_residual(blocks, scales, calibration, validation, rank=1)
    _, r2 = measure_residual(blocks, scales, calibration, validation, rank=2)
    result["rank1_fp16_adapter"] = r1
    result["rank2_fp16_adapter"] = r2
    result.update({
        "sample_row_start": row_start,
        "sample_origin": "pinned_real_20b_GGUF" if real else "synthetic_MXFP4",
        "stored_bytes_after_quality_gate": selected.frame_bytes,
        "stored_saving_pct_after_quality_gate": round(
            100 * (1 - selected.frame_bytes / result["raw_mxfp4_bytes"]), 5),
        "benchmark_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "real_model_activations_operator_attested": bool(real_teacher_inputs),
        "real_model_activations_independently_verified": False,
        "model_inference_or_agentic_equivalence": False,
    })
    return result


def samples(real):
    if real:
        # Existing R04 reader pins revision, file size and HTTP 206 range;
        # it enforces a strict ~120 KB per request and trusted HF redirects.
        from .real_mxfp4_ranges_r04 import TENSORS, sample
        for name in sorted(TENSORS):
            for row_start in (0, 4096):
                blocks, scales, _, _ = sample(
                    name, row_start=row_start, rows=48, groups=16)
                yield name, row_start, blocks, scales
    else:
        rng = np.random.default_rng(443)
        for layer in (0, 1, 2):
            yield (f"blk.{layer}.ffn_up_exps.weight", 0,
                   rng.integers(0, 256, (48, 16, 16), np.uint8),
                   rng.integers(118, 133, (48, 16), np.uint8))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Read bounded public pinned 20B GGUF ranges")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--teacher-calibration", type=Path)
    parser.add_argument("--teacher-validation", type=Path)
    args = parser.parse_args(argv)
    if (args.teacher_calibration is None) != (args.teacher_validation is None):
        parser.error("provide both calibration and validation archives")
    if args.teacher_calibration is not None and not args.real:
        parser.error("--real is required for real teacher activation probes")
    cal = val = None
    if args.teacher_calibration is not None:
        from .expert_trace_input_r09 import load_pair, matched_vectors
        cal, val = load_pair(args.teacher_calibration, args.teacher_validation)
    items, skipped = [], []
    for name, row_start, b, s in samples(args.real):
        pair = None
        if cal is not None:
            try:
                pair = matched_vectors(cal, val, tensor=name, row_start=row_start)
            except ValueError:
                pair = None
            if pair is None:
                skipped.append({"tensor": name, "row_start": row_start,
                                "reason": "no supported 8+8 routed up/gate input samples"})
                continue
        items.append(benchmark_one(name, b, s, row_start=row_start,
                                   real=args.real,
                                   calibration=None if pair is None else pair[0],
                                   validation=None if pair is None else pair[1]))
    result = {
        "experiment": "ZipBurger R08",
        "real_20b_weight_fragments": args.real,
        "real_20b_activations_operator_attested": bool(cal is not None),
        "real_20b_activations_independently_verified": False,
        "skipped_samples": skipped,
        "whole_model_compression_proven": False,
        "agentic_equivalence_proven": False,
        "original_bytes": sum(x["raw_mxfp4_bytes"] for x in items),
        "candidate_3bit_bytes": sum(x["three_bit_bytes"] for x in items),
        "quality_gated_bytes": sum(x["stored_bytes_after_quality_gate"] for x in items),
        "quality_passes": sum(x["quality_gate_passed"] for x in items),
        "rank1_adapter_passes": sum(x["rank1_fp16_adapter"]["accepted_after_quality_and_size_gate"] for x in items),
        "rank2_adapter_passes": sum(x["rank2_fp16_adapter"]["accepted_after_quality_and_size_gate"] for x in items),
        "rank1_adapter_aggregate_bytes": sum(x["rank1_fp16_adapter"]["corrected_bytes"] for x in items),
        "rank2_adapter_aggregate_bytes": sum(x["rank2_fp16_adapter"]["corrected_bytes"] for x in items),
        "samples": len(items),
        "results": items,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return result


if __name__ == "__main__":
    main()
