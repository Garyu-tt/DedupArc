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


def benchmark_one(name, blocks, scales, *, row_start=0, real=False):
    from .mxfp4_palette_r04 import pack as lossless_pack
    seed = 2108 + row_start + sum(ord(ch) for ch in name)
    rng = np.random.default_rng(seed)
    width = scales.shape[1] * 32
    calibration = rng.standard_normal((32, width))
    validation = rng.standard_normal((16, width))
    started = time.perf_counter()
    selected, result = evaluate_expert(
        name, blocks, scales, calibration, validation,
        max_relative_l2=0.04, min_cosine=0.999, fallback_factory=lossless_pack)
    result.update({
        "sample_row_start": row_start,
        "sample_origin": "pinned_real_20b_GGUF" if real else "synthetic_MXFP4",
        "stored_bytes_after_quality_gate": selected.frame_bytes,
        "stored_saving_pct_after_quality_gate": round(
            100 * (1 - selected.frame_bytes / result["raw_mxfp4_bytes"]), 5),
        "benchmark_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "real_model_activations": False,
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
    args = parser.parse_args(argv)
    items = [benchmark_one(name, b, s, row_start=row_start, real=args.real)
             for name, row_start, b, s in samples(args.real)]
    result = {
        "experiment": "ZipBurger R08",
        "real_20b_weight_fragments": args.real,
        "real_20b_activations": False,
        "whole_model_compression_proven": False,
        "agentic_equivalence_proven": False,
        "original_bytes": sum(x["raw_mxfp4_bytes"] for x in items),
        "candidate_3bit_bytes": sum(x["three_bit_bytes"] for x in items),
        "quality_gated_bytes": sum(x["stored_bytes_after_quality_gate"] for x in items),
        "quality_passes": sum(x["quality_gate_passed"] for x in items),
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
