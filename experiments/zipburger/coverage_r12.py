"""ZipBurger R12: conservative teacher routing/activation coverage audit.

Quantifies whether genuine, disjoint R10/R11 teacher captures actually
touch enough MoE experts to evaluate lossily compressed R08 tiles.
NO full model inference, optimization, or model equivalence is claimed.
All counts are per layer and expert, not estimated from weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .expert_trace_input_r09 import load_pair, MIN_EXPERT_SAMPLES
from .layer_trace_r09 import load as load_trace, LAYERS, LOCAL_EXPERTS

MAX_UNIQUE_PROMPTS = 64
ENTROPY_EPS = 1e-300


def _routing_stats(routes):
    # Official GPT-OSS-20B has 4 chosen experts per token from 32 per layer.
    if routes.dtype != np.int16 or routes.ndim != 2 or routes.shape[1] != 4:
        raise ValueError("invalid top-4 router trace")
    n = routes.shape[0]
    counts = np.bincount(routes.ravel().astype(np.int64),
                         minlength=LOCAL_EXPERTS)
    if counts.size != LOCAL_EXPERTS or counts.sum() != 4 * n:
        raise ValueError("unexpected routing cardinality")
    probabilities = counts / max(float(counts.sum()), 1)
    positive = probabilities[probabilities > 0]
    entropy_bits = float(-(positive * np.log2(positive)).sum())
    return counts, round(entropy_bits, 6)


def report(calibration_trace, validation_trace, calibration_archive,
           validation_archive, *, min_samples=MIN_EXPERT_SAMPLES):
    if not 1 <= min_samples <= 256:
        raise ValueError("invalid per-expert minimum sample threshold")
    cal, val = load_pair(calibration_archive, validation_archive)
    teacher = load_trace(calibration_trace)
    heldout = load_trace(validation_trace)
    if teacher.source != "instrumented_teacher" or heldout.source != "instrumented_teacher":
        raise ValueError("routing coverage requires teacher capture provenance")
    if teacher.checkpoint_digest != heldout.checkpoint_digest:
        raise ValueError("checkpoint identity does not match")
    if teacher.prompt_digest == heldout.prompt_digest:
        raise ValueError("duplicate prompt corpus")
    layers = []
    total_covered = 0
    total_routed = 0
    total_observed = 0
    complete = 0
    incomplete = []
    for idx in range(LAYERS):
        cal_counts, cal_entropy = _routing_stats(teacher.routes[idx])
        val_counts, val_entropy = _routing_stats(heldout.routes[idx])
        row = {
            "layer": idx,
            "calibration_tokens": int(teacher.routes[idx].shape[0]),
            "holdout_tokens": int(heldout.routes[idx].shape[0]),
            "calibration_routing_entropy_bits": cal_entropy,
            "holdout_routing_entropy_bits": val_entropy,
            "seen_experts_calibration": int(np.count_nonzero(cal_counts)),
            "seen_experts_holdout": int(np.count_nonzero(val_counts)),
            "experts_meeting_minimum": [],
            "experts_missing_or_underrepresented": [],
        }
        for expert in range(LOCAL_EXPERTS):
            key = f"layer_{idx:02d}_expert_{expert:02d}"
            a, b = cal.get(key), val.get(key)
            n_a, n_b = 0 if a is None else len(a), 0 if b is None else len(b)
            total_routed += int(cal_counts[expert]) + int(val_counts[expert])
            total_observed += n_a + n_b
            # Each router token dispatches to the expert once; one observed
            # activation row must be available for each routed occurrence
            # unless archive intentionally caps to <=256 examples.
            if n_a > min(int(cal_counts[expert]), 256) or n_b > min(int(val_counts[expert]), 256):
                raise ValueError("unexplained extra expert activation observations")
            if n_a >= min_samples and n_b >= min_samples:
                row["experts_meeting_minimum"].append(expert)
                total_covered += 1
            else:
                row["experts_missing_or_underrepresented"].append({
                    "expert": expert, "calibration_observations": n_a,
                    "heldout_observations": n_b,
                    "calibration_router_hits": int(cal_counts[expert]),
                    "heldout_router_hits": int(val_counts[expert]),
                })
        if len(row["experts_meeting_minimum"]) == LOCAL_EXPERTS:
            complete += 1
        else:
            incomplete.append(idx)
        layers.append(row)
    result = {
        "schema": "zipburger-r12-expert-coverage-v1",
        "teacher_layers": LAYERS,
        "experts_per_layer": LOCAL_EXPERTS,
        "routed_experts_per_token": 4,
        "min_calibration_and_holdout_observations_per_expert": min_samples,
        "eligible_layer_expert_pairs": total_covered,
        "total_layer_expert_pairs": LAYERS * LOCAL_EXPERTS,
        "eligible_percent": round(100 * total_covered / (LAYERS * LOCAL_EXPERTS), 4),
        "fully_covered_layers": complete,
        "layers_with_missing_experts": incomplete,
        "total_routed_expert_hits": total_routed,
        "total_stored_expert_observations": total_observed,
        "routing_provenance": "operator_declared_instrumented_teacher",
        "real_full_model_inference_verified": False,
        "model_agentic_equivalence_verified": False,
        "lossy_compression_authorized": False,
        "layers": layers,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-trace", type=Path, required=True)
    parser.add_argument("--val-trace", type=Path, required=True)
    parser.add_argument("--cal-experts", type=Path, required=True)
    parser.add_argument("--val-experts", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=MIN_EXPERT_SAMPLES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = report(args.cal_trace, args.val_trace,
                    args.cal_experts, args.val_experts,
                    min_samples=args.min_samples)
    formatted = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(formatted, encoding="utf-8")
    print(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
