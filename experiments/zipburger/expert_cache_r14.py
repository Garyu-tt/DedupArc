"""ZipBurger R14: offline, route-driven expert paging feasibility simulator.

This is NOT running a GGUF model with paging. It simulates a basic LRU of
whole MoE experts for one-token-at-a-time decode using recorded GPT-OSS-20B
top4 routing, or clearly labeled synthetic routes. Assumes three
2880x2880 MXFP4 projections per expert and 24 layers x 32 experts.
No compute energy, KV cache, OS page faults or phone battery modeled.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .gguf_capture_r10 import load, read_complete_batch
from .real_mxfp4_ranges_r04 import EXPECTED_FILE_BYTES

N_LAYERS = 24
N_EXPERTS = 32
TOPK = 4
DIMS = 2880
MXFP4_BYTES_PER_32 = 17
EXPERT_BYTES = 3 * (DIMS * DIMS // 32) * MXFP4_BYTES_PER_32
TOTAL_EXPERT_BYTES = N_LAYERS * N_EXPERTS * EXPERT_BYTES
ASSUMED_NON_EXPERT_FILE_BYTES = EXPECTED_FILE_BYTES - TOTAL_EXPERT_BYTES


def synth_routes(*, tokens=64, seed=1414):
    if not 1 <= tokens <= 512:
        raise ValueError("simulation token bounds 1..512")
    rng = np.random.default_rng(seed)
    # Four unique expert IDs per layer and token.
    return np.stack([
        np.stack([rng.choice(N_EXPERTS, 4, replace=False)
                  for _ in range(N_LAYERS)])
        for _ in range(tokens)
    ]).astype(np.int16)


def native_routes(captures):
    if not isinstance(captures, (tuple, list)) or not 1 <= len(captures) <= 64:
        raise ValueError("1..64 private native capture paths required")
    all_routes = []
    for p in captures:
        c = load(p)
        if len(c.routes) != N_LAYERS:
            raise ValueError("incomplete 24-layer route capture")
        all_routes.append(np.stack(c.routes, axis=1))
    return np.concatenate(all_routes, axis=0)


def simulate(routes, *, budget_bytes, expert_bytes=EXPERT_BYTES):
    a = np.asarray(routes)
    if (a.dtype != np.int16 or a.ndim != 3 or a.shape[1:] !=
            (N_LAYERS, TOPK) or not 1 <= a.shape[0] <= 512
            or np.any(a < 0) or np.any(a >= N_EXPERTS)
            or np.any(np.diff(np.sort(a, axis=2), axis=2) == 0)):
        raise ValueError("invalid top4 routing events")
    if (type(budget_bytes) is not int or budget_bytes < 0 or
            type(expert_bytes) is not int or expert_bytes < 1):
        raise ValueError("invalid RAM/physical expert-byte budget")

    slots = budget_bytes // expert_bytes
    if slots < 1:
        return {
            "simulation_possible_under_budget": False,
            "reason": "less than one whole expert fits in assumed memory budget",
            "tokens": int(a.shape[0]), "expert_cache_slots": 0,
            "actual_model_inference_measured": False,
        }
    cache = OrderedDict()
    hits = misses = evictions = 0
    per_token_misses = []
    for t in range(a.shape[0]):
        current_misses = 0
        for layer in range(N_LAYERS):
            for expert in a[t, layer]:
                key = (layer, int(expert))
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                else:
                    misses += 1
                    current_misses += 1
                    if len(cache) == slots:
                        cache.popitem(last=False)
                        evictions += 1
                    cache[key] = None
        per_token_misses.append(current_misses)
    total_accesses = hits + misses
    return {
        "schema": "zipburger-r14-expert-lru-feasibility-v1",
        "simulation_possible_under_budget": True,
        "tokens": int(a.shape[0]),
        "expert_bytes_assumed": expert_bytes,
        "expert_cache_budget_bytes": budget_bytes,
        "expert_cache_slots": slots,
        "unique_possible_experts": N_LAYERS * N_EXPERTS,
        "expert_accesses": total_accesses,
        "cache_hits": hits, "cache_misses": misses,
        "cache_hit_fraction": round(hits / total_accesses, 6),
        "cache_evictions": evictions,
        "peak_resident_expert_bytes": len(cache) * expert_bytes,
        "raw_expert_bytes_read_from_storage": misses * expert_bytes,
        "mean_raw_expert_read_bytes_per_token": round(
            misses * expert_bytes / a.shape[0]),
        "max_raw_expert_read_bytes_for_one_token": max(per_token_misses) * expert_bytes,
        "each_access_equals_one_full_expert_assumption": True,
        "decode_order_assumption_not_full_prefill_graph_order": True,
        "no_cache_compression_or_prefetch_modeled": True,
        "actual_model_inference_measured": False,
        "device_energy_or_latency_measured": False,
        "agentic_equivalence_proven": False,
    }


def evaluate(target_bytes=8_000_000_000, *, routes, reserve_extra_bytes=0):
    if (type(target_bytes) is not int or target_bytes < 0 or
            type(reserve_extra_bytes) is not int or reserve_extra_bytes < 0):
        raise ValueError("invalid target RAM/reserve")
    # The "nonexpert" portion includes model file bytes outside the 24x32
    # equal-size expert projections. The *actual* runtime adds OS and KV
    # headroom; this is not RSS independently measured.
    available = max(0, target_bytes - ASSUMED_NON_EXPERT_FILE_BYTES -
                    reserve_extra_bytes)
    result = simulate(routes, budget_bytes=available)
    result.update({
        "target_memory_bytes": target_bytes,
        "modeled_nonexpert_file_bytes": ASSUMED_NON_EXPERT_FILE_BYTES,
        "user_supplied_other_runtime_reserve_bytes": reserve_extra_bytes,
        "modeled_expert_cache_available_bytes": available,
        "original_pinned_GGUF_file_bytes": EXPECTED_FILE_BYTES,
        "only_a_necessary_memory_feasibility_survey": True,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path,
                        help="private completed R11 capture directory, else synthetic routes")
    parser.add_argument("--synthetic-tokens", type=int, default=64)
    parser.add_argument("--target-gb", type=float, default=8.0,
                        help="decimal GB, not GiB")
    parser.add_argument("--reserve-runtime-gb", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.target_gb <= 64 or not 0 <= args.reserve_runtime_gb <= 32:
        parser.error("invalid RAM budget")
    if args.capture_dir is None:
        routes = synth_routes(tokens=args.synthetic_tokens)
        provenance = "synthetic_uniform_routing"
    else:
        routes = native_routes(read_complete_batch(args.capture_dir))
        provenance = "private_genuine_format_callback_traces_operator_attested"
    result = evaluate(int(args.target_gb * 1e9), routes=routes,
                      reserve_extra_bytes=int(args.reserve_runtime_gb * 1e9))
    result["source"] = provenance
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
