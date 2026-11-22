"""ZipBurger R13: bounded cross-expert redundancy survey on real 20B GGUF.

Tests whether shared expert representations deserve further research.
NO inference, no neuron equivalence, no full checkpoint rewrite, no GPU.
Reads only publicly pinned HTTPS range slices when --real is explicitly set.
"""
from __future__ import annotations

import argparse
import itertools
import json
import zlib

import numpy as np

from .palette3_r08 import MANTISSAS, _codes, _check
from .real_mxfp4_ranges_r04 import REVISION, FILE, TENSORS, sample


def examine(name, expert_tiles, *, provenance="synthetic_fixture"):
    if not 2 <= len(expert_tiles) <= 8:
        raise ValueError("R13 needs 2..8 aligned experts")
    if provenance not in ("synthetic_fixture", "pinned_public_gguf_ranges"):
        raise ValueError("unknown source provenance")
    raw_blocks, weighted = [], []
    shape = None
    for blocks, scales in expert_tiles:
        b, s = _check(blocks, scales)
        if shape is None:
            shape = s.shape
        if s.shape != shape:
            raise ValueError("all expert tiles must have identical shapes")
        raw = np.concatenate((s[..., None], b), axis=2).tobytes()
        raw_blocks.append(raw)
        factors = np.ldexp(np.full(s.shape, 0.5), s.astype(np.int16) - 127)
        dense = (MANTISSAS[_codes(b)].astype(np.float64)
                 * factors[..., None]).reshape(-1)
        weighted.append(dense)
    size = len(raw_blocks[0])
    if any(len(raw) != size for raw in raw_blocks):
        raise ValueError("unaligned raw MXFP4 tile lengths")
    equality, cosine = [], []
    for i, j in itertools.combinations(range(len(raw_blocks)), 2):
        xi, xj = np.frombuffer(raw_blocks[i], np.uint8), np.frombuffer(raw_blocks[j], np.uint8)
        identical = np.all(xi.reshape(-1, 17) == xj.reshape(-1, 17), axis=1)
        equality.append(float(np.mean(identical)))
        a, b = weighted[i], weighted[j]
        norma, normb = np.linalg.norm(a), np.linalg.norm(b)
        cosine.append(float(np.dot(a, b) / (norma * normb))
                      if norma > 0 and normb > 0 else
                      (1.0 if norma == 0 and normb == 0 else 0.0))
    vectors = np.stack(weighted)
    # 2..8 dense rows x ~25k weights: small SVD on rows, never model-scale.
    gram = vectors @ vectors.T
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
    total_energy = float(eigenvalues.sum())
    rank1_energy = float(eigenvalues[0] / total_energy) if total_energy else 1.0
    rank2_energy = (float(eigenvalues[:2].sum() / total_energy)
                    if total_energy else 1.0)
    original = len(raw_blocks) * size
    independent = sum(len(zlib.compress(x, 9)) for x in raw_blocks)
    combined = len(zlib.compress(b"".join(raw_blocks), 9))
    base = np.frombuffer(raw_blocks[0], np.uint8)
    delta_zlib = len(zlib.compress(raw_blocks[0], 9))
    for x in raw_blocks[1:]:
        delta = np.bitwise_xor(np.frombuffer(x, np.uint8), base).tobytes()
        delta_zlib += len(zlib.compress(delta, 9))
    best = min(independent, combined, delta_zlib)
    return {
        "schema": "zipburger-r13-cross-expert-v1",
        "tensor": name,
        "provenance": provenance,
        "expert_count": len(expert_tiles),
        "rows_per_expert": shape[0],
        "groups_per_row": shape[1],
        "source_bytes": original,
        "source_mxfp4_bytes_per_block": 17,
        "exact_identical_17byte_block_fraction_pairwise_mean": round(
            float(np.mean(equality)), 8),
        "decoded_weight_cosine_pairwise_mean": round(
            float(np.mean(cosine)), 8),
        "shared_expert_rank1_energy_fraction": round(rank1_energy, 8),
        "shared_expert_rank2_energy_fraction": round(rank2_energy, 8),
        "lossless_zlib9_separate_experts_bytes": independent,
        "lossless_zlib9_concat_experts_bytes": combined,
        "lossless_xor_to_expert0_zlib9_bytes": delta_zlib,
        "best_tested_lossless_bytes": best,
        "best_tested_lossless_saving_pct": round(100 * (1 - best / original), 6),
        "candidate_is_an_inference_codec": False,
        "model_agentic_equivalence_proven": False,
        "whole_model_size_reduction_proven": False,
    }


def slices(name, *, experts=4, synthetic=True):
    if not 2 <= experts <= 8:
        raise ValueError("bounded research needs 2..8 experts")
    if synthetic:
        rng = np.random.default_rng(1300 + sum(ord(c) for c in name))
        return [(rng.integers(0, 256, (48, 16, 16), np.uint8),
                 rng.integers(120, 132, (48, 16), np.uint8))
                for _ in range(experts)]
    return [sample(name, row_start=2880 * i, rows=48, groups=16)[:2]
            for i in range(experts)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="fetch small immutable public GPT-OSS-20B GGUF byte ranges")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--tensor", action="append",
                        help="choose name(s) from pinned up/gate/down tensors (max 9)")
    args = parser.parse_args(argv)
    names = args.tensor or sorted(TENSORS)
    if not 1 <= len(names) <= 9 or len(set(names)) != len(names) or any(
            name not in TENSORS for name in names):
        parser.error("invalid bounded pinned tensor names")
    if not 2 <= args.experts <= 8:
        parser.error("--experts must be 2..8")
    output = [examine(name, slices(name, experts=args.experts,
                                   synthetic=not args.real),
                      provenance="pinned_public_gguf_ranges" if args.real
                      else "synthetic_fixture")
              for name in names]
    report = {
        "schema": "zipburger-r13-cross-expert-results-v1",
        "source_gguf": FILE, "pinned_revision": REVISION,
        "real_public_weights": bool(args.real),
        "real_teacher_activations": False,
        "real_model_inference": False,
        "cross_expert_storage_hypothesis_only": True,
        "results": output,
    }
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
