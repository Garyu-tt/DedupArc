"""ZipBurger R10: strict native llama.cpp GPT-OSS GGUF callback trace importer.

The corresponding C++ probe is local, CPU-only, opt-in and bounded.
R10's binary stores original *token IDs* (possibly reversible to text);
keep every output private. Only the original GGUF model inference is
measured, not ZipBurger candidate equivalence or agentic success.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from .layer_trace_r09 import LAYERS, Trace, save

MAGIC = b"ZBR10BIN"
HEADER = struct.Struct("<8sIIIII")
TOPK = 4
EXPERTS = 32
MAX_TOKENS = 8
MAX_HIDDEN = 8192
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_CAPTURES = 64
PINNED_LLAMA_CPP = "f505c30b92b5579e00532c9e75baa3aa74b70432"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Capture:
    hidden: tuple[np.ndarray, ...]
    expert_inputs: tuple[np.ndarray, ...]
    routes: tuple[np.ndarray, ...]
    router_logits: tuple[np.ndarray, ...]
    router_weights: tuple[np.ndarray, ...]
    predictions: np.ndarray
    token_ids: np.ndarray
    capture_sha256: str

    @property
    def tokens(self):
        return int(self.predictions.size)

    @property
    def width(self):
        return int(self.hidden[0].shape[1])

    @property
    def token_sequence_sha256(self):
        return hashlib.sha256(self.token_ids.astype("<i4").tobytes()).hexdigest()


def _read_array(blob, offset, count, dtype, shape):
    datatype = np.dtype(dtype)
    length = count * datatype.itemsize
    if offset + length > len(blob):
        raise ValueError("truncated native GGUF trace")
    array = np.frombuffer(blob, dtype=datatype, count=count, offset=offset).copy()
    return array.reshape(shape), offset + length


def load(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size < HEADER.size or p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing/oversized native R10 capture")
    raw = p.read_bytes()
    magic, n, width, layers, topk, experts = HEADER.unpack_from(raw)
    if (magic != MAGIC or not 1 <= n <= MAX_TOKENS or
            not 1 <= width <= MAX_HIDDEN or layers != LAYERS or
            topk != TOPK or experts != EXPERTS):
        raise ValueError("unsupported or malformed R10 capture header")
    per_layer = (2 * n * width + n * TOPK + n * EXPERTS + n * TOPK) * 4
    expected = HEADER.size + layers * per_layer + 2 * n * 4
    if len(raw) != expected:
        raise ValueError("R10 capture size does not match declared dimensions")
    pos = HEADER.size
    hidden, expert_inputs, routes, logits, weights = [], [], [], [], []
    for i in range(LAYERS):
        h, pos = _read_array(raw, pos, n * width, "<f4", (n, width))
        e, pos = _read_array(raw, pos, n * width, "<f4", (n, width))
        r, pos = _read_array(raw, pos, n * TOPK, "<i4", (n, TOPK))
        l, pos = _read_array(raw, pos, n * EXPERTS, "<f4", (n, EXPERTS))
        w, pos = _read_array(raw, pos, n * TOPK, "<f4", (n, TOPK))
        if (not np.all(np.isfinite(h)) or not np.all(np.isfinite(e)) or
                not np.all(np.isfinite(l)) or not np.all(np.isfinite(w))):
            raise ValueError(f"nonfinite GGUF activation at layer {i}")
        if (np.any(r < 0) or np.any(r >= EXPERTS) or
                np.any(np.diff(np.sort(r, axis=-1), axis=-1) == 0)):
            raise ValueError(f"invalid MoE top4 routing at layer {i}")
        if not np.allclose(w.sum(axis=1), 1.0, rtol=1e-3, atol=1e-3):
            raise ValueError(f"router weights not normalized at layer {i}")
        if np.any(w < 0) or np.any(w > 1):
            raise ValueError(f"router weights outside probabilities at layer {i}")
        hidden.append(h)
        expert_inputs.append(e)
        routes.append(r)
        logits.append(l)
        weights.append(w)
    pred, pos = _read_array(raw, pos, n, "<i4", (n,))
    tok, pos = _read_array(raw, pos, n, "<i4", (n,))
    if pos != len(raw) or np.any(pred < 0) or np.any(tok < 0):
        raise ValueError("bad token ID or output prediction in GGUF capture")
    return Capture(tuple(hidden), tuple(expert_inputs), tuple(routes),
                   tuple(logits), tuple(weights), pred, tok,
                   hashlib.sha256(raw).hexdigest())


def aggregate(paths, *, checkpoint_sha256, source):
    if (not isinstance(paths, list) or not 1 <= len(paths) <= MAX_CAPTURES or
            not isinstance(checkpoint_sha256, str) or
            not _SHA256.fullmatch(checkpoint_sha256) or
            source not in ("instrumented_teacher", "instrumented_candidate")):
        raise ValueError("invalid trace aggregation metadata")
    captures = [load(p) for p in paths]
    token_hashes = [c.token_sequence_sha256 for c in captures]
    if len(set(token_hashes)) != len(token_hashes):
        raise ValueError("duplicate tokenized prompt; train/holdout separation required")
    width = captures[0].width
    if any(c.width != width for c in captures):
        raise ValueError("inconsistent model hidden widths")
    n = sum(c.tokens for c in captures)
    if not 1 <= n <= 512:
        raise ValueError("total trace exceeds 512 token limit")
    corpus_digest = hashlib.sha256(
        json.dumps(token_hashes, separators=(",", ":")).encode("ascii")).hexdigest()
    sample_ids = np.arange(n, dtype=np.int64)
    trace = Trace(
        sample_ids,
        tuple(np.concatenate([c.hidden[i] for c in captures], axis=0)
              .astype(np.float32) for i in range(LAYERS)),
        tuple(np.concatenate([c.routes[i] for c in captures], axis=0)
              .astype(np.int16) for i in range(LAYERS)),
        np.concatenate([c.predictions for c in captures]).astype(np.int64),
        corpus_digest, checkpoint_sha256, source,
    )
    extra = {}
    experts = {}
    for layer in range(LAYERS):
        extra[f"layer_{layer:02d}_router_logits"] = np.concatenate(
            [c.router_logits[layer] for c in captures], axis=0).astype(np.float32)
        extra[f"layer_{layer:02d}_router_weights"] = np.concatenate(
            [c.router_weights[layer] for c in captures], axis=0).astype(np.float32)
        inputs = np.concatenate([c.expert_inputs[layer] for c in captures], axis=0)
        routes = trace.routes[layer]
        for expert_id in range(EXPERTS):
            chosen = np.any(routes == expert_id, axis=1)
            if np.any(chosen):
                experts[f"layer_{layer:02d}_expert_{expert_id:02d}"] = (
                    inputs[chosen, :512].astype(np.float32)[:256])
    return trace, extra, experts, token_hashes, [c.capture_sha256 for c in captures]


def save_bundle(paths, *, trace_path, checkpoint_sha256, source):
    trace, router, experts, tokens, source_shas = aggregate(
        paths, checkpoint_sha256=checkpoint_sha256, source=source)
    output = Path(trace_path)
    if output.suffix != ".npz":
        raise ValueError("R10 output must be .npz")
    if not 512 <= trace.layers[0].shape[1]:
        raise ValueError("insufficient 512-dim expert input width")
    manifest = save(trace, output)
    router_path = output.with_name(output.stem + "_router_r10.npz")
    np.savez_compressed(router_path, **router)
    extra_path = output.with_name(output.stem + "_expert_upgate.npz")
    if experts:
        np.savez_compressed(extra_path, **experts)
        extra_meta = {
            "schema": "zipburger-r09-expert-inputs-v1",
            "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
            "prompt_digest": manifest["prompt_digest"],
            "checkpoint_digest": manifest["checkpoint_digest"],
            "source": source,
            "input_width": 512, "projections": ["up", "gate"],
            "contains_down_projection_inputs": False,
            "token_sequence_sha256": tokens,
            "provenance": "gguf_llama_cpp_cb_eval_operator_attested",
        }
        extra_path.with_suffix(".json").write_text(
            json.dumps(extra_meta, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
    provenance = {
        "schema": "zipburger-r10-gguf-activation-bundle-v1",
        "native_binary_sha256": source_shas,
        "token_sequence_sha256": tokens,
        "llama_cpp_reference_commit": PINNED_LLAMA_CPP,
        "runtime_commit_independently_verified": False,
        "checkpoint_sha256_operator_supplied": checkpoint_sha256,
        "trace_sha256": manifest["sha256"],
        "router_archive_sha256": hashlib.sha256(router_path.read_bytes()).hexdigest(),
        "expert_inputs": len(experts),
        "has_full_down_projection_inputs": False,
        "full_20b_equivalence_tested": False,
        "agentic_equivalence_tested": False,
    }
    output.with_name(output.stem + "_provenance_r10.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return provenance


def read_complete_batch(directory):
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R11 batch must be a real directory")
    marker = root / "complete.marker"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("R11 incomplete capture batch: no completion marker")
    contents = marker.read_text(encoding="ascii")
    match = re.fullmatch(r"ZBR11 COMPLETE ([1-9][0-9]?)\\n", contents)
    if match is None:
        raise ValueError("invalid R11 batch completion marker")
    files = sorted(root.glob("*.zbr10"))
    n = int(match.group(1))
    if (n != len(files) or not 1 <= n <= MAX_CAPTURES or
            any(p.is_symlink() or not p.is_file() for p in files)
            or [p.name for p in files] != [f"{i:03d}.zbr10" for i in range(n)]):
        raise ValueError("R11 missing/modified native captures")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--capture", type=Path, action="append")
    sources.add_argument("--capture-dir", type=Path,
                         help="completed R11 batch directory with completion marker")
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source", required=True,
                        choices=("instrumented_teacher", "instrumented_candidate"))
    args = parser.parse_args(argv)
    paths = read_complete_batch(args.capture_dir) if args.capture_dir is not None else args.capture
    print(json.dumps(save_bundle(paths, trace_path=args.trace_npz,
                                 checkpoint_sha256=args.checkpoint_sha256,
                                 source=args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
