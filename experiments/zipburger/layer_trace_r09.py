"""R09: strict, bounded 24-layer teacher/student trace contract and drift gate.

Source-independent: model adapters write aligned activations, router indices
and next-token predictions; no model, weights, secrets or prompt text in traces.
Passing this gate does NOT prove the agentic abilities of an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

LAYERS = 24
ROUTED_EXPERTS = 4
LOCAL_EXPERTS = 32
MAX_SAMPLES = 512
MAX_HIDDEN = 8192
MAX_UNCOMPRESSED = 256 * 1024 * 1024
MAX_COMPRESSED = 128 * 1024 * 1024
_HEX64 = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class Trace:
    sample_ids: np.ndarray
    layers: tuple[np.ndarray, ...]
    routes: tuple[np.ndarray, ...]
    predictions: np.ndarray
    prompt_digest: str
    checkpoint_digest: str
    source: str


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _validate(t):
    if not isinstance(t, Trace) or len(t.layers) != LAYERS or len(t.routes) != LAYERS:
        raise ValueError("R09 needs exactly 24 aligned layer and routing outputs")
    if (not isinstance(t.sample_ids, np.ndarray) or t.sample_ids.dtype != np.int64
            or t.sample_ids.ndim != 1 or not 1 <= t.sample_ids.size <= MAX_SAMPLES
            or np.unique(t.sample_ids).size != t.sample_ids.size):
        raise ValueError("sample IDs must be unique int64 rows")
    n = t.sample_ids.size
    if (not isinstance(t.predictions, np.ndarray)
            or t.predictions.dtype != np.int64 or t.predictions.shape != (n,)
            or np.any(t.predictions < 0)):
        raise ValueError("invalid per-token top1 prediction indices")
    width = None
    for i, (h, r) in enumerate(zip(t.layers, t.routes)):
        if (not isinstance(h, np.ndarray) or h.dtype != np.float32 or
                h.ndim != 2 or h.shape[0] != n or
                not 1 <= h.shape[1] <= MAX_HIDDEN or not np.all(np.isfinite(h))):
            raise ValueError(f"invalid layer {i} activations")
        if width is None:
            width = h.shape[1]
        elif h.shape[1] != width:
            raise ValueError("inconsistent hidden width across layers")
        if (not isinstance(r, np.ndarray) or r.dtype != np.int16 or
                r.shape != (n, ROUTED_EXPERTS) or np.any(r < 0)
                or np.any(r >= LOCAL_EXPERTS)
                or np.any(np.diff(np.sort(r, axis=-1), axis=-1) == 0)):
            raise ValueError(f"invalid layer {i} top4 routing")
    if not (_HEX64.fullmatch(t.prompt_digest) and _HEX64.fullmatch(t.checkpoint_digest)):
        raise ValueError("missing verified-format corpus/model digests")
    if t.source not in ("synthetic_fixture", "instrumented_teacher", "instrumented_candidate"):
        raise ValueError("unknown trace provenance")
    return n, width


def save(trace: Trace, path):
    n, width = _validate(trace)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("R09 trace file must end in .npz")
    data = {"sample_ids": trace.sample_ids, "predictions": trace.predictions}
    for idx in range(LAYERS):
        data[f"hidden_{idx:02d}"] = trace.layers[idx]
        data[f"route_{idx:02d}"] = trace.routes[idx]
    np.savez_compressed(path, **data)
    if path.stat().st_size > MAX_COMPRESSED:
        path.unlink(missing_ok=True)
        raise ValueError("R09 trace exceeds compressed size limit")
    meta = {
        "schema": "zipburger-r09-layer-trace-v1",
        "sha256": _hash(path.read_bytes()),
        "sample_count": n, "hidden_width": width,
        "prompt_digest": trace.prompt_digest,
        "checkpoint_digest": trace.checkpoint_digest,
        "source": trace.source,
        "raw_prompt_text_included": False,
        "model_agenticity_proven": False,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n",
                                         encoding="utf-8")
    return meta


def load(path):
    path = Path(path)
    if path.suffix != ".npz" or not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("trace .npz and paired .json manifest are required")
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError("trace compressed size limit exceeded")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("schema") != "zipburger-r09-layer-trace-v1":
        raise ValueError("unsupported trace schema")
    if meta.get("sha256") != _hash(path.read_bytes()):
        raise ValueError("trace/manifest checksum mismatch")
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) != 2 + 2 * LAYERS:
            raise ValueError("incorrect trace array count")
        if sum(x.file_size for x in z.infolist()) > MAX_UNCOMPRESSED:
            raise ValueError("uncompressed trace size budget exceeded")
        if any(x.file_size > MAX_UNCOMPRESSED or x.filename.startswith("/") or
               ".." in x.filename for x in z.infolist()):
            raise ValueError("unsafe trace entry")
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"sample_ids", "predictions"} | {
            f"{kind}_{i:02d}" for kind in ("hidden", "route")
            for i in range(LAYERS)}
        if set(arrays.files) != expected:
            raise ValueError("missing or unexpected trace tensors")
        t = Trace(
            np.array(arrays["sample_ids"]),
            tuple(np.array(arrays[f"hidden_{i:02d}"]) for i in range(LAYERS)),
            tuple(np.array(arrays[f"route_{i:02d}"]) for i in range(LAYERS)),
            np.array(arrays["predictions"]),
            meta["prompt_digest"], meta["checkpoint_digest"], meta["source"],
        )
    n, width = _validate(t)
    if n != meta.get("sample_count") or width != meta.get("hidden_width"):
        raise ValueError("trace metadata shape mismatch")
    return t


def compare(teacher: Trace, candidate: Trace, *, max_l2=0.04,
            min_cosine=0.999, require_router_exact=True,
            require_top1_exact=True):
    _validate(teacher)
    _validate(candidate)
    if teacher.sample_ids.shape != candidate.sample_ids.shape or not np.array_equal(
            teacher.sample_ids, candidate.sample_ids):
        raise ValueError("teacher/candidate sample IDs or order mismatch")
    if teacher.prompt_digest != candidate.prompt_digest:
        raise ValueError("different prompt corpus: cannot align traces")
    if (teacher.layers[0].shape != candidate.layers[0].shape or
            teacher.source == "instrumented_candidate"):
        raise ValueError("invalid teacher/candidate pairing")
    if not (np.isfinite(max_l2) and max_l2 >= 0 and np.isfinite(min_cosine) and
            -1 <= min_cosine <= 1):
        raise ValueError("invalid quality limits")
    layers = []
    for idx, (a, b, ra, rb) in enumerate(zip(
            teacher.layers, candidate.layers, teacher.routes, candidate.routes)):
        base = a.astype(np.float64)
        other = b.astype(np.float64)
        reference_norm = float(np.linalg.norm(base))
        other_norm = float(np.linalg.norm(other))
        diff = float(np.linalg.norm(other - base))
        rel_l2 = diff / max(reference_norm, 1e-30)
        cos = (1.0 if reference_norm < 1e-30 and other_norm < 1e-30
               else float(np.sum(base * other) / max(reference_norm * other_norm, 1e-30)))
        router_equal = float(np.mean(np.all(ra == rb, axis=1)))
        layers.append({
            "layer": idx, "relative_l2": rel_l2, "cosine": cos,
            "routing_top4_exact_fraction": router_equal,
            "passes_local_gate": bool(rel_l2 <= max_l2 and cos >= min_cosine and
                                      (not require_router_exact or router_equal == 1.0)),
        })
    top1_agreement = float(np.mean(teacher.predictions == candidate.predictions))
    gate = all(k["passes_local_gate"] for k in layers)
    if require_top1_exact and top1_agreement < 1:
        gate = False
    return {
        "layers": layers,
        "max_layer_relative_l2": max(k["relative_l2"] for k in layers),
        "minimum_layer_cosine": min(k["cosine"] for k in layers),
        "top1_agreement": top1_agreement,
        "all_24_layer_gates_pass": bool(gate),
        "same_checkpoint_digest": teacher.checkpoint_digest == candidate.checkpoint_digest,
        "genuine_model_traces": (teacher.source == "instrumented_teacher"
                                 and candidate.source == "instrumented_candidate"),
        "agenticity_evaluated": False,
        "agenticity_proven": False,
    }
