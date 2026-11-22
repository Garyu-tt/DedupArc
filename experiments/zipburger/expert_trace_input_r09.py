"""R09 bounded, operator-attested expert activation input archives.

No new synthetic inputs in this module: only aligned per-expert inputs
captured on disjoint prompt corpora can be passed to the R08 tile gate.
The source metadata is a declaration, not cryptographic hardware attestation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

_EXPERT_KEY = re.compile(r"layer_(\d{2})_expert_(\d{2})\Z")
MAX_ARCHIVE_COMPRESSED = 128 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED = 256 * 1024 * 1024
WIDTH = 512
MIN_EXPERT_SAMPLES = 8


def read_archive(path):
    p = Path(path)
    if not p.is_file() or p.suffix != ".npz" or not p.with_suffix(".json").is_file():
        raise ValueError("expert input archive .npz and matching .json required")
    if p.stat().st_size > MAX_ARCHIVE_COMPRESSED:
        raise ValueError("oversized expert archive")
    manifest = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "zipburger-r09-expert-inputs-v1"
            or manifest.get("input_width") != WIDTH
            or manifest.get("projections") != ["up", "gate"]
            or manifest.get("contains_down_projection_inputs") is not False
            or manifest.get("source") != "instrumented_teacher"):
        raise ValueError("archive must be instrumented teacher up/gate only")
    if manifest.get("sha256") != hashlib.sha256(p.read_bytes()).hexdigest():
        raise ValueError("archive provenance manifest checksum mismatch")
    for digest in ("checkpoint_digest", "prompt_digest"):
        value = manifest.get(digest)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid archive digest")
    token_hashes = manifest.get("token_sequence_sha256")
    if (not isinstance(token_hashes, list) or not 1 <= len(token_hashes) <= 8
            or any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
                   for x in token_hashes)
            or len(set(token_hashes)) != len(token_hashes)):
        raise ValueError("missing/invalid tokenized prompt fingerprints")
    with zipfile.ZipFile(p) as archive:
        names = archive.namelist()
        if not 1 <= len(names) <= 24 * 32:
            raise ValueError("wrong archive entry count")
        if sum(z.file_size for z in archive.infolist()) > MAX_ARCHIVE_UNCOMPRESSED:
            raise ValueError("uncompressed archive exceeds budget")
        if any(z.file_size > MAX_ARCHIVE_UNCOMPRESSED for z in archive.infolist()):
            raise ValueError("unsafe archive element")
    with np.load(p, allow_pickle=False) as inp:
        result = {}
        for name in inp.files:
            match = _EXPERT_KEY.fullmatch(name)
            if not match or int(match.group(1)) >= 24 or int(match.group(2)) >= 32:
                raise ValueError("unrecognized expert activation key")
            values = np.array(inp[name])
            if (values.dtype != np.float32 or values.ndim != 2 or
                    values.shape[1] != WIDTH or not 1 <= values.shape[0] <= 256 or
                    not np.all(np.isfinite(values))):
                raise ValueError("malformed expert activation array")
            result[name] = values
    return manifest, result


def load_pair(calibration_path, validation_path, *, return_metadata=False):
    c_meta, cal = read_archive(calibration_path)
    v_meta, val = read_archive(validation_path)
    if c_meta["checkpoint_digest"] != v_meta["checkpoint_digest"]:
        raise ValueError("different teacher checkpoints")
    if c_meta["prompt_digest"] == v_meta["prompt_digest"]:
        raise ValueError("calibration and validation prompt manifests identical")
    if set(c_meta["token_sequence_sha256"]) & set(v_meta["token_sequence_sha256"]):
        raise ValueError("calibration and validation share tokenized prompts")
    return (cal, val, c_meta, v_meta) if return_metadata else (cal, val)


def matched_vectors(cal, val, *, tensor, row_start):
    """Identify true gate/up expert input slices for GGUF 2880-row experts."""
    match = re.fullmatch(r"blk\.(\d+)\.ffn_(up|gate)_exps\.weight", tensor)
    if match is None:
        raise ValueError("down expert inputs are not available in HF pre-hook")
    if (type(row_start) is not int or row_start < 0 or
            row_start % 2880 + 48 > 2880):
        raise ValueError("tile crosses 2880-row expert boundary")
    expert_id = row_start // 2880
    if expert_id >= 32:
        raise ValueError("expert ID exceeds 20B config")
    key = f"layer_{int(match.group(1)):02d}_expert_{expert_id:02d}"
    a, b = cal.get(key), val.get(key)
    if a is None or b is None:
        return None
    if a.shape[0] < MIN_EXPERT_SAMPLES or b.shape[0] < MIN_EXPERT_SAMPLES:
        return None
    return a, b
