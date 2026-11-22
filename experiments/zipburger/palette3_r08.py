"""ZipBurger R08: research-only lossy 3-bit E2M1 MoE tile with exact E8M0 scales.

For each 32-value MXFP4 block retain one scale byte, and encode 32
3-bit global-palette indices in 12 bytes. Only expert tensors qualify.
Attention, router, KV cache, and residual topology are never rewritten.
Neither tile quality nor model agenticity is implied by packing ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
import numpy as np

MAGIC = b"ZB38"
HEADER = struct.Struct("<4sHH8s32s")
MANTISSAS = np.array((0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), dtype=np.int8)
MAX_BLOCKS = 4096
MAX_CALIBRATION_ROWS = 2048
BLOCK_PACKED_BYTES = 13
EXPERT_TENSOR = re.compile(r"blk\.(\d+)\.ffn_(down|gate|up)_exps\.weight\Z")


def expert_layer_id(name):
    match = EXPERT_TENSOR.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError("R08 may only approximate MoE experts; routing and attention protected")
    return int(match.group(1))


def _check(blocks, scales):
    b, s = np.asarray(blocks), np.asarray(scales)
    if b.dtype != np.uint8 or s.dtype != np.uint8 or b.ndim != 3 or s.ndim != 2:
        raise TypeError("expected uint8 blocks [rows,groups,16] and scales [rows,groups]")
    if b.shape[-1] != 16 or b.shape[:2] != s.shape:
        raise ValueError("MXFP4 shape mismatch")
    rows, groups = s.shape
    if not (1 <= rows <= 2048 and 1 <= groups <= 256 and rows * groups <= MAX_BLOCKS):
        raise ValueError("R08 research tile exceeds bounds")
    return np.ascontiguousarray(b), np.ascontiguousarray(s)


def _codes(blocks):
    c = np.empty((*blocks.shape[:2], 32), dtype=np.uint8)
    c[..., ::2], c[..., 1::2] = blocks & 15, blocks >> 4
    return c


def _calibration(x, width):
    a = np.asarray(x)
    if (a.ndim != 2 or not 1 <= a.shape[0] <= MAX_CALIBRATION_ROWS or
            a.shape[1] != width or not np.issubdtype(a.dtype, np.number) or
            not np.all(np.isfinite(a))):
        raise ValueError("invalid bounded activation matrix")
    return a.astype(np.float64, copy=False)


def dense_dot(blocks, scales, x):
    """Offline FLOAT64 MXFP4 reference, never the compressed inference path."""
    b, s = _check(blocks, scales)
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (s.shape[1] * 32,) or not np.all(np.isfinite(v)):
        raise ValueError("invalid dense reference input")
    factors = np.ldexp(np.full(s.shape, 0.5), s.astype(np.int16) - 127)
    return (MANTISSAS[_codes(b)].astype(np.float64) * factors[..., None]).reshape(
        s.shape[0], -1) @ v


def choose_palette(blocks, scales, calibration):
    b, s = _check(blocks, scales)
    a = _calibration(calibration, s.shape[1] * 32)
    energy = np.mean(a ** 2, axis=0).reshape(s.shape[1], 32)
    factors = np.ldexp(np.full(s.shape, 0.5), s.astype(np.int16) - 127)
    importance = (factors / np.max(factors)) ** 2
    codes = _codes(b)
    counts = np.bincount(codes.ravel(),
                         weights=np.broadcast_to(importance[..., None] * energy[None, :, :],
                                                 codes.shape).ravel(),
                         minlength=16)
    counts[0] += counts[8]  # +0 and -0 have the same numerical weight
    counts[8] = 0
    dist = (MANTISSAS[:, None].astype(np.float64) - MANTISSAS[None, :]) ** 2
    selected = [0]
    while len(selected) < 8:
        candidates = [c for c in range(16) if c not in selected and c != 8]
        def score(c):
            return float(counts @ np.minimum(np.min(dist[:, selected], axis=1),
                                               dist[:, c]))
        selected.append(min(candidates, key=lambda c: (score(c), c)))
    return np.array(selected, dtype=np.uint8)


def _pack_3bit(indices):
    values = np.asarray(indices, dtype=np.uint8).ravel()
    if values.size != 32 or np.any(values > 7):
        raise ValueError("invalid 32 3-bit indices")
    data = bytearray(12)
    for i, code in enumerate(values):
        off, shift = divmod(i * 3, 8)
        data[off] |= (int(code) << shift) & 255
        if shift > 5:
            data[off + 1] |= int(code) >> (8 - shift)
    return bytes(data)


def _unpack_3bit(payload):
    if len(payload) != 12:
        raise ValueError("invalid 3-bit block length")
    out = np.empty(32, np.uint8)
    for i in range(32):
        off, shift = divmod(i * 3, 8)
        value = payload[off] >> shift
        if shift > 5:
            value |= payload[off + 1] << (8 - shift)
        out[i] = value & 7
    return out


@dataclass(frozen=True)
class Carrier3Bit:
    payload: bytes
    rows: int
    groups: int
    palette: np.ndarray
    data: bytes

    @property
    def source_bytes(self):
        return self.rows * self.groups * 17

    @property
    def frame_bytes(self):
        return len(self.payload)

    def read_codes(self, row, group):
        if (type(row) is not int or type(group) is not int or
                not (0 <= row < self.rows and 0 <= group < self.groups)):
            raise IndexError((row, group))
        off = (row * self.groups + group) * BLOCK_PACKED_BYTES
        return self.palette[_unpack_3bit(self.data[off + 1:off + 13])]

    def dot(self, x):
        """Direct inference from packed bytes, decoding one 32-value block at a time."""
        v = np.asarray(x, dtype=np.float64)
        if v.ndim != 1 or v.size != self.groups * 32 or not np.all(np.isfinite(v)):
            raise ValueError("invalid vector")
        y = np.zeros(self.rows, np.float64)
        for row in range(self.rows):
            for group in range(self.groups):
                off = (row * self.groups + group) * BLOCK_PACKED_BYTES
                scale = np.ldexp(0.5, int(self.data[off]) - 127)
                codes = self.palette[_unpack_3bit(self.data[off + 1:off + 13])]
                y[row] += scale * (MANTISSAS[codes].astype(np.float64) @
                                   v[group * 32:(group + 1) * 32])
        return y


def pack(blocks, scales, calibration):
    b, s = _check(blocks, scales)
    palette = choose_palette(b, s, calibration)
    values = MANTISSAS[_codes(b)].astype(np.int16)
    pal_values = MANTISSAS[palette].astype(np.int16)
    indices = np.argmin((values[..., None] - pal_values) ** 2, axis=-1)
    data = bytearray()
    for row in range(s.shape[0]):
        for group in range(s.shape[1]):
            data.append(int(s[row, group]))
            data.extend(_pack_3bit(indices[row, group]))
    encoded = bytes(data)
    header = HEADER.pack(MAGIC, s.shape[0], s.shape[1], palette.tobytes(),
                         hashlib.sha256(encoded).digest())
    return load(header + encoded)


def load(payload):
    if not isinstance(payload, bytes) or len(payload) < HEADER.size:
        raise ValueError("truncated R08 frame")
    magic, rows, groups, palette_bytes, digest = HEADER.unpack_from(payload)
    if (magic != MAGIC or not 1 <= rows <= 2048 or not 1 <= groups <= 256 or
            rows * groups > MAX_BLOCKS):
        raise ValueError("invalid R08 frame bounds/header")
    palette = np.frombuffer(palette_bytes, dtype=np.uint8)
    if np.any(palette > 15) or np.unique(palette).size != 8:
        raise ValueError("invalid R08 palette")
    data = payload[HEADER.size:]
    if len(data) != rows * groups * BLOCK_PACKED_BYTES:
        raise ValueError("invalid R08 payload length")
    if hashlib.sha256(data).digest() != digest:
        raise ValueError("corrupt R08 payload")
    return Carrier3Bit(payload, rows, groups, palette, data)


def evaluate(blocks, scales, calibration, validation, *,
             max_relative_l2=0.04, min_cosine=0.999, fallback_factory=None):
    """Held-out tile quality gate; NOT a model/agenticity test."""
    b, s = _check(blocks, scales)
    _calibration(calibration, s.shape[1] * 32)
    holdout = _calibration(validation, s.shape[1] * 32)
    carrier = pack(b, s, calibration)
    reference = np.stack([dense_dot(b, s, x) for x in holdout])
    predicted = np.stack([carrier.dot(x) for x in holdout])
    relative_l2 = float(np.linalg.norm(predicted - reference) /
                        max(np.linalg.norm(reference), 1e-30))
    a, z = np.linalg.norm(reference), np.linalg.norm(predicted)
    cosine = (1.0 if a < 1e-30 and z < 1e-30 else
              float(np.sum(reference * predicted) / max(a * z, 1e-30)))
    ok = relative_l2 <= max_relative_l2 and cosine >= min_cosine
    report = {
        "raw_mxfp4_bytes": carrier.source_bytes,
        "three_bit_bytes": carrier.frame_bytes,
        "three_bit_saving_pct": round(100 * (1 - carrier.frame_bytes / carrier.source_bytes), 5),
        "holdout_relative_l2": round(relative_l2, 8),
        "holdout_cosine": round(cosine, 8),
        "quality_gate_passed": bool(ok),
        "quality_gate_limits": {"max_relative_l2": max_relative_l2,
                                "min_cosine": min_cosine},
        "model_level_or_agentic_equivalence": False,
    }
    if ok:
        return carrier, report
    if fallback_factory is not None:
        original = fallback_factory(b, s)
        report["fallback_bytes"] = original.frame_bytes
        report["fallback_mode"] = getattr(original, "mode", "lossless")
        return original, report
    return None, report


def evaluate_expert(name, blocks, scales, calibration, validation, **kwargs):
    layer = expert_layer_id(name)
    carrier, report = evaluate(blocks, scales, calibration, validation, **kwargs)
    report.update({"tensor": name, "layer": layer,
                   "protected_attention_and_router_unchanged": True})
    return carrier, report
