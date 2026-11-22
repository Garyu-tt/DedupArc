"""R08B: activation-weighted low-rank corrections for frozen 3-bit experts.

The tiny FP16 residual encoder/decoder is trained OFFLINE using original
MXFP4 weights and calibration inputs. During CPU inference, no dense original
weights are needed. All factor bytes must count toward any compression claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import palette3_r08 as p

ADAPTER_HEADER_BYTES = 8

@dataclass(frozen=True)
class CorrectedCarrier:
    base: p.Carrier3Bit
    left: np.ndarray
    right: np.ndarray

    @property
    def frame_bytes(self):
        return self.base.frame_bytes + ADAPTER_HEADER_BYTES + self.left.nbytes + self.right.nbytes

    def dot(self, x):
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.base.groups * 32,) or not np.all(np.isfinite(v)):
            raise ValueError("invalid corrected GEMV vector")
        return self.base.dot(v) + self.left.astype(np.float64) @ (self.right.astype(np.float64) @ v)


def fit(blocks, scales, calibration, rank=2, *, carrier=None):
    b, s = p._check(blocks, scales)
    a = p._calibration(calibration, s.shape[1] * 32)
    if type(rank) is not int or not 1 <= rank <= min(s.shape[0], s.shape[1] * 32):
        raise ValueError("invalid adapter rank")
    factors = np.ldexp(np.full(s.shape, 0.5), s.astype(np.int16) - 127)
    original = (p.MANTISSAS[p._codes(b)].astype(np.float64) *
                factors[..., None]).reshape(s.shape[0], -1)
    base = carrier if carrier is not None else p.pack(b, s, a)
    if base.rows != s.shape[0] or base.groups != s.shape[1]:
        raise ValueError("carrier shape mismatch")
    approximate = np.empty_like(original)
    for row in range(base.rows):
        for group in range(base.groups):
            codes = base.read_codes(row, group)
            approximate[row, group * 32:(group + 1) * 32] = (
                p.MANTISSAS[codes] * factors[row, group])
    energy = np.sqrt(np.maximum(np.mean(a ** 2, axis=0), 1e-8))
    u, singular, vh = np.linalg.svd((original - approximate) * energy[None, :],
                                    full_matrices=False)
    left = (u[:, :rank] * singular[None, :rank]).astype(np.float16)
    right = (vh[:rank, :] / energy[None, :]).astype(np.float16)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("fp16 correction factors overflow")
    return CorrectedCarrier(base, left, right)


def measure(blocks, scales, calibration, validation, rank=2, *,
            max_l2=0.04, min_cosine=0.999):
    b, s = p._check(blocks, scales)
    x = p._calibration(validation, s.shape[1] * 32)
    base = p.pack(b, s, calibration)
    corrected = fit(b, s, calibration, rank=rank, carrier=base)
    desired = np.stack([p.dense_dot(b, s, v) for v in x])
    plain = np.stack([base.dot(v) for v in x])
    fixed = np.stack([corrected.dot(v) for v in x])
    def score(predicted):
        l2 = float(np.linalg.norm(predicted - desired) /
                   max(np.linalg.norm(desired), 1e-30))
        norm = float(np.linalg.norm(predicted) * np.linalg.norm(desired))
        cos = (1.0 if norm < 1e-30 and not np.any(predicted - desired)
               else float(np.sum(predicted * desired) / max(norm, 1e-30)))
        return l2, cos
    plain_l2, _ = score(plain)
    corrected_l2, corrected_cosine = score(fixed)
    ok = (corrected.frame_bytes < base.source_bytes and
          corrected_l2 <= max_l2 and corrected_cosine >= min_cosine)
    return corrected if ok else None, {
        "rank": rank,
        "raw_bytes": base.source_bytes,
        "plain_3bit_bytes": base.frame_bytes,
        "corrected_bytes": corrected.frame_bytes,
        "plain_l2": round(plain_l2, 8),
        "corrected_l2": round(corrected_l2, 8),
        "corrected_cosine": round(corrected_cosine, 8),
        "accepted_after_quality_and_size_gate": bool(ok),
        "actual_model_teacher_activations": False,
    }
