"""NeuroCrunch ZipBurger R01: exact int8 shared-root neural carrier.

Experimental, pure Python+NumPy implementation. No entropy codec is used.
The compressed carrier is a set of root neurons, denominator links and a sparse
residual connection matrix. A fused matrix/vector product uses these structures
without reconstructing the original dense matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Optional

import numpy as np

_MAGIC_GRAPH = b"ZBG1"
_MAGIC_RAW = b"ZBR1"
_HEAD = struct.Struct("<4sIII")
_LINK = struct.Struct("<BHBH")  # tag, root-id, denominator, correction-count
_ENTRY = struct.Struct("<Hh")  # input column, signed exact correction
_MAX_ELEMENTS = 8_388_608  # research parser allocation guard
_DENOMS = (8, 4, 2, 1)  # root activation multiplied by 1,2,4,8


@dataclass(frozen=True)
class Carrier:
    """Self-contained, bit-exact int8 representation (without dense reconstruction)."""

    rows: int
    cols: int
    roots: np.ndarray                 # (k,n) int8: shared learned/selected bases
    root_ids: np.ndarray              # (m,) int32: -1 means raw row in residual matrix
    denominators: np.ndarray          # (m,) int16
    residual_rows: np.ndarray         # COO row coordinates
    residual_cols: np.ndarray         # COO column coordinates
    residual_values: np.ndarray       # COO signed corrections, int16
    digest: bytes
    payload: bytes

    def dot(self, x: np.ndarray) -> np.ndarray:
        """Compute Wx from root neuron activations and sparse connections only.

        Integer inputs use exact int64 arithmetic provided accumulation stays in
        int64 range; caller is responsible for that bound.
        """
        arr = np.asarray(x)
        if arr.ndim != 1 or arr.size != self.cols:
            raise ValueError("x must be a vector of length cols")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("exact dot expects integer input")
        x64 = arr.astype(np.int64)
        y = np.zeros(self.rows, dtype=np.int64)
        if len(self.roots):
            root_activation = self.roots.astype(np.int64) @ x64
            used = self.root_ids >= 0
            y[used] = root_activation[self.root_ids[used]] * (8 // self.denominators[used])
        if self.residual_values.size:
            terms = self.residual_values.astype(np.int64) * x64[self.residual_cols]
            np.add.at(y, self.residual_rows, terms)
        return y

    def verify_exact(self) -> None:
        """Stream SHA256 over reconstructed rows; never allocate a dense weight matrix."""
        h = hashlib.sha256()
        j = 0
        for i in range(self.rows):
            rid = self.root_ids[i]
            row = (self.roots[rid].astype(np.int32) * (8 // int(self.denominators[i]))
                   if rid >= 0 else np.zeros(self.cols, dtype=np.int32))
            while j < len(self.residual_rows) and self.residual_rows[j] == i:
                row[self.residual_cols[j]] += int(self.residual_values[j])
                j += 1
            if np.any((row < -128) | (row > 127)):
                raise ValueError("decoded values exceed signed int8 range")
            h.update(row.astype(np.int8).tobytes())
        if h.digest() != self.digest:
            raise ValueError("SHA256 mismatch in exact streamed verification")

    def restore(self) -> np.ndarray:
        """Explicit reconstruction for validation/export, never used by dot()."""
        w = np.zeros((self.rows, self.cols), dtype=np.int32)
        for i, rid in enumerate(self.root_ids):
            if rid >= 0:
                w[i] = self.roots[rid].astype(np.int32) * (8 // int(self.denominators[i]))
        np.add.at(w, (self.residual_rows, self.residual_cols), self.residual_values.astype(np.int32))
        if np.any((w < -128) | (w > 127)):
            raise ValueError("decoded values exceed signed int8 range")
        out = w.astype(np.int8)
        if hashlib.sha256(out.tobytes()).digest() != self.digest:
            raise ValueError("SHA256 mismatch after exact reconstruction")
        return out

    @property
    def compressed_bytes(self) -> int:
        return len(self.payload)

    @property
    def raw_bytes(self) -> int:
        return self.rows * self.cols

    @property
    def root_count(self) -> int:
        return len(self.roots)

    @property
    def link_count(self) -> int:
        return int(np.count_nonzero(self.root_ids >= 0))

    @property
    def residual_count(self) -> int:
        return int(self.residual_values.size)


def _validate_matrix(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights)
    if w.ndim != 2 or w.dtype != np.int8:
        raise TypeError("expected a 2D int8 matrix (quantize/bitcast explicitly upstream)")
    if not (0 < w.shape[0] <= 65535 and 0 < w.shape[1] <= 65535):
        raise ValueError("shape must be positive and fit 16-bit sparse coordinates")
    if w.size > _MAX_ELEMENTS:
        raise ValueError("research frame exceeds element safety limit")
    return np.ascontiguousarray(w)


def _candidate_roots(w: np.ndarray, limit: int) -> list[np.ndarray]:
    candidates: dict[bytes, np.ndarray] = {}
    # Deterministically sample rows from the complete matrix, not random data.
    indices = list(dict.fromkeys(
        list(range(min(len(w), min(16, max(1, limit // 8)))))
        + np.linspace(0, len(w) - 1, min(len(w), max(1, limit // 4)), dtype=int).tolist()
    ))
    for i in indices:
        row = w[i].astype(np.int16)
        for multiplier in (1, 2, 4, 8):
            if np.all(row % multiplier == 0):
                root = (row // multiplier).astype(np.int8)
                candidates.setdefault(root.tobytes(), root)
                if len(candidates) >= limit:
                    return list(candidates.values())
    return list(candidates.values())


def pack(weights: np.ndarray, *, max_roots: int = 32, max_candidates: int = 256) -> Carrier:
    """Find reusable base neurons and choose them only if net frame size shrinks.

    Explicitly charges for root table, link descriptors, sparse residuals,
    per-row tags, header and SHA256. Never claims ratios from omitted weights.
    """
    w = _validate_matrix(weights)
    m, n = w.shape
    if not (0 <= max_roots <= 65535 and max_candidates >= 1):
        raise ValueError("invalid root/candidate bounds")
    candidates = _candidate_roots(w, max_candidates)
    selected: list[np.ndarray] = []
    chosen = np.full(m, -1, dtype=np.int32)
    denoms = np.ones(m, dtype=np.int16)
    current_cost = np.full(m, n + 1, dtype=np.int64)  # tag + raw row
    while len(selected) < max_roots and candidates:
        best_gain = 0
        best_idx: Optional[int] = None
        best_cost = None
        best_denom = None
        for c_idx, root in enumerate(candidates):
            row_cost = np.full(m, n + 1, dtype=np.int64)
            row_denom = np.ones(m, dtype=np.int16)
            for d in _DENOMS:
                multiplier = 8 // d
                prediction = root.astype(np.int16) * multiplier
                mismatches = np.count_nonzero(w != prediction, axis=1)
                # 6 bytes: link/tag/root-index/denom/count; 4 bytes per sparse delta.
                size = 6 + mismatches.astype(np.int64) * 4
                keep = size < row_cost
                row_cost[keep], row_denom[keep] = size[keep], d
            gains = np.maximum(current_cost - row_cost, 0)
            # Require net savings after paying n bytes for an additional root.
            gain = int(gains.sum()) - n
            if gain > best_gain:
                best_gain, best_idx = gain, c_idx
                best_cost, best_denom = row_cost, row_denom
        if best_idx is None:
            break
        selected.append(candidates.pop(best_idx))
        improved = best_cost < current_cost
        chosen[improved] = len(selected) - 1
        denoms[improved] = best_denom[improved]
        current_cost[improved] = best_cost[improved]

    digest = hashlib.sha256(w.tobytes()).digest()
    roots = np.stack(selected).astype(np.int8) if selected else np.empty((0, n), dtype=np.int8)
    graph = bytearray(_HEAD.pack(_MAGIC_GRAPH, m, n, len(roots)))
    graph += digest
    graph += roots.tobytes()
    for i in range(m):
        rid = int(chosen[i])
        if rid < 0:
            graph += b"\x00" + w[i].tobytes()
            continue
        d = int(denoms[i])
        delta = w[i].astype(np.int16) - roots[rid].astype(np.int16) * (8 // d)
        nz = np.flatnonzero(delta)
        graph += _LINK.pack(1, rid, d, len(nz))
        for col in nz:
            graph += _ENTRY.pack(int(col), int(delta[col]))
    raw = _HEAD.pack(_MAGIC_RAW, m, n, 0) + digest + w.tobytes()
    return load(bytes(graph if len(graph) < len(raw) else raw))


def load(payload: bytes) -> Carrier:
    """Bounded parser for the research format; rejects malformed/truncated frames."""
    if len(payload) < _HEAD.size + 32:
        raise ValueError("truncated ZipBurger header")
    magic, m, n, k = _HEAD.unpack_from(payload)
    if magic not in (_MAGIC_RAW, _MAGIC_GRAPH) or not (0 < m <= 65535 and 0 < n <= 65535):
        raise ValueError("bad magic or shape")
    if m * n > _MAX_ELEMENTS or len(payload) > _MAX_ELEMENTS * 5:
        raise ValueError("research frame exceeds resource bounds")
    if magic == _MAGIC_RAW and k:
        raise ValueError("raw payload cannot contain root table")
    if k > min(m, 65535):
        raise ValueError("invalid root count")
    digest = payload[_HEAD.size:_HEAD.size + 32]
    p = _HEAD.size + 32
    root_size = k * n
    if p + root_size > len(payload):
        raise ValueError("truncated root neurons")
    roots = np.frombuffer(payload, dtype=np.int8, count=root_size, offset=p).reshape(k, n).copy()
    p += root_size
    root_ids = np.full(m, -1, dtype=np.int32)
    denoms = np.ones(m, dtype=np.int16)
    rr: list[int] = []
    cc: list[int] = []
    vv: list[int] = []
    if magic == _MAGIC_RAW:
        if len(payload) - p != m * n:
            raise ValueError("bad raw matrix size")
        dense = np.frombuffer(payload, dtype=np.int8, count=m * n, offset=p).reshape(m, n)
        rows, cols = np.nonzero(dense)
        rr.extend(rows.tolist())
        cc.extend(cols.tolist())
        vv.extend(dense[rows, cols].astype(np.int16).tolist())
    else:
        for i in range(m):
            if p >= len(payload):
                raise ValueError("truncated row")
            tag = payload[p]
            if tag == 0:
                p += 1
                if p + n > len(payload):
                    raise ValueError("truncated raw row")
                row = np.frombuffer(payload, dtype=np.int8, count=n, offset=p)
                cols = np.flatnonzero(row)
                rr.extend([i] * len(cols));cc.extend(cols.tolist());vv.extend(row[cols].astype(np.int16).tolist())
                p += n
            elif tag == 1:
                if p + _LINK.size > len(payload):
                    raise ValueError("truncated link")
                _, rid, d, count = _LINK.unpack_from(payload, p)
                p += _LINK.size
                if rid >= k or d not in _DENOMS or count > n:
                    raise ValueError("invalid denominator link")
                root_ids[i], denoms[i] = rid, d
                previous_col = -1
                for _ in range(count):
                    if p + _ENTRY.size > len(payload):
                        raise ValueError("truncated residual")
                    col, delta = _ENTRY.unpack_from(payload, p)
                    p += _ENTRY.size
                    if col >= n or col <= previous_col or delta == 0:
                        raise ValueError("bad sparse correction order/value")
                    rr.append(i);cc.append(col);vv.append(delta)
                    previous_col = col
            else:
                raise ValueError("invalid row tag")
        if p != len(payload):
            raise ValueError("trailing bytes")
    carrier = Carrier(m, n, roots, root_ids, denoms,
                      np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32),
                      np.asarray(vv, dtype=np.int16), digest, payload)
    carrier.verify_exact()  # streaming O(cols) integrity check, no dense W allocation
    return carrier


def make_torch_module(carrier: Carrier):
    """Return trainable input/output neural adapters around direct carrier compute.

    PyTorch is optional. The core has no dense decoded W Parameter/Buffer.
    Adapters initialize to identity for functional equivalence and can be trained
    on application data later. FP rounding may differ from original GEMM order.
    """
    import torch
    from torch import nn

    class ResidualAdapter(nn.Module):
        """Tiny trainable neural network initialized as an exact identity map."""
        def __init__(self, width: int, rank: int = 2):
            super().__init__()
            self.down = nn.Linear(width, rank, bias=False)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, x):
            return x + self.up(torch.tanh(self.down(x)))

    class ZipBurgerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_net = ResidualAdapter(carrier.cols)
            self.output_net = ResidualAdapter(carrier.rows)
            self.register_buffer("root_neurons", torch.from_numpy(carrier.roots.astype(np.float32)))
            self.register_buffer("row_root_ids", torch.from_numpy(np.maximum(carrier.root_ids, 0).astype(np.int64)))
            multipliers = np.where(carrier.root_ids >= 0, 8 // carrier.denominators, 0).astype(np.float32)
            self.register_buffer("row_gain", torch.from_numpy(multipliers))
            indices = np.stack((carrier.residual_rows, carrier.residual_cols)).astype(np.int64)
            corrections = torch.sparse_coo_tensor(torch.from_numpy(indices),
                        torch.from_numpy(carrier.residual_values.astype(np.float32)),
                        (carrier.rows, carrier.cols)).coalesce()
            self.register_buffer("correction_edges", corrections)

        def forward(self, inputs):
            if inputs.ndim != 2 or inputs.shape[1] != carrier.cols:
                raise ValueError("expected [batch, cols]")
            x = self.input_net(inputs)
            root_act = torch.nn.functional.linear(x, self.root_neurons)
            if self.root_neurons.shape[0]:
                y = root_act[:, self.row_root_ids] * self.row_gain
            else:
                y = x.new_zeros((x.shape[0], carrier.rows))
            # Sparse edges directly contribute to output, no dense weight restore.
            y = y + torch.sparse.mm(self.correction_edges, x.T).T
            return self.output_net(y)

    return ZipBurgerNet()
