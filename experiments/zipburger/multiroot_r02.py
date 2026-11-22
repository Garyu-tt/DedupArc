"""NeuroCrunch ZipBurger R02: exact two-link neural carrier.

Research-only, standard library + NumPy; no entropy coding or PyTorch required.
The data graph is W=A@B+R, where A has at most two signed integer
connections per row, B stores shared root neurons, and R stores exact sparse
residual synapses.  Direct GEMV never materializes dense W.

The packed representation is intended to expose meaningful arithmetic
structure, not to promise compression of encrypted or entropy-coded bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

_HEADER = struct.Struct('<4sIIIQ')  # magic, rows, columns, roots, original data bytes
_SHA_SIZE = 32
_ROW = struct.Struct('<BBH')        # tag, number of links, correction count
_EDGE = struct.Struct('<Hb')        # root index, signed coefficient
_RES = struct.Struct('<Hh')         # column, signed exact residual
_GRAPH = b'ZBG2'
_RAW = b'ZBR2'
_MAX_ELEMENTS = 2_097_152
_COEFFICIENTS = (1, -1, 2, -2, 4, -4)


@dataclass(frozen=True)
class NeuralCarrierR02:
    payload: bytes
    rows: int
    cols: int
    bases: np.ndarray            # [k, n] int8
    index: np.ndarray            # [m,2] int32; -1=unused
    coeff: np.ndarray            # [m,2] int8
    residual_row: np.ndarray     # [e] int32
    residual_col: np.ndarray     # [e] int32
    residual: np.ndarray         # [e] int16
    literal_row: np.ndarray      # [m] bool
    literal_data: bytes          # one row per literal, concatenated
    digest: bytes
    original_length: int

    @property
    def frame_bytes(self) -> int:
        return len(self.payload)

    @property
    def links(self) -> int:
        return int(np.count_nonzero(self.index >= 0))

    @property
    def residuals(self) -> int:
        return int(self.residual.size)

    def _row(self, row: int) -> np.ndarray:
        if not 0 <= row < self.rows:
            raise IndexError(row)
        if self.literal_row[row]:
            position = int(np.count_nonzero(self.literal_row[:row])) * self.cols
            return np.frombuffer(self.literal_data, np.int8, count=self.cols, offset=position).copy()
        data = np.zeros(self.cols, np.int32)
        for j in range(2):
            idx = int(self.index[row, j])
            if idx >= 0:
                data += self.bases[idx].astype(np.int32) * int(self.coeff[row, j])
        start = int(np.searchsorted(self.residual_row, row, side='left'))
        end = int(np.searchsorted(self.residual_row, row, side='right'))
        if end > start:
            data[self.residual_col[start:end]] += self.residual[start:end].astype(np.int32)
        if np.any((data < -128) | (data > 127)):
            raise ValueError('decoded int8 overflow')
        return data.astype(np.int8)

    def read_cell(self, row: int, col: int) -> int:
        """Retrieve one exact original byte value, without restoring a matrix."""
        if not 0 <= row < self.rows or not 0 <= col < self.cols:
            raise IndexError((row, col))
        if self.literal_row[row]:
            position = int(np.count_nonzero(self.literal_row[:row])) * self.cols + col
            return int(np.frombuffer(self.literal_data, np.int8, count=1, offset=position)[0])
        value = 0
        for j in range(2):
            k = int(self.index[row, j])
            if k >= 0:
                value += int(self.coeff[row, j]) * int(self.bases[k, col])
        start = int(np.searchsorted(self.residual_row, row, side='left'))
        end = int(np.searchsorted(self.residual_row, row, side='right'))
        where = np.searchsorted(self.residual_col[start:end], col)
        if where < end - start and self.residual_col[start+where] == col:
            value += int(self.residual[start+where])
        return value

    def verify(self) -> None:
        hasher = hashlib.sha256()
        for i in range(self.rows):
            hasher.update(self._row(i).tobytes())
        if hasher.digest() != self.digest:
            raise ValueError('ZipBurger SHA256 mismatch')

    def restore(self) -> np.ndarray:
        """Explicit export/debug operation only."""
        return np.vstack([self._row(i) for i in range(self.rows)])

    def dot(self, x: np.ndarray) -> np.ndarray:
        """Exact int64 GEMV directly on bases, row links, residual synapses.

        The integer accumulator must not overflow signed 64-bit range.
        """
        vector = np.asarray(x)
        if vector.ndim != 1 or vector.size != self.cols or not np.issubdtype(vector.dtype, np.integer):
            raise TypeError('expected integer vector with cols entries')
        v = vector.astype(np.int64)
        out = np.zeros(self.rows, dtype=np.int64)
        if len(self.bases):
            activated = self.bases.astype(np.int64) @ v
            for j in range(2):
                good = self.index[:, j] >= 0
                out[good] += activated[self.index[good, j]] * self.coeff[good, j].astype(np.int64)
        if self.residuals:
            np.add.at(out, self.residual_row, self.residual.astype(np.int64) * v[self.residual_col])
        if np.any(self.literal_row):
            literal = np.frombuffer(self.literal_data, dtype=np.int8).reshape(-1, self.cols)
            out[self.literal_row] = literal.astype(np.int64) @ v
        return out

    def restore_bytes(self) -> bytes:
        """Return original post-codec payload exactly; strip only framing pad."""
        bits = (self.restore().view(np.uint8).tobytes())[:self.original_length]
        return bits


def _matrix(weights: np.ndarray) -> np.ndarray:
    x = np.asarray(weights)
    if x.ndim != 2 or x.dtype != np.int8:
        raise TypeError('ZipBurger R02 accepts an explicit INT8 2D matrix')
    if not 0 < x.shape[0] <= 65535 or not 0 < x.shape[1] <= 65535 or x.size > _MAX_ELEMENTS:
        raise ValueError('invalid dimensions/over research memory bound')
    return np.ascontiguousarray(x)


def _candidate_roots(w: np.ndarray, maximum: int) -> list[np.ndarray]:
    # Deterministic broad selection. Never search unrestricted pair matrices.
    ids = list(dict.fromkeys(list(range(min(len(w), maximum//2))) +
                              np.linspace(0, len(w)-1, min(len(w), maximum), dtype=int).tolist()))
    seen: set[bytes] = set()
    roots = []
    for row in ids:
        b = w[row].tobytes()
        if b not in seen:
            seen.add(b)
            roots.append(w[row].copy())
        if len(roots) == maximum:
            break
    return roots


def pack(weights: np.ndarray, *, max_roots: int = 16, max_candidates: int = 64,
         original_length: int | None = None) -> NeuralCarrierR02:
    """Greedy dictionary selection with 0-2 links per row and full-size costing."""
    w = _matrix(weights)
    m, n = w.shape
    if original_length is None:
        original_length = m * n
    if not (0 < original_length <= m*n and 0 <= max_roots <= 128 and 1 <= max_candidates <= 256):
        raise ValueError('invalid packing bounds')
    candidates = _candidate_roots(w, max_candidates)
    selected: list[np.ndarray] = []
    links: list[list[tuple[int,int]]] = [[] for _ in range(m)]
    predicted = np.zeros((m,n), dtype=np.int32)
    costs = np.full(m, n + _ROW.size, dtype=np.int32)
    # Our encoder writes an encoded raw row as _ROW + data; the global raw
    # fallback is even smaller and will win if the full graph adds no value.
    for _ in range(max_roots):
        best_gain = 0
        best_index = -1
        best_choice = None
        for candidate_idx, root in enumerate(candidates):
            root32 = root.astype(np.int32)
            local_choice = []
            local_gain = 0
            for i in range(m):
                current = int(costs[i])
                best = (current, 0)
                if len(links[i]) < 2:
                    for coefficient in _COEFFICIENTS:
                        diff = w[i].astype(np.int32) - (predicted[i] + coefficient*root32)
                        mismatches = int(np.count_nonzero(diff))
                        size = _ROW.size + (len(links[i])+1)*_EDGE.size + mismatches*_RES.size
                        if size < best[0]:
                            best = (size, coefficient)
                local_gain += current - best[0]
                local_choice.append(best)
            net = local_gain - n  # bank of shared neurons costs n bytes per root
            if net > best_gain:
                best_gain = net
                best_index = candidate_idx
                best_choice = local_choice
        if best_index < 0:
            break
        new_root = candidates.pop(best_index)
        selected.append(new_root)
        idx = len(selected) - 1
        root32 = new_root.astype(np.int32)
        for i, (cost, coefficient) in enumerate(best_choice):
            if coefficient:
                links[i].append((idx, coefficient))
                predicted[i] += coefficient * root32
                costs[i] = cost
        if not candidates:
            break

    digest = hashlib.sha256(w.tobytes()).digest()
    raw = _HEADER.pack(_RAW, m, n, 0, original_length) + digest + w.tobytes()
    graph = bytearray(_HEADER.pack(_GRAPH, m, n, len(selected), original_length) + digest)
    if selected:
        graph.extend(np.stack(selected).tobytes())
    for i in range(m):
        if not links[i]:
            graph.extend(_ROW.pack(0, 0, 0))
            graph.extend(w[i].tobytes())
            continue
        diff = w[i].astype(np.int32) - predicted[i]
        positions = np.flatnonzero(diff)
        if len(positions) > 65535 or np.any((diff[positions] < -32768) | (diff[positions] > 32767)):
            raise ValueError('int16 residual overflow')
        graph.extend(_ROW.pack(1, len(links[i]), len(positions)))
        for root_idx, coefficient in links[i]:
            graph.extend(_EDGE.pack(root_idx, coefficient))
        for col in positions:
            graph.extend(_RES.pack(int(col), int(diff[col])))
    return load(bytes(graph if len(graph) < len(raw) else raw))


def pack_bytes(data: bytes, width: int = 256, **kwargs) -> NeuralCarrierR02:
    """Post-codec bitstream -> exact arithmetic matrix without entropy recoding."""
    if not isinstance(data, bytes) or not data:
        raise ValueError('nonempty bytes required')
    if width <= 0 or width > 65535:
        raise ValueError('invalid matrix width')
    padded = data + b'\0' * (-len(data) % width)
    return pack(np.frombuffer(padded, np.int8).reshape(-1,width), original_length=len(data), **kwargs)


def load(payload: bytes) -> NeuralCarrierR02:
    if not isinstance(payload, bytes) or len(payload) < _HEADER.size+_SHA_SIZE:
        raise ValueError('truncated ZipBurger R02 header')
    magic, m, n, k, original_length = _HEADER.unpack_from(payload)
    if magic not in (_GRAPH, _RAW) or not 0 < m <= 65535 or not 0 < n <= 65535:
        raise ValueError('bad ZipBurger magic/shape')
    if m*n > _MAX_ELEMENTS or len(payload) > _MAX_ELEMENTS*5 or not 0 < original_length <= m*n:
        raise ValueError('resource bounds/length')
    if (magic == _RAW and k) or k > min(m,128):
        raise ValueError('invalid dictionary size')
    digest = payload[_HEADER.size:_HEADER.size+_SHA_SIZE]
    pos = _HEADER.size + _SHA_SIZE
    if pos + k*n > len(payload):
        raise ValueError('truncated root table')
    bases = np.frombuffer(payload, np.int8, count=k*n, offset=pos).reshape(k,n).copy()
    pos += k*n
    indices = np.full((m,2), -1, np.int32)
    coeffs = np.zeros((m,2), np.int8)
    rr=[];cc=[];vv=[];literal=[];literals=bytearray()
    if magic == _RAW:
        if len(payload)-pos != m*n:
            raise ValueError('invalid raw frame size')
        literal = [True]*m
        literals.extend(payload[pos:])
    else:
        for i in range(m):
            if pos + _ROW.size > len(payload):
                raise ValueError('truncated row tag')
            tag, nlinks, count = _ROW.unpack_from(payload, pos)
            pos += _ROW.size
            if tag == 0:
                if nlinks or count or pos+n > len(payload):
                    raise ValueError('invalid literal row')
                literal.append(True)
                literals.extend(payload[pos:pos+n]); pos+=n
            elif tag == 1:
                literal.append(False)
                if not 1 <= nlinks <= 2 or pos + nlinks*_EDGE.size + count*_RES.size > len(payload) or count > n:
                    raise ValueError('invalid link row')
                used=set()
                for j in range(nlinks):
                    idx,coeff = _EDGE.unpack_from(payload,pos)
                    pos += _EDGE.size
                    if idx >= k or idx in used or coeff not in _COEFFICIENTS:
                        raise ValueError('invalid link')
                    used.add(idx)
                    indices[i,j]=idx
                    coeffs[i,j]=coeff
                previous = -1
                for _ in range(count):
                    col,delta=_RES.unpack_from(payload,pos);pos += _RES.size
                    if col >= n or col <= previous or delta == 0:
                        raise ValueError('invalid residual')
                    rr.append(i);cc.append(col);vv.append(delta)
                    previous=col
            else:
                raise ValueError('unknown row tag')
        if pos != len(payload):
            raise ValueError('trailing payload bytes')
    carrier = NeuralCarrierR02(payload,m,n,bases,indices,coeffs,
        np.asarray(rr,np.int32),np.asarray(cc,np.int32),np.asarray(vv,np.int16),
        np.asarray(literal,np.bool_),bytes(literals),digest,original_length)
    carrier.verify()
    return carrier


def make_torch_module(carrier: NeuralCarrierR02, *, rank: int = 2):
    """Trainable input/core/output network over the encoded graph, no W Parameter.

    The payload and root tensor are frozen. Adapter parameters are *additional*
    storage, never charged as compression savings. Float rounding may differ
    from a dense matmul; exact bitwise guarantees apply to integer dot() only.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    if not 1 <= rank <= 128:
        raise ValueError('rank must be between 1 and 128')

    class Adapter(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.down = nn.Linear(width, rank, bias=False)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, x):
            return x + self.up(torch.tanh(self.down(x)))

    class ZipBurgerNetR02(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_net = Adapter(carrier.cols)
            self.output_net = Adapter(carrier.rows)
            self.register_buffer('root_neurons', torch.from_numpy(carrier.bases.copy()))
            self.register_buffer('row_root_ids', torch.from_numpy(np.maximum(carrier.index,0).astype(np.int64)))
            self.register_buffer('row_root_coeff', torch.from_numpy(carrier.coeff.astype(np.float32)))
            literal_idx = np.flatnonzero(carrier.literal_row).astype(np.int64)
            literal_weight = np.frombuffer(carrier.literal_data,np.int8).reshape(-1,carrier.cols).copy()
            self.register_buffer('literal_indices',torch.from_numpy(literal_idx))
            self.register_buffer('literal_weights',torch.from_numpy(literal_weight))
            coo = np.vstack((carrier.residual_row, carrier.residual_col)).astype(np.int64)
            residuals = torch.sparse_coo_tensor(torch.from_numpy(coo),
                torch.from_numpy(carrier.residual.astype(np.float32)),
                (carrier.rows, carrier.cols)).coalesce()
            self.register_buffer('residual_connections', residuals)

        def forward(self, input_batch):
            if input_batch.ndim != 2 or input_batch.shape[1] != carrier.cols:
                raise ValueError('expected [batch, columns]')
            x = self.input_net(input_batch)
            if self.root_neurons.numel():
                activations = F.linear(x, self.root_neurons.to(x.dtype))
                y = x.new_zeros((x.shape[0],carrier.rows))
                for j in range(2):
                    y = y + activations[:,self.row_root_ids[:,j]] * self.row_root_coeff[:,j]
            else:
                y = x.new_zeros((x.shape[0],carrier.rows))
            if self.residual_connections._nnz():
                y = y + torch.sparse.mm(self.residual_connections, x.T).T
            if self.literal_indices.numel():
                literal_output = F.linear(x,self.literal_weights.to(x.dtype))
                y = y.index_copy(1,self.literal_indices,literal_output)
            return self.output_net(y)

    return ZipBurgerNetR02()
