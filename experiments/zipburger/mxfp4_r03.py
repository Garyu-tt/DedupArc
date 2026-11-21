"""ZipBurger R03: lossless MXFP4 neural carrier with direct sparse-graph matvec.

Research only. Uses the R02 exact signed-int8 graph for *numerical* half-unit
E2M1 mantissas, another graph for original E8M0 scale bytes, and a tightly
packed sideband for IEEE negative zero nibble codes. Unlike a byte-graph, the
mantissa graph can be evaluated directly, before applying each block scale.

Reference model: openai/gpt-oss/gpt_oss/torch/weights.py: 16 packed bytes per
32 mantissas; low nibble first; E8M0 exponent is scale_uint8 - 127.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from . import multiroot_r02 as graph

MAGIC = b'ZBM3'
HEADER = struct.Struct('<4sIIIIIB32s')  # magic, rows, groups, len(M), len(S), zeros, sign_mode, digest
RAW_HEADER = struct.Struct('<4sII32s')     # magic ZBR3, rows, groups, sha256
RAW_MAGIC = b'ZBR3'

MANTISSAS = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int8)
MAG_TO_CODE = {0:0, 1:1,2:2,3:3,4:4,6:5,8:6,12:7,
               -1:9,-2:10,-3:11,-4:12,-6:13,-8:14,-12:15}
MAX_MANTISSAS = 2_097_152


def _check_inputs(blocks: np.ndarray, scales: np.ndarray):
    b = np.asarray(blocks)
    s = np.asarray(scales)
    if b.dtype != np.uint8 or s.dtype != np.uint8 or b.ndim != 3 or s.ndim != 2:
        raise TypeError('expected uint8 blocks [rows,groups,16] and scales [rows,groups]')
    if b.shape[2] != 16 or b.shape[:2] != s.shape or not b.shape[0] or not b.shape[1]:
        raise ValueError('invalid MXFP4 blocks/scales shape')
    if b.shape[0] > 65535 or b.shape[1] > 2048 or b.size*2 > MAX_MANTISSAS:
        raise ValueError('MXFP4 R03 research sample exceeds bounds')
    return np.ascontiguousarray(b), np.ascontiguousarray(s)


def _unpack_codes(b: np.ndarray) -> np.ndarray:
    c = np.empty((*b.shape[:2],32), dtype=np.uint8)
    c[...,0::2] = b & 15
    c[...,1::2] = b >> 4
    return c.reshape(b.shape[0], -1)


def _pack_mantissa_row(m: np.ndarray, zero_neg: np.ndarray) -> bytes:
    """Reconstruct original nibble codes, including the sign of zero."""
    try:
        codes = np.fromiter((MAG_TO_CODE[int(x)] for x in m), dtype=np.uint8, count=m.size)
    except KeyError as exc:
        raise ValueError('invalid E2M1 mantissa in graph payload') from exc
    if zero_neg.size != np.count_nonzero(m == 0):
        raise ValueError('negative-zero sideband count mismatch')
    codes[m == 0] = zero_neg.astype(np.uint8) * 8
    return ((codes[::2] | (codes[1::2] << 4)).astype(np.uint8)).tobytes()


@dataclass(frozen=True)
class Mxfp4CarrierR03:
    payload: bytes
    rows: int
    groups: int
    mantissa: graph.NeuralCarrierR02
    scales: graph.NeuralCarrierR02
    zero_signs: bytes
    zero_count: int
    sign_mode: int  # 0 all +0, 1 all -0, 2 packed mixed sign bits
    digest: bytes

    @property
    def mode(self) -> str:
        return 'neural-graph'

    @property
    def frame_bytes(self) -> int:
        return len(self.payload)

    @property
    def source_bytes(self) -> int:
        return self.rows * self.groups * 17  # 16 nibble bytes + 1 scale byte per block

    def _zero_sign_row(self, start: int, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.uint8)
        if self.sign_mode < 2:
            return np.full(count, self.sign_mode,dtype=np.uint8)
        # Decode only the required bit range, not the full sideband per row.
        first, last = start // 8, (start+count+7)//8
        bits = np.unpackbits(np.frombuffer(self.zero_signs[first:last],dtype=np.uint8),bitorder='big')
        return bits[start%8:start%8+count]

    def stream_original(self):
        """Yield one original packed row and one E8M0 scale row; never dense W."""
        zero_cursor = 0
        for r in range(self.rows):
            mant = self.mantissa._row(r)
            count = int(np.count_nonzero(mant == 0))
            signs = self._zero_sign_row(zero_cursor, count)
            zero_cursor += count
            yield _pack_mantissa_row(mant, signs), self.scales._row(r).view(np.uint8).tobytes()
        if zero_cursor != self.zero_count:
            raise ValueError('unused/insufficient zero-sideband bits')

    def verify(self):
        hasher = hashlib.sha256()
        for packed, scales in self.stream_original():
            hasher.update(packed)
            hasher.update(scales)
        if hasher.digest() != self.digest:
            raise ValueError('MXFP4 original bit-pattern SHA256 mismatch')

    def restore(self) -> tuple[np.ndarray, np.ndarray]:
        """Explicit checkpoint export only, not used by forward()."""
        blocks = np.empty((self.rows,self.groups,16), dtype=np.uint8)
        scales = np.empty((self.rows,self.groups), dtype=np.uint8)
        for r,(packed,scale_row) in enumerate(self.stream_original()):
            blocks[r] = np.frombuffer(packed,np.uint8).reshape(self.groups,16)
            scales[r] = np.frombuffer(scale_row,np.uint8)
        return blocks,scales

    def read_code(self, row: int, col: int) -> int:
        if not 0 <= row < self.rows or not 0 <= col < self.groups*32:
            raise IndexError((row,col))
        val = self.mantissa.read_cell(row,col)
        if val != 0:
            return MAG_TO_CODE[val]
        cursor = 0
        for i in range(row):
            cursor += int(np.count_nonzero(self.mantissa._row(i) == 0))
        cursor += int(np.count_nonzero(self.mantissa._row(row)[:col] == 0))
        return 8 if self._zero_sign_row(cursor,1)[0] else 0

    def dot(self, x: np.ndarray) -> np.ndarray:
        """Forward float64 GEMV without full dequantized weight matrix.

        Each shared basis vector is activated once per 32-value group;
        row links and sparse residuals are applied in the compressed graph.
        This changes float summation order, so numerical results use tolerance,
        not bitwise identity, even while packed weights restore bit-perfectly.
        """
        x = np.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.groups*32 or not np.issubdtype(x.dtype,np.number):
            raise ValueError('expected numerical input vector of logical input width')
        if not np.all(np.isfinite(x)):
            raise ValueError('nonfinite input vector')
        v = x.astype(np.float64)
        c = self.mantissa
        out = np.zeros(self.rows, dtype=np.float64)
        literal = np.frombuffer(c.literal_data, np.int8).reshape(-1, c.cols)
        literal_idx = np.flatnonzero(c.literal_row)
        for g in range(self.groups):
            lo,hi = 32*g,32*(g+1)
            vec = v[lo:hi]
            group = np.zeros(self.rows,dtype=np.float64)
            if c.bases.shape[0]:
                activated = c.bases[:,lo:hi].astype(np.float64) @ vec
                for link in range(2):
                    present = c.index[:,link] >= 0
                    group[present] += activated[c.index[present,link]] * c.coeff[present,link]
            if c.residuals:
                selected = (c.residual_col >= lo) & (c.residual_col < hi)
                np.add.at(group,c.residual_row[selected],
                    c.residual[selected].astype(np.float64) * v[c.residual_col[selected]])
            if literal_idx.size:
                group[literal_idx] = literal[:,lo:hi].astype(np.float64) @ vec
            scales = np.fromiter((int(self.scales.read_cell(i,g)) & 255
                                  for i in range(self.rows)), dtype=np.int16,count=self.rows)
            # E8M0 exponent exactly follows the official openai/gpt-oss torch reference.
            group *= np.ldexp(np.full(self.rows,0.5,dtype=np.float64),scales - 127)
            out += group
        return out


@dataclass(frozen=True)
class RawMxfp4CarrierR03:
    """Non-expanding container fallback; decode one 32-value group at a time."""
    payload: bytes
    rows: int
    groups: int
    digest: bytes

    @property
    def mode(self) -> str:
        return 'raw-fallback'

    @property
    def frame_bytes(self) -> int:
        return len(self.payload)

    @property
    def source_bytes(self) -> int:
        return self.rows*self.groups*17

    @property
    def mantissa(self):
        return None

    @property
    def scales(self):
        return None

    @property
    def zero_count(self):
        count=0
        for packed,_ in self.stream_original():
            block=np.frombuffer(packed,dtype=np.uint8)
            count += int(np.count_nonzero((block & 7)==0))
            count += int(np.count_nonzero(((block>>4)&7)==0))
        return count  # signs are present in original nibbles, no extra sideband

    @classmethod
    def from_payload(cls, payload: bytes):
        if len(payload) < RAW_HEADER.size:
            raise ValueError('truncated MXFP4 raw frame')
        magic,rows,groups,digest=RAW_HEADER.unpack_from(payload)
        if magic != RAW_MAGIC or not 0 < rows <= 65535 or not 0 < groups <= 2048:
            raise ValueError('invalid MXFP4 raw header')
        if rows*groups*32 > MAX_MANTISSAS or len(payload) != RAW_HEADER.size + rows*groups*17:
            raise ValueError('MXFP4 raw shape/resource bounds')
        carrier=cls(payload,rows,groups,digest)
        carrier.verify()
        return carrier

    def _row(self,row):
        if not 0<=row<self.rows:
            raise IndexError(row)
        offset=RAW_HEADER.size+row*self.groups*17
        return (self.payload[offset:offset+self.groups*16],
                self.payload[offset+self.groups*16:offset+self.groups*17])

    def stream_original(self):
        for i in range(self.rows):
            yield self._row(i)

    def verify(self):
        h=hashlib.sha256()
        for b,s in self.stream_original():
            h.update(b);h.update(s)
        if h.digest()!=self.digest:
            raise ValueError('MXFP4 raw frame SHA256 mismatch')

    def restore(self):
        b=np.empty((self.rows,self.groups,16),np.uint8)
        s=np.empty((self.rows,self.groups),np.uint8)
        for i,(bi,si) in enumerate(self.stream_original()):
            b[i]=np.frombuffer(bi,np.uint8).reshape(self.groups,16)
            s[i]=np.frombuffer(si,np.uint8)
        return b,s

    def read_code(self,row,col):
        if not 0<=col<self.groups*32:
            raise IndexError((row,col))
        b,_=self._row(row)
        v=b[col//2]
        return (v&15) if col%2==0 else (v>>4)

    def dot(self,x):
        v=np.asarray(x)
        if v.ndim!=1 or v.size!=self.groups*32 or not np.issubdtype(v.dtype,np.number):
            raise ValueError('expected numerical input vector of logical input width')
        if not np.all(np.isfinite(v)):
            raise ValueError('nonfinite input vector')
        v=v.astype(np.float64)
        out=np.zeros(self.rows,dtype=np.float64)
        for i,(packed,scale_bytes) in enumerate(self.stream_original()):
            b=np.frombuffer(packed,dtype=np.uint8).reshape(self.groups,16)
            s=np.frombuffer(scale_bytes,dtype=np.uint8)
            for g in range(self.groups):
                sub=b[g]
                codes=np.empty(32,dtype=np.uint8)
                codes[0::2]=sub&15
                codes[1::2]=sub>>4
                out[i] += (MANTISSAS[codes].astype(np.float64) @ v[g*32:(g+1)*32]) * np.ldexp(0.5,int(s[g])-127)
        return out


def _make_raw_frame(b,s):
    body=b''.join(p.tobytes()+sc.tobytes() for p,sc in zip(b.reshape(b.shape[0],-1),s))
    return RAW_HEADER.pack(RAW_MAGIC,b.shape[0],b.shape[1],hashlib.sha256(body).digest())+body


def pack(blocks: np.ndarray, scales: np.ndarray, *, max_roots=16, max_candidates=48,
         scale_roots=8) -> Mxfp4CarrierR03 | RawMxfp4CarrierR03:
    b,s = _check_inputs(blocks,scales)
    codes = _unpack_codes(b)
    mantissas = MANTISSAS[codes]
    zeros = codes[mantissas == 0]
    if len(zeros)==0 or np.all(zeros == 0):
        sign_mode, sign_bits = 0, b''
    elif np.all(zeros == 8):
        sign_mode, sign_bits = 1, b''
    else:
        sign_mode = 2
        sign_bits = np.packbits((zeros == 8).astype(np.uint8),bitorder='big').tobytes()
    m = graph.pack(mantissas,max_roots=max_roots,max_candidates=max_candidates)
    # Store E8M0 exponent bytes losslessly in a second neuron graph. View as
    # signed int8 for the R02 arithmetic storage, never numerical casting.
    s_graph = graph.pack(s.view(np.int8),max_roots=scale_roots,max_candidates=max_candidates)
    header = HEADER.pack(MAGIC,b.shape[0],b.shape[1],len(m.payload),len(s_graph.payload),
                         len(zeros),sign_mode,_digest(b,s))
    frame = header + m.payload + s_graph.payload + sign_bits
    raw = _make_raw_frame(b,s)
    return load(frame if len(frame) < len(raw) else raw)


def _digest(blocks, scales):
    h = hashlib.sha256()
    for packed, s in zip(blocks.reshape(blocks.shape[0],-1),scales):
        h.update(packed.tobytes());h.update(s.tobytes())
    return h.digest()


def load(payload: bytes) -> Mxfp4CarrierR03 | RawMxfp4CarrierR03:
    if not isinstance(payload,bytes) or len(payload)<RAW_HEADER.size:
        raise ValueError('truncated MXFP4 frame')
    if payload[:4] == RAW_MAGIC:
        return RawMxfp4CarrierR03.from_payload(payload)
    if len(payload)<HEADER.size:
        raise ValueError('truncated MXFP4 graph frame')
    magic, rows, groups, mlen, slen, zeros, sign_mode, digest = HEADER.unpack_from(payload)
    if magic != MAGIC or not 0 < rows <= 65535 or not 0 < groups <= 2048:
        raise ValueError('bad MXFP4 magic/dimensions')
    if rows*groups*32 > MAX_MANTISSAS or zeros > rows*groups*32:
        raise ValueError('MXFP4 frame resource bounds')
    if sign_mode not in (0,1,2):
        raise ValueError('unsupported zero sign encoding')
    signbytes = (zeros+7)//8 if sign_mode == 2 else 0
    if (mlen < 56 or slen < 56 or len(payload) != HEADER.size+mlen+slen+signbytes):
        raise ValueError('bad frame section lengths')
    pos=HEADER.size
    m=graph.load(payload[pos:pos+mlen]);pos+=mlen
    s=graph.load(payload[pos:pos+slen]);pos+=slen
    if (m.rows,m.cols)!=(rows,groups*32) or (s.rows,s.cols)!=(rows,groups):
        raise ValueError('MXFP4 carrier tensor shape mismatch')
    signs=payload[pos:]
    if sign_mode == 2 and zeros%8 and (signs[-1]&((1<<(8-(zeros%8)))-1)):
        raise ValueError('noncanonical negative zero padding')
    carrier=Mxfp4CarrierR03(payload,rows,groups,m,s,signs,zeros,sign_mode,digest)
    carrier.verify()
    return carrier


def dense_reference_dot(blocks: np.ndarray,scales: np.ndarray,x: np.ndarray):
    """Deliberately materialize dense float64 ONLY for test oracle."""
    b,s=_check_inputs(blocks,scales)
    c=_unpack_codes(b)
    f=MANTISSAS[c].astype(np.float64)*0.5
    f=f.reshape(b.shape[0],b.shape[1],32)
    f*=np.ldexp(np.ones_like(s,dtype=np.float64),s.astype(np.int16)-127)[...,None]
    return f.reshape(b.shape[0],-1) @ np.asarray(x,dtype=np.float64)


def make_torch_module(carrier: Mxfp4CarrierR03, rank:int=2):
    """Differentiable input/core/output graph; no dequantized full-W parameter."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    if not isinstance(carrier,Mxfp4CarrierR03):
        raise ValueError('Neural adapters require a graph frame; input selected raw fallback')
    if not 1 <= rank <= 64:
        raise ValueError('invalid adapter rank')

    class Adapter(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.down=nn.Linear(width,rank,bias=False)
            self.up=nn.Linear(rank,width,bias=False)
            nn.init.zeros_(self.up.weight)
        def forward(self,x):
            return x + self.up(torch.tanh(self.down(x)))

    class Mxfp4Net(nn.Module):
        def __init__(self):
            super().__init__()
            c=carrier.mantissa
            self.input_net=Adapter(c.cols)
            self.output_net=Adapter(c.rows)
            self.register_buffer('bases',torch.from_numpy(c.bases.copy()))
            self.register_buffer('root_index',torch.from_numpy(np.maximum(c.index,0).astype(np.int64)))
            self.register_buffer('coeff',torch.from_numpy(c.coeff.astype(np.float32)))
            self.register_buffer('literal_ids',torch.from_numpy(np.flatnonzero(c.literal_row).astype(np.int64)))
            literal=np.frombuffer(c.literal_data,np.int8).reshape(-1,c.cols).copy()
            self.register_buffer('literal_vals',torch.from_numpy(literal))
            coo=np.stack((c.residual_row,c.residual_col)).astype(np.int64)
            self.register_buffer('residual_rows',torch.from_numpy(coo[0].copy()))
            self.register_buffer('residual_cols',torch.from_numpy(coo[1].copy()))
            self.register_buffer('residual_vals',torch.from_numpy(c.residual.astype(np.float32)))
            scale_rows=np.stack([carrier.scales._row(i).view(np.uint8) for i in range(c.rows)])
            scales=torch.from_numpy(scale_rows.astype(np.int16)-127)
            self.register_buffer('scales',torch.ldexp(torch.full((c.rows,carrier.groups),0.5),scales.to(torch.int32)))
        def forward(self, x):
            if x.ndim!=2 or x.shape[1]!=carrier.groups*32:
                raise ValueError('expected [batch,groups*32]')
            x=self.input_net(x)
            y=x.new_zeros((x.shape[0],carrier.rows))
            for g in range(carrier.groups):
                lo,hi=g*32,(g+1)*32
                v=x[:,lo:hi]
                chunk=x.new_zeros((x.shape[0],carrier.rows))
                if self.bases.shape[0]:
                    z=F.linear(v,self.bases[:,lo:hi].to(v.dtype))
                    for j in range(2):
                        chunk=chunk+z[:,self.root_index[:,j]]*self.coeff[:,j]
                if self.residual_vals.numel():
                    chosen=(self.residual_cols>=lo)&(self.residual_cols<hi)
                    ids=torch.stack((self.residual_rows[chosen],self.residual_cols[chosen]-lo))
                    coo=torch.sparse_coo_tensor(ids,self.residual_vals[chosen].to(v.dtype),
                        (carrier.rows,32)).coalesce()
                    chunk=chunk+torch.sparse.mm(coo,v.T).T
                if self.literal_ids.numel():
                    literals=F.linear(v,self.literal_vals[:,lo:hi].to(v.dtype))
                    chunk=chunk.index_copy(1,self.literal_ids,literals)
                y=y+chunk*self.scales[:,g].to(v.dtype)
            return self.output_net(y)
    return Mxfp4Net()
