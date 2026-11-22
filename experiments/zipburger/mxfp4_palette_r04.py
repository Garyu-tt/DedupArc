"""NeuroCrunch ZipBurger R04: exact scale-bitpacked MXFP4 carrier.

Research-only baseline targeting the *actual* scale entropy of GGUF MXFP4.
Keep the original 16 E2M1 nibble bytes per 32 weights, store E8M0 scales
using either a fixed-base bit-packed delta, a compact palette, or a literal
GGUF fallback. Scale and code reads are O(1). Direct GEMV decodes only
one block of 32 values at a time, never a dense floating-point weight matrix.

Not a production GGML kernel, not a compressed 20B checkpoint, not proof
of improved inference throughput. Input/output adapters are NOT included.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np

from .mxfp4_r03 import MANTISSAS

MAGIC = b"ZBP4"
HEADER = struct.Struct("<4sHHBBHI32s")
RAW, DELTA, PALETTE = 0, 1, 2
MAX_BLOCKS = 4096


def _bits_needed(values: int) -> int:
    return (values - 1).bit_length() if values > 1 else 0


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    seq = np.asarray(values, dtype=np.uint8).ravel()
    if not 0 <= bits <= 8:
        raise ValueError("invalid bit width")
    if bits == 0:
        if np.any(seq):
            raise ValueError("nonzero constant index")
        return b""
    if np.any(seq >= (1 << bits)) and bits < 8:
        raise ValueError("index outside packed width")
    result = bytearray((seq.size * bits + 7) // 8)
    for i, v in enumerate(seq):
        bit = i * bits
        off = bit >> 3
        shift = bit & 7
        result[off] |= (int(v) << shift) & 255
        if shift + bits > 8:
            result[off + 1] |= int(v) >> (8 - shift)
    return bytes(result)


def _get_bitpacked(encoded: bytes, i: int, bits: int) -> int:
    if bits == 0:
        return 0
    bit = i * bits
    off = bit >> 3
    shift = bit & 7
    x = encoded[off] >> shift
    if shift + bits > 8:
        x |= encoded[off + 1] << (8 - shift)
    return x & ((1 << bits) - 1)


def _validate_input(blocks, scales):
    b = np.asarray(blocks)
    s = np.asarray(scales)
    if b.dtype != np.uint8 or s.dtype != np.uint8 or b.ndim != 3 or s.ndim != 2:
        raise TypeError("expected uint8 blocks [rows,groups,16] and scales [rows,groups]")
    if b.shape[2] != 16 or b.shape[:2] != s.shape:
        raise ValueError("MXFP4 shape mismatch")
    rows, groups = s.shape
    if not 1 <= rows <= 2048 or not 1 <= groups <= 256 or rows * groups > MAX_BLOCKS:
        raise ValueError("R04 tile exceeds bounds")
    return np.ascontiguousarray(b), np.ascontiguousarray(s)


def _physical_bytes(blocks, scales):
    rows, groups = scales.shape
    array = np.empty((rows, groups, 17), np.uint8)
    array[:, :, 0] = scales
    array[:, :, 1:] = blocks
    return array.tobytes()


@dataclass(frozen=True)
class CarrierR04:
    payload: bytes
    rows: int
    groups: int
    encoding: int
    bits: int
    meta: bytes
    indexes: bytes
    data: bytes
    digest: bytes

    @property
    def mode(self) -> str:
        return {RAW: "raw-fallback", DELTA: "delta-bitpack",
                PALETTE: "palette-bitpack"}[self.encoding]

    @property
    def frame_bytes(self):
        return len(self.payload)

    @property
    def source_bytes(self):
        return self.rows * self.groups * 17

    def _position(self, row, group):
        if type(row) is not int or type(group) is not int or not (
                0 <= row < self.rows and 0 <= group < self.groups):
            raise IndexError((row, group))
        return row * self.groups + group

    def read_scale(self, row, group) -> int:
        i = self._position(row, group)
        if self.encoding == RAW:
            return self.data[i * 17]
        code = _get_bitpacked(self.indexes, i, self.bits)
        if self.encoding == DELTA:
            value = self.meta[0] + code
            if value > 255:
                raise ValueError("scale delta overflow")
            return value
        if code >= len(self.meta):
            raise ValueError("palette index out of range")
        return self.meta[code]

    def _nibbles(self, row, group):
        i = self._position(row, group)
        if self.encoding == RAW:
            return self.data[i * 17 + 1:i * 17 + 17]
        return self.data[i * 16:i * 16 + 16]

    def read_code(self, row, col) -> int:
        if type(col) is not int or not 0 <= col < self.groups * 32:
            raise IndexError((row, col))
        group, local = divmod(col, 32)
        v = self._nibbles(row, group)[local // 2]
        return (v & 15) if local % 2 == 0 else (v >> 4)

    def stream_original(self):
        for row in range(self.rows):
            for group in range(self.groups):
                yield bytes([self.read_scale(row, group)]) + self._nibbles(row, group)

    def verify(self):
        h = hashlib.sha256()
        for chunk in self.stream_original():
            h.update(chunk)
        if h.digest() != self.digest:
            raise ValueError("R04 original GGUF bit-pattern digest mismatch")

    def restore(self):
        """Explicit export only. Not used by direct dot or integrity verification."""
        b = np.empty((self.rows, self.groups, 16), np.uint8)
        s = np.empty((self.rows, self.groups), np.uint8)
        for i in range(self.rows):
            for g in range(self.groups):
                b[i, g] = np.frombuffer(self._nibbles(i, g), np.uint8)
                s[i, g] = self.read_scale(i, g)
        return b, s

    def dot(self, x):
        v = np.asarray(x)
        if (v.ndim != 1 or v.size != self.groups * 32
                or not np.issubdtype(v.dtype, np.number)
                or not np.all(np.isfinite(v))):
            raise ValueError("invalid float64 GEMV input")
        v = v.astype(np.float64)
        y = np.zeros(self.rows, np.float64)
        for row in range(self.rows):
            for g in range(self.groups):
                p = np.frombuffer(self._nibbles(row, g), np.uint8)
                codes = np.empty(32, np.uint8)
                codes[0::2], codes[1::2] = p & 15, p >> 4
                mantissas = MANTISSAS[codes].astype(np.float64)
                scale = self.read_scale(row, g)
                y[row] += (mantissas @ v[g * 32:(g + 1) * 32]) * (
                    np.ldexp(0.5, scale - 127))
        return y


def _frame(rows, groups, mode, bits, meta, indexes, data, digest):
    return (HEADER.pack(MAGIC, rows, groups, mode, bits, len(meta),
                        len(indexes), digest) + meta + indexes + data)


def pack(blocks, scales):
    b, s = _validate_input(blocks, scales)
    rows, groups = s.shape
    raw = _physical_bytes(b, s)
    digest = hashlib.sha256(raw).digest()
    best = _frame(rows, groups, RAW, 0, b"", b"", raw, digest)
    nibbles = b.tobytes()
    scale_flat = s.ravel()
    lo, hi = int(scale_flat.min()), int(scale_flat.max())

    # Delta encoding performs direct O(1) scale lookup without a palette.
    delta_bits = _bits_needed(hi - lo + 1)
    delta = (scale_flat.astype(np.int16) - lo).astype(np.uint8)
    delta_indexes = _pack_bits(delta, delta_bits)
    candidate = _frame(rows, groups, DELTA, delta_bits,
                       bytes([lo]), delta_indexes, nibbles, digest)
    if len(candidate) < len(best):
        best = candidate

    # Palette mode handles widely separated rare scale codes more compactly.
    palette, codes = np.unique(scale_flat, return_inverse=True)
    palette_bits = _bits_needed(palette.size)
    palette_indexes = _pack_bits(codes.astype(np.uint8), palette_bits)
    candidate = _frame(rows, groups, PALETTE, palette_bits,
                       palette.tobytes(), palette_indexes, nibbles, digest)
    if len(candidate) < len(best):
        best = candidate

    return load(best)


def load(payload: bytes):
    if not isinstance(payload, bytes) or len(payload) < HEADER.size:
        raise ValueError("truncated R04 frame")
    magic, rows, groups, mode, bits, metalen, ixlen, digest = HEADER.unpack_from(payload)
    if magic != MAGIC or mode not in (RAW, DELTA, PALETTE):
        raise ValueError("bad R04 magic/encoding")
    if not 1 <= rows <= 2048 or not 1 <= groups <= 256 or rows * groups > MAX_BLOCKS:
        raise ValueError("R04 frame bounds")
    if not 0 <= bits <= 8 or metalen > 256:
        raise ValueError("invalid R04 index width/metadata")
    n = rows * groups
    expected_indices = (n * bits + 7) // 8
    if mode == RAW:
        if bits or metalen or ixlen or len(payload) != HEADER.size + n * 17:
            raise ValueError("invalid raw R04 section lengths")
    else:
        if ixlen != expected_indices:
            raise ValueError("invalid bit-packed index length")
        if mode == DELTA and metalen != 1:
            raise ValueError("invalid delta metadata")
        if mode == PALETTE and (not 1 <= metalen <= 256
                                or bits != _bits_needed(metalen)):
            raise ValueError("invalid palette metadata")
        if len(payload) != HEADER.size + metalen + ixlen + n * 16:
            raise ValueError("invalid compressed R04 section lengths")
    start = HEADER.size
    meta = payload[start:start + metalen]
    indexes = payload[start + metalen:start + metalen + ixlen]
    data = payload[start + metalen + ixlen:]
    # Verified per-block scale reads reject actual overflow; spare codes in
    # the fixed-width bitplane need not be representable uint8 values.
    if mode != RAW and bits and (n * bits) % 8:
        spare = 8 - ((n * bits) % 8)
        if indexes[-1] >> (8 - spare):
            raise ValueError("noncanonical index padding")
    carrier = CarrierR04(payload, rows, groups, mode, bits,
                         meta, indexes, data, digest)
    carrier.verify()
    return carrier
