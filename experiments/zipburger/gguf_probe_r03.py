"""Bounded, read-only GGUF MXFP4 slices for NeuroCrunch ZipBurger R03.

GGML's block_mxfp4 is 17 bytes: one E8M0 scale, then 16 packed
low-nibble-first E2M1 bytes. No full checkpoint mmap, model execution, GPU
access or writes. Research-only; results cannot establish an LLM speedup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import struct
import time
import zlib

import numpy as np
from .mxfp4_r03 import dense_reference_dot, load, pack

TYPE_MXFP4 = 39
SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
           10: 8, 11: 8, 12: 8}
MAX_HEADER = 128 * 1024 * 1024


class GGUFProbeError(ValueError):
    pass


class GGUFIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            self.size = os.fstat(self.file.fileno()).st_size
            self.limit = min(MAX_HEADER, self.size)
            self.tensors = {}
            if self._get(4) != b"GGUF" or self._u32() != 3:
                raise GGUFProbeError("unsupported GGUF header")
            nt, nk = self._u64(), self._u64()
            if nt > 50000 or nk > 100000:
                raise GGUFProbeError("too many header entries")
            align = 32
            for _ in range(nk):
                key, typ = self._string(), self._u32()
                if key == "general.alignment" and typ == 4:
                    align = self._u32()
                else:
                    self._skipvalue(typ)
            if not 0 < align <= 4096 or align & (align - 1):
                raise GGUFProbeError("bad GGUF alignment")
            for _ in range(nt):
                name, nd = self._string(), self._u32()
                if not 1 <= nd <= 8:
                    raise GGUFProbeError("invalid dimensions")
                dims = tuple(self._u64() for _ in range(nd))
                if any(n == 0 or n > 1000000000 for n in dims):
                    raise GGUFProbeError("invalid shape")
                typ, offset = self._u32(), self._u64()
                if name in self.tensors:
                    raise GGUFProbeError("duplicate tensor")
                self.tensors[name] = (typ, dims, offset)
            self.data_start = (self.file.tell() + align - 1) // align * align
            if self.data_start >= self.size:
                raise GGUFProbeError("missing data section")
        except BaseException:
            self.file.close()
            raise

    def _get(self, n):
        if n < 0 or n > self.limit - self.file.tell():
            raise GGUFProbeError("out-of-bounds GGUF metadata")
        b = self.file.read(n)
        if len(b) != n:
            raise GGUFProbeError("truncated header")
        return b

    def _skip(self, n):
        if n < 0 or n > self.limit - self.file.tell():
            raise GGUFProbeError("out-of-bounds metadata skip")
        self.file.seek(n, os.SEEK_CUR)

    def _u32(self):
        return struct.unpack("<I", self._get(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._get(8))[0]

    def _string(self):
        n = self._u64()
        if n > 65536:
            raise GGUFProbeError("too-long GGUF string")
        return self._get(n).decode("utf-8")

    def _skipvalue(self, typ):
        if typ in SCALARS:
            self._skip(SCALARS[typ])
        elif typ == 8:
            self._skip(self._u64())
        elif typ == 9:
            subtyp, n = self._u32(), self._u64()
            if n > 3000000:
                raise GGUFProbeError("oversize metadata array")
            if subtyp in SCALARS:
                self._skip(n * SCALARS[subtyp])
            elif subtyp == 8:
                for _ in range(n):
                    self._skip(self._u64())
            else:
                raise GGUFProbeError("unknown array type")
        else:
            raise GGUFProbeError("unknown metadata type")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()

    def sample(self, name, rows=48, groups=4, row_start=0):
        if (type(rows) is not int or type(groups) is not int or
            type(row_start) is not int or not 1 <= rows <= 128 or
            not 1 <= groups <= 16 or row_start < 0):
            raise GGUFProbeError("bounded slice required")
        info = self.tensors.get(name)
        if info is None or info[0] != TYPE_MXFP4:
            raise GGUFProbeError("tensor missing or not MXFP4")
        _, dims, offset = info
        if dims[0] % 32:
            raise GGUFProbeError("MXFP4 width not divisible by 32")
        total_groups, total_rows = dims[0] // 32, math.prod(dims[1:])
        if groups > total_groups or row_start + rows > total_rows:
            raise GGUFProbeError("slice exceeds tensor")
        base = self.data_start + offset
        if base < self.data_start or base + total_rows * total_groups * 17 > self.size:
            raise GGUFProbeError("tensor offset invalid")
        raw = bytearray()
        for r in range(row_start, row_start + rows):
            self.file.seek(base + r * total_groups * 17)
            row = self.file.read(groups * 17)
            if len(row) != groups * 17:
                raise GGUFProbeError("short tensor row")
            raw.extend(row)
        x = np.frombuffer(raw, np.uint8).reshape(rows, groups, 17)
        return np.ascontiguousarray(x[:, :, 1:]), np.ascontiguousarray(x[:, :, 0]), bytes(raw), dims


def evaluate(index, tensor, rows=48, groups=4, row_start=0):
    b, s, original, dims = index.sample(tensor, rows, groups, row_start)
    begin = time.perf_counter()
    carrier = pack(b, s, max_roots=8, max_candidates=16)
    pack_ms = (time.perf_counter() - begin) * 1000
    bb, ss = load(carrier.payload).restore()
    interleaved = np.empty((rows, groups, 17), np.uint8)
    interleaved[:, :, 0], interleaved[:, :, 1:] = ss, bb
    if interleaved.tobytes() != original:
        raise AssertionError("exact GGUF packed-bit reconstruction failed")
    v = np.random.default_rng(331).standard_normal(groups * 32)
    begin = time.perf_counter()
    direct = carrier.dot(v)
    dot_ms = (time.perf_counter() - begin) * 1000
    np.testing.assert_allclose(direct, dense_reference_dot(b, s, v),
                               rtol=1e-12, atol=1e-8)
    return dict(tensor=tensor, shape=dims, rows=rows, groups=groups,
                row_start=row_start, source_bytes=len(original),
                source_sha256=hashlib.sha256(original).hexdigest(),
                zipburger_bytes=carrier.frame_bytes,
                zipburger_mode=carrier.mode,
                zlib9_bytes=len(zlib.compress(original, 9)),
                xz6_bytes=len(lzma.compress(original, preset=6)),
                exact_gguf_roundtrip=True, direct_float64_match=True,
                pack_ms=round(pack_ms, 3), direct_dot_ms=round(dot_ms, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--tensor", action="append")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--row-start", type=int, default=0)
    a = ap.parse_args(argv)
    with GGUFIndex(a.gguf) as index:
        names = a.tensor or [k for k, v in index.tensors.items()
                             if v[0] == TYPE_MXFP4][:3]
        results = [evaluate(index, n, a.rows, a.groups, a.row_start)
                   for n in names]
        print(json.dumps(dict(schema="zipburger-real-gguf-r03/1",
                              gguf_file=a.gguf.name,
                              tensor_count=len(index.tensors),
                              mxfp4_tensor_count=sum(v[0] == TYPE_MXFP4
                                                     for v in index.tensors.values()),
                              checkpoint_unchanged=True,
                              full_model_inference=False, results=results),
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
