"""R07: lossless-structure diagnostics for real ALT 120B MXFP4 slices.

Shannon rates describe single-symbol distributions of bitpacked bytes, NOT
guaranteed achievable coding rates, CPU inference savings, or model-level
entropy bounds. Profiling is read-only and never constructs dense floats.
"""
from __future__ import annotations

from collections import Counter
import math
import zlib

import numpy as np


def _entropy(values):
    data = np.asarray(values,dtype=np.uint8).reshape(-1)
    if data.size == 0:
        raise ValueError("empty entropy sample")
    _,counts=np.unique(data,return_counts=True)
    probs=counts/counts.sum()
    return float(-(probs*np.log2(probs)).sum())


def profile(blocks, scales):
    """Statistics of exact 17-byte GGUF blocks, no floating-point matrices."""
    b=np.asarray(blocks)
    s=np.asarray(scales)
    if b.dtype!=np.uint8 or s.dtype!=np.uint8 or b.ndim!=3 or s.ndim!=2:
        raise TypeError("uint8 [rows,groups,16] / uint8 [rows,groups] required")
    if b.shape[-1]!=16 or b.shape[:2]!=s.shape or b.shape[0]==0:
        raise ValueError("invalid MXFP4 block shapes")
    packed=b.reshape(-1,16)
    low=b&15
    high=b>>4
    nibbles=np.stack((low,high),axis=-1).reshape(-1)
    flat_scale=s.reshape(-1)
    exact_blocks=[v.tobytes() for v in packed]
    unique_blocks=len(set(exact_blocks))
    repeated=len(exact_blocks)-unique_blocks
    adjacent={}
    for stride in (1,2,3,4,8):
        if len(exact_blocks)<=stride:
            adjacent[stride]=0
        else:
            adjacent[stride]=sum(x==y for x,y in zip(
                exact_blocks[stride:],exact_blocks[:-stride]))
    flat=b.reshape(-1).copy()
    xored={}
    for stride in (1,2,3,4):
        if packed.shape[0] <= stride:
            xored[stride]=None
            continue
        residual=np.bitwise_xor(packed[stride:],packed[:-stride]).tobytes()
        xored[stride]={"residual_bytes":len(residual),
                       "zlib9_bytes":len(zlib.compress(residual,9))}
    scale_entropy=_entropy(flat_scale)
    nibble_entropy=_entropy(nibbles)
    scale_symbols=len(flat_scale)
    nibble_symbols=len(nibbles)
    symbolic_lower_bytes=math.ceil(
        (scale_entropy*scale_symbols+nibble_entropy*nibble_symbols)/8)
    return {
        "scale_symbols":scale_symbols,
        "scale_unique":int(np.unique(flat_scale).size),
        "scale_symbol_entropy_bits":round(scale_entropy,5),
        "scale_literal_bytes":int(s.nbytes),
        "scale_zlib9_bytes":len(zlib.compress(flat_scale.tobytes(),9)),
        "nibble_symbols":nibble_symbols,
        "nibble_symbol_entropy_bits":round(nibble_entropy,5),
        "nibble_literal_bytes":int(b.nbytes),
        "nibble_zlib9_bytes":len(zlib.compress(flat.tobytes(),9)),
        "mantissa_unique_16byte_blocks":unique_blocks,
        "mantissa_exact_duplicate_blocks":repeated,
        "mantissa_strided_equal_blocks":adjacent,
        "mantissa_xor_stride_zlib9":xored,
        "independent_symbol_shannon_bytes_no_index_or_decoder":symbolic_lower_bytes,
        "single_symbol_entropy_is_not_full_information_bound":True,
    }
