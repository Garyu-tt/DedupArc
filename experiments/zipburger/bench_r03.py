"""Deterministic R03 lab: MXFP4 E2M1 + E8M0 exact frame, graph-based GEMV.

python -m experiments.zipburger.bench_r03
Optional real checkpoint sample: python -m experiments.zipburger.bench_r03 --model-dir /path/gpt-oss-20b
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import zlib
import lzma

import numpy as np
from .mxfp4_r03 import pack, dense_reference_dot


def cases():
    rng=np.random.default_rng(2026)
    rows,groups=48,4
    patterns=rng.integers(0,256,(4,groups,16),dtype=np.uint8)
    b=patterns[np.arange(rows)%4].copy()
    s=rng.integers(116,134,(4,groups),dtype=np.uint8)[np.arange(rows)%4].copy()
    yield 'repeated-E2M1-E8M0',b,s
    b2=b.copy()
    for row in range(rows):
        if row%5==0:
            b2[row,0,0] ^= 8  # negative zero/sign changes, always bit-exact
        if row%7==0:
            b2[row,1,10] ^= 3
    yield 'repeated-with-exceptions',b2,s
    yield 'random-E2M1-and-E8M0',rng.integers(0,256,(rows,groups,16),dtype=np.uint8),rng.integers(116,135,(rows,groups),dtype=np.uint8)
    yield 'all-zero-positive',np.zeros((rows,groups,16),np.uint8),np.full((rows,groups),127,np.uint8)


def evaluate(name,blocks,scales,max_roots=8,max_candidates=16):
    t=time.perf_counter()
    carrier=pack(blocks,scales,max_roots=max_roots,max_candidates=max_candidates)
    pack_ms=(time.perf_counter()-t)*1000
    original=b''.join(b.tobytes()+s.tobytes() for b,s in zip(blocks,scales))
    rb,rs=carrier.restore()
    assert np.array_equal(rb,blocks) and np.array_equal(rs,scales)
    assert carrier.digest==hashlib.sha256(original).digest()
    x=np.random.default_rng(13).standard_normal(blocks.shape[1]*32)
    before=time.perf_counter()
    direct=carrier.dot(x)
    dot_ms=(time.perf_counter()-before)*1000
    oracle=dense_reference_dot(blocks,scales,x)
    np.testing.assert_allclose(direct,oracle,rtol=1e-12,atol=1e-10)
    raw=len(original)
    return {'corpus':name,'rows':blocks.shape[0],'groups':blocks.shape[1],
            'raw_bytes':raw,'r03_frame_bytes':carrier.frame_bytes,'mode':carrier.mode,
            'zlib9_bytes':len(zlib.compress(original,9)),
            'xz6_bytes':len(lzma.compress(original,preset=6)),
            'frame_over_raw':round(carrier.frame_bytes/raw,4),
            'mantissa_root_neurons':len(carrier.mantissa.bases) if carrier.mantissa is not None else 0,
            'mantissa_links':carrier.mantissa.links if carrier.mantissa is not None else 0,
            'mantissa_residuals':carrier.mantissa.residuals if carrier.mantissa is not None else 0,
            'scale_root_neurons':len(carrier.scales.bases) if carrier.scales is not None else 0,
            'negative_and_positive_zero_count':carrier.zero_count,
            'pack_ms':round(pack_ms,2),'one_dot_ms':round(dot_ms,3),
            'exact_packed_bits':True,'direct_float_gemv_match':True}


def sample_real_checkpoint(directory:Path, rows:int, groups:int, tensor:str|None):
    """Read a tiny aligned slice directly from local official safetensors shards.

    Does not download entire model or change it. Choose a tensor.blocks name or
    automatically take the first MXFP4 pair in the safetensors index.
    """
    from safetensors import safe_open
    idx=json.loads((directory/'model.safetensors.index.json').read_text())
    weight_map=idx['weight_map']
    if tensor is None:
        tensor=next((name for name in sorted(weight_map) if name.endswith('.blocks')
                     and name[:-7]+'.scales' in weight_map),None)
    if tensor is None or not tensor.endswith('.blocks'):
        raise ValueError('MXFP4 .blocks tensor not in checkpoint index')
    other=tensor[:-7]+'.scales'
    if other not in weight_map:
        raise ValueError('matching MXFP4 .scales missing')
    if weight_map[tensor]!=weight_map[other]:
        raise ValueError('sample currently requires both tensors in same shard')
    shard=directory/weight_map[tensor]
    with safe_open(str(shard), framework='np',device='cpu') as handle:
        blocks_view=handle.get_slice(tensor)
        scales_view=handle.get_slice(other)
        shape=blocks_view.get_shape()
        if shape[-1]!=16 or len(shape)<3:
            raise ValueError(f'unexpected MXFP4 tensor shape: {shape}')
        if len(shape)==3:
            bs=blocks_view[:min(rows,shape[0]),:min(groups,shape[1]),:]
            ss=scales_view[:min(rows,shape[0]),:min(groups,shape[1])]
        else:
            # One expert, then the first rows, then block-aligned groups.
            prefix=(0,)*(len(shape)-4)
            bs=blocks_view[prefix+(0,slice(0,min(rows,shape[-3])),
                                   slice(0,min(groups,shape[-2])),slice(None))]
            ss=scales_view[prefix+(0,slice(0,min(rows,shape[-3])),
                                   slice(0,min(groups,shape[-2])))]
    blocks=np.asarray(bs,dtype=np.uint8).reshape(-1,min(groups,shape[-2]),16)
    scales=np.asarray(ss,dtype=np.uint8).reshape(-1,min(groups,shape[-2]))
    if blocks.shape[:2]!=scales.shape:
        raise ValueError('MXFP4 sample misaligned')
    return tensor, blocks, scales


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--tensor')
    p.add_argument('--rows',type=int,default=32)
    p.add_argument('--groups',type=int,default=4)
    args=p.parse_args()
    results=[evaluate(*case) for case in cases()]
    if args.model_dir:
        name,b,s=sample_real_checkpoint(args.model_dir,args.rows,args.groups,args.tensor)
        result=evaluate(name,b,s)
        result['model_source']='local checkpoint slice'
        results.append(result)
    print(json.dumps({'schema':'zipburger-mxfp4-r03','dataset':'synthetic unless model_source is set',
                      'results':results},indent=2))

if __name__=='__main__':main()
