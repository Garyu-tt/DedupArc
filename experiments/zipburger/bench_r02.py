"""Reproducible ZipBurger R02 lab: python -m experiments.zipburger.bench_r02"""
from __future__ import annotations

import json
import time
import zlib

import numpy as np

from .neurocrunch import pack as pack_r01
from .multiroot_r02 import pack as pack_r02, pack_bytes


def corpora():
    rng = np.random.default_rng(2026)
    n=256
    dense=rng.integers(-30,31,n,dtype=np.int16).astype(np.int8)
    patterns=np.zeros((7,n),np.int8)
    for i in range(7):
        positions=(np.arange(i*3,n,13)+i*7)%n
        patterns[i,positions]=rng.integers(-6,7,len(positions),dtype=np.int16).astype(np.int8)
    rows=[dense.copy(),*patterns.copy()]
    for i in range(200):
        row=dense.astype(np.int16)+patterns[i%7].astype(np.int16)
        for j in range(i%3):
            row[(i*23+j*53)%n]+=1
        rows.append(row.astype(np.int8))
    yield 'two-shared-neurons-sparse-residuals',np.array(rows,dtype=np.int8)
    yield 'random-int8',rng.integers(-128,128,(208,256),dtype=np.int16).astype(np.int8)
    yield 'zeros',np.zeros((208,256),np.int8)
    yield 'mxfp4-byte-like-control',rng.choice(np.array([-119,-85,-1,0,17,34,68,119],np.int8),size=(208,256))


def main():
    rng=np.random.default_rng(818)
    results=[]
    for name,w in corpora():
        start=time.perf_counter()
        r02=pack_r02(w,max_roots=12,max_candidates=32)
        ms=(time.perf_counter()-start)*1000
        r01=pack_r01(w,max_roots=12,max_candidates=32)
        x=rng.integers(-8,9,w.shape[1],dtype=np.int64)
        dense=w.astype(np.int64)@x
        assert np.array_equal(r02.dot(x),dense)
        assert r02.restore().tobytes()==w.tobytes()
        assert r02.read_cell(100,99)==int(w[100,99])
        results.append({'corpus':name,'raw_bytes':w.nbytes,'zipburger_r01_bytes':r01.frame_bytes if hasattr(r01,'frame_bytes') else r01.compressed_bytes,
            'zipburger_r02_bytes':r02.frame_bytes,'zlib9_bytes':len(zlib.compress(w.tobytes(),9)),
            'root_neurons':len(r02.bases),'links':r02.links,'residuals':r02.residuals,
            'literal_rows':int(np.count_nonzero(r02.literal_row)),
            'r02_pack_ms':round(ms,2),'exact_integer_gemv':True,'exact_direct_cell_read':True,'exact_roundtrip':True})
    post_codec=zlib.compress(next(corpora())[1].tobytes(),level=9)
    carrier=pack_bytes(post_codec,width=256,max_roots=5,max_candidates=12)
    assert carrier.restore_bytes()==post_codec
    results.append({'corpus':'zlib-coded-structured-data','raw_bytes':len(post_codec),
        'zipburger_r02_bytes':carrier.frame_bytes,'zlib9_bytes':len(zlib.compress(post_codec,9)),
        'root_neurons':len(carrier.bases),'links':carrier.links,'exact_roundtrip':True,
        'note':'entropy-coded bytes were bitcast into int8 and preserved; no lossless gain claimed'})
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
