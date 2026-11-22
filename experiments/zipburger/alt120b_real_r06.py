"""R06: bounded public GGUF MXFP4 scale benchmark of real gpt-oss-120b.

Only immutable Hugging Face HTTPS byte-ranges are accessed; no full 63 GB
download, no user VM/ALT/model service interactions, no checkpoint writes.
The header is lazily scanned in <= 24 MiB and sample reads are <= 128 KiB.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import struct
from urllib.request import Request, ProxyHandler, build_opener
import zlib
import lzma

import numpy as np
from .mxfp4_palette_r04 import pack
from .mxfp4_r03 import dense_reference_dot
from .real_mxfp4_ranges_r04 import _TrustedHFRedirect, _trusted_host

REPO = "ggml-org/gpt-oss-120b-GGUF"
REV = "f59990363fbbb67f01bce03e5e7bf0929e3e320b"
FILENAME = "gpt-oss-120b-MXFP4.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{FILENAME}"
FILE_BYTES = 63_387_346_208
UPSTREAM_SHA256 = "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d"
HEADER_LIMIT = 24 * 1024 * 1024
CHUNK = 4 * 1024 * 1024
SAMPLE_LIMIT = 128 * 1024
TYPE_MXFP4 = 39
TYPESIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
RE_NAME = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up)_exps\.weight$")


class RangeError(ValueError):
    pass


def fetch_range(first: int, last: int, *, opener=None) -> bytes:
    if (type(first) is not int or type(last) is not int or first < 0
        or last < first or last >= FILE_BYTES or last-first+1>CHUNK):
        raise RangeError("range outside immutable model / size bound")
    req = Request(URL, headers={"Range": f"bytes={first}-{last}",
           "Accept-Encoding": "identity", "User-Agent": "ZipBurger120B-RangeR06/1.0"})
    client = opener or build_opener(ProxyHandler({}), _TrustedHFRedirect())
    with client.open(req,timeout=30) as resp:
        if not _trusted_host(resp.geturl()):
            raise RangeError("untrusted URL for model bytes")
        if resp.status != 206:
            raise RangeError(f"HTTP {resp.status} is not partial content")
        expected=f"bytes {first}-{last}/{FILE_BYTES}"
        if resp.headers.get("Content-Range","")!=expected:
            raise RangeError("wrong Content-Range for pinned checkpoint")
        length=last-first+1
        if resp.headers.get("Content-Length"):
            if int(resp.headers["Content-Length"])!=length:
                raise RangeError("wrong range body size")
        content=resp.read(length+1)
        if len(content)!=length:
            raise RangeError("truncated/oversized range")
        return content


class Header:
    """Streaming GGUF v3 metadata scanner with a fixed hard HTTP cap."""

    def __init__(self, fetch=fetch_range):
        self.fetch=fetch
        self.buf=bytearray()
        self.pos=0
        self.tensors={}
        self.alignment=32
        if self.read(4)!=b"GGUF" or self.u32()!=3:
            raise RangeError("unsupported GGUF version/magic")
        n_tensors,n_meta=self.u64(),self.u64()
        if not 1 <= n_tensors <= 50000 or n_meta > 100000:
            raise RangeError("GGUF metadata bounds")
        self.metadata_count=n_meta
        for _ in range(n_meta):
            key=self.string()
            typ=self.u32()
            if key=="general.alignment" and typ==4:
                self.alignment=self.u32()
            else:
                self.skip_value(typ)
        if not 0<self.alignment<=4096 or self.alignment&(self.alignment-1):
            raise RangeError("invalid tensor alignment")
        for _ in range(n_tensors):
            name=self.string()
            dims_count=self.u32()
            if not 1<=dims_count<=8:raise RangeError("bad tensor rank")
            dims=tuple(self.u64() for _ in range(dims_count))
            if any(not n or n>1000000000 for n in dims):
                raise RangeError("bad tensor dims")
            typ=self.u32()
            offset=self.u64()
            if name in self.tensors:raise RangeError("duplicate tensor")
            self.tensors[name]=(typ,dims,offset)
        self.data_start=(self.pos+self.alignment-1)//self.alignment*self.alignment
        if self.data_start>HEADER_LIMIT:
            raise RangeError("tensor data starts after header limit")
        self.downloaded_header_bytes=len(self.buf)

    def ensure(self, end: int):
        if not 0<=end<=HEADER_LIMIT:
            raise RangeError("header scan exceeded 24 MiB")
        while len(self.buf)<end:
            start=len(self.buf)
            stop=min(start+CHUNK,HEADER_LIMIT,FILE_BYTES)-1
            if stop<start:raise RangeError("header too large")
            self.buf.extend(self.fetch(start,stop))

    def read(self,n):
        if n<0 or n>HEADER_LIMIT-self.pos:raise RangeError("read exceeds header")
        self.ensure(self.pos+n)
        x=self.buf[self.pos:self.pos+n]
        self.pos+=n
        return x
    def skip(self,n):
        if n<0 or n>HEADER_LIMIT-self.pos:raise RangeError("skip exceeds header")
        self.ensure(self.pos+n)
        self.pos+=n
    def u32(self):return struct.unpack("<I",self.read(4))[0]
    def u64(self):return struct.unpack("<Q",self.read(8))[0]
    def string(self):
        n=self.u64()
        if n>65536:raise RangeError("overlong string")
        return self.read(n).decode("utf-8")
    def skip_string(self):
        n=self.u64()
        if n>65536:raise RangeError("overlong metadata element")
        self.skip(n)
    def skip_value(self,typ):
        if typ in TYPESIZE:
            self.skip(TYPESIZE[typ])
        elif typ==8:
            self.skip_string()
        elif typ==9:
            subtyp=self.u32()
            n=self.u64()
            if n>3000000:raise RangeError("huge metadata array")
            if subtyp in TYPESIZE:
                self.skip(n*TYPESIZE[subtyp])
            elif subtyp==8:
                for _ in range(n):self.skip_string()
            else:raise RangeError("unsupported array type")
        else:raise RangeError("unsupported GGUF metadata type")

    def mx_tensor(self,name):
        item=self.tensors.get(name)
        if item is None or item[0]!=TYPE_MXFP4:
            raise RangeError("missing/incorrect MXFP4 tensor")
        _,dims,offset=item
        if dims[0]%32:raise RangeError("MXFP4 block width")
        groups_per_row=dims[0]//32
        rows=math.prod(dims[1:])
        tensor_bytes=rows*groups_per_row*17
        base=self.data_start+offset
        if base<self.data_start or base+tensor_bytes>FILE_BYTES:
            raise RangeError("tensor extent exceeds file")
        return dims,base,rows,groups_per_row,tensor_bytes


def sample(header: Header, name, row_start=0, rows=48,groups=16):
    if (type(row_start) is not int or type(rows) is not int
        or type(groups) is not int or row_start<0
        or not 1<=rows<=128 or not 1<=groups<=16):
        raise RangeError("sample size/request bounds")
    dims,base,row_count,gpr,_=header.mx_tensor(name)
    if row_start+rows>row_count or groups>gpr:raise RangeError("sample exceeds tensor")
    stride=gpr*17
    first=base+row_start*stride
    last=first+(rows-1)*stride+groups*17-1
    if last-first+1>SAMPLE_LIMIT:raise RangeError("sample fetch exceeds 128 KiB")
    span=header.fetch(first,last)
    if len(span)!=last-first+1:raise RangeError("short sample")
    raw=bytearray()
    for r in range(rows):
        offset=r*stride
        raw.extend(span[offset:offset+groups*17])
    blocks=np.frombuffer(raw,np.uint8).reshape(rows,groups,17)
    return (np.ascontiguousarray(blocks[:,:,1:]),
            np.ascontiguousarray(blocks[:,:,0]),
            bytes(raw),dims,(first,last))


def choose(header):
    layers=sorted({int(m.group(1)) for n,v in header.tensors.items()
                   if v[0]==TYPE_MXFP4 and (m:=RE_NAME.fullmatch(n))})
    if len(layers)<3:raise RangeError("too few MXFP4 MoE layers in GGUF")
    ids=sorted({layers[0],layers[len(layers)//3],
                layers[(2*len(layers))//3],layers[-1]})
    names=[]
    for layer in ids:
        for kind in ("down","gate","up"):
            name=f"blk.{layer}.ffn_{kind}_exps.weight"
            header.mx_tensor(name)
            names.append(name)
    return names,ids


def benchmark(header,name,row_start):
    b,s,raw,dims,byte_range=sample(header,name,row_start)
    carrier=pack(b,s)
    rb,rs=carrier.restore()
    interleaved=np.empty((48,16,17),np.uint8)
    interleaved[:,:,0]=rs
    interleaved[:,:,1:]=rb
    if interleaved.tobytes()!=raw:
        raise AssertionError("lost GGUF bits or signed zero")
    vec=np.random.default_rng(1337).standard_normal(512)
    np.testing.assert_allclose(carrier.dot(vec),
        dense_reference_dot(b,s,vec),rtol=1e-12,atol=1e-8)
    return dict(tensor=name, dims=dims,row_start=row_start,
        official_file_range=byte_range,sample_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),r04_bytes=carrier.frame_bytes,r04_mode=carrier.mode,
        zlib9_bytes=len(zlib.compress(raw,9)),
        xz6_bytes=len(lzma.compress(raw,preset=6)),
        unique_scale_values=int(np.unique(s).size),
        exact_packed_bytes=True,float64_direct_dot_reference_match=True)


def main():
    h=Header()
    names,layers=choose(h)
    stats=Counter(t for t,_,_ in h.tensors.values())
    all_mxfp4=sum(h.mx_tensor(name)[4] for name,v in h.tensors.items() if v[0]==TYPE_MXFP4)
    results=[]
    for name in names:
        _,_,row_count,_,_=h.mx_tensor(name)
        for start in (0,4096):
            if start+48<=row_count:
                results.append(benchmark(h,name,start))
    aggregate={key:sum(x[key] for x in results)
        for key in ("raw_bytes","r04_bytes","zlib9_bytes","xz6_bytes")}
    print(json.dumps(dict(
        schema="neurocrunch-alt120b-real-mxfp4-r06/1",
        repo=REPO,revision=REV,filename=FILENAME,official_file_bytes=FILE_BYTES,
        upstream_whole_file_sha256=UPSTREAM_SHA256,whole_file_sha256_recomputed=False,
        bounded_header_download_bytes=h.downloaded_header_bytes,
        gguf_data_start=h.data_start,metadata_count=h.metadata_count,
        tensor_count=len(h.tensors),tensor_type_histogram=dict(stats),
        total_indexed_mxfp4_bytes=all_mxfp4,selected_layers=layers,
        checkpoint_downloaded=False,checkpoint_modified=False,
        gpu_involved=False,model_inference_performed=False,
        samples=len(results),aggregate=aggregate,results=results),
        indent=2,sort_keys=True))


if __name__=="__main__":
    main()
