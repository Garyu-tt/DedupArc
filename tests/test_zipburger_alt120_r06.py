"""Offline GGUF v3/mxfp4 120B range reader fixture tests. No network."""
from io import BytesIO
import struct
import unittest
from urllib.request import Request

import numpy as np

from experiments.zipburger import alt120b_real_r06 as r


def text(s):
    d=s.encode()
    return struct.pack("<Q",len(d))+d


def fake_gguf():
    rng=np.random.default_rng(5)
    tensors=[]
    payload=bytearray()
    for layer in (0,1,2):
        for kind in ("down","gate","up"):
            name=f"blk.{layer}.ffn_{kind}_exps.weight"
            tensors.append((name, len(payload)))
            b=rng.integers(0,256,(48,16,17),dtype=np.uint8)
            b[:,:,0]=127 + layer
            payload.extend(b.tobytes())
    meta=text("general.alignment")+struct.pack("<II",4,32)
    entries=b"".join(
        text(name)+struct.pack("<IQQIQ",2,512,48,39,offset)
        for name,offset in tensors
    )
    header=struct.pack("<4sIQQ",b"GGUF",3,len(tensors),1)+meta+entries
    align=(len(header)+31)//32*32
    content=header+b"\0"*(align-len(header))+payload
    return content.ljust(r.CHUNK,b"\0")


class FakeResponse:
    def __init__(self,a,b,code=206,bad_range=False,host=None):
        self.status=code
        self.headers={
            "Content-Range": (f"bytes {a}-{b}/{r.FILE_BYTES}"
                              if not bad_range else "bytes 0-4/10"),
            "Content-Length": str(b-a+1)
        }
        self.host=host or r.URL
        self.content=b"x"*(b-a+1)
    def geturl(self):return self.host
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def read(self,n):return self.content[:n]


class FakeOpener:
    def __init__(self,resp):self.resp=resp
    def open(self,req,timeout):
        self.req=req
        return self.resp


class ALT120BTests(unittest.TestCase):
    def test_scan_realistic_metadata_without_reading_full_checkpoint(self):
        content=fake_gguf()
        seen=[]
        def fetch(a,b):
            seen.append((a,b))
            return content[a:b+1]
        idx=r.Header(fetch)
        self.assertEqual(len(idx.tensors),9)
        self.assertLessEqual(idx.downloaded_header_bytes,r.CHUNK)
        self.assertEqual(r.choose(idx)[1],[0,1,2])
        dims,offset,count,gpr,nbytes=idx.mx_tensor("blk.0.ffn_down_exps.weight")
        self.assertEqual(dims,(512,48))
        self.assertEqual(count,48)
        self.assertEqual(gpr,16)
        self.assertEqual(nbytes,13056)
        data,scales,original,shape,span=r.sample(idx,"blk.0.ffn_down_exps.weight")
        self.assertEqual(data.shape,(48,16,16))
        self.assertTrue(np.all(scales==127))
        self.assertEqual(len(original),13056)
        result=r.benchmark(idx,"blk.1.ffn_gate_exps.weight",0)
        self.assertTrue(result["exact_packed_bytes"])
        self.assertTrue(result["float64_direct_dot_reference_match"])
        self.assertLess(result["r04_bytes"],result["raw_bytes"])
        self.assertLessEqual(sum(b-a+1 for a,b in seen), 3*r.CHUNK)

    def test_reject_incorrect_range_or_untrusted_host(self):
        for kw in (dict(code=200),dict(bad_range=True),
                   dict(host="https://example.net/checkpoint")):
            with self.subTest(kw=kw), self.assertRaises(r.RangeError):
                r.fetch_range(0,15,opener=FakeOpener(FakeResponse(0,15,**kw)))
        for a,b in ((-1,100),(2,1),(0,r.CHUNK),(0,r.FILE_BYTES)):
            with self.subTest(a=a,b=b),self.assertRaises(r.RangeError):
                r.fetch_range(a,b,opener=FakeOpener(FakeResponse(0,0)))

    def test_invalid_gguf_is_rejected(self):
        content=bytearray(fake_gguf())
        content[0:4]=b"BAD!"
        with self.assertRaises(r.RangeError):
            r.Header(lambda a,b:bytes(content[a:b+1]))
        content[0:4]=b"GGUF"
        content[4:8]=struct.pack("<I",2)
        with self.assertRaises(r.RangeError):
            r.Header(lambda a,b:bytes(content[a:b+1]))

    def test_sample_bounds(self):
        content=fake_gguf()
        idx=r.Header(lambda a,b:content[a:b+1])
        for args in (dict(row_start=-1),dict(rows=129),
                     dict(groups=17),dict(rows=True),
                     dict(row_start=30,rows=48)):
            with self.subTest(args=args),self.assertRaises(r.RangeError):
                r.sample(idx,"blk.0.ffn_down_exps.weight",**args)
        with self.assertRaises(r.RangeError):
            r.sample(idx,"no-experts-found")


if __name__=="__main__":
    unittest.main()
