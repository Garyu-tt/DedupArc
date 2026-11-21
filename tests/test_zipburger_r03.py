"""MXFP4 exact-bit carrier, negative-zero and direct graph execution tests."""
import hashlib
import unittest
from unittest import mock
import zlib

import numpy as np
from experiments.zipburger import mxfp4_r03 as z


def structured(seed=2026, rows=32, groups=4):
    rng=np.random.default_rng(seed)
    # Repeat actual E2M1 sign/magnitude codes, including -0 nibble values.
    patterns=rng.integers(0,16,(4,groups,16),dtype=np.uint8)
    b=patterns[np.arange(rows)%4].copy()
    b[::5,0,0] ^= 8
    scales=(rng.integers(112,135,(4,groups),dtype=np.uint8))[np.arange(rows)%4].copy()
    scales[::9,0] += 1
    return b,scales

class R03(unittest.TestCase):
    def test_exact_roundtrip_all_nibbles_and_signed_zero(self):
        b=np.arange(256,dtype=np.uint8).reshape(16,1,16)
        s=np.arange(100,116,dtype=np.uint8).reshape(16,1)
        c=z.pack(b,s,max_roots=8,max_candidates=16)
        self.assertEqual(c.rows,16)
        self.assertEqual(c.source_bytes,272)
        rb,rs=c.restore()
        np.testing.assert_array_equal(rb,b)
        np.testing.assert_array_equal(rs,s)
        for row in range(16):
            for col in range(32):
                code=int(b[row,0,col//2])
                self.assertEqual(c.read_code(row,col),(code & 15) if col%2==0 else (code>>4))
        self.assertEqual(z.load(c.payload).digest,c.digest)

    def test_direct_gemv_no_dense_restore(self):
        b,s=structured()
        c=z.pack(b,s,max_roots=8,max_candidates=12)
        vec=np.arange(c.groups*32,dtype=np.float64)%11-5
        with mock.patch.object(z.Mxfp4CarrierR03,'restore',side_effect=AssertionError('dense')):
            with mock.patch.object(z.graph.NeuralCarrierR02,'restore',side_effect=AssertionError('dense')):
                out=c.dot(vec)
        np.testing.assert_allclose(out,z.dense_reference_dot(b,s,vec),rtol=1e-12,atol=1e-10)
        self.assertEqual(c.verify(),None)
        self.assertEqual(c.mode,'neural-graph')
        self.assertGreaterEqual(c.mantissa.links,0)

    def test_random_high_entropy_fallback_and_float(self):
        rng=np.random.default_rng(8)
        b=rng.integers(0,256,(18,2,16),dtype=np.uint8)
        s=rng.integers(115,130,(18,2),dtype=np.uint8)
        c=z.pack(b,s,max_roots=5,max_candidates=8)
        rb,rs=c.restore()
        np.testing.assert_array_equal(rb,b)
        np.testing.assert_array_equal(rs,s)
        self.assertEqual(c.mode,'raw-fallback')
        self.assertLessEqual(c.frame_bytes,c.source_bytes+44)
        x=rng.standard_normal(64)
        np.testing.assert_allclose(c.dot(x),z.dense_reference_dot(b,s,x),rtol=1e-12,atol=1e-9)

    def test_reject_corrupt_payload_and_wrong_shapes(self):
        b,s=structured(rows=12,groups=2)
        c=z.pack(b,s,max_roots=4,max_candidates=8)
        for bad in (c.payload[:-1],c.payload+b'\x00',c.payload[:20]+b'\xff'+c.payload[21:]):
            with self.assertRaises(ValueError):z.load(bad)
        with self.assertRaises(ValueError): z.pack(b,s[:2])

    def test_torch_input_output_adapters(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        b,s=structured(rows=48,groups=4)
        c=z.pack(b,s,max_roots=8,max_candidates=12)
        with mock.patch.object(z.Mxfp4CarrierR03,'restore',side_effect=AssertionError('dense')):
            net=z.make_torch_module(c,rank=2)
            x=torch.randn(3,128,requires_grad=True)
            y=net(x)
        ref=torch.tensor(z.dense_reference_dot(b,s,x.detach().numpy()[0]),dtype=torch.float32)
        torch.testing.assert_close(y[0],ref,rtol=0.0003,atol=0.0005)
        y.square().mean().backward()
        self.assertIsNotNone(net.input_net.up.weight.grad)
        self.assertIsNotNone(net.output_net.up.weight.grad)
        self.assertFalse(any(tuple(b.shape)==(c.rows,128) for b in net.buffers() if not b.is_sparse))

if __name__=='__main__':unittest.main()

class R03FormatAndLocalCheckpoint(unittest.TestCase):
    def test_all_negative_zeros_compact_sideband(self):
        b=np.full((48,2,16),0x88,dtype=np.uint8)
        s=np.full((48,2),127,dtype=np.uint8)
        c=z.pack(b,s,max_roots=4,max_candidates=8)
        self.assertEqual(c.mode,'neural-graph')
        self.assertEqual(c.sign_mode,1)
        self.assertEqual(c.zero_signs,b'')
        np.testing.assert_array_equal(c.restore()[0],b)
        self.assertEqual(c.read_code(47,63),8)

    def test_load_streaming_never_calls_full_restore(self):
        b,s=structured(rows=48,groups=4)
        original=z.pack(b,s,max_roots=8,max_candidates=12)
        with mock.patch.object(z.Mxfp4CarrierR03,'restore',side_effect=AssertionError('dense used')):
            with mock.patch.object(z.graph.NeuralCarrierR02,'restore',side_effect=AssertionError('dense used')):
                loaded=z.load(original.payload)
                self.assertEqual(loaded.read_code(4,12),original.read_code(4,12))
                loaded.dot(np.arange(128,dtype=np.float64))

    def test_benchmark_reads_only_local_safetensors_slice(self):
        try:
            from safetensors.numpy import save_file
        except ImportError:
            self.skipTest('safetensors optional')
        import tempfile
        import json
        from pathlib import Path
        from experiments.zipburger.bench_r03 import sample_real_checkpoint
        b=np.arange(2*8*3*16,dtype=np.uint16).astype(np.uint8).reshape(2,8,3,16)
        s=np.arange(2*8*3,dtype=np.uint8).reshape(2,8,3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_file({'block.0.mlp.mlp1_weight.blocks':b,
                       'block.0.mlp.mlp1_weight.scales':s},str(root/'model-00001-of-00001.safetensors'))
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
                'block.0.mlp.mlp1_weight.blocks':'model-00001-of-00001.safetensors',
                'block.0.mlp.mlp1_weight.scales':'model-00001-of-00001.safetensors'}}))
            name,rb,rs=sample_real_checkpoint(root,rows=5,groups=2,tensor=None)
            self.assertEqual(name,'block.0.mlp.mlp1_weight.blocks')
            np.testing.assert_array_equal(rb,b[0,:5,:2,:])
            np.testing.assert_array_equal(rs,s[0,:5,:2])
