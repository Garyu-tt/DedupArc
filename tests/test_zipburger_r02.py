"""R02 differential, corruption and direct inference test suite."""
import hashlib
import unittest
from unittest import mock

import numpy as np
from experiments.zipburger import multiroot_r02 as z


def corpus(seed=2026):
    rng = np.random.default_rng(seed)
    n = 256
    base = rng.integers(-30,31,(1,n),dtype=np.int16).astype(np.int8)[0]
    # Reused sparse root neurons carry independent residual-like directions.
    patterns=np.zeros((7,n),np.int8)
    for i in range(7):
        spots=(np.arange(i*3, n, 13)+i*7) % n
        patterns[i,spots] = rng.integers(-6,7,len(spots),dtype=np.int16).astype(np.int8)
    rows=[base.copy(),*patterns.copy()]
    for i in range(200):
        row = base.astype(np.int16) + patterns[i%7].astype(np.int16)
        for j in range(i%3):
            row[(i*23+j*53)%n]+=1
        rows.append(row.astype(np.int8))
    return np.asarray(rows,dtype=np.int8)


class MultiRootTests(unittest.TestCase):
    def test_exact_direct_two_root_dot_and_random_cell_lookup(self):
        w=corpus(); c=z.pack(w,max_roots=12,max_candidates=32)
        self.assertLess(c.frame_bytes,w.nbytes)
        self.assertTrue(np.any(np.count_nonzero(c.index>=0,axis=1)==2))
        self.assertTrue(np.array_equal(c.restore(),w))
        with mock.patch.object(z.NeuralCarrierR02,'restore',side_effect=AssertionError('dense restore')):
            for row,col in [(0,0),(7,255),(80,120),(201,201)]:
                self.assertEqual(c.read_cell(row,col),int(w[row,col]))
            x=np.arange(w.shape[1],dtype=np.int64)%13-6
            np.testing.assert_array_equal(c.dot(x),w.astype(np.int64)@x)
        self.assertEqual(z.load(c.payload).digest,hashlib.sha256(w.tobytes()).digest())

    def test_bytes_after_codec_roundtrip(self):
        rng=np.random.default_rng(14)
        # Already entropy-coded: intentionally difficult control, no fake gains.
        stream=rng.integers(0,256,16391,dtype=np.uint8).tobytes()
        c=z.pack_bytes(stream,256,max_roots=3,max_candidates=8)
        self.assertEqual(c.restore_bytes(),stream)
        self.assertEqual(c.frame_bytes, len(c.payload))
        self.assertEqual(c.links,0)

    def test_corrupted_payload_is_detected(self):
        c=z.pack(corpus(),max_roots=12,max_candidates=32)
        for p in [c.payload[:-1], c.payload+b'x',c.payload[:65]+b'\xff'+c.payload[66:]]:
            with self.assertRaises(ValueError): z.load(p)

    def test_literal_fallback_memory(self):
        rng=np.random.default_rng(47)
        w=rng.integers(-128,128,(64,128),dtype=np.int16).astype(np.int8)
        c=z.pack(w,max_roots=4,max_candidates=8)
        self.assertEqual(c.links,0)
        self.assertEqual(c.residuals,0)
        self.assertEqual(len(c.literal_data),w.nbytes)
        self.assertEqual(c.restore().tobytes(),w.tobytes())
        x=rng.integers(-10,11,w.shape[1],dtype=np.int64)
        np.testing.assert_array_equal(c.dot(x),w.astype(np.int64)@x)

if __name__=='__main__': unittest.main()

class TorchMultiRootTests(unittest.TestCase):
    def test_uses_trainable_adapters_with_no_dense_carrier(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        w=corpus();carrier=z.pack(w,max_roots=12,max_candidates=32)
        with mock.patch.object(z.NeuralCarrierR02,'restore',side_effect=AssertionError('full restore called')):
            net=z.make_torch_module(carrier,rank=2)
            inp=torch.randn(3,w.shape[1],requires_grad=True)
            out=net(inp)
        dense=inp @ torch.tensor(w.copy(),dtype=torch.float32).T
        torch.testing.assert_close(out,dense,rtol=0.0002,atol=0.00025)
        out.square().mean().backward()
        self.assertIsNotNone(net.input_net.up.weight.grad)
        self.assertIsNotNone(net.output_net.up.weight.grad)
        self.assertEqual(net.residual_connections.layout,torch.sparse_coo)
        self.assertFalse(any(tuple(b.shape)==w.shape for b in net.buffers() if not b.is_sparse))
