import unittest
from unittest import mock

import numpy as np
from experiments.zipburger.neurocrunch import Carrier, load, pack, make_torch_module


class ZipBurgerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2006)
        self.base = rng.integers(-12, 13, size=(6, 128), dtype=np.int8)
        # Guarantee exact donor rows to discover each root before edited copies.
        rows = [r.copy() for r in self.base]
        for i in range(192):
            b = self.base[i % 6]
            scale = (1, 2, 4, 8)[(i // 6) % 4]
            row = (b.astype(np.int16) * scale).astype(np.int8)
            if i % 5:
                row[(i * 17) % 128] += np.int8(1)
            rows.append(row)
        self.w = np.array(rows, dtype=np.int8)

    def test_lossless_sparse_root_and_exact_direct_dot(self):
        carrier = pack(self.w)
        self.assertLess(carrier.compressed_bytes, carrier.raw_bytes)
        self.assertGreater(carrier.root_count, 0)
        self.assertGreater(carrier.link_count, 50)
        self.assertEqual(carrier.restore().tobytes(), self.w.tobytes())
        self.assertEqual(load(carrier.payload).restore().tobytes(), self.w.tobytes())
        rng = np.random.default_rng(412)
        for _ in range(5):
            x = rng.integers(-8, 9, size=self.w.shape[1], dtype=np.int64)
            np.testing.assert_array_equal(carrier.dot(x), self.w.astype(np.int64) @ x)

    def test_load_does_not_materialize_full_weight(self):
        carrier = pack(self.w)
        with mock.patch.object(Carrier, "restore", side_effect=AssertionError("dense restore called")):
            loaded = load(carrier.payload)
            x = np.ones(self.w.shape[1], dtype=np.int64)
            np.testing.assert_array_equal(loaded.dot(x), self.w.astype(np.int64) @ x)

    def test_random_high_entropy_rejects_graph(self):
        rng = np.random.default_rng(54)
        w = rng.integers(-128, 128, size=(32, 128), dtype=np.int16).astype(np.int8)
        carrier = pack(w)
        self.assertEqual(carrier.root_count, 0)
        np.testing.assert_array_equal(carrier.restore(), w)
        self.assertEqual(carrier.compressed_bytes, w.size + 48)

    def test_uniform_zero_and_invalid_inputs(self):
        w = np.zeros((12, 16), dtype=np.int8)
        carrier = pack(w)
        self.assertEqual(carrier.restore().tobytes(), w.tobytes())
        with self.assertRaises(ValueError): load(carrier.payload[:-1])
        with self.assertRaises(ValueError): load(carrier.payload + b"extra")
        with self.assertRaises(TypeError): pack(w.astype(np.float32))
        with self.assertRaises(ValueError): carrier.dot(np.zeros(3, dtype=np.int64))

    def test_input_output_neural_adapters(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        carrier = pack(self.w)
        net = make_torch_module(carrier)
        self.assertGreater(sum(p.numel() for p in net.parameters()), 0)
        x = torch.randn(4, self.w.shape[1])
        actual = net(x)
        expected = x @ torch.from_numpy(self.w.astype(np.float32)).T
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
        actual.sum().backward()
        self.assertIsNotNone(net.input_net.up.weight.grad)
        self.assertIsNotNone(net.output_net.up.weight.grad)
        # No dense original weight tensor in carrier's registered buffers.
        self.assertEqual(net.correction_edges.layout, torch.sparse_coo)
        self.assertTrue(all(b.shape != self.w.shape for name, b in net.named_buffers() if name != "correction_edges"))


if __name__ == "__main__": unittest.main()
