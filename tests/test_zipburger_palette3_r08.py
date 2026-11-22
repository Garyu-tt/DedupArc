"""ZipBurger R08 isolated CPU tests; no checkpoint download."""
import unittest
from unittest import mock
import numpy as np
from experiments.zipburger import palette3_r08 as r

class Palette3R08Tests(unittest.TestCase):
    def fixture(self, groups=4, seed=8):
        rng = np.random.default_rng(seed)
        blocks = rng.integers(0, 256, (48, groups, 16), dtype=np.uint8)
        scales = rng.integers(120, 130, (48, groups), dtype=np.uint8)
        calibration = rng.standard_normal((24, groups * 32))
        validation = rng.standard_normal((16, groups * 32))
        return blocks, scales, calibration, validation

    def test_3bit_packing_across_byte_boundaries(self):
        idx = np.arange(32, dtype=np.uint8) % 8
        np.testing.assert_array_equal(r._unpack_3bit(r._pack_3bit(idx)), idx)
        self.assertEqual(len(r._pack_3bit(idx)), 12)

    def test_exact_for_source_with_four_distinct_values(self):
        b, s, a, v = self.fixture()
        b &= 0x33
        carrier, report = r.evaluate(b, s, a, v,
                                     max_relative_l2=1e-8, min_cosine=0.9999999)
        self.assertTrue(report["quality_gate_passed"])
        self.assertLess(carrier.frame_bytes, carrier.source_bytes)
        for x in v:
            np.testing.assert_allclose(carrier.dot(x), r.dense_dot(b, s, x),
                                       rtol=1e-12, atol=1e-9)

    def test_fail_closed_on_lossy_random_weights(self):
        b, s, a, v = self.fixture()
        carrier, report = r.evaluate(b, s, a, v,
                                     max_relative_l2=0., min_cosine=1.)
        self.assertIsNone(carrier)
        self.assertFalse(report["quality_gate_passed"])
        self.assertFalse(report["model_level_or_agentic_equivalence"])

    def test_exact_fallback_is_reported(self):
        b, s, a, v = self.fixture()
        class Exact:
            mode = "original"
            frame_bytes = 1234
        factory = mock.Mock(return_value=Exact())
        carrier, report = r.evaluate(b, s, a, v, max_relative_l2=0.,
                                     min_cosine=1., fallback_factory=factory)
        self.assertEqual(carrier.frame_bytes, 1234)
        self.assertEqual(report["fallback_mode"], "original")
        factory.assert_called_once()

    def test_direct_dot_without_dense_restore(self):
        b, s, a, v = self.fixture()
        c = r.pack(b, s, a)
        with mock.patch.object(r, "dense_dot", side_effect=AssertionError("dense path")):
            out = c.dot(v[0])
        self.assertEqual(out.shape, (48,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_bad_frame_and_input_fail_closed(self):
        b, s, a, v = self.fixture()
        c = r.pack(b, s, a)
        bad = bytearray(c.payload)
        bad[-1] ^= 1
        for frame in (bytes(bad), c.payload[:-1], b"XXXX" + c.payload[4:]):
            with self.subTest(frame_len=len(frame)), self.assertRaises(ValueError):
                r.load(frame)
        with self.assertRaises(TypeError):
            r.pack(b.astype(np.int8), s, a)
        with self.assertRaises(ValueError):
            r.pack(b, s, np.zeros((4, 7)))
        with self.assertRaises(IndexError):
            c.read_codes(48, 0)

    def test_protected_attention_and_router_refused(self):
        self.assertEqual(r.expert_layer_id("blk.23.ffn_up_exps.weight"), 23)
        for name in ("blk.23.attn_q.weight", "blk.23.ffn_router.weight",
                     "blk.23.ffn_up.weight", "not_a_tensor"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                r.expert_layer_id(name)

    def test_frame_payload_saves_bytes(self):
        b, s, a, v = self.fixture(groups=16)
        c = r.pack(b, s, a)
        self.assertEqual(c.frame_bytes, r.HEADER.size + 48 * 16 * 13)
        self.assertEqual(c.source_bytes, 48 * 16 * 17)
        self.assertLess(c.frame_bytes, c.source_bytes)

if __name__ == "__main__":
    unittest.main()
