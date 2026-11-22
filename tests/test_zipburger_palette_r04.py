"""Lossless E8M0 palette and delta direct execution; offline, no GPU."""
import hashlib
import struct
import unittest
from unittest import mock

import numpy as np

from experiments.zipburger import mxfp4_palette_r04 as r
from experiments.zipburger.mxfp4_r03 import dense_reference_dot


def fixture(rows=48, groups=16, seed=222):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, (rows, groups, 16), dtype=np.uint8)
    return b, rng.integers(115, 135, (rows, groups), dtype=np.uint8)


class BitplaneTests(unittest.TestCase):
    def test_bitpacking_all_widths_across_byte_boundaries(self):
        for bits in range(9):
            with self.subTest(bits=bits):
                values = np.array([i % (1 << bits) for i in range(37)],
                                  dtype=np.uint8)
                encoded = r._pack_bits(values, bits)
                self.assertEqual(len(encoded), (37 * bits + 7) // 8)
                got = [r._get_bitpacked(encoded, i, bits)
                       for i in range(values.size)]
                self.assertEqual(got, values.tolist())

    def test_constant_scale_signed_zero_and_direct(self):
        b = np.full((48, 4, 16), 0x88, np.uint8)
        s = np.full((48, 4), 127, np.uint8)
        c = r.pack(b, s)
        self.assertEqual(c.mode, "delta-bitpack")
        self.assertEqual(c.bits, 0)
        self.assertLess(c.frame_bytes, b.nbytes + s.nbytes)
        self.assertEqual(c.read_code(47, 127), 8)
        self.assertEqual(c.read_scale(47, 3), 127)
        bb, ss = c.restore()
        np.testing.assert_array_equal(bb, b)
        np.testing.assert_array_equal(ss, s)
        v = np.random.default_rng(9).standard_normal(128)
        with mock.patch.object(r.CarrierR04, "restore",
                               side_effect=AssertionError("dense restore used")):
            actual = c.dot(v)
        np.testing.assert_allclose(actual, dense_reference_dot(b, s, v),
                                   rtol=1e-12, atol=1e-8)

    def test_narrow_high_scale_range_no_false_overflow(self):
        b, _ = fixture(24, 4)
        s = np.random.default_rng(3).integers(200, 256, (24, 4),
                                               dtype=np.uint8)
        c = r.pack(b, s)
        bb, ss = r.load(c.payload).restore()
        np.testing.assert_array_equal(bb, b)
        np.testing.assert_array_equal(ss, s)
        self.assertIn(c.mode, ("delta-bitpack", "palette-bitpack",
                               "raw-fallback"))

    def test_sparse_outliers_palette_beats_range(self):
        b, _ = fixture(48, 4)
        s = np.full((48, 4), 127, np.uint8)
        s[0, 0] = 0
        s[6, 1] = 255
        c = r.pack(b, s)
        self.assertEqual(c.mode, "palette-bitpack")
        self.assertEqual(c.bits, 2)
        np.testing.assert_array_equal(c.restore()[1], s)

    def test_random_scale_selects_nonexpanding_fallback(self):
        b, _ = fixture(48, 4)
        s = np.arange(192, dtype=np.uint8).reshape(48, 4)
        c = r.pack(b, s)
        self.assertEqual(c.mode, "raw-fallback")
        self.assertEqual(c.frame_bytes, b.nbytes + s.nbytes + r.HEADER.size)
        np.testing.assert_array_equal(c.restore()[0], b)

    def test_mixed_sign_zero_integrity_and_numerical_dot(self):
        b, s = fixture()
        b[0, 0, 0] = 0x80
        b[0, 1, 0] = 0x08
        c = r.pack(b, s)
        self.assertLess(c.frame_bytes, b.nbytes + s.nbytes)
        x = np.random.default_rng(5).standard_normal(16 * 32)
        np.testing.assert_allclose(
            c.dot(x), dense_reference_dot(b, s, x),
            rtol=1e-12, atol=1e-8)
        bb, ss = r.load(c.payload).restore()
        np.testing.assert_array_equal(bb, b)
        np.testing.assert_array_equal(ss, s)
        self.assertEqual(c.read_code(0, 0), 0)
        self.assertEqual(c.read_code(0, 1), 8)
        self.assertEqual(c.read_code(0, 32), 8)

    def test_malformed_or_tampered_frame_fails_closed(self):
        b, s = fixture()
        frame = r.pack(b, s).payload
        mutations = [frame[:15], frame[:-1], b"ABCD" + frame[4:],
                     frame[:r.HEADER.size] + frame[r.HEADER.size:-1] + b"\0"]
        # Last mutation must actually change one original nibble for a
        # deterministic digest failure (irrespective of the old last byte).
        tampered = bytearray(frame)
        tampered[-1] ^= 1
        mutations[-1] = bytes(tampered)
        for mutated in mutations:
            with self.subTest(length=len(mutated)), self.assertRaises(ValueError):
                r.load(mutated)

    def test_input_shape_bounds_and_bad_read_indices(self):
        b, s = fixture(8, 4)
        for invalid in (s[:2], s.astype(np.int8)):
            with self.assertRaises((ValueError, TypeError)):
                r.pack(b, invalid)
        c = r.pack(b, s)
        for row, group in ((-1, 0), (8, 0), (0, 4)):
            with self.assertRaises(IndexError):
                c.read_scale(row, group)
        for col in (-1, 128, True):
            with self.assertRaises(IndexError):
                c.read_code(0, col)


if __name__ == "__main__":
    unittest.main()
