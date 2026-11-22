"""CPU-only exact MXFP4 entropy structural profiler tests."""
import unittest
import numpy as np
from experiments.zipburger.alt120b_entropy_r07 import _entropy,profile


class EntropyR07Tests(unittest.TestCase):
    def test_all_zero_symbols_have_zero_entropy_and_repeated_blocks(self):
        b=np.zeros((8,4,16),np.uint8)
        s=np.full((8,4),127,np.uint8)
        d=profile(b,s)
        self.assertEqual(d["scale_symbol_entropy_bits"],0)
        self.assertEqual(d["nibble_symbol_entropy_bits"],0)
        self.assertEqual(d["mantissa_unique_16byte_blocks"],1)
        self.assertEqual(d["mantissa_exact_duplicate_blocks"],31)
        self.assertEqual(d["mantissa_strided_equal_blocks"][1],31)

    def test_uniform_nibbles_are_near_four_bits(self):
        b=np.arange(256,dtype=np.uint8).reshape(1,16,16)
        s=np.arange(16,dtype=np.uint8).reshape(1,16)
        d=profile(b,s)
        self.assertAlmostEqual(d["nibble_symbol_entropy_bits"],4,places=4)
        self.assertAlmostEqual(d["scale_symbol_entropy_bits"],4,places=4)
        self.assertEqual(d["mantissa_exact_duplicate_blocks"],0)

    def test_strided_repeats_are_detected_even_when_nonadjacent(self):
        b=np.zeros((3,4,16),np.uint8)
        b.reshape(-1,16)[::2,:]=0x33
        b.reshape(-1,16)[1::2,:]=0xCC
        s=np.full((3,4),127,np.uint8)
        d=profile(b,s)
        self.assertEqual(d["mantissa_strided_equal_blocks"][1],0)
        self.assertEqual(d["mantissa_strided_equal_blocks"][2],10)
        self.assertTrue(d["mantissa_xor_stride_zlib9"][2]["zlib9_bytes"]>0)

    def test_random_high_entropy_still_bounded_and_read_only(self):
        rng=np.random.default_rng(23)
        b=rng.integers(0,256,(48,16,16),np.uint8)
        s=rng.integers(112,136,(48,16),np.uint8)
        original=b.copy()
        d=profile(b,s)
        self.assertLessEqual(d["nibble_symbol_entropy_bits"],4)
        self.assertGreaterEqual(d["nibble_symbol_entropy_bits"],3.9)
        self.assertTrue(d["single_symbol_entropy_is_not_full_information_bound"])
        np.testing.assert_array_equal(b,original)

    def test_bad_shapes_rejected(self):
        for b,s in (
            (np.zeros((0,4,16),np.uint8),np.zeros((0,4),np.uint8)),
            (np.zeros((4,4,16),np.uint8),np.zeros((4,3),np.uint8)),
            (np.zeros((4,4,17),np.uint8),np.zeros((4,4),np.uint8)),
            (np.zeros((4,4,16),np.int8),np.zeros((4,4),np.uint8)),
        ):
            with self.subTest(shape=b.shape),self.assertRaises((ValueError,TypeError)):
                profile(b,s)


if __name__=="__main__":
    unittest.main()
