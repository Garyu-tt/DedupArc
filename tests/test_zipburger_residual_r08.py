"""R08B CPU adapter correctness, size accounting, and quality gates."""
import unittest
import numpy as np
from experiments.zipburger import residual_adapter_r08 as r
from experiments.zipburger import palette3_r08 as p

class ResidualR08Tests(unittest.TestCase):
    def test_trains_fp16_residual_and_counts_all_bytes(self):
        rng = np.random.default_rng(31)
        b = rng.integers(0,256,(32,8,16),dtype=np.uint8)
        s = rng.integers(120,132,(32,8),dtype=np.uint8)
        a = rng.standard_normal((32,256))
        v = rng.standard_normal((16,256))
        candidate, report = r.measure(b,s,a,v,rank=1,max_l2=1,min_cosine=0)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.left.dtype,np.float16)
        self.assertEqual(candidate.right.dtype,np.float16)
        self.assertEqual(report["corrected_bytes"],
                         p.HEADER.size + 32*8*13 + 8 + (32+256)*2)
        self.assertLessEqual(report["corrected_l2"], report["plain_l2"] + 0.05)
        self.assertTrue(np.all(np.isfinite(candidate.dot(v[0]))))

    def test_rejects_oversized_rank_and_invalid_rank(self):
        rng = np.random.default_rng(50)
        b = rng.integers(0,256,(8,4,16),dtype=np.uint8)
        s = np.full((8,4),127,dtype=np.uint8)
        a = rng.standard_normal((16,128))
        v = rng.standard_normal((8,128))
        carrier, report = r.measure(b,s,a,v,rank=4,max_l2=1,min_cosine=0)
        self.assertIsNone(carrier)
        self.assertFalse(report["accepted_after_quality_and_size_gate"])
        with self.assertRaises(ValueError):
            r.fit(b,s,a,rank=0)

    def test_fails_closed_on_wrong_vector_length(self):
        rng = np.random.default_rng(10)
        b = rng.integers(0,256,(16,4,16),dtype=np.uint8)
        s = np.full((16,4),127,dtype=np.uint8)
        a = rng.standard_normal((16,128))
        carrier = r.fit(b,s,a,rank=1)
        with self.assertRaises(ValueError):
            carrier.dot(np.ones(12))

if __name__ == "__main__":
    unittest.main()
