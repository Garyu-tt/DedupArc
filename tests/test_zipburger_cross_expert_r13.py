"""R13 independent cross-expert research fixtures, no network/no model."""
import unittest

import numpy as np

from experiments.zipburger import cross_expert_r13 as r


class CrossExpertR13Tests(unittest.TestCase):
    def example(self, seed=21):
        rng = np.random.default_rng(seed)
        blocks = rng.integers(0,256,(16,4,16),dtype=np.uint8)
        scales = rng.integers(118,134,(16,4),dtype=np.uint8)
        return blocks,scales

    def test_identical_experts_share_all_blocks_and_rank1_energy(self):
        original=self.example()
        report=r.examine("blk.0.ffn_gate_exps.weight",[original]*4)
        self.assertEqual(report["exact_identical_17byte_block_fraction_pairwise_mean"],1)
        self.assertAlmostEqual(report["decoded_weight_cosine_pairwise_mean"],1,places=7)
        self.assertAlmostEqual(report["shared_expert_rank1_energy_fraction"],1,places=7)
        self.assertLess(report["lossless_xor_to_expert0_zlib9_bytes"],
                        report["lossless_zlib9_separate_experts_bytes"])
        self.assertFalse(report["candidate_is_an_inference_codec"])

    def test_independent_random_experts_are_not_identical(self):
        tiles=[self.example(i+50) for i in range(4)]
        result=r.examine("blk.2.ffn_up_exps.weight",tiles)
        self.assertEqual(result["expert_count"],4)
        self.assertEqual(result["rows_per_expert"],16)
        self.assertEqual(result["groups_per_row"],4)
        self.assertEqual(result["source_bytes"],4*16*4*17)
        self.assertLess(result["exact_identical_17byte_block_fraction_pairwise_mean"],1)
        self.assertGreater(result["shared_expert_rank1_energy_fraction"],0)
        self.assertFalse(result["whole_model_size_reduction_proven"])
        self.assertFalse(result["model_agentic_equivalence_proven"])

    def test_rejects_misaligned_shapes_and_out_of_bounds(self):
        a=self.example(1)
        b=self.example(2)
        bad=(b[0][:8],b[1][:8])
        with self.assertRaises(ValueError):
            r.examine("blk.0.ffn_down_exps.weight",[a,bad])
        with self.assertRaises(ValueError):
            r.examine("blk.0.ffn_down_exps.weight",[a])
        with self.assertRaises(TypeError):
            r.examine("blk.0.ffn_down_exps.weight",[(a[0].astype(np.int8),a[1]),b])
        with self.assertRaises(ValueError):
            r.slices("blk.0.ffn_down_exps.weight",experts=9)

    def test_synthetic_example_is_deterministic(self):
        a=r.slices("blk.1.ffn_gate_exps.weight",synthetic=True,experts=2)
        b=r.slices("blk.1.ffn_gate_exps.weight",synthetic=True,experts=2)
        for (aa,as_), (ba,bs) in zip(a,b):
            np.testing.assert_array_equal(aa,ba)
            np.testing.assert_array_equal(as_,bs)


if __name__ == "__main__":
    unittest.main()
