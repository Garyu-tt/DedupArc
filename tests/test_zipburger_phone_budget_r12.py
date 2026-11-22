"""R12 disk vs RAM feasibility bounds, no real model inference."""
import unittest
from experiments.zipburger.phone_budget_r12 import bound, MXFP4_SOURCE_BYTES


class PhoneBudgetR12Tests(unittest.TestCase):
    def test_even_unrealistically_compressing_entire_file_cannot_reach_8gb(self):
        report = bound(target_bytes=8_000_000_000, expert_fraction=1,
                       extra_runtime_bytes=0)
        self.assertFalse(report["r08_3bit_physically_can_fit_under_assumptions"])
        self.assertGreater(report["hypothetical_min_model_bytes"], 8_000_000_000)
        self.assertEqual(report["lossy_R08_validation_passes_on_real_20b_fragments"], 0)
        self.assertTrue(report["kv_cache_and_runtime_will_add_overhead"])
        self.assertFalse(report["agentic_equivalence_proven"])

    def test_10gb_optimistic_possible_but_not_proven(self):
        optimistic = bound(target_bytes=10_000_000_000, expert_fraction=1.0)
        self.assertTrue(optimistic["r08_3bit_physically_can_fit_under_assumptions"])
        protected = bound(target_bytes=10_000_000_000, expert_fraction=0.5)
        self.assertFalse(protected["r08_3bit_physically_can_fit_under_assumptions"])
        self.assertTrue(protected["no_real_model_or_phone_runtime_measured"])

    def test_extra_runtime_memory_makes_optimistic_fit_harder(self):
        base = bound(target_bytes=10_000_000_000)
        expensive = bound(target_bytes=10_000_000_000,
                          extra_runtime_bytes=2_000_000_000)
        self.assertLess(base["hypothetical_min_total_bytes"],
                        expensive["hypothetical_min_total_bytes"])
        self.assertFalse(expensive["r08_3bit_physically_can_fit_under_assumptions"])

    def test_invalid_inputs_fail(self):
        for kwargs in ({"expert_fraction":1.1}, {"expert_fraction":-0.1},
                       {"extra_runtime_bytes":-2}, {"target_bytes":0},
                       {"model_bytes":0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bound(**kwargs)


if __name__ == "__main__":
    unittest.main()
