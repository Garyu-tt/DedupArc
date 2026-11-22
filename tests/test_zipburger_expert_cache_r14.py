"""R14 cache paging LRU simulation fixtures, no real GGUF inference."""
import unittest

import numpy as np

from experiments.zipburger import expert_cache_r14 as c


class CacheR14Tests(unittest.TestCase):
    def test_cache_size_effect_on_storage_reads(self):
        routes = c.synth_routes(tokens=18, seed=42)
        small = c.simulate(routes, budget_bytes=24 * c.EXPERT_BYTES)
        large = c.simulate(routes, budget_bytes=768 * c.EXPERT_BYTES)
        self.assertTrue(small["simulation_possible_under_budget"])
        self.assertEqual(small["expert_accesses"], 18 * 24 * 4)
        self.assertGreater(small["cache_evictions"], 0)
        self.assertEqual(large["cache_evictions"], 0)
        self.assertGreaterEqual(large["cache_hit_fraction"],
                                small["cache_hit_fraction"])
        self.assertGreater(small["raw_expert_bytes_read_from_storage"],
                           large["raw_expert_bytes_read_from_storage"])
        self.assertFalse(small["actual_model_inference_measured"])
        self.assertFalse(small["device_energy_or_latency_measured"])

    def test_zero_budget_expert_page_is_fail_closed(self):
        result = c.simulate(c.synth_routes(tokens=1),
                            budget_bytes=c.EXPERT_BYTES - 1)
        self.assertFalse(result["simulation_possible_under_budget"])
        self.assertEqual(result["expert_cache_slots"], 0)

    def test_pinned_20b_expert_shape_accounting(self):
        self.assertEqual(c.EXPERT_BYTES, 13_219_200)
        self.assertEqual(c.TOTAL_EXPERT_BYTES, 10_152_345_600)
        self.assertGreater(c.ASSUMED_NON_EXPERT_FILE_BYTES, 1_900_000_000)
        self.assertLess(c.ASSUMED_NON_EXPERT_FILE_BYTES, 2_100_000_000)

    def test_byte_reserve_reduces_slot_budget(self):
        r = c.synth_routes(tokens=4, seed=7)
        loose = c.evaluate(routes=r, reserve_extra_bytes=0)
        tight = c.evaluate(routes=r, reserve_extra_bytes=4_000_000_000)
        self.assertGreater(loose["expert_cache_slots"],
                           tight["expert_cache_slots"])
        self.assertTrue(loose["only_a_necessary_memory_feasibility_survey"])
        self.assertFalse(loose["agentic_equivalence_proven"])

    def test_invalid_routes_and_budget_rejected(self):
        routes = c.synth_routes(tokens=2)
        malformed = routes.copy()
        malformed[0, 0, 1] = malformed[0, 0, 0]
        for candidate in (malformed, routes.astype(np.int32),
                          routes[:, :, :3], np.ones((0,24,4),np.int16)):
            with self.subTest(shape=candidate.shape), self.assertRaises(ValueError):
                c.simulate(candidate, budget_bytes=10 * c.EXPERT_BYTES)
        with self.assertRaises(ValueError):
            c.simulate(routes, budget_bytes=-1)


if __name__ == "__main__":
    unittest.main()
