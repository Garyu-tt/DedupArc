"""R09 CPU-only regression: trace integrity, 24-layer drift, routed expert IO."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.zipburger import layer_trace_r09 as r
from experiments.zipburger.hf_capture_r09 import select_expert_inputs, _digest_prompts


class TraceR09Tests(unittest.TestCase):
    def trace(self, *, source="synthetic_fixture", noise=0.0, route_offset=0,
              pred_offset=0, sample_offset=0, seed=12):
        rng = np.random.default_rng(seed)
        n, width = 8, 128
        h = tuple((rng.standard_normal((n, width)) + noise).astype(np.float32)
                  for _ in range(r.LAYERS))
        routes = tuple(np.broadcast_to(
            ((np.arange(4, dtype=np.int16) + i + route_offset) % 32)[None, :],
            (n, 4)).copy() for i in range(r.LAYERS))
        return r.Trace(np.arange(n, dtype=np.int64) + sample_offset,
                       h, routes, np.arange(n, dtype=np.int64) + pred_offset,
                       hashlib.sha256(b"prompt-manifest-v1").hexdigest(),
                       hashlib.sha256(b"checkpoint").hexdigest(), source)

    def test_24_layer_exact_self_pair_passes_but_not_agenticity(self):
        original = self.trace()
        report = r.compare(original, original)
        self.assertTrue(report["all_24_layer_gates_pass"])
        self.assertEqual(len(report["layers"]), 24)
        self.assertEqual(report["top1_agreement"], 1)
        self.assertFalse(report["agenticity_evaluated"])
        self.assertFalse(report["genuine_model_traces"])

    def test_layer_drift_and_router_mismatch_fail_closed(self):
        a = self.trace()
        b = self.trace(noise=0.1)
        bad = r.compare(a, b)
        self.assertFalse(bad["all_24_layer_gates_pass"])
        self.assertGreater(bad["max_layer_relative_l2"], .04)
        c = self.trace(route_offset=1)
        bad_router = r.compare(a, c)
        self.assertFalse(bad_router["all_24_layer_gates_pass"])
        self.assertEqual(bad_router["layers"][0]["routing_top4_exact_fraction"], 0)

    def test_prediction_mismatch_fail_closed(self):
        a = self.trace()
        b = self.trace(pred_offset=1)
        report = r.compare(a, b)
        self.assertEqual(report["top1_agreement"], 0)
        self.assertFalse(report["all_24_layer_gates_pass"])

    def test_non_aligned_samples_refused(self):
        a = self.trace()
        with self.assertRaises(ValueError):
            r.compare(a, self.trace(sample_offset=20))
        alternate = r.Trace(a.sample_ids, a.layers, a.routes, a.predictions,
                            hashlib.sha256(b"another prompt").hexdigest(),
                            a.checkpoint_digest, a.source)
        with self.assertRaises(ValueError):
            r.compare(a, alternate)

    def test_roundtrip_manifest_and_tamper_rejection(self):
        a = self.trace()
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "teacher.npz"
            manifest = r.save(a, file)
            self.assertEqual(manifest["sample_count"], 8)
            b = r.load(file)
            np.testing.assert_array_equal(a.layers[12], b.layers[12])
            np.testing.assert_array_equal(a.routes[12], b.routes[12])
            file.write_bytes(file.read_bytes() + b"tampering")
            with self.assertRaises(ValueError):
                r.load(file)

    def test_schema_bounds_and_finite_checks(self):
        a = self.trace()
        invalid = r.Trace(a.sample_ids, a.layers[:-1], a.routes,
                          a.predictions, a.prompt_digest,
                          a.checkpoint_digest, a.source)
        with self.assertRaises(ValueError):
            r._validate(invalid)
        h = list(a.layers)
        h[4] = h[4].copy()
        h[4][0, 0] = np.nan
        invalid = r.Trace(a.sample_ids, tuple(h), a.routes, a.predictions,
                          a.prompt_digest, a.checkpoint_digest, a.source)
        with self.assertRaises(ValueError):
            r._validate(invalid)
        routes = list(a.routes)
        routes[0] = routes[0].copy()
        routes[0][0, 1] = routes[0][0, 0]
        invalid = r.Trace(a.sample_ids, a.layers, tuple(routes), a.predictions,
                          a.prompt_digest, a.checkpoint_digest, a.source)
        with self.assertRaises(ValueError):
            r._validate(invalid)

    def test_only_selected_expert_inputs_collected(self):
        x = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
        routes = np.array([[0,1,2,3],[4,5,6,7],[3,8,9,10],[31,0,1,2]],
                          dtype=np.int16)
        actual = select_expert_inputs(x, routes, 3)
        np.testing.assert_array_equal(actual, x[[0, 2]])
        self.assertEqual(select_expert_inputs(x, routes, 30).shape, (0, 512))
        with self.assertRaises(ValueError):
            select_expert_inputs(x, routes[:, :3], 3)

    def test_prompt_fingerprint_stable_and_order_sensitive(self):
        self.assertEqual(_digest_prompts(["a", "b"]), _digest_prompts(["a", "b"]))
        self.assertNotEqual(_digest_prompts(["a", "b"]), _digest_prompts(["b", "a"]))

if __name__ == "__main__":
    unittest.main()
