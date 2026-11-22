"""R12 fail-closed coverage tests using ONLY synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.zipburger import coverage_r12 as c
from experiments.zipburger import layer_trace_r09 as r


class CoverageR12Tests(unittest.TestCase):
    def prepare(self, root):
        pairs = []
        for label in ("cal", "holdout"):
            n = 8
            corpus = hashlib.sha256(label.encode()).hexdigest()
            checkpoint = hashlib.sha256(b"fixture-model").hexdigest()
            sample_ids = np.arange(n, dtype=np.int64)
            layers = tuple(np.full((n, 512), (i + 1) / 100, np.float32)
                           for i in range(r.LAYERS))
            routes = tuple(np.tile(np.arange(4, dtype=np.int16), (n, 1))
                           for _ in range(r.LAYERS))
            trace = r.Trace(sample_ids, layers, routes,
                            np.arange(n, dtype=np.int64), corpus, checkpoint,
                            "instrumented_teacher")
            trace_file = root / f"{label}.npz"
            r.save(trace, trace_file)
            expert_file = root / f"{label}_expert_upgate.npz"
            expert_data = {
                f"layer_{i:02d}_expert_{k:02d}":
                    np.full((n, 512), (i + k + 1) / 100, dtype=np.float32)
                for i in range(r.LAYERS) for k in range(4)
            }
            np.savez_compressed(expert_file, **expert_data)
            meta = {
                "schema": "zipburger-r09-expert-inputs-v1",
                "sha256": hashlib.sha256(expert_file.read_bytes()).hexdigest(),
                "prompt_digest": corpus,
                "checkpoint_digest": checkpoint,
                "source": "instrumented_teacher",
                "input_width": 512,
                "projections": ["up", "gate"],
                "contains_down_projection_inputs": False,
                "token_sequence_sha256": [corpus],
            }
            expert_file.with_suffix(".json").write_text(json.dumps(meta),
                                                       encoding="utf-8")
            pairs.append((trace_file, expert_file))
        return pairs

    def test_limited_coverage_is_reported_and_never_authorizes_lossy(self):
        with tempfile.TemporaryDirectory() as tmp:
            (cal, c_archive), (val, v_archive) = self.prepare(Path(tmp))
            result = c.report(cal, val, c_archive, v_archive)
            self.assertEqual(result["eligible_layer_expert_pairs"], 24 * 4)
            self.assertEqual(result["total_layer_expert_pairs"], 768)
            self.assertEqual(result["eligible_percent"], 12.5)
            self.assertEqual(result["fully_covered_layers"], 0)
            self.assertEqual(result["layers"][0]["calibration_routing_entropy_bits"], 2)
            self.assertFalse(result["lossy_compression_authorized"])
            self.assertFalse(result["model_agentic_equivalence_verified"])

    def test_minimum_sample_gate_can_reject_all_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            (cal, c_archive), (val, v_archive) = self.prepare(Path(tmp))
            result = c.report(cal, val, c_archive, v_archive, min_samples=9)
            self.assertEqual(result["eligible_layer_expert_pairs"], 0)

    def test_mismatched_manifest_trace_digest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            (cal, c_archive), (val, v_archive) = self.prepare(Path(tmp))
            manifest_path = v_archive.with_suffix(".json")
            meta = json.loads(manifest_path.read_text())
            meta["prompt_digest"] = hashlib.sha256(b"other").hexdigest()
            manifest_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not aligned"):
                c.report(cal, val, c_archive, v_archive)

    def test_missing_routed_activations_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            (cal, c_archive), (val, v_archive) = self.prepare(Path(tmp))
            with np.load(v_archive, allow_pickle=False) as f:
                data = {name: np.array(f[name]) for name in f.files}
            data.pop("layer_03_expert_02")
            np.savez_compressed(v_archive, **data)
            manifest_path = v_archive.with_suffix(".json")
            meta = json.loads(manifest_path.read_text())
            meta["sha256"] = hashlib.sha256(v_archive.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete/misaligned"):
                c.report(cal, val, c_archive, v_archive)


if __name__ == "__main__":
    unittest.main()
