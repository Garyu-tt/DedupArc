"""R09 operator-provenance expert input archive and real row alignment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experiments.zipburger import expert_trace_input_r09 as r
from experiments.zipburger import bench_layerwise_r08 as bench


class ExpertArchiveR09Tests(unittest.TestCase):
    def make_archive(self, path, *, prompt=b"cal", source="instrumented_teacher"):
        rng = np.random.default_rng(18 if prompt == b"cal" else 19)
        values = rng.standard_normal((12, 512)).astype(np.float32)
        np.savez_compressed(path, layer_00_expert_00=values,
                            layer_00_expert_01=values * .3)
        meta = {
            "schema": "zipburger-r09-expert-inputs-v1",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_digest": hashlib.sha256(prompt).hexdigest(),
            "token_sequence_sha256": [hashlib.sha256(prompt).hexdigest()],
            "checkpoint_digest": hashlib.sha256(b"checkpoint").hexdigest(),
            "source": source,
            "input_width": 512, "projections": ["up", "gate"],
            "contains_down_projection_inputs": False,
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return values

    def test_calibration_validation_pair_and_matching_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal, val = Path(tmp) / "cal.npz", Path(tmp) / "val.npz"
            a = self.make_archive(cal)
            b = self.make_archive(val, prompt=b"heldout")
            x, y = r.load_pair(cal, val)
            a2, b2 = r.matched_vectors(x, y, tensor="blk.0.ffn_gate_exps.weight",
                                       row_start=0)
            np.testing.assert_array_equal(a2, a)
            np.testing.assert_array_equal(b2, b)
            a3, b3 = r.matched_vectors(x, y, tensor="blk.0.ffn_up_exps.weight",
                                       row_start=4096)
            np.testing.assert_array_equal(a3, a * .3)
            np.testing.assert_array_equal(b3, b * .3)
            self.assertIsNone(r.matched_vectors(x, y, tensor="blk.3.ffn_up_exps.weight",
                                                 row_start=0))
            with self.assertRaises(ValueError):
                r.matched_vectors(x, y, tensor="blk.0.ffn_down_exps.weight",
                                  row_start=0)
            with self.assertRaises(ValueError):
                r.matched_vectors(x, y, tensor="blk.0.ffn_gate_exps.weight",
                                  row_start=2860)

    def test_same_prompt_fingerprint_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal, val = Path(tmp) / "cal.npz", Path(tmp) / "val.npz"
            self.make_archive(cal)
            self.make_archive(val)
            with self.assertRaises(ValueError):
                r.load_pair(cal, val)

    def test_overlap_refused_even_when_corpus_manifests_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal, val = Path(tmp) / "cal.npz", Path(tmp) / "val.npz"
            self.make_archive(cal)
            self.make_archive(val, prompt=b"heldout")
            manifest = json.loads(val.with_suffix(".json").read_text(encoding="utf-8"))
            manifest["token_sequence_sha256"] = [hashlib.sha256(b"cal").hexdigest()]
            val.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                r.load_pair(cal, val)

    def test_bad_provenance_and_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "cal.npz"
            self.make_archive(file, source="synthetic_fixture")
            with self.assertRaises(ValueError):
                r.read_archive(file)
            self.make_archive(file)
            file.write_bytes(file.read_bytes() + b"evil")
            with self.assertRaises(ValueError):
                r.read_archive(file)

    def test_benchmark_accepts_explicit_vectors_without_fake_activations(self):
        rng = np.random.default_rng(99)
        b = rng.integers(0, 256, (8, 4, 16), np.uint8)
        s = np.full((8, 4), 127, np.uint8)
        cal = rng.standard_normal((12, 128)).astype(np.float32)
        val = rng.standard_normal((8, 128)).astype(np.float32)
        report = bench.benchmark_one("blk.0.ffn_up_exps.weight", b, s,
                                     real=True, calibration=cal, validation=val)
        self.assertTrue(report["real_model_activations_operator_attested"])
        self.assertFalse(report["real_model_activations_independently_verified"])
        with self.assertRaises(ValueError):
            bench.benchmark_one("blk.0.ffn_up_exps.weight", b, s,
                                calibration=cal, validation=None)

if __name__ == "__main__":
    unittest.main()
