"""CPU-only R10 native GGUF callback binary fixture tests (NO model inference)."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from experiments.zipburger import gguf_capture_r10 as r
from experiments.zipburger import layer_trace_r09 as tr
from experiments.zipburger import expert_trace_input_r09 as archive


def fixture(path, *, seed=10, tokens=8, width=512, broken_topk=False,
            broken_weights=False, broken_shape=False):
    rng = np.random.default_rng(seed)
    header = r.HEADER.pack(r.MAGIC, tokens, width + (1 if broken_shape else 0),
                           r.LAYERS, r.TOPK, r.EXPERTS)
    data = bytearray(header)
    for layer in range(r.LAYERS):
        hidden = rng.standard_normal((tokens, width)).astype("<f4")
        inp = rng.standard_normal((tokens, width)).astype("<f4")
        routes = np.tile(np.arange(4, dtype="<i4"), (tokens, 1))
        if broken_topk and layer == 0:
            routes[0, 0] = 99
        logits = np.zeros((tokens, r.EXPERTS), dtype="<f4")
        weights = np.full((tokens, 4), .25, dtype="<f4")
        if broken_weights and layer == 0:
            weights[0, 0] = 0
        for array in (hidden, inp, routes, logits, weights):
            data.extend(array.tobytes())
    data.extend(np.arange(tokens, dtype="<i4").tobytes())
    data.extend(np.arange(seed, seed + tokens, dtype="<i4").tobytes())
    Path(path).write_bytes(data)


class GGUFNativeR10Tests(unittest.TestCase):
    def test_parse_strict_all_24_layers_router_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "teacher.zbr10"
            fixture(file)
            obj = r.load(file)
            self.assertEqual(obj.tokens, 8)
            self.assertEqual(obj.width, 512)
            self.assertEqual(len(obj.hidden), 24)
            self.assertEqual(obj.expert_inputs[4].shape, (8, 512))
            self.assertEqual(obj.router_logits[0].shape, (8, 32))
            self.assertEqual(obj.router_weights[15].shape, (8, 4))
            self.assertEqual(obj.routes[23][0].tolist(), [0, 1, 2, 3])
            self.assertEqual(len(obj.token_sequence_sha256), 64)

    def test_reject_truncation_novel_header_router_duplicate_invalid_prob(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "test.zbr10"
            fixture(file)
            data = file.read_bytes()
            for payload in (data[:-1], b"BROKEN!!" + data[8:],
                            data[:8] + b"\xff\xff\xff\xff" + data[12:]):
                file.write_bytes(payload)
                with self.assertRaises(ValueError):
                    r.load(file)
            fixture(file, broken_topk=True)
            with self.assertRaises(ValueError):
                r.load(file)
            fixture(file, broken_weights=True)
            with self.assertRaises(ValueError):
                r.load(file)
            fixture(file, broken_shape=True)
            with self.assertRaises(ValueError):
                r.load(file)

    def test_two_real_format_archives_disjoint_and_r09_consumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.zbr10"
            b = Path(tmp) / "b.zbr10"
            fixture(a, seed=41)
            fixture(b, seed=73)
            root = Path(tmp)
            checkpoint = hashlib.sha256(b"operator checkpoint").hexdigest()
            meta_a = r.save_bundle([a], trace_path=root / "cal.npz",
                                   checkpoint_sha256=checkpoint,
                                   source="instrumented_teacher")
            meta_b = r.save_bundle([b], trace_path=root / "val.npz",
                                   checkpoint_sha256=checkpoint,
                                   source="instrumented_teacher")
            self.assertNotEqual(meta_a["token_sequence_sha256"],
                                meta_b["token_sequence_sha256"])
            teacher = tr.load(root / "cal.npz")
            self.assertEqual(len(teacher.layers), 24)
            self.assertEqual(teacher.sample_ids.shape, (8,))
            self.assertEqual(teacher.routes[0].shape, (8, 4))
            ca, vb = archive.load_pair(root / "cal_expert_upgate.npz",
                                       root / "val_expert_upgate.npz")
            x, y = archive.matched_vectors(ca, vb,
                                           tensor="blk.0.ffn_gate_exps.weight",
                                           row_start=0)
            self.assertEqual(x.shape, (8, 512))
            self.assertEqual(y.shape, (8, 512))
            self.assertNotEqual(meta_a["trace_sha256"], meta_b["trace_sha256"])

    def test_duplicate_token_sequence_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.zbr10"
            b = Path(tmp) / "b.zbr10"
            fixture(a, seed=10)
            fixture(b, seed=10)
            checkpoint = hashlib.sha256(b"fake").hexdigest()
            with self.assertRaises(ValueError):
                r.aggregate([a, b], checkpoint_sha256=checkpoint,
                            source="instrumented_teacher")

    def test_unattested_checkpoint_and_invalid_role_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.zbr10"
            fixture(a)
            with self.assertRaises(ValueError):
                r.aggregate([a], checkpoint_sha256="1234",
                            source="instrumented_teacher")
            with self.assertRaises(ValueError):
                r.aggregate([a], checkpoint_sha256="a" * 64,
                            source="synthetic_fixture")

if __name__ == "__main__":
    unittest.main()
