"""Offline GGUF MXFP4 slicing tests; no real checkpoint, network, or GPU."""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.zipburger.gguf_probe_r03 import (
    GGUFIndex, GGUFProbeError, evaluate,
)


def _string(value):
    b = value.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def fixture(path):
    rng = np.random.default_rng(42)
    blocks = rng.integers(0, 256, (12, 4, 16), dtype=np.uint8)
    scales = rng.integers(115, 140, (12, 4), dtype=np.uint8)
    interleaved = np.empty((12, 4, 17), dtype=np.uint8)
    interleaved[:, :, 0] = scales
    interleaved[:, :, 1:] = blocks
    payload = interleaved.tobytes()
    # One 2D MXFP4 tensor with 12 logical rows; one tiny non-MXFP4 tensor.
    meta = (
        _string("general.alignment") + struct.pack("<II", 4, 32)
        + _string("tokenizer.ggml.tokens") + struct.pack("<IQ", 9, 8)
        + struct.pack("<Q", 2) + _string("a") + _string("b")
    )
    tensor1 = (
        _string("blk.0.ffn_gate_exps.weight")
        + struct.pack("<IQQIQ", 2, 128, 12, 39, 0)
    )
    offset2 = (len(payload) + 31) // 32 * 32
    tensor2 = (
        _string("blk.0.attn_norm.weight")
        + struct.pack("<IQQIQ", 2, 8, 1, 0, offset2)
    )
    header = struct.pack("<4sIQQ", b"GGUF", 3, 2, 2) + meta + tensor1 + tensor2
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\x00" * (data_start - len(header)) +
                     payload + b"\x00" * (offset2 - len(payload)) + b"\0" * 32)
    return blocks, scales, payload


class GGUFProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.gguf"
        self.blocks, self.scales, self.payload = fixture(self.path)

    def test_reads_interleaved_scale_first_and_alignment(self):
        with GGUFIndex(self.path) as index:
            self.assertEqual(len(index.tensors), 2)
            b, s, raw, shape = index.sample(
                "blk.0.ffn_gate_exps.weight", rows=5, groups=2, row_start=3)
        np.testing.assert_array_equal(b, self.blocks[3:8, :2])
        np.testing.assert_array_equal(s, self.scales[3:8, :2])
        self.assertEqual(shape, (128, 12))
        self.assertEqual(len(raw), 5 * 2 * 17)

    def test_zipburger_exact_roundtrip_and_float64_oracle(self):
        with GGUFIndex(self.path) as index:
            result = evaluate(index, "blk.0.ffn_gate_exps.weight",
                              rows=8, groups=4, row_start=2)
        self.assertTrue(result["exact_gguf_roundtrip"])
        self.assertTrue(result["direct_float64_match"])
        self.assertEqual(result["source_bytes"], 8 * 4 * 17)
        self.assertLessEqual(result["zipburger_bytes"],
                             result["source_bytes"] + 44)
        self.assertIn(result["zipburger_mode"],
                      ("raw-fallback", "neural-graph"))

    def test_bad_limits_and_non_mxfp4_fail_closed(self):
        with GGUFIndex(self.path) as index:
            for kwargs in ({"rows": 129}, {"groups": 0},
                           {"row_start": -1}, {"rows": True},
                           {"row_start": 10, "rows": 3},
                           {"groups": 5}):
                with self.subTest(kwargs=kwargs), self.assertRaises(GGUFProbeError):
                    index.sample("blk.0.ffn_gate_exps.weight", **kwargs)
            with self.assertRaises(GGUFProbeError):
                index.sample("blk.0.attn_norm.weight")
            with self.assertRaises(GGUFProbeError):
                index.sample("missing")

    def test_bad_magic_version_and_truncation(self):
        data = self.path.read_bytes()
        for changed in (b"NOPE" + data[4:],
                        data[:4] + struct.pack("<I", 2) + data[8:],
                        data[:25]):
            self.path.write_bytes(changed)
            with self.assertRaises(GGUFProbeError):
                with GGUFIndex(self.path):
                    pass

    def test_reject_tensor_offset_past_eof(self):
        # Corrupt the offset of the MXFP4 tensor (not the non-MXFP4 control).
        data = bytearray(self.path.read_bytes())
        needle = _string("blk.0.ffn_gate_exps.weight")
        pos = data.index(needle) + len(needle) + 4 + 16 + 4
        data[pos:pos + 8] = struct.pack("<Q", 2 ** 40)
        self.path.write_bytes(data)
        with GGUFIndex(self.path) as index:
            with self.assertRaises(GGUFProbeError):
                index.sample("blk.0.ffn_gate_exps.weight")
            # Header parser alone must not scan or load tensor payloads.
            self.assertEqual(len(index.tensors), 2)


if __name__ == "__main__":
    unittest.main()
