"""CPU-native R04 direct GEMV vs real-valued oracle, no GPU or model load."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from experiments.zipburger import mxfp4_palette_r04 as r
from experiments.zipburger.mxfp4_r03 import dense_reference_dot


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"),
                     "native C11 regression requires Linux C compiler")
class NativeGEMV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        source = (Path(__file__).resolve().parents[1] /
                  "experiments/zipburger/native_r04.c")
        output = Path(cls.tmp.name) / "libzipburger_r04.so"
        subprocess.run(["cc", "-O3", "-fPIC", "-shared",
                        "-std=c11", "-Wall", "-Wextra", "-Werror",
                        str(source), "-o", str(output), "-lm"],
                       check=True, capture_output=True, text=True, timeout=30)
        cls.lib = ctypes.CDLL(str(output))
        cls.fn = cls.lib.zipburger_r04_gemv
        cls.fn.argtypes = (
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        )
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def call_native(self, payload, x, rows, *, size_override=None):
        data = ctypes.create_string_buffer(payload, len(payload))
        vec = np.ascontiguousarray(x, dtype=np.float64)
        out = np.zeros(rows, dtype=np.float64)
        rc = self.fn(data, len(payload) if size_override is None else size_override,
                     vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), vec.size,
                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rows)
        return rc, out

    def test_raw_delta_and_palette_are_numerically_exact(self):
        rng = np.random.default_rng(144)
        for kind in ("raw", "delta", "palette"):
            with self.subTest(kind=kind):
                b = rng.integers(0, 256, (48, 16, 16), dtype=np.uint8)
                if kind == "raw":
                    s = np.resize(np.arange(256, dtype=np.uint8),
                                  48 * 16).reshape(48, 16)
                elif kind == "delta":
                    s = rng.integers(114, 134, (48, 16), dtype=np.uint8)
                else:
                    s = rng.choice(np.array([0, 127, 255], np.uint8),
                                   size=(48, 16))
                c = r.pack(b, s)
                self.assertEqual(
                    c.mode, {"raw": "raw-fallback",
                             "delta": "delta-bitpack",
                             "palette": "palette-bitpack"}[kind])
                x = rng.standard_normal(16 * 32)
                code, y = self.call_native(c.payload, x, 48)
                self.assertEqual(code, 0)
                np.testing.assert_allclose(
                    y, dense_reference_dot(b, s, x),
                    rtol=1e-12, atol=1e-8)
                np.testing.assert_allclose(y, c.dot(x),
                                           rtol=1e-12, atol=1e-8)

    def test_rejects_wrong_magic_short_buffer_and_nonfinite_input(self):
        b = np.zeros((3, 2, 16), dtype=np.uint8)
        s = np.full((3, 2), 127, dtype=np.uint8)
        c = r.pack(b, s)
        x = np.ones(64)
        code, _ = self.call_native(c.payload, x, 3, size_override=12)
        self.assertLess(code, 0)
        code, _ = self.call_native(b"NOPE" + c.payload[4:], x, 3)
        self.assertLess(code, 0)
        x[0] = np.nan
        code, _ = self.call_native(c.payload, x, 3)
        self.assertLess(code, 0)

    def test_small_benchmark_no_performance_assertion(self):
        rng = np.random.default_rng(333)
        b = rng.integers(0, 256, (48, 16, 16), dtype=np.uint8)
        s = rng.integers(120, 131, (48, 16), dtype=np.uint8)
        c = r.pack(b, s)
        x = rng.standard_normal(512)
        for _ in range(5):
            status, out = self.call_native(c.payload, x, 48)
            self.assertEqual(status, 0)
        loops = 25
        begin = time.perf_counter_ns()
        for _ in range(loops):
            self.call_native(c.payload, x, 48)
        native_us = (time.perf_counter_ns() - begin) / loops / 1000
        begin = time.perf_counter_ns()
        for _ in range(loops):
            c.dot(x)
        python_us = (time.perf_counter_ns() - begin) / loops / 1000
        print(f"R04_NATIVE_GEMV_MICROBENCH: native={native_us:.2f}us "
              f"python={python_us:.2f}us rows=48 groups=16 "
              "single-thread, excludes model loading, not LLM throughput")


if __name__ == "__main__":
    unittest.main()
