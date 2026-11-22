"""C11 native packed 3-bit CPU research tests."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from experiments.zipburger import palette3_r08 as z

C_FILE = Path(__file__).resolve().parents[1] / "experiments/zipburger/native_palette3_r08.c"

@unittest.skipUnless(shutil.which("cc"), "C compiler unavailable")
class NativePalette3R08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        library = Path(cls.tmp.name) / "libzipburger_r08.so"
        subprocess.run(["cc", "-O3", "-fPIC", "-shared", "-std=c11", "-Wall",
                        "-Wextra", "-Werror", str(C_FILE), "-o", str(library), "-lm"],
                       check=True, capture_output=True, text=True)
        cls.fn = ctypes.CDLL(str(library)).zipburger_r08_gemv
        cls.fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def native(self, payload, x, rows):
        frame = ctypes.create_string_buffer(payload, len(payload))
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.zeros(rows, np.float64)
        rc = self.fn(frame, len(payload),
                     x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.size,
                     y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.size)
        return rc, y

    def test_direct_gemv_matches_python_on_random_and_exact_fixtures(self):
        rng = np.random.default_rng(110)
        for exact in (False, True):
            b = rng.integers(0, 256, (48, 16, 16), np.uint8)
            if exact:
                b &= 0x33
            s = rng.integers(120, 133, (48, 16), np.uint8)
            x = rng.standard_normal(512)
            carrier = z.pack(b, s, rng.standard_normal((16, 512)))
            rc, result = self.native(carrier.payload, x, 48)
            self.assertEqual(rc, 0)
            np.testing.assert_allclose(result, carrier.dot(x), rtol=1e-12, atol=1e-9)

    def test_rejects_malformed_length_and_palette(self):
        rng = np.random.default_rng(72)
        b = rng.integers(0, 256, (8, 4, 16), np.uint8)
        s = np.full((8, 4), 127, np.uint8)
        c = z.pack(b, s, rng.standard_normal((16, 128)))
        x = rng.standard_normal(128)
        self.assertNotEqual(self.native(c.payload[:-1], x, 8)[0], 0)
        bad = bytearray(c.payload)
        bad[8] = 255
        self.assertNotEqual(self.native(bytes(bad), x, 8)[0], 0)

if __name__ == "__main__":
    unittest.main()
