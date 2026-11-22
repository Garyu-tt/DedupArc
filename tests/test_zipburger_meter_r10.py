"""R10 Linux metering without running a real GGUF or claiming battery power."""
from pathlib import Path
import os
import stat
import tempfile
import unittest

from experiments.zipburger import meter_r10 as meter


class MeterR10Tests(unittest.TestCase):
    def stub(self, root, *, sleep=False):
        exe = root / "stub_executable.py"
        exe.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib,sys,time\n"
            + ("time.sleep(3)\n" if sleep else "")
            + "pathlib.Path(sys.argv[sys.argv.index('--out')+1]).write_bytes(b'fixture')\n"
              "print('R10 capture complete: 3 tokens, 24 layers, hidden=512.')\n",
            encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        model = root / "test.gguf"
        model.write_bytes(b"mock only; no actual model")
        prompt = root / "prompt.txt"
        prompt.write_text("stub", encoding="utf-8")
        return exe, model, prompt

    def test_native_stub_reports_cpu_only_cold_start_not_tokens_per_sec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, model, prompt = self.stub(root)
            result = meter.run(exe, model, prompt, root / "capture.zbr10",
                               threads=1, timeout=10)
            self.assertTrue(result["success"])
            self.assertEqual(result["native_reported_tokens"], 3)
            self.assertIsNone(result["cpu_package_energy_joules_optional"])
            self.assertFalse(result["complete_device_energy_included"])
            self.assertFalse(result["model_level_per_token_inference_speed_measured"])
            self.assertGreater(result["wall_seconds_cold_start_including_model_load"], 0)
            with self.assertRaises(ValueError):
                meter.run(exe, model, prompt, root / "capture.zbr10",
                          threads=1, timeout=10)

    def test_timeout_terminates_single_child_and_refuses_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, model, prompt = self.stub(root, sleep=True)
            result = meter.run(exe, model, prompt, root / "capture.zbr10",
                               threads=1, timeout=1)
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["success"])
            self.assertEqual(result["error_class"], "timeout")

    def test_rapl_wrap_counter_is_optional_and_accounted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "max_energy_range_uj").write_text("100", encoding="utf-8")
            (root / "energy_uj").write_text("90", encoding="utf-8")
            counter = meter.EnergyCounter(root)
            (root / "energy_uj").write_text("5", encoding="utf-8")
            counter.sample()
            self.assertAlmostEqual(counter.measured_joules(), 15e-6, places=8)
            (root / "energy_uj").write_text("10", encoding="utf-8")
            counter.sample()
            self.assertAlmostEqual(counter.measured_joules(), 20e-6, places=8)

    def test_bounds_refuse_nonexistent_sources_and_invalid_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, model, prompt = self.stub(root)
            with self.assertRaises(ValueError):
                meter.run(exe, model, prompt, root / "capture.zbr10", timeout=0)
            with self.assertRaises(ValueError):
                meter.run(root / "missing", model, prompt, root / "capture.zbr10")

if __name__ == "__main__":
    unittest.main()
