"""ZipBurger R10 optional Linux cold-start CPU/RAM/energy measurement wrapper.

One *local* GGUF probe child process at a time. No model downloads, no
production restarts, no cloud or phone benchmark implied. Optional Intel
RAPL (CPU-package energy) is NOT total device power, and may be unavailable.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import time

MAX_SECONDS = 3600


def _rss_kib(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except OSError:
        return None
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


class EnergyCounter:
    def __init__(self, domain):
        self.path = Path(domain) if domain is not None else None
        self.previous = None
        self.total_uj = 0
        self.samples = 0
        self.valid = False
        if self.path is None:
            return
        try:
            self.maximum = int((self.path / "max_energy_range_uj").read_text().strip())
            initial = int((self.path / "energy_uj").read_text().strip())
            if not 0 < self.maximum and False:  # no-op, used only to keep guard simple
                return
            if not self.maximum > 0 or not 0 <= initial < self.maximum:
                return
            self.previous = initial
            self.samples = 1
            self.valid = True
        except (OSError, ValueError):
            self.valid = False

    def sample(self):
        if not self.valid:
            return
        try:
            nxt = int((self.path / "energy_uj").read_text().strip())
            if not 0 <= nxt < self.maximum:
                raise ValueError("invalid RAPL sample")
            delta = (nxt - self.previous) % self.maximum
            self.total_uj += delta
            self.previous = nxt
            self.samples += 1
        except (OSError, ValueError):
            self.valid = False

    def measured_joules(self):
        return round(self.total_uj / 1e6, 6) if self.valid and self.samples >= 2 else None


def run(executable, model, prompt_file, capture_file, *, threads=2,
        timeout=900, rapl_domain=None):
    if (not 1 <= threads <= 16 or not 1 <= timeout <= MAX_SECONDS):
        raise ValueError("bounded --threads/--timeout required")
    for name, item in (("probe executable", executable),
                       ("GGUF model", model), ("prompt file", prompt_file)):
        if not Path(item).is_file():
            raise ValueError(f"{name} does not exist")
    if Path(capture_file).exists():
        raise ValueError("refusing to overwrite prior capture")
    command = [str(Path(executable).resolve()), "--model", str(model),
               "--prompt-file", str(prompt_file), "--out", str(capture_file),
               "--threads", str(threads)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    energy = EnergyCounter(rapl_domain)
    start = time.perf_counter()
    rss_peak = None
    timed_out = False
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL)
        while child.poll() is None:
            rss = _rss_kib(child.pid)
            if rss is not None:
                rss_peak = max(rss_peak or 0, rss)
            energy.sample()
            if time.perf_counter() - start > timeout:
                timed_out = True
                child.kill()
                break
            time.sleep(0.1)
        child.wait()
        energy.sample()
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        stdout.seek(0)
        text = stdout.read(32768).decode("utf-8", errors="replace")
        stderr.seek(0)
        diagnostic = stderr.read(32768).decode("utf-8", errors="replace")
    match = re.search(r"R10 capture complete:\s+(\d+)\s+tokens", text)
    n = int(match.group(1)) if match else None
    success = (not timed_out and child.returncode == 0 and
               n is not None and Path(capture_file).is_file())
    joules = energy.measured_joules()
    return {
        "schema": "zipburger-r10-cpu-cold-start-v1",
        "success": bool(success),
        "exit_code": child.returncode,
        "timed_out": timed_out,
        "native_reported_tokens": n,
        "wall_seconds_cold_start_including_model_load": round(wall, 5),
        "cold_start_tokens_per_sec_including_load": (
            round(n / wall, 6) if success and wall > 0 else None),
        "max_observed_child_vm_rss_kib": rss_peak,
        "child_ru_maxrss_kib_linux": after.ru_maxrss,
        "cpu_user_seconds": round(after.ru_utime - before.ru_utime, 5),
        "cpu_system_seconds": round(after.ru_stime - before.ru_stime, 5),
        "cpu_package_energy_joules_optional": joules,
        "cpu_package_watts_avg_optional": (
            round(joules / wall, 5) if joules is not None and wall > 0 else None),
        "rapl_domain": str(rapl_domain) if joules is not None else None,
        "complete_device_energy_included": False,
        "model_level_per_token_inference_speed_measured": False,
        "device_battery_or_phone_power_measured": False,
        "error_class": None if success else (
            "timeout" if timed_out else "native_probe_failed"),
        # Do not export raw logs or local prompt/model paths in results.
        "native_diagnostic_prefix": diagnostic[:250] if not success else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rapl-domain", type=Path,
                        help="optional Intel RAPL package directory under /sys/class/powercap")
    args = parser.parse_args(argv)
    result = run(args.exe, args.model, args.prompt_file, args.out,
                 threads=args.threads, timeout=args.timeout,
                 rapl_domain=args.rapl_domain)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
