"""R09 agentic outcome tests with simulated task records (not model trials)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.zipburger import agentic_gate_r09 as a


class AgenticR09Tests(unittest.TestCase):
    def example(self, suffix):
        cases = []
        for idx, family in enumerate(sorted(a.ALLOWED_FAMILIES)):
            cases.append({
                "case_id": f"{family}-01", "family": family,
                "environment_sha256": hashlib.sha256(f"env:{idx}".encode()).hexdigest(),
                "completed": True, "unsafe_actions": 0,
                "recovered_failures": 1 if family == "error_recovery" else 0,
                "tool_calls": ["inspect", "run_check"] if suffix == 0
                              else ["run_check", "inspect"],
            })
        return cases

    def test_tool_sequences_can_change_without_losing_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            x, y = Path(tmp) / "teacher.jsonl", Path(tmp) / "candidate.jsonl"
            x.write_text("\n".join(json.dumps(v) for v in self.example(0)), encoding="utf-8")
            y.write_text("\n".join(json.dumps(v) for v in self.example(1)), encoding="utf-8")
            report = a.compare(a.load_cases(x), a.load_cases(y), min_cases=4)
            self.assertTrue(report["all_agentic_outcome_gates_pass"])
            self.assertEqual(report["different_tool_trajectories"], 4)
            self.assertFalse(report["model_quality_or_agenticity_proven"])

    def test_regression_caught_even_when_success_count_elsewhere_increases(self):
        baseline = {v["case_id"]: v for v in self.example(0)}
        newer = {v["case_id"]: dict(v) for v in self.example(0)}
        newer["planning-01"]["completed"] = False
        newer["error_recovery-01"]["recovered_failures"] = 0
        newer["tool_use-01"]["unsafe_actions"] = 1
        report = a.compare(baseline, newer, min_cases=4)
        self.assertFalse(report["all_agentic_outcome_gates_pass"])
        self.assertEqual(report["regressed_cases"], ["planning-01"])
        self.assertEqual(report["increased_unsafe_cases"], ["tool_use-01"])
        self.assertEqual(report["regressed_error_recovery_cases"], ["error_recovery-01"])

    def test_mismatched_environment_fails_closed(self):
        baseline = {v["case_id"]: v for v in self.example(0)}
        newer = {k: dict(v) for k, v in baseline.items()}
        newer["planning-01"]["environment_sha256"] = hashlib.sha256(b"changed").hexdigest()
        with self.assertRaises(ValueError):
            a.compare(baseline, newer, min_cases=4)

    def test_raw_args_and_unexpected_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "a.jsonl"
            datum = self.example(0)[0]
            datum["raw_prompt"] = "not allowed in metrics"
            file.write_text(json.dumps(datum), encoding="utf-8")
            with self.assertRaises(ValueError):
                a.load_cases(file)

    def test_default_minimum_cases_is_not_waived(self):
        baseline = {v["case_id"]: v for v in self.example(0)}
        with self.assertRaises(ValueError):
            a.compare(baseline, baseline)

if __name__ == "__main__":
    unittest.main()
