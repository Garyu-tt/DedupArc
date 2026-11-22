"""R09 CPU-only agentic trajectory gate for paired executed tasks.

This is an *outcome evaluator*, NOT an LLM runner: it requires actual
recorded teacher/candidate task executions. Tool sequence may differ while
success is preserved; exact imitation is not a requirement of agenticity.
"""
from __future__ import annotations
import json
from pathlib import Path
import re

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_CASES = 1000
ALLOWED_FAMILIES = {"planning", "tool_use", "error_recovery", "long_context"}
_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,96}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def load_cases(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing/oversized trajectory JSONL")
    result = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or set(obj) != {
                "case_id", "family", "environment_sha256", "completed",
                "unsafe_actions", "recovered_failures", "tool_calls"}:
            raise ValueError("unsupported task result schema; no prompt/raw tool data")
        case_id, family = obj["case_id"], obj["family"]
        if (not isinstance(case_id, str) or not _ID.fullmatch(case_id)
                or case_id in result or family not in ALLOWED_FAMILIES
                or not isinstance(obj["environment_sha256"], str)
                or not _DIGEST.fullmatch(obj["environment_sha256"])
                or type(obj["completed"]) is not bool
                or type(obj["unsafe_actions"]) is not int
                or not 0 <= obj["unsafe_actions"] <= 100
                or type(obj["recovered_failures"]) is not int
                or not 0 <= obj["recovered_failures"] <= 100
                or type(obj["tool_calls"]) is not list
                or len(obj["tool_calls"]) > 64
                or any(not isinstance(v, str) or not _ID.fullmatch(v)
                       for v in obj["tool_calls"])):
            raise ValueError("invalid bounded executed task outcome")
        result[case_id] = obj
        if len(result) > MAX_CASES:
            raise ValueError("too many evaluation cases")
    if not result:
        raise ValueError("empty agentic evaluation corpus")
    return result


def compare(teacher, candidate, *, min_cases=32, require_families=None):
    if not isinstance(teacher, dict) or not isinstance(candidate, dict):
        raise ValueError("paired case maps required")
    if teacher.keys() != candidate.keys() or len(teacher) < min_cases:
        raise ValueError("missing/insufficient paired executed cases")
    families = set(v["family"] for v in teacher.values())
    wanted = set(ALLOWED_FAMILIES if require_families is None else require_families)
    if not wanted <= families:
        raise ValueError("missing required evaluation task families")
    regressions, unsafe_increase, recovery_regressions, tool_diff = [], [], [], 0
    for case_id in teacher:
        a, b = teacher[case_id], candidate[case_id]
        if (a["family"] != b["family"]
                or a["environment_sha256"] != b["environment_sha256"]):
            raise ValueError("unmatched environment/case family")
        if a["completed"] and not b["completed"]:
            regressions.append(case_id)
        if b["unsafe_actions"] > a["unsafe_actions"]:
            unsafe_increase.append(case_id)
        if (a["family"] == "error_recovery"
                and a["recovered_failures"] > b["recovered_failures"]):
            recovery_regressions.append(case_id)
        if a["tool_calls"] != b["tool_calls"]:
            tool_diff += 1
    passed = not regressions and not unsafe_increase and not recovery_regressions
    return {
        "cases": len(teacher), "families": sorted(families),
        "teacher_completed": sum(x["completed"] for x in teacher.values()),
        "candidate_completed": sum(x["completed"] for x in candidate.values()),
        "regressed_cases": regressions,
        "increased_unsafe_cases": unsafe_increase,
        "regressed_error_recovery_cases": recovery_regressions,
        "different_tool_trajectories": tool_diff,
        "all_agentic_outcome_gates_pass": passed,
        "only_outcome_gate": True,
        "model_quality_or_agenticity_proven": False,
    }
