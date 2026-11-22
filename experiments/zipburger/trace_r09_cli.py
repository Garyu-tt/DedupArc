"""R09 offline quality report for paired teacher/student executions.

A clean report is a prerequisite for experiments, NOT deployment approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import layer_trace_r09 as traces
from . import agentic_gate_r09 as agency


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--teacher-tasks", type=Path)
    parser.add_argument("--candidate-tasks", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.teacher_tasks is None) != (args.candidate_tasks is None):
        parser.error("teacher and candidate executed task files are required together")

    original, candidate = traces.load(args.teacher_trace), traces.load(args.candidate_trace)
    layer_report = traces.compare(original, candidate)
    outcome = None
    if args.teacher_tasks is not None:
        outcome = agency.compare(agency.load_cases(args.teacher_tasks),
                                 agency.load_cases(args.candidate_tasks))
    gate = (layer_report["all_24_layer_gates_pass"] and
            (outcome is None or outcome["all_agentic_outcome_gates_pass"]))
    report = {
        "experiment": "ZipBurger R09 24-layer drift and agent outcomes",
        "layer_report": layer_report,
        "agentic_outcomes_report": outcome,
        "all_requested_gates_pass": bool(gate),
        "real_execution_confirmed": bool(layer_report["genuine_model_traces"]
                                         and outcome is not None),
        "model_agenticity_proven": False,
        "production_release_approved": False,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
