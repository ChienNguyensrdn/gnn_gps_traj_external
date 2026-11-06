#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BELIEFMOVE_ROOT="${BELIEFMOVE_ROOT:-results/beliefmove-faithful/v2}"
REPORT="$BELIEFMOVE_ROOT/aggregated/pilot_report.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 2
fi

# Rebuild the aggregate from immutable raw artifacts before inspecting it.
"$PYTHON_BIN" -m hybrid.beliefmove.experiment pilot-report \
  --results-root "$BELIEFMOVE_ROOT" >/dev/null

"$PYTHON_BIN" - "$REPORT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))

operational_checks = (
    "unique_base_cache",
    "cache_complete",
    "belief_schema_valid",
    "fixed_ontology_valid",
    "content_provenance_valid",
    "pilot_budget_contract",
    "valid_output_rate",
    "transport_error_rate",
    "no_prompt_target_leakage",
)
scientific_checks = (
    "teacher_activity_gain",
    "teacher_destination_gain",
    "m2_heldout_nll_improves",
    "repeat_jsd",
)


def journal_ids(path: Path) -> tuple[set[str], str | None]:
    if not path.is_file():
        return set(), f"missing {path.name}"
    ids: set[str] = set()
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in ids:
                return ids, f"duplicate query_id at {path.name}:{line_number}"
            ids.add(query_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ids, f"invalid {path.name}: {error}"
    return ids, None


operational_ready = bool(report.get("all_units_present"))
scientific_ready = operational_ready

print(f"report={report_path}")
print("dataset/unit | artifacts | dangling | scientific | failed checks")
print("-" * 112)

for unit in report.get("units", []):
    label = f"{unit.get('dataset')}:{unit.get('unit')}"
    checks = unit.get("checks", {})
    cache_value = unit.get("cache")
    cache = Path(cache_value) if isinstance(cache_value, str) and cache_value else None

    dangling: int | str = "N/A"
    journal_errors: list[str] = []
    if cache is not None:
        attempts, attempt_error = journal_ids(cache / "attempts.jsonl")
        beliefs, belief_error = journal_ids(cache / "beliefs.jsonl")
        if attempt_error:
            journal_errors.append(attempt_error)
        if belief_error:
            journal_errors.append(belief_error)
        if not journal_errors:
            dangling = len(attempts - beliefs)
            orphan = len(beliefs - attempts)
            if orphan:
                journal_errors.append(f"orphan_beliefs={orphan}")

    failed_operational = [
        name for name in operational_checks if checks.get(name) is not True
    ]
    if unit.get("pilot_evaluation_present") is not True:
        failed_operational.append("pilot_evaluation_present")
    if dangling != 0:
        failed_operational.append(f"dangling_attempts={dangling}")
    failed_operational.extend(journal_errors)

    failed_scientific = [
        name for name in scientific_checks if checks.get(name) is not True
    ]
    unit_operational = not failed_operational
    unit_scientific = unit_operational and not failed_scientific
    operational_ready = operational_ready and unit_operational
    scientific_ready = scientific_ready and unit_scientific

    failed = failed_operational if failed_operational else failed_scientific
    print(
        f"{label:<30} | "
        f"{'complete' if unit_operational else 'incomplete':<10} | "
        f"{str(dangling):<8} | "
        f"{'pass' if unit_scientific else ('fail' if unit_operational else 'pending'):<10} | "
        f"{', '.join(failed) if failed else '-'}"
    )

print("-" * 112)
if not operational_ready:
    print("STATUS=INCOMPLETE_ARTIFACTS")
    print("ACTION=resume/recover only the incomplete unit; do not run protocol lock")
    raise SystemExit(2)
if not scientific_ready:
    print("STATUS=COMPLETE_NEGATIVE")
    print("ACTION=report the negative pilot and stop; do not run protocol lock")
    raise SystemExit(3)

print("STATUS=READY_FOR_LOCK")
print("ACTION=the predeclared pilot gate passed on every required unit")
raise SystemExit(0)
PY
