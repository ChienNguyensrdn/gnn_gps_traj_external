from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def aggregate(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"path": str(path), "status": "invalid", "error": str(exc)})
            continue
        records.append({"path": str(path), "status": "complete", "metrics": metrics})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "completed_metric_files": sum(row["status"] == "complete" for row in records),
        "invalid_metric_files": sum(row["status"] == "invalid" for row in records),
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a traceable index of experiment metrics")
    parser.add_argument("--results-root", default="results"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); payload = aggregate(Path(args.results_root))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("root", "completed_metric_files", "invalid_metric_files")}, indent=2))


if __name__ == "__main__":
    main()
