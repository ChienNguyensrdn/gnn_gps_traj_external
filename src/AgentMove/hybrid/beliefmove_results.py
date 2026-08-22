from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED = {"rq", "experiment", "seed", "git_commit", "dataset", "config", "metrics"}
DISPLAY_METRICS = {
    "acc1", "acc5", "acc10", "acc@1", "acc@5", "acc@10",
    "recall@1", "recall@5", "recall@10", "mrr", "ece", "nll", "brier",
    "cka", "transition_cosine", "llm_call_rate", "p50_latency", "p95_latency",
}


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() or "unknown"


def dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_raw(path: Path, rq: str, experiment: str, seed: int, dataset: str, config: str,
              metrics: dict[str, Any], repository: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"rq": rq.upper(), "experiment": experiment, "seed": seed, "git_commit": git_commit(repository),
               "dataset": dataset, "config": config, "metrics": metrics,
               "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0]}
    if extra: payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_raw(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8")); missing = REQUIRED - row.keys()
        if missing: raise ValueError(f"{path}: missing raw-result fields {sorted(missing)}")
        row["_path"] = str(path); rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        split = row.get("evaluation_split", "validation-legacy")
        groups.setdefault((row["rq"], row["experiment"], row["dataset"], split), []).append(row)
    output = []
    for (rq, experiment, dataset, split), items in sorted(groups.items()):
        names = sorted(set.intersection(*(set(item["metrics"]) for item in items))) if items else []
        metrics = {}
        for name in names:
            values = [item["metrics"][name] for item in items]
            if all(isinstance(value, (int, float)) for value in values):
                array = np.asarray(values, dtype=float)
                if len(array) > 1:
                    rng = np.random.default_rng(42)
                    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
                    ci95 = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
                else:
                    ci95 = None
                metrics[name] = {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
                                 "bootstrap_ci95": ci95}
        seeds = sorted(set(item["seed"] for item in items))
        output.append({"rq": rq, "experiment": experiment, "dataset": dataset, "evaluation_split": split,
                       "seeds": seeds, "publication_ready": split == "test" and len(seeds) >= 3, "metrics": metrics})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "raw_runs": len(rows), "groups": output}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# BeliefMove-Evo — Kết quả tổng hợp", "", "> Sinh tự động từ raw JSON; không chỉnh số liệu bằng tay.", "",
             f"- Raw runs hợp lệ: **{summary['raw_runs']}**", f"- Generated: `{summary['generated_at']}`", ""]
    if not summary["groups"]:
        lines += ["Chưa có raw result. Chạy các phase/RQ rồi chạy lại script aggregate.", ""]
    current = None
    for group in summary["groups"]:
        if group["rq"] != current:
            current = group["rq"]; lines += [f"## {current}", "", "| Experiment | Dataset | Split | Seeds | Evaluation metrics | Gate |", "|---|---|---|---|---|---|"]
        rendered = []
        for name, value in group["metrics"].items():
            if name not in DISPLAY_METRICS:
                continue
            if value["bootstrap_ci95"] is None:
                rendered.append(f"{name}={value['mean']:.6f} (std/CI N/A; cần ≥2 seeds)")
            else:
                rendered.append(
                    f"{name}={value['mean']:.6f} ± {value['std']:.6f} "
                    f"(95% CI {value['bootstrap_ci95'][0]:.6f}–{value['bootstrap_ci95'][1]:.6f})"
                )
        cells = ", ".join(rendered) or "TBD"
        gate = "ready" if group["publication_ready"] else "not ready"
        lines.append(
            f"| {group['experiment']} | {group['dataset']} | {group['evaluation_split']} | "
            f"{', '.join(map(str, group['seeds']))} | {cells} | {gate} |"
        )
    lines += ["", "## Publication gate", "", "Một group chỉ `ready` khi là test split và có ít nhất 3 seeds. Validation dùng chọn checkpoint/hyperparameter, không phải kết quả test cuối cùng.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BeliefMove-Evo raw results and generate ideas/results.md")
    parser.add_argument("--input", type=Path, default=Path("results/beliefmove-evo/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/beliefmove-evo/aggregated"))
    parser.add_argument("--results-md", type=Path, default=Path("../../ideas/results.md"))
    args = parser.parse_args(); summary = aggregate(load_raw(args.input) if args.input.exists() else [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.results_md.parent.mkdir(parents=True, exist_ok=True); args.results_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"raw_runs": summary["raw_runs"], "summary": str(args.output_dir / "summary.json"), "markdown": str(args.results_md)}))


if __name__ == "__main__":
    main()
