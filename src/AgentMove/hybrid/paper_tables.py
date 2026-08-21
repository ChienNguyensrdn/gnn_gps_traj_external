from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .io import write_json


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _value(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def _row(label: str, values) -> str:
    return label + " & " + " & ".join(_value(value) for value in values) + r" \\"


def generate(agentmove: Path, paper: Path, hybrid_results: Path, llm_results: Path) -> Dict[str, Any]:
    output = paper / "generated"; output.mkdir(parents=True, exist_ok=True)
    getnext = agentmove / "data/hybrid/Shanghai/getnext"
    frames = [pd.read_csv(getnext / name) for name in ["train.csv", "val.csv", "test.csv"]]
    all_rows = pd.concat(frames, ignore_index=True)
    lengths = all_rows.groupby("trajectory_id").size()
    dataset_stats = {
        "ISP-Shanghai": {
            "users": int(all_rows.user_id.nunique()), "trajectories": int(all_rows.trajectory_id.nunique()),
            "locations": int(all_rows.POI_id.nunique()), "avg_length": float(lengths.mean()), "cities": 1,
        },
        "Foursquare TIST2015": None, "YJMob100K": None,
    }
    shanghai = dataset_stats["ISP-Shanghai"]
    table1 = "\n".join([
        r"% Generated from Shanghai preprocessing artifacts; blank cells have no result yet.",
        f"ISP-Shanghai & {shanghai['users']} & {shanghai['trajectories']} & {shanghai['locations']} & {shanghai['avg_length']:.4f} & 1 " + r"\\",
        r"Foursquare TIST2015 &  &  &  &  & 12 \\",
        r"YJMob100K &  &  &  &  &  \\", "",
    ])
    (output / "table_1_dataset_stats_rows.tex").write_text(table1, encoding="utf-8")

    rq1 = _load(hybrid_results / "rq1_main_results.json") or {}
    stage_recall = _load(agentmove / "results/hybrid/shanghai-50-seed42/stage1_candidate_recall.json") or {}
    llm = _load(llm_results / "metrics.json")
    markov = stage_recall.get("markov", {})
    stage1 = rq1.get("stage1_only") or {}
    full = rq1.get("ours_full") or {}
    main_rows: Dict[str, Any] = {
        "Markov/Bi-gram": [markov.get("recall@1"), markov.get("recall@5"), markov.get("recall@10"), markov.get("mrr_full_candidate_space")],
        "GETNext": None,
        "CGM (Stage 1 only)": [stage1.get("acc@1"), stage1.get("acc@5"), stage1.get("acc@10"), stage1.get("mrr")],
        "LLM-Mob": None,
        "LLM-ZS": [llm.get("acc@1"), llm.get("acc@5"), llm.get("acc@10"), llm.get("mrr")] if llm else None,
        "AgentMove (original)": None, "NextLocLLM": None, "TrajAgent": None, "DBN": None,
        "Ours (full)": [full.get("acc@1"), full.get("acc@5"), full.get("acc@10"), full.get("mrr")],
    }
    groups = [
        (r"\multirow{3}{*}{Quantitative}", "Markov/Bi-gram"), ("", "GETNext"), ("", "CGM (Stage 1 only)"),
        (r"\multirow{3}{*}{LLM-only}", "LLM-Mob"), ("", "LLM-ZS"), ("", "AgentMove (original)"),
        (r"\multirow{2}{*}{Existing hybrids}", "NextLocLLM"), ("", "TrajAgent"),
        ("Classical Bayes", "DBN"), (r"\textbf{Proposed}", "Ours (full)"),
    ]
    table2 = [r"% Ready-to-paste body rows; Shanghai columns followed by blank TIST2015 columns."]
    for group, label in groups:
        values = main_rows[label] or [None] * 4
        rendered = " & ".join(_value(value) for value in values + [None] * 4)
        table2.append(f"{group} & {label} & {rendered} " + r"\\")
    (output / "table_2_main_results_rows.tex").write_text("\n".join(table2) + "\n", encoding="utf-8")

    ablations = _load(hybrid_results / "rq2_ablation.json") or {}
    mapping = [
        ("Ours (full)", "full"), (r"\quad w/o Calibration (CGM)", "no_temperature"),
        (r"\quad w/o World Knowledge (OSM)", "no_world"),
        (r"\quad w/o Personal Memory (embedding)", "no_embedding_memory"),
        (r"\quad w/o BBN (raw-score proxy)", "no_bbn"),
        (r"\quad w/o Link Calibration $(\alpha,\beta)$", "no_link_calibration"),
        (r"\quad Top-1 only (no Stages 2--3)", "top1_only"),
    ]
    table3 = [r"% no_bbn is a raw-score proxy, not a free-text reranker."]
    for label, key in mapping:
        # These implementations do not match the manuscript row exactly.
        # Keep the manuscript cells blank until the exact experiment exists.
        row = {} if key in {"no_world", "no_bbn"} else ablations.get(key, {})
        table3.append(_row(label, [row.get("acc@1"), row.get("acc@5"), row.get("mrr"), row.get("ece")]))
    (output / "table_3_ablation_rows.tex").write_text("\n".join(table3) + "\n", encoding="utf-8")

    rq3 = _load(hybrid_results / "rq3_efficiency.json") or {}
    ours = rq3.get("full", {})
    table4 = [
        r"% Tokens are len(text)/4 estimates; latency is measured model-call latency.",
        _row("LLM-Mob", [None, None, None]),
        _row("AgentMove (original)", [None, None, None]),
        _row("Ours (full)", [ours.get("input_tokens_mean"), ours.get("output_tokens_mean"), ours.get("llm_latency_mean")]), "",
    ]
    (output / "table_4_efficiency_rows.tex").write_text("\n".join(table4), encoding="utf-8")

    manifest = {
        "scope": "ISP-Shanghai deterministic 50% sample, seed 42",
        "hybrid_results": str(hybrid_results.resolve()), "llm_results": str(llm_results.resolve()),
        "table_1_dataset_stats": dataset_stats, "table_2_main_results": main_rows,
        "table_3_ablation": {key: ablations.get(key) for _, key in mapping},
        "table_4_efficiency": {"llm_zs_history_only": llm, "ours_full": ours},
        "blank_policy": "Missing experiments are emitted as empty LaTeX cells.",
    }
    write_json(output / "table_values.json", manifest)
    (output / "README.md").write_text(
        "# Generated paper table data\n\n"
        "These fragments use the deterministic Shanghai 50% protocol (seed 42). "
        "Empty cells mean the experiment has not been run. Do not present these as full-dataset results.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate traceable LaTeX rows for paper Tables 1--4")
    parser.add_argument("--agentmove-root", default="."); parser.add_argument("--paper-dir", default="../../paper")
    parser.add_argument("--hybrid-results", default="results/hybrid/shanghai-neural-cgm-50-seed42/qwen2-7b")
    parser.add_argument("--llm-results", default="results/llm-only/shanghai-50-seed42/qwen2-7b")
    args = parser.parse_args()
    result = generate(Path(args.agentmove_root), Path(args.paper_dir), Path(args.hybrid_results), Path(args.llm_results))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
