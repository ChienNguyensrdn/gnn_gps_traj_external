from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .io import read_jsonl, write_json


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _metric_table(rows: Dict[str, Dict[str, float]], metrics: Iterable[str]) -> List[str]:
    metrics = list(metrics)
    lines = ["| Variant | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for name, values in rows.items():
        lines.append(f"| `{name}` | " + " | ".join(_fmt(values.get(metric)) for metric in metrics) + " |")
    return lines


def _reliability(prediction_file: Path, bins: int = 15) -> List[Dict[str, float]]:
    confidences, outcomes = [], []
    for row in read_jsonl(prediction_file):
        ranking = row.get("ranking", [])
        probabilities = row.get("probabilities", [])
        confidences.append(float(probabilities[0]) if probabilities else 0.0)
        outcomes.append(float(bool(ranking) and str(ranking[0]) == str(row["true_id"])))
    confidence = np.asarray(confidences)
    outcome = np.asarray(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        mask = (confidence >= edges[index]) & ((confidence < edges[index + 1]) if index < bins - 1 else (confidence <= edges[index + 1]))
        if mask.any():
            result.append({
                "lower": float(edges[index]), "upper": float(edges[index + 1]),
                "queries": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(outcome[mask].mean()),
            })
    return result


def generate(results: Path, manifest_path: Path) -> Path:
    manifest = _load(manifest_path)
    rq1 = _load(results / "rq1_main_results.json")
    rq1_bootstrap = _load(results / "rq1_full_vs_stage1_bootstrap.json")
    rq2 = _load(results / "rq2_ablation.json")
    rq2_bootstrap = _load(results / "rq2_paired_bootstrap.json")
    rq3 = _load(results / "rq3_efficiency.json")
    rq4 = _load(results / "rq4_calibration_and_generalization.json")
    reliability = {
        variant: _reliability(results / variant / "predictions.jsonl")
        for variant in ["stage1_uncalibrated", "stage1_only", "full"]
        if (results / variant / "predictions.jsonl").exists()
    }
    write_json(results / "rq4_reliability_bins.json", reliability)

    full = rq1["ours_full"]
    stage1 = rq1["stage1_only"]
    lines = [
        "# Báo cáo thực nghiệm RQ1–RQ4 — ISP-Shanghai 50%",
        "",
        "## Protocol",
        "",
        f"- Validation: {manifest['validation']['selected_queries']}/{manifest['validation']['source_queries']} query.",
        f"- Test: {manifest['test']['selected_queries']}/{manifest['test']['source_queries']} query.",
        f"- Seed: `{manifest['validation']['seed']}`; selection: `{manifest['validation']['selection']}`.",
        f"- Candidate hash: `{manifest['test']['query_ids_sha256']}`.",
        "- Phạm vi: một thành phố. Không dùng kết quả này để kết luận generalization qua thành phố.",
        "",
        "## RQ1 — Độ chính xác",
        "",
        *_metric_table({"full": full, "stage1_only": stage1}, ["acc@1", "acc@5", "acc@10", "mrr", "candidate_recall"]),
        "",
        f"Delta Acc@1 full − Stage 1 = **{_fmt(rq1_bootstrap['delta'])}**, "
        f"95% CI [{_fmt(rq1_bootstrap['ci95_low'])}, {_fmt(rq1_bootstrap['ci95_high'])}].",
        "",
        "Kết quả này trả lời so sánh Hybrid với Stage 1 trên Shanghai. Các baseline độc lập "
        "(GETNext, AgentMove/LLM-only, DBN) chưa có trong run này nên chưa thể kết luận RQ1 đầy đủ như bảng paper.",
        "",
        "## RQ2 — Ablation",
        "",
        *_metric_table(rq2, ["acc@1", "acc@5", "mrr", "ece", "invalid_evidence_rate"]),
        "",
        "Paired bootstrap Acc@1 (full − variant):",
        "",
        "| Variant | Delta | CI95 low | CI95 high |",
        "|---|---:|---:|---:|",
    ]
    for variant, values in rq2_bootstrap.items():
        lines.append(f"| `{variant}` | {_fmt(values['delta'])} | {_fmt(values['ci95_low'])} | {_fmt(values['ci95_high'])} |")
    lines.extend([
        "",
        "Lưu ý: `no_bbn` là raw-score reranking proxy, chưa phải free-text LLM reranking độc lập.",
        "",
        "## RQ3 — Chi phí và latency",
        "",
        *_metric_table({"full": rq3["full"]}, ["acc@1", "input_tokens_mean", "output_tokens_mean", "api_calls_mean", "llm_latency_mean", "llm_latency_p95"]),
        "",
        "`llm_latency_*` được cộng từ thời gian gọi model đã lưu trong evidence cache. Token hiện là "
        "ước lượng theo độ dài chuỗi, không phải tokenizer usage chính xác của Ollama. Chưa có run LLM-only "
        "cùng protocol nên chưa thể tính phần trăm tiết kiệm so với AgentMove/LLM-Mob.",
        "",
        "## RQ4 — Calibration và generalization",
        "",
        *_metric_table({
            "stage1_uncalibrated": rq4["stage1_before_temperature"],
            "stage1_temperature": rq4["stage1_after_temperature"],
            "stage3_full": rq4["stage3_final"],
        }, ["acc@1", "ece", "nll", "brier"]),
        "",
        "Reliability-bin data: `rq4_reliability_bins.json`.",
        "",
        "Run này trả lời phần calibration trên Shanghai. Cross-city cần TIST2015; cross-backbone cần "
        "chạy cùng sample manifest với ít nhất một model khác.",
        "",
        "## Trạng thái kết luận",
        "",
        "| RQ | Trạng thái |",
        "|---|---|",
        "| RQ1 | Một phần: full vs Stage 1; thiếu baseline độc lập |",
        "| RQ2 | Đầy đủ cho ablation nội bộ; `no_bbn` vẫn là proxy |",
        "| RQ3 | Có measurement của Hybrid; thiếu LLM-only comparison |",
        "| RQ4 | Có calibration Shanghai; thiếu cross-city/cross-backbone |",
        "",
    ])
    report = results / "RQ_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-backed RQ1--RQ4 Markdown report")
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(generate(Path(args.results), Path(args.manifest)))


if __name__ == "__main__":
    main()
