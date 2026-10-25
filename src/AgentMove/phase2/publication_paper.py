from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("tist2015", "www2019", "yjmob100k")
LABELS = {
    "tist2015": "TIST2015 (12-city macro)",
    "www2019": "WWW2019-Shanghai",
    "yjmob100k": "YJMob100K-Dataset1",
}
SHORT_LABELS = {"tist2015": "TIST2015", "www2019": "WWW2019", "yjmob100k": "YJMob100K"}


def load(path: Path, expected_gate: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing aggregate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != expected_gate:
        raise ValueError(f"gate {payload.get('gate')!r} in {path}; expected {expected_gate!r}")
    if payload.get("missing"):
        raise ValueError(f"aggregate still reports {len(payload['missing'])} missing artifacts: {path}")
    missing_datasets = set(DATASETS) - set(payload.get("datasets", {}))
    if missing_datasets:
        raise ValueError(f"missing datasets in {path}: {sorted(missing_datasets)}")
    return payload


def mean(value: dict, digits: int = 4) -> str:
    return f"{float(value['mean']):.{digits}f}"


def metric_row(label: str, name: str, values: dict, metrics: tuple[str, ...],
               digits: tuple[int, ...] | None = None) -> str:
    if digits is None:
        digits = (4,) * len(metrics)
    cells = [label, name] + [mean(values[metric], width) for metric, width in zip(metrics, digits)]
    return " & ".join(cells) + r" \\"


def find_test(payload: dict, dataset: str, comparison: str, metric: str) -> dict:
    for row in payload["paired_tests"]:
        if (row.get("dataset"), row.get("comparison"), row.get("metric")) == (
            dataset,
            comparison,
            metric,
        ):
            return row
    raise KeyError(f"missing paired test: {dataset}/{comparison}/{metric}")


def test_values(row: dict) -> tuple[float, list[float], float, bool]:
    effect = float(row.get("effect_favoring_first", row.get("effect")))
    ci = row.get("bootstrap_ci95", row.get("ci95"))
    p = float(row.get("holm_adjusted_p", row.get("holm_p")))
    sig = bool(row.get("significant_at_0.05", row.get("significant")))
    return effect, [float(ci[0]), float(ci[1])], p, sig


def tex_number(value: float, digits: int = 4) -> str:
    rendered = f"{float(value):.{digits}f}"
    return f"${rendered}$" if float(value) < 0 else rendered


def tex_interval(values: list[float], digits: int = 4) -> str:
    return f"[{tex_number(values[0], digits)}, {tex_number(values[1], digits)}]"


def paired_cells(payload: dict, dataset: str, comparison: str, metric: str,
                 digits: int = 4) -> tuple[str, str, str]:
    effect, ci, p, _ = test_values(find_test(payload, dataset, comparison, metric))
    return tex_number(effect, digits), tex_interval(ci, digits), f"{p:.4g}"


def append_grouped_rows(lines: list[str], payload: dict, container: str,
                        variants: tuple[str, ...], metrics: tuple[str, ...],
                        digits: tuple[int, ...] | None = None) -> None:
    for dataset in DATASETS:
        values = payload["datasets"][dataset][container]
        for index, variant in enumerate(variants):
            label = LABELS[dataset] if index == 0 else ""
            lines.append(metric_row(label, variant, values[variant], metrics, digits))
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")


def render_rq4(lines: list[str], payload: dict) -> None:
    lines += [
        r"\newcommand{\RQFourTables}{%", r"\begin{table*}[!t]", r"\centering",
        r"\caption{RQ4 last-query means. TIST2015 is an equal-weight 12-city macro average; WWW2019 and YJMob100K average student seeds 42--44.}",
        r"\label{tab:rq4_main}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"Dataset & Variant & R@1 & R@5 & R@10 & MRR & NLL$\downarrow$ & Brier$\downarrow$ & ECE$\downarrow$ \\",
        r"\midrule",
    ]
    append_grouped_rows(lines, payload, "variants", ("E0-ce", "E1-kd", "E5-dual"),
                        ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece"))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", "",
              r"\begin{table}[!t]", r"\centering",
              r"\caption{RQ4 paired Recall@1 effects. Positive values favor the first variant; $p_{\mathrm{H}}$ is Holm-adjusted.}",
              r"\label{tab:rq4_tests}", r"\resizebox{\columnwidth}{!}{%",
              r"\begin{tabular}{llrrr}", r"\toprule",
              r"Dataset & Contrast & Effect & 95\% CI & $p_{\mathrm{H}}$ \\", r"\midrule"]
    for dataset in DATASETS:
        for contrast, comparison in (("E1--E0", "E1-kd-vs-E0-ce"),
                                     ("E5--E1", "E5-dual-vs-E1-kd")):
            effect, ci, p = paired_cells(payload, dataset, comparison, "recall@1")
            lines.append(f"{SHORT_LABELS[dataset]} & {contrast} & {effect} & {ci} & {p}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", "}", ""]


def render_rq7(lines: list[str], payload: dict) -> None:
    lines += [r"\newcommand{\RQSevenTables}{%", r"\begin{table*}[!t]", r"\centering",
              r"\caption{RQ7 all-prefix transition refinement: (a) aggregate test metrics and (b) query-level paired B3--B0 effects. B0 is the unmodified frozen E5 distribution; lower NLL, Brier, and ECE are better. Paired intervals do not cluster prefixes by trajectory.}",
              r"\label{tab:rq7}", r"\textit{(a) Aggregate test metrics}\par\smallskip",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
              r"Dataset & Variant & R@1 & R@5 & R@10 & MRR & NLL$\downarrow$ & Brier$\downarrow$ & ECE$\downarrow$ \\", r"\midrule"]
    append_grouped_rows(lines, payload, "variants", ("B0-static", "B3-dbn"),
                        ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece"))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\medskip",
              r"\textit{(b) Paired effects; NLL/Brier signs are reversed so positive favors B3}\par\smallskip",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
              r"Dataset & $\Delta$R@1 & 95\% CI & $p_{\mathrm{H}}$ & $\Delta$MRR & 95\% CI & $p_{\mathrm{H}}$ & $\Delta$NLL & $\Delta$Brier \\", r"\midrule"]
    for dataset in DATASETS:
        r1, r1_ci, p = paired_cells(payload, dataset, "B3-dbn-vs-B0-static", "recall@1")
        mrr, mrr_ci, mrr_p = paired_cells(payload, dataset, "B3-dbn-vs-B0-static", "mrr")
        nll, _, _ = paired_cells(payload, dataset, "B3-dbn-vs-B0-static", "nll")
        brier, _, _ = paired_cells(payload, dataset, "B3-dbn-vs-B0-static", "brier")
        lines.append(f"{SHORT_LABELS[dataset]} & {r1} & {r1_ci} & {p} & {mrr} & {mrr_ci} & {mrr_p} & {nll} & {brier}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", "}", ""]


def render_rq3(lines: list[str], payload: dict) -> None:
    lines += [r"\newcommand{\RQThreeTables}{%", r"\begin{table*}[!t]", r"\centering",
              r"\caption{RQ3 bounded diagnostic (first $n\leq200$ prepared queries per unit, Qwen2:7b): (a) aggregate test metrics and (b) paired M4--M3 effects. M3 is data-prior plus quantitative fusion; M4 adds a validation-selected structured-evidence factor. YJMob100K selected zero LLM weight. Holm adjustment covers all 36 RQ3 tests; Panel (b) shows the primary M4--M3 contrast.}",
              r"\label{tab:rq3}", r"\textit{(a) Aggregate test metrics}\par\smallskip",
              r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Dataset & Variant & R@1 & R@5 & R@10 & MRR \\", r"\midrule"]
    append_grouped_rows(lines, payload, "variants", ("M3-quantitative", "M4-both"),
                        ("recall@1", "recall@5", "recall@10", "mrr"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
              r"\textit{(b) Paired M4--M3 effects with Holm-adjusted $p$-values}\par\smallskip",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Dataset & $\Delta$R@1 & 95\% CI & $p_{\mathrm{H}}$ & $\Delta$MRR & 95\% CI & $p_{\mathrm{H}}$ \\", r"\midrule"]
    for dataset in DATASETS:
        r1, r1_ci, r1_p = paired_cells(payload, dataset, "M4-both-vs-M3-quantitative", "recall@1")
        mrr, mrr_ci, mrr_p = paired_cells(payload, dataset, "M4-both-vs-M3-quantitative", "mrr")
        lines.append(f"{SHORT_LABELS[dataset]} & {r1} & {r1_ci} & {r1_p} & {mrr} & {mrr_ci} & {mrr_p}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", "}", ""]


def render_rq8(lines: list[str], payload: dict) -> None:
    display = {"never": "Never", "always": "Always", "entropy": "Entropy", "margin": "Margin",
               "random-budget-matched": "Random matched"}
    lines += [r"\newcommand{\RQEightTables}{%", r"\begin{table*}[!t]", r"\centering",
              r"\caption{RQ8 bounded routing diagnostic at the primary validation-selected call-rate cap: (a) aggregate test metrics and (b) paired entropy--random effects. Random is the mean of 50 draws matched to the realized entropy call rate.}",
              r"\label{tab:rq8}", r"\textit{(a) Aggregate test metrics}\par\smallskip",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Dataset & Policy & R@1 & MRR & Call rate & Est. tokens/query \\", r"\midrule"]
    for dataset in DATASETS:
        values = payload["datasets"][dataset]["policies"]
        for index, policy in enumerate(("never", "always", "entropy", "margin", "random-budget-matched")):
            label = LABELS[dataset] if index == 0 else ""
            lines.append(metric_row(label, display[policy], values[policy],
                                    ("recall@1", "mrr", "llm_call_rate", "tokens_per_query"),
                                    (4, 4, 4, 1)))
        if dataset != DATASETS[-1]: lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\medskip",
              r"\textit{(b) Paired entropy--random-matched effects with Holm-adjusted $p$-values}\par\smallskip",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Dataset & $\Delta$R@1 & 95\% CI & $p_{\mathrm{H}}$ & $\Delta$MRR & 95\% CI & $p_{\mathrm{H}}$ \\", r"\midrule"]
    for dataset in DATASETS:
        r1, r1_ci, r1_p = paired_cells(payload, dataset, "entropy-vs-random-budget-matched", "recall@1", 5)
        mrr, mrr_ci, mrr_p = paired_cells(payload, dataset, "entropy-vs-random-budget-matched", "mrr", 5)
        lines.append(f"{SHORT_LABELS[dataset]} & {r1} & {r1_ci} & {r1_p} & {mrr} & {mrr_ci} & {mrr_p}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", "}", ""]


def render(rq3: dict, rq4: dict, rq7: dict, rq8: dict) -> str:
    lines = ["% Generated by phase2.publication_paper; do not edit numbers manually.", ""]
    render_rq4(lines, rq4)
    render_rq7(lines, rq7)
    render_rq3(lines, rq3)
    render_rq8(lines, rq8)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase-2 evidence and export IEEE LaTeX tables")
    parser.add_argument("--root", type=Path, default=Path("results/phase2/aggregated"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    rq3 = load(args.root / f"rq3_limit-{args.limit}_summary.json", "ready-bounded")
    rq4 = load(args.root / "rq4_summary.json", "ready")
    rq7 = load(args.root / "rq7_summary.json", "ready")
    rq8 = load(args.root / f"rq8_limit-{args.limit}_summary.json", "ready-bounded")
    if args.audit_only:
        print(json.dumps({"artifact_gate": "ready", "gate_scope": "artifact-completeness only",
                          "scientific_claims": {"RQ3": "unsupported/inconclusive",
                                                "RQ8": "unsupported/inconclusive"},
                          "rqs": ["RQ3", "RQ4", "RQ7", "RQ8"], "limit": args.limit}))
        return
    if args.output is None:
        parser.error("--output is required unless --audit-only is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rq3, rq4, rq7, rq8) + "\n", encoding="utf-8")
    print(json.dumps({"gate": "ready", "output": str(args.output), "limit": args.limit}))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"publication gate failed: {exc}") from None
