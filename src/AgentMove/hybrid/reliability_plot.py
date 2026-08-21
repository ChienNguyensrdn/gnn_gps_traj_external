from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np

from .io import read_jsonl, write_json


def reliability_bins(path: Path, bins: int = 10) -> list[dict[str, float | int]]:
    confidence, correct = [], []
    for row in read_jsonl(path):
        ranking, probabilities = row.get("ranking", []), row.get("probabilities", [])
        confidence.append(float(probabilities[0]) if probabilities else 0.0)
        correct.append(float(bool(ranking) and str(ranking[0]) == str(row["true_id"])))
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        mask = (confidence_array >= edges[index]) & (
            confidence_array < edges[index + 1] if index < bins - 1 else confidence_array <= edges[index + 1]
        )
        if mask.any():
            result.append({
                "lower": float(edges[index]), "upper": float(edges[index + 1]),
                "queries": int(mask.sum()),
                "mean_confidence": float(confidence_array[mask].mean()),
                "accuracy": float(correct_array[mask].mean()),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reliability-bin data and a paper-ready PDF")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.results)
    labels = {
        "stage1_uncalibrated": "Stage 1 (uncalibrated)",
        "stage1_only": "Stage 1 + temperature",
        "full": "Hybrid final",
    }
    data = {
        variant: reliability_bins(root / variant / "predictions.jsonl", args.bins)
        for variant in labels if (root / variant / "predictions.jsonl").exists()
    }
    write_json(args.output_json, data)
    destination = Path(args.output_figure); destination.parent.mkdir(parents=True, exist_ok=True)
    tex_path = destination.with_suffix(".tex")
    colors = ["blue!70!black", "orange!85!black", "green!55!black"]
    marks = ["*", "square*", "triangle*"]
    plot_lines = []
    for (variant, rows), color, mark in zip(data.items(), colors, marks):
        coordinates = " ".join(f"({row['mean_confidence']:.6f},{row['accuracy']:.6f})" for row in rows)
        plot_lines.append(
            rf"\draw[{color},thick] plot[mark={mark},mark size=1.5pt] coordinates {{{coordinates}}};"
        )
    legend_lines = []
    for index, ((variant, _), color, mark) in enumerate(zip(data.items(), colors, marks)):
        y = 0.96 - index * 0.075
        legend_lines.append(rf"\draw[{color},thick] (0.56,{y}) -- plot[mark={mark},mark size=1.5pt] (0.64,{y});")
        legend_lines.append(rf"\node[anchor=west,font=\scriptsize] at (0.66,{y}) {{{labels[variant]}}};")
    tex = rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usepackage{{tikz}}
\begin{{document}}
\begin{{tikzpicture}}[x=10cm,y=7.5cm]
  \draw[step=0.2,gray!18,very thin] (0,0) grid (1,1);
  \draw[->] (0,0) -- (1.04,0) node[below left,font=\small] {{Mean predicted confidence}};
  \draw[->] (0,0) -- (0,1.04) node[above left,rotate=90,font=\small] {{Empirical accuracy}};
  \foreach \x in {{0,0.2,...,1.0}} {{\draw (\x,0) -- (\x,-0.012) node[below,font=\scriptsize] {{\x}};}}
  \foreach \y in {{0,0.2,...,1.0}} {{\draw (0,\y) -- (-0.012,\y) node[left,font=\scriptsize] {{\y}};}}
  \draw[gray!65,dashed] (0,0) -- (1,1);
  {chr(10).join(plot_lines)}
  {chr(10).join(legend_lines)}
\end{{tikzpicture}}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(destination.parent), str(tex_path)],
        check=True, stdout=subprocess.DEVNULL,
    )
    generated = destination.parent / f"{tex_path.stem}.pdf"
    if generated != destination:
        generated.replace(destination)
    print(f"bins={args.output_json} figure={destination}")


if __name__ == "__main__":
    main()
