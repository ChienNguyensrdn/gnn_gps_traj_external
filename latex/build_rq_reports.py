#!/usr/bin/env python3
"""Convert ideas/results_rq1.md ... results_rq13.md into one LaTeX include.

This is intentionally a mechanical converter: report wording, numbers and
conclusions remain in the source Markdown files.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("rq_reports.tex")


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "≤": r"$\leq$", "≥": r"$\geq$",
        "±": r"$\pm$", "→": r"$\rightarrow$", "–": "--", "—": "---",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline(text: str) -> str:
    tokens: list[str] = []

    def hold(value: str) -> str:
        tokens.append(value)
        return f"@@TOKEN{len(tokens)-1}@@"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: hold(r"\[" + m.group(1) + r"\]"), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: hold("$" + m.group(1) + "$"), text)
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + escape_plain(m.group(1)) + "}"), text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", lambda m: hold(r"\href{" + escape_plain(m.group(2)) + "}{" + escape_plain(m.group(1)) + "}"), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: hold(r"\textbf{" + escape_plain(m.group(1)) + "}"), text)
    text = escape_plain(text)
    for index, value in enumerate(tokens):
        text = text.replace(f"@@TOKEN{index}@@", value)
    return text


def table(lines: list[str]) -> str:
    rows = [[inline(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0]] + rows[2:]
    columns = len(rows[0])
    alignment = "@{}" + "l" + "r" * (columns - 1) + "@{}"
    body = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{adjustbox}{max width=\textwidth}", f"\\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    for index, row in enumerate(rows):
        body.append(" & ".join(row) + r" \\")
        if index == 0:
            body.append(r"\midrule")
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    return "\n".join(body)


def convert(path: Path) -> str:
    source = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(inline(" ".join(item.strip() for item in paragraph)))
            output.append("")
            paragraph = []

    index = 0
    while index < len(source):
        line = source[index]
        if line.startswith("|") and index + 1 < len(source) and source[index + 1].startswith("|"):
            flush_paragraph()
            if in_list:
                output.append(r"\end{" + ("enumerate" if in_list == "enumerate" else "itemize") + "}"); in_list = False
            block = []
            while index < len(source) and source[index].startswith("|"):
                block.append(source[index]); index += 1
            output.append(table(block)); output.append("")
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            if in_list:
                output.append(r"\end{" + ("enumerate" if in_list == "enumerate" else "itemize") + "}"); in_list = False
            level = len(heading.group(1))
            command = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}[level]
            output.append(f"\\{command}{{{inline(heading.group(2))}}}"); output.append("")
        elif line.startswith("- "):
            flush_paragraph()
            if not in_list:
                output.append(r"\begin{itemize}"); in_list = True
            output.append(r"\item " + inline(line[2:]))
        elif re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            if not in_list:
                output.append(r"\begin{enumerate}"); in_list = "enumerate"
            output.append(r"\item " + inline(re.sub(r"^\d+\.\s+", "", line)))
        elif not line.strip():
            flush_paragraph()
            if in_list:
                output.append(r"\end{" + ("enumerate" if in_list == "enumerate" else "itemize") + "}")
                output.append(""); in_list = False
        elif line.startswith(">"):
            flush_paragraph(); output.append(r"\begin{quote}\small " + inline(line.lstrip("> ")) + r"\end{quote}")
        else:
            paragraph.append(line)
        index += 1
    flush_paragraph()
    if in_list:
        output.append(r"\end{" + ("enumerate" if in_list == "enumerate" else "itemize") + "}")
    return "\n".join(output)


parts = ["% Generated mechanically from ideas/results_rq1.md ... results_rq13.md.",
         "% Run: python3 build_rq_reports.py", ""]
for rq in range(1, 14):
    parts.append(convert(ROOT / "ideas" / f"results_rq{rq}.md"))
    parts.append(r"\clearpage")
OUTPUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(f"generated {OUTPUT}")
