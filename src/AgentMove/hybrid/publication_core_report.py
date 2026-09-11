"""Combine per-dataset publication-core Markdown reports into one document."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


TIST_CITIES = (
    "Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai "
    "SanFrancisco London SaoPaulo Moscow"
).split()


def expected_reports(source_dir: Path, cities: list[str]) -> list[tuple[str, str, Path]]:
    reports = [
        ("TIST2015", city, source_dir / f"publication_core_tist2015-{city}.md")
        for city in cities
    ]
    reports.extend(
        [
            ("WWW2019", "Shanghai", source_dir / "publication_core_www2019-Shanghai.md"),
            ("YJMob100K", "YJMob", source_dir / "publication_core_yjmob100k-YJMob.md"),
        ]
    )
    return reports


def body_with_shifted_headings(text: str) -> str:
    """Drop the source H1 and shift remaining headings down two levels."""
    lines = text.strip().splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    shifted: list[str] = []
    for line in lines:
        match = re.match(r"^(#{1,4})(\s+.*)$", line)
        if match:
            line = "#" * min(len(match.group(1)) + 2, 6) + match.group(2)
        shifted.append(line)
    return "\n".join(shifted).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("../../ideas"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../../ideas/results_publication_core_3datasets.md"),
    )
    parser.add_argument("--cities", nargs="+", default=TIST_CITIES)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    reports = expected_reports(args.source_dir, args.cities)
    missing = [path for _, _, path in reports if not path.is_file()]
    if missing and not args.allow_incomplete:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing publication reports:\n{formatted}")

    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = [
        "# Tổng hợp kết quả thực nghiệm trên ba bộ dữ liệu",
        "",
        f"> Generated: `{generated}`. Các bảng bên dưới được ghép từ báo cáo sinh tự động; không chỉnh số liệu thủ công.",
        "",
        "## Phạm vi",
        "",
        "- TIST2015: 12 thành phố.",
        "- WWW2019: Shanghai-ISP.",
        "- YJMob100K: Dataset1/YJMob.",
        "- Neural last-query và Bayesian all-prefix là hai protocol khác nhau, không so trực tiếp trị tuyệt đối.",
        "",
        "## Mục lục",
        "",
        "- [TIST2015](#tist2015)",
        "- [WWW2019 — Shanghai](#www2019--shanghai)",
        "- [YJMob100K](#yjmob100k)",
    ]

    current_dataset = None
    included = 0
    for dataset, unit, path in reports:
        if dataset != current_dataset:
            lines.extend(["", f"## {dataset}" + (" — Shanghai" if dataset == "WWW2019" else "")])
            current_dataset = dataset
        lines.extend(["", f"### {unit}", ""])
        if not path.is_file():
            lines.append(f"> Chưa có báo cáo nguồn: `{path}`.")
            continue
        lines.append(body_with_shifted_headings(path.read_text(encoding="utf-8")))
        included += 1

    lines.extend(
        [
            "",
            "## Trạng thái tổng hợp",
            "",
            f"- Báo cáo đã đưa vào: **{included}/{len(reports)}**.",
            f"- Báo cáo nguồn còn thiếu: **{len(missing)}**.",
        ]
    )
    if missing:
        lines.extend(["", "### Tệp còn thiếu", ""])
        lines.extend(f"- `{path}`" for path in missing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"included={included}/{len(reports)} missing={len(missing)}")


if __name__ == "__main__":
    main()
