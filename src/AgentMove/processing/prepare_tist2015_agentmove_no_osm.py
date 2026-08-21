from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CITIES = [
    "Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris",
    "Beijing", "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow",
]


def prepare(input_dir: Path, output_dir: Path, cities: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for city in cities:
        source = input_dir / f"{city}_filtered.csv"
        destination = output_dir / f"{city}_filtered.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing extracted TIST2015 city CSV: {source}")
        frame = pd.read_csv(source, dtype={"venue_id": str})
        category = frame["venue_category_name"].fillna("").astype(str)
        # AgentMove requires four address columns. These deterministic fallbacks
        # keep the run executable without pretending that OSM enrichment exists.
        frame["admin"] = city
        frame["subdistrict"] = ""
        frame["poi"] = category
        frame["street"] = ""
        frame.to_csv(destination, index=False)
        print(f"prepared {city}: rows={len(frame)} output={destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare no-OSM TIST2015 CSVs for original AgentMove")
    parser.add_argument("--input-dir", default="data/input_trajectories")
    parser.add_argument("--output-dir", default="data/input_trajectories_clean")
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    args = parser.parse_args()
    prepare(Path(args.input_dir), Path(args.output_dir), args.cities)


if __name__ == "__main__":
    main()
