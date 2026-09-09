from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "uid": "user_id", "user": "user_id", "userid": "user_id", "user_id": "user_id",
    "d": "day", "day": "day", "day_id": "day",
    "t": "timeslot", "time": "timeslot", "timeslot": "timeslot", "time_slot": "timeslot",
    "x": "x", "y": "y",
}
REQUIRED = {"user_id", "day", "timeslot", "x", "y"}


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {column: ALIASES.get(str(column).strip().lower(), str(column)) for column in frame.columns}
    result = frame.rename(columns=rename)
    missing = REQUIRED - set(result.columns)
    if missing:
        raise ValueError(f"YJMob100K file is missing columns: {sorted(missing)}; found={list(frame.columns)}")
    return result


def read_chunks(path: Path, chunk_size: int):
    for chunk in pd.read_csv(path, chunksize=chunk_size, compression="infer"):
        yield canonical_columns(chunk)


def stable_sample(users: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit >= len(users):
        return sorted(users)
    return sorted(users, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())[:limit]


def select_users(path: Path, chunk_size: int, minimum: int, limit: int | None, seed: int) -> list[str]:
    counts: Counter[str] = Counter()
    for chunk in read_chunks(path, chunk_size):
        counts.update(chunk["user_id"].astype(str).value_counts().to_dict())
    eligible = [user for user, count in counts.items() if count >= minimum]
    selected = stable_sample(eligible, limit, seed)
    if not selected:
        raise ValueError("no YJMob100K users satisfy the selection criteria")
    return selected


def load_category_names(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    frame = pd.read_csv(path, header=None)
    values = frame.iloc[:, -1].dropna().astype(str).tolist()
    return values


def load_poi_labels(path: Path | None, categories_path: Path | None) -> dict[tuple[int, int], str]:
    if path is None or not path.is_file():
        return {}
    frame = pd.read_csv(path, compression="infer")
    lower = {str(column).lower(): column for column in frame.columns}
    x_col = lower.get("x", frame.columns[0]); y_col = lower.get("y", frame.columns[1])
    feature_columns = [column for column in frame.columns if column not in {x_col, y_col}]
    category_names = load_category_names(categories_path)
    labels: dict[tuple[int, int], str] = {}
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    for position, (_, row) in enumerate(frame.iterrows()):
        vector = values[position]
        if len(vector) and float(vector.max()) > 0:
            index = int(np.argmax(vector))
            label = category_names[index] if index < len(category_names) else str(feature_columns[index])
        else:
            label = "grid-cell"
        labels[(int(row[x_col]), int(row[y_col]))] = label
    return labels


def normalize(args: argparse.Namespace) -> dict:
    selected = set(select_users(args.input, args.chunk_size, args.min_observations, args.max_users, args.seed))
    poi_labels = load_poi_labels(args.poi, args.categories)
    pieces = []
    for chunk in read_chunks(args.input, args.chunk_size):
        chunk["user_id"] = chunk["user_id"].astype(str)
        chunk = chunk[chunk["user_id"].isin(selected)][["user_id", "day", "timeslot", "x", "y"]]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        raise ValueError("selected users have no readable observations")
    frame = pd.concat(pieces, ignore_index=True)
    for column in ("day", "timeslot", "x", "y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().astype({"day": int, "timeslot": int, "x": int, "y": int})
    frame = frame[(frame["day"].between(0, 74)) & (frame["timeslot"].between(0, 47))]
    frame = frame.sort_values(["user_id", "day", "timeslot"], kind="stable")
    frame = frame.drop_duplicates(["user_id", "day", "timeslot"], keep="last")
    if args.drop_consecutive_stays:
        groups = frame.groupby(["user_id", "day"], sort=False)
        previous_x = groups["x"].shift()
        previous_y = groups["y"].shift()
        changed = previous_x.isna() | frame["x"].ne(previous_x) | frame["y"].ne(previous_y)
        frame = frame.loc[changed]
    origin = pd.Timestamp("2000-01-03", tz="UTC")
    frame["utc_time"] = origin + pd.to_timedelta(frame["day"], unit="D") + pd.to_timedelta(frame["timeslot"] * 30, unit="m")
    frame["venue_id"] = frame["x"].astype(str) + "_" + frame["y"].astype(str)
    frame["traj_id"] = frame["day"].astype(str)
    frame["city"] = args.city
    frame["admin"] = "YJMob100K anonymous grid"
    frame["poi"] = "grid-cell-" + frame["venue_id"]
    # These are anonymized grid coordinates in kilometres, not geographic latitude/longitude.
    frame["longitude"] = frame["x"] * 0.5
    frame["latitude"] = frame["y"] * 0.5
    frame["venue_category_name"] = [poi_labels.get((x, y), "grid-cell") for x, y in zip(frame["x"], frame["y"])]
    output = frame[["city", "user_id", "traj_id", "utc_time", "venue_id", "longitude", "latitude",
                    "venue_category_name", "admin", "poi", "x", "y", "day", "timeslot"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    result = {
        "dataset": args.dataset_name, "source": str(args.input), "output": str(args.output),
        "users": int(output["user_id"].nunique()), "rows": len(output),
        "trajectories": int(output[["user_id", "traj_id"]].drop_duplicates().shape[0]),
        "grid_cells": int(output["venue_id"].nunique()), "drop_consecutive_stays": args.drop_consecutive_stays,
        "coordinate_semantics": "anonymous 500m grid; latitude/longitude columns store y/x kilometres only",
        "selection": {"max_users": args.max_users, "min_observations": args.min_observations, "seed": args.seed},
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize YJMob100K into the AgentMove trajectory schema")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--poi", type=Path)
    parser.add_argument("--categories", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="YJMob100K-Dataset1")
    parser.add_argument("--city", default="YJMob")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-consecutive-stays", dest="drop_consecutive_stays", action="store_false")
    parser.set_defaults(drop_consecutive_stays=True)
    return parser


if __name__ == "__main__":
    normalize(build_parser().parse_args())
