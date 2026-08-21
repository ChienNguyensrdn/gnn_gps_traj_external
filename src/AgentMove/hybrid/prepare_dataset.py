from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .cgm_adapter import export_queries


REQUIRED = {"user_id", "venue_id", "utc_time", "latitude", "longitude", "venue_category_name"}


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_input(path: str | Path, city: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        "user": "user_id",
        "Venue ID": "venue_id",
        "UTC Time": "utc_time",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Venue Category Name": "venue_category_name",
        "venue_cat_name": "venue_category_name",
    })
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Input {path} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["user_id", "venue_id", "utc_time", "latitude", "longitude"]).copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["venue_id"] = frame["venue_id"].astype(str)
    frame["datetime"] = pd.to_datetime(frame["utc_time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["datetime"])
    if "city" not in frame:
        frame["city"] = city
    else:
        frame["city"] = frame["city"].fillna(city)
    for column in ["admin", "subdistrict", "poi", "street"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return frame.sort_values(["user_id", "datetime"]).reset_index(drop=True)


def assign_trajectories(frame: pd.DataFrame, dataset: str, window_hours: int) -> pd.DataFrame:
    result = frame.copy()
    if dataset == "isp" and "traj_id" in result.columns:
        result["trajectory_id"] = result["user_id"] + "_" + result["traj_id"].astype(str)
        return result
    trajectory_ids: List[str] = [""] * len(result)
    for user_id, indices in result.groupby("user_id", sort=False).groups.items():
        session = 0
        start = None
        for index in indices:
            timestamp = result.at[index, "datetime"]
            if start is None or timestamp > start + pd.Timedelta(hours=window_hours):
                if start is not None:
                    session += 1
                start = timestamp
            trajectory_ids[index] = f"{user_id}_{session}"
    result["trajectory_id"] = trajectory_ids
    return result


def assign_temporal_splits(frame: pd.DataFrame, dataset: str, min_points: int) -> pd.DataFrame:
    valid = frame.groupby("trajectory_id").filter(lambda group: len(group) >= min_points).copy()
    split_by_trajectory: Dict[str, str] = {}
    ratios = (0.4, 0.1, 0.5) if dataset == "isp" else (0.7, 0.1, 0.2)
    for _, user_rows in valid.groupby("user_id", sort=False):
        ordered = (
            user_rows.groupby("trajectory_id")["datetime"].min().sort_values().index.tolist()
        )
        count = len(ordered)
        if count < 3:
            continue
        train_end = max(1, int(ratios[0] * count))
        validation_end = max(train_end + 1, int((ratios[0] + ratios[1]) * count))
        validation_end = min(validation_end, count - 1)
        for trajectory_id in ordered[:train_end]:
            split_by_trajectory[trajectory_id] = "train"
        for trajectory_id in ordered[train_end:validation_end]:
            split_by_trajectory[trajectory_id] = "validation"
        for trajectory_id in ordered[validation_end:]:
            split_by_trajectory[trajectory_id] = "test"
    valid["split"] = valid["trajectory_id"].map(split_by_trajectory)
    return valid.dropna(subset=["split"]).copy()


class MarkovCGM:
    def __init__(self, candidate_ids: Sequence[str], alpha: float = 0.1) -> None:
        self.candidate_ids = list(candidate_ids)
        self.alpha = alpha
        self.transitions: Dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def fit(self, trajectories: Iterable[Sequence[str]]) -> "MarkovCGM":
        for sequence in trajectories:
            self.global_counts.update(sequence)
            for previous, following in zip(sequence[:-1], sequence[1:]):
                self.transitions[str(previous)][str(following)] += 1
        return self

    def logits(self, previous: str) -> np.ndarray:
        transition = self.transitions.get(str(previous), Counter())
        total_global = sum(self.global_counts.values())
        scores = []
        for candidate in self.candidate_ids:
            local = transition[candidate]
            popularity = self.global_counts[candidate] / max(total_global, 1)
            scores.append(math.log(local + self.alpha + popularity))
        return np.asarray(scores, dtype=np.float32)


def candidate_metadata(frame: pd.DataFrame, candidate_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for venue_id, rows in frame[frame["venue_id"].isin(candidate_ids)].groupby("venue_id"):
        last = rows.iloc[-1]
        address = ", ".join(value for value in [last["admin"], last["subdistrict"], last["street"], last["poi"]] if value)
        metadata[str(venue_id)] = {
            "latitude": float(rows["latitude"].mean()),
            "longitude": float(rows["longitude"].mean()),
            "category": str(last["venue_category_name"] or ""),
            "address": address,
        }
    return metadata


def export_split(
    frame: pd.DataFrame,
    split: str,
    model: MarkovCGM,
    output_dir: Path,
    history_limit: int,
    context_limit: int,
) -> int:
    rows: List[Dict[str, Any]] = []
    logits: List[np.ndarray] = []
    for user_id, user_rows in frame.groupby("user_id", sort=False):
        user_rows = user_rows.sort_values("datetime")
        historical = user_rows[user_rows["split"] == "train"]
        history_records = [
            [row.datetime.isoformat(), row.venue_category_name, row.venue_id]
            for row in historical.itertuples()
        ][-history_limit:]
        for trajectory_id, trajectory in user_rows[user_rows["split"] == split].groupby("trajectory_id", sort=False):
            trajectory = trajectory.sort_values("datetime")
            if len(trajectory) < 2:
                continue
            target = trajectory.iloc[-1]
            if str(target["venue_id"]) not in model.candidate_ids:
                continue
            context_frame = trajectory.iloc[:-1].tail(context_limit)
            context = [
                [row.datetime.isoformat(), row.venue_category_name, row.venue_id]
                for row in context_frame.itertuples()
            ]
            if not context:
                continue
            query_id = f"{split}:{trajectory_id}"
            rows.append({
                "query_id": query_id,
                "user_id": str(user_id),
                "city": str(target["city"]),
                "true_id": str(target["venue_id"]),
                "history": history_records,
                "context": context,
                "target_time": target["datetime"].isoformat(),
                "metadata": {"trajectory_id": str(trajectory_id), "stage1": "markov-smoothed"},
            })
            logits.append(model.logits(str(context_frame.iloc[-1]["venue_id"])))
    if not rows:
        raise ValueError(f"No usable {split} queries after filtering")
    np.save(output_dir / f"{split}_logits.npy", np.stack(logits))
    metadata_path = output_dir / f"{split}_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    export_queries(
        str(output_dir / f"{split}_logits.npy"),
        str(metadata_path),
        str(output_dir / "candidate_ids.json"),
        str(output_dir / f"{split}.jsonl"),
        str(output_dir / "candidate_metadata.json"),
    )
    return len(rows)


def export_getnext_csv(frame: pd.DataFrame, output_dir: Path) -> None:
    encoded = frame.copy()
    candidate_ids = json.loads((output_dir / "candidate_ids.json").read_text(encoding="utf-8"))
    poi_map = {candidate: index for index, candidate in enumerate(candidate_ids)}
    user_ids = sorted(encoded["user_id"].unique())
    user_map = {user: index for index, user in enumerate(user_ids)}
    categories = sorted(encoded["venue_category_name"].fillna("").astype(str).unique())
    category_map = {category: index for index, category in enumerate(categories)}
    encoded = encoded[encoded["venue_id"].isin(poi_map)].copy()
    encoded["POI_id"] = encoded["venue_id"].map(poi_map)
    encoded["user_id"] = encoded["user_id"].map(user_map)
    encoded["POI_catname"] = encoded["venue_category_name"].fillna("")
    encoded["POI_catid"] = encoded["POI_catname"].map(category_map)
    encoded["POI_catid_code"] = encoded["POI_catid"]
    encoded["timezone"] = encoded["datetime"].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    encoded["UTC_time"] = encoded["timezone"]
    encoded["norm_in_day_time"] = (encoded["datetime"].dt.hour * 2 + encoded["datetime"].dt.minute // 30) / 48.0
    columns = ["user_id", "POI_id", "POI_catname", "POI_catid", "latitude", "longitude", "trajectory_id", "UTC_time", "timezone", "POI_catid_code", "norm_in_day_time"]
    getnext_dir = output_dir / "getnext"
    getnext_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        name = "val.csv" if split == "validation" else f"{split}.csv"
        encoded[encoded["split"] == split][columns].to_csv(getnext_dir / name, index=False)


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = normalize_input(args.input, args.city)
    frame = assign_trajectories(frame, args.dataset, args.window_hours)
    frame = assign_temporal_splits(frame, args.dataset, args.min_points)
    train = frame[frame["split"] == "train"]
    candidate_ids = sorted(train["venue_id"].unique().tolist())
    if len(candidate_ids) < 2:
        raise ValueError("Training split must contain at least two distinct POIs")
    (output_dir / "candidate_ids.json").write_text(json.dumps(candidate_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_metadata.json").write_text(
        json.dumps(candidate_metadata(frame, candidate_ids), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    trajectories = [group.sort_values("datetime")["venue_id"].tolist() for _, group in train.groupby("trajectory_id")]
    model = MarkovCGM(candidate_ids, args.alpha).fit(trajectories)
    validation_count = export_split(frame, "validation", model, output_dir, args.history_limit, args.context_limit)
    test_count = export_split(frame, "test", model, output_dir, args.history_limit, args.context_limit)
    export_getnext_csv(frame, output_dir)
    statistics = {
        "dataset": args.dataset,
        "city": args.city,
        "rows": len(frame),
        "users": int(frame["user_id"].nunique()),
        "locations_train": len(candidate_ids),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "validation_queries": validation_count,
        "test_queries": test_count,
        "split_rows": {key: int(value) for key, value in frame["split"].value_counts().to_dict().items()},
        "stage1": "smoothed first-order Markov; replace logits with GETNext for main paper results",
    }
    (output_dir / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ISP-Shanghai or TIST2015 for hybrid experiments")
    parser.add_argument("--dataset", required=True, choices=["isp", "tist2015"])
    parser.add_argument("--input", required=True, help="Normalized city CSV from AgentMove preprocessing")
    parser.add_argument("--city", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=40)
    parser.add_argument("--context-limit", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.1)
    return parser


def main() -> None:
    statistics = prepare(build_parser().parse_args())
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
