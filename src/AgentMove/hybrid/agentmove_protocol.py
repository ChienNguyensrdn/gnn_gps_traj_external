from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .io import read_jsonl, write_json
from .llm_only import _select_one_session_per_user


def _write_subset(source: Path, output: Path, rows: List[Dict[str, Any]]) -> None:
    source_rows = list(read_jsonl(source))
    bundle = source_rows[0] if source_rows and "_bundle" in source_rows[0] else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        if bundle is not None:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(validation: Path, test: Path, output_dir: Path, test_users: int) -> dict:
    validation_rows = [row for row in read_jsonl(validation) if "_bundle" not in row]
    test_rows = [row for row in read_jsonl(test) if "_bundle" not in row]
    selected = _select_one_session_per_user(test_rows, test_users)
    if len(selected) != test_users:
        raise ValueError(f"Requested {test_users} users but only selected {len(selected)}")
    _write_subset(validation, output_dir / "validation.jsonl", validation_rows)
    _write_subset(test, output_dir / "test.jsonl", selected)
    ids = [str(row["query_id"]) for row in selected]
    manifest = {
        "protocol": "agentmove-one-session-per-user-v1",
        "validation": {"source": str(validation.resolve()), "queries": len(validation_rows)},
        "test": {
            "source": str(test.resolve()), "source_queries": len(test_rows),
            "selected_queries": len(selected), "unique_users": len({str(row["user_id"]) for row in selected}),
            "selection": "users and trajectories sorted by ID; one session per user; first n users",
            "query_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        },
    }
    write_json(output_dir / "sample_manifest.json", manifest)
    return manifest


def seed_cache(source: Path, destination: Path, allowed_ids: set[str]) -> int:
    if destination.exists():
        return sum(1 for row in read_jsonl(destination) if str(row.get("query_id")) in allowed_ids)
    if not source.exists():
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(source):
            if str(row.get("query_id")) in allowed_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AgentMove-compatible Hybrid evaluation split")
    parser.add_argument("--validation", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--test-users", type=int, default=200)
    parser.add_argument("--seed-cache-from"); parser.add_argument("--result-dir")
    args = parser.parse_args(); output = Path(args.output_dir)
    manifest = prepare(Path(args.validation), Path(args.test), output, args.test_users)
    if args.seed_cache_from and args.result_dir:
        allowed = {
            str(row["query_id"]) for name in ("validation.jsonl", "test.jsonl")
            for row in read_jsonl(output / name) if "_bundle" not in row
        }
        base, result = Path(args.seed_cache_from), Path(args.result_dir)
        manifest["seeded_evidence_rows"] = {
            "embedding": seed_cache(base / "evidence_cache.jsonl", result / "evidence_cache.jsonl", allowed),
            "frequency": seed_cache(base / "evidence_cache_no_embedding_memory.jsonl", result / "evidence_cache_no_embedding_memory.jsonl", allowed),
        }
        write_json(output / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
