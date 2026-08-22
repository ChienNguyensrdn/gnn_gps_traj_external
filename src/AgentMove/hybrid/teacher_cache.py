from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class ImmutableTeacherCache:
    """Content-addressed JSONL cache; existing keys may be reused but never changed."""

    def __init__(self, path: str | Path, namespace: str, version: str) -> None:
        self.path = Path(path); self.namespace = namespace; self.version = version
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.records[str(row["key"])] = row

    def put(self, key: str, value: Any, source_hash: str) -> dict[str, Any]:
        record = {"key": str(key), "namespace": self.namespace, "version": self.version,
                  "source_hash": source_hash, "value": value, "value_hash": canonical_hash(value)}
        existing = self.records.get(str(key))
        if existing:
            if existing != record:
                raise ValueError(f"immutable teacher-cache conflict for key={key}")
            return existing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.records[str(key)] = record
        return record

    def get(self, key: str) -> Any | None:
        record = self.records.get(str(key))
        return None if record is None else record["value"]


def import_jsonl(source: Path, cache: ImmutableTeacherCache, key_field: str, value_fields: Iterable[str]) -> int:
    count = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); key = str(row[key_field])
        value = {field: row[field] for field in value_fields if field in row}
        cache.put(key, value, canonical_hash(row)); count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable quantitative/LLM teacher cache")
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", choices=["quantitative", "llm"], required=True)
    parser.add_argument("--version", required=True); parser.add_argument("--key-field", default="query_id")
    parser.add_argument("--value-fields", nargs="+", required=True)
    args = parser.parse_args(); cache = ImmutableTeacherCache(args.output, args.namespace, args.version)
    count = import_jsonl(args.input, cache, args.key_field, args.value_fields)
    print(json.dumps({"imported": count, "cache": str(cache.path), "records": len(cache.records)}))


if __name__ == "__main__":
    main()
