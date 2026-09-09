from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

import requests


RECORD = "https://zenodo.org/records/10142719/files"
FILES = {
    "dataset1": "yjmob100k-dataset1.csv.gz",
    "dataset2": "yjmob100k-dataset2.csv.gz",
    "poi": "cell_POIcat.csv.gz",
    "categories": "POI_datacategories.csv",
}


def download(name: str, destination: Path) -> Path:
    filename = FILES[name]
    target = destination / filename
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another download is active: {lock_path}") from exc
        downloaded = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        url = f"{RECORD}/{filename}?download=1"
        with requests.get(url, headers=headers, stream=True, timeout=180) as response:
            response.raise_for_status()
            content_range = response.headers.get("content-range", "")
            resumed = downloaded > 0 and response.status_code == 206 and content_range.startswith(f"bytes {downloaded}-")
            mode = "ab" if resumed else "wb"
            if downloaded and not resumed:
                downloaded = 0
            with target.open(mode) as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        print(f"ready={target} bytes={target.stat().st_size}", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YJMob100K from the official Zenodo record")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset_yjmob100k"))
    parser.add_argument("--files", nargs="+", choices=sorted(FILES), default=["dataset1", "poi", "categories"])
    args = parser.parse_args()
    for name in args.files:
        download(name, args.output_dir)


if __name__ == "__main__":
    main()
