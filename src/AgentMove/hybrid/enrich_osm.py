from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def reverse_geocode(base_url: str, latitude: float, longitude: float, user_agent: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/reverse?" + urlencode({
        "format": "jsonv2", "lat": latitude, "lon": longitude,
        "zoom": 18, "addressdetails": 1, "accept-language": "en-US",
    })
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_address(payload: Dict[str, Any]) -> Dict[str, str]:
    address = payload.get("address") or {}
    return {
        "admin": str(address.get("state") or address.get("city") or address.get("province") or ""),
        "subdistrict": str(address.get("suburb") or address.get("quarter") or address.get("neighbourhood") or address.get("district") or ""),
        "poi": str(address.get("amenity") or address.get("building") or address.get("shop") or address.get("tourism") or payload.get("name") or ""),
        "street": str(address.get("road") or address.get("pedestrian") or address.get("footway") or ""),
        "osm_category": str(payload.get("category") or ""),
        "osm_type": str(payload.get("type") or ""),
    }


def enrich(input_path: str, output_path: str, cache_path: str, base_url: str,
           user_agent: str, delay_seconds: float) -> Dict[str, int]:
    frame = pd.read_csv(input_path)
    required = {"venue_id", "latitude", "longitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OSM input is missing columns: {sorted(missing)}")
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Dict[str, str]] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[str(row["venue_id"])] = row["address"]
    venues = frame.groupby("venue_id", sort=False).first().reset_index()
    fetched = failed = 0
    with cache_file.open("a", encoding="utf-8") as handle:
        for row in venues.itertuples():
            venue_id = str(row.venue_id)
            if venue_id in cache:
                continue
            try:
                address = normalize_address(reverse_geocode(base_url, float(row.latitude), float(row.longitude), user_agent))
                fetched += 1
            except Exception as exc:
                address = {"admin": "", "subdistrict": "", "poi": "", "street": "", "osm_category": "", "osm_type": ""}
                address["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            cache[venue_id] = address
            handle.write(json.dumps({"venue_id": venue_id, "address": address}, ensure_ascii=False) + "\n")
            handle.flush()
            if delay_seconds:
                time.sleep(delay_seconds)
    for column in ["admin", "subdistrict", "poi", "street", "osm_category", "osm_type"]:
        frame[column] = frame["venue_id"].astype(str).map(lambda value: cache.get(value, {}).get(column, ""))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {"unique_venues": len(venues), "cached": len(venues) - fetched - failed, "fetched": fetched, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich hybrid trajectory POIs with Nominatim/OSM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--user-agent", default="HybridGPS-Traj/1.0 research")
    parser.add_argument("--delay-seconds", type=float, default=0.0,
                        help="Use >=1.0 with the public Nominatim service")
    args = parser.parse_args()
    print(json.dumps(enrich(args.input, args.output, args.cache, args.base_url, args.user_agent, args.delay_seconds), indent=2))


if __name__ == "__main__":
    main()
