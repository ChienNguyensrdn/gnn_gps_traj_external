from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CITIES = [
    "Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris",
    "Beijing", "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow",
]

# Pipeline/output identifiers follow AgentMove's compact city names, while the
# official TIST2015 catalog stores four multi-word names with spaces.
CITY_CATALOG_ALIASES = {
    "NewYork": "New York",
    "CapeTown": "Cape Town",
    "SanFrancisco": "San Francisco",
    "SaoPaulo": "Sao Paulo",
}


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray, city_lat: np.ndarray, city_lon: np.ndarray) -> np.ndarray:
    lat1 = np.radians(latitudes)[:, None]
    lon1 = np.radians(longitudes)[:, None]
    lat2 = np.radians(city_lat)[None, :]
    lon2 = np.radians(city_lon)[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def extract(input_dir: str, output_dir: str, cities: list[str], chunk_size: int = 20_000) -> None:
    source = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cities_frame = pd.read_csv(
        source / "dataset_TIST2015_Cities.txt", sep="\t", header=None,
        names=["city", "latitude", "longitude", "country", "type", "timezone"],
    )
    catalog_names = {city: CITY_CATALOG_ALIASES.get(city, city) for city in cities}
    missing_cities = sorted(city for city, catalog in catalog_names.items() if catalog not in set(cities_frame["city"]))
    if missing_cities:
        raise ValueError(f"Cities not found in TIST2015 city catalog: {missing_cities}")
    # Assign against the complete catalog first. Restricting the distance search
    # to the 12 experiment cities would incorrectly pull unrelated global POIs
    # into the nearest experiment city.
    complete_catalog = cities_frame.reset_index(drop=True)
    pois = pd.read_csv(
        source / "dataset_TIST2015_POIs.txt", sep="\t", header=None,
        names=["venue_id", "latitude", "longitude", "venue_category_name", "country_code"],
    )
    assignments: list[np.ndarray] = []
    for start in range(0, len(pois), chunk_size):
        chunk = pois.iloc[start:start + chunk_size]
        distances = haversine_matrix(
            chunk["latitude"].to_numpy(), chunk["longitude"].to_numpy(),
            complete_catalog["latitude"].to_numpy(), complete_catalog["longitude"].to_numpy(),
        )
        assignments.append(complete_catalog.iloc[np.argmin(distances, axis=1)]["city"].to_numpy())
    pois["city"] = np.concatenate(assignments)
    catalog_to_output = {catalog: city for city, catalog in catalog_names.items()}
    pois = pois[pois["city"].isin(catalog_to_output)].copy()
    pois["city"] = pois["city"].map(catalog_to_output)
    poi_lookup = pois.set_index("venue_id")
    output_paths = {city: destination / f"{city}_filtered.csv" for city in cities}
    wrote_header = {city: False for city in cities}
    for path in output_paths.values():
        if path.exists():
            path.unlink()
    checkins_path = source / "dataset_TIST2015_Checkins.txt"
    for checkins in pd.read_csv(
        checkins_path, sep="\t", header=None,
        names=["user_id", "venue_id", "utc_time", "timezone_offset"],
        chunksize=chunk_size,
    ):
        joined = checkins.join(poi_lookup, on="venue_id", how="inner")
        for city, group in joined.groupby("city"):
            if city in output_paths:
                output = group[[
                    "city", "user_id", "timezone_offset", "venue_id", "utc_time",
                    "longitude", "latitude", "venue_category_name",
                ]].rename(columns={"timezone_offset": "time"})
                output.to_csv(output_paths[city], mode="a", header=not wrote_header[city], index=False)
                wrote_header[city] = True
    for city in cities:
        if not wrote_header[city]:
            raise ValueError(f"No check-ins found for {city}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the 12 paper cities from raw Foursquare TIST2015")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    extract(args.input_dir, args.output_dir, args.cities, args.chunk_size)


if __name__ == "__main__":
    main()
