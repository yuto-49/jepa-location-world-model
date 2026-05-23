"""Load Foursquare NYC+Tokyo check-in data into a DataFrame.

Expected raw files in data/raw/:
  - dataset_TSMC2014_NYC.txt
  - dataset_TSMC2014_TKY.txt

Columns (tab-separated, no header):
  userId, venueId, venueCategoryId, venueCategoryName,
  latitude, longitude, timezoneOffset, utcTimestamp
"""

import os
import hashlib
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
SALT = b"jepa-location-2026"

COLUMNS = [
    "user_id_raw",
    "venue_id",
    "venue_category_id",
    "category",
    "lat",
    "lng",
    "tz_offset",
    "utc_timestamp",
]


def _hash_user_id(raw_id: str) -> str:
    h = hashlib.sha256(SALT + str(raw_id).encode()).hexdigest()[:12]
    return f"u_{h}"


def load_foursquare(cities: list[str] | None = None) -> pd.DataFrame:
    if cities is None:
        cities = ["NYC", "TKY"]

    frames = []
    for city in cities:
        path = os.path.join(RAW_DIR, f"dataset_TSMC2014_{city}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Download the Foursquare TSMC2014 dataset "
                f"and place files in {RAW_DIR}/"
            )
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=COLUMNS,
            encoding="latin-1",
        )
        df["city"] = city
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["user_id"] = combined["user_id_raw"].apply(_hash_user_id)
    combined["timestamp"] = pd.to_datetime(
        combined["utc_timestamp"], format="mixed", utc=True
    )
    combined = combined[["user_id", "timestamp", "lat", "lng", "category"]].copy()
    combined = combined.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return combined


def print_stats(df: pd.DataFrame) -> None:
    print(f"Total check-ins: {len(df):,}")
    print(f"Unique users:    {df['user_id'].nunique():,}")
    print(f"Date range:      {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Lat range:       [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"Lng range:       [{df['lng'].min():.4f}, {df['lng'].max():.4f}]")
    print(f"Categories:      {df['category'].nunique():,}")


if __name__ == "__main__":
    df = load_foursquare()
    print_stats(df)
