"""Load Gowalla check-in data from Stanford SNAP.

Source: https://snap.stanford.edu/data/loc-gowalla.html
File: loc-gowalla_totalCheckins.txt

Format (tab-separated, no header):
  user_id_raw  check_in_time  latitude  longitude  location_id
  196514       2010-07-24T13:45:06Z  53.36  -6.23  29593
"""

import os
import hashlib
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
SALT = b"jepa-location-2026"

COLUMNS = ["user_id_raw", "utc_timestamp", "lat", "lng", "location_id"]


def _hash_user_id(raw_id: str) -> str:
    h = hashlib.sha256(SALT + str(raw_id).encode()).hexdigest()[:12]
    return f"u_{h}"


def load_gowalla() -> pd.DataFrame:
    """Load Gowalla check-ins.

    Returns:
        DataFrame with columns: user_id, timestamp, lat, lng
    """
    path = os.path.join(RAW_DIR, "loc-gowalla_totalCheckins.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Download from "
            "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
    )

    # Filter out invalid coordinates
    df = df[(df["lat"] != 0.0) & (df["lng"] != 0.0)].copy()
    df = df[(df["lat"].between(-90, 90)) & (df["lng"].between(-180, 180))].copy()

    df["user_id"] = df["user_id_raw"].apply(_hash_user_id)
    df["timestamp"] = pd.to_datetime(df["utc_timestamp"])
    df = df[["user_id", "timestamp", "lat", "lng"]].copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df


def print_stats(df: pd.DataFrame) -> None:
    print(f"Total check-ins: {len(df):,}")
    print(f"Unique users:    {df['user_id'].nunique():,}")
    print(f"Date range:      {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Lat range:       [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"Lng range:       [{df['lng'].min():.4f}, {df['lng'].max():.4f}]")

    # Per-user stats
    user_counts = df.groupby("user_id").size()
    print(f"Check-ins/user:  median={user_counts.median():.0f}, mean={user_counts.mean():.1f}, max={user_counts.max()}")

    # Daily trajectory length
    df_tmp = df.copy()
    df_tmp["date"] = df_tmp["timestamp"].dt.date
    daily = df_tmp.groupby(["user_id", "date"]).size()
    print(f"Check-ins/day:   median={daily.median():.0f}, mean={daily.mean():.1f}, max={daily.max()}")


if __name__ == "__main__":
    df = load_gowalla()
    print_stats(df)
