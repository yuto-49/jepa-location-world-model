"""Generate synthetic Foursquare TSMC2014-format check-in data for testing.

Outputs tab-separated files (no header) matching the real dataset format:
  userId, venueId, venueCategoryId, venueCategoryName,
  latitude, longitude, timezoneOffset, utcTimestamp

Files written to data/raw/:
  - dataset_TSMC2014_NYC.txt
  - dataset_TSMC2014_TKY.txt
"""

import os
import random
import string
from datetime import datetime, timedelta, timezone

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

CATEGORIES = [
    ("4bf58dd8d48988d116941735", "Food"),
    ("4bf58dd8d48988d1d1941735", "Coffee Shop"),
    ("4bf58dd8d48988d1fa931735", "Bar"),
    ("4bf58dd8d48988d1ce941735", "Hotel"),
    ("4bf58dd8d48988d181941735", "Arts & Entertainment"),
    ("4bf58dd8d48988d182941735", "Outdoors & Recreation"),
    ("4bf58dd8d48988d1e2931735", "Shop & Service"),
    ("4bf58dd8d48988d1dc931735", "Travel & Transport"),
    ("4bf58dd8d48988d11e941735", "Nightlife Spot"),
    ("4bf58dd8d48988d124941735", "Gym"),
    ("4bf58dd8d48988d17f941735", "Museum"),
    ("4bf58dd8d48988d1e3931735", "Bookstore"),
    ("4bf58dd8d48988d117941735", "Pizza Place"),
    ("4bf58dd8d48988d110941735", "American Restaurant"),
    ("4bf58dd8d48988d1c1941735", "Park"),
    ("4bf58dd8d48988d165941735", "Office"),
    ("4bf58dd8d48988d178941735", "College & University"),
    ("4bf58dd8d48988d1f2931735", "Pharmacy"),
    ("4bf58dd8d48988d139941735", "Supermarket"),
    ("4bf58dd8d48988d102941735", "Airport"),
]

CITY_CONFIGS = {
    "NYC": {
        "lat_range": (40.6, 40.9),
        "lng_range": (-74.1, -73.7),
        "tz_offset": -300,
        "n_users": 100,
        "n_checkins": 1000,
    },
    "TKY": {
        "lat_range": (35.5, 35.8),
        "lng_range": (139.5, 139.9),
        "tz_offset": 540,
        "n_users": 100,
        "n_checkins": 1000,
    },
}

START_DATE = datetime(2012, 4, 3, tzinfo=timezone.utc)
END_DATE = datetime(2013, 9, 16, tzinfo=timezone.utc)
TOTAL_SECONDS = int((END_DATE - START_DATE).total_seconds())


def _rand_id(prefix: str, length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def _rand_timestamp() -> str:
    offset_secs = random.randint(0, TOTAL_SECONDS)
    dt = START_DATE + timedelta(seconds=offset_secs)
    return dt.strftime("%a %b %d %H:%M:%S +0000 %Y")


def generate_city(city: str) -> list[list]:
    cfg = CITY_CONFIGS[city]
    lat_lo, lat_hi = cfg["lat_range"]
    lng_lo, lng_hi = cfg["lng_range"]
    tz_offset = cfg["tz_offset"]
    n_users = cfg["n_users"]
    n_checkins = cfg["n_checkins"]

    user_ids = [_rand_id("u", 6) for _ in range(n_users)]
    venue_pool = [
        (_rand_id("v", 16), random.choice(CATEGORIES))
        for _ in range(n_checkins // 5)
    ]

    rows = []
    for _ in range(n_checkins):
        user_id = random.choice(user_ids)
        venue_id, (cat_id, cat_name) = random.choice(venue_pool)
        lat = round(random.uniform(lat_lo, lat_hi), 6)
        lng = round(random.uniform(lng_lo, lng_hi), 6)
        ts = _rand_timestamp()
        rows.append([user_id, venue_id, cat_id, cat_name, lat, lng, tz_offset, ts])

    return rows


def write_tsv(rows: list[list], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"Wrote {len(rows):,} rows -> {path}")


def main() -> None:
    random.seed(42)
    for city in ("NYC", "TKY"):
        rows = generate_city(city)
        out_path = os.path.join(RAW_DIR, f"dataset_TSMC2014_{city}.txt")
        write_tsv(rows, out_path)
    print("Done. Synthetic data ready for testing.")


if __name__ == "__main__":
    main()
