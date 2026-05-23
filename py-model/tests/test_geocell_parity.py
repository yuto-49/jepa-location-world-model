import json
import math
import os
import pytest

from data.geocell import quantize, centroid, parse_geocell, LAT_ORIGIN, LNG_ORIGIN, CELL_SIZE_DEG

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "geocell_parity.json")


def generate_parity_fixture(n: int = 1000, seed: int = 42) -> list[dict]:
    import random
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        lat = rng.uniform(-60.0, 70.0)
        lng = rng.uniform(-180.0, 180.0)
        iy = math.floor((lat - LAT_ORIGIN) / CELL_SIZE_DEG)
        ix = math.floor((lng - LNG_ORIGIN) / CELL_SIZE_DEG)
        expected_id = f"g_{iy}_{ix}"
        pairs.append({"lat": lat, "lng": lng, "expected_id": expected_id})
    return pairs


@pytest.fixture(scope="session")
def parity_data() -> list[dict]:
    if os.path.exists(FIXTURE_PATH):
        with open(FIXTURE_PATH) as f:
            return json.load(f)
    data = generate_parity_fixture()
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(data, f)
    return data


def test_quantize_matches_fixture(parity_data):
    for entry in parity_data:
        result = quantize(entry["lat"], entry["lng"])
        assert result == entry["expected_id"], (
            f"quantize({entry['lat']}, {entry['lng']}) = {result}, expected {entry['expected_id']}"
        )


def test_parse_geocell_roundtrip(parity_data):
    for entry in parity_data:
        cell_id = quantize(entry["lat"], entry["lng"])
        iy, ix = parse_geocell(cell_id)
        assert cell_id == f"g_{iy}_{ix}"


def test_centroid_within_cell(parity_data):
    for entry in parity_data:
        cell_id = quantize(entry["lat"], entry["lng"])
        iy, ix = parse_geocell(cell_id)
        clat, clng = centroid(iy, ix)
        cell_min_lat = LAT_ORIGIN + iy * CELL_SIZE_DEG
        cell_max_lat = cell_min_lat + CELL_SIZE_DEG
        cell_min_lng = LNG_ORIGIN + ix * CELL_SIZE_DEG
        cell_max_lng = cell_min_lng + CELL_SIZE_DEG
        assert cell_min_lat < clat < cell_max_lat
        assert cell_min_lng < clng < cell_max_lng


def test_quantize_count(parity_data):
    assert len(parity_data) == 1000
