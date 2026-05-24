"""Deterministic integer-grid geocell quantizer.

Implements the shared geocell contract (shared/geocell-contract.md).
Both this module and ts-service/src/geocell.ts MUST produce identical
cell IDs for identical (lat, lng) inputs.
"""

import math

LAT_ORIGIN: float = 0.0
LNG_ORIGIN: float = -180.0
CELL_SIZE_DEG: float = 0.01  # ~1km cells (coarser = smaller vocab, more signal per cell)


def quantize(lat: float, lng: float) -> str:
    iy = math.floor((lat - LAT_ORIGIN) / CELL_SIZE_DEG)
    ix = math.floor((lng - LNG_ORIGIN) / CELL_SIZE_DEG)
    return f"g_{iy}_{ix}"


def parse_geocell(cell_id: str) -> tuple[int, int]:
    parts = cell_id.split("_")
    return int(parts[1]), int(parts[2])


def centroid(iy: int, ix: int) -> tuple[float, float]:
    lat = LAT_ORIGIN + (iy + 0.5) * CELL_SIZE_DEG
    lng = LNG_ORIGIN + (ix + 0.5) * CELL_SIZE_DEG
    return lat, lng


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
