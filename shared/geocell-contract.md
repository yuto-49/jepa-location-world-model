# Geocell Contract

Both Python (`py-model/data/geocell.py`) and TypeScript (`ts-service/src/geocell.ts`)
MUST implement this exact specification. A 1000-pair parity test guards cross-language agreement.

## Quantization Parameters

| Parameter | Value |
|-----------|-------|
| LAT_ORIGIN | 0.0 |
| LNG_ORIGIN | -180.0 |
| CELL_SIZE_DEG | 0.005 |

## Functions

### quantize(lat: float, lng: float) -> str

```
iy = floor((lat - LAT_ORIGIN) / CELL_SIZE_DEG)
ix = floor((lng - LNG_ORIGIN) / CELL_SIZE_DEG)
return "g_{iy}_{ix}"
```

### centroid(iy: int, ix: int) -> (float, float)

```
lat = LAT_ORIGIN + (iy + 0.5) * CELL_SIZE_DEG
lng = LNG_ORIGIN + (ix + 0.5) * CELL_SIZE_DEG
return (lat, lng)
```

### parse_geocell(cell_id: str) -> (int, int)

```
parts = cell_id.split("_")  # ["g", "{iy}", "{ix}"]
return (int(parts[1]), int(parts[2]))
```

## Sequence Record Schema (JSONL)

One line per trajectory (one user, one day):

```json
{
  "user_id": "u_<salted_hash>",
  "date": "YYYY-MM-DD",
  "tokens": [
    { "geocell": "g_{iy}_{ix}", "hour": 0-23, "dow": 0-6 }
  ]
}
```

- `dow`: Monday=0, Sunday=6 (Python `datetime.weekday()` convention)
- No raw lat/lng ever appears in exported records

## Privacy Rules

1. No raw coordinates in exported JSONL
2. user_id is a salted hash (salt not exported)
3. Cells with < 5 distinct users mapped to `<UNK>` before export
4. Model predicts cells only -- no demographic/identity prediction heads
