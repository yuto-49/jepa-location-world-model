/**
 * Deterministic integer-grid geocell quantizer.
 * MUST produce identical IDs to py-model/data/geocell.py.
 * See shared/geocell-contract.md.
 */

export const LAT_ORIGIN = 0.0;
export const LNG_ORIGIN = -180.0;
export const CELL_SIZE_DEG = 0.005;

export function quantize(lat: number, lng: number): string {
  const iy = Math.floor((lat - LAT_ORIGIN) / CELL_SIZE_DEG);
  const ix = Math.floor((lng - LNG_ORIGIN) / CELL_SIZE_DEG);
  return `g_${iy}_${ix}`;
}

export function parseGeocell(cellId: string): [number, number] {
  const parts = cellId.split("_");
  return [parseInt(parts[1], 10), parseInt(parts[2], 10)];
}

export function centroid(iy: number, ix: number): [number, number] {
  const lat = LAT_ORIGIN + (iy + 0.5) * CELL_SIZE_DEG;
  const lng = LNG_ORIGIN + (ix + 0.5) * CELL_SIZE_DEG;
  return [lat, lng];
}
