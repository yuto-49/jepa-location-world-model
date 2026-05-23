import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { quantize, parseGeocell, centroid, LAT_ORIGIN, LNG_ORIGIN, CELL_SIZE_DEG } from "../src/geocell.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ParityEntry {
  lat: number;
  lng: number;
  expected_id: string;
}

const fixturePath = resolve(__dirname, "../../py-model/tests/fixtures/geocell_parity.json");
const parityData: ParityEntry[] = JSON.parse(readFileSync(fixturePath, "utf-8"));

describe("geocell parity with Python", () => {
  it("should have 1000 test pairs", () => {
    expect(parityData.length).toBe(1000);
  });

  it("quantize matches all 1000 fixture entries", () => {
    for (const entry of parityData) {
      const result = quantize(entry.lat, entry.lng);
      expect(result).toBe(entry.expected_id);
    }
  });

  it("parseGeocell roundtrips", () => {
    for (const entry of parityData) {
      const cellId = quantize(entry.lat, entry.lng);
      const [iy, ix] = parseGeocell(cellId);
      expect(cellId).toBe(`g_${iy}_${ix}`);
    }
  });

  it("centroid is within cell bounds", () => {
    for (const entry of parityData) {
      const cellId = quantize(entry.lat, entry.lng);
      const [iy, ix] = parseGeocell(cellId);
      const [clat, clng] = centroid(iy, ix);
      const minLat = LAT_ORIGIN + iy * CELL_SIZE_DEG;
      const maxLat = minLat + CELL_SIZE_DEG;
      const minLng = LNG_ORIGIN + ix * CELL_SIZE_DEG;
      const maxLng = minLng + CELL_SIZE_DEG;
      expect(clat).toBeGreaterThan(minLat);
      expect(clat).toBeLessThan(maxLat);
      expect(clng).toBeGreaterThan(minLng);
      expect(clng).toBeLessThan(maxLng);
    }
  });
});
