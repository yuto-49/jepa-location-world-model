/**
 * Convert stay points into geocell token sequences.
 */

import { StayPoint } from "./episodes.js";
import { quantize } from "./geocell.js";

export interface GeoToken {
  geocell: string;
  hour: number;
  dow: number;
}

export interface TrajectoryRecord {
  user_id: string;
  date: string;
  tokens: GeoToken[];
}

export function buildSequences(stays: StayPoint[]): TrajectoryRecord[] {
  const byUserDate = new Map<string, StayPoint[]>();
  for (const stay of stays) {
    const date = stay.arrivalTime.slice(0, 10);
    const key = `${stay.userId}|${date}`;
    if (!byUserDate.has(key)) byUserDate.set(key, []);
    byUserDate.get(key)!.push(stay);
  }
  const records: TrajectoryRecord[] = [];
  for (const [key, dayStays] of byUserDate) {
    const [userId, date] = key.split("|");
    const sorted = dayStays.sort(
      (a, b) => Date.parse(a.arrivalTime) - Date.parse(b.arrivalTime)
    );
    const tokens: GeoToken[] = sorted.map((stay) => {
      const dt = new Date(stay.arrivalTime);
      return {
        geocell: quantize(stay.latitude, stay.longitude),
        hour: dt.getUTCHours(),
        dow: (dt.getUTCDay() + 6) % 7,
      };
    });
    if (tokens.length >= 2) {
      records.push({ user_id: userId, date, tokens });
    }
  }
  return records;
}
