/**
 * Cluster consecutive location pings into stay points (episodes).
 */

import { LocationEvent } from "./ingestion.js";

export interface StayPoint {
  userId: string;
  latitude: number;
  longitude: number;
  arrivalTime: string;
  departureTime: string;
  dwellMinutes: number;
}

const DWELL_THRESHOLD_MIN = 5;
const DISTANCE_THRESHOLD_DEG = 0.001;

function degDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
  return Math.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2);
}

export function buildStayPoints(events: LocationEvent[]): StayPoint[] {
  if (events.length === 0) return [];
  const sorted = [...events].sort(
    (a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp)
  );
  const stays: StayPoint[] = [];
  let cluster: LocationEvent[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const prev = cluster[cluster.length - 1];
    const curr = sorted[i];
    const dist = degDistance(prev.latitude, prev.longitude, curr.latitude, curr.longitude);
    if (dist <= DISTANCE_THRESHOLD_DEG) {
      cluster.push(curr);
    } else {
      const stay = finalizeCluster(cluster);
      if (stay) stays.push(stay);
      cluster = [curr];
    }
  }
  const lastStay = finalizeCluster(cluster);
  if (lastStay) stays.push(lastStay);
  return stays;
}

function finalizeCluster(cluster: LocationEvent[]): StayPoint | null {
  if (cluster.length === 0) return null;
  const arrival = Date.parse(cluster[0].timestamp);
  const departure = Date.parse(cluster[cluster.length - 1].timestamp);
  const dwellMin = (departure - arrival) / 60000;
  if (dwellMin < DWELL_THRESHOLD_MIN && cluster.length < 2) return null;
  const avgLat = cluster.reduce((s, e) => s + e.latitude, 0) / cluster.length;
  const avgLng = cluster.reduce((s, e) => s + e.longitude, 0) / cluster.length;
  return {
    userId: cluster[0].userId,
    latitude: avgLat,
    longitude: avgLng,
    arrivalTime: cluster[0].timestamp,
    departureTime: cluster[cluster.length - 1].timestamp,
    dwellMinutes: Math.round(dwellMin),
  };
}
