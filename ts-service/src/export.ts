/**
 * Privacy-filtered JSONL export.
 * No raw coordinates appear in exported records.
 */

import { createHash } from "crypto";
import { TrajectoryRecord } from "./sequence.js";

const SALT = "jepa-location-2026";

export function hashUserId(rawId: string): string {
  const hash = createHash("sha256")
    .update(SALT + rawId)
    .digest("hex")
    .slice(0, 12);
  return `u_${hash}`;
}

export function exportJsonl(records: TrajectoryRecord[]): string {
  return records
    .map((r) => {
      const sanitized = {
        user_id: r.user_id.startsWith("u_") ? r.user_id : hashUserId(r.user_id),
        date: r.date,
        tokens: r.tokens.map(({ geocell, hour, dow }) => ({ geocell, hour, dow })),
      };
      return JSON.stringify(sanitized);
    })
    .join("\n");
}
