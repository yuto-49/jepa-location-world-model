/**
 * Consent-gated location event ingestion.
 */

export interface LocationEvent {
  userId: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  consentVersion: string;
}

export interface IngestResult {
  accepted: boolean;
  reason?: string;
}

const REQUIRED_CONSENT_VERSION = "1.0";

export function ingestLocationEvent(event: LocationEvent): IngestResult {
  if (!event.consentVersion || event.consentVersion < REQUIRED_CONSENT_VERSION) {
    return { accepted: false, reason: "Consent version too old or missing" };
  }
  if (event.latitude < -90 || event.latitude > 90) {
    return { accepted: false, reason: "Invalid latitude" };
  }
  if (event.longitude < -180 || event.longitude > 180) {
    return { accepted: false, reason: "Invalid longitude" };
  }
  if (!event.timestamp || isNaN(Date.parse(event.timestamp))) {
    return { accepted: false, reason: "Invalid timestamp" };
  }
  return { accepted: true };
}
