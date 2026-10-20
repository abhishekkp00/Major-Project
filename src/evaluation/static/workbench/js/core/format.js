/* Pure formatting / status helpers (no DOM access; unit-tested under Node). */

export const NOT_RECORDED = "Not recorded";

export function isNil(v) { return v === null || v === undefined || v === ""; }

export function fmtNumber(v, digits = 3) {
  if (isNil(v) || typeof v === "boolean") return NOT_RECORDED;
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (Number.isInteger(n)) return n.toLocaleString("en-US");
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function fmtPercent(v, digits = 1) {
  if (isNil(v)) return NOT_RECORDED;
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtBytes(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return NOT_RECORDED;
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`;
}

export function fmtDate(v) {
  if (isNil(v)) return NOT_RECORDED;
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtDuration(startOrSeconds, end) {
  if (isNil(startOrSeconds)) return NOT_RECORDED;
  let sec = 0;
  if (typeof startOrSeconds === "number" && end === undefined) {
    sec = startOrSeconds;
  } else if (startOrSeconds && end) {
    const t0 = new Date(startOrSeconds).getTime();
    const t1 = new Date(end).getTime();
    if (!Number.isNaN(t0) && !Number.isNaN(t1)) {
      sec = Math.max(0, (t1 - t0) / 1000);
    } else {
      return NOT_RECORDED;
    }
  } else if (startOrSeconds) {
    const t0 = new Date(startOrSeconds).getTime();
    if (!Number.isNaN(t0)) {
      sec = Math.max(0, (Date.now() - t0) / 1000);
    } else {
      return NOT_RECORDED;
    }
  }
  if (!Number.isFinite(sec) || sec < 0) return NOT_RECORDED;
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const remS = s % 60;
  if (m < 60) return remS > 0 ? `${m}m ${remS}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  return `${h}h ${remM}m`;
}

export function fmtScalar(v) {
  if (isNil(v)) return NOT_RECORDED;
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v === "number") return fmtNumber(v);
  return String(v);
}

export function titleCase(key) {
  return String(key).replace(/[_\-.]+/g, " ").replace(/\s+/g, " ").trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Normalises backend status strings into a UI "kind". Success is granted ONLY
 * for explicit success words; PENDING / SKIPPED / unknown never render green.
 */
const SUCCESS = new Set(["PASSED", "PASS", "SUCCESS", "SUCCEEDED", "COMPLETED", "APPROVED", "SCREENED", "VERIFIED", "OK", "EXECUTED", "READY", "LOADED", "PACKAGED"]);
const DANGER = new Set(["FAILED", "FAIL", "REJECTED", "REJECT", "ERROR", "BLOCKED", "DENIED", "TAMPERED", "SECURITY_POLICY_REJECTED", "SECURITY_SCREENING_FAILED"]);
const WARNING = new Set(["REVIEW", "FLAG", "FLAGGED", "REQUIRES_ADMIN_APPROVAL", "WARNING", "SKIPPED", "PARTIAL", "NOT_EXECUTED", "UNAVAILABLE", "SIMULATED", "HISTORICAL", "CANCELLED", "CANCELED"]);
const RUNNING = new Set(["RUNNING", "IN_PROGRESS", "STARTED", "PROCESSING", "INGESTING", "TRAINING", "STARTING"]);

export function statusKind(status) {
  if (isNil(status)) return "neutral";
  const s = String(status).trim().toUpperCase().replace(/\s+/g, "_");
  if (SUCCESS.has(s)) return "success";
  if (DANGER.has(s)) return "danger";
  if (WARNING.has(s)) return "warning";
  if (RUNNING.has(s)) return "running";
  return "neutral";
}

export function statusLabel(status) {
  if (isNil(status)) return NOT_RECORDED;
  return titleCase(String(status).toLowerCase());
}

/** Validation helpers return an error string, or "" when valid. */
export function validateInteger(raw, { min, max, label }) {
  const text = String(raw ?? "").trim();
  if (text === "") return `${label} is required.`;
  if (!/^-?\d+$/.test(text)) return `${label} must be a whole number.`;
  const n = Number(text);
  if (min !== undefined && n < min) return `${label} must be at least ${min}.`;
  if (max !== undefined && n > max) return `${label} must be at most ${max}.`;
  return "";
}

export function validateFloat(raw, { min, max, label }) {
  const text = String(raw ?? "").trim();
  if (text === "") return `${label} is required.`;
  const n = Number(text);
  if (!Number.isFinite(n)) return `${label} must be a valid number.`;
  if (min !== undefined && n < min) return `${label} must be at least ${min}.`;
  if (max !== undefined && n > max) return `${label} must be at most ${max}.`;
  return "";
}

export const ALLOWED_DATASET_EXTENSIONS = [".txt", ".csv", ".json", ".jsonl", ".md"];

export function validateDatasetFile(file, maxBytes) {
  if (!file) return "Choose a dataset file.";
  const name = String(file.name || "");
  const dot = name.lastIndexOf(".");
  const ext = dot >= 0 ? name.slice(dot).toLowerCase() : "";
  if (!ALLOWED_DATASET_EXTENSIONS.includes(ext)) {
    return `Unsupported format "${ext || "none"}". Allowed: ${ALLOWED_DATASET_EXTENSIONS.join(", ")}.`;
  }
  if (file.size === 0) return "The selected file is empty.";
  if (maxBytes && file.size > maxBytes) return `File is ${fmtBytes(file.size)}; the limit is ${fmtBytes(maxBytes)}.`;
  return "";
}

export function validateJobVersion(raw) {
  const text = String(raw ?? "").trim();
  if (text === "") return "Version is required.";
  if (!/^[0-9A-Za-z][0-9A-Za-z._-]{0,31}$/.test(text)) return "Use letters, digits, dot, dash or underscore (max 32 characters).";
  return "";
}

/** Splits "#/path?x=1&y=2" into { path, query }. */
export function parseHash(hash) {
  let raw = String(hash || "").replace(/^#/, "");
  if (!raw.startsWith("/")) raw = `/${raw}`;
  const [pathPart, queryPart = ""] = raw.split("?");
  const path = pathPart.replace(/\/+$/, "") || "/";
  const query = {};
  for (const pair of queryPart.split("&")) {
    if (!pair) continue;
    const [k, v = ""] = pair.split("=");
    try { query[decodeURIComponent(k)] = decodeURIComponent(v); } catch (_) { /* ignore malformed */ }
  }
  return { path, query };
}

export function buildHash(path, query = {}) {
  const qs = Object.entries(query).filter(([, v]) => !isNil(v))
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
  return `#${path}${qs ? `?${qs}` : ""}`;
}

/** Flattens nested objects to [label, value] rows for generic key/value tables. */
export function flatten(obj, { prefix = "", maxDepth = 3, skip = [] } = {}) {
  const rows = [];
  const walk = (value, path, depth) => {
    if (value && typeof value === "object" && !Array.isArray(value) && depth < maxDepth) {
      for (const [k, v] of Object.entries(value)) {
        if (skip.includes(k)) continue;
        walk(v, path ? `${path}.${k}` : k, depth + 1);
      }
    } else {
      rows.push([path, value]);
    }
  };
  walk(obj, prefix, 0);
  return rows;
}
