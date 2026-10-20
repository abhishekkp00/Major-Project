/* API client.
   Authentication stays compatible with the existing dashboard: the Bearer token
   is read from the same storage keys, in the same precedence order, and is sent
   as `Authorization: Bearer <token>`. Nothing is hard-coded and the token value
   is never logged or rendered. */

const TOKEN_KEY = "SECURELORA_API_TOKEN";
const LEGACY_TOKEN_KEY = "api_bearer_token";

function readStore(store, key) {
  try { return store.getItem(key); } catch (_) { return null; }
}

export function getToken() {
  const found =
    readStore(window.sessionStorage, TOKEN_KEY) ||
    readStore(window.localStorage, TOKEN_KEY) ||
    readStore(window.sessionStorage, LEGACY_TOKEN_KEY) ||
    readStore(window.localStorage, LEGACY_TOKEN_KEY);
  return found ? found.trim() : "";
}

export function hasToken() { return getToken() !== ""; }

export function tokenStorageKind() {
  if (readStore(window.sessionStorage, TOKEN_KEY) || readStore(window.sessionStorage, LEGACY_TOKEN_KEY)) return "session";
  if (readStore(window.localStorage, TOKEN_KEY) || readStore(window.localStorage, LEGACY_TOKEN_KEY)) return "local";
  return "none";
}

export function clearToken() {
  for (const store of [window.sessionStorage, window.localStorage]) {
    for (const key of [TOKEN_KEY, LEGACY_TOKEN_KEY]) {
      try { store.removeItem(key); } catch (_) { /* ignore */ }
    }
  }
  notifyAuthChange();
}

/** persist=false keeps the token for this tab only (recommended). */
export function setToken(token, { persist = false } = {}) {
  clearToken();
  const store = persist ? window.localStorage : window.sessionStorage;
  store.setItem(TOKEN_KEY, String(token).trim());
  notifyAuthChange();
}

export function maskToken(token) {
  if (!token) return "";
  if (token.length <= 8) return "••••••••";
  return `${token.slice(0, 2)}${"•".repeat(10)}${token.slice(-2)}`;
}

const authListeners = new Set();
export function onAuthChange(fn) { authListeners.add(fn); return () => authListeners.delete(fn); }
function notifyAuthChange() { for (const fn of authListeners) fn(); }

export class ApiError extends Error {
  constructor(message, { status = 0, kind = "server", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
    this.payload = payload;
  }
}

function kindFor(status) {
  if (status === 401 || status === 403) return "auth";
  if (status === 404) return "notfound";
  if (status === 413 || status === 400 || status === 422) return "validation";
  if (status >= 500) return "server";
  return "client";
}

/**
 * @param {string} path  same-origin path beginning with /api/
 * @param {{method?:string,json?:any,form?:FormData,signal?:AbortSignal,timeoutMs?:number}} opts
 */
export async function api(path, { method = "GET", json, form, signal, timeoutMs = 30000 } = {}) {
  if (!path.startsWith("/")) throw new Error("api(): path must be same-origin and start with '/'");
  const headers = { Accept: "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  else if (form) body = form;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException("timeout", "TimeoutError")), timeoutMs);
  if (signal) {
    if (signal.aborted) ctrl.abort(signal.reason);
    else signal.addEventListener("abort", () => ctrl.abort(signal.reason), { once: true });
  }

  let res;
  try {
    res = await fetch(path, { method, headers, body, signal: ctrl.signal, credentials: "same-origin" });
  } catch (err) {
    clearTimeout(timer);
    if (signal?.aborted) throw err; // caller-cancelled navigation: let it propagate untouched
    if (err?.name === "TimeoutError" || err?.name === "AbortError") {
      throw new ApiError("The request timed out. The server may be busy.", { kind: "timeout" });
    }
    throw new ApiError("Cannot reach the SecureLoRA server. Check that it is running.", { kind: "network" });
  }
  clearTimeout(timer);

  let payload = null;
  const type = res.headers.get("content-type") || "";
  try {
    payload = type.includes("json") ? await res.json() : await res.text();
  } catch (_) { payload = null; }

  const failedFlag = payload && typeof payload === "object" && payload.success === false;
  if (!res.ok || failedFlag) {
    const detail = payload && typeof payload === "object" ? (payload.error || payload.reason || payload.message) : null;
    const kind = kindFor(res.status || 500);
    const fallback = kind === "auth"
      ? "Authentication is required. Add an API token in Settings."
      : `The server returned HTTP ${res.status}.`;
    throw new ApiError(detail || fallback, { status: res.status, kind, payload });
  }
  return payload;
}

/** Authenticated blob download (a plain <a href> cannot send the Bearer header). */
export async function downloadAuthenticated(path, filename) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let res;
  try { res = await fetch(path, { headers, credentials: "same-origin" }); }
  catch (_) { throw new ApiError("Cannot reach the SecureLoRA server.", { kind: "network" }); }
  if (!res.ok) {
    let detail = null;
    try { detail = (await res.json()).error; } catch (_) { /* not json */ }
    throw new ApiError(detail || `Download failed (HTTP ${res.status}).`, { status: res.status, kind: kindFor(res.status) });
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export function userMessage(err) {
  if (err instanceof ApiError) return err.message;
  return err?.message ? String(err.message) : "An unexpected error occurred.";
}

export const isAbort = (err) => err?.name === "AbortError";

/* ---- Thin endpoint wrappers (existing backend routes, unchanged) ---------- */
export const endpoints = {
  health: (o) => api("/api/health", o),
  jobs: (o) => api("/api/orchestrator/jobs", o),
  job: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}`, o),
  jobLogs: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/logs`, o),
  jobMetrics: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/metrics`, o),
  jobSummary: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/pipeline-summary`, o),
  jobScreening: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/screening`, o),
  jobPackaging: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/packaging`, o),
  jobArtifacts: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/artifacts`, o),
  jobReport: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/report`, o),
  createJob: (json, o) => api("/api/orchestrator/jobs", { method: "POST", json, ...o }),
  uploadToJob: (id, form, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/upload`, { method: "POST", form, timeoutMs: 120000, ...o }),
  startJob: (id, o) => api(`/api/orchestrator/jobs/${encodeURIComponent(id)}/start`, { method: "POST", ...o }),
  validateDataset: (form, o) => api("/api/orchestrator/validate", { method: "POST", form, timeoutMs: 120000, ...o }),
  datasetTemplates: (o) => api("/api/orchestrator/dataset-templates", o),
  datasetDetails: (id, o) => api(`/api/orchestrator/datasets/${encodeURIComponent(id)}`, o),
  transparencyInspect: (json, o) => api("/api/transparency/inspect", { method: "POST", json, ...o }),
  modelStatus: (o) => api("/api/orchestrator/model-status", { timeoutMs: 120000, ...o }),
  phase4Status: (o) => api("/api/phase4/status", o),
  phase4Verify: (json, o) => api("/api/phase4/verify", { method: "POST", json, timeoutMs: 300000, ...o }),
  phase4Generate: (json, o) => api("/api/phase4/generate", { method: "POST", json, timeoutMs: 120000, ...o }),
  trainingConfig: (o) => api("/api/orchestrator/training-config", o),
  research: (name, o) => api(`/api/research/${name}`, o),
};
