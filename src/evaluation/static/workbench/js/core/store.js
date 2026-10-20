/* Namespaced, failure-tolerant localStorage access (private mode / quota safe). */

export const KEYS = Object.freeze({
  theme: "securelora-theme",
  sidebar: "securelora-sidebar",
  selectedJob: "securelora-selected-job",
});

export function readPref(key) {
  try { return window.localStorage.getItem(key); } catch (_) { return null; }
}

export function writePref(key, value) {
  try {
    if (value === null || value === undefined) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
    return true;
  } catch (_) { return false; }
}
