/* Theme management.
   - The very first paint is handled by the inline script in workbench.html
     (same storage key + same resolution rule) so there is no flash.
   - This module owns runtime toggling, persistence and system-change tracking. */
import { KEYS, readPref, writePref } from "./store.js";

export const THEMES = ["light", "dark"];

/** stored: "light" | "dark" | null (follow system). */
export function resolveTheme(stored, systemPrefersDark) {
  if (stored === "light" || stored === "dark") return stored;
  return systemPrefersDark ? "dark" : "light";
}

const mq = typeof window !== "undefined" && window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
const listeners = new Set();

export function getStoredTheme() {
  const v = readPref(KEYS.theme);
  return v === "light" || v === "dark" ? v : null;
}

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || resolveTheme(getStoredTheme(), !!mq?.matches);
}

export const getTheme = currentTheme;
export const isSystemDark = () => !!mq?.matches;

function apply(theme) {
  const root = document.documentElement;
  root.setAttribute("data-theme", theme);
  root.style.colorScheme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", theme === "dark" ? "#060e1d" : "#0a1830");
  for (const fn of listeners) fn(theme);
}

/** theme: "light" | "dark" | null (revert to following the system). */
export function setTheme(theme) {
  if (theme === "light" || theme === "dark") writePref(KEYS.theme, theme);
  else writePref(KEYS.theme, null);
  apply(resolveTheme(getStoredTheme(), !!mq?.matches));
}

export function toggleTheme() {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
}

export function onThemeChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function initTheme() {
  apply(resolveTheme(getStoredTheme(), !!mq?.matches));
  // Follow OS changes only while the user has not chosen explicitly.
  const onSystem = () => { if (getStoredTheme() === null) apply(resolveTheme(null, !!mq.matches)); };
  if (mq?.addEventListener) mq.addEventListener("change", onSystem);
  else if (mq?.addListener) mq.addListener(onSystem);
  // Keep multiple tabs consistent.
  window.addEventListener("storage", (e) => {
    if (e.key === KEYS.theme) apply(resolveTheme(getStoredTheme(), !!mq?.matches));
  });
}
