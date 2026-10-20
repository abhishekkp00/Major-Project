/* Hash router with lazy page modules, focus management and race protection.
   Page contract:  export default { async mount(root, ctx) -> cleanup?: () => void }
   ctx = { query, route, navigate(path, query), signal } */
import { h, replace } from "./dom.js";
import { parseHash, buildHash } from "./format.js";
import { ROUTES, DEFAULT_PATH, routeForPath } from "./routes.js";
import { skeletonLines, skeletonKpis, errorState, emptyState } from "./ui.js";

let current = { cleanup: null, controller: null };
let navSeq = 0;
const listeners = new Set();

export function onRoute(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export function navigate(path, query) {
  const target = buildHash(path, query);
  if (location.hash === target) render();
  else location.hash = target;
}

function pageHead(route) {
  return h("div", null);
}

function loadingView() {
  return h("div", { class: "stack" }, skeletonKpis(3), skeletonLines(6));
}

async function render() {
  const { path, query } = parseHash(location.hash);
  const mine = ++navSeq;
  const root = document.getElementById("view");
  const route = routeForPath(path);

  // Tear down the previous page (timers, listeners, in-flight requests).
  try { current.cleanup?.(); } catch (e) { console.error("page cleanup failed", e); }
  current.controller?.abort();
  current = { cleanup: null, controller: new AbortController() };
  const { signal } = current.controller;

  if (!location.hash || location.hash === "#" || location.hash === "#/") {
    location.replace(buildHash(DEFAULT_PATH));
    return;
  }

  for (const fn of listeners) fn(route, query);

  if (!route) {
    document.title = "Page not found · SecureLoRA Workbench";
    replace(root, h("div", { class: "view" },
      h("div", { class: "page-head" }, h("div", null, h("h1", { tabindex: "-1", id: "page-title" }, "Page not found"))),
      emptyState({
        iconName: "alert", title: "This page does not exist",
        message: `There is no workbench page at "${path}".`,
        action: h("a", { class: "btn btn-primary", href: buildHash(DEFAULT_PATH) }, "Go to Overview"),
      })));
    focusTitle();
    return;
  }

  document.title = `${route.title} · SecureLoRA Workbench`;
  replace(root, h("div", { class: "view", "aria-busy": "true" }, loadingView()));

  let mod;
  try { mod = await route.load(); }
  catch (err) {
    if (mine !== navSeq) return;
    console.error("Failed to load page module", route.id, err);
    replace(root, h("div", { class: "view" }, errorState(new Error("This page could not be loaded. Check your connection and retry."), () => render())));
    return;
  }
  if (mine !== navSeq) return;

  const view = h("div", { class: "view" });
  replace(root, view);
  try {
    const cleanup = await mod.default.mount(view, {
      query, route, signal, navigate,
    });
    if (mine !== navSeq) { try { cleanup?.(); } catch (_) { /* ignore */ } return; }
    current.cleanup = cleanup || null;
  } catch (err) {
    if (mine !== navSeq || err?.name === "AbortError") return;
    console.error("Page failed to render", route.id, err);
    replace(view, errorState(err, () => render()));
  }
  focusTitle();
}

function focusTitle() {
  const title = document.getElementById("page-title");
  if (title) title.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "auto" });
}

export function startRouter() {
  window.addEventListener("hashchange", render);
  render();
}

export { ROUTES };
