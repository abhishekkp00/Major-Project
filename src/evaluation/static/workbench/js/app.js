/* Application bootstrap: theme, sidebar, navigation, breadcrumbs, connection
   status and global keyboard handling. Pages are loaded by the router. */
import { h, icon, replace } from "./core/dom.js";
import { initTheme, toggleTheme, currentTheme, onThemeChange } from "./core/theme.js";
import { KEYS, readPref, writePref } from "./core/store.js";
import { ROUTES, DEFAULT_PATH } from "./core/routes.js";
import { startRouter, onRoute } from "./core/router.js";
import { endpoints, hasToken, onAuthChange } from "./core/api.js";
import { buildHash } from "./core/format.js";

const root = document.documentElement;
const mqMobile = window.matchMedia("(max-width: 768px)");
const mqTablet = window.matchMedia("(max-width: 1024px)");

/* ---- Theme toggle -------------------------------------------------------- */
function syncThemeButton() {
  const btn = document.getElementById("theme-toggle");
  const dark = currentTheme() === "dark";
  btn.setAttribute("aria-pressed", String(dark));
}
function initThemeToggle() {
  initTheme();
  const btn = document.getElementById("theme-toggle");
  btn.addEventListener("click", toggleTheme);
  onThemeChange(syncThemeButton);
  syncThemeButton();
}

/* ---- Sidebar ------------------------------------------------------------- */
function setSidebar(state, { persist = true } = {}) {
  root.setAttribute("data-sidebar", state);
  if (persist) writePref(KEYS.sidebar, state);
  const collapsed = state === "collapsed";
  const btn = document.getElementById("sidebar-toggle");
  btn.setAttribute("aria-expanded", String(!collapsed));
  btn.setAttribute("data-tip", collapsed ? "Expand sidebar (Ctrl+B)" : "Collapse sidebar (Ctrl+B)");
  btn.querySelector("#sidebar-toggle-label").textContent = collapsed ? "Expand" : "Collapse";
}

function openDrawer(open) {
  root.setAttribute("data-nav-open", String(open));
  const menu = document.getElementById("menu-btn");
  menu.setAttribute("aria-expanded", String(open));
  menu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  const sidebar = document.getElementById("sidebar");
  if (mqMobile.matches) {
    // Hide the off-screen drawer from assistive tech and the tab order when closed.
    if (open) { sidebar.removeAttribute("inert"); sidebar.removeAttribute("aria-hidden"); }
    else { sidebar.setAttribute("inert", ""); sidebar.setAttribute("aria-hidden", "true"); }
  } else {
    sidebar.removeAttribute("inert"); sidebar.removeAttribute("aria-hidden");
  }
  if (open) sidebar.querySelector(".nav-link[aria-current='page'], .nav-link")?.focus();
}

function initSidebar() {
  setSidebar(root.getAttribute("data-sidebar") === "collapsed" ? "collapsed" : "expanded", { persist: false });
  document.getElementById("sidebar-toggle").addEventListener("click", () => {
    setSidebar(root.getAttribute("data-sidebar") === "collapsed" ? "expanded" : "collapsed");
  });
  const menu = document.getElementById("menu-btn");
  menu.addEventListener("click", () => openDrawer(root.getAttribute("data-nav-open") !== "true"));
  document.getElementById("scrim").addEventListener("click", () => { openDrawer(false); menu.focus(); });

  // Responsive defaults only apply while the user has no stored preference.
  const applyViewport = () => {
    openDrawer(false);
    if (!readPref(KEYS.sidebar)) setSidebar(mqTablet.matches ? "collapsed" : "expanded", { persist: false });
  };
  mqMobile.addEventListener("change", applyViewport);
  mqTablet.addEventListener("change", applyViewport);
  openDrawer(false);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && root.getAttribute("data-nav-open") === "true") {
      openDrawer(false); menu.focus();
    }
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b") {
      if (mqMobile.matches) { e.preventDefault(); openDrawer(root.getAttribute("data-nav-open") !== "true"); return; }
      e.preventDefault();
      setSidebar(root.getAttribute("data-sidebar") === "collapsed" ? "expanded" : "collapsed");
    }
  });

  // Keep keyboard focus inside the open drawer.
  document.getElementById("sidebar").addEventListener("keydown", (e) => {
    if (e.key !== "Tab" || root.getAttribute("data-nav-open") !== "true") return;
    const focusables = [...document.querySelectorAll("#sidebar a[href], #sidebar button:not([disabled])")];
    if (!focusables.length) return;
    const first = focusables[0]; const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
}

/* ---- Navigation ----------------------------------------------------------- */
function buildNav() {
  const nav = document.getElementById("nav");
  const groups = [];
  for (const r of ROUTES) {
    let g = groups.find((x) => x.name === r.group);
    if (!g) { g = { name: r.group, items: [] }; groups.push(g); }
    g.items.push(r);
  }
  const frag = [];
  for (const g of groups) {
    frag.push(h("div", { class: "nav-group-label", "aria-hidden": "true" }, g.name));
    for (const r of g.items) {
      frag.push(h("a", { class: "nav-link", href: buildHash(r.path), dataset: { route: r.id }, "data-tip": r.label },
        icon(r.icon), h("span", { class: "nav-label" }, r.label)));
    }
  }
  replace(nav, ...frag);

  // Arrow-key / Home / End navigation between items (in addition to Tab).
  nav.addEventListener("keydown", (e) => {
    const links = [...nav.querySelectorAll(".nav-link")];
    const i = links.indexOf(document.activeElement);
    if (i < 0) return;
    let next = null;
    if (e.key === "ArrowDown") next = links[(i + 1) % links.length];
    else if (e.key === "ArrowUp") next = links[(i - 1 + links.length) % links.length];
    else if (e.key === "Home") next = links[0];
    else if (e.key === "End") next = links[links.length - 1];
    if (next) { e.preventDefault(); next.focus(); }
  });
  nav.addEventListener("click", (e) => {
    if (e.target.closest(".nav-link") && mqMobile.matches) openDrawer(false);
  });
}

function onRouteChange(route) {
  for (const a of document.querySelectorAll("#nav .nav-link")) {
    if (route && a.dataset.route === route.id) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  const crumbs = document.getElementById("crumbs");
  replace(crumbs, h("ol", null,
    h("li", null, h("a", { href: buildHash(DEFAULT_PATH) }, "Workbench")),
    route ? h("li", null, route.group) : null,
    h("li", null, h("span", { "aria-current": "page" }, route ? route.title : "Not found"))));
  if (mqMobile.matches) openDrawer(false);
}

/* ---- API connection indicator ---------------------------------------------- */
async function refreshConnection() {
  const pill = document.getElementById("conn-pill");
  const text = document.getElementById("conn-text");
  const set = (state, label, tip) => {
    pill.dataset.state = state; text.textContent = label; pill.setAttribute("data-tip", tip);
  };
  try {
    const health = await endpoints.health({ timeoutMs: 8000 });
    const demo = health.auth_mode === "local-demo";
    if (demo) set("ok", "Local demo", "Tokenless access is limited to loopback clients.");
    else if (hasToken()) set("ok", "Token set", "Bearer-token authentication is enabled; a token is configured in this browser.");
    else set("locked", "Token required", "Bearer-token authentication is enabled. Add a token in Settings.");
  } catch (_) {
    set("offline", "API offline", "The SecureLoRA server could not be reached.");
  }
}

/* ---- Boot -------------------------------------------------------------------- */
function boot() {
  initThemeToggle();
  buildNav();
  initSidebar();
  onRoute(onRouteChange);
  onAuthChange(refreshConnection);
  refreshConnection();
  startRouter();
  // Re-enable transitions after the first paint so reloads never animate.
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("no-transition")));
}

window.addEventListener("error", (e) => console.error("Unhandled error:", e.message));
window.addEventListener("unhandledrejection", (e) => console.error("Unhandled rejection:", e.reason));

boot();
