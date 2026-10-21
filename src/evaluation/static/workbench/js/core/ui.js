/* Shared UI building blocks: toasts, dialogs, skeletons, empty/error states,
   badges, tables, form fields and the async-section loader. */
import { h, icon, clear, replace, nextId } from "./dom.js";
import { ApiError, userMessage, isAbort, endpoints } from "./api.js";
import { statusKind, statusLabel, fmtScalar, NOT_RECORDED } from "./format.js";
import { KEYS, readPref, writePref } from "./store.js";

/* ---- Toasts ------------------------------------------------------------- */
const TOAST_ICON = { success: "check", warning: "alert", danger: "error", info: "info" };

export function toast({ kind = "info", title, message = "", timeout, action } = {}) {
  const host = document.getElementById("toasts");
  if (!host) return null;
  while (host.children.length >= 4) host.firstElementChild.remove();
  const closeBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Dismiss notification", onclick: () => dismiss() }, icon("x", { cls: "" }));
  const el = h("div", { class: "toast", dataset: { kind }, role: kind === "danger" ? "alert" : "status" },
    icon(TOAST_ICON[kind] || "info", { cls: "" }),
    h("div", null,
      h("div", { class: "toast-title" }, title),
      message ? h("div", { class: "toast-msg" }, message) : null,
      action ? h("a", { href: action.href, class: "small" }, action.label) : null),
    closeBtn);
  let timer;
  function dismiss() { clearTimeout(timer); el.remove(); }
  host.append(el);
  const ms = timeout ?? (kind === "danger" ? 9000 : 5000);
  if (ms > 0) timer = setTimeout(dismiss, ms);
  return dismiss;
}

/** One consistent shape for every failure surfaced to the user. */
export function notifyError(err, context = "Request failed") {
  if (isAbort(err)) return;
  const kind = err instanceof ApiError ? err.kind : "client";
  const action = kind === "auth" ? { label: "Open Settings", href: "#/settings" } : undefined;
  toast({ kind: "danger", title: context, message: userMessage(err), action });
}

export const notifySuccess = (title, message) => toast({ kind: "success", title, message });

/** Contextual page title (the single <h1>; focused after each navigation). */
export function pageHead({ title, description, actions }) {
  return h("div", { class: "page-head" },
    h("div", null, h("h1", { id: "page-title", tabindex: "-1" }, title), description ? h("p", null, description) : null),
    actions ? h("div", { class: "page-actions" }, actions) : null);
}

/* ---- Small presentational pieces --------------------------------------- */
export function badge(status, labelOverride) {
  const kind = statusKind(status);
  return h("span", { class: "badge", dataset: { kind } }, labelOverride ?? statusLabel(status));
}

export function alertBox(kind, title, message, extra) {
  return h("div", { class: "alert", dataset: { kind }, role: kind === "danger" ? "alert" : null },
    icon(TOAST_ICON[kind] || "info", { cls: "alert-icon" }),
    h("div", null, title ? h("strong", null, title) : null, message ? h("span", null, message) : null, extra || null));
}

export function emptyState({ iconName = "inbox", title, message, action } = {}) {
  return h("div", { class: "empty" },
    icon(iconName, { cls: "empty-icon" }),
    h("h3", null, title),
    message ? h("p", null, message) : null,
    action || null);
}

export function errorState(err, onRetry) {
  const kind = err instanceof ApiError ? err.kind : "client";
  const action = kind === "auth"
    ? h("a", { class: "btn btn-primary", href: "#/settings" }, "Open Settings")
    : onRetry ? h("button", { class: "btn", type: "button", onclick: onRetry }, icon("refresh", { cls: "" }), "Retry") : null;
  return h("div", { class: "empty", role: "alert" },
    icon("error", { cls: "empty-icon" }),
    h("h3", null, kind === "auth" ? "Authentication required" : "Unable to load this section"),
    h("p", null, userMessage(err)),
    action);
}

export function skeleton(kind = "block") {
  return h("div", { class: `skeleton skeleton-${kind}`, "aria-hidden": "true" });
}

export function skeletonLines(n = 4) {
  return h("div", { role: "status", "aria-label": "Loading" },
    h("span", { class: "sr-only" }, "Loading…"),
    Array.from({ length: n }, (_, i) => h("div", { class: "skeleton skeleton-line", style: `width:${100 - ((i * 13) % 40)}%`, "aria-hidden": "true" })));
}

export function skeletonKpis(n = 4) {
  return h("div", { class: "kpis", role: "status", "aria-label": "Loading metrics" },
    h("span", { class: "sr-only" }, "Loading metrics…"),
    Array.from({ length: n }, () => skeleton("kpi")));
}

export function skeletonCards(n = 3) {
  return h("div", { class: "cards-grid", role: "status", "aria-label": "Loading content" },
    h("span", { class: "sr-only" }, "Loading content…"),
    Array.from({ length: n }, () => skeleton("card")));
}

export function kpi({ label, value, foot, text = false }) {
  return h("div", { class: "card kpi" },
    h("div", { class: "kpi-label" }, label),
    h("div", { class: `kpi-value${text ? " is-text" : ""}` }, value),
    foot ? h("div", { class: "kpi-foot" }, foot) : null);
}

export function card({ title, sub, actions, body, flush = false, id } = {}) {
  return h("section", { class: "card", id: id || null, "aria-labelledby": title ? `${id || "c"}-h` : null },
    title ? h("div", { class: "card-head" },
      h("div", null, h("h2", { id: `${id || "c"}-h` }, title), sub ? h("div", { class: "sub" }, sub) : null),
      actions ? h("div", { class: "row" }, actions) : null) : null,
    h("div", { class: `card-body${flush ? " flush" : ""}` }, body));
}

export function kvList(rows) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of rows) {
    dl.append(h("dt", null, k), h("dd", null, v instanceof Node ? v : fmtScalar(v)));
  }
  return dl;
}

export function button({ label, iconName, kind = "", type = "button", onClick, small = false, id, disabled = false, title } = {}) {
  return h("button", {
    class: `btn${kind ? ` btn-${kind}` : ""}${small ? " btn-sm" : ""}`, type, id, onclick: onClick, disabled, title,
  }, iconName ? icon(iconName, { cls: "" }) : null, label);
}

/** Disables a button and shows a spinner while `fn` runs. */
export async function withBusy(btn, fn) {
  if (!btn || btn.disabled) return undefined;
  const original = Array.from(btn.childNodes);
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  replace(btn, h("span", { class: "spinner", "aria-hidden": "true" }), h("span", null, "Working…"));
  try { return await fn(); }
  finally {
    clear(btn); btn.append(...original);
    btn.disabled = false; btn.removeAttribute("aria-busy");
  }
}

/* ---- Forms -------------------------------------------------------------- */
export function field({ label, hint, control, input, id }) {
  const ctrl = control || input;
  const fid = id || ctrl?.id || nextId("f");
  if (ctrl) {
    ctrl.id = fid;
    const errId = `${fid}-err`;
    const hintId = `${fid}-hint`;
    const described = [hint ? hintId : null, errId].filter(Boolean).join(" ");
    if (typeof ctrl.setAttribute === "function") {
      ctrl.setAttribute("aria-describedby", described);
    }
  }
  const errId = `${fid}-err`;
  const hintId = `${fid}-hint`;
  return h("div", { class: "field" },
    h("label", { for: fid }, label),
    ctrl,
    hint ? h("div", { class: "hint", id: hintId }, hint) : null,
    h("div", { class: "error-text", id: errId, role: "alert" }));
}

/** Show/clear an inline validation message for a control built via field(). */
export function setFieldError(control, message) {
  const wrap = control.closest(".field");
  if (!wrap) return;
  const err = wrap.querySelector(".error-text");
  if (message) {
    wrap.dataset.invalid = "true"; delete wrap.dataset.valid;
    control.setAttribute("aria-invalid", "true");
    err.textContent = message;
  } else {
    delete wrap.dataset.invalid;
    control.removeAttribute("aria-invalid");
    err.textContent = "";
  }
}

export function markValid(control) {
  const wrap = control.closest(".field");
  if (wrap && !wrap.dataset.invalid) wrap.dataset.valid = "true";
}

/* ---- Dialogs ------------------------------------------------------------ */
export function openDialog({ title, content, actions = [], width }) {
  return new Promise((resolve) => {
    const titleId = nextId("dlg");
    const dlg = h("dialog", { class: "modal", "aria-labelledby": titleId, style: width ? `width:min(${width}px, calc(100vw - 32px))` : null },
      h("div", { class: "modal-head" },
        h("h2", { id: titleId }, title),
        h("button", { class: "icon-btn", type: "button", "aria-label": "Close dialog", onclick: () => dlg.close("dismiss") }, icon("x", { cls: "" }))),
      h("div", { class: "modal-body" }, content),
      actions.length ? h("div", { class: "modal-foot" },
        actions.map((a) => h("button", { class: `btn${a.kind ? ` btn-${a.kind}` : ""}`, type: "button", onclick: () => dlg.close(a.value) }, a.label))) : null);
    dlg.addEventListener("close", () => { const v = dlg.returnValue; dlg.remove(); resolve(v || "dismiss"); });
    document.body.append(dlg);
    dlg.showModal();
    (dlg.querySelector("[autofocus]") || dlg.querySelector(".modal-foot .btn:last-child") || dlg.querySelector(".icon-btn"))?.focus();
  });
}

export async function confirmDialog({ title, message, confirmLabel = "Confirm", danger = false, details }) {
  const result = await openDialog({
    title,
    content: h("div", { class: "stack-sm" }, h("p", null, message), details || null),
    actions: [{ label: "Cancel", value: "cancel" }, { label: confirmLabel, value: "ok", kind: danger ? "danger" : "primary" }],
  });
  return result === "ok";
}

/* ---- Tables ------------------------------------------------------------- */
/**
 * columns: [{ key, label, num?, sortable?, render?(row)->Node|string, sortValue?(row) }]
 */
export function dataTable({ columns, rows, caption, onRowClick, initialSort }) {
  let sort = initialSort ? { ...initialSort } : null;
  const wrap = h("div", { class: "table-wrap" });
  const draw = () => {
    const sorted = rows.slice();
    if (sort) {
      const col = columns.find((c) => c.key === sort.key);
      const val = (r) => (col?.sortValue ? col.sortValue(r) : r[sort.key]);
      sorted.sort((a, b) => {
        const av = val(a); const bv = val(b);
        if (av === bv) return 0;
        if (av === undefined || av === null) return 1;
        if (bv === undefined || bv === null) return -1;
        return (av > bv ? 1 : -1) * (sort.dir === "asc" ? 1 : -1);
      });
    }
    const table = h("table", { class: "table" },
      caption ? h("caption", { class: "sr-only" }, caption) : null,
      h("thead", null, h("tr", null, columns.map((c) => h("th", {
        scope: "col", class: c.num ? "num" : null,
        "aria-sort": sort && sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : (c.sortable ? "none" : null),
      }, c.sortable
        ? h("button", {
          class: "th-sort", type: "button", dataset: sort && sort.key === c.key ? { dir: sort.dir } : {},
          onclick: () => { sort = { key: c.key, dir: sort && sort.key === c.key && sort.dir === "asc" ? "desc" : "asc" }; draw(); },
        }, c.label)
        : c.label)))),
      h("tbody", null, sorted.map((row) => {
        const tr = h("tr", {
          dataset: onRowClick ? { clickable: "true" } : {}, tabindex: onRowClick ? "0" : null,
          onclick: onRowClick ? () => onRowClick(row) : null,
          onkeydown: onRowClick ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRowClick(row); } } : null,
        }, columns.map((c) => {
          const content = c.render ? c.render(row) : fmtScalar(row[c.key]);
          return h("td", { class: c.num ? "num" : null }, content);
        }));
        return tr;
      })));
    replace(wrap, table);
  };
  draw();
  return wrap;
}

/* ---- Async section loader ---------------------------------------------- */
/**
 * Renders a skeleton, runs `load`, then `render(data)`; failures become a
 * consistent error state with a Retry action. Returns { reload }.
 */
export function asyncSection(container, { load, render, skeleton: skel = () => skeletonLines(5), signal }) {
  let seq = 0;
  async function run() {
    const mine = ++seq;
    replace(container, skel());
    container.setAttribute("aria-busy", "true");
    try {
      const data = await load();
      if (mine !== seq || signal?.aborted) return;
      replace(container, render(data));
    } catch (err) {
      if (mine !== seq || isAbort(err) || signal?.aborted) return;
      replace(container, errorState(err, run));
    } finally {
      if (mine === seq) container.removeAttribute("aria-busy");
    }
  }
  run();
  return { reload: run };
}

/* ---- Shared job selection ---------------------------------------------- */
export function getSelectedJobId() { return readPref(KEYS.selectedJob) || ""; }
export function setSelectedJobId(id) { writePref(KEYS.selectedJob, id || null); }

/**
 * Builds a labelled <select> populated from the real jobs list. Calls
 * onChange(jobId|"") after load and on every user change.
 */
export function jobPicker({ onChange, preferred, signal }) {
  const select = h("select", { class: "select", id: nextId("job") });
  const wrap = field({ label: "Run", control: select, hint: "Choose a pipeline run recorded by the backend." });
  const status = h("div", { class: "hint" });
  let jobs = [];

  async function load() {
    select.disabled = true;
    replace(select, h("option", { value: "" }, "Loading runs…"));
    try {
      const data = await endpoints.jobs({ signal });
      jobs = (data.jobs || []).slice().sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
      replace(select, h("option", { value: "" }, jobs.length ? "Select a run…" : "No runs available"),
        jobs.map((j) => h("option", { value: j.job_id }, `${j.job_id} · ${statusLabel(j.status)}${j.dataset_name ? ` · ${j.dataset_name}` : ""}`)));
      const want = preferred || getSelectedJobId();
      if (want && jobs.some((j) => j.job_id === want)) select.value = want;
      else if (jobs.length) select.value = jobs[0].job_id;
      select.disabled = jobs.length === 0;
      setSelectedJobId(select.value);
      onChange(select.value, jobs);
    } catch (err) {
      if (isAbort(err)) return;
      replace(select, h("option", { value: "" }, "Unable to load runs"));
      select.disabled = true;
      status.textContent = userMessage(err);
      notifyError(err, "Could not load runs");
      onChange("", []);
    }
  }
  select.addEventListener("change", () => { setSelectedJobId(select.value); onChange(select.value, jobs); });
  load();
  return { el: h("div", null, wrap, status), select, reload: load, jobs: () => jobs };
}

export { NOT_RECORDED };
