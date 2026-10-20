/* Dependency-free SVG charts. Every colour comes from a CSS token via a class,
   so switching theme restyles charts (and their tooltips) instantly without
   re-rendering. Marks are keyboard focusable and expose accessible names. */
import { h } from "./dom.js";
import { fmtNumber } from "./format.js";

let tipEl = null;
function tip() {
  if (!tipEl) {
    tipEl = h("div", { class: "chart-tip", role: "tooltip", id: "chart-tip" });
    document.body.append(tipEl);
  }
  return tipEl;
}
function showTip(text, x, y) {
  const t = tip();
  t.textContent = text;
  t.dataset.show = "true";
  const pad = 12;
  const w = t.offsetWidth;
  t.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, x + pad))}px`;
  t.style.top = `${Math.max(8, y - t.offsetHeight - pad)}px`;
}
function hideTip() { if (tipEl) tipEl.dataset.show = "false"; }

function bind(node, text) {
  node.addEventListener("pointermove", (e) => showTip(text, e.clientX, e.clientY));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    showTip(text, r.left + r.width / 2, r.top);
  });
  node.addEventListener("blur", hideTip);
}

const W = 640;
const H = 260;
const M = { top: 12, right: 12, bottom: 38, left: 48 };

function niceMax(v) {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * p;
}

function axes(svg, { yMax, yFormat, ticks = 4 }) {
  const innerH = H - M.top - M.bottom;
  for (let i = 0; i <= ticks; i += 1) {
    const v = (yMax / ticks) * i;
    const y = M.top + innerH - (v / yMax) * innerH;
    svg.append(
      h("line", { class: "grid-line", x1: M.left, x2: W - M.right, y1: y, y2: y }),
      h("text", { class: "axis-text", x: M.left - 8, y: y + 4, "text-anchor": "end" }, yFormat(v)),
    );
  }
}

/** series: [{ name, values: number|null[] }]; labels: string[]. */
export function barChart({ title, labels, series, yMax, yFormat = (v) => fmtNumber(v, 2), valueFormat }) {
  const fmt = valueFormat || yFormat;
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  const top = yMax ?? niceMax(Math.max(0, ...all));
  const svg = h("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  svg.append(h("title", null, title));
  axes(svg, { yMax: top, yFormat });
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;
  const groupW = innerW / Math.max(1, labels.length);
  const barW = Math.min(46, (groupW * 0.7) / series.length);
  labels.forEach((label, i) => {
    const gx = M.left + groupW * i + groupW / 2;
    svg.append(h("text", { class: "axis-text", x: gx, y: H - M.bottom + 18, "text-anchor": "middle" }, label));
    series.forEach((s, si) => {
      const v = s.values[i];
      if (!Number.isFinite(v)) return;
      const bh = Math.max(1, (v / top) * innerH);
      const x = gx - (barW * series.length) / 2 + barW * si;
      const text = `${s.name} · ${label}: ${fmt(v)}`;
      const rect = h("rect", {
        class: `bar s${(si % 4) + 1}`, x, y: M.top + innerH - bh, width: barW - 3, height: bh, rx: 3,
        tabindex: "0", role: "img", "aria-label": text,
      });
      bind(rect, text);
      svg.append(rect);
    });
  });
  return wrapChart(svg, series);
}

export function lineChart({ title, labels, series, yMax, yMin = 0, yFormat = (v) => fmtNumber(v, 2) }) {
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  const top = yMax ?? niceMax(Math.max(0, ...all));
  const svg = h("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  svg.append(h("title", null, title));
  axes(svg, { yMax: top, yFormat });
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;
  const n = labels.length;
  const xAt = (i) => M.left + (n <= 1 ? innerW / 2 : (innerW * i) / (n - 1));
  const yAt = (v) => M.top + innerH - ((v - yMin) / (top - yMin)) * innerH;
  const step = Math.max(1, Math.ceil(n / 8));
  labels.forEach((label, i) => {
    if (i % step === 0 || i === n - 1) svg.append(h("text", { class: "axis-text", x: xAt(i), y: H - M.bottom + 18, "text-anchor": "middle" }, label));
  });
  series.forEach((s, si) => {
    const cls = `s${(si % 4) + 1}`;
    const pts = s.values.map((v, i) => (Number.isFinite(v) ? [xAt(i), yAt(v), v, i] : null)).filter(Boolean);
    if (pts.length > 1) svg.append(h("polyline", { class: `line ${cls}`, points: pts.map((p) => `${p[0]},${p[1]}`).join(" ") }));
    pts.forEach(([x, y, v, i]) => {
      const text = `${s.name} · ${labels[i]}: ${yFormat(v)}`;
      const c = h("circle", { class: `pt ${cls}`, cx: x, cy: y, r: 4, tabindex: "0", role: "img", "aria-label": text });
      bind(c, text);
      svg.append(c);
    });
  });
  return wrapChart(svg, series);
}

function wrapChart(svg, series) {
  const legend = series.length > 1 || series[0]?.name
    ? h("div", { class: "chart-legend" }, series.map((s, i) => h("span", null, h("i", { class: "legend-swatch", style: `background:var(--chart-${(i % 4) + 1})` }), s.name)))
    : null;
  return h("figure", { class: "chart", style: "margin:0" }, svg, legend);
}
