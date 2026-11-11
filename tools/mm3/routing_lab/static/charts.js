// Small SVG charts for the routing lab. One axis per chart, thin marks, a hover tooltip on every mark,
// a legend whenever there are two or more series, text in ink colours and never in the series colour.
"use strict";

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const tipEl = () => document.getElementById("tip");

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function tip(text, ev) {
    const t = tipEl();
    if (text == null) { t.hidden = true; return; }
    t.textContent = text; t.hidden = false;
    const w = t.offsetWidth, h = t.offsetHeight;
    t.style.left = Math.max(4, Math.min(window.innerWidth - w - 4, ev.clientX + 12)) + "px";
    t.style.top = Math.max(4, Math.min(window.innerHeight - h - 4, ev.clientY + 14)) + "px";
  }
  function hover(node, text) {
    node.addEventListener("mousemove", ev => tip(typeof text === "function" ? text(ev) : text, ev));
    node.addEventListener("mouseleave", () => tip(null));
  }
  function ticks(lo, hi, n = 4) {
    if (!(hi > lo)) hi = lo + 1;
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
    return out;
  }
  const fmt = (v, d = 3) => {
    if (v == null || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && a < 0.001) return v.toExponential(1);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(Math.min(d, 1)) : v.toFixed(d).replace(/\.?0+$/, m => m.includes(".") ? "" : m);
  };
  const pct = v => (v == null ? "–" : (100 * v).toFixed(1) + "%");
  function frame(host, title, note) {
    host.replaceChildren();
    host.classList.add("chart");
    for (const [cls, text] of [["title", title], ["note", note]]) {
      if (!text) continue;
      const d = document.createElement("div");
      d.className = cls; d.textContent = text;      // text only: titles may carry model output
      host.appendChild(d);
    }
    return host;
  }
  const seriesColor = i => css(`--series-${(i % 8) + 1}`);
  // The sequential ramp: t in [0,1] to one of eight steps of a single hue; step 0 is the near-surface neutral.
  function seq(t) {
    const k = Math.max(0, Math.min(7, Math.round(t * 7)));
    return { bg: css(`--seq-${k}`), step: k };
  }

  // Horizontal bars: items = [{label, value, note, mark}]. One hue; `mark` adds a text tag, never a second colour.
  function barsH(host, items, o = {}) {
    frame(host, o.title, o.note);
    const rowH = o.rowH || 18, gap = 2, labelW = o.labelW || 110, valueW = 62, W = o.width || 360;
    const H = items.length * (rowH + gap) + 4, max = o.max || Math.max(1e-12, ...items.map(d => d.value));
    const svg = el("svg", { style: `max-width:${W}px`, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "bar chart" }, host);
    const plotW = W - labelW - valueW;
    items.forEach((d, i) => {
      const y = 2 + i * (rowH + gap), w = Math.max(1, plotW * d.value / max);
      const g = el("g", {}, svg);
      const lab = el("text", { x: labelW - 6, y: y + rowH / 2 + 4, "text-anchor": "end", class: d.mark ? "lbl" : "" }, g);
      const fit = Math.max(8, Math.floor((labelW - 16) / 6.1));
      lab.textContent = (d.mark ? "▸ " : "") + (d.label.length > fit ? d.label.slice(0, fit - 1) + "…" : d.label);
      el("rect", { x: labelW, y, width: w, height: rowH, rx: 3, fill: seriesColor(0), opacity: d.mark ? 1 : 0.62 }, g);
      const val = el("text", { x: labelW + w + 5, y: y + rowH / 2 + 4, class: "lbl" }, g);
      val.textContent = o.format ? o.format(d.value) : fmt(d.value);
      hover(el("rect", { x: 0, y, width: W, height: rowH, class: "hit" }, g), d.note || `${d.label}: ${fmt(d.value)}`);
      if (d.onclick) g.style.cursor = "pointer", g.addEventListener("click", d.onclick);
    });
  }

  // Vertical bars over an index (a histogram, or one bar per layer). values = [{x, y, note}].
  function barsV(host, values, o = {}) {
    frame(host, o.title, o.note);
    const W = o.width || 360, H = o.height || 120, m = { l: 34, r: 6, t: 6, b: 20 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b, max = o.max || Math.max(1e-12, ...values.map(d => d.y));
    const svg = el("svg", { style: `max-width:${W}px`, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "bar chart" }, host);
    for (const t of ticks(0, max, 3)) {
      const y = m.t + ph - ph * t / max;
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, svg);
      el("text", { x: m.l - 5, y: y + 4, "text-anchor": "end" }, svg).textContent = o.format ? o.format(t) : fmt(t, 2);
    }
    const bw = pw / values.length, gap = bw > 4 ? 2 : (bw > 2 ? 1 : 0);
    values.forEach((d, i) => {
      const h = ph * Math.min(d.y, max) / max, x = m.l + i * bw;
      el("rect", { x: x + gap / 2, y: m.t + ph - h, width: Math.max(0.5, bw - gap), height: Math.max(0, h), rx: bw > 6 ? 2 : 0, fill: seriesColor(0) }, svg);
      hover(el("rect", { x, y: m.t, width: bw, height: ph, class: "hit" }, svg), d.note || `${d.x}: ${fmt(d.y)}`);
    });
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, class: "axis" }, svg);
    (o.xticks || []).forEach(t => {
      el("text", { x: m.l + (t.i + 0.5) * bw, y: H - 6, "text-anchor": "middle" }, svg).textContent = t.label;
    });
    if (o.xlabel) el("text", { x: W - m.r, y: H - 6, "text-anchor": "end" }, svg).textContent = o.xlabel;
  }

  // Lines over a shared x: series = [{name, x:[], y:[]}]. Legend above for two or more; crosshair with all values.
  function lines(host, series, o = {}) {
    frame(host, o.title, o.note);
    if (series.length > 1) {
      const lg = document.createElement("div"); lg.className = "chart-legend";
      series.forEach((s, i) => { const sp = document.createElement("span"); sp.style.setProperty("--c", seriesColor(i)); sp.textContent = s.name; lg.appendChild(sp); });
      host.appendChild(lg);
    }
    const W = o.width || 720, H = o.height || 220, m = { l: 46, r: series.length > 1 && series.length <= 4 ? 96 : 10, t: 8, b: 24 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(v => Number.isFinite(v));
    const x0 = Math.min(...xs, 0), x1 = Math.max(...xs, 1);
    // o.log: a logarithmic y axis for values spread over decades; values at or below o.logFloor sit on the floor.
    const floor = o.logFloor || 1e-4, T = o.log ? (v => Math.log10(Math.max(v, floor))) : (v => v);
    const y0 = o.log ? Math.log10(floor) : (o.ymin != null ? o.ymin : Math.min(0, ...ys));
    const y1 = o.log ? Math.max(Math.ceil(T(Math.max(...ys, floor * 10))), y0 + 1) : Math.max(o.ymax || 0, ...ys, 1e-9);
    const X = v => m.l + pw * (v - x0) / (x1 - x0 || 1), Yt = t => m.t + ph - ph * (t - y0) / (y1 - y0 || 1), Y = v => Yt(T(v));
    const svg = el("svg", { style: `max-width:${W}px`, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "line chart" }, host);
    const yticks = o.log ? Array.from({ length: y1 - y0 + 1 }, (_, k) => y0 + k) : ticks(y0, y1, 4);
    for (const t of yticks) {
      el("line", { x1: m.l, x2: m.l + pw, y1: Yt(t), y2: Yt(t), class: "gridline" }, svg);
      el("text", { x: m.l - 6, y: Yt(t) + 4, "text-anchor": "end" }, svg).textContent =
        o.log ? (t === y0 ? "≤" : "") + String(+Math.pow(10, t).toPrecision(1)) : (o.format ? o.format(t) : fmt(t, 2));
    }
    for (const t of ticks(x0, x1, 6)) el("text", { x: X(t), y: H - 6, "text-anchor": "middle" }, svg).textContent = fmt(t, 0);
    el("line", { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: "axis" }, svg);
    if (o.xlabel) el("text", { x: m.l + pw, y: H - 6, "text-anchor": "end" }, svg).textContent = "";
    series.forEach((s, i) => {
      const d = s.x.map((x, k) => `${k ? "L" : "M"}${X(x).toFixed(1)},${Y(s.y[k]).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: seriesColor(i), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    });
    if (series.length > 1 && series.length <= 4) {
      // Direct labels at the line ends, pushed apart so that lines ending close together keep readable names.
      const ends = series.map((s, i) => ({ i, name: s.name, x: s.x.length ? X(s.x[s.x.length - 1]) : 0, y: s.x.length ? Y(s.y[s.y.length - 1]) : 0, has: !!s.x.length }))
        .filter(e => e.has).sort((a, b) => a.y - b.y);
      for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 13) ends[k].y = ends[k - 1].y + 13;
      const over = ends.length ? ends[ends.length - 1].y - (m.t + ph) : 0;
      if (over > 0) ends.forEach(e => { e.y -= over; });
      ends.forEach(e => {
        el("rect", { x: e.x + 6, y: e.y - 4, width: 8, height: 8, rx: 2, fill: seriesColor(e.i) }, svg);
        const t = el("text", { x: e.x + 18, y: e.y + 4, class: "lbl" }, svg);
        t.textContent = e.name.length > 13 ? e.name.slice(0, 12) + "…" : e.name;
      });
    }
    (o.marks || []).forEach(mk => {             // a labelled vertical rule, e.g. where the thinking ends
      if (!(mk.x > x0 && mk.x < x1)) return;
      el("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: m.t + ph, class: "axis", "stroke-dasharray": "3 3" }, svg);
      el("text", { x: X(mk.x) + 4, y: m.t + 10 }, svg).textContent = mk.label;
    });
    const cross = el("line", { y1: m.t, y2: m.t + ph, class: "axis", visibility: "hidden" }, svg);
    const dots = series.map((s, i) => el("circle", { r: 4, fill: seriesColor(i), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, class: "hit" }, svg);
    hit.addEventListener("mousemove", ev => {
      const box = svg.getBoundingClientRect(), px = (ev.clientX - box.left) * W / box.width;
      const xv = x0 + (px - m.l) / pw * (x1 - x0);
      let text = "";
      series.forEach((s, i) => {
        let k = 0, best = Infinity;
        s.x.forEach((x, j) => { const dd = Math.abs(x - xv); if (dd < best) best = dd, k = j; });
        if (!s.x.length) return;
        dots[i].setAttribute("cx", X(s.x[k])); dots[i].setAttribute("cy", Y(s.y[k])); dots[i].setAttribute("visibility", "visible");
        if (i === 0) { cross.setAttribute("x1", X(s.x[k])); cross.setAttribute("x2", X(s.x[k])); cross.setAttribute("visibility", "visible"); text += `${o.xname || "x"} ${fmt(s.x[k], 0)}${o.xnote ? "  " + o.xnote(s.x[k]) : ""}\n`; }
        text += `${s.name}: ${o.format ? o.format(s.y[k]) : fmt(s.y[k])}\n`;
      });
      tip(text.trim(), ev);
      if (o.onhover) o.onhover(Math.round(xv));
    });
    hit.addEventListener("mouseleave", () => { tip(null); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
    if (o.onclick) hit.addEventListener("click", ev => { const box = svg.getBoundingClientRect(); o.onclick(Math.round(x0 + ((ev.clientX - box.left) * W / box.width - m.l) / pw * (x1 - x0))); });
  }

  // Scatter of labelled points in one hue: identity is carried by the label, not by colour.
  function scatter(host, points, o = {}) {
    frame(host, o.title, o.note);
    const W = o.width || 720, H = o.height || 280, m = { l: 54, r: 120, t: 10, b: 36 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const pad = (lo, hi) => { const d = (hi - lo) || Math.abs(hi) || 1; return [lo - d * 0.08, hi + d * 0.08]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(0, ...ys), Math.max(...ys));
    const X = v => m.l + pw * (v - x0) / (x1 - x0), Y = v => m.t + ph - ph * (v - y0) / (y1 - y0);
    const svg = el("svg", { style: `max-width:${W}px`, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "scatter" }, host);
    for (const t of ticks(y0, y1, 4)) { el("line", { x1: m.l, x2: m.l + pw, y1: Y(t), y2: Y(t), class: "gridline" }, svg); el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = fmt(t, 3); }
    for (const t of ticks(x0, x1, 5)) el("text", { x: X(t), y: m.t + ph + 14, "text-anchor": "middle" }, svg).textContent = fmt(t, 2);
    el("line", { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: "axis" }, svg);
    el("text", { x: m.l + pw / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = o.xlabel || "";
    const yl = el("text", { x: 12, y: m.t + ph / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ph / 2})` }, svg); yl.textContent = o.ylabel || "";
    points.forEach(p => {
      const g = el("g", {}, svg);
      el("circle", { cx: X(p.x), cy: Y(p.y), r: 5, fill: seriesColor(0), stroke: css("--surface-1"), "stroke-width": 2, opacity: p.muted ? 0.45 : 1 }, g);
      const t = el("text", { x: X(p.x) + 9, y: Y(p.y) + 4, class: "lbl" }, g); t.textContent = p.label.length > 20 ? p.label.slice(0, 19) + "…" : p.label;
      hover(el("circle", { cx: X(p.x), cy: Y(p.y), r: 12, class: "hit" }, g), p.note || `${p.label}\n${fmt(p.x)} , ${fmt(p.y)}`);
    });
  }

  // Heat map on a canvas: matrix[row][col] in [0,1], one hue. Rows are layers, columns are tokens.
  function heat(host, matrix, o = {}) {
    frame(host, o.title, o.note);
    const rows = matrix.length, cols = rows ? matrix[0].length : 0;
    const canvas = document.createElement("canvas");
    canvas.className = "heat"; canvas.width = Math.max(1, cols); canvas.height = Math.max(1, rows);
    canvas.style.height = Math.min(360, Math.max(120, rows * 4)) + "px";
    host.appendChild(canvas);
    const ctx = canvas.getContext("2d"), ramp = Array.from({ length: 8 }, (_, k) => css(`--seq-${k}`));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      ctx.fillStyle = ramp[Math.max(0, Math.min(7, Math.round(matrix[r][c] * 7)))];
      ctx.fillRect(c, r, 1, 1);
    }
    const at = ev => { const b = canvas.getBoundingClientRect(); return [Math.min(rows - 1, Math.floor((ev.clientY - b.top) / b.height * rows)), Math.min(cols - 1, Math.floor((ev.clientX - b.left) / b.width * cols))]; };
    canvas.addEventListener("mousemove", ev => { const [r, c] = at(ev); tip(o.cell ? o.cell(r, c) : `${r}, ${c}: ${fmt(matrix[r][c])}`, ev); });
    canvas.addEventListener("mouseleave", () => tip(null));
    if (o.onclick) canvas.addEventListener("click", ev => { const [r, c] = at(ev); o.onclick(r, c); });
  }

  return { barsH, barsV, lines, scatter, heat, fmt, pct, seq, seriesColor, tip, css };
})();
