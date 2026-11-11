// Routing lab: the side panel's Token and Analysis pages. Charts take the panel's width, so they follow it when
// the panel is resized.
"use strict";

// Line breaks in the context around a token are shown as a pilcrow, so the context stays on a few lines.
const flat = t => t.replace(/\n+/g, m => " " + "\u00b6".repeat(Math.min(2, m.length)) + " ");
const panelWidth = host => Math.max(260, Math.min(720, host.clientWidth - 24));

function sideTab(name) {
  if (state.sideHidden) setSideHidden(false);
  state.sideTab = name;
  $$("[data-side]").forEach(t => { const on = t.dataset.side === name; t.classList.toggle("on", on); t.setAttribute("aria-selected", on); });
  $$(".side-page").forEach(p => p.classList.toggle("on", p.id === "side-" + name));
  store.set("layout", layoutState());
}

function selectToken(a, i, o = {}) {
  if (!a || i < 0 || i >= a.records.length) return;
  state.selected = { answer: a, i };
  $$(".t.sel").forEach(n => n.classList.remove("sel"));
  const spans = a.el ? [...a.el.querySelectorAll(`.t[data-i="${i}"]`)] : [];
  if (spans.length) {
    const fold = spans[0].closest("details");
    if (fold && !fold.open) { a.thinkOpen = true; fold.open = true; }
    spans.forEach(s => s.classList.add("sel"));
    if (!o.noScroll) spans[0].scrollIntoView({ block: "nearest" });
  }
  renderToken();
  sideTab("token");
}
function stepToken(d) {
  const s = state.selected;
  if (s) selectToken(s.answer, Math.max(0, Math.min(s.answer.records.length - 1, s.i + d)));
}

function renderToken(showAll = false) {
  const host = $("#side-token");
  if (!state.selected) return;
  const { answer: a, i } = state.selected, rec = a.records[i];
  if (!rec) return;
  const W = panelWidth(host);
  const sampled = state.showSampled && rec.samp && rec.samp.length;
  const order = [...((sampled ? rec.samp : rec.top) || [])].sort((x, y) => y[1] - x[1]);
  const nThink = thinkingSpan(a);
  const nav = h("div", { class: "tok-nav" },
    h("button", { type: "button", class: "tbtn", text: "Previous", title: "The token before (left arrow key)", disabled: i === 0, onclick: () => stepToken(-1) }),
    h("button", { type: "button", class: "tbtn", text: "Next", title: "The token after (right arrow key)", disabled: i >= a.records.length - 1, onclick: () => stepToken(1) }),
    h("span", { class: "muted small", text: `token ${i + 1} of ${a.records.length}` + (nThink ? (i < nThink ? ", in the thinking" : ", in the answer") : "") }));
  const context = h("div", { class: "tok-context" },
    h("span", { class: "muted", text: "..." + flat(a.records.slice(Math.max(0, i - 14), i).map(r => r.text).join("")) }),
    h("span", { class: "tokbig", text: rec.text }),
    h("span", { class: "muted", text: flat(a.records.slice(i + 1, i + 6).map(r => r.text).join("")) + "..." }));
  // First what a reader asks of a token: how likely, how sure, how it was routed; then what each rule did to it.
  const tiles = h("div", { class: "tiles compact" });
  const tile = (k, v, title, cls) => tiles.append(h("div", { class: "tile" + (cls ? " " + cls : ""), title: title || null }, h("div", { class: "v", text: v }), h("div", { class: "k", text: k })));
  tile("probability", fmt(praw(rec)), "At temperature 1, before filters" + (rec.p_raw != null && Math.abs(rec.p_raw - rec.p) > 1e-9 ? `; ${fmt(rec.p)} after temperature and top p` : ""));
  tile("rank", rankOf(rec) === 1 ? "first" : String(rankOf(rec)), "Which choice it was among the candidates listed");
  tile("uncertainty", fmt(rec.entropy, 2), "Entropy of the distribution, in nats");
  const r = rec.route, ex = expertsOf(rec);
  if (ex) tile("experts a layer", ex.perLayer.toFixed(2), expertsText(ex), ex.diff ? "flag" : "");
  if (r) {
    tile("picks moved", pct(r.moved / (r.picks || 1)), `${r.moved} of ${r.picks} picks moved off the router's top four`);
    tile("worst layer kept", pct(r.min_kept), `Router mass kept: ${pct(r.mass_kept)} on average, ${pct(r.min_kept)} in the worst layer`);
  }
  const parts = [nav, context, tiles];
  if (r && r.changed) {
    const acted = Object.entries(r.changed).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]);
    const list = h("dl", { class: "kv" });
    for (const [k, v] of acted) list.append(h("dt", { text: (CHANGED[k] || [k])[0] }), h("dd", { text: `${plural(v, "layer")}: ${(CHANGED[k] || [, ""])[1]}` }));
    parts.push(h("h3", { text: "What each rule changed" }),
      acted.length ? list : h("p", { class: "muted small", text: "No rule changed this token's routing: every layer ran the router's own picks." }));
  }
  const kv = h("dl", { class: "kv small" });
  const add = (k, v) => kv.append(h("dt", { text: k }), h("dd", { text: v }));
  if (ex) add("experts run", expertsText(ex));
  if (r) {
    add("router mass kept", `${pct(r.mass_kept)} on average, ${pct(r.min_kept)} in the worst layer; ${r.rows_under_08 != null ? r.rows_under_08 : "-"} layers under 0.8`);
    add("loads, staged", `${r.loads != null ? r.loads : "-"}, ${r.staged != null ? r.staged : "-"}` + (r.in_flight ? ` (${r.in_flight} in flight)` : ""));
    if (r.dropped || r.widened || r.layers_dropped) add("three, five, layer dropped", `${r.dropped || 0}, ${r.widened || 0}, ${r.layers_dropped || 0}`);
  }
  add("came from", (rec.source === "draft" ? "an accepted draft" : "the model's own sample") + (rec.forced ? ", forced" : "") + `; ${plural(rec.rows || 1, "row")} in the pass`);
  if (rec.override) add("replaced by the engine", String(rec.override));
  add("token id", String(rec.id) + (isTag(rec) ? ", a control token" : ""));
  const dist = h("div");
  const outside = sampled ? rec.samp_rest : rec.rest;
  const shown = showAll ? order : order.slice(0, 20);
  Charts.barsH(dist, shown.map(t => ({ label: JSON.stringify(t[2] != null ? t[2] : String(t[0])).slice(1, -1), value: t[1], mark: t[0] === rec.id,
    note: `${JSON.stringify(t[2])}  id ${t[0]}\nprobability ${fmt(t[1])}` + (t[0] === rec.id ? "\nthe token that was committed" : "") })),
    { title: sampled ? "What this token was drawn from" : "The model's next-token distribution here",
      note: (sampled ? `After temperature and filters; ${rec.samp_n != null ? rec.samp_n : order.length} tokens could have been drawn. ` : "At temperature 1, before filters. ") +
        `The committed token is marked. Mass outside the ${order.length} candidates listed: ${fmt(outside || 0)}.`, format: v => fmt(v), width: W, labelW: Math.min(160, W * 0.36) });
  const which = rec.samp && rec.samp.length ? h("div", { class: "chart-tools" }, "Show",
    segmented([["raw", "the model's distribution"], ["samp", "what was sampled from"]], sampled ? "samp" : "raw", v => { state.showSampled = v === "samp"; renderToken(showAll); }, "Which distribution")) : null;
  parts.push(h("h3", { text: "Distribution" }), which, dist);
  if (!showAll && order.length > 20) parts.push(h("button", { type: "button", class: "link small", text: `Show all ${order.length} candidates`, onclick: () => renderToken(true) }));
  parts.push(h("h3", { text: "More about this token" }), kv);
  if (rec.layers && rec.layers.kept && rec.layers.moved) {
    const lost = h("div"), moved = h("div"), L = rec.layers, xt = [0, 14, 28, 42, 56].map(i => ({ i, label: String(i + 1) }));
    if (L.k) {
      const ran = h("div");
      Charts.barsV(ran, L.k.map((v, li) => ({ x: li + 1, y: v, note: `layer ${li + 1}\n${plural(v, "expert")} ran` })),
        { title: "Experts run, by layer", note: "Four is the router's number. Three or fewer: a pick or the layer was dropped. Five: the row was widened.", max: 5, format: v => fmt(v, 0), xticks: xt, xlabel: "layer", height: 90, width: W });
      parts.push(h("h3", { text: "By layer" }), ran);
    }
    Charts.barsV(lost, L.kept.map((v, li) => ({ x: li + 1, y: 1 - v, note: `layer ${li + 1}\nrouter mass kept ${pct(v)}\npicks moved ${L.moved[li]}, executed ${L.k ? L.k[li] : 4}` })),
      { title: "Router mass lost, by layer", max: Math.max(0.25, ...L.kept.map(v => 1 - v)), format: v => pct(v), xticks: xt, xlabel: "layer", width: W });
    Charts.barsV(moved, L.moved.map((v, li) => ({ x: li + 1, y: v, note: `layer ${li + 1}\npicks moved ${v} of ${L.k ? L.k[li] : 4}` })),
      { title: "Picks moved, by layer", max: 5, format: v => fmt(v, 0), xticks: xt, xlabel: "layer", height: 90, width: W });
    parts.push(...(L.k ? [] : [h("h3", { text: "By layer" })]), lost, moved);
  }
  const top = host.scrollTop;
  host.replaceChildren(...parts.filter(Boolean));
  host.scrollTop = top;
}

// ---------------------------------------------------------------------------------------------- analysis
function analyse(a) {
  state.analysed = a;
  sideTab("analysis");
  renderAnalysis();
}
function renderAnalysis() {
  const a = state.analysed, host = $("#side-analysis");
  if (!a) return;
  const W = panelWidth(host), st = a.stats || {}, rt = st.route || {}, recs = a.records;
  const tiles = h("div", { class: "tiles" });
  const tile = (k, v, title) => tiles.append(h("div", { class: "tile", title: title || null }, h("div", { class: "v", text: v }), h("div", { class: "k", text: k })));
  const nThink = thinkingSpan(a), smooth = recs.length > 120 ? 8 : 1;
  tile("tokens", String(st.tokens || recs.length));
  if (nThink) tile("thinking tokens", String(nThink));
  tile("perplexity", fmt(st.perplexity, 2), "exp of the mean surprise of the chosen tokens; lower means the model was surer");
  tile("mean probability", fmt(st.mean_prob, 3));
  tile("uncertainty, nats", fmt(st.entropy_mean, 2));
  tile("first choice taken", pct(st.chosen_top1));
  tile("drafted", pct(st.drafted));
  if (rt.moved_share != null) {
    if (rt.experts_per_layer != null) tile("experts a layer", rt.experts_per_layer.toFixed(2), "experts run per layer, averaged over the tokens; the router's number is four");
    tile("picks moved", pct(rt.moved_share)); tile("router mass kept", pct(rt.mass_kept_mean));
    tile("layers under 0.8, a token", fmt(rt.rows_under_08, 2)); tile("loads a token", fmt(rt.loads, 1));
    const sp = speedOf(st);
    if (sp) tile(sp.name, fmt(sp.v, 1), sp.what + "; an estimate for ordering rules, not a timing.");
    if (st.prompt_route) { tile("prompt: picks moved", pct(st.prompt_route.moved_share), "the prompt's own rows"); tile("prompt: mass kept", pct(st.prompt_route.mass_kept_mean), `the prompt's own rows; lowest ${pct(st.prompt_route.min_kept)}`); }
  }
  const parts = [h("div", { class: "an-head" }, h("span", { class: "chip rule", text: a.describe }), h("span", { class: "muted small", text: a.drafting ? "drafting" : "one row" })), tiles];
  const ch = a.changed || (a.changed = changedOf(recs));
  if (ch) {
    const rows = Object.entries(ch).sort((x, y) => y[1].per - x[1].per);
    parts.push(h("h3", { text: "What each rule changed" }),
      h("p", { class: "muted small", text: "Over the answer's tokens: the mean number of layers (of 57) at which the rule acted, and the share of tokens it acted on at least once." }),
      h("table", { class: "data" }, h("thead", {}, h("tr", {}, h("th", { text: "Rule" }), h("th", { text: "Layers a token" }), h("th", { text: "Tokens" }))),
        h("tbody", {}, rows.map(([k, v]) => h("tr", { class: v.per ? "" : "zero", title: (CHANGED[k] || [, ""])[1] },
          h("td", { text: (CHANGED[k] || [k])[0] }), h("td", { class: "num", text: v.per ? fmt(v.per, 2) : "0" }), h("td", { class: "num", text: pct(v.share) }))))));
  }
  const jump = i => selectToken(a, Math.max(0, Math.min(recs.length - 1, i)));
  const sur = h("div");
  Charts.lines(sur, [{ name: "surprise", x: recs.map((_, i) => i), y: rolling(recs.map(r => -Math.log(praw(r))), smooth) },
                     { name: "uncertainty", x: recs.map((_, i) => i), y: rolling(recs.map(r => r.entropy || 0), smooth) }],
    { title: "Surprise and uncertainty along the answer", note: "In nats. Surprise is minus the log of the chosen token's probability; uncertainty is the entropy of the distribution it came from. Surprise far above uncertainty marks an unlikely choice." +
        (smooth > 1 ? ` Rolling mean over ${smooth} tokens.` : "") + " Click to inspect a token.",
      xname: "token", width: W, height: 170, marks: nThink && nThink < recs.length ? [{ x: nThink, label: "answer" }] : [],
      xnote: i => JSON.stringify((recs[i] || {}).text || ""), onclick: jump });
  const bins = Array.from({ length: 10 }, () => 0);
  recs.forEach(r => bins[Math.min(9, Math.floor(praw(r) * 10))]++);
  const hist = h("div");
  Charts.barsV(hist, bins.map((n, b) => ({ x: b, y: n, note: `probability ${b / 10} to ${(b + 1) / 10}: ${plural(n, "token")}` })),
    { title: "How sure the model was of the tokens it chose", xticks: [0, 5, 9].map(i => ({ i, label: (i / 10).toFixed(1) })), xlabel: "probability", format: v => fmt(v, 0), width: W });
  parts.push(sur, hist);
  if (recs.some(r => r.layers && r.layers.kept)) {            // the first token comes from the prompt and has no routing
    const n = state.info.n_layers, kept = (r, li) => r.layers && r.layers.kept ? r.layers.kept[li] : 1;
    const matrix = Array.from({ length: n }, (_, li) => recs.map(r => Math.min(1, Math.max(0, 1 - kept(r, li)) / 0.5)));
    const heat = h("div");
    Charts.heat(heat, matrix, { title: "Router mass lost: layers down, tokens across", note: "Darkest is half the mass lost or more. Click a cell to inspect that token.",
      cell: (r, c) => `layer ${r + 1}, token ${c + 1} ${JSON.stringify(recs[c].text)}\n` + (recs[c].layers ? `mass kept ${pct(kept(recs[c], r))}, picks moved ${recs[c].layers.moved[r]}` : "no routing record"), onclick: (r, c) => jump(c) });
    parts.push(heat);
  }
  const least = recs.map((r, i) => ({ i, s: -Math.log(praw(r)) })).sort((x, y) => y.s - x.s).slice(0, 8);
  const item = (label, what, meta, i) => h("div", { class: "item" }, h("button", { type: "button", class: "link mono", text: label, onclick: () => jump(i) }),
    h("span", { class: "what", text: what }), meta ? h("span", { class: "meta", text: meta }) : null);
  parts.push(h("h3", { text: "Least likely tokens" }), h("div", { class: "list" }, least.map(x => item(JSON.stringify(recs[x.i].text), "..." + recs.slice(Math.max(0, x.i - 6), x.i).map(r => r.text).join(""), `p ${fmt(praw(recs[x.i]))}`, x.i))));
  parts.push(h("h3", { text: "Doubled words" }), a.slips && a.slips.length ? h("div", { class: "list" }, a.slips.map(s => item(s.text, s.context, null, s.i))) : h("p", { class: "muted small", text: "None found." }));
  parts.push(h("h3", { text: "Repeated phrases" }), a.repeats && a.repeats.length ? h("div", { class: "list" }, a.repeats.map(s => item(s.text, "", `${s.times} times`, s.i))) : h("p", { class: "muted small", text: "No phrase of six words occurs twice." }));
  host.replaceChildren(...parts);
  host.scrollTop = 0;
}
function rolling(y, w) { const out = []; let s = 0; for (let i = 0; i < y.length; i++) { s += y[i]; if (i >= w) s -= y[i - w]; out.push(s / Math.min(i + 1, w)); } return out; }
// The speed estimate of a set of records: the reader's priced token time when there is one, else the engine's
// priced load cost. Both order rules; neither is a timing.
function speedOf(stats) {
  const p = stats && stats.price, r = stats && stats.route;
  if (p && p.ms_per_token) return { v: p.ms_per_token, name: "est. ms a token", what: "token time priced from the pass counts with the measured pass prices" + (p.exact ? "" : " (approximate: per-layer counts were not recorded)") };
  if (r && r.est_ms) return { v: r.est_ms, name: "est. load cost, ms a token", what: "the engine's priced cost of the expert loads behind each token" };
  return null;
}
