// Routing lab page. All model output is placed with textContent; nothing from the model is ever parsed as HTML.
"use strict";

const $ = sel => document.querySelector(sel);
const { fmt, pct } = Charts;

function h(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k in n && typeof v !== "string") n[k] = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid);
  return n;
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({ error: `the lab's server answered ${r.status}` }));
  if (!r.ok && !data.error) data.error = `the lab's server answered ${r.status}`;
  return data;
}

const state = {
  info: null, settings: {}, sampling: {}, turns: [], busy: false, colorBy: "surprisal",
  selected: null,            // {answer, i}
  analysed: null,            // answer shown in the Analysis tab
  arms: [], meterJob: null, presets: { built_in: {}, saved: {} }, ticked: new Set(),
};

// ---------------------------------------------------------------------------------------------------- settings
function describe(s) {
  const d = state.info.defaults, parts = [`bonus ${(+s.routing_prior_bonus).toFixed(2)}`];
  const names = { route_bonus_draft: "draft bonus", route_prompt_bonus: "prompt bonus", route_floor: "budget", route_protect: "protect", route_priced: "priced",
                  route_layer_drop: "layer drop", route_k_drop: "drop pick", route_k_widen: "widen" };
  for (const [k, name] of Object.entries(names)) if (s[k] !== d[k]) parts.push(`${name} ${+(k === "route_floor" ? 1 - s[k] : +s[k]).toFixed(3)}`);
  if (s.route_weights) parts.push(s.route_weights === 1 ? "dampen" : "amplify");
  if ((s.route_layer_scale || []).some(x => Math.abs(x - 1) > 1e-9)) parts.push("by layer");
  return parts.join(", ");
}
const LAYER_GROUPS = [[0, 9], [10, 19], [20, 29], [30, 39], [40, 49], [50, 56]];

function buildRules() {
  const host = $("#rules");
  host.replaceChildren();
  const groups = new Map();
  for (const s of state.info.schema) {
    if (!groups.has(s.group)) groups.set(s.group, { main: [], adv: [] });
    groups.get(s.group)[s.advanced ? "adv" : "main"].push(s);
  }
  for (const [name, g] of groups) {
    const box = h("details", { class: "group", open: true }, h("summary", { text: name }));
    g.main.forEach(s => box.append(field(s)));
    if (g.adv.length) box.append(h("details", { class: "adv" }, h("summary", { text: "More" }), g.adv.map(field)));
    host.append(box);
  }
  const samp = $("#sampling");
  samp.replaceChildren(...state.info.sampling.map(s => field(s, state.sampling)));
  summary();
}

function field(s, target = state.settings) {
  const wrap = h("div", { class: "field" }, h("label", { text: s.label }));
  const set = v => { target[s.key] = v; summary(); };
  if (s.kind === "choice") {
    const sel = h("select", { onchange: e => { const c = s.choices.find(c => String(c[0]) === e.target.value); set(c ? c[0] : e.target.value); } },
      s.choices.map(c => h("option", { value: c[0], text: c[1] })));
    sel.value = target[s.key];
    wrap.append(sel);
  } else if (s.kind === "layers") {
    const grid = h("div", { class: "layer-groups" });
    const all = h("textarea", { rows: 3, style: "width:100%;font:12px var(--mono)", onchange: e => {
      const v = e.target.value.split(/[\s,]+/).filter(Boolean).map(Number);
      if (v.length === state.info.n_layers && v.every(x => Number.isFinite(x) && x >= s.min && x <= s.max)) { target[s.key] = v; sync(); summary(); }
      else e.target.value = target[s.key].join(", ");
    } });
    const inputs = LAYER_GROUPS.map(([a, b]) => {
      const range = h("input", { type: "range", min: s.min, max: s.max, step: s.step });
      const num = h("input", { type: "number", min: s.min, max: s.max, step: s.step });
      const apply = v => { v = Math.max(s.min, Math.min(s.max, +v)); for (let i = a; i <= b; i++) target[s.key][i] = v; sync(); summary(); };
      range.addEventListener("input", e => apply(e.target.value)); num.addEventListener("change", e => apply(e.target.value));
      grid.append(h("span", { text: `layers ${a + 1}-${b + 1}` }), range, num);
      return { a, b, range, num };
    });
    function sync() {
      for (const g of inputs) {
        const vals = target[s.key].slice(g.a, g.b + 1), same = vals.every(x => x === vals[0]);
        g.range.value = vals[0]; g.num.value = same ? vals[0] : ""; g.num.placeholder = same ? "" : "mixed";
      }
      all.value = target[s.key].map(x => +x.toFixed(3)).join(", ");
    }
    sync();
    wrap.append(grid, h("details", { class: "adv" }, h("summary", { text: "All 57 values" }), all));
  } else {
    // `invert`: the control shows 1 minus the stored value (a budget on lost mass, stored as a floor on mass kept).
    const off = s.default, optional = !!s.optional, inv = !!s.invert, conv = v => inv ? +(1 - v).toFixed(6) : v;
    const lo = inv ? conv(s.max) : (optional && s.min < 0 ? 0 : s.min), hi = inv ? conv(s.min) - (optional ? s.step : 0) : s.max;
    const range = h("input", { type: "range", min: lo, max: hi, step: s.step });
    const num = h("input", { type: "number", min: lo, max: hi, step: s.step, class: s.max > 99999 ? "wide" : "" });
    const row = h("div", { class: "row" });
    let box = null, last = conv(s.on_value != null ? s.on_value : (s.key === "route_bonus_draft" ? 0.20 : s.default));
    const show = () => {
      const on = !optional || target[s.key] !== off;
      if (box) box.checked = on;
      wrap.classList.toggle("off", !on);
      range.disabled = num.disabled = !on;
      range.value = num.value = on ? conv(target[s.key]) : last;
    };
    const apply = v => { v = Math.max(lo, Math.min(hi, +v)); if (s.kind === "int") v = Math.round(v); last = v; set(conv(v)); show(); };
    range.addEventListener("input", e => apply(e.target.value)); num.addEventListener("change", e => apply(e.target.value));
    if (optional) { box = h("input", { type: "checkbox", onchange: e => { set(e.target.checked ? conv(last) : off); show(); } }); row.append(box); }
    row.append(range, num);
    wrap.append(row);
    show();
    wrap._show = show;
  }
  if (s.help) wrap.append(h("div", { class: "help", text: s.help }));
  return wrap;
}

function summary() {
  if (!state.info) return;
  const mode = $("#mode").value === "drafting" ? "drafting" : "one row";
  $("#rule-summary").textContent = `Next prompt: ${describe(state.settings)}; ${mode}; temperature ${state.sampling.temperature}`;
}
function loadSettings(s) { state.settings = JSON.parse(JSON.stringify(s)); buildRules(); }

async function loadPresets() {
  state.presets = await api("/api/presets");
  for (const sel of [$("#preset"), $("#arm-preset")]) {
    const first = sel.firstElementChild; sel.replaceChildren(first);
    for (const [label, set] of [["Built in", state.presets.built_in], ["Saved", state.presets.saved]]) {
      const names = Object.keys(set || {});
      if (names.length) sel.append(h("optgroup", { label }, names.map(n => h("option", { value: n, text: n }))));
    }
  }
}
const presetByName = n => state.presets.built_in[n] || state.presets.saved[n];

// -------------------------------------------------------------------------------------------------------- chat
function tokenValue(rec, mode) {
  const r = rec.route || {};
  switch (mode) {
    case "surprisal": return Math.min(1, -Math.log(praw(rec)) / 6);
    case "entropy": return Math.min(1, (rec.entropy || 0) / 4);
    case "rank": { const k = rankOf(rec); return Math.min(1, (k - 1) / 9); }
    case "lost": return r.min_kept == null ? 0 : Math.min(1, (1 - r.min_kept) / 0.5);
    case "moved": return r.picks ? Math.min(1, (r.moved / r.picks) / 0.5) : 0;
    default: return 0;
  }
}
const praw = rec => Math.max(rec.p_raw != null ? rec.p_raw : rec.p, 1e-12);
function rankOf(rec) {
  const order = [...(rec.top || [])].sort((a, b) => b[1] - a[1]);
  const k = order.findIndex(t => t[0] === rec.id);
  return k < 0 ? order.length + 1 : k + 1;
}
function ink(hex) {          // black or white, whichever reads on this background
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return "";
  const [r, g, b] = m.slice(1).map(x => parseInt(x, 16) / 255).map(c => c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.32 ? "#0b0b0b" : "#ffffff";
}
function paint(span, rec) {
  const mode = state.colorBy;
  span.style.background = span.style.color = "";
  if (mode === "none") return;
  if (mode === "source") {
    if (rec.source === "draft") { span.style.background = Charts.css("--seq-2"); span.style.color = ink(Charts.css("--seq-2")); }
    return;
  }
  const { bg, step } = Charts.seq(tokenValue(rec, mode));
  if (step > 0) { span.style.background = bg; span.style.color = ink(bg); }
}
function legend() {
  const host = $("#color-legend"), mode = state.colorBy;
  host.replaceChildren();
  if (mode === "none") return;
  if (mode === "source") { host.append(h("span", { class: "ramp" }, h("i", { style: `background:${Charts.css("--seq-2")}` })), "an accepted draft"); return; }
  const ends = { surprisal: ["likely", "unlikely"], entropy: ["certain", "uncertain"], rank: ["first choice", "tenth or lower"],
                 lost: ["all router mass kept", "half lost in some layer"], moved: ["no pick moved", "half the picks moved"] }[mode];
  host.append(ends[0], h("span", { class: "ramp" }, [1, 2, 3, 4, 5, 6, 7].map(k => h("i", { style: `background:${Charts.css("--seq-" + k)}` }))), ends[1]);
}

// Doubled words ("the the"): a cheap flag for the kind of damage a high bonus produced in the usability blocks.
function findSlips(records) {
  let text = "", spans = [];
  records.forEach((r, i) => { spans.push([text.length, text.length + (r.text || "").length, i]); text += r.text || ""; });
  const hits = [], re = /\b([A-Za-z]{2,})(\s+)\1\b/gi, allowed = new Set(["that", "had", "very", "so", "no", "bye", "ha", "knock"]);
  let m;
  while ((m = re.exec(text))) {
    if (allowed.has(m[1].toLowerCase())) continue;
    const at = m.index + m[1].length + m[2].length;
    const tok = spans.find(s => at >= s[0] && at < s[1]);
    hits.push({ i: tok ? tok[2] : 0, text: m[0], context: text.slice(Math.max(0, m.index - 30), m.index + m[0].length + 30) });
  }
  return hits;
}
// Phrases of six words that occur more than once: the mark of a model that has started to loop.
function findRepeats(records) {
  let text = "", spans = [];
  records.forEach((r, i) => { spans.push([text.length, i]); text += r.text || ""; });
  const words = [...text.matchAll(/[\p{L}\p{N}_']+/gu)].map(m => ({ w: m[0].toLowerCase(), at: m.index }));
  const seen = new Map(), N = 6;
  for (let k = 0; k + N <= words.length; k++) {
    const key = words.slice(k, k + N).map(x => x.w).join(" ");
    if (!seen.has(key)) seen.set(key, []);
    seen.get(key).push(words[k].at);
  }
  const out = [];
  let covered = -1;
  for (const [key, at] of seen) {
    if (at.length < 2 || at[1] <= covered) continue;          // one entry per repeated stretch, not one per shift
    covered = at[1] + key.length;
    let tok = 0;
    for (const sp of spans) { if (sp[0] <= at[1]) tok = sp[1]; else break; }
    out.push({ i: tok, text: key, times: at.length });
  }
  return out.sort((a, b) => b.times - a.times).slice(0, 12);
}
const THINK_END = new Set(["</mm:think>", "</think>"]), THINK_BEGIN = new Set(["<mm:think>", "<think>"]);
// How many leading tokens are the model's thinking (its closing tag included). When thinking is on, the opening tag
// is part of the prompt, so until the closing tag arrives the server's reasoning stream is what says so.
function thinkingSpan(a) {
  let end = -1, begun = false;
  a.records.forEach((r, k) => { const t = (r.text || "").trim(); if (THINK_END.has(t)) end = k; else if (THINK_BEGIN.has(t)) begun = true; });
  return end >= 0 ? end + 1 : (begun || a.reasoning ? a.records.length : 0);
}

function renderConversation() {
  const host = $("#conversation");
  host.replaceChildren();
  if (!state.turns.length) host.append(h("p", { class: "hint", text: "Set the rules on the right, then write a prompt. Each answer keeps the rules it was made with, so you can try the same prompt under different rules and compare." }));
  state.turns.forEach(turn => {
    host.append(h("div", { class: "msg user" }, h("div", { class: "who", text: "You" }), turn.user));
    turn.answers.forEach((a, k) => host.append(answerBlock(turn, a, k)));
  });
  host.scrollTop = host.scrollHeight;
}

function answerBlock(turn, a, k) {
  const box = h("div", { class: "answer" + (turn.answers.length > 1 && turn.active !== k ? " inactive" : "") });
  a.el = box;
  fillAnswer(turn, a, k);
  return box;
}
function answerHead(a) {
  const st = a.stats || {}, rt = st.route || {}, n = st.tokens || a.records.length;
  return h("div", { class: "answer-head" },
    h("span", { class: "chip rule", text: a.describe }), h("span", { class: "chip", text: a.drafting ? "drafting" : "one row" }),
    a.status === "running" ? h("span", { class: "chip", text: "writing" }) : null,
    a.status === "stopped" ? h("span", { class: "chip", text: "stopped" }) : null,
    n ? h("span", { class: "chip", text: `${n} tokens` }) : null,
    st.perplexity ? h("span", { class: "chip", title: "exp of the mean surprise of the chosen tokens", text: `perplexity ${fmt(st.perplexity, 2)}` }) : null,
    rt.moved_share != null ? h("span", { class: "chip", title: "picks moved off the router's top four", text: `moved ${pct(rt.moved_share)}` }) : null,
    rt.mass_kept_mean != null ? h("span", { class: "chip", title: "mean share of router mass kept, and the lowest in any layer of any token", text: `mass kept ${pct(rt.mass_kept_mean)}, lowest ${pct(rt.min_kept)}` }) : null,
    a.slips && a.slips.length ? h("span", { class: "chip warn", text: `${a.slips.length} doubled word${a.slips.length > 1 ? "s" : ""}` }) : null,
    a.repeats && a.repeats.length ? h("span", { class: "chip warn", text: `${a.repeats.length} repeated phrase${a.repeats.length > 1 ? "s" : ""}` }) : null);
}
function tokenSpan(a, rec, i, slipAt) {
  const tag = THINK_END.has((rec.text || "").trim()) || THINK_BEGIN.has((rec.text || "").trim());
  const span = h("span", { class: "t" + (tag ? " tag" : "") + (slipAt && slipAt.has(i) ? " slip" : "") +
    (state.selected && state.selected.answer === a && state.selected.i === i ? " sel" : ""), text: rec.text });
  span.dataset.i = i;
  if (!tag) paint(span, rec);
  return span;
}
function tokenHost(a, cls) {             // one set of listeners per block, whatever the number of tokens
  const host = h("div", { class: cls });
  const at = ev => { const t = ev.target.closest(".t"); return t ? +t.dataset.i : -1; };
  host.addEventListener("click", ev => { const i = at(ev); if (i >= 0) selectToken(a, i); });
  host.addEventListener("mousemove", ev => {
    const i = at(ev), rec = a.records[i];
    if (!rec) { Charts.tip(null); return; }
    Charts.tip(`${JSON.stringify(rec.text)}  p ${fmt(praw(rec))}  rank ${rankOf(rec)}` + (rec.route && rec.route.picks ? `\nmoved ${rec.route.moved} of ${rec.route.picks}, lowest mass kept ${pct(rec.route.min_kept)}` : ""), ev);
  });
  host.addEventListener("mouseleave", () => Charts.tip(null));
  return host;
}
function fillAnswer(turn, a, k) {
  const box = a.el;
  if (!box) return;
  a.dom = null;
  let body;
  if (a.records.length && !a.plain) {
    const nThink = thinkingSpan(a), slipAt = new Set((a.slips || []).map(s => s.i));
    const think = tokenHost(a, "tokens thinking-tokens"), answer = tokenHost(a, "tokens");
    a.records.forEach((rec, i) => (i < nThink ? think : answer).append(tokenSpan(a, rec, i, slipAt)));
    const fold = h("details", { class: "thinking", hidden: !nThink }, h("summary", { text: `Thinking, ${nThink} tokens` }), think);
    fold.open = a.thinkOpen != null ? a.thinkOpen : (a.status === "running" && nThink === a.records.length);
    // A click is the reader's choice and is kept; opening it for them while the thinking streams is not.
    fold.firstElementChild.addEventListener("click", () => { a.thinkOpen = !fold.open; });
    body = h("div", {}, fold, answer);
    a.dom = { fold, think, answer, placed: a.records.length, inThink: Math.min(nThink, a.records.length) };
  } else {
    body = h("div", {},
      a.reasoning ? h("details", { class: "thinking" }, h("summary", { text: "Thinking" }), h("div", { class: "plain thinking-tokens", text: a.reasoning })) : null,
      h("div", { class: "plain", text: a.text || (a.status === "running" ? "…" : "") }));
  }
  const foot = h("div", { class: "answer-foot" });
  if (a.status !== "running") {
    if (a.records.length) foot.append(
      h("button", { class: "link", text: "Analyse", onclick: () => analyse(a) }),
      h("button", { class: "link", text: "Measure fidelity", onclick: () => toMeter(a) }),
      h("button", { class: "link", text: a.plain ? "Show tokens" : "Show plain text", onclick: () => { a.plain = !a.plain; fillAnswer(turn, a, k); } }));
    foot.append(h("button", { class: "link", text: "Try again with the current rules", onclick: () => generate(turn) }));
    if (turn.answers.length > 1 && turn.active !== k) foot.append(h("button", { class: "link", text: "Continue the conversation from this answer", onclick: () => { turn.active = k; renderConversation(); } }));
    foot.append(h("button", { class: "link", text: "Copy", onclick: () => navigator.clipboard && navigator.clipboard.writeText(a.text) }));
    if (a.run && a.records.length) foot.append(h("a", { class: "link", href: `/api/run?id=${encodeURIComponent(a.run)}`, download: `answer-${a.run}.json`, text: "Download the records" }));
  }
  const notes = (a.notes || []).length ? h("div", { class: "notes", text: a.notes.join("; ") }) : null;
  box.replaceChildren(...[answerHead(a), notes, body, a.error ? h("div", { class: "err", text: a.error }) : null, foot].filter(Boolean));
}
// While an answer streams: add the new tokens' spans instead of rebuilding the block for every token.
function growAnswer(turn, a) {
  const d = a.dom, nThink = thinkingSpan(a);
  // The thinking is a prefix of the tokens. Rebuild if a token already placed now belongs to the other part.
  if (!d || a.plain || Math.min(nThink, d.placed) !== d.inThink) { fillAnswer(turn, a, turn.answers.indexOf(a)); return; }
  for (let i = d.placed; i < a.records.length; i++) {
    const thinking = i < nThink;
    (thinking ? d.think : d.answer).append(tokenSpan(a, a.records[i], i, null));
    if (thinking) d.inThink++;
  }
  d.placed = a.records.length;
  d.fold.hidden = !nThink;
  d.fold.firstElementChild.textContent = `Thinking, ${nThink} tokens`;
  if (a.thinkOpen == null) d.fold.open = nThink === a.records.length;
  a.el.firstElementChild.replaceWith(answerHead(a));
}

function history(upTo) {
  const out = [];
  for (const t of state.turns) {
    if (t === upTo) break;
    out.push({ role: "user", content: t.user });
    const a = t.answers[t.active];
    if (a && a.text) out.push({ role: "assistant", content: a.text });
  }
  return out;
}

async function generate(turn) {
  if (state.busy) return;
  if ($("#new-seed").checked) { state.sampling.seed = Math.floor(Math.random() * 2 ** 31); buildRules(); }
  const settings = JSON.parse(JSON.stringify(state.settings)), drafting = $("#mode").value === "drafting";
  const a = { run: null, status: "running", text: "", reasoning: "", records: [], stats: null, error: null, describe: describe(settings), drafting, settings, slips: [], repeats: [], notes: [] };
  turn.answers.push(a); turn.active = turn.answers.length - 1;
  renderConversation();
  setBusy(true);
  const res = await api("/api/chat", { messages: [...history(turn), { role: "user", content: turn.user }], settings, sampling: state.sampling, drafting });
  if (res.error || !res.run) { a.status = "error"; a.error = res.error || "the request was not started"; setBusy(false); renderConversation(); return; }
  a.run = res.run; state.currentRun = res.run;
  let dirty = false;
  const redraw = () => { if (dirty) return; dirty = true; requestAnimationFrame(() => {
    dirty = false;
    const c = $("#conversation"), follow = c.scrollHeight - c.scrollTop - c.clientHeight < 240;
    if (a.status === "running" && a.records.length) growAnswer(turn, a); else fillAnswer(turn, a, turn.answers.indexOf(a));
    if (follow) c.scrollTop = c.scrollHeight;
  }); };
  const es = new EventSource(`/api/events?run=${encodeURIComponent(res.run)}`);
  es.onmessage = ev => {
    const e = JSON.parse(ev.data);
    if (e.type === "delta") { if (e.channel === "reasoning") a.reasoning += e.text; else a.text += e.text; }
    else if (e.type === "token") a.records.push(e.record);
    else if (e.type === "error") a.error = e.message;
    else if (e.type === "note") a.notes.push(e.text);
    else if (e.type === "end") {
      es.close(); a.status = e.status; a.stats = e.stats; a.slips = findSlips(a.records); a.repeats = findRepeats(a.records);
      setBusy(false); refreshAnswers();
    }
    redraw();
  };
  es.onerror = () => { if (a.status === "running") { es.close(); a.status = "error"; a.error = a.error || "the connection to the lab's server was lost"; setBusy(false); redraw(); } };
}
function setBusy(b) { state.busy = b; $("#send").disabled = b; $("#stop").hidden = !b; }

// -------------------------------------------------------------------------------------------- token inspector
function sideTab(name) {
  document.querySelectorAll("[data-side]").forEach(t => t.classList.toggle("on", t.dataset.side === name));
  document.querySelectorAll(".side-page").forEach(p => p.classList.toggle("on", p.id === "side-" + name));
}
function selectToken(a, i) {
  state.selected = { answer: a, i };
  document.querySelectorAll(".tokens .t.sel").forEach(n => n.classList.remove("sel"));
  const span = a.el && a.el.querySelector(`.tokens .t[data-i="${i}"]`);
  if (span) {
    const fold = span.closest("details");
    if (fold && !fold.open) fold.open = true;
    span.classList.add("sel"); span.scrollIntoView({ block: "nearest" });
  }
  renderToken(); sideTab("token");
}
function renderToken(showAll = false) {
  const host = $("#side-token"), { answer: a, i } = state.selected, rec = a.records[i];
  const sampled = state.showSampled && rec.samp && rec.samp.length;
  const order = [...((sampled ? rec.samp : rec.top) || [])].sort((x, y) => y[1] - x[1]);
  const context = a.records.slice(Math.max(0, i - 12), i).map(r => r.text).join("");
  const nThink = thinkingSpan(a);
  const nav = h("div", { class: "bar", style: "padding:0 0 8px" },
    h("button", { class: "ghost", text: "← previous", disabled: i === 0, onclick: () => selectToken(a, i - 1) }),
    h("button", { class: "ghost", text: "next →", disabled: i >= a.records.length - 1, onclick: () => selectToken(a, i + 1) }),
    h("span", { class: "hint", style: "padding:0", text: `token ${i + 1} of ${a.records.length}` + (nThink ? (i < nThink ? ", thinking" : ", answer") : "") }));
  const kv = h("dl", { class: "kv" });
  const add = (k, v) => kv.append(h("dt", { text: k }), h("dd", { text: v }));
  add("probability", `${fmt(praw(rec))}  (rank ${rankOf(rec)} of the candidates listed)`);
  if (rec.p_raw != null && Math.abs(rec.p_raw - rec.p) > 1e-9) add("after temperature and filters", fmt(rec.p));
  add("uncertainty", `${fmt(rec.entropy, 2)} nats`);
  add("came from", (rec.source === "draft" ? "an accepted draft" : "the model's own sample") + (rec.forced ? ", forced" : "") + `; ${rec.rows || 1} row${(rec.rows || 1) > 1 ? "s" : ""} in the pass`);
  if (rec.override) add("replaced by the engine", String(rec.override));
  add("token id", String(rec.id));
  const shown = showAll ? order : order.slice(0, 20);
  const dist = h("div");
  const outside = sampled ? rec.samp_rest : rec.rest;
  Charts.barsH(dist, shown.map(t => ({ label: JSON.stringify(t[2] != null ? t[2] : String(t[0])).slice(1, -1), value: t[1], mark: t[0] === rec.id,
    note: `${JSON.stringify(t[2])}  id ${t[0]}\nprobability ${fmt(t[1])}` + (t[0] === rec.id ? "\nthe token that was committed" : "") })),
    { title: sampled ? "What this token was drawn from" : "The model's next-token distribution here",
      note: (sampled ? `After temperature and filters; ${rec.samp_n != null ? rec.samp_n : order.length} tokens could have been drawn. ` : "At temperature 1, before filters. ") +
        `The committed token is marked. Mass outside the ${order.length} candidates listed: ${fmt(outside || 0)}.`, format: v => fmt(v), width: 370 });
  const which = rec.samp && rec.samp.length ? h("div", { class: "chart-tools" }, "Show",
    h("select", { onchange: e => { state.showSampled = e.target.value === "samp"; renderToken(showAll); } },
      h("option", { value: "raw", text: "the model's distribution", selected: !sampled }),
      h("option", { value: "samp", text: "what was sampled from", selected: !!sampled }))) : null;
  const parts = [nav, h("div", { class: "hint", style: "padding:0 0 4px", text: "…" + context }), h("div", { class: "tokbig", text: rec.text }), kv, which, dist].filter(Boolean);
  if (!showAll && order.length > 20) parts.push(h("button", { class: "link", text: `Show all ${order.length} candidates`, onclick: () => renderToken(true) }));
  if (rec.route) {
    const r = rec.route, kv2 = h("dl", { class: "kv" }), add2 = (k, v) => kv2.append(h("dt", { text: k }), h("dd", { text: v }));
    add2("picks moved", `${r.moved} of ${r.picks} (${pct(r.moved / (r.picks || 1))})`);
    add2("router mass kept", `${pct(r.mass_kept)} on average, ${pct(r.min_kept)} in the worst layer`);
    add2("layers under 0.8", String(r.rows_under_08 != null ? r.rows_under_08 : "–"));
    add2("loads, staged", `${r.loads != null ? r.loads : "–"}, ${r.staged != null ? r.staged : "–"}`);
    if (r.dropped || r.widened || r.layers_dropped) add2("three, five, layer dropped", `${r.dropped || 0}, ${r.widened || 0}, ${r.layers_dropped || 0}`);
    parts.push(h("h3", { text: "How this token's layers were routed" }), kv2);
  }
  if (rec.layers) {
    const lost = h("div"), moved = h("div"), L = rec.layers, xt = [0, 14, 28, 42, 56].map(i => ({ i, label: String(i + 1) }));
    Charts.barsV(lost, L.kept.map((v, li) => ({ x: li + 1, y: 1 - v, note: `layer ${li + 1}\nrouter mass kept ${pct(v)}\npicks moved ${L.moved[li]}, executed ${L.k ? L.k[li] : 4}` })),
      { title: "Router mass lost, by layer", max: Math.max(0.25, ...L.kept.map(v => 1 - v)), format: v => pct(v), xticks: xt, xlabel: "layer", width: 370 });
    Charts.barsV(moved, L.moved.map((v, li) => ({ x: li + 1, y: v, note: `layer ${li + 1}\npicks moved ${v} of ${L.k ? L.k[li] : 4}` })),
      { title: "Picks moved, by layer", max: 4, format: v => fmt(v, 0), xticks: xt, xlabel: "layer", height: 90, width: 370 });
    parts.push(lost, moved);
  }
  host.replaceChildren(...parts);
}

// ---------------------------------------------------------------------------------------------------- analysis
function analyse(a) {
  state.analysed = a;
  const host = $("#side-analysis"), st = a.stats || {}, rt = st.route || {}, recs = a.records;
  const tiles = h("div", { class: "tiles" });
  const tile = (k, v, title) => tiles.append(h("div", { class: "tile", title }, h("div", { class: "v", text: v }), h("div", { class: "k", text: k })));
  tile("tokens", String(st.tokens || recs.length));
  tile("perplexity", fmt(st.perplexity, 2), "exp of the mean surprise of the chosen tokens; lower means the model was surer");
  tile("mean probability", fmt(st.mean_prob, 3));
  tile("uncertainty, nats", fmt(st.entropy_mean, 2));
  tile("first choice taken", pct(st.chosen_top1));
  tile("drafted", pct(st.drafted));
  if (rt.moved_share != null) {
    tile("picks moved", pct(rt.moved_share)); tile("router mass kept", pct(rt.mass_kept_mean));
    tile("layers under 0.8, a token", fmt(rt.rows_under_08, 2)); tile("loads a token", fmt(rt.loads, 1));
    if (rt.est_ms) tile("est. ms a token", fmt(rt.est_ms, 1), "priced from the engine's pass counts; not a timing");
  }
  const jump = i => selectToken(a, Math.max(0, Math.min(recs.length - 1, i)));
  const sur = h("div");
  const nThink = thinkingSpan(a), smooth = recs.length > 120 ? 8 : 1;
  if (nThink) tile("thinking tokens", String(nThink));
  Charts.lines(sur, [{ name: "surprise", x: recs.map((_, i) => i), y: rolling(recs.map(r => -Math.log(praw(r))), smooth) },
                     { name: "uncertainty", x: recs.map((_, i) => i), y: rolling(recs.map(r => r.entropy || 0), smooth) }],
    { title: "Surprise and uncertainty along the answer", note: "In nats. Surprise is minus the log of the chosen token's probability; uncertainty is the entropy of the distribution it came from. Surprise far above uncertainty marks an unlikely choice." +
        (smooth > 1 ? ` Rolling mean over ${smooth} tokens.` : "") + " Click to inspect a token.",
      xname: "token", width: 370, height: 170, marks: nThink && nThink < recs.length ? [{ x: nThink, label: "answer" }] : [],
      xnote: i => JSON.stringify((recs[i] || {}).text || ""), onclick: jump });
  const bins = Array.from({ length: 10 }, () => 0);
  recs.forEach(r => bins[Math.min(9, Math.floor(praw(r) * 10))]++);
  const hist = h("div");
  Charts.barsV(hist, bins.map((n, b) => ({ x: b, y: n, note: `probability ${b / 10} to ${(b + 1) / 10}: ${n} tokens` })),
    { title: "How sure the model was of the tokens it chose", xticks: [0, 5, 9].map(i => ({ i, label: (i / 10).toFixed(1) })), xlabel: "probability", format: v => fmt(v, 0), width: 370 });
  const parts = [h("div", { class: "chip rule", text: a.describe }), tiles, sur, hist];
  if (recs.length && recs[0].layers) {
    const n = state.info.n_layers, matrix = Array.from({ length: n }, (_, li) => recs.map(r => Math.min(1, (1 - r.layers.kept[li]) / 0.5)));
    const heat = h("div");
    Charts.heat(heat, matrix, { title: "Router mass lost: layers down, tokens across", note: "Darkest is half the mass lost or more. Click a cell to inspect that token.",
      cell: (r, c) => `layer ${r + 1}, token ${c + 1} ${JSON.stringify(recs[c].text)}\nmass kept ${pct(recs[c].layers.kept[r])}, picks moved ${recs[c].layers.moved[r]}`, onclick: (r, c) => jump(c) });
    parts.push(heat);
  }
  const least = recs.map((r, i) => ({ i, s: -Math.log(praw(r)) })).sort((x, y) => y.s - x.s).slice(0, 8);
  parts.push(h("h3", { text: "Least likely tokens" }), h("div", { class: "list" }, least.map(x => h("div", { class: "item" },
    h("button", { class: "link mono", text: JSON.stringify(recs[x.i].text), onclick: () => jump(x.i) }),
    h("span", { class: "what", text: "…" + recs.slice(Math.max(0, x.i - 6), x.i).map(r => r.text).join("") }), h("span", { class: "meta", text: `p ${fmt(praw(recs[x.i]))}` })))));
  if (a.slips && a.slips.length) parts.push(h("h3", { text: "Doubled words" }), h("div", { class: "list" }, a.slips.map(s => h("div", { class: "item" },
    h("button", { class: "link mono", text: s.text, onclick: () => jump(s.i) }), h("span", { class: "what", text: s.context })))));
  if (a.repeats && a.repeats.length) parts.push(h("h3", { text: "Repeated phrases" }), h("div", { class: "list" }, a.repeats.map(s => h("div", { class: "item" },
    h("button", { class: "link mono", text: s.text, onclick: () => jump(s.i) }), h("span", { class: "what" }), h("span", { class: "meta", text: `${s.times} times` })))));
  host.replaceChildren(...parts);
  sideTab("analysis");
}

// ------------------------------------------------------------------------------------------------------- meter
function renderArms() {
  const host = $("#meter-arms");
  host.replaceChildren(...state.arms.map((arm, k) => h("div", { class: "arm" },
    h("input", { type: "text", value: arm.name, onchange: e => arm.name = e.target.value }),
    h("span", { class: "desc", text: describe(arm.settings) }),
    h("button", { class: "ghost", text: "Remove", onclick: () => { state.arms.splice(k, 1); renderArms(); } }))));
  if (!state.arms.length) host.append(h("p", { class: "hint", style: "padding:4px 0", text: "No rules yet. Add the current rules, a preset, or a sweep of the bonus." }));
}
const addArm = (name, settings) => { state.arms.push({ name, settings: JSON.parse(JSON.stringify(settings)) }); renderArms(); };
function toMeter(a) {
  showPage("meter");
  refreshAnswers().then(() => { $("#meter-source").value = a.run; });
  $("#meter-mode").value = a.drafting ? "drafting" : "one_row";
}
async function runMeter() {
  const source = $("#meter-source").value;
  if (!source) { alert("Choose an answer to replay. Answers appear here once they have been written in the Chat tab."); return; }
  const res = await api("/api/meter", { source, arms: state.arms, drafting: $("#meter-mode").value === "drafting", noise_floor: $("#meter-noise").checked });
  if (res.error) { alert(res.error); return; }
  pollMeter(res.id);
}
async function pollMeter(id) {
  const job = await api(`/api/meter?id=${encodeURIComponent(id)}`);
  if (job.error && !job.id) return;
  state.meterJob = job;
  const p = $("#meter-progress"), running = job.status === "running";
  p.hidden = !running;
  p.querySelector("i").style.width = `${100 * job.done / Math.max(1, job.total)}%`;
  p.querySelector("span").textContent = `${job.done} of ${job.total}: ${job.progress}`;
  $("#meter-run").disabled = running;
  renderMeter(job);
  if (running) setTimeout(() => pollMeter(id), 1000); else refreshMeters();
}
function rolling(y, w) { const out = []; let s = 0; for (let i = 0; i < y.length; i++) { s += y[i]; if (i >= w) s -= y[i - w]; out.push(s / Math.min(i + 1, w)); } return out; }

function renderMeter(job) {
  const host = $("#meter-results");
  if (!job.arms || !job.arms.length) { host.replaceChildren(job.error ? h("div", { class: "card err", text: job.error }) : ""); return; }
  const card = h("div", { class: "card" }, h("h2", { text: "Result" }),
    h("p", { class: "lede", text: `${job.tokens} tokens replayed, ${job.drafting ? "drafting" : "one row"}. Reference: the unmodified router (bonus 0). ` +
      "Divergence is the Kullback-Leibler divergence from the reference's next-token distribution, in nats. The noise floor is what two runs of the reference differ by; a rule at or below it cannot be told from the unmodified router." }));
  if (job.error) card.append(h("div", { class: "err", text: job.error }));
  if ((job.notes || []).length) card.append(h("p", { class: "notes", text: "Every run started from the saved cache state: " + job.notes.join("; ") + "." }));
  const cols = [["Rule", a => a.name], ["Mean divergence", a => fmt(a.summary.kl_mean, 4)], ["Median", a => fmt(a.summary.kl_median, 4)], ["95th pct", a => fmt(a.summary.kl_p95, 3)],
    ["Positions over 0.1", a => pct(a.summary["kl_over_0.1"])], ["Same first choice", a => pct(a.summary.top1_agreement)],
    ["Forced token's probability", a => "×" + fmt(Math.exp(a.summary.logratio_mean), 3)], ["Halved or worse", a => pct(a.summary.forced_prob_halved)],
    ["Picks moved", a => a.stats.route ? pct(a.stats.route.moved_share) : "–"], ["Mass kept", a => a.stats.route ? pct(a.stats.route.mass_kept_mean) : "–"],
    ["Est. ms a token", a => a.stats.route && a.stats.route.est_ms ? fmt(a.stats.route.est_ms, 1) : "–"]];
  const table = h("table", { class: "data" }, h("thead", {}, h("tr", {}, cols.map(c => h("th", { text: c[0] })))),
    h("tbody", {}, job.arms.map(a => h("tr", { class: a.noise_floor ? "noise" : "" }, cols.map((c, k) => h("td", { class: k ? "num" : "", text: c[1](a), title: k ? "" : (a.describe || "") }))))));
  card.append(h("div", { class: "scroll-x" }, table));
  const rules = job.arms.filter(a => !a.noise_floor), noise = job.arms.find(a => a.noise_floor);
  const bars = h("div"), linesHost = h("div"), sc = h("div");
  Charts.barsH(bars, job.arms.map(a => ({ label: a.name, value: a.summary.kl_mean, mark: !!a.noise_floor, note: `${a.name}\nmean divergence ${fmt(a.summary.kl_mean, 4)} nats\n${a.describe || ""}` })),
    { title: "Mean divergence from the unmodified router", note: noise ? "The marked bar is the noise floor." : "", labelW: 210, width: 720, format: v => fmt(v, 4) });
  const view = { w: 16, log: true };
  const drawLines = () => Charts.lines(linesHost, rules.slice(0, 8).map(a => ({ name: a.name, x: a.positions, y: rolling(a.kl, view.w) })),
    { title: "Divergence along the answer", note: `Rolling mean over ${view.w} token${view.w > 1 ? "s" : ""}` + (view.log ? ", on a logarithmic scale so that small and large divergences both show." : ".") +
        (rules.length > 8 ? " The first eight rules are drawn; the table has them all." : ""), xname: "token", format: v => fmt(v, 4), log: view.log, logFloor: 1e-4 });
  const tools = h("div", { class: "chart-tools" }, "Smooth over",
    h("select", { onchange: e => { view.w = +e.target.value; drawLines(); } }, [1, 8, 16, 32].map(w => h("option", { value: w, text: `${w} tokens`, selected: w === 16 }))),
    h("label", { class: "check", style: "margin:0" }, h("input", { type: "checkbox", checked: true, onchange: e => { view.log = e.target.checked; drawLines(); } }), "logarithmic scale"));
  drawLines();
  card.append(h("div", { class: "two" }, bars, h("div", {}, tools, linesHost)));
  const timed = rules.filter(a => a.stats.route && a.stats.route.est_ms);
  if (timed.length >= 2) {
    Charts.scatter(sc, timed.map(a => ({ label: a.name, x: a.stats.route.est_ms, y: a.summary.kl_mean, note: `${a.name}\nest. ${fmt(a.stats.route.est_ms, 1)} ms a token\nmean divergence ${fmt(a.summary.kl_mean, 4)}` })),
      { title: "Speed against fidelity", note: "Down and to the left is better. Time is priced from the engine's pass counts with the measured pass prices; it is an estimate, not a timing.", xlabel: "estimated ms a token", ylabel: "mean divergence, nats" });
    card.append(sc);
  } else card.append(h("p", { class: "hint", style: "padding:0", text: "No speed estimate: the engine reported no priced pass counts for these runs." }));
  host.replaceChildren(card);
}
async function refreshMeters() {
  const list = await api("/api/meters");
  $("#meter-list").replaceChildren(...(list.length ? list.map(j => h("div", { class: "item" },
    h("span", { class: "when", text: new Date(j.created * 1000).toLocaleString() }),
    h("button", { class: "link what", text: `${j.tokens} tokens, ${j.drafting ? "drafting" : "one row"}: ${j.arms.join("; ") || j.progress}`, onclick: () => pollMeter(j.id) }),
    h("span", { class: "meta", text: j.status }))) : [h("p", { class: "hint", style: "padding:4px 0", text: "None yet." })]));
}

// ----------------------------------------------------------------------------------------------------- answers
async function refreshAnswers() {
  const runs = await api("/api/runs");
  const chat = (Array.isArray(runs) ? runs : []).filter(r => r.kind === "chat" && r.tokens > 0);
  const src = $("#meter-source"), keep = src.value;
  src.replaceChildren(...chat.map(r => h("option", { value: r.id, text: `${new Date(r.created * 1000).toLocaleTimeString()}  ${r.tokens} tokens  [${r.describe}]  ${r.answer.replace(/\s+/g, " ").slice(0, 70)}` })));
  if (keep) src.value = keep;
  $("#answer-list").replaceChildren(...(chat.length ? chat.map(r => h("label", { class: "item" },
    h("input", { type: "checkbox", checked: state.ticked.has(r.id), onchange: e => { e.target.checked ? state.ticked.add(r.id) : state.ticked.delete(r.id); $("#compare").disabled = state.ticked.size !== 2; } }),
    h("span", { class: "when", text: new Date(r.created * 1000).toLocaleString() }),
    h("span", { class: "what", text: `${r.prompt.replace(/\s+/g, " ").slice(0, 60)}  →  ${r.answer.replace(/\s+/g, " ")}` }),
    h("span", { class: "meta", text: `${r.describe}; ${r.drafting ? "drafting" : "one row"}; ${r.tokens} tokens` }))) : [h("p", { class: "hint", style: "padding:4px 0", text: "No answers yet." })]));
}
async function compareTicked() {
  const ids = [...state.ticked];
  const [A, B, C] = await Promise.all([api(`/api/run?id=${encodeURIComponent(ids[0])}`), api(`/api/run?id=${encodeURIComponent(ids[1])}`),
    api(`/api/compare?a=${encodeURIComponent(ids[0])}&b=${encodeURIComponent(ids[1])}`)]);
  const rows = [["Rules", r => r.describe], ["Mode", r => r.drafting ? "drafting" : "one row"], ["Tokens", r => String(r.stats.tokens)],
    ["Thinking tokens", r => String(r.stats.thinking_tokens || 0)],
    ["Perplexity", r => fmt(r.stats.perplexity, 2)], ["Mean probability", r => fmt(r.stats.mean_prob, 3)], ["Uncertainty, nats", r => fmt(r.stats.entropy_mean, 2)],
    ["First choice taken", r => pct(r.stats.chosen_top1)], ["Drafted", r => pct(r.stats.drafted)],
    ["Picks moved", r => r.stats.route ? pct(r.stats.route.moved_share) : "–"], ["Router mass kept", r => r.stats.route ? pct(r.stats.route.mass_kept_mean) : "–"],
    ["Lowest mass kept", r => r.stats.route ? pct(r.stats.route.min_kept) : "–"], ["Loads a token", r => r.stats.route ? fmt(r.stats.route.loads, 1) : "–"],
    ["Doubled words", r => String(findSlips(r.records || []).length)], ["Repeated phrases", r => String(findRepeats(r.records || []).length)]];
  const table = h("table", { class: "data" }, h("thead", {}, h("tr", {}, h("th", { text: "" }), h("th", { text: "First" }), h("th", { text: "Second" }))),
    h("tbody", {}, rows.map(([k, f]) => h("tr", {}, h("td", { text: k }), h("td", { class: "num", text: f(A) }), h("td", { class: "num", text: f(B) })))));
  const card = h("div", { class: "card" }, h("h2", { text: "Comparison" }));
  const samePrompt = JSON.stringify(A.messages) === JSON.stringify(B.messages);
  let lede = samePrompt ? "" : "The two answers are to different prompts, so only their totals can be compared. ";
  if (samePrompt && !C.error) lede += C.identical ? "The two answers are the same tokens throughout." :
    `The two answers share their first ${C.shared} tokens and then choose differently.` +
    (C.shared ? ` Over the shared tokens their next-token distributions differ by ${fmt(C.summary.kl_mean, 4)} nats on average (largest ${fmt(C.summary.kl_max, 3)}).` : "");
  card.append(h("p", { class: "lede", text: lede }), h("div", { class: "scroll-x", style: "max-width:760px" }, table));
  if (samePrompt && !C.error && C.shared > 1) {
    const line = h("div");
    Charts.lines(line, [{ name: "divergence", x: C.positions, y: C.kl }], { title: "Divergence between the two answers' distributions, over the tokens they share",
      note: "Kullback-Leibler divergence of the second from the first, in nats, at each shared position.", xname: "token", height: 170, format: v => fmt(v, 4),
      xnote: i => JSON.stringify(((A.records || [])[i] || {}).text || "") });
    card.append(line);
  }
  if (samePrompt && C.first_a && C.first_b) {
    const side = (rec, name) => { const d = h("div"); const order = [...rec.top].sort((x, y) => y[1] - x[1]).slice(0, 12);
      Charts.barsH(d, order.map(t => ({ label: JSON.stringify(t[2] != null ? t[2] : String(t[0])).slice(1, -1), value: t[1], mark: t[0] === rec.id, note: `${JSON.stringify(t[2])}  probability ${fmt(t[1])}` })),
        { title: `${name}: chose ${JSON.stringify(rec.text)}`, note: "The marked candidate is the one chosen.", width: 420, format: v => fmt(v) }); return d; };
    card.append(h("h3", { text: `Where they part: token ${C.shared + 1}` }),
      h("p", { class: "hint", style: "padding:0", text: "…" + (A.records || []).slice(Math.max(0, C.shared - 14), C.shared).map(r => r.text).join("") +
        `    Divergence between the two distributions here: ${fmt(C.kl_at_split, 4)} nats. A small value means the same distribution and a different draw; a large one means the rule changed what the model wanted to say.` }),
      h("div", { class: "two" }, side(C.first_a, "First"), side(C.first_b, "Second")));
  }
  card.append(h("div", { class: "two", style: "margin-top:12px" }, h("div", {}, h("h3", { text: "First" }), h("div", { class: "plain", text: A.text })), h("div", {}, h("h3", { text: "Second" }), h("div", { class: "plain", text: B.text }))));
  $("#compare-result").replaceChildren(card);
}
function firstDifference(a, b) { const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) if (a[i].id !== b[i].id) return i; return a.length === b.length ? -1 : n; }

// -------------------------------------------------------------------------------------------------------- wire
function showPage(name) {
  document.querySelectorAll("[data-tab]").forEach(t => t.classList.toggle("on", t.dataset.tab === name));
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + name));
  if (name === "answers" || name === "meter") refreshAnswers();
  if (name === "meter") refreshMeters();
}
async function init() {
  state.info = await api("/api/info");
  loadSettings(state.info.defaults);
  state.sampling = { ...state.info.sampling_defaults };
  buildRules();
  const e = state.info.engine, st = $("#engine-status");
  st.className = "status " + (e.engine === "mock" ? "mock" : e.connected ? "ok" : "bad");
  st.textContent = e.engine === "mock" ? "mock engine: synthetic text" : !e.connected ? "engine not reachable" : e.note ? "engine connected, with limits" : "engine connected";
  st.title = e.note || e.error || e.url || "";
  if (e.engine !== "mock" && (e.note || e.error)) $("#engine-note").textContent = e.connected ? e.note : `The engine does not answer at ${e.url}. Start it with launch_lab.py start, then reload this page.`;
  await loadPresets(); renderArms(); legend(); renderConversation(); refreshAnswers();

  document.querySelectorAll("[data-tab]").forEach(t => t.addEventListener("click", () => showPage(t.dataset.tab)));
  document.querySelectorAll("[data-side]").forEach(t => t.addEventListener("click", () => sideTab(t.dataset.side)));
  $("#mode").addEventListener("change", summary);
  $("#color-by").addEventListener("change", e => { state.colorBy = e.target.value; legend(); renderConversation(); });
  $("#new-chat").addEventListener("click", () => { if (!state.busy) { state.turns = []; state.selected = null; renderConversation(); } });
  $("#composer").addEventListener("submit", ev => { ev.preventDefault(); const text = $("#prompt").value.trim(); if (!text || state.busy) return; $("#prompt").value = ""; const turn = { user: text, answers: [], active: 0 }; state.turns.push(turn); generate(turn); });
  $("#prompt").addEventListener("keydown", ev => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) $("#composer").requestSubmit(); });
  $("#stop").addEventListener("click", () => state.currentRun && api("/api/stop", { run: state.currentRun }));
  $("#preset").addEventListener("change", e => { const p = presetByName(e.target.value); if (p) loadSettings(p); e.target.value = ""; });
  $("#reset-rules").addEventListener("click", () => loadSettings(state.info.defaults));
  $("#save-preset").addEventListener("click", async () => { const name = prompt("Name for these rules"); if (name) { await api("/api/presets", { name, settings: state.settings }); loadPresets(); } });
  $("#arm-current").addEventListener("click", () => addArm(describe(state.settings), state.settings));
  $("#arm-preset").addEventListener("change", e => { const p = presetByName(e.target.value); if (p) addArm(e.target.value, p); e.target.value = ""; });
  $("#arm-sweep").addEventListener("click", () => [0.20, 0.25, 0.30, 0.40].forEach(b => addArm(`Bonus ${b.toFixed(2)}`, { ...state.settings, routing_prior_bonus: b })));
  $("#meter-run").addEventListener("click", runMeter);
  $("#compare").addEventListener("click", compareTicked);
  $("#refresh-answers").addEventListener("click", refreshAnswers);
}
init().catch(err => { $("#engine-status").className = "status bad"; $("#engine-status").textContent = "the page could not start"; console.error(err); });
