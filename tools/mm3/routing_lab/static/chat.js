// Routing lab: the conversation. Prompts, their answers (side by side when a prompt has several), streaming, and
// the conversation's memory across reloads: the page keeps the prompts and the ids of the stored answers, and
// fetches the answers' records from the lab's server when it starts again.
"use strict";

// Control tokens: the thinking tags, and MiniMax's end of turn and role markers when the record does not mark them.
const THINK_END = new Set(["</mm:think>", "</think>"]), THINK_BEGIN = new Set(["<mm:think>", "<think>"]);
const CONTROL = new Set(["[e~[", "]~!b[", "]~b]", "]!d~[", "]!p~["]);
const isTag = rec => !!rec.special || THINK_END.has((rec.text || "").trim()) || THINK_BEGIN.has((rec.text || "").trim()) || CONTROL.has(rec.text || "");

// What each rule's `changed` count means (API.md 2.4): the number of layers at which that rule changed the token.
const CHANGED = {
  bonus: ["bonus", "a substitution was made (today's rule makes them too)"],
  protect: ["protect", "the bonus would have displaced a protected pick"],
  floor: ["budget", "the budget on lost mass kept a substitution out"],
  priced: ["price", "the price kept a substitution out"],
  layer_drop: ["layer drop", "the layer's routed experts were dropped"],
  k_drop: ["drop pick", "a weak unready pick was dropped"],
  widen: ["widen", "the row ran five experts"],
  weights: ["weights", "the weights differ from today's for the same picks"],
};

// ---------------------------------------------------------------------------------------------- measures
const praw = rec => Math.max(rec.p_raw != null ? rec.p_raw : (rec.p || 0), 1e-12);
function rankOf(rec) {
  const order = [...(rec.top || [])].sort((a, b) => b[1] - a[1]);
  const k = order.findIndex(t => t[0] === rec.id);
  return k < 0 ? order.length + 1 : k + 1;
}
// How many experts ran for a token: the total over its layers, and how that differs from four a layer.
function expertsOf(rec) {
  const r = rec.route;
  if (!r || r.picks == null) return null;
  const layers = r.layer_count || state.info.n_layers, k = rec.layers && rec.layers.k;
  const count = n => k ? k.filter(x => x === n).length : null;
  return { total: r.picks, layers, perLayer: r.picks / layers, diff: r.picks - 4 * layers,
           three: k ? k.filter(x => x > 0 && x < 4).length : (r.dropped || 0), five: k ? count(5) : (r.widened || 0), none: k ? count(0) : (r.layers_dropped || 0) };
}
function expertsText(x) {
  const parts = [];
  if (x.three) parts.push(`${plural(x.three, "layer")} ran fewer than four`);
  if (x.five) parts.push(`${x.five} ran five`);
  if (x.none) parts.push(`${x.none} ran none`);
  return `${x.total} over ${x.layers} layers (${x.perLayer.toFixed(2)} a layer)` + (parts.length ? "; " + parts.join(", ") : "; four in every layer");
}
// Over an answer's tokens: for each rule, the mean number of layers it changed a token, and the share of tokens it touched.
function changedOf(records) {
  const sum = {}, touched = {};
  let n = 0;
  for (const r of records) {
    const c = r.route && r.route.changed;
    if (!c) continue;
    n++;
    for (const [k, v] of Object.entries(c)) { sum[k] = (sum[k] || 0) + v; if (v > 0) touched[k] = (touched[k] || 0) + 1; }
  }
  if (!n) return null;
  const out = {};
  for (const k of Object.keys(sum)) out[k] = { per: sum[k] / n, share: (touched[k] || 0) / n };
  return out;
}
function changedText(ch) {
  const acted = Object.entries(ch).filter(([, v]) => v.per > 0);
  if (!acted.length) return "no rule changed the routing";
  return acted.map(([k, v]) => `${(CHANGED[k] || [k])[0]} ${v.per >= 10 ? v.per.toFixed(0) : v.per.toFixed(1)}`).join(", ");
}
// Highlighting marks the tokens worth a look, not every token: below `from` a token is left plain, and from there
// to `to` it takes the seven steps of the sequential ramp. The legend states both ends.
const SHADE = {
  surprisal: { v: r => -Math.log(praw(r)), from: Math.log(2), to: Math.log(200), ends: ["probability below 0.5", "0.005"] },
  entropy: { v: r => r.entropy || 0, from: 0.5, to: 3.5, ends: ["uncertainty above 0.5 nats", "3.5"] },
  rank: { v: r => rankOf(r), from: 1, to: 10, ends: ["second choice", "tenth"] },
  lost: { v: r => (r.route && r.route.min_kept != null ? 1 - r.route.min_kept : 0), from: 0.1, to: 0.5, ends: ["a layer lost over 10%", "half"] },
  moved: { v: r => (r.route && r.route.picks ? r.route.moved / r.route.picks : 0), from: 0.2, to: 0.6, ends: ["over 20% of picks moved", "60%"] },
};
function shadeStep(rec, mode) {
  const m = SHADE[mode];
  if (!m) return 0;
  const x = m.v(rec);
  if (!(x > m.from)) return 0;
  return 1 + Math.round(6 * Math.min(1, (x - m.from) / (m.to - m.from)));
}
const colours = {};          // read once per colour change, not once per token
function colour(v) { if (!(v in colours)) colours[v] = Charts.css(v); return colours[v]; }
// A highlighted token keeps its ink; a bar under it carries the value (thicker and darker for more), so a
// highlighted paragraph still reads as text.
function paint(token, rec) {
  const mode = state.colorBy, last = token.lastElementChild;
  const span = last && last.classList.contains("tw") ? last : token;
  span.style.backgroundImage = "";
  span.classList.remove("hl");
  if (mode === "none") return;
  let bg = null, k = 0;
  if (mode === "source") { if (rec.source === "draft") { bg = colour("--seq-4"); k = 4; } }
  else if (mode === "experts") {            // two directions from four a layer: warm for fewer, blue for more
    const x = expertsOf(rec);
    if (x && x.diff) { const step = Math.abs(x.diff) <= 2 ? 1 : Math.abs(x.diff) <= 8 ? 2 : 3; bg = colour(x.diff < 0 ? `--warm-${step}` : `--seq-${step * 2 + 1}`); k = 2 + step * 2; }
  } else {
    k = shadeStep(rec, mode);
    if (k > 0) bg = colour(`--seq-${k}`);
  }
  if (bg) {
    span.classList.add("hl");
    span.style.backgroundImage = `linear-gradient(${bg}, ${bg})`;
    span.style.backgroundSize = `100% ${k >= 5 ? 4 : k >= 3 ? 3 : 2}px`;
  }
}
function repaint() {
  for (const k of Object.keys(colours)) delete colours[k];
  for (const turn of state.turns) for (const a of turn.answers) {
    if (!a.el) continue;
    a.el.querySelectorAll(".t").forEach(span => { const rec = a.records[+span.dataset.i]; if (rec && !span.classList.contains("tag") && /\S/.test(span.textContent)) paint(span, rec); });
  }
}
function legend() {
  const host = $("#color-legend"), mode = state.colorBy;
  const ramp = vars => h("span", { class: "ramp", "aria-hidden": "true" }, vars.map(v => h("i", { style: `background:${Charts.css(v)}` })));
  host.replaceChildren();
  if (mode === "none") return;
  if (mode === "source") { host.append(ramp(["--seq-4"]), "an accepted draft"); return; }
  if (mode === "experts") { host.append("fewer", ramp(["--warm-3", "--warm-2", "--warm-1"]), "or more", ramp(["--seq-3", "--seq-5", "--seq-7"]), "than four a layer"); return; }
  const ends = SHADE[mode].ends;
  host.append(ends[0], ramp([1, 2, 3, 4, 5, 6, 7].map(k => "--seq-" + k)), ends[1]);
}

// Doubled words ("the the"): a cheap flag for the kind of damage a high bonus produced in the usability blocks.
const prose = r => (isTag(r) ? " " : (r.text || ""));
function findSlips(records) {
  let text = "";
  const spans = [];
  records.forEach((r, i) => { const t = prose(r); spans.push([text.length, text.length + t.length, i]); text += t; });
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
  let text = "";
  const spans = [];
  records.forEach((r, i) => { spans.push([text.length, i]); text += prose(r); });
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
    if (at.length < 2 || at[1] <= covered) continue;
    covered = at[1] + key.length;
    let tok = 0;
    for (const sp of spans) { if (sp[0] <= at[1]) tok = sp[1]; else break; }
    out.push({ i: tok, text: key, times: at.length });
  }
  return out.sort((a, b) => b.times - a.times).slice(0, 12);
}
// How many leading tokens are the model's thinking (its closing tag included). When thinking is on, the opening tag
// is part of the prompt, so until the closing tag arrives the server's reasoning stream is what says so.
function thinkingSpan(a) {
  let end = -1, begun = false;
  a.records.forEach((r, k) => { const t = (r.text || "").trim(); if (THINK_END.has(t)) end = k; else if (THINK_BEGIN.has(t)) begun = true; });
  return end >= 0 ? end + 1 : (begun || a.reasoning ? a.records.length : 0);
}

// ---------------------------------------------------------------------------------------------- memory
let saveTimer = 0;
function saveConversation() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store.set("conversation", pack(state.turns)), 150);
}
function pack(turns) {
  return { seq: state.seq, turns: turns.map(t => ({ id: t.id, user: t.user, active: t.active, collapsed: !!t.collapsed,
    answers: t.answers.map(a => ({ key: a.key, run: a.run, describe: a.describe, drafting: a.drafting, cap: a.cap, thinkOpen: a.thinkOpen,
      status: a.run ? undefined : a.status, error: a.run ? undefined : a.error, text: a.run ? undefined : a.text })) })) };
}
function newAnswer(props) {
  return Object.assign({ key: "a" + (++state.seq), run: null, status: "running", text: "", reasoning: "", records: [], stats: null, error: null,
    describe: "", drafting: false, settings: null, slips: [], repeats: [], notes: [], cap: 0 }, props);
}
function fromRun(a, r) {
  Object.assign(a, { run: r.id, status: r.status, text: r.text || "", reasoning: r.reasoning || "", records: r.records || [], stats: r.stats || null,
    error: r.error, describe: r.describe, drafting: !!r.drafting, settings: r.settings, notes: r.notes || [], cap: (r.sampling || {}).max_tokens || 0, loading: false });
  a.slips = findSlips(a.records); a.repeats = findRepeats(a.records);
  return a;
}
// The conversation kept from the last visit: prompts at once, answers as their records arrive.
async function restoreConversation(key = "conversation") {
  const got = store.get(key, null);
  if (!got || !Array.isArray(got.turns)) return;
  state.seq = Math.max(state.seq, got.seq || 0);
  state.turns = got.turns.map(t => ({ id: t.id || "t" + (++state.seq), user: String(t.user || ""), active: t.active || 0, collapsed: !!t.collapsed,
    answers: (t.answers || []).map(x => newAnswer({ key: x.key || "a" + (++state.seq), run: x.run || null, describe: x.describe || "", drafting: !!x.drafting, cap: x.cap || 0,
      thinkOpen: x.thinkOpen,
      // Without a run id an answer is only what was kept here: an error, or one cut off by a reload before it started.
      status: x.run ? "loading" : (["done", "stopped", "error"].includes(x.status) ? x.status : "error"),
      error: x.error || (x.run || ["done", "stopped"].includes(x.status) ? null : "This answer was not started: the page was reloaded before the lab's server took it."),
      text: x.text || "", loading: !!x.run })) }));
  renderConversation({ bottom: true });
  const stick = () => { const c = $("#conversation"); return c.scrollHeight - c.scrollTop - c.clientHeight < 40; };
  // All at once, newest first: an answer still being written is picked up while it is being written.
  const jobs = [];
  for (const turn of [...state.turns].reverse()) for (const a of [...turn.answers].reverse()) {
    if (!a.run) continue;
    jobs.push(api(`/api/run?id=${encodeURIComponent(a.run)}`).then(r => {
      if (r.error) { a.loading = false; a.status = "error"; a.error = `This answer is no longer stored on the lab's server (${r.error}).`; }
      else if (r.status === "running") { a.loading = false; a.status = "running"; a.describe = r.describe; a.drafting = !!r.drafting; attach(turn, a, r.id); return; }
      else fromRun(a, r);
      const bottom = stick();
      keepPlace($("#conversation"), () => fillAnswer(turn, a));
      if (bottom) $("#conversation").scrollTop = $("#conversation").scrollHeight;
    }));
  }
  await Promise.all(jobs);
}
function newConversation() {
  if (state.busy) { toast("Wait for the answer to finish, or stop it first."); return; }
  if (state.turns.length) store.set("previous", pack(state.turns));
  state.turns = []; state.selected = null;
  renderConversation();
  saveConversation();
  $("#prompt").focus();
}

// ---------------------------------------------------------------------------------------------- rendering
function renderConversation(o = {}) {
  const host = $("#conversation");
  const build = () => {
    host.replaceChildren();
    if (!state.turns.length) {
      const prev = store.get("previous", null);
      host.append(h("div", { class: "empty" },
        h("h2", { text: "Judge a routing rule by reading what it writes" }),
        h("p", { text: "Set the rule in the panel, write a prompt below and send it. Change the rule and press Rerun: the new answer appears beside the last one, so the two can be read side by side." }),
        h("p", { class: "muted", text: "Click any word of an answer to see the distribution it came from, how many experts ran for it and what each rule changed." }),
        prev && prev.turns && prev.turns.length ? h("p", {}, h("button", { type: "button", class: "ghost", text: `Bring back the previous conversation (${plural(prev.turns.length, "prompt")})`,
          onclick: () => { store.set("conversation", prev); store.set("previous", null); restoreConversation(); } })) : null));
    }
    state.turns.forEach((turn, n) => host.append(turnBlock(turn, n)));
  };
  if (o.bottom) { build(); host.scrollTop = host.scrollHeight; }
  else keepPlace(host, build);
  updateRerunLast();
}
function turnBlock(turn, n) {
  const many = turn.answers.length > 1;
  const box = h("section", { class: "turn", dataset: { key: turn.id } });
  turn.el = box;
  const text = h("div", { class: "prompt-text", text: turn.user });
  const long = turn.user.length > 600 || turn.user.split("\n").length > 8;
  if (long) text.classList.add("clamped");
  const tools = h("span", { class: "tools" },
    h("button", { type: "button", class: "tbtn rerun", disabled: state.busy, title: "Answer this prompt again under the rules as they are set now, with the same seed: differences come from the rules",
      text: "Rerun", onclick: () => generate(turn) }),
    h("button", { type: "button", class: "tbtn rerun", disabled: state.busy, title: "Answer this prompt again under the current rules with a new random seed: another sample of the same rule",
      text: "Rerun, new seed", onclick: () => generate(turn, { newSeed: true }) }),
    h("button", { type: "button", class: "tbtn", title: "Put this prompt in the box below to change it", text: "Edit",
      onclick: () => { const b = $("#prompt"); b.value = turn.user; autosize(); b.focus(); saveDraft(); } }),
    !turn.answers.length ? h("button", { type: "button", class: "tbtn", text: "Remove", title: "Remove this prompt from the conversation",
      onclick: () => { state.turns.splice(state.turns.indexOf(turn), 1); renderConversation(); saveConversation(); updateRerunLast(); } }) : null);
  box.append(h("div", { class: "prompt" },
    h("div", { class: "p-head" }, h("span", { class: "eyebrow", text: `Prompt ${n + 1}` }),
      many ? h("span", { class: "p-count", text: `${turn.answers.length} answers` }) : null, h("span", { class: "grow" }), tools),
    text, long ? h("button", { type: "button", class: "link small", text: "Show all of the prompt", onclick: e => { const on = text.classList.toggle("clamped"); e.target.textContent = on ? "Show all of the prompt" : "Show less"; } }) : null));
  const answers = h("div", { class: "answers" + (many && state.layout === "side" ? " cols" : "") });
  if (!turn.answers.length) answers.append(h("p", { class: "muted small", text: "No answer kept. Rerun to answer this prompt." }));
  turn.answers.forEach(a => { a.el = h("article", { class: "answer", dataset: { key: a.key } }); answers.append(a.el); fillAnswer(turn, a); });
  box.append(answers);
  return box;
}

const ruleLabel = d => (d || "rules unknown").split(", ").join(" · ");
function setActive(turn, k) {
  turn.active = k;
  keepPlace($("#conversation"), () => turn.answers.forEach(x => x.el && fillAnswer(turn, x)));
  saveConversation();
}
// The header says, at a glance: which rule, whether the answer is whole, how much was rerouted and which rules
// acted. Everything else is one click away (Details here, or the Analysis panel).
function answerHead(turn, a) {
  const k = turn.answers.indexOf(a), many = turn.answers.length > 1, st = a.stats || {}, rt = st.route || {};
  const n = a.records.length || st.tokens || 0, settled = a.status !== "running" && a.status !== "loading";
  const head = h("header", { class: "a-head" });
  const top = h("div", { class: "a-top" },
    many ? h("span", { class: "nth", title: `Answer ${k + 1} of ${turn.answers.length}`, text: String(k + 1) }) : null,
    h("span", { class: "a-rule", title: "The rules this answer was written under", text: ruleLabel(a.describe) }));
  const tools = h("span", { class: "tools" });
  top.append(tools);
  if (a.status === "running") tools.append(h("button", { type: "button", class: "tbtn", text: "Stop", title: "Stop this answer (Escape)", onclick: stopRun }));
  else if (settled) {
    if (a.records.length) tools.append(h("button", { type: "button", class: "tbtn", text: "Analysis", title: "Charts and lists for this answer, in the panel", onclick: () => analyse(a) }));
    tools.append(menuButton("More", () => [
      a.records.length ? { text: "Measure fidelity", title: "Replay this answer under several rules on the Fidelity meter page", onclick: () => toMeter(a) } : null,
      a.text ? { text: "Copy the text", onclick: () => copyText(a.text) } : null,
      a.run && a.records.length ? { text: "Download the records", title: "Every token record, as JSON", href: `/api/run?id=${encodeURIComponent(a.run)}`, download: `answer-${a.run}.json` } : null,
      many && turn.active !== k ? { text: "Continue the conversation from this answer", onclick: () => setActive(turn, k) } : null,
      "-",
      { text: "Remove from the conversation", title: "It stays stored and listed under Answers", danger: true, onclick: () => removeAnswer(turn, a) }], { title: "More for this answer" }));
  }
  head.append(top);
  const facts = h("div", { class: "a-facts" });
  const fact = (text, cls, title, onclick) => facts.append(h(onclick ? "button" : "span", { type: onclick ? "button" : null, class: "fact" + (cls ? " " + cls : ""), title: title || null, text, onclick: onclick || null }));
  const nThink = thinkingSpan(a);
  fact(a.drafting ? "drafting" : "one row", "muted");
  if (a.status === "running") fact(n ? `Writing, ${plural(n, "token")}` + (nThink === n ? ", still thinking" : "") : "Waiting for the first token", "live");
  if (a.status === "loading") fact("Loading the records", "muted");
  if (a.status === "stopped") fact(`Stopped after ${plural(n, "token")}`, "warn");
  if (a.status === "error" && !a.records.length) fact("Not answered", "bad");
  if (st.finish === "length") fact(`Cut at the cap of ${a.cap || "?"} tokens`, "warn", "The answer did not end by itself: it reached the cap on generated tokens, which counts the thinking too. Raise or switch off the cap under Sampling and rerun.");
  if (settled && a.records.length) {
    if (rt.moved_share != null) fact(`${pct(rt.moved_share)} rerouted`, "", "Share of expert picks moved off the router's own top four");
    const ch = a.changed || (a.changed = changedOf(a.records));
    if (ch) {
      const acted = Object.entries(ch).filter(([, v]) => v.per > 0).sort((x, y) => y[1].per - x[1].per);
      fact(acted.length ? "acted: " + acted.map(([r]) => (CHANGED[r] || [r])[0]).join(", ") : "no rule acted", "",
        "Layers a token at which each rule changed the routing (of 57): " + changedText(ch));
    }
    if (rt.experts_per_layer != null && Math.abs(rt.experts_per_layer - 4) > 0.004)
      fact(`${rt.experts_per_layer.toFixed(2)} experts a layer`, "", "Experts run per layer, averaged over the tokens. The router's number is four. Highlight by Experts to see which tokens differ.");
    if (a.repeats.length) fact(plural(a.repeats.length, "repeated phrase"), "warn", "List them in the Analysis panel", () => analyse(a));
    if (a.slips.length) fact(plural(a.slips.length, "doubled word"), "warn", "Show the first one", () => selectToken(a, a.slips[0].i));
    if (st.unreadable_rows) fact(plural(st.unreadable_rows, "unreadable row"), "warn", "Tokens whose distribution the engine could not read; they are left out of the statistics");
    if (many && turn.active === k) fact("the next prompt builds on this", "accent", "Choose another answer's More menu to continue from it instead");
    facts.append(h("button", { type: "button", class: "fact-more", "aria-expanded": String(!!a.details), text: a.details ? "Fewer details" : "Details",
      onclick: () => { a.details = !a.details; keepPlace($("#conversation"), () => fillAnswer(turn, a)); } }));
  }
  if (facts.childNodes.length) head.append(facts);
  if (a.details && settled && a.records.length) head.append(detailsOf(a));
  return head;
}
// Everything about an answer that is not needed at a glance.
function detailsOf(a) {
  const st = a.stats || {}, rt = st.route || {}, pr = st.prompt_route, dl = h("dl", { class: "a-details" });
  const add = (k, v, title) => { if (v != null) dl.append(h("dt", { text: k, title: title || null }), h("dd", { text: v })); };
  const nThink = thinkingSpan(a);
  add("tokens", `${a.records.length}` + (nThink ? `, ${nThink} of them thinking` : ""));
  if (rt.experts_per_layer != null) add("experts a layer", rt.experts_per_layer.toFixed(2), "the router's number is four");
  if (rt.moved_share != null) add("picks moved", pct(rt.moved_share));
  if (rt.mass_kept_mean != null) add("router mass kept", `${pct(rt.mass_kept_mean)} on average, ${pct(rt.min_kept)} at the lowest`);
  const ch = a.changed;
  if (ch) add("rules acted, layers a token", changedText(ch), "the mean number of layers (of 57) at which each rule changed a token");
  if (pr) add("the prompt's rows", `${pct(pr.moved_share)} moved, ${pct(pr.mass_kept_mean)} mass kept`);
  add("perplexity", st.perplexity ? fmt(st.perplexity, 2) : null);
  add("first choice taken", st.chosen_top1 != null ? pct(st.chosen_top1) : null);
  if (rt.loads != null) add("loads a token", fmt(rt.loads, 1));
  const sp = speedOf(st);
  if (sp) add(sp.name, fmt(sp.v, 1), sp.what + "; an estimate for ordering rules, not a timing");
  const notes = (a.notes || []).filter(x => !x.startsWith("ROUTING CHECK FAILED"));
  if (notes.length) add("notes", notes.join("; "));
  return dl;
}
function copyText(text) {
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => toast("Copied."), () => toast("The browser did not allow copying.", "bad"));
}
function removeAnswer(turn, a) {
  const k = turn.answers.indexOf(a);
  if (k < 0 || a.status === "running") return;
  turn.answers.splice(k, 1);
  if (turn.active >= turn.answers.length || turn.active > k) turn.active = Math.max(0, turn.active - (turn.active >= k ? 1 : 0));
  if (state.selected && state.selected.answer === a) state.selected = null;
  keepPlace($("#conversation"), () => turn.el.replaceWith(turnBlock(turn, state.turns.indexOf(turn))));
  saveConversation();
}

function tokenSpan(a, i, text, slipAt) {
  const rec = a.records[i], tag = isTag(rec), t = text != null ? text : (rec.text || "");
  const span = h("span", { class: "t" + (tag ? " tag" : "") + (slipAt && slipAt.has(i) ? " slip" : "") +
    (state.selected && state.selected.answer === a && state.selected.i === i ? " sel" : "") });
  span.dataset.i = i;
  // A token's leading space stays outside its highlight bar, so the bar sits under the word alone.
  const lead = tag ? null : /^\s+(?=\S)/.exec(t);
  if (lead) span.append(lead[0], h("span", { class: "tw", text: t.slice(lead[0].length) }));
  else span.textContent = t;
  if (!tag && /\S/.test(t)) paint(span, rec);
  return span;
}
// The text of tokens [from, to): formatted (Markdown laid out) or every token as it was written.
function textBlock(a, from, to, slipAt, cls) {
  const host = h("div", { class: cls });
  if (state.view === "formatted") {
    host.classList.add("formatted");
    host.append(Formatted.render(a.records, from, to, { hidden: isTag, span: (i, text) => tokenSpan(a, i, text, slipAt) }));
  } else {
    host.classList.add("tokens");
    for (let i = from; i < to; i++) host.append(tokenSpan(a, i, null, slipAt));
  }
  return host;
}
const skeleton = n => h("div", { class: "skeleton", "aria-hidden": "true" }, Array.from({ length: n }, (_, i) => h("i", { style: `width:${[92, 100, 76, 88, 60][i % 5]}%` })));
function bodyOf(turn, a) {
  if (a.status === "loading") return skeleton(4);
  if (a.status === "error" && !a.records.length) {
    return h("div", { class: "refused" }, h("p", { text: a.error || "The lab's server did not start this answer." }),
      h("button", { type: "button", class: "ghost small rerun", disabled: state.busy, text: "Try again under the current rules", onclick: () => { removeAnswer(turn, a); generate(turn); } }));
  }
  if (!a.records.length) {
    return h("div", { class: "body" },
      a.reasoning ? h("details", { class: "thinking" }, h("summary", {}, h("span", { class: "fold-title", text: "Thinking" })), h("div", { class: "plain thinking-text", text: a.reasoning })) : null,
      a.text ? h("div", { class: "plain", text: a.text }) : (a.status === "running" ? skeleton(2) : null));
  }
  const nThink = thinkingSpan(a), slipAt = new Set((a.slips || []).map(s => s.i));
  const body = h("div", { class: "body" });
  if (nThink) {
    const still = nThink === a.records.length && a.status === "running";
    const fold = h("details", { class: "thinking" },
      h("summary", {}, h("span", { class: "fold-title", text: still ? "Thinking" : "Thought" }), h("span", { class: "fold-n", text: plural(nThink, "token") })),
      textBlock(a, 0, nThink, slipAt, "thinking-text"));
    const open = a.thinkOpen != null ? a.thinkOpen : still;
    if (open) { fold.dataset.auto = "1"; fold.open = true; }          // opening it here fires "toggle" too: not the reader's choice
    // A click is the reader's choice and is kept; opening it for them while the thinking streams is not.
    fold.addEventListener("toggle", () => { if (fold.dataset.auto !== "1") { a.thinkOpen = fold.open; saveConversation(); } fold.dataset.auto = ""; });
    body.append(fold);
  }
  if (nThink < a.records.length) body.append(textBlock(a, nThink, a.records.length, slipAt, "answer-text"));
  if ((a.stats || {}).finish === "length") body.append(h("div", { class: "cut", text: `Cut here: the answer reached the cap of ${a.cap ? plural(a.cap, "generated token") : "generated tokens"} before it ended.` }));
  if (a.status === "stopped") body.append(h("div", { class: "cut", text: "Stopped here." }));
  return body;
}
function fillAnswer(turn, a) {
  const box = a.el;
  if (!box) return;
  const k = turn.answers.indexOf(a), many = turn.answers.length > 1;
  box.className = "answer" + (many && turn.active === k ? " active" : "") + (a.status === "running" ? " running" : "") + (a.status === "error" && !a.records.length ? " failed" : "");
  const failed = (a.notes || []).filter(n => n.startsWith("ROUTING CHECK FAILED"));
  box.replaceChildren(...[answerHead(turn, a),
    failed.length ? h("div", { class: "alarm", text: failed.join("; ") }) : null,
    bodyOf(turn, a),
    a.error && a.records.length ? h("div", { class: "err small", text: a.error }) : null].filter(Boolean));
}

// ---------------------------------------------------------------------------------------------- generation
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
async function generate(turn, opts = {}) {
  if (state.busy) return;
  if (state.newSeed || opts.newSeed) { state.sampling.seed = Math.floor(Math.random() * 2 ** 31); const f = document.querySelector('#rules .field[data-key="seed"]'); if (f && f._show) f._show(); rulesChanged(); }
  const settings = clone(state.settings), drafting = state.mode === "drafting";
  const a = newAnswer({ describe: describe(settings), drafting, settings, cap: state.sampling.max_tokens || 0 });
  turn.answers.push(a); turn.active = turn.answers.length - 1;
  setBusy(true);
  const fresh = !turn.el || !turn.el.isConnected;
  if (fresh) renderConversation({ bottom: true });
  else keepPlace($("#conversation"), () => turn.el.replaceWith(turnBlock(turn, state.turns.indexOf(turn))));
  a.el.scrollIntoView({ block: "nearest" });
  const res = await api("/api/chat", { messages: [...history(turn), { role: "user", content: turn.user }], settings, sampling: state.sampling, drafting });
  if (res.error || !res.run) { a.status = "error"; a.error = res.error || "the request was not started"; setBusy(false); fillAnswer(turn, a); saveConversation(); return; }
  a.run = res.run;
  saveConversation();
  attach(turn, a, res.run);
}
// Follow a run's events from its first: used for a new answer and for one still being written after a reload.
function attach(turn, a, run) {
  state.currentRun = run;
  a.records = []; a.text = ""; a.reasoning = ""; a.notes = [];
  setBusy(true);
  let timer = 0, lastDraw = 0;
  const draw = () => {
    timer = 0; lastDraw = performance.now();
    const c = $("#conversation"), follow = c.scrollHeight - c.scrollTop - c.clientHeight < 160;
    fillAnswer(turn, a);
    if (follow && a.status === "running") c.scrollTop = c.scrollHeight;
  };
  // At most one redraw every 150 ms while tokens stream: the formatted view is rebuilt as a whole.
  const redraw = () => { if (!timer) timer = setTimeout(draw, Math.max(0, 150 - (performance.now() - lastDraw))); };
  const es = new EventSource(`/api/events?run=${encodeURIComponent(run)}`);
  es.onmessage = ev => {
    const e = JSON.parse(ev.data);
    if (e.type === "delta") { if (e.channel === "reasoning") a.reasoning += e.text; else a.text += e.text; }
    else if (e.type === "token") a.records.push(e.record);
    else if (e.type === "error") a.error = e.message;
    else if (e.type === "note") a.notes.push(e.text);
    else if (e.type === "end") {
      es.close(); a.status = e.status; a.stats = e.stats; a.slips = findSlips(a.records); a.repeats = findRepeats(a.records); a.changed = null;
      setBusy(false); saveConversation(); refreshAnswers();
      clearTimeout(timer); draw();
      if (state.selected && state.selected.answer === a) renderToken();
      return;
    }
    redraw();
  };
  es.onerror = () => {
    if (a.status !== "running") return;
    es.close(); a.status = "error"; a.error = a.error || "The connection to the lab's server was lost. Reload the page: an answer still being written is picked up again.";
    setBusy(false); clearTimeout(timer); draw();
  };
}
function setBusy(b) {
  state.busy = b;
  const send = $("#send");
  send.textContent = b ? "Stop" : "Send";
  send.classList.toggle("stop", b);
  send.title = b ? "Stop the answer being written (Escape)" : "Send the prompt (Ctrl+Enter)";
  $$("button.rerun").forEach(x => { x.disabled = b; });
  updateRerunLast();
}
function stopRun() { if (state.currentRun) api("/api/stop", { run: state.currentRun }); }
// The composer's Rerun: the last prompt again, under the rules shown beside it.
function updateRerunLast() {
  const btn = $("#rerun-last"), last = state.turns[state.turns.length - 1];
  btn.disabled = state.busy || !last;
  btn.textContent = last && state.turns.length > 1 ? `Rerun prompt ${state.turns.length}` : "Rerun";
}

// One set of listeners for every token on the page: click to inspect, hover for a short readout.
function answerOf(node) {
  const el = node && node.closest("article.answer");
  if (!el) return null;
  for (const t of state.turns) for (const a of t.answers) if (a.key === el.dataset.key) return a;
  return null;
}
function wireConversation() {
  const c = $("#conversation");
  c.addEventListener("click", ev => {
    const t = ev.target.closest(".t");
    if (!t || window.getSelection().toString()) return;          // selecting text to copy is not a click on a token
    const a = answerOf(t);
    if (a) selectToken(a, +t.dataset.i, { noScroll: true });
  });
  c.addEventListener("mousemove", ev => {
    const t = ev.target.closest(".t"), a = t && answerOf(t), rec = a && a.records[+t.dataset.i];
    if (!rec) { Charts.tip(null); return; }
    const ex = expertsOf(rec);
    Charts.tip(`${JSON.stringify(rec.text)}  p ${fmt(praw(rec))}  rank ${rankOf(rec)}` +
      (ex ? `\n${ex.total} experts ran (${ex.perLayer.toFixed(2)} a layer), ${rec.route.moved} moved, lowest mass kept ${pct(rec.route.min_kept)}` : "") +
      "\nclick to inspect", ev);
  });
  c.addEventListener("mouseleave", () => Charts.tip(null));
  c.addEventListener("scroll", () => Charts.tip(null), { passive: true });
}

// ---------------------------------------------------------------------------------------------- composer
function autosize() {
  const t = $("#prompt");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight + 2, Math.max(60, Math.min(260, window.innerHeight * 0.26))) + "px";
}
let draftTimer = 0;
function saveDraft() { clearTimeout(draftTimer); draftTimer = setTimeout(() => store.set("draft", $("#prompt").value), 300); }
function send() {
  const text = $("#prompt").value.trim();
  if (!text || state.busy) return;
  $("#prompt").value = ""; autosize(); store.set("draft", "");
  const turn = { id: "t" + (++state.seq), user: text, answers: [], active: 0 };
  state.turns.push(turn);
  turn.el = null;
  generate(turn);
}
