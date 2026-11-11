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
function tokenValue(rec, mode) {
  const r = rec.route || {};
  switch (mode) {
    case "surprisal": return Math.min(1, -Math.log(praw(rec)) / 6);
    case "entropy": return Math.min(1, (rec.entropy || 0) / 4);
    case "rank": return Math.min(1, (rankOf(rec) - 1) / 9);
    case "lost": return r.min_kept == null ? 0 : Math.min(1, (1 - r.min_kept) / 0.5);
    case "moved": return r.picks ? Math.min(1, (r.moved / r.picks) / 0.5) : 0;
    default: return 0;
  }
}
function ink(hex) {          // black or white, whichever reads on this background
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return "";
  const [r, g, b] = m.slice(1).map(x => parseInt(x, 16) / 255).map(c => c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.32 ? "#0b0b0b" : "#ffffff";
}
const colours = {};          // read once per colour change, not once per token
function colour(v) { if (!(v in colours)) colours[v] = Charts.css(v); return colours[v]; }
function paint(span, rec) {
  const mode = state.colorBy;
  span.style.background = span.style.color = "";
  if (mode === "none") return;
  let bg = null;
  if (mode === "source") { if (rec.source === "draft") bg = colour("--seq-2"); }
  else if (mode === "experts") {            // two directions from four a layer: warm for fewer, blue for more
    const x = expertsOf(rec);
    if (x && x.diff) { const step = Math.abs(x.diff) <= 2 ? 1 : Math.abs(x.diff) <= 8 ? 2 : 3; bg = colour(x.diff < 0 ? `--warm-${step}` : `--seq-${step * 2}`); }
  } else {
    const k = Math.max(0, Math.min(7, Math.round(tokenValue(rec, mode) * 7)));
    if (k > 0) bg = colour(`--seq-${k}`);
  }
  if (bg) { span.style.background = bg; span.style.color = ink(bg); }
}
function repaint() {
  for (const k of Object.keys(colours)) delete colours[k];
  for (const turn of state.turns) for (const a of turn.answers) {
    if (!a.el) continue;
    a.el.querySelectorAll(".t").forEach(span => { const rec = a.records[+span.dataset.i]; if (rec && !span.classList.contains("tag")) paint(span, rec); });
  }
}
function legend() {
  const host = $("#color-legend"), mode = state.colorBy;
  const ramp = vars => h("span", { class: "ramp", "aria-hidden": "true" }, vars.map(v => h("i", { style: `background:${Charts.css(v)}` })));
  host.replaceChildren();
  if (mode === "none") return;
  if (mode === "source") { host.append(ramp(["--seq-2"]), "an accepted draft"); return; }
  if (mode === "experts") { host.append("fewer", ramp(["--warm-3", "--warm-2", "--warm-1", "--seq-0", "--seq-2", "--seq-4", "--seq-6"]), "more than four a layer"); return; }
  const ends = { surprisal: ["likely", "unlikely"], entropy: ["certain", "uncertain"], rank: ["first choice", "tenth or lower"],
                 lost: ["all mass kept", "half lost in a layer"], moved: ["no pick moved", "half moved"] }[mode];
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
      thinkOpen: x.thinkOpen, status: x.run ? "loading" : (x.status || "error"), error: x.error || null, text: x.text || "", loading: !!x.run })) }));
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
        h("p", { text: "Set the rules on the right, then write a prompt below. Each answer keeps the rules it was made with." }),
        h("p", { text: "Rerun a prompt under other rules and its answers sit side by side. Click any token to see its distribution and how its layers were routed." }),
        prev && prev.turns && prev.turns.length ? h("p", {}, h("button", { type: "button", class: "ghost", text: `Bring back the previous conversation (${plural(prev.turns.length, "prompt")})`,
          onclick: () => { store.set("conversation", prev); store.set("previous", null); restoreConversation(); } })) : null));
    }
    state.turns.forEach((turn, n) => host.append(turnBlock(turn, n)));
  };
  if (o.bottom) { build(); host.scrollTop = host.scrollHeight; }
  else keepPlace(host, build);
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
      onclick: () => { state.turns.splice(state.turns.indexOf(turn), 1); renderConversation(); saveConversation(); } }) : null);
  box.append(h("div", { class: "prompt" },
    h("div", { class: "prompt-head" }, h("span", { class: "who", text: `Prompt ${n + 1}` }),
      many ? h("span", { class: "muted small", text: `${turn.answers.length} answers` }) : null, tools),
    text, long ? h("button", { type: "button", class: "link small", text: "Show all of the prompt", onclick: e => { const on = text.classList.toggle("clamped"); e.target.textContent = on ? "Show all of the prompt" : "Show less"; } }) : null));
  const answers = h("div", { class: "answers" + (many && state.layout === "side" ? " cols" : "") });
  if (!turn.answers.length) answers.append(h("p", { class: "muted", text: "No answer kept. Rerun to answer this prompt." }));
  turn.answers.forEach(a => { a.el = h("article", { class: "answer", dataset: { key: a.key } }); answers.append(a.el); fillAnswer(turn, a); });
  box.append(answers);
  return box;
}

function metric(text, title) { return h("span", { class: "m", title: title || null, text }); }
function answerHead(turn, a) {
  const k = turn.answers.indexOf(a), many = turn.answers.length > 1, st = a.stats || {}, rt = st.route || {};
  const n = a.records.length || st.tokens || 0;
  const head = h("header", { class: "answer-head" });
  const line = h("div", { class: "line" },
    many ? h("span", { class: "nth", text: `${k + 1}` }) : null,
    h("span", { class: "chip rule", title: "The rules this answer was written under", text: a.describe || "rules unknown" }),
    h("span", { class: "chip", text: a.drafting ? "drafting" : "one row" }),
    a.status === "running" ? h("span", { class: "chip live", text: "writing" }) : null,
    a.status === "loading" ? h("span", { class: "chip", text: "loading" }) : null,
    a.status === "stopped" ? h("span", { class: "chip warn", text: "stopped" }) : null,
    st.finish === "length" ? h("span", { class: "chip warn", title: "The answer did not end by itself: it reached the cap on generated tokens, which counts the thinking too. Raise or switch off the cap under Sampling and rerun.",
      text: "cut at the token cap" }) : null);
  head.append(line);
  if (a.status !== "running" && a.status !== "loading" && a.records.length) {
    line.append(h("span", { class: "tools" },
      h("button", { type: "button", class: "tbtn", text: "Analysis", title: "Charts and lists for this answer, in the panel on the right", onclick: () => analyse(a) }),
      h("button", { type: "button", class: "tbtn", text: "Fidelity", title: "Replay this answer under several rules on the Fidelity meter page", onclick: () => toMeter(a) }),
      h("button", { type: "button", class: "tbtn", text: "Copy", title: "Copy the answer's text", onclick: () => copyText(a.text) }),
      a.run ? h("a", { class: "tbtn", href: `/api/run?id=${encodeURIComponent(a.run)}`, download: `answer-${a.run}.json`, title: "Download every token record as JSON", text: "Records" }) : null));
  }
  if (a.status !== "running") head.prepend(h("button", { type: "button", class: "chip-x close", text: "×", "aria-label": "Remove this answer from the conversation",
    title: "Take this answer out of the conversation. It stays stored and listed under Answers.", onclick: () => removeAnswer(turn, a) }));
  if (n) {
    const nThink = st.thinking_tokens != null && a.status !== "running" ? st.thinking_tokens : thinkingSpan(a);
    const ex = rt.experts_per_layer != null ? rt.experts_per_layer : null;
    const ms = h("div", { class: "metrics" },
      metric(plural(n, "token") + (nThink ? `, ${nThink} thinking` : "")),
      ex != null ? metric(`${ex.toFixed(2)} experts a layer`, "Experts run per layer, averaged over this answer's tokens. The router's number is four. Colour the tokens by Experts to see which tokens differ.") : null,
      rt.moved_share != null ? metric(`moved ${pct(rt.moved_share)}`, "Share of picks moved off the router's top four") : null,
      rt.mass_kept_mean != null ? metric(`mass kept ${pct(rt.mass_kept_mean)}, lowest ${pct(rt.min_kept)}`, "Mean share of router mass kept, and the lowest in any layer of any token") : null,
      st.perplexity ? metric(`perplexity ${fmt(st.perplexity, 2)}`, "exp of the mean surprise of the chosen tokens") : null,
      st.prompt_route ? metric(`prompt: moved ${pct(st.prompt_route.moved_share)}, kept ${pct(st.prompt_route.mass_kept_mean)}`, "The prompt's own rows: picks moved off the router's top four, and the mean share of router mass kept") : null);
    head.append(ms);
    const ch = a.status !== "running" ? (a.changed || (a.changed = changedOf(a.records))) : null;
    if (ch) head.append(h("div", { class: "metrics acted", title: "For each rule, the mean number of layers (of 57) at which it changed a token's routing. Click a token for its own counts." },
      h("span", { class: "k", text: "Rules acted, layers a token:" }), metric(changedText(ch))));
  }
  const flags = [];
  if (st.unreadable_rows) flags.push(h("span", { class: "chip warn", title: "Tokens whose distribution the engine could not read (NaN or infinite logits); they are left out of the statistics", text: plural(st.unreadable_rows, "unreadable row") }));
  if (a.slips && a.slips.length) flags.push(h("button", { type: "button", class: "chip warn", title: "Show the first one", text: plural(a.slips.length, "doubled word"), onclick: () => selectToken(a, a.slips[0].i) }));
  if (a.repeats && a.repeats.length) flags.push(h("button", { type: "button", class: "chip warn", title: "List them in the Analysis panel", text: plural(a.repeats.length, "repeated phrase"), onclick: () => analyse(a) }));
  if (many) flags.push(turn.active === k ? h("span", { class: "chip on", title: "The next prompt builds on this answer", text: "the conversation continues from here" })
    : h("button", { type: "button", class: "chip", title: "Make the next prompt build on this answer", text: "continue from this one", onclick: () => { turn.active = k; keepPlace($("#conversation"), () => turn.answers.forEach(x => x.el && fillAnswer(turn, x))); saveConversation(); } }));
  if (flags.length) line.after(h("div", { class: "line flags" }, flags));          // warnings right under the rule
  return head;
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
  const rec = a.records[i], tag = isTag(rec);
  const span = h("span", { class: "t" + (tag ? " tag" : "") + (slipAt && slipAt.has(i) ? " slip" : "") +
    (state.selected && state.selected.answer === a && state.selected.i === i ? " sel" : ""), text: text != null ? text : rec.text });
  span.dataset.i = i;
  if (!tag) paint(span, rec);
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
function bodyOf(turn, a) {
  if (a.status === "loading") return h("div", { class: "muted", text: "Loading the answer's records..." });
  if (!a.records.length) {
    return h("div", {},
      a.reasoning ? h("details", { class: "thinking" }, h("summary", { text: "Thinking" }), h("div", { class: "plain thinking-text", text: a.reasoning })) : null,
      h("div", { class: "plain", text: a.text || (a.status === "running" ? "Waiting for the first token..." : "") }));
  }
  const nThink = thinkingSpan(a), slipAt = new Set((a.slips || []).map(s => s.i));
  const body = h("div", { class: "body" });
  if (nThink) {
    const fold = h("details", { class: "thinking" },
      h("summary", {}, h("span", { class: "fold-title", text: "Thinking" }), h("span", { class: "muted", text: ` ${plural(nThink, "token")}` + (nThink === a.records.length && a.status === "running" ? ", still thinking" : "") })),
      textBlock(a, 0, nThink, slipAt, "thinking-text"));
    const open = a.thinkOpen != null ? a.thinkOpen : (a.status === "running" && nThink === a.records.length);
    if (open) { fold.dataset.auto = "1"; fold.open = true; }          // opening it here fires "toggle" too: not the reader's choice
    // A click is the reader's choice and is kept; opening it for them while the thinking streams is not.
    fold.addEventListener("toggle", () => { if (fold.dataset.auto !== "1") { a.thinkOpen = fold.open; saveConversation(); } fold.dataset.auto = ""; });
    body.append(fold);
  }
  if (nThink < a.records.length) body.append(textBlock(a, nThink, a.records.length, slipAt, "answer-text"));
  if ((a.stats || {}).finish === "length") body.append(h("div", { class: "cut", text: `Cut here: the answer reached the cap of ${a.cap ? plural(a.cap, "generated token") : "generated tokens"} before it ended.` }));
  return body;
}
function fillAnswer(turn, a) {
  const box = a.el;
  if (!box) return;
  const k = turn.answers.indexOf(a), many = turn.answers.length > 1;
  box.className = "answer" + (many && turn.active === k ? " active" : "") + (many && turn.active !== k ? " inactive" : "");
  const failed = (a.notes || []).some(n => n.startsWith("ROUTING CHECK FAILED"));
  box.replaceChildren(...[answerHead(turn, a),
    (a.notes || []).length ? h("div", { class: "notes" + (failed ? " err" : ""), text: a.notes.join("; ") }) : null,
    bodyOf(turn, a),
    a.error ? h("div", { class: "err", text: a.error }) : null].filter(Boolean));
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
  state.busy = b; $("#send").disabled = b; $("#stop").hidden = !b;
  $$("button.rerun").forEach(x => { x.disabled = b; });
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
