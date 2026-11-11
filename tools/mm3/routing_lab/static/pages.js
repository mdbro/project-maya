// Routing lab: the Fidelity meter page and the Answers page.
"use strict";

// ---------------------------------------------------------------------------------------------- meter
function renderArms() {
  const host = $("#meter-arms");
  host.replaceChildren(...state.arms.map((arm, k) => h("div", { class: "arm" },
    h("input", { type: "text", value: arm.name, "aria-label": "Name of this rule", onchange: e => { arm.name = e.target.value; } }),
    h("span", { class: "desc", title: describe(arm.settings), text: describe(arm.settings) }),
    h("button", { type: "button", class: "chip-x", "aria-label": `Remove ${arm.name}`, title: "Remove", text: "×", onclick: () => { state.arms.splice(k, 1); renderArms(); } }))));
  if (!state.arms.length) host.append(h("p", { class: "muted small", text: "No rules yet. Add the rules set in the Chat panel, a preset, or a sweep of the bonus." }));
  $("#meter-run").disabled = !state.arms.length || (state.meterJob && state.meterJob.status === "running");
  store.set("arms", state.arms);
}
const addArm = (name, settings) => { state.arms.push({ name, settings: clone(settings) }); renderArms(); };
function toMeter(a) {
  showPage("meter");
  refreshAnswers().then(() => { $("#meter-source").value = a.run; });
  $("#meter-mode").mark(a.drafting ? "drafting" : "one_row");
  state.meterMode = a.drafting ? "drafting" : "one_row";
}
async function runMeter() {
  const source = $("#meter-source").value, msg = $("#meter-msg");
  msg.textContent = "";
  if (!source) { msg.textContent = "Choose an answer to replay. Answers appear here once they have been written on the Chat page."; return; }
  if (!state.arms.length) { msg.textContent = "Add at least one rule to measure."; return; }
  const res = await api("/api/meter", { source, arms: state.arms, drafting: state.meterMode === "drafting", noise_floor: $("#meter-noise").checked });
  if (res.error) { msg.textContent = res.error; return; }
  pollMeter(res.id);
}
async function pollMeter(id) {
  const job = await api(`/api/meter?id=${encodeURIComponent(id)}`);
  if (job.error && !job.id) { $("#meter-msg").textContent = job.error; return; }
  state.meterJob = job;
  const p = $("#meter-progress"), running = job.status === "running";
  p.hidden = !running;
  p.querySelector("i").style.width = `${100 * job.done / Math.max(1, job.total)}%`;
  p.querySelector("span").textContent = `${job.done} of ${job.total}: ${job.progress}`;
  $("#meter-run").disabled = running || !state.arms.length;
  $$("#meter-list .item").forEach(n => n.classList.toggle("on", n.dataset.id === id));
  renderMeter(job);
  if (running) setTimeout(() => pollMeter(id), 1000); else refreshMeters();
}
function renderMeter(job) {
  const host = $("#meter-results");
  if (!job.arms || !job.arms.length) {
    host.replaceChildren(job.error ? h("div", { class: "card err", text: job.error }) : h("div", { class: "card muted", text: job.status === "running" ? "Waiting for the first rule's replay." : "No result." }));
    return;
  }
  const W = Math.max(480, Math.min(1100, host.clientWidth - 40));
  const card = h("div", { class: "card" }, h("h2", { text: "Result" }),
    h("p", { class: "lede", text: `${plural(job.tokens, "token")} replayed, ${job.drafting ? "drafting" : "one row"}. Reference: the unmodified router (bonus 0). ` +
      "Divergence is the Kullback-Leibler divergence from the reference's next-token distribution, in nats. The noise floor is what two runs of the reference differ by: a rule at or below it cannot be told from the unmodified router, and two rules whose divergences differ by less than it cannot be told from each other. The floor is not constant (0 to about 0.015 nats in the first checks), so read it in every run." }));
  if (job.source_text) card.append(h("details", { class: "source" }, h("summary", { text: "The answer that was replayed" }), h("div", { class: "plain small", text: job.source_text })));
  if (job.error) card.append(h("div", { class: "err", text: job.error }));
  if ((job.notes || []).length) card.append(h("p", { class: "notes", text: "Every run started from the saved cache state: " + job.notes.join("; ") + "." }));
  const ch = a => a.stats && a.stats.route && a.stats.route.changed;
  const cols = [["Rule", a => a.name], ["Mean divergence", a => fmt(a.summary.kl_mean, 4)], ["Median", a => fmt(a.summary.kl_median, 4)], ["95th pct", a => fmt(a.summary.kl_p95, 3)],
    ["Positions over 0.1", a => pct(a.summary["kl_over_0.1"])], ["Same first choice", a => pct(a.summary.top1_agreement)],
    ["Forced token's probability", a => "×" + fmt(Math.exp(a.summary.logratio_mean), 3)], ["Halved or worse", a => pct(a.summary.forced_prob_halved)],
    ["Experts a layer", a => a.stats.route && a.stats.route.experts_per_layer != null ? a.stats.route.experts_per_layer.toFixed(2) : "-"],
    ["Picks moved", a => a.stats.route ? pct(a.stats.route.moved_share) : "-"], ["Mass kept", a => a.stats.route ? pct(a.stats.route.mass_kept_mean) : "-"]];
  if (job.arms.some(ch)) cols.push(["Rules acted, layers a token", a => ch(a) ? changedText(Object.fromEntries(Object.entries(ch(a)).map(([k, v]) => [k, { per: v }]))) : "-"]);
  cols.push([(job.arms.map(a => speedOf(a.stats)).find(Boolean) || { name: "Est. ms a token" }).name.replace(/^e/, "E"), a => { const sp = speedOf(a.stats); return sp ? fmt(sp.v, 1) : "-"; }]);
  const table = h("table", { class: "data" }, h("thead", {}, h("tr", {}, cols.map(c => h("th", { text: c[0] })))),
    h("tbody", {}, job.arms.map(a => h("tr", { class: a.noise_floor ? "noise" : "" }, cols.map((c, k) => h("td", { class: k ? "num" : "name", text: c[1](a), title: k ? null : (a.describe || "") }))))));
  card.append(h("div", { class: "scroll-x" }, table));
  const rules = job.arms.filter(a => !a.noise_floor), noise = job.arms.find(a => a.noise_floor);
  const bars = h("div"), linesHost = h("div"), sc = h("div");
  const half = W >= 900 ? Math.floor(W / 2) - 12 : W;
  Charts.barsH(bars, job.arms.map(a => ({ label: a.name, value: a.summary.kl_mean, mark: !!a.noise_floor, note: `${a.name}\nmean divergence ${fmt(a.summary.kl_mean, 4)} nats\n${a.describe || ""}` })),
    { title: "Mean divergence from the unmodified router", note: noise ? "The marked bar is the noise floor." : "", labelW: Math.min(220, half * 0.4), width: half, format: v => fmt(v, 4) });
  const view = { w: 16, log: true };
  const drawLines = () => Charts.lines(linesHost, rules.slice(0, 8).map(a => ({ name: a.name, x: a.positions, y: rolling(a.kl, view.w) })),
    { title: "Divergence along the answer", note: `Rolling mean over ${plural(view.w, "token")}` + (view.log ? ", on a logarithmic scale so that small and large divergences both show." : ".") +
        (rules.length > 8 ? " The first eight rules are drawn; the table has them all." : ""), xname: "token", format: v => fmt(v, 4), log: view.log, logFloor: 1e-4, width: half });
  const tools = h("div", { class: "chart-tools" }, "Smooth over",
    segmented([1, 8, 16, 32].map(w => [w, String(w)]), 16, v => { view.w = +v; drawLines(); }, "Smoothing in tokens"), "tokens",
    h("label", { class: "check" }, h("input", { type: "checkbox", checked: true, onchange: e => { view.log = e.target.checked; drawLines(); } }), "logarithmic scale"));
  drawLines();
  card.append(h("div", { class: W >= 900 ? "two" : "" }, bars, h("div", {}, tools, linesHost)));
  const timed = rules.filter(a => speedOf(a.stats));
  if (timed.length >= 2) {
    const kind = speedOf(timed[0].stats);
    Charts.scatter(sc, timed.map(a => ({ label: a.name, x: speedOf(a.stats).v, y: a.summary.kl_mean, note: `${a.name}\n${kind.name} ${fmt(speedOf(a.stats).v, 1)}\nmean divergence ${fmt(a.summary.kl_mean, 4)}` })),
      { title: "Estimated cost against fidelity", note: `Down and to the left is better. Across: ${kind.what}. It orders rules; it is not a timing, and speed is decided by the timed tests.`, xlabel: kind.name, ylabel: "mean divergence, nats", width: Math.min(W, 820) });
    card.append(sc);
  } else card.append(h("p", { class: "muted small", text: "No cost estimate: the engine reported none for these runs." }));
  host.replaceChildren(card);
}
async function refreshMeters() {
  const list = await api("/api/meters");
  const jobs = Array.isArray(list) ? list : [];
  $("#meter-list").replaceChildren(...(jobs.length ? jobs.map(j => h("button", { type: "button", class: "item" + (state.meterJob && state.meterJob.id === j.id ? " on" : ""), dataset: { id: j.id },
    onclick: () => pollMeter(j.id) },
    h("span", { class: "when", text: when(j.created) }),
    h("span", { class: "what", text: `${plural(j.tokens, "token")}, ${j.drafting ? "drafting" : "one row"}: ${j.arms.join("; ") || j.progress}` }),
    h("span", { class: "meta", text: j.status === "done" ? "" : j.status }))) : [h("p", { class: "muted small", text: "None yet." })]));
}
const when = t => { const d = new Date(t * 1000), now = new Date(); return d.toDateString() === now.toDateString() ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); };

// ---------------------------------------------------------------------------------------------- answers
async function refreshAnswers() {
  const runs = await api("/api/runs");
  state.runs = (Array.isArray(runs) ? runs : []).filter(r => r.kind === "chat" && r.tokens > 0);
  const src = $("#meter-source"), keep = src.value;
  src.replaceChildren(...state.runs.map(r => h("option", { value: r.id, text: `${when(r.created)}  ${plural(r.tokens, "token")}  [${r.describe}]  ${r.prompt.replace(/\s+/g, " ").slice(0, 50)}` })));
  if (keep) src.value = keep;
  renderAnswerList();
}
function renderAnswerList() {
  const q = ($("#answer-filter").value || "").toLowerCase().trim();
  const shown = state.runs.filter(r => !q || `${r.prompt} ${r.answer} ${r.describe}`.toLowerCase().includes(q));
  $("#answer-count").textContent = q ? `${shown.length} of ${state.runs.length}` : plural(state.runs.length, "answer");
  $("#answer-list").replaceChildren(...(shown.length ? shown.map(r => h("div", { class: "item" + (state.ticked.has(r.id) ? " on" : "") },
    h("input", { type: "checkbox", "aria-label": "Tick to compare", checked: state.ticked.has(r.id), onchange: e => {
      if (e.target.checked) state.ticked.add(r.id); else state.ticked.delete(r.id);
      e.target.closest(".item").classList.toggle("on", e.target.checked);
      $("#compare").disabled = state.ticked.size !== 2;
      $("#compare-hint").textContent = state.ticked.size === 2 ? "" : `${state.ticked.size} of 2 ticked`;
    } }),
    h("div", { class: "what" },
      h("div", { class: "line1" }, h("span", { class: "chip rule", text: r.describe }), h("span", { class: "muted small", text: `${when(r.created)}, ${r.drafting ? "drafting" : "one row"}, ${plural(r.tokens, "token")}` })),
      h("div", { class: "prompt-line", text: r.prompt.replace(/\s+/g, " ") }),
      h("div", { class: "answer-line muted", text: r.answer.replace(/\s+/g, " ") })),
    h("button", { type: "button", class: "ghost small", text: "Open in chat", title: "Add this answer to the conversation on the Chat page, under its prompt", onclick: () => openInChat(r.id) })))
    : [h("p", { class: "muted small", text: q ? "No answer matches." : "No answers yet." })]));
}
// A stored answer into the conversation: under the same prompt if the conversation has it, else as a new prompt.
async function openInChat(id) {
  const r = await api(`/api/run?id=${encodeURIComponent(id)}`);
  if (r.error) { toast(r.error, "bad"); return; }
  const user = (r.messages || []).filter(m => m.role === "user").map(m => m.content).pop() || "";
  let turn = state.turns.find(t => t.user === user);
  if (!turn) { turn = { id: "t" + (++state.seq), user, answers: [], active: 0 }; state.turns.push(turn); }
  if (turn.answers.some(a => a.run === id)) { showPage("chat"); turn.el && turn.el.scrollIntoView(); return; }
  const a = fromRun(newAnswer({}), r);
  turn.answers.push(a);
  if ((r.messages || []).length > 1) toast("This answer continued an earlier conversation; only its last prompt is shown.");
  showPage("chat");
  renderConversation();
  a.el.scrollIntoView({ block: "start" });
  saveConversation();
}
async function compareTicked() {
  const ids = [...state.ticked];
  const [A, B, C] = await Promise.all([api(`/api/run?id=${encodeURIComponent(ids[0])}`), api(`/api/run?id=${encodeURIComponent(ids[1])}`),
    api(`/api/compare?a=${encodeURIComponent(ids[0])}&b=${encodeURIComponent(ids[1])}`)]);
  if (A.error || B.error) { toast(A.error || B.error, "bad"); return; }
  const W = Math.max(420, $("#compare-result").clientWidth - 40);
  const rows = [["Rules", r => r.describe], ["Mode", r => r.drafting ? "drafting" : "one row"], ["Tokens", r => String(r.stats.tokens)],
    ["Thinking tokens", r => String(r.stats.thinking_tokens || 0)],
    ["Ended", r => r.stats.finish === "length" ? "cut at the token cap" : r.stats.finish === "stop" ? "by itself" : (r.stats.finish || "-")],
    ["Perplexity", r => fmt(r.stats.perplexity, 2)], ["Mean probability", r => fmt(r.stats.mean_prob, 3)], ["Uncertainty, nats", r => fmt(r.stats.entropy_mean, 2)],
    ["First choice taken", r => pct(r.stats.chosen_top1)], ["Drafted", r => pct(r.stats.drafted)],
    ["Experts a layer", r => r.stats.route && r.stats.route.experts_per_layer != null ? r.stats.route.experts_per_layer.toFixed(2) : "-"],
    ["Picks moved", r => r.stats.route ? pct(r.stats.route.moved_share) : "-"], ["Router mass kept", r => r.stats.route ? pct(r.stats.route.mass_kept_mean) : "-"],
    ["Lowest mass kept", r => r.stats.route ? pct(r.stats.route.min_kept) : "-"], ["Loads a token", r => r.stats.route ? fmt(r.stats.route.loads, 1) : "-"],
    ["Rules acted, layers a token", r => { const c = changedOf(r.records || []); return c ? changedText(c) : "-"; }],
    ["Prompt: picks moved", r => r.stats.prompt_route ? pct(r.stats.prompt_route.moved_share) : "-"], ["Prompt: mass kept", r => r.stats.prompt_route ? pct(r.stats.prompt_route.mass_kept_mean) : "-"],
    ["Doubled words", r => String(findSlips(r.records || []).length)], ["Repeated phrases", r => String(findRepeats(r.records || []).length)]];
  const table = h("table", { class: "data" }, h("thead", {}, h("tr", {}, h("th", { text: "" }), h("th", { text: "First" }), h("th", { text: "Second" }))),
    h("tbody", {}, rows.map(([k, f]) => h("tr", {}, h("td", { text: k }), h("td", { class: "num", text: f(A) }), h("td", { class: "num", text: f(B) })))));
  const card = h("div", { class: "card" }, h("h2", { text: "Comparison" }));
  const samePrompt = same(A.messages, B.messages);
  let lede = samePrompt ? "" : "The two answers are to different prompts, so only their totals can be compared. ";
  if (samePrompt && !C.error) lede += C.identical ? "The two answers are the same tokens throughout." :
    `The two answers share their first ${plural(C.shared, "token")} and then choose differently.` +
    (C.shared ? ` Over the shared tokens their next-token distributions differ by ${fmt(C.summary.kl_mean, 4)} nats on average (largest ${fmt(C.summary.kl_max, 3)}).` : "");
  card.append(h("p", { class: "lede", text: lede }), h("div", { class: "scroll-x" }, table));
  if (samePrompt && !C.error && C.shared > 1) {
    const line = h("div");
    Charts.lines(line, [{ name: "divergence", x: C.positions, y: C.kl }], { title: "Divergence between the two answers' distributions, over the tokens they share",
      note: "Kullback-Leibler divergence of the second from the first, in nats, at each shared position.", xname: "token", height: 170, format: v => fmt(v, 4), width: W,
      xnote: i => JSON.stringify(((A.records || [])[i] || {}).text || "") });
    card.append(line);
  }
  if (samePrompt && C.first_a && C.first_b) {
    const side = (rec, name) => { const d = h("div"); const order = [...rec.top].sort((x, y) => y[1] - x[1]).slice(0, 12);
      Charts.barsH(d, order.map(t => ({ label: JSON.stringify(t[2] != null ? t[2] : String(t[0])).slice(1, -1), value: t[1], mark: t[0] === rec.id, note: `${JSON.stringify(t[2])}  probability ${fmt(t[1])}` })),
        { title: `${name}: chose ${JSON.stringify(rec.text)}`, note: "The marked candidate is the one chosen.", width: Math.min(460, W / 2 - 10), format: v => fmt(v) }); return d; };
    card.append(h("h3", { text: `Where they part: token ${C.shared + 1}` }),
      h("p", { class: "muted", text: "..." + (A.records || []).slice(Math.max(0, C.shared - 14), C.shared).map(r => r.text).join("") }),
      h("p", { class: "muted small", text: `Divergence between the two distributions here: ${fmt(C.kl_at_split, 4)} nats. A small value means the same distribution and a different draw; a large one means the rule changed what the model wanted to say.` }),
      h("div", { class: "two" }, side(C.first_a, "First"), side(C.first_b, "Second")));
  }
  card.append(h("div", { class: "two", style: "margin-top:12px" }, h("div", {}, h("h3", { text: "First" }), h("div", { class: "plain small", text: A.text })), h("div", {}, h("h3", { text: "Second" }), h("div", { class: "plain small", text: B.text }))));
  $("#compare-result").replaceChildren(card);
  $("#compare-result").scrollTop = 0;
}
