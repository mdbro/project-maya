// Routing lab page: start-up and wiring. The parts live in util.js, markdown.js, rules.js, chat.js, inspect.js and
// pages.js. All model output is placed with textContent; nothing from the model is ever parsed as HTML.
"use strict";

const state = {
  info: null, settings: {}, sampling: {}, mode: "one_row", newSeed: false,
  turns: [], seq: 0, busy: false, currentRun: null,
  colorBy: "surprisal", view: "formatted", layout: "side",
  selected: null,            // {answer, i}
  analysed: null,            // the answer shown in the Analysis panel
  sideTab: "rules", sideHidden: false, sideWidth: 400, pinned: new Set(),
  arms: [], meterJob: null, meterMode: "one_row", presets: { built_in: {}, saved: {} }, ticked: new Set(), runs: [],
};

// ---------------------------------------------------------------------------------------------- pages
function showPage(name) {
  $$("[data-tab]").forEach(t => { const on = t.dataset.tab === name; t.classList.toggle("on", on); t.setAttribute("aria-selected", on); });
  $$(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + name));
  if (name === "answers" || name === "meter") refreshAnswers();
  if (name === "meter") refreshMeters();
  store.set("page", name);
}

function showEngine(e) {
  const st = $("#engine-status"), L = e.loading, mock = e.engine === "mock";
  const mmss = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  st.className = "status " + (mock ? "mock" : e.connected ? "ok" : L ? "wait" : "bad");
  st.textContent = mock ? "mock engine" : e.connected ? (e.note ? "engine connected, with limits" : "engine connected") : L ? "engine loading" : "engine not reachable";
  st.title = mock ? "Mock engine: the text and numbers are synthetic" : (e.note || e.error || e.url || "");
  const note = $("#engine-note");
  note.className = "engine-note" + (mock || (e.connected && !e.note) ? "" : e.connected ? " info" : L ? " info" : " warn");
  if (mock || (e.connected && !e.note)) { note.replaceChildren(); return; }
  if (e.connected) { note.replaceChildren(h("span", { text: e.note })); return; }
  if (L) {
    const share = L.phase === "setup" ? 1 : Math.min(0.98, (L.elapsed_s || 0) / Math.max(1, L.typical_s || 1));
    note.replaceChildren(h("strong", { text: L.phase === "setup" ? "Preparing the engine" : "Loading the model" }),
      h("span", { text: L.phase === "setup" ? "Placement and the saved starting state, about 20 seconds." : `${mmss(L.elapsed_s)} of about ${mmss(L.typical_s)}.` }),
      h("span", { class: "bar", "aria-hidden": "true" }, h("i", { style: `width:${(100 * share).toFixed(1)}%` })),
      h("span", { class: "muted", text: "The page connects by itself; set the rules meanwhile." }));
    return;
  }
  note.replaceChildren(h("strong", { text: "Not connected" }),
    h("span", { text: e.blocked ? e.blocked : `The engine does not answer at ${e.url}. Start it with launch_lab.py start; this page connects by itself.` }));
}
// Until the engine answers, ask again every three seconds, so that the page connects without a reload.
function watchEngine() {
  const e = state.info.engine;
  if (e.engine === "mock" || e.connected) return;
  const again = async () => {
    const info = await api("/api/info");
    if (info && info.engine) { state.info.engine = info.engine; showEngine(info.engine); if (info.engine.connected) return; }
    setTimeout(again, 3000);
  };
  setTimeout(again, 3000);
}

// ---------------------------------------------------------------------------------------------- side panel
const defaultSideWidth = () => Math.round(Math.max(320, Math.min(420, window.innerWidth * 0.3)));
const layoutState = () => ({ sideTab: state.sideTab, sideHidden: state.sideHidden, sideWidth: state.sideWidth, colorBy: state.colorBy, view: state.view, layout: state.layout });
function applySideWidth() {
  const max = Math.max(300, Math.floor(window.innerWidth * 0.6));
  state.sideWidth = Math.max(280, Math.min(max, state.sideWidth));
  document.documentElement.style.setProperty("--side-w", state.sideWidth + "px");
}
function setSideHidden(hidden) {
  state.sideHidden = hidden;
  $("#page-chat").classList.toggle("side-hidden", hidden);
  $("#side-show").hidden = !hidden;
  $("#side-toggle").setAttribute("aria-expanded", !hidden);
  store.set("layout", layoutState());
}
let redrawTimer = 0;
function panelResized() {          // charts in the panel follow its width, once the drag has settled
  clearTimeout(redrawTimer);
  redrawTimer = setTimeout(() => {
    if (state.sideTab === "token" && state.selected) renderToken();
    if (state.sideTab === "analysis" && state.analysed) { const t = $("#side-analysis").scrollTop; renderAnalysis(); $("#side-analysis").scrollTop = t; }
  }, 120);
}
function wireSplitter() {
  const bar = $("#splitter");
  let startX = 0, startW = 0;
  const move = ev => { state.sideWidth = startW + (startX - ev.clientX); applySideWidth(); };
  const up = () => { document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up); document.body.classList.remove("dragging"); store.set("layout", layoutState()); panelResized(); };
  bar.addEventListener("pointerdown", ev => {
    if (state.sideHidden) return;
    startX = ev.clientX; startW = state.sideWidth;
    document.body.classList.add("dragging");
    document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
    ev.preventDefault();
  });
  bar.addEventListener("keydown", ev => {             // the splitter can be moved from the keyboard too
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    state.sideWidth += ev.key === "ArrowLeft" ? 32 : -32;
    applySideWidth(); store.set("layout", layoutState()); panelResized(); ev.preventDefault();
  });
  bar.addEventListener("dblclick", () => { state.sideWidth = defaultSideWidth(); applySideWidth(); store.set("layout", layoutState()); panelResized(); });
}

// ---------------------------------------------------------------------------------------------- start
async function init() {
  const info = await api("/api/info");
  if (info.error) throw new Error(info.error);
  state.info = info;
  const lay = store.get("layout", {});
  Object.assign(state, {
    colorBy: lay.colorBy || "surprisal", view: lay.view === "tokens" ? "tokens" : "formatted", layout: lay.layout === "stack" ? "stack" : "side",
    sideTab: ["rules", "token", "analysis"].includes(lay.sideTab) ? lay.sideTab : "rules", sideWidth: +lay.sideWidth || defaultSideWidth() });
  state.arms = (store.get("arms", []) || []).filter(a => a && a.settings);
  restoreRules();
  applySideWidth();
  if (lay.sideHidden) setSideHidden(true);

  // Controls built once.
  $("#mode-host").replaceChildren(Object.assign(segmented([["one_row", "One row", "No drafting: one token a pass"], ["drafting", "Drafting", "The drafter proposes tokens and the model checks them"]], state.mode, setMode, "Mode"), { id: "mode" }));
  $("#meter-mode-host").replaceChildren(Object.assign(segmented([["one_row", "One row"], ["drafting", "Drafting"]], state.meterMode, v => { state.meterMode = v; }, "Mode"), { id: "meter-mode" }));
  $("#color-host").replaceChildren(segmented([
    ["surprisal", "Surprise", "How unlikely the chosen token was: minus the log of its probability"],
    ["entropy", "Uncertainty", "How uncertain the model was at this position: the entropy of its distribution"],
    ["rank", "Rank", "Which choice the token was: first, second, ... among the candidates"],
    ["lost", "Mass lost", "Router mass lost in this token's worst layer"],
    ["moved", "Picks moved", "Share of this token's expert picks that were moved off the router's own"],
    ["experts", "Experts", "How many experts ran for this token: fewer or more than four a layer"],
    ["source", "Drafted", "Whether the token was an accepted draft"],
    ["none", "Off", "No highlighting"]], state.colorBy, v => { state.colorBy = v; legend(); repaint(); store.set("layout", layoutState()); }, "Colour tokens by"));
  // Settings chosen once and rarely changed live in one View menu, not on the toolbar.
  const setView = v => { state.view = v; keepPlace($("#conversation"), () => state.turns.forEach(t => t.answers.forEach(a => fillAnswer(t, a)))); store.set("layout", layoutState()); };
  const setLayout = v => { state.layout = v; renderConversation(); store.set("layout", layoutState()); };
  $("#view-host").replaceChildren(menuButton("View", () => [
    { text: "Formatted text", title: "Markdown laid out: headings, lists, tables and code", checked: state.view === "formatted", onclick: () => setView("formatted") },
    { text: "Every token as written", title: "Control tokens and Markdown markers included", checked: state.view === "tokens", onclick: () => setView("tokens") },
    "-",
    { text: "Answers to one prompt side by side", checked: state.layout === "side", onclick: () => setLayout("side") },
    { text: "Answers to one prompt stacked", checked: state.layout === "stack", onclick: () => setLayout("stack") }], { title: "How answers are shown" }));

  buildRules();
  showEngine(state.info.engine); watchEngine();
  wirePresetSave();
  await loadPresets();
  renderArms(); legend();
  sideTab(state.sideTab);
  const draft = store.get("draft", "");
  if (draft) $("#prompt").value = draft;
  autosize();

  $$("[data-tab]").forEach(t => t.addEventListener("click", () => showPage(t.dataset.tab)));
  $$("[data-side]").forEach(t => t.addEventListener("click", () => sideTab(t.dataset.side)));
  $("#side-toggle").addEventListener("click", () => setSideHidden(true));
  $("#side-show").addEventListener("click", () => setSideHidden(false));
  wireSplitter();
  wireConversation();
  window.addEventListener("resize", () => { applySideWidth(); autosize(); panelResized(); });
  $("#new-chat").addEventListener("click", newConversation);
  $("#composer").addEventListener("submit", ev => { ev.preventDefault(); if (state.busy) stopRun(); else send(); });
  $("#rerun-last").addEventListener("click", () => { const last = state.turns[state.turns.length - 1]; if (last) generate(last); });
  $("#prompt").addEventListener("input", () => { autosize(); saveDraft(); });
  $("#prompt").addEventListener("keydown", ev => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); send(); } });
  const fold = $("#samp-fold");
  fold.open = !!store.get("samplingOpen", false);
  fold.addEventListener("toggle", () => store.set("samplingOpen", fold.open));
  $("#reset-rules").addEventListener("click", () => { state.pinned.clear(); loadSettings(state.info.defaults); state.sampling = Object.assign(clone(state.info.sampling_defaults), { seed: state.sampling.seed }); buildRules(); });
  $("#explain").checked = !!store.get("explain", false);
  $("#side-rules").classList.toggle("explain", $("#explain").checked);
  $("#explain").addEventListener("change", e => { $("#side-rules").classList.toggle("explain", e.target.checked); store.set("explain", e.target.checked); });
  $("#arm-current").addEventListener("click", () => addArm(describe(state.settings), state.settings));
  $("#arm-sweep").addEventListener("click", () => [0.20, 0.25, 0.30, 0.40].forEach(b => addArm(`Bonus ${b.toFixed(2)}`, { ...state.settings, routing_prior_bonus: b })));
  $("#arm-clear").addEventListener("click", () => { state.arms = []; renderArms(); });
  $("#meter-run").addEventListener("click", runMeter);
  $("#compare").addEventListener("click", compareTicked);
  $("#refresh-answers").addEventListener("click", refreshAnswers);
  $("#answer-filter").addEventListener("input", renderAnswerList);

  // Keys: left and right step through tokens once one is selected (not while typing); Escape stops an answer.
  document.addEventListener("keydown", ev => {
    const typing = ev.target.closest("input, textarea, select, [contenteditable]");
    if (typing || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && state.selected && $("#page-chat").classList.contains("on") && !ev.target.closest("#splitter, .seg, [role=radiogroup]")) {
      stepToken(ev.key === "ArrowLeft" ? -1 : 1); ev.preventDefault();
    } else if (ev.key === "Escape" && state.busy && !openMenuState) { stopRun(); }
  });

  const page = store.get("page", "chat");
  showPage(["chat", "meter", "answers"].includes(page) ? page : "chat");
  refreshAnswers();
  await restoreConversation();
}
init().catch(err => { $("#engine-status").className = "status bad"; $("#engine-status").textContent = "the page could not start"; console.error(err); });
