// Routing lab: the rules panel. Every control is built from the schema the backend sends (settings_schema.py).
// The top of the panel says what the next prompt will run with and how that differs from today's serving rule;
// each difference can be cleared with one click, and presets are one click away.
"use strict";

const LAYER_GROUPS = [[0, 9], [10, 19], [20, 29], [30, 39], [40, 49], [50, 56]];
const SHORT = { routing_prior_bonus: "bonus", route_bonus_draft: "draft bonus", route_prompt_bonus: "prompt at", route_floor: "budget",
                route_protect: "protect", route_priced: "priced", route_layer_drop: "layer drop", route_k_drop: "drop pick", route_k_widen: "widen",
                route_price_first: "price of a lone load", route_price_more: "price of a further load", route_price_staged: "price of a staged copy",
                route_k_drop_ranks: "drop applies to", route_weights: "weights", route_lab_prompt: "prompt", route_layer_scale: "bonus by layer",
                temperature: "temperature", top_p: "top p", max_tokens: "cap", reasoning_effort: "thinking", same_start: "start", lab_path: "path" };

// A short label for a set of rules: only what differs from today's rule (the backend's describe() says the same).
function describe(s) {
  const d = state.info.defaults, parts = [`bonus ${(+s.routing_prior_bonus).toFixed(2)}`];
  for (const k of ["route_bonus_draft", "route_prompt_bonus", "route_floor", "route_protect", "route_priced", "route_layer_drop", "route_k_drop", "route_k_widen"])
    if (s[k] !== d[k]) parts.push(`${SHORT[k]} ${+(k === "route_floor" ? 1 - s[k] : +s[k]).toFixed(3)}`);
  if (s.route_weights) parts.push(s.route_weights === 1 ? "dampen" : "amplify");
  if (!s.route_lab_prompt) parts.push("rules skip the prompt");
  if ((s.route_layer_scale || []).some(x => Math.abs(x - 1) > 1e-9)) parts.push("by layer");
  return parts.join(", ");
}

const schemaOf = key => state.info.schema.find(s => s.key === key) || state.info.sampling.find(s => s.key === key);
function valueText(s, v) {
  if (s.kind === "choice") { const c = s.choices.find(c => c[0] === v); return c ? c[1].replace(/\s*\(.*\)$/, "") : String(v); }
  if (s.kind === "layers") return "changed";
  // As the answers' labels write it (settings_schema.describe): the bonus with two decimals, the rest as short as it goes.
  const x = s.invert ? 1 - v : +v;
  return s.key === "routing_prior_bonus" ? x.toFixed(2) : String(+x.toFixed(3));
}
// What the next prompt runs with that is not today's: [{key, text, sampling}].
function differences() {
  const out = [], d = state.info.defaults, sd = state.info.sampling_defaults;
  for (const s of state.info.schema) {
    if (same(state.settings[s.key], d[s.key])) continue;
    let text = s.optional && state.settings[s.key] === s.default ? `${SHORT[s.key] || s.label} off` : `${SHORT[s.key] || s.label} ${valueText(s, state.settings[s.key])}`;
    if (s.key === "route_weights") text = valueText(s, state.settings[s.key]).split(":")[0];
    if (s.key === "route_lab_prompt") text = "rules skip the prompt";
    if (s.key === "route_layer_scale") text = "bonus by layer";
    out.push({ key: s.key, text });
  }
  for (const s of state.info.sampling) {
    if (s.key === "seed" || same(state.sampling[s.key], sd[s.key])) continue;
    const v = state.sampling[s.key];
    const text = s.key === "max_tokens" ? `cap ${v} tokens` : s.key === "same_start" ? (v ? "restore the cache first" : "carry the cache on")
      : s.key === "lab_path" ? (v ? "lab path" : "serving kernel") : `${SHORT[s.key] || s.label} ${valueText(s, v)}`;
    out.push({ key: s.key, sampling: true, text });
  }
  return out;
}

function buildRules() {
  const host = $("#rules");
  const scroll = $("#side-rules").scrollTop;
  host.replaceChildren();
  const groups = new Map();
  for (const s of state.info.schema) {
    if (!groups.has(s.group)) groups.set(s.group, { main: [], adv: [] });
    groups.get(s.group)[s.advanced ? "adv" : "main"].push(s);
  }
  let first = true;
  for (const [name, g] of groups) {
    const title = h("h3", { class: "group-title" }, h("span", { text: name }));
    if (first) { title.append($("#explain-wrap")); first = false; }       // the switch for every rule's explanation
    const box = h("section", { class: "group" }, title);
    g.main.forEach(s => box.append(field(s, state.settings)));
    if (g.adv.length) box.append(h("details", { class: "adv", open: g.adv.some(s => !same(state.settings[s.key], state.info.defaults[s.key])) },
      h("summary", { title: g.adv.map(s => s.label).join("; "), text: g.adv.every(s => s.label.startsWith("Price:")) ? "More: the three prices" : `More: ${g.adv.map(s => s.label.toLowerCase()).join("; ")}` }), g.adv.map(s => field(s, state.settings))));
    host.append(box);
  }
  // Sampling is set once and seldom touched: folded, with what it is set to in the fold's title.
  const samp = h("details", { class: "group samp", open: !!store.get("samplingOpen", false) },
    h("summary", {}, h("span", { class: "group-title", text: "Sampling and thinking" }), h("span", { class: "samp-sum", id: "samp-sum" })),
    state.info.sampling.map(s => field(s, state.sampling)));
  samp.addEventListener("toggle", () => store.set("samplingOpen", samp.open));
  host.append(samp);
  $("#side-rules").scrollTop = scroll;
  rulesChanged();
}

// One control. Rules that are off show only their name and a switch; their slider appears when switched on.
function field(s, target) {
  const defaults = target === state.sampling ? state.info.sampling_defaults : state.info.defaults;
  const id = "f-" + s.key;
  const wrap = h("div", { class: "field" + (s.key === "routing_prior_bonus" ? " hero" : ""), dataset: { key: s.key } });
  const head = h("div", { class: "field-head" });
  const reset = h("button", { type: "button", class: "reset-one", text: "reset", title: "Back to today's value",
    onclick: () => { target[s.key] = clone(defaults[s.key]); wrap.replaceWith(field(s, target)); rulesChanged(); } });
  const set = v => { target[s.key] = v; rulesChanged(); mark(); };
  const mark = () => wrap.classList.toggle("changed", !same(target[s.key], defaults[s.key]));
  wrap._mark = mark;
  if (s.kind === "choice") {
    head.append(h("span", { class: "label", id: id + "-l", text: s.label }), h("span", { class: "grow" }), reset);
    const short = s.choices.every(c => c[1].length <= 14);
    let ctl;
    if (short) ctl = segmented(s.choices.map(c => [c[0], c[1]]), target[s.key], v => set(s.choices.find(c => String(c[0]) === String(v))[0]), s.label);
    else {
      ctl = h("div", { class: "choices", role: "radiogroup", "aria-labelledby": id + "-l" });
      for (const [v, text] of s.choices)
        ctl.append(h("label", { class: "choice" }, h("input", { type: "radio", name: id, checked: target[s.key] === v, onchange: () => set(v) }), h("span", { text })));
    }
    wrap.append(head, ctl);
  } else if (s.kind === "layers") {
    head.append(h("span", { class: "label", text: s.label }), h("span", { class: "grow" }), reset);
    const grid = h("div", { class: "layer-groups" });
    const state_ = h("span", { class: "muted" });
    const all = h("textarea", { rows: 3, class: "mono small", "aria-label": "All 57 values", onchange: e => {
      const v = e.target.value.split(/[\s,]+/).filter(Boolean).map(Number);
      if (v.length === state.info.n_layers && v.every(x => Number.isFinite(x) && x >= s.min && x <= s.max)) { target[s.key] = v; sync(); set(v); }
      else { toast(`Give ${state.info.n_layers} numbers from ${s.min} to ${s.max}.`, "bad"); e.target.value = target[s.key].join(", "); }
    } });
    const inputs = LAYER_GROUPS.map(([a, b]) => {
      const range = h("input", { type: "range", min: s.min, max: s.max, step: s.step, "aria-label": `layers ${a + 1} to ${b + 1}` });
      const num = h("input", { type: "number", min: s.min, max: s.max, step: s.step, "aria-label": `layers ${a + 1} to ${b + 1}` });
      const apply = v => { v = Math.max(s.min, Math.min(s.max, +v)); const arr = target[s.key].slice(); for (let i = a; i <= b; i++) arr[i] = v; target[s.key] = arr; sync(); set(arr); };
      range.addEventListener("input", e => apply(e.target.value)); num.addEventListener("change", e => apply(e.target.value));
      grid.append(h("span", { text: `${a + 1} to ${b + 1}` }), range, num);
      return { a, b, range, num };
    });
    function sync() {
      for (const g of inputs) {
        const vals = target[s.key].slice(g.a, g.b + 1), one = vals.every(x => x === vals[0]);
        g.range.value = vals[0]; g.num.value = one ? vals[0] : ""; g.num.placeholder = one ? "" : "mixed";
      }
      all.value = target[s.key].map(x => +x.toFixed(3)).join(", ");
      const flat = target[s.key].every(x => x === 1);
      state_.textContent = flat ? "1 in every layer" : "set per layer";
    }
    sync();
    const changed = !same(target[s.key], defaults[s.key]);
    wrap.append(head, h("details", { class: "layers-edit", open: changed }, h("summary", {}, "Layer groups, ", state_), grid,
      h("details", { class: "adv" }, h("summary", { text: "All 57 values" }), all)));
  } else {
    // `invert`: the control shows 1 minus the stored value (a budget on lost mass, stored as a floor on mass kept).
    const off = s.default, optional = !!s.optional, inv = !!s.invert, conv = v => inv ? +(1 - v).toFixed(6) : v;
    const lo = inv ? conv(s.max) : (optional && s.min < 0 ? 0 : s.min), hi = inv ? conv(s.min) - (optional ? s.step : 0) : s.max;
    const slider = s.max - s.min <= 10000;
    const range = slider ? h("input", { type: "range", min: lo, max: hi, step: s.step, "aria-label": s.label }) : null;
    const num = h("input", { type: "number", id, min: lo, max: hi, step: s.step, class: slider ? "" : "wide" });
    let box = null, last = conv(s.on_value != null ? s.on_value : (s.key === "route_bonus_draft" ? 0.20 : s.default));
    if (optional && target[s.key] !== off) last = conv(target[s.key]);
    const row = h("div", { class: "row" }, range, num);
    const show = () => {
      const on = !optional || target[s.key] !== off;
      if (box) box.checked = on;
      wrap.classList.toggle("off", !on);
      row.hidden = !on;
      const v = on ? conv(target[s.key]) : last;
      if (range) range.value = v;
      num.value = v;
    };
    const apply = v => { v = Math.max(lo, Math.min(hi, +v)); if (!Number.isFinite(v)) return show(); if (s.kind === "int") v = Math.round(v); last = v; set(conv(v)); show(); };
    if (range) range.addEventListener("input", e => apply(e.target.value));
    num.addEventListener("change", e => apply(e.target.value));
    if (optional) {
      box = h("input", { type: "checkbox", id: id + "-on", onchange: e => { set(e.target.checked ? conv(last) : off); show(); if (e.target.checked) (range || num).focus(); } });
      head.append(h("label", { class: "switch", for: id + "-on" }, box, h("span", { class: "label", text: s.label })));
    } else head.append(h("label", { class: "label", for: id, text: s.label }));
    head.append(h("span", { class: "grow" }), reset);
    if (s.key === "seed") row.append(h("button", { type: "button", class: "ghost small", text: "New seed", title: "Draw a new random seed",
      onclick: () => { target.seed = Math.floor(Math.random() * 2 ** 31); show(); rulesChanged(); } }),
      h("label", { class: "check inline-check" }, h("input", { type: "checkbox", id: "new-seed", checked: state.newSeed,
        onchange: e => { state.newSeed = e.target.checked; saveRules(); } }), "new for every prompt"));
    wrap.append(head, row);
    wrap._show = show;
    show();
  }
  if (s.help) wrap.append(h("div", { class: "help", text: s.help }));
  mark();
  return wrap;
}

// After any change: the differences, the matching preset, the summary under the prompt box, and memory.
function rulesChanged() {
  if (!state.info) return;
  const diffs = differences();
  const chips = $("#diff-chips");
  chips.replaceChildren(...(diffs.length ? diffs.map(d => h("span", { class: "chip diff" + (d.sampling ? " samp" : "") },
    h("button", { type: "button", class: "chip-go", text: d.text, title: "Show this control", onclick: () => focusField(d.key) }),
    h("button", { type: "button", class: "chip-x", "aria-label": `Clear ${d.text}`, title: "Back to today's value", text: "×", onclick: () => clearKey(d.key, d.sampling) })))
    : []));
  // The chips are the rule when it differs from today's; the headline names it when it does not.
  const own = diffs.some(d => !d.sampling);
  $("#next-rule").hidden = own;
  $("#next-rule").textContent = "Today's serving rule";
  $("#next-rule").title = describe(state.settings);
  $("#reset-rules").hidden = !diffs.length;
  const ss = $("#samp-sum");
  if (ss) {
    const sp = state.sampling, eff = (state.info.sampling.find(x => x.key === "reasoning_effort").choices.find(c => c[0] === sp.reasoning_effort) || [, ""])[1].replace(/\s*\(.*\)$/, "");
    ss.textContent = `temperature ${sp.temperature}, top p ${sp.top_p}, thinking ${eff}, ` + (sp.max_tokens ? `cap ${sp.max_tokens}` : "no cap") + `, seed ${sp.seed}`;
  }
  for (const b of $$("#preset-list [data-preset]")) b.classList.toggle("on", same(presetByName(b.dataset.preset), state.settings));
  composerSummary(diffs);
  saveRules();
}
function clearKey(key, sampling) {
  const target = sampling ? state.sampling : state.settings, d = sampling ? state.info.sampling_defaults : state.info.defaults;
  target[key] = clone(d[key]);
  const old = document.querySelector(`#rules .field[data-key="${key}"]`);
  if (old) old.replaceWith(field(schemaOf(key), target));
  rulesChanged();
}
function focusField(key) {
  sideTab("rules");
  const f = document.querySelector(`#rules .field[data-key="${key}"]`);
  if (!f) return;
  const d = f.closest("details");
  if (d) d.open = true;
  f.scrollIntoView({ block: "center" });
  f.classList.remove("flash"); void f.offsetWidth; f.classList.add("flash");
  const input = f.querySelector("input:not([type=checkbox]):not([hidden]), select, textarea") || f.querySelector("input");
  if (input) input.focus({ preventScroll: true });
}

// Under the prompt box: what Send and Rerun will run with, in one line. A click opens the rules.
function composerSummary(diffs) {
  const host = $("#rule-summary");
  const parts = [state.mode === "drafting" ? "drafting" : "one row", ...diffs.filter(d => d.sampling).map(d => d.text)];
  host.replaceChildren(h("button", { type: "button", class: "next-link", title: "Change the rules for the next prompt", onclick: () => { sideTab("rules"); $("#side-rules").scrollTop = 0; } },
    h("span", { class: "muted", text: "Runs with " }), h("strong", { text: ruleLabel(describe(state.settings)) }), h("span", { class: "muted", text: " \u00b7 " + parts.join(" \u00b7 ") })));
}

function loadSettings(s) { state.settings = Object.assign(clone(state.info.defaults), clone(s)); buildRules(); }
function setMode(m) { state.mode = m; $("#mode").mark(m); rulesChanged(); }

function saveRules() {
  store.set("rules", { settings: state.settings, sampling: state.sampling, mode: state.mode, newSeed: state.newSeed });
}
// Rules kept from the last visit, cleaned against today's schema: unknown keys dropped, out-of-range values reset.
function restoreRules() {
  const got = store.get("rules", null);
  state.settings = clone(state.info.defaults);
  state.sampling = clone(state.info.sampling_defaults);
  state.mode = "one_row";
  if (!got) return;
  const ok = (s, v) => {
    if (s.kind === "choice") return s.choices.some(c => c[0] === v);
    if (s.kind === "layers") return Array.isArray(v) && v.length === state.info.n_layers && v.every(x => typeof x === "number" && x >= s.min && x <= s.max);
    return typeof v === "number" && v >= s.min && v <= s.max;
  };
  for (const s of state.info.schema) if (got.settings && ok(s, got.settings[s.key])) state.settings[s.key] = got.settings[s.key];
  for (const s of state.info.sampling) if (got.sampling && ok(s, got.sampling[s.key])) state.sampling[s.key] = got.sampling[s.key];
  if (got.mode === "drafting") state.mode = "drafting";
  state.newSeed = !!got.newSeed;
}

// ------------------------------------------------------------------------------------------------------ presets
async function loadPresets() {
  const got = await api("/api/presets");
  if (!got.error) state.presets = got;
  const host = $("#preset-list");
  const btn = (name, saved) => h("span", { class: "preset" + (saved ? " saved" : "") },
    h("button", { type: "button", class: "preset-go", dataset: { preset: name }, text: name, title: describe(presetByName(name)), onclick: () => { loadSettings(presetByName(name)); toast(`Rules set to "${name}".`); } }),
    saved ? h("button", { type: "button", class: "chip-x", "aria-label": `Delete the preset ${name}`, title: "Delete this preset", text: "×",
      onclick: async ev => {
        const b = ev.currentTarget;
        if (!b.classList.contains("confirm")) { b.classList.add("confirm"); b.textContent = "delete?"; setTimeout(() => { b.classList.remove("confirm"); b.textContent = "×"; }, 3000); return; }
        await api("/api/presets", { name, settings: null }); loadPresets();
      } }) : null);
  host.replaceChildren(...Object.keys(state.presets.built_in || {}).map(n => btn(n, false)), ...Object.keys(state.presets.saved || {}).map(n => btn(n, true)));
  // The meter's quick-add row uses the same presets.
  $("#arm-presets").replaceChildren(...[...Object.keys(state.presets.built_in || {}), ...Object.keys(state.presets.saved || {})].map(n =>
    h("button", { type: "button", class: "ghost small", text: n, title: describe(presetByName(n)), onclick: () => addArm(n, presetByName(n)) })));
  rulesChanged();
}
const presetByName = n => (state.presets.built_in || {})[n] || (state.presets.saved || {})[n];

function wirePresetSave() {
  const form = $("#save-form"), name = $("#preset-name");
  $("#save-preset").addEventListener("click", () => { form.hidden = false; $("#save-preset").hidden = true; name.value = describe(state.settings); name.select(); name.focus(); });
  const close = () => { form.hidden = true; $("#save-preset").hidden = false; };
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const n = name.value.trim();
    if (!n) return;
    const got = await api("/api/presets", { name: n, settings: state.settings });
    if (got.error) { toast(got.error, "bad"); return; }
    close(); loadPresets(); toast(`Saved as "${n}".`);
  });
  $("#save-cancel").addEventListener("click", close);
  name.addEventListener("keydown", ev => { if (ev.key === "Escape") close(); });
}
