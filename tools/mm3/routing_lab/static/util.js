// Routing lab: small helpers shared by every part of the page. Model output is only ever placed with textContent
// or as DOM nodes; nothing on this page parses a string as HTML.
"use strict";

const $ = sel => document.querySelector(sel);
const $$ = sel => [...document.querySelectorAll(sel)];
const { fmt, pct } = Charts;

function h(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k === "dataset") Object.assign(n.dataset, v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k in n && typeof v !== "string") n[k] = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid);
  return n;
}

async function api(path, body) {
  let r;
  try {
    r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (err) {
    return { error: "the lab's server does not answer" };
  }
  const data = await r.json().catch(() => ({ error: `the lab's server answered ${r.status}` }));
  if (!r.ok && data && !data.error) data.error = `the lab's server answered ${r.status}`;
  return data;
}

// The browser's own storage: the conversation, the rules and the layout survive a reload. It can be missing or
// full (a private window); the page then works as before, without memory.
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem("lab." + key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem("lab." + key, JSON.stringify(value)); } catch (e) { /* no storage: nothing is kept */ }
  },
};

const clone = x => JSON.parse(JSON.stringify(x));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + "s")}`;

// A segmented control: one click per choice, the current one marked. `choices` = [[value, label, title]].
function segmented(choices, value, onpick, label) {
  const box = h("div", { class: "seg", role: "radiogroup", "aria-label": label || "" });
  const mark = v => box.querySelectorAll("button").forEach(b => { const on = b.dataset.v === String(v); b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
  for (const [v, text, title] of choices) {
    box.append(h("button", { type: "button", role: "radio", dataset: { v: String(v) }, text, title: title || null,
      onclick: () => { mark(v); onpick(v); } }));
  }
  mark(value);
  box.mark = mark;
  return box;
}

// A short message in place of alert(): shown at the bottom of the window for a few seconds.
let toastTimer = 0;
function toast(text, kind = "") {
  const t = $("#toast");
  t.textContent = text;
  t.className = "toast " + kind;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === "bad" ? 7000 : 3500);
}

// Rebuild part of a scrolling area without moving what the reader is looking at: the first keyed element that is
// visible stays at the same height on the screen.
function keepPlace(scroller, fn) {
  const top = scroller.getBoundingClientRect().top;
  const anchor = [...scroller.querySelectorAll("[data-key]")].find(n => n.getBoundingClientRect().bottom > top + 4);
  const key = anchor && anchor.dataset.key, offset = anchor ? anchor.getBoundingClientRect().top - top : 0;
  fn();
  if (!key) return;
  const again = scroller.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (again) scroller.scrollTop += again.getBoundingClientRect().top - top - offset;
}
