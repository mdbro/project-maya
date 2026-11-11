// Formatted view of an answer: Markdown laid out as headings, lists, tables and code blocks, built from the
// answer's own tokens. Every visible character stays inside a span of the token that wrote it (class "t",
// data-i = the token's index), so clicking and colouring work exactly as in the token view. Markdown markers
// (#, **, `, |, list bullets) are left out of the text; a token made only of markers is not shown here, and the
// token view shows every token. Only DOM nodes are built: the model's text is never parsed as HTML.
"use strict";

const Formatted = (() => {
  const BLOCK = {
    fence: /^(\s{0,3})(`{3,}|~{3,})[ \t]*([^\s`]*)/,
    heading: /^\s{0,3}(#{1,6})[ \t]+/,
    hr: /^\s{0,3}([-*_])([ \t]*\1){2,}[ \t]*$/,
    quote: /^\s{0,3}>[ \t]?/,
    item: /^(\s*)([-*+]|\d{1,3}[.)])[ \t]+/,
    tableSep: /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/,
  };

  function render(records, from, to, o) {
    let text = "";
    const owner = [];
    for (let i = from; i < to; i++) {
      if (o.hidden(records[i])) continue;
      const t = records[i].text || "";
      text += t;
      for (let c = 0; c < t.length; c++) owner.push(i);
    }
    const out = document.createDocumentFragment();

    // Characters [a, b) as token fragments, one span per run of characters from the same token.
    function emit(parent, a, b) {
      let s = a;
      while (s < b) {
        const i = owner[s];
        let e = s + 1;
        while (e < b && owner[e] === i) e++;
        parent.append(o.span(i, text.slice(s, e)));
        s = e;
      }
    }
    // Inline Markdown in [a, b): code spans, strong and emphasis. Anything unmatched is plain text.
    function inline(parent, a, b) {
      let i = a, plain = a;
      const flush = to => { if (to > plain) emit(parent, plain, to); };
      while (i < b) {
        const c = text[i];
        if (c === "`") {
          let n = 1;
          while (i + n < b && text[i + n] === "`") n++;
          const close = text.indexOf("`".repeat(n), i + n);
          if (close > i + n && close + n <= b && text[close + n] !== "`") {
            flush(i);
            const code = h("code", { class: "md-ic" });
            emit(code, i + n, close);
            parent.append(code);
            i = plain = close + n;
            continue;
          }
          i += n;
          continue;
        }
        if (c === "*" || c === "_") {
          let n = 1;
          while (n < 3 && text[i + n] === c) n++;
          const next = text[i + n], prev = i > a ? text[i - 1] : " ";
          const opens = next && !/\s/.test(next) && !(c === "_" && /[\p{L}\p{N}]/u.test(prev));
          if (opens) {
            const marker = c.repeat(n);
            let s = i + n + 1, close = -1;
            while (s < b) {
              s = text.indexOf(marker, s);
              if (s < 0 || s + n > b) break;
              const after = text[s + n] || " ";
              if (!/\s/.test(text[s - 1]) && after !== c && !(c === "_" && /[\p{L}\p{N}]/u.test(after))) { close = s; break; }
              s++;
            }
            if (close > 0) {
              flush(i);
              let node = h(n === 1 ? "em" : "strong"), inner = node;
              if (n === 3) { inner = h("em"); node.append(inner); }
              inline(inner, i + n, close);
              parent.append(node);
              i = plain = close + n;
              continue;
            }
          }
          i += n;
          continue;
        }
        i++;
      }
      flush(b);
    }

    const lines = [];
    for (let a = 0; a <= text.length;) {
      let b = text.indexOf("\n", a);
      if (b < 0) b = text.length;
      lines.push({ a, b, s: text.slice(a, b) });
      a = b + 1;
    }
    const startsBlock = s => BLOCK.fence.test(s) || BLOCK.heading.test(s) || BLOCK.hr.test(s) || BLOCK.quote.test(s) || BLOCK.item.test(s);
    const blank = s => /^\s*$/.test(s);

    for (let k = 0; k < lines.length;) {
      const L = lines[k], s = L.s;
      let m;
      if (blank(s)) { k++; continue; }
      if ((m = BLOCK.fence.exec(s))) {
        const fence = m[2];
        const closing = new RegExp(`^\\s{0,3}${fence[0] === "`" ? "`" : "~"}{${fence.length},}\\s*$`);
        let j = k + 1;
        while (j < lines.length && !closing.test(lines[j].s)) j++;
        const a = k + 1 < lines.length ? lines[k + 1].a : L.b, b = j > k + 1 ? lines[j - 1].b : a;
        const pre = h("pre"), code = h("code");
        emit(code, a, Math.max(a, b));
        pre.append(code);
        out.append(h("div", { class: "md-code" }, m[3] ? h("div", { class: "md-lang", text: m[3] }) : null, pre));
        k = j + 1;
        continue;
      }
      if ((m = BLOCK.heading.exec(s))) {
        let b = L.b;
        const tail = /[ \t]+#+[ \t]*$/.exec(s);
        if (tail) b = L.a + tail.index;
        const node = h("div", { class: `md-h md-h${m[1].length}` });
        inline(node, L.a + m[0].length, b);
        out.append(node);
        k++;
        continue;
      }
      if (BLOCK.hr.test(s)) { out.append(h("hr", { class: "md-hr" })); k++; continue; }
      if (BLOCK.quote.test(s)) {
        const node = h("div", { class: "md-quote" });
        while (k < lines.length && (m = BLOCK.quote.exec(lines[k].s))) {
          if (node.childNodes.length) node.append("\n");
          inline(node, lines[k].a + m[0].length, lines[k].b);
          k++;
        }
        out.append(node);
        continue;
      }
      if ((m = BLOCK.item.exec(s))) {
        // One block per item, indented by its leading spaces; indented lines that follow belong to it.
        let b = L.b, j = k + 1;
        while (j < lines.length && !blank(lines[j].s) && /^\s{2,}/.test(lines[j].s) && !startsBlock(lines[j].s)) { b = lines[j].b; j++; }
        const ordered = /\d/.test(m[2]);
        const node = h("div", { class: "md-li", style: `--indent:${Math.min(6, Math.floor(m[1].replace(/\t/g, "    ").length / 2))}` },
          h("span", { class: "md-bullet", "aria-hidden": "true", text: ordered ? m[2] : "•" }));
        const body = h("div", { class: "md-li-body" });
        inline(body, L.a + m[0].length, b);
        node.append(body);
        out.append(node);
        k = j;
        continue;
      }
      if (/^\s*\|/.test(s) && k + 1 < lines.length && BLOCK.tableSep.test(lines[k + 1].s)) {
        const table = h("table", { class: "md-table" });
        let j = k, head = true;
        while (j < lines.length && /^\s*\|/.test(lines[j].s)) {
          if (j === k + 1) { j++; continue; }                         // the separator row
          const row = h("tr");
          const line = lines[j];
          const cuts = [];
          for (let c = 0; c < line.s.length; c++) if (line.s[c] === "|" && line.s[c - 1] !== "\\") cuts.push(c);
          for (let q = 0; q + 1 < cuts.length; q++) {
            let a = line.a + cuts[q] + 1, b = line.a + cuts[q + 1];
            while (a < b && text[a] === " ") a++;
            while (b > a && text[b - 1] === " ") b--;
            const cell = h(head ? "th" : "td");
            inline(cell, a, b);
            row.append(cell);
          }
          table.append(row);
          head = false;
          j++;
        }
        out.append(h("div", { class: "md-table-wrap" }, table));
        k = j;
        continue;
      }
      // A paragraph: lines up to a blank line or the start of another block. Its line breaks are kept.
      let j = k + 1;
      while (j < lines.length && !blank(lines[j].s) && !startsBlock(lines[j].s) && !(/^\s*\|/.test(lines[j].s) && j + 1 < lines.length && BLOCK.tableSep.test(lines[j + 1].s))) j++;
      const node = h("div", { class: "md-p" });
      inline(node, L.a, lines[j - 1].b);
      out.append(node);
      k = j;
    }
    return out;
  }
  return { render };
})();
