// Notes tab: the editor (views/notes.py).
//
// The note is edited as formatted text in a contenteditable; the server
// keeps it as Markdown. What the page shows is the server's HTML of it,
// carried in data-html and taken whenever data-rev moves (a regeneration,
// another tab, an agent) -- otherwise the editor is the user's and every
// re-render leaves it alone (data-keep, see app.js morphNode).
//
// Saved as the user types (SAVE_MS after the last key) and when the editor
// loses the focus. A "- ", "* ", "1. " or "#".."###" typed at the start of
// a line turns it into a list or a heading, as a chat box does. A paste is
// taken as plain text: formatting from another page would only be lost at
// the next save anyway.
(function () {
  "use strict";

  const SAVE_MS = 700;
  const BLOCKS = { h1: "H1", h2: "H2", h3: "H3", p: "P", blockquote: "BLOCKQUOTE" };
  let timer = null;

  function editorOf(el) {
    const card = el && el.closest ? el.closest(".card") : null;
    return card ? card.querySelector(".notes-editor") : document.querySelector(".notes-editor");
  }

  function payload(ed) {
    return { html: ed.innerHTML, rev: parseInt(ed.getAttribute("data-rev") || "0", 10) };
  }

  function save(ed, action) {
    clearTimeout(timer);
    timer = null;
    ed._lkDirty = false;
    return window.lookinAct("notes", action || "save", payload(ed));
  }

  function load(root) {
    (root || document).querySelectorAll(".notes-editor").forEach((ed) => {
      const rev = ed.getAttribute("data-rev");
      if (ed._lkRev === rev) return;
      ed._lkRev = rev;
      ed.innerHTML = ed.getAttribute("data-html") || "";
      ed._lkDirty = false;
    });
  }
  window.lookinInit(load);

  function inEditor(node) {
    const el = node && (node.nodeType === 1 ? node : node.parentElement);
    return el ? el.closest(".notes-editor") : null;
  }

  document.addEventListener("input", (ev) => {
    const ed = inEditor(ev.target);
    if (!ed) return;
    ed._lkDirty = true;
    clearTimeout(timer);
    timer = setTimeout(() => save(ed), SAVE_MS);
  });

  document.addEventListener("focusout", (ev) => {
    const ed = inEditor(ev.target);
    if (ed && ed._lkDirty) save(ed);
  });

  document.addEventListener("focusin", (ev) => {
    if (inEditor(ev.target)) {
      try { document.execCommand("defaultParagraphSeparator", false, "p"); } catch (e) { /* older engines */ }
    }
  });

  document.addEventListener("paste", (ev) => {
    if (!inEditor(ev.target)) return;
    const text = ev.clipboardData ? ev.clipboardData.getData("text/plain") : "";
    if (!text) return;
    ev.preventDefault();
    document.execCommand("insertText", false, text);
  });

  // ---- the toolbar ----------------------------------------------------------

  // Keep the selection in the editor when a toolbar button is pressed.
  document.addEventListener("mousedown", (ev) => {
    if (ev.target.closest("[data-notes-cmd]")) ev.preventDefault();
  });

  function wrapCode() {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return;
    const range = sel.getRangeAt(0);
    const code = document.createElement("code");
    code.textContent = range.toString();
    range.deleteContents();
    range.insertNode(code);
    sel.removeAllRanges();
  }

  function run(cmd, ed) {
    ed.focus();
    if (BLOCKS[cmd]) {
      document.execCommand("formatBlock", false, BLOCKS[cmd]);
    } else if (cmd === "link") {
      const url = window.prompt("Link to (https://...)", "https://");
      if (url && /^(https?:\/\/|mailto:)/i.test(url.trim())) document.execCommand("createLink", false, url.trim());
    } else if (cmd === "code") {
      wrapCode();
    } else {
      document.execCommand(cmd, false, null);
      if (cmd === "removeFormat") document.execCommand("unlink", false, null);
    }
    ed._lkDirty = true;
    clearTimeout(timer);
    timer = setTimeout(() => save(ed), SAVE_MS);
  }

  document.addEventListener("click", (ev) => {
    const tool = ev.target.closest("[data-notes-cmd]");
    if (tool) {
      const ed = editorOf(tool);
      if (ed) run(tool.getAttribute("data-notes-cmd"), ed);
      return;
    }
    // Buttons that need what is in the editor now, not what was saved.
    const btn = ev.target.closest("[data-notes-act]");
    if (btn && !btn.disabled) {
      const ed = editorOf(btn);
      if (ed) save(ed, btn.getAttribute("data-notes-act"));
    }
  });

  // ---- what a chat box does with a few characters at the start of a line ----

  const SHORTCUTS = [
    [/^[-*]$/, "insertUnorderedList"],
    [/^1[.)]$/, "insertOrderedList"],
    [/^###$/, "h3"],
    [/^##$/, "h2"],
    [/^#$/, "h1"],
    [/^>$/, "blockquote"],
  ];

  document.addEventListener("keydown", (ev) => {
    const ed = inEditor(ev.target);
    if (!ed) return;
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === "s" || ev.key === "S")) {
      ev.preventDefault();      // the browser's "save page": save the note instead
      save(ed);
      return;
    }
    if (ev.key !== " " || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const sel = window.getSelection();
    if (!sel || !sel.isCollapsed || sel.rangeCount === 0) return;
    const node = sel.anchorNode;
    if (!node || node.nodeType !== 3) return;
    // Already a list item, a heading or code: a "- " there is text.
    if (node.parentElement.closest("li,h1,h2,h3,h4,h5,h6,pre,code")) return;
    const before = node.nodeValue.slice(0, sel.anchorOffset);
    const lineStart = !node.previousSibling || node.previousSibling.nodeName === "BR";
    if (!lineStart) return;
    for (const [re, cmd] of SHORTCUTS) {
      if (!re.test(before)) continue;
      ev.preventDefault();
      // Delete the marker through the selection, as a keystroke would:
      // emptying the text node by hand left the caret nowhere, and the
      // list or heading then landed on the paragraph above.
      const range = document.createRange();
      range.setStart(node, 0);
      range.setEnd(node, sel.anchorOffset);
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand("delete", false, null);
      run(cmd, ed);
      return;
    }
  });
})();
