// Cleaning tab: the triage panel's keyboard (views/triage.py).
// Acts only while the triage panel is the top dialog and focus is not in a
// field; sends the key by name ("Space", "Arrow Right"), as views/triage.py
// expects it.
(function () {
  "use strict";

  const NAMES = { " ": "Space", "ArrowRight": "Arrow Right", "ArrowLeft": "Arrow Left" };
  const LETTERS = ["K", "X", "U", "R", "Z"];

  function topTriage() {
    const top = document.querySelector("#dialogs .dlg-back:last-child");
    if (!top) return null;
    return top.querySelector(".triage[data-triage]");
  }

  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const panel = topTriage();
    if (!panel) return;
    if (ev.target.closest("input,textarea,select,[contenteditable]")) return;
    if (document.querySelector(".zoom-back")) return;   // the full-size view is on top
    let key = NAMES[ev.key];
    if (!key) {
      const upper = (ev.key || "").length === 1 ? ev.key.toUpperCase() : "";
      if (LETTERS.includes(upper)) key = upper;
      else if (/^[1-9]$/.test(ev.key)) key = ev.key;
    }
    if (!key) return;
    ev.preventDefault();
    ev.stopPropagation();
    // A button clicked a moment ago keeps the focus, and Space would click it again.
    const active = document.activeElement;
    if (active && active.tagName === "BUTTON") active.blur();
    window.lookinAct("dlg:" + panel.getAttribute("data-triage"), "key", { key: key });
  });

  // A picker used with the mouse keeps the focus, and the keys would then
  // drive the picker instead of the queue: hand the focus back.
  document.addEventListener("change", (ev) => {
    const el = ev.target;
    if (el && el.tagName === "SELECT" && el.closest(".triage[data-triage]")) {
      setTimeout(() => el.blur(), 0);
    }
  });
})();
