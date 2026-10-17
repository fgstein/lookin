/* Lookin web runtime.
 *
 * The server renders HTML; this file only
 *   - keeps one session id per browser tab (sessionStorage, with a check
 *     that a duplicated tab does not share its parent's session),
 *   - turns data-act / data-change / data-input / data-submit attributes
 *     into POST /api/act,
 *   - applies what comes back by morphing the named regions in place (so
 *     scroll positions, focus and half-typed text survive a re-render),
 *   - listens to /api/events for what changed in the background,
 *   - and does the few purely local things: image zoom, drag and drop,
 *     toasts, copy to clipboard.
 *
 * Nothing here talks to any host but the one that served the page.
 */
(function () {
  "use strict";

  // ---- session id -------------------------------------------------------
  const SID_KEY = "lookin-sid";
  const tabToken = Math.random().toString(36).slice(2);
  let sid = null;

  function newSid() {
    const a = new Uint8Array(12);
    crypto.getRandomValues(a);
    return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function claimSid() {
    // A duplicated tab copies sessionStorage, and with it the session id:
    // two tabs on one session would fight over the same filters. Ask the
    // other tabs whether one of them already holds this id.
    return new Promise((resolve) => {
      let stored = null;
      try { stored = sessionStorage.getItem(SID_KEY); } catch (e) { stored = null; }
      if (!stored || typeof BroadcastChannel === "undefined") {
        resolve(stored || newSid());
        return;
      }
      let taken = false;
      const ch = new BroadcastChannel("lookin-sessions");
      ch.onmessage = (ev) => {
        if (ev.data && ev.data.type === "taken" && ev.data.sid === stored && ev.data.token !== tabToken) taken = true;
      };
      ch.postMessage({ type: "claim", sid: stored, token: tabToken });
      setTimeout(() => { ch.close(); resolve(taken ? newSid() : stored); }, 120);
    });
  }

  function listenForClaims() {
    if (typeof BroadcastChannel === "undefined") return;
    const ch = new BroadcastChannel("lookin-sessions");
    ch.onmessage = (ev) => {
      if (ev.data && ev.data.type === "claim" && ev.data.sid === sid && ev.data.token !== tabToken) {
        ch.postMessage({ type: "taken", sid: sid, token: tabToken });
      }
    };
  }

  // ---- requests ---------------------------------------------------------
  let inflight = 0;
  let slowTimer = null;

  function busyOn() {
    inflight += 1;
    if (!slowTimer) slowTimer = setTimeout(() => document.body.classList.add("req"), 350);
  }
  function busyOff() {
    inflight = Math.max(0, inflight - 1);
    if (!inflight) {
      clearTimeout(slowTimer); slowTimer = null;
      document.body.classList.remove("req");
    }
  }

  async function post(url, body) {
    busyOn();
    try {
      const r = await fetch(url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({ vw: window.innerWidth, vh: window.innerHeight }, body)),
      });
      if (r.status === 410) { await reboot(); return null; }
      if (!r.ok) { toast("The server answered " + r.status, "error"); return null; }
      return await r.json();
    } catch (e) {
      toast("Lost contact with the Lookin server.", "error");
      return null;
    } finally {
      busyOff();
    }
  }

  // Actions are sent one at a time, in order: two quick clicks must reach
  // the server in the order they were made.
  let chain = Promise.resolve();
  function act(target, action, args) {
    chain = chain.then(async () => {
      const resp = await post("/api/act", { sid, target, action, args: args || {} });
      if (resp) apply(resp);
    }).catch((e) => {
      // One failed apply must not stall every later action behind it.
      console.error("lookin: applying " + target + ":" + action + " failed", e);
    });
    return chain;
  }
  window.lookinAct = act;

  async function reboot() {
    sid = newSid();
    try { sessionStorage.setItem(SID_KEY, sid); } catch (e) { /* private mode */ }
    await boot();
  }

  let booted = false;
  async function boot() {
    connectEvents();
    const resp = await post("/api/boot", { sid, search: location.search });
    if (resp) apply(resp);
    booted = true;
  }

  // ---- applying a response --------------------------------------------------
  let currentView = null;

  function apply(resp) {
    if (resp.page !== undefined) {
      const app = document.getElementById("app");
      if (currentView !== resp.view || !app.firstElementChild || app.querySelector(".boot")) {
        app.innerHTML = resp.page;
        app.querySelectorAll("[autofocus]").forEach((el) => el.focus());
      } else {
        morphInner(app, resp.page);
      }
      currentView = resp.view;
    }
    if (resp.view) currentView = resp.view;
    if (resp.html) {
      for (const id of Object.keys(resp.html)) {
        const el = document.getElementById(id);
        if (el) {
          const wasEmpty = id === "dialogs" && !el.firstElementChild;
          morphInner(el, resp.html[id]);
          if (id === "dialogs") afterDialogs(wasEmpty);
        }
      }
    }
    if (resp.title) document.title = resp.title;
    if (resp.cmds) resp.cmds.forEach(runCmd);
    document.body.dataset.mode = resp.mode || document.body.dataset.mode || "";
    initDynamic();
  }

  function afterDialogs(wasEmpty) {
    const layer = document.getElementById("dialogs");
    const top = layer.lastElementChild;
    if (top && wasEmpty) {
      const f = top.querySelector("[autofocus]") || top.querySelector("input,select,textarea,button:not(.dlg-x)");
      if (f) f.focus();
    }
  }

  function runCmd(c) {
    if (c.toast) toast(c.toast, c.kind || "ok");
    if (c.open) window.open(c.open, "_blank", "noopener");
    if (c.copy !== undefined) copyText(c.copy);
    if (c.download) {
      const a = document.createElement("a");
      a.href = c.download; a.download = c.filename || "";
      document.body.appendChild(a); a.click(); a.remove();
    }
    if (c.navigate) location.href = c.navigate;
    if (c.replace_url) history.replaceState(null, "", c.replace_url);
    if (c.focus) { const el = document.querySelector(c.focus); if (el) el.focus(); }
    if (c.scroll) { const el = document.querySelector(c.scroll); if (el) el.scrollIntoView({ block: "nearest" }); }
    if (c.event) document.dispatchEvent(new CustomEvent("lookin:" + c.event, { detail: c.detail }));
  }

  // ---- morphing -----------------------------------------------------------
  // Patches `el`'s children to match `html`, keeping every element that
  // can be kept: what the user is typing, where a list is scrolled, which
  // <details> are open, a canvas that draws itself.
  function morphInner(el, html) {
    const tpl = document.createElement("template");
    tpl.innerHTML = html;
    morphChildren(el, tpl.content);
  }

  function keyOf(n) {
    if (n.nodeType !== 1) return null;
    return n.id || n.getAttribute("data-key") || null;
  }

  function morphChildren(from, to) {
    const oldKeyed = new Map();
    for (const c of Array.from(from.childNodes)) {
      const k = keyOf(c);
      if (k) oldKeyed.set(k, c);
    }
    let cursor = from.firstChild;
    for (const nc of Array.from(to.childNodes)) {
      const k = keyOf(nc);
      let match = null;
      if (k && oldKeyed.has(k)) {
        match = oldKeyed.get(k);
        oldKeyed.delete(k);
      } else if (!k && cursor && !keyOf(cursor) && sameKind(cursor, nc)) {
        match = cursor;
      }
      if (match) {
        if (match !== cursor) from.insertBefore(match, cursor);
        morphNode(match, nc);
        cursor = match.nextSibling;
      } else {
        from.insertBefore(nc, cursor);
      }
    }
    while (cursor) {
      const next = cursor.nextSibling;
      from.removeChild(cursor);
      cursor = next;
    }
  }

  function sameKind(a, b) {
    if (a.nodeType !== b.nodeType) return false;
    if (a.nodeType === 1) return a.tagName === b.tagName && (a.type || "") === (b.type || "");
    return true;
  }

  function morphNode(a, b) {
    if (a.nodeType === 3 || a.nodeType === 8) {
      if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue;
      return;
    }
    if (a.nodeType !== 1) return;
    if (a.hasAttribute("data-static") && b.hasAttribute("data-static") &&
        a.getAttribute("data-static") === b.getAttribute("data-static")) return;
    // attributes
    for (const at of Array.from(a.attributes)) {
      if (!b.hasAttribute(at.name)) a.removeAttribute(at.name);
    }
    for (const at of Array.from(b.attributes)) {
      if (a.getAttribute(at.name) !== at.value) a.setAttribute(at.name, at.value);
    }
    const tag = a.tagName;
    if (tag === "INPUT") {
      const focused = document.activeElement === a;
      if (a.type === "checkbox" || a.type === "radio") {
        a.checked = b.hasAttribute("checked");
      } else if (!focused || b.hasAttribute("data-force")) {
        const v = b.getAttribute("value") || "";
        if (a.value !== v) a.value = v;
      }
      return;
    }
    if (tag === "TEXTAREA") {
      if (document.activeElement !== a) {
        const v = b.textContent;
        if (a.value !== v) a.value = v;
      }
      return;
    }
    if (tag === "DETAILS") {
      // the server decides only when it says so
      if (b.hasAttribute("data-force")) a.open = b.hasAttribute("open");
    }
    if (a.hasAttribute("data-keep")) return;   // e.g. a canvas that draws itself
    morphChildren(a, b);
    if (tag === "SELECT") {
      const sel = b.querySelector("option[selected]");
      const v = sel ? sel.getAttribute("value") : (a.options[0] ? a.options[0].value : "");
      if (a.value !== v) a.value = v;
    }
  }

  // ---- events from the server -----------------------------------------------
  let es = null;
  let pendingNames = new Set();
  let renderTimer = null;

  function connectEvents() {
    if (es) es.close();
    es = new EventSource("/api/events?sid=" + encodeURIComponent(sid));
    es.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.cmds) msg.cmds.forEach(runCmd);
      if (msg.dirty) {
        msg.dirty.forEach((n) => pendingNames.add(n));
        if (!renderTimer) renderTimer = setTimeout(flushRender, 60);
      }
    };
    es.onerror = () => { /* the browser reconnects by itself */ };
    es.onopen = () => {
      // A reconnect may have missed pushes: redraw what is on screen.
      if (booted) { pendingNames.add("*"); if (!renderTimer) renderTimer = setTimeout(flushRender, 60); }
    };
  }

  async function flushRender() {
    renderTimer = null;
    const names = Array.from(pendingNames);
    pendingNames = new Set();
    const relevant = names.filter((n) => n === "*" || n === "page" || n === "chrome" || n === "dialogs" || n === currentView);
    if (!relevant.length) return;
    try {
      const r = await fetch("/api/render?sid=" + encodeURIComponent(sid) + "&names=" + encodeURIComponent(relevant.join(",")));
      if (r.status === 410) { await reboot(); return; }
      if (r.ok) apply(await r.json());
    } catch (e) { /* next push will retry */ }
  }

  // ---- delegated DOM events ------------------------------------------------
  function parseTarget(spec) {
    const i = spec.indexOf(":");
    if (spec.startsWith("dlg:")) {
      const j = spec.indexOf(":", 4);
      return [spec.slice(0, j), spec.slice(j + 1)];
    }
    return [spec.slice(0, i), spec.slice(i + 1)];
  }

  function argsOf(el) {
    const raw = el.getAttribute("data-args");
    if (!raw) return {};
    try { return JSON.parse(raw); } catch (e) { return {}; }
  }

  function fire(el, attr, extra) {
    const spec = el.getAttribute(attr);
    if (!spec) return;
    const [target, action] = parseTarget(spec);
    act(target, action, Object.assign(argsOf(el), extra || {}));
  }

  function valueOf(el) {
    if (el.type === "checkbox") return el.checked;
    if (el.multiple) return Array.from(el.selectedOptions).map((o) => o.value);
    return el.value;
  }

  document.addEventListener("click", (ev) => {
    const zoomEl = ev.target.closest("img[data-zoom]");
    if (zoomEl && !ev.target.closest("[data-act]")) { ev.preventDefault(); openZoom(zoomEl); return; }
    const tabLink = ev.target.closest("a[data-tab]");
    if (tabLink) {
      ev.preventDefault();
      act("app", "tab", { tab: tabLink.getAttribute("data-tab") });
      return;
    }
    const homeLink = ev.target.closest("a[data-home]");
    if (homeLink && !ev.ctrlKey && !ev.metaKey && !ev.shiftKey && ev.button === 0) {
      ev.preventDefault();
      act("app", "home", {});
      return;
    }
    const copyEl = ev.target.closest("[data-copy]");
    if (copyEl) { copyText(copyEl.getAttribute("data-copy")); toast("Copied.", "ok"); return; }
    const menuBtn = ev.target.closest("[data-menu]");
    if (menuBtn) { toggleMenu(menuBtn); ev.preventDefault(); return; }
    const el = ev.target.closest("[data-act]");
    closeMenus(el);
    if (!el || el.disabled) return;
    if (el.tagName === "A" && !el.getAttribute("href")) ev.preventDefault();
    if (el.tagName === "INPUT" && (el.type === "checkbox" || el.type === "radio")) return;
    ev.preventDefault();
    const extra = {};
    if (ev.shiftKey) extra.shift = true;
    if (ev.ctrlKey || ev.metaKey) extra.ctrl = true;
    fire(el, "data-act", extra);
  });

  document.addEventListener("change", (ev) => {
    const el = ev.target.closest("[data-change]");
    if (el) fire(el, "data-change", { value: valueOf(ev.target) });
  });

  const inputTimers = new WeakMap();
  document.addEventListener("input", (ev) => {
    // A dual range slider shows "[start, end]" while it is dragged; the
    // server only hears about it on release (data-change).
    const rng = ev.target.closest("input[type=range][data-out]");
    if (rng) {
      const pair = rng.parentElement.querySelectorAll("input[type=range]");
      const out = document.getElementById(rng.getAttribute("data-out"));
      if (pair.length === 2) {
        let a = parseFloat(pair[0].value), b = parseFloat(pair[1].value);
        if (a > b) { if (rng === pair[0]) { pair[0].value = b; a = b; } else { pair[1].value = a; b = a; } }
        if (out) out.textContent = "[" + a.toFixed(2) + ", " + b.toFixed(2) + "]";
      } else if (out) out.textContent = parseFloat(rng.value).toFixed(2);
    }
    const el = ev.target.closest("[data-input]");
    if (!el) return;
    clearTimeout(inputTimers.get(el));
    inputTimers.set(el, setTimeout(() => fire(el, "data-input", { value: valueOf(el) }), 320));
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      if (closeZoom() || closeMenus()) { ev.preventDefault(); return; }
      const top = document.querySelector("#dialogs .dlg-back:last-child");
      if (top) { ev.preventDefault(); act("dlg:" + top.getAttribute("data-dlg"), "close", {}); return; }
    }
    if (ev.key === "Enter") {
      const el = ev.target.closest("[data-enter]");
      if (el) {
        ev.preventDefault();
        clearTimeout(inputTimers.get(el));
        fire(el, "data-enter", { value: valueOf(el) });
        return;
      }
    }
    // view shortcuts (e.g. the carousel's arrows), never while typing
    const typing = ev.target.closest("input,textarea,select,[contenteditable]");
    if (typing || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const scopes = Array.from(document.querySelectorAll("[data-keys]"));
    const scope = scopes.length ? scopes[scopes.length - 1] : null;
    if (!scope) return;
    const allowed = (scope.getAttribute("data-keylist") || "").split(" ");
    if (!allowed.includes(ev.key)) return;
    ev.preventDefault();
    const [target, action] = parseTarget(scope.getAttribute("data-keys"));
    act(target, action, { key: ev.key });
  });

  document.addEventListener("submit", (ev) => {
    const form = ev.target.closest("form[data-submit]");
    if (!form) return;
    ev.preventDefault();
    const values = {};
    for (const el of Array.from(form.elements)) {
      if (!el.name || el.disabled) continue;
      if (el.type === "checkbox") {
        if (el.hasAttribute("data-many")) {
          values[el.name] = values[el.name] || [];
          if (el.checked) values[el.name].push(el.value);
        } else values[el.name] = el.checked;
      } else if (el.type === "radio") {
        if (el.checked) values[el.name] = el.value;
      } else if (el.multiple) {
        values[el.name] = Array.from(el.selectedOptions).map((o) => o.value);
      } else if (el.tagName !== "BUTTON") {
        values[el.name] = el.value;
      }
    }
    fire(form, "data-submit", values);
  });

  // ---- popup menus ----------------------------------------------------------
  function toggleMenu(btn) {
    const menu = btn.nextElementSibling;
    if (!menu || !menu.classList.contains("menu")) return;
    const open = !menu.classList.contains("open");
    closeMenus();
    if (open) {
      menu.classList.add("open");
      const r = btn.getBoundingClientRect();
      menu.style.minWidth = "180px";
      const right = window.innerWidth - r.right;
      if (r.left + 220 > window.innerWidth) { menu.style.right = "0"; menu.style.left = "auto"; }
      if (right < 0) menu.style.right = "0";
    }
  }
  function closeMenus(keep) {
    let closed = false;
    document.querySelectorAll(".menu.open").forEach((m) => {
      if (keep && m.contains(keep)) return;
      m.classList.remove("open"); closed = true;
    });
    return closed;
  }
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest("[data-menu]") && !ev.target.closest(".menu")) closeMenus();
  }, true);

  // ---- drag and drop ------------------------------------------------------
  document.addEventListener("dragstart", (ev) => {
    const el = ev.target.closest("[data-drag]");
    if (!el) return;
    ev.dataTransfer.effectAllowed = "copyMove";
    ev.dataTransfer.setData("application/x-lookin", JSON.stringify({
      kind: el.getAttribute("data-drag-kind") || "", key: el.getAttribute("data-drag"),
    }));
    ev.dataTransfer.setData("text/plain", el.getAttribute("data-drag-label") || el.textContent.trim().slice(0, 60));
    document.body.classList.add("dragging");
  });
  document.addEventListener("dragend", () => {
    document.body.classList.remove("dragging");
    document.querySelectorAll(".drop-on").forEach((d) => d.classList.remove("drop-on"));
  });
  function dropTarget(ev) {
    const el = ev.target.closest("[data-drop]");
    if (!el) return null;
    const kinds = (el.getAttribute("data-drop-kinds") || "").split(" ").filter(Boolean);
    return { el, kinds };
  }
  document.addEventListener("dragover", (ev) => {
    const t = dropTarget(ev);
    if (!t) return;
    ev.preventDefault();
    document.querySelectorAll(".drop-on").forEach((d) => { if (d !== t.el) d.classList.remove("drop-on"); });
    t.el.classList.add("drop-on");
  });
  document.addEventListener("dragleave", (ev) => {
    const t = dropTarget(ev);
    if (t && !t.el.contains(ev.relatedTarget)) t.el.classList.remove("drop-on");
  });
  document.addEventListener("drop", (ev) => {
    const t = dropTarget(ev);
    if (!t) return;
    ev.preventDefault();
    t.el.classList.remove("drop-on");
    let payload = null;
    try { payload = JSON.parse(ev.dataTransfer.getData("application/x-lookin")); } catch (e) { return; }
    if (!payload) return;
    if (t.kinds.length && !t.kinds.includes(payload.kind)) {
      // dropped on the wrong kind of target: hand it to the closest ancestor that takes it
      let up = t.el.parentElement ? t.el.parentElement.closest("[data-drop]") : null;
      while (up) {
        const k = (up.getAttribute("data-drop-kinds") || "").split(" ").filter(Boolean);
        if (!k.length || k.includes(payload.kind)) break;
        up = up.parentElement ? up.parentElement.closest("[data-drop]") : null;
      }
      if (!up) return;
      fire(up, "data-drop", { kind: payload.kind, key: payload.key });
      return;
    }
    fire(t.el, "data-drop", { kind: payload.kind, key: payload.key });
  });

  // ---- zoom -------------------------------------------------------------
  function openZoom(img) {
    const src = img.getAttribute("data-zoom-src") || img.currentSrc || img.src;
    const title = img.getAttribute("data-title") || img.alt || "image";
    const z = document.getElementById("zoom");
    const fname = "lookin_" + title.replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 60) + ".png";
    z.innerHTML =
      '<div class="zoom-back"><div class="zoom-bar"><span class="zoom-t"></span>' +
      '<a class="btn btn-sm" target="_blank" rel="noopener">Open in a new tab</a>' +
      '<a class="btn btn-sm">Save</a>' +
      '<button type="button" class="btn btn-sm btn-primary zoom-close">Close</button></div>' +
      '<div class="zoom-img"><img alt=""></div></div>';
    z.querySelector(".zoom-t").textContent = title;
    const links = z.querySelectorAll("a");
    links[0].href = src;
    links[1].href = src; links[1].download = fname;
    const big = z.querySelector(".zoom-img img");
    big.src = src;
    big.alt = title;
    if (img.hasAttribute("data-pixelated")) big.classList.add("pix");
    z.querySelector(".zoom-close").onclick = closeZoom;
    z.querySelector(".zoom-back").onclick = (ev) => { if (ev.target.classList.contains("zoom-back")) closeZoom(); };
  }
  function closeZoom() {
    const z = document.getElementById("zoom");
    if (!z.firstChild) return false;
    z.innerHTML = "";
    return true;
  }

  // ---- toasts and clipboard ----------------------------------------------
  function toast(text, kind) {
    const box = document.getElementById("toasts");
    if (!box) return;
    const t = document.createElement("div");
    t.className = "toast toast-" + (kind || "ok");
    t.textContent = text;
    t.onclick = () => t.remove();
    box.appendChild(t);
    setTimeout(() => t.remove(), kind === "error" ? 9000 : 4500);
  }
  window.lookinToast = toast;

  function copyText(text) {
    // navigator.clipboard only exists on https or localhost; Lookin is
    // often reached as http://host:port, so fall back to execCommand.
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
    } else legacyCopy(text);
  }
  function legacyCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing more to try */ }
    ta.remove();
  }

  // ---- widgets that need a little script after each render ---------------
  const initializers = [];
  window.lookinInit = (fn) => initializers.push(fn);
  function initDynamic() {
    initializers.forEach((fn) => { try { fn(document); } catch (e) { console.error(e); } });
  }

  // ---- start --------------------------------------------------------------
  window.addEventListener("DOMContentLoaded", async () => {
    sid = await claimSid();
    try { sessionStorage.setItem(SID_KEY, sid); } catch (e) { /* private mode */ }
    window.lookinSid = () => sid;
    listenForClaims();
    await boot();
  });
})();
