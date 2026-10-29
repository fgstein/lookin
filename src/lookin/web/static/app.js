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
    if (!document.hidden) connectEvents();
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
    if (c.zoom) {
      // An image drawn on demand (too slow to draw with the page), shown
      // in the zoom layer as if it had been clicked.
      const img = document.createElement("img");
      img.src = c.zoom; img.setAttribute("data-title", c.title || "image");
      if (c.pixelated) img.setAttribute("data-pixelated", "1");
      openZoom(img);
    }
    if (c.append) appendItems(c.append);
    if (c.scroll_item) scrollToItem(c.scroll_item);
    if (c.scroll_top) {
      const el = document.querySelector(c.scroll_top);
      const sp = el ? scrollParent(el) : null;
      if (sp) sp.scrollTop = 0; else window.scrollTo(0, 0);
    }
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

  // An <img> kept by the morph but given another src (the gallery after a
  // shuffle) would show its old picture until the new one arrives, so it
  // goes black -- as a tile still loading does -- until then.
  function imgWait(img) {
    img.classList.add("img-wait");
    if (img._lkWait) return;          // already waiting: the same listeners cover the new src
    img._lkWait = true;
    const done = () => {
      img._lkWait = false;
      img.classList.remove("img-wait");
      img.removeEventListener("load", done);
      img.removeEventListener("error", done);
    };
    img.addEventListener("load", done);
    img.addEventListener("error", done);
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
    const tag = a.tagName;
    const newSrc = tag === "IMG" && b.hasAttribute("src") && a.getAttribute("src") !== b.getAttribute("src");
    for (const at of Array.from(b.attributes)) {
      if (a.getAttribute(at.name) !== at.value) a.setAttribute(at.name, at.value);
    }
    if (newSrc) imgWait(a);
    else if (a._lkWait) a.classList.add("img-wait");  // a later morph must not reveal it early
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
    // Read before morphing the children: an option that is new is moved out
    // of `b` into `a`, and would no longer be found in `b` afterwards -- the
    // select then fell back to its first option ("None (no id)" shown for a
    // uuid column the server had picked).
    const wanted = tag === "SELECT" ? b.querySelector("option[selected]") : null;
    const wantedValue = wanted ? wanted.getAttribute("value") : null;
    morphChildren(a, b);
    if (tag === "SELECT") {
      const v = wantedValue !== null ? wantedValue : (a.options[0] ? a.options[0].value : "");
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

  // A browser opens at most six connections to one server, and an open
  // event stream holds one for good: with six Lookin tabs open (every
  // workspace opens in a tab of its own) the seventh never even loaded. So
  // only a tab on screen keeps its stream; a hidden one lets it go, and
  // reconnecting redraws whatever changed meanwhile (onopen above). What
  // the server had to say to it meanwhile -- a toast, a download -- waits
  // for it there (Session.push).
  // ...and says it is still there now and then, or its session -- holding
  // a loaded workspace -- would be dropped as abandoned (server.py).
  const PING_MS = 10 * 60 * 1000;
  setInterval(() => {
    if (!document.hidden || !sid || !booted) return;
    fetch("/api/ping", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify({ sid }) }).catch(() => { /* the next one will do */ });
  }, PING_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (es) { es.close(); es = null; }
    } else if (!es && sid) {
      connectEvents();
    }
  });

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
    const args = Object.assign(argsOf(el), extra || {});
    // data-anchor="<list id>": send which item of that infinite list is at
    // the top of its scroll area, so a re-layout (a zoom step) can put it
    // back there.
    const anchorList = el.getAttribute("data-anchor");
    if (anchorList) args.anchor = firstVisible(document.getElementById(anchorList));
    return act(target, action, args);
  }

  // ---- infinite lists --------------------------------------------------
  // A list (an element with an id and data-gen) followed somewhere by a
  // sentinel [data-more="target:action"][data-list="<id>"]. When the
  // sentinel comes within one screen of the bottom of its scroll area, the
  // server is asked for the next batch ({gen, loaded}); it answers with an
  // `append` command, so only the new items cross the wire.
  function scrollParent(el) {
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      const s = getComputedStyle(p).overflowY;
      if (s === "auto" || s === "scroll") return p;
    }
    return null;
  }

  function firstVisible(list) {
    if (!list) return 0;
    const sp = scrollParent(list);
    const top = sp ? sp.getBoundingClientRect().top : 0;
    const items = list.children;
    for (let i = 0; i < items.length; i++) {
      if (items[i].getBoundingClientRect().bottom > top + 2) return +(items[i].getAttribute("data-i") || i);
    }
    return 0;
  }

  function nearBottom(sentinel) {
    const sp = scrollParent(sentinel);
    const r = sentinel.getBoundingClientRect();
    const bottom = sp ? sp.getBoundingClientRect().bottom : window.innerHeight;
    const height = sp ? sp.clientHeight : window.innerHeight;
    return r.top <= bottom + height && r.bottom >= 0 && sentinel.offsetParent !== null;
  }

  function requestMore(sentinel) {
    if (sentinel._busy || sentinel.hidden) return null;
    const list = document.getElementById(sentinel.getAttribute("data-list"));
    if (!list) return null;
    sentinel._busy = true;
    const [target, action] = parseTarget(sentinel.getAttribute("data-more"));
    return act(target, action, Object.assign(argsOf(sentinel), {
      gen: list.getAttribute("data-gen") || "", loaded: list.children.length,
    })).then(() => {
      sentinel._busy = false;
      // Still in reach (a tall window, a fast scroll): keep going.
      if (document.contains(sentinel) && !sentinel.hidden && nearBottom(sentinel)) requestMore(sentinel);
    });
  }

  function wireInfiniteLists(root) {
    root.querySelectorAll("[data-more]").forEach((s) => {
      if (s._io) { if (nearBottom(s)) requestMore(s); return; }
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) requestMore(s);
      }, { root: scrollParent(s), rootMargin: "0px 0px 100% 0px" });
      io.observe(s);
      s._io = io;
    });
  }

  function appendItems(c) {
    const list = document.getElementById(c.to);
    if (!list || String(list.getAttribute("data-gen")) !== String(c.gen)) return;
    if (list.children.length !== c.start) return;     // out of step: the next render fixes it
    list.insertAdjacentHTML("beforeend", c.html);
    const count = document.getElementById(c.to + "-count");
    if (count && c.count !== undefined) count.textContent = c.count;
    document.querySelectorAll('[data-list="' + c.to + '"]').forEach((s) => { s.hidden = !!c.done; });
  }

  function scrollToItem(c) {
    const list = document.getElementById(c.list);
    if (!list) return;
    const item = list.querySelector('[data-i="' + c.index + '"]');
    const sp = scrollParent(list);
    if (!sp) return;
    if (!item) { sp.scrollTop = 0; return; }
    sp.scrollTop += item.getBoundingClientRect().top - sp.getBoundingClientRect().top;
  }

  function valueOf(el) {
    if (el.type === "checkbox") return el.checked;
    if (el.multiple) return Array.from(el.selectedOptions).map((o) => o.value);
    return el.value;
  }

  // ---- light / dark ------------------------------------------------------
  // The <head> script (server.py, THEME_SCRIPT) applied the saved choice, or
  // the system's, before the first paint; this saves a click on the button,
  // and the other tabs of this browser follow it.
  const THEME_KEY = "lookin-theme";

  function toggleTheme() {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode: this tab only */ }
  }

  window.addEventListener("storage", (ev) => {
    if (ev.key === THEME_KEY && (ev.newValue === "light" || ev.newValue === "dark")) {
      document.documentElement.setAttribute("data-theme", ev.newValue);
    }
  });

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
    if (ev.target.closest("[data-theme-toggle]")) { toggleTheme(); ev.preventDefault(); return; }
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
    // An experiment's name field follows the length of what is typed, so
    // a long name is read whole (see sources.py _exp_tbody).
    if (ev.target.classList && ev.target.classList.contains("exp-name")) {
      const n = Math.min(Math.max(ev.target.value.length, 12), 48);
      ev.target.style.width = "calc(" + n + "ch + 20px)";
    }
    const el = ev.target.closest("[data-input]");
    if (!el) return;
    clearTimeout(inputTimers.get(el));
    inputTimers.set(el, setTimeout(() => fire(el, "data-input", { value: valueOf(el) }), 320));
  });

  document.addEventListener("keydown", (ev) => {
    if (zoomState && !ev.ctrlKey && !ev.metaKey && !ev.altKey && zoomKey(ev)) { ev.preventDefault(); return; }
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
      if (menu.classList.contains("menu-fixed")) {
        // Opened from inside a scrolling list (an experiment's color, see
        // widgets.experiment_color_menu), which would clip it: placed on
        // the page instead, under its button.
        menu.style.position = "fixed"; menu.style.right = "auto"; menu.style.minWidth = "0";
        menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)) + "px";
        menu.style.top = Math.min(r.bottom + 4, window.innerHeight - menu.offsetHeight - 8) + "px";
      }
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
  // What is being dragged is only readable on drop (dataTransfer hides its
  // data during dragover), so its kind is kept here from dragstart: a target
  // lights up, and takes the drop, only for the kinds it lists -- a run
  // dragged over a group header must not look droppable there.
  let dragKind = null;
  document.addEventListener("dragstart", (ev) => {
    const el = ev.target.closest("[data-drag]");
    if (!el) return;
    dragKind = el.getAttribute("data-drag-kind") || "";
    ev.dataTransfer.effectAllowed = "copyMove";
    ev.dataTransfer.setData("application/x-lookin", JSON.stringify({
      kind: dragKind, key: el.getAttribute("data-drag"),
    }));
    ev.dataTransfer.setData("text/plain", el.getAttribute("data-drag-label") || el.textContent.trim().slice(0, 60));
    document.body.classList.add("dragging");
  });
  document.addEventListener("dragend", () => {
    dragKind = null;
    document.body.classList.remove("dragging");
    document.querySelectorAll(".drop-on").forEach((d) => d.classList.remove("drop-on"));
  });
  function accepts(el, kind) {
    const kinds = (el.getAttribute("data-drop-kinds") || "").split(" ").filter(Boolean);
    return !kinds.length || kinds.includes(kind);
  }
  // The innermost [data-drop] under the pointer that takes `kind`: a run
  // over a workspace row is that row's, a workspace over the same row goes
  // up to the row's group.
  function dropTarget(ev, kind) {
    if (kind === null) return null;
    let el = ev.target.closest ? ev.target.closest("[data-drop]") : null;
    while (el && !accepts(el, kind)) el = el.parentElement ? el.parentElement.closest("[data-drop]") : null;
    return el;
  }
  document.addEventListener("dragover", (ev) => {
    const el = dropTarget(ev, dragKind);
    document.querySelectorAll(".drop-on").forEach((d) => { if (d !== el) d.classList.remove("drop-on"); });
    if (!el) return;
    ev.preventDefault();
    el.classList.add("drop-on");
  });
  document.addEventListener("dragleave", (ev) => {
    const el = dropTarget(ev, dragKind);
    if (el && !el.contains(ev.relatedTarget)) el.classList.remove("drop-on");
  });
  document.addEventListener("drop", (ev) => {
    let payload = null;
    try { payload = JSON.parse(ev.dataTransfer.getData("application/x-lookin")); } catch (e) { payload = null; }
    const el = payload ? dropTarget(ev, payload.kind || "") : null;
    if (!el) return;
    ev.preventDefault();
    el.classList.remove("drop-on");
    fire(el, "data-drop", { kind: payload.kind, key: payload.key });
  });

  // ---- zoom -------------------------------------------------------------
  // The full-screen view of a chart or a crop: it opens fitted to the
  // window, the wheel zooms about the pointer (as far in as wanted -- a crop
  // is shown with its pixels, a chart as drawn), a drag pans, a double click
  // goes back and forth between fitted and 2x at that point. + - 0 on the
  // keyboard do the same as the buttons.
  //
  // A crop with a `data-info` URL (views/details.py) also gets an (i)
  // button: a panel on the left -- the margin a fitted crop leaves unused
  // -- with its dataset row, its review decision and every experiment's
  // prediction, fetched when the panel is opened. Whether it is open is
  // remembered from one crop to the next; "i" toggles it.
  //
  // A crop opened from a list (the Gallery's carousel: an element with
  // data-gen; the Embeddings gallery's tiles: data-zoom-list) gets < > in
  // the middle of the bar, and the arrow keys: they
  // step through the crops of that list, loading its next page when the
  // last loaded crop is reached.
  let zoomState = null;
  let zoomInfoOpen = false;
  const INFO_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.3"/></svg>';

  function openZoom(img) {
    const src = img.getAttribute("data-zoom-src") || img.currentSrc || img.src;
    const title = img.getAttribute("data-title") || img.alt || "image";
    const info = img.getAttribute("data-info") || "";
    const list = img.closest(".car-list[data-gen], [data-zoom-list]");
    const z = document.getElementById("zoom");
    const fname = "lookin_" + title.replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 60) + ".png";
    z.innerHTML =
      '<div class="zoom-back"><div class="zoom-bar">' +
      (info ? '<button type="button" class="btn btn-sm btn-icon zoom-info-b" title="Details: metadata and predictions (i)">' + INFO_ICON + '</button>' : '') +
      '<span class="zoom-t"></span>' +
      (list ? '<span class="zoom-nav"><button type="button" class="btn btn-sm zoom-prev" title="Previous image (Left arrow)">&lt;</button>' +
        '<span class="zoom-pos mono small"></span>' +
        '<button type="button" class="btn btn-sm zoom-next" title="Next image (Right arrow)">&gt;</button></span>' : '') +
      '<button type="button" class="btn btn-sm zoom-out" title="Zoom out (-)">-</button>' +
      '<span class="zoom-pct mono small"></span>' +
      '<button type="button" class="btn btn-sm zoom-in" title="Zoom in (+)">+</button>' +
      '<button type="button" class="btn btn-sm zoom-fit" title="Fit to the window (0)">Fit</button>' +
      '<button type="button" class="btn btn-sm zoom-one" title="Actual size">1:1</button>' +
      '<span class="vsep"></span>' +
      '<a class="btn btn-sm" target="_blank" rel="noopener">Open in a new tab</a>' +
      '<a class="btn btn-sm">Save</a>' +
      '<button type="button" class="btn btn-sm btn-primary zoom-close">Close</button></div>' +
      '<div class="zoom-main"><div class="zoom-side" hidden></div>' +
      '<div class="zoom-img"><img alt="" draggable="false"></div></div></div>';
    z.querySelector(".zoom-t").textContent = title;
    const links = z.querySelectorAll("a");
    links[0].href = src;
    links[1].href = src; links[1].download = fname;
    const stage = z.querySelector(".zoom-img");
    const big = stage.querySelector("img");
    big.alt = title;
    if (img.hasAttribute("data-pixelated")) big.classList.add("pix");
    const st = { stage, big, scale: 1, x: 0, y: 0, fit: 1, w: 0, h: 0, info, infoLoaded: false, list, from: img };
    zoomState = st;
    if (list) {
      z.querySelector(".zoom-prev").onclick = () => zoomStep(-1);
      z.querySelector(".zoom-next").onclick = () => zoomStep(1);
      zoomNavSync();
    }
    if (info) {
      z.querySelector(".zoom-info-b").onclick = () => zoomInfo(!zoomInfoOpen);
      if (zoomInfoOpen) zoomInfo(true);
    }
    big.onload = () => { st.w = big.naturalWidth; st.h = big.naturalHeight; zoomFit(); };
    big.src = src;

    z.querySelector(".zoom-close").onclick = closeZoom;
    z.querySelector(".zoom-in").onclick = () => zoomBy(1.5);
    z.querySelector(".zoom-out").onclick = () => zoomBy(1 / 1.5);
    z.querySelector(".zoom-fit").onclick = zoomFit;
    z.querySelector(".zoom-one").onclick = () => zoomTo(1);
    stage.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const r = stage.getBoundingClientRect();
      zoomBy(Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015)), ev.clientX - r.left, ev.clientY - r.top);
    }, { passive: false });
    let drag = null;
    stage.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      drag = { px: ev.clientX, py: ev.clientY, x: st.x, y: st.y, moved: false };
      stage.setPointerCapture(ev.pointerId);
      stage.classList.add("drag");
    });
    stage.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.px, dy = ev.clientY - drag.py;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      st.x = drag.x + dx; st.y = drag.y + dy;
      zoomApply();
    });
    const endDrag = (ev) => {
      if (!drag) return;
      const wasClick = !drag.moved;
      drag = null;
      stage.classList.remove("drag");
      // A click on the dark margin, not on the picture, closes as before.
      if (wasClick && ev.type === "pointerup" && ev.target === stage && !zoomOnImage(ev)) closeZoom();
    };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("dblclick", (ev) => {
      const r = stage.getBoundingClientRect();
      if (st.scale > st.fit * 1.05) zoomFit();
      else zoomTo(Math.max(st.fit * 2, 1), ev.clientX - r.left, ev.clientY - r.top);
    });
  }

  function zoomInfo(open) {
    const st = zoomState;
    if (!st || !st.info) return;
    zoomInfoOpen = open;
    const side = document.querySelector("#zoom .zoom-side");
    const btn = document.querySelector("#zoom .zoom-info-b");
    side.hidden = !open;
    if (btn) btn.classList.toggle("on", open);
    if (open && !st.infoLoaded) {
      st.infoLoaded = true;
      side.innerHTML = '<div class="muted small">Loading...</div>';
      fetch(st.info).then((r) => r.ok ? r.text() : Promise.reject(r.status)).then((html) => {
        if (zoomState === st) side.innerHTML = html;
      }).catch(() => {
        if (zoomState === st) { side.textContent = "Could not load the details of this crop."; st.infoLoaded = false; }
      });
    }
    zoomFit();
  }

  function zoomSiblings(st) {
    return Array.from(st.list.querySelectorAll("img[data-zoom]"));
  }

  function zoomMoreSentinel(st) {
    if (!st.list.id) return null;                 // a list with no further pages (the map's tiles)
    const s = document.querySelector('[data-list="' + st.list.id + '"]');
    return s && !s.hidden ? s : null;
  }

  function zoomNavSync() {
    const st = zoomState;
    if (!st || !st.list) return;
    const all = zoomSiblings(st), k = all.indexOf(st.from);
    const more = zoomMoreSentinel(st);
    const z = document.getElementById("zoom");
    z.querySelector(".zoom-prev").disabled = k <= 0;
    z.querySelector(".zoom-next").disabled = k < 0 || (k >= all.length - 1 && !more);
    z.querySelector(".zoom-pos").textContent = k < 0 ? "" : (k + 1) + " / " + all.length + (more ? "+" : "");
  }

  function zoomStep(delta) {
    const st = zoomState;
    if (!st || !st.list || st.stepping) return;
    const all = zoomSiblings(st), k = all.indexOf(st.from);
    if (k < 0) return;
    const go = (img) => {
      if (zoomState !== st || !img) return;
      img.scrollIntoView({ block: "nearest" });
      openZoom(img);
    };
    if (all[k + delta]) { go(all[k + delta]); return; }
    if (delta < 0) return;
    // past the last loaded crop: fetch the list's next page, then step
    const more = zoomMoreSentinel(st);
    const pending = more ? requestMore(more) : null;
    if (!pending) return;
    st.stepping = true;
    pending.then(() => { st.stepping = false; go(zoomSiblings(st)[k + 1]); });
  }

  function zoomOnImage(ev) {
    const st = zoomState;
    if (!st) return false;
    const r = st.stage.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    return x >= st.x && y >= st.y && x <= st.x + st.w * st.scale && y <= st.y + st.h * st.scale;
  }

  function zoomLimits(st) {
    return [Math.min(st.fit, 1) / 4, Math.max(32, st.fit * 16)];
  }

  function zoomApply() {
    const st = zoomState;
    if (!st) return;
    st.big.style.transform = "translate(" + st.x + "px," + st.y + "px) scale(" + st.scale + ")";
    const pct = document.querySelector("#zoom .zoom-pct");
    if (pct) pct.textContent = Math.round(st.scale * 100) + "%";
  }

  function zoomFit() {
    const st = zoomState;
    if (!st || !st.w) return;
    const W = st.stage.clientWidth, H = st.stage.clientHeight;
    let fit = Math.min(W / st.w, H / st.h);
    // A chart is drawn at twice the screen resolution: shown no larger than
    // that to start with. A crop is small: enlarged to fill the window.
    if (!st.big.classList.contains("pix")) fit = Math.min(fit, 1);
    st.fit = fit;
    st.scale = fit;
    st.x = (W - st.w * fit) / 2;
    st.y = (H - st.h * fit) / 2;
    zoomApply();
  }

  function zoomTo(scale, cx, cy) {
    const st = zoomState;
    if (!st || !st.w) return;
    const [lo, hi] = zoomLimits(st);
    const next = Math.max(lo, Math.min(hi, scale));
    if (cx === undefined) { cx = st.stage.clientWidth / 2; cy = st.stage.clientHeight / 2; }
    st.x = cx - (cx - st.x) * (next / st.scale);
    st.y = cy - (cy - st.y) * (next / st.scale);
    st.scale = next;
    zoomApply();
  }

  function zoomBy(factor, cx, cy) {
    if (zoomState) zoomTo(zoomState.scale * factor, cx, cy);
  }

  function zoomKey(ev) {
    if (!zoomState) return false;
    if (ev.key === "+" || ev.key === "=") zoomBy(1.5);
    else if (ev.key === "-" || ev.key === "_") zoomBy(1 / 1.5);
    else if (ev.key === "0") zoomFit();
    else if (ev.key === "i" && zoomState.info) zoomInfo(!zoomInfoOpen);
    else if (ev.key === "ArrowLeft" && zoomState.list) zoomStep(-1);
    else if (ev.key === "ArrowRight" && zoomState.list) zoomStep(1);
    else return false;
    return true;
  }

  function closeZoom() {
    const z = document.getElementById("zoom");
    zoomState = null;
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
    wireInfiniteLists(document);
    // "Half ticked" is a property, not an attribute, so the server asks for
    // it with data-indeterminate (a run with some of its CSVs selected).
    document.querySelectorAll("input[type=checkbox]").forEach((el) => {
      el.indeterminate = el.hasAttribute("data-indeterminate");
    });
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
