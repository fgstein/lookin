/* The Embeddings tab's interactive map -- the browser port of
 * lookin.ui.interactive_projection.
 *
 * The server sends the projected cloud once (a JSON file: coordinates in
 * data space, a colour and a group per point, the prototypes); everything a
 * gesture does happens here, on a canvas: hover shows the point's crop (the
 * original file, through the session's display setting, fetched from
 * /pimg only for the points the pointer lands on), a click opens it in the
 * carousel, a lasso selects a group and opens it there, the wheel / the
 * box / the buttons zoom, a drag pans. The legend hides groups and brings
 * them to the front in place. One transform (toScreen) both places the dots
 * and answers "which point is under the cursor", so the two cannot drift.
 *
 * Talks to the server only through window.lookinAct.
 */
(function () {
  "use strict";

  const POINT_DIAMETER = 5.0;
  const PICK_RADIUS = 11.0;
  const HOVER_THUMB = 120;
  const HOVER_OFFSET = 16;
  const LASSO_COLOR = "#1f6feb";
  const ZOOM_BOX_COLOR = "#a04ec9";
  const MARKER_COLOR = "#111111";
  const LASSO_MIN_STEP = 4.0;
  const LASSO_MIN_VERTICES = 4;
  const ZOOM_BOX_MIN_SIDE = 12.0;
  const ZOOM_STEP = 1.25;
  const ZOOM_MAX = 100.0;
  const PROTOTYPE_RADIUS = 9.0;
  const SUBCENTER_SHAPES = ["star", "plus", "cross", "diamond", "triangle_up", "triangle_down", "square", "pentagon"];
  const PLOT_PADDING_MIN = 22;
  const PLOT_PADDING_FRAC = 0.06;
  const MIN_PLOT_SIZE = 420;

  function plotPadding(size) { return Math.max(PLOT_PADDING_MIN, size * PLOT_PADDING_FRAC); }

  function markerPolygon(kind, cx, cy, r) {
    const ring = (n, start, radius) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const a = (start + i * 360 / n) * Math.PI / 180;
        out.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a)]);
      }
      return out;
    };
    if (kind === "star") {
      const out = [];
      for (let i = 0; i < 10; i++) {
        const rr = i % 2 === 0 ? r : r * 0.45;
        const a = (-90 + i * 36) * Math.PI / 180;
        out.push([cx + rr * Math.cos(a), cy + rr * Math.sin(a)]);
      }
      return out;
    }
    if (kind === "plus" || kind === "cross") {
      const a = r * 0.34;
      let pts = [[-a, -r], [a, -r], [a, -a], [r, -a], [r, a], [a, a], [a, r], [-a, r], [-a, a], [-r, a], [-r, -a], [-a, -a]];
      if (kind === "cross") {
        const c = Math.cos(Math.PI / 4), s = Math.sin(Math.PI / 4);
        pts = pts.map(([x, y]) => [x * c - y * s, x * s + y * c]);
      }
      return pts.map(([x, y]) => [cx + x, cy + y]);
    }
    if (kind === "diamond") return ring(4, -90, r);
    if (kind === "triangle_up") return ring(3, -90, r);
    if (kind === "triangle_down") return ring(3, 90, r);
    if (kind === "square") return ring(4, -45, r);
    return ring(5, -90, r);
  }

  function insidePolygon(poly, x, y) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }

  class PMap {
    constructor(root, data) {
      this.root = root;
      this.d = data;
      this.target = root.getAttribute("data-target") || "embeddings";
      this.imgBase = root.getAttribute("data-pimg");
      this.n = data.x.length;
      this.x = Float64Array.from(data.x);
      this.y = Float64Array.from(data.y);
      this.sx = new Float64Array(this.n);
      this.sy = new Float64Array(this.n);
      this.hidden = new Set();          // group indices
      this.front = [];                  // group indices, last on top
      this.selected = [];
      this.lassoData = null;            // [[x, y], ...] in data space
      this.lasso = [];
      this.zoomBox = null;
      this.mode = "pan";
      this.current = null;
      this.sprites = {};
      this.buildDom();
      this.size = this.measure();
      this.fit();
      this.wire();
      this.redraw();
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.box);
    }

    // ---- DOM ----
    buildDom() {
      const r = this.root;
      this.box = r.querySelector(".pmap-box");
      this.canvas = document.createElement("canvas");
      this.overlay = document.createElement("canvas");
      this.canvas.className = "pmap-c";
      this.overlay.className = "pmap-o";
      this.card = document.createElement("div");
      this.card.className = "pmap-card";
      this.card.innerHTML = '<img alt=""><b></b><span></span>';
      this.box.innerHTML = "";
      this.box.append(this.canvas, this.overlay, this.card);
      this.zoomText = r.querySelector(".pm-zoomtext");
      this.selText = r.querySelector(".pm-seltext");
      this.btnReset = r.querySelector("[data-pm=reset]");
      this.btnShow = r.querySelector("[data-pm=show]");
      this.btnClear = r.querySelector("[data-pm=clear]");
    }

    measure() {
      const w = this.box.parentElement.clientWidth || 600;
      const top = this.box.getBoundingClientRect().top;
      const h = window.innerHeight - Math.max(0, top) - 16;
      return Math.max(MIN_PLOT_SIZE, Math.floor(Math.min(w, h > 200 ? h : w)));
    }

    setupCanvas() {
      const dpr = window.devicePixelRatio || 1;
      for (const c of [this.canvas, this.overlay]) {
        c.width = Math.round(this.size * dpr);
        c.height = Math.round(this.size * dpr);
        c.style.width = this.size + "px";
        c.style.height = this.size + "px";
        c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      this.box.style.width = this.size + "px";
      this.box.style.height = this.size + "px";
      this.sprites = {};
    }

    // ---- geometry ----
    fit() {
      let xs = Array.from(this.x), ys = Array.from(this.y);
      const p = this.d.protos;
      if (p && p.x.length) { xs = xs.concat(p.x); ys = ys.concat(p.y); }
      let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
      for (let i = 0; i < xs.length; i++) {
        if (xs[i] < minx) minx = xs[i]; if (xs[i] > maxx) maxx = xs[i];
        if (ys[i] < miny) miny = ys[i]; if (ys[i] > maxy) maxy = ys[i];
      }
      if (!isFinite(minx)) { minx = 0; maxx = 1; miny = 0; maxy = 1; }
      this.span = Math.max(maxx - minx, maxy - miny) || 1.0;
      this.setupCanvas();
      this.fitScale = (this.size - 2 * plotPadding(this.size)) / this.span;
      this.scale = this.fitScale;
      this.cx = (minx + maxx) / 2;
      this.cy = (miny + maxy) / 2;
      this.base = [this.cx, this.cy];
    }

    toScreen(x, y) {
      const half = this.size / 2;
      return [half + (x - this.cx) * this.scale, half - (y - this.cy) * this.scale];
    }

    toData(px, py) {
      const half = this.size / 2;
      return [this.cx + (px - half) / this.scale, this.cy - (py - half) / this.scale];
    }

    project() {
      const half = this.size / 2, s = this.scale, cx = this.cx, cy = this.cy;
      for (let i = 0; i < this.n; i++) {
        this.sx[i] = half + (this.x[i] - cx) * s;
        this.sy[i] = half - (this.y[i] - cy) * s;
      }
    }

    clampCentre() {
      const halfSpan = (this.size / 2) / this.scale;
      const fitHalf = (this.size / 2 - plotPadding(this.size)) / this.fitScale;
      const limit = Math.max(0, fitHalf - halfSpan);
      this.cx = Math.min(this.base[0] + limit, Math.max(this.base[0] - limit, this.cx));
      this.cy = Math.min(this.base[1] + limit, Math.max(this.base[1] - limit, this.cy));
    }

    applyZoom(newScale, anchor) {
      newScale = Math.max(this.fitScale, Math.min(ZOOM_MAX * this.fitScale, newScale));
      if (newScale === this.scale) return;
      const half = this.size / 2;
      const [ax, ay] = anchor || [half, half];
      const [dx, dy] = this.toData(ax, ay);
      this.scale = newScale;
      this.cx = dx - (ax - half) / newScale;
      this.cy = dy + (ay - half) / newScale;
      this.clampCentre();
      this.requestRedraw();
    }

    resize() {
      const size = this.measure();
      if (Math.abs(size - this.size) < 2) return;
      const ratio = this.scale / this.fitScale;
      this.size = size;
      this.setupCanvas();
      this.fitScale = (size - 2 * plotPadding(size)) / this.span;
      this.scale = this.fitScale * ratio;
      this.clampCentre();
      this.requestRedraw();
    }

    // ---- what is shown ----
    shown(i) { return !this.hidden.size || !this.hidden.has(this.d.g[i]); }

    rankOf(i) {
      const k = this.front.indexOf(this.d.g[i]);
      return k < 0 ? 0 : k + 1;
    }

    sprite(color, kind) {
      const key = color + "|" + kind;
      if (this.sprites[key]) return this.sprites[key];
      const dpr = window.devicePixelRatio || 1;
      const d = POINT_DIAMETER * (kind === 1 ? 1.3 : 1.0);
      const c = document.createElement("canvas");
      const px = Math.ceil((d + 2) * dpr);
      c.width = c.height = px;
      const g = c.getContext("2d");
      g.scale(dpr, dpr);
      g.fillStyle = color;
      if (kind === 1) {
        g.fillRect(1, 1, d, d);
        g.strokeStyle = "#000"; g.lineWidth = 0.6; g.strokeRect(1, 1, d, d);
      } else {
        g.beginPath(); g.arc(1 + d / 2, 1 + d / 2, d / 2, 0, Math.PI * 2); g.fill();
      }
      this.sprites[key] = { c, d: d + 2 };
      return this.sprites[key];
    }

    // ---- drawing ----
    requestRedraw() {
      if (this.raf) return;
      this.raf = requestAnimationFrame(() => { this.raf = null; this.redraw(); });
    }

    redraw() {
      this.project();
      const g = this.canvas.getContext("2d");
      g.clearRect(0, 0, this.size, this.size);
      g.fillStyle = "#ffffff";
      g.fillRect(0, 0, this.size, this.size);
      const pal = this.d.palette, cols = this.d.c, kinds = this.d.k;
      const layers = [[]];
      this.front.forEach(() => layers.push([]));
      const m = 24;
      for (let i = 0; i < this.n; i++) {
        if (!this.shown(i)) continue;
        const x = this.sx[i], y = this.sy[i];
        if (x < -m || y < -m || x > this.size + m || y > this.size + m) continue;
        layers[this.rankOf(i)].push(i);
      }
      // the selection halo goes under the selected points, in their colours
      const sel = new Set(this.selected);
      for (const layer of layers) {
        for (const i of layer) {
          const s = this.sprite(pal[cols[i]], kinds ? kinds[i] : 0);
          g.drawImage(s.c, this.sx[i] - s.d / 2, this.sy[i] - s.d / 2, s.d, s.d);
        }
      }
      if (sel.size) {
        g.fillStyle = MARKER_COLOR;
        for (const i of this.selected) {
          if (!this.shown(i)) continue;
          g.beginPath(); g.arc(this.sx[i], this.sy[i], (POINT_DIAMETER + 4) / 2, 0, Math.PI * 2); g.fill();
        }
        for (const i of this.selected) {
          if (!this.shown(i)) continue;
          const s = this.sprite(pal[cols[i]], kinds ? kinds[i] : 0);
          g.drawImage(s.c, this.sx[i] - s.d / 2, this.sy[i] - s.d / 2, s.d, s.d);
        }
      }
      this.drawPrototypes(g);
      if (this.d.circle) {
        const [ox, oy] = this.toScreen(0, 0);
        g.strokeStyle = "gray"; g.setLineDash([4, 3]); g.lineWidth = 0.8;
        g.beginPath(); g.arc(ox, oy, this.scale, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
      }
      this.drawOverlay();
      this.syncToolbar();
    }

    drawPrototypes(g) {
      const p = this.d.protos;
      if (!p || !p.x.length) return;
      this.psx = []; this.psy = [];
      for (let j = 0; j < p.x.length; j++) {
        const [px, py] = this.toScreen(p.x[j], p.y[j]);
        this.psx.push(px); this.psy.push(py);
        const kind = SUBCENTER_SHAPES[p.k[j] % SUBCENTER_SHAPES.length];
        const poly = markerPolygon(kind, px, py, PROTOTYPE_RADIUS);
        g.beginPath();
        g.moveTo(poly[0][0], poly[0][1]);
        for (let q = 1; q < poly.length; q++) g.lineTo(poly[q][0], poly[q][1]);
        g.closePath();
        g.fillStyle = p.color[j]; g.fill();
        g.strokeStyle = "#000"; g.lineWidth = 1.2; g.stroke();
      }
    }

    drawOverlay() {
      const g = this.overlay.getContext("2d");
      g.clearRect(0, 0, this.size, this.size);
      let poly = null;
      if (this.lasso.length) poly = this.lasso;
      else if (this.lassoData) poly = this.lassoData.map(([x, y]) => this.toScreen(x, y));
      if (poly && poly.length > 1) {
        g.strokeStyle = LASSO_COLOR; g.lineWidth = 1.8;
        g.beginPath(); g.moveTo(poly[0][0], poly[0][1]);
        for (let q = 1; q < poly.length; q++) g.lineTo(poly[q][0], poly[q][1]);
        if (!this.lasso.length) g.closePath();
        g.stroke();
      }
      if (this.zoomBox) {
        const [x0, y0, x1, y1] = this.zoomBox;
        g.strokeStyle = ZOOM_BOX_COLOR; g.lineWidth = 1.5;
        g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      }
      if (this.current) {
        let x, y, r;
        if (this.current[0] === "p") { x = this.sx[this.current[1]]; y = this.sy[this.current[1]]; r = POINT_DIAMETER * 1.8; }
        else { x = this.psx[this.current[1]]; y = this.psy[this.current[1]]; r = PROTOTYPE_RADIUS + 4; }
        g.strokeStyle = MARKER_COLOR; g.lineWidth = 2;
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke();
      }
    }

    syncToolbar() {
      const z = this.scale / this.fitScale;
      if (this.zoomText) this.zoomText.textContent = "×" + z.toFixed(1);
      if (this.btnReset) this.btnReset.hidden = !(this.scale > this.fitScale * 1.001);
      const has = this.lassoData !== null;
      if (this.selText) this.selText.textContent = has ? this.selected.length.toLocaleString("en-US") + " points selected" : "";
      if (this.btnShow) this.btnShow.hidden = !this.selected.length;
      if (this.btnClear) this.btnClear.hidden = !has;
      this.root.querySelectorAll("[data-pm-mode]").forEach((b) => b.classList.toggle("on", b.getAttribute("data-pm-mode") === this.mode));
    }

    // ---- picking ----
    pick(px, py) {
      let best = -1, bestD = PICK_RADIUS * PICK_RADIUS, bestRank = -1;
      const useRank = this.front.length > 0;
      for (let i = 0; i < this.n; i++) {
        const dx = this.sx[i] - px, dy = this.sy[i] - py;
        const d2 = dx * dx + dy * dy;
        if (d2 > PICK_RADIUS * PICK_RADIUS || !this.shown(i)) continue;
        if (useRank) {
          const r = this.rankOf(i);
          if (r > bestRank || (r === bestRank && d2 < bestD)) { best = i; bestD = d2; bestRank = r; }
        } else if (d2 <= bestD) { best = i; bestD = d2; }
      }
      return best < 0 ? null : best;
    }

    pickPrototype(px, py) {
      if (!this.psx || !this.psx.length) return null;
      let best = -1, bestD = (PROTOTYPE_RADIUS + 4) ** 2;
      for (let j = 0; j < this.psx.length; j++) {
        const d2 = (this.psx[j] - px) ** 2 + (this.psy[j] - py) ** 2;
        if (d2 <= bestD) { best = j; bestD = d2; }
      }
      return best < 0 ? null : best;
    }

    // ---- hover ----
    hover(px, py) {
      const j = this.pickPrototype(px, py);
      let target = j !== null ? ["q", j] : null;
      if (!target) { const i = this.pick(px, py); target = i !== null ? ["p", i] : null; }
      const same = (a, b) => (a === b) || (a && b && a[0] === b[0] && a[1] === b[1]);
      if (same(target, this.current)) return;
      this.current = target;
      const card = this.card, img = card.querySelector("img"), note = card.querySelector("b"), label = card.querySelector("span");
      if (!target) { card.style.display = "none"; this.drawOverlay(); return; }
      let x0, y0, w, h;
      if (target[0] === "p") {
        const i = target[1];
        img.style.display = ""; note.style.display = "none";
        img.src = this.imgBase + i;
        label.textContent = this.d.hl[this.d.hi[i]] || "";
        x0 = this.sx[i]; y0 = this.sy[i]; w = HOVER_THUMB + 12; h = HOVER_THUMB + 28;
      } else {
        const q = target[1], p = this.d.protos;
        img.style.display = "none"; note.style.display = "";
        note.textContent = p.label[q] != null ? String(p.label[q]) : "prototype";
        label.textContent = "subcenter " + p.k[q];
        x0 = this.psx[q]; y0 = this.psy[q]; w = 170; h = 46;
      }
      let x = x0 + HOVER_OFFSET, y = y0 + HOVER_OFFSET;
      if (x + w > this.size) x = x0 - HOVER_OFFSET - w;
      if (y + h > this.size) y = y0 - HOVER_OFFSET - h;
      card.style.left = Math.max(2, Math.min(this.size - w - 2, x)) + "px";
      card.style.top = Math.max(2, Math.min(this.size - h - 2, y)) + "px";
      card.style.display = "flex";
      this.drawOverlay();
    }

    hideHover() {
      this.current = null;
      this.card.style.display = "none";
      this.drawOverlay();
    }

    // ---- gestures ----
    wire() {
      const o = this.overlay;
      const local = (ev) => { const r = o.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
      let down = null, moved = false, last = null;
      o.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        o.setPointerCapture(ev.pointerId);
        down = local(ev); last = down; moved = false;
        if (this.mode === "zoom") this.zoomBox = [down[0], down[1], down[0], down[1]];
        else if (this.mode === "lasso") { this.lasso = [down]; this.hideHover(); }
      });
      o.addEventListener("pointermove", (ev) => {
        const p = local(ev);
        if (!down) { this.hover(p[0], p[1]); return; }
        if (Math.hypot(p[0] - down[0], p[1] - down[1]) > 3) moved = true;
        if (this.mode === "pan") {
          if (!moved) return;
          this.cx -= (p[0] - last[0]) / this.scale;
          this.cy += (p[1] - last[1]) / this.scale;
          this.clampCentre();
          last = p;
          this.card.style.display = "none"; this.current = null;
          this.requestRedraw();
        } else if (this.mode === "zoom") {
          this.zoomBox = [this.zoomBox[0], this.zoomBox[1], p[0], p[1]];
          this.drawOverlay();
        } else {
          const prev = this.lasso[this.lasso.length - 1];
          if (Math.hypot(p[0] - prev[0], p[1] - prev[1]) >= LASSO_MIN_STEP) { this.lasso.push(p); this.drawOverlay(); }
        }
      });
      const end = (ev) => {
        if (!down) return;
        const p = local(ev);
        const wasClick = !moved;
        down = null;
        if (this.mode === "zoom") { this.finishZoomBox(); if (!wasClick) return; }
        else if (this.mode === "lasso") { if (!wasClick) { this.finishLasso(); return; } this.lasso = []; this.drawOverlay(); }
        if (wasClick) this.tap(p[0], p[1]);
      };
      o.addEventListener("pointerup", end);
      o.addEventListener("pointercancel", () => { down = null; this.zoomBox = null; this.lasso = []; this.drawOverlay(); });
      o.addEventListener("pointerleave", () => { if (!down) this.hideHover(); });
      o.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        if (!ev.deltaY) return;
        const factor = ev.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
        this.applyZoom(this.scale * factor, local(ev));
      }, { passive: false });
      this.root.querySelectorAll("[data-pm-mode]").forEach((b) => b.addEventListener("click", () => {
        this.mode = b.getAttribute("data-pm-mode"); this.syncToolbar();
      }));
      const on = (sel, fn) => { const b = this.root.querySelector(sel); if (b) b.addEventListener("click", fn); };
      on("[data-pm=in]", () => this.applyZoom(this.scale * ZOOM_STEP));
      on("[data-pm=out]", () => this.applyZoom(this.scale / ZOOM_STEP));
      on("[data-pm=reset]", () => { this.fit(); this.requestRedraw(); });
      on("[data-pm=show]", () => this.emitSelection());
      on("[data-pm=clear]", () => { this.selected = []; this.lassoData = null; this.requestRedraw(); });
      on("[data-pm=full]", () => {
        const view = this.root.closest(".emb");
        if (view) view.classList.toggle("emb-full");
        const b = this.root.querySelector("[data-pm=full]");
        if (b && view) b.title = view.classList.contains("emb-full") ? "Leave full screen" : "Full screen -- hide the settings column and fill the window";
        setTimeout(() => this.resize(), 30);
      });
      this.root.querySelectorAll("[data-pm-render]").forEach((b) => b.addEventListener("click", () => {
        const which = b.getAttribute("data-pm-render");
        this.root.classList.toggle("pm-static", which === "static");
        this.root.querySelectorAll("[data-pm-render]").forEach((x) => x.classList.toggle("on", x === b));
      }));
      this.wireLegend();
    }

    tap(px, py) {
      if (this.pickPrototype(px, py) !== null) return;
      const i = this.pick(px, py);
      if (i === null) return;
      window.lookinAct(this.target, "point", { i: i });
    }

    finishZoomBox() {
      const box = this.zoomBox; this.zoomBox = null;
      if (!box) { this.drawOverlay(); return; }
      const [x0, y0, x1, y1] = box;
      if (Math.abs(x1 - x0) < ZOOM_BOX_MIN_SIDE || Math.abs(y1 - y0) < ZOOM_BOX_MIN_SIDE) { this.drawOverlay(); return; }
      const [dx0, dy0] = this.toData(x0, y0), [dx1, dy1] = this.toData(x1, y1);
      const span = Math.max(Math.abs(dx1 - dx0), Math.abs(dy1 - dy0)) || 1e-9;
      this.cx = (dx0 + dx1) / 2; this.cy = (dy0 + dy1) / 2;
      this.scale = Math.max(this.fitScale, Math.min(ZOOM_MAX * this.fitScale, (this.size - 2 * plotPadding(this.size)) / span));
      this.clampCentre();
      this.requestRedraw();
    }

    finishLasso() {
      const v = this.lasso; this.lasso = [];
      if (v.length < LASSO_MIN_VERTICES) { this.drawOverlay(); return; }
      this.lassoData = v.map(([x, y]) => this.toData(x, y));
      this.reselect();
      this.requestRedraw();
      if (this.selected.length) this.emitSelection();
    }

    reselect() {
      if (!this.lassoData) return;
      this.project();
      const poly = this.lassoData.map(([x, y]) => this.toScreen(x, y));
      const out = [];
      for (let i = 0; i < this.n; i++) if (this.shown(i) && insidePolygon(poly, this.sx[i], this.sy[i])) out.push(i);
      this.selected = out;
    }

    emitSelection() {
      if (!this.selected.length) return;
      window.lookinAct(this.target, "select", { idx: this.selected });
    }

    // ---- legend: show / hide / bring to front, in place ----
    wireLegend() {
      const key = this.root.getAttribute("data-map");
      const legend = document.querySelector('.pm-legend[data-map="' + key + '"]');
      if (!legend) return;
      this.legend = legend;
      legend.querySelectorAll("input[data-g]").forEach((cb) => cb.addEventListener("change", () => {
        const g = +cb.getAttribute("data-g");
        if (cb.checked) this.hidden.delete(g); else this.hidden.add(g);
        this.afterShownChange();
      }));
      legend.querySelectorAll("button[data-front]").forEach((b) => b.addEventListener("click", () => {
        const g = +b.getAttribute("data-front");
        const k = this.front.indexOf(g);
        if (k >= 0) this.front.splice(k, 1); else this.front.push(g);
        legend.querySelectorAll("button[data-front]").forEach((x) => x.classList.toggle("on", this.front.includes(+x.getAttribute("data-front"))));
        this.requestRedraw();
      }));
      const all = legend.querySelector("[data-pm=all]"), none = legend.querySelector("[data-pm=none]");
      const setAll = (vis) => {
        legend.querySelectorAll("input[data-g]").forEach((cb) => {
          cb.checked = vis;
          const g = +cb.getAttribute("data-g");
          if (vis) this.hidden.delete(g); else this.hidden.add(g);
        });
        this.afterShownChange();
      };
      if (all) all.addEventListener("click", () => setAll(true));
      if (none) none.addEventListener("click", () => setAll(false));
    }

    afterShownChange() {
      this.reselect();
      this.hideHover();
      this.requestRedraw();
    }
  }

  const maps = new WeakMap();

  window.lookinInit((doc) => {
    doc.querySelectorAll(".pmap[data-src]").forEach((root) => {
      const src = root.getAttribute("data-src");
      if (root.getAttribute("data-loaded") === src) return;
      root.setAttribute("data-loaded", src);
      fetch(src).then((r) => r.json()).then((data) => {
        const old = maps.get(root);
        if (old && old.ro) old.ro.disconnect();
        maps.set(root, new PMap(root, data));
      }).catch((e) => console.error("lookin: map", e));
    });
  });
})();
