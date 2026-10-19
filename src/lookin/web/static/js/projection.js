/* The Embeddings tab's interactive map -- the browser port of
 * lookin.ui.interactive_projection.
 *
 * The server sends the projected cloud once (a JSON file: coordinates in
 * data space, a colour and a group per point, the prototypes); everything a
 * gesture does happens here, on a canvas: hover shows the point's crop (the
 * original file, through the session's display setting, fetched from
 * /pimg only for the points the pointer lands on), a click opens it in the
 * carousel, a lasso selects a group and opens it there, the wheel / the
 * box / the buttons zoom, a drag pans -- past the edge of the cloud too, so
 * a point on the rim can be brought to the middle. The legend hides groups
 * and brings them to the front in place.
 *
 * One transform (toScreen) both places the dots and answers "which point is
 * under the cursor", so the two cannot drift.
 *
 * Gallery mode: the button itself shrinks the map into the top-left corner
 * (so drawing or clearing a lasso never moves the layout); a lasso drawn
 * there fills the rest of the panel with the crops of a random sample of
 * the selected points -- as many as fit. The tiles keep one size (a
 * seventh of the panel's shorter side) and the mosaic goes on as far as the
 * panel does: drag the splitter to the left and it fills the whole width.
 *
 * Capture: "Copy" / "Save" draw what is shown -- the map with its lasso,
 * and in gallery mode the tiles -- again, off screen, at a fixed scale
 * whatever the window or the screen's pixel ratio: the map is always
 * EXPORT_SIZE (3/7 of it in gallery mode) and so is a tile. The same view
 * of the same data always gives the same picture, pixel for pixel, and two
 * captures line up.
 *
 * The splitter between the settings column and the map (.emb-split) is
 * wired at the bottom of this file.
 *
 * Talks to the server only through window.lookinAct.
 */
(function () {
  "use strict";

  const POINT_DIAMETER = 5.0;
  const PICK_RADIUS = 11.0;
  const HOVER_THUMB = 240;
  const HOVER_OFFSET = 16;
  const LASSO_COLOR = "#1f6feb";
  const LASSO_FILL = "rgba(31, 111, 235, 0.10)";
  const ZOOM_BOX_COLOR = "#a04ec9";
  const MARKER_COLOR = "#111111";
  const LASSO_MIN_STEP = 4.0;
  const LASSO_MIN_VERTICES = 4;
  const ZOOM_BOX_MIN_SIDE = 12.0;
  const ZOOM_STEP = 1.25;
  const ZOOM_MAX = 100.0;
  const ZOOM_MIN = 0.5;                 // relative to the fitted view
  // how far a drag may push the cloud off the canvas: its bounding box
  // keeps at least this fraction of the view's half-width on screen
  const PAN_KEEP = 0.1;
  const GALLERY_COLS = 7;               // tiles along the panel's shorter side, by default ...
  const GALLERY_COLS_MIN = 3;           // ... and as the -/+ buttons may set it
  const GALLERY_COLS_MAX = 16;
  const GALLERY_MAP_CELLS = 3;          // the corner map spans 3 x 3 tiles by default
  const GALLERY_MAP_CELLS_MIN = 2;      // ... and its grip snaps it to whole tiles
  const GALLERY_KEY = "lookin.embGallery";
  const GALLERY_SAMPLE_MAX = 2000;      // crops drawn per selection
  const EXPORT_SIZE = 800;              // capture side, CSS px ...
  const EXPORT_RATIO = 2;               // ... drawn at 2x: a 1600 x 1600 PNG
  const EXPORT_LINE = "#dfe2e8";        // the box's border colour (--line)
  const EXPORT_IMAGE_WAIT_MS = 5000;
  const PROTOTYPE_RADIUS = 9.0;
  const SUBCENTER_SHAPES = ["star", "plus", "cross", "diamond", "triangle_up", "triangle_down", "square", "pentagon"];
  const PLOT_PADDING_MIN = 22;
  const PLOT_PADDING_FRAC = 0.06;
  const MIN_PLOT_SIZE = 420;

  function plotPadding(size) { return Math.max(PLOT_PADDING_MIN, size * PLOT_PADDING_FRAC); }
  function galleryMapSide(cell, cells) { return Math.floor(cells * cell); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  // tile density and map size: kept in this browser, like the splitter
  function loadGalleryPrefs() {
    try { return JSON.parse(localStorage.getItem(GALLERY_KEY) || "{}") || {}; } catch (e) { return {}; }
  }
  function saveGalleryPrefs(p) {
    try { localStorage.setItem(GALLERY_KEY, JSON.stringify(p)); } catch (e) { /* private mode */ }
  }

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
      this.gallery = false;             // gallery layout: map in the corner, crops around
      const gp = loadGalleryPrefs();
      this.density = clamp(+gp.density || GALLERY_COLS, GALLERY_COLS_MIN, GALLERY_COLS_MAX);
      this.wantCells = Math.max(GALLERY_MAP_CELLS_MIN, +gp.cells || GALLERY_MAP_CELLS);
      this.pinfoBase = root.getAttribute("data-pinfo") || "";
      this.buildDom();
      this.room = this.measure();       // [w, h] the panel leaves for the box
      this.boxW = this.boxH = Math.min(this.room[0], this.room[1]);
      this.size = this.boxW;            // side of the map canvas inside the box
      this.fit();
      this.wire();
      this.redraw();
      // the panel's width (the splitter) and the window's height both size the box
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.box.parentElement);
      this.onWinResize = () => this.resize();
      window.addEventListener("resize", this.onWinResize);
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
      this.gal = document.createElement("div");
      this.gal.className = "pmap-gal";
      this.gal.setAttribute("data-zoom-list", "");
      this.grip = document.createElement("div");
      this.grip.className = "pmap-grip";
      this.grip.title = "Drag to make the map bigger or smaller";
      this.grip.hidden = true;
      this.box.innerHTML = "";
      this.box.append(this.gal, this.canvas, this.overlay, this.grip, this.card);
      this.zoomText = r.querySelector(".pm-zoomtext");
      this.selText = r.querySelector(".pm-seltext");
      this.btnReset = r.querySelector("[data-pm=reset]");
      this.btnShow = r.querySelector("[data-pm=show]");
      this.btnClear = r.querySelector("[data-pm=clear]");
      this.btnGallery = r.querySelector("[data-pm=gallery]");
      this.btnShuffle = r.querySelector("[data-pm=shuffle]");
    }

    // [width, height] available for the box: the panel's width, and the
    // window below the box's top
    measure() {
      const w = this.box.parentElement.clientWidth || 600;
      const top = this.box.getBoundingClientRect().top;
      const h = window.innerHeight - Math.max(0, top) - 16;
      return [Math.max(MIN_PLOT_SIZE, Math.floor(w)), Math.max(MIN_PLOT_SIZE, Math.floor(h > 200 ? h : w))];
    }

    setupCanvas() {
      const dpr = this.dpr = window.devicePixelRatio || 1;
      for (const c of [this.canvas, this.overlay]) {
        c.width = Math.round(this.size * dpr);
        c.height = Math.round(this.size * dpr);
        c.style.width = this.size + "px";
        c.style.height = this.size + "px";
        c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      this.box.style.width = this.boxW + "px";
      this.box.style.height = this.boxH + "px";
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
      // the centre may wander until only a sliver of the cloud's box is
      // left on screen -- enough room to centre any point of the rim
      const halfSpan = (this.size / 2) / this.scale;
      const fitHalf = (this.size / 2 - plotPadding(this.size)) / this.fitScale;
      const limit = fitHalf + halfSpan * (1 - PAN_KEEP);
      this.cx = Math.min(this.base[0] + limit, Math.max(this.base[0] - limit, this.cx));
      this.cy = Math.min(this.base[1] + limit, Math.max(this.base[1] - limit, this.cy));
    }

    applyZoom(newScale, anchor) {
      newScale = Math.max(ZOOM_MIN * this.fitScale, Math.min(ZOOM_MAX * this.fitScale, newScale));
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
      if (!document.contains(this.box)) {      // the tab was redrawn without this map
        this.ro.disconnect();
        window.removeEventListener("resize", this.onWinResize);
        return;
      }
      const room = this.measure();
      if (Math.abs(room[0] - this.room[0]) < 2 && Math.abs(room[1] - this.room[1]) < 2) return;
      this.room = room;
      this.layout();
    }

    // Size the map canvas for the current layout (the whole box, or its
    // corner in gallery mode) keeping the same view, then lay out the tiles.
    layout() {
      const [W, H] = this.room, square = Math.min(W, H);
      let size = square;
      this.boxW = this.boxH = square;
      if (this.gallery) {
        // tiles of about 1/density of the shorter side, as many whole
        // columns and rows as fit, sized so the grid fills both the width
        // and the height to within a fraction of a tile; the map spans the
        // number of tiles asked for, as long as a column of crops is left
        const c0 = square / this.density;
        this.cols = Math.max(GALLERY_MAP_CELLS_MIN + 1, Math.round(W / c0));
        this.rows = Math.max(GALLERY_MAP_CELLS_MIN, Math.round(H / c0));
        this.mapCells = clamp(this.wantCells, GALLERY_MAP_CELLS_MIN, Math.min(this.rows, this.cols - 1));
        this.cell = Math.min(W / this.cols, H / this.rows);
        this.boxW = Math.floor(this.cols * this.cell);
        this.boxH = Math.floor(this.rows * this.cell);
        size = galleryMapSide(this.cell, this.mapCells);
      }
      const ratio = this.scale / this.fitScale;
      this.size = size;
      this.setupCanvas();
      this.fitScale = (size - 2 * plotPadding(size)) / this.span;
      this.scale = this.fitScale * ratio;
      this.clampCentre();
      this.box.classList.toggle("pm-gallery", this.gallery);
      this.root.classList.toggle("pm-gallery-on", this.gallery);
      this.placeGrip();
      this.fillGallery();
      this.requestRedraw();
    }

    // the corner map's grip: drag it to give the map more or fewer tiles
    placeGrip() {
      this.grip.hidden = !this.gallery;
      if (!this.gallery) return;
      this.grip.style.left = this.size + "px";
      this.grip.style.top = this.size + "px";
    }

    setMapCells(n) {
      n = clamp(n, GALLERY_MAP_CELLS_MIN, Math.min(this.rows, this.cols - 1));
      this.wantCells = n;
      if (n === this.mapCells) return;
      this.hideHover();
      this.layout();
    }

    setDensity(d) {
      d = clamp(d, GALLERY_COLS_MIN, GALLERY_COLS_MAX);
      if (d === this.density) return;
      // the map keeps about the same size on screen: more tiles, more of them under it
      const side = this.mapCells * this.cell;
      this.density = d;
      this.layout();
      this.wantCells = Math.max(GALLERY_MAP_CELLS_MIN, Math.round(side / this.cell));
      this.layout();
      saveGalleryPrefs({ density: this.density, cells: this.wantCells });
    }

    // ---- gallery ----
    // `reshuffle`: draw a new sample. Without it (a resize, the toggle) the
    // tiles keep their crops, so the layout can change size without the
    // pictures changing under the user -- or between two captures.
    fillGallery(reshuffle) {
      if (reshuffle) this.sample = null;
      this.gal.innerHTML = "";
      if (!this.gallery) return;
      const cols = this.cols, rows = this.rows, m = this.mapCells, cell = this.cell;
      const slots = [];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (r >= m || c >= m) slots.push([r, c]);
      this.nShown = 0;
      if (!this.selected.length) {
        const hint = document.createElement("div");
        hint.className = "pmap-gal-hint";
        hint.style.left = (m * cell) + "px";
        hint.style.top = "0px";
        hint.style.width = ((cols - m) * cell) + "px";
        hint.style.height = (m * cell) + "px";
        hint.textContent = "Draw a lasso on the map: the crops it holds fill this space.";
        this.gal.appendChild(hint);
        this.syncToolbar();
        return;
      }
      if (!this.sample) {
        // more than any panel shows, so widening it brings in more crops
        // of the same draw instead of a new one
        const pool = this.selected.slice();
        const n = Math.min(GALLERY_SAMPLE_MAX, pool.length);
        for (let k = 0; k < n; k++) {      // partial Fisher-Yates
          const j = k + Math.floor(Math.random() * (pool.length - k));
          const t = pool[k]; pool[k] = pool[j]; pool[j] = t;
        }
        this.sample = pool.slice(0, n);
      }
      const pool = this.sample;
      const take = Math.min(slots.length, pool.length);
      const frag = document.createDocumentFragment();
      for (let k = 0; k < take; k++) {
        const i = pool[k], [r, c] = slots[k];
        const img = document.createElement("img");
        img.alt = "";
        img.src = this.imgBase + i;
        const label = this.d.hl[this.d.hi[i]] || "";
        img.title = label;
        img.dataset.i = i;
        // a click opens the full-size view (app.js), with < > through the
        // tiles and the (i) details of this point
        img.setAttribute("data-zoom", "");
        img.setAttribute("data-pixelated", "");
        img.setAttribute("data-title", label || "crop");
        if (this.pinfoBase) img.setAttribute("data-info", this.pinfoBase + i);
        img.dataset.r = r; img.dataset.c = c;
        img.style.left = (c * cell) + "px";
        img.style.top = (r * cell) + "px";
        img.style.width = img.style.height = cell + "px";
        frag.appendChild(img);
      }
      this.gal.appendChild(frag);
      this.nShown = take;
      this.syncToolbar();
    }

    setGallery(on) {
      this.gallery = on;
      // the gallery is driven by the lasso: switch to it straight away
      if (on) this.mode = "lasso";
      this.hideHover();
      this.layout();
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
      const dpr = this.dpr || 1;
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
        g.beginPath(); g.moveTo(poly[0][0], poly[0][1]);
        for (let q = 1; q < poly.length; q++) g.lineTo(poly[q][0], poly[q][1]);
        if (!this.lasso.length) {
          g.closePath();
          g.fillStyle = LASSO_FILL; g.fill();
        }
        // on the corner map the selection halos crowd the outline: draw it
        // white under blue so it still reads
        if (this.gallery) { g.strokeStyle = "#ffffff"; g.lineWidth = 5; g.stroke(); }
        g.strokeStyle = LASSO_COLOR; g.lineWidth = this.gallery ? 2.5 : 1.8;
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
      if (this.btnReset) this.btnReset.hidden = Math.abs(this.scale / this.fitScale - 1) < 0.001
        && Math.abs(this.cx - this.base[0]) * this.scale < 1 && Math.abs(this.cy - this.base[1]) * this.scale < 1;
      const has = this.lassoData !== null;
      const n = this.selected.length.toLocaleString("en-US");
      if (this.selText) {
        // short, the toolbar is one row: "40/452" in gallery mode (crops shown / selected)
        const partial = this.gallery && this.nShown && this.nShown < this.selected.length;
        this.selText.textContent = !has ? "" : partial ? this.nShown.toLocaleString("en-US") + "/" + n : n;
        this.selText.title = !has ? "" : n + " points selected" +
          (partial ? ", " + this.nShown.toLocaleString("en-US") + " of them shown, drawn at random" : "");
      }
      if (this.btnShow) this.btnShow.hidden = !this.selected.length;
      if (this.btnClear) this.btnClear.hidden = !has;
      if (this.btnGallery) this.btnGallery.classList.toggle("on", this.gallery);
      if (this.btnShuffle) this.btnShuffle.hidden = !(this.gallery && this.selected.length);
      const less = this.root.querySelector("[data-pm=crop-larger]"), more = this.root.querySelector("[data-pm=crop-smaller]");
      if (less) less.disabled = this.density <= GALLERY_COLS_MIN;
      if (more) more.disabled = this.density >= GALLERY_COLS_MAX;
      this.root.querySelectorAll("[data-pm-mode]").forEach((b) => b.classList.toggle("on", b.getAttribute("data-pm-mode") === this.mode));
      const sel = this.root.querySelector(".pm-g-sel");
      if (sel) sel.classList.toggle("live", this.mode === "lasso" || has);
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
      // the card may spill out of the corner map onto the tiles
      const rw = this.boxW, rh = this.boxH;
      let x = x0 + HOVER_OFFSET, y = y0 + HOVER_OFFSET;
      if (x + w > rw) x = x0 - HOVER_OFFSET - w;
      if (y + h > rh) y = y0 - HOVER_OFFSET - h;
      card.style.left = Math.max(2, Math.min(rw - w - 2, x)) + "px";
      card.style.top = Math.max(2, Math.min(rh - h - 2, y)) + "px";
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
      on("[data-pm=clear]", () => { this.selected = []; this.sample = null; this.lassoData = null; this.fillGallery(true); this.requestRedraw(); });
      on("[data-pm=gallery]", () => this.setGallery(!this.gallery));
      on("[data-pm=shuffle]", () => this.fillGallery(true));
      on("[data-pm=copy]", () => this.copyImage());
      on("[data-pm=save]", () => this.saveImage());
      const copyBtn = this.root.querySelector("[data-pm=copy]");
      if (copyBtn && !this.canCopy()) copyBtn.hidden = true;
      // a tile: hovering rings its point on the map; a click opens the
      // full-size view (its data-zoom, handled by app.js)
      this.gal.addEventListener("pointerover", (ev) => {
        const t = ev.target.closest("img[data-i]");
        this.current = t ? ["p", +t.dataset.i] : null;
        this.drawOverlay();
      });
      this.gal.addEventListener("pointerleave", () => { this.current = null; this.drawOverlay(); });
      on("[data-pm=crop-larger]", () => this.setDensity(this.density - 1));
      on("[data-pm=crop-smaller]", () => this.setDensity(this.density + 1));
      this.wireGrip();
      this.root.querySelectorAll("[data-pm-render]").forEach((b) => b.addEventListener("click", () => {
        const which = b.getAttribute("data-pm-render");
        this.root.classList.toggle("pm-static", which === "static");
        this.root.querySelectorAll("[data-pm-render]").forEach((x) => x.classList.toggle("on", x === b));
      }));
      this.wireLegend();
    }

    wireGrip() {
      const g = this.grip;
      g.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault(); ev.stopPropagation();
        try { g.setPointerCapture(ev.pointerId); } catch (e) { /* a pointer already gone */ }
        this.hideHover();
        const box = this.box.getBoundingClientRect();
        const move = (e) => {
          // the corner follows the pointer, snapped to whole tiles
          const reach = Math.max(e.clientX - box.left, e.clientY - box.top);
          this.setMapCells(Math.round(reach / this.cell));
        };
        const up = () => {
          g.removeEventListener("pointermove", move);
          g.removeEventListener("pointerup", up);
          g.removeEventListener("pointercancel", up);
          saveGalleryPrefs({ density: this.density, cells: this.wantCells });
        };
        g.addEventListener("pointermove", move);
        g.addEventListener("pointerup", up);
        g.addEventListener("pointercancel", up);
      });
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
      this.scale = Math.max(ZOOM_MIN * this.fitScale, Math.min(ZOOM_MAX * this.fitScale, (this.size - 2 * plotPadding(this.size)) / span));
      this.clampCentre();
      this.requestRedraw();
    }

    finishLasso() {
      const v = this.lasso; this.lasso = [];
      if (v.length < LASSO_MIN_VERTICES) { this.drawOverlay(); return; }
      this.lassoData = v.map(([x, y]) => this.toData(x, y));
      this.reselect();
      this.requestRedraw();
      if (this.gallery) { this.fillGallery(true); return; }
      if (this.selected.length) this.emitSelection();
    }

    reselect() {
      if (!this.lassoData) return;
      this.project();
      const poly = this.lassoData.map(([x, y]) => this.toScreen(x, y));
      const out = [];
      for (let i = 0; i < this.n; i++) if (this.shown(i) && insidePolygon(poly, this.sx[i], this.sy[i])) out.push(i);
      this.selected = out;
      this.sample = null;                 // a new selection: a new gallery sample
    }

    emitSelection() {
      if (!this.selected.length) return;
      window.lookinAct(this.target, "select", { idx: this.selected });
    }

    // ---- capture ----
    // The map drawn again at a fixed size: the view (centre, zoom relative
    // to the fit) is kept, everything that depends on the window -- canvas
    // size, pixel ratio -- is swapped for fixed values for one synchronous
    // redraw, then put back.
    renderMap(side, ratio) {
      const keep = {
        size: this.size, canvas: this.canvas, overlay: this.overlay, dpr: this.dpr,
        sprites: this.sprites, fitScale: this.fitScale, scale: this.scale, current: this.current,
      };
      const zoom = this.scale / this.fitScale;
      const make = () => {
        const c = document.createElement("canvas");
        c.width = c.height = Math.round(side * ratio);
        c.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
        return c;
      };
      if (this.raf) { cancelAnimationFrame(this.raf); this.raf = null; }
      this.canvas = make(); this.overlay = make();
      this.size = side; this.dpr = ratio; this.sprites = {}; this.current = null;
      this.fitScale = (side - 2 * plotPadding(side)) / this.span;
      this.scale = this.fitScale * zoom;
      let out;
      try {
        this.redraw();
        out = this.canvas;
        out.getContext("2d").setTransform(1, 0, 0, 1, 0, 0);
        out.getContext("2d").drawImage(this.overlay, 0, 0);
      } finally {
        Object.assign(this, keep);
        this.redraw();
      }
      return out;
    }

    async capture() {
      const W = EXPORT_SIZE, R = EXPORT_RATIO;
      const out = document.createElement("canvas");
      const g = out.getContext("2d");
      if (!this.gallery) {
        out.width = out.height = W * R;
        g.fillStyle = "#ffffff";
        g.fillRect(0, 0, out.width, out.height);
        g.drawImage(this.renderMap(W, R), 0, 0);
        return out;
      }
      // the grid shown (its columns and rows), at a fixed tile size
      const cell = W / GALLERY_COLS, side = galleryMapSide(cell, this.mapCells);
      out.width = Math.round(this.cols * cell * R);
      out.height = Math.round(this.rows * cell * R);
      g.fillStyle = "#f7f8fa";                      // the tiles' backdrop (--surface-2)
      g.fillRect(0, 0, out.width, out.height);
      g.drawImage(this.renderMap(side, R), 0, 0);
      g.fillStyle = EXPORT_LINE;
      g.fillRect(side * R, 0, R, (side + 1) * R);
      g.fillRect(0, side * R, (side + 1) * R, R);
      // each crop loaded again (from the cache) rather than read off the
      // page's <img>, which may not have loaded yet; one that does not
      // come within a few seconds is left black rather than holding up
      // the capture
      const tiles = await Promise.all(Array.from(this.gal.querySelectorAll("img[data-i]")).map((el) => {
        const im = new Image();
        im.dataset.r = el.dataset.r; im.dataset.c = el.dataset.c;
        const loaded = new Promise((ok) => { im.onload = im.onerror = () => ok(im); });
        im.src = el.src;
        return Promise.race([loaded, new Promise((ok) => setTimeout(() => ok(im), EXPORT_IMAGE_WAIT_MS))]);
      }));
      g.imageSmoothingEnabled = false;              // crops are shown with their pixels
      for (const im of tiles) {
        // device-pixel edges from the cell grid, so tiles never overlap or gap
        const x0 = Math.round(+im.dataset.c * cell * R), y0 = Math.round(+im.dataset.r * cell * R);
        const x1 = Math.round((+im.dataset.c + 1) * cell * R), y1 = Math.round((+im.dataset.r + 1) * cell * R);
        g.fillStyle = "#ffffff"; g.fillRect(x0, y0, x1 - x0, y1 - y0);
        const b = R;                                // the tile's 1 px white border
        const iw = x1 - x0 - 2 * b, ih = y1 - y0 - 2 * b;
        g.fillStyle = "#000000"; g.fillRect(x0 + b, y0 + b, iw, ih);
        if (!im.complete || !im.naturalWidth) continue;
        const k = Math.min(iw / im.naturalWidth, ih / im.naturalHeight);
        const w = Math.round(im.naturalWidth * k), h = Math.round(im.naturalHeight * k);
        g.drawImage(im, x0 + b + Math.floor((iw - w) / 2), y0 + b + Math.floor((ih - h) / 2), w, h);
      }
      return out;
    }

    fileName() {
      const d = new Date(), p = (n) => String(n).padStart(2, "0");
      return "lookin_embeddings" + (this.gallery ? "_gallery" : "") + "_" + d.getFullYear() + p(d.getMonth() + 1) +
        p(d.getDate()) + "_" + p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds()) + ".png";
    }

    canCopy() {
      return !!(window.isSecureContext && navigator.clipboard && navigator.clipboard.write && window.ClipboardItem);
    }

    async copyImage() {
      const toast = window.lookinToast || (() => {});
      if (!this.canCopy()) { this.saveImage(); return; }
      try {
        // the blob is handed over as a promise: Safari wants the write
        // started within the click, before the drawing is done
        const blob = this.capture().then((c) => new Promise((ok) => c.toBlob(ok, "image/png")));
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        toast("Image copied to the clipboard");
      } catch (e) {
        console.error("lookin: copy", e);
        toast("The browser refused the clipboard: the image is saved instead", "warn");
        this.saveImage();
      }
    }

    async saveImage() {
      const c = await this.capture();
      c.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = this.fileName();
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      }, "image/png");
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
      if (this.gallery) this.fillGallery(true);
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

  // ---- the splitter between the settings column and the map ----------------
  // Drag it to size the map; drag it (nearly) to the left edge, or
  // double-click it, and the settings column folds away for a full-width
  // map. Kept on <html> -- a CSS variable and a class -- rather than on the
  // tab's elements, which the server's re-renders rewrite; remembered in
  // this browser only.
  const SPLIT_KEY = "lookin.embSplit";
  const SPLIT_FOLD_PX = 200;           // a settings column dragged narrower than this folds
  const html = document.documentElement;

  function applySplit(st) {
    if (st.right) html.style.setProperty("--emb-right", st.right + "px");
    else html.style.removeProperty("--emb-right");
    html.classList.toggle("emb-folded", !!st.folded);
  }
  function loadSplit() {
    try { return JSON.parse(localStorage.getItem(SPLIT_KEY) || "{}") || {}; } catch (e) { return {}; }
  }
  function saveSplit(st) {
    try { localStorage.setItem(SPLIT_KEY, JSON.stringify(st)); } catch (e) { /* private mode */ }
  }
  let split = loadSplit();
  applySplit(split);

  document.addEventListener("pointerdown", (ev) => {
    const bar = ev.target.closest(".emb-split");
    if (!bar || ev.button !== 0) return;
    const emb = bar.closest(".emb");
    if (!emb) return;
    ev.preventDefault();
    bar.setPointerCapture(ev.pointerId);
    html.classList.add("emb-dragging");
    const move = (e) => {
      const r = emb.getBoundingClientRect();
      const left = e.clientX - r.left;
      if (left < SPLIT_FOLD_PX) split = { right: split.right, folded: true };
      else split = { right: Math.round(r.right - e.clientX - bar.offsetWidth / 2), folded: false };
      applySplit(split);
    };
    const up = () => {
      bar.removeEventListener("pointermove", move);
      bar.removeEventListener("pointerup", up);
      bar.removeEventListener("pointercancel", up);
      html.classList.remove("emb-dragging");
      saveSplit(split);
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up);
    bar.addEventListener("pointercancel", up);
  });
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest("[data-emb-unfold]")) return;
    split = { right: split.right, folded: false };
    applySplit(split);
    saveSplit(split);
  });
  document.addEventListener("dblclick", (ev) => {
    if (!ev.target.closest(".emb-split")) return;
    split = { right: split.right, folded: !split.folded };
    applySplit(split);
    saveSplit(split);
  });
})();
