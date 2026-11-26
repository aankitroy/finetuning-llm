// M6's video: one drawing per scene in script.toml. Numbers come from the reference runs
// (outputs/m06/run.json and the rank sweep) and from Qwen2.5-0.5B's real layer sizes.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // a grid of square cells; fill(r, c) gives each cell's colour, op(r, c) its opacity
  function grid(g, K, x, y, rows, cols, cell, fill, op = () => 1) {
    const gg = K.el(g, "g", { transform: `translate(${x} ${y})` });
    const cells = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        cells.push(K.el(gg, "rect", {
          x: c * cell + 1, y: r * cell + 1, width: cell - 2, height: cell - 2, rx: 3,
          fill: fill(r, c), opacity: op(r, c),
        }));
      }
    }
    return { gg, cells };
  }

  // a labelled rounded box
  function box(g, K, x, y, w, h, label, { fill, stroke, ink, size = 24, dash } = {}) {
    const b = K.el(g, "g", { transform: `translate(${x} ${y})` });
    const rect = K.el(b, "rect", {
      width: w, height: h, rx: 12, fill: fill || K.color.paper, stroke: stroke || K.color.line, "stroke-width": 2,
      ...(dash ? { "stroke-dasharray": dash } : {}),
    });
    const txt = K.text(b, w / 2, h / 2 + size * 0.35, label, { "font-size": size, "text-anchor": "middle", fill: ink || K.color.ink, "font-weight": 600 });
    return { b, rect, txt };
  }

  // a small deterministic "random" number in [0, 1) for cell k
  const rnd = (k) => {
    const x = Math.sin(k * 12.9898 + 78.233) * 43758.5453;
    return x - Math.floor(x);
  };

  // ---------------------------------------------------------------- s01
  S.recap = (g, K) => {
    const h = heading(g, K, "In M5, every weight changed");
    const sub = K.text(g, 80, 150, "A full fine-tune of Qwen2.5-0.5B: all 494 million weights trained", { "font-size": 26, fill: K.color.muted });
    const COLS = 14;
    const ROWS = 9;
    const dots = [];
    const dotsG = K.el(g, "g", {});
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) dots.push(K.el(dotsG, "circle", { cx: 100 + c * 32, cy: 220 + r * 32, r: 10, fill: K.color.cobalt }));
    }
    const dotsNote = K.text(g, 80, 530, "changed by training", { "font-size": 22, fill: K.color.amber, "font-weight": 600 });
    const stats = [
      ["Peak memory", "about 24 GB", K.color.ink],
      ["AdamW's notes alone", "3.95 GB", K.color.ink],
      ["A 7B model, the same way", "about 112 GB", K.color.rose],
    ].map(([label, value, color], i) => {
      const s = K.el(g, "g", { transform: `translate(660 ${220 + i * 100})` });
      K.text(s, 0, 0, label, { "font-size": 24, fill: K.color.muted });
      K.text(s, 0, 46, value, { "font-size": 40, "font-weight": 700, fill: color });
      return s;
    });
    const q = K.text(g, 640, 640, "Do we really need to change every weight?", {
      "font-size": 38, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600,
    });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.03));
      const sweep = P(K, t, d, 0.08, 0.2) * dots.length;
      dots.forEach((dot, k) => dot.setAttribute("fill", k < sweep ? K.color.amber : K.color.cobalt));
      K.show(dotsNote, P(K, t, d, 0.22));
      [0.3, 0.45, 0.62].forEach((a, i) => K.show(stats[i], P(K, t, d, a), 14));
      K.show(q, P(K, t, d, 0.8), 14);
    };
  };

  // ---------------------------------------------------------------- s02
  S.idea = (g, K) => {
    const h = heading(g, K, "The idea: keep W, learn a small change");
    const N = 6;
    const C = 40;
    const Y = 200;
    const shade = (r, c) => 0.35 + 0.6 * rnd(r * N + c);
    const nudge = (r, c) => 0.15 + 0.5 * rnd(100 + r * N + c);
    const W = grid(g, K, 110, Y, N, N, C, () => K.color.cobalt, shade);
    const dW = grid(g, K, 510, Y, N, N, C, () => K.color.moss, nudge);
    const sum = grid(g, K, 910, Y, N, N, C, () => K.color.cobalt, (r, c) => Math.min(1, shade(r, c) + 0.2 * nudge(r, c)));
    sum.cells.forEach((c) => c.setAttribute("stroke", K.color.moss) || c.setAttribute("stroke-width", 3));
    const sign = (x, s) => K.text(g, x, Y + 140, s, { "font-size": 64, "text-anchor": "middle", fill: K.color.muted });
    const plus = sign(430, "+");
    const eq = sign(830, "=");
    const label = (x, top, bottom, color) => {
      const l = K.el(g, "g", {});
      K.text(l, x + 120, Y + 290, top, { "font-size": 34, "text-anchor": "middle", "font-weight": 700, fill: color });
      K.text(l, x + 120, Y + 326, bottom, { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
      return l;
    };
    const lW = label(110, "W", "frozen: never changes", K.color.cobalt);
    const lD = label(510, "ΔW", "learned: a small change", K.color.moss);
    const lS = label(910, "W + ΔW", "what the layer uses", K.color.ink);
    const lock = K.el(g, "g", { transform: `translate(${110 + 240 - 30} ${Y - 40})` });
    K.el(lock, "rect", { x: 0, y: 12, width: 28, height: 22, rx: 4, fill: K.color.muted });
    K.el(lock, "path", { d: "M6 12 v-5 a8 8 0 0 1 16 0 v5", fill: "none", stroke: K.color.muted, "stroke-width": 4 });
    const note = K.text(g, 640, 640, "Only ΔW is trained, and it can be made from far fewer numbers than W.", {
      "font-size": 28, "text-anchor": "middle", "font-weight": 600,
    });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(W.gg, P(K, t, d, 0.05), 14);
      K.show(lW, P(K, t, d, 0.12));
      K.show(lock, P(K, t, d, 0.14));
      K.show(plus, P(K, t, d, 0.3));
      K.show(dW.gg, P(K, t, d, 0.32), 14);
      K.show(lD, P(K, t, d, 0.36));
      K.show(eq, P(K, t, d, 0.5));
      K.show(sum.gg, P(K, t, d, 0.52), 14);
      K.show(lS, P(K, t, d, 0.56));
      K.show(note, P(K, t, d, 0.78), 12);
    };
  };

  // ---------------------------------------------------------------- s03
  S.low_rank = (g, K) => {
    const h = heading(g, K, "Low rank: ΔW = B · A");
    // left: a rank-1 example, one column times one row
    const u = [1, 0.6, 0.3, 0.85, 0.5];
    const v = [0.9, 0.4, 1, 0.25, 0.7];
    const C = 44;
    const GX = 160;
    const GY = 250;
    const demo = K.el(g, "g", {});
    K.text(demo, 80, 158, "Rank 1: one column × one row", { "font-size": 26, "font-weight": 600 });
    const row = grid(demo, K, GX, GY - C - 10, 1, 5, C, () => K.color.amber, (r, c) => 0.3 + 0.7 * v[c]);
    const col = grid(demo, K, GX - C - 10, GY, 5, 1, C, () => K.color.amber, (r) => 0.3 + 0.7 * u[r]);
    const cellOp = (r, c) => 0.12 + 0.88 * u[r] * v[c];
    const prod = grid(demo, K, GX, GY, 5, 5, C, () => K.color.moss, cellOp);
    const demoNote = K.el(g, "g", {});
    K.text(demoNote, 80, 510, "5 + 5 numbers fill all 25 cells.", { "font-size": 24 });
    K.text(demoNote, 80, 548, "The rank is how many column-row pairs you use.", { "font-size": 22, fill: K.color.muted });

    // right: Qwen's query projection at rank 8 (the thin matrices are drawn wider than to scale)
    const R = K.el(g, "g", {});
    K.text(R, 640, 158, "q_proj: 896 × 896", { "font-size": 26, "font-weight": 600, "font-family": K.color.mono });
    K.el(R, "rect", { x: 640, y: 190, width: 220, height: 220, rx: 6, fill: K.color.cobaltSoft, stroke: K.color.cobalt, "stroke-width": 2 });
    K.text(R, 750, 296, "W", { "font-size": 40, "text-anchor": "middle", "font-weight": 700, fill: K.color.cobalt });
    K.text(R, 750, 330, "802,816 numbers", { "font-size": 20, "text-anchor": "middle", fill: K.color.cobalt });
    const thin = K.el(g, "g", {});
    K.el(thin, "rect", { x: 920, y: 190, width: 18, height: 220, rx: 4, fill: K.color.moss });
    K.text(thin, 929, 440, "B", { "font-size": 26, "text-anchor": "middle", "font-weight": 700, fill: K.color.moss });
    K.text(thin, 929, 466, "896 × 8", { "font-size": 18, "text-anchor": "middle", fill: K.color.muted, "font-family": K.color.mono });
    K.text(thin, 964, 214, "·", { "font-size": 56, "text-anchor": "middle", fill: K.color.muted });
    K.el(thin, "rect", { x: 990, y: 190, width: 220, height: 18, rx: 4, fill: K.color.moss });
    K.text(thin, 1100, 244, "A", { "font-size": 26, "text-anchor": "middle", "font-weight": 700, fill: K.color.moss });
    K.text(thin, 1100, 270, "8 × 896", { "font-size": 18, "text-anchor": "middle", fill: K.color.muted, "font-family": K.color.mono });
    const sum = K.el(g, "g", {});
    K.text(sum, 640, 520, "7,168 + 7,168 = 14,336 numbers", { "font-size": 28, "font-weight": 600 });
    K.text(sum, 640, 562, "under 2% of 802,816 (1.8%)", { "font-size": 26, fill: K.color.moss, "font-weight": 600 });
    const foot = K.text(g, 640, 655, "B · A still has the full 896 × 896 shape.", { "font-size": 28, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(demo, P(K, t, d, 0.12));
      // the grid fills cell by cell: each cell is its row's number times its column's number
      const filled = P(K, t, d, 0.18, 0.18) * prod.cells.length;
      prod.cells.forEach((cell, k) => (cell.style.opacity = k < filled ? cellOp(Math.floor(k / 5), k % 5) : 0.06));
      const r = Math.min(4, Math.floor(filled / 5));
      const c = Math.min(4, Math.floor(filled) % 5);
      const live = filled > 0 && filled < prod.cells.length;
      row.cells.forEach((cell, k) => cell.setAttribute("stroke", live && k === c ? K.color.ink : "none"));
      col.cells.forEach((cell, k) => cell.setAttribute("stroke", live && k === r ? K.color.ink : "none"));
      K.show(demoNote, P(K, t, d, 0.3));
      K.show(R, P(K, t, d, 0.42), 14);
      K.show(thin, P(K, t, d, 0.6), 14);
      K.show(sum, P(K, t, d, 0.72), 12);
      K.show(foot, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s04
  S.layer = (g, K) => {
    const h = heading(g, K, "A LoRA layer: frozen base plus a side path");
    const formula = K.text(g, 640, 170, "output = base(x) + (alpha / r) · B · A · x", {
      "font-size": 32, "text-anchor": "middle", "font-family": K.color.mono,
    });
    const wires = K.el(g, "g", {});
    const wire = (dd, color) => K.el(wires, "path", { d: dd, fill: "none", stroke: color || K.color.muted, "stroke-width": 3 });
    wire("M180 355 H240 V285 H300");
    wire("M680 285 H940 V327");
    wire("M968 355 H1010");
    const side = K.el(g, "g", {});
    const sw = (dd) => K.el(side, "path", { d: dd, fill: "none", stroke: K.color.moss, "stroke-width": 3 });
    sw("M240 355 V435 H300");
    sw("M470 435 H510");
    sw("M680 435 H720");
    sw("M860 435 H940 V383");
    const x = box(g, K, 90, 325, 90, 60, "x", { size: 28 });
    const base = box(g, K, 300, 250, 380, 70, "base: W, frozen", { fill: K.color.cobaltSoft, stroke: K.color.cobalt, ink: K.color.cobalt });
    const A = box(side, K, 300, 400, 170, 70, "A: random", { stroke: K.color.moss, ink: K.color.moss, size: 22 });
    const B = box(side, K, 510, 400, 170, 70, "B: zeros", { stroke: K.color.moss, ink: K.color.moss, size: 22 });
    const sc = box(side, K, 720, 400, 140, 70, "× alpha/r", { stroke: K.color.moss, ink: K.color.moss, size: 22 });
    const plus = K.el(g, "g", {});
    K.el(plus, "circle", { cx: 940, cy: 355, r: 28, fill: K.color.paper, stroke: K.color.ink, "stroke-width": 2 });
    K.text(plus, 940, 366, "+", { "font-size": 34, "text-anchor": "middle" });
    const out = box(g, K, 1010, 325, 170, 60, "output", { size: 24 });
    const zero = K.text(g, 80, 540, "At the start, B = 0, so B · A = 0: the output is exactly base(x).", { "font-size": 28, fill: K.color.moss, "font-weight": 600 });
    const test = K.text(g, 80, 582, "The checkpoint tests this.", { "font-size": 24, fill: K.color.muted });
    const both = K.text(g, 80, 650, "If A and B both started at zero, neither would get a gradient: nothing would train.", { "font-size": 24, fill: K.color.rose });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(formula, P(K, t, d, 0.04));
      K.show(x.b, P(K, t, d, 0.08));
      K.show(base.b, P(K, t, d, 0.1));
      K.show(wires, P(K, t, d, 0.1));
      K.show(plus, P(K, t, d, 0.12));
      K.show(out.b, P(K, t, d, 0.14));
      K.show(side, P(K, t, d, 0.24), 12);
      // B lights up when the narration says it starts at zero
      const z = P(K, t, d, 0.4) > 0.5;
      B.rect.setAttribute("fill", z ? K.color.moss : K.color.paper);
      B.txt.setAttribute("fill", z ? K.color.surface : K.color.moss);
      K.show(zero, P(K, t, d, 0.5), 10);
      K.show(test, P(K, t, d, 0.66));
      K.show(both, P(K, t, d, 0.78), 10);
    };
  };

  // ---------------------------------------------------------------- s05
  S.scale = (g, K) => {
    const h = heading(g, K, "alpha and r: a volume knob on the update");
    const f = K.el(g, "text", { x: 640, y: 220, "font-size": 48, "text-anchor": "middle", "font-family": K.color.mono, fill: K.color.ink });
    const parts = ["scale = alpha / r", " = 16 / 8", " = 2"].map((s, i) => K.el(f, "tspan", i === 2 ? { fill: K.color.moss, "font-weight": 700 } : {}, s));
    // the knob
    const kx = 330;
    const ky = 420;
    const knob = K.el(g, "g", {});
    K.el(knob, "circle", { cx: kx, cy: ky, r: 100, fill: K.color.paper, stroke: K.color.line, "stroke-width": 4 });
    const ang = (s) => (-135 + (270 * s) / 4) * (Math.PI / 180);
    [0, 1, 2, 3, 4].forEach((s) => {
      const a = ang(s);
      K.el(knob, "line", { x1: kx + 110 * Math.sin(a), y1: ky - 110 * Math.cos(a), x2: kx + 122 * Math.sin(a), y2: ky - 122 * Math.cos(a), stroke: K.color.muted, "stroke-width": 3 });
      K.text(knob, kx + 146 * Math.sin(a), ky - 146 * Math.cos(a) + 8, String(s), { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
    });
    const pointer = K.el(knob, "line", { x1: kx, y1: ky, x2: kx, y2: ky - 80, stroke: K.color.moss, "stroke-width": 8, "stroke-linecap": "round" });
    K.el(knob, "circle", { cx: kx, cy: ky, r: 12, fill: K.color.moss });
    const knobLbl = K.text(g, kx, ky + 140, "", { "font-size": 26, "text-anchor": "middle", "font-weight": 600 });
    // the update it controls
    const upd = K.el(g, "g", {});
    K.text(upd, 640, 330, "B · A · x, as A and B make it", { "font-size": 22, fill: K.color.muted });
    K.el(upd, "rect", { x: 640, y: 346, width: 140, height: 36, rx: 6, fill: K.color.line });
    K.text(upd, 640, 440, "what the layer adds: scale × B · A · x", { "font-size": 22, fill: K.color.muted });
    const bar = K.el(upd, "rect", { x: 640, y: 456, width: 0, height: 36, rx: 6, fill: K.color.moss });
    const foot = K.text(g, 640, 635, "Turn it up, and the same small matrices push the model harder.", { "font-size": 28, "text-anchor": "middle", "font-weight": 600 });
    const base = K.text(g, 640, 676, "The base model is never touched.", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      parts.forEach((p, i) => (p.style.opacity = P(K, t, d, [0.04, 0.3, 0.42][i])));
      K.show(knob, P(K, t, d, 0.45));
      K.show(upd, P(K, t, d, 0.5));
      // the knob turns to 2, then up to 3 and back while the narration says "turn it up"
      let s = 2 * P(K, t, d, 0.46, 0.1);
      const wig = P(K, t, d, 0.68, 0.08) - P(K, t, d, 0.84, 0.08);
      s += wig;
      const a = ang(s);
      pointer.setAttribute("x2", kx + 80 * Math.sin(a));
      pointer.setAttribute("y2", ky - 80 * Math.cos(a));
      knobLbl.textContent = `scale ${s.toFixed(1)}`;
      knobLbl.style.opacity = P(K, t, d, 0.46);
      bar.setAttribute("width", 140 * s);
      K.show(foot, P(K, t, d, 0.68));
      K.show(base, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s06
  S.inject = (g, K) => {
    const h = heading(g, K, "Into the model: 96 LoRA layers");
    const cols = [
      { x: 150, w: 50, label: "q" }, { x: 210, w: 50, label: "k" }, { x: 270, w: 50, label: "v" }, { x: 330, w: 50, label: "o" },
      { x: 410, w: 130, label: "MLP, rest" },
    ];
    const stack = K.el(g, "g", {});
    cols.forEach((c) => K.text(stack, c.x + c.w / 2, 168, c.label, { "font-size": 20, "text-anchor": "middle", fill: K.color.muted, "font-family": K.color.mono }));
    const chips = [];
    for (let r = 0; r < 24; r++) {
      cols.forEach((c, i) => {
        const rect = K.el(stack, "rect", { x: c.x, y: 180 + r * 17, width: c.w, height: 13, rx: 3, fill: K.color.line });
        if (i < 4) chips.push(rect);
      });
    }
    K.text(stack, 80, 625, "24 layers, each with 4 attention projections", { "font-size": 22, fill: K.color.muted });
    const counter = K.text(g, 640, 200, "", { "font-size": 34, "font-weight": 700, fill: K.color.moss });
    const sizes = K.el(g, "g", {});
    K.text(sizes, 640, 260, "q, o: 896 → 896, +14,336 each", { "font-size": 24, "font-family": K.color.mono });
    K.text(sizes, 640, 300, "k, v: 896 → 128, +8,192 each", { "font-size": 24, "font-family": K.color.mono });
    const frozen = K.text(g, 640, 360, "Everything else: frozen", { "font-size": 26, fill: K.color.muted });
    const total = K.el(g, "g", {});
    K.text(total, 640, 450, "1,081,344 trainable", { "font-size": 42, "font-weight": 700, fill: K.color.moss });
    K.text(total, 640, 494, "of about 495 million: 0.22%", { "font-size": 28 });
    K.el(total, "rect", { x: 640, y: 530, width: 560, height: 40, rx: 6, fill: K.color.line });
    K.el(total, "rect", { x: 640, y: 530, width: 4, height: 40, fill: K.color.moss });
    K.text(total, 652, 600, "← this sliver is the part that trains", { "font-size": 20, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(stack, P(K, t, d, 0.04));
      const on = P(K, t, d, 0.5, 0.14) * chips.length;
      chips.forEach((c, k) => c.setAttribute("fill", k < on ? K.color.moss : K.color.cobalt));
      counter.textContent = on > 0 ? `LoRA layers: ${Math.round(on)}` : "";
      K.show(sizes, P(K, t, d, 0.2));
      K.show(frozen, P(K, t, d, 0.6));
      K.show(total, P(K, t, d, 0.68), 14);
    };
  };

  // ---------------------------------------------------------------- s07
  S.training = (g, K) => {
    const h = heading(g, K, "Training: same task, same 400 examples");
    const sub = K.text(g, 80, 150, "jargon → plain English, the data from M5", { "font-size": 26, fill: K.color.muted });
    const chart = K.el(g, "g", { transform: "translate(100 0)" });
    const BASE = 560;
    const scale = 320 / 1.6;
    K.text(chart, 0, 205, "held-out loss (lower is better)", { "font-size": 22, fill: K.color.muted });
    K.el(chart, "line", { x1: 0, y1: BASE, x2: 500, y2: BASE, stroke: K.color.line, "stroke-width": 2 });
    const bars = [
      { x: 20, v: 1.554, label: "before", sub: "", color: K.color.cobalt },
      { x: 180, v: 0.94, label: "M5: full", sub: "fine-tune", color: K.color.amber },
      { x: 340, v: 0.903, label: "LoRA", sub: "rank 8", color: K.color.moss },
    ].map((b) => {
      const r = K.el(chart, "rect", { x: b.x, width: 130, rx: 6, fill: b.color });
      const v = K.text(chart, b.x + 65, 0, String(b.v), { "font-size": 30, "text-anchor": "middle", "font-weight": 600 });
      K.text(chart, b.x + 65, BASE + 30, b.label, { "font-size": 22, "text-anchor": "middle" });
      K.text(chart, b.x + 65, BASE + 56, b.sub, { "font-size": 20, "text-anchor": "middle", fill: K.color.muted });
      return { r, v, b };
    });
    const card = (y, label, value, note) => {
      const c = K.el(g, "g", { transform: `translate(700 ${y})` });
      K.el(c, "rect", { width: 500, height: 120, rx: 14, fill: K.color.paper, stroke: K.color.line });
      K.text(c, 24, 36, label, { "font-size": 22, fill: K.color.muted });
      K.text(c, 24, 78, value, { "font-size": 32, "font-weight": 700 });
      if (note) K.text(c, 24, 106, note, { "font-size": 18, fill: K.color.muted });
      return c;
    };
    const cards = [
      card(180, "AdamW's notes", "3.95 GB → 8.7 MB"),
      card(315, "Peak memory", "about 24 GB → about 20 GB", "frozen weights and activations still there (M8)"),
      card(450, "Training time", "about 22 seconds"),
    ];
    const foot = K.text(g, 640, 670, "A little lower than the full fine-tune, training 0.22% of the weights.", { "font-size": 28, "text-anchor": "middle", "font-weight": 600, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.03));
      K.show(chart, P(K, t, d, 0.14));
      bars.forEach(({ r, v, b }, i) => {
        const p = P(K, t, d, [0.16, 0.3, 0.24][i], 0.08);
        const hgt = b.v * scale * p;
        r.setAttribute("y", BASE - hgt);
        r.setAttribute("height", hgt);
        v.setAttribute("y", BASE - hgt - 12);
        v.style.opacity = p;
      });
      [0.4, 0.55, 0.88].forEach((a, i) => K.show(cards[i], P(K, t, d, a), 14));
      K.show(foot, P(K, t, d, 0.93));
    };
  };

  // ---------------------------------------------------------------- s08
  // An example from the reference run's held-out rewrites, and the 10 ordinary questions.
  S.example = (g, K) => {
    const h = heading(g, K, "On a sentence it never saw");
    const card = (y, hgt, label, text, color, size) => {
      const c = K.el(g, "g", { transform: `translate(80 ${y})` });
      K.el(c, "rect", { width: 1120, height: hgt, rx: 14, fill: K.color.paper, stroke: color, "stroke-width": 2 });
      K.text(c, 28, 42, label, { "font-size": 22, fill: color, "font-weight": 600 });
      return { c, p: K.para(c, 28, 88, text, { chars: 64, lh: 40, "font-size": size }) };
    };
    const src = card(140, 160, "Jargon", "As a value-add, all orders over $75 now qualify for free next-day delivery within the continental US.", K.color.muted, 30);
    const out = card(340, 130, "LoRA, rank 8", "All orders over $75 get free next-day shipping within the US.", K.color.moss, 30);
    const qs = K.el(g, "g", {});
    K.text(qs, 80, 528, "Ten ordinary questions, asked afterwards", { "font-size": 22, fill: K.color.muted });
    const marks = [];
    for (let i = 0; i < 10; i++) {
      const m = K.el(qs, "g", { transform: `translate(${100 + i * 56} 570)` });
      K.el(m, "circle", { r: 22, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
      K.text(m, 0, 9, "✓", { "font-size": 24, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      marks.push(m);
    }
    const score = K.text(g, 690, 580, "10 of 10 right", { "font-size": 32, "font-weight": 700, fill: K.color.moss });
    const foot = K.text(g, 80, 660, "The original weights never changed.", { "font-size": 26, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(src.c, P(K, t, d, 0.04), 14);
      K.show(out.c, P(K, t, d, 0.42), 14);
      const typed = P(K, t, d, 0.42, 0.22);
      out.p.words.forEach((w, i) => (w.style.opacity = i < typed * out.p.words.length ? 1 : 0));
      K.show(qs, P(K, t, d, 0.72));
      marks.forEach((m, i) => K.show(m, P(K, t, d, 0.73 + i * 0.012, 0.03)));
      K.show(score, P(K, t, d, 0.86));
      K.show(foot, P(K, t, d, 0.92));
    };
  };

  // ---------------------------------------------------------------- s09
  S.merge = (g, K) => {
    const h = heading(g, K, "Merging: fold the change back in");
    const formula = K.text(g, 340, 200, "W′ = W + 2 · B · A", { "font-size": 46, "text-anchor": "middle", "font-weight": 600 });
    // left: a LoRA layer becomes one plain layer
    const lora = K.el(g, "g", { transform: "translate(80 250)" });
    K.el(lora, "rect", { width: 230, height: 130, rx: 14, fill: K.color.paper, stroke: K.color.line, "stroke-width": 2 });
    K.el(lora, "rect", { x: 20, y: 20, width: 110, height: 90, rx: 8, fill: K.color.cobaltSoft, stroke: K.color.cobalt });
    K.text(lora, 75, 74, "W", { "font-size": 30, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 700 });
    K.el(lora, "rect", { x: 150, y: 20, width: 60, height: 90, rx: 8, fill: K.color.moss });
    K.text(lora, 180, 74, "B·A", { "font-size": 20, "text-anchor": "middle", fill: K.color.surface, "font-weight": 700 });
    K.text(lora, 115, 160, "LoRA layer", { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
    const arrow = K.text(g, 350, 330, "→", { "font-size": 56, "text-anchor": "middle", fill: K.color.muted });
    const plain = K.el(g, "g", { transform: "translate(390 250)" });
    K.el(plain, "rect", { width: 230, height: 130, rx: 14, fill: K.color.cobaltSoft, stroke: K.color.cobalt, "stroke-width": 2 });
    K.text(plain, 115, 78, "W′", { "font-size": 36, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 700 });
    K.text(plain, 115, 160, "one plain layer", { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
    const facts = K.el(g, "g", {});
    K.text(facts, 80, 480, "held-out loss: still 0.903", { "font-size": 26, "font-weight": 600 });
    K.text(facts, 80, 520, "just as fast as the original", { "font-size": 26 });
    // right: or keep the adapter on its own
    const files = K.el(g, "g", {});
    K.text(files, 700, 200, "Or save the adapter on its own", { "font-size": 26, "font-weight": 600 });
    K.el(files, "rect", { x: 700, y: 230, width: 500, height: 40, rx: 6, fill: K.color.cobalt });
    K.text(files, 700, 300, "the model (fp32): about 2 GB", { "font-size": 22, fill: K.color.muted });
    K.el(files, "rect", { x: 700, y: 330, width: 4, height: 40, fill: K.color.moss });
    K.text(files, 716, 358, "the adapter: 4.4 MB", { "font-size": 24, fill: K.color.moss, "font-weight": 600 });
    const many = K.el(g, "g", {});
    K.el(many, "rect", { x: 700, y: 440, width: 200, height: 100, rx: 12, fill: K.color.cobaltSoft, stroke: K.color.cobalt, "stroke-width": 2 });
    K.text(many, 800, 498, "base model", { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600 });
    const adapters = [0, 1, 2].map((i) => {
      const a = K.el(many, "g", {});
      const y = 420 + i * 52;
      K.el(a, "path", { d: `M900 490 C 940 490, 940 ${y + 20}, 980 ${y + 20}`, fill: "none", stroke: K.color.moss, "stroke-width": 2 });
      K.el(a, "rect", { x: 980, y, width: 220, height: 40, rx: 20, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
      K.text(a, 1090, y + 27, `adapter for job ${i + 1}`, { "font-size": 20, "text-anchor": "middle", fill: K.color.moss });
      return a;
    });
    const foot = K.text(g, 640, 660, "One base model, many small adapters, one for each job.", { "font-size": 28, "text-anchor": "middle", "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(lora, P(K, t, d, 0.06));
      K.show(formula, P(K, t, d, 0.14));
      K.show(arrow, P(K, t, d, 0.28));
      K.show(plain, P(K, t, d, 0.3), 12);
      K.show(facts, P(K, t, d, 0.4));
      K.show(files, P(K, t, d, 0.6));
      K.show(many, P(K, t, d, 0.78));
      adapters.forEach((a, i) => K.show(a, P(K, t, d, 0.8 + i * 0.03)));
      K.show(foot, P(K, t, d, 0.9));
    };
  };

  // ---------------------------------------------------------------- s10
  S.choices = (g, K) => {
    const h = heading(g, K, "Which rank? Bigger isn't automatically better");
    const sub = K.text(g, 80, 150, "Held-out loss after training (lower is better), and the share of weights trained", { "font-size": 24, fill: K.color.muted });
    const head = K.el(g, "g", {});
    K.text(head, 420, 205, "held-out loss", { "font-size": 20, fill: K.color.muted });
    K.text(head, 1200, 205, "trained", { "font-size": 20, fill: K.color.muted, "text-anchor": "end" });
    const runs = [
      ["rank 2", 0.94, "0.05%"],
      ["rank 8", 0.903, "0.22%"],
      ["rank 32", 0.907, "0.88%"],
      ["rank 8, attention + MLP", 0.913, "0.89%"],
    ];
    const rows = runs.map(([label, loss, pct], i) => {
      const y = 225 + i * 95;
      const r = K.el(g, "g", {});
      const best = i === 1;
      K.text(r, 80, y + 38, label, { "font-size": 26, "font-weight": best ? 700 : 400 });
      const bar = K.el(r, "rect", { x: 420, y, width: 0, height: 56, rx: 6, fill: best ? K.color.moss : K.color.cobalt });
      const val = K.text(r, 420, y + 38, loss.toFixed(3), { "font-size": 26, "font-weight": 600 });
      K.text(r, 1200, y + 38, pct, { "font-size": 26, "text-anchor": "end", fill: K.color.muted });
      return { r, bar, val, loss };
    });
    const foot = K.text(g, 640, 650, "Here, rank 8 on attention alone did best.", { "font-size": 30, "text-anchor": "middle", "font-weight": 600, fill: K.color.moss });
    const W = 600;
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.03));
      K.show(head, P(K, t, d, 0.05));
      [0.1, 0.32, 0.45, 0.6].forEach((a, i) => {
        const row = rows[i];
        const p = P(K, t, d, a, 0.08);
        K.show(row.r, Math.min(1, p * 3));
        const w = row.loss * W * p;
        row.bar.setAttribute("width", w);
        row.val.setAttribute("x", 420 + w + 12);
      });
      K.show(foot, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s11
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      "1. LoRALinear: the layer",
      "2. lora_params: count its numbers",
      "3. inject_lora: put it in the model",
      "4. count_trainable: what will train",
      "5. merge_lora: fold it back in",
      "train.py: train your adapter",
      "make check M=06",
    ];
    const rows = items.map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(${i < 5 ? 80 : 760} ${i < 5 ? 170 + i * 70 : 170 + (i - 5) * 70})` });
      const bx = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 24, "font-family": K.color.mono });
      return { r, bx, tick };
    });
    const first = K.text(g, 760, 400, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    const real = K.text(g, 760, 442, "It builds every piece with real code.", { "font-size": 24, fill: K.color.muted });
    const next = K.text(g, 80, 600, "Next, in M7: the same thing with the real tools.", { "font-size": 30, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(first, P(K, t, d, 0.06));
      K.show(real, P(K, t, d, 0.08));
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, 0.2 + i * 0.075));
        const done = P(K, t, d, 0.26 + i * 0.075, 0.03);
        row.tick.style.opacity = done;
        row.bx.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(next, P(K, t, d, 0.84));
    };
  };
})();
