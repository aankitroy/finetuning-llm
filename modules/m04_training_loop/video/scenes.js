// M4's video: one drawing per scene in script.toml. Every number and example comes from
// the lesson, the README or train.py (the curves in s05 and s09 are the lesson's own runs).
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // a polyline through points [[x, y], ...], drawn up to fraction p of its length
  function line(g, K, pts, attrs) {
    const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
    const path = K.el(g, "path", { d, fill: "none", "stroke-width": 4, "stroke-linejoin": "round", "stroke-linecap": "round", ...attrs });
    const len = path.getTotalLength ? path.getTotalLength() || 1 : 1;
    path.setAttribute("stroke-dasharray", `${len} ${len}`);
    return (p) => path.setAttribute("stroke-dashoffset", len * (1 - p));
  }

  // simple chart axes: returns functions mapping data to stage coordinates
  function axes(g, K, { x, y, w, h, xmax, ymax, xlabel, ylabel, xticks, yticks }) {
    const ax = K.el(g, "g", {});
    K.el(ax, "line", { x1: x, y1: y + h, x2: x + w, y2: y + h, stroke: K.color.line, "stroke-width": 2 });
    K.el(ax, "line", { x1: x, y1: y, x2: x, y2: y + h, stroke: K.color.line, "stroke-width": 2 });
    const X = (v) => x + (v / xmax) * w;
    const Y = (v) => y + h - (v / ymax) * h;
    (xticks || []).forEach(([v, s]) => K.text(ax, X(v), y + h + 30, s, { "font-size": 22, fill: K.color.muted, "text-anchor": "middle" }));
    (yticks || []).forEach(([v, s]) => K.text(ax, x - 12, Y(v) + 8, s, { "font-size": 22, fill: K.color.muted, "text-anchor": "end" }));
    if (xlabel) K.text(ax, x + w, y + h + 62, xlabel, { "font-size": 22, fill: K.color.muted, "text-anchor": "end" });
    if (ylabel) K.text(ax, x, y - 18, ylabel, { "font-size": 22, fill: K.color.muted });
    return { ax, X, Y };
  }

  // ---------------------------------------------------------------- s01
  // The untrained model's 80 characters, exactly as the lesson's first cell prints them.
  const GIBBERISH = ["AJJ?z,eqERkFNnBg-XNR3rTBVSgcgRExBico$dDe", "oSz?q$KttvJUxi ;TPSZjjJHSYIQhdM'HrRjeOp"];
  const SAMPLE = ["First Senator:", "I will live carry my life, and bound I have,", "For then strew-struction the happy offence,"];

  S.intro = (g, K) => {
    const h = heading(g, K, "Training: four beats, repeated");
    const sub = K.text(g, 80, 150, "From random numbers to something that writes like Shakespeare", { "font-size": 26, fill: K.color.muted });
    const beats = ["make a guess", "measure how wrong: the loss", "work out the gradients", "nudge every weight a little"].map((s, i) => {
      const b = K.el(g, "g", { transform: `translate(80 ${210 + i * 80})` });
      const dot = K.el(b, "circle", { cx: 24, cy: 24, r: 24, fill: K.color.cobaltSoft });
      K.text(b, 24, 33, String(i + 1), { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 700 });
      K.text(b, 64, 33, s, { "font-size": 26 });
      return { b, dot };
    });
    const loop = K.el(g, "g", {});
    K.el(loop, "path", { d: "M470 234 C 530 234, 530 474, 470 474", fill: "none", stroke: K.color.muted, "stroke-width": 3, "stroke-dasharray": "6 6" });
    const stepTxt = K.text(g, 80, 580, "", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });

    const panel = (y, hgt, title, color) => {
      const p = K.el(g, "g", { transform: `translate(600 ${y})` });
      K.el(p, "rect", { width: 600, height: hgt, rx: 14, fill: K.color.paper, stroke: color, "stroke-width": 2 });
      K.text(p, 28, 42, title, { "font-size": 22, fill: color, "font-weight": 600 });
      return p;
    };
    const before = panel(200, 150, "Before training: random weights", K.color.muted);
    GIBBERISH.forEach((s, i) => K.text(before, 28, 88 + i * 34, s, { "font-size": 22, "font-family": K.color.mono, fill: K.color.muted }));
    const after = panel(380, 190, "After 2,000 steps", K.color.moss);
    SAMPLE.forEach((s, i) => K.text(after, 28, 88 + i * 36, s, { "font-size": 25, "font-style": "italic" }));
    const model = K.text(g, 600, 640, "TinyGPT: 4.8 million parameters, one character at a time", { "font-size": 22, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.03));
      const on = [0.33, 0.39, 0.46, 0.55];
      beats.forEach((b, i) => K.show(b.b, P(K, t, d, on[i]), 12));
      K.show(loop, P(K, t, d, 0.6));
      // once the loop is complete, the beats light up in turn, and the step counter runs
      const cyc = t > 0.62 * d ? Math.floor(t * 2.5) % 4 : -1;
      beats.forEach((b, i) => b.dot.setAttribute("fill", i === cyc ? K.color.amber : K.color.cobaltSoft));
      const n = Math.round(1 + 1999 * K.clamp((t - 0.62 * d) / (0.3 * d)));
      stepTxt.textContent = t > 0.62 * d ? `step ${n.toLocaleString("en-US")}` : "";
      K.show(before, P(K, t, d, 0.68), 16);
      K.show(after, P(K, t, d, 0.84), 16);
      K.show(model, P(K, t, d, 0.9));
    };
  };

  // ---------------------------------------------------------------- s02
  const CHARS = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  S.chars_split = (g, K) => {
    const h = heading(g, K, "One character is one token");
    const tiles = [...CHARS].map((c, i) => {
      const label = c === "\n" ? "\\n" : c === " " ? "␣" : c;
      return K.tile(g, 80 + (i % 13) * 54, 140 + Math.floor(i / 13) * 54, 48, 46, label, { size: 24 });
    });
    const count = K.el(g, "g", { transform: "translate(830 160)" });
    K.text(count, 0, 70, "65", { "font-size": 96, "font-weight": 700, fill: K.color.cobalt });
    K.text(count, 0, 120, "different characters,", { "font-size": 26 });
    K.text(count, 0, 156, "so 65 choices at every", { "font-size": 26 });
    K.text(count, 0, 192, "guess", { "font-size": 26 });

    const W = 1120;
    const split = K.el(g, "g", { transform: "translate(80 450)" });
    const trainBar = K.el(split, "rect", { x: 0, y: 0, width: 0, height: 60, rx: 8, fill: K.color.cobalt });
    const valBar = K.el(split, "rect", { x: W * 0.9 + 4, y: 0, width: W * 0.1 - 4, height: 60, rx: 8, fill: K.color.moss });
    const trainLbl = K.el(split, "g", {});
    K.text(trainLbl, 20, 40, "training set: the first 90%", { "font-size": 26, fill: "#fff", "font-weight": 600 });
    K.text(trainLbl, 0, 100, "1,003,854 characters the model learns from", { "font-size": 24 });
    const valLbl = K.el(split, "g", {});
    K.text(valLbl, W, 100, "validation set: the last 10%", { "font-size": 24, "text-anchor": "end", fill: K.color.moss, "font-weight": 600 });
    K.text(valLbl, W, 136, "111,540 characters, never trained on", { "font-size": 24, "text-anchor": "end", fill: K.color.moss });
    const exam = K.text(g, 80, 660, "The validation set is the honest exam: the questions aren't in the textbook.", { "font-size": 26, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      tiles.forEach((tl, i) => K.show(tl, P(K, t, d, 0.04 + i * 0.003, 0.03)));
      K.show(count, P(K, t, d, 0.25));
      K.show(split, P(K, t, d, 0.44));
      trainBar.setAttribute("width", W * 0.9 * P(K, t, d, 0.46, 0.08));
      K.show(trainLbl, P(K, t, d, 0.52));
      K.show(valBar, P(K, t, d, 0.68));
      K.show(valLbl, P(K, t, d, 0.7));
      K.show(exam, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s03
  // The lesson's window at position 0, and its seeded batch of 4 random windows.
  const WINDOW = "First Cit";
  const BATCH = [["he gives", "e gives "], ["m the yo", " the you"], ["all you ", "ll you h"], ["own soul", "wn soul "]];

  S.windows = (g, K) => {
    const h = heading(g, K, "One window, eight guessing exercises");
    const TW = 84;
    const X0 = 300;
    const shown = (c) => (c === " " ? "␣" : c);
    const text = K.el(g, "g", {});
    K.text(text, 80, 180, "the text", { "font-size": 24, fill: K.color.muted, "font-weight": 600 });
    const textTiles = [...WINDOW].map((c, i) => K.tile(text, X0 + i * TW, 140, TW - 8, 56, shown(c), { size: 28 }));

    const xg = K.el(g, "g", {});
    K.text(xg, 80, 290, "input", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600 });
    const xTiles = [...WINDOW.slice(0, 8)].map((c, i) => K.tile(xg, X0 + i * TW, 250, TW - 8, 56, shown(c), { size: 28, fill: K.color.cobaltSoft }));
    const yg = K.el(g, "g", {});
    K.text(yg, 80, 386, "target", { "font-size": 24, fill: K.color.moss, "font-weight": 600 });
    K.text(yg, 80, 414, "moved by one", { "font-size": 20, fill: K.color.muted });
    const yTiles = [...WINDOW.slice(1)].map((c, i) => K.tile(yg, X0 + (i + 1) * TW, 346, TW - 8, 56, shown(c), { size: 28 }));
    const pair = K.text(g, 640, 470, "", { "font-size": 30, "text-anchor": "middle", "font-weight": 600 });

    const batch = K.el(g, "g", { transform: "translate(80 520)" });
    K.text(batch, 0, 0, "A batch: 4 windows from random places in the text", { "font-size": 24, fill: K.color.muted });
    BATCH.forEach(([x, y], i) => {
      const bx = (i % 2) * 560;
      const by = 46 + Math.floor(i / 2) * 50;
      K.text(batch, bx, by, `"${x}"`, { "font-size": 24, "font-family": K.color.mono, fill: K.color.cobalt });
      K.text(batch, bx + 190, by, "→", { "font-size": 24, fill: K.color.muted });
      K.text(batch, bx + 230, by, `"${y}"`, { "font-size": 24, "font-family": K.color.mono, fill: K.color.moss });
    });
    const rect = (tile) => tile.querySelector("rect");
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(text, P(K, t, d, 0.14));
      textTiles.forEach((tl, i) => K.show(tl, P(K, t, d, 0.14 + i * 0.012, 0.03)));
      K.show(xg, P(K, t, d, 0.26), 12);
      K.show(yg, P(K, t, d, 0.36), 12);
      // walk through the 8 exercises: x[0..k] is the context, y[k] the answer
      const walk = K.clamp((t - 0.5 * d) / (0.3 * d));
      const k = t > 0.5 * d ? Math.min(7, Math.floor(walk * 8)) : -1;
      xTiles.forEach((tl, i) => rect(tl).setAttribute("fill", k >= 0 && i <= k ? K.color.cobalt : K.color.cobaltSoft));
      xTiles.forEach((tl, i) => tl.querySelector("text").setAttribute("fill", k >= 0 && i <= k ? "#fff" : K.color.ink));
      yTiles.forEach((tl, i) => rect(tl).setAttribute("fill", i === k ? K.color.moss : K.color.surface));
      yTiles.forEach((tl, i) => tl.querySelector("text").setAttribute("fill", i === k ? "#fff" : K.color.ink));
      pair.textContent = k >= 0 ? `after "${[...WINDOW.slice(0, k + 1)].map(shown).join("")}" comes "${shown(WINDOW[k + 1])}"   (${k + 1} of 8)` : "";
      K.show(batch, P(K, t, d, 0.84), 14);
    };
  };

  // ---------------------------------------------------------------- s04
  S.loss = (g, K) => {
    const h = heading(g, K, "The causal LM loss: how surprised the model is");
    const { ax, X, Y } = axes(g, K, {
      x: 140, y: 190, w: 480, h: 360, xmax: 1, ymax: 5,
      xlabel: "probability given to the right character", ylabel: "loss",
      xticks: [[0, "0"], [0.5, "0.5"], [1, "1"]], yticks: [[0, "0"], [2, "2"], [4, "4"]],
    });
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const p = 0.0067 + (i / 60) * (1 - 0.0067);
      pts.push([X(p), Y(-Math.log(p))]);
    }
    const curve = line(g, K, pts, { stroke: K.color.cobalt });
    const mark = (p, label, dx, anchor, dy = -16) => {
      const m = K.el(g, "g", {});
      K.el(m, "circle", { cx: X(p), cy: Y(-Math.log(p)), r: 9, fill: K.color.amber });
      K.text(m, X(p) + dx, Y(-Math.log(p)) + dy, label, { "font-size": 22, "text-anchor": anchor, "font-weight": 600 });
      return m;
    };
    const m1 = mark(1, "sure and right: 0", 6, "end", -60);
    const mEven = mark(1 / 65, "even guess, 1 in 65: 4.17", 22, "start");

    const R = 700;
    const formula = K.el(g, "g", {});
    K.text(formula, R, 210, "loss = −ln(probability of", { "font-size": 30, "font-weight": 600 });
    K.text(formula, R, 250, "the right character)", { "font-size": 30, "font-weight": 600 });
    const avg = K.text(g, R, 310, "averaged over every position", { "font-size": 26, fill: K.color.muted });
    const check = K.el(g, "g", { transform: `translate(${R} 380)` });
    K.el(check, "rect", { width: 500, height: 170, rx: 14, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
    K.text(check, 24, 44, "An untrained model starts near", { "font-size": 26 });
    K.text(check, 24, 84, "ln 65 ≈ 4.17", { "font-size": 30, fill: K.color.moss, "font-weight": 700 });
    K.text(check, 24, 126, "(the lesson's small model: about 4.6)", { "font-size": 22, fill: K.color.muted });
    const wrong = K.text(g, R, 610, "Starts at 10, or 50? Something is wrong.", { "font-size": 26, fill: K.color.rose, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(ax, P(K, t, d, 0.08));
      K.show(formula, P(K, t, d, 0.2));
      curve(P(K, t, d, 0.24, 0.12));
      K.show(avg, P(K, t, d, 0.38));
      K.show(m1, P(K, t, d, 0.52));
      K.show(mEven, P(K, t, d, 0.6));
      K.show(check, P(K, t, d, 0.72), 14);
      K.show(wrong, P(K, t, d, 0.88));
    };
  };

  // ---------------------------------------------------------------- s05
  // The lesson's 200 steps at learning rate 3e-3: the loss at every 20th step, and the last.
  const STEP_LOSS = [4.31, 2.86, 2.73, 2.61, 2.56, 2.54, 2.48, 2.5, 2.39, 2.45, 2.4];

  S.step = (g, K) => {
    const h = heading(g, K, "One training step, in code");
    const code = K.el(g, "g", { transform: "translate(80 150)" });
    K.el(code, "rect", { width: 580, height: 400, rx: 14, fill: K.color.paper, stroke: K.color.line });
    const lines = [
      ["optimizer.zero_grad()", "clear the old gradients"],
      ["loss = F.cross_entropy(...)", "guess, and measure the loss"],
      ["loss.backward()", "work out every gradient"],
      ["optimizer.step()", "AdamW nudges every weight"],
    ].map(([c, s], i) => {
      const r = K.el(code, "g", { transform: `translate(0 ${30 + i * 92})` });
      const bg = K.el(r, "rect", { x: 12, y: 0, width: 556, height: 80, rx: 8, fill: K.color.cobaltSoft, opacity: 0 });
      K.text(r, 32, 36, c, { "font-size": 26, "font-family": K.color.mono, fill: K.color.cobalt });
      K.text(r, 32, 68, s, { "font-size": 22, fill: K.color.muted });
      return { r, bg };
    });
    const lr = K.text(g, 80, 610, "The learning rate sets how big each nudge is.", { "font-size": 28, "font-weight": 600 });

    const { ax, X, Y } = axes(g, K, {
      x: 780, y: 190, w: 400, h: 320, xmax: 200, ymax: 5, ylabel: "loss, a small model, learning rate 0.003",
      xticks: [[0, "0"], [100, "100"], [200, "200"]], yticks: [[0, "0"], [2, "2"], [4, "4"]], xlabel: "step",
    });
    const curve = line(g, K, STEP_LOSS.map((v, i) => [X(i * 20), Y(v)]), { stroke: K.color.moss });
    const end = K.text(g, X(200), Y(2.4) - 24, "about 2.4", { "font-size": 26, "text-anchor": "end", fill: K.color.moss, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(code, P(K, t, d, 0.05));
      const on = [0.15, 0.24, 0.35, 0.47];
      const loopAt = 0.58;
      lines.forEach((l, i) => {
        K.show(l.r, P(K, t, d, on[i]));
        const cur = t < loopAt * d ? (i === on.filter((a) => t >= a * d).length - 1) : i === Math.floor(t * 1.5) % 4;
        l.bg.setAttribute("opacity", cur ? 1 : 0);
      });
      K.show(lr, P(K, t, d, 0.6));
      K.show(ax, P(K, t, d, 0.7));
      curve(P(K, t, d, 0.74, 0.16));
      K.show(end, P(K, t, d, 0.9));
    };
  };

  // ---------------------------------------------------------------- s06
  S.schedule = (g, K) => {
    const h = heading(g, K, "Warmup, then cosine decay");
    // two little valleys: small steps creep, big steps jump past the bottom
    const valley = (x0, title, color, steps) => {
      const v = K.el(g, "g", { transform: `translate(${x0} 130)` });
      const pts = [];
      for (let i = 0; i <= 40; i++) {
        const u = i / 40;
        pts.push([30 + u * 460, 140 - 120 * (2 * u - 1) ** 2]);
      }
      K.el(v, "path", { d: pts.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" "), fill: "none", stroke: K.color.line, "stroke-width": 3 });
      const dots = steps.map((u) => K.el(v, "circle", { cx: 30 + u * 460, cy: 140 - 120 * (2 * u - 1) ** 2, r: 9, fill: color }));
      K.text(v, 260, 185, title, { "font-size": 24, "text-anchor": "middle", fill: color, "font-weight": 600 });
      return { v, dots };
    };
    const small = valley(80, "too small: learns slowly", K.color.amber, [0.04, 0.07, 0.1, 0.13, 0.16, 0.19]);
    const big = valley(700, "too big: jumps past the low point", K.color.rose, [0.06, 0.9, 0.12, 0.95, 0.04, 0.97]);

    const MAX = 1e-3;
    const MIN = 1e-4;
    const lrAt = (s) => (s < 100 ? (MAX * (s + 1)) / 100 : MIN + 0.5 * (MAX - MIN) * (1 + Math.cos((Math.PI * (s - 100)) / 1900)));
    const { ax, X, Y } = axes(g, K, {
      x: 200, y: 380, w: 960, h: 220, xmax: 2000, ymax: 0.0011, ylabel: "learning rate",
      xticks: [[0, "0"], [1000, "1,000"], [2000, "2,000 steps"]], yticks: [[0.001, "0.001"], [0.0001, "0.0001"]],
    });
    const warm = [];
    for (let s = 0; s <= 100; s += 5) warm.push([X(s), Y(lrAt(s))]);
    const cos = [];
    for (let s = 100; s <= 2000; s += 25) cos.push([X(s), Y(lrAt(s))]);
    const warmLine = line(g, K, warm, { stroke: K.color.amber, "stroke-width": 6 });
    const cosLine = line(g, K, cos, { stroke: K.color.cobalt, "stroke-width": 5 });
    const warmLbl = K.text(g, X(100) + 18, Y(0.0004), "warmup: 100 steps, small first steps", { "font-size": 24, fill: K.color.amber, "font-weight": 600 });
    const cosLbl = K.text(g, X(1300), Y(lrAt(1300)) - 22, "cosine decay: the last steps settle in", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(small.v, P(K, t, d, 0.08));
      small.dots.forEach((dot, i) => K.show(dot, P(K, t, d, 0.1 + i * 0.015, 0.02)));
      K.show(big.v, P(K, t, d, 0.18));
      big.dots.forEach((dot, i) => K.show(dot, P(K, t, d, 0.2 + i * 0.015, 0.02)));
      K.show(ax, P(K, t, d, 0.34));
      warmLine(P(K, t, d, 0.42, 0.1));
      K.show(warmLbl, P(K, t, d, 0.5));
      cosLine(P(K, t, d, 0.68, 0.18));
      K.show(cosLbl, P(K, t, d, 0.82));
    };
  };

  // ---------------------------------------------------------------- s07
  S.clipping = (g, K) => {
    const h = heading(g, K, "Gradient clipping: shorter, same direction");
    const O = [160, 610];
    const U = 80;
    const pt = (x, y) => [O[0] + x * U, O[1] - y * U];
    const grid = K.el(g, "g", {});
    for (let i = 0; i <= 5; i++) {
      K.el(grid, "line", { x1: pt(i, 0)[0], y1: pt(0, 0)[1], x2: pt(i, 0)[0], y2: pt(0, 5)[1], stroke: K.color.line, "stroke-width": 1 });
      K.el(grid, "line", { x1: pt(0, 0)[0], y1: pt(0, i)[1], x2: pt(5, 0)[0], y2: pt(0, i)[1], stroke: K.color.line, "stroke-width": 1 });
    }
    const circle = K.el(g, "g", {});
    K.el(circle, "path", { d: `M${pt(1, 0)[0]} ${pt(1, 0)[1]} A ${U} ${U} 0 0 0 ${pt(0, 1)[0]} ${pt(0, 1)[1]}`, fill: "none", stroke: K.color.moss, "stroke-width": 3, "stroke-dasharray": "8 6" });
    K.text(circle, pt(1, 0)[0] + 10, pt(1, 0)[1] - 10, "limit 1", { "font-size": 22, fill: K.color.moss });
    const ghost = K.el(g, "g", {});
    K.el(ghost, "line", { x1: O[0], y1: O[1], x2: pt(3, 4)[0], y2: pt(3, 4)[1], stroke: K.color.rose, "stroke-width": 3, "stroke-dasharray": "8 6" });
    K.text(ghost, pt(3, 4)[0] + 18, pt(3, 4)[1] + 8, "[3, 4]: norm 5", { "font-size": 24, "font-family": K.color.mono, fill: K.color.rose });
    const arrow = K.el(g, "line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: K.color.rose, "stroke-width": 7, "stroke-linecap": "round" });
    const tip = K.el(g, "circle", { r: 10, fill: K.color.rose });
    const arrowLbl = K.text(g, pt(3, 4)[0] + 18, pt(3, 4)[1] + 8, "[3, 4]", { "font-size": 26, "font-family": K.color.mono, fill: K.color.rose, "font-weight": 600 });

    const R = 680;
    const norm = K.el(g, "g", {});
    K.text(norm, R, 190, "gradient norm: the length of", { "font-size": 28 });
    K.text(norm, R, 228, "all the gradients together", { "font-size": 28 });
    const lines = [
      ["norm of [3, 4] = √(3² + 4²) = 5", K.color.ink],
      ["5 is over the limit of 1,", K.color.ink],
      ["so multiply everything by 1/5", K.color.ink],
      ["[0.6, 0.8]: norm 1, same direction", K.color.moss],
    ].map(([s, c], i) => K.text(g, R, 320 + i * 56, s, { "font-size": 28, fill: c, "font-weight": i === 3 ? 700 : 400 }));
    const run = K.text(g, R, 610, "train.py clips at a norm of 1.0", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(grid, P(K, t, d, 0.04));
      K.show(norm, P(K, t, d, 0.22));
      const grow = P(K, t, d, 0.56, 0.06);
      const shrink = P(K, t, d, 0.8, 0.08);
      const len = K.lerp(1, 0.2, shrink) * grow;
      const [ex, ey] = pt(3 * len, 4 * len);
      arrow.setAttribute("x2", ex);
      arrow.setAttribute("y2", ey);
      tip.setAttribute("cx", ex);
      tip.setAttribute("cy", ey);
      tip.style.opacity = grow;
      const done = shrink > 0.5;
      arrow.setAttribute("stroke", done ? K.color.moss : K.color.rose);
      tip.setAttribute("fill", done ? K.color.moss : K.color.rose);
      arrowLbl.textContent = done ? "[0.6, 0.8]" : "[3, 4]";
      arrowLbl.setAttribute("x", (done ? pt(0.6, 0.8)[0] : pt(3, 4)[0]) + 18);
      arrowLbl.setAttribute("y", (done ? pt(0.6, 0.8)[1] - 12 : pt(3, 4)[1] + 8));
      arrowLbl.setAttribute("fill", done ? K.color.moss : K.color.rose);
      arrowLbl.style.opacity = grow;
      ghost.style.opacity = done ? 0.6 : 0;
      K.show(circle, P(K, t, d, 0.7));
      [0.62, 0.7, 0.76, 0.86].forEach((a, i) => K.show(lines[i], P(K, t, d, a)));
      K.show(run, P(K, t, d, 0.92));
    };
  };

  // ---------------------------------------------------------------- s08
  S.accumulation = (g, K) => {
    const h = heading(g, K, "Gradient accumulation: big batch, small memory");
    const BW = 30;
    const row = (y, groups, title) => {
      const r = K.el(g, "g", { transform: `translate(80 ${y})` });
      K.text(r, 0, 0, title, { "font-size": 26, "font-weight": 600 });
      const cells = [];
      let x = 0;
      groups.forEach((n, gi) => {
        if (gi) x += 30;
        for (let i = 0; i < n; i++) {
          cells.push({ el: K.el(r, "rect", { x, y: 22, width: BW - 6, height: 52, rx: 5, fill: K.color.cobaltSoft }), gi });
          x += BW;
        }
      });
      K.text(r, 680, 58, "gradients", { "font-size": 24, fill: K.color.muted, "text-anchor": "end" });
      K.el(r, "rect", { x: 700, y: 26, width: 420, height: 44, rx: 8, fill: K.color.paper, stroke: K.color.line });
      const fill = K.el(r, "rect", { x: 700, y: 26, width: 0, height: 44, rx: 8, fill: K.color.amber });
      return { r, cells, fill };
    };
    const A = row(170, [16], "One batch of 16: all 16 in memory at once");
    const B = row(330, [8, 8], "Two micro-batches of 8: half the memory");
    const halves = [0, 1].map((i) => K.text(g, 200 + i * 270, 450, "loss ÷ 2, backward", { "font-size": 22, "text-anchor": "middle", fill: K.color.muted }));
    const adds = K.text(g, 1200, 450, "each backward adds to it", { "font-size": 22, "text-anchor": "end", fill: K.color.muted });
    const same = K.text(g, 640, 520, "Same gradients, then one optimizer step.", { "font-size": 30, "text-anchor": "middle", "font-weight": 600, fill: K.color.moss });
    const run = K.el(g, "g", { transform: "translate(80 580)" });
    K.el(run, "rect", { width: 1120, height: 80, rx: 14, fill: K.color.cobaltSoft });
    K.text(run, 560, 51, "The real run: 2 micro-batches of 16 = 32 windows per step", { "font-size": 28, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(A.r, P(K, t, d, 0.05));
      const a = P(K, t, d, 0.14, 0.08);
      A.fill.setAttribute("width", 420 * a);
      A.cells.forEach((c) => c.el.setAttribute("fill", a > 0 && a < 1 ? K.color.cobalt : K.color.cobaltSoft));
      K.show(B.r, P(K, t, d, 0.3));
      const b1 = P(K, t, d, 0.4, 0.08);
      const b2 = P(K, t, d, 0.52, 0.08);
      B.fill.setAttribute("width", 210 * b1 + 210 * b2);
      B.cells.forEach((c) => {
        const p = c.gi === 0 ? b1 : b2;
        c.el.setAttribute("fill", p > 0 && p < 1 ? K.color.cobalt : K.color.cobaltSoft);
      });
      K.show(adds, P(K, t, d, 0.5));
      halves.forEach((hl, i) => K.show(hl, P(K, t, d, 0.62 + i * 0.03)));
      K.show(same, P(K, t, d, 0.72));
      K.show(run, P(K, t, d, 0.82), 14);
    };
  };

  // ---------------------------------------------------------------- s09
  // The lesson's overfitting run: 5,000 training characters, both losses every 100 steps.
  const TRAIN = [4.33, 2.33, 2.13, 1.95, 1.7, 1.42, 1.16, 0.93, 0.73];
  const VAL = [4.34, 2.85, 2.91, 2.98, 3.07, 3.28, 3.62, 4.07, 4.56];

  S.curves = (g, K) => {
    const h = heading(g, K, "Loss curves, and overfitting");
    const { ax, X, Y } = axes(g, K, {
      x: 140, y: 190, w: 560, h: 360, xmax: 800, ymax: 5, xlabel: "step", ylabel: "loss, training on only 5,000 characters",
      xticks: [[0, "0"], [400, "400"], [800, "800"]], yticks: [[0, "0"], [2, "2"], [4, "4"]],
    });
    const tr = line(g, K, TRAIN.map((v, i) => [X(i * 100), Y(v)]), { stroke: K.color.cobalt });
    const va = line(g, K, VAL.map((v, i) => [X(i * 100), Y(v)]), { stroke: K.color.amber });
    const trLbl = K.text(g, X(800) - 10, Y(0.73) + 40, "training", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600, "text-anchor": "end" });
    const vaLbl = K.text(g, X(800) - 10, Y(4.56) - 16, "validation", { "font-size": 24, fill: K.color.amber, "font-weight": 600, "text-anchor": "end" });
    const split = K.el(g, "g", {});
    K.el(split, "line", { x1: X(550), y1: Y(3.4) + 8, x2: X(550), y2: Y(1.3) - 8, stroke: K.color.rose, "stroke-width": 3, "stroke-dasharray": "6 5" });
    K.text(split, X(530), Y(2.35) + 8, "the lines split:", { "font-size": 22, fill: K.color.rose, "font-weight": 600, "text-anchor": "end" });
    K.text(split, X(530), Y(2.35) + 36, "overfitting", { "font-size": 22, fill: K.color.rose, "font-weight": 600, "text-anchor": "end" });

    const card = K.el(g, "g", { transform: "translate(780 190)" });
    K.el(card, "rect", { width: 420, height: 360, rx: 14, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
    K.text(card, 28, 50, "The real run, 2,000 steps", { "font-size": 26, "font-weight": 600, fill: K.color.moss });
    [
      ["training loss", "about 1.27"],
      ["validation loss", "about 1.53"],
      ["target", "below 1.70"],
    ].forEach(([a, b], i) => {
      K.text(card, 28, 120 + i * 60, a, { "font-size": 24, fill: K.color.muted });
      K.text(card, 392, 120 + i * 60, b, { "font-size": 26, "text-anchor": "end", "font-weight": 600 });
    });
    K.text(card, 28, 300, "Validation still falling", { "font-size": 24, fill: K.color.moss });
    K.text(card, 28, 332, "at the end: no overfitting", { "font-size": 24, fill: K.color.moss });
    const memo = K.text(g, 140, 660, "Memorising the page doesn't carry over to new text.", { "font-size": 28, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(ax, P(K, t, d, 0.04));
      const p = P(K, t, d, 0.28, 0.3);
      tr(p);
      va(p);
      K.show(trLbl, P(K, t, d, 0.36));
      K.show(vaLbl, P(K, t, d, 0.5));
      K.show(memo, P(K, t, d, 0.6));
      K.show(split, P(K, t, d, 0.68));
      K.show(card, P(K, t, d, 0.8), 16);
    };
  };

  // ---------------------------------------------------------------- s10
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      "1. get_batch: shifted windows",
      "2. causal_lm_loss",
      "3. lr_at: warmup, then cosine",
      "4. clip_grad_norm",
      "5. train_step: one full step",
      "6. estimate_loss: measure honestly",
    ];
    const rows = items.map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(80 ${170 + i * 70})` });
      const box = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 24, "font-family": K.color.mono });
      return { r, box, tick };
    });
    const R = 740;
    const first = K.el(g, "g", {});
    K.text(first, R, 196, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    K.text(first, R, 236, "It runs every piece with real code.", { "font-size": 24, fill: K.color.muted });
    const run = K.el(g, "g", { transform: `translate(${R} 300)` });
    K.el(run, "rect", { width: 460, height: 250, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(run, 24, 46, "Then train.py:", { "font-size": 26, "font-weight": 600 });
    K.text(run, 24, 84, "2,000 steps, about 2 minutes", { "font-size": 24 });
    const goals = [
      K.text(run, 24, 150, "validation loss below 1.70", { "font-size": 24, fill: K.color.moss, "font-weight": 600 }),
      K.text(run, 24, 196, "80% real words in the sample", { "font-size": 24, fill: K.color.moss, "font-weight": 600 }),
    ];
    const qs = K.text(g, 80, 640, "Plus six understanding checks in answers.toml.", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(first, P(K, t, d, 0.06));
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, 0.2 + i * 0.06));
        const done = P(K, t, d, 0.24 + i * 0.06, 0.03);
        row.tick.style.opacity = done;
        row.box.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(run, P(K, t, d, 0.6), 14);
      goals.forEach((gl, i) => K.show(gl, P(K, t, d, [0.74, 0.8][i])));
      K.show(qs, P(K, t, d, 0.86));
    };
  };
})();
