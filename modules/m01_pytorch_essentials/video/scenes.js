// M1's video: one drawing per scene in script.toml. Every number is one the lesson prints.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // a small deterministic random source, so the dots look the same every frame and every run
  function rng(seed) {
    let s = seed >>> 0;
    const u = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return (s + 0.5) / 4294967296;
    };
    const n = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
    return { u, n };
  }

  function card(g, K, x, y, w, h, stroke) {
    const c = K.el(g, "g", { transform: `translate(${x} ${y})` });
    K.el(c, "rect", { width: w, height: h, rx: 14, fill: K.color.paper, stroke: stroke || K.color.line, "stroke-width": 2 });
    return c;
  }

  // ---------------------------------------------------------------- s01
  S.model_loop = (g, K) => {
    const h = heading(g, K, "A model is a function with adjustable numbers");

    const formula = card(g, K, 80, 140, 540, 110);
    const ftext = K.text(formula, 270, 70, "", { "font-size": 40, "text-anchor": "middle", "font-family": K.color.mono });
    ["price = ", "w", " × size + ", "b"].forEach((s, i) => {
      K.el(ftext, "tspan", i % 2 ? { fill: K.color.cobalt, "font-weight": 700 } : {}, s);
    });

    const params = K.el(g, "g", { transform: "translate(80 280)" });
    K.text(params, 0, 26, "parameters:", { "font-size": 24, fill: K.color.muted });
    K.text(params, 150, 26, "w, a weight (scales the size)", { "font-size": 24, fill: K.color.cobalt });
    K.text(params, 150, 62, "b, a bias (added at the end)", { "font-size": 24, fill: K.color.cobalt });

    // a tiny picture: house sizes and prices, and the model's line getting better
    const plot = K.el(g, "g", { transform: "translate(80 380)" });
    const PW = 540;
    const PH = 240;
    K.el(plot, "rect", { width: PW, height: PH, rx: 10, fill: K.color.paper, stroke: K.color.line });
    K.text(plot, 16, PH - 14, "size", { "font-size": 22, fill: K.color.muted });
    K.text(plot, 16, 32, "price", { "font-size": 22, fill: K.color.muted });
    const r = rng(7);
    for (let i = 0; i < 16; i++) {
      const sx = 0.1 + 0.8 * (i / 15);
      const sy = 0.15 + 0.7 * sx + 0.06 * r.n();
      K.el(plot, "circle", { cx: sx * PW, cy: PH - sy * PH, r: 7, fill: K.color.muted, opacity: 0.7 });
    }
    const line = K.el(plot, "line", { stroke: K.color.cobalt, "stroke-width": 4, "stroke-linecap": "round" });

    // the four-step loop
    const cx = 950;
    const cy = 420;
    const R = 165;
    const ring = K.el(g, "g", {});
    K.el(ring, "circle", { cx, cy, r: R, fill: "none", stroke: K.color.line, "stroke-width": 4 });
    const dot = K.el(ring, "circle", { r: 12, fill: K.color.amber });
    const steps = [
      ["1. Guess", "run the model"],
      ["2. Measure", "the loss"],
      ["3. Direction", "the gradients"],
      ["4. Nudge", "the parameters"],
    ].map(([a, b], i) => {
      const ang = -Math.PI / 2 + (i * Math.PI) / 2;
      const x = cx + R * Math.cos(ang);
      const y = cy + R * Math.sin(ang);
      const sg = K.el(g, "g", {});
      K.el(sg, "rect", { x: x - 95, y: y - 38, width: 190, height: 76, rx: 18, fill: K.color.cobaltSoft, stroke: K.color.cobalt });
      K.text(sg, x, y - 6, a, { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 700 });
      K.text(sg, x, y + 24, b, { "font-size": 20, "text-anchor": "middle", fill: K.color.ink });
      return sg;
    });
    const loopTitle = K.text(g, cx, cy + 8, "Training", { "font-size": 30, "text-anchor": "middle", "font-weight": 600 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(formula, P(K, t, d, 0.1), 16);
      K.show(plot, P(K, t, d, 0.2), 16);
      K.show(params, P(K, t, d, 0.42));
      ftext.querySelectorAll("tspan").forEach((s, i) => {
        if (i % 2) s.setAttribute("fill", P(K, t, d, 0.42) > 0.5 ? K.color.cobalt : K.color.ink);
      });
      // a bad line first, then it settles onto the dots as training runs
      const fit = P(K, t, d, 0.78, 0.2);
      const slope = K.lerp(-0.2, 0.7, fit);
      const icept = K.lerp(0.7, 0.15, fit);
      line.setAttribute("x1", 0.05 * PW);
      line.setAttribute("x2", 0.95 * PW);
      line.setAttribute("y1", PH - (icept + slope * 0.05) * PH);
      line.setAttribute("y2", PH - (icept + slope * 0.95) * PH);
      line.style.opacity = P(K, t, d, 0.25);
      K.show(ring, P(K, t, d, 0.62));
      K.show(loopTitle, P(K, t, d, 0.62));
      steps.forEach((s, i) => K.show(s, P(K, t, d, [0.72, 0.78, 0.84, 0.9][i])));
      const a = -Math.PI / 2 + Math.max(0, t - 0.72 * d) * 1.8;
      dot.setAttribute("cx", cx + R * Math.cos(a));
      dot.setAttribute("cy", cy + R * Math.sin(a));
      dot.style.opacity = P(K, t, d, 0.72);
    };
  };

  // ---------------------------------------------------------------- s02
  S.tensors = (g, K) => {
    const h = heading(g, K, "Tensors: grids of numbers");
    const kinds = K.el(g, "g", {});
    K.text(kinds, 80, 160, "one number", { "font-size": 22, fill: K.color.muted });
    K.tile(kinds, 80, 176, 90, 50, "7.0", { size: 24 });
    K.text(kinds, 240, 160, "a list", { "font-size": 22, fill: K.color.muted });
    [1, 2, 3].forEach((v, i) => K.tile(kinds, 240 + i * 66, 176, 60, 50, `${v}.`, { size: 24 }));

    const C = 110;
    const GX = 130;
    const GY = 330;
    const grid = K.el(g, "g", {});
    K.text(grid, GX, GY - 20, "a table", { "font-size": 22, fill: K.color.muted });
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        K.tile(grid, GX + c * C, GY + r * 90, C - 10, 80, `${r * 3 + c + 1}.`, { size: 34, fill: K.color.cobaltSoft, ink: K.color.cobalt });
      }
    }
    const rows = K.el(g, "g", {});
    K.el(rows, "path", { d: `M${GX - 20} ${GY + 4} h-12 v162 h12`, fill: "none", stroke: K.color.amber, "stroke-width": 3 });
    K.text(rows, GX - 44, GY + 92, "2 rows", { "font-size": 24, fill: K.color.amber, "text-anchor": "end", "font-weight": 600 });
    const cols = K.el(g, "g", {});
    K.el(cols, "path", { d: `M${GX + 4} ${GY + 190} v12 h312 v-12`, fill: "none", stroke: K.color.amber, "stroke-width": 3 });
    K.text(cols, GX + 160, GY + 236, "3 columns", { "font-size": 24, fill: K.color.amber, "text-anchor": "middle", "font-weight": 600 });

    const props = [
      { y: 170, k: "shape", v: "(2, 3)", note: "2 rows, 3 columns" },
      { y: 320, k: "dtype", v: "float32", note: "4 bytes per number: 6 × 4 = 24 bytes" },
      { y: 470, k: "device", v: "cpu or mps", note: "the chip that holds it (mps: your Mac's GPU)" },
    ].map((p) => {
      const c = card(g, K, 600, p.y, 600, 120);
      K.text(c, 28, 50, p.k, { "font-size": 24, fill: K.color.muted });
      K.text(c, 160, 52, p.v, { "font-size": 34, "font-family": K.color.mono, "font-weight": 600, fill: K.color.cobalt });
      K.text(c, 28, 94, p.note, { "font-size": 24 });
      return c;
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(kinds, P(K, t, d, 0.04));
      K.show(grid, P(K, t, d, 0.12), 16);
      K.show(rows, P(K, t, d, 0.25));
      K.show(cols, P(K, t, d, 0.32));
      [0.38, 0.5, 0.84].forEach((a, i) => K.show(props[i], P(K, t, d, a), 16));
    };
  };

  // ---------------------------------------------------------------- s03
  S.matmul = (g, K) => {
    const h = heading(g, K, "Matrix multiply: row times column, added up");
    const C = 70;
    const A = [[1, 2, 3], [4, 5, 6]];
    const B = [[1, 0], [0, 1], [1, 1]];
    const Rv = [[4, 5], [10, 11]];
    const grid = (x, y, M, fill) => {
      const gg = K.el(g, "g", {});
      const cells = M.map((row, r) => row.map((v, c) => {
        const cell = K.tile(gg, x + c * C, y + r * C, C - 6, C - 6, v === null ? "" : String(v), { size: 30, fill });
        return cell;
      }));
      return { gg, cells };
    };
    const ga = grid(80, 200, A);
    const gb = grid(390, 165, B);
    const gr = grid(650, 200, [["", ""], ["", ""]]);
    const ops = [K.text(g, 330, 285, "@", { "font-size": 44, "text-anchor": "middle", fill: K.color.muted }),
      K.text(g, 600, 285, "=", { "font-size": 44, "text-anchor": "middle", fill: K.color.muted })];
    const labels = [
      K.text(g, 80 + 1.5 * C - 3, 360, "A", { "font-size": 26, "text-anchor": "middle", fill: K.color.muted }),
      K.text(g, 390 + C - 3, 395, "B", { "font-size": 26, "text-anchor": "middle", fill: K.color.muted }),
      K.text(g, 650 + C - 3, 360, "A @ B", { "font-size": 26, "text-anchor": "middle", fill: K.color.muted }),
    ];
    const resultText = gr.cells.map((row) => row.map((cell) => cell.querySelector("text")));
    const resultRect = gr.cells.map((row) => row.map((cell) => cell.querySelector("rect")));
    const sums = [];
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const terms = [0, 1, 2].map((k) => `${A[r][k]}×${B[k][c]}`).join(" + ");
        sums.push({ r, c, node: K.text(g, 840, 200 + sums.length * 52, `${terms} = ${Rv[r][c]}`, { "font-size": 26, "font-family": K.color.mono }) });
      }
    }
    const rule = K.el(g, "g", { transform: "translate(80 470)" });
    const rt = K.text(rule, 0, 50, "", { "font-size": 44, "font-family": K.color.mono, "font-weight": 600 });
    const parts = ["(2, ", "3", ") @ (", "3", ", 2)  →  (2, 2)"].map((s, i) => K.el(rt, "tspan", i % 2 ? { fill: K.color.amber } : {}, s));
    const ruleNote = K.text(rule, 0, 120, "The middle numbers must match, and they disappear.", { "font-size": 28, fill: K.color.muted });

    // which result cell is being worked out, and when
    const when = [0.4, 0.62, 0.67, 0.72];
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(ga.gg, P(K, t, d, 0.03));
      K.show(gb.gg, P(K, t, d, 0.06));
      K.show(gr.gg, P(K, t, d, 0.12));
      ops.forEach((o) => K.show(o, P(K, t, d, 0.06)));
      labels.forEach((l) => K.show(l, P(K, t, d, 0.08)));
      let active = -1;
      when.forEach((w, i) => { if (t >= w * d) active = i; });
      if (t > 0.78 * d) active = -1;
      ga.cells.forEach((row, r) => row.forEach((cell, c) => {
        const on = active >= 0 && sums[active].r === r;
        cell.querySelector("rect").setAttribute("fill", on ? K.color.cobaltSoft : K.color.surface);
        cell.querySelector("rect").setAttribute("stroke", on ? K.color.cobalt : K.color.line);
        void c;
      }));
      gb.cells.forEach((row) => row.forEach((cell, c) => {
        const on = active >= 0 && sums[active].c === c;
        cell.querySelector("rect").setAttribute("fill", on ? K.color.cobaltSoft : K.color.surface);
        cell.querySelector("rect").setAttribute("stroke", on ? K.color.cobalt : K.color.line);
      }));
      sums.forEach((s, i) => {
        const p = P(K, t, d, when[i] + (i === 0 ? 0.08 : 0.02), 0.04);
        s.node.style.opacity = p;
        resultText[s.r][s.c].textContent = p > 0.5 ? String(Rv[s.r][s.c]) : "";
        resultText[s.r][s.c].setAttribute("fill", K.color.moss);
        resultText[s.r][s.c].setAttribute("font-weight", 700);
        resultRect[s.r][s.c].setAttribute("stroke", active === i ? K.color.moss : K.color.line);
        resultRect[s.r][s.c].setAttribute("stroke-width", active === i ? 3 : 1);
      });
      K.show(rule, P(K, t, d, 0.78), 16);
      K.show(ruleNote, P(K, t, d, 0.92));
      const fade = P(K, t, d, 0.92);
      [parts[1], parts[3]].forEach((p) => p.setAttribute("opacity", 1 - 0.65 * fade));
    };
  };

  // ---------------------------------------------------------------- the curve loss = (w - 5)^2 (s04, s05)
  function lossCurve(g, K, x, y, w, h) {
    const pg = K.el(g, "g", { transform: `translate(${x} ${y})` });
    const X = (v) => (v / 10) * w;
    const TOP = 12; // the curve is cut off above this loss, so the bottom of the bowl is big
    const Y = (l) => h - (l / TOP) * h;
    K.el(pg, "line", { x1: 0, y1: h, x2: w, y2: h, stroke: K.color.muted, "stroke-width": 2 });
    K.el(pg, "line", { x1: 0, y1: 0, x2: 0, y2: h, stroke: K.color.muted, "stroke-width": 2 });
    [0, 3, 5, 10].forEach((v) => K.text(pg, X(v), h + 30, String(v), { "font-size": 22, "text-anchor": "middle", fill: K.color.muted }));
    K.text(pg, w + 14, h + 8, "w", { "font-size": 24, fill: K.color.muted });
    K.text(pg, 12, 4, "loss", { "font-size": 22, fill: K.color.muted });
    let d = "";
    const reach = Math.sqrt(TOP);
    for (let i = 0; i <= 100; i++) {
      const v = 5 - reach + (i / 100) * 2 * reach;
      d += `${i ? "L" : "M"}${X(v).toFixed(1)} ${Y((v - 5) ** 2).toFixed(1)}`;
    }
    const curve = K.el(pg, "path", { d, fill: "none", stroke: K.color.cobalt, "stroke-width": 4 });
    return { pg, X, Y, curve };
  }

  // ---------------------------------------------------------------- s04
  S.gradient = (g, K) => {
    const h = heading(g, K, "Gradients: which way is downhill?");
    const c = lossCurve(g, K, 110, 160, 520, 300);
    const formula = K.text(c.pg, 300, 40, "loss = (w − 5)²", { "font-size": 28, "font-family": K.color.mono, "text-anchor": "middle" });
    const min = K.el(c.pg, "g", {});
    K.el(min, "circle", { cx: c.X(5), cy: c.Y(0), r: 9, fill: K.color.moss });
    K.text(min, c.X(5), c.Y(0) + 62, "smallest at w = 5", { "font-size": 22, "text-anchor": "middle", fill: K.color.moss, "font-weight": 600 });
    const pt = K.el(c.pg, "g", {});
    K.el(pt, "line", { x1: c.X(2), y1: c.Y(8), x2: c.X(4), y2: c.Y(0), stroke: K.color.rose, "stroke-width": 4 });
    K.el(pt, "circle", { cx: c.X(3), cy: c.Y(4), r: 10, fill: K.color.rose });
    const step = K.el(c.pg, "g", {});
    K.el(step, "path", { d: `M${c.X(3.2)} ${c.Y(7)} H${c.X(4.4)} l-12 -10 m12 10 l-12 10`, fill: "none", stroke: K.color.moss, "stroke-width": 4, "stroke-linecap": "round" });
    K.text(step, c.X(3.8), c.Y(7) - 16, "downhill", { "font-size": 22, "text-anchor": "middle", fill: K.color.moss, "font-weight": 600 });

    const side = K.el(g, "g", { transform: "translate(720 170)" });
    const lines = [
      { y: 0, s: "The gradient says which way is", c: K.color.ink },
      { y: 40, s: "uphill, and how steeply.", c: K.color.ink },
      { y: 100, s: "Step the other way: downhill.", c: K.color.moss },
      { y: 190, s: "At w = 3:  gradient = −4", c: K.color.rose, mono: true },
      { y: 240, s: "negative, so a bigger w", c: K.color.ink },
      { y: 280, s: "lowers the loss.", c: K.color.ink },
    ].map((l) => K.text(side, 0, l.y, l.s, { "font-size": l.mono ? 28 : 28, fill: l.c, "font-family": l.mono ? K.color.mono : K.color.font, "font-weight": l.mono ? 600 : 400 }));

    const code = card(g, K, 80, 560, 1120, 130, K.color.cobalt);
    K.text(code, 28, 40, "autograd does it for you", { "font-size": 22, fill: K.color.muted });
    K.text(code, 28, 80, "w = torch.tensor(3.0, requires_grad=True)", { "font-size": 24, "font-family": K.color.mono });
    K.text(code, 28, 112, "loss = (w - 5) ** 2;  loss.backward()", { "font-size": 24, "font-family": K.color.mono });
    const grad = K.text(code, 1092, 98, "w.grad  →  −4", { "font-size": 32, "font-family": K.color.mono, "text-anchor": "end", fill: K.color.rose, "font-weight": 700 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(c.pg, P(K, t, d, 0.08));
      K.show(lines[0], P(K, t, d, 0.12));
      K.show(lines[1], P(K, t, d, 0.12));
      K.show(lines[2], P(K, t, d, 0.33));
      K.show(formula, P(K, t, d, 0.42));
      K.show(min, P(K, t, d, 0.54));
      K.show(pt, P(K, t, d, 0.62));
      K.show(lines[3], P(K, t, d, 0.64));
      K.show(step, P(K, t, d, 0.72));
      K.show(lines[4], P(K, t, d, 0.72));
      K.show(lines[5], P(K, t, d, 0.72));
      K.show(code, P(K, t, d, 0.84), 16);
      K.show(grad, P(K, t, d, 0.92));
    };
  };

  // ---------------------------------------------------------------- s05
  S.descent = (g, K) => {
    const h = heading(g, K, "Gradient descent: nudge, again and again");
    const c = lossCurve(g, K, 110, 150, 500, 240);
    // w after k steps with learning rate 0.1: the gap to 5 shrinks by 0.8 each step
    const W = (k) => 5 - 2 * Math.pow(0.8, k);
    const dots = [];
    for (let k = 0; k <= 25; k++) dots.push(K.el(c.pg, "circle", { cx: c.X(W(k)), cy: c.Y((W(k) - 5) ** 2), r: 7, fill: K.color.amber }));

    const rule = card(g, K, 680, 150, 520, 130, K.color.cobalt);
    K.text(rule, 260, 56, "w ← w − lr × gradient", { "font-size": 32, "font-family": K.color.mono, "text-anchor": "middle", "font-weight": 600 });
    const lrNote = K.text(rule, 260, 100, "lr, the learning rate: the step size", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });
    const counter = K.text(g, 700, 340, "", { "font-size": 30, "font-family": K.color.mono });
    const best = K.text(g, 700, 386, "lr = 0.1 · the best value is 5", { "font-size": 24, fill: K.color.muted });

    const add = K.el(g, "g", { transform: "translate(80 520)" });
    K.text(add, 0, 0, "Gradients add up:", { "font-size": 28, "font-weight": 600 });
    const once = K.text(add, 0, 52, "backward() once:  w.grad = −4", { "font-size": 26, "font-family": K.color.mono });
    const twice = K.text(add, 0, 100, "twice, no reset:  w.grad = −8", { "font-size": 26, "font-family": K.color.mono, fill: K.color.rose, "font-weight": 600 });
    const fix = card(g, K, 700, 520, 500, 110, K.color.moss);
    K.text(fix, 250, 46, "So clear them every step:", { "font-size": 26, "text-anchor": "middle" });
    K.text(fix, 250, 86, "w.grad = None", { "font-size": 28, "text-anchor": "middle", "font-family": K.color.mono, fill: K.color.moss, "font-weight": 700 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(rule, P(K, t, d, 0.03), 16);
      K.show(lrNote, P(K, t, d, 0.15));
      K.show(c.pg, P(K, t, d, 0.2));
      const k = Math.round(25 * P(K, t, d, 0.33, 0.22));
      dots.forEach((dot, i) => {
        dot.style.opacity = i <= k ? (i === k ? 1 : 0.45) : 0;
        dot.setAttribute("r", i === k ? 10 : 6);
      });
      counter.textContent = `step ${String(k).padStart(2, " ")}:  w = ${W(k).toFixed(4)}`;
      K.show(counter, P(K, t, d, 0.33));
      K.show(best, P(K, t, d, 0.36));
      K.show(add, P(K, t, d, 0.62));
      K.show(once, P(K, t, d, 0.66));
      K.show(twice, P(K, t, d, 0.76));
      K.show(fix, P(K, t, d, 0.86), 16);
    };
  };

  // ---------------------------------------------------------------- s06
  S.five_beats = (g, K) => {
    const h = heading(g, K, "A real model, and the five beats");
    const plot = K.el(g, "g", { transform: "translate(80 140)" });
    const PW = 500;
    const PH = 340;
    K.el(plot, "rect", { width: PW, height: PH, rx: 10, fill: K.color.paper, stroke: K.color.line });
    // 100 noisy points around y = 2x + 1, x from 0 to 4 (a picture; the lesson makes its own)
    const X = (x) => 20 + (x / 4) * (PW - 40);
    const Y = (y) => PH - 20 - (y / 10) * (PH - 40);
    const r = rng(3);
    for (let i = 0; i < 100; i++) {
      const x = 4 * r.u();
      const y = 2 * x + 1 + 0.2 * r.n();
      K.el(plot, "circle", { cx: X(x), cy: Y(y), r: 4, fill: K.color.muted, opacity: 0.6 });
    }
    const line = K.el(plot, "line", { x1: X(0), x2: X(4), stroke: K.color.cobalt, "stroke-width": 4 });
    const learned = K.el(g, "g", {});
    K.text(learned, 80, 520, "learned w = 1.966, b = 1.066", { "font-size": 26, "font-family": K.color.mono, fill: K.color.cobalt, "font-weight": 600 });
    K.text(learned, 520, 520, "(true: 2, 1)", { "font-size": 24, fill: K.color.muted });

    const layer = card(g, K, 660, 140, 540, 150);
    K.text(layer, 28, 52, "nn.Linear", { "font-size": 32, "font-family": K.color.mono, fill: K.color.cobalt, "font-weight": 600 });
    K.text(layer, 28, 94, "a layer: computes x @ w + b", { "font-size": 24 });
    K.text(layer, 28, 128, "and keeps its own w and b", { "font-size": 24 });
    const opt = card(g, K, 660, 320, 540, 150);
    K.text(opt, 28, 52, "torch.optim.SGD", { "font-size": 32, "font-family": K.color.mono, fill: K.color.cobalt, "font-weight": 600 });
    K.text(opt, 28, 94, "an optimizer: nudges every", { "font-size": 24 });
    K.text(opt, 28, 128, "parameter at once", { "font-size": 24 });

    const beats = ["forward", "loss", "zero_grad()", "backward()", "step()"].map((s, i) => {
      const bx = 80 + i * 228;
      const bg = K.el(g, "g", {});
      K.el(bg, "rect", { x: bx, y: 580, width: 208, height: 70, rx: 35, fill: K.color.cobaltSoft, stroke: K.color.cobalt });
      K.text(bg, bx + 104, 625, `${i + 1}. ${s}`, { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600, "font-family": K.color.mono });
      return bg;
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(plot, P(K, t, d, 0.04));
      const fit = P(K, t, d, 0.12, 0.2);
      const w = K.lerp(0, 1.966, fit);
      const b = K.lerp(0, 1.066, fit);
      line.setAttribute("y1", Y(b));
      line.setAttribute("y2", Y(b + 4 * w));
      K.show(learned, P(K, t, d, 0.32));
      K.show(layer, P(K, t, d, 0.4), 16);
      K.show(opt, P(K, t, d, 0.52), 16);
      beats.forEach((bt, i) => K.show(bt, P(K, t, d, 0.8 + i * 0.03, 0.04), 12));
    };
  };

  // ---------------------------------------------------------------- s07
  S.classify = (g, K) => {
    const h = heading(g, K, "Picking a class: logits, softmax, argmax");
    const logits = [2.0, 1.0, -1.0];
    const probs = [0.7054, 0.2595, 0.0351];
    const head = K.el(g, "g", {});
    K.text(head, 80, 170, "class", { "font-size": 22, fill: K.color.muted });
    K.text(head, 200, 170, "logit", { "font-size": 22, fill: K.color.muted });
    const headP = K.text(g, 420, 170, "softmax: probability", { "font-size": 22, fill: K.color.muted });
    const BAR = 560;
    const rows = logits.map((v, i) => {
      const y = 190 + i * 90;
      const rg = K.el(g, "g", {});
      K.text(rg, 110, y + 46, String(i), { "font-size": 32, "text-anchor": "middle", "font-weight": 600 });
      const tl = K.tile(rg, 190, y, 130, 66, v.toFixed(1), { size: 30 });
      const arrow = K.text(g, 365, y + 46, "→", { "font-size": 36, "text-anchor": "middle", fill: K.color.muted });
      const bar = K.el(g, "rect", { x: 420, y: y + 8, height: 50, rx: 8, fill: i === 0 ? K.color.moss : K.color.cobalt });
      const val = K.text(g, 0, y + 45, probs[i].toFixed(2), { "font-size": 30, "font-weight": 600 });
      return { rg, tl, arrow, bar, val };
    });
    const sum = K.text(g, 420, 480, "they add up to 1", { "font-size": 24, fill: K.color.muted });
    const arg = K.el(g, "g", {});
    K.el(arg, "rect", { x: 1010, y: 194, width: 190, height: 58, rx: 29, fill: K.color.moss });
    K.text(arg, 1105, 232, "argmax: 0", { "font-size": 26, "text-anchor": "middle", fill: K.color.paper, "font-weight": 700 });

    const ce = card(g, K, 80, 530, 1120, 140);
    K.text(ce, 28, 50, "cross-entropy: small when the right class gets a high probability", { "font-size": 26 });
    K.text(ce, 28, 104, "right answer: class 0", { "font-size": 26, fill: K.color.muted });
    const ceVal = K.text(ce, 1092, 106, "−log(0.7054) = 0.3490", { "font-size": 34, "font-family": K.color.mono, "text-anchor": "end", "font-weight": 700, fill: K.color.moss });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(head, P(K, t, d, 0.04));
      rows.forEach((r, i) => K.show(r.rg, P(K, t, d, 0.08 + i * 0.04)));
      K.show(headP, P(K, t, d, 0.24));
      rows.forEach((r, i) => {
        K.show(r.arrow, P(K, t, d, 0.26));
        const p = P(K, t, d, 0.42 + i * 0.06, 0.1);
        const w = probs[i] * BAR * p;
        r.bar.setAttribute("width", Math.max(0.01, w));
        r.bar.style.opacity = p > 0 ? 1 : 0;
        r.val.setAttribute("x", 420 + w + 16);
        r.val.style.opacity = p;
        const picked = i === 0 && P(K, t, d, 0.7) > 0.5;
        r.tl.querySelector("rect").setAttribute("stroke", picked ? K.color.moss : K.color.line);
        r.tl.querySelector("rect").setAttribute("stroke-width", picked ? 3 : 1);
      });
      K.show(sum, P(K, t, d, 0.62));
      K.show(arg, P(K, t, d, 0.7), 12);
      K.show(ce, P(K, t, d, 0.8), 16);
      K.show(ceVal, P(K, t, d, 0.92));
    };
  };

  // ---------------------------------------------------------------- s08
  S.moons = (g, K) => {
    const h = heading(g, K, "Two moons: no straight line will do");
    const plot = K.el(g, "g", { transform: "translate(80 140)" });
    const PW = 560;
    const PH = 440;
    K.el(plot, "rect", { width: PW, height: PH, rx: 10, fill: K.color.paper, stroke: K.color.line });
    const X = (x) => ((x + 1.6) / 4.2) * PW;
    const Y = (y) => PH - ((y + 1.1) / 2.7) * PH;
    // the same shape as data.make_moons: two half circles, plus noise
    const r = rng(11);
    const dots = [];
    for (let i = 0; i < 300; i++) {
      const upper = i % 2 === 0;
      const th = Math.PI * r.u();
      const x = (upper ? Math.cos(th) : 1 - Math.cos(th)) + 0.2 * r.n();
      const y = (upper ? Math.sin(th) : 0.5 - Math.sin(th)) + 0.2 * r.n();
      dots.push(K.el(plot, "circle", { cx: X(x), cy: Y(y), r: 5, fill: upper ? K.color.cobalt : K.color.amber, opacity: 0.8 }));
    }
    const straight = K.el(plot, "g", {});
    K.el(straight, "line", { x1: X(-1.5), y1: Y(1.0), x2: X(2.5), y2: Y(-0.5), stroke: K.color.rose, "stroke-width": 5, "stroke-dasharray": "14 8" });
    const cross = K.text(plot, X(2.25), Y(-0.75), "✗", { "font-size": 40, fill: K.color.rose, "text-anchor": "middle", "font-weight": 700 });
    let path = "";
    for (let i = 0; i <= 60; i++) {
      const x = -1.3 + (i / 60) * 3.6;
      path += `${i ? "L" : "M"}${X(x).toFixed(1)} ${Y(0.25 + 0.5 * Math.cos(Math.PI * x)).toFixed(1)}`;
    }
    const curved = K.el(plot, "path", { d: path, fill: "none", stroke: K.color.moss, "stroke-width": 6, "stroke-linecap": "round" });

    const net = K.el(g, "g", { transform: "translate(700 150)" });
    ["Linear", "ReLU", "Linear"].forEach((s, i) => {
      const x = i * 175;
      K.el(net, "rect", { x, y: 0, width: 150, height: 64, rx: 12, fill: i === 1 ? K.color.moss : K.color.cobaltSoft, stroke: i === 1 ? K.color.moss : K.color.cobalt });
      K.text(net, x + 75, 42, s, { "font-size": 26, "text-anchor": "middle", "font-weight": 600, fill: i === 1 ? K.color.paper : K.color.cobalt });
      if (i < 2) K.text(net, x + 162, 42, "→", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });
    });

    const relu = K.el(g, "g", { transform: "translate(700 290)" });
    K.text(relu, 0, 0, "ReLU keeps positives, turns negatives into 0", { "font-size": 24 });
    const zin = [-2.0, -0.5, 0.0, 1.5, 3.0];
    const outs = [];
    zin.forEach((v, i) => {
      K.tile(relu, i * 100, 30, 88, 58, v.toFixed(1), { size: 24, ink: v < 0 ? K.color.rose : K.color.ink });
      const o = K.tile(relu, i * 100, 140, 88, 58, Math.max(0, v).toFixed(1), { size: 24, fill: v < 0 ? K.color.cobaltSoft : K.color.surface });
      K.text(relu, i * 100 + 44, 124, "↓", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });
      outs.push(o);
    });
    const bend = K.text(g, 700, 580, "That bend lets the network", { "font-size": 30, fill: K.color.moss, "font-weight": 600 });
    const bend2 = K.text(g, 700, 620, "learn a curved boundary.", { "font-size": 30, fill: K.color.moss, "font-weight": 600 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(plot, P(K, t, d, 0.04));
      const lineP = P(K, t, d, 0.3);
      const gone = P(K, t, d, 0.86);
      K.show(straight, lineP * (1 - 0.75 * gone));
      K.show(cross, P(K, t, d, 0.42) * (1 - 0.75 * gone));
      K.show(net, P(K, t, d, 0.52), 12);
      K.show(relu, P(K, t, d, 0.6), 12);
      outs.forEach((o, i) => K.show(o, P(K, t, d, 0.68 + i * 0.02)));
      curved.style.opacity = gone;
      curved.setAttribute("stroke-dasharray", "2000");
      curved.setAttribute("stroke-dashoffset", String(2000 * (1 - P(K, t, d, 0.86, 0.1))));
      K.show(bend, P(K, t, d, 0.88));
      K.show(bend2, P(K, t, d, 0.88));
    };
  };

  // ---------------------------------------------------------------- s09
  S.batches = (g, K) => {
    const h = heading(g, K, "Batches and epochs");
    const order = [7, 2, 9, 0, 4, 1, 8, 5, 3, 6]; // one shuffle; every epoch gets a new one
    const TW = 86;
    const GAP = 14;
    const X0 = 100;
    const tiles = [];
    for (let i = 0; i < 10; i++) tiles.push(K.tile(g, 0, 0, TW, 70, String(i), { size: 30 }));
    const slotX = (pos, sliced) => X0 + pos * (TW + GAP) + sliced * 30 * Math.floor(pos / 4);
    const shufLbl = K.text(g, 80, 165, "", { "font-size": 24, fill: K.color.muted });
    const groups = [[0, 4], [4, 8], [8, 10]].map(([a, b], i) => {
      const gg = K.el(g, "g", {});
      const x1 = X0 + a * (TW + GAP) + 30 * i;
      const x2 = X0 + (b - 1) * (TW + GAP) + 30 * i + TW;
      K.el(gg, "path", { d: `M${x1} 300 v12 H${x2} v-12`, fill: "none", stroke: K.color.cobalt, "stroke-width": 3 });
      K.text(gg, (x1 + x2) / 2, 345, `batch ${i + 1}: ${b - a}`, { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600 });
      return gg;
    });
    const epoch = K.el(g, "g", {});
    const sweep = K.el(epoch, "rect", { x: X0 - 10, y: 380, height: 10, rx: 5, fill: K.color.moss });
    K.text(epoch, 80, 430, "One epoch: every batch, once.", { "font-size": 28, fill: K.color.moss, "font-weight": 600 });

    const run = card(g, K, 80, 480, 1120, 190, K.color.cobalt);
    K.text(run, 28, 48, "Your training run (train.py)", { "font-size": 24, fill: K.color.muted });
    const facts = [
      { x: 28, big: "150", small: "epochs" },
      { x: 330, big: "64", small: "examples per batch" },
      { x: 720, big: "mps", small: "your Mac's GPU" },
    ].map((f) => {
      const fg = K.el(run, "g", {});
      K.text(fg, f.x, 124, f.big, { "font-size": 56, "font-weight": 700, fill: K.color.cobalt, "font-family": K.color.mono });
      K.text(fg, f.x, 164, f.small, { "font-size": 24 });
      return fg;
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      const shuf = P(K, t, d, 0.18, 0.1);
      const sliced = P(K, t, d, 0.3, 0.06);
      tiles.forEach((tl, i) => {
        const pos = K.lerp(i, order.indexOf(i), shuf);
        const x = slotX(pos, sliced);
        const lift = Math.sin(Math.PI * shuf) * (i % 2 ? -40 : 40);
        tl.setAttribute("transform", `translate(${x} ${215 + lift})`);
        tl.style.opacity = P(K, t, d, 0.02 + i * 0.008, 0.04);
      });
      shufLbl.textContent = shuf > 0.5 ? "shuffled, then sliced into batches of 4" : "10 examples";
      K.show(shufLbl, P(K, t, d, 0.04));
      groups.forEach((gr, i) => K.show(gr, P(K, t, d, 0.36 + i * 0.05)));
      K.show(epoch, P(K, t, d, 0.5));
      sweep.setAttribute("width", Math.max(0.01, (slotX(9, 1) + TW - X0 + 20) * P(K, t, d, 0.5, 0.12)));
      K.show(run, P(K, t, d, 0.64), 16);
      facts.forEach((f, i) => K.show(f, P(K, t, d, [0.68, 0.78, 0.88][i])));
    };
  };

  // ---------------------------------------------------------------- s10
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const lesson = card(g, K, 80, 140, 1120, 84, K.color.cobalt);
    K.text(lesson, 28, 52, "First, run the lesson. Section 7's worked example shows the pattern.", { "font-size": 28, "font-weight": 600, fill: K.color.cobalt });
    const items = [
      "1. linear_forward",
      "2. mse_loss",
      "3. sgd_step",
      "4. MLP",
      "5. train_one_epoch",
      "6. accuracy",
    ];
    K.text(g, 80, 278, "exercises.py", { "font-size": 22, fill: K.color.muted });
    const rows = items.map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(80 ${300 + i * 62})` });
      const box = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 28, "font-family": K.color.mono });
      return { r, box, tick };
    });
    const qs = card(g, K, 620, 300, 580, 110);
    K.text(qs, 28, 46, "answers.toml", { "font-size": 26, "font-family": K.color.mono, "font-weight": 600 });
    K.text(qs, 28, 84, "six short questions", { "font-size": 24, fill: K.color.muted });
    const train = card(g, K, 620, 440, 580, 220, K.color.moss);
    K.text(train, 28, 50, "train.py", { "font-size": 26, "font-family": K.color.mono, "font-weight": 600 });
    K.text(train, 28, 90, "your network on the two moons", { "font-size": 24, fill: K.color.muted });
    K.text(train, 28, 128, "about 5 seconds", { "font-size": 24, fill: K.color.muted });
    K.text(train, 28, 186, "needs 95% accuracy", { "font-size": 40, "font-weight": 700, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(lesson, P(K, t, d, 0.06), 12);
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, 0.36 + i * 0.045));
        const done = P(K, t, d, 0.4 + i * 0.045, 0.03);
        row.tick.style.opacity = done;
        row.box.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(qs, P(K, t, d, 0.64), 12);
      K.show(train, P(K, t, d, 0.72), 12);
    };
  };
})();
