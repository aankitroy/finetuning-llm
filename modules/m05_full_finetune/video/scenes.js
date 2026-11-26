// M5's video: one drawing per scene in script.toml. Numbers come from the measured runs.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // ---------------------------------------------------------------- s01
  S.intro = (g, K) => {
    const h = heading(g, K, "Fine-tuning: start from a model that already knows a lot");
    const panels = [
      { x: 80, title: "Random weights", sub: ["Where M4 began.", "2,000 steps just", "to learn to spell."], kind: "noise" },
      { x: 480, title: "Pretrained", sub: ["Trained on trillions", "of words: grammar,", "facts, instructions."], kind: "full" },
      { x: 880, title: "Fine-tuned", sub: ["The same model,", "trained a little more", "on your examples."], kind: "tuned" },
    ];
    const built = panels.map((p) => {
      const pg = K.el(g, "g", { transform: `translate(${p.x} 170)` });
      K.el(pg, "rect", { width: 320, height: 380, rx: 14, fill: K.color.paper, stroke: K.color.line });
      K.text(pg, 24, 52, p.title, { "font-size": 32, "font-weight": 600 });
      const dots = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 8; c++) {
          dots.push(K.el(pg, "circle", { cx: 40 + c * 34, cy: 100 + r * 30, r: 9, fill: K.color.cobalt }));
        }
      }
      p.sub.forEach((line, i) => K.text(pg, 24, 300 + i * 30, line, { "font-size": 22, fill: K.color.muted }));
      let badge = null;
      if (p.kind === "tuned") {
        badge = K.el(pg, "g", {});
        K.el(badge, "rect", { x: 24, y: 240, width: 272, height: 34, rx: 17, fill: K.color.moss });
        K.text(badge, 160, 264, "+ your house style", { "font-size": 20, fill: "#fff", "text-anchor": "middle", "font-weight": 600 });
      }
      return { pg, dots, badge, kind: p.kind };
    });
    const arrows = [400, 800].map((x) => K.text(g, x + 40, 380, "→", { "font-size": 56, fill: K.color.muted, "text-anchor": "middle" }));
    const job = K.text(g, 640, 640, "This module's job: rewrite jargon in plain English", {
      "font-size": 32, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600,
    });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      built.forEach((b, i) => {
        K.show(b.pg, P(K, t, d, [0.05, 0.32, 0.55][i], 0.08), 20);
        b.dots.forEach((dot, k) => {
          // random weights flicker; trained ones settle into a pattern
          const settled = b.kind === "noise" ? 0 : 1;
          const flicker = 0.25 + 0.6 * Math.abs(Math.sin(t * 2.3 + k * 1.7));
          const pattern = 0.2 + 0.75 * ((k * 7) % 5 === 0 || (k % 8) === (Math.floor(k / 8) * 2) % 8 ? 1 : 0.15);
          dot.setAttribute("opacity", settled ? pattern : flicker);
          if (b.kind === "tuned") dot.setAttribute("fill", (k % 8) > 5 && P(K, t, d, 0.62) > 0.5 ? K.color.moss : K.color.cobalt);
        });
        if (b.badge) K.show(b.badge, P(K, t, d, 0.66));
      });
      arrows.forEach((a, i) => K.show(a, P(K, t, d, [0.28, 0.52][i])));
      K.show(job, P(K, t, d, 0.8), 16);
    };
  };

  // ---------------------------------------------------------------- s02
  S.base_try = (g, K) => {
    const h = heading(g, K, "The original model tries the task");
    const chip = K.el(g, "g", { transform: "translate(80 140)" });
    K.el(chip, "rect", { width: 520, height: 52, rx: 26, fill: K.color.cobaltSoft });
    K.text(chip, 26, 35, "Qwen2.5-0.5B-Instruct, 494M parameters", { "font-size": 22, fill: K.color.cobalt, "font-weight": 600 });

    const inBox = K.el(g, "g", { transform: "translate(80 230)" });
    K.el(inBox, "rect", { width: 1120, height: 130, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(inBox, 28, 40, "You: rewrite in plain English", { "font-size": 20, fill: K.color.muted });
    K.para(inBox, 28, 80, "Our goal is to operationalize the new sales playbook across all twelve regional offices by the end of Q2.", { chars: 70, lh: 36, "font-size": 27 });

    const outBox = K.el(g, "g", { transform: "translate(80 390)" });
    K.el(outBox, "rect", { width: 1120, height: 130, rx: 14, fill: K.color.surface, stroke: K.color.cobalt });
    K.text(outBox, 28, 40, "The original model", { "font-size": 20, fill: K.color.muted });
    const out = K.para(outBox, 28, 80, "Our objective is to implement the new sales playbook at all twelve regional offices by the end of Q2.", { chars: 70, lh: 36, "font-size": 27 });
    const formal = out.words.filter((w) => /objective|implement/.test(w.textContent));

    const note = K.text(g, 80, 590, "It rewrites, but in its own formal style, not yours.", { "font-size": 30, fill: K.color.amber, "font-weight": 600 });
    const plan = K.text(g, 80, 650, "Fine-tuning teaches the house style from 400 examples.", { "font-size": 30 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(chip, P(K, t, d, 0.04));
      K.show(inBox, P(K, t, d, 0.18), 16);
      K.show(outBox, P(K, t, d, 0.3), 16);
      const typed = P(K, t, d, 0.3, 0.2);
      out.words.forEach((w, i) => (w.style.opacity = i < typed * out.words.length ? 1 : 0));
      const hi = P(K, t, d, 0.5);
      formal.forEach((w) => {
        w.setAttribute("fill", hi > 0.5 ? K.color.amber : K.color.ink);
        w.setAttribute("font-weight", hi > 0.5 ? 700 : 400);
      });
      K.show(note, P(K, t, d, 0.55));
      K.show(plan, P(K, t, d, 0.82));
    };
  };

  // ---------------------------------------------------------------- the 66-token bar (s03, s04)
  const PROMPT_TOKENS = 48;
  const ANSWER_TOKENS = 18;
  const CELL = 16;
  const BAR_X = (1280 - (PROMPT_TOKENS + ANSWER_TOKENS) * CELL) / 2;

  function tokenBar(g, K, y) {
    const cells = [];
    for (let i = 0; i < PROMPT_TOKENS + ANSWER_TOKENS; i++) {
      const answer = i >= PROMPT_TOKENS;
      cells.push(
        K.el(g, "rect", {
          x: BAR_X + i * CELL + 1, y, width: CELL - 2, height: 54, rx: 3,
          fill: answer ? K.color.moss : K.color.cobalt,
        }),
      );
    }
    return cells;
  }

  // ---------------------------------------------------------------- s03
  S.sequence = (g, K) => {
    const h = heading(g, K, "One example becomes one sequence of tokens");
    const cells = tokenBar(g, K, 300);
    const spans = [
      { from: 0, to: 21, label: "system message" },
      { from: 21, to: 45, label: "user's jargon" },
      { from: 45, to: 48, label: "assistant:" },
      { from: 48, to: 66, label: "plain rewrite" },
    ].map((s) => {
      const sg = K.el(g, "g", {});
      const x1 = BAR_X + s.from * CELL;
      const x2 = BAR_X + s.to * CELL;
      K.el(sg, "path", { d: `M${x1 + 2} 286 v-12 H${x2 - 2} v12`, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      K.text(sg, (x1 + x2) / 2, 262, s.label, { "font-size": 20, "text-anchor": "middle", fill: K.color.muted });
      return sg;
    });
    const promptLbl = K.text(g, BAR_X, 400, "Prompt: 48 tokens, formatted with the chat template (M3)", { "font-size": 26, fill: K.color.cobalt, "font-weight": 600 });
    const answerLbl = K.text(g, BAR_X, 446, "Answer: 18 tokens", { "font-size": 26, fill: K.color.moss, "font-weight": 600 });
    const endTok = K.tile(g, BAR_X + 66 * CELL - 120, 500, 160, 46, "<|im_end|>", { size: 20 });
    const endNote = K.text(g, BAR_X + 66 * CELL - 150, 580, "the last token: \"my turn is over\"", { "font-size": 22, "text-anchor": "end", fill: K.color.muted });
    const pairs = K.el(g, "g", {});
    K.para(pairs, 80, 160, "Jargon: \"Our model was optimized using cross-validation in order to minimize overfitting.\"", { chars: 90, "font-size": 22, fill: K.color.muted });
    K.para(pairs, 80, 196, "Plain: \"We tuned our model with cross-validation so it would not overfit.\"", { chars: 90, "font-size": 22, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(pairs, P(K, t, d, 0.04));
      const shown = P(K, t, d, 0.2, 0.5) * cells.length;
      cells.forEach((c, i) => (c.style.opacity = i < shown ? 1 : 0.08));
      spans.forEach((s, i) => K.show(s, P(K, t, d, [0.32, 0.45, 0.58, 0.66][i])));
      K.show(promptLbl, P(K, t, d, 0.4));
      K.show(answerLbl, P(K, t, d, 0.68));
      K.show(endTok, P(K, t, d, 0.82), 12);
      K.show(endNote, P(K, t, d, 0.86));
      cells[cells.length - 1].setAttribute("fill", P(K, t, d, 0.82) > 0.5 && Math.sin(t * 6) > 0 ? K.color.amber : K.color.moss);
    };
  };

  // ---------------------------------------------------------------- s04
  S.masking = (g, K) => {
    const h = heading(g, K, "Loss masking: only the answer is graded");
    K.text(g, BAR_X, 210, "input", { "font-size": 22, fill: K.color.muted });
    const cells = tokenBar(g, K, 222);
    K.text(g, BAR_X, 348, "labels", { "font-size": 22, fill: K.color.muted });
    const labels = [];
    for (let i = 0; i < PROMPT_TOKENS + ANSWER_TOKENS; i++) {
      labels.push(K.el(g, "rect", { x: BAR_X + i * CELL + 1, y: 360, width: CELL - 2, height: 54, rx: 3, fill: i >= PROMPT_TOKENS ? K.color.moss : K.color.cobalt }));
    }
    const legend = K.el(g, "g", { transform: `translate(${BAR_X} 470)` });
    K.el(legend, "rect", { width: 30, height: 30, rx: 4, fill: K.color.line });
    K.text(legend, 44, 25, "-100 means \"don't grade this position\"", { "font-size": 26 });
    const why = K.text(g, BAR_X, 560, "The model is always given the prompt; it never has to write it.", { "font-size": 26, fill: K.color.muted });
    const count = K.text(g, 640, 650, "", { "font-size": 40, "text-anchor": "middle", "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      cells.forEach((c) => (c.style.opacity = 1));
      const sweep = P(K, t, d, 0.4, 0.22) * PROMPT_TOKENS;
      labels.forEach((l, i) => {
        if (i < PROMPT_TOKENS) l.setAttribute("fill", i < sweep ? K.color.line : K.color.cobalt);
        else l.setAttribute("opacity", 0.6 + 0.4 * P(K, t, d, 0.64));
      });
      K.show(legend, P(K, t, d, 0.45));
      K.show(why, P(K, t, d, 0.18));
      const n = P(K, t, d, 0.82);
      count.textContent = n > 0 ? `66 tokens in, ${Math.round(18 + (1 - n) * 48)} graded` : "";
      count.setAttribute("fill", n >= 1 ? K.color.moss : K.color.ink);
    };
  };

  // ---------------------------------------------------------------- s05
  S.padding = (g, K) => {
    const h = heading(g, K, "Padding: every row of a batch the same length");
    const lens = [9, 6, 4];
    const C = 40;
    const grids = [
      { x: 80, title: "input_ids", real: (r, c) => K.color.cobalt, pad: "pad" },
      { x: 480, title: "attention_mask", real: () => K.color.cobaltSoft, pad: "0", realText: "1" },
      { x: 880, title: "labels", real: (r, c) => (c < [4, 3, 2][r] ? K.color.line : K.color.moss), pad: "-100" },
    ].map((gr) => {
      const gg = K.el(g, "g", { transform: `translate(${gr.x} 190)` });
      K.text(gg, 0, 0, gr.title, { "font-size": 24, "font-family": K.color.mono, fill: K.color.muted });
      const pads = [];
      lens.forEach((len, r) => {
        for (let c = 0; c < 9; c++) {
          const real = c < len;
          const cell = K.el(gg, "g", { transform: `translate(${c * C} ${30 + r * 56})` });
          K.el(cell, "rect", {
            width: C - 4, height: 46, rx: 5,
            fill: real ? gr.real(r, c) : K.color.paper,
            stroke: real ? "none" : K.color.muted, "stroke-dasharray": real ? "" : "4 3",
          });
          const label = real ? gr.realText || "" : gr.pad;
          if (label) K.text(cell, (C - 4) / 2, 30, label, { "font-size": label.length > 2 ? 12 : 18, "text-anchor": "middle", fill: real ? K.color.cobalt : K.color.muted, "font-family": K.color.mono });
          if (!real) pads.push(cell);
        }
      });
      return { gg, pads };
    });
    const notes = [
      K.text(g, 80, 440, "Shorter rows are padded at the end, to make a rectangle.", { "font-size": 27 }),
      K.text(g, 80, 500, "The attention mask: 1 for real tokens, 0 for padding, so the model never looks at it.", { "font-size": 27 }),
      K.text(g, 80, 560, "Padding's labels are -100 too, so padding is never graded.", { "font-size": 27 }),
    ];
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(grids[0].gg, P(K, t, d, 0.05));
      grids[0].pads.forEach((p, i) => K.show(p, P(K, t, d, 0.2 + i * 0.012, 0.05)));
      K.show(grids[1].gg, P(K, t, d, 0.42));
      K.show(grids[2].gg, P(K, t, d, 0.72));
      [0.2, 0.42, 0.72].forEach((a, i) => K.show(notes[i], P(K, t, d, a), 10));
    };
  };

  // ---------------------------------------------------------------- s06
  S.memory = (g, K) => {
    const h = heading(g, K, "Where the memory goes in a full fine-tune");
    const X = 80;
    const W = 1100;
    const GB = W / 40;
    const parts = [
      { gb: 1.98, label: "weights 2 GB", color: K.color.cobalt },
      { gb: 1.98, label: "gradients 2 GB", color: K.color.amber },
      { gb: 3.95, label: "AdamW's state 4 GB", color: K.color.moss },
      { gb: 16.1, label: "activations and the rest, ~16 GB", color: K.color.line },
    ];
    const bar = K.el(g, "g", { transform: "translate(0 200)" });
    K.el(bar, "rect", { x: X, y: 0, width: W, height: 70, rx: 8, fill: K.color.paper, stroke: K.color.line });
    let x = X;
    const legendX = [X, X + 230, X + 460, X + 740];
    const segs = parts.map((p, i) => {
      const s = K.el(bar, "g", {});
      K.el(s, "rect", { x, y: 0, width: p.gb * GB, height: 70, fill: p.color, rx: i === 0 ? 8 : 0 });
      K.el(s, "rect", { x: legendX[i], y: 96, width: 20, height: 20, rx: 4, fill: p.color });
      K.text(s, legendX[i] + 30, 113, p.label, { "font-size": 20, fill: K.color.ink });
      x += p.gb * GB;
      return s;
    });
    const mac = K.el(bar, "g", {});
    K.el(mac, "line", { x1: X + 36 * GB, y1: -24, x2: X + 36 * GB, y2: 94, stroke: K.color.rose, "stroke-width": 3 });
    K.text(mac, X + 36 * GB, -34, "this Mac: 36 GB", { "font-size": 20, "text-anchor": "middle", fill: K.color.rose });
    const sum = K.text(g, X, 410, "16 bytes x 494M weights = 7.9 GB, before anything else", { "font-size": 30, "font-weight": 600 });
    const peak = K.text(g, X, 460, "The real run peaks at about 24 GB.", { "font-size": 30 });
    const big = K.el(g, "g", { transform: "translate(0 530)" });
    K.text(big, X, -14, "A 7B model, same sum:", { "font-size": 24, fill: K.color.muted });
    K.el(big, "rect", { x: X, y: 0, width: W, height: 56, rx: 8, fill: K.color.rose, opacity: 0.85 });
    K.text(big, X + W - 16, 38, "112 GB before activations →", { "font-size": 26, "text-anchor": "end", fill: "#fff", "font-weight": 600 });
    const partC = K.text(g, X, 660, "So Part C trains far fewer weights.", { "font-size": 30, fill: K.color.cobalt, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      [0.12, 0.2, 0.3, 0.55].forEach((a, i) => K.show(segs[i], P(K, t, d, a)));
      K.show(sum, P(K, t, d, 0.4));
      K.show(peak, P(K, t, d, 0.56));
      K.show(mac, P(K, t, d, 0.58));
      K.show(big, P(K, t, d, 0.72), 16);
      K.show(partC, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s07
  S.training = (g, K) => {
    const h = heading(g, K, "Training: the loop you wrote in M4");
    const cx = 330;
    const cy = 400;
    const R = 170;
    const steps = ["batch", "loss", "backward", "step"];
    const ring = K.el(g, "g", {});
    K.el(ring, "circle", { cx, cy, r: R, fill: "none", stroke: K.color.line, "stroke-width": 4 });
    steps.forEach((s, i) => {
      const a = -Math.PI / 2 + (i * Math.PI) / 2;
      const x = cx + R * Math.cos(a);
      const y = cy + R * Math.sin(a);
      K.el(ring, "rect", { x: x - 70, y: y - 26, width: 140, height: 52, rx: 26, fill: K.color.cobaltSoft });
      K.text(ring, x, y + 9, s, { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600 });
    });
    const dot = K.el(ring, "circle", { r: 12, fill: K.color.amber });
    const time = K.text(g, cx, cy + 10, "", { "font-size": 26, "text-anchor": "middle", fill: K.color.muted });

    const chart = K.el(g, "g", { transform: "translate(700 170)" });
    const H = 400;
    const scale = H / 1.8;
    K.text(chart, 0, -10, "loss on unseen rewrites", { "font-size": 22, fill: K.color.muted });
    K.el(chart, "line", { x1: 0, y1: H, x2: 480, y2: H, stroke: K.color.line, "stroke-width": 2 });
    const bars = [
      { x: 40, v: 1.55, label: "before", color: K.color.cobalt },
      { x: 280, v: 0.94, label: "after 2 epochs", color: K.color.moss },
    ].map((b) => {
      const r = K.el(chart, "rect", { x: b.x, width: 160, rx: 6, fill: b.color });
      const v = K.text(chart, b.x + 80, 0, b.v.toFixed(2), { "font-size": 34, "text-anchor": "middle", "font-weight": 600 });
      K.text(chart, b.x + 80, H + 34, b.label, { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
      return { r, v, b };
    });
    const note = K.text(g, 700, 660, "Plain English is now far more likely.", { "font-size": 30, "font-weight": 600, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(ring, P(K, t, d, 0.03));
      const a = -Math.PI / 2 + t * 1.6;
      dot.setAttribute("cx", cx + R * Math.cos(a));
      dot.setAttribute("cy", cy + R * Math.sin(a));
      const secs = Math.round(40 * K.clamp((t - 0.15 * d) / (0.3 * d)));
      time.textContent = t > 0.15 * d ? `${secs} s of training` : "";
      K.show(chart, P(K, t, d, 0.42));
      bars.forEach(({ r, v, b }, i) => {
        const p = P(K, t, d, [0.46, 0.66][i], 0.1);
        const hgt = b.v * scale * p;
        r.setAttribute("y", H - hgt);
        r.setAttribute("height", hgt);
        v.setAttribute("y", H - hgt - 14);
        v.style.opacity = p;
      });
      K.show(note, P(K, t, d, 0.84));
    };
  };

  // ---------------------------------------------------------------- s08
  S.before_after = (g, K) => {
    const h = heading(g, K, "Before and after, on a sentence it never saw");
    const src = K.para(g, 80, 160, "Jargon: Residents are requested to place recycling bins at the curbside prior to 7 a.m. on collection days.", { chars: 80, "font-size": 24, fill: K.color.muted, lh: 34 });
    const card = (y, label, text, color) => {
      const c = K.el(g, "g", { transform: `translate(80 ${y})` });
      K.el(c, "rect", { width: 1120, height: 150, rx: 14, fill: K.color.paper, stroke: color, "stroke-width": 2 });
      K.text(c, 28, 44, label, { "font-size": 22, fill: color, "font-weight": 600 });
      return { c, p: K.para(c, 28, 92, text, { chars: 66, lh: 38, "font-size": 30 }) };
    };
    const before = card(250, "Original model", "Residents should place recycling bins at curb-side locations by 7 a.m. on collection days.", K.color.muted);
    const after = card(440, "Fine-tuned", "Please put recycling bins at the curb before 7 a.m. on collection days.", K.color.moss);
    const plainWords = after.p.words.filter((w) => /Please|put|curb|before/.test(w.textContent));
    const formalWords = before.p.words.filter((w) => /Residents|should|place|curb-side|locations|by/.test(w.textContent));
    const tag = K.text(g, 80, 660, "Shorter, plainer, in the voice of the training examples.", { "font-size": 30, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(src.node, P(K, t, d, 0.04));
      K.show(before.c, P(K, t, d, 0.18), 16);
      K.show(after.c, P(K, t, d, 0.52), 16);
      const hi = P(K, t, d, 0.7) > 0.5;
      plainWords.forEach((w) => w.setAttribute("fill", hi ? K.color.moss : K.color.ink));
      formalWords.forEach((w) => w.setAttribute("fill", hi ? K.color.amber : K.color.ink));
      K.show(tag, P(K, t, d, 0.8));
    };
  };

  // ---------------------------------------------------------------- s09
  // What the reference run with 3 epochs at lr 2e-5 answered wrongly (it got the other 7 right)
  const QUESTIONS = [
    "Capital of France", "12 times 12", "The Red Planet", "Gas plants take in", "\"Thank you\" in Spanish",
    "Days in a leap year", "Length of a list in Python", "Symbol for gold", "Opposite of \"ancient\"", "Sides of a hexagon",
  ];
  const FORGOT = { 3: "\"oxygen\"", 4: "\"Thanks.\"", 6: "\"5\"" };

  S.forgetting = (g, K) => {
    const h = heading(g, K, "The cost: catastrophic forgetting");
    const sub = K.text(g, 80, 150, "Ten ordinary questions, before and after fine-tuning", { "font-size": 26, fill: K.color.muted });
    const chips = QUESTIONS.map((q, i) => {
      const c = K.el(g, "g", { transform: `translate(${80 + (i % 2) * 570} ${200 + Math.floor(i / 2) * 76})` });
      const box = K.el(c, "rect", { width: 540, height: 60, rx: 30, fill: K.color.paper, stroke: K.color.line });
      const mark = K.text(c, 34, 40, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      const label = K.text(c, 64, 39, q, { "font-size": 24 });
      const wrong = i in FORGOT ? K.el(label, "tspan", { fill: K.color.rose, "font-weight": 600 }, `  →  ${FORGOT[i]}`) : null;
      return { c, box, mark, wrong, forgot: i in FORGOT };
    });
    const after = K.text(g, 80, 620, "", { "font-size": 30, "font-weight": 600 });
    const fix = K.text(g, 80, 670, "Fewer epochs, a lower learning rate, or some general examples in the mix all help.", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.04));
      const lost = P(K, t, d, 0.62) > 0.5;
      chips.forEach((ch, i) => {
        K.show(ch.c, P(K, t, d, 0.3 + i * 0.015, 0.05));
        const bad = lost && ch.forgot;
        ch.mark.textContent = bad ? "✗" : "✓";
        ch.mark.setAttribute("fill", bad ? K.color.rose : K.color.moss);
        ch.box.setAttribute("stroke", bad ? K.color.rose : K.color.line);
        if (ch.wrong) ch.wrong.style.opacity = bad ? 1 : 0;
      });
      after.textContent = lost ? `After 3 epochs at learning rate 2e-5: ${10 - Object.keys(FORGOT).length} of 10 right` : "Before fine-tuning: 10 of 10 right";
      after.setAttribute("fill", lost ? K.color.rose : K.color.moss);
      K.show(after, P(K, t, d, 0.4));
      K.show(fix, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s10
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      "1. build_example: mask the prompt",
      "2. collate: pad a batch",
      "3. masked_lm_loss: grade only the answer",
      "4. optimizer_state_bytes: measure AdamW",
      "5. chrf: score a rewrite",
      "6. count_buzzwords: a fair style check",
      "train.py --loop: your own loop",
      "train.py --trainer: Hugging Face's Trainer",
    ];
    const rows = items.map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(${i < 6 ? 80 : 700} ${i < 6 ? 170 + i * 70 : 170 + (i - 6) * 70})` });
      const box = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 26, "font-family": K.color.mono });
      return { r, box, tick };
    });
    const first = K.text(g, 700, 420, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    const cmd = K.text(g, 700, 462, "It runs every step with real code.", { "font-size": 26, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, 0.18 + i * 0.07));
        const done = P(K, t, d, 0.25 + i * 0.07, 0.03);
        row.tick.style.opacity = done;
        row.box.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(first, P(K, t, d, 0.1));
      K.show(cmd, P(K, t, d, 0.12));
    };
  };
})();
