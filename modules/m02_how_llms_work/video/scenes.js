// M2's video: one drawing per scene in script.toml. Every number is one the lesson prints
// for SmolLM2-135M on "The capital of France is" (float32, greedy), or its worked examples.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  function arrow(g, K, x1, y1, x2, y2, color) {
    const a = K.el(g, "g", {});
    const c = color || K.color.muted;
    K.el(a, "line", { x1, y1, x2, y2, stroke: c, "stroke-width": 3 });
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const s = 12;
    const p1 = [x2 - s * Math.cos(ang - 0.45), y2 - s * Math.sin(ang - 0.45)];
    const p2 = [x2 - s * Math.cos(ang + 0.45), y2 - s * Math.sin(ang + 0.45)];
    K.el(a, "path", { d: `M${x2} ${y2} L${p1[0]} ${p1[1]} L${p2[0]} ${p2[1]} Z`, fill: c });
    return a;
  }

  // width of a mono tile for a label
  const tileW = (label, size = 24) => Math.max(56, Math.round(label.length * size * 0.62 + 28));

  // ---------------------------------------------------------------- s01
  S.intro_loop = (g, K) => {
    const h = heading(g, K, "A language model predicts what comes next");
    const sub = K.text(g, 80, 150, "M1 picked one of 2 classes. An LLM picks one of about 50,000 tokens.", {
      "font-size": 26, fill: K.color.muted,
    });

    const loop = K.el(g, "g", { transform: "translate(80 190)" });
    K.el(loop, "rect", { width: 560, height: 300, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(loop, 28, 50, "The loop", { "font-size": 30, "font-weight": 600 });
    const steps = [
      ["1", "Look at the tokens so far."],
      ["2", "Score every token in the vocabulary."],
      ["3", "Pick one, stick it on the end."],
    ].map(([n, s], i) => {
      const r = K.el(loop, "g", { transform: `translate(28 ${100 + i * 60})` });
      K.el(r, "circle", { cx: 18, cy: -8, r: 18, fill: K.color.cobalt });
      K.text(r, 18, 0, n, { "font-size": 22, fill: "#fff", "text-anchor": "middle", "font-weight": 700 });
      K.text(r, 50, 0, s, { "font-size": 24 });
      return r;
    });
    const again = K.text(loop, 28, 275, "... and go back to step 1.", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600 });

    const table = K.el(g, "g", { transform: "translate(680 190)" });
    K.el(table, "rect", { width: 520, height: 300, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(table, 28, 50, "Toy model: a lookup table", { "font-size": 30, "font-weight": 600 });
    const pairs = [["the", "cat"], ["cat", "sat"], ["sat", "on"], ["on", "the"]];
    const rows = pairs.map(([a, b], i) => {
      const r = K.el(table, "g", { transform: `translate(28 ${80 + i * 52})` });
      const bg = K.el(r, "rect", { width: 464, height: 44, rx: 8, fill: K.color.cobaltSoft, opacity: 0 });
      K.text(r, 20, 31, a, { "font-size": 26, "font-family": K.color.mono });
      K.text(r, 170, 31, "→", { "font-size": 26, fill: K.color.muted });
      K.text(r, 240, 31, b, { "font-size": 26, "font-family": K.color.mono, fill: K.color.cobalt });
      return bg;
    });

    const words = ["the", "cat", "sat", "on", "the", "cat", "sat"];
    const tw = 130;
    const gap = 14;
    const x0 = (1280 - (words.length * tw + (words.length - 1) * gap)) / 2;
    const tiles = words.map((w, i) =>
      K.tile(g, x0 + i * (tw + gap), 560, tw, 64, w, { size: 28, fill: i === 0 ? K.color.paper : K.color.cobaltSoft }),
    );
    const promptLbl = K.text(g, x0, 660, "prompt", { "font-size": 22, fill: K.color.muted });
    const rowOf = { the: 0, cat: 1, sat: 2, on: 3 };

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.03));
      K.show(loop, P(K, t, d, 0.45), 16);
      steps.forEach((s, i) => K.show(s, P(K, t, d, 0.53 + i * 0.07)));
      K.show(again, P(K, t, d, 0.72));
      K.show(table, P(K, t, d, 0.78), 16);
      const start = 0.85;
      const each = 0.018;
      K.show(tiles[0], P(K, t, d, start - 0.02));
      K.show(promptLbl, P(K, t, d, start - 0.02));
      let active = -1;
      for (let i = 1; i < tiles.length; i++) {
        const p = P(K, t, d, start + i * each, 0.025);
        K.show(tiles[i], p, 14);
        if (p > 0 && p < 1) active = rowOf[words[i - 1]];
      }
      rows.forEach((bg, i) => bg.setAttribute("opacity", i === active ? 1 : 0));
    };
  };

  // ---------------------------------------------------------------- s02
  S.real_model = (g, K) => {
    const h = heading(g, K, "A real model: SmolLM2-135M");
    const chip = K.el(g, "g", { transform: "translate(80 130)" });
    K.el(chip, "rect", { width: 640, height: 52, rx: 26, fill: K.color.cobaltSoft });
    K.text(chip, 26, 35, "134,515,008 parameters · built like ChatGPT", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600 });

    const prompt = K.text(g, 640, 260, "The capital of France is", { "font-size": 40, "text-anchor": "middle", "font-weight": 600 });
    const tokLbl = K.el(g, "g", {});
    arrow(tokLbl, K, 640, 285, 640, 335);
    K.text(tokLbl, 660, 318, "tokenizer", { "font-size": 22, fill: K.color.muted });

    const pieces = ["The", "␣capital", "␣of", "␣France", "␣is"];
    const ids = ["504", "3575", "282", "4649", "314"];
    const widths = pieces.map((p) => tileW(p, 28));
    const gap = 20;
    const total = widths.reduce((a, b) => a + b, 0) + gap * (pieces.length - 1);
    let x = (1280 - total) / 2;
    const toks = pieces.map((p, i) => {
      const tg = K.el(g, "g", {});
      K.tile(tg, x, 360, widths[i], 64, p, { size: 28, fill: K.color.paper });
      const idT = K.text(tg, x + widths[i] / 2, 470, ids[i], {
        "font-size": 28, "text-anchor": "middle", fill: K.color.cobalt, "font-family": K.color.mono, "font-weight": 600,
      });
      x += widths[i] + gap;
      return { tg, idT };
    });
    const idLbl = K.text(g, (1280 - total) / 2 - 20, 470, "id", { "font-size": 22, fill: K.color.muted, "text-anchor": "end" });
    const space = K.text(g, 640, 540, "␣ marks a space: it usually belongs to the word after it.", {
      "font-size": 24, fill: K.color.muted, "text-anchor": "middle",
    });
    const vocab = K.text(g, 640, 630, "Vocabulary: 49,152 tokens, each with its own id", {
      "font-size": 32, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600,
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(chip, P(K, t, d, 0.04));
      K.show(prompt, P(K, t, d, 0.3));
      K.show(tokLbl, P(K, t, d, 0.36));
      toks.forEach((tk, i) => {
        K.show(tk.tg, P(K, t, d, 0.5 + i * 0.04), 14);
        tk.idT.style.opacity = P(K, t, d, 0.56 + i * 0.04);
      });
      K.show(idLbl, P(K, t, d, 0.56));
      K.show(space, P(K, t, d, 0.74));
      K.show(vocab, P(K, t, d, 0.84), 16);
    };
  };

  // ---------------------------------------------------------------- s03
  S.inside = (g, K) => {
    const h = heading(g, K, "Inside: a transformer");
    const box = (x, y, w, hgt, title, lines, fill) => {
      const b = K.el(g, "g", { transform: `translate(${x} ${y})` });
      K.el(b, "rect", { width: w, height: hgt, rx: 12, fill: fill || K.color.paper, stroke: K.color.line });
      K.text(b, w / 2, 44, title, { "font-size": 26, "font-weight": 600, "text-anchor": "middle" });
      lines.forEach((l, i) => K.text(b, w / 2, 84 + i * 30, l, { "font-size": 22, fill: K.color.muted, "text-anchor": "middle" }));
      return b;
    };
    const Y = 200;
    const ids = box(60, Y, 150, 150, "Token ids", ["5 of them"]);
    const emb = box(250, Y, 200, 150, "Embedding", ["576 numbers", "per token"]);
    const fin = box(870, Y, 150, 150, "Last layer", ["576 in,", "49,152 out"]);
    const out = box(1060, Y, 160, 150, "Logits", ["49,152", "scores"], K.color.cobaltSoft);

    // the stack of 30 blocks
    const stack = K.el(g, "g", { transform: "translate(490 170)" });
    const layers = [];
    for (let i = 0; i < 30; i++) {
      layers.push(K.el(stack, "rect", { x: 10, y: 200 - i * 6.4, width: 320, height: 5, rx: 2, fill: K.color.cobalt }));
    }
    K.text(stack, 170, 240, "30 identical blocks", { "font-size": 26, "font-weight": 600, "text-anchor": "middle" });

    const arrows = [
      [210, 250], [450, 490], [830, 870], [1020, 1060],
    ].map(([a, b]) => arrow(g, K, a + 4, Y + 75, b - 4, Y + 75));

    const blk = K.el(g, "g", { transform: "translate(250 460)" });
    K.el(blk, "rect", { width: 780, height: 190, rx: 14, fill: K.color.paper, stroke: K.color.cobalt, "stroke-width": 2 });
    K.text(blk, 30, 48, "Each block does two things:", { "font-size": 28, "font-weight": 600 });
    const att = K.el(blk, "g", {});
    K.text(att, 30, 100, "1. Attention:", { "font-size": 26, "font-weight": 600, fill: K.color.cobalt });
    K.text(att, 210, 100, "each token gathers information from earlier ones", { "font-size": 24 });
    const mlp = K.el(blk, "g", {});
    K.text(mlp, 30, 150, "2. MLP:", { "font-size": 26, "font-weight": 600, fill: K.color.moss });
    K.text(mlp, 210, 150, "a small network, like the one you built in M1", { "font-size": 24 });
    const link = K.el(g, "path", { d: "M660 432 L640 460 L620 432", fill: "none", stroke: K.color.cobalt, "stroke-width": 2 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(ids, P(K, t, d, 0.12));
      K.show(arrows[0], P(K, t, d, 0.16));
      K.show(emb, P(K, t, d, 0.17), 14);
      K.show(arrows[1], P(K, t, d, 0.3));
      K.show(stack, P(K, t, d, 0.3));
      const grow = P(K, t, d, 0.3, 0.12);
      layers.forEach((l, i) => (l.style.opacity = i < grow * 30 ? 1 : 0));
      K.show(blk, P(K, t, d, 0.42), 14);
      K.show(link, P(K, t, d, 0.42));
      K.show(att, P(K, t, d, 0.46));
      K.show(mlp, P(K, t, d, 0.6));
      K.show(arrows[2], P(K, t, d, 0.72));
      K.show(fin, P(K, t, d, 0.72), 14);
      K.show(arrows[3], P(K, t, d, 0.86));
      K.show(out, P(K, t, d, 0.86), 14);
    };
  };

  // ---------------------------------------------------------------- s04
  S.attention_idea = (g, K) => {
    const h = heading(g, K, "Attention: a token looks at the others");

    // a sentence as word tiles, with an arc from "it" to the word it means
    function sentence(y, words, itIdx, refIdx) {
      const sg = K.el(g, "g", {});
      const size = 24;
      const gap = 10;
      const ws = words.map((w) => tileW(w, size));
      let x = 80;
      const xs = ws.map((w) => {
        const x1 = x;
        x += w + gap;
        return x1;
      });
      words.forEach((w, i) => {
        const hi = i === itIdx ? K.color.amber : i === refIdx ? K.color.cobaltSoft : K.color.paper;
        K.tile(sg, xs[i], y, ws[i], 50, w, { size, fill: hi, ink: i === itIdx ? "#fff" : K.color.ink });
      });
      const ax = xs[itIdx] + ws[itIdx] / 2;
      const bx = xs[refIdx] + ws[refIdx] / 2;
      const arc = K.el(sg, "path", {
        d: `M${ax} ${y - 4} C${ax} ${y - 50} ${bx} ${y - 50} ${bx} ${y - 4}`,
        fill: "none", stroke: K.color.amber, "stroke-width": 3, "stroke-dasharray": "600", "stroke-dashoffset": "600",
      });
      return { sg, arc };
    }
    const s1 = sentence(190, ["the", "cat", "drank", "because", "it", "was", "thirsty"], 4, 1);
    const s2 = sentence(310, ["the", "cat", "drank", "the", "milk", "because", "it", "was", "there"], 6, 4);

    const lib = K.text(g, 80, 430, "Like a library search, each token makes three lists of numbers:", { "font-size": 26, fill: K.color.muted });
    const cards = [
      ["Query", "What am I", "looking for?", K.color.cobalt],
      ["Key", "What do I", "contain?", K.color.moss],
      ["Value", "What I hand over", "if attended to", K.color.amber],
    ].map(([title, a, b, c], i) => {
      const cg = K.el(g, "g", { transform: `translate(${80 + i * 380} 460)` });
      K.el(cg, "rect", { width: 350, height: 180, rx: 14, fill: K.color.paper, stroke: c, "stroke-width": 3 });
      K.text(cg, 28, 56, title, { "font-size": 34, "font-weight": 700, fill: c });
      K.text(cg, 28, 110, a, { "font-size": 26 });
      K.text(cg, 28, 146, b, { "font-size": 26 });
      return cg;
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(s1.sg, P(K, t, d, 0.06));
      s1.arc.setAttribute("stroke-dashoffset", 600 * (1 - P(K, t, d, 0.18, 0.08)));
      K.show(s2.sg, P(K, t, d, 0.22));
      s2.arc.setAttribute("stroke-dashoffset", 600 * (1 - P(K, t, d, 0.34, 0.08)));
      K.show(lib, P(K, t, d, 0.5));
      cards.forEach((c, i) => K.show(c, P(K, t, d, [0.64, 0.75, 0.84][i]), 16));
    };
  };

  // ---------------------------------------------------------------- s05
  S.attention_by_hand = (g, K) => {
    const h = heading(g, K, "Attention by hand: 3 tokens");
    const given = K.el(g, "g", {});
    const gAttrs = { "font-size": 24, "font-family": K.color.mono, fill: K.color.muted };
    K.text(given, 80, 150, "queries = keys: [1, 0] [0, 1] [1, 1]", gAttrs);
    K.text(given, 700, 150, "values: [10, 0] [0, 10] [5, 5]", gAttrs);
    const C = 76;
    const Y = 230;
    function grid(x, vals, title, sub) {
      const gg = K.el(g, "g", { transform: `translate(${x} ${Y})` });
      const cells = [];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const rect = K.el(gg, "rect", { x: c * C, y: r * C, width: C - 4, height: C - 4, rx: 6, fill: K.color.paper, stroke: K.color.line });
          const txt = K.text(gg, c * C + (C - 4) / 2, r * C + C / 2 + 6, vals[r][c], {
            "font-size": 24, "text-anchor": "middle", "font-family": K.color.mono,
          });
          cells.push({ r, c, rect, txt });
        }
      }
      K.text(gg, 0, 3 * C + 40, title, { "font-size": 26, "font-weight": 600 });
      K.text(gg, 0, 3 * C + 72, sub, { "font-size": 22, fill: K.color.muted });
      return { gg, cells };
    }
    const scores = grid(110, [["0.71", "0", "0.71"], ["0", "0.71", "0.71"], ["0.71", "0.71", "1.41"]], "1. Score", "query · key");
    const masked = grid(420, [["0.71", "−∞", "−∞"], ["0", "0.71", "−∞"], ["0.71", "0.71", "1.41"]], "2. Hide the future", "the causal mask");
    const weights = grid(730, [["1", "0", "0"], ["0.33", "0.67", "0"], ["0.25", "0.25", "0.50"]], "3. Softmax", "each row adds up to 1");
    const rowLbls = K.el(g, "g", {});
    ["1", "2", "3"].forEach((n, i) => K.text(rowLbls, 84, Y + i * C + C / 2 + 6, n, { "font-size": 22, fill: K.color.muted, "text-anchor": "middle" }));

    const outG = K.el(g, "g", { transform: `translate(1040 ${Y})` });
    const outs = ["10, 0", "3.3, 6.7", "5, 5"].map((s, r) => {
      const og = K.el(outG, "g", {});
      K.el(og, "rect", { x: 0, y: r * C, width: 150, height: C - 4, rx: 6, fill: r === 0 ? K.color.cobaltSoft : K.color.paper, stroke: K.color.line });
      K.text(og, 75, r * C + C / 2 + 6, s, { "font-size": 24, "text-anchor": "middle", "font-family": K.color.mono });
      return og;
    });
    K.text(outG, 0, 3 * C + 40, "4. Mix", { "font-size": 26, "font-weight": 600 });
    K.text(outG, 0, 3 * C + 72, "weights × values", { "font-size": 22, fill: K.color.muted });

    const arrows = [[350, 412], [660, 722], [970, 1032]].map(([a, b]) => arrow(g, K, a, Y + 1.5 * C, b, Y + 1.5 * C));
    const heads = K.text(g, 640, 650, "SmolLM2 runs 9 heads like this side by side, in each of its 30 blocks.", {
      "font-size": 28, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600,
    });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(given, P(K, t, d, 0.03));
      K.show(scores.gg, P(K, t, d, 0.12), 14);
      K.show(rowLbls, P(K, t, d, 0.12));
      K.show(arrows[0], P(K, t, d, 0.24));
      K.show(masked.gg, P(K, t, d, 0.25), 14);
      const m = P(K, t, d, 0.34);
      masked.cells.forEach((cl) => {
        if (cl.c > cl.r) {
          cl.rect.setAttribute("fill", m > 0.5 ? K.color.line : K.color.paper);
          cl.txt.setAttribute("fill", m > 0.5 ? K.color.rose : K.color.ink);
        }
      });
      K.show(arrows[1], P(K, t, d, 0.42));
      K.show(weights.gg, P(K, t, d, 0.43), 14);
      const z = P(K, t, d, 0.5);
      weights.cells.forEach((cl) => {
        if (cl.c > cl.r) cl.txt.setAttribute("fill", z > 0.5 ? K.color.muted : K.color.ink);
        if (cl.r === 0 && cl.c === 0) cl.rect.setAttribute("fill", P(K, t, d, 0.66) > 0.5 ? K.color.cobaltSoft : K.color.paper);
      });
      K.show(arrows[2], P(K, t, d, 0.56));
      outs.forEach((o, i) => K.show(o, P(K, t, d, 0.58 + i * 0.03), 10));
      K.show(outG, P(K, t, d, 0.56));
      K.show(heads, P(K, t, d, 0.84), 14);
    };
  };

  // ---------------------------------------------------------------- s06
  S.next_token = (g, K) => {
    const h = heading(g, K, "From logits to the next token");
    const pieces = ["The", "␣capital", "␣of", "␣France", "␣is"];
    const ws = pieces.map((p) => tileW(p, 24));
    let x = 80;
    const toks = pieces.map((p, i) => {
      const tl = K.tile(g, x, 140, ws[i], 52, p, { size: 24, fill: i === 4 ? K.color.cobaltSoft : K.color.paper });
      x += ws[i] + 12;
      return tl;
    });
    const nextX = x;
    const nextTile = K.tile(g, nextX, 140, tileW("␣the", 24), 52, "␣the", { size: 24, fill: K.color.moss, ink: "#fff" });
    const lastLbl = K.text(g, 80, 240, "Logits at the last position: 49,152 scores → softmax → probabilities", { "font-size": 24, fill: K.color.muted });

    const top = [["the", 26.2], ["Paris", 9.4], ["located", 7.3], ["called", 4.4], ["a", 3.9]];
    const BX = 300;
    const scale = 26;
    const bars = top.map(([w, p], i) => {
      const bg = K.el(g, "g", { transform: `translate(0 ${290 + i * 64})` });
      K.text(bg, BX - 24, 34, `"${w}"`, { "font-size": 28, "text-anchor": "end", "font-family": K.color.mono });
      const bar = K.el(bg, "rect", { x: BX, y: 6, width: 0, height: 40, rx: 6, fill: i === 0 ? K.color.cobalt : K.color.cobaltSoft });
      const lbl = K.text(bg, BX, 36, `${p.toFixed(1)}%`, { "font-size": 26, "font-weight": 600 });
      return { bg, bar, lbl, w: p * scale };
    });
    const greedy = K.text(g, 80, 650, "Greedy decoding: always take the most likely token.", { "font-size": 30, fill: K.color.moss, "font-weight": 600 });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      toks.forEach((tl, i) => K.show(tl, P(K, t, d, 0.03 + i * 0.02)));
      K.show(lastLbl, P(K, t, d, 0.18));
      bars.forEach((b, i) => {
        const p = P(K, t, d, 0.4 + i * 0.08, 0.06);
        K.show(b.bg, p > 0 ? 1 : 0);
        b.bar.setAttribute("width", b.w * p);
        b.lbl.setAttribute("x", BX + b.w * p + 14);
      });
      K.show(greedy, P(K, t, d, 0.8));
      K.show(nextTile, P(K, t, d, 0.9), -14);
    };
  };

  // ---------------------------------------------------------------- s07
  S.gen_loop = (g, K) => {
    const h = heading(g, K, "Predict, append, repeat");
    const prompt = ["The", "␣capital", "␣of", "␣France", "␣is"];
    const added = ["␣the", "␣capital", "␣of", "␣the", "␣country", ".", "↵", "↵"];
    const all = prompt.concat(added);
    const size = 26;
    const gap = 12;
    let x = 80;
    let y = 170;
    const tiles = all.map((p, i) => {
      const w = tileW(p, size);
      if (x + w > 1200) {
        x = 80;
        y += 80;
      }
      const tl = K.tile(g, x, y, w, 60, p, { size, fill: i < prompt.length ? K.color.paper : K.color.cobaltSoft });
      x += w + gap;
      return tl;
    });
    const counter = K.text(g, 80, 380, "", { "font-size": 26, fill: K.color.muted });
    const read = K.text(g, 80, 440, "\"The capital of France is the capital of the country.\" + 2 line breaks", { "font-size": 26 });

    const match = K.el(g, "g", { transform: "translate(80 490)" });
    K.el(match, "rect", { width: 1120, height: 64, rx: 32, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
    K.text(match, 32, 42, "✓  The library's model.generate() writes exactly the same tokens.", { "font-size": 26, fill: K.color.moss, "font-weight": 600 });
    const check = K.text(g, 80, 600, "Matching it token for token is M2's checkpoint.", { "font-size": 28, "font-weight": 600 });
    const eos = K.text(g, 80, 660, "If the model picks the end-of-text token, the loop stops early.", { "font-size": 26, fill: K.color.muted });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      prompt.forEach((_, i) => K.show(tiles[i], P(K, t, d, 0.02)));
      let n = 0;
      added.forEach((_, i) => {
        const p = P(K, t, d, 0.2 + i * 0.04, 0.03);
        K.show(tiles[prompt.length + i], p, 14);
        if (p > 0.5) n = i + 1;
      });
      counter.textContent = n ? `step ${n} of 8` : "";
      K.show(read, P(K, t, d, 0.5));
      K.show(match, P(K, t, d, 0.62), 14);
      K.show(check, P(K, t, d, 0.74));
      K.show(eos, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s08
  S.sampling = (g, K) => {
    const h = heading(g, K, "Sampling: temperature and top-p");
    const sub = K.text(g, 80, 150, "Pick at random, weighted by probability.", { "font-size": 26, fill: K.color.muted });

    // left: temperature
    const left = K.el(g, "g", { transform: "translate(80 190)" });
    K.text(left, 0, 30, "Temperature: logits ÷ t, then softmax", { "font-size": 26, "font-weight": 600 });
    const temps = [
      ["t = 0.5", [77.2, 9.9, 6.0], "sharper, safer"],
      ["t = 1.0", [26.2, 9.4, 7.3], ""],
      ["t = 2.0", [1.2, 0.7, 0.6], "flatter, more surprising"],
    ];
    const words = ["the", "Paris", "located"];
    const rows = temps.map(([lbl, ps, note], r) => {
      const rg = K.el(left, "g", { transform: `translate(0 ${70 + r * 150})` });
      K.text(rg, 0, 26, lbl, { "font-size": 24, "font-family": K.color.mono, "font-weight": 600 });
      if (note) K.text(rg, 150, 26, note, { "font-size": 22, fill: r === 0 ? K.color.cobalt : K.color.amber });
      ps.forEach((p, i) => {
        K.text(rg, 0, 64 + i * 30, words[i], { "font-size": 22, "font-family": K.color.mono, fill: K.color.muted });
        K.el(rg, "rect", { x: 130, y: 46 + i * 30, width: Math.max(3, p * 4.2), height: 22, rx: 4, fill: i === 0 ? K.color.cobalt : K.color.cobaltSoft });
        K.text(rg, 140 + Math.max(3, p * 4.2), 64 + i * 30, `${p.toFixed(1)}%`, { "font-size": 22 });
      });
      return rg;
    });

    // right: top-p
    const right = K.el(g, "g", { transform: "translate(700 190)" });
    K.el(right, "rect", { width: 500, height: 470, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(right, 28, 50, "Top-p: cut the long tail", { "font-size": 28, "font-weight": 600 });
    K.text(right, 28, 100, "Keep the most likely tokens", { "font-size": 24 });
    K.text(right, 28, 134, "until they add up to p = 0.9.", { "font-size": 24 });
    const keep = K.el(right, "g", {});
    K.text(keep, 28, 210, "This prompt needs", { "font-size": 24, fill: K.color.muted });
    K.text(keep, 28, 270, "405 tokens", { "font-size": 48, "font-weight": 700, fill: K.color.moss });
    K.text(keep, 28, 310, "out of 49,152", { "font-size": 24, fill: K.color.muted });
    const strip = K.el(right, "g", { transform: "translate(28 360)" });
    K.el(strip, "rect", { width: 444, height: 40, rx: 6, fill: K.color.line });
    K.el(strip, "rect", { width: 6, height: 40, rx: 2, fill: K.color.moss });
    K.text(strip, 0, 76, "kept", { "font-size": 22, fill: K.color.moss, "font-weight": 600 });
    K.text(strip, 444, 76, "all the rest: dropped", { "font-size": 22, fill: K.color.muted, "text-anchor": "end" });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.1));
      K.show(left, P(K, t, d, 0.22));
      rows.forEach((rg, i) => K.show(rg, P(K, t, d, [0.36, 0.25, 0.5][i]), 12));
      K.show(right, P(K, t, d, 0.68), 16);
      K.show(keep, P(K, t, d, 0.8));
      K.show(strip, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s09
  S.kv_cache = (g, K) => {
    const h = heading(g, K, "The KV cache");
    const SQ = 30;
    function panel(x, title, cached) {
      const pg = K.el(g, "g", { transform: `translate(${x} 150)` });
      K.el(pg, "rect", { width: 540, height: 380, rx: 14, fill: K.color.paper, stroke: cached ? K.color.moss : K.color.line, "stroke-width": cached ? 2 : 1 });
      K.text(pg, 28, 50, title, { "font-size": 28, "font-weight": 600 });
      const rows = [];
      for (let s = 0; s < 6; s++) {
        const rg = K.el(pg, "g", { transform: `translate(28 ${84 + s * 46})` });
        K.text(rg, 0, 26, `step ${s + 1}`, { "font-size": 22, fill: K.color.muted });
        const n = 5 + s;
        for (let i = 0; i < n; i++) {
          const fresh = !cached || i === n - 1;
          K.el(rg, "rect", { x: 110 + i * (SQ + 4), y: 2, width: SQ, height: SQ, rx: 4, fill: fresh ? K.color.cobalt : K.color.cobaltSoft });
        }
        rows.push(rg);
      }
      return { pg, rows };
    }
    const without = panel(80, "Your loop: every token, every step", false);
    const withc = panel(660, "With the KV cache: just the new one", true);
    const legend = K.el(g, "g", { transform: "translate(80 556)" });
    K.el(legend, "rect", { width: 26, height: 26, rx: 4, fill: K.color.cobalt });
    K.text(legend, 40, 22, "run through the model", { "font-size": 22 });
    K.el(legend, "rect", { x: 340, width: 26, height: 26, rx: 4, fill: K.color.cobaltSoft });
    K.text(legend, 380, 22, "keys and values kept in memory", { "font-size": 22 });
    const step20 = K.text(g, 80, 628, "By step 20: about 25 tokens reprocessed to get one new one.", { "font-size": 26, fill: K.color.muted });
    const same = K.text(g, 80, 684, "Same output, much faster.", { "font-size": 30, "font-weight": 600, fill: K.color.moss });

    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(without.pg, P(K, t, d, 0.02));
      without.rows.forEach((r, i) => K.show(r, P(K, t, d, 0.05 + i * 0.03)));
      K.show(step20, P(K, t, d, 0.18));
      K.show(withc.pg, P(K, t, d, 0.5), 14);
      withc.rows.forEach((r, i) => K.show(r, P(K, t, d, 0.55 + i * 0.03)));
      K.show(legend, P(K, t, d, 0.5));
      K.show(same, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s10
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      "1. next_token_logits",
      "2. greedy",
      "3. softmax_with_temperature",
      "4. top_p_filter",
      "5. causal_attention",
      "6. generate",
      "answers.toml: 6 questions",
      "generate.py: same 20 tokens",
    ];
    const rows = items.map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(${i < 6 ? 80 : 700} ${i < 6 ? 170 + i * 70 : 400 + (i - 6) * 70})` });
      const box = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 26, "font-family": K.color.mono });
      return { r, box, tick };
    });
    const first = K.text(g, 700, 196, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    const cmd = K.text(g, 700, 238, "It runs every step on the real model.", { "font-size": 26, fill: K.color.muted });
    const goal = K.text(g, 700, 372, "Then:", { "font-size": 26, fill: K.color.muted });
    const end = K.text(g, 640, 650, "Your loop matches model.generate() token for token.", {
      "font-size": 30, "text-anchor": "middle", "font-weight": 600, fill: K.color.moss,
    });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(first, P(K, t, d, 0.08));
      K.show(cmd, P(K, t, d, 0.1));
      const starts = [0.24, 0.32, 0.37, 0.44, 0.5, 0.58, 0.7, 0.8];
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, starts[i]));
        const done = P(K, t, d, starts[i] + 0.05, 0.03);
        row.tick.style.opacity = done;
        row.box.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(goal, P(K, t, d, 0.68));
      K.show(end, P(K, t, d, 0.9));
    };
  };
})();
