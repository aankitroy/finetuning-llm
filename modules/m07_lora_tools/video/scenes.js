// M7's video: one drawing per scene in script.toml. Numbers come from the reference runs of
// train.py (outputs/m07/run.json), the lesson's padding and sequence-length measurements, and
// Qwen2.5-1.5B-Instruct's real layer sizes.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // a labelled rounded box
  function box(g, K, x, y, w, h, label, { fill, stroke, ink, size = 24, dash, mono } = {}) {
    const b = K.el(g, "g", { transform: `translate(${x} ${y})` });
    const rect = K.el(b, "rect", {
      width: w, height: h, rx: 12, fill: fill || K.color.paper, stroke: stroke || K.color.line, "stroke-width": 2,
      ...(dash ? { "stroke-dasharray": dash } : {}),
    });
    const txt = K.text(b, w / 2, h / 2 + size * 0.35, label, {
      "font-size": size, "text-anchor": "middle", fill: ink || K.color.ink, "font-weight": 600,
      ...(mono ? { "font-family": K.color.mono } : {}),
    });
    return { b, rect, txt };
  }

  // a card with a coloured title and lines of monospace code; returns the group and one node per line
  function codeCard(g, K, x, y, w, title, lines, color, { size = 22, lh = 36 } = {}) {
    const c = K.el(g, "g", { transform: `translate(${x} ${y})` });
    const hgt = 70 + lines.length * lh;
    K.el(c, "rect", { width: w, height: hgt, rx: 14, fill: K.color.paper, stroke: color, "stroke-width": 2 });
    K.text(c, 24, 40, title, { "font-size": 22, fill: color, "font-weight": 600 });
    const rows = lines.map((ln, i) => {
      const [code, note] = Array.isArray(ln) ? ln : [ln, ""];
      const r = K.el(c, "g", {});
      // SVG drops leading spaces, so indent with no-break spaces
      const shown = code.replace(/^ +/, (sp) => "\u00a0".repeat(sp.length));
      K.text(r, 24, 82 + i * lh, shown, { "font-size": size, "font-family": K.color.mono });
      if (note) K.text(r, w - 24, 82 + i * lh, note, { "font-size": 20, fill: K.color.muted, "text-anchor": "end" });
      return r;
    });
    return { c, rows, hgt };
  }

  // a thin matrix drawn as a rectangle, with its name and shape under it
  function matrix(g, K, x, y, w, h, name, shape, color, { below = true } = {}) {
    const m = K.el(g, "g", {});
    const rect = K.el(m, "rect", { x, y, width: w, height: h, rx: 4, fill: color });
    const ty = below ? y + h + 32 : y - 40;
    K.text(m, x + w / 2, ty, name, { "font-size": 24, "text-anchor": "middle", "font-weight": 700, fill: color, "font-family": K.color.mono });
    K.text(m, x + w / 2, ty + 26, shape, { "font-size": 20, "text-anchor": "middle", fill: K.color.muted, "font-family": K.color.mono });
    return { m, rect };
  }

  // ---------------------------------------------------------------- s01
  S.recap = (g, K) => {
    const h = heading(g, K, "In M6, you built LoRA by hand");
    const diag = K.el(g, "g", {});
    const wire = (dd, color) => K.el(diag, "path", { d: dd, fill: "none", stroke: color || K.color.muted, "stroke-width": 3 });
    wire("M150 245 H190 V200 H220");
    wire("M480 200 H560 V222");
    wire("M190 245 V290 H220", K.color.moss);
    wire("M480 290 H560 V268", K.color.moss);
    wire("M583 245 H610");
    box(diag, K, 80, 220, 70, 50, "x", { size: 24 });
    box(diag, K, 220, 170, 260, 60, "W, frozen", { fill: K.color.cobaltSoft, stroke: K.color.cobalt, ink: K.color.cobalt, size: 22 });
    box(diag, K, 220, 260, 260, 60, "scale · B · A", { stroke: K.color.moss, ink: K.color.moss, size: 22 });
    K.el(diag, "circle", { cx: 560, cy: 245, r: 23, fill: K.color.paper, stroke: K.color.ink, "stroke-width": 2 });
    K.text(diag, 560, 255, "+", { "font-size": 30, "text-anchor": "middle" });
    K.text(diag, 220, 360, "rank 8, written by you, in plain PyTorch", { "font-size": 22, fill: K.color.muted });
    const now = K.text(g, 720, 180, "Now the real tools do it, and add:", { "font-size": 26, "font-weight": 600, fill: K.color.cobalt });
    const adds = ["saving and loading", "a speed you can trust", "a 3× bigger model"].map((s, i) => {
      const r = K.el(g, "g", { transform: `translate(720 ${235 + i * 56})` });
      K.el(r, "circle", { cx: 10, cy: -8, r: 7, fill: K.color.moss });
      K.text(r, 32, 0, s, { "font-size": 26 });
      return r;
    });
    // model size, to scale: 494 million vs 1.55 billion weights
    const sizes = K.el(g, "g", {});
    const W = 700 / 1.55;
    K.text(sizes, 80, 470, "M5 and M6: Qwen2.5-0.5B-Instruct, 494 million weights", { "font-size": 22, fill: K.color.muted });
    K.el(sizes, "rect", { x: 80, y: 486, width: 0.494 * W, height: 36, rx: 6, fill: K.color.cobalt });
    const big = K.el(g, "g", {});
    K.text(big, 80, 570, "M7: Qwen2.5-1.5B-Instruct, 1.55 billion weights", { "font-size": 22, "font-weight": 600 });
    const bigBar = K.el(big, "rect", { x: 80, y: 586, width: 0, height: 36, rx: 6, fill: K.color.moss });
    const gb = K.text(g, 800, 612, "about 3 GB in bf16", { "font-size": 28, "font-weight": 700, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(diag, P(K, t, d, 0.05), 14);
      K.show(now, P(K, t, d, 0.36));
      [0.48, 0.57, 0.66].forEach((a, i) => K.show(adds[i], P(K, t, d, a), 10));
      K.show(sizes, P(K, t, d, 0.68));
      K.show(big, P(K, t, d, 0.72));
      bigBar.setAttribute("width", 1.55 * W * P(K, t, d, 0.73, 0.1));
      K.show(gb, P(K, t, d, 0.86));
    };
  };

  // ---------------------------------------------------------------- s02
  S.tools = (g, K) => {
    const h = heading(g, K, "Two tools for the same job");
    const card = (x, color, name, from, lines) => {
      const c = K.el(g, "g", { transform: `translate(${x} 150)` });
      K.el(c, "rect", { width: 540, height: 470, rx: 16, fill: K.color.paper, stroke: color, "stroke-width": 2 });
      K.text(c, 30, 64, name, { "font-size": 44, "font-weight": 700, fill: color, "font-family": K.color.mono });
      K.text(c, 30, 104, from, { "font-size": 24, fill: K.color.muted });
      lines.forEach((s, i) => K.text(c, 30, 164 + i * 44, s, { "font-size": 26 }));
      return c;
    };
    const mlx = card(80, K.color.cobalt, "mlx-lm", "from Apple", ["built on MLX", "trains on your Mac's GPU"]);
    // unified memory: CPU and GPU draw on one pool
    const uni = K.el(g, "g", { transform: "translate(110 380)" });
    box(uni, K, 0, 0, 160, 60, "CPU", { size: 24 });
    box(uni, K, 320, 0, 160, 60, "GPU", { size: 24, stroke: K.color.cobalt, ink: K.color.cobalt });
    K.el(uni, "path", { d: "M80 60 V100 M400 60 V100", stroke: K.color.muted, "stroke-width": 3 });
    box(uni, K, 0, 100, 480, 70, "one shared pool of memory", { fill: K.color.cobaltSoft, stroke: K.color.cobalt, ink: K.color.cobalt, size: 24 });
    K.text(uni, 240, 210, "unified memory", { "font-size": 22, "text-anchor": "middle", fill: K.color.muted });
    const peft = card(660, K.color.moss, "PEFT", "from Hugging Face", ["built on PyTorch", "runs almost anywhere:"]);
    const where = ["NVIDIA GPU", "Mac", "CPU"].map((s, i) => {
      const w = K.el(g, "g", {});
      box(w, K, 690 + i * 165, 420, 150, 60, s, { stroke: K.color.moss, ink: K.color.moss, size: 22 });
      return w;
    });
    const foot = K.text(g, 640, 680, "Same LoRA inside both.", { "font-size": 28, "text-anchor": "middle", "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(mlx, P(K, t, d, 0.04), 14);
      K.show(uni, P(K, t, d, 0.3), 12);
      K.show(peft, P(K, t, d, 0.6), 14);
      where.forEach((w, i) => K.show(w, P(K, t, d, 0.8 + i * 0.03)));
      K.show(foot, P(K, t, d, 0.9));
    };
  };

  // ---------------------------------------------------------------- s03
  S.chat_data = (g, K) => {
    const h = heading(g, K, "The data, as chat records");
    const rec = K.el(g, "g", { transform: "translate(80 140)" });
    K.el(rec, "rect", { width: 1120, height: 300, rx: 14, fill: K.color.paper, stroke: K.color.line, "stroke-width": 2 });
    K.text(rec, 24, 48, '{"messages": [', { "font-size": 24, "font-family": K.color.mono });
    const msgs = [
      ["system", "Rewrite the user's text in plain English. Reply with the rewrite only.", K.color.muted],
      ["user", "As a value-add, all orders over $75 now qualify for free next-day ...", K.color.cobalt],
      ["assistant", "All orders over $75 get free next-day shipping ...", K.color.moss],
    ].map(([role, text, color], i) => {
      const m = K.el(rec, "g", { transform: `translate(48 ${96 + i * 62})` });
      K.el(m, "rect", { x: 0, y: -30, width: 150, height: 42, rx: 21, fill: color });
      K.text(m, 75, -1, role, { "font-size": 20, "text-anchor": "middle", fill: K.color.surface, "font-weight": 700, "font-family": K.color.mono });
      K.text(m, 168, 0, text, { "font-size": 21 });
      return m;
    });
    K.text(rec, 24, 280, "]}", { "font-size": 24, "font-family": K.color.mono });
    // the file: one record per line
    const file = K.el(g, "g", { transform: "translate(80 480)" });
    K.text(file, 0, 0, "train.jsonl: one record per line", { "font-size": 24, "font-weight": 600, "font-family": K.color.mono });
    const lines = [0, 1, 2, 3].map((i) => {
      const l = K.el(file, "g", {});
      K.text(l, 0, 44 + i * 36, String(i + 1), { "font-size": 18, fill: K.color.muted, "font-family": K.color.mono });
      K.el(l, "rect", { x: 40, y: 28 + i * 36, width: 820 - (i % 2) * 140 - i * 30, height: 20, rx: 5, fill: K.color.line });
      return l;
    });
    const tag = K.text(g, 1000, 590, "JSON lines", { "font-size": 30, "text-anchor": "middle", "font-weight": 700, fill: K.color.cobalt });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(rec, P(K, t, d, 0.12), 12);
      [0.3, 0.45, 0.55].forEach((a, i) => K.show(msgs[i], P(K, t, d, a), 10));
      K.show(file, P(K, t, d, 0.72));
      lines.forEach((l, i) => K.show(l, P(K, t, d, 0.74 + i * 0.04, 0.04)));
      K.show(tag, P(K, t, d, 0.88));
    };
  };

  // ---------------------------------------------------------------- s04
  S.mlx_config = (g, K) => {
    const h = heading(g, K, "mlx-lm: give it the scale");
    const cfg = codeCard(g, K, 80, 140, 560, "the adapter's config", [
      ["num_layers: 28", "all 28 blocks"],
      ["lora_parameters:", ""],
      ["  rank: 8", ""],
      ["  scale: 2.0", "alpha / r = 16 / 8"],
      ["  keys:", ""],
      ["    self_attn.q_proj", ""],
      ["    ...k_proj, v_proj, o_proj", ""],
    ], K.color.cobalt, { size: 21, lh: 40 });
    // inside: mlx's LoRALinear, A and B turned on their side
    const R = K.el(g, "g", {});
    K.text(R, 700, 170, "Inside its LoRALinear", { "font-size": 26, "font-weight": 600 });
    const a = matrix(R, K, 720, 210, 22, 220, "lora_a", "1536 × 8", K.color.moss);
    const b = matrix(R, K, 820, 300, 220, 22, "lora_b", "8 × 1536", K.color.moss);
    const yours = K.el(g, "g", {});
    K.text(yours, 700, 560, "Your A was 8 × 1536 and B was 1536 × 8:", { "font-size": 22, fill: K.color.muted });
    K.text(yours, 700, 592, "the same numbers, turned on their side.", { "font-size": 22, fill: K.color.muted });
    const zero = K.text(g, 820, 450, "starts at zero, like yours", { "font-size": 22, "font-weight": 600, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(cfg.c, P(K, t, d, 0.04), 12);
      cfg.rows.forEach((r, i) => K.show(r, P(K, t, d, [0.5, 0.1, 0.12, 0.18, 0.36, 0.38, 0.4][i])));
      K.show(R, P(K, t, d, 0.6), 12);
      K.show(yours, P(K, t, d, 0.72));
      const z = P(K, t, d, 0.86) > 0.5;
      b.rect.setAttribute("fill", z ? K.color.paper : K.color.moss);
      b.rect.setAttribute("stroke", K.color.moss);
      b.rect.setAttribute("stroke-width", 2);
      K.show(zero, P(K, t, d, 0.86));
      a.rect.style.opacity = 1;
    };
  };

  // ---------------------------------------------------------------- s05
  S.peft_config = (g, K) => {
    const h = heading(g, K, "PEFT: give it alpha");
    const cfg = codeCard(g, K, 80, 140, 560, "LoraConfig", [
      ["r=8,", ""],
      ["lora_alpha=16,", "not the scale"],
      ['target_modules=["q_proj",', ""],
      ['  "k_proj", "v_proj", "o_proj"]', ""],
    ], K.color.moss, { size: 21, lh: 40 });
    const div = K.text(g, 80, 440, "PEFT works out the scale itself:", { "font-size": 24, fill: K.color.muted });
    const sc = K.text(g, 80, 490, "scaling = 16 / 8 = 2.0", { "font-size": 34, "font-weight": 700, "font-family": K.color.mono, fill: K.color.moss });
    const R = K.el(g, "g", {});
    K.text(R, 700, 170, "Inside each layer", { "font-size": 26, "font-weight": 600 });
    matrix(R, K, 720, 230, 220, 22, "lora_A", "8 × 1536", K.color.moss);
    matrix(R, K, 1040, 230, 22, 190, "lora_B", "1536 × 8", K.color.moss);
    K.text(R, 720, 380, "the same way round", { "font-size": 22, fill: K.color.muted });
    K.text(R, 720, 408, "as your A and B", { "font-size": 22, fill: K.color.muted });
    const f32 = K.el(g, "g", {});
    box(f32, K, 700, 520, 500, 64, "A and B kept in float32", { stroke: K.color.amber, ink: K.color.amber, size: 24 });
    K.text(f32, 700, 620, "while the base model's weights stay in bf16", { "font-size": 20, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(cfg.c, P(K, t, d, 0.04), 12);
      cfg.rows.forEach((r, i) => K.show(r, P(K, t, d, [0.1, 0.16, 0.48, 0.5][i])));
      K.show(div, P(K, t, d, 0.26));
      K.show(sc, P(K, t, d, 0.32), 10);
      K.show(R, P(K, t, d, 0.58), 12);
      K.show(f32, P(K, t, d, 0.84), 10);
    };
  };

  // ---------------------------------------------------------------- s06
  S.same_adapter = (g, K) => {
    const h = heading(g, K, "The same adapter in both");
    const sub = K.text(g, 80, 150, "Per block, rank 8 times (inputs + outputs) for each projection:", { "font-size": 24, fill: K.color.muted });
    const parts = [
      ["q", "1536 + 1536", "3072"], ["k", "1536 + 256", "1792"], ["v", "1536 + 256", "1792"], ["o", "1536 + 1536", "3072"],
    ].map(([n, sum, tot], i) => {
      const p = K.el(g, "g", { transform: `translate(${80 + i * 285} 190)` });
      K.el(p, "rect", { width: 265, height: 110, rx: 12, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
      K.text(p, 20, 42, `${n}_proj`, { "font-size": 24, "font-weight": 700, fill: K.color.moss, "font-family": K.color.mono });
      K.text(p, 20, 84, `${sum} = ${tot}`, { "font-size": 22, "font-family": K.color.mono });
      return p;
    });
    const formula = K.text(g, 640, 380, "28 × 8 × (3072 + 1792 + 1792 + 3072)", { "font-size": 34, "text-anchor": "middle", "font-family": K.color.mono });
    const total = K.text(g, 640, 450, "= 2,179,072 trainable numbers", { "font-size": 42, "text-anchor": "middle", "font-weight": 700, fill: K.color.moss });
    const bar = K.el(g, "g", {});
    K.el(bar, "rect", { x: 80, y: 520, width: 1120, height: 44, rx: 6, fill: K.color.line });
    K.el(bar, "rect", { x: 80, y: 520, width: 3, height: 44, fill: K.color.moss });
    K.text(bar, 80, 600, "← 0.141% of the model's 1.55 billion: the only part that trains", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.04));
      parts.forEach((p, i) => K.show(p, P(K, t, d, 0.1 + i * 0.05), 12));
      K.show(formula, P(K, t, d, 0.32));
      K.show(total, P(K, t, d, 0.42), 12);
      K.show(bar, P(K, t, d, 0.7));
    };
  };

  // ---------------------------------------------------------------- s07
  S.race = (g, K) => {
    const h = heading(g, K, "A fair race: 100 steps each");
    const rules = ["same batches", "same order", "same loss", "100 steps"].map((s, i) => {
      const r = K.el(g, "g", {});
      box(r, K, 80 + i * 285, 130, 265, 50, s, { size: 22 });
      return r;
    });
    const head = K.el(g, "g", {});
    K.text(head, 640, 240, "mlx-lm", { "font-size": 28, "font-weight": 700, fill: K.color.cobalt, "font-family": K.color.mono });
    K.text(head, 960, 240, "PEFT", { "font-size": 28, "font-weight": 700, fill: K.color.moss, "font-family": K.color.mono });
    const rows = [
      ["held-out loss", "1.159 → 0.682", "1.163 → 0.685"],
      ["training time", "94 s", "86 s"],
      ["chrF vs reference rewrites", "54.8 → 63.8", "55.1 → 61.9"],
      ["ordinary questions right", "10 / 10", "10 / 10"],
    ].map(([label, a, b], i) => {
      const y = 300 + i * 82;
      const r = K.el(g, "g", {});
      K.el(r, "line", { x1: 80, y1: y + 26, x2: 1200, y2: y + 26, stroke: K.color.line, "stroke-width": 1 });
      K.text(r, 80, y, label, { "font-size": 24 });
      K.text(r, 640, y, a, { "font-size": 28, "font-weight": 600, "font-family": K.color.mono });
      K.text(r, 960, y, b, { "font-size": 28, "font-weight": 600, "font-family": K.color.mono });
      return r;
    });
    const note = K.text(g, 80, 650, "chrF: how many characters match a reference rewrite, out of 100", { "font-size": 20, fill: K.color.muted });
    const foot = K.text(g, 80, 690, "Same result, from two different tools.", { "font-size": 26, "font-weight": 600, fill: K.color.moss });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      rules.forEach((r, i) => K.show(r, P(K, t, d, 0.04 + i * 0.05), 10));
      K.show(head, P(K, t, d, 0.26));
      [0.3, 0.5, 0.66, 0.86].forEach((a, i) => K.show(rows[i], P(K, t, d, a), 10));
      K.show(note, P(K, t, d, 0.68));
      K.show(foot, P(K, t, d, 0.93));
    };
  };

  // ---------------------------------------------------------------- s08
  S.tokens = (g, K) => {
    const h = heading(g, K, "Tokens per second: count the real ones");
    // a batch padded to its longest row; grey cells are padding (the mask is 0 there)
    const lens = [16, 11, 14, 8, 13];
    const COLS = 16;
    const C = 30;
    const batch = K.el(g, "g", { transform: "translate(80 160)" });
    K.text(batch, 0, -14, "one batch, padded to its longest row", { "font-size": 22, fill: K.color.muted });
    const pads = [];
    lens.forEach((n, r) => {
      for (let c = 0; c < COLS; c++) {
        const cell = K.el(batch, "rect", { x: c * C + 1, y: r * C + 8, width: C - 2, height: C - 2, rx: 3, fill: c < n ? K.color.cobalt : K.color.line });
        if (c >= n) pads.push(cell);
      }
    });
    const legend = K.el(g, "g", { transform: "translate(80 360)" });
    K.el(legend, "rect", { x: 0, y: -18, width: 22, height: 22, rx: 3, fill: K.color.cobalt });
    K.text(legend, 32, 0, "real token", { "font-size": 22 });
    K.el(legend, "rect", { x: 170, y: -18, width: 22, height: 22, rx: 3, fill: K.color.line });
    K.text(legend, 202, 0, "padding", { "font-size": 22 });
    const stat = K.el(g, "g", {});
    K.text(stat, 640, 200, "In 20 sample batches:", { "font-size": 24, fill: K.color.muted });
    K.text(stat, 640, 260, "27% of positions were padding", { "font-size": 34, "font-weight": 700, fill: K.color.amber });
    K.text(stat, 640, 300, "padded speed looks about 35% higher", { "font-size": 24 });
    K.text(stat, 640, 336, "but the model learns nothing from padding", { "font-size": 24, fill: K.color.muted });
    // the full mlx-lm run, two ways
    const W = 560 / 824;
    const bars = [
      ["real tokens only", 605, K.color.moss],
      ["counting padding too", 824, K.color.line],
    ].map(([label, v, color], i) => {
      const y = 450 + i * 90;
      const r = K.el(g, "g", {});
      K.text(r, 80, y + 34, label, { "font-size": 24, "font-weight": i ? 400 : 700 });
      const bar = K.el(r, "rect", { x: 380, y, width: 0, height: 50, rx: 6, fill: color });
      const val = K.text(r, 380, y + 36, `${v} tokens/s`, { "font-size": 26, "font-weight": 700, fill: i ? K.color.muted : K.color.moss });
      return { r, bar, val, v };
    });
    const runLbl = K.text(g, 80, 430, "The full mlx-lm run:", { "font-size": 22, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(batch, P(K, t, d, 0.04));
      K.show(legend, P(K, t, d, 0.08));
      const hl = P(K, t, d, 0.3) - P(K, t, d, 0.62);
      pads.forEach((c) => c.setAttribute("fill", hl > 0.5 ? K.color.amber : K.color.line));
      K.show(stat, P(K, t, d, 0.3), 12);
      K.show(runLbl, P(K, t, d, 0.68));
      bars.forEach((b, i) => {
        const p = P(K, t, d, [0.72, 0.84][i], 0.08);
        K.show(b.r, Math.min(1, p * 3));
        const w = b.v * W * p;
        b.bar.setAttribute("width", w);
        b.val.setAttribute("x", 380 + w + 14);
      });
    };
  };

  // ---------------------------------------------------------------- s09
  S.memory = (g, K) => {
    const h = heading(g, K, "Memory: framework and sequence length");
    // left: peak memory of the full runs, the same measure (macOS footprint) for both
    const L = K.el(g, "g", {});
    K.text(L, 80, 160, "Peak memory, same adapter", { "font-size": 24, "font-weight": 600 });
    K.text(L, 80, 190, "measured the same way (macOS footprint)", { "font-size": 20, fill: K.color.muted });
    const BASE = 560;
    const sc = 300 / 13;
    const peak = [
      ["mlx-lm", 9.3, K.color.cobalt, 110],
      ["PEFT", 12.8, K.color.moss, 290],
    ].map(([label, v, color, x]) => {
      const r = K.el(L, "rect", { x, width: 130, rx: 6, fill: color });
      const val = K.text(L, x + 65, 0, `${v} GB`, { "font-size": 28, "text-anchor": "middle", "font-weight": 700 });
      K.text(L, x + 65, BASE + 32, label, { "font-size": 22, "text-anchor": "middle", "font-family": K.color.mono });
      return { r, val, v };
    });
    K.el(L, "line", { x1: 90, y1: BASE, x2: 440, y2: BASE, stroke: K.color.line, "stroke-width": 2 });
    const why = K.el(g, "g", {});
    K.text(why, 80, 632, "PyTorch keeps freed memory cached,", { "font-size": 20, fill: K.color.muted });
    K.text(why, 80, 658, "and the two use different kernels", { "font-size": 20, fill: K.color.muted });
    // right: one training step on 2 sequences, memory above the 3.1 GB of weights
    const R = K.el(g, "g", {});
    K.text(R, 620, 160, "One step on 2 sequences", { "font-size": 24, "font-weight": 600 });
    K.text(R, 620, 190, "memory above the 3.1 GB of weights", { "font-size": 20, fill: K.color.muted });
    const seq = [[64, 0.8], [128, 1.2], [256, 2.5], [512, 5.0]];
    const SW = 320 / 5;
    const sbars = seq.map(([n, v], i) => {
      const y = 230 + i * 80;
      const r = K.el(R, "g", {});
      K.text(r, 620, y + 34, `${n} tokens`, { "font-size": 22, "font-family": K.color.mono });
      const bar = K.el(r, "rect", { x: 780, y, width: 0, height: 50, rx: 6, fill: K.color.amber });
      const val = K.text(r, 780, y + 34, `${v.toFixed(1)} GB`, { "font-size": 24, "font-weight": 700 });
      return { r, bar, val, v };
    });
    const act = K.el(g, "g", {});
    K.text(act, 620, 590, "Each layer keeps its in-between results for the", { "font-size": 20 });
    K.text(act, 620, 616, "backward pass, including the logits:", { "font-size": 20 });
    K.text(act, 620, 642, "151,936 scores for every position.", { "font-size": 20, "font-weight": 600, fill: K.color.amber });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(L, P(K, t, d, 0.03));
      peak.forEach((p, i) => {
        const q = P(K, t, d, [0.06, 0.16][i], 0.08);
        const hgt = p.v * sc * q;
        p.r.setAttribute("y", BASE - hgt);
        p.r.setAttribute("height", hgt);
        p.val.setAttribute("y", BASE - hgt - 12);
        p.val.style.opacity = q;
      });
      K.show(why, P(K, t, d, 0.28));
      K.show(R, P(K, t, d, 0.38));
      sbars.forEach((b, i) => {
        const p = P(K, t, d, 0.5 + i * 0.07, 0.07);
        K.show(b.r, Math.min(1, p * 3));
        const w = b.v * SW * p;
        b.bar.setAttribute("width", w);
        b.val.setAttribute("x", 780 + w + 12);
      });
      K.show(act, P(K, t, d, 0.84));
    };
  };

  // ---------------------------------------------------------------- s10
  S.save_load = (g, K) => {
    const h = heading(g, K, "Saving and loading an adapter");
    const sizes = K.el(g, "g", {});
    K.el(sizes, "rect", { x: 80, y: 140, width: 700, height: 36, rx: 6, fill: K.color.cobalt });
    K.text(sizes, 80, 206, "the model: about 3,100 MB", { "font-size": 22, fill: K.color.muted });
    K.el(sizes, "rect", { x: 80, y: 226, width: 3, height: 36, fill: K.color.moss });
    K.text(sizes, 96, 252, "the adapter: 8.7 MB", { "font-size": 24, fill: K.color.moss, "font-weight": 700 });
    const folder = K.el(g, "g", { transform: "translate(840 140)" });
    K.el(folder, "rect", { width: 360, height: 124, rx: 12, fill: K.color.paper, stroke: K.color.line, "stroke-width": 2 });
    K.text(folder, 20, 34, "adapter folder/", { "font-size": 20, fill: K.color.muted, "font-family": K.color.mono });
    K.text(folder, 20, 70, "the A and B weights", { "font-size": 20, "font-family": K.color.mono });
    K.text(folder, 20, 104, "adapter_config.json", { "font-size": 20, "font-family": K.color.mono, fill: K.color.cobalt });
    const load = codeCard(g, K, 80, 300, 1120, "loading it back onto the base model", [
      ["load(MODEL, adapter_path=folder)", "mlx-lm"],
      ["PeftModel.from_pretrained(base, folder)", "PEFT"],
    ], K.color.cobalt, { size: 22, lh: 40 });
    const same = K.text(g, 80, 490, "Reloaded replies: identical on 60 of 60 prompts", { "font-size": 26, "font-weight": 600, fill: K.color.moss });
    const keep = K.el(g, "g", {});
    box(keep, K, 80, 540, 540, 110, "", { stroke: K.color.moss });
    K.text(keep, 104, 584, "Keep it separate", { "font-size": 24, "font-weight": 700, fill: K.color.moss });
    K.text(keep, 104, 622, "one base, many adapters, swap per job", { "font-size": 22 });
    const merge = K.el(g, "g", {});
    box(merge, K, 660, 540, 540, 110, "", { stroke: K.color.cobalt });
    K.text(merge, 684, 584, "Or merge it in", { "font-size": 24, "font-weight": 700, fill: K.color.cobalt });
    K.text(merge, 684, 622, "mlx_lm.fuse / merge_and_unload", { "font-size": 20, "font-family": K.color.mono });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sizes, P(K, t, d, 0.04));
      K.show(folder, P(K, t, d, 0.22), 10);
      K.show(load.c, P(K, t, d, 0.36), 10);
      load.rows.forEach((r, i) => K.show(r, P(K, t, d, 0.38 + i * 0.06)));
      K.show(same, P(K, t, d, 0.52));
      K.show(keep, P(K, t, d, 0.7), 10);
      K.show(merge, P(K, t, d, 0.86), 10);
    };
  };

  // ---------------------------------------------------------------- s11
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      "1. to_chat_record",
      "2. mlx_lora_config",
      "3. peft_lora_config",
      "4. tokens_per_sec",
      "5. load_peft_adapter",
      "6. load_mlx_adapter",
      "six questions in answers.toml",
      "train.py: both tools, ~8 minutes",
      "make check M=07",
    ];
    const rows = items.map((s, i) => {
      const left = i < 6;
      const r = K.el(g, "g", { transform: `translate(${left ? 80 : 640} ${left ? 160 + i * 64 : 160 + (i - 6) * 64})` });
      const bx = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, s, { "font-size": 24, "font-family": K.color.mono });
      return { r, bx, tick };
    });
    const first = K.text(g, 640, 400, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    const real = K.text(g, 640, 442, "It runs both tools, side by side.", { "font-size": 24, fill: K.color.muted });
    const next = K.text(g, 80, 620, "Next, in M8: QLoRA, a 7B model in 4 bits.", { "font-size": 30, "font-weight": 600 });
    const starts = [0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.42, 0.52, 0.66];
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(first, P(K, t, d, 0.04));
      K.show(real, P(K, t, d, 0.06));
      rows.forEach((row, i) => {
        K.show(row.r, P(K, t, d, starts[i]));
        const done = P(K, t, d, starts[i] + 0.04, 0.03);
        row.tick.style.opacity = done;
        row.bx.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(next, P(K, t, d, 0.8));
    };
  };
})();
