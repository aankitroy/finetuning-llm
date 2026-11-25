// M3's video: one drawing per scene in script.toml. Every id, count and string comes from the
// lesson's own cells, run against the SmolLM2-135M-Instruct tokenizer.
// Each scene is (g, K) => (t, d) => { ... }; see dashboard/video.js for the kit K.

"use strict";

(() => {
  const S = window.VIDEO_SCENES;

  // progress (0 to 1) of something that starts at fraction `a` of the scene and lasts `len`
  const P = (K, t, d, a, len = 0.06) => K.at(t, a * d, len * d);

  function heading(g, K, str) {
    return K.text(g, 80, 96, str, { "font-size": 44, "font-weight": 600 });
  }

  // a mono line built from coloured pieces: [[text, colour, bold?], ...]
  function segLine(g, K, x, y, segs, size = 24) {
    const node = K.text(g, x, y, "", { "font-size": size, "font-family": K.color.mono, "xml:space": "preserve" });
    node.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
    return segs.map(([str, fill, bold]) =>
      K.el(node, "tspan", { fill: fill || K.color.ink, "font-weight": bold ? 700 : 400 }, str),
    );
  }

  // the player's stylesheet sets one font for all stage text, which beats the font-family
  // attribute; an inline style beats the stylesheet, so mono text gets one
  function monoFix(g, K) {
    g.querySelectorAll("text").forEach((n) => {
      if (n.getAttribute("font-family") === K.color.mono) n.style.fontFamily = K.color.mono;
    });
  }
  const before = new Set(Object.keys(S));

  function card(g, K, x, y, w, h, stroke) {
    const c = K.el(g, "g", { transform: `translate(${x} ${y})` });
    K.el(c, "rect", { width: w, height: h, rx: 14, fill: K.color.paper, stroke: stroke || K.color.line, "stroke-width": 2 });
    return c;
  }

  // ---------------------------------------------------------------- s01
  S.intro = (g, K) => {
    const h = heading(g, K, "What the model sees: numbers");
    const sentence = K.text(g, 640, 190, "Hello world, hello!", { "font-size": 40, "text-anchor": "middle", "font-family": K.color.mono });
    const pieces = ["\"Hello\"", "\" world\"", "\",\"", "\" hello\"", "\"!\""];
    const ids = [19556, 905, 28, 33662, 17];
    const widths = [150, 160, 90, 160, 90];
    const gap = 24;
    let x = (1280 - widths.reduce((a, b) => a + b, 0) - gap * 4) / 2;
    const toks = pieces.map((p, i) => {
      const tg = K.el(g, "g", {});
      K.tile(tg, x, 226, widths[i], 56, p, { size: 24, fill: K.color.cobaltSoft });
      K.text(tg, x + widths[i] / 2, 322, String(ids[i]), { "font-size": 26, "text-anchor": "middle", fill: K.color.cobalt, "font-weight": 600, "font-family": K.color.mono });
      x += widths[i] + gap;
      return tg;
    });
    const lblTok = K.text(g, 80, 262, "tokens", { "font-size": 22, fill: K.color.muted });
    const lblId = K.text(g, 80, 320, "token ids", { "font-size": 22, fill: K.color.muted });
    const vocab = K.text(g, 640, 382, "Each id is a token's place in the vocabulary: SmolLM2's has 49,152.", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });

    const cards = [
      { x: 80, title: "One per character", lines: ["Every text becomes", "very long."], color: K.color.rose },
      { x: 460, title: "One per word", lines: ["Millions of tokens, and", "new words get none."], color: K.color.rose },
      { x: 840, title: "Real tokenizers", lines: ["Common words: one token.", "Rare words: a few pieces."], color: K.color.moss },
    ].map((c) => {
      const cg = card(g, K, c.x, 430, 360, 190, c.color);
      K.text(cg, 24, 52, c.title, { "font-size": 30, "font-weight": 600, fill: c.color });
      c.lines.forEach((l, i) => K.text(cg, 24, 104 + i * 36, l, { "font-size": 24 }));
      return cg;
    });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sentence, P(K, t, d, 0.02));
      K.show(lblTok, P(K, t, d, 0.2));
      toks.forEach((tk, i) => K.show(tk, P(K, t, d, 0.1 + i * 0.03), 12));
      K.show(lblId, P(K, t, d, 0.12));
      K.show(vocab, P(K, t, d, 0.36));
      [0.62, 0.76, 0.86].forEach((a, i) => K.show(cards[i], P(K, t, d, a), 16));
    };
  };

  // ---------------------------------------------------------------- s02
  S.bytes = (g, K) => {
    const h = heading(g, K, "Any text becomes bytes, with UTF-8");
    const sub = K.text(g, 80, 150, "A byte is a number from 0 to 255.", { "font-size": 28, fill: K.color.muted });
    const rows = [
      { ch: "A", b: [65], a: 0.38 },
      { ch: "é", b: [195, 169], a: 0.44 },
      { ch: "न", b: [224, 164, 168], a: 0.5 },
      { ch: "🙂", b: [240, 159, 153, 130], a: 0.55 },
    ].map((r, i) => {
      const y = 200 + i * 86;
      const rg = K.el(g, "g", {});
      K.text(rg, 140, y + 46, r.ch, { "font-size": 48, "text-anchor": "middle" });
      K.text(rg, 220, y + 42, "→", { "font-size": 36, fill: K.color.muted });
      r.b.forEach((v, j) => K.tile(rg, 290 + j * 120, y, 104, 60, String(v), { size: 26, fill: K.color.cobaltSoft }));
      K.text(rg, 790, y + 42, `${r.b.length} byte${r.b.length > 1 ? "s" : ""}`, { "font-size": 28, "font-weight": 600, fill: K.color.cobalt });
      return { rg, a: r.a };
    });
    const utf = K.text(g, 1200, 150, "UTF-8: the standard recipe", { "font-size": 24, "text-anchor": "end", fill: K.color.cobalt, "font-weight": 600 });
    const only = K.el(g, "g", { transform: "translate(80 570)" });
    K.el(only, "rect", { width: 1120, height: 84, rx: 14, fill: K.color.paper, stroke: K.color.moss, "stroke-width": 2 });
    K.text(only, 560, 53, "Only 256 different bytes, so nothing is ever unknown.", { "font-size": 30, "text-anchor": "middle", fill: K.color.moss, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(sub, P(K, t, d, 0.06));
      K.show(utf, P(K, t, d, 0.25));
      rows.forEach((r) => K.show(r.rg, P(K, t, d, r.a), 12));
      K.show(only, P(K, t, d, 0.68), 16);
    };
  };

  // ---------------------------------------------------------------- s03
  const LETTERS = "aaabdaaabac".split("");
  const CODE = { a: 97, b: 98, c: 99, d: 100 };

  S.pairs = (g, K) => {
    const h = heading(g, K, "BPE: count the pairs, merge the top one");
    const TW = 80;
    const TG = 12;
    const X0 = (1280 - LETTERS.length * TW - (LETTERS.length - 1) * TG) / 2;
    const tiles = LETTERS.map((ch, i) => {
      const tg = K.el(g, "g", {});
      const box = K.el(tg, "rect", { x: X0 + i * (TW + TG), y: 140, width: TW, height: 60, rx: 8, fill: K.color.surface, stroke: K.color.line, "stroke-width": 2 });
      K.text(tg, X0 + i * (TW + TG) + TW / 2, 182, ch, { "font-size": 32, "text-anchor": "middle", "font-family": K.color.mono });
      K.text(tg, X0 + i * (TW + TG) + TW / 2, 236, String(CODE[ch]), { "font-size": 24, "text-anchor": "middle", fill: K.color.cobalt, "font-family": K.color.mono });
      return { tg, box };
    });
    const count11 = K.text(g, 640, 280, "11 letters, so 11 ids", { "font-size": 24, "text-anchor": "middle", fill: K.color.muted });
    // the sliding window that looks at each neighbouring pair
    const win = K.el(g, "rect", { y: 132, width: 2 * TW + TG + 16, height: 76, rx: 12, fill: "none", stroke: K.color.amber, "stroke-width": 4 });

    const counts = [["aa", "(97, 97)", 4], ["ab", "(97, 98)", 2], ["bd", "(98, 100)", 1], ["da", "(100, 97)", 1], ["ba", "(98, 97)", 1], ["ac", "(97, 99)", 1]];
    const table = K.el(g, "g", { transform: "translate(80 320)" });
    K.text(table, 0, 0, "pair counts", { "font-size": 22, fill: K.color.muted });
    const bars = counts.map(([s, p, n], i) => {
      const y = 24 + i * 40;
      const top = i === 0;
      K.text(table, 0, y + 24, s, { "font-size": 24, "font-family": K.color.mono, "font-weight": top ? 700 : 400 });
      K.text(table, 50, y + 24, p, { "font-size": 22, "font-family": K.color.mono, fill: K.color.muted });
      const bar = K.el(table, "rect", { x: 200, y: y + 4, width: n * 60, height: 26, rx: 4, fill: top ? K.color.amber : K.color.line });
      K.text(table, 200 + n * 60 + 12, y + 25, String(n), { "font-size": 24, "font-weight": top ? 700 : 400 });
      return bar;
    });

    const merge = K.el(g, "g", { transform: "translate(620 320)" });
    K.text(merge, 0, 0, "one merge", { "font-size": 22, fill: K.color.muted });
    K.tile(merge, 0, 24, 120, 60, "aa", { size: 26 });
    K.text(merge, 140, 66, "→", { "font-size": 36, fill: K.color.muted });
    K.tile(merge, 190, 24, 120, 60, "256", { size: 28, fill: K.color.amber, ink: "#fff" });
    K.text(merge, 330, 64, "a new id", { "font-size": 26 });

    const after = ["256", "97", "98", "100", "256", "97", "98", "97", "99"];
    const out = K.el(g, "g", { transform: "translate(620 450)" });
    K.text(out, 0, 0, "replaced everywhere", { "font-size": 22, fill: K.color.muted });
    after.forEach((s, i) => K.tile(out, i * 64, 24, 58, 52, s, { size: 20, fill: s === "256" ? K.color.amber : K.color.surface, ink: s === "256" ? "#fff" : K.color.ink }));
    const nine = K.text(g, 640, 640, "11 ids become 9", { "font-size": 40, "text-anchor": "middle", "font-weight": 600, fill: K.color.cobalt });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      tiles.forEach((tl, i) => K.show(tl.tg, P(K, t, d, 0.16 + i * 0.012, 0.04)));
      K.show(count11, P(K, t, d, 0.26));
      // the window sweeps over the 10 pairs while we count
      const sweep = P(K, t, d, 0.33, 0.12);
      const k = Math.min(9, Math.floor(sweep * 10));
      win.setAttribute("x", X0 + k * (TW + TG) - 8);
      win.style.opacity = sweep > 0 && sweep < 1 ? 1 : 0;
      K.show(table, P(K, t, d, 0.33));
      bars.forEach((b, i) => (b.style.opacity = P(K, t, d, 0.36 + i * 0.012, 0.03)));
      // the four "aa" pairs light up
      const hi = P(K, t, d, 0.46) > 0.5;
      [0, 1, 5, 6].forEach((i) => tiles[i].box.setAttribute("stroke", hi ? K.color.amber : K.color.line));
      [0, 1, 5, 6].forEach((i) => tiles[i].box.setAttribute("stroke-width", hi ? 4 : 2));
      K.show(merge, P(K, t, d, 0.6), 12);
      K.show(out, P(K, t, d, 0.74), 12);
      K.show(nine, P(K, t, d, 0.88));
    };
  };

  // ---------------------------------------------------------------- s04
  S.rounds = (g, K) => {
    const h = heading(g, K, "Repeat, and the text gets shorter");
    const NEW = { 256: K.color.amber, 257: K.color.cobalt, 258: K.color.moss };
    const rows = [
      { label: "start", note: "", ids: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99], a: 0.0 },
      { label: "round 1", note: "aa → 256", ids: [256, 97, 98, 100, 256, 97, 98, 97, 99], a: 0.04 },
      { label: "round 2", note: "aaa → 257", ids: [257, 98, 100, 257, 98, 97, 99], a: 0.17 },
      { label: "round 3", note: "aaab → 258", ids: [258, 100, 258, 97, 99], a: 0.31 },
    ].map((r, i) => {
      const y = 150 + i * 106;
      const rg = K.el(g, "g", {});
      K.text(rg, 80, y + 30, r.label, { "font-size": 26, "font-weight": 600 });
      if (r.note) K.text(rg, 80, y + 62, r.note, { "font-size": 22, "font-family": K.color.mono, fill: K.color.muted });
      r.ids.forEach((v, j) => {
        const c = NEW[v];
        K.tile(rg, 300 + j * 72, y, 66, 56, String(v), { size: 22, fill: c || K.color.surface, ink: c ? "#fff" : K.color.ink });
      });
      const n = K.text(rg, 1200, y + 38, `${r.ids.length} ids`, { "font-size": 28, "text-anchor": "end", "font-weight": 600, fill: K.color.cobalt });
      return { rg, n, a: r.a };
    });
    const tok = K.el(g, "g", { transform: "translate(80 580)" });
    K.text(tok, 0, 0, "The whole tokenizer is these merges:", { "font-size": 26, "font-weight": 600 });
    K.text(tok, 0, 44, "{(97, 97): 256, (256, 97): 257, (257, 98): 258}", { "font-size": 26, "font-family": K.color.mono, fill: K.color.cobalt });
    const real = K.text(g, 80, 680, "Real models learn tens of thousands of merges, from huge amounts of text.", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      rows.forEach((r) => K.show(r.rg, P(K, t, d, r.a), 12));
      // the counts flash in turn as the narration counts down 11, 9, 7, 5
      rows.forEach((r, i) => r.n.setAttribute("fill", P(K, t, d, 0.42 + i * 0.04, 0.02) > 0.5 && P(K, t, d, 0.46 + i * 0.04, 0.02) < 0.5 ? K.color.amber : K.color.cobalt));
      K.show(tok, P(K, t, d, 0.6), 12);
      K.show(real, P(K, t, d, 0.76));
    };
  };

  // ---------------------------------------------------------------- s05
  S.encode_decode = (g, K) => {
    const h = heading(g, K, "Decode, and encode");
    const divider = K.el(g, "line", { x1: 640, y1: 140, x2: 640, y2: 680, stroke: K.color.line, "stroke-width": 2 });

    // left: decoding
    const dec = K.el(g, "g", { transform: "translate(80 150)" });
    K.text(dec, 0, 20, "Decode: ids → bytes → text", { "font-size": 28, "font-weight": 600, fill: K.color.cobalt });
    const vocab = K.el(dec, "g", {});
    [["0 to 255", "one byte each"], ["256", "aa"], ["257", "aaa"], ["258", "aaab"]].forEach(([id, b], i) => {
      K.text(vocab, 0, 80 + i * 44, id, { "font-size": 24, "font-family": K.color.mono });
      K.text(vocab, 140, 80 + i * 44, "→", { "font-size": 24, fill: K.color.muted });
      K.text(vocab, 180, 80 + i * 44, b, { "font-size": 24, "font-family": i ? K.color.mono : K.color.font, fill: i ? K.color.ink : K.color.muted });
    });
    const toks = K.el(dec, "g", {});
    [["258", "aaab"], ["100", "d"], ["258", "aaab"], ["97", "a"], ["99", "c"]].forEach(([id, s], i) => {
      const x = [0, 110, 190, 300, 380][i];
      const w = [100, 70, 100, 70, 70][i];
      K.tile(toks, x, 270, w, 50, id, { size: 22, fill: K.color.cobaltSoft });
      K.text(toks, x + w / 2, 356, s, { "font-size": 24, "text-anchor": "middle", "font-family": K.color.mono });
    });
    const back = K.text(dec, 0, 430, "aaabdaaabac", { "font-size": 34, "font-family": K.color.mono, fill: K.color.moss, "font-weight": 700 });
    const backNote = K.text(dec, 0, 470, "exactly the text we started with", { "font-size": 22, fill: K.color.muted });

    // right: encoding
    const enc = K.el(g, "g", { transform: "translate(690 150)" });
    K.text(enc, 0, 20, "Encode: order matters", { "font-size": 28, "font-weight": 600, fill: K.color.cobalt });
    const learned = K.el(enc, "g", {});
    K.text(learned, 0, 80, "learned 1st:  bc → 256", { "font-size": 24, "font-family": K.color.mono });
    K.text(learned, 0, 124, "learned 2nd:  ab → 257", { "font-size": 24, "font-family": K.color.mono });
    const word = K.text(enc, 0, 196, "encode  abc", { "font-size": 28, "font-family": K.color.mono, "font-weight": 600 });
    const right = K.el(enc, "g", {});
    K.tile(right, 0, 230, 70, 54, "a", { size: 26 });
    K.tile(right, 82, 230, 110, 54, "bc", { size: 26, fill: K.color.moss, ink: "#fff" });
    K.text(right, 212, 266, "✓ [97, 256]", { "font-size": 26, fill: K.color.moss, "font-weight": 600, "font-family": K.color.mono });
    K.text(right, 0, 318, "first-learned merge first", { "font-size": 22, fill: K.color.muted });
    const wrong = K.el(enc, "g", {});
    K.tile(wrong, 0, 360, 110, 54, "ab", { size: 26, fill: K.color.rose, ink: "#fff" });
    K.tile(wrong, 122, 360, 70, 54, "c", { size: 26 });
    K.text(wrong, 212, 396, "✗ [257, 99]", { "font-size": 26, fill: K.color.rose, "font-weight": 600, "font-family": K.color.mono });
    K.text(wrong, 0, 448, "leftmost pair first: different tokens", { "font-size": 22, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(divider, P(K, t, d, 0.02));
      K.show(dec, P(K, t, d, 0.02));
      K.show(vocab, P(K, t, d, 0.08));
      K.show(toks, P(K, t, d, 0.22), 12);
      K.show(back, P(K, t, d, 0.36));
      K.show(backNote, P(K, t, d, 0.4));
      K.show(enc, P(K, t, d, 0.5));
      K.show(learned, P(K, t, d, 0.62));
      K.show(word, P(K, t, d, 0.72));
      K.show(right, P(K, t, d, 0.76), 12);
      K.show(wrong, P(K, t, d, 0.84), 12);
    };
  };

  // ---------------------------------------------------------------- s06
  S.real = (g, K) => {
    const h = heading(g, K, "A real tokenizer: SmolLM2");
    const chip = K.el(g, "g", { transform: "translate(80 130)" });
    K.el(chip, "rect", { width: 560, height: 50, rx: 25, fill: K.color.cobaltSoft });
    K.text(chip, 26, 34, "byte-level BPE, 49,152 tokens", { "font-size": 24, fill: K.color.cobalt, "font-weight": 600 });
    const rows = [
      { text: "Hello world, hello!", pieces: ["Hello", "Ġworld", ",", "Ġhello", "!"], note: "Ġ is a space", a: 0.28 },
      { text: "hello", pieces: ["hello"], note: "not the same token as Hello", a: 0.38 },
      { text: "12345", pieces: ["1", "2", "3", "4", "5"], note: "single digits", a: 0.48 },
      { text: "unbelievable", pieces: ["un", "belie", "vable"], note: "built from pieces", a: 0.56 },
    ].map((r, i) => {
      const y = 214 + i * 72;
      const rg = K.el(g, "g", {});
      K.text(rg, 80, y + 34, r.text, { "font-size": 24, "font-family": K.color.mono });
      let x = 410;
      const tiles = r.pieces.map((p) => {
        const w = Math.max(44, p.length * 15 + 24);
        const tl = K.tile(rg, x, y, w, 50, p, { size: 22, fill: K.color.cobaltSoft });
        x += w + 8;
        return { tl, p };
      });
      K.text(rg, 1200, y + 34, r.note, { "font-size": 22, "text-anchor": "end", fill: K.color.muted });
      return { rg, a: r.a, tiles };
    });
    const cmp = K.el(g, "g", { transform: "translate(80 520)" });
    K.el(cmp, "rect", { width: 1120, height: 150, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(cmp, 28, 52, "The bread is ready.", { "font-size": 26 });
    K.text(cmp, 400, 52, "19 bytes → 5 tokens", { "font-size": 26, "font-family": K.color.mono });
    K.text(cmp, 1092, 52, "3.8 bytes per token", { "font-size": 26, "text-anchor": "end", fill: K.color.moss, "font-weight": 600 });
    K.text(cmp, 28, 112, "रोटी तैयार है।", { "font-size": 28 });
    K.text(cmp, 400, 112, "38 bytes → 18 tokens", { "font-size": 26, "font-family": K.color.mono });
    K.text(cmp, 1092, 112, "2.1 bytes per token", { "font-size": 26, "text-anchor": "end", fill: K.color.rose, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(chip, P(K, t, d, 0.1));
      rows.forEach((r) => K.show(r.rg, P(K, t, d, r.a), 12));
      // capitals: Hello and hello light up together
      const cap = P(K, t, d, 0.38) > 0.5;
      const pick = [rows[0].tiles[0], rows[1].tiles[0]];
      pick.forEach((p) => p.tl.firstChild.setAttribute("stroke", cap ? K.color.amber : K.color.line));
      pick.forEach((p) => p.tl.firstChild.setAttribute("stroke-width", cap ? 3 : 1));
      K.show(cmp, P(K, t, d, 0.68), 16);
    };
  };

  // ---------------------------------------------------------------- s07
  S.special = (g, K) => {
    const h = heading(g, K, "Special tokens, padding, attention mask");
    const marks = [
      { tok: "<|im_start|>", txt: "id 1: start of a message", a: 0.12 },
      { tok: "<|im_end|>", txt: "id 2: end of a message (EOS), stops generation", a: 0.28 },
    ].map((m, i) => {
      const mg = K.el(g, "g", {});
      K.tile(mg, 80, 140 + i * 74, 230, 56, m.tok, { size: 24, fill: K.color.cobalt, ink: "#fff" });
      K.text(mg, 334, 177 + i * 74, m.txt, { "font-size": 26 });
      return { mg, a: m.a };
    });
    const never = K.text(g, 80, 312, "Each is one token, and never split.", { "font-size": 22, fill: K.color.muted });

    const batch = K.el(g, "g", { transform: "translate(80 364)" });
    K.text(batch, 0, 0, "a batch", { "font-size": 22, fill: K.color.muted });
    K.text(batch, 0, 40, "input_ids", { "font-size": 24, "font-family": K.color.mono, fill: K.color.cobalt });
    K.text(batch, 450, 40, "attention_mask", { "font-size": 24, "font-family": K.color.mono, fill: K.color.cobalt });
    const texts = ["\"Hi\"", "\"Hello there friend\""];
    const rowIds = [[26843, 2, 2], [19556, 665, 3289]];
    const cells = rowIds.map((ids, r) =>
      ids.map((v, c) => {
        const pad = r === 0 && c > 0;
        const cg = K.el(batch, "g", {});
        const box = K.el(cg, "rect", { width: 120, height: 54, rx: 8, fill: pad ? K.color.paper : K.color.cobaltSoft, stroke: pad ? K.color.muted : "none", "stroke-dasharray": pad ? "5 4" : "" });
        K.text(cg, 60, 36, String(v), { "font-size": 24, "text-anchor": "middle", "font-family": K.color.mono, fill: pad ? K.color.muted : K.color.ink });
        const mg = K.el(batch, "g", {});
        K.el(mg, "rect", { width: 64, height: 54, rx: 8, fill: pad ? K.color.paper : K.color.moss, stroke: pad ? K.color.muted : "none", "stroke-dasharray": pad ? "5 4" : "" });
        K.text(mg, 32, 36, pad ? "0" : "1", { "font-size": 26, "text-anchor": "middle", "font-family": K.color.mono, fill: pad ? K.color.muted : "#fff", "font-weight": 600 });
        return { cg, mg, box, pad, r, c };
      }),
    ).flat();
    const labels = texts.map((s, r) => K.text(batch, 720, 98 + r * 70, s, { "font-size": 24, "font-family": K.color.mono, fill: K.color.muted }));
    const padNote = K.text(batch, 0, 228, "Padding uses id 2, the same token as EOS.", { "font-size": 22, fill: K.color.muted });
    const side = K.text(g, 80, 664, "", { "font-size": 28, "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(never, P(K, t, d, 0.36));
      marks.forEach((m) => K.show(m.mg, P(K, t, d, m.a), 10));
      K.show(batch, P(K, t, d, 0.48));
      labels.forEach((l) => K.show(l, P(K, t, d, 0.48)));
      K.show(padNote, P(K, t, d, 0.56));
      const left = P(K, t, d, 0.84, 0.05);
      cells.forEach((cl) => {
        // with left padding, the "Hi" row moves its real token to the end
        let col = cl.c;
        if (cl.r === 0) col = K.lerp(cl.c, cl.pad ? cl.c - 1 : 2, left);
        const y = 62 + cl.r * 70;
        cl.cg.setAttribute("transform", `translate(${col * 132} ${y})`);
        cl.mg.setAttribute("transform", `translate(${450 + col * 76} ${y})`);
        cl.mg.style.opacity = P(K, t, d, 0.62 + cl.c * 0.03, 0.04);
      });
      side.textContent = left > 0.5 ? "Left padding: every row ends with real text, ready to generate." : "Right padding (the default): fine for training.";
      side.setAttribute("fill", left > 0.5 ? K.color.moss : K.color.ink);
      K.show(side, P(K, t, d, 0.72));
    };
  };

  // ---------------------------------------------------------------- s08
  S.chat = (g, K) => {
    const h = heading(g, K, "Chat templates: messages become one string");
    const msg = K.el(g, "g", {});
    K.text(msg, 80, 150, "you send", { "font-size": 22, fill: K.color.muted });
    K.text(msg, 80, 190, "[{\"role\": \"user\", \"content\": \"Hi!\"}]", { "font-size": 26, "font-family": K.color.mono });
    const tpl = K.text(g, 1200, 190, "the chat template (ChatML) →", { "font-size": 24, "text-anchor": "end", fill: K.color.cobalt, "font-weight": 600 });

    const box = K.el(g, "g", { transform: "translate(80 222)" });
    K.el(box, "rect", { width: 1120, height: 300, rx: 14, fill: K.color.paper, stroke: K.color.line });
    K.text(box, 24, 36, "the model receives", { "font-size": 22, fill: K.color.muted });
    const SP = K.color.cobalt;
    const sysBar = K.el(box, "rect", { x: 10, y: 56, width: 6, height: 92, rx: 3, fill: K.color.amber });
    const genBar = K.el(box, "rect", { x: 10, y: 240, width: 6, height: 42, rx: 3, fill: K.color.moss });
    const L = (y, segs) => {
      const lg = K.el(box, "g", {});
      segLine(lg, K, 30, y, segs, 22);
      return lg;
    };
    const sys = [
      L(86, [["<|im_start|>", SP, 1], ["system"]]),
      L(132, [["You are a helpful AI assistant named SmolLM, trained by Hugging Face"], ["<|im_end|>", SP, 1]]),
    ];
    const user = [L(178, [["<|im_start|>", SP, 1], ["user"]]), L(224, [["Hi!"], ["<|im_end|>", SP, 1]])];
    const gen = L(270, [["<|im_start|>", SP, 1], ["assistant"]]);
    const tags = K.el(g, "g", { transform: "translate(80 556)" });
    const tSys = K.el(tags, "g", {});
    K.el(tSys, "rect", { width: 20, height: 20, rx: 4, fill: K.color.amber });
    K.text(tSys, 32, 18, "a system message you never wrote", { "font-size": 24 });
    const tGen = K.el(tags, "g", { transform: "translate(560 0)" });
    K.el(tGen, "rect", { width: 20, height: 20, rx: 4, fill: K.color.moss });
    K.text(tGen, 32, 18, "the generation prompt: your turn, model", { "font-size": 24 });
    const total = K.text(g, 640, 660, "32 tokens in all", { "font-size": 36, "text-anchor": "middle", "font-weight": 600, fill: K.color.cobalt });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(msg, P(K, t, d, 0.04));
      K.show(tpl, P(K, t, d, 0.16));
      K.show(box, P(K, t, d, 0.2));
      user.forEach((u) => K.show(u, P(K, t, d, 0.26)));
      sys.forEach((s) => K.show(s, P(K, t, d, 0.5)));
      K.show(sysBar, P(K, t, d, 0.5));
      K.show(tSys, P(K, t, d, 0.52));
      K.show(gen, P(K, t, d, 0.78));
      K.show(genBar, P(K, t, d, 0.78));
      K.show(tGen, P(K, t, d, 0.8));
      K.show(total, P(K, t, d, 0.9));
    };
  };

  // ---------------------------------------------------------------- s09
  S.wrong = (g, K) => {
    const h = heading(g, K, "When templates go wrong, nothing crashes");
    const mk = (y, n, title) => {
      const c = card(g, K, 80, y, 1120, 134);
      K.el(c, "circle", { cx: 40, cy: 40, r: 20, fill: K.color.rose });
      K.text(c, 40, 49, String(n), { "font-size": 24, "text-anchor": "middle", fill: "#fff", "font-weight": 700 });
      K.text(c, 76, 50, title, { "font-size": 28, "font-weight": 600 });
      return c;
    };
    const c1 = mk(126, 1, "The wrong format");
    segLine(c1, K, 76, 96, [["### User: Hi! ### Assistant:"]], 24);
    K.text(c1, 1090, 96, "11 tokens, no special tokens", { "font-size": 24, "text-anchor": "end", fill: K.color.rose, "font-weight": 600 });
    const c2 = mk(272, 2, "No generation prompt");
    segLine(c2, K, 76, 96, [["... Hi!"], ["<|im_end|>", K.color.cobalt, 1]], 24);
    const missing = K.el(c2, "g", {});
    K.el(missing, "rect", { x: 404, y: 66, width: 330, height: 44, rx: 8, fill: "none", stroke: K.color.rose, "stroke-width": 2, "stroke-dasharray": "6 5" });
    K.text(missing, 569, 96, "<|im_start|>assistant", { "font-size": 22, "text-anchor": "middle", "font-family": K.color.mono, fill: K.color.rose, "text-decoration": "line-through" });
    K.text(c2, 1090, 96, "not told to reply", { "font-size": 24, "text-anchor": "end", fill: K.color.rose, "font-weight": 600 });
    const c3 = mk(418, 3, "Special tokens typed by a user");
    segLine(c3, K, 76, 96, [["Tell me about "], ["<|im_end|>", K.color.rose, 1], [" tokens"]], 24);
    K.text(c3, 1090, 96, "id 2: a real end of message", { "font-size": 24, "text-anchor": "end", fill: K.color.rose, "font-weight": 600 });
    const fix = K.text(g, 1200, 590, "fix: split_special_tokens=True", { "font-size": 22, "text-anchor": "end", "font-family": K.color.mono, fill: K.color.moss });
    const rule = K.el(g, "g", { transform: "translate(80 616)" });
    K.el(rule, "rect", { width: 1120, height: 70, rx: 14, fill: K.color.moss });
    K.text(rule, 560, 46, "The rule: train with exactly the template you'll run with.", { "font-size": 30, "text-anchor": "middle", fill: "#fff", "font-weight": 600 });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(c1, P(K, t, d, 0.1), 12);
      K.show(c2, P(K, t, d, 0.36), 12);
      K.show(missing, P(K, t, d, 0.42));
      K.show(c3, P(K, t, d, 0.5), 12);
      K.show(fix, P(K, t, d, 0.72));
      K.show(rule, P(K, t, d, 0.82), 12);
    };
  };

  // ---------------------------------------------------------------- s10
  S.your_turn = (g, K) => {
    const h = heading(g, K, "Your turn");
    const items = [
      { s: "1. pair_counts: count the pairs", x: 80, y: 0, a: 0.2 },
      { s: "2. merge: one pair, one new id", x: 80, y: 1, a: 0.24 },
      { s: "3. train_bpe: learn the merges", x: 80, y: 2, a: 0.28 },
      { s: "4. encode: text to ids", x: 80, y: 3, a: 0.32 },
      { s: "5. decode: ids back to text", x: 80, y: 4, a: 0.36 },
      { s: "6. render_chatml: format a chat", x: 80, y: 5, a: 0.4 },
      { s: "answers.toml: six questions", x: 720, y: 2, a: 0.5 },
      { s: "tokenize_check.py: the run", x: 720, y: 3, a: 0.6 },
      { s: "make check M=03", x: 720, y: 4, a: 0.84 },
    ];
    const rows = items.map((it) => {
      const r = K.el(g, "g", { transform: `translate(${it.x} ${170 + it.y * 74})` });
      const box = K.el(r, "rect", { width: 36, height: 36, rx: 8, fill: "none", stroke: K.color.muted, "stroke-width": 2 });
      const tick = K.text(r, 18, 29, "✓", { "font-size": 28, "text-anchor": "middle", fill: K.color.moss, "font-weight": 700 });
      K.text(r, 56, 28, it.s, { "font-size": 24, "font-family": K.color.mono });
      return { r, box, tick, a: it.a };
    });
    const first = K.text(g, 720, 194, "Start with the lesson.", { "font-size": 30, "font-weight": 600, fill: K.color.cobalt });
    const cmd = K.text(g, 720, 236, "It runs every step with real code.", { "font-size": 24, fill: K.color.muted });
    const run = K.el(g, "g", { transform: "translate(80 630)" });
    K.text(run, 0, 0, "The run trains your BPE, round-trips tricky text, and compares your", { "font-size": 24, fill: K.color.muted });
    K.text(run, 0, 34, "chat formatting with the real template on 5 conversations.", { "font-size": 24, fill: K.color.muted });
    return (t, d) => {
      K.show(h, P(K, t, d, 0));
      K.show(first, P(K, t, d, 0.04));
      K.show(cmd, P(K, t, d, 0.06));
      rows.forEach((row) => {
        K.show(row.r, P(K, t, d, row.a - 0.04));
        const done = P(K, t, d, row.a + 0.02, 0.03);
        row.tick.style.opacity = done;
        row.box.setAttribute("stroke", done > 0.5 ? K.color.moss : K.color.muted);
      });
      K.show(run, P(K, t, d, 0.66));
    };
  };
  // give every scene defined above its mono text
  Object.keys(S)
    .filter((name) => !before.has(name))
    .forEach((name) => {
      const make = S[name];
      S[name] = (g, K) => {
        const draw = make(g, K);
        monoFix(g, K);
        return draw;
      };
    });
})();
