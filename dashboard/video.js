// Module video player. Plays a module's scenes in order, each with its narration clip.
// A module's video/scenes.js registers one draw function per visual:
//   VIDEO_SCENES[name] = (g, kit) => (t, d) => { ... }
// The outer function builds the scene inside the SVG group g; the function it returns is called
// every frame with t (seconds into the scene) and d (the scene's length), and must depend only on
// those, so pausing and scrubbing always show the right picture.

"use strict";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
window.VIDEO_SCENES = {};

// ---------- the kit scenes draw with ----------

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const kit = {
  color: {},
  clamp: (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x)),
  lerp: (a, b, p) => a + (b - a) * p,
  ease: (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2),
  // eased progress of something that starts at `start` seconds and lasts `len` seconds
  at(t, start, len = 0.6) {
    return kit.ease(kit.clamp((t - start) / len));
  },
  el(parent, tag, attrs = {}, text) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  },
  text(parent, x, y, str, attrs = {}) {
    return kit.el(parent, "text", { x, y, fill: kit.color.ink, "font-size": 30, ...attrs }, str);
  },
  // a rounded token tile with a label; returns the group so it can be moved and faded
  tile(parent, x, y, w, h, label, { fill, ink, size = 22, mono = true } = {}) {
    const g = kit.el(parent, "g", { transform: `translate(${x} ${y})` });
    kit.el(g, "rect", { width: w, height: h, rx: 8, fill: fill || kit.color.surface, stroke: kit.color.line });
    kit.el(
      g,
      "text",
      {
        x: w / 2, y: h / 2 + size * 0.35, "text-anchor": "middle", "font-size": size,
        fill: ink || kit.color.ink, "font-family": mono ? kit.color.mono : kit.color.font,
      },
      label,
    );
    return g;
  },
  // wrapped text; returns the <text> node and one <tspan> per word, so words can be highlighted
  para(parent, x, y, str, { chars = 48, lh = 40, ...attrs } = {}) {
    const node = kit.el(parent, "text", { x, y, fill: kit.color.ink, "font-size": 30, ...attrs });
    const words = [];
    let line = 0;
    let len = 0;
    for (const w of str.split(" ")) {
      if (len && len + w.length + 1 > chars) {
        line += 1;
        len = 0;
      }
      const attrsW = len === 0 && line > 0 ? { x, dy: lh } : {};
      words.push(kit.el(node, "tspan", attrsW, (len ? " " : "") + w));
      len += w.length + (len ? 1 : 0);
    }
    return { node, words };
  },
  // fade a node in (p from 0 to 1), sliding it up by dy as it comes, on top of its own position
  show(node, p, dy = 0) {
    node.style.opacity = p;
    if (!dy) return;
    if (node.dataset.base === undefined) node.dataset.base = node.getAttribute("transform") || "";
    node.setAttribute("transform", `${node.dataset.base} translate(0 ${(1 - p) * dy})`.trim());
  },
};

function readColors() {
  for (const k of ["paper", "surface", "ink", "muted", "line", "cobalt", "cobalt-soft", "moss", "amber"]) {
    kit.color[k.replace("-soft", "Soft")] = cssVar(`--${k}`);
  }
  kit.color.rose = matchMedia("(prefers-color-scheme: dark)").matches ? "#f07a7a" : "#c43d3d";
  kit.color.font = cssVar("--font");
  kit.color.mono = cssVar("--mono");
}

// ---------- the player ----------

const state = { scenes: [], audios: [], i: 0, t: 0, playing: false, started: false, watched: false, draw: null };

function fmt(s) {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function sceneLength(i) {
  const s = state.scenes[i];
  const a = state.audios[i];
  return s.seconds || (a && isFinite(a.duration) ? a.duration : 5);
}

function totalLength() {
  return state.scenes.reduce((sum, _, i) => sum + sceneLength(i), 0);
}

function sentences(text) {
  return text.split(/(?<=[.!?])\s+/).filter(Boolean);
}

function captionAt(scene, t, d) {
  const parts = sentences(scene.narration);
  const total = parts.reduce((n, p) => n + p.length, 0) || 1;
  let used = 0;
  for (const p of parts) {
    used += p.length;
    if (t / d <= used / total) return p;
  }
  return parts[parts.length - 1] || "";
}

function buildScene(i) {
  const svg = $("stage");
  svg.replaceChildren();
  kit.el(svg, "rect", { width: 1280, height: 720, fill: kit.color.surface });
  const g = kit.el(svg, "g", { class: "scene" });
  const make = window.VIDEO_SCENES[state.scenes[i].visual];
  state.draw = make ? make(g, kit) : () => {};
  if (!make) kit.text(g, 640, 360, `(no drawing for "${state.scenes[i].visual}")`, { "text-anchor": "middle", fill: kit.color.muted });
  $("scene-name").textContent = `Scene ${i + 1} of ${state.scenes.length}`;
  $("scrub").querySelectorAll(".seg").forEach((s, k) => s.classList.toggle("now", k === i));
}

function render() {
  const s = state.scenes[state.i];
  const d = sceneLength(state.i);
  state.draw(Math.min(state.t, d), d);
  $("caption").textContent = $("cc").getAttribute("aria-pressed") === "true" ? captionAt(s, state.t, d) : "";
  let before = 0;
  state.scenes.forEach((_, k) => {
    const seg = $("scrub").children[k];
    const len = sceneLength(k);
    const fill = k < state.i ? 1 : k === state.i ? Math.min(1, state.t / len) : 0;
    seg.firstChild.style.width = `${fill * 100}%`;
    if (k < state.i) before += len;
  });
  const now = before + state.t;
  $("clock").textContent = `${fmt(now)} / ${fmt(totalLength())}`;
  $("scrub").setAttribute("aria-valuenow", String(Math.round((100 * now) / totalLength())));
}

function setPlaying(on) {
  state.playing = on;
  $("play-icon").setAttribute("d", on ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z");
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  const a = state.audios[state.i];
  if (a) on ? a.play().catch(() => {}) : a.pause();
  if (on) {
    $("big-play").hidden = true;
    $("end-card").hidden = true;
    state.started = true;
  }
}

function goTo(i, t = 0) {
  const old = state.audios[state.i];
  if (old) old.pause();
  state.i = Math.max(0, Math.min(state.scenes.length - 1, i));
  state.t = t;
  buildScene(state.i);
  const a = state.audios[state.i];
  if (a) {
    try { a.currentTime = t; } catch { /* not loaded yet */ }
    if (state.playing) a.play().catch(() => {});
  }
  render();
}

function finish() {
  setPlaying(false);
  state.t = sceneLength(state.i);
  render();
  $("end-card").hidden = false;
  if (!state.watched) {
    state.watched = true;
    fetch("/api/watched", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ module: state.module }),
    }).catch(() => {});
  }
}

let last = null;
function frame(now) {
  if (state.playing) {
    const a = state.audios[state.i];
    const d = sceneLength(state.i);
    if (a && !a.paused && isFinite(a.currentTime)) state.t = a.currentTime;
    else if (last !== null) state.t += (now - last) / 1000;  // no audio: keep time ourselves
    if (state.t >= d - 0.02 || (a && a.ended)) {
      if (state.i + 1 < state.scenes.length) goTo(state.i + 1);
      else finish();
    }
    render();
  }
  last = now;
  requestAnimationFrame(frame);
}

function seekToFraction(f) {
  let target = f * totalLength();
  for (let k = 0; k < state.scenes.length; k++) {
    const len = sceneLength(k);
    if (target <= len || k === state.scenes.length - 1) return goTo(k, Math.min(target, len - 0.05));
    target -= len;
  }
}

function wire() {
  $("play").onclick = $("big-play").onclick = () => setPlaying(!state.playing);
  $("replay").onclick = () => { goTo(0); setPlaying(true); };
  $("prev").onclick = () => goTo(state.t > 1.5 ? state.i : state.i - 1);
  $("next").onclick = () => (state.i + 1 < state.scenes.length ? goTo(state.i + 1) : finish());
  $("cc").onclick = () => {
    const on = $("cc").getAttribute("aria-pressed") !== "true";
    $("cc").setAttribute("aria-pressed", String(on));
    render();
  };
  const scrub = $("scrub");
  const seek = (e) => {
    const r = scrub.getBoundingClientRect();
    seekToFraction(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  scrub.addEventListener("pointerdown", (e) => {
    seek(e);
    scrub.setPointerCapture(e.pointerId);
    scrub.onpointermove = seek;
  });
  scrub.addEventListener("pointerup", () => (scrub.onpointermove = null));
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("button, a")) return;
    if (e.key === " ") { e.preventDefault(); setPlaying(!state.playing); }
    if (e.key === "ArrowRight") $("next").onclick();
    if (e.key === "ArrowLeft") $("prev").onclick();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readColors();
    buildScene(state.i);
    render();
  });
}

async function main() {
  readColors();
  let video;
  try {
    const r = await fetch("script.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    video = await r.json();
  } catch (e) {
    $("title").textContent = "This video couldn't be loaded.";
    $("caption").textContent = `Is make progress still running? (${e.message})`;
    $("big-play").hidden = true;
    return;
  }
  await new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = "scenes.js";
    s.onload = ok;
    s.onerror = () => fail(new Error("scenes.js"));
    document.head.append(s);
  }).catch(() => {});
  state.module = video.module;
  state.scenes = video.scenes;
  state.audios = video.scenes.map((s) => {
    const a = new Audio(s.audio);
    a.preload = "auto";
    return a;
  });
  document.title = video.title;
  $("title").textContent = video.title;
  $("next-cmd").textContent = video.next_command || "";
  if (video.problems?.length) {
    $("problems").hidden = false;
    $("problems").textContent = "Course authors: " + video.problems.join("; ");
  }
  const scrub = $("scrub");
  scrub.replaceChildren(
    ...video.scenes.map((s, k) => {
      const seg = document.createElement("span");
      seg.className = "seg";
      seg.style.flexGrow = String(sceneLength(k));
      seg.title = `Scene ${k + 1}`;
      seg.append(document.createElement("i"));
      return seg;
    }),
  );
  wire();
  // #scene=3&t=10 opens scene 3 at 10 seconds, paused (handy for sharing one moment)
  const hash = new URLSearchParams(location.hash.slice(1));
  const start = Math.max(1, parseInt(hash.get("scene") || "1", 10) || 1);
  goTo(start - 1, parseFloat(hash.get("t") || "0") || 0);
  if (hash.has("still")) $("big-play").hidden = true;
  requestAnimationFrame(frame);
}

main();
