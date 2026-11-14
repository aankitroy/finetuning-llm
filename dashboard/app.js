// Progress dashboard. Polls /api/state and renders it. No build step, no network beyond localhost.
"use strict";

const POLL_MS = 3000;
const COMPARE_MODULES = ["m05", "m07", "m08", "m10", "m12"];
const COMPARE_KEYS = [
  ["peak_mem_gb", "Peak memory (GB)", "lower"],
  ["tokens_per_sec", "Training speed (tokens/sec)", "higher"],
  ["trainable_pct", "Trainable parameters (%)", "lower"],
  ["regression_score", "General-capability score", "higher"],
];

let lastJson = "";
let prevDone = null;
const openIds = new Set();
const charts = new Map();

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const c of children) if (c != null) n.append(c);
  return n;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmt(v) {
  if (typeof v !== "number") return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function when(ts) {
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// ---------- header ----------

function renderHeader(s) {
  const { done, total, est_minutes_done } = s.course;
  const hours = Math.round(est_minutes_done / 6) / 10;
  $("tally").textContent =
    `${done} of ${total} modules done` + (done ? `, about ${hours} h of coursework (estimated)` : "");
  const pct = total ? Math.round((100 * done) / total) : 0;
  $("bar").setAttribute("aria-valuenow", String(pct));
  $("bar").firstElementChild.style.width = pct + "%";
}

// ---------- copy helpers ----------

async function copyText(text, btn, idle) {
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied";
  } catch {
    btn.textContent = "Select and copy";
  }
  setTimeout(() => (btn.textContent = idle), 1500);
}

function cmdBox(command) {
  const btn = el("button", { type: "button", text: "Copy" });
  btn.addEventListener("click", () => copyText(command, btn, "Copy"));
  return el("div", { class: "cmd" }, el("code", { text: command }), btn);
}

function miniCmd(command) {
  const btn = el("button", { type: "button", class: "mini", title: "Copy command" }, el("code", { text: command }));
  btn.addEventListener("click", () => {
    const code = btn.firstChild;
    copyText(command, code, command);
  });
  return btn;
}

// ---------- next step ----------

const KICKER = {
  empty: "Start here",
  start: "Up next",
  finish: "Finish what you started",
  retry: "Try the checkpoint again",
  complete: "All done",
};

function renderNext(n) {
  const box = $("next");
  box.replaceChildren(
    el("p", { class: "kicker", text: KICKER[n.kind] || "Up next" }),
    el("h3", { text: n.title }),
    el("p", { text: n.instruction }),
  );
  if (n.command) box.append(cmdBox(n.command));
  if (n.why) box.append(el("p", { class: "why", text: "Why: " + n.why }));
}

// ---------- step-by-step guide ----------

const SELECT_KEY = "ftcourse.guide.module";
let selectedId = null;
try {
  selectedId = localStorage.getItem(SELECT_KEY);
} catch {}
const stepOpen = new Map();
let lastState = null;

function selectModule(id) {
  selectedId = id;
  try {
    localStorage.setItem(SELECT_KEY, id);
  } catch {}
  if (lastState) renderGuide(lastState);
  $("guide").scrollIntoView({ behavior: "smooth", block: "start" });
}

function itemRow(kind, it) {
  const done = kind === "exercises" ? it.written : it.answered;
  // Not written yet beats "failed": a stub failing its test isn't news.
  const state = !done ? "todo" : it.last === "pass" ? "pass" : it.last === "fail" ? "fail" : "written";
  const mark = { pass: "✔", fail: "✗", written: "●", todo: "○" }[state];
  const label =
    kind === "exercises" ? el("span", { class: "iname", text: `${it.id}. ${it.name}` }) : el("span", { text: `${it.id.toUpperCase()} ${it.question}` });
  const body = el("div", {}, label);
  if (state === "fail" && it.message) body.append(el("span", { class: "imsg", text: it.message }));
  if (state === "written") body.append(el("span", { class: "imsg", text: "Done. Run the check to see if it's right." }));
  if (state === "todo") body.append(el("span", { class: "imsg", text: kind === "exercises" ? "Not written yet" : "Not answered yet" }));
  if (state === "fail" || state === "todo") body.append(el("span", { class: "ihint" }, miniCmd(it.hint)));
  return el("li", { class: state, "aria-label": state }, el("span", { class: "mark", text: mark }), body);
}

function stepProgress(st) {
  if (st.kind === "train" && typeof st.value === "number") {
    const v = /accuracy|rate|pct/.test(st.metric) && st.value <= 1 ? (st.value * 100).toFixed(1) + "%" : fmt(st.value);
    return `${st.progress}: ${st.metric.replace(/_/g, " ")} ${v}`;
  }
  return st.progress || "";
}

function renderGuide(s) {
  lastState = s;
  const box = $("guide");
  const byId = new Map(s.modules.map((m) => [m.id, m]));
  const m = byId.get(selectedId) || byId.get(s.next.module) || s.modules.find((x) => x.guide);
  if (!m) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const head = el(
    "div",
    { class: "guide-head" },
    el("h2", { id: "guide-h", text: `Steps for ${m.id.toUpperCase()} · ${m.title}` }),
    m.guide ? el("span", { class: "count", text: `${m.guide.done} of ${m.guide.total} steps done` }) : null,
  );
  if (!m.guide) {
    box.replaceChildren(head, el("p", { class: "empty-guide", text: "This module's lessons are still being written. Step-by-step guidance will appear here when they're ready." }));
    return;
  }
  const current = m.guide.steps.findIndex((st) => st.status !== "done");
  const list = el("ol", { class: "steps" });
  m.guide.steps.forEach((st, i) => {
    const key = `${m.id}:${i}`;
    const det = el(
      "details",
      {},
      el(
        "summary",
        {},
        el("span", { class: "gnum", text: st.status === "done" ? "✔" : String(i + 1) }),
        el("span", { class: "gtitle", text: st.title }),
        el("span", { class: "gprog", text: stepProgress(st) }),
      ),
    );
    det.open = stepOpen.has(key) ? stepOpen.get(key) : i === current;
    det.addEventListener("toggle", () => stepOpen.set(key, det.open));
    const body = el("div", { class: "gbody" }, el("p", { text: st.detail }));
    if (st.note) body.append(el("p", { class: "gnote", text: st.note }));
    if (st.items?.length) body.append(el("ul", { class: "items" }, ...st.items.map((it) => itemRow(st.kind, it))));
    if (st.command) body.append(cmdBox(st.command));
    det.append(body);
    const classes = ["gstep", st.status, i === current ? "current" : ""].join(" ");
    list.append(el("li", { class: classes }, det));
  });
  box.replaceChildren(head, list);
}

// ---------- route ----------

function revealNote(r) {
  const parts = [];
  if (r?.hints) parts.push(`${r.hints} hint${r.hints === 1 ? "" : "s"}`);
  if (r?.solutions) parts.push(`${r.solutions} solution${r.solutions === 1 ? "" : "s"} viewed`);
  return parts.join(", ");
}

function statusText(m) {
  const base = statusBase(m);
  const note = revealNote(m.reveals);
  return note ? (base ? `${base}, ${note}` : note) : base;
}

function statusBase(m) {
  if (m.status === "done") {
    const keys = m.metrics.map((x) => x.key).filter((k) => k in m.latest_metrics);
    if (keys.length) {
      const k = keys[0];
      const label = m.metrics.find((x) => x.key === k).label;
      return `${label}: ${fmt(m.latest_metrics[k])}`;
    }
    return "Done";
  }
  if (m.status === "in_progress") return m.attempts.length ? "Check failed" : "In progress";
  return "";
}

function attemptsTable(m) {
  const wrap = el("div", { class: "attempts" }, el("p", { text: m.summary }));
  const open = el("button", { type: "button", class: "mini open-guide", text: "Show the steps for this module" });
  open.addEventListener("click", () => selectModule(m.id));
  wrap.append(open);
  const note = revealNote(m.reveals);
  if (note) wrap.append(el("p", { text: `Help used: ${note}.` }));
  if (!m.attempts.length) {
    wrap.append(el("p", { text: "No checkpoint runs yet." }));
    return wrap;
  }
  const rows = [...m.attempts].reverse().map((a) => {
    const metrics = Object.entries(a.metrics || {}).map(([k, v]) => `${k} ${fmt(v)}`).join(", ");
    const t = a.tests || {};
    return el(
      "tr",
      {},
      el("td", { text: when(a.ts) }),
      el("td", { class: a.passed ? "ok" : "bad", text: a.passed ? "Passed" : `Failed (${t.failed ?? "?"})` }),
      el("td", { text: a.run_label || "" }),
      el("td", { text: metrics }),
    );
  });
  wrap.append(
    el(
      "table",
      {},
      el("thead", {}, el("tr", {}, ...["When", "Result", "Label", "Metrics"].map((h) => el("th", { text: h })))),
      el("tbody", {}, ...rows),
    ),
  );
  return wrap;
}

function renderRoute(s, newlyDone) {
  const parts = [];
  for (const m of s.modules) {
    let p = parts[parts.length - 1];
    if (!p || p.name !== m.part) parts.push((p = { name: m.part, mods: [] }));
    p.mods.push(m);
  }
  const nextId = s.next.module;
  $("route").replaceChildren(
    ...parts.map((p) =>
      el(
        "div",
        { class: "part" },
        el("h3", { text: p.name }),
        el(
          "ol",
          {},
          ...p.mods.map((m) => {
            const det = el(
              "details",
              {},
              el(
                "summary",
                {},
                el("span", { class: "mid", text: m.id.toUpperCase() }),
                el("span", { class: "mtitle", text: m.title }),
                el("span", { class: "meta", text: statusText(m) }),
              ),
              attemptsTable(m),
            );
            det.open = openIds.has(m.id);
            det.addEventListener("toggle", () => (det.open ? openIds.add(m.id) : openIds.delete(m.id)));
            const li = el("li", { class: `stop ${m.status}${m.id === nextId ? " is-next" : ""}` }, det);
            if (newlyDone.has(m.id)) li.classList.add("flash");
            return li;
          }),
        ),
      ),
    ),
  );
}

// ---------- charts ----------

function baseOptions(yTitle) {
  const ink = css("--muted");
  const grid = css("--line");
  return {
    responsive: true,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: ink, maxRotation: 0, autoSkip: true }, grid: { display: false } },
      y: { ticks: { color: ink }, grid: { color: grid }, title: { display: !!yTitle, text: yTitle, color: ink } },
    },
  };
}

function upsertChart(key, canvas, type, labels, data, color, yTitle, logY = false) {
  let c = charts.get(key);
  if (c && c.canvas !== canvas) {
    c.destroy();
    c = null;
  }
  if (!c) {
    c = new Chart(canvas, {
      type,
      data: { labels, datasets: [{ data, borderColor: color, backgroundColor: color, pointRadius: 3, tension: 0.25 }] },
      options: baseOptions(yTitle),
    });
    if (logY) c.options.scales.y.type = "logarithmic";
    c.update("none");
    charts.set(key, c);
  } else {
    c.data.labels = labels;
    c.data.datasets[0].data = data;
    c.update("none");
  }
}

function chartBox(caption, hint) {
  const canvas = el("canvas", { height: "150" });
  const fig = el("figure", { class: "chart-box" }, el("figcaption", { text: caption }));
  if (hint) fig.append(el("p", { class: "hint", text: hint }));
  fig.append(canvas);
  return { fig, canvas };
}

// Rebuild chart DOM only when the set of charts changes; otherwise update data in place.
const layoutKeys = { compare: "", perModule: "" };
const canvases = new Map();

function renderResults(s) {
  const firstDone = new Map();
  for (const r of s.history) if (r.passed && !firstDone.has(r.module)) firstDone.set(r.module, r.ts);
  const anyMetrics = s.history.some((r) => r.metrics && Object.keys(r.metrics).length);
  $("results").hidden = !(firstDone.size || anyMetrics);
  if ($("results").hidden) return;

  // Timeline: modules completed, cumulative, by the time each was first passed.
  const tl = [...firstDone.entries()].sort((a, b) => (a[1] < b[1] ? -1 : 1));
  upsertChart(
    "timeline",
    $("chart-timeline"),
    "line",
    tl.map(([id, ts]) => `${id.toUpperCase()} ${when(ts)}`),
    tl.map((_, i) => i + 1),
    css("--moss"),
    "Modules done",
  );

  // Comparison across techniques — only metrics that mean the same thing everywhere.
  const byId = new Map(s.modules.map((m) => [m.id, m]));
  const latestPass = (id) => [...(byId.get(id)?.attempts || [])].reverse().find((a) => a.passed);
  const compare = COMPARE_KEYS.map(([key, label, better]) => {
    const pts = COMPARE_MODULES.map((id) => [id, latestPass(id)?.metrics?.[key]]).filter(([, v]) => typeof v === "number");
    return { key, label, better, pts };
  }).filter((c) => c.pts.length);

  // Per-module: each declared metric across attempts.
  const perModule = s.modules
    .map((m) => ({
      m,
      series: m.metrics
        .map((d) => ({ d, pts: m.attempts.filter((a) => typeof a.metrics?.[d.key] === "number") }))
        .filter((x) => x.pts.length),
    }))
    .filter((x) => x.series.length);

  const cKey = compare.map((c) => c.key).join("|");
  if (cKey !== layoutKeys.compare) {
    layoutKeys.compare = cKey;
    const root = $("compare");
    root.replaceChildren();
    if (compare.length) {
      const grid = el("div", { class: "small-multiples" });
      for (const c of compare) {
        const scale = c.key === "trainable_pct" ? " (log scale)" : "";
        const { fig, canvas } = chartBox(c.label, `Across techniques, ${c.better} is better${scale}`);
        canvases.set("cmp:" + c.key, canvas);
        grid.append(fig);
      }
      root.append(el("h3", { text: "Full fine-tune vs LoRA vs QLoRA and beyond" }), grid);
    }
  }
  for (const c of compare) {
    upsertChart(
      "cmp:" + c.key,
      canvases.get("cmp:" + c.key),
      "bar",
      c.pts.map(([id]) => id.toUpperCase()),
      c.pts.map(([, v]) => v),
      css("--cobalt"),
      undefined,
      c.key === "trainable_pct",
    );
  }

  const pKey = perModule.map((x) => x.m.id + ":" + x.series.map((s) => s.d.key).join(",")).join("|");
  if (pKey !== layoutKeys.perModule) {
    layoutKeys.perModule = pKey;
    const root = $("per-module");
    root.replaceChildren();
    for (const { m, series } of perModule) {
      const grid = el("div", { class: "small-multiples" });
      for (const { d } of series) {
        const { fig, canvas } = chartBox(d.label, `Each attempt, ${d.better} is better`);
        canvases.set(`mod:${m.id}:${d.key}`, canvas);
        grid.append(fig);
      }
      root.append(el("h3", { text: `${m.id.toUpperCase()} ${m.title}` }), grid);
    }
  }
  for (const { m, series } of perModule) {
    for (const { d, pts } of series) {
      upsertChart(
        `mod:${m.id}:${d.key}`,
        canvases.get(`mod:${m.id}:${d.key}`),
        "line",
        pts.map((a, i) => a.run_label || `#${i + 1}`),
        pts.map((a) => a.metrics[d.key]),
        css("--cobalt"),
      );
    }
  }
}

// ---------- warnings ----------

let warningsHiddenFor = "";
function renderWarnings(ws) {
  const key = ws.join("\n");
  const box = $("warnings");
  box.hidden = !ws.length || key === warningsHiddenFor;
  $("warnings-text").textContent = ws.length === 1 ? ws[0] : `${ws.length} problems: ${ws.join("; ")}`;
  $("warnings-close").onclick = () => {
    warningsHiddenFor = key;
    box.hidden = true;
  };
}

// ---------- loop ----------

async function tick() {
  try {
    let res;
    try {
      res = await fetch("/api/state", { cache: "no-store" });
    } catch {
      $("tally").textContent = "Can't reach the progress server. Is `make progress` still running?";
      return;
    }
    const text = await res.text();
    if (!res.ok) {
      let detail = "";
      try {
        detail = JSON.parse(text).error || "";
      } catch {}
      $("tally").textContent =
        "The progress server hit an error. Check the terminal running `make progress`." +
        (detail ? ` (${detail})` : "");
      lastJson = "";
      return;
    }
    if (text !== lastJson) {
      const s = JSON.parse(text);
      const done = new Set(s.modules.filter((m) => m.status === "done").map((m) => m.id));
      const newly = prevDone ? new Set([...done].filter((id) => !prevDone.has(id))) : new Set();
      prevDone = done;
      renderHeader(s);
      renderNext(s.next);
      renderGuide(s);
      renderRoute(s, newly);
      renderResults(s);
      renderWarnings(s.warnings);
      lastJson = text; // only after a successful render, so a bad payload is retried
    }
  } catch (e) {
    $("tally").textContent = "The dashboard couldn't read your progress: " + e.message;
  } finally {
    setTimeout(tick, POLL_MS);
  }
}

tick();
