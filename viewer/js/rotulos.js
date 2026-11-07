// Hand labelling inside the viewer: the stoppage queue of analise/out/rotulos_manuais.csv
// (analise/rotular_paradas.py) and possession changes (analise/out/rotulos_posse.csv).
// Read and written through serve.py (/api/rotulos); a draft stays in localStorage if a save fails.
//
// Keys (when "atalhos de rotulação" is on): I start · F end · V "viva" · P new stoppage here ·
// H / A possession change to that team · Enter next unlabelled row.
// The queue is blind on purpose: only id and time are shown, never the category or the prediction.

const API = "../api/rotulos?arquivo=";
const FILES = {
  manuais: { name: "rotulos_manuais",
    cols: ["id", "categoria", "frame_ref", "tempo_video", "previsto_inicio", "previsto_fim", "info", "inicio_real", "fim_real", "obs"] },
  posse: { name: "rotulos_posse", cols: ["frame", "time", "obs"] },
};
const TEAM_COLOR = { H: "#1f6fd1", A: "#ee2523" };
const DONE_COLOR = "#3ddc84";

const lab = { manuais: [], posse: [], sel: -1, ok: { manuais: false, posse: false }, keys: true };
let api = null;                      // {getFrame, setFrame, fps, redraw, setStatus}
const timers = {};
const $ = (id) => document.getElementById(id);

// ---- CSV -------------------------------------------------------------------------------------

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows.filter((r) => r.some((x) => x !== ""));
  return body.map((r) => Object.fromEntries(head.map((h, j) => [h, r[j] ?? ""])));
}

function toCSV(cols, rows) {
  const q = (v) => { v = v == null ? "" : String(v); return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v; };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => q(r[c])).join(","))].join("\n") + "\n";
}

// ---- storage ---------------------------------------------------------------------------------

function draftKey(kind) { return "rotulos_rascunho_" + kind; }
function readDraft(kind) {
  try { return JSON.parse(localStorage.getItem(draftKey(kind)) || "null"); } catch { return null; }
}
function writeDraft(kind, pending) {
  try { localStorage.setItem(draftKey(kind), JSON.stringify({ pending, rows: lab[kind] })); } catch { /* private mode */ }
}

async function load(kind) {
  const draft = readDraft(kind);
  try {
    const res = await fetch(API + FILES[kind].name, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 120)}`);
    lab[kind] = parseCSV(await res.text());
    lab.ok[kind] = true;
    if (draft && draft.pending) {    // a save failed last time: the draft is newer than the disk
      lab[kind] = draft.rows;
      save(kind);
    }
  } catch (err) {
    lab.ok[kind] = false;
    if (draft) lab[kind] = draft.rows;
    console.warn("rótulos", kind, err);
    setSave(`⚠ ${FILES[kind].name}: ${err.message}`);
  }
}

function save(kind) {
  writeDraft(kind, true);
  clearTimeout(timers[kind]);
  timers[kind] = setTimeout(async () => {
    try {
      const res = await fetch(API + FILES[kind].name, { method: "POST", body: toCSV(FILES[kind].cols, lab[kind]) });
      if (!res.ok) throw new Error(`${res.status}`);
      writeDraft(kind, false);
      setSave(`salvo ✓ ${new Date().toLocaleTimeString()}`);
    } catch (err) {
      setSave(`⚠ não salvo (${err.message}) — rascunho guardado no navegador`);
    }
  }, 400);
}

function setSave(text) { const el = $("rotSave"); if (el) el.textContent = text; }

// ---- helpers ---------------------------------------------------------------------------------

const num = (v) => (v === "" || v == null || isNaN(+v) ? null : +v);
function mmss(f) {
  const s = f / api.fps;
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(2).padStart(5, "0")}`;
}
function status(r) {
  if (/viva/i.test(r.obs || "")) return "viva";
  if (num(r.inicio_real) != null && num(r.fim_real) != null) return "ok";
  if (num(r.inicio_real) != null || num(r.fim_real) != null) return "meio";
  return "";
}
const STATUS_TXT = { ok: "✓", viva: "viva", meio: "…", "": "" };

function changed(kind) {
  save(kind);
  render();
  api.redraw();
}

// ---- actions ---------------------------------------------------------------------------------

function select(i, go = true) {
  lab.sel = i;
  const r = lab.manuais[i];
  if (r && go) api.setFrame(num(r.inicio_real) ?? num(r.frame_ref));
  render();
}

function mark(field) {
  const r = lab.manuais[lab.sel];
  if (!r) return api.setStatus("rotular: escolha uma linha da fila (ou P para uma parada nova)");
  r[field] = String(api.getFrame());
  if (field !== "obs" && /viva/i.test(r.obs || "")) r.obs = "";
  changed("manuais");
}

function setViva() {
  const r = lab.manuais[lab.sel];
  if (!r) return;
  r.inicio_real = ""; r.fim_real = ""; r.obs = "viva";
  changed("manuais");
}

function clearRow() {
  const r = lab.manuais[lab.sel];
  if (!r) return;
  r.inicio_real = ""; r.fim_real = ""; r.obs = "";
  changed("manuais");
}

function newStoppage() {
  const f = api.getFrame();
  const id = Math.max(0, ...lab.manuais.map((r) => num(r.id) || 0)) + 1;
  lab.manuais.push({ id: String(id), categoria: "livre", frame_ref: String(f), tempo_video: mmss(f),
    previsto_inicio: "", previsto_fim: "", info: "marcada no viewer", inicio_real: String(f), fim_real: "", obs: "" });
  lab.sel = lab.manuais.length - 1;
  changed("manuais");
}

function next() {
  const n = lab.manuais.length;
  for (let k = 1; k <= n; k++) {
    const i = (lab.sel + k + n) % n;
    if (!status(lab.manuais[i])) return select(i);
  }
  api.setStatus("rotular: todas as linhas estão preenchidas");
}

function possession(team) {
  const f = api.getFrame();
  const i = lab.posse.findIndex((p) => num(p.frame) === f);
  if (i >= 0) lab.posse[i].time = team;
  else lab.posse.push({ frame: String(f), time: team, obs: "" });
  lab.posse.sort((a, b) => num(a.frame) - num(b.frame));
  changed("posse");
}

// ---- panel -----------------------------------------------------------------------------------

function render() {
  const list = $("rotList");
  if (!list) return;
  const rows = lab.manuais;
  const done = rows.filter((r) => status(r) === "ok" || status(r) === "viva").length;
  $("rotProgress").textContent = rows.length
    ? `${done} / ${rows.length} paradas rotuladas · ${lab.posse.length} trocas de posse`
    : "sem fila: rode python analise/rotular_paradas.py (ou use P para marcar paradas livres)";
  list.innerHTML = rows.map((r, i) => `
    <div class="rotitem ${i === lab.sel ? "sel" : ""}" data-i="${i}">
      <span>#${r.id}</span><span>${r.tempo_video || mmss(num(r.frame_ref) || 0)}</span>
      ${r.categoria === "livre" ? '<span class="help">livre</span>' : ""}
      <span class="st ${status(r)}">${STATUS_TXT[status(r)]}</span></div>`).join("");
  const sel = list.querySelector(".sel");
  if (sel) sel.scrollIntoView({ block: "nearest" });

  const r = rows[lab.sel];
  $("rotCur").innerHTML = !r ? '<div class="help">clique numa linha da fila</div>' : `
    <div class="row"><b>#${r.id}</b><span class="help">referência ${r.frame_ref}</span></div>
    <div class="row">início <input type="number" data-f="inicio_real" value="${r.inicio_real || ""}">
      <button data-a="ini" title="frame atual (I)">⟵ frame</button></div>
    <div class="row">fim&nbsp;&nbsp;&nbsp; <input type="number" data-f="fim_real" value="${r.fim_real || ""}">
      <button data-a="fim" title="frame atual (F)">⟵ frame</button></div>
    <div class="row">obs <input type="text" data-f="obs" value="${(r.obs || "").replace(/"/g, "&quot;")}" style="width:110px">
      <button data-a="viva" title="não houve parada (V)">viva</button><button data-a="limpar">limpar</button></div>
    <div class="row"><button data-a="ref">ir à referência</button><button data-a="proxima" class="primary">próxima ⏎</button></div>`;

  $("rotPosse").innerHTML = lab.posse.slice().reverse().map((p) => `
    <div class="rotitem" data-f="${p.frame}"><i class="sw" style="background:${TEAM_COLOR[p.time] || "#888"}"></i>
      <span>${mmss(num(p.frame))}</span><span>${p.time}</span><span class="help">${p.frame}</span>
      <button class="st" data-del="${p.frame}" title="apagar">✕</button></div>`).join("");
}

function bind() {
  $("rotList").onclick = (e) => {
    const it = e.target.closest(".rotitem");
    if (it) select(+it.dataset.i);
  };
  $("rotCur").onclick = (e) => {
    const a = e.target.dataset.a;
    const r = lab.manuais[lab.sel];
    if (!a || !r) return;
    if (a === "ini") mark("inicio_real");
    else if (a === "fim") mark("fim_real");
    else if (a === "viva") setViva();
    else if (a === "limpar") clearRow();
    else if (a === "ref") api.setFrame(num(r.frame_ref));
    else if (a === "proxima") next();
  };
  $("rotCur").onchange = (e) => {
    const f = e.target.dataset.f;
    const r = lab.manuais[lab.sel];
    if (!f || !r) return;
    r[f] = e.target.value.trim();
    changed("manuais");
  };
  $("rotPosse").onclick = (e) => {
    if (e.target.dataset.del) {
      lab.posse = lab.posse.filter((p) => p.frame !== e.target.dataset.del);
      return changed("posse");
    }
    const it = e.target.closest(".rotitem");
    if (it) api.setFrame(num(it.dataset.f));
  };
  $("rotNew").onclick = newStoppage;
  $("rotH").onclick = () => possession("H");
  $("rotA").onclick = () => possession("A");
  $("rotDownload").onclick = () => {
    for (const kind of ["manuais", "posse"]) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([toCSV(FILES[kind].cols, lab[kind])], { type: "text/csv" }));
      a.download = FILES[kind].name + ".csv";
      a.click();
    }
  };
  const keys = $("rotKeys");
  try { lab.keys = localStorage.getItem("rotulos_atalhos") !== "0"; } catch { /* default on */ }
  keys.checked = lab.keys;
  keys.onchange = () => {
    lab.keys = keys.checked;
    try { localStorage.setItem("rotulos_atalhos", lab.keys ? "1" : "0"); } catch { /* ignore */ }
  };
}

// ---- public ----------------------------------------------------------------------------------

export async function initRotulos(hooks) {
  api = hooks;
  bind();
  await Promise.all([load("manuais"), load("posse")]);
  if (lab.manuais.length) lab.sel = Math.max(0, lab.manuais.findIndex((r) => !status(r)));
  render();
  api.redraw();
}

// true when the key was a labelling shortcut (the caller then ignores it)
export function rotulosKey(e) {
  if (!lab.keys || e.ctrlKey || e.metaKey || e.altKey) return false;
  const k = e.key.toLowerCase();
  const act = { i: () => mark("inicio_real"), f: () => mark("fim_real"), v: setViva, p: newStoppage,
    h: () => possession("H"), a: () => possession("A"), enter: next }[k];
  if (!act) return false;
  e.preventDefault();
  act();
  return true;
}

// one timeline row: labelled stoppages (green), "viva" checks (grey tick), possession changes (team tick)
export function drawRotulos(ctx, a, b, w, row) {
  const span = b - a + 1;
  const x = (f) => ((f - a) / span) * w;
  for (const r of lab.manuais) {
    const s = num(r.inicio_real), e = num(r.fim_real);
    if (s != null && e != null && e >= a && s <= b) {
      ctx.fillStyle = DONE_COLOR;
      ctx.fillRect(x(s), row[0], Math.max(2, x(e) - x(s)), row[1]);
    } else if (s != null && s >= a && s <= b) {
      ctx.fillStyle = DONE_COLOR;
      ctx.fillRect(x(s) - 1, row[0], 2, row[1]);
    } else if (/viva/i.test(r.obs || "") && num(r.frame_ref) >= a && num(r.frame_ref) <= b) {
      ctx.fillStyle = "#8b95a3";
      ctx.fillRect(x(num(r.frame_ref)) - 1, row[0] + row[1] / 2, 2, row[1] / 2);
    }
  }
  for (const p of lab.posse) {
    const f = num(p.frame);
    if (f < a || f > b) continue;
    ctx.fillStyle = TEAM_COLOR[p.time] || "#888";
    ctx.fillRect(x(f) - 1, row[0] - 1, 3, row[1] + 2);
  }
}
