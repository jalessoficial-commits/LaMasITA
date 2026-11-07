// Hand labelling inside the viewer: the stoppage queue of analise/out/rotulos_manuais.csv
// (analise/rotular_paradas.py) and possession changes (analise/out/rotulos_posse.csv).
// Read and written through serve.py (/api/rotulos); a draft stays in localStorage if a save fails.
//
// Keys (when "atalhos de rotulação" is on): I start · F end · V "viva" · P new stoppage here ·
// Enter next unlabelled row · H / A / D possession from this frame (Barcelona, Athletic, dead ball) ·
// Z undo the last possession mark.
// The queue is blind on purpose: only id and time are shown, never the category or the prediction.
//
// Possession blocks (rotulos_blocos.csv): a stretch [inicio, fim] labelled completely, mark by mark.
// Inside a block every frame takes the label of the last mark at or before it, so a block is a small
// ground truth in the official format (analise/rotular_paradas.gt_blocos). Playback stops at its end.

const API = "../api/rotulos?arquivo=";
const FILES = {
  manuais: { name: "rotulos_manuais",
    cols: ["id", "categoria", "frame_ref", "tempo_video", "previsto_inicio", "previsto_fim", "info", "inicio_real", "fim_real", "obs"] },
  posse: { name: "rotulos_posse", cols: ["frame", "time", "obs"] },
  blocos: { name: "rotulos_blocos", cols: ["bloco", "inicio", "fim", "obs"] },
};
const TEAM_COLOR = { H: "#1f6fd1", A: "#ee2523", D: "#9aa3ad" };
const DONE_COLOR = "#3ddc84";
const BLOCK_S = 300;                 // a new block is 5 min long until its end is moved

const lab = { manuais: [], posse: [], blocos: [], sel: -1, bloco: -1, hist: [],
  ok: { manuais: false, posse: false, blocos: false }, keys: true };
let api = null;                      // {getFrame, setFrame, fps, redraw, setStatus, pause, isPlaying, showBlock}
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
  lab.hist.push({ frame: String(f), before: i >= 0 ? lab.posse[i].time : null });
  if (i >= 0) lab.posse[i].time = team;
  else lab.posse.push({ frame: String(f), time: team, obs: "" });
  lab.posse.sort((a, b) => num(a.frame) - num(b.frame));
  changed("posse");
}

function undo() {
  const h = lab.hist.pop();
  if (!h) return api.setStatus("posse: nada para desfazer");
  if (h.before) lab.posse.find((p) => p.frame === h.frame).time = h.before;
  else lab.posse = lab.posse.filter((p) => p.frame !== h.frame);
  api.setFrame(num(h.frame));
  changed("posse");
}

// ---- possession blocks -------------------------------------------------------------------------

const activeBlock = () => lab.blocos[lab.bloco];

function blockMarks(b) {
  const a = num(b.inicio), e = num(b.fim);
  return lab.posse.filter((p) => num(p.frame) >= a && num(p.frame) <= e);
}

function newBlock() {
  const f = api.getFrame();
  const id = Math.max(0, ...lab.blocos.map((b) => num(b.bloco) || 0)) + 1;
  lab.blocos.push({ bloco: String(id), inicio: String(f), fim: String(f + BLOCK_S * api.fps), obs: "" });
  lab.blocos.sort((a, b) => num(a.inicio) - num(b.inicio));
  lab.bloco = lab.blocos.findIndex((b) => b.bloco === String(id));
  api.showBlock();
  changed("blocos");
  api.setStatus("bloco novo: marque já quem tem a bola (H / A / D) e dê play");
}

function setBlockEdge(edge) {
  const b = activeBlock();
  if (!b) return api.setStatus("posse: crie ou escolha um bloco");
  const f = api.getFrame();
  if (edge === "inicio" ? f >= num(b.fim) : f <= num(b.inicio)) return api.setStatus("posse: o início precisa vir antes do fim");
  b[edge] = String(f);
  lab.blocos.sort((x, y) => num(x.inicio) - num(y.inicio));
  lab.bloco = lab.blocos.indexOf(b);
  api.showBlock();
  changed("blocos");
}

function selectBlock(i) {
  lab.bloco = i;
  const b = activeBlock();
  if (b) api.setFrame(num(b.inicio));
  api.showBlock();
  render();
  api.redraw();
}

function blockStatus(b, f) {
  const a = num(b.inicio), e = num(b.fim);
  const marks = blockMarks(b);
  const first = marks.length && num(marks[0].frame) <= a + api.fps;
  const pct = Math.round(100 * Math.min(1, Math.max(0, (f - a) / Math.max(1, e - a))));
  return { marks, first, pct, text: `bloco ${b.bloco} · ${mmss(a)}–${mmss(e)} · ${pct}% · ${marks.length} marcas`
    + (first ? "" : " · ⚠ marque quem tem a bola no início") };
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

  $("rotBlocos").innerHTML = lab.blocos.map((b, i) => {
    const s = blockStatus(b, i === lab.bloco ? api.getFrame() : num(b.inicio));
    return `<div class="rotitem ${i === lab.bloco ? "sel" : ""}" data-b="${i}">
      <span>#${b.bloco}</span><span>${mmss(num(b.inicio))}–${mmss(num(b.fim))}</span>
      <span class="help">${s.marks.length} marcas</span>
      <span class="st ${s.first ? "ok" : "meio"}">${s.first ? "✓" : "⚠"}</span>
      <button data-bdel="${i}" title="apagar o bloco (as marcas ficam)">✕</button></div>`;
  }).join("") || '<div class="help">nenhum bloco: vá ao início do trecho e clique em "novo bloco aqui"</div>';
  const b = activeBlock();
  $("rotBlocoStatus").textContent = b ? blockStatus(b, api.getFrame()).text : "";

  const shown = b ? blockMarks(b) : lab.posse;
  $("rotPosse").innerHTML = shown.slice().reverse().map((p) => `
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
  $("rotD").onclick = () => possession("D");
  $("rotUndo").onclick = undo;
  $("rotBlocoNovo").onclick = newBlock;
  $("rotBlocoIni").onclick = () => setBlockEdge("inicio");
  $("rotBlocoFim").onclick = () => setBlockEdge("fim");
  $("rotBlocoIr").onclick = () => { const b = activeBlock(); if (b) api.setFrame(num(b.inicio)); };
  $("rotBlocos").onclick = (e) => {
    if (e.target.dataset.bdel !== undefined) {
      lab.blocos.splice(+e.target.dataset.bdel, 1);
      lab.bloco = -1;
      return changed("blocos");
    }
    const it = e.target.closest(".rotitem");
    if (it) selectBlock(+it.dataset.b);
  };
  $("rotDownload").onclick = () => {
    for (const kind of ["manuais", "posse", "blocos"]) {
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
  await Promise.all([load("manuais"), load("posse"), load("blocos")]);
  if (lab.manuais.length) lab.sel = Math.max(0, lab.manuais.findIndex((r) => !status(r)));
  render();
  api.redraw();
}

// called on every frame: progress of the active block, and playback stops at its end
let lastPct = -1;
export function rotulosFrame(f) {
  const b = activeBlock();
  if (!b || !api) return;
  if (api.isPlaying() && f >= num(b.fim)) {
    api.pause();
    api.setStatus(`fim do bloco ${b.bloco}: confira as marcas ou crie o próximo bloco`);
  }
  const s = blockStatus(b, f);
  if (s.pct !== lastPct) { lastPct = s.pct; $("rotBlocoStatus").textContent = s.text; }
}

// [inicio, fim] of the active block, for the "bloco ativo" timeline zoom (null if none)
export function rotulosRange() {
  const b = activeBlock();
  if (!b) return null;
  const pad = 5 * (api ? api.fps : 25);
  return [Math.max(0, num(b.inicio) - pad), num(b.fim) + pad];
}

// true when the key was a labelling shortcut (the caller then ignores it)
export function rotulosKey(e) {
  if (!lab.keys || e.ctrlKey || e.metaKey || e.altKey) return false;
  const k = e.key.toLowerCase();
  const act = { i: () => mark("inicio_real"), f: () => mark("fim_real"), v: setViva, p: newStoppage,
    h: () => possession("H"), a: () => possession("A"), d: () => possession("D"), z: undo, enter: next }[k];
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
  // inside each block: the possession sequence as filled segments, and the block edges as brackets
  const inBlock = new Set();
  for (const blk of lab.blocos) {
    const s = num(blk.inicio), e = num(blk.fim);
    if (e < a || s > b) continue;
    const marks = blockMarks(blk);
    marks.forEach((p, k) => {
      inBlock.add(p);
      const f0 = num(p.frame), f1 = k + 1 < marks.length ? num(marks[k + 1].frame) : e;
      ctx.fillStyle = TEAM_COLOR[p.time] || "#888";
      ctx.fillRect(x(f0), row[0], Math.max(1, x(f1) - x(f0)), row[1]);
    });
    ctx.fillStyle = "#ffffff";
    for (const f of [s, e]) ctx.fillRect(x(f) - 1, row[0] - 3, 2, row[1] + 6);
  }
  for (const p of lab.posse) {
    const f = num(p.frame);
    if (inBlock.has(p) || f < a || f > b) continue;
    ctx.fillStyle = TEAM_COLOR[p.time] || "#888";
    ctx.fillRect(x(f) - 1, row[0] - 1, 3, row[1] + 2);
  }
}
