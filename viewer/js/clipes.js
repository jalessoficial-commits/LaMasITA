// "Clipes D": every dead-ball stretch of the loaded prediction as a clip, to review them one by one.
// The clips are the D runs of the prediction (🎯 or ?pred=<name>); without a prediction, the stoppages
// of eventos_*.json. Each clip shows its start, duration and the stoppage type that opened it.
//
// Keys: K next clip · J previous · R replay · 1 ✓ right · 2 ✗ false (it was live play) ·
// the buttons "início real aqui" / "fim real aqui" mark the true edges ("ajustado").
// Verdicts are saved through serve.py (/api/rotulos) in analise/out/rotulos_clipes.csv, keyed by the
// prediction name and the clip start, so analise/avaliar_v3.py can score any detector against them.

import { parseCSV, toCSV } from "./rotulos.js";

const API = "../api/rotulos?arquivo=rotulos_clipes";
const COLS = ["inicio", "fim", "fonte", "veredito", "inicio_real", "fim_real", "obs"];
const PRE_S = 3;                     // a clip starts playing 3 s before the dead ball
const POST_S = 2;                    // ... and stops 2 s after it
const D_CODE = 3;                    // LABELS = X H A D
const MIN_CLIP_S = 0.4;
const VERDICT = { certo: ["✓", "#3ddc84"], falso: ["✗", "#ff5252"], ajustado: ["±", "#ffb74d"] };

const st = { clips: [], sel: -1, fonte: "", rows: [], ok: false, playingClip: false };
let api = null;   // {getFrame, setFrame, play, pause, isPlaying, fps, setStatus, redraw, getPred, getParadas}
let timer = null;
const $ = (id) => document.getElementById(id);

const num = (v) => (v === "" || v == null || isNaN(+v) ? null : +v);
function mmss(f) {
  const s = f / api.fps;
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

// ---- storage -----------------------------------------------------------------------------------

async function load() {
  try {
    const res = await fetch(API, { cache: "no-store" });
    if (res.status === 404) { st.rows = []; st.ok = true; return; }   // nenhum veredito ainda: o 1º POST cria
    if (!res.ok) throw new Error(`${res.status}`);
    st.rows = parseCSV(await res.text());
    st.ok = true;
  } catch (err) {
    st.rows = [];
    st.ok = false;
    setInfo(`⚠ rotulos_clipes.csv: ${err.message} (rode viewer/serve.py)`);
  }
}

function save() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    try {
      const res = await fetch(API, { method: "POST", body: toCSV(COLS, st.rows) });
      if (!res.ok) throw new Error(`${res.status}`);
      $("clipSave").textContent = `salvo ✓ ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      $("clipSave").textContent = `⚠ não salvo (${err.message})`;
    }
  }, 300);
}

function rowOf(clip, create = false) {
  let r = st.rows.find((x) => x.fonte === st.fonte && num(x.inicio) === clip.s);
  if (!r && create) {
    r = { inicio: String(clip.s), fim: String(clip.e), fonte: st.fonte, veredito: "", inicio_real: "", fim_real: "", obs: "" };
    st.rows.push(r);
  }
  return r;
}

// ---- clips -------------------------------------------------------------------------------------

function buildClips() {
  const pred = api.getPred();
  const paradas = (api.getParadas() || []).slice().sort((a, b) => a.start_frame - b.start_frame);
  const clips = [];
  if (pred.labels) {
    st.fonte = pred.name || "predição";
    const L = pred.labels;
    let s = -1;
    for (let f = 0; f <= L.length; f++) {
      const d = f < L.length && L[f] === D_CODE;
      if (d && s < 0) s = f;
      else if (!d && s >= 0) {
        if (f - s >= MIN_CLIP_S * api.fps) clips.push({ s, e: f });
        s = -1;
      }
    }
  } else {
    st.fonte = new URLSearchParams(location.search).get("eventos") || "eventos";
    for (const p of paradas) clips.push({ s: p.start_frame, e: p.end_frame });
  }
  // the stoppage that opened each clip (largest overlap), for its type
  for (const c of clips) {
    let best = null, bestOv = 0;
    for (const p of paradas) {
      if (p.start_frame > c.e) break;
      const ov = Math.min(c.e, p.end_frame) - Math.max(c.s, p.start_frame);
      if (ov > bestOv) { bestOv = ov; best = p; }
    }
    c.tipo = best ? best.tipo : "—";
    c.fonteIni = best ? best.fonte_inicio : "";
  }
  st.clips = clips;
  st.sel = -1;
}

function visible() {
  const only = $("clipPend").checked;
  return st.clips.map((c, i) => [c, i]).filter(([c]) => !only || !(rowOf(c) || {}).veredito);
}

function select(i, playIt = true) {
  if (i < 0 || i >= st.clips.length) return;
  st.sel = i;
  const c = st.clips[i];
  api.setFrame(Math.max(0, c.s - PRE_S * api.fps));
  if (playIt) { st.playingClip = true; api.play(); }
  render();
  const el = document.querySelector("#clipList .rotitem.sel");
  if (el) el.scrollIntoView({ block: "nearest" });
}

function move(dir) {
  const vis = visible().map(([, i]) => i);
  if (!vis.length) return api.setStatus("clipes: nenhum clipe (carregue uma predição ou ?eventos=)");
  let target;
  if (st.sel < 0) {
    const f = api.getFrame();
    target = dir > 0 ? vis.find((i) => st.clips[i].s > f) : [...vis].reverse().find((i) => st.clips[i].s < f);
  } else {
    target = dir > 0 ? vis.find((i) => i > st.sel) : [...vis].reverse().find((i) => i < st.sel);
  }
  if (target === undefined) return api.setStatus(dir > 0 ? "clipes: fim da lista" : "clipes: começo da lista");
  select(target);
}

function verdict(v) {
  const c = st.clips[st.sel];
  if (!c) return api.setStatus("clipes: escolha um clipe (K / J ou clique na lista)");
  const r = rowOf(c, true);
  r.veredito = r.veredito === v ? "" : v;
  if (v === "falso") { r.inicio_real = ""; r.fim_real = ""; }
  changed();
}

function markEdge(field) {
  const c = st.clips[st.sel];
  if (!c) return api.setStatus("clipes: escolha um clipe primeiro");
  const r = rowOf(c, true);
  r[field] = String(api.getFrame());
  r.veredito = "ajustado";
  changed();
}

function changed() {
  save();
  render();
  api.redraw();
}

// ---- UI ----------------------------------------------------------------------------------------

function setInfo(text) { const el = $("clipInfo"); if (el) el.textContent = text; }

function render() {
  const n = st.clips.length;
  const done = st.clips.filter((c) => (rowOf(c) || {}).veredito).length;
  const falsos = st.clips.filter((c) => (rowOf(c) || {}).veredito === "falso").length;
  setInfo(n ? `${st.fonte}: ${n} clipes · revisados ${done} · falsos ${falsos}`
    : "nenhum clipe: carregue uma predição (🎯) ou abra com ?pred=pred_v3");
  $("clipList").innerHTML = visible().map(([c, i]) => {
    const r = rowOf(c) || {};
    const [sym, col] = VERDICT[r.veredito] || ["", ""];
    return `<div class="rotitem${i === st.sel ? " sel" : ""}" data-i="${i}">
      <span>#${i + 1}</span><span>${mmss(c.s)}</span><span>${((c.e - c.s) / api.fps).toFixed(1)} s</span>
      <span class="help">${c.tipo}</span><span class="st" style="color:${col}">${sym}</span></div>`;
  }).join("");
  const c = st.clips[st.sel];
  const r = c ? rowOf(c) || {} : {};
  $("clipCur").innerHTML = c
    ? `<b>#${st.sel + 1}</b> frames ${c.s}–${c.e} (${((c.e - c.s) / api.fps).toFixed(1)} s) · ${c.tipo}`
      + (c.fonteIni ? ` · início por ${c.fonteIni}` : "")
      + (r.inicio_real || r.fim_real ? `<br>real: ${r.inicio_real || "?"}–${r.fim_real || "?"}` : "")
    : "";
  const obs = $("clipObs");
  if (document.activeElement !== obs) obs.value = r.obs || "";
  obs.disabled = !c;
}

function bind() {
  $("clipList").addEventListener("click", (e) => {
    const it = e.target.closest(".rotitem");
    if (it) select(+it.dataset.i);
  });
  $("clipPrev").onclick = () => move(-1);
  $("clipNext").onclick = () => move(1);
  $("clipReplay").onclick = () => select(st.sel);
  $("clipOk").onclick = () => verdict("certo");
  $("clipBad").onclick = () => verdict("falso");
  $("clipIni").onclick = () => markEdge("inicio_real");
  $("clipFim").onclick = () => markEdge("fim_real");
  $("clipPend").onchange = render;
  $("clipObs").onchange = (e) => {
    const c = st.clips[st.sel];
    if (!c) return;
    rowOf(c, true).obs = e.target.value;
    changed();
  };
}

// ---- public ------------------------------------------------------------------------------------

export async function initClipes(hooks) {
  api = hooks;
  bind();
  await load();
  buildClips();
  render();
}

// the prediction or the events changed: rebuild the list (verdicts stay, keyed by prediction name)
export function clipesRefresh() {
  if (!api) return;
  buildClips();
  render();
}

// every frame: stop at the end of the clip being played
export function clipesFrame(f) {
  const c = st.clips[st.sel];
  if (!c || !st.playingClip || !api.isPlaying() || !$("clipStop").checked) return;
  if (f >= c.e + POST_S * api.fps) {
    st.playingClip = false;
    api.pause();
    api.setStatus(`fim do clipe #${st.sel + 1}: 1 ✓ certo · 2 ✗ falso · K próximo · R repetir`);
  }
}

export function clipesKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  const act = { k: () => move(1), j: () => move(-1), r: () => select(st.sel),
    1: () => verdict("certo"), 2: () => verdict("falso") }[e.key.toLowerCase()];
  if (!act) return false;
  e.preventDefault();
  act();
  return true;
}

// ticks on the prediction row: verdict of each reviewed clip, and a bracket on the selected one
export function drawClipes(ctx, a, b, w, row) {
  const span = b - a + 1;
  const x = (f) => ((f - a) / span) * w;
  st.clips.forEach((c, i) => {
    if (c.e < a || c.s > b) return;
    const r = rowOf(c);
    if (r && VERDICT[r.veredito]) {
      ctx.fillStyle = VERDICT[r.veredito][1];
      ctx.fillRect(x(c.s), row[0], Math.max(2, x(c.e) - x(c.s)), 3);
    }
    if (i === st.sel) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.strokeRect(x(c.s) - 1, row[0] - 1, Math.max(3, x(c.e) - x(c.s)) + 2, row[1] + 2);
    }
  });
}
