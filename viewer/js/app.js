import {
  LABELS, loadATD, loadHomography, loadSmartTagging, loadEvents, fetchPossession, parsePossession, parseTrackingCSV,
} from "./loaders.js";
import { Pitch } from "./pitch.js";
import { Overlay } from "./overlay.js";
import { initRotulos, rotulosKey, drawRotulos, rotulosFrame, rotulosRange } from "./rotulos.js";
import { initClipes, clipesRefresh, clipesFrame, clipesKey, drawClipes } from "./clipes.js";

const GT_URL = "../evaluation/gt/BAR-ATH_possession_gt_public.csv";
const RESULTS_URL = "../evaluation/results/";
const SERVED_VIDEO = "../video/tactical.mp4";
const GT_WINDOW = [11185, 18684];
// kickoffs, for the match clock (starter/load_smart_tagging.py KICKOFF_VIDEO_FRAMES)
const KICKOFFS = { 1: 11185, 2: 113742 };

const TEAM = {
  H: { name: "Barcelona", crest: "assets/barcelona.svg", color: "#a50044" },
  A: { name: "Athletic Club", crest: "assets/athletic.svg", color: "#ee2523" },
};
const LABEL_COLORS = ["#22272e", "#1f6fd1", "#ee2523", "#9aa3ad"]; // X H(azul Barça) A(vermelho Athletic) D
const TAG_COLORS = { Barcelona: "#1f6fd1", "Atletic Club": "#ee2523" };
// stoppages (analise/paradas.py) by restart type, and whistles
const STOP_COLORS = { apito: "#c0ca33", linha_de_fundo: "#ab47bc", lateral: "#26a69a", centro: "#ff7043",
  campo: "#8d6e63", desconhecida: "#78909c", sem_evento: "#546e7a", gol: "#ffffff", inicio_de_tempo: "#5c6bc0" };
const WHISTLE_COLOR = "#ffeb3b";

// Default look of each new layer; extra layers cycle through the alternatives so they stand out.
const LAYER_STYLES = [
  { colorH: "#2f86ff", colorA: "#ff2a2a", colorBall: "#ffe600", shape: "circle", size: 7 },
  { colorH: "#00e5ff", colorA: "#ffa000", colorBall: "#76ff03", shape: "ring", size: 10 },
  { colorH: "#b388ff", colorA: "#69f0ae", colorBall: "#ff4081", shape: "cross", size: 8 },
  { colorH: "#ffd740", colorA: "#40c4ff", colorBall: "#ff6e40", shape: "square", size: 6 },
];

const $ = (id) => document.getElementById(id);
const state = {
  frame: 0, playing: false, rate: 1, fps: 25, nFrames: 199319,
  hasVideo: false, gt: null, pred: null, predName: "", predAcc: null,
  tags: [], events: { apitos: [], paradas: [] }, layers: [], range: "gt", halves: [],
};
let pitch, overlay;

// ---- startup -------------------------------------------------------------------------------

async function init() {
  setStatus("carregando tracking oficial…");
  let atd, homography;
  try {
    [atd, homography, state.tags, state.events] = await Promise.all(
      [loadATD(), loadHomography(), loadSmartTagging(), loadEvents()]);
  } catch (err) {
    setStatus("⚠ dados não encontrados — rode python viewer/preprocess.py (" + err.message + ")");
    console.error(err);
    return;
  }
  state.fps = atd.meta.fps;
  state.nFrames = atd.meta.n_frames;
  state.halves = atd.meta.halves;
  $("scrub").max = state.nFrames - 1;

  pitch = new Pitch($("pitch"), atd.meta.pitch);
  overlay = new Overlay($("overlay"), homography);
  addLayer("ATD oficial (Metrica)", atd);

  try {
    state.gt = await fetchPossession(GT_URL, state.nFrames);
  } catch (err) {
    console.warn("ground truth:", err);
  }

  bindUI();
  onResize();
  tryServedVideo();
  setFrame(GT_WINDOW[0]);
  setStatus("pronto");
  initRotulos({
    getFrame: () => state.frame, setFrame: (f) => { if (state.playing) pause(); setFrame(f); }, fps: state.fps,
    redraw: () => { drawTimelineBase(); drawTimeline(); }, setStatus,
    pause, isPlaying: () => state.playing,
    showBlock: () => { state.range = "bloco"; $("range").value = "bloco"; drawTimelineBase(); drawTimeline(); },
  });
  await initClipes({
    getFrame: () => state.frame, setFrame: (f) => { if (state.playing) pause(); setFrame(f); },
    play, pause, isPlaying: () => state.playing, fps: state.fps, setStatus,
    redraw: () => { drawTimelineBase(); drawTimeline(); },
    getPred: () => ({ labels: state.pred, name: state.predName }),
    getParadas: () => state.events.paradas,
  });
  // ?pred=<name>: loads viewer/data/<name>.csv as the prediction (e.g. ?eventos=eventos_v3&pred=pred_v3)
  const predParam = new URLSearchParams(location.search).get("pred");
  if (predParam) {
    const name = predParam.replace(/[^\w-]/g, "");
    try {
      setPrediction(await fetchPossession(`data/${name}.csv`, state.nFrames), name);
    } catch (err) {
      setStatus(`⚠ ?pred=${name}: ${err.message}`);
    }
  }
}

function setStatus(text) { $("status").textContent = text; }

// ---- layers --------------------------------------------------------------------------------

function addLayer(name, data) {
  const base = LAYER_STYLES[state.layers.length % LAYER_STYLES.length];
  const layer = {
    name, data,
    style: { ...base, opacity: 0.9, showVideo: true, showPitch: true, showIds: false, trail: 0 },
  };
  state.layers.push(layer);
  renderLayerPanel();
  render();
}

function renderLayerPanel() {
  const root = $("layers");
  root.innerHTML = "";
  state.layers.forEach((layer, index) => {
    const s = layer.style;
    const el = document.createElement("div");
    el.className = "layer";
    el.innerHTML = `
      <div class="title"><span title="${layer.name}">${layer.name}</span>
        ${index ? '<button data-k="remove" title="remover">✕</button>' : ""}</div>
      <div class="row">
        <label><input type="checkbox" data-k="showVideo" ${s.showVideo ? "checked" : ""}> vídeo</label>
        <label><input type="checkbox" data-k="showPitch" ${s.showPitch ? "checked" : ""}> 2D</label>
        <label><input type="checkbox" data-k="showIds" ${s.showIds ? "checked" : ""}> IDs</label>
      </div>
      <div class="row">
        <label>H <input type="color" data-k="colorH" value="${s.colorH}"></label>
        <label>A <input type="color" data-k="colorA" value="${s.colorA}"></label>
        <label>bola <input type="color" data-k="colorBall" value="${s.colorBall}"></label>
      </div>
      <div class="row">
        <select data-k="shape">
          ${["circle", "ring", "cross", "square"].map((v) =>
            `<option value="${v}" ${v === s.shape ? "selected" : ""}>${
              { circle: "● círculo", ring: "○ anel", cross: "✕ cruz", square: "■ quadrado" }[v]}</option>`).join("")}
        </select>
        <label>tam <input type="range" data-k="size" min="2" max="18" value="${s.size}"></label>
      </div>
      <div class="row">
        <label>opac <input type="range" data-k="opacity" min="0.1" max="1" step="0.05" value="${s.opacity}"></label>
        <label>rastro <input type="range" data-k="trail" min="0" max="125" step="5" value="${s.trail}"></label>
      </div>`;
    el.addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      s[k] = e.target.type === "checkbox" ? e.target.checked
        : e.target.type === "range" ? +e.target.value : e.target.value;
      render();
    });
    el.addEventListener("click", (e) => {
      if (e.target.dataset.k !== "remove") return;
      state.layers.splice(index, 1);
      renderLayerPanel();
      render();
    });
    root.appendChild(el);
  });
}

// ---- possession ----------------------------------------------------------------------------

function setPrediction(labels, name) {
  state.pred = labels;
  state.predName = name;
  state.predAcc = null;
  if (state.gt) {
    let n = 0, ok = 0;
    for (let f = 0; f < state.nFrames; f++) {
      if (!state.gt[f]) continue;
      n++;
      if (labels[f] === state.gt[f]) ok++;
    }
    state.predAcc = n ? { ok, n } : null;
  }
  clipesRefresh();
  drawTimelineBase();
  render();
}

function renderPossession() {
  const f = state.frame;
  const code = state.gt ? state.gt[f] : 0;
  const label = LABELS[code];
  const box = $("possession");
  box.dataset.label = label;
  const crest = $("crest");
  if (TEAM[label]) {
    if (!crest.src.endsWith(TEAM[label].crest)) crest.src = TEAM[label].crest;
    crest.classList.remove("hidden");
  } else {
    crest.classList.add("hidden");
  }
  $("who").textContent = TEAM[label] ? TEAM[label].name
    : label === "D" ? "Bola morta" : state.gt ? "Sem gabarito" : "Gabarito não carregado";
  $("whoSub").textContent = TEAM[label] ? `Gabarito · posse ${label === "H" ? "Barcelona (H)" : "Athletic (A)"}`
    : label === "D" ? "Gabarito · bola fora de jogo (D)"
    : `Gabarito público só cobre os frames ${GT_WINDOW[0]}–${GT_WINDOW[1]}`;

  const predCrest = $("predCrest");
  if (!state.pred) {
    $("predVal").textContent = "sem predição";
    $("predAcc").textContent = "carregue um CSV frame,label";
    predCrest.style.visibility = "hidden";
    return;
  }
  const p = LABELS[state.pred[f]];
  predCrest.style.visibility = TEAM[p] ? "visible" : "hidden";
  if (TEAM[p]) predCrest.src = TEAM[p].crest;
  const mark = code === 0 ? "" : state.pred[f] === code ? ' <span class="ok">✓</span>' : ' <span class="bad">✗</span>';
  $("predVal").innerHTML = (TEAM[p] ? TEAM[p].name : p === "D" ? "Bola morta" : "X") + mark;
  $("predAcc").textContent = `${state.predName}` + (state.predAcc
    ? ` · acurácia no gabarito: ${(100 * state.predAcc.ok / state.predAcc.n).toFixed(1)}%` : "");
}

// ---- timeline ------------------------------------------------------------------------------

const timelineBase = document.createElement("canvas");
const ROWS = { gt: [2, 18], pred: [22, 16], err: [40, 5], tags: [48, 20], ev: [70, 12], lab: [86, 10] };

function viewRange() {
  const [h1, h2] = state.halves.length ? state.halves : [[1, 11185, 90651], [2, 110808, 190197]];
  switch (state.range) {
    case "h1": return [h1[1], h1[2]];
    case "h2": return [h2[1], h2[2]];
    case "all": return [0, state.nFrames - 1];
    case "bloco": return rotulosRange() || GT_WINDOW;
    default: return GT_WINDOW;
  }
}

function drawTimelineBase() {
  const canvas = $("timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  timelineBase.width = canvas.width;
  timelineBase.height = canvas.height;
  const ctx = timelineBase.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#12151a";
  ctx.fillRect(0, 0, w, h);
  const [a, b] = viewRange();
  const span = b - a + 1;
  for (let x = 0; x < w; x++) {
    const f = Math.min(b, Math.floor(a + (x / w) * span));
    const g = state.gt ? state.gt[f] : 0;
    ctx.fillStyle = LABEL_COLORS[g];
    ctx.fillRect(x, ROWS.gt[0], 1, ROWS.gt[1]);
    if (state.pred) {
      const p = state.pred[f];
      ctx.fillStyle = LABEL_COLORS[p];
      ctx.fillRect(x, ROWS.pred[0], 1, ROWS.pred[1]);
      if (g && p !== g) {
        ctx.fillStyle = "#ff3b3b";
        ctx.fillRect(x, ROWS.err[0], 1, ROWS.err[1]);
      }
    }
  }
  for (const t of state.tags) {
    if (t.end_frame < a || t.start_frame > b) continue;
    const x0 = ((t.start_frame - a) / span) * w;
    const x1 = ((t.end_frame - a) / span) * w;
    ctx.fillStyle = TAG_COLORS[t.team] || "#8b95a3";
    ctx.globalAlpha = 0.55;
    ctx.fillRect(x0, ROWS.tags[0], Math.max(1, x1 - x0), ROWS.tags[1]);
    ctx.globalAlpha = 1;
  }
  // stoppages (bars by type) and whistles (ticks) -- only when viewer/data/eventos.json exists
  for (const p of state.events.paradas || []) {
    if (p.end_frame < a || p.start_frame > b) continue;
    const x0 = ((p.start_frame - a) / span) * w;
    const x1 = ((p.end_frame - a) / span) * w;
    ctx.fillStyle = STOP_COLORS[p.tipo] || "#78909c";
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x0, ROWS.ev[0], Math.max(1, x1 - x0), ROWS.ev[1]);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = WHISTLE_COLOR;
  for (const [s] of state.events.apitos || []) {
    if (s < a || s > b) continue;
    ctx.fillRect(((s - a) / span) * w - 1, ROWS.ev[0] - 2, 2, ROWS.ev[1] + 4);
  }
  drawRotulos(ctx, a, b, w, ROWS.lab);
  drawClipes(ctx, a, b, w, ROWS.pred);
  ctx.fillStyle = "#8b95a3";
  ctx.font = "10px system-ui";
  ctx.fillText(`${a}`, 3, h - 2);
  ctx.textAlign = "right";
  ctx.fillText(`${b}`, w - 3, h - 2);
}

function drawTimeline() {
  const canvas = $("timeline");
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(timelineBase, 0, 0);
  const dpr = window.devicePixelRatio || 1;
  const [a, b] = viewRange();
  if (state.frame < a || state.frame > b) return;
  const x = ((state.frame - a) / (b - a + 1)) * canvas.width;
  ctx.fillStyle = "#fff";
  ctx.fillRect(Math.round(x) - dpr, 0, 2 * dpr, canvas.height);
}

function timelineSeek(e) {
  const rect = $("timeline").getBoundingClientRect();
  const [a, b] = viewRange();
  const t = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  setFrame(Math.round(a + t * (b - a)));
}

// ---- clock ---------------------------------------------------------------------------------

const video = $("video");

function setFrame(f, fromVideo = false) {
  f = Math.max(0, Math.min(state.nFrames - 1, Math.round(f)));
  state.frame = f;
  if (state.hasVideo && !fromVideo) {
    // aim at the middle of the frame so rounding never lands on its neighbour
    video.currentTime = (f + 0.5) / state.fps;
  }
  render();
}

function step(delta) {
  if (state.playing) pause();
  setFrame(state.frame + delta);
}

function play() {
  state.playing = true;
  $("play").textContent = "⏸ Pause";
  if (state.hasVideo) {
    video.playbackRate = state.rate;
    video.play().catch((err) => {
      if (err.name !== "NotAllowedError" || video.muted) return;
      video.muted = true;                    // the browser refused sound without a click: play muted
      updateSoundUI();
      setStatus("som bloqueado pelo navegador: clique em 🔇 para ligar");
      video.play().catch(() => {});
    });
  } else {
    let last = performance.now();
    let acc = 0;
    const tick = (now) => {
      if (!state.playing || state.hasVideo) return;
      acc += ((now - last) / 1000) * state.fps * state.rate;
      last = now;
      if (acc >= 1) {
        const n = Math.floor(acc);
        acc -= n;
        setFrame(state.frame + n);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

function pause() {
  state.playing = false;
  $("play").textContent = "▶ Play";
  if (state.hasVideo) video.pause();
}

function onVideoFrame(_now, metadata) {
  const f = Math.round(metadata.mediaTime * state.fps);
  if (f !== state.frame) setFrame(f, true);
  video.requestVideoFrameCallback(onVideoFrame);
}

// ---- sound: on by default, the choice is remembered per browser ---------------------------------

function loadSoundPrefs() {
  let prefs = { muted: false, volume: 0.8 };
  try { prefs = { ...prefs, ...JSON.parse(localStorage.getItem("viewer_som") || "{}") }; } catch { /* default */ }
  video.muted = prefs.muted;
  video.volume = Math.min(1, Math.max(0, prefs.volume));
  updateSoundUI();
}

function saveSoundPrefs() {
  try { localStorage.setItem("viewer_som", JSON.stringify({ muted: video.muted, volume: video.volume })); } catch { /* ignore */ }
}

function updateSoundUI() {
  const off = video.muted || video.volume === 0;
  $("mute").textContent = off ? "🔇" : "🔊";
  $("volume").value = video.volume;
}

function toggleSound() {
  video.muted = !video.muted;
  if (!video.muted && video.volume === 0) video.volume = 0.8;
  updateSoundUI();
  saveSoundPrefs();
}

function attachVideo(src, label) {
  video.src = src;
  video.addEventListener("loadedmetadata", () => {
    state.hasVideo = true;
    $("novideo").style.display = "none";
    setStatus(`vídeo: ${label}`);
    setFrame(state.frame);
  }, { once: true });
  video.addEventListener("error", () => {
    setStatus(`⚠ vídeo não pôde ser lido (${video.error ? video.error.message || video.error.code : "?"})`);
  }, { once: true });
  if ("requestVideoFrameCallback" in video) {
    video.requestVideoFrameCallback(onVideoFrame);
  } else {
    const poll = () => {
      if (state.hasVideo) {
        const f = Math.floor(video.currentTime * state.fps + 1e-3);
        if (f !== state.frame) setFrame(f, true);
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }
}

async function tryServedVideo() {
  try {
    const res = await fetch(SERVED_VIDEO, { method: "HEAD" });
    if (res.ok) attachVideo(SERVED_VIDEO, "servido pelo serve.py");
  } catch { /* no server-side video: the file picker still works */ }
}

// ---- render --------------------------------------------------------------------------------

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function matchClock(f) {
  for (const [half, first, last] of state.halves) {
    if (f >= first && f <= last) {
      const offset = half === 2 ? 45 * 60 : 0;
      return `${half}º T ${fmtTime(Math.max(0, offset + (f - KICKOFFS[half]) / state.fps))}`;
    }
  }
  return "fora de jogo";
}

function render() {
  if (!pitch) return;
  const f = state.frame;
  const layers = state.layers;
  pitch.draw(f, layers);
  overlay.draw(f, $("showOverlay").checked ? layers : [], $("showLines").checked);
  renderPossession();
  drawTimeline();
  rotulosFrame(f);
  clipesFrame(f);
  $("scrub").value = f;
  $("clock").textContent = `frame ${f} · ${fmtTime(f / state.fps)} · ${matchClock(f)}`;
  const active = state.tags.filter((t) => t.start_frame <= f && f < t.end_frame);
  $("tags").innerHTML = active.length
    ? active.map((t) => `<span class="tag ${t.team === "Barcelona" ? "H" : t.team === "Atletic Club" ? "A" : ""}">
        ${t.code}${t.type ? " · " + t.type : ""} · ${t.team || "?"}</span>`).join("")
    : '<span class="help">smart tagging: nenhuma fase ativa</span>';
}

function onResize() {
  pitch.resize();
  overlay.resize();
  drawTimelineBase();
  render();
}

// ---- UI bindings ---------------------------------------------------------------------------

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function parseTime(text) {
  const parts = text.trim().split(":").map(Number);
  if (parts.some(Number.isNaN)) return null;
  return parts.reduce((acc, v) => acc * 60 + v, 0);
}

function bindUI() {
  $("play").onclick = () => (state.playing ? pause() : play());
  document.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => step(+b.dataset.step)));
  $("rate").onchange = (e) => {
    state.rate = +e.target.value;
    video.playbackRate = state.rate;
  };
  $("scrub").oninput = (e) => setFrame(+e.target.value);
  $("gotoFrame").onchange = (e) => e.target.value !== "" && setFrame(+e.target.value);
  $("gotoTime").onchange = (e) => {
    const s = parseTime(e.target.value);
    if (s !== null) setFrame(s * state.fps);
  };
  $("goGT").onclick = () => setFrame(GT_WINDOW[0]);
  $("goH2").onclick = () => setFrame(state.halves[1] ? state.halves[1][1] : 110808);
  $("range").onchange = (e) => {
    state.range = e.target.value;
    drawTimelineBase();
    drawTimeline();
  };
  ["showOverlay", "showLines"].forEach((id) => ($(id).onchange = render));
  $("flipX").onchange = (e) => { pitch.flipX = e.target.checked; render(); };
  $("flipY").onchange = (e) => { pitch.flipY = e.target.checked; render(); };

  let dragging = false;
  const tl = $("timeline");
  tl.addEventListener("pointerdown", (e) => { dragging = true; tl.setPointerCapture(e.pointerId); timelineSeek(e); });
  tl.addEventListener("pointermove", (e) => dragging && timelineSeek(e));
  tl.addEventListener("pointerup", () => (dragging = false));

  video.addEventListener("click", () => (state.playing ? pause() : play()));
  video.addEventListener("pause", () => { if (state.playing) pause(); });

  $("videoFile").onchange = (e) => {
    const file = e.target.files[0];
    if (file) attachVideo(URL.createObjectURL(file), file.name);
  };
  $("trackFile").onchange = async (e) => {
    for (const file of e.target.files) {
      setStatus(`lendo ${file.name}…`);
      try {
        addLayer(file.name, parseTrackingCSV(await readFile(file)));
        setStatus(`camada adicionada: ${file.name}`);
      } catch (err) {
        setStatus(`⚠ ${file.name}: ${err.message}`);
      }
    }
    e.target.value = "";
  };
  $("predFile").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPrediction(parsePossession(await readFile(file), state.nFrames), file.name);
    e.target.value = "";
  };
  $("predBaseline").onchange = async (e) => {
    if (!e.target.value) return;
    setStatus(`carregando ${e.target.value}…`);
    try {
      setPrediction(await fetchPossession(`${RESULTS_URL}${e.target.value}.csv`, state.nFrames), e.target.value);
      setStatus("pronto");
    } catch (err) {
      setStatus(`⚠ ${err.message}`);
    }
  };

  loadSoundPrefs();
  $("mute").onclick = toggleSound;
  $("volume").oninput = (e) => {
    video.volume = +e.target.value;
    video.muted = video.volume === 0;
    updateSoundUI();
    saveSoundPrefs();
  };

  window.addEventListener("resize", onResize);
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input[type=text], input[type=number], select")) return;
    if (clipesKey(e) || rotulosKey(e)) return;
    const big = e.ctrlKey || e.metaKey ? 250 : e.shiftKey ? 25 : 1;
    if (e.code === "Space") { e.preventDefault(); state.playing ? pause() : play(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(big); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-big); }
    else if (e.key === "g" || e.key === "G") setFrame(GT_WINDOW[0]);
    else if (e.key === "m" || e.key === "M") toggleSound();
    else if (e.key === "n" || e.key === "N") jumpToEvent(e.shiftKey ? -1 : 1);
    else if (e.key === "[" || e.key === "]") {
      const sel = $("rate");
      sel.selectedIndex = Math.max(0, Math.min(sel.options.length - 1, sel.selectedIndex + (e.key === "]" ? 1 : -1)));
      sel.onchange({ target: sel });
    }
  });
}

// next/previous stoppage start or whistle (N / Shift+N), for checking and hand-labelling
function jumpToEvent(dir) {
  const marks = [...(state.events.paradas || []).map((p) => p.start_frame),
                 ...(state.events.apitos || []).map((x) => x[0])].sort((x, y) => x - y);
  const f = state.frame;
  const target = dir > 0 ? marks.find((m) => m > f + 1) : [...marks].reverse().find((m) => m < f - 1);
  if (target !== undefined) setFrame(Math.max(0, target - 3 * state.fps));   // 3 s before, to see it coming
}

init();
