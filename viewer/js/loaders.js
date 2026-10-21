// Everything that turns a file into something indexed by video frame (0-based).

export const LABELS = ["X", "H", "A", "D"];
const LABEL_CODE = { X: 0, H: 1, A: 2, D: 3 };

export async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.arrayBuffer();
}

// ---- official ATD, produced by viewer/preprocess.py --------------------------------------------

export async function loadATD(source = "tactical") {
  const meta = await fetchJSON(`data/meta_${source}.json`);
  const atd = new Uint16Array(await fetchBuffer(`data/atd_${source}.bin`));
  const nObj = meta.slots.length + 1;
  const stride = nObj * 2;
  return {
    meta,
    nFrames: meta.n_frames,
    // [{id, team, x, y}] for one frame; team is H, A or B (ball)
    frame(f) {
      if (f < 0 || f >= meta.n_frames) return [];
      const out = [];
      const base = f * stride;
      for (let i = 0; i < nObj; i++) {
        const xq = atd[base + 2 * i];
        if (xq === meta.missing) continue;
        const yq = atd[base + 2 * i + 1];
        if (yq === meta.missing) continue;
        const isBall = i === nObj - 1;
        out.push({
          id: isBall ? "ball" : String(meta.slots[i]),
          team: isBall ? "B" : meta.slot_teams[i],
          x: xq / meta.scale,
          y: yq / meta.scale,
        });
      }
      return out;
    },
  };
}

export async function loadHomography(source = "tactical") {
  return new Float32Array(await fetchBuffer(`data/homography_${source}.bin`));
}

export async function loadSmartTagging() {
  try {
    return await fetchJSON("data/smart_tagging.json");
  } catch {
    return [];
  }
}

// ---- possession CSVs: frame,label --------------------------------------------------------------

export function parsePossession(text, nFrames) {
  const labels = new Uint8Array(nFrames); // 0 = X
  const lines = text.split(/\r?\n/);
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const comma = line.indexOf(",");
    if (comma < 0) continue;
    const f = +line.slice(0, comma);
    if (f >= 0 && f < nFrames) labels[f] = LABEL_CODE[line.slice(comma + 1).trim()] ?? 0;
  }
  return labels;
}

export async function fetchPossession(url, nFrames) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return parsePossession(await res.text(), nFrames);
}

// ---- extra tracking: frame,track_id,team,x,y ---------------------------------------------------
// team is H, A or B (ball); x, y normalised 0-1 like the ATD; frame is the 0-based video frame.

export function parseTrackingCSV(text) {
  const byFrame = new Map();
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(",").map((s) => s.trim().toLowerCase());
  const col = (name) => header.indexOf(name);
  const [cf, ci, ct, cx, cy] = ["frame", "track_id", "team", "x", "y"].map(col);
  if ([cf, ci, ct, cx, cy].some((c) => c < 0)) {
    throw new Error("CSV precisa das colunas frame,track_id,team,x,y");
  }
  let maxFrame = 0;
  for (let i = 1; i < lines.length; i++) {
    const p = lines[i].split(",");
    if (p.length < header.length) continue;
    const x = parseFloat(p[cx]);
    const y = parseFloat(p[cy]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const f = parseInt(p[cf], 10);
    const team = p[ct].trim().toUpperCase();
    let rows = byFrame.get(f);
    if (!rows) byFrame.set(f, (rows = []));
    rows.push({ id: team === "B" ? "ball" : p[ci].trim(), team, x, y });
    if (f > maxFrame) maxFrame = f;
  }
  return {
    nFrames: maxFrame + 1,
    frame: (f) => byFrame.get(f) || [],
  };
}
