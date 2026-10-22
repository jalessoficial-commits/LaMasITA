// 2D tactical map, plus the marker drawing shared with the video overlay.

const LINE = "rgba(255,255,255,0.85)";

// One marker in canvas pixels, in the layer's style.
export function drawMarker(ctx, px, py, obj, style) {
  const isBall = obj.team === "B";
  const color = isBall ? style.colorBall : obj.team === "H" ? style.colorH : style.colorA;
  const r = isBall ? Math.max(3, style.size * 0.6) : style.size;
  ctx.globalAlpha = style.opacity;
  ctx.lineWidth = Math.max(1.5, r * 0.3);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  if (style.shape === "cross") {
    ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r);
    ctx.moveTo(px + r, py - r); ctx.lineTo(px - r, py + r);
    ctx.stroke();
  } else if (style.shape === "square") {
    ctx.rect(px - r, py - r, 2 * r, 2 * r);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 1; ctx.stroke();
  } else {
    ctx.arc(px, py, r, 0, Math.PI * 2);
    if (style.shape === "ring") ctx.stroke();
    else { ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 1; ctx.stroke(); }
  }
  if (style.showIds && !isBall) {
    ctx.globalAlpha = 1;
    ctx.font = `bold ${Math.max(9, r * 1.1)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    ctx.strokeText(obj.id, px, py - r - 1);
    ctx.fillStyle = "#fff";
    ctx.fillText(obj.id, px, py - r - 1);
  }
  ctx.globalAlpha = 1;
}

// Trails: for each id, a polyline over the last `style.trail` frames.
export function drawTrails(ctx, layer, frame, toPx) {
  const n = layer.style.trail;
  if (!n) return;
  const paths = new Map();
  for (let f = Math.max(0, frame - n); f <= frame; f++) {
    for (const o of layer.data.frame(f)) {
      const p = toPx(o.x, o.y, f);
      if (!p) continue;
      let entry = paths.get(o.id);
      if (!entry) paths.set(o.id, (entry = { team: o.team, pts: [] }));
      entry.pts.push(p);
    }
  }
  const s = layer.style;
  ctx.lineWidth = 2;
  ctx.globalAlpha = s.opacity * 0.6;
  for (const { team, pts } of paths.values()) {
    if (pts.length < 2) continue;
    ctx.strokeStyle = team === "B" ? s.colorBall : team === "H" ? s.colorH : s.colorA;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export class Pitch {
  constructor(canvas, pitchSize) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    [this.L, this.W] = pitchSize; // metres
    this.flipX = false;
    this.flipY = false;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const margin = 14;
    const scale = Math.min((rect.width - 2 * margin) / this.L, (rect.height - 2 * margin) / this.W);
    this.scale = scale;
    this.ox = (rect.width - this.L * scale) / 2;
    this.oy = (rect.height - this.W * scale) / 2;
    this.cssW = rect.width;
    this.cssH = rect.height;
  }

  // normalised 0-1 pitch coordinates -> canvas css pixels
  toPx(x, y) {
    const nx = this.flipX ? 1 - x : x;
    const ny = this.flipY ? 1 - y : y;
    return [this.ox + nx * this.L * this.scale, this.oy + ny * this.W * this.scale];
  }

  drawField() {
    const { ctx, L, W, scale: s, ox, oy } = this;
    ctx.fillStyle = "#1d5e33";
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    // mowing stripes
    for (let i = 0; i < 12; i++) {
      if (i % 2) continue;
      ctx.fillStyle = "#226a3a";
      ctx.fillRect(ox + (i * L / 12) * s, oy, (L / 12) * s, W * s);
    }
    const m = (x, y) => [ox + x * s, oy + y * s];
    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, L * s, W * s);
    ctx.beginPath();
    ctx.moveTo(...m(L / 2, 0)); ctx.lineTo(...m(L / 2, W));
    ctx.stroke();
    ctx.beginPath(); ctx.arc(...m(L / 2, W / 2), 9.15 * s, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = LINE;
    ctx.beginPath(); ctx.arc(...m(L / 2, W / 2), 2, 0, Math.PI * 2); ctx.fill();
    for (const side of [0, 1]) {
      const x0 = side ? L : 0;
      const dir = side ? -1 : 1;
      const box = (depth, width) => {
        const [ax, ay] = m(x0, W / 2 - width / 2);
        ctx.strokeRect(Math.min(ax, ax + dir * depth * s), ay, depth * s, width * s);
      };
      box(16.5, 40.32);
      box(5.5, 18.32);
      ctx.beginPath(); ctx.arc(...m(x0 + dir * 11, W / 2), 2, 0, Math.PI * 2); ctx.fill();
      // penalty arc, the part outside the box
      const a = Math.acos(5.5 / 9.15);
      ctx.beginPath();
      if (side) ctx.arc(...m(x0 - 11, W / 2), 9.15 * s, Math.PI - a, Math.PI + a);
      else ctx.arc(...m(x0 + 11, W / 2), 9.15 * s, -a, a);
      ctx.stroke();
      // goal
      const [gx, gy] = m(x0, W / 2 - 3.66);
      ctx.fillRect(side ? gx : gx - 4, gy, 4, 7.32 * s);
    }
  }

  draw(frame, layers) {
    this.drawField();
    const toPx = (x, y) => this.toPx(x, y);
    for (const layer of layers) {
      if (!layer.style.showPitch) continue;
      drawTrails(this.ctx, layer, frame, toPx);
      const objs = layer.data.frame(frame);
      // ball last so it sits on top
      for (const o of objs) if (o.team !== "B") drawMarker(this.ctx, ...toPx(o.x, o.y), o, layer.style);
      for (const o of objs) if (o.team === "B") drawMarker(this.ctx, ...toPx(o.x, o.y), o, layer.style);
    }
  }
}
