// Tracking drawn on top of the video, through the per-frame homography (pitch 0-1 -> pixels).

import { drawMarker, drawTrails } from "./pitch.js";

const VIDEO_W = 1920;
const VIDEO_H = 1080;

export class Overlay {
  constructor(canvas, homography) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.H = homography; // Float32Array, 9 per frame, NaN when invalid
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.sx = rect.width / VIDEO_W;
    this.sy = rect.height / VIDEO_H;
    this.cssW = rect.width;
    this.cssH = rect.height;
  }

  project(x, y, frame) {
    const h = this.H;
    const b = frame * 9;
    if (!this.H || b + 8 >= h.length || Number.isNaN(h[b])) return null;
    const d = h[b + 6] * x + h[b + 7] * y + h[b + 8];
    if (d <= 0) return null;
    return [
      ((h[b] * x + h[b + 1] * y + h[b + 2]) / d) * this.sx,
      ((h[b + 3] * x + h[b + 4] * y + h[b + 5]) / d) * this.sy,
    ];
  }

  draw(frame, layers, showPitchLines) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    if (showPitchLines) this.drawPitchOutline(frame);
    const toPx = (x, y, f) => this.project(x, y, f);
    for (const layer of layers) {
      if (!layer.style.showVideo) continue;
      drawTrails(ctx, layer, frame, toPx);
      const objs = layer.data.frame(frame);
      for (const o of [...objs.filter((o) => o.team !== "B"), ...objs.filter((o) => o.team === "B")]) {
        const p = this.project(o.x, o.y, frame);
        if (p) drawMarker(ctx, p[0], p[1], o, layer.style);
      }
    }
  }

  // Touchlines and halfway line, sampled so the projection's curvature-free lines stay exact.
  drawPitchOutline(frame) {
    const ctx = this.ctx;
    const segs = [
      [[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]], [[0.5, 0], [0.5, 1]],
    ];
    ctx.strokeStyle = "rgba(0,255,200,0.55)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    for (const [[x0, y0], [x1, y1]] of segs) {
      const a = this.project(x0, y0, frame);
      const b = this.project(x1, y1, frame);
      if (!a || !b) continue;
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
}
