/*
 * Decision-boundary rendering.
 *
 * Spec §6.4: render via ImageData, upscale, composite under the training
 * points, ring the misclassified ones.
 *
 * The grid is painted into an offscreen canvas at its native resolution and
 * then drawn scaled. Letting the browser do the upscale is both faster than
 * doing it per-pixel and gives smooth interpolation for free, which is what
 * makes a 120x120 grid read as a continuous surface rather than as tiles.
 */

import {
  classColor as classToken,
  COLORS,
  FONTS,
  parseHex,
  WEIGHT_NEGATIVE,
  WEIGHT_POSITIVE,
} from '../theme';
import type { Ctx2D } from './context';
import { TAU } from './context';

export interface BoundaryData {
  readonly values: Float32Array;
  readonly resolution: number;
  readonly classes: number;
}

export interface DataBounds {
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
}

const NEG = parseHex(WEIGHT_NEGATIVE);
const POS = parseHex(WEIGHT_POSITIVE);

/*
 * Class colours come from the validated categorical palette in theme.ts, and
 * are parsed once at module load rather than per pixel.
 *
 * Slots 0 and 1 ARE the weight poles, so a binary boundary speaks the same
 * visual language as the weights; further classes step down in lightness,
 * which is what keeps them separable under colour-vision deficiency.
 */
const CLASS_RGB: readonly { r: number; g: number; b: number }[] = [0, 1, 2, 3, 4].map((k) =>
  parseHex(classToken(k)),
);

function classColor(k: number): { r: number; g: number; b: number } {
  return CLASS_RGB[k % CLASS_RGB.length] as { r: number; g: number; b: number };
}

/**
 * Paint the grid into an ImageData.
 *
 * Alpha carries confidence, so the surface is strongest where the network is
 * sure and fades toward the boundary itself. That fade IS the decision
 * boundary: it is where the network is least certain, and drawing it as a hard
 * line would claim a precision the network does not have.
 */
export function paintBoundary(image: ImageData, data: BoundaryData): void {
  const { values, classes } = data;
  const pixels = image.data;

  for (let i = 0; i < values.length; i++) {
    const value = values[i] as number;
    let color: { r: number; g: number; b: number };
    let confidence: number;

    if (classes === 1) {
      // Sigmoid output: 0.5 is the boundary, distance from it is confidence.
      confidence = Math.abs(value - 0.5) * 2;
      color = value >= 0.5 ? POS : NEG;
    } else {
      const k = Math.floor(value);
      const probability = value - k;
      // Chance level for K classes is 1/K, so confidence is measured from there.
      confidence = Math.max(0, (probability - 1 / classes) / (1 - 1 / classes));
      color = classColor(k);
    }

    const o = i * 4;
    pixels[o] = color.r;
    pixels[o + 1] = color.g;
    pixels[o + 2] = color.b;
    pixels[o + 3] = Math.round(28 + 132 * Math.min(1, confidence));
  }
}

/** Data-space point to screen pixel within a plot rectangle. */
export function projectX(bounds: DataBounds, x: number, left: number, width: number): number {
  return left + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * width;
}

export function projectY(bounds: DataBounds, y: number, top: number, height: number): number {
  // Screen y grows downward.
  return top + (1 - (y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * height;
}

export interface PointStyle {
  readonly radius: number;
  /** Ring the points the network gets wrong (§6.4). */
  readonly showMisclassified: boolean;
}

/**
 * The training points on top of the surface.
 *
 * Misclassified points get a ring rather than a different colour: the colour
 * already means "which class this point belongs to", and overloading it would
 * make a wrong point unreadable as either.
 */
export function drawPoints(
  ctx: Ctx2D,
  x: Float64Array,
  labels: Int32Array,
  predictions: Int32Array | null,
  bounds: DataBounds,
  left: number,
  top: number,
  width: number,
  height: number,
  style: PointStyle,
): void {
  ctx.setLineDash([]);
  const count = labels.length;
  for (let i = 0; i < count; i++) {
    const px = projectX(bounds, x[i * 2] as number, left, width);
    const py = projectY(bounds, x[i * 2 + 1] as number, top, height);
    const label = labels[i] as number;
    const color = classColor(label);

    ctx.beginPath();
    ctx.arc(px, py, style.radius, 0, TAU);
    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.bgCanvas;
    ctx.stroke();

    if (style.showMisclassified && predictions !== null && predictions[i] !== label) {
      ctx.beginPath();
      ctx.arc(px, py, style.radius + 3, 0, TAU);
      ctx.strokeStyle = COLORS.focus;
      ctx.lineWidth = 1.25;
      ctx.stroke();
    }
  }
}

/** Bounds covering the data with a margin, so no point sits on the edge. */
export function boundsFor(x: Float64Array, count: number, margin = 0.15): DataBounds {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < count; i++) {
    const px = x[i * 2] as number;
    const py = x[i * 2 + 1] as number;
    if (px < xMin) xMin = px;
    if (px > xMax) xMax = px;
    if (py < yMin) yMin = py;
    if (py > yMax) yMax = py;
  }
  if (!Number.isFinite(xMin)) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };

  const padX = (xMax - xMin) * margin || 1;
  const padY = (yMax - yMin) * margin || 1;
  return { xMin: xMin - padX, xMax: xMax + padX, yMin: yMin - padY, yMax: yMax + padY };
}

/* ------------------------------------------------------------------ *
 * Confusion matrix (§4.10, §6.4)
 * ------------------------------------------------------------------ */

/**
 * Rows are the true class, columns the prediction, so the diagonal is correct.
 *
 * Cells are shaded by their share of the ROW rather than of the whole matrix.
 * With an imbalanced validation split, shading by the global maximum makes
 * every small class look empty and hides exactly the confusions worth seeing.
 */
export function drawConfusion(
  ctx: Ctx2D,
  x: number,
  y: number,
  size: number,
  confusion: readonly (readonly number[])[],
  labels: readonly string[],
): void {
  const n = confusion.length;
  if (n === 0) return;
  ctx.setLineDash([]);

  const gutter = 18;
  const cell = Math.max(8, (size - gutter) / n);
  const originX = x + gutter;
  const originY = y + gutter;

  ctx.font = `400 8px ${FONTS.mono}`;
  ctx.textBaseline = 'middle';

  for (let row = 0; row < n; row++) {
    const total = (confusion[row] as readonly number[]).reduce((a, b) => a + b, 0);
    for (let col = 0; col < n; col++) {
      const count = (confusion[row] as readonly number[])[col] ?? 0;
      const share = total === 0 ? 0 : count / total;
      const cx = originX + col * cell;
      const cy = originY + row * cell;

      // The diagonal is success and takes the warm pole; off-diagonal cells are
      // errors and take the error colour, so a confusion is legible at a glance.
      const base = row === col ? parseHex(COLORS.weightPositive) : parseHex(COLORS.statusBad);
      ctx.fillStyle = `rgba(${base.r}, ${base.g}, ${base.b}, ${(0.06 + 0.84 * share).toFixed(3)})`;
      ctx.fillRect(cx, cy, cell - 1, cell - 1);

      if (count > 0 && cell >= 16) {
        ctx.fillStyle = share > 0.5 ? COLORS.bgCanvas : COLORS.textMid;
        ctx.textAlign = 'center';
        ctx.fillText(String(count), cx + (cell - 1) / 2, cy + (cell - 1) / 2);
      }
    }
  }

  ctx.fillStyle = COLORS.textLo;
  for (let i = 0; i < n; i++) {
    const label = labels[i] ?? String(i);
    ctx.textAlign = 'right';
    ctx.fillText(label, originX - 4, originY + i * cell + (cell - 1) / 2);
    ctx.textAlign = 'center';
    ctx.fillText(label, originX + i * cell + (cell - 1) / 2, y + gutter / 2);
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}
