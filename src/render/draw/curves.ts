/*
 * Loss and accuracy curves.
 *
 * Spec §6.4: train and validation, log-scale toggle, epoch markers, early-stop
 * marker. Drawn on canvas rather than with a charting library, per §2, which
 * also means the curves use the same palette rules as everything else.
 *
 * Train and validation are told apart by LIGHTNESS AND DASH, never by hue:
 * §9 reserves the two hues for weight sign, and a coloured series here would be
 * read as a weight value.
 */

import { COLORS, FONTS, formatLoss, parseHex } from '../theme';
import type { Ctx2D } from './context';

export interface Series {
  readonly values: readonly number[];
  readonly label: string;
  readonly dashed: boolean;
  readonly dim: boolean;
  /**
   * Position in a sequential ramp, 0 (faint) to 1 (bright).
   *
   * Used where the series index is ORDINAL rather than categorical: per-layer
   * gradient norms are ordered by depth, so a light-to-dark ramp is the correct
   * encoding and it costs no hue budget. Categorical hues would be both wrong
   * for ordered data and in breach of §9.
   */
  readonly tone?: number | undefined;
}

export interface ChartOptions {
  readonly logScale: boolean;
  /**
   * Never extend the axis below zero when padding.
   *
   * A loss cannot be negative, so an axis tick reading −0.0373 states something
   * impossible about the quantity being plotted.
   */
  readonly nonNegative?: boolean | undefined;
  /** Epoch to mark, for example where early stopping fired. */
  readonly marker?: number | undefined;
  readonly markerLabel?: string | undefined;
  readonly yLabel?: string | undefined;
  /** Clamp the y-axis, for accuracy which is always 0..1. */
  readonly yRange?: readonly [number, number] | undefined;
  /**
   * Window for a centred moving average, in samples. 0 disables.
   *
   * Per-layer gradient norms are measured on the last batch of each epoch, so
   * they fluctuate by an order of magnitude between neighbouring epochs. Drawn
   * raw at one pixel per epoch, eight layers become a single grey mass and the
   * thing worth seeing — that layer 1 receives a thousandth of what layer 8
   * does — is lost inside the noise. Smoothing shows the trend, and the axis
   * label says it is smoothed.
   */
  readonly smooth?: number | undefined;
}

/** Centred moving average, preserving length and skipping non-finite samples. */
export function movingAverage(values: readonly number[], window: number): number[] {
  if (window <= 1 || values.length === 0) return [...values];
  const half = Math.floor(window / 2);
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j] as number;
      if (!Number.isFinite(v)) continue;
      sum += v;
      count++;
    }
    // A window of entirely non-finite samples stays non-finite, so a divergence
    // still breaks the line rather than being smoothed away.
    out.push(count === 0 ? (values[i] as number) : sum / count);
  }
  return out;
}

const PAD_LEFT = 44;
const PAD_RIGHT = 10;
const PAD_TOP = 16;
const PAD_BOTTOM = 20;

/** Smallest positive value plotted on a log axis. */
const LOG_FLOOR = 1e-6;

export function drawChart(
  ctx: Ctx2D,
  x: number,
  y: number,
  width: number,
  height: number,
  series: readonly Series[],
  options: ChartOptions,
): void {
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.bgCanvas;
  ctx.fillRect(x, y, width, height);

  const plotLeft = x + PAD_LEFT;
  const plotTop = y + PAD_TOP;
  const plotWidth = Math.max(1, width - PAD_LEFT - PAD_RIGHT);
  const plotHeight = Math.max(1, height - PAD_TOP - PAD_BOTTOM);

  const finite = (v: number): boolean => Number.isFinite(v);
  const window = options.smooth ?? 0;
  const plotted: Series[] =
    window > 1 ? series.map((s) => ({ ...s, values: movingAverage(s.values, window) })) : [...series];

  let maxLength = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of plotted) {
    maxLength = Math.max(maxLength, s.values.length);
    for (const v of s.values) {
      if (!finite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }

  if (options.yRange !== undefined) {
    lo = options.yRange[0];
    hi = options.yRange[1];
  }
  if (!finite(lo) || !finite(hi)) {
    drawEmpty(ctx, x, y, width, height);
    return;
  }
  if (options.logScale) {
    lo = Math.max(LOG_FLOOR, lo);
    hi = Math.max(lo * 10, hi);
  } else if (hi - lo < 1e-9) {
    // A flat series would otherwise divide by zero; give it a visible band.
    hi = lo + 1;
  } else {
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    if (options.nonNegative === true && lo < 0) lo = 0;
  }

  const toY = (v: number): number => {
    if (options.logScale) {
      const clamped = Math.max(LOG_FLOOR, v);
      const t = (Math.log10(clamped) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
      return plotTop + (1 - t) * plotHeight;
    }
    return plotTop + (1 - (v - lo) / (hi - lo)) * plotHeight;
  };
  const toX = (i: number): number =>
    plotLeft + (maxLength <= 1 ? 0 : (i / (maxLength - 1)) * plotWidth);

  /* Axes and ticks. */
  ctx.strokeStyle = COLORS.lineGrid;
  ctx.lineWidth = 1;
  ctx.font = `400 9px ${FONTS.mono}`;
  ctx.fillStyle = COLORS.textLo;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  const ticks = options.logScale
    ? logTicks(lo, hi)
    : [lo, lo + (hi - lo) / 2, hi];
  for (const tick of ticks) {
    const ty = Math.round(toY(tick)) + 0.5;
    if (ty < plotTop - 1 || ty > plotTop + plotHeight + 1) continue;
    ctx.beginPath();
    ctx.moveTo(plotLeft, ty);
    ctx.lineTo(plotLeft + plotWidth, ty);
    ctx.stroke();
    ctx.fillText(tickLabel(tick, options.logScale), plotLeft - 6, ty);
  }

  /* The marker, drawn under the series so it never obscures data. */
  if (options.marker !== undefined && maxLength > 1) {
    const mx = Math.round(toX(options.marker)) + 0.5;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = COLORS.lineEdge;
    ctx.beginPath();
    ctx.moveTo(mx, plotTop);
    ctx.lineTo(mx, plotTop + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    if (options.markerLabel !== undefined) {
      ctx.textAlign = 'left';
      ctx.fillStyle = COLORS.textLo;
      ctx.fillText(options.markerLabel, mx + 4, plotTop + 6);
    }
  }

  /* Series. */
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const s of plotted) {
    if (s.values.length === 0) continue;
    ctx.strokeStyle = seriesStroke(s);
    ctx.lineWidth = s.dim ? 1 : 1.5;
    ctx.setLineDash(s.dashed ? [4, 3] : []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.values.length; i++) {
      const v = s.values[i] as number;
      if (!finite(v)) {
        // A NaN breaks the line rather than being interpolated across, so a
        // divergence is visible as a curve that stops (§7.4).
        started = false;
        continue;
      }
      const px = toX(i);
      const py = toY(v);
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  /*
   * Legend, bottom-left, inside the plot.
   *
   * Beyond four series only the first and last are labelled. Eight labels do
   * not fit and would run off the edge, and with an ordinal ramp the ends are
   * what carry the meaning: everything between them is interpolated, and the
   * reader can see that from the ramp itself.
   */
  ctx.font = `400 9px ${FONTS.mono}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  const drawn = plotted.filter((s) => s.values.length > 0);
  const legendSeries =
    drawn.length > 4 ? [drawn[0] as Series, drawn[drawn.length - 1] as Series] : drawn;
  let legendX = plotLeft + 2;
  for (const s of legendSeries) {
    ctx.strokeStyle = seriesStroke(s);
    ctx.lineWidth = s.dim ? 1 : 1.5;
    ctx.setLineDash(s.dashed ? [3, 2] : []);
    ctx.beginPath();
    ctx.moveTo(legendX, y + height - 6);
    ctx.lineTo(legendX + 12, y + height - 6);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.textLo;
    const label =
      drawn.length > 4
        ? `${s.label} ${s === legendSeries[0] ? '(first)' : '(last)'}`
        : s.label;
    ctx.fillText(label, legendX + 16, y + height - 2);
    legendX += 16 + label.length * 5.4 + 12;
  }

  if (options.yLabel !== undefined) {
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = COLORS.textLo;
    ctx.fillText(options.yLabel, x + 4, y + 4);
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

/**
 * A series' stroke: a neutral ink, or a step on the sequential ramp.
 *
 * The ramp runs through the text tokens rather than introducing a hue, so an
 * eight-layer gradient chart still leaves the two weight poles unique.
 */
function seriesStroke(s: Series): string {
  if (s.tone === undefined) return s.dim ? COLORS.textLo : COLORS.textHi;
  const t = Math.max(0, Math.min(1, s.tone));
  // Spans lineEdge to focus rather than textLo to textHi: the narrower text
  // range could not separate eight layers, and depth is the whole point here.
  const lo = parseHex(COLORS.lineEdge);
  const hi = parseHex(COLORS.focus);
  const mix = (a: number, b: number): number => Math.round(a + (b - a) * t);
  return `rgb(${mix(lo.r, hi.r)}, ${mix(lo.g, hi.g)}, ${mix(lo.b, hi.b)})`;
}

function drawEmpty(ctx: Ctx2D, x: number, y: number, width: number, height: number): void {
  ctx.font = `400 10px ${FONTS.body}`;
  ctx.fillStyle = COLORS.textLo;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('No data yet', x + width / 2, y + height / 2);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

function logTicks(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  const start = Math.floor(Math.log10(lo));
  const end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e++) {
    const v = Math.pow(10, e);
    if (v >= lo && v <= hi) ticks.push(v);
  }
  return ticks.length >= 2 ? ticks : [lo, hi];
}

function tickLabel(value: number, log: boolean): string {
  if (log) return value >= 0.01 ? value.toFixed(2) : value.toExponential(0);
  return Math.abs(value) >= 100 ? value.toFixed(0) : formatLoss(value);
}


/* ------------------------------------------------------------------ *
 * Histogram (§6.4 — weight distributions per layer)
 * ------------------------------------------------------------------ */

export interface HistogramBin {
  readonly from: number;
  readonly to: number;
  readonly count: number;
}

/** Bin values into a fixed number of buckets spanning their range. */
export function histogram(values: Float64Array, bins = 24): HistogramBin[] {
  if (values.length === 0) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [];
  // A layer whose weights are all identical (zero init, §7.3) has no range;
  // give it a nominal one so the single spike is still drawn.
  if (hi - lo < 1e-12) {
    lo -= 0.5;
    hi += 0.5;
  }
  const width = (hi - lo) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const index = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[index] = (counts[index] as number) + 1;
  }
  return counts.map((count, i) => ({ from: lo + i * width, to: lo + (i + 1) * width, count }));
}

/**
 * Draw a histogram, coloured by the sign of each bin.
 *
 * Sign uses the weight poles because these ARE weights: the histogram and the
 * edges on the canvas describe the same numbers and should read alike.
 */
export function drawHistogram(
  ctx: Ctx2D,
  x: number,
  y: number,
  width: number,
  height: number,
  bins: readonly HistogramBin[],
  label: string,
): void {
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.bgCanvas;
  ctx.fillRect(x, y, width, height);

  ctx.font = `500 9px ${FONTS.mono}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText(label, x + 4, y + 3);

  if (bins.length === 0) return;
  const plotTop = y + 14;
  const plotHeight = Math.max(1, height - 14 - 10);
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const barWidth = width / bins.length;

  bins.forEach((bin, i) => {
    if (bin.count === 0) return;
    const h = (bin.count / maxCount) * plotHeight;
    const centre = (bin.from + bin.to) / 2;
    ctx.fillStyle = centre < 0 ? COLORS.weightNegative : COLORS.weightPositive;
    // A 1px gap between bars, so adjacent bins read as separate.
    ctx.fillRect(x + i * barWidth, plotTop + plotHeight - h, Math.max(1, barWidth - 1), h);
  });

  // Zero line, so the split between negative and positive is locatable.
  const lo = bins[0]?.from ?? 0;
  const hi = bins[bins.length - 1]?.to ?? 1;
  if (lo < 0 && hi > 0) {
    const zx = Math.round(x + ((0 - lo) / (hi - lo)) * width) + 0.5;
    ctx.strokeStyle = COLORS.lineEdge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(zx, plotTop);
    ctx.lineTo(zx, plotTop + plotHeight);
    ctx.stroke();
  }

  ctx.font = `400 8px ${FONTS.mono}`;
  ctx.fillStyle = COLORS.textLo;
  ctx.textBaseline = 'bottom';
  ctx.fillText(lo.toFixed(2), x + 3, y + height - 1);
  ctx.textAlign = 'right';
  ctx.fillText(hi.toFixed(2), x + width - 3, y + height - 1);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}
