/*
 * Edge rendering — the diverging weight scale, which is the visual core (§9).
 *
 * Sign is hue, magnitude is alpha AND stroke width, near-zero fades almost out
 * so pruning becomes visible (§6.2). Colourblind-safe mode adds a dash pattern
 * for negative weights on top of the hue difference.
 *
 * Everything is drawn in SCREEN coordinates: world positions are converted
 * explicitly rather than by a canvas transform, so a 0.5px hairline is a real
 * 0.5px at every zoom level instead of vanishing when zoomed out.
 */

import type { EdgeLayout, Layout, Viewport } from '../layout';
import { worldToScreenX, worldToScreenY } from '../layout';
import {
  COLORBLIND_NEGATIVE_DASH,
  COLORS,
  weightStroke,
  weightWidth,
} from '../theme';
import type { Ctx2D } from './context';

export interface EdgeStyleContext {
  /** 95th percentile of |w|, smoothed over time so the picture cannot strobe. */
  readonly wRef: number;
  readonly colorblindSafe: boolean;
  /** Index of the hovered edge, or -1. */
  readonly hoveredEdge: number;
  /** Index of the selected edge, or -1. */
  readonly selectedEdge: number;
  /** Layers excluded from updates render with a distinct dashed treatment (§6.2). */
  readonly frozenLayers: ReadonlySet<number>;
}

/** Reads a weight for an edge. Bias edges use `row === -1`. */
export type WeightLookup = (edge: EdgeLayout) => number;

const SOLID: number[] = [];
const FROZEN_DASH = [2, 4];

export function drawEdges(
  ctx: Ctx2D,
  layout: Layout,
  viewport: Viewport,
  weightOf: WeightLookup,
  style: EdgeStyleContext,
): void {
  ctx.lineCap = 'round';

  for (let i = 0; i < layout.edges.length; i++) {
    const edge = layout.edges[i] as EdgeLayout;
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) continue;

    const w = weightOf(edge);
    const emphasised = i === style.hoveredEdge || i === style.selectedEdge;

    ctx.strokeStyle = emphasised ? COLORS.focus : weightStroke(w, style.wRef);
    // Width is a screen-space quantity, so it is NOT scaled by the viewport:
    // the encoding must mean the same thing however far the user has zoomed.
    ctx.lineWidth = weightWidth(w, style.wRef) + (emphasised ? 1.5 : 0);

    if (style.frozenLayers.has(edge.layer)) {
      ctx.setLineDash(FROZEN_DASH);
    } else if (style.colorblindSafe && w < 0) {
      ctx.setLineDash(COLORBLIND_NEGATIVE_DASH as number[]);
    } else {
      ctx.setLineDash(SOLID);
    }

    ctx.beginPath();
    ctx.moveTo(worldToScreenX(viewport, a.x), worldToScreenY(viewport, a.y));
    ctx.lineTo(worldToScreenX(viewport, b.x), worldToScreenY(viewport, b.y));
    ctx.stroke();
  }

  ctx.setLineDash(SOLID);
}

/**
 * The 95th percentile of |w| across the network — the reference the whole edge
 * encoding is relative to, shown in the legend so the scale is never a mystery.
 *
 * A percentile rather than the max: one runaway weight would otherwise flatten
 * every other edge to invisibility, which is exactly what happens right before
 * a divergence, when the picture matters most.
 */
export function weightReference(weights: Float64Array, percentile = 0.95): number {
  if (weights.length === 0) return 1;
  const magnitudes = Array.from(weights, Math.abs).sort((a, b) => a - b);
  const index = Math.min(magnitudes.length - 1, Math.floor(percentile * (magnitudes.length - 1)));
  const value = magnitudes[index] ?? 0;
  // A zero-initialised network (lesson 3) has no scale at all; fall back to 1
  // so every edge renders at its minimum width rather than dividing by zero.
  return value > 1e-9 ? value : 1;
}

/**
 * Exponentially smoothed reference, so the picture does not strobe while
 * training (§6.2). Returns the new smoothed value.
 */
export function smoothReference(previous: number, target: number, alpha = 0.12): number {
  if (!Number.isFinite(previous) || previous <= 0) return target;
  return previous + (target - previous) * alpha;
}
