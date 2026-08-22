/*
 * Edge pulses.
 *
 * Spec §6.3 A.2: "pulse speed constant, pulse size ∝ |w · x| contribution,
 * pulse colour by contribution sign. Contribution magnitude is what matters
 * here, not the raw weight."
 *
 * That last sentence is the whole point of this file. A large weight on a
 * near-zero input contributes nothing, and drawing it as a fat pulse would
 * teach the wrong thing. Size follows |w·a|.
 *
 * §6.1 requires zero allocation in the hot path, so pulses come from a
 * pre-allocated pool and are never created per frame.
 */

import type { Layout, Viewport } from '../layout';
import { worldToScreenX, worldToScreenY } from '../layout';
import { COLORS, WEIGHT_NEGATIVE, WEIGHT_POSITIVE, parseHex } from '../theme';
import type { Ctx2D } from './context';
import { TAU } from './context';

export interface Pulse {
  active: boolean;
  edge: number;
  /** 0..1 along the edge. */
  t: number;
  /** |contribution|, normalized to the largest in its layer. */
  magnitude: number;
  sign: number;
}

export const PULSE_MIN_RADIUS = 1.6;
export const PULSE_MAX_RADIUS = 7;

/**
 * A fixed pool. Capacity is the largest number of simultaneous pulses the
 * layout can produce — one per edge into a single layer — so acquire() never
 * has to allocate and never has to drop a pulse.
 */
export class PulsePool {
  private readonly pulses: Pulse[];

  constructor(capacity: number) {
    this.pulses = Array.from({ length: Math.max(1, capacity) }, () => ({
      active: false,
      edge: -1,
      t: 0,
      magnitude: 0,
      sign: 1,
    }));
  }

  get capacity(): number {
    return this.pulses.length;
  }

  /** Deactivate everything without discarding the backing objects. */
  clear(): void {
    for (const pulse of this.pulses) pulse.active = false;
  }

  /** The next free pulse, already marked active, or null if the pool is full. */
  acquire(): Pulse | null {
    for (const pulse of this.pulses) {
      if (!pulse.active) {
        pulse.active = true;
        return pulse;
      }
    }
    return null;
  }

  release(pulse: Pulse): void {
    pulse.active = false;
  }

  get activeCount(): number {
    let n = 0;
    for (const pulse of this.pulses) if (pulse.active) n++;
    return n;
  }

  forEachActive(visit: (pulse: Pulse) => void): void {
    for (const pulse of this.pulses) if (pulse.active) visit(pulse);
  }
}

const NEG = parseHex(WEIGHT_NEGATIVE);
const POS = parseHex(WEIGHT_POSITIVE);

/** Radius for a normalized contribution magnitude. */
export function pulseRadius(magnitude: number): number {
  const m = magnitude < 0 ? 0 : magnitude > 1 ? 1 : magnitude;
  // sqrt so AREA tracks magnitude: a pulse twice as significant should look
  // twice as big, and area is what the eye actually compares.
  return PULSE_MIN_RADIUS + (PULSE_MAX_RADIUS - PULSE_MIN_RADIUS) * Math.sqrt(m);
}

export function drawPulses(
  ctx: Ctx2D,
  layout: Layout,
  viewport: Viewport,
  pool: PulsePool,
): void {
  ctx.setLineDash([]);
  pool.forEachActive((pulse) => {
    const edge = layout.edges[pulse.edge];
    if (edge === undefined) return;
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) return;

    const x = worldToScreenX(viewport, a.x + (b.x - a.x) * pulse.t);
    const y = worldToScreenY(viewport, a.y + (b.y - a.y) * pulse.t);
    const r = pulseRadius(pulse.magnitude);
    const c = pulse.sign < 0 ? NEG : POS;

    // A soft halo under a solid core: the core carries the value, the halo
    // makes a 2px dot findable against a busy field of edges.
    ctx.beginPath();
    ctx.arc(x, y, r * 2.1, 0, TAU);
    ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, 0.14)`;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, 0.95)`;
    ctx.fill();
  });
}

/**
 * Discrete stage highlight for prefers-reduced-motion (§6.3).
 *
 * Instead of travelling dots, the edges feeding the active layer are
 * over-stroked as a set. The learner still sees WHICH connections are
 * contributing and how much — only the travel is gone.
 */
export function drawStageHighlight(
  ctx: Ctx2D,
  layout: Layout,
  viewport: Viewport,
  layer: number,
  contributionOf: (edgeIndex: number) => number,
): void {
  ctx.setLineDash([]);
  ctx.lineCap = 'round';
  for (let i = 0; i < layout.edges.length; i++) {
    const edge = layout.edges[i];
    if (edge === undefined || edge.layer !== layer) continue;
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) continue;

    const magnitude = contributionOf(i);
    if (magnitude <= 0.02) continue;
    const c = magnitude < 0 ? NEG : POS;
    ctx.strokeStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${(0.25 + 0.6 * Math.min(1, Math.abs(magnitude))).toFixed(3)})`;
    ctx.lineWidth = 1 + 4 * Math.min(1, Math.abs(magnitude));
    ctx.beginPath();
    ctx.moveTo(worldToScreenX(viewport, a.x), worldToScreenY(viewport, a.y));
    ctx.lineTo(worldToScreenX(viewport, b.x), worldToScreenY(viewport, b.y));
    ctx.stroke();
  }
}

/** Ring drawn on a node as its stage becomes active. */
export function drawActiveRing(
  ctx: Ctx2D,
  x: number,
  y: number,
  radius: number,
  strength: number,
): void {
  ctx.beginPath();
  ctx.arc(x, y, radius + 5, 0, TAU);
  ctx.strokeStyle = COLORS.focus;
  ctx.globalAlpha = 0.55 * Math.min(1, Math.max(0, strength));
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.stroke();
  ctx.globalAlpha = 1;
}
