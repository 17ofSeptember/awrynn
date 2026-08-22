/*
 * Node rendering.
 *
 * Spec §6.2: fill ∝ activation through the activation's own range; ring ∝ |δ|
 * during and after backprop; bias satellites drawn explicitly; dead units
 * hollow and desaturated; frozen parameters dashed.
 */

import type { Layout, NodeLayout, Viewport } from '../layout';
import { worldToScreenX, worldToScreenY } from '../layout';
import { activationFill, COLORS, FONTS } from '../theme';
import type { Ctx2D } from './context';
import { TAU } from './context';

export interface NodeStyleContext {
  /** Activation normalized to [-1, 1], or null when nothing has run yet. */
  readonly normalizedActivation: (node: NodeLayout) => number | null;
  /** |δ| normalized to [0, 1], or null outside a backward pass. */
  readonly normalizedDelta: (node: NodeLayout) => number | null;
  /** Zero activation across the whole last epoch (§6.2). */
  readonly isDead: (node: NodeLayout) => boolean;
  readonly isFrozen: (node: NodeLayout) => boolean;
  readonly isAblated: (node: NodeLayout) => boolean;
  readonly hoveredNode: number;
  readonly selectedNode: number;
}

const FROZEN_DASH = [2, 3];
const SOLID: number[] = [];

export function drawNodes(
  ctx: Ctx2D,
  layout: Layout,
  viewport: Viewport,
  style: NodeStyleContext,
): void {
  for (let i = 0; i < layout.nodes.length; i++) {
    const node = layout.nodes[i] as NodeLayout;
    const x = worldToScreenX(viewport, node.x);
    const y = worldToScreenY(viewport, node.y);
    // Radius follows zoom — a node is an object in the scene, unlike a stroke
    // width, which is an encoding.
    const r = node.radius * viewport.scale;
    const dead = style.isDead(node);
    const ablated = style.isAblated(node);

    // Fill.
    const activation = style.normalizedActivation(node);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    if (dead || ablated) {
      // Hollow and desaturated: a dead unit must look inert, not just dim.
      ctx.fillStyle = COLORS.bgCanvas;
    } else {
      ctx.fillStyle = activation === null ? COLORS.bgRaised : activationFill(activation);
    }
    ctx.fill();

    // Outline.
    ctx.setLineDash(style.isFrozen(node) ? FROZEN_DASH : SOLID);
    ctx.lineWidth = node.kind === 'bias' ? 1 : 1.25;
    ctx.strokeStyle =
      i === style.hoveredNode || i === style.selectedNode
        ? COLORS.focus
        : dead || ablated
          ? COLORS.lineEdge
          : COLORS.lineHair;
    ctx.stroke();
    ctx.setLineDash(SOLID);

    // Delta ring: which neurons are actually learning (§6.2).
    const delta = style.normalizedDelta(node);
    if (delta !== null && delta > 0.01 && node.kind !== 'bias') {
      ctx.beginPath();
      ctx.arc(x, y, r + 3.5, 0, TAU);
      ctx.strokeStyle = COLORS.focus;
      ctx.globalAlpha = Math.min(1, delta) * 0.7;
      ctx.lineWidth = 1 + Math.min(1, delta) * 2.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
}

/**
 * Layer captions and the bias glyph.
 *
 * Drawn after every node so no fill can land on top of a label.
 */
export function drawNodeLabels(
  ctx: Ctx2D,
  layout: Layout,
  viewport: Viewport,
  captions: readonly string[],
): void {
  ctx.font = `500 11px ${FONTS.mono}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = COLORS.textLo;

  for (let i = 0; i < layout.nodes.length; i++) {
    const node = layout.nodes[i] as NodeLayout;
    if (node.kind !== 'bias') continue;
    const r = node.radius * viewport.scale;
    if (r < 4) continue;
    ctx.fillText(
      'b',
      worldToScreenX(viewport, node.x),
      worldToScreenY(viewport, node.y) + 3.5,
    );
  }

  ctx.font = `600 11px ${FONTS.display}`;
  ctx.fillStyle = COLORS.textMid;
  ctx.textBaseline = 'bottom';
  layout.columnX.forEach((x, layer) => {
    const caption = captions[layer];
    if (caption === undefined) return;
    ctx.fillText(caption, worldToScreenX(viewport, x), 22);
  });
}
