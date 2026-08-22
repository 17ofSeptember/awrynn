/*
 * Dissection view controller.
 *
 * Ties the choreography (animation.ts), the engine truth (dissection.ts) and
 * the drawing (draw/pulses.ts, draw/cards.ts) into the mode §6.3 A describes.
 *
 * ONE DELIBERATE DEVIATION, and the reason for it.
 *
 * §6.3 describes the choreography "per neuron". Taken literally on a 6-unit
 * layer that means six 208px cards competing for a column whose nodes are ~64px
 * apart: they overlap into illegibility, and a card nobody can read teaches
 * nothing. So a layer resolves as a unit, with the FULL card shown for one
 * focused neuron and compact z/a chips for its neighbours. Clicking any node
 * moves the focus. Every neuron still gets its card — you choose which one you
 * are reading, instead of being shown all of them at once and none of them
 * clearly.
 */

import type { Dissection, NeuronDissection } from './dissection';
import type { Layout, Viewport } from './layout';
import { worldToScreenX, worldToScreenY } from './layout';
import { buildChoreography, Choreography, clamp01, easeOutCubic, stagger } from './animation';
import type { Beat } from './animation';
import { getActivation, isElementwise } from '../engine/activations';
import type { ActivationName } from '../engine/activations';
import type { Ctx2D } from './draw/context';
import {
  drawActiveRing,
  drawPulses,
  drawStageHighlight,
  PulsePool,
} from './draw/pulses';
import {
  drawActivationPlot,
  drawFormulaCard,
  drawGradientCard,
  drawLossCard,
  GRADIENT_CARD_HEIGHT,
  GRADIENT_CARD_WIDTH,
  LOSS_CARD_WIDTH,
  measureFormulaCard,
  MINI_PLOT_SIZE,
} from './draw/cards';
import { COLORS, FONTS } from './theme';

export interface TransportStatus {
  readonly label: string;
  readonly beatIndex: number;
  readonly beatCount: number;
  readonly playing: boolean;
  readonly complete: boolean;
  readonly speed: number;
}

const CARD_GAP = 18;
/** Keep-out margin so a card never touches the viewport edge. */
const EDGE_MARGIN = 12;

export class DissectionView {
  private dissection: Dissection | null = null;
  private choreography: Choreography;
  private pool: PulsePool;
  private layout: Layout | null = null;
  private reducedMotion = false;

  speed = 1;
  /** Viewport size for the current frame, so cards can avoid its edges. */
  private viewWidth = 0;
  private viewHeight = 0;
  /** Which unit of the active layer shows its full card. */
  focusUnit = 0;
  /** Which edge shows its gradient card during the backward pass. */
  focusEdgeRow = 0;

  constructor() {
    this.choreography = new Choreography(
      buildChoreography({ layerCount: 1, longestEdgePerLayer: [200] }),
    );
    this.pool = new PulsePool(1);
  }

  get active(): boolean {
    return this.dissection !== null;
  }

  get current(): Beat {
    return this.choreography.current;
  }

  /** Point the view at a fresh dissection, rebuilding the timeline for it. */
  load(dissection: Dissection, layout: Layout, layerCount: number, reducedMotion: boolean): void {
    this.dissection = dissection;
    this.layout = layout;
    this.reducedMotion = reducedMotion;

    const longest = longestEdgePerLayer(layout, layerCount);
    this.choreography.replace(buildChoreography({ layerCount, longestEdgePerLayer: longest, reducedMotion }));

    // Capacity is the widest fan-in the layout can produce, so acquire() never
    // has to drop a pulse mid-beat.
    const widest = widestLayerEdgeCount(layout, layerCount);
    if (this.pool.capacity < widest) this.pool = new PulsePool(widest);
    this.pool.clear();
  }

  clear(): void {
    this.dissection = null;
    this.layout = null;
    this.pool.clear();
  }

  advance(deltaMs: number): void {
    this.choreography.advance(deltaMs, this.speed);
  }

  play(): void {
    this.choreography.play();
  }
  pause(): void {
    this.choreography.pause();
  }
  stepBeat(): void {
    this.choreography.stepBeat();
  }
  stepBack(): void {
    this.choreography.stepBack();
  }
  restart(): void {
    this.choreography.reset();
  }
  seek(index: number): void {
    this.choreography.seek(index);
  }

  status(): TransportStatus {
    const state = this.choreography.state;
    return {
      label: this.choreography.current.label,
      beatIndex: state.beatIndex,
      beatCount: this.choreography.beatCount,
      playing: !this.choreography.isPaused,
      complete: state.complete,
      speed: this.speed,
    };
  }

  /** Neurons of a layer, in unit order. */
  /**
   * Place a card beside a node, flipping to the other side when it would
   * overflow, and clamping vertically.
   *
   * The output column sits at the right edge of the layout, so a card placed
   * blindly to its right is half off-screen — which is how a chip ends up
   * reading "a +0." instead of a number.
   */
  private place(
    x: number,
    y: number,
    radius: number,
    width: number,
    height: number,
  ): { x: number; y: number; width: number; height: number } {
    let left = x + radius + CARD_GAP;
    if (left + width > this.viewWidth - EDGE_MARGIN) {
      const flipped = x - radius - CARD_GAP - width;
      // Only flip if the other side actually fits; otherwise clamp, so a very
      // narrow viewport degrades to "slightly overlapping" not "off-screen".
      left = flipped >= EDGE_MARGIN ? flipped : Math.max(EDGE_MARGIN, this.viewWidth - EDGE_MARGIN - width);
    }
    const top = Math.max(
      EDGE_MARGIN,
      Math.min(this.viewHeight - EDGE_MARGIN - height, y - height / 2),
    );
    return { x: left, y: top, width, height };
  }

  private neuronsOf(layer: number): NeuronDissection[] {
    if (this.dissection === null) return [];
    return this.dissection.neurons.filter((n) => n.layer === layer);
  }

  /**
   * The whole dissection overlay for this frame.
   *
   * Drawn after nodes and before hover overlays, so cards sit above the network
   * but below anything the pointer is doing.
   */
  draw(ctx: Ctx2D, viewport: Viewport, width: number, height: number): void {
    const dissection = this.dissection;
    const layout = this.layout;
    if (dissection === null || layout === null) return;

    this.viewWidth = width;
    this.viewHeight = height;

    const beat = this.choreography.current;
    const { progress } = this.choreography.state;

    switch (beat.kind) {
      case 'input':
        this.drawInputs(ctx, viewport, progress);
        break;
      case 'forward-pulse':
        this.drawForwardPulses(ctx, viewport, beat.layer, progress);
        this.drawLayerCards(ctx, viewport, beat.layer, progress, 0, false);
        break;
      case 'forward-activate':
        this.drawLayerCards(ctx, viewport, beat.layer, 1, progress, false);
        break;
      case 'loss':
        this.drawCollapsedLayers(ctx, viewport, Number.POSITIVE_INFINITY);
        drawLossCard(ctx, width - LOSS_CARD_WIDTH - 20, 64, dissection.output, progress);
        break;
      case 'backward-delta':
        this.drawCollapsedLayers(ctx, viewport, Number.POSITIVE_INFINITY);
        this.drawDeltaPulses(ctx, viewport, beat.layer, progress);
        break;
      case 'backward-grad':
        this.drawCollapsedLayers(ctx, viewport, Number.POSITIVE_INFINITY);
        this.drawGradient(ctx, viewport, beat.layer, progress, width, height);
        break;
      case 'backward-update':
        this.drawCollapsedLayers(ctx, viewport, Number.POSITIVE_INFINITY);
        this.drawUpdate(ctx, viewport, beat.layer, progress);
        break;
      case 'complete':
        this.drawCollapsedLayers(ctx, viewport, Number.POSITIVE_INFINITY);
        drawLossCard(ctx, width - LOSS_CARD_WIDTH - 20, 64, dissection.output, 1);
        break;
    }

    this.drawBeatCaption(ctx, width, height, beat);
  }

  private drawInputs(ctx: Ctx2D, viewport: Viewport, progress: number): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;

    ctx.font = `500 11px ${FONTS.mono}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const count = dissection.inputs.length;
    for (let i = 0; i < count; i++) {
      const node = layout.nodes[i];
      if (node === undefined || node.layer !== 0) continue;
      const appear = easeOutCubic(stagger(progress, i, count));
      if (appear <= 0) continue;
      const x = worldToScreenX(viewport, node.x);
      const y = worldToScreenY(viewport, node.y);
      drawActiveRing(ctx, x, y, node.radius * viewport.scale, appear);
      ctx.globalAlpha = appear;
      ctx.fillStyle = COLORS.textHi;
      ctx.fillText(formatValue(dissection.inputs[i] as number), x - node.radius * viewport.scale - 8, y);
      ctx.globalAlpha = 1;
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  private drawForwardPulses(
    ctx: Ctx2D,
    viewport: Viewport,
    layer: number,
    progress: number,
  ): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;

    const neurons = this.neuronsOf(layer);
    let maxContribution = 1e-12;
    for (const n of neurons) {
      for (const t of n.terms) maxContribution = Math.max(maxContribution, Math.abs(t.contribution));
    }

    if (this.reducedMotion) {
      // §6.3: discrete stage highlights rather than travelling pulses.
      drawStageHighlight(ctx, layout, viewport, layer, (edgeIndex) => {
        const edge = layout.edges[edgeIndex];
        if (edge === undefined || edge.isBias) return 0;
        const neuron = neurons[edge.col];
        const term = neuron?.terms[edge.row];
        return term === undefined ? 0 : term.contribution / maxContribution;
      });
      return;
    }

    this.pool.clear();
    for (let i = 0; i < layout.edges.length; i++) {
      const edge = layout.edges[i];
      if (edge === undefined || edge.layer !== layer || edge.isBias) continue;
      const neuron = neurons[edge.col];
      const term = neuron?.terms[edge.row];
      if (term === undefined) continue;

      const pulse = this.pool.acquire();
      if (pulse === null) break;
      pulse.edge = i;
      // Constant speed: a pulse on a longer edge is still travelling when a
      // short one has landed, which is what makes the card fill unevenly.
      pulse.t = clamp01(progress);
      pulse.magnitude = Math.abs(term.contribution) / maxContribution;
      pulse.sign = term.contribution < 0 ? -1 : 1;
    }
    drawPulses(ctx, layout, viewport, this.pool);
  }

  /** Cards for the active layer: full card for the focused unit, chips beside. */
  private drawLayerCards(
    ctx: Ctx2D,
    viewport: Viewport,
    layer: number,
    assemble: number,
    activate: number,
    collapsed: boolean,
  ): void {
    const layout = this.layout;
    if (layout === null) return;
    const neurons = this.neuronsOf(layer);
    if (neurons.length === 0) return;

    this.drawCollapsedLayers(ctx, viewport, layer);

    const focus = Math.min(this.focusUnit, neurons.length - 1);

    /*
     * Neighbours first, the focused card last.
     *
     * Nothing here avoids collisions, and the focused card is the tallest thing
     * on the canvas — taller still on a normalized layer, which adds two rows.
     * Drawing in unit order let a lower neighbour's chip paint over the card
     * being read. Painting order is the whole fix: the card the reader is
     * looking at goes on top of anything it happens to overlap.
     */
    neurons.forEach((neuron, unit) => {
      if (unit === focus || assemble < 1) return;
      const node = nodeFor(layout, layer + 1, unit);
      if (node === undefined) return;
      const state = { neuron, assembleProgress: 1, activateProgress: activate, collapsed: true };
      const chip = measureFormulaCard(state);
      drawFormulaCard(
        ctx,
        state,
        this.place(
          worldToScreenX(viewport, node.x),
          worldToScreenY(viewport, node.y),
          node.radius * viewport.scale,
          chip.width,
          chip.height,
        ),
      );
    });

    const focused = neurons[focus];
    const focusNode = nodeFor(layout, layer + 1, focus);
    if (focused !== undefined && focusNode !== undefined) {
      const x = worldToScreenX(viewport, focusNode.x);
      const y = worldToScreenY(viewport, focusNode.y);
      const r = focusNode.radius * viewport.scale;
      const neuron = focused;

      const state = { neuron, assembleProgress: assemble, activateProgress: activate, collapsed };
      const size = measureFormulaCard(state);
      const place = this.place(x, y, r, size.width, size.height);
      drawActiveRing(ctx, x, y, r, 1);
      drawFormulaCard(ctx, state, place);

      if (activate > 0) {
        const activation = getActivation(neuron.activation as ActivationName);
        if (isElementwise(activation)) {
          drawActivationPlot(
            ctx,
            place.x,
            place.y + place.height + 8,
            neuron,
            (z) => activation.f(z),
            activation.range,
            activate,
          );
        } else {
          // Softmax is row-wise; a 1-D curve would be a lie, so show the
          // probability it produced instead.
          drawSoftmaxNote(ctx, place.x, place.y + place.height + 8, neuron);
        }
      }
    }
  }

  /** z/a chips for every layer already resolved, so the trail stays visible. */
  private drawCollapsedLayers(ctx: Ctx2D, viewport: Viewport, before: number): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;
    for (const neuron of dissection.neurons) {
      if (neuron.layer >= before) continue;
      const node = nodeFor(layout, neuron.layer + 1, neuron.unit);
      if (node === undefined) continue;
      const x = worldToScreenX(viewport, node.x);
      const y = worldToScreenY(viewport, node.y);
      const r = node.radius * viewport.scale;
      const chipState = { neuron, assembleProgress: 1, activateProgress: 1, collapsed: true };
      const chip = measureFormulaCard(chipState);
      drawFormulaCard(ctx, chipState, this.place(x, y, r, chip.width, chip.height));
    }
  }

  private drawDeltaPulses(ctx: Ctx2D, viewport: Viewport, layer: number, progress: number): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;

    const gradients = dissection.gradients.filter((g) => g.layer === layer);
    let maxDelta = 1e-12;
    for (const g of gradients) maxDelta = Math.max(maxDelta, Math.abs(g.delta));

    if (this.reducedMotion) {
      drawStageHighlight(ctx, layout, viewport, layer, (edgeIndex) => {
        const edge = layout.edges[edgeIndex];
        if (edge === undefined || edge.isBias) return 0;
        const g = gradients.find((x) => x.row === edge.row && x.col === edge.col);
        return g === undefined ? 0 : g.delta / maxDelta;
      });
      return;
    }

    this.pool.clear();
    for (let i = 0; i < layout.edges.length; i++) {
      const edge = layout.edges[i];
      if (edge === undefined || edge.layer !== layer || edge.isBias) continue;
      const g = gradients.find((x) => x.row === edge.row && x.col === edge.col);
      if (g === undefined) continue;
      const pulse = this.pool.acquire();
      if (pulse === null) break;
      pulse.edge = i;
      // Right to left: δ flows backward, so t runs from 1 to 0.
      pulse.t = 1 - clamp01(progress);
      pulse.magnitude = Math.abs(g.delta) / maxDelta;
      pulse.sign = g.delta < 0 ? -1 : 1;
    }
    drawPulses(ctx, layout, viewport, this.pool);
  }

  private drawGradient(
    ctx: Ctx2D,
    viewport: Viewport,
    layer: number,
    progress: number,
    width: number,
    height: number,
  ): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;
    const gradients = dissection.gradients.filter((g) => g.layer === layer);
    if (gradients.length === 0) return;

    const focus = gradients[Math.min(this.focusEdgeRow, gradients.length - 1)];
    if (focus === undefined) return;

    const edgeIndex = layout.edges.findIndex(
      (e) => e.layer === layer && e.row === focus.row && e.col === focus.col && !e.isBias,
    );
    const edge = layout.edges[edgeIndex];
    let cx = width / 2;
    let cy = height / 2;
    if (edge !== undefined) {
      const a = layout.nodes[edge.from];
      const b = layout.nodes[edge.to];
      if (a !== undefined && b !== undefined) {
        cx = worldToScreenX(viewport, (a.x + b.x) / 2);
        cy = worldToScreenY(viewport, (a.y + b.y) / 2);
      }
    }
    // Kept inside the viewport, so a card never half-vanishes off an edge.
    const x = Math.max(12, Math.min(width - GRADIENT_CARD_WIDTH - 12, cx - GRADIENT_CARD_WIDTH / 2));
    const y = Math.max(12, Math.min(height - GRADIENT_CARD_HEIGHT - 12, cy - GRADIENT_CARD_HEIGHT - 12));
    drawGradientCard(ctx, x, y, focus, dissection.learningRate, progress);
  }

  private drawUpdate(ctx: Ctx2D, viewport: Viewport, layer: number, progress: number): void {
    const layout = this.layout;
    const dissection = this.dissection;
    if (layout === null || dissection === null) return;

    // The edge visibly re-weighting: the stroke widens toward w + Δw.
    const gradients = dissection.gradients.filter((g) => g.layer === layer);
    let maxStep = 1e-12;
    for (const g of gradients) maxStep = Math.max(maxStep, Math.abs(g.step));

    ctx.setLineDash([]);
    ctx.lineCap = 'round';
    for (let i = 0; i < layout.edges.length; i++) {
      const edge = layout.edges[i];
      if (edge === undefined || edge.layer !== layer || edge.isBias) continue;
      const g = gradients.find((x) => x.row === edge.row && x.col === edge.col);
      if (g === undefined) continue;
      const a = layout.nodes[edge.from];
      const b = layout.nodes[edge.to];
      if (a === undefined || b === undefined) continue;

      const strength = (Math.abs(g.step) / maxStep) * easeOutCubic(progress);
      if (strength < 0.02) continue;
      ctx.strokeStyle = g.step < 0 ? COLORS.weightNegative : COLORS.weightPositive;
      ctx.globalAlpha = 0.8 * strength;
      ctx.lineWidth = 1 + 5 * strength;
      ctx.beginPath();
      ctx.moveTo(worldToScreenX(viewport, a.x), worldToScreenY(viewport, a.y));
      ctx.lineTo(worldToScreenX(viewport, b.x), worldToScreenY(viewport, b.y));
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  private drawBeatCaption(ctx: Ctx2D, width: number, height: number, beat: Beat): void {
    // The learner should never have to guess which stage they are watching.
    ctx.font = `500 10px ${FONTS.mono}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = COLORS.textMid;
    ctx.fillText(beat.label.toUpperCase(), width / 2, height - 18);
    ctx.textAlign = 'left';
  }
}

function drawSoftmaxNote(ctx: Ctx2D, x: number, y: number, neuron: NeuronDissection): void {
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(x, y, MINI_PLOT_SIZE * 2, 30);
  ctx.strokeStyle = COLORS.lineHair;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x + 0.5, y + 0.5, MINI_PLOT_SIZE * 2 - 1, 29);
  ctx.stroke();
  ctx.font = `400 10px ${FONTS.mono}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText('softmax over the layer  p =', x + 8, y + 15);
  ctx.fillStyle = COLORS.textHi;
  ctx.fillText(neuron.a.toFixed(3), x + 128, y + 15);
  ctx.textBaseline = 'alphabetic';
}

function nodeFor(layout: Layout, column: number, unit: number): Layout['nodes'][number] | undefined {
  const base = layout.layerOffsets[column];
  if (base === undefined) return undefined;
  return layout.nodes[base + unit];
}

function longestEdgePerLayer(layout: Layout, layerCount: number): number[] {
  const longest = new Array<number>(layerCount).fill(0);
  for (const edge of layout.edges) {
    if (edge.layer < 0 || edge.layer >= layerCount) continue;
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) continue;
    longest[edge.layer] = Math.max(longest[edge.layer] as number, Math.hypot(b.x - a.x, b.y - a.y));
  }
  return longest;
}

function widestLayerEdgeCount(layout: Layout, layerCount: number): number {
  const counts = new Array<number>(layerCount).fill(0);
  for (const edge of layout.edges) {
    if (edge.isBias || edge.layer < 0 || edge.layer >= layerCount) continue;
    counts[edge.layer] = (counts[edge.layer] as number) + 1;
  }
  return Math.max(1, ...counts);
}

function formatValue(v: number): string {
  return (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
}
