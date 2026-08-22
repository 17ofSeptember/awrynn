/*
 * The render loop.
 *
 * Spec §6.1:
 *   - own requestAnimationFrame loop
 *   - React must NOT re-render on animation frames; the canvas reads a mutable
 *     frame-state object and subscribes to the store directly
 *   - fixed draw order
 *   - zero allocation in the hot path
 *   - devicePixelRatio scaling so nothing is blurry on retina
 *
 * The loop is dirty-tracked: an idle scene costs one comparison per frame and
 * no drawing at all, which is what makes "60fps idle" mean an idle CPU rather
 * than a busy one hitting 60.
 */

import type { Hit } from './hit';
import type { Layout, NodeLayout, Viewport } from './layout';
import { IDENTITY_VIEWPORT } from './layout';
import { COLORS, FONTS } from './theme';
import type { Ctx2D } from './draw/context';
import { drawGrid } from './draw/grid';
import { drawEdges, type WeightLookup } from './draw/edges';
import { drawNodes, drawNodeLabels } from './draw/nodes';
import { drawLegend, LEGEND_HEIGHT } from './draw/legend';
import { drawThumbnails, ThumbnailSheet } from './draw/thumbnails';
import type { BlitContext, ThumbnailSource } from './draw/thumbnails';
import type { DissectionView } from './dissectionView';

/**
 * Everything a frame needs, as one MUTABLE object.
 *
 * Mutable on purpose: the store mutates fields in place and flips `dirty`, so a
 * frame reads plain properties instead of allocating a new state object 60
 * times a second (§6.1).
 */
export interface FrameState {
  layout: Layout;
  viewport: Viewport;
  weightOf: WeightLookup;
  normalizedActivation: (node: NodeLayout) => number | null;
  normalizedDelta: (node: NodeLayout) => number | null;
  isDead: (node: NodeLayout) => boolean;
  isFrozen: (node: NodeLayout) => boolean;
  isAblated: (node: NodeLayout) => boolean;
  frozenLayers: ReadonlySet<number>;
  captions: readonly string[];
  wRef: number;
  colorblindSafe: boolean;
  /** Edges encode Δw against a pinned snapshot rather than w (§6.6). */
  diffing: boolean;
  hover: Hit | null;
  selection: Hit | null;
  /**
   * The dissection overlay, or null when the mode is off.
   *
   * While a dissection is playing the scene redraws every frame regardless of
   * `dirty`, because the choreography is advancing whether or not the store
   * changed.
   */
  dissection: DissectionView | null;
  /** Per-unit activation maps, or null when the option is off (§6.4). */
  thumbnails: ThumbnailSource | null;
  /** Set to request a redraw. The scene clears it after drawing. */
  dirty: boolean;
}

export function createFrameState(layout: Layout): FrameState {
  return {
    layout,
    viewport: IDENTITY_VIEWPORT,
    weightOf: () => 0,
    normalizedActivation: () => null,
    normalizedDelta: () => null,
    isDead: () => false,
    isFrozen: () => false,
    isAblated: () => false,
    frozenLayers: new Set<number>(),
    captions: [],
    wRef: 1,
    colorblindSafe: false,
    diffing: false,
    hover: null,
    selection: null,
    dissection: null,
    thumbnails: null,
    dirty: true,
  };
}

export interface SceneMetrics {
  /** Frames actually drawn in the last second (skipped frames are not counted). */
  readonly fps: number;
  readonly lastFrameMs: number;
  readonly framesDrawn: number;
  readonly framesSkipped: number;
}

/*
 * Reused across frames so the hot path allocates nothing. The style objects
 * hold closures that read `state`, which never changes identity.
 */
interface SceneInternals {
  hoveredNode: number;
  hoveredEdge: number;
  selectedNode: number;
  selectedEdge: number;
}

export class Scene {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: Ctx2D;
  private readonly state: FrameState;
  private raf = 0;
  private running = false;

  private cssWidth = 0;
  private cssHeight = 0;
  private dpr = 1;

  private framesDrawn = 0;
  private framesSkipped = 0;
  private lastFrameMs = 0;
  private fps = 0;
  private fpsWindowStart = 0;
  private fpsWindowFrames = 0;
  private lastTimestamp = 0;

  /** Rebuilt only when the thumbnail data changes, not per frame. */
  private readonly thumbnailSheet = new ThumbnailSheet();

  private readonly internals: SceneInternals = {
    hoveredNode: -1,
    hoveredEdge: -1,
    selectedNode: -1,
    selectedEdge: -1,
  };

  private readonly edgeStyle: {
    wRef: number;
    colorblindSafe: boolean;
    hoveredEdge: number;
    selectedEdge: number;
    frozenLayers: ReadonlySet<number>;
  };

  private readonly nodeStyle: {
    normalizedActivation: (node: NodeLayout) => number | null;
    normalizedDelta: (node: NodeLayout) => number | null;
    isDead: (node: NodeLayout) => boolean;
    isFrozen: (node: NodeLayout) => boolean;
    isAblated: (node: NodeLayout) => boolean;
    hoveredNode: number;
    selectedNode: number;
  };

  constructor(canvas: HTMLCanvasElement, state: FrameState) {
    const context = canvas.getContext('2d');
    if (context === null) {
      throw new Error('Scene: this browser did not provide a 2D canvas context.');
    }
    this.canvas = canvas;
    this.ctx = context as unknown as Ctx2D;
    this.state = state;

    // Allocated once; fields are overwritten per frame.
    this.edgeStyle = {
      wRef: 1,
      colorblindSafe: false,
      hoveredEdge: -1,
      selectedEdge: -1,
      frozenLayers: state.frozenLayers,
    };
    this.nodeStyle = {
      normalizedActivation: state.normalizedActivation,
      normalizedDelta: state.normalizedDelta,
      isDead: state.isDead,
      isFrozen: state.isFrozen,
      isAblated: state.isAblated,
      hoveredNode: -1,
      selectedNode: -1,
    };
  }

  /** Match the backing store to the CSS size and devicePixelRatio (§2). */
  resize(cssWidth: number, cssHeight: number, devicePixelRatio: number): void {
    const dpr = Math.max(1, devicePixelRatio);
    if (cssWidth === this.cssWidth && cssHeight === this.cssHeight && dpr === this.dpr) return;
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.dpr = dpr;
    this.canvas.width = Math.round(cssWidth * dpr);
    this.canvas.height = Math.round(cssHeight * dpr);
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
    // The only transform in the system: everything else draws in CSS pixels.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.state.dirty = true;
  }

  get width(): number {
    return this.cssWidth;
  }

  get height(): number {
    return this.cssHeight;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.fpsWindowStart = now();
    const tick = (): void => {
      if (!this.running) return;
      this.frame();
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    if (this.raf !== 0) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  /** Draw once, regardless of the dirty flag. */
  renderNow(): void {
    this.state.dirty = true;
    this.frame();
  }

  private frame(): void {
    const timestamp = now();
    const delta = this.lastTimestamp === 0 ? 16 : timestamp - this.lastTimestamp;
    this.lastTimestamp = timestamp;

    /*
     * A playing choreography advances on wall-clock time, not on store changes,
     * so it marks the frame dirty itself. A paused one costs one comparison and
     * draws nothing — which is what keeps single-stepping and idle both free.
     */
    const dissection = this.state.dissection;
    if (dissection !== null && dissection.active) {
      dissection.advance(delta);
      if (dissection.status().playing) this.state.dirty = true;
    }

    if (!this.state.dirty) {
      this.framesSkipped++;
      return;
    }
    const started = now();
    this.draw();
    this.state.dirty = false;
    this.lastFrameMs = now() - started;
    this.framesDrawn++;
    this.fpsWindowFrames++;

    const elapsed = now() - this.fpsWindowStart;
    if (elapsed >= 1000) {
      this.fps = (this.fpsWindowFrames * 1000) / elapsed;
      this.fpsWindowFrames = 0;
      this.fpsWindowStart = now();
    }
  }

  /**
   * Fixed draw order (§6.1):
   *   background grid → edges → nodes → node glyphs → labels → overlays
   *
   * Pulses, formula cards and hover cards slot into this in Phase 4; the order
   * is fixed now so they have a defined place rather than being appended.
   */
  private draw(): void {
    const state = this.state;
    const ctx = this.ctx;

    resolveSelection(state, this.internals);

    drawGrid(ctx, state.viewport, this.cssWidth, this.cssHeight);

    this.edgeStyle.wRef = state.wRef;
    this.edgeStyle.colorblindSafe = state.colorblindSafe;
    this.edgeStyle.hoveredEdge = this.internals.hoveredEdge;
    this.edgeStyle.selectedEdge = this.internals.selectedEdge;
    this.edgeStyle.frozenLayers = state.frozenLayers;
    drawEdges(ctx, state.layout, state.viewport, state.weightOf, this.edgeStyle);

    this.nodeStyle.normalizedActivation = state.normalizedActivation;
    this.nodeStyle.normalizedDelta = state.normalizedDelta;
    this.nodeStyle.isDead = state.isDead;
    this.nodeStyle.isFrozen = state.isFrozen;
    this.nodeStyle.isAblated = state.isAblated;
    this.nodeStyle.hoveredNode = this.internals.hoveredNode;
    this.nodeStyle.selectedNode = this.internals.selectedNode;
    drawNodes(ctx, state.layout, state.viewport, this.nodeStyle);

    if (state.thumbnails !== null) {
      if (!this.thumbnailSheet.matches(state.thumbnails)) {
        this.thumbnailSheet.build(state.thumbnails, () => document.createElement('canvas'));
      }
      drawThumbnails(ctx as BlitContext, state.layout, state.viewport, this.thumbnailSheet);
    }

    drawNodeLabels(ctx, state.layout, state.viewport, state.captions);

    // Dissection sits above the network but below anything the pointer is
    // doing, per the fixed draw order.
    if (state.dissection !== null && state.dissection.active) {
      state.dissection.draw(ctx, state.viewport, this.cssWidth, this.cssHeight);
    }

    drawLegend(
      ctx,
      12,
      this.cssHeight - LEGEND_HEIGHT - 12,
      state.wRef,
      state.colorblindSafe,
      state.diffing,
    );

    if (state.layout.nodes.length === 0) {
      drawEmptyState(ctx, this.cssWidth, this.cssHeight);
    }
  }

  metrics(): SceneMetrics {
    return {
      fps: this.fps,
      lastFrameMs: this.lastFrameMs,
      framesDrawn: this.framesDrawn,
      framesSkipped: this.framesSkipped,
    };
  }
}

/** Split a Hit into the four indices the draw calls want, without allocating. */
export function resolveSelection(state: FrameState, out: SceneInternals): void {
  out.hoveredNode = -1;
  out.hoveredEdge = -1;
  out.selectedNode = -1;
  out.selectedEdge = -1;
  if (state.hover !== null) {
    if (state.hover.kind === 'node') out.hoveredNode = state.hover.index;
    else out.hoveredEdge = state.hover.index;
  }
  if (state.selection !== null) {
    if (state.selection.kind === 'node') out.selectedNode = state.selection.index;
    else out.selectedEdge = state.selection.index;
  }
}

function drawEmptyState(ctx: Ctx2D, width: number, height: number): void {
  // §9: empty states give direction.
  ctx.font = `400 13px ${FONTS.body}`;
  ctx.fillStyle = COLORS.textMid;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('No network yet. Choose an architecture to begin.', width / 2, height / 2);
  ctx.textAlign = 'left';
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
