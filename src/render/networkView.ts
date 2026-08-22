/*
 * The adapter between a Network and what the canvas draws.
 *
 * Not in §3's file skeleton, but the seam has to live somewhere: the draw
 * modules take plain callbacks so they stay testable without an engine, and
 * something has to turn a Network into those callbacks. Keeping it here rather
 * than in the store means the canvas can be driven by a snapshot from the
 * history scrubber (§6.6) just as easily as by the live network.
 *
 * Nothing in this file mutates the network. It reads cached forward/backward
 * state and normalizes it for display.
 */

import type { Network } from '../engine/network';
import type { Matrix } from '../engine/tensor';
import type { EdgeLayout, NodeLayout } from './layout';
import { smoothReference, weightReference } from './draw/edges';

export class NetworkView {
  private network: Network;

  /** Smoothed 95th-percentile |w|, so the picture cannot strobe (§6.2). */
  private smoothedRef = 0;

  /**
   * Per-layer activation for the displayed sample. Index 0 is the input layer,
   * so this is one longer than the network's layer list.
   */
  private activations: (Float64Array | null)[] = [];
  /** Per-layer |δ| for the displayed sample. Index 0 is unused. */
  private deltas: (Float64Array | null)[] = [];
  /**
   * Running max |a| per layer for UNBOUNDED activations, smoothed. Bounded
   * activations use their own range instead (§6.2).
   */
  private layerMax: number[] = [];
  /** Units with zero activation across the last epoch (§6.2). */
  private deadUnits: Set<string> = new Set();

  /**
   * Baseline for the A/B diff (§6.6).
   *
   * When set, edges encode Δw against this snapshot instead of w. That answers
   * a different question: not "what does the network weigh now" but "what did
   * training actually change", which is what makes an A/B comparison legible.
   */
  private diffBase: Float64Array | null = null;

  constructor(network: Network) {
    this.network = network;
    this.reset();
  }

  private reset(): void {
    const columns = this.network.layers.length + 1;
    this.activations = new Array<Float64Array | null>(columns).fill(null);
    this.deltas = new Array<Float64Array | null>(columns).fill(null);
    this.layerMax = new Array<number>(columns).fill(1);
    this.smoothedRef = 0;
    this.refreshWeightReference();
  }

  /**
   * Point the view at a different network without reallocating it. The layout
   * is not needed here: every callback addresses units by (layer, unit), which
   * is exactly what a NodeLayout carries.
   */
  retarget(network: Network): void {
    this.network = network;
    this.diffBase = null;
    this.weightDeltas = null;
    this.reset();
  }

  get weightRef(): number {
    return this.smoothedRef;
  }

  get diffing(): boolean {
    return this.diffBase !== null;
  }

  setDiffBase(base: Float64Array | null): void {
    // A base from a differently shaped network describes nothing here.
    this.diffBase = base !== null && base.length === this.network.params.length ? base : null;
    this.refreshWeightReference();
  }

  /** Recompute the smoothed weight reference. Call once per visual update. */
  refreshWeightReference(): void {
    // While diffing, the scale must describe the DELTAS, not the weights, or
    // every change would render as a hairline against a much larger reference.
    const source = this.diffBase === null ? this.network.params : this.deltaBuffer();
    const target = weightReference(source);
    this.smoothedRef = smoothReference(this.smoothedRef, target);
  }

  /**
   * Δ = current − base, into a reused buffer so no frame allocates.
   *
   * Named apart from `deltas`, which is the per-unit backprop δ cache: two very
   * different quantities that would be easy to confuse at a glance.
   */
  private weightDeltas: Float64Array | null = null;

  private deltaBuffer(): Float64Array {
    const base = this.diffBase as Float64Array;
    if (this.weightDeltas === null || this.weightDeltas.length !== base.length) {
      this.weightDeltas = new Float64Array(base.length);
    }
    for (let i = 0; i < base.length; i++) {
      this.weightDeltas[i] = (this.network.params[i] as number) - (base[i] as number);
    }
    return this.weightDeltas;
  }

  /**
   * Run one sample through the network and cache every layer's activation for
   * display. Eval mode, so dropout is the identity and what is shown is what
   * inference would produce.
   */
  captureSample(x: Matrix, row: number): boolean {
    if (row < 0 || row >= x.rows) {
      throw new Error(`NetworkView.captureSample: row ${row} is outside [0, ${x.rows}).`);
    }
    /*
     * Refuse a sample the network cannot accept, instead of letting forward()
     * throw. Store updates are synchronous and subscribers run inside set(), so
     * an exception raised here would abort whatever was mid-update — the render
     * layer must never be able to break the state layer.
     */
    if (x.cols !== this.network.inputSize) return false;
    const single: Matrix = {
      rows: 1,
      cols: x.cols,
      data: x.data.subarray(row * x.cols, (row + 1) * x.cols),
    };
    this.network.forward(single, false);

    this.activations[0] = Float64Array.from(single.data);
    this.network.layers.forEach((layer, i) => {
      const a = layer.A;
      this.activations[i + 1] = a === null ? null : Float64Array.from(a.data);
    });
    this.refreshLayerMaxima();
    return true;
  }

  /** Cache |δ| per unit from the last backward pass, for the node rings (§6.2). */
  captureDeltas(): void {
    this.network.layers.forEach((layer, i) => {
      const dZ = layer.dZ;
      if (dZ === null || dZ.rows === 0) {
        this.deltas[i + 1] = null;
        return;
      }
      // Mean |δ| across the batch: a single row would be an arbitrary sample,
      // and the question the ring answers is "is this unit learning", which is
      // a property of the batch.
      const out = new Float64Array(dZ.cols);
      for (let r = 0; r < dZ.rows; r++) {
        for (let c = 0; c < dZ.cols; c++) out[c] = out[c]! + Math.abs(dZ.data[r * dZ.cols + c]!);
      }
      for (let c = 0; c < out.length; c++) out[c] = out[c]! / dZ.rows;
      this.deltas[i + 1] = out;
    });
  }

  clearActivations(): void {
    this.activations.fill(null);
    this.deltas.fill(null);
  }

  setDeadUnits(units: Iterable<{ layer: number; unit: number }>): void {
    this.deadUnits = new Set();
    for (const { layer, unit } of units) this.deadUnits.add(`${layer}:${unit}`);
  }

  private refreshLayerMaxima(): void {
    this.network.layers.forEach((layer, i) => {
      if (layer.activation.range !== null) return; // bounded: uses its own range
      const values = this.activations[i + 1];
      if (values === undefined || values === null) return;
      let max = 0;
      for (const v of values) max = Math.max(max, Math.abs(v));
      // Smoothed for the same reason the weight reference is: an unbounded
      // layer's scale jumps around, and a jumping scale reads as the network
      // flickering rather than the scale moving.
      const previous = this.layerMax[i + 1] ?? 1;
      this.layerMax[i + 1] = max <= 0 ? previous : previous + (max - previous) * 0.2;
    });
  }

  /** Current display max for a layer, shown next to it when unbounded (§6.2). */
  layerScale(column: number): number | null {
    const layer = this.network.layers[column - 1];
    if (layer === undefined || layer.activation.range !== null) return null;
    return this.layerMax[column] ?? 1;
  }

  /* ---------------- the callbacks the draw layer wants ---------------- */

  weightOf = (edge: EdgeLayout): number => {
    const layer = this.network.layers[edge.layer];
    if (layer === undefined) return 0;
    const value = edge.isBias
      ? (layer.b.data[edge.col] ?? 0)
      : (layer.W.data[edge.row * layer.units + edge.col] ?? 0);
    if (this.diffBase === null) return value;

    // Δw against the pinned snapshot. Resolved through the flat index so the
    // W-then-b-per-layer layout is honoured rather than re-derived.
    let offset = 0;
    for (let i = 0; i < edge.layer; i++) {
      offset += this.network.layers[i]?.parameterCount ?? 0;
    }
    const index = edge.isBias
      ? offset + layer.inputs * layer.units + edge.col
      : offset + edge.row * layer.units + edge.col;
    return value - (this.diffBase[index] ?? 0);
  };

  normalizedActivation = (node: NodeLayout): number | null => {
    if (node.kind === 'bias') {
      // A bias satellite shows its own value, on the weight scale.
      const layer = this.network.layers[node.layer - 1];
      if (layer === undefined) return null;
      const b = layer.b.data[node.unit] ?? 0;
      return this.smoothedRef > 0 ? clamp(b / this.smoothedRef, -1, 1) : 0;
    }

    const values = this.activations[node.layer];
    if (values === undefined || values === null) return null;
    const a = values[node.unit];
    if (a === undefined) return null;

    if (node.layer === 0) {
      // Inputs have no activation function; normalize against the layer max.
      const max = this.layerMax[0] ?? 1;
      return max > 0 ? clamp(a / max, -1, 1) : 0;
    }

    const layer = this.network.layers[node.layer - 1];
    if (layer === undefined) return null;
    const range = layer.activation.range;
    if (range !== null) {
      // Bounded: map through the activation's TRUE range, so a sigmoid at 0.5
      // reads as mid-scale rather than as whatever this batch happened to hit.
      const [lo, hi] = range;
      return clamp((2 * (a - lo)) / (hi - lo) - 1, -1, 1);
    }
    const max = this.layerMax[node.layer] ?? 1;
    return max > 0 ? clamp(a / max, -1, 1) : 0;
  };

  normalizedDelta = (node: NodeLayout): number | null => {
    if (node.kind === 'bias' || node.layer === 0) return null;
    const values = this.deltas[node.layer];
    if (values === undefined || values === null) return null;
    const d = values[node.unit];
    if (d === undefined) return null;
    // Normalized against the largest |δ| in the same layer, so the ring answers
    // "which units in this layer are learning" rather than being swamped by a
    // different layer's scale.
    let max = 0;
    for (const v of values) max = Math.max(max, v);
    return max > 0 ? d / max : 0;
  };

  isDead = (node: NodeLayout): boolean => {
    if (node.kind === 'bias' || node.layer === 0) return false;
    return this.deadUnits.has(`${node.layer - 1}:${node.unit}`);
  };

  isFrozen = (node: NodeLayout): boolean => {
    if (node.layer === 0) return false;
    return this.network.layers[node.layer - 1]?.frozen ?? false;
  };

  isAblated = (node: NodeLayout): boolean => {
    if (node.layer === 0 || node.kind === 'bias') return false;
    return this.network.layers[node.layer - 1]?.ablated ?? false;
  };

  frozenLayers(): ReadonlySet<number> {
    const frozen = new Set<number>();
    this.network.layers.forEach((layer, i) => {
      if (layer.frozen) frozen.add(i);
    });
    return frozen;
  }

  /**
   * Column captions: "input", each layer's activation, "output".
   *
   * A normalized layer says so. Batch norm changes what the column computes and
   * leaves no other mark on the picture, so without the caption the canvas
   * would be showing two different networks identically.
   */
  captions(): string[] {
    const out: string[] = [`input ×${this.network.inputSize}`];
    this.network.layers.forEach((layer, i) => {
      const last = i === this.network.layers.length - 1;
      const norm = layer.batchNorm ? ' ·bn' : '';
      out.push(`${last ? 'output' : `h${i + 1}`} ·${layer.activationName}${norm} ×${layer.units}`);
    });
    return out;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
