/*
 * Decision-boundary worker.
 *
 * Spec §6.4: "evaluate the network on a 120x120 grid over the input range,
 * render via ImageData, upscale, composite under the training points."
 *
 * The grid is evaluated as ONE batch of 14,400 rows rather than 14,400 forward
 * passes. The engine is already built around batched matrix multiplication, so
 * this is a single matmul chain rather than a loop, which is the difference
 * between comfortably inside the 16ms budget and nowhere near it.
 */

import { Network } from '../engine/network';
import type { SerializedNetwork } from '../engine/network';
import { createMatrix } from '../engine/tensor';

export interface BoundaryRequest {
  readonly type: 'boundary';
  /** Serialized so the worker can rebuild an identical network. */
  readonly network: SerializedNetwork;
  readonly resolution: number;
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
  /**
   * Also return a per-unit activation map for every hidden unit (§6.4).
   *
   * Computed in the SAME pass as the boundary: the grid has already been pushed
   * through the network, so every layer's activations are sitting in the
   * engine's caches and reading them out is nearly free. Evaluating them
   * separately would double the cost for no benefit.
   */
  readonly thumbnails: boolean;
  readonly thumbnailResolution: number;
  /** Echoed back so a stale reply can be discarded. */
  readonly nonce: number;
}

/** Where one unit's tile sits in the packed thumbnail buffer. */
export interface ThumbnailSlot {
  readonly layer: number;
  readonly unit: number;
  /** Offset into `values`, in samples. */
  readonly offset: number;
}

export interface BoundaryResponse {
  readonly type: 'boundary';
  readonly nonce: number;
  readonly resolution: number;
  /**
   * One value per grid cell.
   *
   * Binary: the sigmoid output in [0, 1]. Multi-class: the winning class index
   * plus its confidence, packed as `class + confidence`, so a single Float32
   * carries both what was predicted and how sure it was.
   */
  readonly values: Float32Array;
  readonly classes: number;
  /** All thumbnails packed end to end, normalized to [-1, 1]. */
  readonly thumbnails: Float32Array | null;
  readonly thumbnailSlots: readonly ThumbnailSlot[];
  readonly thumbnailResolution: number;
}

self.onmessage = (event: MessageEvent<BoundaryRequest>): void => {
  const request = event.data;
  if (request.type !== 'boundary') return;

  try {
    /*
     * Deserialize every time rather than caching an instance and swapping the
     * flat parameter array in.
     *
     * The flat layout is W-then-b PER LAYER, so reconstructing it from the
     * serialized form means interleaving two arrays in exactly the right order.
     * Getting that subtly wrong would not throw; it would silently draw a
     * boundary for a network that does not exist, which is the worst kind of
     * bug in a tool whose whole claim is fidelity. Construction costs about a
     * thousand RNG draws against a 14,400-row forward pass, so the saving was
     * never worth the risk.
     */
    const net = Network.deserialize(request.network);
    const n = request.resolution;
    const grid = createMatrix(n * n, 2);
    for (let row = 0; row < n; row++) {
      // Screen y grows downward while the data's y grows upward, so the grid is
      // built flipped here rather than flipping the image afterwards.
      const y = request.yMax - ((request.yMax - request.yMin) * row) / (n - 1);
      for (let col = 0; col < n; col++) {
        const x = request.xMin + ((request.xMax - request.xMin) * col) / (n - 1);
        grid.data[(row * n + col) * 2] = x;
        grid.data[(row * n + col) * 2 + 1] = y;
      }
    }

    const predictions = net.forward(grid, false);
    const classes = predictions.cols;
    const values = new Float32Array(n * n);

    if (classes === 1) {
      for (let i = 0; i < values.length; i++) values[i] = predictions.data[i] as number;
    } else {
      for (let i = 0; i < values.length; i++) {
        let best = 0;
        let bestValue = -Infinity;
        for (let c = 0; c < classes; c++) {
          const v = predictions.data[i * classes + c] as number;
          if (v > bestValue) {
            bestValue = v;
            best = c;
          }
        }
        // Confidence clamped below 1 so it can never carry into the class index.
        values[i] = best + Math.min(0.999, Math.max(0, bestValue));
      }
    }

    /*
     * Thumbnails, read from the caches the boundary pass just filled.
     *
     * Each unit's map is normalized by ITS OWN peak rather than a global one.
     * A unit that only ever fires weakly still has a shape worth seeing, and
     * normalizing globally would flatten it into a blank square next to a
     * louder neighbour. The point of these is the shape, not the magnitude.
     */
    let thumbnails: Float32Array | null = null;
    const slots: ThumbnailSlot[] = [];
    let thumbnailResolution = 0;

    if (request.thumbnails) {
      const t = Math.max(4, Math.min(n, request.thumbnailResolution));
      thumbnailResolution = t;
      const hidden = net.layers.slice(0, -1);
      const unitCount = hidden.reduce((sum, l) => sum + l.units, 0);
      thumbnails = new Float32Array(unitCount * t * t);

      const buffer = thumbnails;
      let offset = 0;
      hidden.forEach((layer, layerIndex) => {
        const a = layer.A;
        if (a === null) return;
        for (let unit = 0; unit < layer.units; unit++) {
          slots.push({ layer: layerIndex, unit, offset });
          let peak = 1e-12;
          for (let i = 0; i < a.rows; i++) {
            peak = Math.max(peak, Math.abs(a.data[i * a.cols + unit] as number));
          }
          // Subsampled from the n x n grid rather than recomputed at t x t.
          for (let row = 0; row < t; row++) {
            const sourceRow = Math.min(n - 1, Math.round((row * (n - 1)) / (t - 1)));
            for (let col = 0; col < t; col++) {
              const sourceCol = Math.min(n - 1, Math.round((col * (n - 1)) / (t - 1)));
              const value = a.data[(sourceRow * n + sourceCol) * a.cols + unit] as number;
              buffer[offset + row * t + col] = value / peak;
            }
          }
          offset += t * t;
        }
      });
    }

    const response: BoundaryResponse = {
      type: 'boundary',
      nonce: request.nonce,
      resolution: n,
      values,
      classes,
      thumbnails,
      thumbnailSlots: slots,
      thumbnailResolution,
    };
    // Transferred rather than copied: the worker has no further use for them.
    const transfer: Transferable[] = [values.buffer as ArrayBuffer];
    if (thumbnails !== null) transfer.push(thumbnails.buffer as ArrayBuffer);
    self.postMessage(response, { transfer });
  } catch (error) {
    self.postMessage({
      type: 'error',
      nonce: request.nonce,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
