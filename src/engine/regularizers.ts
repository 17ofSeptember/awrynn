/*
 * Regularization.
 *
 * Spec §4.9. Each piece lives where it can least easily go wrong:
 *
 *   L2         DenseLayer.applyL2 — it touches gradients that never leave the layer
 *   clipping   optimizers.ts — it acts on the flat gradient vector before an update
 *   dropout    DenseLayer — forward and backward share one mask buffer, so the
 *              two ends cannot drift apart
 *
 * What remains here is standardization, which is a property of the DATA rather
 * than of the network, and has to be fitted before a network exists.
 */

import type { Matrix } from './tensor';

/**
 * Per-feature standardization (x − μ)/σ (§4.9).
 *
 * μ and σ MUST come from the training split alone. Fitting them on all data
 * leaks validation statistics into training and inflates validation accuracy —
 * which is a lesson in the app (§7.8), so the leak is available as an explicit,
 * labelled option rather than being impossible to express.
 */
export interface Standardizer {
  readonly mean: Float64Array;
  readonly std: Float64Array;
}

/** Fit per-column mean and standard deviation. */
export function fitStandardizer(x: Matrix): Standardizer {
  const mean = new Float64Array(x.cols);
  const std = new Float64Array(x.cols);
  if (x.rows === 0) return { mean, std: std.fill(1) };

  for (let r = 0; r < x.rows; r++) {
    for (let c = 0; c < x.cols; c++) mean[c] = mean[c]! + x.data[r * x.cols + c]!;
  }
  for (let c = 0; c < x.cols; c++) mean[c] = mean[c]! / x.rows;

  for (let r = 0; r < x.rows; r++) {
    for (let c = 0; c < x.cols; c++) {
      const d = x.data[r * x.cols + c]! - mean[c]!;
      std[c] = std[c]! + d * d;
    }
  }
  for (let c = 0; c < x.cols; c++) {
    const variance = std[c]! / x.rows;
    // A constant feature has zero variance; dividing by it would produce NaN
    // for every sample. Leaving it at 1 passes the feature through unchanged,
    // which is the only sensible reading of "standardize a constant".
    std[c] = variance > 0 ? Math.sqrt(variance) : 1;
  }
  return { mean, std };
}

/** Apply a fitted standardizer in place. */
export function applyStandardizer(x: Matrix, stats: Standardizer): void {
  if (stats.mean.length !== x.cols) {
    throw new Error(
      `applyStandardizer: statistics cover ${stats.mean.length} features but the data has ${x.cols}.`,
    );
  }
  for (let r = 0; r < x.rows; r++) {
    const row = r * x.cols;
    for (let c = 0; c < x.cols; c++) {
      x.data[row + c] = (x.data[row + c]! - stats.mean[c]!) / stats.std[c]!;
    }
  }
}
