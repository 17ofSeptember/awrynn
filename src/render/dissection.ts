/*
 * Dissection: the exact arithmetic behind one sample's journey through the
 * network.
 *
 * Spec §6.3 A, and the Phase 4 gate: "every number shown during a dissection
 * matches the engine's cached values exactly."
 *
 * That constraint drives the whole design of this file. Nothing here recomputes
 * the network. Every weight, bias, pre-activation, activation and delta is READ
 * from the engine's caches after a real forward/backward pass. The only
 * arithmetic performed is the per-term product w·a, because the engine never
 * materialises individual products — it goes straight to a dot product.
 *
 * So that one computation is checked rather than trusted: `residual` is the gap
 * between the terms as displayed and the z the engine actually cached. If a
 * card ever showed numbers that did not add up to the engine's z, the residual
 * would be non-zero and both the test suite and the dev-mode assertion catch
 * it. That is the difference between "the UI shows plausible arithmetic" and
 * "the UI shows the arithmetic that happened".
 */

import type { Network } from '../engine/network';
import type { Matrix } from '../engine/tensor';

/** One `(w)(a)` product in a neuron's sum. */
export interface Term {
  /** Unit in the previous layer that this term comes from. */
  readonly source: number;
  /** W[source, unit], read from the engine. */
  readonly weight: number;
  /** A^{l-1}[source], read from the engine. */
  readonly input: number;
  /** weight × input — the only value computed here rather than read. */
  readonly contribution: number;
}

/**
 * The normalization step, when a layer has one.
 *
 * Every field is read from the engine, including which statistics the last
 * forward actually used. Saying "running" or "batch" is not decoration: a
 * single sample stepped through the network is normalized by the running
 * estimate, and a reader comparing these numbers against a training-time batch
 * would otherwise be chasing a difference that is not an error.
 */
export interface Normalization {
  /** Σ w·a before the normalization, the sum the terms above add up to. */
  readonly u: number;
  /** The μ used, read from the engine. */
  readonly mean: number;
  /** The σ behind the 1/σ the engine used. */
  readonly sigma: number;
  /** (u − μ)/σ, as the engine computed it. */
  readonly normalized: number;
  /** γ[unit], read from the engine. */
  readonly gamma: number;
  /** Whether those statistics came from this batch or from the running estimate. */
  readonly fromBatch: boolean;
}

export interface NeuronDissection {
  /** DenseLayer index. */
  readonly layer: number;
  readonly unit: number;
  readonly terms: readonly Term[];
  /** The normalization between the sum and the bias, or null when there is none. */
  readonly normalization: Normalization | null;
  /** b[unit], read from the engine. With batch norm this is β. */
  readonly bias: number;
  /** Z[unit], read from the engine cache — authoritative. */
  readonly z: number;
  /** A[unit], read from the engine cache — authoritative. */
  readonly a: number;
  /**
   * The card's own arithmetic, carried through to the end. Should equal `z`.
   *
   * Without normalization that is Σ contributions + bias. With it, the sum is
   * normalized and scaled first, exactly as the card displays it, so the
   * residual keeps checking the whole chain rather than only its first link.
   */
  readonly assembled: number;
  /** |assembled − z|. Non-zero means the card would be lying. */
  readonly residual: number;
  readonly activation: string;
}

export interface OutputDissection {
  readonly prediction: number;
  readonly target: number;
  /** The loss the engine computed for this single sample. */
  readonly loss: number;
  readonly lossName: string;
}

export interface EdgeGradient {
  readonly layer: number;
  readonly row: number;
  readonly col: number;
  /** δ for the destination unit, from the engine's cached dZ. */
  readonly delta: number;
  /** A^{l-1}[row], from the engine. */
  readonly input: number;
  /** dW[row, col] from the engine — authoritative, already divided by B. */
  readonly gradient: number;
  /** −η · ∂L/∂w. */
  readonly step: number;
  readonly weightBefore: number;
}

export interface Dissection {
  /** Input values, index 0 of the column addressing used by the layout. */
  readonly inputs: readonly number[];
  /** Forward arithmetic, ordered layer by layer then unit by unit. */
  readonly neurons: readonly NeuronDissection[];
  readonly output: OutputDissection;
  readonly gradients: readonly EdgeGradient[];
  /** Largest residual across every neuron — the gate's diff, in one number. */
  readonly maxResidual: number;
  readonly learningRate: number;
}

/**
 * Run one sample forward and backward, then read out everything the dissection
 * view needs.
 *
 * The batch is deliberately a single row: §6.3 A is "a single sample walks
 * through the network", and a batch mean would make the numbers on the cards
 * unverifiable against any one sample.
 */
export function dissect(
  network: Network,
  x: Matrix,
  y: Matrix,
  row: number,
  learningRate: number,
): Dissection {
  if (row < 0 || row >= x.rows) {
    throw new Error(`dissect: row ${row} is outside [0, ${x.rows}).`);
  }
  if (x.cols !== network.inputSize) {
    throw new Error(
      `dissect: sample has ${x.cols} features but the network takes ${network.inputSize}.`,
    );
  }
  if (y.rows !== x.rows || y.cols !== network.outputSize) {
    throw new Error(
      `dissect: target [${y.rows}, ${y.cols}] does not match [${x.rows}, ${network.outputSize}].`,
    );
  }

  const sample: Matrix = {
    rows: 1,
    cols: x.cols,
    data: x.data.subarray(row * x.cols, (row + 1) * x.cols),
  };
  const targetRow: Matrix = {
    rows: 1,
    cols: y.cols,
    data: y.data.subarray(row * y.cols, (row + 1) * y.cols),
  };

  // training=true so A^{l-1} is retained for the backward pass, and nothing
  // else: inspect() silences dropout, which would make the displayed arithmetic
  // unreproducible, and freezes batch norm's running statistics, which would
  // otherwise drift every time someone stepped through a sample.
  const prediction: Matrix = network.inspect(() => {
    const out = network.forward(sample, true);
    network.backward(targetRow);
    return out;
  });

  const inputs = Array.from(sample.data);
  const neurons: NeuronDissection[] = [];
  let maxResidual = 0;

  network.layers.forEach((layer, layerIndex) => {
    const aPrev = layer.inputActivations;
    const z = layer.Z;
    const a = layer.A;
    if (aPrev === null || z === null || a === null) {
      throw new Error(`dissect: layer ${layerIndex} has no cached forward state.`);
    }

    for (let unit = 0; unit < layer.units; unit++) {
      const terms: Term[] = [];
      let sum = 0;
      for (let source = 0; source < layer.inputs; source++) {
        const weight = layer.W.data[source * layer.units + unit] as number;
        const input = aPrev.data[source] as number;
        const contribution = weight * input;
        sum += contribution;
        terms.push({ source, weight, input, contribution });
      }
      const bias = layer.b.data[unit] as number;
      const stats = layer.statistics;
      let normalization: Normalization | null = null;
      let assembled: number;
      if (stats === null) {
        assembled = sum + bias;
      } else {
        // Rebuilt from the engine's own μ and 1/σ rather than recomputed from
        // the batch, so the residual below compares the card against the
        // engine and not against a second implementation of the same formula.
        const mean = stats.mean[unit] as number;
        const invStd = stats.invStd[unit] as number;
        const gamma = layer.gamma.data[unit] as number;
        const normalized = (sum - mean) * invStd;
        normalization = {
          u: sum,
          mean,
          sigma: 1 / invStd,
          normalized,
          gamma,
          fromBatch: stats.fromBatch,
        };
        assembled = gamma * normalized + bias;
      }
      const cachedZ = z.data[unit] as number;
      const residual = Math.abs(assembled - cachedZ);
      maxResidual = Math.max(maxResidual, residual);

      neurons.push({
        layer: layerIndex,
        unit,
        terms,
        normalization,
        bias,
        z: cachedZ,
        a: a.data[unit] as number,
        assembled,
        residual,
        activation: layer.activationName,
      });
    }
  });

  const gradients: EdgeGradient[] = [];
  network.layers.forEach((layer, layerIndex) => {
    const aPrev = layer.inputActivations;
    const dZ = layer.dZ;
    if (aPrev === null || dZ === null) return;
    for (let row_ = 0; row_ < layer.inputs; row_++) {
      for (let col = 0; col < layer.units; col++) {
        const gradient = layer.dW.data[row_ * layer.units + col] as number;
        gradients.push({
          layer: layerIndex,
          row: row_,
          col,
          delta: dZ.data[col] as number,
          input: aPrev.data[row_] as number,
          gradient,
          step: -learningRate * gradient,
          weightBefore: layer.W.data[row_ * layer.units + col] as number,
        });
      }
    }
  });

  // Single-row batch, so the batch mean IS this sample's loss.
  const loss = network.dataLoss(prediction, targetRow);
  const output: OutputDissection = {
    prediction: prediction.data[0] as number,
    target: targetRow.data[0] as number,
    loss,
    lossName: network.lossName,
  };

  return { inputs, neurons, output, gradients, maxResidual, learningRate };
}

/**
 * Tolerance for the assembled-vs-cached comparison.
 *
 * Not zero: the engine sums a dot product in a different order than this file
 * accumulates terms, and float addition is not associative, so the two agree to
 * rounding rather than bit-for-bit. Anything above this is a real disagreement,
 * not reassociation.
 */
export const RESIDUAL_TOLERANCE = 1e-9;

/** True when every card would show arithmetic that actually adds up. */
export function dissectionIsFaithful(dissection: Dissection): boolean {
  return dissection.maxResidual <= RESIDUAL_TOLERANCE;
}
