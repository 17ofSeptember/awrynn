/*
 * DenseLayer — forward, backward, and parameter access.
 *
 * Spec §4.2, §4.3. Shapes, fixed by §4.1:
 *
 *   W  [n_{l-1}, n_l]     b  [1, n_l]
 *   A^{l-1} [B, n_{l-1}]  Z, A  [B, n_l]
 *
 * THE DIVIDE-BY-B CONVENTION (§4.3), which is the single easiest thing to get
 * wrong in this file: dZ flowing through the network is the UNAVERAGED
 * per-sample gradient. The division by B happens exactly once, at the moment
 * the parameter gradients are formed:
 *
 *   dW = (A^{l-1})ᵀ · dZ / B          <- divided here
 *   db = colSum(dZ) / B               <- and here
 *   dA^{l-1} = dZ · Wᵀ                <- NOT divided; it keeps flowing unaveraged
 *
 * Divide in dA^{l-1} as well and every gradient picks up an extra 1/B per
 * layer, which looks like "training is just slow" rather than like a bug.
 *
 * BATCH NORMALIZATION, and why it does not add a second bias.
 *
 * With `batchNorm` on, the layer computes
 *
 *   U = A^{l-1} · W                     (no bias yet)
 *   Z = γ ⊙ (U − μ) / √(σ² + ε) + b
 *
 * where μ and σ² are per-unit statistics over the batch. Textbooks call that
 * trailing shift β and drop the dense layer's bias, because adding a constant
 * before a step that subtracts the mean does exactly nothing: shift every u in
 * a column by δ and μ shifts by δ too, so U − μ is unchanged and ∂L/∂b is
 * identically zero. Carrying both would put a parameter on screen that provably
 * cannot affect anything.
 *
 * So this layer does not add β. Its EXISTING bias moves: instead of being added
 * before the activation it is added after the normalization, which is the same
 * position β occupies. It keeps its name, its storage slot, its bias satellite
 * on the canvas and its line in the dissection card, and it keeps meaning "the
 * shift applied just before the activation". The layer gains exactly one new
 * parameter vector, γ.
 *
 * Z remains the true pre-activation, so φ′(Z, A) is untouched and the dead-unit
 * and saturation metrics keep measuring what they always measured.
 */

import type { Activation, ActivationName, ElementwiseActivation } from './activations';
import { applyActivation, applyActivationDerivative, getActivation, isElementwise } from './activations';
import type { InitScheme } from './init';
import { initBiases, initWeights } from './init';
import type { Rng } from './rng';
import type { Matrix } from './tensor';
import {
  addRowVector,
  colSum,
  createMatrix,
  ensureShape,
  matmul,
  matmulAT,
  matmulBT,
  scaleInPlace,
} from './tensor';

export interface LayerSpec {
  readonly units: number;
  readonly activation: ActivationName;
  readonly leakyAlpha?: number | undefined;
  /**
   * Normalize the affine output across the batch before the activation.
   *
   * When on, the layer's bias becomes β. See BATCH NORMALIZATION above.
   */
  readonly batchNorm?: boolean | undefined;
}

/**
 * Momentum for the running statistics: new ← (1−m)·old + m·batch.
 *
 * 0.1 is the usual default and is what PyTorch and Keras both settle on. It
 * corresponds to an effective window of about ten batches, which is short
 * enough to follow a network that is still moving and long enough that one
 * unlucky batch does not define the eval-time behaviour.
 */
export const BATCH_NORM_MOMENTUM = 0.1;

/**
 * The ε inside √(σ² + ε).
 *
 * Guards the division when a unit is constant across the batch, which is not a
 * corner case: a dead ReLU is exactly that.
 */
export const BATCH_NORM_EPSILON = 1e-5;

export class DenseLayer {
  readonly inputs: number;
  readonly units: number;
  readonly activationName: ActivationName;
  readonly leakyAlpha: number | undefined;
  readonly activation: Activation;

  /*
   * W, b, dW and db are rebound onto network-owned contiguous storage by
   * bindStorage() immediately after construction, so they are views rather than
   * independent allocations. Never reassign `.data` directly — use
   * setWeights()/bindStorage(), which preserve the aliasing that optimizers,
   * gradient clipping and history snapshots depend on.
   */
  /** W: [inputs, units] */
  W: Matrix;
  /** b: [1, units] */
  b: Matrix;
  /** dW: [inputs, units] — already divided by B. */
  dW: Matrix;
  /** db: [1, units] — already divided by B. */
  db: Matrix;

  /** Whether this layer normalizes across the batch before the activation. */
  readonly batchNorm: boolean;
  /**
   * γ: [1, units], and its gradient. Zero-length views when batchNorm is off,
   * so the storage layout stays exactly what it was before this existed.
   */
  gamma: Matrix;
  /** dγ: [1, units] — already divided by B. */
  dGamma: Matrix;

  /*
   * Running statistics: MEASURED, not learned.
   *
   * They are updated by the forward pass rather than by an optimizer, which is
   * why they live outside `params` in the network's parallel `buffers` array.
   * Putting them in with the weights would have them counted as parameters on
   * screen, decayed by AdamW, and clipped by the global-norm clipper, none of
   * which is meaningful for an estimate of a mean.
   *
   * They still have to travel: a link or a save file that omitted them would
   * reopen a trained network whose eval-mode predictions, and therefore whose
   * decision boundary, are wrong.
   */
  runningMean: Float64Array;
  runningVar: Float64Array;

  /** Excluded from optimizer updates but still forward-propagates (§6.5). */
  frozen = false;
  /** Output forced to zero, to demonstrate redundancy (§6.5). */
  ablated = false;
  /**
   * Inverted-dropout probability for THIS layer's output (§4.9). 0 disables.
   *
   * Dropout lives on the layer rather than in the trainer so that the forward
   * mask and the backward mask cannot drift apart — they are the same buffer,
   * applied at both ends of the same object.
   */
  dropout = 0;

  /* Forward caches. Backprop needs A^{l-1} and Z (§4.2). */
  private aPrev: Matrix | null = null;
  private zCache: Matrix | null = null;
  private aCache: Matrix | null = null;

  /*
   * `aCache` always holds the TRUE activation φ(Z), because df(z, a) reads it
   * and tanh/sigmoid derivatives are expressed in terms of a. Anything that
   * modifies what the next layer sees — dropout or ablation — goes into a
   * separate masked buffer, never into aCache. Multiplying the mask into
   * aCache would silently corrupt φ′ for every bounded activation.
   */
  private outputCache: Matrix | null = null;
  private maskBuffer: Float64Array | null = null;
  private maskActive = false;

  /* Backward caches. */
  private dZCache: Matrix | null = null;
  private dAPrevCache: Matrix | null = null;

  /* Batch-norm caches, all null unless batchNorm is on. */
  /** U = A^{l-1}·W, before normalization and before the bias. */
  private uCache: Matrix | null = null;
  /** X̂ = (U − μ)/√(σ²+ε), the normalized pre-activation. */
  private xHatCache: Matrix | null = null;
  /** The μ ACTUALLY used by the last forward, batch or running. */
  private usedMean: Float64Array | null = null;
  /** 1/√(σ²+ε) for that same μ. */
  private usedInvStd: Float64Array | null = null;
  /** Whether the last forward normalized by batch statistics or running ones. */
  private usedBatchStatistics = false;
  /** dU: [B, units], the gradient that reaches W once normalization is undone. */
  private dUCache: Matrix | null = null;
  /**
   * Suppresses the running-statistics update.
   *
   * The gradient check runs thousands of extra forward passes to build its
   * numerical estimates. Those passes must not move an estimate the network
   * will later predict with, exactly as the check disables dropout so that L is
   * a function rather than a distribution.
   */
  freezeStatistics = false;

  constructor(inputs: number, spec: LayerSpec, scheme: InitScheme, rng: Rng) {
    if (inputs <= 0 || spec.units <= 0) {
      throw new Error(
        `DenseLayer: inputs and units must be positive, got inputs=${inputs}, units=${spec.units}.`,
      );
    }
    this.inputs = inputs;
    this.units = spec.units;
    this.activationName = spec.activation;
    this.leakyAlpha = spec.leakyAlpha;
    this.activation = getActivation(spec.activation, { leakyAlpha: spec.leakyAlpha });
    this.W = initWeights(scheme, inputs, spec.units, rng);
    this.b = initBiases(spec.units);
    this.dW = createMatrix(inputs, spec.units);
    this.db = createMatrix(1, spec.units);

    this.batchNorm = spec.batchNorm === true;
    const gammaSize = this.batchNorm ? spec.units : 0;
    this.gamma = { rows: 1, cols: gammaSize, data: new Float64Array(gammaSize).fill(1) };
    this.dGamma = { rows: 1, cols: gammaSize, data: new Float64Array(gammaSize) };
    // γ starts at 1 and b at 0, so a freshly built batch-norm layer passes the
    // normalized values straight through. Any other choice would make turning
    // the switch on change the network's function for two unrelated reasons.
    this.runningMean = new Float64Array(gammaSize);
    this.runningVar = new Float64Array(gammaSize).fill(1);
  }

  /**
   * Z^l = A^{l-1} · W^l + 1_B · b^l ; A^l = φ(Z^l)
   *
   * `rng` supplies the dropout mask and must come from the 'dropout' stream
   * (§4.7). Pass null — or training=false — to run in eval mode, where dropout
   * is the identity.
   */
  forward(aPrev: Matrix, training: boolean, rng: Rng | null = null): Matrix {
    if (aPrev.cols !== this.inputs) {
      throw new Error(
        `DenseLayer.forward: input [${aPrev.rows}, ${aPrev.cols}] does not match W [${this.inputs}, ${this.units}].`,
      );
    }
    const z = ensureShape(this.zCache, aPrev.rows, this.units);
    this.zCache = z;
    if (this.batchNorm) {
      this.normalizedForward(aPrev, training, z);
    } else {
      matmul(aPrev, this.W, z);
      addRowVector(z, this.b, z);
    }

    const a = ensureShape(this.aCache, aPrev.rows, this.units);
    this.aCache = a;
    applyActivation(this.activation, z, a);

    // Only training needs A^{l-1} retained; inference can drop it immediately.
    this.aPrev = training ? aPrev : null;

    const wantsDropout = training && this.dropout > 0 && rng !== null;
    if (!this.ablated && !wantsDropout) {
      this.maskActive = false;
      this.outputCache = a;
      return a;
    }

    // One mask covers both effects: ablation is a permanent all-zero mask, and
    // the two compose correctly when both are on.
    const size = a.data.length;
    if (this.maskBuffer === null || this.maskBuffer.length !== size) {
      this.maskBuffer = new Float64Array(size);
    }
    const mask = this.maskBuffer;
    if (this.ablated) {
      mask.fill(0);
    } else {
      const keep = 1 - this.dropout;
      const scale = 1 / keep;
      const stream = rng as Rng;
      for (let i = 0; i < size; i++) {
        // Inverted dropout: kept units scale up by 1/(1−p) NOW, so eval-time
        // inference needs no compensation at all.
        mask[i] = stream.next() < keep ? scale : 0;
      }
    }

    const out = ensureShape(this.outputCache === a ? null : this.outputCache, a.rows, a.cols);
    this.outputCache = out;
    for (let i = 0; i < size; i++) out.data[i] = a.data[i]! * mask[i]!;
    this.maskActive = true;
    return out;
  }

  /**
   * Z = γ ⊙ (U − μ)/√(σ² + ε) + b, where U = A^{l-1}·W.
   *
   * WHICH μ AND σ² are used is the whole subject. Training uses the statistics
   * of the batch in front of it, which is what makes the trick work and what
   * makes every sample's output depend on its neighbours. Evaluation uses a
   * running estimate, because a single sample has no spread of its own and
   * because a prediction that changed depending on what you batched it with
   * would not be a prediction.
   *
   * A batch of one is the awkward case: its variance is zero, so normalizing by
   * it would map every unit to exactly zero and cut the gradient at that layer
   * outright, silently. The last batch of an epoch can easily have one sample
   * in it, so this cannot be an error. Such a batch is normalized by the
   * running estimate instead, and contributes nothing to it.
   */
  private normalizedForward(aPrev: Matrix, training: boolean, z: Matrix): void {
    const batch = aPrev.rows;
    const units = this.units;

    const u = ensureShape(this.uCache, batch, units);
    this.uCache = u;
    matmul(aPrev, this.W, u);

    const xHat = ensureShape(this.xHatCache, batch, units);
    this.xHatCache = xHat;
    if (this.usedMean === null || this.usedMean.length !== units) {
      this.usedMean = new Float64Array(units);
      this.usedInvStd = new Float64Array(units);
    }
    const mean = this.usedMean;
    const invStd = this.usedInvStd as Float64Array;

    const useBatch = training && batch >= 2;
    this.usedBatchStatistics = useBatch;

    if (useBatch) {
      for (let j = 0; j < units; j++) {
        let sum = 0;
        for (let i = 0; i < batch; i++) sum += u.data[i * units + j] as number;
        const m = sum / batch;
        let sq = 0;
        for (let i = 0; i < batch; i++) {
          const d = (u.data[i * units + j] as number) - m;
          sq += d * d;
        }
        // The BIASED variance normalizes the batch, because dividing by B is
        // what makes X̂ have unit variance within it.
        const variance = sq / batch;
        mean[j] = m;
        invStd[j] = 1 / Math.sqrt(variance + BATCH_NORM_EPSILON);

        if (!this.freezeStatistics) {
          // The running estimate takes the UNBIASED variance, because there it
          // is estimating the population rather than describing this batch.
          // PyTorch draws the same distinction; it is worth a line of code.
          const unbiased = sq / (batch - 1);
          this.runningMean[j] =
            (1 - BATCH_NORM_MOMENTUM) * (this.runningMean[j] as number) + BATCH_NORM_MOMENTUM * m;
          this.runningVar[j] =
            (1 - BATCH_NORM_MOMENTUM) * (this.runningVar[j] as number) +
            BATCH_NORM_MOMENTUM * unbiased;
        }
      }
    } else {
      for (let j = 0; j < units; j++) {
        mean[j] = this.runningMean[j] as number;
        invStd[j] = 1 / Math.sqrt((this.runningVar[j] as number) + BATCH_NORM_EPSILON);
      }
    }

    for (let i = 0; i < batch; i++) {
      const row = i * units;
      for (let j = 0; j < units; j++) {
        const hat = ((u.data[row + j] as number) - (mean[j] as number)) * (invStd[j] as number);
        xHat.data[row + j] = hat;
        z.data[row + j] = (this.gamma.data[j] as number) * hat + (this.b.data[j] as number);
      }
    }
  }

  /**
   * Backward from dA (the general path): dZ = dA ⊙ φ'(Z), then parameters.
   * Returns dA^{l-1}.
   */
  backwardFromDA(dA: Matrix): Matrix {
    if (!isElementwise(this.activation)) {
      // Softmax only ever appears as the output layer paired with cce, where
      // network.ts takes the fused path. Reaching here means a validation hole.
      throw new Error(
        'DenseLayer.backwardFromDA: softmax has a dense Jacobian and must use the fused output path (§4.3).',
      );
    }
    const z = this.requireCache(this.zCache, 'Z');
    const a = this.requireCache(this.aCache, 'A');

    // The mask applies to the incoming dA, BEFORE φ′. A unit that was dropped
    // made no contribution to the loss, so it must receive no gradient — and
    // the gradient that flows on to earlier layers has to be masked too, which
    // is only true if this happens here rather than after the fact.
    if (this.maskActive && this.maskBuffer !== null) {
      if (this.maskBuffer.length !== dA.data.length) {
        throw new Error(
          `DenseLayer.backwardFromDA: mask holds ${this.maskBuffer.length} values but dA holds ${dA.data.length}. Forward and backward saw different batch sizes.`,
        );
      }
      for (let i = 0; i < dA.data.length; i++) {
        dA.data[i] = dA.data[i]! * this.maskBuffer[i]!;
      }
    }

    const dZ = ensureShape(this.dZCache, z.rows, z.cols);
    this.dZCache = dZ;
    // `a` is the PRISTINE activation, never the masked output.
    applyActivationDerivative(this.activation as ElementwiseActivation, z, a, dA, dZ);
    return this.backwardFromDZ(dZ);
  }

  /**
   * Backward from dZ directly (the fused output path, §4.3), and the shared
   * tail of both routes. `dZ` is the unaveraged per-sample gradient.
   */
  backwardFromDZ(dZ: Matrix): Matrix {
    const aPrev = this.requireCache(this.aPrev, 'A^{l-1}');
    const batch = dZ.rows;
    if (batch === 0) {
      throw new Error('DenseLayer.backwardFromDZ: batch size is zero.');
    }

    if (dZ !== this.dZCache) {
      // Retain the fused dZ so the dissection view can show the real values.
      const held = ensureShape(this.dZCache, dZ.rows, dZ.cols);
      held.data.set(dZ.data);
      this.dZCache = held;
    }

    // With batch norm, the gradient that reaches W is not dZ: the normalization
    // sits between them. dU replaces it from here down, and dγ and db are taken
    // there rather than here.
    const dU = this.batchNorm ? this.normalizedBackward(dZ) : dZ;

    // The one and only division by B (§4.3).
    matmulAT(aPrev, dU, this.dW);
    scaleInPlace(this.dW, 1 / batch);

    if (!this.batchNorm) {
      colSum(dZ, this.db);
      scaleInPlace(this.db, 1 / batch);
    }

    // dA^{l-1} keeps the per-sample scale — deliberately NOT divided.
    const dAPrev = ensureShape(this.dAPrevCache, batch, this.inputs);
    this.dAPrevCache = dAPrev;
    return matmulBT(dU, this.W, dAPrev);
  }

  /**
   * Differentiate the normalization: dZ in, dU out, with dγ and db taken here.
   *
   * The compact form of the batch-statistics case,
   *
   *   dU_i = (s/B)·( B·dX̂_i − Σ_k dX̂_k − X̂_i·Σ_k dX̂_k·X̂_k )
   *
   * looks nothing like the chain rule until you notice what the two sums are:
   * every sample's u affects every other sample's z, through μ and through σ².
   * The middle term is the mean's share and the last is the variance's. Both
   * vanish in the running-statistics case, where μ and σ² are constants that
   * this batch had no part in, which is why the eval branch is two lines.
   *
   * The 1/B here is the statistics' own, from ∂μ/∂u_i = 1/B. It is NOT the
   * loss-averaging division, which still happens exactly once, later, at dW.
   */
  private normalizedBackward(dZ: Matrix): Matrix {
    const batch = dZ.rows;
    const units = this.units;
    const xHat = this.requireCache(this.xHatCache, 'X̂');
    const invStd = this.usedInvStd as Float64Array;

    // dγ_j = Σ_i dZ_ij·X̂_ij and db_j = Σ_i dZ_ij, both then averaged over B.
    for (let j = 0; j < units; j++) {
      let dg = 0;
      let dbeta = 0;
      for (let i = 0; i < batch; i++) {
        const dz = dZ.data[i * units + j] as number;
        dg += dz * (xHat.data[i * units + j] as number);
        dbeta += dz;
      }
      this.dGamma.data[j] = dg / batch;
      this.db.data[j] = dbeta / batch;
    }

    const dU = ensureShape(this.dUCache, batch, units);
    this.dUCache = dU;

    for (let j = 0; j < units; j++) {
      const g = this.gamma.data[j] as number;
      const s = invStd[j] as number;

      if (!this.usedBatchStatistics) {
        // μ and σ² were constants: dU = dZ·γ·s, and nothing couples the rows.
        for (let i = 0; i < batch; i++) {
          dU.data[i * units + j] = (dZ.data[i * units + j] as number) * g * s;
        }
        continue;
      }

      let sumDxHat = 0;
      let sumDxHatXHat = 0;
      for (let i = 0; i < batch; i++) {
        const dxHat = (dZ.data[i * units + j] as number) * g;
        sumDxHat += dxHat;
        sumDxHatXHat += dxHat * (xHat.data[i * units + j] as number);
      }
      const scale = s / batch;
      for (let i = 0; i < batch; i++) {
        const dxHat = (dZ.data[i * units + j] as number) * g;
        dU.data[i * units + j] =
          scale * (batch * dxHat - sumDxHat - (xHat.data[i * units + j] as number) * sumDxHatXHat);
      }
    }
    return dU;
  }

  private requireCache(m: Matrix | null, what: string): Matrix {
    if (m === null) {
      throw new Error(
        `DenseLayer: ${what} is not cached. forward(x, true) must run before backward().`,
      );
    }
    return m;
  }

  /** Cached pre-activations from the last forward pass. Null before any forward. */
  get Z(): Matrix | null {
    return this.zCache;
  }

  /** Cached activations φ(Z) from the last forward pass, before any mask. */
  get A(): Matrix | null {
    return this.aCache;
  }

  /** What was actually passed downstream — A after dropout/ablation. */
  get output(): Matrix | null {
    return this.outputCache;
  }

  /** The dropout/ablation mask from the last forward pass, or null (§6.2). */
  get lastMask(): Float64Array | null {
    return this.maskActive ? this.maskBuffer : null;
  }

  /** Cached input activations, retained only during training. */
  get inputActivations(): Matrix | null {
    return this.aPrev;
  }

  /** Cached dZ from the last backward pass — what the canvas draws as δ (§6.2). */
  get dZ(): Matrix | null {
    return this.dZCache;
  }

  /** U = A^{l-1}·W before normalization, or null when batch norm is off. */
  get U(): Matrix | null {
    return this.batchNorm ? this.uCache : null;
  }

  /** X̂, the normalized pre-activation, or null when batch norm is off. */
  get normalized(): Matrix | null {
    return this.batchNorm ? this.xHatCache : null;
  }

  /**
   * The statistics the LAST forward actually used, batch or running.
   *
   * Deliberately not "the batch statistics": the dissection view has to be able
   * to say which of the two produced the number on screen, and a single sample
   * stepped through the network is normalized by the running estimate.
   */
  get statistics(): {
    readonly mean: Float64Array;
    readonly invStd: Float64Array;
    readonly fromBatch: boolean;
  } | null {
    if (!this.batchNorm || this.usedMean === null || this.usedInvStd === null) return null;
    return {
      mean: this.usedMean,
      invStd: this.usedInvStd,
      fromBatch: this.usedBatchStatistics,
    };
  }

  /** Number of trainable scalars: |W| + |b|, plus |γ| when batch norm is on. */
  get parameterCount(): number {
    return this.W.data.length + this.b.data.length + this.gamma.data.length;
  }

  /**
   * Measured scalars carried alongside the parameters: μ̂ and σ̂², or none.
   *
   * Separate from parameterCount so that "51 params" on screen keeps counting
   * only the numbers gradient descent is actually moving.
   */
  get bufferCount(): number {
    return this.runningMean.length + this.runningVar.length;
  }

  /** Σ W² — the L2 objective term. Biases are excluded by construction (§4.9). */
  weightSumSquares(): number {
    let sum = 0;
    for (let i = 0; i < this.W.data.length; i++) {
      const v = this.W.data[i]!;
      sum += v * v;
    }
    return sum;
  }

  /** Add λ·W to dW. Never touches biases (§4.9). */
  applyL2(lambda: number): void {
    if (lambda === 0) return;
    for (let i = 0; i < this.W.data.length; i++) {
      this.dW.data[i] = this.dW.data[i]! + lambda * this.W.data[i]!;
    }
  }

  /** Discard forward/backward caches without touching parameters. */
  clearCaches(): void {
    this.aPrev = null;
    this.zCache = null;
    this.aCache = null;
    this.dZCache = null;
    this.dAPrevCache = null;
    this.outputCache = null;
    this.maskActive = false;
    this.uCache = null;
    this.xHatCache = null;
    this.dUCache = null;
  }

  spec(): LayerSpec {
    const base: LayerSpec = { units: this.units, activation: this.activationName };
    const withAlpha =
      this.leakyAlpha === undefined ? base : { ...base, leakyAlpha: this.leakyAlpha };
    // Omitted rather than written as false, so a spec round-trips to the same
    // object it came from and a share link made before batch norm existed still
    // encodes identically.
    return this.batchNorm ? { ...withAlpha, batchNorm: true } : withAlpha;
  }

  /** Deep copy of parameters; caches are not carried over. */
  copyParametersFrom(other: DenseLayer): void {
    if (other.inputs !== this.inputs || other.units !== this.units) {
      throw new Error(
        `DenseLayer.copyParametersFrom: shape mismatch [${this.inputs}, ${this.units}] vs [${other.inputs}, ${other.units}].`,
      );
    }
    this.W.data.set(other.W.data);
    this.b.data.set(other.b.data);
    if (this.batchNorm !== other.batchNorm) {
      throw new Error(
        'DenseLayer.copyParametersFrom: one layer normalizes across the batch and the other does not.',
      );
    }
    this.gamma.data.set(other.gamma.data);
    this.runningMean.set(other.runningMean);
    this.runningVar.set(other.runningVar);
  }

  /**
   * Copy values in place. Deliberately NOT a reassignment: the matrices are
   * views into the network's contiguous parameter storage, and replacing them
   * would silently detach this layer from the optimizer.
   */
  setWeights(w: Matrix, b: Matrix): void {
    if (w.rows !== this.inputs || w.cols !== this.units) {
      throw new Error(
        `DenseLayer.setWeights: W must be [${this.inputs}, ${this.units}], got [${w.rows}, ${w.cols}].`,
      );
    }
    if (b.rows !== 1 || b.cols !== this.units) {
      throw new Error(`DenseLayer.setWeights: b must be [1, ${this.units}], got [${b.rows}, ${b.cols}].`);
    }
    this.W.data.set(w.data);
    this.b.data.set(b.data);
  }

  /**
   * Copy γ and the running statistics in place.
   *
   * In place for the same reason setWeights is: these are views into the
   * network's storage, and reassigning them would detach the layer from the
   * optimizer and from every snapshot taken of it.
   */
  setNormalization(
    gamma: readonly number[] | Float64Array,
    runningMean: readonly number[] | Float64Array,
    runningVar: readonly number[] | Float64Array,
  ): void {
    if (!this.batchNorm) {
      throw new Error('DenseLayer.setNormalization: this layer does not normalize across the batch.');
    }
    const expect = (values: readonly number[] | Float64Array, what: string): void => {
      if (values.length !== this.units) {
        throw new Error(
          `DenseLayer.setNormalization: ${what} must hold ${this.units} values, got ${values.length}.`,
        );
      }
    };
    expect(gamma, 'γ');
    expect(runningMean, 'μ̂');
    expect(runningVar, 'σ̂²');
    this.gamma.data.set(gamma);
    this.runningMean.set(runningMean);
    this.runningVar.set(runningVar);
  }

  /**
   * Rebind the parameter and gradient matrices onto externally-owned storage,
   * preserving whatever values they currently hold.
   *
   * Called once by the Network constructor so that every parameter in the
   * network lives in one contiguous Float64Array. That is what lets optimizers
   * take the flat arrays §4.8 describes, lets gradient clipping compute a
   * global norm without gathering, and makes a history snapshot a memcpy.
   */
  bindStorage(storage: {
    readonly w: Float64Array;
    readonly b: Float64Array;
    readonly dw: Float64Array;
    readonly db: Float64Array;
    /** γ and dγ, zero-length when batch norm is off. */
    readonly gamma: Float64Array;
    readonly dgamma: Float64Array;
    /** μ̂ and σ̂², slices of the network's `buffers`, zero-length when off. */
    readonly runningMean: Float64Array;
    readonly runningVar: Float64Array;
  }): void {
    const expect = (actual: Float64Array, wanted: number, what: string): void => {
      if (actual.length !== wanted) {
        throw new Error(
          `DenseLayer.bindStorage: ${what} storage must hold ${wanted} values, got ${actual.length}.`,
        );
      }
    };
    const gammaSize = this.gamma.data.length;
    expect(storage.w, this.inputs * this.units, 'W');
    expect(storage.b, this.units, 'b');
    expect(storage.dw, this.inputs * this.units, 'dW');
    expect(storage.db, this.units, 'db');
    expect(storage.gamma, gammaSize, 'γ');
    expect(storage.dgamma, gammaSize, 'dγ');
    expect(storage.runningMean, gammaSize, 'μ̂');
    expect(storage.runningVar, gammaSize, 'σ̂²');

    storage.w.set(this.W.data);
    storage.b.set(this.b.data);
    storage.dw.set(this.dW.data);
    storage.db.set(this.db.data);
    storage.gamma.set(this.gamma.data);
    storage.dgamma.set(this.dGamma.data);
    storage.runningMean.set(this.runningMean);
    storage.runningVar.set(this.runningVar);

    this.W = { rows: this.inputs, cols: this.units, data: storage.w };
    this.b = { rows: 1, cols: this.units, data: storage.b };
    this.dW = { rows: this.inputs, cols: this.units, data: storage.dw };
    this.db = { rows: 1, cols: this.units, data: storage.db };
    this.gamma = { rows: 1, cols: gammaSize, data: storage.gamma };
    this.dGamma = { rows: 1, cols: gammaSize, data: storage.dgamma };
    this.runningMean = storage.runningMean;
    this.runningVar = storage.runningVar;
  }
}
