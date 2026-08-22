/*
 * Network — forward, backward, serialize, clone.
 *
 * Spec §4.2, §4.3, §4.5. This module owns three things the layers cannot decide
 * for themselves: configuration validation, fused output-gradient detection,
 * and the L2 objective.
 */

import type { ActivationName } from './activations';
import { hasKink } from './activations';
import type { InitScheme } from './init';
import type { LayerSpec } from './layers';
import { DenseLayer } from './layers';
import type { Loss, LossName } from './losses';
import { fusedOutputGradient, getLoss } from './losses';
import type { RngSource } from './rng';
import { createRng } from './rng';
import type { Matrix } from './tensor';
import { cloneMatrix, ensureShape, fromArray, sumSquares } from './tensor';

export interface NetworkConfig {
  readonly inputSize: number;
  readonly layers: readonly LayerSpec[];
  readonly loss: LossName;
  readonly seed: number;
  readonly init: InitScheme;
  /** L2 coefficient λ. 0 disables. Applied to weights only, never biases (§4.9). */
  readonly l2?: number | undefined;
}

/** How the output-layer gradient is computed — surfaced so the UI can say which route ran. */
export type OutputGradientMode = 'fused-softmax-cce' | 'fused-sigmoid-bce' | 'general';

/**
 * A network's transferable state: what it learned, and what it measured.
 *
 * `buffers` is empty for a network with no batch normalization anywhere, which
 * is why almost nothing in the app had to know this type existed until one did.
 */
export interface NetworkState {
  readonly parameters: Float64Array;
  readonly buffers: Float64Array;
}

/** The current serialization format. Version 1 predates batch normalization. */
export const SERIALIZATION_VERSION = 2;

export interface SerializedNetwork {
  readonly version: 1 | 2;
  readonly inputSize: number;
  readonly layers: readonly LayerSpec[];
  readonly loss: LossName;
  readonly seed: number;
  readonly init: InitScheme;
  readonly l2: number;
  readonly weights: readonly (readonly number[])[];
  readonly biases: readonly (readonly number[])[];
  /**
   * γ per layer, empty for a layer that does not normalize. Absent in v1 files,
   * which could not have had a normalizing layer to write.
   */
  readonly gammas?: readonly (readonly number[])[] | undefined;
  /** μ̂ and σ̂² per layer, likewise empty where the layer does not normalize. */
  readonly runningMeans?: readonly (readonly number[])[] | undefined;
  readonly runningVars?: readonly (readonly number[])[] | undefined;
}

/**
 * Validate a configuration, returning every problem found rather than the first.
 * Spec §4.4 requires softmax misuse to surface as a clear error in the UI, not
 * a silent fallback, so this is exported for the UI to call before constructing.
 */
export function validateConfig(config: NetworkConfig): string[] {
  const problems: string[] = [];

  if (!Number.isInteger(config.inputSize) || config.inputSize <= 0) {
    problems.push(`Input size must be a positive integer, got ${config.inputSize}.`);
  }
  if (config.layers.length === 0) {
    problems.push('A network needs at least one layer.');
  }

  config.layers.forEach((layer, i) => {
    if (!Number.isInteger(layer.units) || layer.units <= 0) {
      problems.push(`Layer ${i + 1} must have a positive integer number of units, got ${layer.units}.`);
    }
    // Spec §4.4: softmax is ONLY legal as the final layer's activation.
    if (layer.activation === 'softmax' && i !== config.layers.length - 1) {
      problems.push(
        `Layer ${i + 1} uses softmax, but softmax is only valid on the output layer. Use tanh, ReLU or sigmoid for hidden layers.`,
      );
    }
  });

  const output = config.layers[config.layers.length - 1];
  if (output !== undefined) {
    // Spec §4.4: softmax is ONLY legal with categorical cross-entropy.
    if (output.activation === 'softmax' && config.loss !== 'cce') {
      problems.push(
        `Softmax output requires the categorical cross-entropy loss, but this network uses ${config.loss}. Softmax produces a probability distribution; only cross-entropy scores one.`,
      );
    }
    if (config.loss === 'cce' && output.activation !== 'softmax') {
      problems.push(
        `Categorical cross-entropy requires a softmax output, but the output layer uses ${output.activation}. Cross-entropy assumes the outputs are probabilities that sum to 1.`,
      );
    }
    if (config.loss === 'bce' && output.units !== 1) {
      problems.push(
        `Binary cross-entropy expects a single output unit, but the output layer has ${output.units}. Use categorical cross-entropy with softmax for ${output.units} classes.`,
      );
    }
    if (config.loss === 'bce' && output.activation !== 'sigmoid') {
      problems.push(
        `Binary cross-entropy expects a sigmoid output, but the output layer uses ${output.activation}. Without sigmoid the prediction is not a probability and the loss is undefined outside [0, 1].`,
      );
    }
  }

  if (config.l2 !== undefined && (config.l2 < 0 || !Number.isFinite(config.l2))) {
    problems.push(`L2 coefficient must be a non-negative finite number, got ${config.l2}.`);
  }

  return problems;
}

export class Network {
  readonly inputSize: number;
  readonly lossName: LossName;
  readonly seed: number;
  readonly initScheme: InitScheme;
  readonly layers: readonly DenseLayer[];
  readonly rng: RngSource;
  readonly outputGradientMode: OutputGradientMode;

  /** L2 coefficient λ (§4.9). Mutable so the UI can scrub it live. */
  l2: number;

  /**
   * Every parameter in the network, contiguous, in layer order: W, then b, then
   * γ where a layer normalizes. Each layer's matrices are subarray VIEWS into
   * this, so writing here writes the network and vice versa — there is no copy
   * to keep in sync.
   *
   * This is what §4.8's "flat parameter/gradient arrays" refers to. It also
   * makes gradient clipping by global norm (§4.9) a single pass and a history
   * snapshot (§6.6) a memcpy.
   */
  readonly params: Float64Array;
  /** Gradients, laid out identically to `params`. */
  readonly grads: Float64Array;
  /**
   * What the forward pass MEASURES rather than what gradient descent learns:
   * μ̂ and σ̂² per normalizing layer, in the same layer order. Empty for a
   * network with no batch normalization anywhere.
   *
   * Kept apart from `params` so no optimizer, weight decay or gradient clip can
   * reach it, and so the parameter count stays a count of parameters. It still
   * has to travel with them: restoring weights without their statistics gives a
   * network whose eval-mode predictions, and so whose decision boundary, are
   * wrong in a way nothing on screen would explain.
   */
  readonly buffers: Float64Array;

  private readonly lossFn: Loss;
  private dZOutCache: Matrix | null = null;

  constructor(config: NetworkConfig) {
    const problems = validateConfig(config);
    if (problems.length > 0) {
      throw new Error(`Invalid network configuration:\n  - ${problems.join('\n  - ')}`);
    }

    this.inputSize = config.inputSize;
    this.lossName = config.loss;
    this.seed = config.seed;
    this.initScheme = config.init;
    this.l2 = config.l2 ?? 0;
    this.lossFn = getLoss(config.loss);
    this.rng = createRng(config.seed);

    const initStream = this.rng.stream('init');
    const built: DenseLayer[] = [];
    let fanIn = config.inputSize;
    for (const spec of config.layers) {
      built.push(new DenseLayer(fanIn, spec, config.init, initStream));
      fanIn = spec.units;
    }
    this.layers = built;

    /*
     * One contiguous allocation for the whole network, then rebind every layer
     * onto slices of it. Order matches parameterHandles() in gradcheck.ts.
     *
     * Per layer the parameters run [W | b | γ], with γ empty unless that layer
     * normalizes. Appending γ rather than interleaving it means a network with
     * no batch normalization has byte-for-byte the layout it had before batch
     * normalization existed, so every share link and save file made until now
     * still restores exactly.
     *
     * `buffers` is the parallel array for what is MEASURED rather than learned:
     * [μ̂ | σ̂²] per normalizing layer, and empty otherwise. It is separate
     * because an optimizer must never touch it, because AdamW would otherwise
     * decay a running mean toward zero, and because the parameter count on
     * screen should keep meaning what a reader thinks it means.
     */
    const total = built.reduce((sum, l) => sum + l.parameterCount, 0);
    const bufferTotal = built.reduce((sum, l) => sum + l.bufferCount, 0);
    this.params = new Float64Array(total);
    this.grads = new Float64Array(total);
    this.buffers = new Float64Array(bufferTotal);
    let offset = 0;
    let bufferOffset = 0;
    for (const layer of built) {
      const weightCount = layer.inputs * layer.units;
      const biasCount = layer.units;
      const gammaCount = layer.batchNorm ? layer.units : 0;
      const wEnd = offset + weightCount;
      const bEnd = wEnd + biasCount;
      const gEnd = bEnd + gammaCount;
      const meanEnd = bufferOffset + gammaCount;
      layer.bindStorage({
        w: this.params.subarray(offset, wEnd),
        dw: this.grads.subarray(offset, wEnd),
        b: this.params.subarray(wEnd, bEnd),
        db: this.grads.subarray(wEnd, bEnd),
        gamma: this.params.subarray(bEnd, gEnd),
        dgamma: this.grads.subarray(bEnd, gEnd),
        runningMean: this.buffers.subarray(bufferOffset, meanEnd),
        runningVar: this.buffers.subarray(meanEnd, meanEnd + gammaCount),
      });
      offset = gEnd;
      bufferOffset = meanEnd + gammaCount;
    }

    const output = built[built.length - 1] as DenseLayer;
    this.outputGradientMode = resolveOutputMode(output.activationName, config.loss);
  }

  get outputSize(): number {
    return (this.layers[this.layers.length - 1] as DenseLayer).units;
  }

  get parameterCount(): number {
    return this.layers.reduce((sum, l) => sum + l.parameterCount, 0);
  }

  /**
   * A^0 = X, then Z^l / A^l layer by layer. Returns Ŷ = A^L.
   *
   * `training` retains A^{l-1} for backprop AND enables dropout. Eval-mode
   * forward is a plain pass with no masks, which is what inference, the
   * decision boundary and every metric use.
   */
  forward(x: Matrix, training = false): Matrix {
    if (x.cols !== this.inputSize) {
      throw new Error(
        `Network.forward: input [${x.rows}, ${x.cols}] does not match the network's input size ${this.inputSize}.`,
      );
    }
    // Masks draw from the named 'dropout' stream so that turning dropout on
    // does not shift init or shuffling (§4.7).
    const dropoutRng = training ? this.rng.stream('dropout') : null;
    let a = x;
    for (const layer of this.layers) a = layer.forward(a, training, dropoutRng);
    return a;
  }

  /**
   * Run `body` with the network safe to LOOK AT: no dropout, no running-statistic
   * updates. Both settings are restored afterwards, whatever `body` does.
   *
   * Displaying a network must not change it, and both hazards are easy to walk
   * into because both require training=true to reach. The dissection view and
   * the live-math tab need a training-mode forward for one reason only, that it
   * retains A^{l-1} for the backward pass, and they get two side effects they
   * never asked for: a dropout mask that makes the arithmetic on screen
   * unreproducible, and a nudge to the running statistics that quietly moves the
   * decision boundary every time a panel re-renders.
   */
  inspect<T>(body: () => T): T {
    const savedDropout = this.layers.map((l) => l.dropout);
    const savedFreeze = this.layers.map((l) => l.freezeStatistics);
    for (const layer of this.layers) {
      layer.dropout = 0;
      layer.freezeStatistics = true;
    }
    try {
      return body();
    } finally {
      this.layers.forEach((layer, i) => {
        layer.dropout = savedDropout[i] as number;
        layer.freezeStatistics = savedFreeze[i] as boolean;
      });
    }
  }

  /** True when any layer would apply a dropout mask during a training forward. */
  hasDropout(): boolean {
    return this.layers.some((l) => l.dropout > 0);
  }

  /**
   * Set dropout on the hidden layers only.
   *
   * Never on the output: dropping outputs corrupts the loss itself rather than
   * regularising a representation, and with a fused softmax/CCE output it would
   * break the Ŷ − Y identity outright.
   */
  setHiddenDropout(p: number): void {
    if (!(p >= 0) || p >= 1) {
      throw new Error(
        `Network.setHiddenDropout: p must be in [0, 1), got ${p}. At p = 1 every unit is dropped and the 1/(1−p) rescaling divides by zero.`,
      );
    }
    this.layers.forEach((layer, i) => {
      layer.dropout = i < this.layers.length - 1 ? p : 0;
    });
  }

  /** Data loss only — the number labelled "loss" in the UI (§4.5). */
  dataLoss(yHat: Matrix, y: Matrix): number {
    return this.lossFn.loss(yHat, y);
  }

  /** (λ/2)·Σ‖W‖² across every layer. Biases excluded (§4.9). */
  l2Penalty(): number {
    if (this.l2 === 0) return 0;
    let total = 0;
    for (const layer of this.layers) total += sumSquares(layer.W);
    return (this.l2 / 2) * total;
  }

  /**
   * The quantity actually being minimised: data loss + L2 term.
   *
   * Spec §4.5 insists these be labelled distinctly, because a learner watching
   * "loss" plateau above zero while L2 is on will otherwise think training has
   * stalled. gradcheck differentiates THIS, not dataLoss (§4.11).
   */
  objective(yHat: Matrix, y: Matrix): number {
    return this.dataLoss(yHat, y) + this.l2Penalty();
  }

  /**
   * Backward pass. Requires a preceding forward(x, true).
   *
   * Populates every layer's dW and db, already divided by B and already
   * including the L2 term when λ > 0.
   */
  backward(y: Matrix): void {
    const output = this.layers[this.layers.length - 1] as DenseLayer;
    const yHat = output.A;
    if (yHat === null) {
      throw new Error('Network.backward: call forward(x, true) before backward(y).');
    }
    if (y.rows !== yHat.rows || y.cols !== yHat.cols) {
      throw new Error(
        `Network.backward: target [${y.rows}, ${y.cols}] does not match prediction [${yHat.rows}, ${yHat.cols}].`,
      );
    }

    let downstream: Matrix;
    if (this.outputGradientMode === 'general') {
      // dA^L = ∂ℓ/∂A^L, unaveraged per-sample (§4.3).
      downstream = output.backwardFromDA(this.lossFn.dA(yHat, y));
    } else {
      // Fused: dZ^L = Ŷ − Y. See the derivation above fusedOutputGradient().
      const dZ = ensureShape(this.dZOutCache, yHat.rows, yHat.cols);
      this.dZOutCache = dZ;
      fusedOutputGradient(yHat, y, dZ);
      downstream = output.backwardFromDZ(dZ);
    }

    for (let i = this.layers.length - 2; i >= 0; i--) {
      downstream = (this.layers[i] as DenseLayer).backwardFromDA(downstream);
    }

    if (this.l2 !== 0) {
      // After the batch average, per §4.9.
      for (const layer of this.layers) layer.applyL2(this.l2);
    }
  }

  /** True when any layer's activation is non-differentiable at zero (§4.11). */
  hasKinkedActivation(): boolean {
    return this.layers.some((l) => hasKink(l.activationName));
  }

  clearCaches(): void {
    for (const layer of this.layers) layer.clearCaches();
    this.dZOutCache = null;
  }

  config(): NetworkConfig {
    return {
      inputSize: this.inputSize,
      layers: this.layers.map((l) => l.spec()),
      loss: this.lossName,
      seed: this.seed,
      init: this.initScheme,
      l2: this.l2,
    };
  }

  serialize(): SerializedNetwork {
    return {
      version: SERIALIZATION_VERSION,
      inputSize: this.inputSize,
      layers: this.layers.map((l) => l.spec()),
      loss: this.lossName,
      seed: this.seed,
      init: this.initScheme,
      l2: this.l2,
      weights: this.layers.map((l) => Array.from(l.W.data)),
      biases: this.layers.map((l) => Array.from(l.b.data)),
      gammas: this.layers.map((l) => Array.from(l.gamma.data)),
      runningMeans: this.layers.map((l) => Array.from(l.runningMean)),
      runningVars: this.layers.map((l) => Array.from(l.runningVar)),
    };
  }

  static deserialize(data: SerializedNetwork): Network {
    if (data.version !== 1 && data.version !== 2) {
      throw new Error(`Network.deserialize: unsupported format version ${String(data.version)}.`);
    }
    const net = new Network({
      inputSize: data.inputSize,
      layers: data.layers,
      loss: data.loss,
      seed: data.seed,
      init: data.init,
      l2: data.l2,
    });
    if (data.weights.length !== net.layers.length || data.biases.length !== net.layers.length) {
      throw new Error(
        `Network.deserialize: expected ${net.layers.length} weight and bias arrays, got ${data.weights.length} and ${data.biases.length}.`,
      );
    }
    net.layers.forEach((layer, i) => {
      const w = data.weights[i] as readonly number[];
      const b = data.biases[i] as readonly number[];
      layer.setWeights(
        fromArray(layer.inputs, layer.units, w),
        fromArray(1, layer.units, b),
      );
      if (!layer.batchNorm) return;
      // A layer that normalizes needs γ and its statistics or it is not the
      // network that was saved: without them its eval-mode output is wrong,
      // and nothing on screen would say why.
      const gamma = data.gammas?.[i];
      const mean = data.runningMeans?.[i];
      const variance = data.runningVars?.[i];
      if (gamma === undefined || mean === undefined || variance === undefined) {
        throw new Error(
          `Network.deserialize: layer ${i} normalizes across the batch, but the file carries no γ or running statistics for it.`,
        );
      }
      layer.setNormalization(gamma, mean, variance);
    });
    return net;
  }

  /** Independent copy with identical parameters. Caches are not carried over. */
  clone(): Network {
    const copy = new Network(this.config());
    copy.layers.forEach((layer, i) => layer.copyParametersFrom(this.layers[i] as DenseLayer));
    return copy;
  }

  /** Reset to the initial parameters for this seed (§6.3 transport: reset-to-init). */
  resetToInit(): void {
    this.rng.reset();
    const initStream = this.rng.stream('init');
    let fanIn = this.inputSize;
    for (const layer of this.layers) {
      const fresh = new DenseLayer(fanIn, layer.spec(), this.initScheme, initStream);
      layer.copyParametersFrom(fresh);
      fanIn = layer.units;
    }
    this.clearCaches();
  }

  /**
   * Everything needed to reproduce this network's behaviour, as one object.
   *
   * Prefer this over captureParameters wherever a network is being MOVED, which
   * is every worker message, history snapshot, share link and save file. The
   * pair travels together because the same weights with different statistics is
   * a different network, and a positional (parameters, buffers) argument list
   * is an invitation to pass one and forget the other.
   */
  captureState(): NetworkState {
    return { parameters: this.captureParameters(), buffers: this.captureBuffers() };
  }

  restoreState(state: NetworkState): void {
    this.restoreParameters(state.parameters);
    this.restoreBuffers(state.buffers);
  }

  /** Snapshot of the measured statistics, flattened in layer order. */
  captureBuffers(): Float64Array {
    return Float64Array.from(this.buffers);
  }

  /**
   * Restore measured statistics.
   *
   * Separate from restoreParameters because they arrive from different places:
   * an optimizer step writes parameters and never buffers, while a forward pass
   * writes buffers and never parameters. Anything that moves a network from one
   * place to another — a worker message, a history snapshot, a share link —
   * has to carry both or the network that arrives is not the one that left.
   */
  restoreBuffers(flat: Float64Array): void {
    if (flat.length !== this.buffers.length) {
      throw new Error(
        `Network.restoreBuffers: expected ${this.buffers.length} values, got ${flat.length}.`,
      );
    }
    this.buffers.set(flat);
  }

  /** Snapshot of all parameters, flattened in layer order: W, b, then γ. */
  captureParameters(): Float64Array {
    return Float64Array.from(this.params);
  }

  restoreParameters(flat: Float64Array): void {
    if (flat.length !== this.parameterCount) {
      throw new Error(
        `Network.restoreParameters: expected ${this.parameterCount} values, got ${flat.length}.`,
      );
    }
    this.params.set(flat);
  }

  /** Zero every gradient — the trainer calls this before accumulating a batch. */
  zeroGradients(): void {
    this.grads.fill(0);
  }

  /** ‖g‖₂ across every parameter, for gradient clipping (§4.9). */
  gradientNorm(): number {
    let sum = 0;
    for (let i = 0; i < this.grads.length; i++) {
      const v = this.grads[i]!;
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /** Composed weight product W¹W²…W^L — proves a linear stack is one linear map (§7.2). */
  composedLinearMap(): Matrix | null {
    if (this.layers.some((l) => l.activationName !== 'linear')) return null;
    let product = cloneMatrix((this.layers[0] as DenseLayer).W);
    for (let i = 1; i < this.layers.length; i++) {
      const next = (this.layers[i] as DenseLayer).W;
      const out = { rows: product.rows, cols: next.cols, data: new Float64Array(product.rows * next.cols) };
      for (let r = 0; r < product.rows; r++) {
        for (let c = 0; c < next.cols; c++) {
          let sum = 0;
          for (let k = 0; k < product.cols; k++) {
            sum += product.data[r * product.cols + k]! * next.data[k * next.cols + c]!;
          }
          out.data[r * next.cols + c] = sum;
        }
      }
      product = out;
    }
    return product;
  }
}

function resolveOutputMode(activation: ActivationName, loss: LossName): OutputGradientMode {
  if (activation === 'softmax' && loss === 'cce') return 'fused-softmax-cce';
  if (activation === 'sigmoid' && loss === 'bce') return 'fused-sigmoid-bce';
  return 'general';
}
