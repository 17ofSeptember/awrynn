/*
 * Training loop: batching, epochs, early stopping, metrics.
 *
 * Spec §4.9, §4.10. The trainer owns everything that turns a single
 * forward/backward pass into training: how the data is split and shuffled, when
 * the optimizer steps, what the learning rate is, and what gets measured.
 *
 * It stays pure TypeScript with no timers and no callbacks into the DOM —
 * `runEpoch()` does exactly one epoch and returns, so the caller decides
 * whether that happens in a worker loop, on a requestAnimationFrame, or one
 * epoch per click of a transport button (§6.3).
 */

import { hasKink } from './activations';
import type { Dataset, DatasetSplit } from './datasets/index';
import { splitDataset } from './datasets/index';
import { layerGradientNorms } from './gradcheck';
import { Network } from './network';
import type { NetworkConfig } from './network';
import { clipGradientsByNorm, createOptimizer } from './optimizers';
import type { Optimizer, OptimizerConfig } from './optimizers';
import { applyStandardizer, fitStandardizer } from './regularizers';
import type { Standardizer } from './regularizers';
import { learningRateAt } from './schedules';
import type { ScheduleConfig } from './schedules';
import type { Matrix } from './tensor';
import { createMatrix, isFinite as matrixIsFinite, rowView } from './tensor';

export interface TrainerConfig {
  readonly network: NetworkConfig;
  readonly optimizer: OptimizerConfig;
  readonly learningRate: number;
  readonly schedule?: ScheduleConfig | undefined;
  readonly batchSize: number;
  /** Fraction held out for validation, in [0, 1). Defaults to 0.2. */
  readonly validationFraction?: number | undefined;
  /** Dropout probability per hidden layer. 0 disables. */
  readonly dropout?: number | undefined;
  /** Clip by global L2 norm when > 0 (§4.9). */
  readonly gradientClip?: number | undefined;
  readonly standardize?: boolean | undefined;
  /**
   * Deliberately fit standardization statistics on ALL data instead of the
   * training split — the leak lesson 8 exposes (§7.8). Off by default, and
   * labelled wherever it is surfaced.
   */
  readonly leakStandardization?: boolean | undefined;
  readonly earlyStopping?: EarlyStoppingConfig | undefined;
}

export interface EarlyStoppingConfig {
  /** Epochs to wait for a validation-loss improvement before stopping. */
  readonly patience: number;
  /** Minimum change that counts as an improvement. */
  readonly minDelta?: number | undefined;
  /** Restore the best-scoring weights when stopping (§4.9). */
  readonly restoreBest?: boolean | undefined;
}

export interface EpochMetrics {
  readonly epoch: number;
  readonly learningRate: number;
  /** Data loss on the training split. */
  readonly trainLoss: number;
  /** Data loss + L2 term — the quantity actually minimised (§4.5). */
  readonly trainObjective: number;
  readonly trainAccuracy: number | null;
  readonly validationLoss: number | null;
  readonly validationAccuracy: number | null;
  /** Per-layer ‖dW, db‖₂ from the last batch — the vanishing-gradient chart. */
  readonly gradientNorms: readonly number[];
  /** Global gradient norm before clipping. */
  readonly gradientNorm: number;
  /** Units with zero activation across the whole epoch (§4.10). */
  readonly deadUnits: number;
  /** Fraction of saturated units in bounded activations (§4.10). */
  readonly saturation: number;
  readonly diverged: boolean;
}

export interface ClassificationReport {
  readonly confusion: readonly (readonly number[])[];
  readonly precision: readonly number[];
  readonly recall: readonly number[];
  readonly accuracy: number;
}

export type StopReason = 'running' | 'early-stopping' | 'diverged';

export const DEFAULT_VALIDATION_FRACTION = 0.2;

/** Distance from a bounded activation's rail that counts as saturated (§4.10). */
export const SATURATION_MARGIN = 0.01;

export class Trainer {
  readonly network: Network;
  readonly optimizer: Optimizer;
  readonly config: TrainerConfig;
  readonly split: DatasetSplit;
  readonly standardizer: Standardizer | null;

  private readonly order: Int32Array;
  private readonly batchX: Matrix;
  private readonly batchY: Matrix;

  private epochIndex = 0;
  private bestValidationLoss = Infinity;
  private bestParameters: Float64Array | null = null;
  /**
   * The running statistics that went with `bestParameters`.
   *
   * Restoring the best weights while leaving the current statistics in place
   * would produce a network that existed at no point during training, and it
   * would go unnoticed: the weights would be right, the shapes would be right,
   * and only the eval-mode predictions would be quietly wrong.
   */
  private bestBuffers: Float64Array | null = null;
  private epochsSinceImprovement = 0;
  private stopReason: StopReason = 'running';
  private stoppedAtEpoch: number | null = null;

  /** Activation accumulator for dead-unit detection across a full epoch. */
  private readonly activityCounts: Int32Array[];
  private saturatedCount = 0;
  /**
   * Denominator for the saturation fraction: only units that CAN saturate.
   * Counting every activation would dilute the figure by however many ReLU or
   * linear units happen to be in the network, so a fully-saturated tanh layer
   * behind a linear output would report well under 100%.
   */
  private saturableCount = 0;

  constructor(config: TrainerConfig, dataset: Dataset) {
    this.config = config;
    this.network = new Network(config.network);
    this.optimizer = createOptimizer(config.optimizer);

    // The split draws from the 'shuffle' stream so that changing the split
    // fraction does not perturb weight initialisation (§4.7).
    const splitRng = this.network.rng.stream('shuffle');
    this.split = splitDataset(dataset, config.validationFraction ?? DEFAULT_VALIDATION_FRACTION, splitRng);

    if (config.standardize === true) {
      // Spec §4.9: fit on the TRAINING split only. The leak option exists so
      // lesson 8 can show what fitting on everything does to validation.
      const source = config.leakStandardization === true ? dataset.x : this.split.train.x;
      this.standardizer = fitStandardizer(source);
      applyStandardizer(this.split.train.x, this.standardizer);
      if (this.split.validation.x.rows > 0) {
        applyStandardizer(this.split.validation.x, this.standardizer);
      }
    } else {
      this.standardizer = null;
    }

    // Dropout lives on the layers, which apply the same mask on both passes.
    this.network.setHiddenDropout(config.dropout ?? 0);

    const trainCount = this.split.train.x.rows;
    if (trainCount === 0) {
      throw new Error('Trainer: the training split is empty. Lower validationFraction.');
    }
    this.order = new Int32Array(trainCount);
    for (let i = 0; i < trainCount; i++) this.order[i] = i;

    // Allocated once at full size; short final batches are row views (§4.9).
    const capacity = Math.min(Math.max(1, config.batchSize), trainCount);
    this.batchX = createMatrix(capacity, this.split.train.x.cols);
    this.batchY = createMatrix(capacity, this.split.train.y.cols);

    this.activityCounts = this.network.layers.map((l) => new Int32Array(l.units));
  }

  get epoch(): number {
    return this.epochIndex;
  }

  get status(): StopReason {
    return this.stopReason;
  }

  get stoppedAt(): number | null {
    return this.stoppedAtEpoch;
  }

  get currentLearningRate(): number {
    return this.config.schedule === undefined
      ? this.config.learningRate
      : learningRateAt(this.config.schedule, this.config.learningRate, this.epochIndex);
  }

  /**
   * One epoch: shuffle, walk the batches, step the optimizer, then measure.
   *
   * Returns the metrics for the epoch just completed. Safe to call after a stop
   * — it returns the last metrics unchanged rather than training further.
   */
  runEpoch(): EpochMetrics {
    if (this.stopReason !== 'running') {
      return this.measure(this.currentLearningRate, 0, [], false);
    }

    const learningRate = this.currentLearningRate;
    const n = this.split.train.x.rows;

    // Spec §4.9: shuffle the training set each epoch, from the 'shuffle' stream.
    const shuffleRng = this.network.rng.stream('shuffle');
    for (let i = n - 1; i > 0; i--) {
      const j = shuffleRng.int(i + 1);
      const tmp = this.order[i]!;
      this.order[i] = this.order[j]!;
      this.order[j] = tmp;
    }

    for (const counts of this.activityCounts) counts.fill(0);
    this.saturatedCount = 0;
    this.saturableCount = 0;

    let lastNorms: number[] = [];
    let lastGlobalNorm = 0;
    let diverged = false;

    for (let start = 0; start < n; start += this.config.batchSize) {
      // Spec §4.9: the final partial batch is KEPT and averaged over its true
      // size. Dropping it would silently discard up to batchSize-1 samples
      // every epoch; padding it would weight some samples twice.
      const size = Math.min(this.config.batchSize, n - start);
      this.fillBatch(start, size);

      const batchX = rowView(this.batchX, size);
      const batchY = rowView(this.batchY, size);
      const yHat = this.network.forward(batchX, true);
      this.network.backward(batchY);
      this.accumulateActivationStats();

      if (!matrixIsFinite(yHat)) {
        diverged = true;
        break;
      }

      const clip = this.config.gradientClip ?? 0;
      lastGlobalNorm =
        clip > 0
          ? clipGradientsByNorm(this.network.grads, clip)
          : this.network.gradientNorm();
      if (!Number.isFinite(lastGlobalNorm)) {
        diverged = true;
        break;
      }

      lastNorms = layerGradientNorms(this.network);
      this.optimizer.step(this.network.params, this.network.grads, learningRate, this.frozenMask());
    }

    const metrics = this.measure(learningRate, lastGlobalNorm, lastNorms, diverged);
    this.epochIndex++;

    if (metrics.diverged) {
      this.stopReason = 'diverged';
      this.stoppedAtEpoch = metrics.epoch;
    } else {
      this.considerEarlyStopping(metrics);
    }
    return metrics;
  }

  /** Copy `size` shuffled rows into the reusable batch buffers. */
  private fillBatch(start: number, size: number): void {
    const trainX = this.split.train.x;
    const trainY = this.split.train.y;
    const xCols = trainX.cols;
    const yCols = trainY.cols;
    for (let i = 0; i < size; i++) {
      const src = this.order[start + i]!;
      for (let c = 0; c < xCols; c++) {
        this.batchX.data[i * xCols + c] = trainX.data[src * xCols + c]!;
      }
      for (let c = 0; c < yCols; c++) {
        this.batchY.data[i * yCols + c] = trainY.data[src * yCols + c]!;
      }
    }
  }

  /** Frozen parameters are skipped entirely by the optimizer (§6.5). */
  private frozenMask(): Uint8Array | null {
    let any = false;
    for (const layer of this.network.layers) {
      if (layer.frozen) {
        any = true;
        break;
      }
    }
    if (!any) return null;
    const mask = new Uint8Array(this.network.params.length);
    let offset = 0;
    for (const layer of this.network.layers) {
      const size = layer.parameterCount;
      if (layer.frozen) mask.fill(1, offset, offset + size);
      offset += size;
    }
    return mask;
  }

  private accumulateActivationStats(): void {
    this.network.layers.forEach((layer, li) => {
      const a = layer.A;
      if (a === null) return;
      const counts = this.activityCounts[li] as Int32Array;
      const bounded = layer.activation.range;
      for (let r = 0; r < a.rows; r++) {
        for (let c = 0; c < a.cols; c++) {
          const v = a.data[r * a.cols + c]!;
          if (v !== 0) counts[c] = counts[c]! + 1;
          if (bounded === null) continue;

          this.saturableCount++;
          /*
           * Spec §4.10: |tanh(z)| > 0.99, or sigmoid outside [0.01, 0.99].
           *
           * A FIXED 0.01 margin from each end, not a fraction of the range.
           * Scaling by the range would put tanh's threshold at 0.98 (span 2 ×
           * 0.01 = 0.02 from each rail), which is not what the spec says.
           */
          if (v < bounded[0] + SATURATION_MARGIN || v > bounded[1] - SATURATION_MARGIN) {
            this.saturatedCount++;
          }
        }
      }
    });
  }

  /** Units that produced zero activation across the entire epoch (§4.10). */
  deadUnitCount(): number {
    let dead = 0;
    this.network.layers.forEach((layer, li) => {
      // Only meaningful for activations that can actually output zero and stay
      // there. A tanh unit sitting at 0 is not dead, it is centred.
      if (!hasKink(layer.activationName)) return;
      const counts = this.activityCounts[li] as Int32Array;
      for (let u = 0; u < counts.length; u++) {
        if (counts[u] === 0) dead++;
      }
    });
    return dead;
  }

  private measure(
    learningRate: number,
    gradientNorm: number,
    gradientNorms: readonly number[],
    diverged: boolean,
  ): EpochMetrics {
    const train = this.split.train;
    const validation = this.split.validation;

    // Eval mode: dropout is identity, so this is a plain forward pass.
    const trainPred = this.network.forward(train.x, false);
    const trainLoss = this.network.dataLoss(trainPred, train.y);
    const trainObjective = trainLoss + this.network.l2Penalty();
    const finite = Number.isFinite(trainLoss) && matrixIsFinite(trainPred);

    let validationLoss: number | null = null;
    let validationAccuracy: number | null = null;
    if (validation.x.rows > 0) {
      const valPred = this.network.forward(validation.x, false);
      validationLoss = this.network.dataLoss(valPred, validation.y);
      validationAccuracy = accuracy(valPred, validation.labels);
    }

    return {
      epoch: this.epochIndex,
      learningRate,
      trainLoss,
      trainObjective,
      trainAccuracy: accuracy(trainPred, train.labels),
      validationLoss,
      validationAccuracy,
      gradientNorms,
      gradientNorm,
      deadUnits: this.deadUnitCount(),
      // 0 when nothing in the network can saturate (an all-ReLU stack), which
      // is the honest reading rather than a division by zero.
      saturation: this.saturableCount === 0 ? 0 : this.saturatedCount / this.saturableCount,
      diverged: diverged || !finite,
    };
  }

  private considerEarlyStopping(metrics: EpochMetrics): void {
    const config = this.config.earlyStopping;
    if (config === undefined || metrics.validationLoss === null) return;

    const minDelta = config.minDelta ?? 0;
    if (metrics.validationLoss < this.bestValidationLoss - minDelta) {
      this.bestValidationLoss = metrics.validationLoss;
      this.epochsSinceImprovement = 0;
      if (config.restoreBest !== false) {
        this.bestParameters = this.network.captureParameters();
        this.bestBuffers = this.network.captureBuffers();
      }
      return;
    }

    this.epochsSinceImprovement++;
    if (this.epochsSinceImprovement >= config.patience) {
      this.stopReason = 'early-stopping';
      this.stoppedAtEpoch = metrics.epoch;
      if (config.restoreBest !== false && this.bestParameters !== null) {
        this.network.restoreParameters(this.bestParameters);
        if (this.bestBuffers !== null) this.network.restoreBuffers(this.bestBuffers);
      }
    }
  }

  /** Run until `maxEpochs` or an early stop. Returns every epoch's metrics. */
  run(maxEpochs: number): EpochMetrics[] {
    const history: EpochMetrics[] = [];
    for (let i = 0; i < maxEpochs && this.stopReason === 'running'; i++) {
      history.push(this.runEpoch());
    }
    return history;
  }

  /** Confusion matrix, precision and recall on the validation split (§4.10). */
  classificationReport(): ClassificationReport | null {
    const validation = this.split.validation;
    if (validation.labels === null || validation.x.rows === 0) return null;
    const predictions = this.network.forward(validation.x, false);
    const classes = Math.max(predictions.cols, 2);
    return buildReport(predictions, validation.labels, classes);
  }
}

/** Predicted class for one row: argmax for K>1, threshold at 0.5 for K=1. */
export function predictedClass(predictions: Matrix, row: number): number {
  if (predictions.cols === 1) {
    return predictions.data[row]! >= 0.5 ? 1 : 0;
  }
  let best = 0;
  let bestValue = -Infinity;
  for (let c = 0; c < predictions.cols; c++) {
    const v = predictions.data[row * predictions.cols + c]!;
    if (v > bestValue) {
      bestValue = v;
      best = c;
    }
  }
  return best;
}

export function accuracy(predictions: Matrix, labels: Int32Array | null): number | null {
  if (labels === null) return null; // regression has no accuracy
  if (predictions.rows === 0) return null;
  let correct = 0;
  for (let r = 0; r < predictions.rows; r++) {
    if (predictedClass(predictions, r) === labels[r]!) correct++;
  }
  return correct / predictions.rows;
}

export function buildReport(
  predictions: Matrix,
  labels: Int32Array,
  classes: number,
): ClassificationReport {
  const confusion: number[][] = Array.from({ length: classes }, () =>
    new Array<number>(classes).fill(0),
  );
  let correct = 0;
  for (let r = 0; r < predictions.rows; r++) {
    const actual = labels[r]!;
    const predicted = predictedClass(predictions, r);
    (confusion[actual] as number[])[predicted] = ((confusion[actual] as number[])[predicted] ?? 0) + 1;
    if (actual === predicted) correct++;
  }

  const precision: number[] = [];
  const recall: number[] = [];
  for (let c = 0; c < classes; c++) {
    let predictedPositive = 0;
    let actualPositive = 0;
    for (let k = 0; k < classes; k++) {
      predictedPositive += (confusion[k] as number[])[c] ?? 0;
      actualPositive += (confusion[c] as number[])[k] ?? 0;
    }
    const truePositive = (confusion[c] as number[])[c] ?? 0;
    // A class the model never predicts has undefined precision; reporting 0
    // is the honest reading and keeps the confusion matrix arithmetic total.
    precision.push(predictedPositive === 0 ? 0 : truePositive / predictedPositive);
    recall.push(actualPositive === 0 ? 0 : truePositive / actualPositive);
  }

  return {
    confusion,
    precision,
    recall,
    accuracy: predictions.rows === 0 ? 0 : correct / predictions.rows,
  };
}
