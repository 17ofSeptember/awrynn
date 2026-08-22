/*
 * The configuration types, in their own module.
 *
 * These are the shapes that describe a network before it exists: what the store
 * holds, what a lesson preset supplies, and what a share link carries. They sit
 * below the store rather than inside it so that a module the store imports can
 * still name them.
 *
 * This is the same fix applied to `datasets/types.ts`: shareLink.ts needs
 * `Architecture` and `toNetworkConfig`, the store needs shareLink's decoder,
 * and with both in store.ts the two would import each other. A cycle that
 * happens to work today because of module evaluation order is a bug waiting
 * for a different entry point.
 */

import type { InitScheme } from '../engine/init';
import type { LayerSpec } from '../engine/layers';
import type { LossName } from '../engine/losses';
import type { NetworkConfig } from '../engine/network';
import type { OptimizerConfig } from '../engine/optimizers';

export interface Architecture {
  readonly inputSize: number;
  readonly layers: readonly LayerSpec[];
  readonly loss: LossName;
  readonly seed: number;
  readonly init: InitScheme;
  readonly l2: number;
}

export interface TrainingSettings {
  readonly optimizer: OptimizerConfig;
  readonly learningRate: number;
  readonly batchSize: number;
  readonly maxEpochs: number;
  readonly dropout: number;
  readonly gradientClip: number;
  readonly standardize: boolean;
}

export const DEFAULT_TRAINING: TrainingSettings = {
  optimizer: { name: 'adam' },
  learningRate: 0.03,
  batchSize: 16,
  maxEpochs: 500,
  dropout: 0,
  gradientClip: 0,
  standardize: false,
};

export function toNetworkConfig(architecture: Architecture): NetworkConfig {
  return {
    inputSize: architecture.inputSize,
    layers: architecture.layers,
    loss: architecture.loss,
    seed: architecture.seed,
    init: architecture.init,
    l2: architecture.l2,
  };
}
