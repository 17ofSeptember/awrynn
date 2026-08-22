/*
 * 1D regression datasets.
 *
 * Spec §5: noisy sine, noisy cubic, step function. The step function is here
 * specifically because ReLU fits it beautifully — the piecewise-linear
 * approximation is visible as actual straight segments in the prediction curve,
 * which makes "what does a ReLU network actually compute" concrete.
 */

import type { Rng } from '../rng';
import { createMatrix } from '../tensor';
import type { Dataset, BaseDatasetOptions as DatasetOptions } from './types';

export type Regression1dName = 'sine' | 'cubic' | 'step';

export const REGRESSION_1D: readonly Regression1dName[] = ['sine', 'cubic', 'step'];

const DEFAULT_SAMPLES = 120;
const DEFAULT_NOISE = 0.1;

function target(name: Regression1dName, x: number): number {
  switch (name) {
    case 'sine':
      return Math.sin(x * Math.PI);
    case 'cubic':
      return 0.5 * x * x * x - 0.6 * x;
    case 'step':
      return x < -0.5 ? -0.8 : x < 0.3 ? 0.2 : 0.9;
  }
}

export function generateRegression1d(
  name: Regression1dName,
  options: DatasetOptions,
  rng: Rng,
): Dataset {
  const samples = options.samples ?? DEFAULT_SAMPLES;
  const noise = options.noise ?? DEFAULT_NOISE;
  if (samples <= 0) {
    throw new Error(`generateRegression1d: samples must be positive, got ${samples}.`);
  }

  const x = createMatrix(samples, 1);
  const y = createMatrix(samples, 1);
  for (let i = 0; i < samples; i++) {
    // Jittered uniform coverage rather than pure uniform: pure random sampling
    // leaves visible gaps at this sample count, and a gap in the data reads on
    // screen as the network being wrong rather than uninformed.
    const t = (i + rng.uniform(0, 1)) / samples;
    const xi = -2 + 4 * t;
    x.data[i] = xi;
    y.data[i] = target(name, xi) + rng.normal(0, noise);
  }

  return {
    name,
    kind: 'regression1d',
    x,
    y,
    labels: null,
    featureCount: 1,
    classCount: 0,
    suggestedLoss: 'mse',
    featureNames: ['x'],
    classNames: [],
  };
}

/** The noiseless target, for drawing the true curve behind the fit. */
export function regressionTarget(name: Regression1dName, x: number): number {
  return target(name, x);
}
