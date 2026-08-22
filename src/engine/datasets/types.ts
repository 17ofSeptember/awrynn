/*
 * Shared dataset types and helpers.
 *
 * These live in their own module to break a cycle: the generators need the
 * Dataset shape and oneHot, while the registry in index.ts needs the
 * generators. With both in index.ts the two imported each other, which happened
 * to work only because of the order modules were first evaluated — importing
 * the engine from a different entry point was enough to produce
 * "CLASSIFICATION_2D is not iterable" at load time.
 */

import type { LossName } from '../losses';
import type { Matrix } from '../tensor';
import { createMatrix } from '../tensor';

export type DatasetKind = 'classification2d' | 'regression1d' | 'glyphs';

/**
 * Generator-facing options, with `name` left as a plain string.
 *
 * The registry re-exports a narrowed `DatasetOptions` whose name is the union
 * of known datasets. The loose form exists so a generator module never has to
 * import the registry, which is what created the import cycle.
 */
export interface BaseDatasetOptions {
  readonly name: string;
  /** Total samples generated, before the train/validation split. */
  readonly samples?: number | undefined;
  /** Noise level; meaning is per-dataset but always "0 = clean". */
  readonly noise?: number | undefined;
  readonly seed?: number | undefined;
  /** Fraction held out for validation, in [0, 1). */
  readonly validationFraction?: number | undefined;
  /** Classes, where the dataset supports a variable count (spiral, glyphs). */
  readonly classes?: number | undefined;
  /**
   * Per-feature multipliers applied after generation.
   *
   * Exists for the feature-scaling lesson (§7.8), which needs "the same data
   * with one feature multiplied by 100". Applied as a post-step rather than
   * baked into each generator, so it composes with every dataset and does not
   * change what any of them mean.
   */
  readonly featureScale?: readonly number[] | undefined;
}

export interface Dataset {
  readonly name: string;
  readonly kind: DatasetKind;
  /** Inputs, [N, features]. */
  readonly x: Matrix;
  /** Targets: one-hot [N, K] for multi-class, [N, 1] otherwise. */
  readonly y: Matrix;
  /** Integer class label per sample, or null for regression. */
  readonly labels: Int32Array | null;
  readonly featureCount: number;
  readonly classCount: number;
  /** The loss this dataset is built for. */
  readonly suggestedLoss: LossName;
  /** Human-readable axis names, for the inspector. */
  readonly featureNames: readonly string[];
  readonly classNames: readonly string[];
}

/** One-hot encode integer labels into [N, K]. */
export function oneHot(labels: Int32Array, classes: number): Matrix {
  const y = createMatrix(labels.length, classes);
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]!;
    if (label < 0 || label >= classes) {
      throw new Error(`oneHot: label ${label} is outside [0, ${classes}).`);
    }
    y.data[i * classes + label] = 1;
  }
  return y;
}
