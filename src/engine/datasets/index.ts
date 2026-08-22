/*
 * Dataset registry and the shared dataset interface.
 *
 * Spec §5. Everything is generated procedurally in-code from the seeded RNG:
 * no fetches, no bundled data files. Changing any knob regenerates
 * deterministically, which is what makes a lesson's stored seed reproduce the
 * phenomenon it promises (§7).
 */

import type { Matrix } from '../tensor';
import { createMatrix } from '../tensor';
import { createRng } from '../rng';
import type { Rng } from '../rng';
import type { BaseDatasetOptions, Dataset, DatasetKind } from './types';
import { oneHot } from './types';
import { CLASSIFICATION_2D, generateClassification2d } from './classification2d';
import type { Classification2dName } from './classification2d';
import { generateRegression1d, REGRESSION_1D } from './regression1d';
import type { Regression1dName } from './regression1d';
import { generateGlyphs, GLYPH_LABELS, GLYPH_HEIGHT, GLYPH_WIDTH } from './glyphs';

export type DatasetName = Classification2dName | Regression1dName | 'glyphs';

/**
 * Options for a dataset the registry knows about.
 *
 * This is the type every consumer should use; the loose `BaseDatasetOptions`
 * exists only so the generator modules can avoid importing this file.
 */
export interface DatasetOptions extends BaseDatasetOptions {
  readonly name: DatasetName;
}

export type { Dataset, DatasetKind };
export { oneHot };

export interface DatasetSplit {
  readonly train: { readonly x: Matrix; readonly y: Matrix; readonly labels: Int32Array | null };
  readonly validation: {
    readonly x: Matrix;
    readonly y: Matrix;
    readonly labels: Int32Array | null;
  };
}

export const DATASET_NAMES: readonly DatasetName[] = [
  ...CLASSIFICATION_2D,
  ...REGRESSION_1D,
  'glyphs',
];

export function datasetKind(name: DatasetName): DatasetKind {
  if ((CLASSIFICATION_2D as readonly string[]).includes(name)) return 'classification2d';
  if ((REGRESSION_1D as readonly string[]).includes(name)) return 'regression1d';
  return 'glyphs';
}

export function generateDataset(options: DatasetOptions): Dataset {
  const seed = options.seed ?? 0;
  const rng = createRng(seed).stream('data');
  const kind = datasetKind(options.name);

  const dataset = ((): Dataset => {
    switch (kind) {
      case 'classification2d':
        return generateClassification2d(options.name as Classification2dName, options, rng);
      case 'regression1d':
        return generateRegression1d(options.name as Regression1dName, options, rng);
      case 'glyphs':
        return generateGlyphs(options, rng);
    }
  })();

  return applyFeatureScale(dataset, options.featureScale);
}

/**
 * Multiply each feature column by its scale.
 *
 * Mutates the freshly generated matrix rather than copying: it was created a
 * few lines ago and nothing else holds a reference.
 */
function applyFeatureScale(dataset: Dataset, scale: readonly number[] | undefined): Dataset {
  if (scale === undefined || scale.length === 0) return dataset;
  const { x } = dataset;
  for (let r = 0; r < x.rows; r++) {
    for (let c = 0; c < x.cols; c++) {
      const factor = scale[c];
      if (factor === undefined || factor === 1) continue;
      x.data[r * x.cols + c] = x.data[r * x.cols + c]! * factor;
    }
  }
  return dataset;
}

/**
 * Split into train and validation.
 *
 * Stratified for classification (§4.9): each class is shuffled and split
 * independently, so a 20-sample 3-class problem cannot end up with a validation
 * set missing a class entirely — which would make validation accuracy
 * meaningless in exactly the small-data regime lesson 7 uses.
 */
export function splitDataset(
  dataset: Dataset,
  validationFraction: number,
  rng: Rng,
): DatasetSplit {
  if (validationFraction < 0 || validationFraction >= 1) {
    throw new Error(
      `splitDataset: validationFraction must be in [0, 1), got ${validationFraction}.`,
    );
  }
  const total = dataset.x.rows;
  const indices: number[] = [];

  if (dataset.labels !== null && dataset.classCount > 1) {
    const byClass = new Map<number, number[]>();
    for (let i = 0; i < total; i++) {
      const label = dataset.labels[i]!;
      const bucket = byClass.get(label);
      if (bucket === undefined) byClass.set(label, [i]);
      else bucket.push(i);
    }
    // Sorted so the split does not depend on Map insertion order.
    const classes = [...byClass.keys()].sort((a, b) => a - b);
    const trainIndices: number[] = [];
    const valIndices: number[] = [];
    for (const cls of classes) {
      const bucket = byClass.get(cls) as number[];
      rng.shuffle(bucket);
      const valCount = Math.round(bucket.length * validationFraction);
      valIndices.push(...bucket.slice(0, valCount));
      trainIndices.push(...bucket.slice(valCount));
    }
    rng.shuffle(trainIndices);
    rng.shuffle(valIndices);
    indices.push(...trainIndices, ...valIndices);
    return materialise(dataset, indices, trainIndices.length);
  }

  for (let i = 0; i < total; i++) indices.push(i);
  rng.shuffle(indices);
  const valCount = Math.round(total * validationFraction);
  return materialise(dataset, indices, total - valCount);
}

function materialise(dataset: Dataset, order: readonly number[], trainCount: number): DatasetSplit {
  const take = (from: number, count: number): {
    x: Matrix;
    y: Matrix;
    labels: Int32Array | null;
  } => {
    const x = createMatrix(count, dataset.x.cols);
    const y = createMatrix(count, dataset.y.cols);
    const labels = dataset.labels === null ? null : new Int32Array(count);
    for (let i = 0; i < count; i++) {
      const src = order[from + i]!;
      x.data.set(dataset.x.data.subarray(src * dataset.x.cols, (src + 1) * dataset.x.cols), i * dataset.x.cols);
      y.data.set(dataset.y.data.subarray(src * dataset.y.cols, (src + 1) * dataset.y.cols), i * dataset.y.cols);
      if (labels !== null && dataset.labels !== null) labels[i] = dataset.labels[src]!;
    }
    return { x, y, labels };
  };

  return {
    train: take(0, trainCount),
    validation: take(trainCount, order.length - trainCount),
  };
}

export { GLYPH_HEIGHT, GLYPH_LABELS, GLYPH_WIDTH };
export type { Classification2dName, Regression1dName };
export { CLASSIFICATION_2D, REGRESSION_1D };
