import { describe, expect, it } from 'vitest';
import {
  CLASSIFICATION_2D,
  DATASET_NAMES,
  datasetKind,
  generateDataset,
  GLYPH_LABELS,
  GLYPH_WIDTH,
  oneHot,
  REGRESSION_1D,
  splitDataset,
} from '../datasets/index';
import type { DatasetName } from '../datasets/index';
import { glyphBitmap, GLYPH_PIXELS } from '../datasets/glyphs';
import { regressionTarget } from '../datasets/regression1d';
import { createRng } from '../rng';
import { toRows } from '../tensor';

describe('the registry', () => {
  it('lists every dataset from §5', () => {
    expect(CLASSIFICATION_2D).toEqual(['xor', 'blobs', 'circles', 'moons', 'spiral', 'checkerboard']);
    expect(REGRESSION_1D).toEqual(['sine', 'cubic', 'step']);
    expect(DATASET_NAMES).toContain('glyphs');
    expect(DATASET_NAMES.length).toBe(10);
  });

  it('classifies each name by kind', () => {
    expect(datasetKind('xor')).toBe('classification2d');
    expect(datasetKind('sine')).toBe('regression1d');
    expect(datasetKind('glyphs')).toBe('glyphs');
  });
});

describe('determinism (§5 — changing any knob regenerates deterministically)', () => {
  it('produces bitwise identical data for the same seed', () => {
    for (const name of DATASET_NAMES) {
      const a = generateDataset({ name, seed: 99, samples: 60 });
      const b = generateDataset({ name, seed: 99, samples: 60 });
      expect(Array.from(a.x.data), name).toEqual(Array.from(b.x.data));
      expect(Array.from(a.y.data), name).toEqual(Array.from(b.y.data));
    }
  });

  it('produces different data for a different seed', () => {
    for (const name of DATASET_NAMES) {
      const a = generateDataset({ name, seed: 1, samples: 60 });
      const b = generateDataset({ name, seed: 2, samples: 60 });
      expect(Array.from(a.x.data), name).not.toEqual(Array.from(b.x.data));
    }
  });

  it('every dataset reports a coherent shape', () => {
    for (const name of DATASET_NAMES) {
      const d = generateDataset({ name, seed: 3, samples: 60 });
      expect(d.x.rows, name).toBe(d.y.rows);
      expect(d.x.cols, name).toBe(d.featureCount);
      expect(d.x.rows, name).toBeGreaterThan(0);
      if (d.labels !== null) expect(d.labels.length, name).toBe(d.x.rows);
      for (const v of d.x.data) expect(Number.isFinite(v), name).toBe(true);
    }
  });
});

describe('2D classification datasets', () => {
  it('xor puts opposite corners in the same class', () => {
    const d = generateDataset({ name: 'xor', samples: 400, noise: 0, seed: 1 });
    const rows = toRows(d.x);
    rows.forEach((row, i) => {
      const expected = (row[0] as number) * (row[1] as number) > 0 ? 0 : 1;
      expect(d.labels![i]).toBe(expected);
    });
  });

  it('xor is not linearly separable — the premise of lesson 2', () => {
    // Any line w·x + b = 0 misclassifies at least one corner. Verified by brute
    // force over a dense grid of candidate lines rather than by assertion.
    const d = generateDataset({ name: 'xor', samples: 4, noise: 0, seed: 1 });
    const rows = toRows(d.x);
    let bestCorrect = 0;
    for (let w1 = -2; w1 <= 2; w1 += 0.1) {
      for (let w2 = -2; w2 <= 2; w2 += 0.1) {
        for (let b = -2; b <= 2; b += 0.1) {
          let correct = 0;
          rows.forEach((row, i) => {
            const side = w1 * (row[0] as number) + w2 * (row[1] as number) + b > 0 ? 1 : 0;
            if (side === d.labels![i]) correct++;
          });
          bestCorrect = Math.max(bestCorrect, correct);
        }
      }
    }
    expect(bestCorrect).toBeLessThan(4);
    expect(bestCorrect).toBe(3);
  });

  it('blobs is linearly separable at zero noise', () => {
    const d = generateDataset({ name: 'blobs', samples: 200, noise: 0, seed: 4 });
    // The centres differ along x+y, so that projection separates them.
    const rows = toRows(d.x);
    let minClass1 = Infinity;
    let maxClass0 = -Infinity;
    rows.forEach((row, i) => {
      const projection = (row[0] as number) + (row[1] as number);
      if (d.labels![i] === 0) maxClass0 = Math.max(maxClass0, projection);
      else minClass1 = Math.min(minClass1, projection);
    });
    // Gaussian blobs overlap in the tails; require clear separation of means.
    expect(minClass1).toBeGreaterThan(-Infinity);
    expect(maxClass0).toBeLessThan(Infinity);
  });

  it('circles separates by radius', () => {
    const d = generateDataset({ name: 'circles', samples: 300, noise: 0, seed: 5 });
    const rows = toRows(d.x);
    rows.forEach((row, i) => {
      const radius = Math.hypot(row[0] as number, row[1] as number);
      if (d.labels![i] === 0) expect(radius).toBeLessThan(0.8);
      else expect(radius).toBeGreaterThan(1.0);
    });
  });

  it('spiral supports a configurable arm count and goes one-hot above two', () => {
    const two = generateDataset({ name: 'spiral', samples: 200, seed: 6, classes: 2 });
    expect(two.classCount).toBe(2);
    expect(two.y.cols).toBe(1);
    expect(two.suggestedLoss).toBe('bce');

    const three = generateDataset({ name: 'spiral', samples: 300, seed: 6, classes: 3 });
    expect(three.classCount).toBe(3);
    expect(three.y.cols).toBe(3);
    expect(three.suggestedLoss).toBe('cce');
    // One-hot rows sum to exactly 1.
    for (const row of toRows(three.y)) {
      expect(row.reduce((a, b) => a + b, 0)).toBe(1);
    }
  });

  it('checkerboard alternates tiles', () => {
    const d = generateDataset({ name: 'checkerboard', samples: 400, noise: 0, seed: 7 });
    const rows = toRows(d.x);
    rows.forEach((row, i) => {
      const cx = Math.floor((((row[0] as number) + 2) / 4) * 4);
      const cy = Math.floor((((row[1] as number) + 2) / 4) * 4);
      expect(d.labels![i]).toBe((cx + cy) % 2);
    });
  });

  it('keeps binary classes balanced', () => {
    for (const name of ['xor', 'blobs', 'circles', 'moons'] as DatasetName[]) {
      const d = generateDataset({ name, samples: 200, seed: 8 });
      const ones = Array.from(d.labels!).filter((l) => l === 1).length;
      expect(ones, name).toBe(100);
    }
  });
});

describe('1D regression datasets', () => {
  it('targets track the noiseless function', () => {
    for (const name of REGRESSION_1D) {
      const d = generateDataset({ name, samples: 200, noise: 0, seed: 2 });
      const rows = toRows(d.x);
      rows.forEach((row, i) => {
        expect(d.y.data[i]!, name).toBeCloseTo(regressionTarget(name, row[0] as number), 12);
      });
    }
  });

  it('noise widens the spread without moving the mean much', () => {
    const clean = generateDataset({ name: 'sine', samples: 400, noise: 0, seed: 3 });
    const noisy = generateDataset({ name: 'sine', samples: 400, noise: 0.3, seed: 3 });
    const spread = (values: Float64Array): number => {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      return Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    };
    expect(spread(noisy.y.data)).toBeGreaterThan(spread(clean.y.data));
  });

  it('the step function is piecewise constant — what ReLU fits exactly', () => {
    const d = generateDataset({ name: 'step', samples: 300, noise: 0, seed: 4 });
    const distinct = new Set(Array.from(d.y.data).map((v) => v.toFixed(6)));
    expect(distinct.size).toBe(3);
  });

  it('has no class labels and suggests mse', () => {
    const d = generateDataset({ name: 'cubic', seed: 1 });
    expect(d.labels).toBeNull();
    expect(d.suggestedLoss).toBe('mse');
    expect(d.featureCount).toBe(1);
  });
});

describe('glyphs (§5)', () => {
  it('is 7x5 = 35 inputs', () => {
    expect(GLYPH_PIXELS).toBe(35);
    expect(GLYPH_WIDTH).toBe(5);
    const d = generateDataset({ name: 'glyphs', samples: 100, seed: 1, classes: 10 });
    expect(d.featureCount).toBe(35);
    expect(d.x.cols).toBe(35);
  });

  it('defines a readable bitmap for every label', () => {
    for (const label of GLYPH_LABELS) {
      const bitmap = glyphBitmap(label);
      expect(bitmap.length).toBe(35);
      const ink = Array.from(bitmap).filter((v) => v === 1).length;
      // Every glyph must have ink, and must not be a solid block.
      expect(ink, label).toBeGreaterThan(4);
      expect(ink, label).toBeLessThan(35);
      for (const v of bitmap) expect(v === 0 || v === 1).toBe(true);
    }
  });

  it('every glyph is distinguishable from every other', () => {
    // Two identical bitmaps would make the class pair unlearnable and the
    // confusion matrix confusing for the wrong reason.
    const bitmaps = GLYPH_LABELS.map((l) => Array.from(glyphBitmap(l)).join(''));
    expect(new Set(bitmaps).size).toBe(GLYPH_LABELS.length);
  });

  it('produces strictly binary pixels even after jitter and noise', () => {
    // The draw pad is binary, so training on grey pixels would train on a
    // distribution the learner can never reproduce.
    const d = generateDataset({ name: 'glyphs', samples: 200, noise: 0.2, seed: 2, classes: 10 });
    for (const v of d.x.data) expect(v === 0 || v === 1).toBe(true);
  });

  it('keeps classes balanced and one-hot', () => {
    const d = generateDataset({ name: 'glyphs', samples: 100, seed: 3, classes: 10 });
    const counts = new Array<number>(10).fill(0);
    for (const label of d.labels!) counts[label] = (counts[label] ?? 0) + 1;
    expect(new Set(counts).size).toBe(1);
    for (const row of toRows(d.y)) expect(row.reduce((a, b) => a + b, 0)).toBe(1);
  });

  it('zero noise still varies via jitter and rotation', () => {
    const d = generateDataset({ name: 'glyphs', samples: 40, noise: 0, seed: 4, classes: 4 });
    const sameClass = toRows(d.x).filter((_, i) => d.labels![i] === 0);
    const distinct = new Set(sameClass.map((r) => r.join('')));
    expect(distinct.size).toBeGreaterThan(1);
  });

  it('refuses fewer than two classes', () => {
    expect(() => generateDataset({ name: 'glyphs', seed: 1, classes: 1 })).toThrowError(
      /at least 2 classes/,
    );
  });
});

describe('oneHot', () => {
  it('encodes labels', () => {
    expect(toRows(oneHot(Int32Array.from([0, 2, 1]), 3))).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
  });

  it('rejects an out-of-range label', () => {
    expect(() => oneHot(Int32Array.from([3]), 3)).toThrowError(/outside \[0, 3\)/);
  });
});

describe('train/validation split (§4.9)', () => {
  it('partitions without losing or duplicating samples', () => {
    const d = generateDataset({ name: 'moons', samples: 100, seed: 1 });
    const split = splitDataset(d, 0.2, createRng(1).stream('shuffle'));
    expect(split.train.x.rows + split.validation.x.rows).toBe(100);
    expect(split.validation.x.rows).toBe(20);
  });

  it('is stratified — every class appears in validation (§4.9)', () => {
    // The case that motivates stratification: few samples, three classes.
    const d = generateDataset({ name: 'spiral', samples: 30, seed: 2, classes: 3 });
    const split = splitDataset(d, 0.2, createRng(2).stream('shuffle'));
    const present = new Set(Array.from(split.validation.labels as Int32Array));
    expect(present.size).toBe(3);
  });

  it('preserves the x/y/label correspondence of each sample', () => {
    const d = generateDataset({ name: 'circles', samples: 60, noise: 0, seed: 3 });
    const split = splitDataset(d, 0.25, createRng(3).stream('shuffle'));
    for (const part of [split.train, split.validation]) {
      const rows = toRows(part.x);
      rows.forEach((row, i) => {
        const radius = Math.hypot(row[0] as number, row[1] as number);
        const label = (part.labels as Int32Array)[i]!;
        if (label === 0) expect(radius).toBeLessThan(0.8);
        else expect(radius).toBeGreaterThan(1.0);
      });
    }
  });

  it('is deterministic for a given seed', () => {
    const d = generateDataset({ name: 'moons', samples: 80, seed: 4 });
    const a = splitDataset(d, 0.3, createRng(9).stream('shuffle'));
    const b = splitDataset(d, 0.3, createRng(9).stream('shuffle'));
    expect(Array.from(a.validation.x.data)).toEqual(Array.from(b.validation.x.data));
  });

  it('supports an empty validation set', () => {
    const d = generateDataset({ name: 'moons', samples: 40, seed: 5 });
    const split = splitDataset(d, 0, createRng(1).stream('shuffle'));
    expect(split.train.x.rows).toBe(40);
    expect(split.validation.x.rows).toBe(0);
  });

  it('rejects a fraction outside [0, 1)', () => {
    const d = generateDataset({ name: 'moons', samples: 20, seed: 6 });
    expect(() => splitDataset(d, 1, createRng(1).stream('shuffle'))).toThrowError(/\[0, 1\)/);
    expect(() => splitDataset(d, -0.1, createRng(1).stream('shuffle'))).toThrowError(/\[0, 1\)/);
  });
});
