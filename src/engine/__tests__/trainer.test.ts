import { describe, expect, it } from 'vitest';
import { buildReport, predictedClass, Trainer } from '../trainer';
import type { TrainerConfig } from '../trainer';
import { generateDataset } from '../datasets/index';
import { fromRows } from '../tensor';

/*
 * Spec §11 Phase 2 gate: "convergence smoke tests pass; training runs off the
 * main thread; XOR and moons solvable from a headless script."
 *
 * The convergence block below is that gate. Its thresholds come from §10 and
 * are not to be softened; the seeds are fixed so a pass means the same thing
 * every run.
 */

function xorConfig(overrides: Partial<TrainerConfig> = {}): TrainerConfig {
  return {
    network: {
      inputSize: 2,
      layers: [
        { units: 8, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 7,
      init: { kind: 'glorot_uniform' },
    },
    optimizer: { name: 'adam' },
    learningRate: 0.05,
    batchSize: 16,
    validationFraction: 0.2,
    ...overrides,
  };
}

describe('convergence smoke tests (§10) — the Phase 2 gate', () => {
  it('XOR reaches loss < 0.02 well within 2000 epochs', () => {
    const trainer = new Trainer(xorConfig(), generateDataset({ name: 'xor', samples: 200, noise: 0.1, seed: 1 }));
    let epochs = 0;
    let loss = Infinity;
    while (epochs < 2000 && loss >= 0.02) {
      loss = trainer.runEpoch().trainLoss;
      epochs++;
    }
    expect(loss).toBeLessThan(0.02);
    expect(epochs).toBeLessThan(2000);
    expect(trainer.status).toBe('running');
  });

  it('moons reaches validation accuracy > 0.95', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 12, activation: 'tanh' },
            { units: 1, activation: 'sigmoid' },
          ],
          loss: 'bce',
          seed: 3,
          init: { kind: 'glorot_uniform' },
        },
        optimizer: { name: 'adam' },
        learningRate: 0.03,
        batchSize: 16,
        validationFraction: 0.2,
      },
      generateDataset({ name: 'moons', samples: 300, noise: 0.12, seed: 2 }),
    );
    const history = trainer.run(300);
    const best = Math.max(...history.map((m) => m.validationAccuracy ?? 0));
    expect(best).toBeGreaterThan(0.95);
  });

  it('3-arm spiral reaches validation accuracy > 0.90 with adequate capacity', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 24, activation: 'tanh' },
            { units: 24, activation: 'tanh' },
            { units: 3, activation: 'softmax' },
          ],
          loss: 'cce',
          seed: 11,
          init: { kind: 'glorot_uniform' },
        },
        optimizer: { name: 'adam' },
        learningRate: 0.02,
        batchSize: 32,
        validationFraction: 0.2,
      },
      generateDataset({ name: 'spiral', samples: 600, noise: 0.06, seed: 4, classes: 3 }),
    );
    const history = trainer.run(600);
    const best = Math.max(...history.map((m) => m.validationAccuracy ?? 0));
    expect(best).toBeGreaterThan(0.9);
  });
});

describe('batching (§4.9)', () => {
  it('keeps the final partial batch and averages over its true size', () => {
    const batchSize = 10;
    const dataset = generateDataset({ name: 'moons', samples: 32, seed: 1 });
    const trainer = new Trainer(xorConfig({ batchSize, validationFraction: 0.25 }), dataset);

    // 24 training samples at batchSize 10 gives batches of 10, 10, 4. Dropping
    // the remainder would silently discard 4 samples every epoch; padding it
    // would weight some samples twice.
    const trainCount = trainer.split.train.x.rows;
    expect(trainCount).toBe(24);
    expect(trainCount % batchSize).not.toBe(0);

    expect(() => trainer.runEpoch()).not.toThrow();
    // Gradients stay finite, which a mis-sized final batch would break.
    expect(Number.isFinite(trainer.network.gradientNorm())).toBe(true);
  });

  it('a batch larger than the dataset is clamped to full-batch', () => {
    const dataset = generateDataset({ name: 'xor', samples: 20, seed: 1 });
    const trainer = new Trainer(xorConfig({ batchSize: 10_000 }), dataset);
    const metrics = trainer.runEpoch();
    expect(Number.isFinite(metrics.trainLoss)).toBe(true);
  });

  it('reshuffles every epoch from the shuffle stream', () => {
    // Two trainers with identical seeds must agree; that is only true if the
    // shuffle is seeded rather than incidental.
    const dataset = generateDataset({ name: 'moons', samples: 60, seed: 1 });
    const a = new Trainer(xorConfig(), dataset);
    const b = new Trainer(xorConfig(), generateDataset({ name: 'moons', samples: 60, seed: 1 }));
    a.run(5);
    b.run(5);
    expect(Array.from(a.network.captureParameters())).toEqual(
      Array.from(b.network.captureParameters()),
    );
  });
});

describe('early stopping (§4.9)', () => {
  it('stops after patience epochs without improvement and restores the best weights', () => {
    const trainer = new Trainer(
      xorConfig({
        // A high LR on noisy data makes validation loss wander, so patience
        // is genuinely exercised rather than never triggered.
        learningRate: 0.5,
        optimizer: { name: 'sgd' },
        earlyStopping: { patience: 3, restoreBest: true },
      }),
      generateDataset({ name: 'moons', samples: 80, noise: 0.4, seed: 5 }),
    );
    const history = trainer.run(400);
    expect(trainer.status).toBe('early-stopping');
    expect(trainer.stoppedAt).toBe(history.length - 1);

    // The restored weights must score at least as well as the final epoch did.
    const restoredLoss = trainer.network.dataLoss(
      trainer.network.forward(trainer.split.validation.x, false),
      trainer.split.validation.y,
    );
    const bestSeen = Math.min(
      ...history.map((m) => m.validationLoss ?? Infinity),
    );
    expect(restoredLoss).toBeCloseTo(bestSeen, 6);
  });

  it('does not stop while validation loss keeps improving', () => {
    const trainer = new Trainer(
      xorConfig({ earlyStopping: { patience: 5 } }),
      generateDataset({ name: 'xor', samples: 200, noise: 0.05, seed: 6 }),
    );
    trainer.run(20);
    expect(trainer.status).toBe('running');
  });

  it('running past a stop is a no-op rather than an error', () => {
    const trainer = new Trainer(
      xorConfig({
        learningRate: 0.5,
        optimizer: { name: 'sgd' },
        earlyStopping: { patience: 2 },
      }),
      generateDataset({ name: 'moons', samples: 60, noise: 0.4, seed: 7 }),
    );
    trainer.run(400);
    const stoppedAt = trainer.epoch;
    trainer.run(50);
    expect(trainer.epoch).toBe(stoppedAt);
  });
});

describe('divergence (§7.4) — recover gracefully, do not crash', () => {
  it('detects divergence and reports it as a state', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 8, activation: 'relu' },
            { units: 1, activation: 'sigmoid' },
          ],
          loss: 'bce',
          seed: 1,
          init: { kind: 'he_normal' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 1e6,
        batchSize: 16,
        validationFraction: 0.2,
      },
      generateDataset({ name: 'moons', samples: 200, seed: 2 }),
    );
    const history = trainer.run(50);
    expect(trainer.status).toBe('diverged');
    expect(history[history.length - 1]!.diverged).toBe(true);
    // Stopped early rather than burning all 50 epochs on garbage.
    expect(history.length).toBeLessThan(50);
  });

  it('catches an unbounded regression blow-up on the first epoch', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 1,
          layers: [
            { units: 8, activation: 'relu' },
            { units: 1, activation: 'linear' },
          ],
          loss: 'mse',
          seed: 1,
          init: { kind: 'he_normal' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 50,
        batchSize: 16,
        validationFraction: 0.2,
      },
      generateDataset({ name: 'sine', samples: 200, seed: 2 }),
    );
    trainer.run(20);
    expect(trainer.status).toBe('diverged');
    expect(trainer.stoppedAt).toBe(0);
  });

  it('tanh with a fused BCE output saturates instead of diverging, even at LR 1e6', () => {
    // Worth pinning: dZ = ŷ − y is bounded in [−1, 1] and tanh saturates, so
    // there is no path to NaN here however large the learning rate. Lesson 4
    // therefore needs ReLU or an unbounded output to show a divergence, and
    // this test is what stops that being rediscovered by accident.
    const trainer = new Trainer(
      xorConfig({ optimizer: { name: 'sgd' }, learningRate: 1e6 }),
      generateDataset({ name: 'moons', samples: 200, seed: 2 }),
    );
    const history = trainer.run(30);
    expect(trainer.status).toBe('running');
    expect(history.every((m) => Number.isFinite(m.trainLoss))).toBe(true);
  });

  it('a sane learning rate does not report divergence', () => {
    const trainer = new Trainer(xorConfig(), generateDataset({ name: 'xor', samples: 100, seed: 1 }));
    const history = trainer.run(20);
    expect(history.every((m) => !m.diverged)).toBe(true);
  });
});

describe('metrics (§4.10)', () => {
  it('separates data loss from the full objective when L2 is on', () => {
    const trainer = new Trainer(
      xorConfig({
        network: { ...xorConfig().network, l2: 0.1 },
      }),
      generateDataset({ name: 'xor', samples: 100, seed: 1 }),
    );
    const metrics = trainer.runEpoch();
    expect(metrics.trainObjective).toBeGreaterThan(metrics.trainLoss);
    expect(metrics.trainObjective - metrics.trainLoss).toBeCloseTo(
      trainer.network.l2Penalty(),
      12,
    );
  });

  it('counts dead ReLU units across the whole epoch (§7.6)', () => {
    // A large negative bias on every hidden unit kills all of them.
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 6, activation: 'relu' },
            { units: 1, activation: 'sigmoid' },
          ],
          loss: 'bce',
          seed: 1,
          init: { kind: 'glorot_uniform' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 0,
        batchSize: 16,
      },
      generateDataset({ name: 'moons', samples: 60, seed: 1 }),
    );
    trainer.network.layers[0]!.b.data.fill(-100);
    const metrics = trainer.runEpoch();
    expect(metrics.deadUnits).toBe(6);
  });

  it('reports zero dead units for a healthy ReLU layer', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 6, activation: 'relu' },
            { units: 1, activation: 'sigmoid' },
          ],
          loss: 'bce',
          seed: 2,
          init: { kind: 'he_normal' },
        },
        optimizer: { name: 'adam' },
        learningRate: 0.01,
        batchSize: 16,
      },
      generateDataset({ name: 'moons', samples: 120, seed: 1 }),
    );
    expect(trainer.runEpoch().deadUnits).toBeLessThan(6);
  });

  it('measures saturation against saturable units only, not every unit', () => {
    const trainer = new Trainer(
      xorConfig({ learningRate: 0 }),
      generateDataset({ name: 'xor', samples: 60, seed: 1 }),
    );
    // Saturate via the BIAS, not the weights: a large shared weight makes
    // z ∝ (x₁+x₂), which is ≈ 0 on XOR's anti-diagonal, so those units would
    // legitimately sit unsaturated and the test would measure the dataset
    // rather than the metric. A large bias saturates every unit for every input.
    for (const layer of trainer.network.layers) {
      layer.W.data.fill(0);
      layer.b.data.fill(10);
    }
    // tanh(10) ≈ 1 and sigmoid(10) ≈ 0.99995, both bounded, both saturated.
    expect(trainer.runEpoch().saturation).toBe(1);
  });

  it('excludes unbounded activations from the saturation denominator', () => {
    // tanh hidden behind a LINEAR output: the linear units can never saturate,
    // so counting them in the denominator would cap the reported figure at the
    // fraction of units that happen to be bounded.
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 8, activation: 'tanh' },
            { units: 1, activation: 'linear' },
          ],
          loss: 'mse',
          seed: 7,
          init: { kind: 'glorot_uniform' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 0,
        batchSize: 16,
      },
      generateDataset({ name: 'moons', samples: 60, seed: 1 }),
    );
    trainer.network.layers[0]!.W.data.fill(0);
    trainer.network.layers[0]!.b.data.fill(10);
    // Only the 8 tanh units count. Including the linear output in the
    // denominator would cap this at 8/9.
    expect(trainer.runEpoch().saturation).toBe(1);
  });

  it('reports zero saturation for a network that cannot saturate', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 6, activation: 'relu' },
            { units: 1, activation: 'linear' },
          ],
          loss: 'mse',
          seed: 2,
          init: { kind: 'he_normal' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 0.01,
        batchSize: 16,
      },
      generateDataset({ name: 'moons', samples: 60, seed: 1 }),
    );
    expect(trainer.runEpoch().saturation).toBe(0);
  });

  it('uses a fixed 0.01 margin, so tanh saturates at 0.99 and not 0.98', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 1,
          layers: [{ units: 1, activation: 'tanh' }],
          loss: 'mse',
          seed: 1,
          init: { kind: 'constant', value: 1 },
        },
        optimizer: { name: 'sgd' },
        learningRate: 0,
        batchSize: 4,
      },
      generateDataset({ name: 'sine', samples: 8, seed: 1 }),
    );
    // atanh(0.985) ≈ 2.4022 — inside a 0.98 threshold, outside a 0.99 one.
    // A span-scaled margin would call this saturated; the spec does not.
    const layer = trainer.network.layers[0]!;
    layer.W.data.fill(0);
    layer.b.data.fill(Math.atanh(0.985));
    expect(trainer.runEpoch().saturation).toBe(0);

    layer.b.data.fill(Math.atanh(0.995));
    expect(trainer.runEpoch().saturation).toBe(1);
  });

  it('reports per-layer gradient norms, which is what makes vanishing visible (§7.5)', () => {
    const trainer = new Trainer(
      {
        network: {
          inputSize: 2,
          layers: [
            { units: 6, activation: 'sigmoid' },
            { units: 6, activation: 'sigmoid' },
            { units: 6, activation: 'sigmoid' },
            { units: 6, activation: 'sigmoid' },
            { units: 1, activation: 'sigmoid' },
          ],
          loss: 'bce',
          seed: 3,
          init: { kind: 'glorot_uniform' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 0.1,
        batchSize: 32,
      },
      generateDataset({ name: 'moons', samples: 120, seed: 1 }),
    );
    const metrics = trainer.runEpoch();
    expect(metrics.gradientNorms.length).toBe(5);
    // The first layer receives dramatically less than the last.
    const first = metrics.gradientNorms[0] as number;
    const last = metrics.gradientNorms[4] as number;
    expect(first).toBeLessThan(last);
    expect(first).toBeGreaterThan(0);
  });

  it('tracks the learning rate through a schedule', () => {
    const trainer = new Trainer(
      xorConfig({
        learningRate: 0.1,
        schedule: { schedule: { kind: 'exponential', gamma: 0.5 } },
      }),
      generateDataset({ name: 'xor', samples: 60, seed: 1 }),
    );
    const history = trainer.run(4);
    expect(history[0]!.learningRate).toBeCloseTo(0.1, 12);
    expect(history[1]!.learningRate).toBeCloseTo(0.05, 12);
    expect(history[3]!.learningRate).toBeCloseTo(0.0125, 12);
  });
});

describe('gradient clipping (§4.9)', () => {
  it('keeps training finite at a learning rate that otherwise diverges', () => {
    const dataset = generateDataset({ name: 'moons', samples: 200, seed: 2 });
    const wild = new Trainer(
      xorConfig({ optimizer: { name: 'sgd' }, learningRate: 50 }),
      dataset,
    );
    wild.run(30);

    const clipped = new Trainer(
      xorConfig({ optimizer: { name: 'sgd' }, learningRate: 50, gradientClip: 1 }),
      generateDataset({ name: 'moons', samples: 200, seed: 2 }),
    );
    const history = clipped.run(30);
    expect(history.every((m) => Number.isFinite(m.trainLoss))).toBe(true);
    expect(clipped.status).not.toBe('diverged');
  });
});

describe('standardization and the deliberate leak (§7.8)', () => {
  it('fits statistics on the training split only, by default', () => {
    const dataset = generateDataset({ name: 'moons', samples: 200, seed: 1 });
    const trainer = new Trainer(xorConfig({ standardize: true }), dataset);
    expect(trainer.standardizer).not.toBeNull();

    // Training features are standardized; validation is transformed by the
    // SAME constants, so its mean is near zero but not exactly zero.
    const trainMean = columnMean(trainer.split.train.x.data, 2, 0);
    expect(trainMean).toBeCloseTo(0, 10);
    const valMean = columnMean(trainer.split.validation.x.data, 2, 0);
    expect(Math.abs(valMean)).toBeGreaterThan(0);
  });

  it('the leak option fits on all data, which is the point of the lesson', () => {
    const dataset = generateDataset({ name: 'moons', samples: 200, seed: 1 });
    const honest = new Trainer(xorConfig({ standardize: true }), dataset);
    const leaky = new Trainer(
      xorConfig({ standardize: true, leakStandardization: true }),
      generateDataset({ name: 'moons', samples: 200, seed: 1 }),
    );
    // Different constants: the leak used validation samples to compute them.
    expect(Array.from(honest.standardizer!.mean)).not.toEqual(
      Array.from(leaky.standardizer!.mean),
    );
  });

  it('rescues convergence on badly scaled features (§7.8)', () => {
    const scale = (seed: number) => {
      const d = generateDataset({ name: 'moons', samples: 200, noise: 0.1, seed });
      for (let r = 0; r < d.x.rows; r++) d.x.data[r * 2] = d.x.data[r * 2]! * 100;
      return d;
    };
    const config = xorConfig({ optimizer: { name: 'sgd' }, learningRate: 0.05 });

    const raw = new Trainer(config, scale(1));
    const rawBest = Math.max(...raw.run(120).map((m) => m.validationAccuracy ?? 0));

    const standardized = new Trainer({ ...config, standardize: true }, scale(1));
    const stdBest = Math.max(...standardized.run(120).map((m) => m.validationAccuracy ?? 0));

    expect(stdBest).toBeGreaterThan(rawBest);
  });
});

describe('classification report (§4.10)', () => {
  it('builds a confusion matrix with precision and recall', () => {
    // 3 classes, hand-built predictions.
    const predictions = fromRows([
      [0.9, 0.05, 0.05], // predicts 0
      [0.1, 0.8, 0.1], // predicts 1
      [0.1, 0.7, 0.2], // predicts 1
      [0.2, 0.2, 0.6], // predicts 2
    ]);
    const labels = Int32Array.from([0, 1, 2, 2]);
    const report = buildReport(predictions, labels, 3);

    expect(report.confusion).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 1],
    ]);
    expect(report.accuracy).toBeCloseTo(0.75, 12);
    // class 1: predicted twice, correct once
    expect(report.precision[1]!).toBeCloseTo(0.5, 12);
    // class 2: two actual, one recalled
    expect(report.recall[2]!).toBeCloseTo(0.5, 12);
  });

  it('reports zero precision for a class the model never predicts', () => {
    const predictions = fromRows([
      [0.9, 0.1],
      [0.8, 0.2],
    ]);
    const report = buildReport(predictions, Int32Array.from([0, 1]), 2);
    expect(report.precision[1]).toBe(0);
    expect(report.recall[1]).toBe(0);
  });

  it('predictedClass thresholds a single sigmoid output at 0.5', () => {
    const predictions = fromRows([[0.49], [0.5], [0.51]]);
    expect(predictedClass(predictions, 0)).toBe(0);
    expect(predictedClass(predictions, 1)).toBe(1);
    expect(predictedClass(predictions, 2)).toBe(1);
  });

  it('predictedClass takes the argmax for multi-class', () => {
    const predictions = fromRows([[0.2, 0.5, 0.3]]);
    expect(predictedClass(predictions, 0)).toBe(1);
  });

  it('is available from the trainer for the validation split', () => {
    const trainer = new Trainer(xorConfig(), generateDataset({ name: 'xor', samples: 100, seed: 1 }));
    trainer.run(30);
    const report = trainer.classificationReport();
    expect(report).not.toBeNull();
    expect(report!.confusion.length).toBeGreaterThanOrEqual(2);
  });
});

describe('frozen layers (§6.5)', () => {
  it('leaves a frozen layer bit-identical while the rest trains', () => {
    const trainer = new Trainer(xorConfig(), generateDataset({ name: 'xor', samples: 100, seed: 1 }));
    trainer.network.layers[0]!.frozen = true;
    const before = Array.from(trainer.network.layers[0]!.W.data);
    const outputBefore = Array.from(trainer.network.layers[1]!.W.data);

    trainer.run(20);

    expect(Array.from(trainer.network.layers[0]!.W.data)).toEqual(before);
    expect(Array.from(trainer.network.layers[1]!.W.data)).not.toEqual(outputBefore);
  });
});

function columnMean(data: Float64Array, cols: number, col: number): number {
  let sum = 0;
  let count = 0;
  for (let i = col; i < data.length; i += cols) {
    sum += data[i]!;
    count++;
  }
  return count === 0 ? 0 : sum / count;
}
