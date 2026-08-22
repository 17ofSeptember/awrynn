import { describe, expect, it } from 'vitest';
import { BATCH_NORM_EPSILON, BATCH_NORM_MOMENTUM } from '../layers';
import type { DenseLayer } from '../layers';
import { Network } from '../network';
import type { NetworkConfig } from '../network';
import { createRng } from '../rng';
import { createMatrix, fromRows, rowView } from '../tensor';
import type { Matrix } from '../tensor';

/*
 * The gradient check proves the derivative. It says nothing about whether the
 * forward pass is batch normalization rather than some other affine map, and
 * nothing about the train/eval split, which is the part people actually get
 * wrong. That is what this file is for.
 */

const config = (overrides: Partial<NetworkConfig> = {}): NetworkConfig => ({
  inputSize: 2,
  layers: [
    { units: 4, activation: 'tanh', batchNorm: true },
    { units: 1, activation: 'sigmoid' },
  ],
  loss: 'bce',
  seed: 11,
  init: { kind: 'glorot_uniform' },
  l2: 0,
  ...overrides,
});

function randomBatch(rows: number, cols: number, seed: number): Matrix {
  const rng = createRng(seed).stream('data');
  const m = createMatrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) m.data[i] = rng.uniform(-2, 2);
  return m;
}

function columnStats(m: Matrix, col: number): { mean: number; variance: number } {
  let sum = 0;
  for (let i = 0; i < m.rows; i++) sum += m.data[i * m.cols + col] as number;
  const mean = sum / m.rows;
  let sq = 0;
  for (let i = 0; i < m.rows; i++) {
    const d = (m.data[i * m.cols + col] as number) - mean;
    sq += d * d;
  }
  return { mean, variance: sq / m.rows };
}

describe('the normalization itself', () => {
  it('leaves each unit with zero mean and unit variance across the batch', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    net.forward(randomBatch(32, 2, 4), true);

    const u = layer.U as Matrix;
    const xHat = layer.normalized as Matrix;
    for (let j = 0; j < layer.units; j++) {
      const { mean, variance } = columnStats(xHat, j);
      expect(mean, `unit ${j} mean`).toBeCloseTo(0, 12);

      /*
       * Not exactly 1, and the shortfall is not slop: dividing by √(σ² + ε)
       * rather than by σ leaves a variance of σ²/(σ² + ε) exactly. Asserting
       * that identity rather than "near enough to 1" is what pins ε's role,
       * and it is the difference between a test that would notice a wrong ε
       * and one that would not.
       */
      const source = columnStats(u, j).variance;
      expect(variance, `unit ${j} variance`).toBeCloseTo(
        source / (source + BATCH_NORM_EPSILON),
        12,
      );
    }
  });

  it('is a normalization, not a rescaling: shifting every input leaves it unchanged', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(16, 2, 5);

    net.forward(x, true);
    const before = Float64Array.from((layer.normalized as Matrix).data);

    // Shift every sample by the same vector. U shifts, μ shifts with it, and
    // U − μ does not move at all.
    const shifted = createMatrix(x.rows, x.cols);
    for (let i = 0; i < x.data.length; i++) {
      shifted.data[i] = (x.data[i] as number) + (i % 2 === 0 ? 3.5 : -1.25);
    }
    net.forward(shifted, true);
    const after = layer.normalized as Matrix;

    for (let i = 0; i < before.length; i++) {
      expect(after.data[i], `element ${i}`).toBeCloseTo(before[i] as number, 10);
    }
  });

  it('γ = 1 and b = 0 at initialization, so switching it on is not also a rescale', () => {
    const layer = new Network(config()).layers[0] as DenseLayer;
    expect(Array.from(layer.gamma.data)).toEqual([1, 1, 1, 1]);
    expect(Array.from(layer.b.data)).toEqual([0, 0, 0, 0]);
    expect(Array.from(layer.runningMean)).toEqual([0, 0, 0, 0]);
    expect(Array.from(layer.runningVar)).toEqual([1, 1, 1, 1]);
  });

  it('survives a unit that is constant across the batch, which a dead ReLU is', () => {
    const net = new Network(
      config({ layers: [{ units: 3, activation: 'relu', batchNorm: true }, { units: 1, activation: 'sigmoid' }] }),
    );
    const layer = net.layers[0] as DenseLayer;
    // Zero the weights into unit 1, so its U column is identically zero and its
    // variance is exactly zero.
    for (let r = 0; r < layer.inputs; r++) layer.W.data[r * layer.units + 1] = 0;

    const yHat = net.forward(randomBatch(8, 2, 6), true);
    expect(Array.from(yHat.data).every(Number.isFinite)).toBe(true);
    const xHat = layer.normalized as Matrix;
    for (let i = 0; i < xHat.rows; i++) {
      // 0 / √(0 + ε) is 0, not NaN. That is what the ε is for.
      expect(xHat.data[i * xHat.cols + 1]).toBe(0);
    }
  });
});

describe('training and evaluation are different functions', () => {
  it('a sample gets a different answer depending on the mode', () => {
    const net = new Network(config());
    const x = randomBatch(24, 2, 7);
    // Train the running statistics away from their defaults.
    for (let i = 0; i < 30; i++) net.forward(x, true);

    const training = Float64Array.from(net.forward(x, true).data);
    const evaluating = Float64Array.from(net.forward(x, false).data);

    const differences = training.filter((v, i) => Math.abs(v - (evaluating[i] as number)) > 1e-6);
    expect(differences.length, 'the two modes should not agree').toBeGreaterThan(0);
  });

  it('eval mode does not depend on who else is in the batch; training mode does', () => {
    const net = new Network(config());
    const x = randomBatch(20, 2, 8);
    for (let i = 0; i < 20; i++) net.forward(x, true);

    const first = rowView(x, 4);

    // Eval: the running statistics are fixed, so the first four rows predict the
    // same whether or not the other sixteen are present. This is the property
    // that makes a prediction a prediction.
    const alone = Float64Array.from(net.forward(first, false).data);
    const together = net.forward(x, false);
    for (let i = 0; i < alone.length; i++) {
      expect(alone[i], `row ${i}`).toBeCloseTo(together.data[i] as number, 12);
    }

    // Training: the statistics come from the batch, so they do not.
    const aloneTrain = Float64Array.from(net.forward(first, true).data);
    const togetherTrain = net.forward(x, true);
    const moved = aloneTrain.filter(
      (v, i) => Math.abs(v - (togetherTrain.data[i] as number)) > 1e-6,
    );
    expect(moved.length, 'batch statistics should couple the samples').toBeGreaterThan(0);
  });

  it('reports which statistics the last forward actually used', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(8, 2, 9);

    net.forward(x, true);
    expect(layer.statistics?.fromBatch).toBe(true);

    net.forward(x, false);
    expect(layer.statistics?.fromBatch).toBe(false);

    // A batch of one has no spread of its own and falls back, even in training.
    net.forward(rowView(x, 1), true);
    expect(layer.statistics?.fromBatch).toBe(false);
  });

  it('a batch of one falls back to the running statistics rather than collapsing', () => {
    /*
     * Normalizing a single sample by its own statistics would map every unit to
     * exactly zero and cut the gradient at that layer, silently. The last batch
     * of an epoch can easily hold one sample, so this cannot be an error.
     */
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(16, 2, 10);
    for (let i = 0; i < 20; i++) net.forward(x, true);
    const savedStatistics = net.captureBuffers();

    const one = rowView(x, 1);
    net.forward(one, true);
    const xHat = layer.normalized as Matrix;
    expect(Array.from(xHat.data).some((v) => Math.abs(v) > 1e-6)).toBe(true);

    // And it contributes nothing to the estimate it just borrowed.
    expect(Array.from(net.buffers)).toEqual(Array.from(savedStatistics));
  });
});

describe('the running statistics', () => {
  it('converge toward the statistics of what the layer actually sees', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(64, 2, 11);

    for (let i = 0; i < 400; i++) net.forward(x, true);

    // With the same batch every time, the exponential average converges to that
    // batch's own statistics.
    const u = layer.U as Matrix;
    for (let j = 0; j < layer.units; j++) {
      const { mean, variance } = columnStats(u, j);
      expect(layer.runningMean[j], `unit ${j} mean`).toBeCloseTo(mean, 8);
      // The running estimate takes the UNBIASED variance, so it lands above the
      // biased one used to normalize, by exactly B/(B−1).
      const unbiased = variance * (u.rows / (u.rows - 1));
      expect(layer.runningVar[j], `unit ${j} variance`).toBeCloseTo(unbiased, 8);
    }
  });

  it('moves by exactly the documented momentum', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(8, 2, 12);

    net.forward(x, true);
    const u = layer.U as Matrix;
    const { mean } = columnStats(u, 0);
    // One step from the initial 0: (1−m)·0 + m·μ.
    expect(layer.runningMean[0] as number).toBeCloseTo(BATCH_NORM_MOMENTUM * mean, 12);
  });

  it('does not move in eval mode, or when frozen', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(8, 2, 13);

    net.forward(x, false);
    expect(Array.from(net.buffers)).toEqual(new Array(8).fill(0).map((_, i) => (i < 4 ? 0 : 1)));

    layer.freezeStatistics = true;
    net.forward(x, true);
    expect(Array.from(net.buffers)).toEqual(new Array(8).fill(0).map((_, i) => (i < 4 ? 0 : 1)));
  });

  it('lives outside params, so no optimizer can reach it', () => {
    const net = new Network(config());
    // 2·4 W + 4 b + 4 γ, then 4·1 W + 1 b.
    expect(net.parameterCount).toBe(8 + 4 + 4 + 4 + 1);
    expect(net.params.length).toBe(net.parameterCount);
    // μ̂ and σ̂² for the one normalizing layer, and nothing for the other.
    expect(net.buffers.length).toBe(8);
  });

  it('is empty for a network with no batch normalization anywhere', () => {
    const net = new Network(
      config({
        layers: [
          { units: 4, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
      }),
    );
    expect(net.buffers.length).toBe(0);
    expect(net.parameterCount).toBe(8 + 4 + 4 + 1);
  });
});

describe('carrying a normalized network from one place to another', () => {
  const trained = (): Network => {
    const net = new Network(config());
    const x = randomBatch(32, 2, 14);
    for (let i = 0; i < 25; i++) net.forward(x, true);
    return net;
  };

  it('serialize and deserialize preserve γ and the statistics exactly', () => {
    const net = trained();
    const x = randomBatch(6, 2, 15);
    const before = Float64Array.from(net.forward(x, false).data);

    const restored = Network.deserialize(JSON.parse(JSON.stringify(net.serialize())));
    expect(Array.from(restored.buffers)).toEqual(Array.from(net.buffers));
    expect(Array.from(restored.params)).toEqual(Array.from(net.params));

    const after = restored.forward(x, false);
    for (let i = 0; i < before.length; i++) {
      expect(after.data[i], `row ${i}`).toBe(before[i] as number);
    }
  });

  it('a file missing the statistics is refused rather than silently wrong', () => {
    const net = trained();
    const stripped = { ...net.serialize(), runningMeans: undefined, runningVars: undefined };
    expect(() => Network.deserialize(stripped)).toThrowError(/running statistics/);
  });

  it('parameters alone are not enough, and that is why buffers travel too', () => {
    // Restore the weights but not the statistics: the network is a different
    // function, and nothing about it looks broken.
    const net = trained();
    const x = randomBatch(6, 2, 16);
    const expected = Float64Array.from(net.forward(x, false).data);

    const fresh = new Network(config());
    fresh.restoreParameters(net.captureParameters());
    const wrong = fresh.forward(x, false);
    expect(Array.from(wrong.data)).not.toEqual(Array.from(expected));

    fresh.restoreBuffers(net.captureBuffers());
    const right = fresh.forward(x, false);
    for (let i = 0; i < expected.length; i++) {
      expect(right.data[i], `row ${i}`).toBe(expected[i] as number);
    }
  });

  it('clone carries both', () => {
    const net = trained();
    const copy = net.clone();
    expect(Array.from(copy.buffers)).toEqual(Array.from(net.buffers));
    const x = randomBatch(6, 2, 17);
    expect(Array.from(copy.forward(x, false).data)).toEqual(Array.from(net.forward(x, false).data));
  });

  it('resetToInit puts γ and the statistics back to their starting values', () => {
    const net = trained();
    expect(Array.from(net.buffers)).not.toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    net.resetToInit();
    expect(Array.from(net.buffers)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(Array.from((net.layers[0] as DenseLayer).gamma.data)).toEqual([1, 1, 1, 1]);
  });

  it('restoreBuffers rejects the wrong length', () => {
    const net = trained();
    expect(() => net.restoreBuffers(new Float64Array(3))).toThrowError(/expected 8 values, got 3/);
  });
});

describe('the bias is β, and there is only one of it', () => {
  it('is added after the normalization, so it survives where a pre-norm bias would not', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(12, 2, 18);

    net.forward(x, true);
    const withoutBias = Float64Array.from((layer.Z as Matrix).data);

    for (let j = 0; j < layer.units; j++) layer.b.data[j] = 0.75;
    net.forward(x, true);
    const withBias = layer.Z as Matrix;

    // Every Z moves by exactly the bias. A bias added BEFORE the normalization
    // would have been subtracted straight back out again by μ.
    for (let i = 0; i < withoutBias.length; i++) {
      expect(withBias.data[i], `element ${i}`).toBeCloseTo((withoutBias[i] as number) + 0.75, 10);
    }
  });

  it('Z is still the true pre-activation, so φ(Z) is still A', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    net.forward(randomBatch(10, 2, 19), true);

    const z = layer.Z as Matrix;
    const a = layer.A as Matrix;
    for (let i = 0; i < z.data.length; i++) {
      expect(a.data[i], `element ${i}`).toBeCloseTo(Math.tanh(z.data[i] as number), 12);
    }
  });

  it('the layer reports the statistics behind its own Z, so the arithmetic closes', () => {
    // The dissection view rebuilds Z from these. If they did not close to the
    // engine's cached Z, every formula card on a normalized layer would be
    // showing numbers that do not add up.
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    for (let i = 0; i < 5; i++) net.forward(randomBatch(9, 2, 20), true);

    const u = layer.U as Matrix;
    const z = layer.Z as Matrix;
    const stats = layer.statistics;
    expect(stats).not.toBeNull();
    if (stats === null) return;

    let worst = 0;
    for (let i = 0; i < u.rows; i++) {
      for (let j = 0; j < layer.units; j++) {
        const hat = ((u.data[i * layer.units + j] as number) - (stats.mean[j] as number)) *
          (stats.invStd[j] as number);
        const rebuilt = (layer.gamma.data[j] as number) * hat + (layer.b.data[j] as number);
        worst = Math.max(worst, Math.abs(rebuilt - (z.data[i * layer.units + j] as number)));
      }
    }
    expect(worst).toBeLessThan(1e-12);
  });
});

describe('spec round-trips', () => {
  it('a normalized layer reports batchNorm, a plain one omits it entirely', () => {
    const net = new Network(config());
    expect((net.layers[0] as DenseLayer).spec()).toEqual({
      units: 4,
      activation: 'tanh',
      batchNorm: true,
    });
    // Omitted, not false: a spec made before batch norm existed must still
    // compare and encode identically.
    expect((net.layers[1] as DenseLayer).spec()).toEqual({ units: 1, activation: 'sigmoid' });
  });

  it('ε is the documented one', () => {
    const net = new Network(config());
    const layer = net.layers[0] as DenseLayer;
    // Feed a constant column so σ² is exactly 0 and only ε is left.
    for (let r = 0; r < layer.inputs; r++) layer.W.data[r * layer.units] = 0;
    net.forward(fromRows([[1, 1], [2, 2], [3, 3]]), true);
    expect(layer.statistics?.invStd[0]).toBeCloseTo(1 / Math.sqrt(BATCH_NORM_EPSILON), 6);
  });
});

describe('looking at a network does not change it', () => {
  it('inspect() silences dropout and freezes the statistics, then restores both', () => {
    const net = new Network(config());
    net.setHiddenDropout(0.4);
    const layer = net.layers[0] as DenseLayer;
    const x = randomBatch(16, 2, 21);
    for (let i = 0; i < 10; i++) net.forward(x, true);

    const before = net.captureBuffers();
    const inside = net.inspect(() => {
      net.forward(x, true);
      return { dropout: layer.dropout, frozen: layer.freezeStatistics };
    });

    expect(inside.dropout).toBe(0);
    expect(inside.frozen).toBe(true);
    expect(Array.from(net.buffers)).toEqual(Array.from(before));

    // And the caller's configuration is back.
    expect(layer.dropout).toBeCloseTo(0.4, 12);
    expect(layer.freezeStatistics).toBe(false);
  });

  it('restores the configuration even when the body throws', () => {
    const net = new Network(config());
    net.setHiddenDropout(0.25);
    const layer = net.layers[0] as DenseLayer;

    expect(() =>
      net.inspect(() => {
        throw new Error('boom');
      }),
    ).toThrowError('boom');

    expect(layer.dropout).toBeCloseTo(0.25, 12);
    expect(layer.freezeStatistics).toBe(false);
  });

  it('a training-mode forward outside inspect() DOES move them, which is the point', () => {
    const net = new Network(config());
    const before = net.captureBuffers();
    net.forward(randomBatch(16, 2, 22), true);
    expect(Array.from(net.buffers)).not.toEqual(Array.from(before));
  });
});

describe('flipping the switch changes one thing only', () => {
  it('leaves W and b bit-identical, because γ draws no randomness', () => {
    /*
     * This is what makes an honest A/B comparison possible: at the same seed,
     * the normalized network and the plain one start from exactly the same
     * weights, so any difference in how they train is the normalization and
     * nothing else. It holds because γ is filled with ones rather than sampled,
     * so it never touches the init stream and never shifts what W draws.
     */
    const build = (batchNorm: boolean): Network =>
      new Network(
        config({
          layers: [
            { units: 6, activation: 'tanh', batchNorm },
            { units: 4, activation: 'tanh', batchNorm },
            { units: 1, activation: 'sigmoid' },
          ],
        }),
      );
    const plain = build(false);
    const normalized = build(true);

    const learned = (net: Network): number[] =>
      net.layers.flatMap((l) => [...Array.from(l.W.data), ...Array.from(l.b.data)]);
    expect(learned(normalized)).toEqual(learned(plain));

    // And the only difference in the storage is γ, plus its statistics.
    expect(normalized.params.length - plain.params.length).toBe(6 + 4);
    expect(normalized.buffers.length).toBe(2 * (6 + 4));
    expect(plain.buffers.length).toBe(0);
  });
});
