import { describe, expect, it } from 'vitest';
import { applyStandardizer, fitStandardizer } from '../regularizers';
import { Network } from '../network';
import { createRng } from '../rng';
import { createMatrix, fromRows, toRows } from '../tensor';
import { gradientCheck } from '../gradcheck';

describe('inverted dropout (§4.9)', () => {
  function net(dropout: number): Network {
    const n = new Network({
      inputSize: 4,
      layers: [
        { units: 6, activation: 'linear' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 5,
      init: { kind: 'constant', value: 0.5 },
    });
    n.setHiddenDropout(dropout);
    return n;
  }

  it('is the identity at eval time', () => {
    const n = net(0.5);
    const x = fromRows([[1, 1, 1, 1]]);
    const a = Array.from(n.forward(x, false).data);
    const b = Array.from(n.forward(x, false).data);
    expect(a).toEqual(b);
  });

  it('drops units during training and rescales the survivors by 1/(1−p)', () => {
    const n = net(0.5);
    const x = fromRows([[1, 1, 1, 1]]);
    n.forward(x, true);
    const mask = n.layers[0]!.lastMask;
    expect(mask).not.toBeNull();
    // Inverted dropout: every entry is either 0 or exactly 1/(1−p) = 2.
    for (const m of mask as Float64Array) expect(m === 0 || Math.abs(m - 2) < 1e-12).toBe(true);
    expect(Array.from(mask as Float64Array).some((m) => m === 0)).toBe(true);
  });

  it('preserves the expected activation — that is what "inverted" buys', () => {
    // With E[m] = 1, the average of many masked forward passes must match the
    // undropped one. If the 1/(1−p) rescaling were missing or applied at eval
    // instead, this average would be low by a factor of (1−p).
    const n = net(0.5);
    const x = fromRows([[1, 1, 1, 1]]);
    const clean = n.forward(x, false).data[0] as number;

    let total = 0;
    const trials = 4000;
    for (let i = 0; i < trials; i++) total += n.forward(x, true).data[0] as number;
    expect(total / trials).toBeCloseTo(clean, 1);
  });

  it('never applies to the output layer', () => {
    const n = net(0.5);
    expect(n.layers[0]!.dropout).toBe(0.5);
    expect(n.layers[n.layers.length - 1]!.dropout).toBe(0);
  });

  it('leaves the cached activation A pristine so φ′ stays correct', () => {
    // The subtle bug this guards: multiplying the mask into A would corrupt
    // df(z, a) for tanh and sigmoid, which read `a` rather than recomputing it.
    const n = new Network({
      inputSize: 3,
      layers: [
        { units: 5, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 9,
      init: { kind: 'glorot_uniform' },
    });
    n.setHiddenDropout(0.5);
    const x = fromRows([[0.4, -0.7, 1.1]]);
    n.forward(x, true);

    const layer = n.layers[0]!;
    const z = layer.Z as { data: Float64Array };
    const a = layer.A as { data: Float64Array };
    for (let i = 0; i < a.data.length; i++) {
      // A must still equal tanh(Z) exactly, mask or no mask.
      expect(a.data[i]!).toBeCloseTo(Math.tanh(z.data[i]!), 12);
    }
    // And the masked output must differ from A on at least one unit.
    const out = layer.output as { data: Float64Array };
    expect(Array.from(out.data)).not.toEqual(Array.from(a.data));
  });

  it('masks the gradient with the SAME mask, including upstream layers', () => {
    // A dropped unit contributed nothing to the loss, so it must receive no
    // gradient — and neither must the weights feeding it.
    const n = new Network({
      inputSize: 3,
      layers: [
        { units: 6, activation: 'linear' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 4,
      init: { kind: 'glorot_uniform' },
    });
    n.setHiddenDropout(0.5);
    const x = fromRows([[1, 1, 1]]);
    const y = fromRows([[0]]);
    n.forward(x, true);
    n.backward(y);

    const layer = n.layers[0]!;
    const mask = layer.lastMask as Float64Array;
    const dZ = layer.dZ as { data: Float64Array };
    for (let u = 0; u < mask.length; u++) {
      if (mask[u] === 0) {
        // Math.abs so that a legitimate -0 does not fail Object.is equality.
        expect(Math.abs(dZ.data[u]!), `dropped unit ${u} received gradient`).toBe(0);
        // Its incoming weights must therefore get no gradient either.
        for (let i = 0; i < layer.inputs; i++) {
          expect(Math.abs(layer.dW.data[i * layer.units + u]!)).toBe(0);
        }
      }
    }
  });

  it('draws from the dropout stream, leaving init and shuffling untouched', () => {
    const withDropout = new Network({
      inputSize: 2,
      layers: [{ units: 4, activation: 'tanh' }, { units: 1, activation: 'linear' }],
      loss: 'mse',
      seed: 42,
      init: { kind: 'he_normal' },
    });
    const without = new Network({
      inputSize: 2,
      layers: [{ units: 4, activation: 'tanh' }, { units: 1, activation: 'linear' }],
      loss: 'mse',
      seed: 42,
      init: { kind: 'he_normal' },
    });
    // Same seed ⇒ same init, regardless of dropout usage afterwards.
    expect(Array.from(withDropout.captureParameters())).toEqual(
      Array.from(without.captureParameters()),
    );

    withDropout.setHiddenDropout(0.5);
    for (let i = 0; i < 20; i++) withDropout.forward(fromRows([[1, 1]]), true);
    // Shuffle stream is unaffected by all that dropout activity.
    expect(withDropout.rng.stream('shuffle').next()).toBe(without.rng.stream('shuffle').next());
  });

  it('is disabled by gradientCheck, and restored afterwards (§4.11)', () => {
    // Dropout makes L stochastic; the check must not see it.
    const n = new Network({
      inputSize: 2,
      layers: [
        { units: 5, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 17,
      init: { kind: 'glorot_uniform' },
    });
    n.setHiddenDropout(0.5);

    const rng = createRng(1).stream('data');
    const x = createMatrix(6, 2);
    for (let i = 0; i < x.data.length; i++) x.data[i] = rng.uniform(-1, 1);
    const y = createMatrix(6, 1);
    for (let i = 0; i < y.data.length; i++) y.data[i] = rng.uniform(-1, 1);

    const result = gradientCheck(n, x, y);
    expect(result.passed).toBe(true);
    // Restored, not clobbered.
    expect(n.layers[0]!.dropout).toBe(0.5);
  });

  it('rejects an out-of-range probability', () => {
    const n = net(0);
    // p = 1 would drop every unit and make the 1/(1−p) rescaling divide by zero.
    expect(() => n.setHiddenDropout(1)).toThrowError(/must be in \[0, 1\)/);
    expect(() => n.setHiddenDropout(-0.1)).toThrowError(/must be in \[0, 1\)/);
    expect(() => n.setHiddenDropout(0.99)).not.toThrow();
    expect(() => n.setHiddenDropout(0)).not.toThrow();
  });
});

describe('ablation (§6.5)', () => {
  it('forces a layer output to zero while leaving A intact', () => {
    const n = new Network({
      inputSize: 2,
      layers: [
        { units: 4, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 3,
      init: { kind: 'glorot_uniform' },
    });
    const x = fromRows([[1, -1]]);
    const before = n.forward(x, false).data[0] as number;

    n.layers[0]!.ablated = true;
    const after = n.forward(x, false).data[0] as number;
    expect(after).not.toBeCloseTo(before, 6);

    // With the hidden layer ablated the output is just the output bias.
    expect(after).toBeCloseTo(n.layers[1]!.b.data[0] as number, 12);
    // A still holds the true activation.
    const a = n.layers[0]!.A as { data: Float64Array };
    expect(Array.from(a.data).some((v) => v !== 0)).toBe(true);
  });
});

describe('standardization (§4.9)', () => {
  it('produces zero mean and unit variance per feature', () => {
    const x = fromRows([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
    ]);
    const stats = fitStandardizer(x);
    expect(stats.mean[0]).toBeCloseTo(2.5, 12);
    expect(stats.mean[1]).toBeCloseTo(250, 12);

    applyStandardizer(x, stats);
    const rows = toRows(x);
    for (let c = 0; c < 2; c++) {
      const column = rows.map((r) => r[c] as number);
      const mean = column.reduce((a, b) => a + b, 0) / column.length;
      const variance = column.reduce((a, b) => a + (b - mean) ** 2, 0) / column.length;
      expect(mean).toBeCloseTo(0, 12);
      expect(Math.sqrt(variance)).toBeCloseTo(1, 12);
    }
  });

  it('leaves a constant feature alone instead of producing NaN', () => {
    const x = fromRows([
      [5, 1],
      [5, 2],
      [5, 3],
    ]);
    const stats = fitStandardizer(x);
    expect(stats.std[0]).toBe(1);
    applyStandardizer(x, stats);
    for (const v of x.data) expect(Number.isNaN(v)).toBe(false);
    // Centred to zero, not divided by zero.
    expect(toRows(x).map((r) => r[0])).toEqual([0, 0, 0]);
  });

  it('applies training statistics to validation, not validation statistics', () => {
    // The leak §7.8 exists to demonstrate: validation must be transformed by
    // constants it had no part in computing.
    const train = fromRows([
      [0],
      [10],
    ]);
    const validation = fromRows([
      [100],
      [200],
    ]);
    const stats = fitStandardizer(train);
    applyStandardizer(validation, stats);
    // mean 5, std 5 → (100−5)/5 = 19, (200−5)/5 = 39
    expect(toRows(validation)).toEqual([[19], [39]]);
  });

  it('rejects statistics of the wrong width', () => {
    const stats = fitStandardizer(fromRows([[1, 2]]));
    expect(() => applyStandardizer(fromRows([[1, 2, 3]]), stats)).toThrowError(
      /statistics cover 2 features but the data has 3/,
    );
  });
});
