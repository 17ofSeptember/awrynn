import { describe, expect, it } from 'vitest';
import { getLoss, fusedOutputGradient, LOG_EPSILON } from '../losses';
import { getActivation, isElementwise, softmaxRows } from '../activations';
import { fromRows, toRows } from '../tensor';

/* Spec §10: "values and gradients against hand-computed values, including the fused paths." */

describe('mse', () => {
  const loss = getLoss('mse');

  it('computes ½Σ(â−y)² averaged over the batch', () => {
    const yHat = fromRows([
      [1, 2],
      [3, 4],
    ]);
    const y = fromRows([
      [0, 0],
      [1, 1],
    ]);
    // sample 0: ½(1² + 2²) = 2.5
    // sample 1: ½(2² + 3²) = 6.5
    // mean over B=2: 4.5
    expect(loss.loss(yHat, y)).toBeCloseTo(4.5, 15);
  });

  it('gradient is â − y, unaveraged', () => {
    const yHat = fromRows([
      [1, 2],
      [3, 4],
    ]);
    const y = fromRows([
      [0, 0],
      [1, 1],
    ]);
    // Deliberately NOT divided by B — §4.3 divides exactly once, later.
    expect(toRows(loss.dA(yHat, y))).toEqual([
      [1, 2],
      [2, 3],
    ]);
  });

  it('is zero at a perfect prediction', () => {
    const m = fromRows([[0.3, -1.2]]);
    expect(loss.loss(m, m)).toBe(0);
    expect(toRows(loss.dA(m, m))).toEqual([[0, 0]]);
  });
});

describe('bce', () => {
  const loss = getLoss('bce');

  it('matches a hand-computed value', () => {
    const yHat = fromRows([[0.9], [0.2]]);
    const y = fromRows([[1], [0]]);
    // sample 0: −log(0.9) = 0.105360515...
    // sample 1: −log(0.8) = 0.223143551...
    // mean = 0.164252033...
    const expected = (-Math.log(0.9) - Math.log(0.8)) / 2;
    expect(loss.loss(yHat, y)).toBeCloseTo(expected, 15);
    expect(loss.loss(yHat, y)).toBeCloseTo(0.164252033486018, 12);
  });

  it('clamps log(0) instead of returning Infinity', () => {
    const yHat = fromRows([[0]]);
    const y = fromRows([[1]]);
    expect(Number.isFinite(loss.loss(yHat, y))).toBe(true);
    expect(loss.loss(yHat, y)).toBeCloseTo(-Math.log(LOG_EPSILON), 12);
  });

  it('unfused gradient is (â−y)/(â(1−â))', () => {
    const yHat = fromRows([[0.9], [0.2]]);
    const y = fromRows([[1], [0]]);
    const g = toRows(loss.dA(yHat, y));
    expect(g[0]![0]!).toBeCloseTo((0.9 - 1) / (0.9 * 0.1), 12);
    expect(g[1]![0]!).toBeCloseTo((0.2 - 0) / (0.2 * 0.8), 12);
  });
});

describe('cce', () => {
  const loss = getLoss('cce');

  it('matches a hand-computed value on one-hot targets', () => {
    const yHat = fromRows([
      [0.7, 0.2, 0.1],
      [0.1, 0.1, 0.8],
    ]);
    const y = fromRows([
      [1, 0, 0],
      [0, 0, 1],
    ]);
    // −log(0.7) = 0.356674944, −log(0.8) = 0.223143551, mean = 0.289909248
    const expected = (-Math.log(0.7) - Math.log(0.8)) / 2;
    expect(loss.loss(yHat, y)).toBeCloseTo(expected, 15);
    expect(loss.loss(yHat, y)).toBeCloseTo(0.2899092476264711, 12);
  });

  it('only the true class contributes', () => {
    const a = fromRows([[0.5, 0.3, 0.2]]);
    const b = fromRows([[0.5, 0.2, 0.3]]);
    const y = fromRows([[1, 0, 0]]);
    expect(loss.loss(a, y)).toBeCloseTo(loss.loss(b, y), 15);
  });

  it('unfused gradient is −y_k/â_k on the true class and 0 elsewhere', () => {
    const yHat = fromRows([[0.7, 0.2, 0.1]]);
    const y = fromRows([[1, 0, 0]]);
    const g = toRows(loss.dA(yHat, y))[0] as number[];
    expect(g[0]!).toBeCloseTo(-1 / 0.7, 12);
    expect(g[1]!).toBe(0);
    expect(g[2]!).toBe(0);
  });
});

/*
 * The fused paths (§4.3). These are the tests that matter most in this file:
 * they check the analytic cancellation numerically instead of trusting the
 * derivation in the comment above fusedOutputGradient().
 */
describe('fused output gradients — verifying the cancellation, not assuming it', () => {
  it('sigmoid + bce: dZ = â − y equals dA ⊙ σ′(z)', () => {
    const sigmoidAct = getActivation('sigmoid');
    if (!isElementwise(sigmoidAct)) throw new Error('sigmoid must be elementwise');

    const z = fromRows([[2.0], [-1.3], [0.4], [-0.05]]);
    const yHat = fromRows(toRows(z).map((row) => [sigmoidAct.f(row[0] as number)]));
    const y = fromRows([[1], [0], [1], [0]]);

    const fused = toRows(fusedOutputGradient(yHat, y));
    const unfusedDA = toRows(getLoss('bce').dA(yHat, y));

    for (let i = 0; i < fused.length; i++) {
      const zi = z.data[i] as number;
      const ai = yHat.data[i] as number;
      const chained = (unfusedDA[i]![0] as number) * sigmoidAct.df(zi, ai);
      expect(fused[i]![0]!).toBeCloseTo(chained, 12);
    }
  });

  it('softmax + cce: dZ = Ŷ − Y equals the full Jacobian product', () => {
    const z = fromRows([
      [1.0, 2.0, 0.5],
      [-1.0, 0.3, 2.2],
    ]);
    const yHat = softmaxRows(z);
    const y = fromRows([
      [1, 0, 0],
      [0, 0, 1],
    ]);

    const fused = toRows(fusedOutputGradient(yHat, y));
    const dA = toRows(getLoss('cce').dA(yHat, y));
    const p = toRows(yHat);

    // dZ_j = Σ_k dA_k · ∂â_k/∂z_j, with ∂â_k/∂z_j = â_k(δ_kj − â_j).
    for (let r = 0; r < z.rows; r++) {
      for (let j = 0; j < z.cols; j++) {
        let sum = 0;
        for (let k = 0; k < z.cols; k++) {
          const delta = k === j ? 1 : 0;
          sum += (dA[r]![k] as number) * (p[r]![k] as number) * (delta - (p[r]![j] as number));
        }
        expect(fused[r]![j]!).toBeCloseTo(sum, 12);
      }
    }
  });

  it('is numerically stable where the unfused path is not', () => {
    // A confident, correct sigmoid output: â(1−â) underflows toward zero, so
    // the unfused gradient divides by ~0 while the fused one stays tiny and finite.
    const yHat = fromRows([[1 - 1e-15]]);
    const y = fromRows([[1]]);
    const fused = fusedOutputGradient(yHat, y).data[0] as number;
    expect(Number.isFinite(fused)).toBe(true);
    expect(Math.abs(fused)).toBeLessThan(1e-14);
  });
});
