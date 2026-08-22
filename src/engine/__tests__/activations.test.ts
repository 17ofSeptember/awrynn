import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEAKY_ALPHA,
  getActivation,
  hasKink,
  isElementwise,
  sigmoid,
  softmaxRows,
} from '../activations';
import type { ElementwiseActivation } from '../activations';
import { fromRows, toRows } from '../tensor';

/*
 * Spec §10: "f and df against hand-computed values at z ∈ {−3, −1, −0.001, 0,
 * 0.001, 1, 3}". That grid is chosen to straddle the ReLU kink from both sides
 * at ±0.001 and to sit well into saturation at ±3.
 */
const Z: readonly number[] = [-3, -1, -0.001, 0, 0.001, 1, 3];

function elementwise(name: 'linear' | 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid'): ElementwiseActivation {
  const act = getActivation(name);
  if (!isElementwise(act)) throw new Error(`${name} should be elementwise`);
  return act;
}

describe('linear', () => {
  const act = elementwise('linear');
  it('f(z) = z and φ′ = 1 everywhere', () => {
    for (const z of Z) {
      expect(act.f(z)).toBe(z);
      expect(act.df(z, act.f(z))).toBe(1);
    }
  });
});

describe('relu', () => {
  const act = elementwise('relu');

  it('f = max(0, z)', () => {
    expect(Z.map((z) => act.f(z))).toEqual([0, 0, 0, 0, 0.001, 1, 3]);
  });

  it('φ′ = 1 for z > 0 and 0 otherwise, including exactly 0 at the kink', () => {
    expect(Z.map((z) => act.df(z, act.f(z)))).toEqual([0, 0, 0, 0, 1, 1, 1]);
  });

  it('is flagged as kinked so gradcheck knows to guard it', () => {
    expect(hasKink('relu')).toBe(true);
  });
});

describe('leaky_relu', () => {
  const act = elementwise('leaky_relu');
  const a = DEFAULT_LEAKY_ALPHA;

  it('f = z for z > 0, αz otherwise', () => {
    expect(act.f(-3)).toBeCloseTo(-3 * a, 15);
    expect(act.f(-1)).toBeCloseTo(-1 * a, 15);
    expect(act.f(-0.001)).toBeCloseTo(-0.001 * a, 15);
    expect(act.f(0)).toBe(0);
    expect(act.f(0.001)).toBe(0.001);
    expect(act.f(3)).toBe(3);
  });

  it('φ′ = 1 for z > 0, α otherwise (α at z = 0, per the spec table)', () => {
    expect(Z.map((z) => act.df(z, act.f(z)))).toEqual([a, a, a, a, 1, 1, 1]);
  });

  it('honours a configured α', () => {
    const custom = getActivation('leaky_relu', { leakyAlpha: 0.2 });
    if (!isElementwise(custom)) throw new Error('expected elementwise');
    expect(custom.f(-1)).toBeCloseTo(-0.2, 15);
    expect(custom.df(-1, -0.2)).toBe(0.2);
  });
});

describe('tanh', () => {
  const act = elementwise('tanh');

  it('f matches Math.tanh', () => {
    for (const z of Z) expect(act.f(z)).toBeCloseTo(Math.tanh(z), 15);
  });

  it("φ′ = 1 − a², reusing the activation", () => {
    for (const z of Z) {
      const a = act.f(z);
      expect(act.df(z, a)).toBeCloseTo(1 - Math.tanh(z) ** 2, 15);
    }
  });

  it('saturates: φ′(3) is small, and 0.25 is nowhere near the tanh maximum', () => {
    expect(act.df(3, act.f(3))).toBeLessThan(0.01);
    expect(act.df(0, act.f(0))).toBe(1);
  });
});

describe('sigmoid', () => {
  const act = elementwise('sigmoid');

  it('f matches the closed form on the grid', () => {
    for (const z of Z) {
      expect(act.f(z)).toBeCloseTo(1 / (1 + Math.exp(-z)), 15);
    }
  });

  it('φ′ = a(1 − a), peaking at exactly 0.25 (the vanishing-gradient number)', () => {
    expect(act.df(0, act.f(0))).toBeCloseTo(0.25, 15);
    for (const z of Z) {
      const a = act.f(z);
      expect(act.df(z, a)).toBeCloseTo(a * (1 - a), 15);
      expect(act.df(z, a)).toBeLessThanOrEqual(0.25);
    }
  });

  it('is stable for large |z| — the naive form overflows here (§4.4)', () => {
    expect(sigmoid(-800)).toBe(0);
    expect(sigmoid(800)).toBe(1);
    expect(Number.isNaN(sigmoid(-800))).toBe(false);
    expect(Number.isFinite(sigmoid(-800))).toBe(true);
    // The failure mode being guarded against: exp(800) is Infinity.
    expect(Math.exp(800)).toBe(Infinity);
    expect(sigmoid(-745)).toBeGreaterThanOrEqual(0);
  });

  it('is symmetric about 0.5', () => {
    for (const z of Z) expect(sigmoid(z) + sigmoid(-z)).toBeCloseTo(1, 15);
  });
});

describe('softmax', () => {
  it('rows sum to 1', () => {
    const z = fromRows([
      [1, 2, 3],
      [-1, 0, 4],
      [0, 0, 0],
    ]);
    for (const row of toRows(softmaxRows(z))) {
      expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 15);
    }
  });

  it('matches a hand-computed row', () => {
    // z = [1, 2, 3]; shift by max 3 -> [-2, -1, 0]
    // exps: e^-2 = 0.135335..., e^-1 = 0.367879..., e^0 = 1
    // sum = 1.503214...
    const s = toRows(softmaxRows(fromRows([[1, 2, 3]])))[0] as number[];
    const e = [Math.exp(-2), Math.exp(-1), 1];
    const sum = e[0]! + e[1]! + e[2]!;
    expect(s[0]!).toBeCloseTo(e[0]! / sum, 15);
    expect(s[1]!).toBeCloseTo(e[1]! / sum, 15);
    expect(s[2]!).toBeCloseTo(e[2]! / sum, 15);
    expect(s[2]!).toBeCloseTo(0.6652409557748219, 12);
  });

  it('is invariant to adding a constant across a row', () => {
    const base = toRows(softmaxRows(fromRows([[1, 2, 3]])))[0] as number[];
    const shifted = toRows(softmaxRows(fromRows([[101, 102, 103]])))[0] as number[];
    for (let i = 0; i < base.length; i++) expect(shifted[i]!).toBeCloseTo(base[i]!, 15);
  });

  it('survives logits that would overflow the naive form (§4.4)', () => {
    // Without the row-max shift this is exp(1000)/exp(1000) = Infinity/Infinity = NaN.
    const s = toRows(softmaxRows(fromRows([[1000, 999, 998]])))[0] as number[];
    expect(s.every((v) => Number.isFinite(v))).toBe(true);
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15);
    expect(s[0]!).toBeGreaterThan(s[1]!);
    expect(Math.exp(1000)).toBe(Infinity);
  });

  it('handles very negative logits without producing NaN', () => {
    const s = toRows(softmaxRows(fromRows([[-1000, -1001, -1002]])))[0] as number[];
    expect(s.every((v) => Number.isFinite(v))).toBe(true);
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15);
  });

  it('treats each row independently', () => {
    const two = toRows(softmaxRows(fromRows([[1, 2, 3], [1, 2, 3]])));
    expect(two[0]).toEqual(two[1]);
  });

  it('is not elementwise, and is not flagged as kinked', () => {
    expect(isElementwise(getActivation('softmax'))).toBe(false);
    expect(hasKink('softmax')).toBe(false);
  });
});
