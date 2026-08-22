import { describe, expect, it } from 'vitest';
import { createRng } from '../rng';
import type { StreamName } from '../rng';
import { initWeights } from '../init';
import { toRows } from '../tensor';

const ALL_STREAMS: readonly StreamName[] = ['init', 'shuffle', 'dropout', 'data'];

describe('seeded PRNG', () => {
  it('produces identical sequences for identical seeds', () => {
    const a = createRng(1234).stream('init');
    const b = createRng(1234).stream('init');
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
  });

  it('produces different sequences for different seeds', () => {
    const a = createRng(1).stream('init');
    const b = createRng(2).stream('init');
    let differences = 0;
    for (let i = 0; i < 50; i++) if (a.next() !== b.next()) differences++;
    expect(differences).toBe(50);
  });

  it('decorrelates adjacent small seeds — learners type 1, 2, 3 (§4.7)', () => {
    // Without the splitmix32 expansion, mulberry32 states differing by 1 give
    // visibly similar first draws. Check the first draw of nearby seeds spreads.
    const firsts = [1, 2, 3, 4, 5].map((s) => createRng(s).stream('init').next());
    for (let i = 1; i < firsts.length; i++) {
      expect(Math.abs(firsts[i]! - firsts[i - 1]!)).toBeGreaterThan(0.01);
    }
  });

  it('stays within [0, 1)', () => {
    const r = createRng(7).stream('data');
    for (let i = 0; i < 10_000; i++) {
      const v = r.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('named streams', () => {
  it('returns the same object for repeated requests', () => {
    const src = createRng(42);
    expect(src.stream('init')).toBe(src.stream('init'));
  });

  it('gives every stream a distinct sequence', () => {
    const src = createRng(42);
    const firstDraws = ALL_STREAMS.map((name) => createRng(42).stream(name).next());
    expect(new Set(firstDraws).size).toBe(ALL_STREAMS.length);
    expect(src.seed).toBe(42);
  });

  it('does not let one stream disturb another (§4.7)', () => {
    // The whole point of named streams: changing dropout must not shift init.
    const withoutDropout = createRng(99);
    const initOnly: number[] = [];
    for (let i = 0; i < 20; i++) initOnly.push(withoutDropout.stream('init').next());

    const withDropout = createRng(99);
    // Heavily exercise an unrelated stream first.
    for (let i = 0; i < 500; i++) withDropout.stream('dropout').next();
    const initAfter: number[] = [];
    for (let i = 0; i < 20; i++) initAfter.push(withDropout.stream('init').next());

    expect(initAfter).toEqual(initOnly);
  });

  it('is insensitive to the order streams are first requested', () => {
    const forward = createRng(5);
    forward.stream('init');
    forward.stream('shuffle');
    const a = forward.stream('data').next();

    const backward = createRng(5);
    backward.stream('data');
    backward.stream('shuffle');
    const b = createRng(5).stream('data').next();

    expect(a).toBe(b);
    expect(backward.stream('data').getState()).not.toBe(0);
  });

  it('reset() rewinds every stream', () => {
    const src = createRng(11);
    const first = [src.stream('init').next(), src.stream('data').next()];
    src.reset();
    expect([src.stream('init').next(), src.stream('data').next()]).toEqual(first);
  });
});

describe('derived distributions', () => {
  it('uniform(min, max) stays in range and has the right mean', () => {
    const r = createRng(3).stream('data');
    let sum = 0;
    const n = 20_000;
    for (let i = 0; i < n; i++) {
      const v = r.uniform(-2, 6);
      expect(v).toBeGreaterThanOrEqual(-2);
      expect(v).toBeLessThan(6);
      sum += v;
    }
    expect(sum / n).toBeCloseTo(2, 1);
  });

  it('normal(mean, std) has approximately the requested moments', () => {
    const r = createRng(4).stream('init');
    const n = 50_000;
    const values: number[] = [];
    for (let i = 0; i < n; i++) values.push(r.normal(1.5, 2));
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    expect(mean).toBeCloseTo(1.5, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(2, 1);
  });

  it('normal never returns a non-finite value (the log(0) guard)', () => {
    const r = createRng(8).stream('init');
    for (let i = 0; i < 50_000; i++) {
      expect(Number.isFinite(r.normal(0, 1))).toBe(true);
    }
  });

  it('int(n) covers [0, n) and never returns n', () => {
    const r = createRng(6).stream('shuffle');
    const seen = new Set<number>();
    for (let i = 0; i < 5000; i++) {
      const v = r.int(5);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(5);
      seen.add(v);
    }
    expect(seen.size).toBe(5);
  });

  it('shuffle permutes without losing or duplicating elements', () => {
    const r = createRng(2).stream('shuffle');
    const items = Array.from({ length: 50 }, (_, i) => i);
    const original = [...items];
    r.shuffle(items);
    expect([...items].sort((a, b) => a - b)).toEqual(original);
    expect(items).not.toEqual(original);
  });

  it('shuffle is deterministic for a given seed', () => {
    const a = Array.from({ length: 30 }, (_, i) => i);
    const b = Array.from({ length: 30 }, (_, i) => i);
    createRng(77).stream('shuffle').shuffle(a);
    createRng(77).stream('shuffle').shuffle(b);
    expect(a).toEqual(b);
  });
});

describe('state and cloning', () => {
  it('clone() continues the same sequence independently', () => {
    const a = createRng(21).stream('init');
    a.next();
    a.next();
    const b = a.clone();
    const fromA = [a.next(), a.next(), a.next()];
    const fromB = [b.next(), b.next(), b.next()];
    expect(fromB).toEqual(fromA);
  });

  it('setState round-trips', () => {
    const r = createRng(31).stream('data');
    for (let i = 0; i < 10; i++) r.next();
    const saved = r.getState();
    const expected = [r.next(), r.next()];
    r.setState(saved);
    expect([r.next(), r.next()]).toEqual(expected);
  });
});

describe('initialization schemes (§4.6)', () => {
  it('glorot_uniform respects its limit', () => {
    const rng = createRng(1).stream('init');
    const fanIn = 4;
    const fanOut = 6;
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    const w = initWeights({ kind: 'glorot_uniform' }, fanIn, fanOut, rng);
    expect(w.rows).toBe(fanIn);
    expect(w.cols).toBe(fanOut);
    for (const v of w.data) {
      expect(Math.abs(v)).toBeLessThanOrEqual(limit);
    }
  });

  it('he_normal has std ≈ sqrt(2 / fan_in)', () => {
    const rng = createRng(2).stream('init');
    const fanIn = 64;
    const w = initWeights({ kind: 'he_normal' }, fanIn, 64, rng);
    const values = Array.from(w.data);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    expect(std).toBeCloseTo(Math.sqrt(2 / fanIn), 2);
  });

  it('lecun_normal has std ≈ sqrt(1 / fan_in)', () => {
    const rng = createRng(3).stream('init');
    const fanIn = 64;
    const w = initWeights({ kind: 'lecun_normal' }, fanIn, 64, rng);
    const values = Array.from(w.data);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    expect(std).toBeCloseTo(Math.sqrt(1 / fanIn), 2);
  });

  it('zeros produces an all-zero matrix (lesson 3 depends on this)', () => {
    const w = initWeights({ kind: 'zeros' }, 3, 2, createRng(1).stream('init'));
    expect(toRows(w)).toEqual([
      [0, 0],
      [0, 0],
      [0, 0],
    ]);
  });

  it('constant fills with the given value', () => {
    const w = initWeights({ kind: 'constant', value: 0.25 }, 2, 3, createRng(1).stream('init'));
    expect(toRows(w)).toEqual([
      [0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25],
    ]);
  });

  it('uniform respects its bounds', () => {
    const w = initWeights({ kind: 'uniform', min: -0.5, max: 0.5 }, 8, 8, createRng(1).stream('init'));
    for (const v of w.data) {
      expect(v).toBeGreaterThanOrEqual(-0.5);
      expect(v).toBeLessThan(0.5);
    }
  });

  it('is bitwise reproducible for a fixed seed', () => {
    const a = initWeights({ kind: 'he_normal' }, 5, 7, createRng(123).stream('init'));
    const b = initWeights({ kind: 'he_normal' }, 5, 7, createRng(123).stream('init'));
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
  });
});
