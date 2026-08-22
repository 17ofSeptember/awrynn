/*
 * Seeded pseudo-random number generation.
 *
 * Spec §4.7. Math.random() is banned in this directory and the ban is enforced
 * by src/engine/__tests__/purity.test.ts, by tsconfig.engine.json, and by
 * ESLint. Every stochastic quantity in AwryNN — weight init, batch shuffling,
 * dropout masks, dataset noise — draws from a *named* stream so that changing
 * one does not shift the others.
 *
 * Construction: a user seed is expanded by splitmix32, and each named stream
 * takes its own mulberry32 state derived from that expansion mixed with a hash
 * of the stream name. Streams are therefore independent and order-insensitive:
 * asking for 'dropout' before or after 'shuffle' yields the same sequences.
 */

/** The named streams. Adding one here is the only way to get randomness. */
export type StreamName = 'init' | 'shuffle' | 'dropout' | 'data';

export interface Rng {
  /** Uniform in [0, 1). */
  next(): number;
  /** Uniform in [min, max). */
  uniform(min: number, max: number): number;
  /** Normal via Box-Muller. */
  normal(mean: number, std: number): number;
  /** Integer in [0, maxExclusive). */
  int(maxExclusive: number): number;
  /** Fisher-Yates, in place, drawing from this stream. */
  shuffle<T>(items: T[]): void;
  /** Raw uint32 state — exposed so determinism tests can assert on it. */
  getState(): number;
  setState(state: number): void;
  /** Independent copy at the current position. */
  clone(): Rng;
}

/**
 * splitmix32 — used only to expand a user seed into well-mixed 32-bit values.
 * A raw seed like 1 or 42 has almost no entropy in its high bits, and feeding
 * that straight into mulberry32 makes the first few draws visibly correlated
 * across nearby seeds. Learners will type 1, 2, 3, so this matters.
 */
function splitmix32(seed: number): () => number {
  let a = seed | 0;
  return function nextUint32(): number {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return t >>> 0;
  };
}

/** FNV-1a over the stream name, so stream identity does not depend on order. */
function hashName(name: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

class Mulberry32 implements Rng {
  private state: number;

  constructor(state: number) {
    this.state = state | 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /*
   * Box-Muller. The second normal of the pair is DISCARDED rather than cached.
   *
   * Caching it would make the stream's state a pair (counter, pending value),
   * and every clone/serialize path would have to carry that flag or silently
   * desynchronise. Determinism is worth more here than one saved sin(); these
   * networks have tens of parameters, not millions.
   *
   * `1 - next()` puts u1 in (0, 1] so log(u1) is never -Infinity.
   */
  normal(mean: number, std: number): number {
    const u1 = 1 - this.next();
    const u2 = this.next();
    const mag = Math.sqrt(-2 * Math.log(u1));
    return mean + std * mag * Math.cos(2 * Math.PI * u2);
  }

  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }

  shuffle<T>(items: T[]): void {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const a = items[i] as T;
      const b = items[j] as T;
      items[i] = b;
      items[j] = a;
    }
  }

  getState(): number {
    return this.state >>> 0;
  }

  setState(state: number): void {
    this.state = state | 0;
  }

  clone(): Rng {
    return new Mulberry32(this.state);
  }
}

export interface RngSource {
  readonly seed: number;
  /** The stream for `name`. Repeated calls return the SAME stream object. */
  stream(name: StreamName): Rng;
  /** Rewind every stream to its initial state for this seed. */
  reset(): void;
}

class SeededSource implements RngSource {
  readonly seed: number;
  private readonly streams = new Map<StreamName, Mulberry32>();

  constructor(seed: number) {
    this.seed = seed | 0;
  }

  private initialState(name: StreamName): number {
    const expand = splitmix32(this.seed ^ hashName(name));
    // Two draws: the first of a splitmix32 sequence is the least mixed.
    expand();
    return expand() | 0;
  }

  stream(name: StreamName): Rng {
    let existing = this.streams.get(name);
    if (existing === undefined) {
      existing = new Mulberry32(this.initialState(name));
      this.streams.set(name, existing);
    }
    return existing;
  }

  reset(): void {
    for (const [name, stream] of this.streams) {
      stream.setState(this.initialState(name));
    }
  }
}

export function createRng(seed: number): RngSource {
  if (!Number.isFinite(seed)) {
    throw new Error(`createRng: seed must be a finite number, received ${String(seed)}.`);
  }
  return new SeededSource(Math.trunc(seed));
}
