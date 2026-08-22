/*
 * Weight initialization schemes.
 *
 * Spec §4.6. fan_in = n_{l-1} (rows of W), fan_out = n_l (cols of W). Biases
 * default to zeros. All normals come from Box-Muller on the seeded 'init'
 * stream — Math.random() is banned here (§4.7).
 *
 * The scaling factors are not arbitrary. Each keeps the variance of the
 * activations roughly constant as signal moves through the layers; when that
 * fails, deep stacks either saturate or vanish, which is exactly what lessons
 * 3 and 5 (§7) put on screen. `zeros` is included precisely so the symmetry
 * lesson can be demonstrated, not because it is ever a good idea.
 */

import type { Rng } from './rng';
import type { Matrix } from './tensor';
import { createMatrix } from './tensor';

export type InitScheme =
  /** U(−limit, limit), limit = sqrt(6 / (fan_in + fan_out)). Good for tanh/sigmoid. */
  | { readonly kind: 'glorot_uniform' }
  /** N(0, sqrt(2 / fan_in)). The sensible default for ReLU. */
  | { readonly kind: 'he_normal' }
  /** N(0, sqrt(1 / fan_in)). */
  | { readonly kind: 'lecun_normal' }
  | { readonly kind: 'normal'; readonly std: number }
  | { readonly kind: 'uniform'; readonly min: number; readonly max: number }
  /** Every hidden unit gets an identical gradient forever — see lesson 3 (§7.3). */
  | { readonly kind: 'zeros' }
  | { readonly kind: 'constant'; readonly value: number };

export type InitKind = InitScheme['kind'];

export const INIT_KINDS: readonly InitKind[] = [
  'glorot_uniform',
  'he_normal',
  'lecun_normal',
  'normal',
  'uniform',
  'zeros',
  'constant',
];

export function describeInit(scheme: InitScheme): string {
  switch (scheme.kind) {
    case 'glorot_uniform':
      return 'Glorot uniform';
    case 'he_normal':
      return 'He normal';
    case 'lecun_normal':
      return 'LeCun normal';
    case 'normal':
      return `Normal (σ = ${scheme.std})`;
    case 'uniform':
      return `Uniform (${scheme.min}, ${scheme.max})`;
    case 'zeros':
      return 'Zeros';
    case 'constant':
      return `Constant (${scheme.value})`;
  }
}

/** Draw a single weight value. Exposed so the UI can show what a scheme produces. */
export function sampleWeight(
  scheme: InitScheme,
  fanIn: number,
  fanOut: number,
  rng: Rng,
): number {
  switch (scheme.kind) {
    case 'glorot_uniform': {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      return rng.uniform(-limit, limit);
    }
    case 'he_normal':
      return rng.normal(0, Math.sqrt(2 / fanIn));
    case 'lecun_normal':
      return rng.normal(0, Math.sqrt(1 / fanIn));
    case 'normal':
      return rng.normal(0, scheme.std);
    case 'uniform':
      return rng.uniform(scheme.min, scheme.max);
    case 'zeros':
      return 0;
    case 'constant':
      return scheme.value;
  }
}

/** W with shape [fan_in, fan_out] (§4.1). */
export function initWeights(
  scheme: InitScheme,
  fanIn: number,
  fanOut: number,
  rng: Rng,
): Matrix {
  if (fanIn <= 0 || fanOut <= 0) {
    throw new Error(`init.initWeights: fanIn and fanOut must be positive, got (${fanIn}, ${fanOut}).`);
  }
  const w = createMatrix(fanIn, fanOut);
  // Filled in row-major order so a given seed produces the same matrix
  // regardless of how the caller later reads it.
  for (let i = 0; i < w.data.length; i++) {
    w.data[i] = sampleWeight(scheme, fanIn, fanOut, rng);
  }
  return w;
}

/** b with shape [1, units]. Spec §4.6: "Bias defaults to zeros." */
export function initBiases(units: number): Matrix {
  if (units <= 0) {
    throw new Error(`init.initBiases: units must be positive, got ${units}.`);
  }
  return createMatrix(1, units);
}
