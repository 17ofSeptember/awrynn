/*
 * Activation functions and their derivatives.
 *
 * Spec §4.4. Each elementwise activation is a pair { f(z), df(z, a) }. Passing
 * the already-computed activation `a` into df is what lets sigmoid and tanh
 * reuse it: sigmoid' = a(1-a) and tanh' = 1 - a², neither of which needs z.
 *
 * softmax is NOT elementwise — it is row-wise and its Jacobian is dense — so it
 * is modelled separately and is legal only as the final activation paired with
 * categorical cross-entropy, where the fused gradient dZ = Ŷ - Y makes the
 * Jacobian disappear (§4.3).
 */

import type { Matrix } from './tensor';
import { ensureShape } from './tensor';

export type ActivationName = 'linear' | 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid' | 'softmax';

export const ACTIVATION_NAMES: readonly ActivationName[] = [
  'linear',
  'relu',
  'leaky_relu',
  'tanh',
  'sigmoid',
  'softmax',
];

export const DEFAULT_LEAKY_ALPHA = 0.01;

export interface ElementwiseActivation {
  readonly name: Exclude<ActivationName, 'softmax'>;
  readonly elementwise: true;
  /** Output range, or null when unbounded. The canvas maps node fill through this (§6.2). */
  readonly range: readonly [number, number] | null;
  f(z: number): number;
  /** φ'(z). `a` is φ(z), already computed. */
  df(z: number, a: number): number;
}

export interface SoftmaxActivation {
  readonly name: 'softmax';
  readonly elementwise: false;
  readonly range: readonly [number, number];
}

export type Activation = ElementwiseActivation | SoftmaxActivation;

/* ------------------------------------------------------------------ *
 * Numerically stable sigmoid
 *
 * Spec §4.4 makes this mandatory. The naive 1/(1+exp(-z)) overflows exp() for
 * large NEGATIVE z (exp(710) is Infinity in float64), yielding 1/Infinity = 0 —
 * which is the right answer by luck. The real failure is that intermediate
 * Infinity propagates through anything that touches it first. The branchless
 * two-sided form never exponentiates a positive argument.
 * ------------------------------------------------------------------ */
export function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

const linear: ElementwiseActivation = {
  name: 'linear',
  elementwise: true,
  range: null,
  f: (z) => z,
  df: () => 1,
};

const relu: ElementwiseActivation = {
  name: 'relu',
  elementwise: true,
  range: null,
  f: (z) => (z > 0 ? z : 0),
  // Defined as 0 at z = 0, per spec §4.4. The true subderivative at the kink is
  // any value in [0, 1]; 0 is the conventional choice and gradcheck skips
  // coordinates whose perturbation straddles the kink (§4.11).
  df: (z) => (z > 0 ? 1 : 0),
};

function leakyRelu(alpha: number): ElementwiseActivation {
  return {
    name: 'leaky_relu',
    elementwise: true,
    range: null,
    f: (z) => (z > 0 ? z : alpha * z),
    df: (z) => (z > 0 ? 1 : alpha),
  };
}

const tanhActivation: ElementwiseActivation = {
  name: 'tanh',
  elementwise: true,
  range: [-1, 1],
  f: (z) => Math.tanh(z),
  df: (_z, a) => 1 - a * a,
};

const sigmoidActivation: ElementwiseActivation = {
  name: 'sigmoid',
  elementwise: true,
  range: [0, 1],
  f: sigmoid,
  df: (_z, a) => a * (1 - a),
};

const softmaxActivation: SoftmaxActivation = {
  name: 'softmax',
  elementwise: false,
  range: [0, 1],
};

export interface ActivationOptions {
  readonly leakyAlpha?: number | undefined;
}

export function getActivation(name: ActivationName, options: ActivationOptions = {}): Activation {
  switch (name) {
    case 'linear':
      return linear;
    case 'relu':
      return relu;
    case 'leaky_relu':
      return leakyRelu(options.leakyAlpha ?? DEFAULT_LEAKY_ALPHA);
    case 'tanh':
      return tanhActivation;
    case 'sigmoid':
      return sigmoidActivation;
    case 'softmax':
      return softmaxActivation;
  }
}

export function isElementwise(a: Activation): a is ElementwiseActivation {
  return a.elementwise;
}

/**
 * Activations whose derivative is discontinuous. gradcheck must detect when a
 * perturbation moves a unit across the kink (§4.11).
 */
export function hasKink(name: ActivationName): boolean {
  return name === 'relu' || name === 'leaky_relu';
}

/* ------------------------------------------------------------------ *
 * Matrix-level application
 * ------------------------------------------------------------------ */

/** A = φ(Z), row-wise for softmax and elementwise otherwise. */
export function applyActivation(act: Activation, z: Matrix, out: Matrix | null = null): Matrix {
  const a = ensureShape(out, z.rows, z.cols);
  if (act.elementwise) {
    for (let i = 0; i < z.data.length; i++) a.data[i] = act.f(z.data[i]!);
    return a;
  }
  return softmaxRows(z, a);
}

/**
 * Row-wise softmax with the max subtracted before exponentiating (§4.4).
 *
 * Subtracting the row max is algebraically a no-op — the shared factor cancels
 * between numerator and denominator — but it caps the largest exponent at
 * exp(0) = 1, so a logit of 800 produces a probability instead of Infinity/Infinity = NaN.
 */
export function softmaxRows(z: Matrix, out: Matrix | null = null): Matrix {
  const a = ensureShape(out, z.rows, z.cols);
  for (let r = 0; r < z.rows; r++) {
    const row = r * z.cols;
    let max = -Infinity;
    for (let j = 0; j < z.cols; j++) {
      const v = z.data[row + j]!;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let j = 0; j < z.cols; j++) {
      const e = Math.exp(z.data[row + j]! - max);
      a.data[row + j] = e;
      sum += e;
    }
    for (let j = 0; j < z.cols; j++) {
      a.data[row + j] = a.data[row + j]! / sum;
    }
  }
  return a;
}

/**
 * dZ = dA ⊙ φ'(Z), elementwise. Softmax never reaches here — it is always fused
 * with categorical cross-entropy, which is why `act` is the elementwise type.
 */
export function applyActivationDerivative(
  act: ElementwiseActivation,
  z: Matrix,
  a: Matrix,
  dA: Matrix,
  out: Matrix | null = null,
): Matrix {
  const dZ = ensureShape(out, z.rows, z.cols);
  for (let i = 0; i < z.data.length; i++) {
    dZ.data[i] = dA.data[i]! * act.df(z.data[i]!, a.data[i]!);
  }
  return dZ;
}
