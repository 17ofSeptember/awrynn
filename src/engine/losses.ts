/*
 * Loss functions.
 *
 * Spec §4.5. Two things here are easy to get subtly wrong, so both are spelled
 * out:
 *
 * 1. `loss()` returns the BATCH MEAN L = (1/B) Σ_b ℓ_b, because that is the
 *    number learners see. But `dA()` returns the UNAVERAGED per-sample
 *    gradients ∂ℓ_b/∂â, shape [B, n_L]. Per §4.3 the division by B happens
 *    exactly once, later, when the parameter gradients are formed. Averaging
 *    here as well would make every gradient off by a factor of B.
 *
 * 2. Every logarithm is clamped: log(max(x, 1e-12)). An unclamped log(0) is
 *    -Infinity, which turns the reported loss into NaN and destroys the loss
 *    curve even though the gradients may be perfectly fine.
 */

import type { Matrix } from './tensor';
import { ensureShape } from './tensor';

export type LossName = 'mse' | 'bce' | 'cce';

export const LOSS_NAMES: readonly LossName[] = ['mse', 'bce', 'cce'];

/** Spec §4.5: "Clamp inside every logarithm". */
export const LOG_EPSILON = 1e-12;

function safeLog(x: number): number {
  return Math.log(Math.max(x, LOG_EPSILON));
}

export interface Loss {
  readonly name: LossName;
  /** Batch mean L = (1/B) Σ_b ℓ_b. */
  loss(yHat: Matrix, y: Matrix): number;
  /**
   * ∂ℓ/∂â, per sample and UNAVERAGED — shape [B, n_L].
   *
   * For bce and cce the network never actually calls this: those pair with
   * sigmoid and softmax respectively and take the fused dZ = Ŷ - Y path
   * (§4.3). It is implemented anyway so the test suite can verify that the
   * fused form really equals dA ⊙ φ'(z) instead of taking the cancellation on
   * trust, and so gradcheck can exercise the unfused route.
   */
  dA(yHat: Matrix, y: Matrix, out?: Matrix | null): Matrix;
}

function assertMatching(name: string, yHat: Matrix, y: Matrix): void {
  if (yHat.rows !== y.rows || yHat.cols !== y.cols) {
    throw new Error(
      `losses.${name}: prediction [${yHat.rows}, ${yHat.cols}] and target [${y.rows}, ${y.cols}] must have the same shape.`,
    );
  }
}

/* ------------------------------------------------------------------ *
 * Mean squared error
 *   ℓ_b = ½ Σ_j (â_j − y_j)²      ∂ℓ/∂â_j = â_j − y_j
 *
 * The ½ is what makes the derivative come out clean; without it every gradient
 * carries a stray 2 that gets absorbed into the learning rate and makes the
 * on-screen arithmetic disagree with the textbook.
 * ------------------------------------------------------------------ */
const mse: Loss = {
  name: 'mse',
  loss(yHat, y) {
    assertMatching('mse', yHat, y);
    let total = 0;
    for (let i = 0; i < yHat.data.length; i++) {
      const d = yHat.data[i]! - y.data[i]!;
      total += 0.5 * d * d;
    }
    return total / yHat.rows;
  },
  dA(yHat, y, out = null) {
    assertMatching('mse', yHat, y);
    const g = ensureShape(out, yHat.rows, yHat.cols);
    for (let i = 0; i < yHat.data.length; i++) g.data[i] = yHat.data[i]! - y.data[i]!;
    return g;
  },
};

/* ------------------------------------------------------------------ *
 * Binary cross-entropy
 *   ℓ_b = −[y log â + (1−y) log(1−â)]
 *   ∂ℓ/∂â = (â − y) / (â(1−â))          <- unfused; blows up as â -> 0 or 1
 *   fused with sigmoid: dZ = â − y      <- the â(1−â) cancels σ'(z) exactly
 * ------------------------------------------------------------------ */
const bce: Loss = {
  name: 'bce',
  loss(yHat, y) {
    assertMatching('bce', yHat, y);
    let total = 0;
    for (let i = 0; i < yHat.data.length; i++) {
      const p = yHat.data[i]!;
      const t = y.data[i]!;
      total -= t * safeLog(p) + (1 - t) * safeLog(1 - p);
    }
    return total / yHat.rows;
  },
  dA(yHat, y, out = null) {
    assertMatching('bce', yHat, y);
    const g = ensureShape(out, yHat.rows, yHat.cols);
    for (let i = 0; i < yHat.data.length; i++) {
      const p = yHat.data[i]!;
      const t = y.data[i]!;
      // Clamped denominator: this is exactly the quantity the fused path exists
      // to avoid dividing by.
      const denom = Math.max(p * (1 - p), LOG_EPSILON);
      g.data[i] = (p - t) / denom;
    }
    return g;
  },
};

/* ------------------------------------------------------------------ *
 * Categorical cross-entropy
 *   ℓ_b = −Σ_k y_k log â_k               (y one-hot)
 *   ∂ℓ/∂â_k = −y_k / â_k                 <- unfused
 *   fused with softmax: dZ = â − y
 * ------------------------------------------------------------------ */
const cce: Loss = {
  name: 'cce',
  loss(yHat, y) {
    assertMatching('cce', yHat, y);
    let total = 0;
    for (let i = 0; i < yHat.data.length; i++) {
      const t = y.data[i]!;
      if (t !== 0) total -= t * safeLog(yHat.data[i]!);
    }
    return total / yHat.rows;
  },
  dA(yHat, y, out = null) {
    assertMatching('cce', yHat, y);
    const g = ensureShape(out, yHat.rows, yHat.cols);
    for (let i = 0; i < yHat.data.length; i++) {
      const t = y.data[i]!;
      g.data[i] = t === 0 ? 0 : -t / Math.max(yHat.data[i]!, LOG_EPSILON);
    }
    return g;
  },
};

export function getLoss(name: LossName): Loss {
  switch (name) {
    case 'mse':
      return mse;
    case 'bce':
      return bce;
    case 'cce':
      return cce;
  }
}

/**
 * dZ^L = Ŷ − Y, the fused output-layer gradient (§4.3).
 *
 * Why this is legal, for both pairings:
 *
 *   sigmoid + bce:  ∂ℓ/∂â = (â−y)/(â(1−â)) and σ'(z) = â(1−â),
 *                   so dZ = ∂ℓ/∂â · σ'(z) = (â−y)/(â(1−â)) · â(1−â) = â−y.
 *
 *   softmax + cce:  ∂ℓ/∂â_k = −y_k/â_k and ∂â_k/∂z_j = â_k(δ_kj − â_j),
 *                   so dZ_j = Σ_k (−y_k/â_k)·â_k(δ_kj − â_j)
 *                           = −y_j + â_j Σ_k y_k = â_j − y_j   (Σ_k y_k = 1).
 *
 * Both cancellations remove a division by a quantity that goes to zero exactly
 * when the network is confident, which is why the fused path is the stable one
 * as well as the fast one. src/engine/__tests__/losses.test.ts checks the
 * identity numerically rather than trusting this comment.
 */
export function fusedOutputGradient(yHat: Matrix, y: Matrix, out: Matrix | null = null): Matrix {
  assertMatching('fusedOutputGradient', yHat, y);
  const dZ = ensureShape(out, yHat.rows, yHat.cols);
  for (let i = 0; i < yHat.data.length; i++) dZ.data[i] = yHat.data[i]! - y.data[i]!;
  return dZ;
}
