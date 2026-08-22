/*
 * Optimizers.
 *
 * Spec §4.8. All operate on flat parameter/gradient arrays with per-parameter
 * state — Network.params and Network.grads are exactly that, contiguous and
 * aliased to every layer's W and b, so there is no gather or scatter per step.
 *
 * `t` is a global step counter incremented ONCE per update, before the
 * parameter loop, because Adam's bias correction reads it and expects t = 1 on
 * the first update.
 *
 * The learning rate is passed in per step rather than stored, because LR
 * schedules (§4.9) change it every step and the optimizer should not have to
 * know which schedule is driving it.
 */

export type OptimizerName = 'sgd' | 'momentum' | 'nesterov' | 'rmsprop' | 'adam' | 'adamw';

export const OPTIMIZER_NAMES: readonly OptimizerName[] = [
  'sgd',
  'momentum',
  'nesterov',
  'rmsprop',
  'adam',
  'adamw',
];

export const OPTIMIZER_DEFAULTS = {
  /** μ for momentum and Nesterov. */
  momentum: 0.9,
  /** ρ for RMSProp. */
  rho: 0.9,
  /** β₁ for Adam and AdamW. */
  beta1: 0.9,
  /** β₂ for Adam and AdamW. */
  beta2: 0.999,
  /** ε in the denominators. Guards against division by zero on a dead unit. */
  epsilon: 1e-8,
  /** λ for AdamW's decoupled decay. */
  weightDecay: 0.01,
} as const;

export interface OptimizerConfig {
  readonly name: OptimizerName;
  readonly momentum?: number | undefined;
  readonly rho?: number | undefined;
  readonly beta1?: number | undefined;
  readonly beta2?: number | undefined;
  readonly epsilon?: number | undefined;
  readonly weightDecay?: number | undefined;
}

export interface Optimizer {
  readonly name: OptimizerName;
  /** The global step counter `t`, incremented once per update. */
  readonly stepCount: number;
  /**
   * Apply one update in place.
   *
   * `skip` marks coordinates to leave completely untouched — frozen parameters
   * (§6.5). Zeroing their gradients instead would NOT be equivalent: a
   * momentum or Adam state still carries a frozen parameter forward on
   * accumulated velocity, so it would drift rather than hold.
   */
  step(
    params: Float64Array,
    grads: Float64Array,
    learningRate: number,
    skip?: Uint8Array | null,
  ): void;
  /** Clear all accumulated state and reset t to 0. */
  reset(): void;
  /** Human-readable hyperparameters, for the UI. */
  describe(): string;
}

function assertSameLength(params: Float64Array, grads: Float64Array): void {
  if (params.length !== grads.length) {
    throw new Error(
      `Optimizer.step: parameters (${params.length}) and gradients (${grads.length}) must have the same length.`,
    );
  }
}

abstract class BaseOptimizer implements Optimizer {
  abstract readonly name: OptimizerName;
  protected t = 0;

  get stepCount(): number {
    return this.t;
  }

  step(
    params: Float64Array,
    grads: Float64Array,
    learningRate: number,
    skip: Uint8Array | null = null,
  ): void {
    assertSameLength(params, grads);
    this.ensureState(params.length);
    // Incremented once per update, not once per parameter.
    this.t++;
    this.update(params, grads, learningRate, skip);
  }

  reset(): void {
    this.t = 0;
    this.clearState();
  }

  abstract describe(): string;
  protected abstract ensureState(size: number): void;
  protected abstract clearState(): void;
  protected abstract update(
    params: Float64Array,
    grads: Float64Array,
    learningRate: number,
    skip: Uint8Array | null,
  ): void;
}

/** θ ← θ − η·g */
class Sgd extends BaseOptimizer {
  readonly name = 'sgd' as const;

  protected ensureState(): void {
    /* stateless */
  }
  protected clearState(): void {
    /* stateless */
  }

  protected update(
    params: Float64Array,
    grads: Float64Array,
    lr: number,
    skip: Uint8Array | null,
  ): void {
    for (let i = 0; i < params.length; i++) {
      if (skip !== null && skip[i] === 1) continue;
      params[i] = params[i]! - lr * grads[i]!;
    }
  }

  describe(): string {
    return 'SGD';
  }
}

/**
 * Momentum, in the accumulate-then-scale form:
 *
 *   v ← μ·v + g
 *   θ ← θ − η·v
 *
 * Spec §4.8 pins this variant explicitly. The alternative `v ← μv − ηg;
 * θ ← θ + v` folds η into the velocity, which rescales the effective learning
 * rate by 1/(1−μ) and makes an LR that works for SGD diverge here. Lesson 12's
 * optimizer race compares these at a shared LR, so the distinction is visible
 * in the product, not just in a comment.
 */
class Momentum extends BaseOptimizer {
  readonly name: OptimizerName;
  private velocity: Float64Array | null = null;

  constructor(
    private readonly mu: number,
    private readonly nesterov: boolean,
  ) {
    super();
    this.name = nesterov ? 'nesterov' : 'momentum';
  }

  protected ensureState(size: number): void {
    if (this.velocity === null || this.velocity.length !== size) {
      this.velocity = new Float64Array(size);
    }
  }
  protected clearState(): void {
    this.velocity = null;
  }

  protected update(
    params: Float64Array,
    grads: Float64Array,
    lr: number,
    skip: Uint8Array | null,
  ): void {
    const v = this.velocity as Float64Array;
    for (let i = 0; i < params.length; i++) {
      if (skip !== null && skip[i] === 1) continue;
      const g = grads[i]!;
      const vi = this.mu * v[i]! + g;
      v[i] = vi;
      // Nesterov looks ahead: the step uses the gradient plus the velocity it
      // is about to acquire, rather than the velocity it already had.
      params[i] = params[i]! - lr * (this.nesterov ? g + this.mu * vi : vi);
    }
  }

  describe(): string {
    return `${this.nesterov ? 'Nesterov' : 'Momentum'} (μ = ${this.mu})`;
  }
}

/**
 * RMSProp:
 *   s ← ρ·s + (1−ρ)·g²
 *   θ ← θ − η·g / (√s + ε)
 */
class RmsProp extends BaseOptimizer {
  readonly name = 'rmsprop' as const;
  private squared: Float64Array | null = null;

  constructor(
    private readonly rho: number,
    private readonly epsilon: number,
  ) {
    super();
  }

  protected ensureState(size: number): void {
    if (this.squared === null || this.squared.length !== size) {
      this.squared = new Float64Array(size);
    }
  }
  protected clearState(): void {
    this.squared = null;
  }

  protected update(
    params: Float64Array,
    grads: Float64Array,
    lr: number,
    skip: Uint8Array | null,
  ): void {
    const s = this.squared as Float64Array;
    for (let i = 0; i < params.length; i++) {
      if (skip !== null && skip[i] === 1) continue;
      const g = grads[i]!;
      const si = this.rho * s[i]! + (1 - this.rho) * g * g;
      s[i] = si;
      params[i] = params[i]! - (lr * g) / (Math.sqrt(si) + this.epsilon);
    }
  }

  describe(): string {
    return `RMSProp (ρ = ${this.rho}, ε = ${this.epsilon})`;
  }
}

/**
 * Adam, and AdamW with decoupled weight decay.
 *
 *   m ← β₁m + (1−β₁)g
 *   v ← β₂v + (1−β₂)g²
 *   m̂ = m/(1−β₁ᵗ)      v̂ = v/(1−β₂ᵗ)
 *   θ ← θ − η·m̂/(√v̂ + ε)                          [Adam]
 *   θ ← θ − η·m̂/(√v̂ + ε) − η·λ·θ                  [AdamW]
 *
 * BIAS CORRECTION IS NOT OPTIONAL. m and v start at zero, so without the
 * 1/(1−βᵗ) factors the first steps are biased toward zero by roughly (1−β₁) and
 * (1−β₂) — at t = 1 the uncorrected update is about 10x too small in m and
 * 1000x too small in v, which is a real bug rather than a slow start. Lesson 12
 * demonstrates the difference, and optimizers.test.ts pins t = 1 by hand.
 */
class Adam extends BaseOptimizer {
  readonly name: OptimizerName;
  private m: Float64Array | null = null;
  private v: Float64Array | null = null;

  constructor(
    private readonly beta1: number,
    private readonly beta2: number,
    private readonly epsilon: number,
    private readonly weightDecay: number,
  ) {
    super();
    this.name = weightDecay > 0 ? 'adamw' : 'adam';
  }

  protected ensureState(size: number): void {
    if (this.m === null || this.m.length !== size) {
      this.m = new Float64Array(size);
      this.v = new Float64Array(size);
    }
  }
  protected clearState(): void {
    this.m = null;
    this.v = null;
  }

  protected update(
    params: Float64Array,
    grads: Float64Array,
    lr: number,
    skip: Uint8Array | null,
  ): void {
    const m = this.m as Float64Array;
    const v = this.v as Float64Array;
    // Computed once per step, not once per parameter: t is global.
    const correction1 = 1 - Math.pow(this.beta1, this.t);
    const correction2 = 1 - Math.pow(this.beta2, this.t);

    for (let i = 0; i < params.length; i++) {
      if (skip !== null && skip[i] === 1) continue;
      const g = grads[i]!;
      const mi = this.beta1 * m[i]! + (1 - this.beta1) * g;
      const vi = this.beta2 * v[i]! + (1 - this.beta2) * g * g;
      m[i] = mi;
      v[i] = vi;

      const mHat = mi / correction1;
      const vHat = vi / correction2;
      let next = params[i]! - (lr * mHat) / (Math.sqrt(vHat) + this.epsilon);
      if (this.weightDecay > 0) {
        // Decoupled: applied to θ directly, NOT folded into g, so it is not
        // scaled by the adaptive 1/√v̂ term. That decoupling is the whole point
        // of AdamW — L2-in-the-gradient behaves differently under Adam.
        next -= lr * this.weightDecay * params[i]!;
      }
      params[i] = next;
    }
  }

  describe(): string {
    const base = `β₁ = ${this.beta1}, β₂ = ${this.beta2}, ε = ${this.epsilon}`;
    return this.weightDecay > 0 ? `AdamW (${base}, λ = ${this.weightDecay})` : `Adam (${base})`;
  }
}

export function createOptimizer(config: OptimizerConfig): Optimizer {
  const mu = config.momentum ?? OPTIMIZER_DEFAULTS.momentum;
  const rho = config.rho ?? OPTIMIZER_DEFAULTS.rho;
  const beta1 = config.beta1 ?? OPTIMIZER_DEFAULTS.beta1;
  const beta2 = config.beta2 ?? OPTIMIZER_DEFAULTS.beta2;
  const epsilon = config.epsilon ?? OPTIMIZER_DEFAULTS.epsilon;

  switch (config.name) {
    case 'sgd':
      return new Sgd();
    case 'momentum':
      return new Momentum(mu, false);
    case 'nesterov':
      return new Momentum(mu, true);
    case 'rmsprop':
      return new RmsProp(rho, epsilon);
    case 'adam':
      // Explicit 0: an Adam that quietly decayed weights would not be Adam.
      return new Adam(beta1, beta2, epsilon, 0);
    case 'adamw':
      return new Adam(
        beta1,
        beta2,
        epsilon,
        config.weightDecay ?? OPTIMIZER_DEFAULTS.weightDecay,
      );
  }
}

/**
 * Clip gradients by global L2 norm (§4.9): if ‖g‖ > c, scale ALL gradients by
 * c/‖g‖. Scaling globally rather than per-parameter preserves the gradient's
 * direction; clipping each coordinate independently would bend it.
 *
 * Returns the norm before clipping, so the UI can chart it.
 */
export function clipGradientsByNorm(grads: Float64Array, maxNorm: number): number {
  let sum = 0;
  for (let i = 0; i < grads.length; i++) {
    const v = grads[i]!;
    sum += v * v;
  }
  const norm = Math.sqrt(sum);
  if (maxNorm > 0 && norm > maxNorm && Number.isFinite(norm)) {
    const scale = maxNorm / norm;
    for (let i = 0; i < grads.length; i++) grads[i] = grads[i]! * scale;
  }
  return norm;
}
