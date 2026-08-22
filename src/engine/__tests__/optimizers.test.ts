import { describe, expect, it } from 'vitest';
import { clipGradientsByNorm, createOptimizer, OPTIMIZER_DEFAULTS } from '../optimizers';
import type { OptimizerConfig } from '../optimizers';

/*
 * Spec §10: "three manual steps of each, hand-computed, including Adam's bias
 * correction at t = 1."
 *
 * Every expectation below is worked out longhand in the comments from a fixed
 * gradient sequence, so a regression points at which term drifted rather than
 * just reporting a number mismatch.
 */

/** A fixed, deliberately asymmetric gradient sequence. */
const G: readonly number[] = [0.1, -0.2, 0.3];
const LR = 0.1;

function runSteps(config: OptimizerConfig, start: number, grads: readonly number[]): number[] {
  const opt = createOptimizer(config);
  const params = new Float64Array([start]);
  const out: number[] = [];
  for (const g of grads) {
    opt.step(params, new Float64Array([g]), LR);
    out.push(params[0] as number);
  }
  return out;
}

describe('SGD', () => {
  it('θ ← θ − η·g for three steps', () => {
    // 1.0 − 0.1(0.1)  = 0.99
    // 0.99 − 0.1(−0.2) = 1.01
    // 1.01 − 0.1(0.3)  = 0.98
    const steps = runSteps({ name: 'sgd' }, 1, G);
    expect(steps[0]!).toBeCloseTo(0.99, 12);
    expect(steps[1]!).toBeCloseTo(1.01, 12);
    expect(steps[2]!).toBeCloseTo(0.98, 12);
  });

  it('is stateless — reset changes nothing about its trajectory', () => {
    const a = runSteps({ name: 'sgd' }, 1, G);
    const b = runSteps({ name: 'sgd' }, 1, G);
    expect(a).toEqual(b);
  });
});

describe('Momentum — v ← μv + g ; θ ← θ − ηv', () => {
  it('accumulates velocity across three steps', () => {
    const mu = OPTIMIZER_DEFAULTS.momentum; // 0.9
    // v1 = 0.9(0)     + 0.1  =  0.1     θ = 1     − 0.1(0.1)     = 0.99
    // v2 = 0.9(0.1)   − 0.2  = -0.11    θ = 0.99  − 0.1(−0.11)   = 1.001
    // v3 = 0.9(−0.11) + 0.3  =  0.201   θ = 1.001 − 0.1(0.201)   = 0.9809
    expect(mu).toBe(0.9);
    const steps = runSteps({ name: 'momentum' }, 1, G);
    expect(steps[0]!).toBeCloseTo(0.99, 12);
    expect(steps[1]!).toBeCloseTo(1.001, 12);
    expect(steps[2]!).toBeCloseTo(0.9809, 12);
  });

  it('uses the accumulate-then-scale variant, not the LR-folding one (§4.8)', () => {
    // The alternative form v ← μv − ηg ; θ ← θ + v reaches a different place
    // after two steps. Pin ours so a "harmless" rewrite is caught.
    const steps = runSteps({ name: 'momentum' }, 1, [0.1, 0.1]);
    // v1 = 0.1              θ = 1    − 0.01  = 0.99
    // v2 = 0.9(0.1) + 0.1 = 0.19     θ = 0.99 − 0.019 = 0.971
    expect(steps[1]!).toBeCloseTo(0.971, 12);
    // The folded variant would give: v1 = −0.01, θ = 0.99;
    // v2 = 0.9(−0.01) − 0.01 = −0.019, θ = 0.971 — same here, so also check
    // that a larger μ diverges between the two forms.
    const heavy = runSteps({ name: 'momentum', momentum: 0.99 }, 1, [0.1, 0.1, 0.1]);
    // v1 = 0.1 ; v2 = 0.199 ; v3 = 0.29701
    // θ = 1 − 0.01 − 0.0199 − 0.029701 = 0.940399
    expect(heavy[2]!).toBeCloseTo(0.940399, 12);
  });
});

describe('Nesterov — v ← μv + g ; θ ← θ − η(g + μv)', () => {
  it('looks ahead by the velocity it is about to acquire', () => {
    // v1 = 0.1                    θ = 1      − 0.1(0.1  + 0.9(0.1))   = 0.981
    // v2 = 0.9(0.1) − 0.2 = −0.11  θ = 0.981  − 0.1(−0.299)            = 1.0109
    // v3 = 0.9(−0.11) + 0.3 = 0.201
    //                              θ = 1.0109 − 0.1(0.4809)            = 0.96281
    const steps = runSteps({ name: 'nesterov' }, 1, G);
    expect(steps[0]!).toBeCloseTo(0.981, 12);
    expect(steps[1]!).toBeCloseTo(1.0109, 12);
    expect(steps[2]!).toBeCloseTo(0.96281, 12);
  });

  it('differs from plain momentum', () => {
    const plain = runSteps({ name: 'momentum' }, 1, G);
    const nesterov = runSteps({ name: 'nesterov' }, 1, G);
    expect(nesterov[0]!).not.toBeCloseTo(plain[0]!, 6);
  });
});

describe('RMSProp — s ← ρs + (1−ρ)g² ; θ ← θ − ηg/(√s + ε)', () => {
  it('matches three hand-computed steps', () => {
    const rho = 0.9;
    const eps = OPTIMIZER_DEFAULTS.epsilon;
    let s = 0;
    let theta = 1;
    const expected: number[] = [];
    for (const g of G) {
      s = rho * s + (1 - rho) * g * g;
      theta = theta - (LR * g) / (Math.sqrt(s) + eps);
      expected.push(theta);
    }
    // s1 = 0.1(0.01) = 0.001        θ = 1 − 0.1(0.1)/(0.0316227…) ≈ 0.68377233
    expect(expected[0]!).toBeCloseTo(0.6837723339831305, 12);

    const steps = runSteps({ name: 'rmsprop' }, 1, G);
    for (let i = 0; i < 3; i++) expect(steps[i]!).toBeCloseTo(expected[i]!, 12);
  });

  it('normalises step size — a gradient 10⁶x larger takes the same size step', () => {
    // On the first step s = (1−ρ)g², so the update is η·g/(|g|√(1−ρ) + ε),
    // which tends to η/√(1−ρ) = 0.3162… regardless of |g|. This is RMSProp's
    // whole point and the reason it survives the badly-scaled data in §7.8.
    const asymptote = LR / Math.sqrt(0.1);
    for (const g of [0.001, 0.1, 1000]) {
      const step = Math.abs(1 - (runSteps({ name: 'rmsprop' }, 1, [g])[0] as number));
      // Within 0.1%: ε keeps the agreement from being exact, and it bites
      // hardest for the smallest gradient, where √s is closest to ε itself.
      expect(Math.abs(step - asymptote) / asymptote, `g = ${g}`).toBeLessThan(1e-3);
    }
  });
});

describe('Adam — bias correction is not optional (§4.8)', () => {
  it('at t = 1 the corrected update is exactly −η·sign(g), independent of |g|', () => {
    // m1 = (1−β₁)g = 0.1g          m̂ = m1/(1−β₁) = g
    // v1 = (1−β₂)g² = 0.001g²      v̂ = v1/(1−β₂) = g²
    // θ ← θ − η·g/(√(g²) + ε) = θ − η·g/|g| ≈ θ − η·sign(g)
    // This clean cancellation is the signature of correct bias correction and
    // is why t = 1 is the right place to pin it.
    const up = runSteps({ name: 'adam' }, 1, [0.5]);
    expect(up[0]!).toBeCloseTo(1 - LR, 7);

    const down = runSteps({ name: 'adam' }, 1, [-0.5]);
    expect(down[0]!).toBeCloseTo(1 + LR, 7);

    // Independent of magnitude: a gradient 1000x larger takes the same step.
    const large = runSteps({ name: 'adam' }, 1, [500]);
    expect(large[0]!).toBeCloseTo(1 - LR, 7);
  });

  it('matches three hand-computed steps', () => {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = OPTIMIZER_DEFAULTS.epsilon;
    let m = 0;
    let v = 0;
    let theta = 1;
    const expected: number[] = [];
    G.forEach((g, idx) => {
      const t = idx + 1;
      m = b1 * m + (1 - b1) * g;
      v = b2 * v + (1 - b2) * g * g;
      const mHat = m / (1 - Math.pow(b1, t));
      const vHat = v / (1 - Math.pow(b2, t));
      theta = theta - (LR * mHat) / (Math.sqrt(vHat) + eps);
      expected.push(theta);
    });
    const steps = runSteps({ name: 'adam' }, 1, G);
    for (let i = 0; i < 3; i++) expect(steps[i]!).toBeCloseTo(expected[i]!, 12);
  });

  it('WITHOUT bias correction the first step would be ~10x too small — the bug being prevented', () => {
    const g = 0.5;
    // Uncorrected: m = 0.05, v = 0.00025, step = η·0.05/√0.00025 = η·3.1623…
    // Corrected:   step = η·1. The uncorrected step is off by 3.16x here and
    // the discrepancy grows as β₂ᵗ approaches 1 for small t.
    const uncorrectedStep = (LR * 0.1 * g) / Math.sqrt(0.001 * g * g);
    const corrected = 1 - (runSteps({ name: 'adam' }, 1, [g])[0] as number);
    expect(corrected).toBeCloseTo(LR, 7);
    expect(uncorrectedStep).not.toBeCloseTo(corrected, 3);
  });

  it('increments t once per update, not once per parameter', () => {
    const opt = createOptimizer({ name: 'adam' });
    const params = new Float64Array([1, 2, 3, 4, 5]);
    const grads = new Float64Array([0.1, 0.1, 0.1, 0.1, 0.1]);
    expect(opt.stepCount).toBe(0);
    opt.step(params, grads, LR);
    expect(opt.stepCount).toBe(1);
    opt.step(params, grads, LR);
    expect(opt.stepCount).toBe(2);
  });
});

describe('AdamW — decoupled weight decay', () => {
  it('applies λ to θ directly, not through the adaptive denominator', () => {
    const lambda = 0.01;
    const g = 0.5;
    const theta0 = 2;
    // At t = 1 the Adam part is −η·sign(g) = −0.1; decay adds −η·λ·θ.
    const expected = theta0 - LR - LR * lambda * theta0;
    const steps = runSteps({ name: 'adamw', weightDecay: lambda }, theta0, [g]);
    expect(steps[0]!).toBeCloseTo(expected, 7);
  });

  it('decays even when the gradient is zero', () => {
    // The defining behaviour: plain Adam would not move a zero-gradient
    // parameter at all, AdamW shrinks it toward the origin.
    const decayed = runSteps({ name: 'adamw', weightDecay: 0.5 }, 3, [0]);
    expect(decayed[0]!).toBeCloseTo(3 - LR * 0.5 * 3, 10);

    const plain = runSteps({ name: 'adam' }, 3, [0]);
    expect(plain[0]!).toBeCloseTo(3, 10);
  });

  it('reports as adam when weight decay is zero', () => {
    expect(createOptimizer({ name: 'adam' }).name).toBe('adam');
    expect(createOptimizer({ name: 'adamw', weightDecay: 0 }).name).toBe('adam');
    expect(createOptimizer({ name: 'adamw' }).name).toBe('adamw');
  });
});

describe('shared optimizer behaviour', () => {
  const configs: OptimizerConfig[] = [
    { name: 'sgd' },
    { name: 'momentum' },
    { name: 'nesterov' },
    { name: 'rmsprop' },
    { name: 'adam' },
    { name: 'adamw' },
  ];

  it('every optimizer reduces a simple quadratic', () => {
    // f(θ) = ½θ², g = θ, minimum at 0. Any correct optimizer must approach it.
    for (const config of configs) {
      const opt = createOptimizer(config);
      const params = new Float64Array([1]);
      for (let i = 0; i < 200; i++) {
        opt.step(params, new Float64Array([params[0] as number]), 0.05);
      }
      expect(Math.abs(params[0] as number), config.name).toBeLessThan(0.2);
    }
  });

  it('reset() clears state and the step counter', () => {
    for (const config of configs) {
      const opt = createOptimizer(config);
      const params = new Float64Array([1]);
      opt.step(params, new Float64Array([0.5]), LR);
      opt.step(params, new Float64Array([0.5]), LR);
      expect(opt.stepCount).toBe(2);

      opt.reset();
      expect(opt.stepCount).toBe(0);

      // After reset the next step must match a fresh optimizer's first step.
      const fresh = createOptimizer(config);
      const a = new Float64Array([1]);
      const b = new Float64Array([1]);
      opt.step(a, new Float64Array([0.3]), LR);
      fresh.step(b, new Float64Array([0.3]), LR);
      expect(a[0], config.name).toBeCloseTo(b[0] as number, 15);
    }
  });

  it('leaves skipped (frozen) coordinates completely untouched', () => {
    for (const config of configs) {
      const opt = createOptimizer(config);
      const params = new Float64Array([1, 1, 1]);
      const skip = new Uint8Array([0, 1, 0]);
      for (let i = 0; i < 25; i++) {
        opt.step(params, new Float64Array([0.4, 0.4, 0.4]), LR, skip);
      }
      expect(params[1], `${config.name} frozen coordinate`).toBe(1);
      expect(params[0]).not.toBe(1);
      expect(params[0]).toBeCloseTo(params[2] as number, 15);
    }
  });

  it('zeroing a gradient is NOT equivalent to freezing, for stateful optimizers', () => {
    // The reason `skip` exists: a momentum or Adam state keeps carrying a
    // parameter forward on accumulated velocity even when its gradient is zero.
    const opt = createOptimizer({ name: 'momentum' });
    const params = new Float64Array([1]);
    opt.step(params, new Float64Array([1]), LR);
    const afterFirst = params[0] as number;
    opt.step(params, new Float64Array([0]), LR); // zero gradient
    expect(params[0]).not.toBeCloseTo(afterFirst, 6);
  });

  it('rejects mismatched parameter and gradient lengths', () => {
    const opt = createOptimizer({ name: 'sgd' });
    expect(() => opt.step(new Float64Array(3), new Float64Array(4), LR)).toThrowError(
      /\(3\).*\(4\).*same length/,
    );
  });

  it('describe() names the hyperparameters', () => {
    expect(createOptimizer({ name: 'sgd' }).describe()).toBe('SGD');
    expect(createOptimizer({ name: 'momentum' }).describe()).toContain('μ = 0.9');
    expect(createOptimizer({ name: 'adam' }).describe()).toContain('β₁ = 0.9');
    expect(createOptimizer({ name: 'adamw' }).describe()).toContain('λ =');
  });
});

describe('gradient clipping by global L2 norm (§4.9)', () => {
  it('leaves gradients alone when the norm is under the threshold', () => {
    const g = new Float64Array([0.3, 0.4]); // norm 0.5
    const norm = clipGradientsByNorm(g, 1);
    expect(norm).toBeCloseTo(0.5, 12);
    expect(Array.from(g)).toEqual([0.3, 0.4]);
  });

  it('scales all gradients by c/‖g‖ when over', () => {
    const g = new Float64Array([3, 4]); // norm 5
    const norm = clipGradientsByNorm(g, 1);
    expect(norm).toBeCloseTo(5, 12);
    // scale = 1/5
    expect(g[0]).toBeCloseTo(0.6, 12);
    expect(g[1]).toBeCloseTo(0.8, 12);
    expect(Math.hypot(g[0] as number, g[1] as number)).toBeCloseTo(1, 12);
  });

  it('preserves direction — clipping per-coordinate would not', () => {
    const g = new Float64Array([3, 4]);
    const beforeRatio = (g[0] as number) / (g[1] as number);
    clipGradientsByNorm(g, 1);
    expect((g[0] as number) / (g[1] as number)).toBeCloseTo(beforeRatio, 12);
  });

  it('does not touch a non-finite gradient set (divergence is reported, not hidden)', () => {
    const g = new Float64Array([Infinity, 1]);
    const norm = clipGradientsByNorm(g, 1);
    expect(norm).toBe(Infinity);
    expect(g[0]).toBe(Infinity);
  });
});
