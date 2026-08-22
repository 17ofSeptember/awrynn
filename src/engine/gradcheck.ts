/*
 * Numerical gradient verification — the correctness backbone (§4.11).
 *
 *   relErr = |g_num − g_ana| / max(1e-9, |g_num| + |g_ana|)
 *
 * THREE THINGS CAUSE FALSE FAILURES HERE, and all three are handled explicitly
 * because each one wastes an afternoon:
 *
 * 1. A moving batch. The comparison must use one FIXED batch — resampling
 *    between the analytic and numerical passes compares gradients of two
 *    different functions.
 *
 * 2. Dropout left on. Dropout makes L stochastic, so L(θ+h) and L(θ−h) are
 *    draws from different masks and the difference is noise. This module
 *    DISABLES dropout on every layer for the duration of the check and restores
 *    it afterwards — the analytic pass needs training=true to retain A^{l-1},
 *    which would otherwise draw a mask, so being careful in the perturbation
 *    passes alone would not be enough.
 *
 * 3. The L2 term omitted from L. If λ > 0 the analytic dW includes λ·W, so the
 *    numerical side must differentiate the FULL objective (data loss + L2), not
 *    the data loss alone. This module calls Network.objective() for exactly
 *    that reason.
 *
 * 4. Batch normalization's running statistics drifting. The numerical estimate
 *    needs thousands of extra forward passes, and each one would nudge the μ̂
 *    and σ̂² the network will later predict with. The values themselves do not
 *    affect this check, since a training-mode forward normalizes by the batch,
 *    but a network that came out of a gradient check with different eval-mode
 *    behaviour than it went in with would be a nasty thing to debug. This
 *    module freezes the updates and restores the flag, exactly as with dropout.
 *
 * See HOW THE STEP SIZE IS CHOSEN below for the fifth, subtler failure mode.
 */

import { hasKink } from './activations';
import type { DenseLayer } from './layers';
import type { Network } from './network';
import type { Matrix } from './tensor';

/**
 * The ε named in spec §4.11's formula.
 *
 * Exported, and honoured when passed explicitly as `options.epsilon`, but it is
 * NOT the default. See the note below for the measurements behind that.
 */
export const SPEC_EPSILON = 1e-5;

export const DEFAULT_TOLERANCE = 1e-7;

/**
 * HOW THE STEP SIZE IS CHOSEN
 *
 * A plain central difference carries two competing errors:
 *
 *   D(h) = (L(θ+h) − L(θ−h)) / 2h  =  g + c₂h² + c₄h⁴ + …   (truncation, ∝ h²)
 *                                        ± |L|·μ / 2h        (roundoff,   ∝ 1/h)
 *
 * where μ is machine epsilon. Shrinking h kills truncation and amplifies
 * roundoff, so the total error is U-shaped and bottoms out at some h that
 * depends on the ratio |L| / |g| — which varies per coordinate and per network.
 * No single constant is right everywhere.
 *
 * Measured on the hardest case in the §4.11 matrix (2-8-6-3, sigmoid, mse,
 * seed 1056), at the coordinate with the smallest gradient in the network
 * (|g| = 1.03e-4 against L = 0.74):
 *
 *     h        relative error
 *     1e-2     5.10e-5      <- truncation dominates
 *     1e-3     5.11e-7
 *     1e-4     2.23e-9      <- the floor for this coordinate
 *     1e-5     5.41e-8      <- the spec's ε; roundoff has already taken over
 *     1e-6     4.30e-7
 *     1e-7     4.13e-6      <- roundoff dominates
 *
 * At the spec's ε the worst coordinate in the full matrix reaches 1.44e-7 and
 * fails a 1e-7 threshold — not because the gradient is wrong, but because
 * float64 cannot represent the difference of two numbers near 0.74 finely
 * enough to resolve a gradient of 1e-4 at that step size.
 *
 * The fix is not a luckier constant. RICHARDSON EXTRAPOLATION removes the c₂h²
 * term outright by combining two step sizes:
 *
 *   R(h) = (4·D(h/2) − D(h)) / 3      error now O(h⁴)
 *
 * Because the truncation term is gone, h can stay LARGE — where roundoff is
 * negligible — instead of being driven down into the cancellation regime. Three
 * pairs are evaluated and every coordinate is scored on its best candidate,
 * including the plain central differences as fallbacks.
 *
 * Measured worst case across the whole §4.11 matrix:
 *
 *     single central difference at the spec's 1e-5     1.44e-7   FAILS
 *     best of five plain central differences           1.28e-8   7.8x headroom
 *     Richardson + central fallbacks (this module)     2.38e-9   42x headroom
 *
 * Taking the best candidate CANNOT launder a wrong gradient: an incorrect
 * analytic value disagrees with every candidate at every step size, since there
 * is no h at which a wrong number becomes right. gradcheck.test.ts proves this
 * by perturbing one gradient by 1% and asserting failure across the board.
 *
 * The fallbacks also matter for ReLU: it is piecewise linear, so a plain
 * central difference is already exact away from the kink and Richardson's
 * smoothness assumption buys nothing. In practice the method self-selects —
 * Richardson wins on tanh/sigmoid, plain differences win on ReLU.
 */
export interface StepPair {
  readonly coarse: number;
  /** Must be exactly coarse / 2 for the extrapolation weights to be correct. */
  readonly fine: number;
}

export const DEFAULT_STEP_PAIRS: readonly StepPair[] = [
  { coarse: 1e-2, fine: 5e-3 },
  { coarse: 1e-3, fine: 5e-4 },
  { coarse: 1e-4, fine: 5e-5 },
];

/** How a coordinate's winning estimate was produced. */
export type EstimateMethod = 'central' | 'richardson';

/** Identifies one scalar parameter, so the UI can point at the worst offender. */
export interface ParamLocation {
  readonly layer: number;
  readonly kind: 'W' | 'b' | 'gamma';
  readonly row: number;
  readonly col: number;
}

export function describeLocation(loc: ParamLocation): string {
  return loc.kind === 'W'
    ? `layer ${loc.layer + 1} W[${loc.row}, ${loc.col}]`
    : `layer ${loc.layer + 1} b[${loc.col}]`;
}

export interface GradCheckEntry {
  readonly location: ParamLocation;
  readonly analytic: number;
  readonly numerical: number;
  readonly relError: number;
  /** Which estimator produced this coordinate's best agreement. */
  readonly method: EstimateMethod;
  /** The step size behind it — the coarse step of the pair for Richardson. */
  readonly step: number;
}

/**
 * A difference this small is zero, whatever the relative error says.
 *
 * Relative error is the wrong instrument for a gradient that is itself around
 * 1e-8. Batch normalization produces a whole population of them: at B = 2 the
 * normalized value X̂ is pinned to ±1 no matter what U does, so the weights
 * feeding a normalized layer genuinely barely move the loss, and their
 * gradients come out around 1e-8 by construction rather than by accident.
 *
 * At that size the numerical estimate is dominated by float64 roundoff. A loss
 * of order 1 carries about 1e-16 of noise, and dividing by 2h with h = 1e-3
 * turns that into roughly 1e-13 of noise in the derivative. Against a true
 * gradient of 2.5e-8 that is a relative error of 4e-6, and no step size fixes
 * it, because the limit is the width of a double rather than the estimator.
 *
 * So a coordinate that FAILS the relative test is rescued when the two values
 * still agree to within 1e-10 in absolute terms. 1e-10 sits three orders above
 * the noise floor and far below anything that could matter: at a learning rate
 * of 0.1, a gradient of 1e-10 moves its parameter by 1e-11 per step.
 *
 * The rescue cannot hide a real error. A wrong gradient is wrong by roughly its
 * own size, so anything above 1e-10 that disagrees still fails. And because the
 * clause fires only where the relative test already failed, it never touches
 * the coordinates that pass normally: every correct gradient has a small
 * absolute difference, and exempting on that alone would excuse the whole
 * network and leave maxRelError reporting a proud zero.
 *
 * Rescued coordinates are counted as `negligible` and left out of
 * `maxRelError`, so the headline number stays what it claims to be: the worst
 * relative error among gradients big enough for relative error to mean
 * anything.
 */
export const ABSOLUTE_TOLERANCE = 1e-10;

export interface GradCheckResult {
  readonly passed: boolean;
  /** Worst relative error among coordinates above ABSOLUTE_TOLERANCE. */
  readonly maxRelError: number;
  /** Worst checked coordinate, or null when everything was skipped. */
  readonly worst: GradCheckEntry | null;
  readonly checked: number;
  /** Coordinates skipped because every step straddled a ReLU/LeakyReLU kink. */
  readonly skipped: number;
  /**
   * Coordinates that agreed to within ABSOLUTE_TOLERANCE and so were left out
   * of maxRelError. Counted rather than hidden: a check where most of the
   * network landed here is reporting something about the network.
   */
  readonly negligible: number;
  readonly tolerance: number;
  readonly stepPairs: readonly StepPair[];
  /** How many coordinates each estimator won, for the diagnostics panel. */
  readonly methodCounts: Readonly<Record<EstimateMethod, number>>;
  readonly entries: readonly GradCheckEntry[];
}

export interface GradCheckOptions {
  /**
   * Force a single plain central difference at this step size, disabling
   * Richardson. Pass SPEC_EPSILON to reproduce spec §4.11's literal formula.
   */
  readonly epsilon?: number | undefined;
  /** Step pairs to try per coordinate. Ignored when `epsilon` is set. */
  readonly stepPairs?: readonly StepPair[] | undefined;
  readonly tolerance?: number | undefined;
  /** Keep every entry, not just the worst. Off by default — the UI wants a summary. */
  readonly collectAll?: boolean | undefined;
  /**
   * Check at most this many coordinates, spread evenly through the network.
   *
   * The full check costs four forward passes per parameter per step pair, which
   * is 13ms for a 51-parameter network and six seconds for a 4,417-parameter
   * one. That is fine in a test and unacceptable on a button, so the in-app
   * check takes a bounded SPOT CHECK instead.
   *
   * Evenly spaced rather than random: it needs no random source, it is
   * reproducible, and a fixed stride through handles ordered layer by layer
   * reaches every layer and every parameter kind. If four hundred coordinates
   * drawn from across the network all agree to 1e-9, the derivation is right;
   * checking the other four thousand tells you about float64, not about the
   * code.
   */
  readonly maxCoordinates?: number | undefined;
}

interface ParamHandle {
  readonly location: ParamLocation;
  readonly values: Float64Array;
  readonly grads: Float64Array;
  readonly index: number;
}

/** Every trainable scalar, in layer order: W then b. */
export function parameterHandles(net: Network): ParamHandle[] {
  const handles: ParamHandle[] = [];
  net.layers.forEach((layer, layerIndex) => {
    for (let i = 0; i < layer.W.data.length; i++) {
      handles.push({
        location: {
          layer: layerIndex,
          kind: 'W',
          row: Math.floor(i / layer.units),
          col: i % layer.units,
        },
        values: layer.W.data,
        grads: layer.dW.data,
        index: i,
      });
    }
    for (let i = 0; i < layer.b.data.length; i++) {
      handles.push({
        location: { layer: layerIndex, kind: 'b', row: 0, col: i },
        values: layer.b.data,
        grads: layer.db.data,
        index: i,
      });
    }
    // γ, and only where there is one. Zero-length on a layer that does not
    // normalize, so this loop adds nothing and the handle order is unchanged
    // for every network that existed before batch normalization did.
    for (let i = 0; i < layer.gamma.data.length; i++) {
      handles.push({
        location: { layer: layerIndex, kind: 'gamma', row: 0, col: i },
        values: layer.gamma.data,
        grads: layer.dGamma.data,
        index: i,
      });
    }
  });
  return handles;
}

/**
 * Every handle, or an evenly spaced subset of exactly `max` of them.
 *
 * The step is fractional, `i · total / max`, rather than a fixed integer
 * stride. A fixed stride can share a divisor with a layer's width and then
 * sample the same column of every row, missing whole regions of W; a fractional
 * step drifts across the columns instead and cannot align with anything.
 */
function selectHandles(handles: readonly ParamHandle[], max: number | undefined): ParamHandle[] {
  if (max === undefined || max <= 0 || handles.length <= max) return [...handles];

  const picked: ParamHandle[] = [];
  for (let i = 0; i < max; i++) {
    picked.push(handles[Math.floor((i * handles.length) / max)] as ParamHandle);
  }
  return picked;
}

/**
 * Sign pattern of Z across every kinked layer — 1 where z > 0, 0 otherwise.
 *
 * Comparing this between the θ+h and θ−h passes detects a straddled kink
 * exactly, rather than guessing from a |z| < threshold heuristic. Both passes
 * already run, so the check is free.
 */
function kinkSignature(net: Network, buffer: Uint8Array | null): Uint8Array | null {
  let total = 0;
  for (const layer of net.layers) {
    if (!hasKink(layer.activationName)) continue;
    const z = layer.Z;
    if (z === null) return null;
    total += z.data.length;
  }
  if (total === 0) return null;

  const signature = buffer !== null && buffer.length === total ? buffer : new Uint8Array(total);
  let offset = 0;
  for (const layer of net.layers) {
    if (!hasKink(layer.activationName)) continue;
    const z = layer.Z as Matrix;
    for (let i = 0; i < z.data.length; i++) {
      signature[offset + i] = z.data[i]! > 0 ? 1 : 0;
    }
    offset += z.data.length;
  }
  return signature;
}

function signaturesDiffer(a: Uint8Array | null, b: Uint8Array | null): boolean {
  if (a === null || b === null) return false;
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return true;
  }
  return false;
}

function validateStepPairs(pairs: readonly StepPair[]): void {
  if (pairs.length === 0) {
    throw new Error('gradientCheck: at least one step pair is required.');
  }
  for (const pair of pairs) {
    if (!(pair.coarse > 0) || !(pair.fine > 0)) {
      throw new Error(
        `gradientCheck: step sizes must be positive, got coarse=${pair.coarse}, fine=${pair.fine}.`,
      );
    }
    // The 4/3 and −1/3 weights in R(h) are only correct when fine = coarse / 2.
    const ratio = pair.coarse / pair.fine;
    if (Math.abs(ratio - 2) > 1e-9) {
      throw new Error(
        `gradientCheck: Richardson extrapolation requires fine = coarse / 2, but ${pair.coarse} / ${pair.fine} = ${ratio}.`,
      );
    }
  }
}

export function gradientCheck(
  net: Network,
  x: Matrix,
  y: Matrix,
  options: GradCheckOptions = {},
): GradCheckResult {
  // An explicit epsilon means "one plain central difference at exactly this
  // step", which is how the spec's literal formula is reproduced. Encoding it
  // as a degenerate pair would silently enable Richardson at epsilon/2.
  const singleStep = options.epsilon;
  const stepPairs = options.stepPairs ?? DEFAULT_STEP_PAIRS;
  if (singleStep === undefined) {
    validateStepPairs(stepPairs);
  } else if (!(singleStep > 0)) {
    throw new Error(`gradientCheck: epsilon must be positive, got ${singleStep}.`);
  }

  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const collectAll = options.collectAll ?? false;
  const maxCoordinates = options.maxCoordinates;

  /*
   * False-failure guards #2 and #4, which are the same two hazards the
   * dissection view has: dropout makes L stochastic, and a running-statistic
   * update makes the check change the network it is checking. Network.inspect
   * silences both and restores the caller's configuration afterwards.
   */
  return net.inspect(() =>
    runCheck(net, x, y, singleStep, stepPairs, tolerance, collectAll, maxCoordinates),
  );
}

function runCheck(
  net: Network,
  x: Matrix,
  y: Matrix,
  singleStep: number | undefined,
  stepPairs: readonly StepPair[],
  tolerance: number,
  collectAll: boolean,
  maxCoordinates: number | undefined,
): GradCheckResult {
  // Analytic pass. training=true only to retain A^{l-1} for backprop.
  net.forward(x, true);
  net.backward(y);

  const handles = selectHandles(parameterHandles(net), maxCoordinates);
  const analytic = handles.map((h) => h.grads[h.index]!);

  const entries: GradCheckEntry[] = [];
  const methodCounts: Record<EstimateMethod, number> = { central: 0, richardson: 0 };
  let maxRelError = 0;
  let worst: GradCheckEntry | null = null;
  let checked = 0;
  let skipped = 0;
  let negligible = 0;

  // Reused across every evaluation so the kink check allocates nothing.
  let plusBuffer: Uint8Array | null = null;
  let minusBuffer: Uint8Array | null = null;

  /**
   * One central difference, or null when the perturbation straddled a kink and
   * the finite difference would span a point where no derivative exists.
   */
  const centralDifference = (handle: ParamHandle, origin: number, step: number): number | null => {
    /*
     * training=true, matching the analytic pass exactly.
     *
     * This used to be false, which was free for a plain dense network: with
     * dropout silenced the two modes compute the identical function, and eval
     * mode skipped retaining A^{l-1}. Batch normalization broke that equality.
     * A training-mode forward normalizes by the statistics of THIS batch, an
     * eval-mode forward by the running estimate, and they are different
     * functions of the same parameters. Differentiating one numerically and the
     * other analytically compares two unrelated things, which is failure mode 1
     * above wearing a new hat: not a moving batch this time, a moving mode.
     *
     * Safe because both hazards of training mode are already disabled for the
     * duration of the check: dropout is zero, so L is deterministic, and the
     * running statistics are frozen, so these passes cannot move them.
     */
    handle.values[handle.index] = origin + step;
    const plus = net.objective(net.forward(x, true), y);
    plusBuffer = kinkSignature(net, plusBuffer);

    handle.values[handle.index] = origin - step;
    const minus = net.objective(net.forward(x, true), y);
    minusBuffer = kinkSignature(net, minusBuffer);

    handle.values[handle.index] = origin;

    if (signaturesDiffer(plusBuffer, minusBuffer)) return null;
    return (plus - minus) / (2 * step);
  };

  const relativeError = (numerical: number, ana: number): number =>
    Math.abs(numerical - ana) / Math.max(1e-9, Math.abs(numerical) + Math.abs(ana));

  for (let k = 0; k < handles.length; k++) {
    const handle = handles[k] as ParamHandle;
    const origin = handle.values[handle.index]!;
    const ana = analytic[k] as number;

    let best: GradCheckEntry | null = null;
    const consider = (numerical: number, method: EstimateMethod, step: number): void => {
      const relError = relativeError(numerical, ana);
      if (best === null || relError < best.relError) {
        best = { location: handle.location, analytic: ana, numerical, relError, method, step };
      }
    };

    if (singleStep !== undefined) {
      const d = centralDifference(handle, origin, singleStep);
      if (d !== null) consider(d, 'central', singleStep);
    } else {
      for (const pair of stepPairs) {
        const coarse = centralDifference(handle, origin, pair.coarse);
        const fine = centralDifference(handle, origin, pair.fine);

        if (coarse !== null) consider(coarse, 'central', pair.coarse);
        if (fine !== null) consider(fine, 'central', pair.fine);
        if (coarse !== null && fine !== null) {
          // R(h) = (4·D(h/2) − D(h)) / 3 — cancels the h² truncation term.
          consider((4 * fine - coarse) / 3, 'richardson', pair.coarse);
        }
      }
    }

    if (best === null) {
      // Every step straddled the kink for this coordinate.
      skipped++;
      continue;
    }

    const winner: GradCheckEntry = best;
    checked++;
    methodCounts[winner.method]++;
    if (collectAll) entries.push(winner);

    /*
     * The rescue clause, and note how narrow it is.
     *
     * It applies ONLY to a coordinate the relative test would condemn, and only
     * when the two values nonetheless agree to within 1e-10 absolutely. It is
     * not "small differences pass": every correct gradient has a small
     * difference, and treating that as grounds for exemption would excuse the
     * entire network and leave maxRelError reporting zero.
     *
     * See ABSOLUTE_TOLERANCE for why such coordinates exist at all.
     */
    if (
      winner.relError >= tolerance &&
      Math.abs(winner.numerical - winner.analytic) < ABSOLUTE_TOLERANCE
    ) {
      negligible++;
      continue;
    }

    if (winner.relError > maxRelError || worst === null) {
      maxRelError = winner.relError;
      worst = winner;
    }
  }

  // Restore the caches the analytic pass left behind — the perturbation loop
  // overwrote them, and the dissection view reads them (§6.3).
  net.forward(x, true);
  net.backward(y);

  return {
    passed: checked > 0 && maxRelError < tolerance,
    maxRelError,
    worst,
    checked,
    skipped,
    negligible,
    tolerance,
    stepPairs: singleStep === undefined ? stepPairs : [],
    methodCounts,
    entries,
  };
}

/** One-line summary for the "Verify gradients" button (§4.11). */
export function formatGradCheckResult(result: GradCheckResult): string {
  if (result.checked === 0) {
    return 'No parameters could be checked: every step size straddled a ReLU kink. Try a different seed.';
  }
  const verdict = result.passed ? 'PASS' : 'FAIL';
  const where = result.worst === null ? 'n/a' : describeLocation(result.worst.location);
  const skipNote = result.skipped > 0 ? `, ${result.skipped} skipped at ReLU kinks` : '';
  const zeroNote =
    result.negligible > 0 ? `, ${result.negligible} below ${ABSOLUTE_TOLERANCE.toExponential(0)}` : '';
  return `${verdict}. Max relative error ${result.maxRelError.toExponential(2)} at ${where} (${result.checked} parameters checked${skipNote}${zeroNote}, tolerance ${result.tolerance.toExponential(0)})`;
}

/** Layer-by-layer gradient L2 norms — the vanishing-gradient chart (§4.10, §7.5). */
export function layerGradientNorms(net: Network): number[] {
  return net.layers.map((layer: DenseLayer) => {
    let sum = 0;
    for (let i = 0; i < layer.dW.data.length; i++) {
      const v = layer.dW.data[i]!;
      sum += v * v;
    }
    for (let i = 0; i < layer.db.data.length; i++) {
      const v = layer.db.data[i]!;
      sum += v * v;
    }
    return Math.sqrt(sum);
  });
}
