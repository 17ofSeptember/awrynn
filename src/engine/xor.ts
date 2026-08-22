/*
 * The XOR challenge checker.
 *
 * Spec §6.5: "Manual mode: pause training entirely and build a solution by
 * hand. The XOR challenge checks your hand-built network against all four cases
 * and tells you which ones you've got."
 *
 * Pure engine code, so it runs headlessly and in tests: it takes a network and
 * reports, case by case, what the network actually outputs for each of the four
 * XOR corners. There is no partial credit and no fuzz — the four corners are
 * exact, and the network either separates them or it does not.
 */

import type { Network } from './network';
import { fromRows } from './tensor';
import { XOR_CORNERS } from './datasets/classification2d';

/**
 * The four corners, in a fixed order so the UI can label them stably.
 *
 * Derived from the dataset's own definition rather than restated here. When
 * these were two separate lists they drifted — the checker tested the unit
 * square while the data sat on the signed square — and the checker cheerfully
 * reported a network as solving XOR while the boundary showed it failing on
 * half the data.
 */
export const XOR_CASES: readonly { readonly inputs: readonly [number, number]; readonly target: number }[] =
  XOR_CORNERS.map((c) => ({ inputs: [c[0], c[1]] as readonly [number, number], target: c[2] }));

export interface XorCaseResult {
  readonly inputs: readonly [number, number];
  readonly target: number;
  /** The network's raw output. */
  readonly output: number;
  /** Thresholded at 0.5. */
  readonly predicted: number;
  readonly correct: boolean;
  /**
   * Distance from the 0.5 threshold.
   *
   * Reported because a network scraping past on 0.501 has not really solved
   * XOR, and a learner deserves to see that rather than a bare tick.
   */
  readonly margin: number;
}

export interface XorReport {
  readonly cases: readonly XorCaseResult[];
  readonly solvedCount: number;
  readonly solved: boolean;
  /** Smallest margin across the four, the honest measure of how solid it is. */
  readonly worstMargin: number;
  /** Null when the network is shaped for XOR; otherwise why it cannot be checked. */
  readonly problem: string | null;
}

/** How far from the threshold every case must sit to count as convincing. */
export const CONVINCING_MARGIN = 0.25;

export function checkXor(network: Network): XorReport {
  const empty: XorCaseResult[] = [];

  if (network.inputSize !== 2) {
    return {
      cases: empty,
      solvedCount: 0,
      solved: false,
      worstMargin: 0,
      problem: `XOR takes two inputs, but this network takes ${network.inputSize}.`,
    };
  }
  if (network.outputSize !== 1) {
    return {
      cases: empty,
      solvedCount: 0,
      solved: false,
      worstMargin: 0,
      problem: `XOR has one output, but this network has ${network.outputSize}.`,
    };
  }

  // All four corners as one batch: same arithmetic, one pass.
  const x = fromRows(XOR_CASES.map((c) => [...c.inputs]));
  const predictions = network.forward(x, false);

  const cases: XorCaseResult[] = XOR_CASES.map((testCase, i) => {
    const output = predictions.data[i] as number;
    const predicted = output >= 0.5 ? 1 : 0;
    return {
      inputs: testCase.inputs,
      target: testCase.target,
      output,
      predicted,
      correct: predicted === testCase.target,
      margin: Math.abs(output - 0.5),
    };
  });

  const solvedCount = cases.filter((c) => c.correct).length;
  return {
    cases,
    solvedCount,
    solved: solvedCount === XOR_CASES.length,
    worstMargin: Math.min(...cases.map((c) => c.margin)),
    problem: null,
  };
}

/**
 * A known-good hand-built solution, for the "show me" affordance.
 *
 * Two hidden tanh units acting as an OR gate and a NAND gate, combined by the
 * output: XOR is exactly `OR AND NAND`. Weights are large so tanh saturates and
 * the four corners land firmly rather than near the threshold.
 */
export interface HandBuiltSolution {
  readonly hidden: { readonly weights: readonly number[][]; readonly biases: readonly number[] };
  readonly output: { readonly weights: readonly number[][]; readonly biases: readonly number[] };
  readonly explanation: string;
}

export const XOR_SOLUTION: HandBuiltSolution = {
  hidden: {
    /*
     * Weights for inputs in {−1, +1}, matching XOR_CORNERS.
     *
     *   unit 0:  z = 6x₁ + 6x₂ + 6   →  −6 only at (−1, −1), so it is OR
     *   unit 1:  z = −6x₁ − 6x₂ + 6  →  −6 only at (+1, +1), so it is NAND
     *
     * tanh then saturates both to about ±1, which is what makes the output's
     * job a clean AND rather than a balancing act.
     */
    weights: [
      [6, -6],
      [6, -6],
    ],
    biases: [6, 6],
  },
  output: {
    // z = 6h₀ + 6h₁ − 6: positive only when BOTH are on, which is exactly XOR.
    weights: [[6], [6]],
    biases: [-6],
  },
  explanation:
    'One hidden unit computes OR, the other computes NAND, and the output takes their AND. XOR is exactly “either one, but not both”.',
};
