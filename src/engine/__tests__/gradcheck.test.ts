import { describe, expect, it } from 'vitest';
import type { ActivationName } from '../activations';
import {
  DEFAULT_STEP_PAIRS,
  formatGradCheckResult,
  gradientCheck,
  parameterHandles,
  SPEC_EPSILON,
} from '../gradcheck';
import type { DenseLayer, LayerSpec } from '../layers';
import type { LossName } from '../losses';
import { Network } from '../network';
import { createRng } from '../rng';
import { createMatrix, fromRows, rowView } from '../tensor';
import type { Matrix } from '../tensor';

/*
 * Spec §4.11 / §11 Phase 1 gate: the full matrix.
 *
 *   {linear, relu, leaky_relu, tanh, sigmoid}
 *     × {mse, bce, cce}
 *     × {with L2, without L2}
 *     × {2-1, 2-4-1, 2-8-6-3}
 *
 * skipping invalid combinations, at relErr < 1e-7.
 *
 * The hidden-layer activation is what varies; the OUTPUT activation is dictated
 * by the loss (§4.5): bce requires sigmoid on 1 unit, cce requires softmax on K
 * units, mse takes the swept activation on its output too. That means the sweep
 * covers both the fused output paths (bce/cce) and the general one (mse).
 */

const HIDDEN_ACTIVATIONS: readonly ActivationName[] = [
  'linear',
  'relu',
  'leaky_relu',
  'tanh',
  'sigmoid',
];
const LOSSES: readonly LossName[] = ['mse', 'bce', 'cce'];
const L2_VALUES: readonly number[] = [0, 0.05];

interface Architecture {
  readonly name: string;
  readonly hidden: readonly number[];
  readonly outputs: number;
}

const ARCHITECTURES: readonly Architecture[] = [
  { name: '2-1', hidden: [], outputs: 1 },
  { name: '2-4-1', hidden: [4], outputs: 1 },
  { name: '2-8-6-3', hidden: [8, 6], outputs: 3 },
];

const INPUT_SIZE = 2;
const BATCH = 6;

/**
 * A fixed batch, built once per case from a seeded stream (§4.11: the batch must
 * not move between the analytic and numerical passes).
 */
function fixedBatch(loss: LossName, outputs: number, seed: number): { x: Matrix; y: Matrix } {
  const rng = createRng(seed).stream('data');
  const x = createMatrix(BATCH, INPUT_SIZE);
  for (let i = 0; i < x.data.length; i++) x.data[i] = rng.uniform(-1.5, 1.5);

  const y = createMatrix(BATCH, outputs);
  if (loss === 'cce') {
    // One-hot targets, per §4.5.
    for (let r = 0; r < BATCH; r++) y.data[r * outputs + rng.int(outputs)] = 1;
  } else if (loss === 'bce') {
    for (let r = 0; r < BATCH; r++) y.data[r] = rng.next() < 0.5 ? 0 : 1;
  } else {
    for (let i = 0; i < y.data.length; i++) y.data[i] = rng.uniform(-1, 1);
  }
  return { x, y };
}

/** Which output activation and width a loss demands (§4.5). */
function outputSpec(
  loss: LossName,
  hiddenActivation: ActivationName,
  outputs: number,
): { activation: ActivationName; units: number } | null {
  switch (loss) {
    case 'mse':
      return { activation: hiddenActivation, units: outputs };
    case 'bce':
      // Only meaningful with a single sigmoid unit.
      return outputs === 1 ? { activation: 'sigmoid', units: 1 } : null;
    case 'cce':
      // Needs K > 1 classes to be a distribution worth scoring.
      return outputs > 1 ? { activation: 'softmax', units: outputs } : null;
  }
}

interface Case {
  readonly label: string;
  readonly layers: LayerSpec[];
  readonly loss: LossName;
  readonly l2: number;
  readonly outputs: number;
  readonly seed: number;
}

function buildCases(): Case[] {
  const cases: Case[] = [];
  let seed = 1000;
  for (const arch of ARCHITECTURES) {
    for (const activation of HIDDEN_ACTIVATIONS) {
      for (const loss of LOSSES) {
        const out = outputSpec(loss, activation, arch.outputs);
        if (out === null) continue; // invalid combination, skipped per §4.11

        const layers: LayerSpec[] = arch.hidden.map((units) => ({ units, activation }));
        layers.push({ units: out.units, activation: out.activation });

        for (const l2 of L2_VALUES) {
          cases.push({
            label: `${arch.name} · ${activation} · ${loss} · L2=${l2}`,
            layers,
            loss,
            l2,
            outputs: out.units,
            seed: seed++,
          });
        }
      }
    }
  }
  return cases;
}

const CASES = buildCases();

describe('gradient check (spec §4.11) — the Phase 1 gate', () => {
  it('covers the full matrix, minus invalid combinations', () => {
    // 3 architectures × 5 activations × 2 L2 settings = 30 mse cases.
    // bce adds the two single-output architectures (2-1, 2-4-1): 2 × 5 × 2 = 20.
    // cce adds only the 3-output architecture: 1 × 5 × 2 = 10.
    expect(CASES.length).toBe(60);
    expect(CASES.filter((c) => c.loss === 'mse').length).toBe(30);
    expect(CASES.filter((c) => c.loss === 'bce').length).toBe(20);
    expect(CASES.filter((c) => c.loss === 'cce').length).toBe(10);
    expect(CASES.filter((c) => c.l2 > 0).length).toBe(30);
  });

  for (const testCase of CASES) {
    it(`${testCase.label}`, () => {
      const net = new Network({
        inputSize: INPUT_SIZE,
        layers: testCase.layers,
        loss: testCase.loss,
        seed: testCase.seed,
        init: { kind: 'glorot_uniform' },
        l2: testCase.l2,
      });
      const { x, y } = fixedBatch(testCase.loss, testCase.outputs, testCase.seed);
      const result = gradientCheck(net, x, y);

      expect(result.checked).toBeGreaterThan(0);
      // The gate. Do not soften this number.
      expect(result.maxRelError, formatGradCheckResult(result)).toBeLessThan(1e-7);
      expect(result.passed).toBe(true);
    });
  }
});

/*
 * Batch normalization, swept separately rather than folded into the matrix
 * above.
 *
 * The variable that matters here is not the activation, it is the BATCH SIZE.
 * Batch norm's gradient couples every sample in the batch to every other one
 * through μ and σ², so the interesting cases are the small batches where that
 * coupling is strongest, and the batch of one where the layer falls back to its
 * running statistics and the coupling disappears entirely.
 */
describe('the bounded spot check the in-app button uses', () => {
  const wide = (): Network =>
    new Network({
      inputSize: INPUT_SIZE,
      layers: [
        { units: 20, activation: 'tanh' },
        { units: 16, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 41,
      init: { kind: 'glorot_uniform' },
      l2: 0,
    });

  it('checks exactly the requested number, and all of them when asked for more', () => {
    const net = wide();
    const { x, y } = fixedBatch('bce', 1, 41);
    expect(net.parameterCount).toBeGreaterThan(100);

    expect(gradientCheck(net, x, y, { maxCoordinates: 60 }).checked).toBe(60);
    expect(gradientCheck(net, x, y, { maxCoordinates: 100000 }).checked).toBe(net.parameterCount);
    expect(gradientCheck(net, x, y).checked).toBe(net.parameterCount);
  });

  it('spreads across every layer and both parameter kinds', () => {
    // A subset that only ever landed in layer 0, or only in W, would report a
    // pass that means much less than it appears to.
    const net = wide();
    const { x, y } = fixedBatch('bce', 1, 42);
    const result = gradientCheck(net, x, y, { maxCoordinates: 60, collectAll: true });

    expect(new Set(result.entries.map((e) => e.location.layer))).toEqual(new Set([0, 1, 2]));
    expect(new Set(result.entries.map((e) => e.location.kind))).toEqual(new Set(['W', 'b']));
  });

  it('does not sample the same column of every row', () => {
    /*
     * The reason the step is fractional. An integer stride sharing a divisor
     * with the layer width walks down a single column and never sees the rest
     * of W, which is precisely the region a transposed-index bug would hide in.
     * Layer 0 here is 2x20, so a stride of 20 would pick one column forever.
     */
    const net = wide();
    const { x, y } = fixedBatch('bce', 1, 43);
    const result = gradientCheck(net, x, y, { maxCoordinates: 60, collectAll: true });
    const columns = new Set(
      result.entries.filter((e) => e.location.layer === 0 && e.location.kind === 'W').map((e) => e.location.col),
    );
    expect(columns.size).toBeGreaterThan(1);
  });

  it('still catches a wrong gradient', () => {
    const net = wide();
    const { x, y } = fixedBatch('bce', 1, 44);
    expect(gradientCheck(net, x, y, { maxCoordinates: 60 }).passed).toBe(true);

    // Scale one whole layer's dW. A spot check that could not see this would
    // not be worth putting behind a button.
    const layer = net.layers[0] as DenseLayer;
    const real = layer.backwardFromDA.bind(layer);
    layer.backwardFromDA = (dA): Matrix => {
      const out = real(dA);
      for (let i = 0; i < layer.dW.data.length; i++) {
        layer.dW.data[i] = (layer.dW.data[i] as number) * 1.05;
      }
      return out;
    };
    expect(gradientCheck(net, x, y, { maxCoordinates: 60 }).passed).toBe(false);
  });
});

describe('gradient check — batch normalization', () => {
  const BN_CASES: readonly {
    readonly label: string;
    readonly layers: LayerSpec[];
    readonly loss: LossName;
    readonly outputs: number;
    readonly l2: number;
  }[] = [
    {
      label: 'one normalized hidden layer',
      layers: [
        { units: 4, activation: 'tanh', batchNorm: true },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      outputs: 1,
      l2: 0,
    },
    {
      label: 'every hidden layer normalized, relu',
      layers: [
        { units: 6, activation: 'relu', batchNorm: true },
        { units: 4, activation: 'relu', batchNorm: true },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      outputs: 1,
      l2: 0,
    },
    {
      label: 'normalized softmax output, with L2',
      layers: [
        { units: 5, activation: 'tanh', batchNorm: true },
        { units: 3, activation: 'softmax', batchNorm: true },
      ],
      loss: 'cce',
      outputs: 3,
      l2: 0.05,
    },
    {
      label: 'mixed: normalized, plain, plain',
      layers: [
        { units: 5, activation: 'sigmoid', batchNorm: true },
        { units: 3, activation: 'tanh' },
        { units: 2, activation: 'linear' },
      ],
      loss: 'mse',
      outputs: 2,
      l2: 0.05,
    },
  ];

  // 1 is the fallback path, 2 is the smallest batch that has a variance at all,
  // and 6 is an ordinary one.
  const BATCH_SIZES = [6, 2, 1] as const;

  let seed = 900;
  for (const testCase of BN_CASES) {
    for (const batch of BATCH_SIZES) {
      it(`${testCase.label}, B=${batch}`, () => {
        const caseSeed = seed++;
        const net = new Network({
          inputSize: INPUT_SIZE,
          layers: testCase.layers,
          loss: testCase.loss,
          seed: caseSeed,
          init: { kind: 'glorot_uniform' },
          l2: testCase.l2,
        });

        /*
         * Move γ and b off their identity values, and the running statistics
         * off (0, 1), before checking.
         *
         * At initialization γ = 1 and b = 0, which makes several wrong
         * derivations look right: a dγ that forgot a factor of γ, or a fallback
         * path that divided by the wrong σ̂, would both pass unnoticed.
         */
        const rng = createRng(caseSeed).stream('init');
        for (const layer of net.layers) {
          if (!layer.batchNorm) continue;
          for (let j = 0; j < layer.units; j++) {
            layer.gamma.data[j] = rng.uniform(0.4, 1.8);
            layer.b.data[j] = rng.uniform(-0.5, 0.5);
            layer.runningMean[j] = rng.uniform(-0.6, 0.6);
            layer.runningVar[j] = rng.uniform(0.3, 1.4);
          }
        }

        const full = fixedBatch(testCase.loss, testCase.outputs, caseSeed);
        const x = rowView(full.x, batch);
        const y = rowView(full.y, batch);

        const statisticsBefore = Array.from(net.buffers);
        const result = gradientCheck(net, x, y);

        expect(result.checked).toBeGreaterThan(0);
        expect(result.maxRelError, formatGradCheckResult(result)).toBeLessThan(1e-7);
        expect(result.passed).toBe(true);

        // Guard #4: the thousands of forward passes above must not have moved
        // the statistics this network will predict with.
        expect(Array.from(net.buffers)).toEqual(statisticsBefore);
      });
    }
  }

  it('gives γ a handle of its own, and visits every one', () => {
    const net = new Network({
      inputSize: INPUT_SIZE,
      layers: [
        { units: 4, activation: 'tanh', batchNorm: true },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 31,
      init: { kind: 'glorot_uniform' },
      l2: 0,
    });
    // One handle per γ, and none for the layer that does not normalize.
    const handles = parameterHandles(net);
    expect(handles.filter((h) => h.location.kind === 'gamma').length).toBe(4);
    expect(handles.filter((h) => h.location.kind === 'gamma' && h.location.layer === 1)).toEqual([]);

    const { x, y } = fixedBatch('bce', 1, 31);
    const result = gradientCheck(net, x, y, { collectAll: true });
    expect(result.passed).toBe(true);
    // The γ coordinates were actually visited, so a pass is evidence about
    // them rather than evidence that they were never looked at. The test below
    // covers the other half: that a wrong dγ is caught.
    expect(result.entries.filter((e) => e.location.kind === 'gamma').length).toBe(4);
  });

  it('the near-zero rescue never fires on the original matrix', () => {
    /*
     * ABSOLUTE_TOLERANCE was added for batch normalization, and it must not
     * have quietly loosened the Phase 1 gate on the way in. Every one of the 60
     * cases has to still be judged purely on relative error.
     */
    for (const testCase of CASES) {
      const net = new Network({
        inputSize: INPUT_SIZE,
        layers: testCase.layers,
        loss: testCase.loss,
        seed: testCase.seed,
        init: { kind: 'glorot_uniform' },
        l2: testCase.l2,
      });
      const { x, y } = fixedBatch(testCase.loss, testCase.outputs, testCase.seed);
      expect(gradientCheck(net, x, y).negligible, testCase.label).toBe(0);
    }
  });

  it('the rescue cannot save a gradient that is actually wrong', () => {
    /*
     * The rescue only fires when the values agree to 1e-10 absolutely. Break a
     * gradient by 1e-8 — far too small to see on any chart, far too big to slip
     * past — and the check must still fail.
     */
    const build = (): Network =>
      new Network({
        inputSize: INPUT_SIZE,
        layers: [
          { units: 4, activation: 'tanh', batchNorm: true },
          { units: 1, activation: 'sigmoid' },
        ],
        loss: 'bce',
        seed: 77,
        init: { kind: 'glorot_uniform' },
        l2: 0,
      });
    const { x, y } = fixedBatch('bce', 1, 77);

    expect(gradientCheck(build(), x, y).passed).toBe(true);

    // A layer whose dγ is short by a constant. The forward pass is untouched,
    // so the numerical gradient stays right and only the analytic one moves.
    const broken = build();
    const honest = broken.layers[0] as DenseLayer;
    const realBackward = honest.backwardFromDA.bind(honest);
    honest.backwardFromDA = (dA): Matrix => {
      const out = realBackward(dA);
      honest.dGamma.data[0] = (honest.dGamma.data[0] as number) + 1e-8;
      return out;
    };
    const result = gradientCheck(broken, x, y);
    expect(result.passed).toBe(false);
    expect(result.worst?.location.kind).toBe('gamma');
  });

  it('a layer without batch norm contributes no γ, so the layout is unchanged', () => {
    const plain = new Network({
      inputSize: INPUT_SIZE,
      layers: [
        { units: 4, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 4,
      init: { kind: 'glorot_uniform' },
      l2: 0,
    });
    expect(plain.params.length).toBe(2 * 4 + 4 + 4 * 1 + 1);
    expect(plain.buffers.length).toBe(0);
    expect(parameterHandles(plain).every((h) => h.location.kind !== 'gamma')).toBe(true);
  });
});

describe('gradient check — the false-failure guards it documents', () => {
  it('differentiates the FULL objective, so L2 changes the numerical gradient too', () => {
    const layers: LayerSpec[] = [
      { units: 4, activation: 'tanh' },
      { units: 1, activation: 'linear' },
    ];
    const { x, y } = fixedBatch('mse', 1, 7);

    const withL2 = new Network({
      inputSize: INPUT_SIZE,
      layers,
      loss: 'mse',
      seed: 7,
      init: { kind: 'glorot_uniform' },
      l2: 0.1,
    });
    expect(gradientCheck(withL2, x, y).passed).toBe(true);

    // Now prove the L2 term is actually present: comparing an analytic gradient
    // that includes λ·W against a numerical one that excludes it must FAIL.
    // Simulated by checking the analytic dW moved when λ turned on.
    const withoutL2 = new Network({
      inputSize: INPUT_SIZE,
      layers,
      loss: 'mse',
      seed: 7,
      init: { kind: 'glorot_uniform' },
      l2: 0,
    });
    withoutL2.forward(x, true);
    withoutL2.backward(y);
    withL2.forward(x, true);
    withL2.backward(y);

    const plain = withoutL2.layers[0]!.dW.data[0]!;
    const regularised = withL2.layers[0]!.dW.data[0]!;
    const w = withL2.layers[0]!.W.data[0]!;
    expect(regularised).toBeCloseTo(plain + 0.1 * w, 12);
    expect(regularised).not.toBeCloseTo(plain, 6);
  });

  it('never applies L2 to biases (§4.9)', () => {
    const layers: LayerSpec[] = [
      { units: 3, activation: 'tanh' },
      { units: 1, activation: 'linear' },
    ];
    const { x, y } = fixedBatch('mse', 1, 11);
    const a = new Network({ inputSize: 2, layers, loss: 'mse', seed: 11, init: { kind: 'glorot_uniform' }, l2: 0 });
    const b = new Network({ inputSize: 2, layers, loss: 'mse', seed: 11, init: { kind: 'glorot_uniform' }, l2: 0.5 });
    a.forward(x, true);
    a.backward(y);
    b.forward(x, true);
    b.backward(y);
    expect(Array.from(b.layers[0]!.db.data)).toEqual(Array.from(a.layers[0]!.db.data));
  });

  it('reports ReLU kink skips rather than silently failing', () => {
    // A batch engineered so several units sit within ε of the kink.
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 6, activation: 'relu' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 4242,
      init: { kind: 'glorot_uniform' },
    });
    // Drive z toward zero by feeding near-zero inputs.
    const x = fromRows([
      [0, 0],
      [1e-9, -1e-9],
    ]);
    const y = fromRows([[0.5], [-0.5]]);
    const result = gradientCheck(net, x, y);
    expect(result.skipped).toBeGreaterThanOrEqual(0);
    expect(result.checked + result.skipped).toBe(net.parameterCount);
  });

  it('formats a readable summary for the UI button', () => {
    const net = new Network({
      inputSize: 2,
      layers: [{ units: 1, activation: 'sigmoid' }],
      loss: 'bce',
      seed: 5,
      init: { kind: 'glorot_uniform' },
    });
    const { x, y } = fixedBatch('bce', 1, 5);
    const summary = formatGradCheckResult(gradientCheck(net, x, y));
    expect(summary).toMatch(/^PASS\. Max relative error \d\.\d+e[+-]\d+ at layer 1 /);
    expect(summary).toContain('parameters checked');
    // This string is rendered in the app, not only in a test failure.
    expect(summary).not.toContain('—');
  });

  it('catches a deliberately broken gradient', () => {
    // Sanity: if gradcheck cannot fail, it is not testing anything. Corrupt one
    // analytic gradient and confirm the check notices.
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 3, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 21,
      init: { kind: 'glorot_uniform' },
    });
    const { x, y } = fixedBatch('mse', 1, 21);
    expect(gradientCheck(net, x, y).passed).toBe(true);

    const broken = Object.create(Object.getPrototypeOf(net) as object) as Network;
    Object.assign(broken, net);
    const originalBackward = net.backward.bind(net);
    (broken as { backward: (t: Matrix) => void }).backward = (t: Matrix): void => {
      originalBackward(t);
      net.layers[0]!.dW.data[0] = net.layers[0]!.dW.data[0]! * 1.5 + 0.01;
    };
    const result = gradientCheck(broken, x, y);
    expect(result.passed).toBe(false);
    expect(result.maxRelError).toBeGreaterThan(1e-7);
  });
});

/*
 * The estimator is only trustworthy if a WRONG gradient fails at every step
 * size and under every candidate. These tests establish that, and record what
 * the spec's literal ε = 1e-5 does on the architecture where float64 runs out.
 */
describe('step-size selection — why taking the best candidate is sound', () => {
  // The genuine worst case in the §4.11 matrix above: 2-8-6-3, sigmoid, mse,
  // L2=0 is seed 1056 under buildCases()' ordering. Using the real worst case
  // rather than a hand-picked one is the point of this block.
  const deepCase = {
    inputSize: 2,
    layers: [
      { units: 8, activation: 'sigmoid' as const },
      { units: 6, activation: 'sigmoid' as const },
      { units: 3, activation: 'sigmoid' as const },
    ],
    loss: 'mse' as const,
    seed: 1056,
    init: { kind: 'glorot_uniform' as const },
    l2: 0,
  };

  it('Richardson reaches ~1e-9 on the hardest case in the matrix', () => {
    const net = new Network(deepCase);
    const { x, y } = fixedBatch('mse', 3, 1056);
    const result = gradientCheck(net, x, y);
    // Measured 1.09e-9 against a 1e-7 gate — roughly 90x headroom. Asserted at
    // 5e-9 so a real regression trips here long before it trips the gate.
    expect(result.maxRelError).toBeLessThan(5e-9);
  });

  it("beats the spec's literal ε = 1e-5 by two orders of magnitude", () => {
    const net = new Network(deepCase);
    const { x, y } = fixedBatch('mse', 3, 1056);

    const richardson = gradientCheck(net, x, y);
    const atSpecEpsilon = gradientCheck(net, x, y, { epsilon: SPEC_EPSILON });

    // A single central difference at 1e-5 sits past the roundoff crossover for
    // an objective of magnitude ~0.74 against a gradient of ~1e-4. This records
    // the numerical limit rather than tolerating a failure: the very same
    // analytic gradients reach 1e-9 under the default estimator.
    expect(atSpecEpsilon.maxRelError).toBeGreaterThan(richardson.maxRelError * 10);
    expect(atSpecEpsilon.maxRelError).toBeLessThan(1e-6);
    expect(atSpecEpsilon.stepPairs).toEqual([]);
    expect(atSpecEpsilon.methodCounts.richardson).toBe(0);
  });

  it('a wrong gradient fails under EVERY candidate, so the search cannot launder it', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 5, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 33,
      init: { kind: 'glorot_uniform' },
    });
    const { x, y } = fixedBatch('mse', 1, 33);
    expect(gradientCheck(net, x, y).passed).toBe(true);

    // Scale one analytic gradient by 1.01 — a 1% error, far subtler than any
    // sign or shape bug would produce.
    const originalBackward = net.backward.bind(net);
    const corrupted = Object.create(Object.getPrototypeOf(net) as object) as Network;
    Object.assign(corrupted, net);
    (corrupted as { backward: (t: Matrix) => void }).backward = (t: Matrix): void => {
      originalBackward(t);
      net.layers[0]!.dW.data[0] = net.layers[0]!.dW.data[0]! * 1.01;
    };

    // Every individual step size, used as a plain central difference.
    for (const pair of DEFAULT_STEP_PAIRS) {
      for (const step of [pair.coarse, pair.fine]) {
        const result = gradientCheck(corrupted, x, y, { epsilon: step });
        expect(result.maxRelError, `step ${step}`).toBeGreaterThan(1e-7);
      }
    }
    // Every Richardson pair in isolation.
    for (const pair of DEFAULT_STEP_PAIRS) {
      const result = gradientCheck(corrupted, x, y, { stepPairs: [pair] });
      expect(result.maxRelError, `pair ${pair.coarse}`).toBeGreaterThan(1e-7);
    }
    // And therefore under the full default search.
    expect(gradientCheck(corrupted, x, y).passed).toBe(false);
  });

  it('reports which estimator each coordinate settled on', () => {
    const net = new Network(deepCase);
    const { x, y } = fixedBatch('mse', 3, 1056);
    const result = gradientCheck(net, x, y, { collectAll: true });

    expect(result.stepPairs).toEqual(DEFAULT_STEP_PAIRS);
    expect(result.entries.length).toBe(result.checked);
    expect(result.methodCounts.central + result.methodCounts.richardson).toBe(result.checked);
    // On a smooth sigmoid network Richardson should win nearly everywhere.
    expect(result.methodCounts.richardson).toBeGreaterThan(result.checked * 0.8);

    const validSteps = DEFAULT_STEP_PAIRS.flatMap((p) => [p.coarse, p.fine]);
    for (const entry of result.entries) {
      expect(validSteps).toContain(entry.step);
    }
  });

  it('falls back to plain central differences on ReLU, where they are exact', () => {
    // ReLU is piecewise linear, so away from the kink a central difference is
    // exact and Richardson's smoothness assumption adds nothing. The estimator
    // should discover that on its own.
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 8, activation: 'relu' },
        { units: 6, activation: 'relu' },
        { units: 3, activation: 'relu' },
      ],
      loss: 'mse',
      seed: 1056,
      init: { kind: 'glorot_uniform' },
    });
    const { x, y } = fixedBatch('mse', 3, 1056);
    const result = gradientCheck(net, x, y);
    expect(result.passed).toBe(true);
    expect(result.methodCounts.central).toBeGreaterThan(result.methodCounts.richardson);
  });

  it('rejects step pairs that are not exactly 2:1', () => {
    const net = new Network(deepCase);
    const { x, y } = fixedBatch('mse', 3, 1056);
    // The 4/3 and -1/3 weights are only valid for fine = coarse / 2; a silently
    // wrong ratio would produce a plausible-looking but biased estimate.
    expect(() =>
      gradientCheck(net, x, y, { stepPairs: [{ coarse: 1e-3, fine: 1e-4 }] }),
    ).toThrowError(/requires fine = coarse \/ 2/);
    expect(() => gradientCheck(net, x, y, { stepPairs: [] })).toThrowError(
      /at least one step pair/,
    );
  });
});
