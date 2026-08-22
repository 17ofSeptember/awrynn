import { describe, expect, it } from 'vitest';
import { evaluateGeneratedJs, generateJavaScript, generateNumpy } from '../codegen';
import { Network } from '../network';
import type { LayerSpec } from '../layers';
import type { LossName } from '../losses';
import { createRng } from '../rng';
import { createMatrix, fromRows } from '../tensor';

/*
 * Spec §10: "Codegen: generated JS inference output matches the engine's output
 * to 1e-12 on 100 random inputs."
 *
 * The test runs the ACTUAL emitted source rather than a reimplementation of it,
 * so a bug in the text that is generated is caught, not just a bug in the idea
 * of generating it.
 */

interface Case {
  readonly label: string;
  readonly layers: LayerSpec[];
  readonly loss: LossName;
  readonly inputSize: number;
}

const CASES: readonly Case[] = [
  { label: 'linear', inputSize: 2, loss: 'mse', layers: [{ units: 1, activation: 'linear' }] },
  {
    label: 'tanh + sigmoid',
    inputSize: 2,
    loss: 'bce',
    layers: [
      { units: 6, activation: 'tanh' },
      { units: 1, activation: 'sigmoid' },
    ],
  },
  {
    label: 'relu deep',
    inputSize: 3,
    loss: 'mse',
    layers: [
      { units: 5, activation: 'relu' },
      { units: 4, activation: 'relu' },
      { units: 2, activation: 'linear' },
    ],
  },
  {
    label: 'leaky relu',
    inputSize: 2,
    loss: 'mse',
    layers: [
      { units: 4, activation: 'leaky_relu', leakyAlpha: 0.02 },
      { units: 1, activation: 'linear' },
    ],
  },
  {
    label: 'softmax + cce',
    inputSize: 2,
    loss: 'cce',
    layers: [
      { units: 6, activation: 'tanh' },
      { units: 3, activation: 'softmax' },
    ],
  },
  {
    label: 'batch-normalised hidden layer',
    inputSize: 2,
    loss: 'bce',
    layers: [
      { units: 6, activation: 'tanh', batchNorm: true },
      { units: 1, activation: 'sigmoid' },
    ],
  },
  {
    label: 'batch norm all the way through, including the softmax output',
    inputSize: 3,
    loss: 'cce',
    layers: [
      { units: 5, activation: 'relu', batchNorm: true },
      { units: 4, activation: 'leaky_relu', batchNorm: true, leakyAlpha: 0.05 },
      { units: 3, activation: 'softmax', batchNorm: true },
    ],
  },
  {
    label: 'wide input',
    inputSize: 35,
    loss: 'cce',
    layers: [
      { units: 8, activation: 'relu' },
      { units: 10, activation: 'softmax' },
    ],
  },
];

function build(testCase: Case, seed: number): Network {
  const network = new Network({
    inputSize: testCase.inputSize,
    layers: testCase.layers,
    loss: testCase.loss,
    seed,
    init: { kind: 'glorot_uniform' },
  });

  /*
   * A freshly built normalised layer is very nearly the identity: γ = 1, b = 0,
   * μ̂ = 0 and σ̂² = 1. Exporting that would prove nothing, because an export
   * that dropped the normalisation entirely would still agree. So move γ and b
   * off their identity values and run real batches through to give the running
   * statistics something to be.
   */
  if (network.layers.some((l) => l.batchNorm)) {
    const rng = createRng(seed).stream('init');
    for (const layer of network.layers) {
      if (!layer.batchNorm) continue;
      for (let j = 0; j < layer.units; j++) {
        layer.gamma.data[j] = rng.uniform(0.5, 1.7);
        layer.b.data[j] = rng.uniform(-0.4, 0.4);
      }
    }
    const batch = createMatrix(24, testCase.inputSize);
    for (let i = 0; i < batch.data.length; i++) batch.data[i] = rng.uniform(-2, 2);
    for (let i = 0; i < 30; i++) network.forward(batch, true);
  }
  return network;
}

describe('generated JavaScript matches the engine (§10)', () => {
  CASES.forEach((testCase, index) => {
    it(`${testCase.label}: 100 random inputs agree to 1e-12`, () => {
      const network = build(testCase, 100 + index);
      const forward = evaluateGeneratedJs(generateJavaScript(network));
      const rng = createRng(200 + index).stream('data');

      let worst = 0;
      for (let trial = 0; trial < 100; trial++) {
        const input: number[] = [];
        for (let i = 0; i < testCase.inputSize; i++) input.push(rng.uniform(-3, 3));

        const expected = network.forward(fromRows([input]), false);
        const actual = forward(input);

        expect(actual.length).toBe(expected.cols);
        for (let i = 0; i < actual.length; i++) {
          worst = Math.max(worst, Math.abs((actual[i] as number) - (expected.data[i] as number)));
        }
      }
      // Full-precision literals, so this is far tighter than the 1e-12 required.
      expect(worst).toBeLessThan(1e-12);
    });
  });

  it('round-trips float64 exactly, not just closely', () => {
    // 17 significant digits is the shortest precision that recovers a float64
    // bit for bit; at 15 the generated code would disagree in the last decimal.
    const network = build(CASES[1] as Case, 7);
    const source = generateJavaScript(network);
    const first = network.params[0] as number;
    expect(source).toContain(first.toPrecision(17));
  });

  it('handles a saturating input without producing NaN', () => {
    // The generated sigmoid must use the same two-sided form as the engine.
    const network = build(CASES[1] as Case, 9);
    const forward = evaluateGeneratedJs(generateJavaScript(network));
    for (const magnitude of [1e3, 1e6, -1e6]) {
      const out = forward([magnitude, -magnitude]);
      expect(out.every((v) => Number.isFinite(v)), `magnitude ${magnitude}`).toBe(true);
    }
  });

  it('produces softmax rows that sum to 1', () => {
    const network = build(CASES[4] as Case, 11);
    const forward = evaluateGeneratedJs(generateJavaScript(network));
    const out = forward([0.5, -1.25]);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
});

describe('generated NumPy', () => {
  it('is emitted for every activation, with the weights inlined', () => {
    for (const testCase of CASES) {
      const network = build(testCase, 3);
      const source = generateNumpy(network);
      expect(source, testCase.label).toContain('import numpy as np');
      expect(source, testCase.label).toContain('def forward(x):');
      // Every layer's parameters appear as literals, not as a loader.
      network.layers.forEach((_, i) => {
        expect(source, `${testCase.label} W${i + 1}`).toContain(`W${i + 1} = np.array([`);
        expect(source, `${testCase.label} b${i + 1}`).toContain(`b${i + 1} = np.array([`);
      });
    }
  });

  it('defines softmax only when the network uses it', () => {
    expect(generateNumpy(build(CASES[4] as Case, 1))).toContain('def softmax(z):');
    expect(generateNumpy(build(CASES[1] as Case, 1))).not.toContain('def softmax(z):');
  });

  it('states the shapes and the convention, so the code can be read', () => {
    // §8: the point is that the learner sees the animated thing is thirty lines
    // of matrix multiplication.
    const source = generateNumpy(build(CASES[1] as Case, 1));
    expect(source).toContain('Z^l = A^{l-1} @ W^l + b^l');
    expect(source).toContain('one row per sample');
  });

  it('carries the configured leaky alpha rather than assuming 0.01', () => {
    const source = generateNumpy(build(CASES[3] as Case, 1));
    expect(source).toContain('LEAKY_ALPHA = 0.020000000000000000');
  });
});

describe('JSON round-trip (§10 regression guard)', () => {
  it('serialize then deserialize predicts identically', () => {
    for (const testCase of CASES) {
      const network = build(testCase, 21);
      const restored = Network.deserialize(
        JSON.parse(JSON.stringify(network.serialize())) as ReturnType<Network['serialize']>,
      );
      const rng = createRng(5).stream('data');
      const input: number[] = [];
      for (let i = 0; i < testCase.inputSize; i++) input.push(rng.uniform(-2, 2));
      const before = Array.from(network.forward(fromRows([input]), false).data);
      const after = Array.from(restored.forward(fromRows([input]), false).data);
      expect(after, testCase.label).toEqual(before);
    }
  });
});
