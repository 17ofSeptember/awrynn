import { describe, expect, it } from 'vitest';
import { Network, validateConfig } from '../network';
import type { NetworkConfig } from '../network';
import { createMatrix, fromRows, toRows } from '../tensor';
import { createRng } from '../rng';
import { sigmoid } from '../activations';

function base(overrides: Partial<NetworkConfig> = {}): NetworkConfig {
  return {
    inputSize: 2,
    layers: [{ units: 1, activation: 'linear' }],
    loss: 'mse',
    seed: 1,
    init: { kind: 'glorot_uniform' },
    ...overrides,
  };
}

describe('forward pass — hand-computed (§4.2)', () => {
  it('computes Z = A·W + 1·b and A = φ(Z) for a 2-2-1 network', () => {
    const net = new Network(
      base({ layers: [{ units: 2, activation: 'relu' }, { units: 1, activation: 'linear' }] }),
    );
    // Hand-set every parameter so the arithmetic is fully determined.
    net.layers[0]!.setWeights(
      fromRows([
        [1, -1],
        [2, 0.5],
      ]),
      fromRows([[0.1, -0.2]]),
    );
    net.layers[1]!.setWeights(fromRows([[3], [-1]]), fromRows([[0.5]]));

    const x = fromRows([[1, 2]]);
    // Layer 1: z = [1*1 + 2*2 + 0.1, 1*(-1) + 2*0.5 + (-0.2)] = [5.1, -0.2]
    //          a = relu(z) = [5.1, 0]
    // Layer 2: z = 5.1*3 + 0*(-1) + 0.5 = 15.8 ; a = 15.8 (linear)
    const yHat = net.forward(x, true);
    expect(toRows(net.layers[0]!.Z!)).toEqual([[5.1, -0.2]]);
    expect(toRows(net.layers[0]!.A!)[0]![0]!).toBeCloseTo(5.1, 12);
    expect(toRows(net.layers[0]!.A!)[0]![1]!).toBe(0);
    expect(toRows(yHat)[0]![0]!).toBeCloseTo(15.8, 12);
  });

  it('broadcasts the bias down every row of the batch', () => {
    const net = new Network(base({ layers: [{ units: 2, activation: 'linear' }] }));
    net.layers[0]!.setWeights(
      fromRows([
        [0, 0],
        [0, 0],
      ]),
      fromRows([[7, -3]]),
    );
    const out = net.forward(fromRows([[1, 1], [2, 2], [3, 3]]));
    expect(toRows(out)).toEqual([
      [7, -3],
      [7, -3],
      [7, -3],
    ]);
  });

  it('rejects an input whose width does not match', () => {
    const net = new Network(base());
    expect(() => net.forward(createMatrix(3, 5))).toThrowError(/\[3, 5\].*input size 2/);
  });
});

describe('backward pass — the divide-by-B convention (§4.3)', () => {
  it('computes dW = A^T·dZ / B and db = colSum(dZ) / B by hand', () => {
    const net = new Network(base({ layers: [{ units: 1, activation: 'linear' }], loss: 'mse' }));
    net.layers[0]!.setWeights(fromRows([[1], [1]]), fromRows([[0]]));

    // Two samples. ŷ = x1 + x2.
    const x = fromRows([
      [1, 2],
      [3, 4],
    ]);
    const y = fromRows([[0], [0]]);
    // ŷ = [3], [7]; dA = ŷ − y = [3], [7]; linear so dZ = dA.
    // dW = Xᵀ·dZ / B = [[1*3 + 3*7], [2*3 + 4*7]] / 2 = [[24], [34]] / 2 = [[12], [17]]
    // db = (3 + 7) / 2 = 5
    net.forward(x, true);
    net.backward(y);
    expect(toRows(net.layers[0]!.dW)).toEqual([[12], [17]]);
    expect(toRows(net.layers[0]!.db)).toEqual([[5]]);
  });

  it('divides by B exactly once — doubling the batch with identical rows leaves gradients unchanged', () => {
    // If dA^{l-1} were also divided by B, this invariant would break for any
    // network deeper than one layer.
    const config = base({
      layers: [
        { units: 3, activation: 'tanh' },
        { units: 1, activation: 'linear' },
      ],
    });
    const single = new Network(config);
    const doubled = new Network(config);

    const x1 = fromRows([[0.5, -0.25]]);
    const y1 = fromRows([[1]]);
    const x2 = fromRows([
      [0.5, -0.25],
      [0.5, -0.25],
    ]);
    const y2 = fromRows([[1], [1]]);

    single.forward(x1, true);
    single.backward(y1);
    doubled.forward(x2, true);
    doubled.backward(y2);

    for (let l = 0; l < single.layers.length; l++) {
      const a = Array.from(single.layers[l]!.dW.data);
      const b = Array.from(doubled.layers[l]!.dW.data);
      for (let i = 0; i < a.length; i++) expect(b[i]!).toBeCloseTo(a[i]!, 12);
    }
  });

  it('requires a training forward pass first', () => {
    const net = new Network(base());
    expect(() => net.backward(fromRows([[0]]))).toThrowError(/forward\(x, true\)/);
  });

  it('rejects a target whose shape does not match the prediction', () => {
    const net = new Network(base());
    net.forward(fromRows([[1, 2]]), true);
    expect(() => net.backward(fromRows([[0, 0]]))).toThrowError(/\[1, 2\].*\[1, 1\]/);
  });
});

describe('fused output-gradient detection (§4.3)', () => {
  it('detects softmax + cce', () => {
    const net = new Network(
      base({ layers: [{ units: 3, activation: 'softmax' }], loss: 'cce' }),
    );
    expect(net.outputGradientMode).toBe('fused-softmax-cce');
  });

  it('detects sigmoid + bce', () => {
    const net = new Network(
      base({ layers: [{ units: 1, activation: 'sigmoid' }], loss: 'bce' }),
    );
    expect(net.outputGradientMode).toBe('fused-sigmoid-bce');
  });

  it('uses the general path for mse', () => {
    expect(new Network(base()).outputGradientMode).toBe('general');
    expect(
      new Network(base({ layers: [{ units: 1, activation: 'sigmoid' }], loss: 'mse' }))
        .outputGradientMode,
    ).toBe('general');
  });

  it('fused sigmoid + bce produces dZ = ŷ − y', () => {
    const net = new Network(base({ layers: [{ units: 1, activation: 'sigmoid' }], loss: 'bce' }));
    net.layers[0]!.setWeights(fromRows([[0.5], [-0.5]]), fromRows([[0.25]]));
    const x = fromRows([[1, 2]]);
    const y = fromRows([[1]]);
    const yHat = net.forward(x, true);
    net.backward(y);
    // z = 0.5 - 1 + 0.25 = -0.25 ; ŷ = sigmoid(-0.25)
    expect(toRows(yHat)[0]![0]!).toBeCloseTo(sigmoid(-0.25), 12);
    expect(net.layers[0]!.dZ!.data[0]!).toBeCloseTo(sigmoid(-0.25) - 1, 12);
  });
});

describe('configuration validation (§4.4) — clear errors, never a silent fallback', () => {
  it('rejects softmax on a hidden layer', () => {
    const problems = validateConfig(
      base({
        layers: [
          { units: 4, activation: 'softmax' },
          { units: 3, activation: 'softmax' },
        ],
        loss: 'cce',
      }),
    );
    expect(problems.some((p) => /only valid on the output layer/.test(p))).toBe(true);
  });

  it('rejects softmax with anything but categorical cross-entropy', () => {
    const problems = validateConfig(
      base({ layers: [{ units: 3, activation: 'softmax' }], loss: 'mse' }),
    );
    expect(problems.some((p) => /requires the categorical cross-entropy loss/.test(p))).toBe(true);
  });

  it('rejects cce without softmax', () => {
    const problems = validateConfig(
      base({ layers: [{ units: 3, activation: 'sigmoid' }], loss: 'cce' }),
    );
    expect(problems.some((p) => /requires a softmax output/.test(p))).toBe(true);
  });

  it('rejects bce with more than one output unit', () => {
    const problems = validateConfig(
      base({ layers: [{ units: 3, activation: 'sigmoid' }], loss: 'bce' }),
    );
    expect(problems.some((p) => /single output unit/.test(p))).toBe(true);
  });

  it('rejects bce without a sigmoid output', () => {
    const problems = validateConfig(
      base({ layers: [{ units: 1, activation: 'relu' }], loss: 'bce' }),
    );
    expect(problems.some((p) => /expects a sigmoid output/.test(p))).toBe(true);
  });

  it('reports every problem at once, not just the first', () => {
    const problems = validateConfig(
      base({ layers: [{ units: 3, activation: 'relu' }], loss: 'bce', l2: -1 }),
    );
    expect(problems.length).toBeGreaterThanOrEqual(3);
  });

  it('accepts valid configurations', () => {
    expect(validateConfig(base())).toEqual([]);
    expect(
      validateConfig(base({ layers: [{ units: 3, activation: 'softmax' }], loss: 'cce' })),
    ).toEqual([]);
  });

  it('throws with all problems listed when constructing', () => {
    expect(() => new Network(base({ layers: [{ units: 2, activation: 'relu' }], loss: 'bce' })))
      .toThrowError(/Invalid network configuration/);
  });
});

describe('determinism (§4.7) — the Phase 1 gate', () => {
  /**
   * Spec §10: "identical seed twice ⇒ bitwise identical parameter arrays after
   * 100 steps". Optimizers arrive in Phase 2, so the update here is plain SGD
   * applied by hand — the point of the test is the engine's determinism, not
   * the optimizer's.
   */
  function train(seed: number, steps: number): Float64Array {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 6, activation: 'tanh' },
        { units: 4, activation: 'relu' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed,
      init: { kind: 'he_normal' },
      l2: 0.01,
    });
    const rng = createRng(seed).stream('data');
    const x = createMatrix(8, 2);
    for (let i = 0; i < x.data.length; i++) x.data[i] = rng.uniform(-2, 2);
    const y = createMatrix(8, 1);
    for (let i = 0; i < y.data.length; i++) y.data[i] = rng.next() < 0.5 ? 0 : 1;

    for (let step = 0; step < steps; step++) {
      net.forward(x, true);
      net.backward(y);
      for (const layer of net.layers) {
        for (let i = 0; i < layer.W.data.length; i++) {
          layer.W.data[i] = layer.W.data[i]! - 0.1 * layer.dW.data[i]!;
        }
        for (let i = 0; i < layer.b.data.length; i++) {
          layer.b.data[i] = layer.b.data[i]! - 0.1 * layer.db.data[i]!;
        }
      }
    }
    return net.captureParameters();
  }

  it('is bitwise identical for the same seed after 100 steps', () => {
    const a = train(20259, 100);
    const b = train(20259, 100);
    expect(a.length).toBeGreaterThan(0);
    // Bitwise, not approximate: compare the raw bytes.
    expect(new Uint8Array(a.buffer)).toEqual(new Uint8Array(b.buffer));
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('differs for a different seed', () => {
    const a = train(1, 100);
    const b = train(2, 100);
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  it('resetToInit returns exactly to the initial parameters', () => {
    const net = new Network(
      base({ layers: [{ units: 5, activation: 'tanh' }, { units: 1, activation: 'linear' }], seed: 9 }),
    );
    const initial = net.captureParameters();
    const x = fromRows([[1, -1]]);
    const y = fromRows([[0.5]]);
    for (let i = 0; i < 10; i++) {
      net.forward(x, true);
      net.backward(y);
      for (const layer of net.layers) {
        for (let j = 0; j < layer.W.data.length; j++) {
          layer.W.data[j] = layer.W.data[j]! - 0.5 * layer.dW.data[j]!;
        }
      }
    }
    expect(Array.from(net.captureParameters())).not.toEqual(Array.from(initial));
    net.resetToInit();
    expect(Array.from(net.captureParameters())).toEqual(Array.from(initial));
  });
});

describe('serialization (§10 regression guard)', () => {
  it('round-trips through JSON and predicts identically', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 5, activation: 'leaky_relu', leakyAlpha: 0.02 },
        { units: 3, activation: 'softmax' },
      ],
      loss: 'cce',
      seed: 314,
      init: { kind: 'he_normal' },
      l2: 0.03,
    });
    const x = createMatrix(10, 2);
    const rng = createRng(1).stream('data');
    for (let i = 0; i < x.data.length; i++) x.data[i] = rng.uniform(-3, 3);
    const before = Array.from(net.forward(x).data);

    const json = JSON.stringify(net.serialize());
    const restored = Network.deserialize(JSON.parse(json) as ReturnType<Network['serialize']>);
    const after = Array.from(restored.forward(x).data);

    expect(after).toEqual(before);
    expect(restored.l2).toBe(0.03);
    expect(restored.layers[0]!.leakyAlpha).toBe(0.02);
    expect(restored.outputGradientMode).toBe('fused-softmax-cce');
  });

  it('rejects an unknown format version', () => {
    const net = new Network(base());
    const data = { ...net.serialize(), version: 7 as unknown as 1 };
    expect(() => Network.deserialize(data)).toThrowError(/unsupported format version 7/);
  });

  it('still reads a version 1 file, which predates batch normalization', () => {
    // v1 had no gammas or running statistics, and could not have had a layer
    // that needed them. Files saved before batch norm existed must still open.
    const net = new Network(base({ seed: 12 }));
    const current = net.serialize();
    const legacy = {
      version: 1 as const,
      inputSize: current.inputSize,
      layers: current.layers,
      loss: current.loss,
      seed: current.seed,
      init: current.init,
      l2: current.l2,
      weights: current.weights,
      biases: current.biases,
    };
    const restored = Network.deserialize(legacy);
    expect(Array.from(restored.params)).toEqual(Array.from(net.params));
  });

  it('clone() produces an independent network with identical predictions', () => {
    const net = new Network(
      base({ layers: [{ units: 4, activation: 'tanh' }, { units: 1, activation: 'linear' }], seed: 8 }),
    );
    const copy = net.clone();
    const x = fromRows([[0.3, -0.7]]);
    expect(Array.from(copy.forward(x).data)).toEqual(Array.from(net.forward(x).data));

    copy.layers[0]!.W.data[0] = 99;
    expect(net.layers[0]!.W.data[0]).not.toBe(99);
  });
});

describe('a stack of linear layers is one linear layer (§7.2)', () => {
  it('composes W¹W²W³ into a single map that reproduces the network', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 8, activation: 'linear' },
        { units: 16, activation: 'linear' },
        { units: 1, activation: 'linear' },
      ],
      loss: 'mse',
      seed: 77,
      init: { kind: 'glorot_uniform' },
    });
    // Zero the biases so the composition is purely W¹W²W³.
    for (const layer of net.layers) layer.b.data.fill(0);

    const composed = net.composedLinearMap();
    expect(composed).not.toBeNull();
    expect(composed!.rows).toBe(2);
    expect(composed!.cols).toBe(1);

    const x = fromRows([[1.5, -2.25]]);
    const viaNetwork = toRows(net.forward(x))[0]![0]!;
    const viaComposed =
      1.5 * composed!.data[0]! + -2.25 * composed!.data[1]!;
    expect(viaComposed).toBeCloseTo(viaNetwork, 10);
  });

  it('returns null when any activation is non-linear', () => {
    const net = new Network(
      base({ layers: [{ units: 4, activation: 'tanh' }, { units: 1, activation: 'linear' }] }),
    );
    expect(net.composedLinearMap()).toBeNull();
  });
});

describe('objective reporting (§4.5)', () => {
  it('separates data loss from the L2 term', () => {
    const net = new Network(base({ layers: [{ units: 1, activation: 'linear' }], l2: 0.5 }));
    net.layers[0]!.setWeights(fromRows([[2], [0]]), fromRows([[0]]));
    const x = fromRows([[1, 0]]);
    const y = fromRows([[0]]);
    const yHat = net.forward(x);
    // ŷ = 2 ; data loss = ½(2)² = 2 ; L2 = (0.5/2)·(2² + 0²) = 1
    expect(net.dataLoss(yHat, y)).toBeCloseTo(2, 12);
    expect(net.l2Penalty()).toBeCloseTo(1, 12);
    expect(net.objective(yHat, y)).toBeCloseTo(3, 12);
  });

  it('reports zero penalty when L2 is off', () => {
    const net = new Network(base());
    expect(net.l2Penalty()).toBe(0);
  });
});
