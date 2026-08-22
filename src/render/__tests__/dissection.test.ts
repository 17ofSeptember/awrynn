import { describe, expect, it } from 'vitest';
import { dissect, dissectionIsFaithful, RESIDUAL_TOLERANCE } from '../dissection';
import { Network } from '../../engine/network';
import type { LayerSpec } from '../../engine/layers';
import type { LossName } from '../../engine/losses';
import { createRng } from '../../engine/rng';
import { createMatrix, fromRows } from '../../engine/tensor';

/*
 * Spec §11 Phase 4 gate:
 *
 *   "every number shown during a dissection matches the engine's cached values
 *    exactly (verify by logging both and diffing)"
 *
 * `residual` IS that diff, computed per neuron: the sum of the terms a formula
 * card would display, minus the z the engine actually cached. These tests
 * assert it across the same matrix of configurations the gradient check uses,
 * because a card that shows arithmetic which does not add up is exactly the
 * "fabricated number" §0.6 forbids.
 */

function build(layers: LayerSpec[], loss: LossName, seed: number, inputSize = 2): Network {
  return new Network({ inputSize, layers, loss, seed, init: { kind: 'glorot_uniform' } });
}

function sample(net: Network, seed: number, samples = 4): { x: ReturnType<typeof createMatrix>; y: ReturnType<typeof createMatrix> } {
  const rng = createRng(seed).stream('data');
  const x = createMatrix(samples, net.inputSize);
  for (let i = 0; i < x.data.length; i++) x.data[i] = rng.uniform(-1.5, 1.5);
  const y = createMatrix(samples, net.outputSize);
  if (net.lossName === 'cce') {
    for (let r = 0; r < samples; r++) y.data[r * net.outputSize + rng.int(net.outputSize)] = 1;
  } else if (net.lossName === 'bce') {
    for (let r = 0; r < samples; r++) y.data[r] = rng.next() < 0.5 ? 0 : 1;
  } else {
    for (let i = 0; i < y.data.length; i++) y.data[i] = rng.uniform(-1, 1);
  }
  return { x, y };
}

describe('fidelity — the Phase 4 gate', () => {
  it('every displayed term sums to the engine’s cached z, across the full matrix', () => {
    const cases: { layers: LayerSpec[]; loss: LossName }[] = [];
    for (const activation of ['linear', 'relu', 'leaky_relu', 'tanh', 'sigmoid'] as const) {
      cases.push({
        layers: [
          { units: 6, activation },
          { units: 1, activation: 'sigmoid' },
        ],
        loss: 'bce',
      });
      cases.push({
        layers: [
          { units: 5, activation },
          { units: 4, activation },
          { units: 3, activation: 'softmax' },
        ],
        loss: 'cce',
      });
      cases.push({
        layers: [
          { units: 4, activation },
          { units: 1, activation },
        ],
        loss: 'mse',
      });
    }

    // The same matrix again with batch normalization on, because a normalized
    // layer's z is NOT the sum of the terms above it. The card has to show the
    // normalization for the arithmetic on screen to close, and the residual is
    // what proves it does.
    const withBatchNorm = cases.map((testCase) => ({
      loss: testCase.loss,
      layers: testCase.layers.map((layer, i) =>
        i < testCase.layers.length - 1 ? { ...layer, batchNorm: true } : layer,
      ),
    }));

    let worst = 0;
    [...cases, ...withBatchNorm].forEach((testCase, i) => {
      const net = build(testCase.layers, testCase.loss, 100 + i);
      const { x, y } = sample(net, 200 + i);
      const d = dissect(net, x, y, 0, 0.1);
      worst = Math.max(worst, d.maxResidual);
      expect(dissectionIsFaithful(d), `${testCase.loss} case ${i}`).toBe(true);
    });
    // Reassociation only; nothing near a real disagreement.
    expect(worst).toBeLessThan(RESIDUAL_TOLERANCE);
  });

  it('reads z and a straight from the engine caches, not from its own sum', () => {
    const net = build([{ units: 3, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }], 'bce', 5);
    const { x, y } = sample(net, 6);
    const d = dissect(net, x, y, 0, 0.1);

    net.layers.forEach((layer, layerIndex) => {
      for (let unit = 0; unit < layer.units; unit++) {
        const neuron = d.neurons.find((n) => n.layer === layerIndex && n.unit === unit);
        expect(neuron).toBeDefined();
        // Identity with the cache, not approximate agreement.
        expect(neuron!.z).toBe(layer.Z!.data[unit]);
        expect(neuron!.a).toBe(layer.A!.data[unit]);
        expect(neuron!.bias).toBe(layer.b.data[unit]);
      }
    });
  });

  it('reads every weight straight from the engine', () => {
    const net = build([{ units: 4, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }], 'bce', 7);
    const { x, y } = sample(net, 8);
    const d = dissect(net, x, y, 0, 0.1);
    for (const neuron of d.neurons) {
      const layer = net.layers[neuron.layer]!;
      for (const term of neuron.terms) {
        expect(term.weight).toBe(layer.W.data[term.source * layer.units + neuron.unit]);
      }
    }
  });

  it('reads gradients from the engine’s dW, already divided by B', () => {
    const net = build([{ units: 3, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }], 'bce', 9);
    const { x, y } = sample(net, 10);
    const eta = 0.05;
    const d = dissect(net, x, y, 0, eta);
    for (const g of d.gradients) {
      const layer = net.layers[g.layer]!;
      expect(g.gradient).toBe(layer.dW.data[g.row * layer.units + g.col]);
      expect(g.step).toBeCloseTo(-eta * g.gradient, 15);
      expect(g.weightBefore).toBe(layer.W.data[g.row * layer.units + g.col]);
    }
  });

  it('shows ∂L/∂w = δ·a for a single sample, which is what the card claims', () => {
    // With B = 1 the batch average is the identity, so the edge card's
    // "∂L/∂w = δ·a" is literally true rather than an approximation.
    const net = build([{ units: 3, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }], 'bce', 11);
    const { x, y } = sample(net, 12);
    const d = dissect(net, x, y, 0, 0.1);
    for (const g of d.gradients) {
      expect(g.gradient).toBeCloseTo(g.delta * g.input, 12);
    }
  });

  it('hand-checks one neuron end to end', () => {
    const net = build([{ units: 1, activation: 'linear' }], 'mse', 1);
    net.layers[0]!.setWeights(fromRows([[2], [-3]]), fromRows([[0.5]]));
    const x = fromRows([[1.5, 0.25]]);
    const y = fromRows([[0]]);
    const d = dissect(net, x, y, 0, 0.1);

    // z = (2)(1.5) + (−3)(0.25) + 0.5 = 3 − 0.75 + 0.5 = 2.75
    const neuron = d.neurons[0]!;
    expect(neuron.terms[0]!.contribution).toBeCloseTo(3, 12);
    expect(neuron.terms[1]!.contribution).toBeCloseTo(-0.75, 12);
    expect(neuron.assembled).toBeCloseTo(2.75, 12);
    expect(neuron.z).toBeCloseTo(2.75, 12);
    expect(neuron.a).toBeCloseTo(2.75, 12); // linear
    expect(neuron.residual).toBeLessThan(RESIDUAL_TOLERANCE);

    // mse: ℓ = ½(2.75 − 0)² = 3.78125
    expect(d.output.loss).toBeCloseTo(3.78125, 12);
    expect(d.output.prediction).toBeCloseTo(2.75, 12);
  });
});

describe('dissection mechanics', () => {
  it('silences dropout and restores it afterwards', () => {
    // Dropout would make the displayed arithmetic irreproducible.
    const net = build([{ units: 6, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }], 'bce', 13);
    net.setHiddenDropout(0.5);
    const { x, y } = sample(net, 14);

    const a = dissect(net, x, y, 0, 0.1);
    const b = dissect(net, x, y, 0, 0.1);
    // Same sample twice must give identical numbers.
    expect(a.neurons.map((n) => n.z)).toEqual(b.neurons.map((n) => n.z));
    expect(net.layers[0]!.dropout).toBe(0.5);
  });

  it('dissects the row it was asked for', () => {
    const net = build([{ units: 2, activation: 'linear' }], 'mse', 15);
    const x = fromRows([
      [1, 0],
      [0, 1],
    ]);
    const y = fromRows([[0, 0], [0, 0]]);
    const first = dissect(net, x, y, 0, 0.1);
    const second = dissect(net, x, y, 1, 0.1);
    expect(first.inputs).toEqual([1, 0]);
    expect(second.inputs).toEqual([0, 1]);
    expect(first.neurons[0]!.z).not.toBe(second.neurons[0]!.z);
  });

  it('rejects a row, width or target that does not fit', () => {
    const net = build([{ units: 1, activation: 'sigmoid' }], 'bce', 16);
    const { x, y } = sample(net, 17);
    expect(() => dissect(net, x, y, 99, 0.1)).toThrowError(/outside \[0, 4\)/);
    expect(() => dissect(net, createMatrix(2, 9), y, 0, 0.1)).toThrowError(/9 features/);
    expect(() => dissect(net, x, createMatrix(4, 7), 0, 0.1)).toThrowError(/does not match/);
  });

  it('covers every neuron and every weight', () => {
    const net = build(
      [
        { units: 5, activation: 'tanh' },
        { units: 3, activation: 'softmax' },
      ],
      'cce',
      18,
    );
    const { x, y } = sample(net, 19);
    const d = dissect(net, x, y, 0, 0.1);
    expect(d.neurons.length).toBe(5 + 3);
    expect(d.gradients.length).toBe(2 * 5 + 5 * 3);
    for (const neuron of d.neurons) {
      expect(neuron.terms.length).toBe(net.layers[neuron.layer]!.inputs);
    }
  });
});

describe('the dissection of a normalized layer', () => {
  const build2 = (): Network =>
    new Network({
      inputSize: 2,
      layers: [
        { units: 4, activation: 'tanh', batchNorm: true },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 51,
      init: { kind: 'glorot_uniform' },
    });

  it('reports the normalization, and only where there is one', () => {
    const net = build2();
    const { x, y } = sample(net, 52);
    const d = dissect(net, x, y, 0, 0.1);

    const normalized = d.neurons.filter((n) => n.layer === 0);
    const plain = d.neurons.filter((n) => n.layer === 1);
    expect(normalized.every((n) => n.normalization !== null)).toBe(true);
    expect(plain.every((n) => n.normalization === null)).toBe(true);
  });

  it('closes: γ·(u − μ)/σ + b equals the engine’s cached z', () => {
    const net = build2();
    const { x, y } = sample(net, 53);
    const d = dissect(net, x, y, 0, 0.1);

    for (const neuron of d.neurons) {
      const norm = neuron.normalization;
      if (norm === null) continue;
      // u is what the terms add up to, before anything is done to it.
      const summed = neuron.terms.reduce((total, t) => total + t.contribution, 0);
      expect(summed).toBeCloseTo(norm.u, 12);
      const rebuilt = norm.gamma * ((norm.u - norm.mean) / norm.sigma) + neuron.bias;
      expect(rebuilt, `layer ${neuron.layer} unit ${neuron.unit}`).toBeCloseTo(neuron.z, 10);
    }
  });

  it('says the statistics came from the running estimate, because one sample has none', () => {
    // The dissection steps ONE sample through, so there is no batch to take a
    // mean over. Reporting that honestly is what stops a reader comparing these
    // numbers against a training batch and concluding something is broken.
    const net = build2();
    const { x, y } = sample(net, 54);
    const d = dissect(net, x, y, 0, 0.1);
    const first = d.neurons.find((n) => n.normalization !== null);
    expect(first?.normalization?.fromBatch).toBe(false);
  });
});
