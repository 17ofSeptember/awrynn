import { describe, expect, it } from 'vitest';
import { checkXor, CONVINCING_MARGIN, XOR_CASES, XOR_SOLUTION } from '../xor';
import { generateDataset } from '../datasets/index';
import { Network } from '../network';
import { fromRows } from '../tensor';

function xorNetwork(): Network {
  return new Network({
    inputSize: 2,
    layers: [
      { units: 2, activation: 'tanh' },
      { units: 1, activation: 'sigmoid' },
    ],
    loss: 'bce',
    seed: 1,
    init: { kind: 'zeros' },
  });
}

function applySolution(net: Network): void {
  net.layers[0]!.setWeights(
    fromRows(XOR_SOLUTION.hidden.weights.map((r) => [...r])),
    fromRows([[...XOR_SOLUTION.hidden.biases]]),
  );
  net.layers[1]!.setWeights(
    fromRows(XOR_SOLUTION.output.weights.map((r) => [...r])),
    fromRows([[...XOR_SOLUTION.output.biases]]),
  );
}

describe('the checker and the dataset describe the SAME problem', () => {
  /*
   * The regression this exists for: XOR_CASES once tested the unit square while
   * the xor dataset generated the signed square. The checker reported a
   * hand-built network as solving all four corners while the decision boundary
   * visibly misclassified half the points. Two definitions of "the four
   * corners" is one too many.
   */
  it('checks the corners the dataset actually generates', () => {
    const dataset = generateDataset({ name: 'xor', samples: 400, noise: 0, seed: 1 });
    const seen = new Map<string, number>();
    for (let i = 0; i < dataset.x.rows; i++) {
      const key = `${dataset.x.data[i * 2]},${dataset.x.data[i * 2 + 1]}`;
      seen.set(key, dataset.labels![i] as number);
    }
    expect(seen.size).toBe(4);
    for (const testCase of XOR_CASES) {
      const key = `${testCase.inputs[0]},${testCase.inputs[1]}`;
      expect(seen.has(key), `dataset has no corner at ${key}`).toBe(true);
      expect(seen.get(key), `label mismatch at ${key}`).toBe(testCase.target);
    }
  });

  it('the hand-built solution also classifies the real dataset correctly', () => {
    // The end-to-end claim: solving the four corners means the boundary drawn
    // over the actual data is right too.
    const net = xorNetwork();
    applySolution(net);
    const dataset = generateDataset({ name: 'xor', samples: 200, noise: 0.05, seed: 2 });
    const output = net.forward(dataset.x, false);
    let correct = 0;
    for (let i = 0; i < dataset.x.rows; i++) {
      const predicted = (output.data[i] as number) >= 0.5 ? 1 : 0;
      if (predicted === dataset.labels![i]) correct++;
    }
    expect(correct / dataset.x.rows).toBeGreaterThan(0.98);
  });
});

describe('the XOR challenge (§6.5)', () => {
  it('checks all four corners', () => {
    const report = checkXor(xorNetwork());
    expect(report.cases.length).toBe(4);
    expect(report.cases.map((c) => c.inputs)).toEqual(XOR_CASES.map((c) => c.inputs));
  });

  it('a zero network gets exactly the two cases whose target is 0', () => {
    // Every output is 0.5, which thresholds to 1, so only the target-1 cases
    // are "right" and for the wrong reason. The margin is what exposes that.
    const report = checkXor(xorNetwork());
    expect(report.solved).toBe(false);
    expect(report.worstMargin).toBeCloseTo(0, 10);
  });

  it('the hand-built solution solves all four, convincingly', () => {
    const net = xorNetwork();
    applySolution(net);
    const report = checkXor(net);
    expect(report.solved).toBe(true);
    expect(report.solvedCount).toBe(4);
    // Not merely past the threshold: firmly on the right side of it.
    expect(report.worstMargin).toBeGreaterThan(CONVINCING_MARGIN);
  });

  it('reports the margin, so a 0.501 win is visible as a weak one', () => {
    const net = xorNetwork();
    applySolution(net);
    const strong = checkXor(net).worstMargin;

    // Shrink the output weights so the sigmoid sits nearer the threshold.
    net.layers[1]!.setWeights(fromRows([[0.35], [0.35]]), fromRows([[-0.52]]));
    const weak = checkXor(net);
    expect(weak.worstMargin).toBeLessThan(strong);
    expect(weak.worstMargin).toBeLessThan(CONVINCING_MARGIN);
  });

  it('names each case it gets wrong', () => {
    const net = xorNetwork();
    applySolution(net);
    // Break the NAND unit.
    net.layers[0]!.setWeights(fromRows([[6, 0], [6, 0]]), fromRows([[-3, 0]]));
    const report = checkXor(net);
    expect(report.solved).toBe(false);
    const wrong = report.cases.filter((c) => !c.correct);
    expect(wrong.length).toBeGreaterThan(0);
    for (const c of wrong) expect(c.predicted).not.toBe(c.target);
  });

  it('refuses a network that is not shaped for XOR, and says why', () => {
    const wide = new Network({
      inputSize: 5,
      layers: [{ units: 1, activation: 'sigmoid' }],
      loss: 'bce',
      seed: 1,
      init: { kind: 'zeros' },
    });
    expect(checkXor(wide).problem).toMatch(/two inputs, but this network takes 5/);

    const multi = new Network({
      inputSize: 2,
      layers: [{ units: 3, activation: 'softmax' }],
      loss: 'cce',
      seed: 1,
      init: { kind: 'zeros' },
    });
    expect(checkXor(multi).problem).toMatch(/one output, but this network has 3/);
  });

  it('a single layer cannot solve XOR, however it is weighted (§7.2)', () => {
    // The premise of the whole lesson: no line separates the four corners.
    const linear = new Network({
      inputSize: 2,
      layers: [{ units: 1, activation: 'sigmoid' }],
      loss: 'bce',
      seed: 1,
      init: { kind: 'zeros' },
    });
    let bestSolved = 0;
    for (let w1 = -8; w1 <= 8; w1 += 0.5) {
      for (let w2 = -8; w2 <= 8; w2 += 0.5) {
        for (let b = -8; b <= 8; b += 0.5) {
          linear.layers[0]!.setWeights(fromRows([[w1], [w2]]), fromRows([[b]]));
          bestSolved = Math.max(bestSolved, checkXor(linear).solvedCount);
        }
      }
    }
    expect(bestSolved).toBe(3);
  });
});
