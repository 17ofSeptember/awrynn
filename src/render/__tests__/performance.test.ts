import { describe, expect, it } from 'vitest';
import { computeLayout } from '../layout';
import { HitIndex } from '../hit';
import { Network } from '../../engine/network';
import { createMatrix } from '../../engine/tensor';
import { NetworkView } from '../networkView';
import { weightStroke, weightWidth } from '../theme';

/*
 * Spec §10 performance targets, as tests rather than as ad-hoc measurements:
 *
 *   60fps sustained with 6 layers x 12 units in full-speed training
 *   boundary recompute < 16ms
 *   interaction latency (hover highlight, weight scrub) < 16ms
 *   main thread never blocked more than 8ms by training
 *
 * These are budgets, not benchmarks. The bounds below are generous multiples of
 * what a developer machine measures, because a CI runner is slower and a test
 * that fails on someone else's hardware is a test people delete. They exist to
 * catch an accidental O(n²) rewrite, not to police milliseconds.
 */

/**
 * Wall-clock cost of `runs` iterations, in milliseconds each.
 *
 * The warm-up is not a formality. A single call primes the interpreter but not
 * the optimising compiler, and a ten-iteration measurement taken straight
 * afterwards is dominated by V8 tiering up rather than by the code under test:
 * the boundary-grid budget measured 5.5ms in steady state and intermittently
 * 16.2ms cold, which made it flaky against its own 16ms bound.
 *
 * Steady state is also the honest thing to measure here. These budgets are
 * about whether the app holds 60fps during SUSTAINED interaction, which is a
 * hot path by definition. A cold first call happens once.
 */
function timePer(runs: number, body: () => void): number {
  for (let i = 0; i < Math.min(runs, 20); i++) body();
  const started = performance.now();
  for (let i = 0; i < runs; i++) body();
  return (performance.now() - started) / runs;
}

describe('§10 performance budgets', () => {
  it('a 6x12 forward pass leaves room for hundreds of epochs per second', () => {
    // The §10 reference architecture for the 60fps target.
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 12, activation: 'tanh' },
        { units: 12, activation: 'tanh' },
        { units: 12, activation: 'tanh' },
        { units: 12, activation: 'tanh' },
        { units: 12, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 1,
      init: { kind: 'glorot_uniform' },
    });
    const x = createMatrix(32, 2);
    for (let i = 0; i < x.data.length; i++) x.data[i] = (i % 13) / 13 - 0.5;
    const y = createMatrix(32, 1);

    const perStep = timePer(200, () => {
      net.forward(x, true);
      net.backward(y);
    });
    // One batch of 32 through six layers. At 1ms a 240-sample epoch is ~8
    // batches, comfortably inside "hundreds of epochs per second".
    expect(perStep).toBeLessThan(2);
  });

  it('a 120x120 boundary grid stays inside the 16ms recompute budget', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 8, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 2,
      init: { kind: 'glorot_uniform' },
    });
    const grid = createMatrix(120 * 120, 2);
    for (let i = 0; i < grid.data.length; i++) grid.data[i] = (i % 97) / 97 - 0.5;
    expect(timePer(10, () => void net.forward(grid, false))).toBeLessThan(16);
  });

  it('hit-testing is far inside the 16ms interaction budget', () => {
    // The glyph network is the largest layout the app produces.
    const layout = computeLayout({ sizes: [35, 24, 24, 10] });
    const index = new HitIndex(layout);
    const perPick = timePer(2000, () => {
      const node = layout.nodes[0];
      if (node !== undefined) index.pick(node.x, node.y);
    });
    expect(perPick).toBeLessThan(1);
  });

  it('building the hit index after a layout change is not a stall', () => {
    const layout = computeLayout({ sizes: [35, 24, 24, 10] });
    expect(timePer(20, () => void new HitIndex(layout))).toBeLessThan(50);
  });

  it('the weight encoding allocates nothing per edge', () => {
    // §6.1. Identical inputs must return the identical string instance, which
    // is only true if it came from the precomputed ramp.
    expect(weightStroke(0.4, 1)).toBe(weightStroke(0.4, 1));
    const perCall = timePer(200_000, () => {
      weightStroke(0.37, 1.2);
      weightWidth(0.37, 1.2);
    });
    expect(perCall).toBeLessThan(0.001);
  });

  it('a full frame of edge styling stays well inside one frame', () => {
    // 35-24-24-10 is 35*24 + 24*24 + 24*10 = 1656 weight edges.
    const layout = computeLayout({ sizes: [35, 24, 24, 10] });
    const perFrame = timePer(200, () => {
      for (const edge of layout.edges) {
        weightStroke(edge.row * 0.01 - 0.5, 1);
        weightWidth(edge.row * 0.01 - 0.5, 1);
      }
    });
    expect(layout.edges.length).toBeGreaterThan(1500);
    expect(perFrame).toBeLessThan(4);
  });

  it('capturing a sample for the canvas is not a per-frame cost', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 12, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 3,
      init: { kind: 'glorot_uniform' },
    });
    const view = new NetworkView(net);
    const x = createMatrix(64, 2);
    for (let i = 0; i < x.data.length; i++) x.data[i] = (i % 11) / 11;
    expect(timePer(500, () => void view.captureSample(x, 0))).toBeLessThan(1);
  });
});
