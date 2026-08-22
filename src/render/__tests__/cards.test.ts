import { describe, expect, it } from 'vitest';
import { dissect } from '../dissection';
import type { NeuronDissection } from '../dissection';
import {
  drawActivationPlot,
  drawFormulaCard,
  drawGradientCard,
  drawLossCard,
  MAX_VISIBLE_TERMS,
  measureFormulaCard,
} from '../draw/cards';
import { PulsePool, pulseRadius, PULSE_MAX_RADIUS, PULSE_MIN_RADIUS, drawPulses } from '../draw/pulses';
import { computeLayout } from '../layout';
import { Network } from '../../engine/network';
import { fromRows } from '../../engine/tensor';
import type { Ctx2D } from '../draw/context';

class Recorder implements Ctx2D {
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 1;
  lineCap: CanvasLineCap = 'butt';
  lineJoin: CanvasLineJoin = 'miter';
  font = '';
  textAlign: CanvasTextAlign = 'start';
  textBaseline: CanvasTextBaseline = 'alphabetic';
  globalAlpha = 1;
  readonly texts: string[] = [];
  readonly arcs: { x: number; y: number; r: number }[] = [];
  save(): void {}
  restore(): void {}
  beginPath(): void {}
  closePath(): void {}
  moveTo(): void {}
  lineTo(): void {}
  arc(x: number, y: number, r: number): void { this.arcs.push({ x, y, r }); }
  rect(): void {}
  fill(): void {}
  stroke(): void {}
  fillRect(): void {}
  clearRect(): void {}
  fillText(text: string): void { this.texts.push(text); }
  measureText(t: string): { width: number } { return { width: t.length * 6 }; }
  setLineDash(): void {}
  setTransform(): void {}
}

function twoInputNeuron(): NeuronDissection {
  const net = new Network({
    inputSize: 2,
    layers: [{ units: 1, activation: 'tanh' }],
    loss: 'mse',
    seed: 1,
    init: { kind: 'glorot_uniform' },
  });
  net.layers[0]!.setWeights(fromRows([[0.82], [-0.31]]), fromRows([[0.1]]));
  const d = dissect(net, fromRows([[1.4, 0.55]]), fromRows([[1]]), 0, 0.1);
  return d.neurons[0] as NeuronDissection;
}

const PLACE = { x: 0, y: 0, width: 208, height: 140 };

describe('formula card (§6.3 A.3)', () => {
  it('shows the spec’s worked example verbatim', () => {
    // §6.3: "z = (0.82)(1.40) + (−0.31)(0.55) + 0.10"
    const neuron = twoInputNeuron();
    const ctx = new Recorder();
    drawFormulaCard(ctx, { neuron, assembleProgress: 1, activateProgress: 0, collapsed: false }, PLACE);
    const joined = ctx.texts.join(' ');
    expect(joined).toContain('(+0.82)(+1.40)');
    expect(joined).toContain('(−0.31)(+0.55)');
    expect(joined).toContain('+0.100'); // the bias
  });

  it('assembles term by term rather than appearing whole', () => {
    const neuron = twoInputNeuron();
    const early = new Recorder();
    drawFormulaCard(early, { neuron, assembleProgress: 0.1, activateProgress: 0, collapsed: false }, PLACE);
    const late = new Recorder();
    drawFormulaCard(late, { neuron, assembleProgress: 1, activateProgress: 0, collapsed: false }, PLACE);
    expect(early.texts.length).toBeLessThan(late.texts.length);
  });

  it('resolves to the engine’s cached z, not to its own running total', () => {
    const neuron = twoInputNeuron();
    const ctx = new Recorder();
    drawFormulaCard(ctx, { neuron, assembleProgress: 1, activateProgress: 0, collapsed: false }, PLACE);
    // z = 0.82(1.4) + (−0.31)(0.55) + 0.1 = 1.148 − 0.1705 + 0.1 = 1.0775
    expect(neuron.z).toBeCloseTo(1.0775, 12);
    expect(ctx.texts.join(' ')).toContain('+1.0775');
  });

  it('summarises a wide layer instead of drawing a card taller than the screen', () => {
    const net = new Network({
      inputSize: 35,
      layers: [{ units: 1, activation: 'relu' }],
      loss: 'mse',
      seed: 2,
      init: { kind: 'he_normal' },
    });
    const x = fromRows([Array.from({ length: 35 }, (_, i) => (i % 2 === 0 ? 1 : 0))]);
    const d = dissect(net, x, fromRows([[1]]), 0, 0.1);
    const neuron = d.neurons[0] as NeuronDissection;

    const measured = measureFormulaCard({
      neuron,
      assembleProgress: 1,
      activateProgress: 0,
      collapsed: false,
    });
    expect(measured.height).toBeLessThan(200);

    const ctx = new Recorder();
    drawFormulaCard(ctx, { neuron, assembleProgress: 1, activateProgress: 0, collapsed: false }, PLACE);
    const joined = ctx.texts.join(' ');
    // Summarised, not omitted — the arithmetic on screen still balances.
    expect(joined).toContain(`+ ${35 - MAX_VISIBLE_TERMS} more terms`);
  });

  it('collapses to a compact z / a chip (§6.3 A.5)', () => {
    const neuron = twoInputNeuron();
    const ctx = new Recorder();
    drawFormulaCard(ctx, { neuron, assembleProgress: 1, activateProgress: 1, collapsed: true }, PLACE);
    const joined = ctx.texts.join(' ');
    expect(joined).toContain('z');
    expect(joined).toContain('a');
    expect(ctx.texts.length).toBeLessThan(8);
    expect(measureFormulaCard({ neuron, assembleProgress: 1, activateProgress: 1, collapsed: true }).height)
      .toBeLessThan(40);
  });
});

describe('mini activation plot (§6.3 A.4)', () => {
  it('draws the curve and a travelling dot', () => {
    const neuron = twoInputNeuron();
    const ctx = new Recorder();
    drawActivationPlot(ctx, 0, 0, neuron, Math.tanh, [-1, 1], 1);
    expect(ctx.arcs.length).toBe(1); // the dot
    expect(ctx.texts.join(' ')).toContain('TANH');
  });

  it('moves the dot along the axis then up the curve', () => {
    const neuron = twoInputNeuron();
    const early = new Recorder();
    drawActivationPlot(early, 0, 0, neuron, Math.tanh, [-1, 1], 0.25);
    const late = new Recorder();
    drawActivationPlot(late, 0, 0, neuron, Math.tanh, [-1, 1], 1);
    // The dot rises only in the second half.
    expect(late.arcs[0]!.y).toBeLessThan(early.arcs[0]!.y);
  });

  it('keeps an unbounded activation on the plot', () => {
    // A ReLU with a large z must not put the dot outside the box.
    const net = new Network({
      inputSize: 1,
      layers: [{ units: 1, activation: 'relu' }],
      loss: 'mse',
      seed: 3,
      init: { kind: 'constant', value: 12 },
    });
    const d = dissect(net, fromRows([[5]]), fromRows([[0]]), 0, 0.1);
    const ctx = new Recorder();
    drawActivationPlot(ctx, 0, 0, d.neurons[0]!, (z) => Math.max(0, z), null, 1);
    const dot = ctx.arcs[0]!;
    expect(dot.x).toBeGreaterThanOrEqual(0);
    expect(dot.x).toBeLessThanOrEqual(74);
    expect(dot.y).toBeGreaterThanOrEqual(0);
    expect(dot.y).toBeLessThanOrEqual(74);
  });
});

describe('loss and gradient cards', () => {
  it('shows prediction, target and the real loss', () => {
    const net = new Network({
      inputSize: 2,
      layers: [{ units: 1, activation: 'linear' }],
      loss: 'mse',
      seed: 4,
      init: { kind: 'glorot_uniform' },
    });
    net.layers[0]!.setWeights(fromRows([[1], [0]]), fromRows([[0]]));
    const d = dissect(net, fromRows([[2, 0]]), fromRows([[0]]), 0, 0.1);
    const ctx = new Recorder();
    drawLossCard(ctx, 0, 0, d.output, 1);
    const joined = ctx.texts.join(' ');
    expect(joined).toContain('MSE');
    expect(joined).toContain('prediction');
    // ŷ = 2, y = 0 → ℓ = ½(2)² = 2
    expect(joined).toContain('2.0000');
  });

  it('shows ∂L/∂w and Δw as equations with their operands', () => {
    const net = new Network({
      inputSize: 2,
      layers: [{ units: 1, activation: 'linear' }],
      loss: 'mse',
      seed: 5,
      init: { kind: 'glorot_uniform' },
    });
    const d = dissect(net, fromRows([[1, 1]]), fromRows([[0]]), 0, 0.25);
    const ctx = new Recorder();
    drawGradientCard(ctx, 0, 0, d.gradients[0]!, 0.25, 1);
    const joined = ctx.texts.join(' ');
    expect(joined).toContain('∂L/∂w = (');
    expect(joined).toContain('Δw = −0.250 × ∂L/∂w');
  });
});

describe('pulse pool (§6.1 — zero allocation in the hot path)', () => {
  it('reuses objects rather than allocating', () => {
    const pool = new PulsePool(3);
    const a = pool.acquire();
    pool.release(a!);
    const b = pool.acquire();
    expect(b).toBe(a);
  });

  it('reports a full pool instead of growing', () => {
    const pool = new PulsePool(2);
    expect(pool.acquire()).not.toBeNull();
    expect(pool.acquire()).not.toBeNull();
    expect(pool.acquire()).toBeNull();
    expect(pool.capacity).toBe(2);
  });

  it('clears without discarding its objects', () => {
    const pool = new PulsePool(4);
    const first = pool.acquire();
    pool.acquire();
    expect(pool.activeCount).toBe(2);
    pool.clear();
    expect(pool.activeCount).toBe(0);
    expect(pool.acquire()).toBe(first);
  });

  it('sizes pulses by contribution magnitude, not by weight', () => {
    // §6.3 A.2: a large weight on a near-zero input contributes nothing.
    expect(pulseRadius(0)).toBe(PULSE_MIN_RADIUS);
    expect(pulseRadius(1)).toBe(PULSE_MAX_RADIUS);
    expect(pulseRadius(5)).toBe(PULSE_MAX_RADIUS);
    expect(pulseRadius(-1)).toBe(PULSE_MIN_RADIUS);
    // Area tracks magnitude, so radius grows as sqrt.
    expect(pulseRadius(0.25) - PULSE_MIN_RADIUS).toBeCloseTo(
      (pulseRadius(1) - PULSE_MIN_RADIUS) * 0.5,
      10,
    );
  });

  it('colours pulses by contribution sign', () => {
    const layout = computeLayout({ sizes: [2, 2], showBiases: false });
    const pool = new PulsePool(4);
    const negative = pool.acquire()!;
    Object.assign(negative, { edge: 0, t: 0.5, magnitude: 1, sign: -1 });
    const positive = pool.acquire()!;
    Object.assign(positive, { edge: 1, t: 0.5, magnitude: 1, sign: 1 });

    const styles: string[] = [];
    const ctx = new Recorder();
    Object.defineProperty(ctx, 'fillStyle', {
      get: () => '',
      set: (v: string) => { styles.push(v); },
    });
    drawPulses(ctx, layout, { scale: 1, offsetX: 0, offsetY: 0 }, pool);
    expect(styles.some((s) => s.includes('62, 197, 232'))).toBe(true);
    expect(styles.some((s) => s.includes('242, 163, 60'))).toBe(true);
  });
});
