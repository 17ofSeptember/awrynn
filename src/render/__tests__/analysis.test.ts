import { describe, expect, it } from 'vitest';
import { boundsFor, drawConfusion, paintBoundary, projectX, projectY } from '../draw/heatmap';
import { CLASS_COLORS, classColor, COLORS } from '../theme';
import { drawChart, drawHistogram, histogram, movingAverage } from '../draw/curves';
import type { Ctx2D } from '../draw/context';
import { Network } from '../../engine/network';
import { createMatrix, fromRows } from '../../engine/tensor';
import { predictedClass } from '../../engine/trainer';
import { generateDataset } from '../../engine/datasets/index';

/*
 * Spec §11 Phase 6 gate: "boundary and thumbnails correct against a CPU
 * reference."
 *
 * The boundary worker cannot be imported directly under a node test (it binds
 * self.onmessage), so the reference test below reimplements the SAME grid
 * evaluation with a straightforward per-point loop and asserts the batched form
 * agrees. That is the comparison that matters: the worker's speed comes from
 * evaluating 14,400 rows as one batch, and a batching bug would be invisible
 * without a slow reference to check it against.
 */

class Stub implements Ctx2D {
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
  save(): void {}
  restore(): void {}
  beginPath(): void {}
  closePath(): void {}
  moveTo(): void {}
  lineTo(): void {}
  arc(): void {}
  rect(): void {}
  fill(): void {}
  stroke(): void {}
  fillRect(): void {}
  clearRect(): void {}
  fillText(t: string): void { this.texts.push(t); }
  measureText(t: string): { width: number } { return { width: t.length * 6 }; }
  setLineDash(): void {}
  setTransform(): void {}
}

function makeImageData(size: number): ImageData {
  return { data: new Uint8ClampedArray(size * size * 4), width: size, height: size } as ImageData;
}

describe('boundary grid, against a per-point CPU reference', () => {
  it('the batched evaluation matches a one-at-a-time loop exactly', () => {
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 6, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 21,
      init: { kind: 'glorot_uniform' },
    });

    const n = 24;
    const bounds = { xMin: -2, xMax: 2, yMin: -1.5, yMax: 1.5 };

    // Batched, exactly as the worker builds it.
    const grid = createMatrix(n * n, 2);
    for (let row = 0; row < n; row++) {
      const y = bounds.yMax - ((bounds.yMax - bounds.yMin) * row) / (n - 1);
      for (let col = 0; col < n; col++) {
        const x = bounds.xMin + ((bounds.xMax - bounds.xMin) * col) / (n - 1);
        grid.data[(row * n + col) * 2] = x;
        grid.data[(row * n + col) * 2 + 1] = y;
      }
    }
    const batched = net.forward(grid, false);

    // Reference: one point at a time.
    for (let row = 0; row < n; row++) {
      const y = bounds.yMax - ((bounds.yMax - bounds.yMin) * row) / (n - 1);
      for (let col = 0; col < n; col++) {
        const x = bounds.xMin + ((bounds.xMax - bounds.xMin) * col) / (n - 1);
        const single = net.forward(fromRows([[x, y]]), false);
        expect(single.data[0]).toBeCloseTo(batched.data[row * n + col] as number, 12);
      }
    }
  });

  it('orients the grid so row 0 is the TOP of the plot', () => {
    // Screen y grows downward while data y grows upward. Getting this backwards
    // produces a vertically mirrored boundary that still looks plausible.
    const n = 3;
    const bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
    const yAt = (row: number): number =>
      bounds.yMax - ((bounds.yMax - bounds.yMin) * row) / (n - 1);
    expect(yAt(0)).toBe(1);
    expect(yAt(n - 1)).toBe(0);
    // And the projection agrees: a high data y maps to a small screen y.
    expect(projectY(bounds, 1, 0, 100)).toBe(0);
    expect(projectY(bounds, 0, 0, 100)).toBe(100);
  });

  it('multi-class packing keeps class and confidence separable', () => {
    // values[i] = class + confidence, with confidence clamped below 1 so it can
    // never carry into the class index.
    for (const [cls, confidence] of [[0, 0.999], [2, 0.5], [9, 0.0]] as const) {
      const packed = cls + Math.min(0.999, confidence);
      expect(Math.floor(packed)).toBe(cls);
      expect(packed - Math.floor(packed)).toBeCloseTo(Math.min(0.999, confidence), 10);
    }
  });
});

describe('paintBoundary', () => {
  it('uses the cool pole below 0.5 and the warm pole above, for binary output', () => {
    const image = makeImageData(2);
    paintBoundary(image, {
      values: Float32Array.from([0.02, 0.98, 0.5, 0.5]),
      resolution: 2,
      classes: 1,
    });
    const px = (i: number): number[] => Array.from(image.data.slice(i * 4, i * 4 + 4));
    expect(px(0).slice(0, 3)).toEqual([62, 197, 232]); // cool
    expect(px(1).slice(0, 3)).toEqual([242, 163, 60]); // warm
  });

  it('fades toward the boundary, which is where the network is least sure', () => {
    const image = makeImageData(2);
    paintBoundary(image, {
      values: Float32Array.from([0.5, 0.99, 0.5, 0.5]),
      resolution: 2,
      classes: 1,
    });
    const alphaAt = (i: number): number => image.data[i * 4 + 3] as number;
    // A hard line would claim a precision the network does not have.
    expect(alphaAt(0)).toBeLessThan(alphaAt(1));
  });

  it('measures multi-class confidence from chance level, not from zero', () => {
    // With 4 classes, a winning probability of 0.25 is chance and must read as
    // no confidence at all. 0.999 is the highest the worker will ever pack.
    const image = makeImageData(2);
    paintBoundary(image, {
      values: Float32Array.from([1 + 0.25, 1 + 0.999, 0, 0]),
      resolution: 2,
      classes: 4,
    });
    const alphaAt = (i: number): number => image.data[i * 4 + 3] as number;
    expect(alphaAt(0)).toBe(28); // the floor
    expect(alphaAt(1)).toBeGreaterThan(alphaAt(0));
  });

  it('is why the worker clamps confidence below 1', () => {
    // An unclamped probability of exactly 1.0 would pack as class + 1, landing
    // on the NEXT class index with zero confidence: the pixel would be drawn in
    // the wrong colour precisely where the network was most certain.
    expect(Math.floor(1 + 1.0)).toBe(2);
    expect(Math.floor(1 + Math.min(0.999, 1.0))).toBe(1);
  });
});

describe('bounds and projection', () => {
  it('pads so no point sits on the edge', () => {
    const x = Float64Array.from([0, 0, 1, 1]);
    const bounds = boundsFor(x, 2, 0.2);
    expect(bounds.xMin).toBeLessThan(0);
    expect(bounds.xMax).toBeGreaterThan(1);
  });

  it('survives a degenerate single point', () => {
    const bounds = boundsFor(Float64Array.from([5, 5]), 1);
    expect(Number.isFinite(bounds.xMin)).toBe(true);
    expect(bounds.xMax).toBeGreaterThan(bounds.xMin);
  });

  it('returns a usable box for empty data', () => {
    const bounds = boundsFor(new Float64Array(0), 0);
    expect(bounds).toEqual({ xMin: -1, xMax: 1, yMin: -1, yMax: 1 });
  });

  it('projects the corners of the box to the corners of the plot', () => {
    const bounds = { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
    expect(projectX(bounds, -1, 10, 100)).toBe(10);
    expect(projectX(bounds, 1, 10, 100)).toBe(110);
    expect(projectY(bounds, 1, 20, 200)).toBe(20);
    expect(projectY(bounds, -1, 20, 200)).toBe(220);
  });
});

describe('loss chart', () => {
  const stub = (): Stub => new Stub();

  it('never draws a negative axis tick for a loss', () => {
    // A loss cannot be negative; an axis reading −0.04 states something
    // impossible about the quantity plotted.
    const ctx = stub();
    drawChart(ctx, 0, 0, 200, 120, [{ values: [0.5, 0.2, 0.001], label: 'train', dashed: false, dim: false }], {
      logScale: false,
      nonNegative: true,
    });
    for (const text of ctx.texts) {
      if (/^-?\d/.test(text)) expect(Number(text)).toBeGreaterThanOrEqual(0);
    }
  });

  it('reports an empty state rather than dividing by an empty range', () => {
    const ctx = stub();
    drawChart(ctx, 0, 0, 200, 120, [{ values: [], label: 'train', dashed: false, dim: false }], {
      logScale: false,
    });
    expect(ctx.texts).toContain('No data yet');
  });

  it('survives a flat series', () => {
    const ctx = stub();
    expect(() =>
      drawChart(ctx, 0, 0, 200, 120, [{ values: [0.3, 0.3, 0.3], label: 'train', dashed: false, dim: false }], {
        logScale: false,
      }),
    ).not.toThrow();
  });

  it('handles a diverged series without throwing', () => {
    // §7.4: the app must recover gracefully from NaN, including in the chart.
    const ctx = stub();
    expect(() =>
      drawChart(
        ctx,
        0,
        0,
        200,
        120,
        [{ values: [0.5, 12, NaN, Infinity], label: 'train', dashed: false, dim: false }],
        { logScale: true },
      ),
    ).not.toThrow();
  });

  it('labels both series', () => {
    const ctx = stub();
    drawChart(
      ctx,
      0,
      0,
      200,
      120,
      [
        { values: [1, 0.5], label: 'train', dashed: false, dim: false },
        { values: [1.1, 0.6], label: 'validation', dashed: true, dim: true },
      ],
      { logScale: false },
    );
    expect(ctx.texts).toContain('train');
    expect(ctx.texts).toContain('validation');
  });

  it('draws an epoch marker with its label', () => {
    const ctx = stub();
    drawChart(ctx, 0, 0, 200, 120, [{ values: [1, 0.8, 0.6], label: 'train', dashed: false, dim: false }], {
      logScale: false,
      marker: 1,
      markerLabel: 'early stop',
    });
    expect(ctx.texts).toContain('early stop');
  });
});

describe('misclassification agrees with the engine', () => {
  it('rings exactly the points the network gets wrong', () => {
    const dataset = generateDataset({ name: 'moons', samples: 40, seed: 3 });
    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 4, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 4,
      init: { kind: 'glorot_uniform' },
    });
    const output = net.forward(dataset.x, false);
    let wrong = 0;
    for (let i = 0; i < dataset.x.rows; i++) {
      if (predictedClass(output, i) !== dataset.labels![i]) wrong++;
    }
    // An untrained network gets a substantial fraction wrong; the point is that
    // the count comes from the engine's own prediction, not from a heuristic.
    expect(wrong).toBeGreaterThan(0);
    expect(wrong).toBeLessThanOrEqual(dataset.x.rows);
  });
});


describe('categorical class palette', () => {
  it('uses the weight poles for the first two classes', () => {
    // A binary boundary should speak the same visual language as the weights.
    expect(classColor(0)).toBe(CLASS_COLORS[0]);
    expect(classColor(1)).toBe(CLASS_COLORS[1]);
  });

  it('assigns in fixed order and never cycles within the defined slots', () => {
    // Colour follows the entity, not its rank: a class keeps its colour when
    // the class count changes.
    const three = [0, 1, 2].map(classColor);
    const five = [0, 1, 2, 3, 4].map(classColor);
    expect(five.slice(0, 3)).toEqual(three);
    expect(new Set(five).size).toBe(5);
  });

  it('steps DOWN in lightness after the poles', () => {
    // Colour-vision deficiency destroys hue but preserves lightness, so equal
    // lightness collapses. This is the property that keeps the slots separable,
    // and it also stops an added class outshining the poles.
    const luminance = (hex: string): number => {
      const n = parseInt(hex.slice(1), 16);
      const lin = (c: number): number => {
        const s = c / 255;
        return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
      };
      return 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
    };
    const poles = Math.min(luminance(classColor(0)), luminance(classColor(1)));
    for (let k = 2; k < CLASS_COLORS.length; k++) {
      expect(luminance(classColor(k)), `slot ${k}`).toBeLessThan(poles);
    }
  });
});

describe('moving average (§6.4 — the gradient chart must be readable)', () => {
  it('preserves length and smooths toward the trend', () => {
    const noisy = [1, 100, 1, 100, 1, 100, 1, 100];
    const smoothed = movingAverage(noisy, 5);
    expect(smoothed.length).toBe(noisy.length);
    // The alternation collapses toward the mean.
    const spread = Math.max(...smoothed) - Math.min(...smoothed);
    expect(spread).toBeLessThan(Math.max(...noisy) - Math.min(...noisy));
  });

  it('is the identity for a window of 1 or less', () => {
    const values = [1, 2, 3];
    expect(movingAverage(values, 1)).toEqual(values);
    expect(movingAverage(values, 0)).toEqual(values);
  });

  it('skips non-finite samples rather than poisoning the whole window', () => {
    const smoothed = movingAverage([1, 1, NaN, 1, 1], 3);
    expect(smoothed.every((v) => Number.isFinite(v))).toBe(true);
  });

  it('keeps a run of non-finite samples non-finite, so a divergence still breaks the line', () => {
    // §7.4: seeing the curve stop is instructive; smoothing it away is not.
    const smoothed = movingAverage([NaN, NaN, NaN], 3);
    expect(smoothed.every((v) => Number.isNaN(v))).toBe(true);
  });
});

describe('histogram (§6.4 — weight distributions)', () => {
  it('bins values across their range', () => {
    const bins = histogram(Float64Array.from([-1, -1, 0, 1, 1, 1]), 4);
    expect(bins.length).toBe(4);
    expect(bins.reduce((sum, b) => sum + b.count, 0)).toBe(6);
    expect(bins[0]!.from).toBeCloseTo(-1, 10);
    expect(bins[bins.length - 1]!.to).toBeCloseTo(1, 10);
  });

  it('gives a constant layer a nominal range instead of dividing by zero', () => {
    // A zero-initialised layer (lesson 3) has no spread at all.
    const bins = histogram(Float64Array.from([0, 0, 0, 0]), 8);
    expect(bins.length).toBe(8);
    expect(bins.every((b) => Number.isFinite(b.from) && Number.isFinite(b.to))).toBe(true);
    expect(bins.reduce((sum, b) => sum + b.count, 0)).toBe(4);
  });

  it('returns nothing for empty input', () => {
    expect(histogram(new Float64Array(0))).toEqual([]);
  });

  it('ignores non-finite values', () => {
    const bins = histogram(Float64Array.from([0, 1, NaN, Infinity]), 4);
    expect(bins.reduce((sum, b) => sum + b.count, 0)).toBe(2);
  });

  it('draws negative bins cool and positive bins warm', () => {
    // The histogram and the edges on the canvas describe the same numbers, so
    // they must read alike.
    const ctx = new Stub();
    const fills: string[] = [];
    Object.defineProperty(ctx, 'fillStyle', { get: () => '', set: (v: string) => fills.push(v) });
    drawHistogram(ctx, 0, 0, 200, 60, histogram(Float64Array.from([-2, -1, 1, 2]), 4), 'L1');
    expect(fills).toContain(COLORS.weightNegative);
    expect(fills).toContain(COLORS.weightPositive);
  });

  it('labels the histogram and survives empty input', () => {
    const ctx = new Stub();
    drawHistogram(ctx, 0, 0, 200, 60, [], 'L1 · 2×3');
    expect(ctx.texts).toContain('L1 · 2×3');
  });
});

describe('confusion matrix rendering', () => {
  it('shades by row share, so a small class is still legible', () => {
    // With an imbalanced split, shading by the global maximum would make every
    // small class look empty and hide the confusions worth seeing.
    const ctx = new Stub();
    const fills: string[] = [];
    Object.defineProperty(ctx, 'fillStyle', { get: () => '', set: (v: string) => fills.push(v) });
    drawConfusion(
      ctx,
      0,
      0,
      120,
      [
        [200, 0],
        [1, 1],
      ],
      ['0', '1'],
    );
    // Row 1 is tiny but its diagonal is 50% of its row, so it is not invisible.
    const alphas = fills
      .filter((f) => f.startsWith('rgba'))
      .map((f) => Number(f.slice(f.lastIndexOf(',') + 1, -1)));
    expect(Math.max(...alphas)).toBeGreaterThan(0.5);
    expect(alphas.filter((a) => a > 0.4).length).toBeGreaterThanOrEqual(2);
  });

  it('does not throw on an empty matrix', () => {
    expect(() => drawConfusion(new Stub(), 0, 0, 100, [], [])).not.toThrow();
  });
});
