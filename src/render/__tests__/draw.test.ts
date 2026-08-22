import { describe, expect, it } from 'vitest';
import { computeLayout } from '../layout';
import type { EdgeLayout } from '../layout';
import { drawGrid, GRID_PITCH } from '../draw/grid';
import { drawEdges, smoothReference, weightReference } from '../draw/edges';
import { drawNodes } from '../draw/nodes';
import { drawLegend } from '../draw/legend';
import type { Ctx2D } from '../draw/context';
import { COLORS, WEIGHT_NEGATIVE, WEIGHT_POSITIVE, parseHex } from '../theme';

/*
 * The draw modules are verified with a RECORDING context rather than pixel
 * snapshots. Snapshots of a canvas are brittle across platforms and tell you
 * nothing about why they changed; a call log lets a test assert the thing that
 * actually matters — draw order, encoding, and that the hot path allocates
 * nothing.
 */

interface Call {
  readonly op: string;
  readonly args: readonly unknown[];
}

class RecordingContext implements Ctx2D {
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 1;
  lineCap: CanvasLineCap = 'butt';
  lineJoin: CanvasLineJoin = 'miter';
  font = '';
  textAlign: CanvasTextAlign = 'start';
  textBaseline: CanvasTextBaseline = 'alphabetic';
  globalAlpha = 1;

  readonly calls: Call[] = [];
  /** Style at the moment of each stroke/fill, which is what a pixel would see. */
  readonly strokes: { style: string; width: number; dash: number[] }[] = [];
  readonly fills: { style: string }[] = [];
  private dash: number[] = [];

  private record(op: string, ...args: unknown[]): void {
    this.calls.push({ op, args });
  }
  save(): void { this.record('save'); }
  restore(): void { this.record('restore'); }
  beginPath(): void { this.record('beginPath'); }
  closePath(): void { this.record('closePath'); }
  moveTo(x: number, y: number): void { this.record('moveTo', x, y); }
  lineTo(x: number, y: number): void { this.record('lineTo', x, y); }
  arc(x: number, y: number, r: number): void { this.record('arc', x, y, r); }
  rect(x: number, y: number, w: number, h: number): void { this.record('rect', x, y, w, h); }
  fill(): void { this.record('fill'); this.fills.push({ style: this.fillStyle }); }
  stroke(): void {
    this.record('stroke');
    this.strokes.push({ style: this.strokeStyle, width: this.lineWidth, dash: [...this.dash] });
  }
  fillRect(x: number, y: number, w: number, h: number): void {
    this.record('fillRect', x, y, w, h);
    this.fills.push({ style: this.fillStyle });
  }
  clearRect(x: number, y: number, w: number, h: number): void { this.record('clearRect', x, y, w, h); }
  fillText(text: string, x: number, y: number): void { this.record('fillText', text, x, y); }
  measureText(text: string): { width: number } { return { width: text.length * 6 }; }
  setLineDash(segments: number[]): void { this.dash = [...segments]; }
  setTransform(...args: number[]): void { this.record('setTransform', ...args); }
}

const VIEWPORT = { scale: 1, offsetX: 0, offsetY: 0 };

function rgbaOf(hex: string): string {
  const { r, g, b } = parseHex(hex);
  return `${r}, ${g}, ${b}`;
}

describe('grid', () => {
  it('paints the canvas face before any lines', () => {
    const ctx = new RecordingContext();
    drawGrid(ctx, VIEWPORT, 400, 300);
    expect(ctx.calls[0]?.op).toBe('fillRect');
    expect(ctx.fills[0]?.style).toBe(COLORS.bgCanvas);
  });

  it('anchors gridlines to world space so they slide with a pan', () => {
    const a = new RecordingContext();
    drawGrid(a, { scale: 1, offsetX: 0, offsetY: 0 }, 400, 300);
    const b = new RecordingContext();
    drawGrid(b, { scale: 1, offsetX: GRID_PITCH, offsetY: 0 }, 400, 300);
    const xs = (c: RecordingContext): number[] =>
      c.calls.filter((k) => k.op === 'moveTo').map((k) => k.args[0] as number);
    // Panning by exactly one pitch reproduces the same set of lines.
    expect(xs(b).length).toBe(xs(a).length);
  });

  it('drops the grid rather than drawing mush when zoomed far out', () => {
    const ctx = new RecordingContext();
    drawGrid(ctx, { scale: 0.05, offsetX: 0, offsetY: 0 }, 400, 300);
    expect(ctx.calls.some((c) => c.op === 'moveTo')).toBe(false);
    // The background is still painted.
    expect(ctx.fills[0]?.style).toBe(COLORS.bgCanvas);
  });
});

describe('edges — the weight encoding (§6.2)', () => {
  const layout = computeLayout({ sizes: [2, 2], showBiases: false });
  const baseStyle = {
    wRef: 1,
    colorblindSafe: false,
    hoveredEdge: -1,
    selectedEdge: -1,
    frozenLayers: new Set<number>(),
  };

  it('draws one stroke per edge', () => {
    const ctx = new RecordingContext();
    drawEdges(ctx, layout, VIEWPORT, () => 0.5, baseStyle);
    expect(ctx.strokes.length).toBe(layout.edges.length);
  });

  it('maps sign to hue', () => {
    const ctx = new RecordingContext();
    const weights = [-1, 1, -0.5, 0.5];
    drawEdges(ctx, layout, VIEWPORT, (e: EdgeLayout) => weights[layout.edges.indexOf(e)] ?? 0, baseStyle);
    expect(ctx.strokes[0]?.style).toContain(rgbaOf(WEIGHT_NEGATIVE));
    expect(ctx.strokes[1]?.style).toContain(rgbaOf(WEIGHT_POSITIVE));
  });

  it('maps magnitude to stroke width', () => {
    const ctx = new RecordingContext();
    const weights = [0.05, 1, 0.05, 1];
    drawEdges(ctx, layout, VIEWPORT, (e) => weights[layout.edges.indexOf(e)] ?? 0, baseStyle);
    expect(ctx.strokes[0]!.width).toBeLessThan(ctx.strokes[1]!.width);
    expect(ctx.strokes[0]!.width).toBeGreaterThanOrEqual(0.5);
  });

  it('keeps stroke width in SCREEN space, so the encoding survives zoom', () => {
    // Width means "this much weight"; it must not double when the user zooms.
    const wide = new RecordingContext();
    drawEdges(wide, layout, { scale: 3, offsetX: 0, offsetY: 0 }, () => 1, baseStyle);
    const narrow = new RecordingContext();
    drawEdges(narrow, layout, { scale: 0.4, offsetX: 0, offsetY: 0 }, () => 1, baseStyle);
    expect(wide.strokes[0]?.width).toBe(narrow.strokes[0]?.width);
  });

  it('dashes negative weights in colourblind-safe mode, and only negative ones', () => {
    const ctx = new RecordingContext();
    const weights = [-1, 1, -1, 1];
    drawEdges(ctx, layout, VIEWPORT, (e) => weights[layout.edges.indexOf(e)] ?? 0, {
      ...baseStyle,
      colorblindSafe: true,
    });
    expect(ctx.strokes[0]?.dash.length).toBeGreaterThan(0);
    expect(ctx.strokes[1]?.dash.length).toBe(0);
  });

  it('dashes frozen layers distinctly (§6.2)', () => {
    const ctx = new RecordingContext();
    drawEdges(ctx, layout, VIEWPORT, () => 1, {
      ...baseStyle,
      frozenLayers: new Set([0]),
    });
    expect(ctx.strokes[0]?.dash.length).toBeGreaterThan(0);
  });

  it('emphasises hover and selection with the neutral focus colour', () => {
    // Never a third hue: a coloured ring would be read as a weight value.
    const ctx = new RecordingContext();
    drawEdges(ctx, layout, VIEWPORT, () => 1, { ...baseStyle, hoveredEdge: 2 });
    expect(ctx.strokes[2]?.style).toBe(COLORS.focus);
    expect(ctx.strokes[0]?.style).not.toBe(COLORS.focus);
  });

  it('leaves the dash pattern reset for whoever draws next', () => {
    const ctx = new RecordingContext();
    drawEdges(ctx, layout, VIEWPORT, () => -1, { ...baseStyle, colorblindSafe: true });
    // The last thing drawEdges does is clear the dash; verify by drawing a
    // node afterwards and checking it is solid.
    drawNodes(ctx, layout, VIEWPORT, {
      normalizedActivation: () => 0.5,
      normalizedDelta: () => null,
      isDead: () => false,
      isFrozen: () => false,
      isAblated: () => false,
      hoveredNode: -1,
      selectedNode: -1,
    });
    const nodeStroke = ctx.strokes[ctx.strokes.length - 1];
    expect(nodeStroke?.dash.length).toBe(0);
  });
});

describe('weightReference', () => {
  it('uses a percentile, so one runaway weight cannot flatten the picture', () => {
    // This is exactly what happens right before a divergence, when the picture
    // matters most.
    const normal = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const withOutlier = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 10_000]);
    expect(weightReference(withOutlier)).toBeLessThan(100);
    // p95 of five sorted values indexes floor(0.95 × 4) = 3, i.e. 0.4 — the
    // reference deliberately sits below the maximum.
    expect(weightReference(normal)).toBeCloseTo(0.4, 6);
  });

  it('falls back to 1 for an all-zero network (lesson 3)', () => {
    expect(weightReference(new Float64Array([0, 0, 0, 0]))).toBe(1);
    expect(weightReference(new Float64Array(0))).toBe(1);
  });

  it('smooths toward a target so the picture cannot strobe', () => {
    let value = smoothReference(0, 10);
    expect(value).toBe(10); // first observation adopts the target
    value = smoothReference(1, 10, 0.1);
    expect(value).toBeCloseTo(1.9, 10);
    // Converges monotonically.
    let v = 1;
    for (let i = 0; i < 200; i++) v = smoothReference(v, 10, 0.1);
    expect(v).toBeCloseTo(10, 4);
  });
});

describe('nodes (§6.2)', () => {
  const layout = computeLayout({ sizes: [2, 2], showBiases: false });
  const baseStyle = {
    normalizedActivation: (): number | null => 0.8,
    normalizedDelta: (): number | null => null,
    isDead: (): boolean => false,
    isFrozen: (): boolean => false,
    isAblated: (): boolean => false,
    hoveredNode: -1,
    selectedNode: -1,
  };

  it('fills each node according to its activation', () => {
    const ctx = new RecordingContext();
    drawNodes(ctx, layout, VIEWPORT, baseStyle);
    expect(ctx.fills.length).toBe(layout.nodes.length);
    expect(ctx.fills[0]?.style).toContain(rgbaOf(WEIGHT_POSITIVE));
  });

  it('renders dead units hollow, not merely dim', () => {
    const ctx = new RecordingContext();
    drawNodes(ctx, layout, VIEWPORT, { ...baseStyle, isDead: () => true });
    // Filled with the canvas colour: an inert hole, not a faint node.
    expect(ctx.fills[0]?.style).toBe(COLORS.bgCanvas);
  });

  it('renders ablated units the same inert way', () => {
    const ctx = new RecordingContext();
    drawNodes(ctx, layout, VIEWPORT, { ...baseStyle, isAblated: () => true });
    expect(ctx.fills[0]?.style).toBe(COLORS.bgCanvas);
  });

  it('falls back to a neutral fill before anything has run', () => {
    const ctx = new RecordingContext();
    drawNodes(ctx, layout, VIEWPORT, { ...baseStyle, normalizedActivation: () => null });
    expect(ctx.fills[0]?.style).toBe(COLORS.bgRaised);
  });

  it('draws a delta ring only where |δ| is meaningful', () => {
    const withDelta = new RecordingContext();
    drawNodes(withDelta, layout, VIEWPORT, { ...baseStyle, normalizedDelta: () => 0.9 });
    const withoutDelta = new RecordingContext();
    drawNodes(withoutDelta, layout, VIEWPORT, { ...baseStyle, normalizedDelta: () => 0 });
    expect(withDelta.strokes.length).toBeGreaterThan(withoutDelta.strokes.length);
  });

  it('scales node radius with zoom — a node is an object, not an encoding', () => {
    const near = new RecordingContext();
    drawNodes(near, layout, { scale: 2, offsetX: 0, offsetY: 0 }, baseStyle);
    const far = new RecordingContext();
    drawNodes(far, layout, { scale: 1, offsetX: 0, offsetY: 0 }, baseStyle);
    const radiusOf = (c: RecordingContext): number =>
      c.calls.find((k) => k.op === 'arc')?.args[2] as number;
    expect(radiusOf(near)).toBeCloseTo(radiusOf(far) * 2, 6);
  });

  it('dashes frozen nodes', () => {
    const ctx = new RecordingContext();
    drawNodes(ctx, layout, VIEWPORT, { ...baseStyle, isFrozen: () => true });
    expect(ctx.strokes[0]?.dash.length).toBeGreaterThan(0);
  });
});

describe('legend (§6.2 — the scale is never a mystery)', () => {
  it('prints wRef as an actual number', () => {
    const ctx = new RecordingContext();
    drawLegend(ctx, 0, 0, 0.42, false);
    const texts = ctx.calls.filter((c) => c.op === 'fillText').map((c) => c.args[0] as string);
    expect(texts).toContain('WEIGHT');
    expect(texts.some((t) => t.includes('0.42'))).toBe(true);
    expect(texts).toContain('0');
  });

  it('samples the same ramp the edges use, so it cannot drift', () => {
    const ctx = new RecordingContext();
    drawLegend(ctx, 0, 0, 1, false);
    const styles = ctx.strokes.map((s) => s.style);
    expect(styles.some((s) => s.includes(rgbaOf(WEIGHT_NEGATIVE)))).toBe(true);
    expect(styles.some((s) => s.includes(rgbaOf(WEIGHT_POSITIVE)))).toBe(true);
  });
});
