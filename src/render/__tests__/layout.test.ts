import { describe, expect, it } from 'vitest';
import {
  computeLayout,
  fitToView,
  IDENTITY_VIEWPORT,
  layerUnitCount,
  nodeIndex,
  panBy,
  screenToWorldX,
  screenToWorldY,
  worldToScreenX,
  worldToScreenY,
  ZOOM_MAX,
  ZOOM_MIN,
  zoomAt,
} from '../layout';

describe('determinism (§6.1 — same architecture ⇒ same picture, every time)', () => {
  it('produces identical geometry for identical input', () => {
    const a = computeLayout({ sizes: [2, 8, 6, 3] });
    const b = computeLayout({ sizes: [2, 8, 6, 3] });
    expect(a.nodes).toEqual(b.nodes);
    expect(a.edges).toEqual(b.edges);
    expect(a.width).toBe(b.width);
    expect(a.height).toBe(b.height);
  });

  it('does not depend on call order or previous layouts', () => {
    const first = computeLayout({ sizes: [2, 4, 1] });
    computeLayout({ sizes: [35, 24, 10] });
    const again = computeLayout({ sizes: [2, 4, 1] });
    expect(again.nodes).toEqual(first.nodes);
  });
});

describe('structure', () => {
  it('places layers as evenly spaced columns', () => {
    const layout = computeLayout({ sizes: [2, 4, 3] });
    expect(layout.columnX.length).toBe(3);
    const gap0 = (layout.columnX[1] as number) - (layout.columnX[0] as number);
    const gap1 = (layout.columnX[2] as number) - (layout.columnX[1] as number);
    expect(gap0).toBeCloseTo(gap1, 12);
  });

  it('centres each column vertically, independent of the others', () => {
    const layout = computeLayout({ sizes: [2, 9, 1], showBiases: false });
    for (let layer = 0; layer < 3; layer++) {
      const start = layout.layerOffsets[layer] as number;
      const count = layerUnitCount(layout, layer);
      const ys = Array.from({ length: count }, (_, i) => (layout.nodes[start + i] as { y: number }).y);
      const mid = (Math.min(...ys) + Math.max(...ys)) / 2;
      expect(mid).toBeCloseTo(layout.height / 2, 6);
    }
  });

  it('keeps unit indices addressable as layerOffsets[layer] + unit', () => {
    // Bias satellites are appended after every unit node precisely so this
    // arithmetic stays valid; interleaving them would break it everywhere.
    const layout = computeLayout({ sizes: [3, 5, 2] });
    for (let layer = 0; layer < 3; layer++) {
      for (let unit = 0; unit < layerUnitCount(layout, layer); unit++) {
        const node = layout.nodes[nodeIndex(layout, layer, unit)];
        expect(node?.layer).toBe(layer);
        expect(node?.unit).toBe(unit);
        expect(node?.kind).not.toBe('bias');
      }
    }
  });

  it('creates one edge per weight, plus one per bias', () => {
    const sizes = [2, 4, 3];
    const withBias = computeLayout({ sizes, showBiases: true });
    const withoutBias = computeLayout({ sizes, showBiases: false });
    // 2*4 + 4*3 = 20 weights; 4 + 3 = 7 biases.
    expect(withoutBias.edges.length).toBe(20);
    expect(withBias.edges.length).toBe(27);
    expect(withBias.edges.filter((e) => e.isBias).length).toBe(7);
  });

  it('addresses each weight edge by its (layer, row, col) in W', () => {
    const layout = computeLayout({ sizes: [2, 3], showBiases: false });
    const seen = new Set<string>();
    for (const edge of layout.edges) {
      seen.add(`${edge.layer}:${edge.row}:${edge.col}`);
      expect(edge.layer).toBe(0);
    }
    expect(seen.size).toBe(6);
    expect(seen.has('0:1:2')).toBe(true);
  });

  it('labels input, hidden and output nodes', () => {
    const layout = computeLayout({ sizes: [2, 4, 3], showBiases: false });
    expect(layout.nodes[0]?.kind).toBe('input');
    expect(layout.nodes[nodeIndex(layout, 1, 0)]?.kind).toBe('hidden');
    expect(layout.nodes[nodeIndex(layout, 2, 0)]?.kind).toBe('output');
  });

  it('gives every non-input node exactly one bias satellite (§6.2)', () => {
    // Half of learners think neurons are just weighted sums; biases must not
    // be invisible.
    const layout = computeLayout({ sizes: [2, 4, 3] });
    const biases = layout.nodes.filter((n) => n.kind === 'bias');
    expect(biases.length).toBe(7);
    const keys = new Set(biases.map((b) => `${b.layer}:${b.unit}`));
    expect(keys.size).toBe(7);
  });
});

describe('spacing adapts to unit count (§6.1)', () => {
  it('compresses a tall column rather than overflowing', () => {
    const small = computeLayout({ sizes: [2, 4, 1], showBiases: false });
    const large = computeLayout({ sizes: [2, 40, 1], showBiases: false });
    expect(large.height).toBeGreaterThan(small.height);
    // But not 10x taller: pitch shrinks as the count grows.
    expect(large.height).toBeLessThan(small.height * 10);
  });

  it('never lets nodes in a column overlap', () => {
    for (const count of [1, 2, 5, 12, 30, 64]) {
      const layout = computeLayout({ sizes: [2, count], showBiases: false });
      const start = layout.layerOffsets[1] as number;
      for (let i = 1; i < count; i++) {
        const a = layout.nodes[start + i - 1] as { y: number; radius: number };
        const b = layout.nodes[start + i] as { y: number; radius: number };
        expect(b.y - a.y, `count ${count}`).toBeGreaterThan(a.radius + b.radius);
      }
    }
  });

  it('keeps radii within the documented bounds', () => {
    for (const count of [1, 8, 64]) {
      const layout = computeLayout({ sizes: [2, count], showBiases: false });
      for (const node of layout.nodes) {
        expect(node.radius).toBeGreaterThanOrEqual(5);
        expect(node.radius).toBeLessThanOrEqual(17);
      }
    }
  });

  it('handles the glyph input layer (35 inputs) without collapsing', () => {
    const layout = computeLayout({ sizes: [35, 24, 10] });
    expect(layout.nodes.length).toBeGreaterThan(35);
    expect(Number.isFinite(layout.height)).toBe(true);
    expect(layout.height).toBeGreaterThan(0);
  });
});

describe('validation', () => {
  it('requires an input layer and at least one dense layer', () => {
    expect(() => computeLayout({ sizes: [2] })).toThrowError(/at least one dense layer/);
    expect(() => computeLayout({ sizes: [] })).toThrowError(/at least one dense layer/);
  });

  it('rejects a non-positive or fractional layer size', () => {
    expect(() => computeLayout({ sizes: [2, 0] })).toThrowError(/positive integer/);
    expect(() => computeLayout({ sizes: [2, 2.5] })).toThrowError(/positive integer/);
  });
});

describe('viewport', () => {
  it('round-trips world and screen coordinates', () => {
    const viewport = { scale: 1.7, offsetX: -40, offsetY: 25 };
    for (const [x, y] of [[0, 0], [100, 250], [-30, 12.5]] as const) {
      expect(screenToWorldX(viewport, worldToScreenX(viewport, x))).toBeCloseTo(x, 10);
      expect(screenToWorldY(viewport, worldToScreenY(viewport, y))).toBeCloseTo(y, 10);
    }
  });

  it('zooms about the cursor, keeping the world point under it fixed', () => {
    // This is the property that makes wheel-zoom feel right; without it the
    // picture slides out from under the pointer.
    const before = IDENTITY_VIEWPORT;
    const screenX = 320;
    const screenY = 180;
    const worldX = screenToWorldX(before, screenX);
    const worldY = screenToWorldY(before, screenY);

    const after = zoomAt(before, screenX, screenY, 1.6);
    expect(worldToScreenX(after, worldX)).toBeCloseTo(screenX, 8);
    expect(worldToScreenY(after, worldY)).toBeCloseTo(screenY, 8);
    expect(after.scale).toBeCloseTo(1.6, 10);
  });

  it('clamps zoom to its bounds', () => {
    let viewport = IDENTITY_VIEWPORT;
    for (let i = 0; i < 50; i++) viewport = zoomAt(viewport, 0, 0, 2);
    expect(viewport.scale).toBe(ZOOM_MAX);
    for (let i = 0; i < 100; i++) viewport = zoomAt(viewport, 0, 0, 0.5);
    expect(viewport.scale).toBe(ZOOM_MIN);
  });

  it('keeps the cursor anchored even at the zoom limit', () => {
    let viewport = { scale: ZOOM_MAX, offsetX: 10, offsetY: 10 };
    const worldX = screenToWorldX(viewport, 200);
    viewport = zoomAt(viewport, 200, 100, 2); // refused, already at max
    expect(worldToScreenX(viewport, worldX)).toBeCloseTo(200, 8);
  });

  it('fits a layout inside the view with padding', () => {
    const layout = computeLayout({ sizes: [2, 8, 6, 3] });
    const viewport = fitToView(layout, 900, 600, 24);
    expect(worldToScreenX(viewport, 0)).toBeGreaterThanOrEqual(0);
    expect(worldToScreenY(viewport, 0)).toBeGreaterThanOrEqual(0);
    expect(worldToScreenX(viewport, layout.width)).toBeLessThanOrEqual(900);
    expect(worldToScreenY(viewport, layout.height)).toBeLessThanOrEqual(600);
  });

  it('centres the layout when it fits', () => {
    const layout = computeLayout({ sizes: [2, 2] });
    const viewport = fitToView(layout, 1200, 800, 24);
    const left = worldToScreenX(viewport, 0);
    const right = 1200 - worldToScreenX(viewport, layout.width);
    expect(left).toBeCloseTo(right, 6);
  });

  it('pans by a screen delta', () => {
    const moved = panBy({ scale: 2, offsetX: 5, offsetY: 7 }, 10, -3);
    expect(moved).toEqual({ scale: 2, offsetX: 15, offsetY: 4 });
  });
});
