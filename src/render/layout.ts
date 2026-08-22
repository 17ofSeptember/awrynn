/*
 * Deterministic node and edge geometry.
 *
 * Spec §6.1: "layers as evenly spaced columns, neurons vertically centered
 * within a column, spacing adapting to count. Same architecture ⇒ same picture,
 * every time."
 *
 * Layout is computed in a fixed WORLD space, independent of canvas size, zoom
 * and pan. The viewport transform is applied at draw time. Keeping them
 * separate means hit-testing can work in world coordinates — so a hit test is
 * exact rather than accumulating rounding through a screen-space round trip —
 * and a resize never changes the picture's proportions.
 */

export interface NodeLayout {
  /** Layer index: 0 is the input layer, so this is one more than a DenseLayer index. */
  readonly layer: number;
  readonly unit: number;
  readonly x: number;
  readonly y: number;
  readonly radius: number;
  /** Bias satellites are drawn and hit-tested like nodes but are not units. */
  readonly kind: 'input' | 'hidden' | 'output' | 'bias';
}

export interface EdgeLayout {
  /** Index into `nodes` of the source. */
  readonly from: number;
  readonly to: number;
  /** Index of the DenseLayer that owns this weight. */
  readonly layer: number;
  /** Row within W (the source unit). -1 for a bias edge. */
  readonly row: number;
  /** Column within W (the destination unit). */
  readonly col: number;
  readonly isBias: boolean;
}

export interface Layout {
  readonly nodes: readonly NodeLayout[];
  readonly edges: readonly EdgeLayout[];
  /** Index of the first node of each layer, plus a terminal entry. */
  readonly layerOffsets: readonly number[];
  readonly width: number;
  readonly height: number;
  /** Column x for each layer, for headers and labels. */
  readonly columnX: readonly number[];
}

export interface LayoutOptions {
  /** Units per layer, INCLUDING the input layer at index 0. */
  readonly sizes: readonly number[];
  /** Draw a bias satellite feeding each non-input node (§6.2). */
  readonly showBiases?: boolean | undefined;
}

export const LAYOUT_CONSTANTS = {
  columnGap: 190,
  /** Vertical pitch when a layer is small enough to use it. */
  maxRowGap: 64,
  minRowGap: 26,
  maxRadius: 17,
  minRadius: 5,
  /** Tallest a column is allowed to get before spacing starts compressing. */
  maxColumnHeight: 520,
  margin: 72,
  biasOffsetX: -46,
  biasOffsetY: -40,
  biasRadius: 6,
} as const;

/**
 * Row pitch and node radius for a layer of `count` units.
 *
 * Spacing adapts to count so a 2-unit layer does not look sparse and a 32-unit
 * layer still fits: the pitch shrinks toward `minRowGap` and the radius with
 * it, both clamped so nodes never overlap or vanish.
 */
function columnMetrics(count: number): { rowGap: number; radius: number } {
  const c = LAYOUT_CONSTANTS;
  if (count <= 1) return { rowGap: c.maxRowGap, radius: c.maxRadius };
  const naturalHeight = (count - 1) * c.maxRowGap;
  const scale = naturalHeight > c.maxColumnHeight ? c.maxColumnHeight / naturalHeight : 1;
  const rowGap = Math.max(c.minRowGap, c.maxRowGap * scale);
  // Radius tracks pitch so the gap between two nodes stays proportional.
  const radius = Math.max(c.minRadius, Math.min(c.maxRadius, rowGap * 0.27));
  return { rowGap, radius };
}

export function computeLayout(options: LayoutOptions): Layout {
  const { sizes } = options;
  if (sizes.length < 2) {
    throw new Error(
      `layout.computeLayout: need an input layer and at least one dense layer, got ${sizes.length} column(s).`,
    );
  }
  for (const [i, size] of sizes.entries()) {
    if (!Number.isInteger(size) || size <= 0) {
      throw new Error(`layout.computeLayout: layer ${i} must have a positive integer size, got ${size}.`);
    }
  }

  const c = LAYOUT_CONSTANTS;
  const showBiases = options.showBiases ?? true;

  const metrics = sizes.map((n) => columnMetrics(n));
  const columnHeights = sizes.map((n, i) => (n - 1) * (metrics[i] as { rowGap: number }).rowGap);
  const contentHeight = Math.max(...columnHeights);
  const height = contentHeight + c.margin * 2;
  const width = (sizes.length - 1) * c.columnGap + c.margin * 2;
  const centerY = height / 2;

  const nodes: NodeLayout[] = [];
  const layerOffsets: number[] = [];
  const columnX: number[] = [];
  // Bias node index per (layer, unit), so edges can reference them.
  const biasIndex = new Map<string, number>();

  for (let layer = 0; layer < sizes.length; layer++) {
    layerOffsets.push(nodes.length);
    const count = sizes[layer] as number;
    const { rowGap, radius } = metrics[layer] as { rowGap: number; radius: number };
    const x = c.margin + layer * c.columnGap;
    columnX.push(x);
    // Vertically centred within the column, independent of other columns.
    const top = centerY - ((count - 1) * rowGap) / 2;

    for (let unit = 0; unit < count; unit++) {
      const kind: NodeLayout['kind'] =
        layer === 0 ? 'input' : layer === sizes.length - 1 ? 'output' : 'hidden';
      nodes.push({ layer, unit, x, y: top + unit * rowGap, radius, kind });
    }
  }
  layerOffsets.push(nodes.length);

  /*
   * Bias satellites are appended AFTER every unit node, so unit indices stay
   * equal to `layerOffsets[layer] + unit`. Interleaving them would make that
   * arithmetic wrong everywhere it is used.
   */
  if (showBiases) {
    for (let layer = 1; layer < sizes.length; layer++) {
      const count = sizes[layer] as number;
      const base = layerOffsets[layer] as number;
      for (let unit = 0; unit < count; unit++) {
        const owner = nodes[base + unit] as NodeLayout;
        biasIndex.set(`${layer}:${unit}`, nodes.length);
        nodes.push({
          layer,
          unit,
          x: owner.x + c.biasOffsetX,
          y: owner.y + c.biasOffsetY,
          radius: c.biasRadius,
          kind: 'bias',
        });
      }
    }
  }

  const edges: EdgeLayout[] = [];
  for (let layer = 1; layer < sizes.length; layer++) {
    const fromBase = layerOffsets[layer - 1] as number;
    const toBase = layerOffsets[layer] as number;
    const fromCount = sizes[layer - 1] as number;
    const toCount = sizes[layer] as number;
    for (let row = 0; row < fromCount; row++) {
      for (let col = 0; col < toCount; col++) {
        edges.push({
          from: fromBase + row,
          to: toBase + col,
          layer: layer - 1,
          row,
          col,
          isBias: false,
        });
      }
    }
    if (showBiases) {
      for (let col = 0; col < toCount; col++) {
        const from = biasIndex.get(`${layer}:${col}`);
        if (from === undefined) continue;
        edges.push({ from, to: toBase + col, layer: layer - 1, row: -1, col, isBias: true });
      }
    }
  }

  return { nodes, edges, layerOffsets, width, height, columnX };
}

/** Index of the node for a unit. Bias satellites are not addressed this way. */
export function nodeIndex(layout: Layout, layer: number, unit: number): number {
  const base = layout.layerOffsets[layer];
  if (base === undefined) {
    throw new Error(`layout.nodeIndex: layer ${layer} is out of range.`);
  }
  return base + unit;
}

export function layerUnitCount(layout: Layout, layer: number): number {
  const start = layout.layerOffsets[layer];
  const end = layout.layerOffsets[layer + 1];
  if (start === undefined || end === undefined) return 0;
  return end - start;
}

/* ------------------------------------------------------------------ *
 * Viewport
 * ------------------------------------------------------------------ */

export interface Viewport {
  /** World units per screen pixel is 1/scale. */
  readonly scale: number;
  readonly offsetX: number;
  readonly offsetY: number;
}

export const IDENTITY_VIEWPORT: Viewport = { scale: 1, offsetX: 0, offsetY: 0 };

export function worldToScreenX(viewport: Viewport, x: number): number {
  return x * viewport.scale + viewport.offsetX;
}

export function worldToScreenY(viewport: Viewport, y: number): number {
  return y * viewport.scale + viewport.offsetY;
}

export function screenToWorldX(viewport: Viewport, x: number): number {
  return (x - viewport.offsetX) / viewport.scale;
}

export function screenToWorldY(viewport: Viewport, y: number): number {
  return (y - viewport.offsetY) / viewport.scale;
}

export const ZOOM_MIN = 0.25;
export const ZOOM_MAX = 4;

/** Zoom about a screen point, so the world point under the cursor stays put. */
export function zoomAt(
  viewport: Viewport,
  screenX: number,
  screenY: number,
  factor: number,
): Viewport {
  const scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, viewport.scale * factor));
  // Solve for the offset that keeps (screenX, screenY) over the same world point.
  const worldX = screenToWorldX(viewport, screenX);
  const worldY = screenToWorldY(viewport, screenY);
  return {
    scale,
    offsetX: screenX - worldX * scale,
    offsetY: screenY - worldY * scale,
  };
}

/** "Fit to view" (§6.1): centre the layout with a margin. */
export function fitToView(
  layout: Layout,
  viewWidth: number,
  viewHeight: number,
  padding = 24,
): Viewport {
  const usableWidth = Math.max(1, viewWidth - padding * 2);
  const usableHeight = Math.max(1, viewHeight - padding * 2);
  const scale = Math.max(
    ZOOM_MIN,
    Math.min(ZOOM_MAX, Math.min(usableWidth / layout.width, usableHeight / layout.height)),
  );
  return {
    scale,
    offsetX: (viewWidth - layout.width * scale) / 2,
    offsetY: (viewHeight - layout.height * scale) / 2,
  };
}

export function panBy(viewport: Viewport, dx: number, dy: number): Viewport {
  return { scale: viewport.scale, offsetX: viewport.offsetX + dx, offsetY: viewport.offsetY + dy };
}
