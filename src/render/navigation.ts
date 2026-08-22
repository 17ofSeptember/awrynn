/*
 * Keyboard traversal of the network (§9 quality floor).
 *
 * "full keyboard navigation of nodes and edges (arrow keys traverse, Enter
 * edits)".
 *
 * The canvas is a picture, so none of its contents are focusable on their own.
 * The canvas itself takes focus and this module decides what the arrow keys
 * mean, which keeps the traversal pure and testable rather than tangled into
 * pointer handling.
 *
 * The model is deliberately small enough to hold in your head:
 *
 *   ← →            move between columns
 *   ↑ ↓            move between units in the current column
 *   shift + ↑ ↓    walk the connections arriving at the current unit
 *   enter          edit the selected value
 *   escape         clear the selection
 *
 * Column-then-unit rather than a single flat order, because the network IS a
 * grid and a flat traversal of 300 edges is not navigation, it is a list.
 */

import type { EdgeLayout, Layout, NodeLayout } from './layout';
import type { Hit } from './hit';

export type NavigationKey =
  | 'left'
  | 'right'
  | 'up'
  | 'down'
  | 'edge-prev'
  | 'edge-next'
  | 'clear';

/** Units in a column, excluding bias satellites. */
function unitsIn(layout: Layout, column: number): number {
  const start = layout.layerOffsets[column];
  const end = layout.layerOffsets[column + 1];
  if (start === undefined || end === undefined) return 0;
  return end - start;
}

function nodeAt(layout: Layout, column: number, unit: number): { index: number; node: NodeLayout } | null {
  const base = layout.layerOffsets[column];
  if (base === undefined) return null;
  const count = unitsIn(layout, column);
  if (unit < 0 || unit >= count) return null;
  const node = layout.nodes[base + unit];
  if (node === undefined) return null;
  return { index: base + unit, node };
}

function nodeHit(layout: Layout, column: number, unit: number): Hit | null {
  const found = nodeAt(layout, column, unit);
  if (found === null) return null;
  return { kind: 'node', index: found.index, node: found.node, distance: 0 };
}

/** Edges arriving at a unit, in row order. */
function incomingEdges(layout: Layout, column: number, unit: number): { index: number; edge: EdgeLayout }[] {
  const denseLayer = column - 1;
  const found: { index: number; edge: EdgeLayout }[] = [];
  layout.edges.forEach((edge, index) => {
    if (edge.layer === denseLayer && edge.col === unit) found.push({ index, edge });
  });
  return found;
}

/** Column count, including the input column. */
export function columnCount(layout: Layout): number {
  return Math.max(0, layout.layerOffsets.length - 1);
}

/**
 * Where the selection goes next.
 *
 * Returns null when the move is not possible, so the caller can leave the
 * selection alone rather than wrapping. Wrapping from the output column back to
 * the inputs would be disorienting: the picture has a direction and the
 * traversal should respect it.
 */
export function navigate(layout: Layout, current: Hit | null, key: NavigationKey): Hit | null {
  if (key === 'clear') return null;

  const columns = columnCount(layout);
  if (columns === 0) return null;

  // Nothing selected: any arrow key enters at the first input unit.
  if (current === null) return nodeHit(layout, 0, 0);

  // Resolve the current position to a (column, unit), whether a node or an edge
  // is selected. From an edge, movement continues from its destination unit.
  let column: number;
  let unit: number;
  if (current.kind === 'node') {
    column = current.node.layer;
    unit = current.node.unit;
  } else {
    column = current.edge.layer + 1;
    unit = current.edge.col;
  }

  switch (key) {
    case 'left':
      return nodeHit(layout, column - 1, Math.min(unit, Math.max(0, unitsIn(layout, column - 1) - 1)));
    case 'right':
      return nodeHit(layout, column + 1, Math.min(unit, Math.max(0, unitsIn(layout, column + 1) - 1)));
    case 'up':
      return nodeHit(layout, column, unit - 1);
    case 'down':
      return nodeHit(layout, column, unit + 1);
    case 'edge-prev':
    case 'edge-next': {
      const edges = incomingEdges(layout, column, unit);
      if (edges.length === 0) return null; // the input column has none
      const position =
        current.kind === 'edge' ? edges.findIndex((e) => e.index === current.index) : -1;
      const step = key === 'edge-next' ? 1 : -1;
      // Wraps within one unit's own connections, which is a short, closed list
      // the reader is deliberately cycling through.
      const next = position === -1
        ? key === 'edge-next'
          ? 0
          : edges.length - 1
        : (position + step + edges.length) % edges.length;
      const target = edges[next];
      if (target === undefined) return null;
      return { kind: 'edge', index: target.index, edge: target.edge, distance: 0 };
    }
  }
}

/**
 * A spoken description of the selection, for the live region.
 *
 * The canvas conveys everything through colour and position, none of which a
 * screen reader can see. This is the text equivalent, and it names the value as
 * well as the location because the value is the point.
 */
export function describeSelection(
  layout: Layout,
  hit: Hit | null,
  valueOf: (hit: Hit) => number | null,
  columnLabels: readonly string[],
): string {
  if (hit === null) return 'Nothing selected.';

  const value = valueOf(hit);
  const formatted = value === null ? '' : ` Value ${value.toFixed(4)}.`;

  if (hit.kind === 'node') {
    const label = columnLabels[hit.node.layer] ?? `column ${hit.node.layer}`;
    if (hit.node.kind === 'bias') {
      return `Bias for unit ${hit.node.unit} of ${label}.${formatted}`;
    }
    const count = unitsIn(layout, hit.node.layer);
    return `${label}, unit ${hit.node.unit + 1} of ${count}.${formatted}`;
  }

  const edge = hit.edge;
  if (edge.isBias) {
    return `Bias into unit ${edge.col} of layer ${edge.layer + 1}.${formatted}`;
  }
  return `Connection from unit ${edge.row} into unit ${edge.col} of layer ${edge.layer + 1}.${formatted}`;
}

/** Map a keyboard event to a navigation key, or null if it is not one. */
export function keyToNavigation(key: string, shift: boolean): NavigationKey | null {
  switch (key) {
    case 'ArrowLeft':
      return 'left';
    case 'ArrowRight':
      return 'right';
    case 'ArrowUp':
      return shift ? 'edge-prev' : 'up';
    case 'ArrowDown':
      return shift ? 'edge-next' : 'down';
    case 'Escape':
      return 'clear';
    default:
      return null;
  }
}
