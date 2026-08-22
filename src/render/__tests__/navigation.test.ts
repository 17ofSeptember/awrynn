import { describe, expect, it } from 'vitest';
import { columnCount, describeSelection, keyToNavigation, navigate } from '../navigation';
import { computeLayout, nodeIndex } from '../layout';
import type { Hit } from '../hit';

/*
 * §9 quality floor: "full keyboard navigation of nodes and edges (arrow keys
 * traverse, Enter edits)".
 */

const layout = computeLayout({ sizes: [2, 3, 2] });

function node(column: number, unit: number): Hit {
  const index = nodeIndex(layout, column, unit);
  return { kind: 'node', index, node: layout.nodes[index]!, distance: 0 };
}

describe('entering the picture', () => {
  it('any arrow key selects the first input when nothing is selected', () => {
    for (const key of ['left', 'right', 'up', 'down'] as const) {
      const hit = navigate(layout, null, key);
      expect(hit?.kind, key).toBe('node');
      expect(hit?.kind === 'node' && hit.node.layer, key).toBe(0);
      expect(hit?.kind === 'node' && hit.node.unit, key).toBe(0);
    }
  });

  it('counts columns including the inputs', () => {
    expect(columnCount(layout)).toBe(3);
  });
});

describe('moving between columns and units', () => {
  it('right and left step across columns', () => {
    const start = node(0, 0);
    const right = navigate(layout, start, 'right');
    expect(right?.kind === 'node' && right.node.layer).toBe(1);
    const back = navigate(layout, right, 'left');
    expect(back?.kind === 'node' && back.node.layer).toBe(0);
  });

  it('up and down step within a column', () => {
    const start = node(1, 1);
    expect((navigate(layout, start, 'up') as Hit & { node: { unit: number } }).node.unit).toBe(0);
    expect((navigate(layout, start, 'down') as Hit & { node: { unit: number } }).node.unit).toBe(2);
  });

  it('stops at the edges rather than wrapping', () => {
    // The picture has a direction; wrapping from the output back to the inputs
    // would be disorienting.
    expect(navigate(layout, node(0, 0), 'left')).toBeNull();
    expect(navigate(layout, node(2, 0), 'right')).toBeNull();
    expect(navigate(layout, node(1, 0), 'up')).toBeNull();
    expect(navigate(layout, node(1, 2), 'down')).toBeNull();
  });

  it('clamps the unit when moving into a shorter column', () => {
    // Unit 2 of a 3-unit column moving into a 2-unit column must land on unit 1,
    // not vanish.
    const hit = navigate(layout, node(1, 2), 'right');
    expect(hit?.kind === 'node' && hit.node.layer).toBe(2);
    expect(hit?.kind === 'node' && hit.node.unit).toBe(1);
  });

  it('never selects a bias satellite by arrow key', () => {
    // Biases are reachable as edges; including them in the unit traversal would
    // double the length of every column for no gain.
    let hit: Hit | null = node(0, 0);
    for (let i = 0; i < 30; i++) {
      for (const key of ['right', 'down', 'left', 'up'] as const) {
        const next = navigate(layout, hit, key);
        if (next !== null) hit = next;
        if (hit.kind === 'node') expect(hit.node.kind).not.toBe('bias');
      }
    }
  });
});

describe('walking a unit’s connections', () => {
  it('shift+down enters the first incoming edge', () => {
    const hit = navigate(layout, node(1, 0), 'edge-next');
    expect(hit?.kind).toBe('edge');
    expect(hit?.kind === 'edge' && hit.edge.col).toBe(0);
    expect(hit?.kind === 'edge' && hit.edge.layer).toBe(0);
  });

  it('cycles through exactly the edges arriving at that unit', () => {
    // Layer 1 unit 0 has two weight edges plus one bias edge.
    const seen = new Set<number>();
    let hit: Hit | null = navigate(layout, node(1, 0), 'edge-next');
    for (let i = 0; i < 10; i++) {
      if (hit === null || hit.kind !== 'edge') break;
      expect(hit.edge.col).toBe(0);
      seen.add(hit.index);
      hit = navigate(layout, hit, 'edge-next');
    }
    expect(seen.size).toBe(3);
  });

  it('wraps within one unit’s own connections', () => {
    // A short closed list the reader is deliberately cycling, unlike the
    // columns, which have a direction.
    const first = navigate(layout, node(1, 0), 'edge-next');
    let hit: Hit | null = first;
    for (let i = 0; i < 3; i++) hit = navigate(layout, hit, 'edge-next');
    expect(hit?.index).toBe(first?.index);
  });

  it('goes backwards too', () => {
    const last = navigate(layout, node(1, 0), 'edge-prev');
    const first = navigate(layout, node(1, 0), 'edge-next');
    expect(last?.index).not.toBe(first?.index);
    expect(navigate(layout, last, 'edge-next')?.index).toBe(first?.index);
  });

  it('has nothing to walk at the input column', () => {
    expect(navigate(layout, node(0, 0), 'edge-next')).toBeNull();
  });

  it('arrow keys from an edge continue from its destination unit', () => {
    const edge = navigate(layout, node(1, 1), 'edge-next');
    const up = navigate(layout, edge, 'up');
    expect(up?.kind === 'node' && up.node.layer).toBe(1);
    expect(up?.kind === 'node' && up.node.unit).toBe(0);
  });
});

describe('key mapping', () => {
  it('maps arrows, with shift switching to edges', () => {
    expect(keyToNavigation('ArrowLeft', false)).toBe('left');
    expect(keyToNavigation('ArrowUp', false)).toBe('up');
    expect(keyToNavigation('ArrowUp', true)).toBe('edge-prev');
    expect(keyToNavigation('ArrowDown', true)).toBe('edge-next');
    expect(keyToNavigation('Escape', false)).toBe('clear');
  });

  it('ignores everything else, so typing is not swallowed', () => {
    for (const key of ['a', 'Enter', 'Tab', ' ', 'F5']) {
      expect(keyToNavigation(key, false), key).toBeNull();
    }
  });

  it('clear deselects', () => {
    expect(navigate(layout, node(1, 1), 'clear')).toBeNull();
  });
});

describe('spoken description (the canvas is invisible to a screen reader)', () => {
  const labels = ['input', 'hidden 1', 'output'];

  it('names a unit, its column, and its value', () => {
    const text = describeSelection(layout, node(1, 1), () => 0.5, labels);
    expect(text).toContain('hidden 1');
    expect(text).toContain('unit 2 of 3');
    expect(text).toContain('0.5000');
  });

  it('names a connection by both endpoints', () => {
    const edge = navigate(layout, node(1, 0), 'edge-next');
    const text = describeSelection(layout, edge, () => -0.25, labels);
    expect(text).toMatch(/from unit \d+ into unit 0 of layer 1/);
    expect(text).toContain('-0.2500');
  });

  it('says so when nothing is selected', () => {
    expect(describeSelection(layout, null, () => null, labels)).toBe('Nothing selected.');
  });

  it('omits a value it cannot read rather than inventing one', () => {
    const text = describeSelection(layout, node(0, 0), () => null, labels);
    expect(text).not.toContain('Value');
  });
});
