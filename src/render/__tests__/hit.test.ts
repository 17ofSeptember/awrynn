import { describe, expect, it } from 'vitest';
import { computeLayout, nodeIndex } from '../layout';
import { distanceToSegment, EDGE_PICK_SLACK, HitIndex, NODE_PICK_SLACK } from '../hit';

/*
 * Spec §11 Phase 3 gate: "hit-testing is pixel-accurate".
 *
 * The strongest form of that is an exhaustive sweep: for every node and every
 * edge, a pick at its exact position must return that item and nothing else.
 * These tests do exactly that rather than sampling a few hand-picked points.
 */

describe('node picking', () => {
  const layout = computeLayout({ sizes: [3, 5, 2] });
  const index = new HitIndex(layout);

  it('returns the exact node for a pick at its centre — every node', () => {
    layout.nodes.forEach((node, i) => {
      const hit = index.pick(node.x, node.y);
      expect(hit?.kind, `node ${i}`).toBe('node');
      expect(hit?.kind === 'node' && hit.index, `node ${i}`).toBe(i);
    });
  });

  it('picks within the radius plus slack, and not beyond', () => {
    const node = layout.nodes[nodeIndex(layout, 1, 2)];
    if (node === undefined) throw new Error('missing node');
    const reach = node.radius + NODE_PICK_SLACK;

    const inside = index.pick(node.x + reach - 0.5, node.y);
    expect(inside?.kind === 'node' && inside.node).toEqual(node);

    // Just outside: must not report this node. It may report an edge, which is
    // correct — edges terminate here — so assert on identity, not on null.
    const outside = index.pick(node.x, node.y - reach - 2);
    expect(outside?.kind === 'node' && outside.node === node).toBe(false);
  });

  it('prefers the nearer node when two are close', () => {
    const a = layout.nodes[nodeIndex(layout, 1, 0)];
    const b = layout.nodes[nodeIndex(layout, 1, 1)];
    if (a === undefined || b === undefined) throw new Error('missing nodes');
    const justBelowA = index.pick(a.x, a.y + 1);
    expect(justBelowA?.kind === 'node' && justBelowA.node).toEqual(a);
    const justAboveB = index.pick(b.x, b.y - 1);
    expect(justAboveB?.kind === 'node' && justAboveB.node).toEqual(b);
  });

  it('picks bias satellites, which are not addressable as units', () => {
    const bias = layout.nodes.find((n) => n.kind === 'bias');
    if (bias === undefined) throw new Error('no bias node');
    const hit = index.pick(bias.x, bias.y);
    expect(hit?.kind === 'node' && hit.node.kind).toBe('bias');
  });

  it('returns null well outside the layout', () => {
    expect(index.pick(-10_000, -10_000)).toBeNull();
    expect(index.pick(layout.width + 5000, layout.height + 5000)).toBeNull();
  });
});

describe('edge picking', () => {
  const layout = computeLayout({ sizes: [2, 3, 2], showBiases: false });
  const index = new HitIndex(layout);

  it('returns a NEAREST edge for a pick on any edge — every edge', () => {
    /*
     * Not "the exact edge": edges in a fully-connected layout genuinely cross,
     * and at a crossing two edges are exactly equidistant. Asserting identity
     * there would be asserting an arbitrary tie-break. The real invariant is
     * that nothing closer was missed.
     */
    layout.edges.forEach((edge, i) => {
      const a = layout.nodes[edge.from];
      const b = layout.nodes[edge.to];
      if (a === undefined || b === undefined) throw new Error('missing endpoint');
      const px = (a.x + b.x) / 2;
      const py = (a.y + b.y) / 2;

      const hit = index.pick(px, py);
      expect(hit?.kind, `edge ${i}`).toBe('edge');
      if (hit?.kind !== 'edge') return;

      // Distance to the edge we sampled from is 0, so the winner must be 0 too.
      expect(hit.distance, `edge ${i}`).toBeLessThanOrEqual(1e-9);
    });
  });

  it('resolves a crossing to the edge drawn on top, deterministically', () => {
    // Edges 1 and 3 of a 2→3 layer share an identical midpoint — they cross
    // exactly there. The later edge is drawn last, so it is what the pointer
    // is visually over.
    const e1 = layout.edges[1];
    const e3 = layout.edges[3];
    if (e1 === undefined || e3 === undefined) throw new Error('missing edges');
    const a1 = layout.nodes[e1.from];
    const b1 = layout.nodes[e1.to];
    const a3 = layout.nodes[e3.from];
    const b3 = layout.nodes[e3.to];
    if (!a1 || !b1 || !a3 || !b3) throw new Error('missing endpoints');

    const mid1x = (a1.x + b1.x) / 2;
    const mid1y = (a1.y + b1.y) / 2;
    expect(mid1x).toBeCloseTo((a3.x + b3.x) / 2, 9);
    expect(mid1y).toBeCloseTo((a3.y + b3.y) / 2, 9);

    const hit = index.pick(mid1x, mid1y);
    expect(hit?.kind === 'edge' && hit.index).toBe(3);
    // Deterministic across repeated picks.
    expect(index.pick(mid1x, mid1y)).toEqual(hit);
  });

  it('returns the exact edge away from any crossing', () => {
    // Sampled close to the destination node, where edges into different units
    // have fanned apart, but outside the node's own pick radius.
    layout.edges.forEach((edge, i) => {
      const a = layout.nodes[edge.from];
      const b = layout.nodes[edge.to];
      if (a === undefined || b === undefined) throw new Error('missing endpoint');
      const reach = (b.radius + NODE_PICK_SLACK + 2) / Math.hypot(b.x - a.x, b.y - a.y);
      const t = 1 - reach;
      const hit = index.pick(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
      expect(hit?.kind, `edge ${i}`).toBe('edge');
      expect(hit?.kind === 'edge' && hit.index, `edge ${i}`).toBe(i);
    });
  });

  it('is grabbable within the slack, so a 0.5px hairline can be scrubbed (§6.5)', () => {
    const edge = layout.edges[0];
    if (edge === undefined) throw new Error('no edges');
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) throw new Error('missing endpoint');
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;

    // Perpendicular offset within the slack still hits.
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    const nx = -dy / len;
    const ny = dx / len;
    const near = index.pick(mx + nx * (EDGE_PICK_SLACK - 1), my + ny * (EDGE_PICK_SLACK - 1));
    expect(near?.kind).toBe('edge');

    const far = index.pick(mx + nx * (EDGE_PICK_SLACK + 8), my + ny * (EDGE_PICK_SLACK + 8));
    expect(far?.kind === 'edge' && far.index === 0).toBe(false);
  });

  it('does not hit past the end of a segment', () => {
    // The clamped projection matters: measuring to the infinite line would
    // report a hit far beyond the edge's endpoints.
    const edge = layout.edges[0];
    if (edge === undefined) throw new Error('no edges');
    const a = layout.nodes[edge.from];
    const b = layout.nodes[edge.to];
    if (a === undefined || b === undefined) throw new Error('missing endpoint');
    const beyond = index.pick(b.x + 400, b.y + (b.y - a.y) * 4);
    expect(beyond).toBeNull();
  });

  it('carries the weight address so the inspector knows what was clicked', () => {
    const target = layout.edges[4];
    if (target === undefined) throw new Error('no edge 4');
    const a = layout.nodes[target.from];
    const b = layout.nodes[target.to];
    if (a === undefined || b === undefined) throw new Error('missing endpoint');
    // Sampled near the destination, away from crossings, so this addresses the
    // edge we mean rather than whichever edge happens to cross it mid-span.
    const reach = (b.radius + NODE_PICK_SLACK + 2) / Math.hypot(b.x - a.x, b.y - a.y);
    const t = 1 - reach;
    const hit = index.pick(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    expect(hit?.kind === 'edge' && hit.edge.layer).toBe(target.layer);
    expect(hit?.kind === 'edge' && hit.edge.row).toBe(target.row);
    expect(hit?.kind === 'edge' && hit.edge.col).toBe(target.col);
  });
});

describe('nodes win over edges', () => {
  it('a pick on a node never returns one of the edges terminating there', () => {
    // Every node sits on top of its own connections; clicking a neuron must
    // select the neuron.
    const layout = computeLayout({ sizes: [4, 6, 3] });
    const index = new HitIndex(layout);
    for (const node of layout.nodes) {
      const hit = index.pick(node.x, node.y);
      expect(hit?.kind).toBe('node');
    }
  });
});

describe('screen-space picking through a viewport', () => {
  const layout = computeLayout({ sizes: [2, 4, 1] });
  const index = new HitIndex(layout);

  it('is unaffected by zoom and pan — picking is done in world space', () => {
    const node = layout.nodes[nodeIndex(layout, 1, 1)];
    if (node === undefined) throw new Error('missing node');

    for (const viewport of [
      { scale: 1, offsetX: 0, offsetY: 0 },
      { scale: 2.5, offsetX: -120, offsetY: 40 },
      { scale: 0.4, offsetX: 300, offsetY: -60 },
    ]) {
      const screenX = node.x * viewport.scale + viewport.offsetX;
      const screenY = node.y * viewport.scale + viewport.offsetY;
      const hit = index.pickScreen(viewport, screenX, screenY);
      expect(hit?.kind === 'node' && hit.node, `scale ${viewport.scale}`).toEqual(node);
    }
  });
});

describe('distanceToSegment', () => {
  it('measures perpendicular distance inside the span', () => {
    expect(distanceToSegment(5, 3, 0, 0, 10, 0)).toBeCloseTo(3, 12);
  });

  it('measures to the nearer endpoint outside the span', () => {
    expect(distanceToSegment(-4, 3, 0, 0, 10, 0)).toBeCloseTo(5, 12);
    expect(distanceToSegment(14, 3, 0, 0, 10, 0)).toBeCloseTo(5, 12);
  });

  it('handles a degenerate zero-length segment', () => {
    expect(distanceToSegment(3, 4, 1, 1, 1, 1)).toBeCloseTo(Math.hypot(2, 3), 12);
  });
});

describe('performance shape', () => {
  it('stays fast on a large architecture', () => {
    // §6.1: "must feel instant". 35-24-24-10 is the glyph network.
    const layout = computeLayout({ sizes: [35, 24, 24, 10] });
    const started = performance.now();
    const index = new HitIndex(layout);
    const built = performance.now() - started;

    const queryStart = performance.now();
    for (let i = 0; i < 2000; i++) {
      const node = layout.nodes[i % layout.nodes.length];
      if (node !== undefined) index.pick(node.x, node.y);
    }
    const queried = performance.now() - queryStart;

    // Generous bounds: this is a regression guard against an accidental
    // O(n²) rewrite, not a benchmark.
    expect(built).toBeLessThan(250);
    expect(queried / 2000).toBeLessThan(1);
  });
});
