/*
 * Pointer hit-testing.
 *
 * Spec §6.1: "against a rebuilt-on-layout-change spatial index: nodes by
 * radius, edges by distance-to-segment with a generous threshold. Must feel
 * instant."
 *
 * A uniform grid rather than a quadtree. The layouts here are a few hundred
 * items in a known bounding box, which is exactly the regime where a flat grid
 * beats a tree: building it is a single linear pass, and a query touches a
 * handful of buckets with no pointer chasing.
 *
 * Everything works in WORLD coordinates. The caller converts the pointer once,
 * which keeps the test exact instead of accumulating rounding through a
 * screen-space round trip, and means zoom never changes what is under the
 * cursor.
 */

import type { EdgeLayout, Layout, NodeLayout, Viewport } from './layout';
import { screenToWorldX, screenToWorldY } from './layout';

export type HitKind = 'node' | 'edge';

export interface NodeHit {
  readonly kind: 'node';
  readonly index: number;
  readonly node: NodeLayout;
  /** World distance from the pointer to the node centre. */
  readonly distance: number;
}

export interface EdgeHit {
  readonly kind: 'edge';
  readonly index: number;
  readonly edge: EdgeLayout;
  /** World distance from the pointer to the segment. */
  readonly distance: number;
}

export type Hit = NodeHit | EdgeHit;

/**
 * Extra world-space slack around an edge, so a 0.5px hairline is still
 * grabbable. §6.5 has learners scrubbing weights by dragging edges, which is
 * unusable if the target is literally the stroke.
 */
export const EDGE_PICK_SLACK = 6;
/** Slack around a node, on top of its radius. */
export const NODE_PICK_SLACK = 4;

const CELL_SIZE = 64;

/** World-space slop within which two edge distances count as equal. */
const TIE_EPSILON = 1e-9;

interface Bucket {
  readonly nodes: number[];
  readonly edges: number[];
}

export class HitIndex {
  private readonly layout: Layout;
  private readonly cols: number;
  private readonly rows: number;
  private readonly minX: number;
  private readonly minY: number;
  private readonly buckets: Bucket[];

  constructor(layout: Layout) {
    this.layout = layout;
    // Pad the bounds so a pick just outside the layout still lands in a bucket.
    const pad = CELL_SIZE;
    this.minX = -pad;
    this.minY = -pad;
    this.cols = Math.max(1, Math.ceil((layout.width + pad * 2) / CELL_SIZE));
    this.rows = Math.max(1, Math.ceil((layout.height + pad * 2) / CELL_SIZE));
    this.buckets = Array.from({ length: this.cols * this.rows }, () => ({
      nodes: [] as number[],
      edges: [] as number[],
    }));

    layout.nodes.forEach((node, index) => {
      const reach = node.radius + NODE_PICK_SLACK;
      this.forEachCellInBox(node.x - reach, node.y - reach, node.x + reach, node.y + reach, (b) =>
        b.nodes.push(index),
      );
    });

    layout.edges.forEach((edge, index) => {
      const a = layout.nodes[edge.from];
      const b = layout.nodes[edge.to];
      if (a === undefined || b === undefined) return;
      // An edge is registered in every cell its bounding box touches. Edges here
      // are short and mostly horizontal, so the box is a tight approximation.
      this.forEachCellInBox(
        Math.min(a.x, b.x) - EDGE_PICK_SLACK,
        Math.min(a.y, b.y) - EDGE_PICK_SLACK,
        Math.max(a.x, b.x) + EDGE_PICK_SLACK,
        Math.max(a.y, b.y) + EDGE_PICK_SLACK,
        (bucket) => bucket.edges.push(index),
      );
    });
  }

  private forEachCellInBox(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    visit: (bucket: Bucket) => void,
  ): void {
    const c0 = Math.max(0, Math.floor((x0 - this.minX) / CELL_SIZE));
    const c1 = Math.min(this.cols - 1, Math.floor((x1 - this.minX) / CELL_SIZE));
    const r0 = Math.max(0, Math.floor((y0 - this.minY) / CELL_SIZE));
    const r1 = Math.min(this.rows - 1, Math.floor((y1 - this.minY) / CELL_SIZE));
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const bucket = this.buckets[r * this.cols + c];
        if (bucket !== undefined) visit(bucket);
      }
    }
  }

  private bucketAt(x: number, y: number): Bucket | undefined {
    const c = Math.floor((x - this.minX) / CELL_SIZE);
    const r = Math.floor((y - this.minY) / CELL_SIZE);
    if (c < 0 || c >= this.cols || r < 0 || r >= this.rows) return undefined;
    return this.buckets[r * this.cols + c];
  }

  /**
   * Nearest thing under a world-space point, or null.
   *
   * Nodes win ties against edges regardless of distance, because every node sits
   * on top of the edges that terminate at it and clicking a neuron must never
   * select one of its own connections.
   */
  pick(worldX: number, worldY: number): Hit | null {
    const bucket = this.bucketAt(worldX, worldY);
    if (bucket === undefined) return null;

    let bestNode: NodeHit | null = null;
    for (const index of bucket.nodes) {
      const node = this.layout.nodes[index];
      if (node === undefined) continue;
      const distance = Math.hypot(worldX - node.x, worldY - node.y);
      if (distance <= node.radius + NODE_PICK_SLACK) {
        if (bestNode === null || distance < bestNode.distance) {
          bestNode = { kind: 'node', index, node, distance };
        }
      }
    }
    if (bestNode !== null) return bestNode;

    /*
     * Edges in a fully-connected layout genuinely cross, and at a crossing two
     * edges are exactly equidistant — this is geometry, not a tolerance
     * problem. Ties go to the LATER edge in the array, because that is the one
     * drawn on top (draw order follows array order), so the pick matches what
     * the pointer is actually over. Without a defined tie-break the winner
     * would depend on spatial-bucket iteration order.
     */
    let bestEdge: EdgeHit | null = null;
    for (const index of bucket.edges) {
      const edge = this.layout.edges[index];
      if (edge === undefined) continue;
      const a = this.layout.nodes[edge.from];
      const b = this.layout.nodes[edge.to];
      if (a === undefined || b === undefined) continue;
      const distance = distanceToSegment(worldX, worldY, a.x, a.y, b.x, b.y);
      if (distance > EDGE_PICK_SLACK) continue;
      if (
        bestEdge === null ||
        distance < bestEdge.distance - TIE_EPSILON ||
        (Math.abs(distance - bestEdge.distance) <= TIE_EPSILON && index > bestEdge.index)
      ) {
        bestEdge = { kind: 'edge', index, edge, distance };
      }
    }
    return bestEdge;
  }

  /** Convenience: pick from screen coordinates through a viewport. */
  pickScreen(viewport: Viewport, screenX: number, screenY: number): Hit | null {
    return this.pick(screenToWorldX(viewport, screenX), screenToWorldY(viewport, screenY));
  }
}

/** Perpendicular distance from a point to a line SEGMENT, not an infinite line. */
export function distanceToSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(px - ax, py - ay);
  // Clamped projection: without the clamp this measures to the infinite line,
  // and a point far off the end of an edge would register as a hit.
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
