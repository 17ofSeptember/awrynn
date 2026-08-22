/*
 * Per-neuron activation thumbnails (§6.4).
 *
 * "every hidden unit gets a small heatmap of its own activation across the
 * input plane, drawn right at the node. This is the moment learners understand
 * composition — early layers make lines, later layers make regions out of them."
 *
 * All tiles are baked into ONE offscreen canvas as a sprite sheet, then each
 * node draws its own tile with a single drawImage. Building an ImageData per
 * node per frame would allocate and upload dozens of small textures every
 * frame; a sprite sheet is rebuilt only when the data actually changes.
 */

import type { Layout, NodeLayout, Viewport } from '../layout';
import { worldToScreenX, worldToScreenY } from '../layout';
import { COLORS, WEIGHT_NEGATIVE, WEIGHT_POSITIVE, parseHex } from '../theme';
import type { Ctx2D } from './context';

export interface ThumbnailSource {
  readonly values: Float32Array;
  readonly slots: readonly { readonly layer: number; readonly unit: number; readonly offset: number }[];
  readonly resolution: number;
}

const NEG = parseHex(WEIGHT_NEGATIVE);
const POS = parseHex(WEIGHT_POSITIVE);

/** Screen size of a tile, before viewport scaling. */
export const THUMBNAIL_SIZE = 34;

export class ThumbnailSheet {
  private canvas: HTMLCanvasElement | null = null;
  private index = new Map<string, number>();
  private resolution = 0;
  private built: ThumbnailSource | null = null;

  /** True when the sheet already reflects this data. */
  matches(source: ThumbnailSource | null): boolean {
    return this.built === source;
  }

  /**
   * Bake every tile into one sprite sheet, laid out in a single row.
   *
   * Uses the same two poles as the weights: a unit's map and the edges leaving
   * it describe the same sign convention, so they should read alike.
   */
  build(source: ThumbnailSource | null, createCanvas: () => HTMLCanvasElement): void {
    this.built = source;
    if (source === null || source.slots.length === 0 || source.resolution === 0) {
      this.canvas = null;
      this.index.clear();
      return;
    }

    const t = source.resolution;
    const count = source.slots.length;
    const canvas = this.canvas ?? createCanvas();
    if (canvas.width !== t * count || canvas.height !== t) {
      canvas.width = t * count;
      canvas.height = t;
    }
    this.canvas = canvas;
    this.resolution = t;

    const ctx = canvas.getContext('2d');
    if (ctx === null) return;
    const image = ctx.createImageData(t * count, t);
    const pixels = image.data;

    this.index.clear();
    source.slots.forEach((slot, tile) => {
      this.index.set(`${slot.layer}:${slot.unit}`, tile);
      for (let row = 0; row < t; row++) {
        for (let col = 0; col < t; col++) {
          const value = source.values[slot.offset + row * t + col] ?? 0;
          const magnitude = Math.min(1, Math.abs(value));
          const color = value < 0 ? NEG : POS;
          const o = (row * (t * count) + tile * t + col) * 4;
          pixels[o] = color.r;
          pixels[o + 1] = color.g;
          pixels[o + 2] = color.b;
          pixels[o + 3] = Math.round(24 + 210 * magnitude);
        }
      }
    });
    ctx.putImageData(image, 0, 0);
  }

  get ready(): boolean {
    return this.canvas !== null;
  }

  tileFor(layer: number, unit: number): number | undefined {
    return this.index.get(`${layer}:${unit}`);
  }

  get sheet(): HTMLCanvasElement | null {
    return this.canvas;
  }

  get tileSize(): number {
    return this.resolution;
  }
}

/** A canvas context that can also blit another canvas. */
export interface BlitContext extends Ctx2D {
  drawImage(
    image: CanvasImageSource,
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    dx: number,
    dy: number,
    dw: number,
    dh: number,
  ): void;
  imageSmoothingEnabled: boolean;
}

/**
 * Draw each hidden unit's map beside its node.
 *
 * Placed above-right of the node rather than centred on it, so the node's own
 * fill and its delta ring stay readable underneath.
 */
export function drawThumbnails(
  ctx: BlitContext,
  layout: Layout,
  viewport: Viewport,
  sheet: ThumbnailSheet,
): void {
  const image = sheet.sheet;
  if (image === null) return;
  const t = sheet.tileSize;
  const size = THUMBNAIL_SIZE;

  ctx.setLineDash([]);
  ctx.imageSmoothingEnabled = true;

  for (const node of layout.nodes as readonly NodeLayout[]) {
    // Input and output columns are not hidden units; a bias has no map.
    if (node.kind !== 'hidden') continue;
    const tile = sheet.tileFor(node.layer - 1, node.unit);
    if (tile === undefined) continue;

    const x = worldToScreenX(viewport, node.x) + node.radius * viewport.scale + 4;
    const y = worldToScreenY(viewport, node.y) - size - 4;

    ctx.fillStyle = COLORS.bgCanvas;
    ctx.fillRect(x, y, size, size);
    ctx.drawImage(image, tile * t, 0, t, t, x, y, size, size);
    ctx.strokeStyle = COLORS.lineHair;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(x + 0.5, y + 0.5, size - 1, size - 1);
    ctx.stroke();
  }
}
