import { describe, expect, it } from 'vitest';
import { drawThumbnails, THUMBNAIL_SIZE, ThumbnailSheet } from '../draw/thumbnails';
import type { BlitContext, ThumbnailSource } from '../draw/thumbnails';
import { computeLayout } from '../layout';
import { WEIGHT_NEGATIVE, WEIGHT_POSITIVE, parseHex } from '../theme';

/*
 * §6.4: "every hidden unit gets a small heatmap of its own activation across
 * the input plane, drawn right at the node."
 */

interface FakeImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** A canvas stand-in that records what was baked and blitted. */
class FakeCanvas {
  width = 0;
  height = 0;
  readonly puts: FakeImageData[] = [];
  getContext(): {
    createImageData: (w: number, h: number) => FakeImageData;
    putImageData: (image: FakeImageData) => void;
  } {
    return {
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      putImageData: (image: FakeImageData) => this.puts.push(image),
    };
  }
}

class BlitRecorder {
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 1;
  lineCap: CanvasLineCap = 'butt';
  lineJoin: CanvasLineJoin = 'miter';
  font = '';
  textAlign: CanvasTextAlign = 'start';
  textBaseline: CanvasTextBaseline = 'alphabetic';
  globalAlpha = 1;
  imageSmoothingEnabled = false;
  readonly blits: { dx: number; dy: number; dw: number; sx: number }[] = [];
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
  fillText(): void {}
  measureText(t: string): { width: number } { return { width: t.length }; }
  setLineDash(): void {}
  setTransform(): void {}
  drawImage(
    _image: unknown,
    sx: number,
    _sy: number,
    _sw: number,
    _sh: number,
    dx: number,
    dy: number,
    dw: number,
  ): void {
    this.blits.push({ dx, dy, dw, sx });
  }
}

function source(resolution = 4): ThumbnailSource {
  const slots = [
    { layer: 0, unit: 0, offset: 0 },
    { layer: 0, unit: 1, offset: resolution * resolution },
  ];
  const values = new Float32Array(slots.length * resolution * resolution);
  // Unit 0 fully negative, unit 1 fully positive, so the poles are checkable.
  values.fill(-1, 0, resolution * resolution);
  values.fill(1, resolution * resolution);
  return { values, slots, resolution };
}

describe('thumbnail sprite sheet', () => {
  it('bakes every tile into one canvas laid out in a row', () => {
    const sheet = new ThumbnailSheet();
    const canvas = new FakeCanvas();
    sheet.build(source(4), () => canvas as unknown as HTMLCanvasElement);
    // One sheet, two tiles wide: a drawImage per node beats an ImageData per node.
    expect(canvas.width).toBe(8);
    expect(canvas.height).toBe(4);
    expect(canvas.puts.length).toBe(1);
    expect(sheet.ready).toBe(true);
  });

  it('uses the weight poles, so a unit map and its edges read alike', () => {
    const sheet = new ThumbnailSheet();
    const canvas = new FakeCanvas();
    sheet.build(source(2), () => canvas as unknown as HTMLCanvasElement);
    const pixels = canvas.puts[0]!.data;
    const neg = parseHex(WEIGHT_NEGATIVE);
    const pos = parseHex(WEIGHT_POSITIVE);
    // Tile 0 is the negative unit, tile 1 the positive one.
    expect([pixels[0], pixels[1], pixels[2]]).toEqual([neg.r, neg.g, neg.b]);
    const secondTile = 2 * 4;
    expect([pixels[secondTile], pixels[secondTile + 1], pixels[secondTile + 2]]).toEqual([
      pos.r,
      pos.g,
      pos.b,
    ]);
  });

  it('maps each unit to its own tile', () => {
    const sheet = new ThumbnailSheet();
    sheet.build(source(), () => new FakeCanvas() as unknown as HTMLCanvasElement);
    expect(sheet.tileFor(0, 0)).toBe(0);
    expect(sheet.tileFor(0, 1)).toBe(1);
    expect(sheet.tileFor(9, 9)).toBeUndefined();
  });

  it('rebuilds only when the data changes', () => {
    // The sheet is baked from an ImageData; doing that every frame would
    // allocate and upload on every tick.
    const sheet = new ThumbnailSheet();
    const data = source();
    expect(sheet.matches(data)).toBe(false);
    sheet.build(data, () => new FakeCanvas() as unknown as HTMLCanvasElement);
    expect(sheet.matches(data)).toBe(true);
    expect(sheet.matches(source())).toBe(false);
  });

  it('clears cleanly when thumbnails are switched off', () => {
    const sheet = new ThumbnailSheet();
    sheet.build(source(), () => new FakeCanvas() as unknown as HTMLCanvasElement);
    sheet.build(null, () => new FakeCanvas() as unknown as HTMLCanvasElement);
    expect(sheet.ready).toBe(false);
    expect(sheet.tileFor(0, 0)).toBeUndefined();
  });
});

describe('drawing thumbnails at nodes', () => {
  it('draws one per hidden unit, and none for inputs, outputs or biases', () => {
    const layout = computeLayout({ sizes: [2, 2, 1] });
    const sheet = new ThumbnailSheet();
    sheet.build(source(), () => new FakeCanvas() as unknown as HTMLCanvasElement);

    const ctx = new BlitRecorder();
    drawThumbnails(ctx as unknown as BlitContext, layout, { scale: 1, offsetX: 0, offsetY: 0 }, sheet);

    // 2 hidden units only: the input column, the output and every bias are skipped.
    expect(ctx.blits.length).toBe(2);
    expect(ctx.blits.every((b) => b.dw === THUMBNAIL_SIZE)).toBe(true);
    // Each blit reads a different tile.
    expect(new Set(ctx.blits.map((b) => b.sx)).size).toBe(2);
  });

  it('draws nothing when the sheet is empty', () => {
    const layout = computeLayout({ sizes: [2, 3, 1] });
    const ctx = new BlitRecorder();
    drawThumbnails(
      ctx as unknown as BlitContext,
      layout,
      { scale: 1, offsetX: 0, offsetY: 0 },
      new ThumbnailSheet(),
    );
    expect(ctx.blits.length).toBe(0);
  });

  it('places the tile clear of the node so its fill stays readable', () => {
    const layout = computeLayout({ sizes: [2, 1, 1] });
    const sheet = new ThumbnailSheet();
    sheet.build(
      { values: new Float32Array(16), slots: [{ layer: 0, unit: 0, offset: 0 }], resolution: 4 },
      () => new FakeCanvas() as unknown as HTMLCanvasElement,
    );
    const ctx = new BlitRecorder();
    drawThumbnails(ctx as unknown as BlitContext, layout, { scale: 1, offsetX: 0, offsetY: 0 }, sheet);
    const node = layout.nodes.find((n) => n.kind === 'hidden');
    expect(node).toBeDefined();
    const blit = ctx.blits[0]!;
    // Above and to the right, not centred on the node.
    expect(blit.dx).toBeGreaterThan(node!.x);
    expect(blit.dy).toBeLessThan(node!.y);
  });
});
