/*
 * Procedural 7x5 binary glyph bitmaps.
 *
 * Spec §5. 35 inputs, K classes, with jitter (±1px shift), rotation and
 * salt-and-pepper noise. This is the bridge from toy 2D to "oh, that's what
 * MNIST is" — same shape of problem, small enough that every input pixel still
 * gets its own visible node on the canvas.
 *
 * The base glyphs are hand-drawn below as ASCII so they can be read and edited
 * as pictures rather than as bit patterns. '#' is ink, '.' is background.
 */

import type { Rng } from '../rng';
import { createMatrix } from '../tensor';
import type { Dataset, BaseDatasetOptions as DatasetOptions } from './types';
import { oneHot } from './types';

export const GLYPH_WIDTH = 5;
export const GLYPH_HEIGHT = 7;
export const GLYPH_PIXELS = GLYPH_WIDTH * GLYPH_HEIGHT;

/** Digits 0-9 plus a few letters, per §5. */
export const GLYPH_LABELS: readonly string[] = [
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  'A',
  'E',
  'H',
  'L',
];

const GLYPH_ART: Readonly<Record<string, readonly string[]>> = {
  '0': ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
  '1': ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  '2': ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
  '3': ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
  '4': ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
  '5': ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
  '6': ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
  '7': ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
  '8': ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
  '9': ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
  H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
};

/** The clean bitmap for a label, row-major, 1 = ink. */
export function glyphBitmap(label: string): Float64Array {
  const art = GLYPH_ART[label];
  if (art === undefined) {
    throw new Error(`glyphBitmap: no glyph defined for ${JSON.stringify(label)}.`);
  }
  const out = new Float64Array(GLYPH_PIXELS);
  for (let r = 0; r < GLYPH_HEIGHT; r++) {
    const row = art[r] as string;
    for (let c = 0; c < GLYPH_WIDTH; c++) {
      out[r * GLYPH_WIDTH + c] = row.charAt(c) === '#' ? 1 : 0;
    }
  }
  return out;
}

/**
 * Sample the glyph at a rotated, shifted position.
 *
 * Nearest-neighbour rather than bilinear: the input is binary and the draw pad
 * the learner uses is binary, so producing grey training pixels would train the
 * network on a distribution it never sees at inference.
 */
function transform(
  source: Float64Array,
  dx: number,
  dy: number,
  angle: number,
  out: Float64Array,
): void {
  const cx = (GLYPH_WIDTH - 1) / 2;
  const cy = (GLYPH_HEIGHT - 1) / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let r = 0; r < GLYPH_HEIGHT; r++) {
    for (let c = 0; c < GLYPH_WIDTH; c++) {
      // Inverse-map the destination pixel back into the source.
      const ox = c - cx - dx;
      const oy = r - cy - dy;
      const sx = Math.round(cx + ox * cos + oy * sin);
      const sy = Math.round(cy - ox * sin + oy * cos);
      out[r * GLYPH_WIDTH + c] =
        sx >= 0 && sx < GLYPH_WIDTH && sy >= 0 && sy < GLYPH_HEIGHT
          ? source[sy * GLYPH_WIDTH + sx]!
          : 0;
    }
  }
}

export function generateGlyphs(options: DatasetOptions, rng: Rng): Dataset {
  const classCount = Math.min(options.classes ?? 10, GLYPH_LABELS.length);
  const samples = options.samples ?? classCount * 20;
  const noise = options.noise ?? 0.05;
  if (classCount < 2) {
    throw new Error(`generateGlyphs: need at least 2 classes, got ${classCount}.`);
  }

  const labels = GLYPH_LABELS.slice(0, classCount);
  const clean = labels.map((label) => glyphBitmap(label));

  const count = Math.max(classCount, samples);
  const x = createMatrix(count, GLYPH_PIXELS);
  const classIndices = new Int32Array(count);
  const scratch = new Float64Array(GLYPH_PIXELS);

  for (let i = 0; i < count; i++) {
    // Round-robin so classes stay balanced regardless of sample count.
    const cls = i % classCount;
    classIndices[i] = cls;

    const dx = rng.int(3) - 1; // ±1px shift, per §5
    const dy = rng.int(3) - 1;
    const angle = rng.uniform(-0.22, 0.22); // ≈ ±12.5°
    transform(clean[cls] as Float64Array, dx, dy, angle, scratch);

    // Salt and pepper: flip each pixel with probability `noise`.
    const offset = i * GLYPH_PIXELS;
    for (let p = 0; p < GLYPH_PIXELS; p++) {
      const value = scratch[p]!;
      x.data[offset + p] = rng.next() < noise ? 1 - value : value;
    }
  }

  return {
    name: 'glyphs',
    kind: 'glyphs',
    x,
    y: oneHot(classIndices, classCount),
    labels: classIndices,
    featureCount: GLYPH_PIXELS,
    classCount,
    suggestedLoss: 'cce',
    featureNames: Array.from({ length: GLYPH_PIXELS }, (_, i) => {
      return `p${Math.floor(i / GLYPH_WIDTH)},${i % GLYPH_WIDTH}`;
    }),
    classNames: labels,
  };
}
