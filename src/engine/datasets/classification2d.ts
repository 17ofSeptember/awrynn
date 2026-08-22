/*
 * 2D classification datasets — the ones the decision-boundary view is built for.
 *
 * Spec §5. Each is chosen to fail against a specific architecture, so the
 * lessons have something honest to demonstrate:
 *
 *   xor           four corners; unsolvable without a hidden layer (§7.2)
 *   blobs         linearly separable; one neuron is a line (§7.1)
 *   circles       concentric rings; needs a curved boundary
 *   moons         interleaving half-moons; the learning-rate lesson (§7.4)
 *   spiral        the classic hard one, and the reason for cce (§7.9, §7.10)
 *   checkerboard  4x4 alternating; needs real capacity
 */

import type { Rng } from '../rng';
import { createMatrix } from '../tensor';
import type { Dataset, BaseDatasetOptions as DatasetOptions } from './types';
import { oneHot } from './types';

export type Classification2dName =
  | 'xor'
  | 'blobs'
  | 'circles'
  | 'moons'
  | 'spiral'
  | 'checkerboard';

export const CLASSIFICATION_2D: readonly Classification2dName[] = [
  'xor',
  'blobs',
  'circles',
  'moons',
  'spiral',
  'checkerboard',
];

const DEFAULT_SAMPLES = 200;
const DEFAULT_NOISE = 0.1;

interface Generated {
  readonly points: number[][];
  readonly labels: number[];
  readonly classes: number;
}

/**
 * The four XOR corners, as (x1, x2, label).
 *
 * Exported because the XOR challenge in xor.ts must check the SAME four points
 * the dataset generates. They were once defined separately, and drifted: the
 * checker tested the unit square (0/1) while the data sat on the signed square
 * (±1). The checker then reported a hand-built network as solving all four
 * while the decision boundary visibly misclassified half the points. One
 * definition, imported by both, is what stops that recurring.
 *
 * Signed rather than unit coordinates because the data is centred on the
 * origin, which is what tanh and the weight initialisers assume.
 */
export const XOR_CORNERS: readonly (readonly [number, number, number])[] = [
  [-1, -1, 0],
  [-1, 1, 1],
  [1, -1, 1],
  [1, 1, 0],
];

/** Four corners, optionally fuzzed into Gaussian clusters. */
function xor(samples: number, noise: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  const corners = XOR_CORNERS;
  for (let i = 0; i < samples; i++) {
    const corner = corners[i % 4] as readonly [number, number, number];
    points.push([corner[0] + rng.normal(0, noise), corner[1] + rng.normal(0, noise)]);
    labels.push(corner[2]);
  }
  return { points, labels, classes: 2 };
}

/** Two linearly separable Gaussians. */
function blobs(samples: number, noise: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  const centres: readonly (readonly [number, number])[] = [
    [-1.2, -0.8],
    [1.2, 0.8],
  ];
  for (let i = 0; i < samples; i++) {
    const cls = i % 2;
    const c = centres[cls] as readonly [number, number];
    points.push([c[0] + rng.normal(0, 0.4 + noise), c[1] + rng.normal(0, 0.4 + noise)]);
    labels.push(cls);
  }
  return { points, labels, classes: 2 };
}

/** Concentric rings. */
function circles(samples: number, noise: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < samples; i++) {
    const cls = i % 2;
    const radius = cls === 0 ? rng.uniform(0, 0.7) : rng.uniform(1.1, 1.6);
    const angle = rng.uniform(0, 2 * Math.PI);
    points.push([
      radius * Math.cos(angle) + rng.normal(0, noise),
      radius * Math.sin(angle) + rng.normal(0, noise),
    ]);
    labels.push(cls);
  }
  return { points, labels, classes: 2 };
}

/** Two interleaving half-moons. */
function moons(samples: number, noise: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < samples; i++) {
    const cls = i % 2;
    const angle = rng.uniform(0, Math.PI);
    if (cls === 0) {
      points.push([
        Math.cos(angle) + rng.normal(0, noise),
        Math.sin(angle) - 0.25 + rng.normal(0, noise),
      ]);
    } else {
      points.push([
        1 - Math.cos(angle) + rng.normal(0, noise),
        0.25 - Math.sin(angle) + rng.normal(0, noise),
      ]);
    }
    labels.push(cls);
  }
  return { points, labels, classes: 2 };
}

/**
 * How far each arm winds, in full turns.
 *
 * This was 3.2 radians, barely half a turn, which does not interleave the arms
 * at all: measured, a network with a SINGLE hidden unit classified it at 100%.
 * That made it useless as "the classic hard one" (§5) and destroyed the
 * capacity lesson (§7.9), whose whole point is that one unit cannot do it.
 *
 * One full turn was chosen by sweeping turn count against hidden-unit count.
 * At 1.0 the ladder is exactly what lesson 9 needs — 1 and 2 units reach ~0.60,
 * 4 and above reach 1.00 — while staying solvable, which lesson 10 and the
 * convergence smoke tests require. Beyond 1.5 turns even 16 units stall around
 * 0.7, which teaches nothing but patience.
 */
const SPIRAL_TURNS = 1;

/**
 * Spiral with `classes` arms. The hard one — and the reason cce exists in this
 * app, since three arms means three outputs and a softmax (§7.10).
 */
function spiral(samples: number, noise: number, classes: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  const perClass = Math.max(1, Math.floor(samples / classes));
  for (let cls = 0; cls < classes; cls++) {
    for (let i = 0; i < perClass; i++) {
      const t = i / perClass;
      const radius = 0.2 + 1.4 * t;
      const angle =
        t * SPIRAL_TURNS * 2 * Math.PI + (cls * 2 * Math.PI) / classes + rng.normal(0, noise);
      points.push([
        radius * Math.sin(angle) + rng.normal(0, noise * 0.5),
        radius * Math.cos(angle) + rng.normal(0, noise * 0.5),
      ]);
      labels.push(cls);
    }
  }
  return { points, labels, classes };
}

/** 4x4 alternating tiles. Needs real capacity — a small net underfits visibly. */
function checkerboard(samples: number, noise: number, rng: Rng): Generated {
  const points: number[][] = [];
  const labels: number[] = [];
  const tiles = 4;
  const extent = 2;
  for (let i = 0; i < samples; i++) {
    const x = rng.uniform(-extent, extent);
    const y = rng.uniform(-extent, extent);
    const cx = Math.floor(((x + extent) / (2 * extent)) * tiles);
    const cy = Math.floor(((y + extent) / (2 * extent)) * tiles);
    let cls = (cx + cy) % 2;
    // Label noise rather than positional noise: the tiles stay crisp, which is
    // what makes an overfitting boundary obvious when it wraps a stray point.
    if (noise > 0 && rng.next() < noise * 0.5) cls = 1 - cls;
    points.push([x, y]);
    labels.push(cls);
  }
  return { points, labels, classes: 2 };
}

export function generateClassification2d(
  name: Classification2dName,
  options: DatasetOptions,
  rng: Rng,
): Dataset {
  const samples = options.samples ?? DEFAULT_SAMPLES;
  const noise = options.noise ?? DEFAULT_NOISE;
  const requestedClasses = options.classes ?? 2;

  if (samples <= 0) {
    throw new Error(`generateClassification2d: samples must be positive, got ${samples}.`);
  }
  if (noise < 0) {
    throw new Error(`generateClassification2d: noise must be non-negative, got ${noise}.`);
  }

  let generated: Generated;
  switch (name) {
    case 'xor':
      generated = xor(samples, noise, rng);
      break;
    case 'blobs':
      generated = blobs(samples, noise, rng);
      break;
    case 'circles':
      generated = circles(samples, noise, rng);
      break;
    case 'moons':
      generated = moons(samples, noise, rng);
      break;
    case 'spiral':
      generated = spiral(samples, noise, Math.max(2, requestedClasses), rng);
      break;
    case 'checkerboard':
      generated = checkerboard(samples, noise, rng);
      break;
  }

  const count = generated.points.length;
  const x = createMatrix(count, 2);
  for (let i = 0; i < count; i++) {
    const p = generated.points[i] as number[];
    x.data[i * 2] = p[0] as number;
    x.data[i * 2 + 1] = p[1] as number;
  }
  const labels = Int32Array.from(generated.labels);

  // Binary problems stay [N, 1] for sigmoid + bce; 3+ classes go one-hot for
  // softmax + cce. That choice is what decides which loss the UI offers.
  const binary = generated.classes === 2;
  const y = binary ? createMatrix(count, 1) : oneHot(labels, generated.classes);
  if (binary) {
    for (let i = 0; i < count; i++) y.data[i] = labels[i]!;
  }

  return {
    name,
    kind: 'classification2d',
    x,
    y,
    labels,
    featureCount: 2,
    classCount: generated.classes,
    suggestedLoss: binary ? 'bce' : 'cce',
    featureNames: ['x₁', 'x₂'],
    classNames: Array.from({ length: generated.classes }, (_, i) => `class ${i}`),
  };
}
