/*
 * Lessons are DATA, not code paths (§7).
 *
 * Every lesson is a record: a preset that puts the app in a known state, a
 * description of what to watch, and a predicate evaluated against live metrics
 * that unlocks the written explanation.
 *
 * Two separate ideas live here, and keeping them apart is what makes the Phase
 * 7 gate meaningful:
 *
 *   successPredicate  what the LEARNER has to achieve. Evaluated against
 *                     whatever they have on screen, however they got there.
 *
 *   evidence          what the TEST runs to prove the phenomenon reproduces
 *                     from the stored seed. Often several configurations,
 *                     because most of these lessons are a contrast: this fails,
 *                     that works, and the difference is the point.
 *
 * A lesson whose successPredicate passes but whose evidence does not is a
 * lesson that no longer teaches what it claims.
 */

import type { InitScheme } from '../engine/init';
import type { LayerSpec } from '../engine/layers';
import type { LossName } from '../engine/losses';
import type { OptimizerConfig } from '../engine/optimizers';
import type { DatasetOptions } from '../engine/datasets/index';
import type { EpochMetrics } from '../engine/trainer';
import type { Network } from '../engine/network';

export interface LessonArchitecture {
  readonly inputSize: number;
  readonly layers: readonly LayerSpec[];
  readonly loss: LossName;
  readonly seed: number;
  readonly init: InitScheme;
  readonly l2: number;
}

export interface LessonTraining {
  readonly optimizer: OptimizerConfig;
  readonly learningRate: number;
  readonly batchSize: number;
  readonly maxEpochs: number;
  readonly dropout: number;
  readonly gradientClip: number;
  readonly standardize: boolean;
}

export interface LessonPreset {
  readonly architecture: LessonArchitecture;
  readonly dataset: DatasetOptions;
  readonly training: LessonTraining;
}

/** What a lesson's predicate gets to look at. */
export interface LessonEvidence {
  readonly metrics: readonly EpochMetrics[];
  readonly latest: EpochMetrics | null;
  readonly network: Network;
  readonly epoch: number;
  /** 'diverged' when the run blew up, which several lessons want. */
  readonly status: string;
}

/** One configuration the test runs to prove the lesson still holds. */
export interface EvidenceCase {
  readonly label: string;
  /** Overrides applied on top of the lesson's preset. */
  readonly preset: DeepPartial<LessonPreset>;
  /** What must be true after running it. Return a reason on failure. */
  readonly expect: (evidence: LessonEvidence) => string | null;
}

export interface Variant {
  readonly label: string;
  readonly note: string;
  readonly preset: DeepPartial<LessonPreset>;
}

export interface Lesson {
  readonly id: string;
  readonly number: number;
  readonly title: string;
  /** One sentence: what the learner is here to see. */
  readonly goal: string;
  readonly preset: LessonPreset;
  /** Things to look at while it runs. */
  readonly whatToWatch: readonly string[];
  /** Other configurations worth trying, offered as buttons. */
  readonly variants?: readonly Variant[];
  readonly successLabel: string;
  readonly successPredicate: (evidence: LessonEvidence) => boolean;
  /** Unlocked once the predicate passes. Never gated behind a wrong answer. */
  readonly explanation: string;
  /** What the suite runs to prove the phenomenon reproduces. */
  readonly evidence: readonly EvidenceCase[];
}

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

/** Apply a variant's overrides to a preset. */
export function mergePreset(base: LessonPreset, override: DeepPartial<LessonPreset>): LessonPreset {
  return {
    architecture: { ...base.architecture, ...(override.architecture ?? {}) } as LessonArchitecture,
    dataset: { ...base.dataset, ...(override.dataset ?? {}) } as DatasetOptions,
    training: { ...base.training, ...(override.training ?? {}) } as LessonTraining,
  };
}

/* ------------------------------------------------------------------ *
 * Predicate helpers
 *
 * Small named functions rather than inline arrow soup: a lesson's success
 * condition is part of what it teaches, and it should be readable as a sentence.
 * ------------------------------------------------------------------ */

export function bestValidationAccuracy(evidence: LessonEvidence): number {
  let best = 0;
  for (const m of evidence.metrics) best = Math.max(best, m.validationAccuracy ?? 0);
  return best;
}

export function finalTrainLoss(evidence: LessonEvidence): number {
  return evidence.latest?.trainLoss ?? Infinity;
}

/**
 * Ratio of the last layer's gradient norm to the first's, EARLY in training.
 *
 * Measured over the first few epochs rather than at the end, because a
 * converged network has small gradients everywhere and the ratio collapses
 * toward 1. Measuring at the end conflated "the gradient vanished on the way
 * back" with "the network finished learning", and reported a deep sigmoid stack
 * as perfectly healthy. Vanishing is a property of the backward pass, and it is
 * visible from the first step.
 */
export const GRADIENT_RATIO_WINDOW = 5;

export function gradientRatio(evidence: LessonEvidence): number {
  const window = evidence.metrics.slice(0, GRADIENT_RATIO_WINDOW);
  let total = 0;
  let counted = 0;
  for (const m of window) {
    const norms = m.gradientNorms;
    const first = norms[0] ?? 0;
    const last = norms[norms.length - 1] ?? 0;
    if (first <= 0 || last <= 0) continue;
    total += last / first;
    counted++;
  }
  // NaN, not 1, when nothing has run. Returning a healthy-looking default made
  // "get the first layer within 100x of the last" true before training even
  // started, so the lesson congratulated the reader for doing nothing.
  return counted === 0 ? Number.NaN : total / counted;
}

/** How far validation loss rose from its best, the signature of overfitting. */
export function validationRebound(evidence: LessonEvidence): number {
  let best = Infinity;
  let final = Infinity;
  for (const m of evidence.metrics) {
    if (m.validationLoss === null) continue;
    best = Math.min(best, m.validationLoss);
    final = m.validationLoss;
  }
  // NaN rather than 0 with no data, for the same reason as gradientRatio: a
  // rebound of zero reads as "did not overfit", which is a claim about a run
  // that has not happened.
  if (!Number.isFinite(best) || !Number.isFinite(final)) return Number.NaN;
  return final - best;
}

/** Spread of the training loss, which is what batch size changes (§7.11). */
/** True once there is at least one epoch of evidence to judge. */
export function hasEvidence(evidence: LessonEvidence): boolean {
  return evidence.metrics.length > 0;
}

export function lossNoise(evidence: LessonEvidence): number {
  const tail = evidence.metrics.slice(Math.floor(evidence.metrics.length / 2));
  const values = tail.map((m) => m.trainLoss).filter((v) => Number.isFinite(v));
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
}
