import { describe, expect, it } from 'vitest';
import { LESSONS } from '../index';
import { mergePreset } from '../types';
import type { LessonEvidence, LessonPreset } from '../types';
import { generateDataset } from '../../engine/datasets/index';
import { Trainer } from '../../engine/trainer';
import type { TrainerConfig } from '../../engine/trainer';

/*
 * Spec §11 Phase 7 gate:
 *
 *   "every lesson reproducibly demonstrates its phenomenon from its stored
 *    seed — run each one and confirm."
 *
 * This is that. Each lesson's `evidence` cases are actually trained, headlessly,
 * from the seeds stored in the lesson, and the assertions are the lesson's own.
 * A lesson that stops teaching what it claims fails here rather than quietly
 * misleading a reader.
 */

function run(preset: LessonPreset): LessonEvidence {
  const config: TrainerConfig = {
    network: {
      inputSize: preset.architecture.inputSize,
      layers: preset.architecture.layers,
      loss: preset.architecture.loss,
      seed: preset.architecture.seed,
      init: preset.architecture.init,
      l2: preset.architecture.l2,
    },
    optimizer: preset.training.optimizer,
    learningRate: preset.training.learningRate,
    batchSize: preset.training.batchSize,
    validationFraction: preset.dataset.validationFraction ?? 0.2,
    dropout: preset.training.dropout,
    gradientClip: preset.training.gradientClip,
    standardize: preset.training.standardize,
  };
  const trainer = new Trainer(config, generateDataset(preset.dataset));
  const metrics = trainer.run(preset.training.maxEpochs);
  return {
    metrics,
    latest: metrics[metrics.length - 1] ?? null,
    network: trainer.network,
    epoch: trainer.epoch,
    status: trainer.status,
  };
}

describe('the lessons (§7)', () => {
  it('the twelve §7 asks for are all present, numbered 1 to 12', () => {
    /*
     * Pinned by id rather than by count. The spec names twelve; Phase 9 added a
     * thirteenth for batch normalization. An extension must not be able to
     * quietly displace one of the twelve, so the twelve are listed.
     */
    const required = [
      'neuron-is-a-line',
      'xor-needs-a-hidden-layer',
      'zero-init-never-breaks-symmetry',
      'learning-rate',
      'vanishing-gradients',
      'dead-relus',
      'overfitting',
      'feature-scaling',
      'capacity-vs-data',
      'softmax-and-cross-entropy',
      'batch-size',
      'optimizer-race',
    ];
    expect(LESSONS.slice(0, 12).map((l) => l.id)).toEqual(required);
    expect(LESSONS.slice(0, 12).map((l) => l.number)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
  });

  it('numbers are consecutive from 1 and ids are unique', () => {
    expect(LESSONS.map((l) => l.number)).toEqual(LESSONS.map((_, i) => i + 1));
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
  });

  it('every lesson is complete data, with no empty prose', () => {
    for (const lesson of LESSONS) {
      expect(lesson.title.length, lesson.id).toBeGreaterThan(0);
      expect(lesson.goal.length, lesson.id).toBeGreaterThan(20);
      expect(lesson.explanation.length, lesson.id).toBeGreaterThan(120);
      expect(lesson.whatToWatch.length, lesson.id).toBeGreaterThanOrEqual(2);
      expect(lesson.successLabel.length, lesson.id).toBeGreaterThan(0);
      expect(lesson.evidence.length, lesson.id).toBeGreaterThan(0);
    }
  });

  it('every preset stores an explicit seed, so a lesson is reproducible', () => {
    for (const lesson of LESSONS) {
      expect(Number.isInteger(lesson.preset.architecture.seed), lesson.id).toBe(true);
      expect(lesson.preset.dataset.seed, lesson.id).toBeDefined();
    }
  });

  it('no lesson reports success before anything has been trained', () => {
    /*
     * The bug this exists for: several helpers returned a healthy-looking
     * default with no data — a gradient ratio of 1, a validation rebound of 0 —
     * so the predicate was true on an untrained network and the lesson
     * congratulated the reader for doing nothing. A success condition that is
     * already met teaches the opposite of what it was written for.
     */
    for (const lesson of LESSONS) {
      const untrained = run({
        ...lesson.preset,
        training: { ...lesson.preset.training, maxEpochs: 0 },
      });
      expect(untrained.metrics.length, lesson.id).toBe(0);
      expect(lesson.successPredicate(untrained), `${lesson.id} claims success untrained`).toBe(false);
    }
  });

  it('every preset is a valid configuration', () => {
    for (const lesson of LESSONS) {
      expect(() => run({ ...lesson.preset, training: { ...lesson.preset.training, maxEpochs: 1 } }), lesson.id).not.toThrow();
    }
  });
});

/*
 * The gate itself. One test per evidence case, so a failure names the exact
 * claim that stopped being true rather than "lesson 7 broke".
 */
for (const lesson of LESSONS) {
  describe(`${lesson.number}. ${lesson.title}`, () => {
    for (const evidence of lesson.evidence) {
      it(evidence.label, () => {
        const result = run(mergePreset(lesson.preset, evidence.preset));
        const failure = evidence.expect(result);
        expect(failure, failure ?? undefined).toBeNull();
      });
    }
  });
}
