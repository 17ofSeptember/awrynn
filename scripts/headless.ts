/*
 * Headless training runs — the Phase 2 gate (§11).
 *
 * "XOR and moons solvable from a headless script." Nothing here touches the
 * DOM, a canvas or React: it constructs the same Trainer the worker uses and
 * prints what it measured. If this file ever needs a browser shim to run, the
 * engine has stopped being pure and §0.5 has been violated.
 *
 *   npm run headless
 */

import { generateDataset } from '../src/engine/datasets/index';
import { gradientCheck, formatGradCheckResult } from '../src/engine/gradcheck';
import { Trainer } from '../src/engine/trainer';
import type { TrainerConfig } from '../src/engine/trainer';
import type { DatasetOptions } from '../src/engine/datasets/index';

interface Task {
  readonly label: string;
  readonly config: TrainerConfig;
  readonly dataset: DatasetOptions;
  readonly maxEpochs: number;
  /** Passing criterion, evaluated against the run. */
  readonly target: string;
  readonly check: (best: Summary) => boolean;
}

interface Summary {
  readonly epochs: number;
  readonly finalLoss: number;
  readonly bestValidationAccuracy: number;
  readonly wallClockMs: number;
  readonly epochsPerSecond: number;
}

const TASKS: readonly Task[] = [
  {
    label: 'XOR',
    target: 'train loss < 0.02 within 2000 epochs',
    maxEpochs: 2000,
    dataset: { name: 'xor', samples: 200, noise: 0.1, seed: 1 },
    config: {
      network: {
        inputSize: 2,
        layers: [
          { units: 8, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        loss: 'bce',
        seed: 7,
        init: { kind: 'glorot_uniform' },
      },
      optimizer: { name: 'adam' },
      learningRate: 0.05,
      batchSize: 16,
      validationFraction: 0.2,
    },
    check: (s) => s.finalLoss < 0.02,
  },
  {
    label: 'moons',
    target: 'validation accuracy > 0.95',
    maxEpochs: 300,
    dataset: { name: 'moons', samples: 300, noise: 0.12, seed: 2 },
    config: {
      network: {
        inputSize: 2,
        layers: [
          { units: 12, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        loss: 'bce',
        seed: 3,
        init: { kind: 'glorot_uniform' },
      },
      optimizer: { name: 'adam' },
      learningRate: 0.03,
      batchSize: 16,
      validationFraction: 0.2,
    },
    check: (s) => s.bestValidationAccuracy > 0.95,
  },
  {
    label: 'spiral (3 arms)',
    target: 'validation accuracy > 0.90',
    maxEpochs: 600,
    dataset: { name: 'spiral', samples: 600, noise: 0.06, seed: 4, classes: 3 },
    config: {
      network: {
        inputSize: 2,
        layers: [
          { units: 24, activation: 'tanh' },
          { units: 24, activation: 'tanh' },
          { units: 3, activation: 'softmax' },
        ],
        loss: 'cce',
        seed: 11,
        init: { kind: 'glorot_uniform' },
      },
      optimizer: { name: 'adam' },
      learningRate: 0.02,
      batchSize: 32,
      validationFraction: 0.2,
    },
    check: (s) => s.bestValidationAccuracy > 0.9,
  },
  {
    label: 'glyphs (10 classes)',
    target: 'validation accuracy > 0.70',
    maxEpochs: 300,
    dataset: { name: 'glyphs', samples: 600, noise: 0.03, seed: 6, classes: 10 },
    config: {
      network: {
        inputSize: 35,
        layers: [
          { units: 24, activation: 'relu' },
          { units: 10, activation: 'softmax' },
        ],
        loss: 'cce',
        seed: 5,
        init: { kind: 'he_normal' },
      },
      optimizer: { name: 'adam' },
      learningRate: 0.01,
      batchSize: 32,
      validationFraction: 0.2,
    },
    check: (s) => s.bestValidationAccuracy > 0.7,
  },
];

function run(task: Task): Summary {
  const trainer = new Trainer(task.config, generateDataset(task.dataset));
  const started = performance.now();
  let best = 0;
  let finalLoss = Infinity;
  let epochs = 0;
  for (let i = 0; i < task.maxEpochs; i++) {
    const metrics = trainer.runEpoch();
    epochs++;
    finalLoss = metrics.trainLoss;
    best = Math.max(best, metrics.validationAccuracy ?? 0);
    if (trainer.status !== 'running') break;
    if (task.label === 'XOR' && finalLoss < 0.02) break;
  }
  const wallClockMs = performance.now() - started;
  return {
    epochs,
    finalLoss,
    bestValidationAccuracy: best,
    wallClockMs,
    epochsPerSecond: (epochs / wallClockMs) * 1000,
  };
}

function main(): void {
  process.stdout.write('AwryNN — headless training (engine only, no DOM)\n\n');
  let failures = 0;

  for (const task of TASKS) {
    const summary = run(task);
    const passed = task.check(summary);
    if (!passed) failures++;
    process.stdout.write(
      `${passed ? 'PASS' : 'FAIL'}  ${task.label.padEnd(20)} ` +
        `epochs ${String(summary.epochs).padStart(5)}  ` +
        `loss ${summary.finalLoss.toFixed(6)}  ` +
        `val acc ${summary.bestValidationAccuracy.toFixed(4)}  ` +
        `${summary.epochsPerSecond.toFixed(0)} epoch/s\n` +
        `      ${task.target}\n`,
    );
  }

  // The correctness backbone, on a live network rather than a fixture.
  const verifyTrainer = new Trainer(
    TASKS[1]!.config,
    generateDataset(TASKS[1]!.dataset),
  );
  verifyTrainer.run(10);
  const check = gradientCheck(
    verifyTrainer.network,
    verifyTrainer.split.train.x,
    verifyTrainer.split.train.y,
  );
  if (!check.passed) failures++;
  process.stdout.write(`\n${check.passed ? 'PASS' : 'FAIL'}  gradient check on a trained network\n`);
  process.stdout.write(`      ${formatGradCheckResult(check)}\n`);

  process.stdout.write(`\n${failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`}\n`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main();
