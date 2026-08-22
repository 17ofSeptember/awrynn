import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WorkerRequest, WorkerResponse } from '../protocol';
import type { TrainerConfig } from '../../engine/trainer';

/*
 * Spec §11 Phase 2 gate: "training runs off the main thread."
 *
 * A real Worker needs a browser, so this drives the worker MODULE directly
 * through a stand-in `self`. That verifies the part that can actually be wrong
 * — the message protocol, the chunked loop, and whether pause/stop are honoured
 * — rather than verifying that the platform can spawn a thread.
 */

interface WorkerGlobal {
  postMessage: (message: WorkerResponse) => void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
}

let received: WorkerResponse[] = [];
let workerSelf: WorkerGlobal;

function send(message: WorkerRequest): void {
  const handler = workerSelf.onmessage;
  if (handler === null) throw new Error('worker did not install an onmessage handler');
  handler({ data: message } as MessageEvent<WorkerRequest>);
}

/** Let the worker's setTimeout-chunked loop run until it settles. */
async function settle(maxTicks = 500): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (received.some((m) => m.type === 'done' || m.type === 'error')) return;
  }
}

const CONFIG: TrainerConfig = {
  network: {
    inputSize: 2,
    layers: [
      { units: 6, activation: 'tanh' },
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
};

beforeEach(async () => {
  received = [];
  workerSelf = {
    postMessage: (message) => received.push(message),
    onmessage: null,
  };
  (globalThis as { self?: WorkerGlobal }).self = workerSelf;
  vi.resetModules();
  await import('../trainer.worker');
});

afterEach(() => {
  delete (globalThis as { self?: WorkerGlobal }).self;
});

describe('training worker', () => {
  it('acknowledges a start with a dataset summary before training', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, noise: 0.1, seed: 1 },
      maxEpochs: 5,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    await settle();

    const ready = received.find((m) => m.type === 'ready');
    expect(ready).toBeDefined();
    expect(ready?.type === 'ready' && ready.datasetSummary).toEqual({
      name: 'xor',
      samples: 100,
      features: 2,
      classes: 2,
    });
    // The summary arrives first, so the UI can lay out before any metrics.
    expect(received[0]?.type).toBe('ready');
  });

  it('streams progress and finishes with done', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 6,
      reportEvery: 2,
      snapshotEvery: 0,
    });
    await settle();

    const progress = received.filter((m) => m.type === 'progress');
    expect(progress.length).toBeGreaterThan(0);

    const epochs = progress.flatMap((m) => (m.type === 'progress' ? m.metrics : []));
    expect(epochs.length).toBe(6);
    expect(epochs.map((e) => e.epoch)).toEqual([0, 1, 2, 3, 4, 5]);

    const done = received.find((m) => m.type === 'done');
    expect(done?.type === 'done' && done.status).toBe('running');
    expect(done?.type === 'done' && done.epoch).toBe(6);
  });

  it('reports in chunks of reportEvery, so the UI is not flooded', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 20,
      reportEvery: 10,
      snapshotEvery: 0,
    });
    await settle();

    const progress = received.filter((m) => m.type === 'progress');
    // 20 epochs at 10 per report: two chunks, not twenty messages.
    expect(progress.length).toBe(2);
    for (const message of progress) {
      if (message.type === 'progress') expect(message.metrics.length).toBe(10);
    }
  });

  it('carries parameters back as a Float64Array snapshot', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 3,
      reportEvery: 3,
      snapshotEvery: 0,
    });
    await settle();

    const done = received.find((m) => m.type === 'done');
    expect(done?.type === 'done' && done.parameters).toBeInstanceOf(Float64Array);
    // 2*6 + 6 weights/biases in layer 1, 6*1 + 1 in layer 2 = 25 parameters.
    expect(done?.type === 'done' && done.parameters.length).toBe(25);
  });

  it('emits snapshots on the requested cadence (§6.6)', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 10,
      reportEvery: 10,
      snapshotEvery: 3,
    });
    await settle();

    const snapshots = received.filter((m) => m.type === 'snapshot');
    // epochs 0, 3, 6, 9
    expect(snapshots.map((m) => (m.type === 'snapshot' ? m.epoch : -1))).toEqual([0, 3, 6, 9]);
  });

  it('honours pause, and resume continues where it left off', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 1000,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    // Let a couple of chunks run, then pause mid-flight.
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));
    send({ type: 'pause' });

    const atPause = received.filter((m) => m.type === 'progress').length;
    await new Promise((resolve) => setTimeout(resolve, 5));
    // Nothing further arrived while paused.
    expect(received.filter((m) => m.type === 'progress').length).toBe(atPause);
    expect(received.some((m) => m.type === 'done')).toBe(false);

    send({ type: 'resume' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(received.filter((m) => m.type === 'progress').length).toBeGreaterThan(atPause);

    send({ type: 'stop' });
  });

  it('step-epoch advances exactly one epoch while paused', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 1000,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    send({ type: 'pause' });
    await new Promise((resolve) => setTimeout(resolve, 2));

    const before = lastEpoch();
    send({ type: 'step-epoch' });
    expect(lastEpoch()).toBe(before + 1);
    send({ type: 'step-epoch' });
    expect(lastEpoch()).toBe(before + 2);

    send({ type: 'stop' });
  });

  it('stops and stays stopped', async () => {
    send({
      type: 'start',
      config: CONFIG,
      dataset: { name: 'xor', samples: 100, seed: 1 },
      maxEpochs: 1000,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    send({ type: 'stop' });
    const count = received.length;
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(received.length).toBe(count);
  });

  it('reports a divergence as a status rather than throwing (§7.4)', async () => {
    // An unbounded regression head, which blows up for ANY init at this LR.
    // A bounded sigmoid/BCE head would not: whether it diverges depends on the
    // init as much as the learning rate, so it makes a flaky test.
    send({
      type: 'start',
      config: {
        ...CONFIG,
        network: {
          inputSize: 1,
          layers: [
            { units: 8, activation: 'relu' },
            { units: 1, activation: 'linear' },
          ],
          loss: 'mse',
          seed: 7,
          init: { kind: 'he_normal' },
        },
        optimizer: { name: 'sgd' },
        learningRate: 50,
      },
      dataset: { name: 'sine', samples: 200, seed: 2 },
      maxEpochs: 100,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    await settle();

    const done = received.find((m) => m.type === 'done');
    expect(done?.type === 'done' && done.status).toBe('diverged');
    expect(received.some((m) => m.type === 'error')).toBe(false);
  });

  it('turns an invalid configuration into a readable error message (§9)', async () => {
    send({
      type: 'start',
      config: {
        ...CONFIG,
        network: {
          ...CONFIG.network,
          // softmax on a hidden layer — rejected by validateConfig (§4.4)
          layers: [
            { units: 4, activation: 'softmax' },
            { units: 1, activation: 'sigmoid' },
          ],
        },
      },
      dataset: { name: 'xor', samples: 50, seed: 1 },
      maxEpochs: 5,
      reportEvery: 1,
      snapshotEvery: 0,
    });
    await settle();

    const error = received.find((m) => m.type === 'error');
    expect(error).toBeDefined();
    expect(error?.type === 'error' && error.message).toMatch(/only valid on the output layer/);
  });
});

function lastEpoch(): number {
  for (let i = received.length - 1; i >= 0; i--) {
    const message = received[i];
    if (message?.type === 'progress') return message.epoch;
  }
  return -1;
}
