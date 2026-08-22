/*
 * Training worker.
 *
 * Spec §11 Phase 2 gate: "training runs off the main thread". The main thread
 * must never block for more than 8ms (§10), which it cannot guarantee if it is
 * also running epochs — so the whole Trainer lives here and the main thread
 * only ever receives metrics and a parameter snapshot.
 *
 * The loop yields between chunks rather than running to completion, so pause,
 * stop and single-step messages are actually delivered instead of queueing
 * behind a synchronous run.
 */

import { generateDataset } from '../engine/datasets/index';
import { Trainer } from '../engine/trainer';
import type { EpochMetrics } from '../engine/trainer';
import type { WorkerRequest, WorkerResponse } from './protocol';

let trainer: Trainer | null = null;
let running = false;
let maxEpochs = 0;
let reportEvery = 1;
let snapshotEvery = 0;
let pending: EpochMetrics[] = [];

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function fail(error: unknown): void {
  post({
    type: 'error',
    message: error instanceof Error ? error.message : String(error),
  });
  running = false;
}

function flush(): void {
  if (trainer === null) return;
  post({
    type: 'progress',
    metrics: pending,
    parameters: trainer.network.captureParameters(),
    buffers: trainer.network.captureBuffers(),
    epoch: trainer.epoch,
    status: trainer.status,
  });
  pending = [];
}

/** One chunk of epochs, then yield so control messages can be delivered. */
function tick(): void {
  if (!running || trainer === null) return;
  try {
    for (let i = 0; i < reportEvery; i++) {
      if (trainer.epoch >= maxEpochs || trainer.status !== 'running') break;
      const metrics = trainer.runEpoch();
      pending.push(metrics);
      if (snapshotEvery > 0 && metrics.epoch % snapshotEvery === 0) {
        post({
          type: 'snapshot',
          epoch: metrics.epoch,
          parameters: trainer.network.captureParameters(),
          buffers: trainer.network.captureBuffers(),
        });
      }
    }
    flush();

    if (trainer.epoch >= maxEpochs || trainer.status !== 'running') {
      running = false;
      post({
        type: 'done',
        status: trainer.status,
        epoch: trainer.epoch,
        parameters: trainer.network.captureParameters(),
        buffers: trainer.network.captureBuffers(),
      });
      return;
    }
    // Yield to the event loop so pause/stop actually arrive.
    setTimeout(tick, 0);
  } catch (error) {
    fail(error);
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const message = event.data;
  try {
    switch (message.type) {
      case 'start': {
        const dataset = generateDataset(message.dataset);
        trainer = new Trainer(message.config, dataset);
        maxEpochs = message.maxEpochs;
        reportEvery = Math.max(1, message.reportEvery);
        snapshotEvery = message.snapshotEvery;
        pending = [];
        post({
          type: 'ready',
          datasetSummary: {
            name: dataset.name,
            samples: dataset.x.rows,
            features: dataset.featureCount,
            classes: dataset.classCount,
          },
        });
        running = true;
        tick();
        break;
      }
      case 'pause':
        running = false;
        flush();
        break;
      case 'resume':
        if (trainer !== null && !running && trainer.status === 'running') {
          running = true;
          tick();
        }
        break;
      case 'step-epoch':
        if (trainer !== null && !running && trainer.status === 'running') {
          pending.push(trainer.runEpoch());
          flush();
        }
        break;
      case 'stop':
        running = false;
        trainer = null;
        pending = [];
        break;
    }
  } catch (error) {
    fail(error);
  }
};
