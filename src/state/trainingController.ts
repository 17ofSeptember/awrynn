/*
 * Owns the training worker and forwards its messages into the store.
 *
 * Kept out of the store itself because starting a worker is a side effect, and
 * a store that spawns threads is a store you cannot test. The store holds the
 * results; this holds the machinery.
 *
 * Spec §11 Phase 2 gate requires training off the main thread, and §10 caps
 * main-thread blocking at 8ms. A worker satisfies both. If one cannot be
 * constructed the controller falls back to running the same Trainer on the main
 * thread in small timed chunks: slower and less smooth, but the app still
 * teaches rather than showing a dead button.
 */

import type { DatasetOptions } from '../engine/datasets/index';
import { generateDataset } from '../engine/datasets/index';
import { Trainer } from '../engine/trainer';
import type { EpochMetrics, StopReason, TrainerConfig } from '../engine/trainer';
import type { NetworkState } from '../engine/network';
import type { WorkerRequest, WorkerResponse } from '../worker/protocol';

export interface TrainingCallbacks {
  onReady: (summary: { samples: number; features: number; classes: number }) => void;
  onProgress: (metrics: readonly EpochMetrics[], state: NetworkState, epoch: number) => void;
  onDone: (status: StopReason, epoch: number, state: NetworkState) => void;
  onError: (message: string) => void;
}

/** Epochs run per main-thread chunk in the fallback path. */
const FALLBACK_CHUNK = 4;

export class TrainingController {
  private worker: Worker | null = null;
  private callbacks: TrainingCallbacks;

  /* Main-thread fallback state. */
  private fallbackTrainer: Trainer | null = null;
  private fallbackTimer = 0;
  private fallbackMaxEpochs = 0;
  private fallbackReportEvery = 1;
  private usingFallback = false;

  constructor(callbacks: TrainingCallbacks) {
    this.callbacks = callbacks;
  }

  get isFallback(): boolean {
    return this.usingFallback;
  }

  start(config: TrainerConfig, dataset: DatasetOptions, maxEpochs: number, reportEvery: number): void {
    this.stop();
    try {
      this.worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), {
        type: 'module',
      });
      this.usingFallback = false;
      this.worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
        this.handle(event.data);
      };
      this.worker.onerror = (): void => {
        // A worker that fails after construction is not recoverable by retrying
        // it, so drop to the main thread rather than leaving training stuck.
        this.disposeWorker();
        this.startFallback(config, dataset, maxEpochs, reportEvery);
      };
      this.post({
        type: 'start',
        config,
        dataset,
        maxEpochs,
        reportEvery,
        snapshotEvery: 0,
      });
    } catch {
      this.startFallback(config, dataset, maxEpochs, reportEvery);
    }
  }

  pause(): void {
    if (this.usingFallback) {
      this.clearFallbackTimer();
      return;
    }
    this.post({ type: 'pause' });
  }

  resume(): void {
    if (this.usingFallback) {
      this.scheduleFallback();
      return;
    }
    this.post({ type: 'resume' });
  }

  stepEpoch(): void {
    if (this.usingFallback) {
      this.runFallbackChunk(1);
      return;
    }
    this.post({ type: 'step-epoch' });
  }

  stop(): void {
    if (this.worker !== null) this.post({ type: 'stop' });
    this.disposeWorker();
    this.clearFallbackTimer();
    this.fallbackTrainer = null;
    this.usingFallback = false;
  }

  dispose(): void {
    this.stop();
  }

  private post(message: WorkerRequest): void {
    this.worker?.postMessage(message);
  }

  private disposeWorker(): void {
    if (this.worker === null) return;
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.terminate();
    this.worker = null;
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'ready':
        this.callbacks.onReady({
          samples: message.datasetSummary.samples,
          features: message.datasetSummary.features,
          classes: message.datasetSummary.classes,
        });
        break;
      case 'progress':
        this.callbacks.onProgress(
          message.metrics,
          { parameters: message.parameters, buffers: message.buffers },
          message.epoch,
        );
        break;
      case 'done':
        this.callbacks.onDone(message.status, message.epoch, {
          parameters: message.parameters,
          buffers: message.buffers,
        });
        break;
      case 'error':
        this.callbacks.onError(message.message);
        break;
      case 'snapshot':
        break;
    }
  }

  /* ---------------- main-thread fallback ---------------- */

  private startFallback(
    config: TrainerConfig,
    dataset: DatasetOptions,
    maxEpochs: number,
    reportEvery: number,
  ): void {
    this.usingFallback = true;
    try {
      const data = generateDataset(dataset);
      this.fallbackTrainer = new Trainer(config, data);
      this.fallbackMaxEpochs = maxEpochs;
      this.fallbackReportEvery = Math.max(1, reportEvery);
      this.callbacks.onReady({
        samples: data.x.rows,
        features: data.featureCount,
        classes: data.classCount,
      });
      this.scheduleFallback();
    } catch (error) {
      this.callbacks.onError(error instanceof Error ? error.message : String(error));
    }
  }

  private scheduleFallback(): void {
    this.clearFallbackTimer();
    // A timer rather than a tight loop: each tick returns to the event loop so
    // the canvas keeps painting and the UI stays responsive.
    this.fallbackTimer = window.setInterval(() => {
      this.runFallbackChunk(Math.min(FALLBACK_CHUNK, this.fallbackReportEvery));
    }, 0);
  }

  private clearFallbackTimer(): void {
    if (this.fallbackTimer !== 0) {
      window.clearInterval(this.fallbackTimer);
      this.fallbackTimer = 0;
    }
  }

  private runFallbackChunk(count: number): void {
    const trainer = this.fallbackTrainer;
    if (trainer === null) return;
    try {
      const batch: EpochMetrics[] = [];
      for (let i = 0; i < count; i++) {
        if (trainer.epoch >= this.fallbackMaxEpochs || trainer.status !== 'running') break;
        batch.push(trainer.runEpoch());
      }
      if (batch.length > 0) {
        this.callbacks.onProgress(batch, trainer.network.captureState(), trainer.epoch);
      }
      if (trainer.epoch >= this.fallbackMaxEpochs || trainer.status !== 'running') {
        this.clearFallbackTimer();
        this.callbacks.onDone(trainer.status, trainer.epoch, trainer.network.captureState());
      }
    } catch (error) {
      this.clearFallbackTimer();
      this.callbacks.onError(error instanceof Error ? error.message : String(error));
    }
  }
}
