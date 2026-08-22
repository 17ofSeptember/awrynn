/*
 * The message protocol between the main thread and the training worker.
 *
 * Deliberately plain structured-cloneable objects and raw postMessage rather
 * than comlink (§2 allows either). The protocol is small, and keeping it
 * explicit means the worker boundary is visible in the type system: anything
 * that cannot cross it will not typecheck.
 *
 * Parameters travel as Float64Array, which structured clone handles natively.
 * They are COPIED rather than transferred: transferring would detach the
 * worker's own buffer, and the worker needs to keep training from it.
 */

import type { Dataset, DatasetOptions } from '../engine/datasets/index';
import type { EpochMetrics, StopReason, TrainerConfig } from '../engine/trainer';

export interface StartMessage {
  readonly type: 'start';
  readonly config: TrainerConfig;
  readonly dataset: DatasetOptions;
  readonly maxEpochs: number;
  /**
   * Epochs to run between progress messages. Higher values mean less
   * postMessage traffic and a faster wall-clock run; the UI picks it from the
   * current animation mode (§6.3).
   */
  readonly reportEvery: number;
  /** Snapshot parameters every K epochs for the history scrubber (§6.6). */
  readonly snapshotEvery: number;
}

export interface ControlMessage {
  readonly type: 'pause' | 'resume' | 'stop' | 'step-epoch';
}

export type WorkerRequest = StartMessage | ControlMessage;

export interface ProgressMessage {
  readonly type: 'progress';
  readonly metrics: readonly EpochMetrics[];
  readonly parameters: Float64Array;
  /**
   * The running statistics that go with those parameters, empty unless some
   * layer normalizes across the batch. Sent together because they are only
   * meaningful together: the same weights with different statistics is a
   * different network.
   */
  readonly buffers: Float64Array;
  readonly epoch: number;
  readonly status: StopReason;
}

export interface SnapshotMessage {
  readonly type: 'snapshot';
  readonly epoch: number;
  readonly parameters: Float64Array;
  /**
   * The running statistics that go with those parameters, empty unless some
   * layer normalizes across the batch. Sent together because they are only
   * meaningful together: the same weights with different statistics is a
   * different network.
   */
  readonly buffers: Float64Array;
}

export interface DoneMessage {
  readonly type: 'done';
  readonly status: StopReason;
  readonly epoch: number;
  readonly parameters: Float64Array;
  /**
   * The running statistics that go with those parameters, empty unless some
   * layer normalizes across the batch. Sent together because they are only
   * meaningful together: the same weights with different statistics is a
   * different network.
   */
  readonly buffers: Float64Array;
}

export interface ErrorMessage {
  readonly type: 'error';
  /** Already human-readable — §9 requires errors that give direction. */
  readonly message: string;
}

export interface ReadyMessage {
  readonly type: 'ready';
  readonly datasetSummary: {
    readonly name: Dataset['name'];
    readonly samples: number;
    readonly features: number;
    readonly classes: number;
  };
}

export type WorkerResponse =
  | ReadyMessage
  | ProgressMessage
  | SnapshotMessage
  | DoneMessage
  | ErrorMessage;
