/*
 * Training history (§6.6).
 *
 * "Snapshot all parameters + metrics every K steps into a capped ring buffer
 * (~500 snapshots; the parameter vectors are tiny). A timeline scrubber replays
 * training: drag it and the edges, boundary, and thumbnails animate through
 * their real history."
 *
 * Snapshots are taken on the main thread as each progress message arrives,
 * rather than asking the worker for extra ones. The worker already sends the
 * parameter vector with every progress message, so the snapshot is a copy of a
 * buffer that has already crossed the boundary; requesting separate snapshots
 * would double the traffic to store the same numbers.
 *
 * A ring rather than a growing array: at a few hundred epochs per second an
 * unbounded history would be tens of megabytes within a minute, and nobody
 * scrubs through ten thousand frames.
 */

export interface Snapshot {
  readonly epoch: number;
  /** A COPY. The source buffer is reused by the next message. */
  readonly parameters: Float64Array;
  /**
   * The running statistics from the same moment, empty without batch norm.
   *
   * Scrubbing back writes a snapshot into the live network, and the canvas and
   * the decision boundary then read it in eval mode. With only the parameters
   * restored, every frame of the scrub would be drawn with today's statistics
   * over an older network, which is a picture of nothing that ever existed.
   */
  readonly buffers: Float64Array;
  readonly trainLoss: number;
  readonly validationLoss: number | null;
  readonly validationAccuracy: number | null;
}

export const HISTORY_CAPACITY = 500;

export class HistoryBuffer {
  private readonly entries: (Snapshot | undefined)[];
  private start = 0;
  private count = 0;

  constructor(private readonly capacity: number = HISTORY_CAPACITY) {
    if (capacity <= 0) throw new Error('HistoryBuffer: capacity must be positive.');
    this.entries = new Array<Snapshot | undefined>(capacity);
  }

  get length(): number {
    return this.count;
  }

  get full(): boolean {
    return this.count === this.capacity;
  }

  /**
   * Append a snapshot, dropping the oldest once full.
   *
   * The parameters are copied here rather than by the caller, so a caller
   * handing over a buffer it intends to reuse cannot silently corrupt history.
   */
  push(snapshot: Snapshot): void {
    // Both arrays are copied: they arrive from a worker message whose buffers
    // are reused by the next one, so storing the references would leave the
    // whole history pointing at the newest epoch.
    const stored: Snapshot = {
      ...snapshot,
      parameters: Float64Array.from(snapshot.parameters),
      buffers: Float64Array.from(snapshot.buffers),
    };
    const index = (this.start + this.count) % this.capacity;
    this.entries[index] = stored;
    if (this.count === this.capacity) this.start = (this.start + 1) % this.capacity;
    else this.count++;
  }

  /** Oldest is 0, newest is length − 1. */
  at(index: number): Snapshot | undefined {
    if (index < 0 || index >= this.count) return undefined;
    return this.entries[(this.start + index) % this.capacity];
  }

  get newest(): Snapshot | undefined {
    return this.at(this.count - 1);
  }

  clear(): void {
    this.entries.fill(undefined);
    this.start = 0;
    this.count = 0;
  }

  /** Every snapshot, oldest first. For charting. */
  toArray(): Snapshot[] {
    const out: Snapshot[] = [];
    for (let i = 0; i < this.count; i++) {
      const entry = this.at(i);
      if (entry !== undefined) out.push(entry);
    }
    return out;
  }
}

/**
 * Elementwise difference between two parameter vectors.
 *
 * Used by the A/B diff (§6.6): with a snapshot pinned, edges colour by Δw
 * instead of w, so the picture shows what training CHANGED rather than what the
 * weights currently are.
 */
export function parameterDelta(current: Float64Array, base: Float64Array): Float64Array {
  if (current.length !== base.length) {
    throw new Error(
      `parameterDelta: length mismatch (${current.length} vs ${base.length}). The networks are not the same shape.`,
    );
  }
  const out = new Float64Array(current.length);
  for (let i = 0; i < current.length; i++) out[i] = (current[i] as number) - (base[i] as number);
  return out;
}
