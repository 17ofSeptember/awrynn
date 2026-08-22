/*
 * Choreography timeline.
 *
 * Spec §6.3. The dissection is a sequence of BEATS, not a continuous animation
 * that happens to pause. That distinction is what lets §6.3's two hard
 * requirements both work from one structure:
 *
 *   - true single-stepping ("space bar advances one choreography beat")
 *   - prefers-reduced-motion, where continuous pulses are replaced by discrete
 *     stage highlights that advance on a timer
 *
 * With motion off, the beats are exactly the stages; nothing about the pedagogy
 * is carried by the tweening.
 */

export type BeatKind =
  | 'input'
  | 'forward-pulse'
  | 'forward-activate'
  | 'loss'
  | 'backward-delta'
  | 'backward-grad'
  | 'backward-update'
  | 'complete';

export interface Beat {
  readonly kind: BeatKind;
  /** DenseLayer index, or −1 where the beat is not layer-specific. */
  readonly layer: number;
  readonly durationMs: number;
  /** Shown in the transport readout, so the learner knows what they are watching. */
  readonly label: string;
}

export interface TimelineState {
  readonly beatIndex: number;
  /** 0..1 within the current beat. Always 1 when paused on a step. */
  readonly progress: number;
  readonly complete: boolean;
}

export const SPEED_MIN = 0.1;
export const SPEED_MAX = 10;

/** World units per second a pulse travels. Constant, per §6.3 A.2. */
export const PULSE_SPEED = 320;

/**
 * Largest wall-clock delta honoured in one advance().
 *
 * A backgrounded tab hands back a multi-second delta on its first frame, which
 * without this would fast-forward the entire choreography the learner was
 * halfway through watching. Real frames are ~16ms, so this never binds in
 * normal playback.
 */
export const MAX_FRAME_DELTA_MS = 250;

export interface ChoreographyOptions {
  readonly layerCount: number;
  /** Longest edge into each layer, in world units, so pulse speed can be constant. */
  readonly longestEdgePerLayer: readonly number[];
  readonly reducedMotion?: boolean | undefined;
}

/** Fixed beat durations, in milliseconds at 1× speed. */
const DURATION = {
  input: 700,
  activate: 900,
  loss: 1200,
  delta: 700,
  grad: 900,
  update: 700,
  /** With motion off every beat is a discrete stage on a uniform timer. */
  reducedStage: 900,
} as const;

export function buildChoreography(options: ChoreographyOptions): Beat[] {
  const { layerCount, longestEdgePerLayer } = options;
  if (layerCount <= 0) {
    throw new Error(`buildChoreography: layerCount must be positive, got ${layerCount}.`);
  }
  const reduced = options.reducedMotion ?? false;
  const beats: Beat[] = [];

  const travelMs = (layer: number): number => {
    if (reduced) return DURATION.reducedStage;
    const length = longestEdgePerLayer[layer] ?? 200;
    // Constant speed means a longer edge takes longer; the beat lasts until the
    // last pulse lands, otherwise a term would arrive after its card resolved.
    return Math.max(320, (length / PULSE_SPEED) * 1000);
  };
  const fixed = (ms: number): number => (reduced ? DURATION.reducedStage : ms);

  beats.push({ kind: 'input', layer: -1, durationMs: fixed(DURATION.input), label: 'Inputs' });

  for (let l = 0; l < layerCount; l++) {
    beats.push({
      kind: 'forward-pulse',
      layer: l,
      durationMs: travelMs(l),
      label: `Layer ${l + 1}: weighted sum`,
    });
    beats.push({
      kind: 'forward-activate',
      layer: l,
      durationMs: fixed(DURATION.activate),
      label: `Layer ${l + 1}: activation`,
    });
  }

  beats.push({ kind: 'loss', layer: -1, durationMs: fixed(DURATION.loss), label: 'Loss' });

  // Right to left, per §6.3.
  for (let l = layerCount - 1; l >= 0; l--) {
    beats.push({
      kind: 'backward-delta',
      layer: l,
      durationMs: fixed(DURATION.delta),
      label: `Layer ${l + 1}: δ`,
    });
    beats.push({
      kind: 'backward-grad',
      layer: l,
      durationMs: fixed(DURATION.grad),
      label: `Layer ${l + 1}: ∂L/∂w`,
    });
    beats.push({
      kind: 'backward-update',
      layer: l,
      durationMs: fixed(DURATION.update),
      label: `Layer ${l + 1}: update`,
    });
  }

  beats.push({ kind: 'complete', layer: -1, durationMs: 0, label: 'Complete' });
  return beats;
}

export class Choreography {
  private beats: Beat[];
  private index = 0;
  private elapsed = 0;
  private paused = true;

  constructor(beats: Beat[]) {
    if (beats.length === 0) {
      throw new Error('Choreography: needs at least one beat.');
    }
    this.beats = beats;
  }

  replace(beats: Beat[]): void {
    if (beats.length === 0) {
      throw new Error('Choreography: needs at least one beat.');
    }
    this.beats = beats;
    this.reset();
  }

  get isPaused(): boolean {
    return this.paused;
  }

  play(): void {
    if (this.state.complete) this.reset();
    this.paused = false;
  }

  pause(): void {
    this.paused = true;
  }

  reset(): void {
    this.index = 0;
    this.elapsed = 0;
  }

  get beatCount(): number {
    return this.beats.length;
  }

  get current(): Beat {
    return this.beats[Math.min(this.index, this.beats.length - 1)] as Beat;
  }

  get state(): TimelineState {
    const beat = this.current;
    const complete = beat.kind === 'complete';
    return {
      beatIndex: this.index,
      progress: complete || beat.durationMs === 0 ? 1 : Math.min(1, this.elapsed / beat.durationMs),
      complete,
    };
  }

  /** Advance by wall-clock time. `speed` is the 0.1×–10× transport control. */
  advance(deltaMs: number, speed: number): void {
    if (this.paused) return;
    const clamped = Math.max(SPEED_MIN, Math.min(SPEED_MAX, speed));
    let remaining = Math.min(deltaMs, MAX_FRAME_DELTA_MS) * clamped;

    while (remaining > 0) {
      const beat = this.current;
      if (beat.kind === 'complete') {
        this.paused = true;
        return;
      }
      const left = beat.durationMs - this.elapsed;
      if (remaining < left) {
        this.elapsed += remaining;
        return;
      }
      remaining -= left;
      this.index++;
      this.elapsed = 0;
    }
  }

  /**
   * Advance exactly one beat, landing on its END so the learner sees the
   * completed stage. This is the space-bar behaviour in §6.3, and it is why
   * beats exist as discrete units rather than as keyframes on a curve.
   */
  stepBeat(): void {
    this.paused = true;
    if (this.current.kind === 'complete') return;
    // Mid-beat, the first press finishes the beat rather than skipping past it.
    if (this.elapsed > 0 && this.elapsed < this.current.durationMs) {
      this.elapsed = this.current.durationMs;
      return;
    }
    this.index = Math.min(this.index + 1, this.beats.length - 1);
    this.elapsed = this.current.durationMs;
  }

  stepBack(): void {
    this.paused = true;
    this.index = Math.max(0, this.index - 1);
    this.elapsed = this.current.durationMs;
  }

  /** Jump to a beat index — the scrubber. */
  seek(beatIndex: number): void {
    this.index = Math.max(0, Math.min(beatIndex, this.beats.length - 1));
    this.elapsed = this.current.durationMs;
  }

  /** True once `beat` has finished, so completed stages stay on screen. */
  hasPassed(beatIndex: number): boolean {
    return this.index > beatIndex;
  }

  beatAt(index: number): Beat | undefined {
    return this.beats[index];
  }

  /** Index of the first beat matching a kind and layer, or −1. */
  findBeat(kind: BeatKind, layer: number): number {
    return this.beats.findIndex((b) => b.kind === kind && b.layer === layer);
  }
}

/* ------------------------------------------------------------------ *
 * Easing
 *
 * One family, used consistently: a fast start that settles. Instruments
 * acknowledge input immediately; they do not ease in.
 * ------------------------------------------------------------------ */

export function easeOutCubic(t: number): number {
  const c = clamp01(t);
  return 1 - Math.pow(1 - c, 3);
}

export function easeInOutCubic(t: number): number {
  const c = clamp01(t);
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
}

export function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * Staggered progress for item `i` of `count` within a beat.
 *
 * Terms land one after another rather than all at once — that staggering is
 * what makes a formula card assemble term by term (§6.3 A.3) instead of
 * appearing whole.
 */
export function stagger(progress: number, index: number, count: number, overlap = 0.55): number {
  if (count <= 1) return clamp01(progress);
  const span = 1 / (count - (count - 1) * overlap);
  const start = index * span * (1 - overlap);
  return clamp01((progress - start) / span);
}
