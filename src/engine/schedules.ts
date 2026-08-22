/*
 * Learning-rate schedules.
 *
 * Spec §4.9: `constant`, `step(drop, every)`, `exponential(γ)`, `cosine(T)`,
 * `warmup(n)`, composable with the above.
 *
 * A schedule is a pure function of the epoch: `rate(base, epoch) -> number`.
 * Purity matters more than it looks — the history scrubber (§6.6) replays
 * training by epoch index, so a schedule that accumulated internal state would
 * produce a different curve on replay than it did live.
 */

export type ScheduleKind = 'constant' | 'step' | 'exponential' | 'cosine';

export type Schedule =
  | { readonly kind: 'constant' }
  /** Multiply by `drop` every `every` epochs: η·drop^⌊epoch/every⌋ */
  | { readonly kind: 'step'; readonly drop: number; readonly every: number }
  /** η·γ^epoch */
  | { readonly kind: 'exponential'; readonly gamma: number }
  /** Cosine anneal from η to `min` over `period` epochs. */
  | { readonly kind: 'cosine'; readonly period: number; readonly min?: number | undefined };

export interface ScheduleConfig {
  readonly schedule: Schedule;
  /**
   * Linear warmup over the first `warmup` epochs, composed with the schedule
   * above (§4.9). Epoch 0 starts at base/warmup rather than 0, because a
   * genuinely zero first step is indistinguishable from a hung trainer on the
   * loss curve.
   */
  readonly warmup?: number | undefined;
}

export const SCHEDULE_KINDS: readonly ScheduleKind[] = [
  'constant',
  'step',
  'exponential',
  'cosine',
];

function baseRate(schedule: Schedule, base: number, epoch: number): number {
  switch (schedule.kind) {
    case 'constant':
      return base;
    case 'step': {
      if (schedule.every <= 0) return base;
      return base * Math.pow(schedule.drop, Math.floor(epoch / schedule.every));
    }
    case 'exponential':
      return base * Math.pow(schedule.gamma, epoch);
    case 'cosine': {
      if (schedule.period <= 0) return base;
      const min = schedule.min ?? 0;
      // Clamped at the period so the rate holds at `min` afterwards rather
      // than climbing back up — a restart should be an explicit choice.
      const progress = Math.min(epoch / schedule.period, 1);
      return min + (base - min) * 0.5 * (1 + Math.cos(Math.PI * progress));
    }
  }
}

/**
 * The learning rate for `epoch` (0-based), warmup composed over the schedule.
 *
 * Warmup multiplies rather than replaces, so `warmup(5)` over `cosine(50)`
 * ramps into the cosine curve instead of stepping onto it.
 */
export function learningRateAt(config: ScheduleConfig, base: number, epoch: number): number {
  if (epoch < 0) {
    throw new Error(`schedules.learningRateAt: epoch must be non-negative, got ${epoch}.`);
  }
  const rate = baseRate(config.schedule, base, epoch);
  const warmup = config.warmup ?? 0;
  if (warmup > 0 && epoch < warmup) {
    return rate * ((epoch + 1) / warmup);
  }
  return rate;
}

export function describeSchedule(config: ScheduleConfig): string {
  const warmup = config.warmup ?? 0;
  const prefix = warmup > 0 ? `warmup ${warmup} → ` : '';
  const s = config.schedule;
  switch (s.kind) {
    case 'constant':
      return `${prefix}constant`;
    case 'step':
      return `${prefix}step (×${s.drop} every ${s.every})`;
    case 'exponential':
      return `${prefix}exponential (γ = ${s.gamma})`;
    case 'cosine':
      return `${prefix}cosine (T = ${s.period}${s.min !== undefined ? `, min ${s.min}` : ''})`;
  }
}
