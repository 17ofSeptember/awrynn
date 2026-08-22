import { describe, expect, it } from 'vitest';
import { describeSchedule, learningRateAt } from '../schedules';
import type { ScheduleConfig } from '../schedules';

const BASE = 0.1;
const at = (config: ScheduleConfig, epoch: number): number => learningRateAt(config, BASE, epoch);

describe('constant', () => {
  it('never changes', () => {
    const config: ScheduleConfig = { schedule: { kind: 'constant' } };
    for (const epoch of [0, 1, 50, 1000]) expect(at(config, epoch)).toBe(BASE);
  });
});

describe('step(drop, every)', () => {
  const config: ScheduleConfig = { schedule: { kind: 'step', drop: 0.5, every: 10 } };

  it('halves every 10 epochs, holding flat between drops', () => {
    expect(at(config, 0)).toBeCloseTo(0.1, 12);
    expect(at(config, 9)).toBeCloseTo(0.1, 12);
    expect(at(config, 10)).toBeCloseTo(0.05, 12);
    expect(at(config, 19)).toBeCloseTo(0.05, 12);
    expect(at(config, 20)).toBeCloseTo(0.025, 12);
    expect(at(config, 30)).toBeCloseTo(0.0125, 12);
  });

  it('treats a zero period as constant rather than dividing by zero', () => {
    expect(at({ schedule: { kind: 'step', drop: 0.5, every: 0 } }, 5)).toBe(BASE);
  });
});

describe('exponential(γ)', () => {
  it('decays as base·γ^epoch', () => {
    const config: ScheduleConfig = { schedule: { kind: 'exponential', gamma: 0.9 } };
    expect(at(config, 0)).toBeCloseTo(0.1, 12);
    expect(at(config, 1)).toBeCloseTo(0.09, 12);
    expect(at(config, 2)).toBeCloseTo(0.081, 12);
    expect(at(config, 10)).toBeCloseTo(0.1 * Math.pow(0.9, 10), 12);
  });
});

describe('cosine(T)', () => {
  const config: ScheduleConfig = { schedule: { kind: 'cosine', period: 100 } };

  it('starts at base, halves at the midpoint and reaches the floor at T', () => {
    expect(at(config, 0)).toBeCloseTo(0.1, 12);
    expect(at(config, 50)).toBeCloseTo(0.05, 12);
    expect(at(config, 100)).toBeCloseTo(0, 12);
  });

  it('holds at the floor past T rather than climbing back up', () => {
    // A restart should be an explicit choice, not an accident of the formula.
    expect(at(config, 150)).toBeCloseTo(0, 12);
    expect(at(config, 1000)).toBeCloseTo(0, 12);
  });

  it('anneals to a configured minimum', () => {
    const withFloor: ScheduleConfig = {
      schedule: { kind: 'cosine', period: 10, min: 0.01 },
    };
    expect(at(withFloor, 0)).toBeCloseTo(0.1, 12);
    expect(at(withFloor, 10)).toBeCloseTo(0.01, 12);
    expect(at(withFloor, 5)).toBeCloseTo(0.055, 12);
  });

  it('decreases monotonically over the period', () => {
    let previous = Infinity;
    for (let epoch = 0; epoch <= 100; epoch++) {
      const rate = at(config, epoch);
      expect(rate).toBeLessThanOrEqual(previous + 1e-15);
      previous = rate;
    }
  });
});

describe('warmup composed over a schedule (§4.9)', () => {
  it('ramps linearly and never starts at exactly zero', () => {
    const config: ScheduleConfig = { schedule: { kind: 'constant' }, warmup: 5 };
    // A genuinely zero first step is indistinguishable from a hung trainer.
    expect(at(config, 0)).toBeCloseTo(0.1 * (1 / 5), 12);
    expect(at(config, 1)).toBeCloseTo(0.1 * (2 / 5), 12);
    expect(at(config, 4)).toBeCloseTo(0.1, 12);
    expect(at(config, 5)).toBeCloseTo(0.1, 12);
    expect(at(config, 0)).toBeGreaterThan(0);
  });

  it('multiplies the underlying schedule rather than replacing it', () => {
    // Ramping INTO a cosine, not stepping onto it.
    const config: ScheduleConfig = {
      schedule: { kind: 'cosine', period: 100 },
      warmup: 4,
    };
    const cosineOnly: ScheduleConfig = { schedule: { kind: 'cosine', period: 100 } };
    expect(at(config, 2)).toBeCloseTo(at(cosineOnly, 2) * (3 / 4), 12);
    // Past warmup the two agree exactly.
    expect(at(config, 40)).toBeCloseTo(at(cosineOnly, 40), 12);
  });

  it('composes with exponential decay', () => {
    const config: ScheduleConfig = {
      schedule: { kind: 'exponential', gamma: 0.9 },
      warmup: 2,
    };
    expect(at(config, 0)).toBeCloseTo(0.1 * 1 * 0.5, 12);
    expect(at(config, 1)).toBeCloseTo(0.1 * 0.9 * 1, 12);
    expect(at(config, 2)).toBeCloseTo(0.1 * 0.81, 12);
  });
});

describe('schedules are pure functions of the epoch', () => {
  it('repeated and out-of-order queries agree — the history scrubber depends on it', () => {
    const config: ScheduleConfig = {
      schedule: { kind: 'cosine', period: 37, min: 0.001 },
      warmup: 3,
    };
    const forward = [0, 1, 2, 3, 10, 20, 36].map((e) => at(config, e));
    const backward = [36, 20, 10, 3, 2, 1, 0].map((e) => at(config, e)).reverse();
    expect(backward).toEqual(forward);
  });

  it('rejects a negative epoch', () => {
    expect(() => at({ schedule: { kind: 'constant' } }, -1)).toThrowError(/non-negative/);
  });
});

describe('describeSchedule', () => {
  it('names the schedule and its warmup', () => {
    expect(describeSchedule({ schedule: { kind: 'constant' } })).toBe('constant');
    expect(describeSchedule({ schedule: { kind: 'step', drop: 0.5, every: 10 } })).toBe(
      'step (×0.5 every 10)',
    );
    expect(describeSchedule({ schedule: { kind: 'constant' }, warmup: 5 })).toBe(
      'warmup 5 → constant',
    );
    expect(describeSchedule({ schedule: { kind: 'cosine', period: 50, min: 0.01 } })).toContain(
      'min 0.01',
    );
  });
});
