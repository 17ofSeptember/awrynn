import { describe, expect, it } from 'vitest';
import { HistoryBuffer, parameterDelta } from '../history';

import type { Snapshot } from '../history';

function snapshot(epoch: number, value = epoch): Snapshot {
  return {
    epoch,
    parameters: Float64Array.from([value, value * 2]),
    buffers: Float64Array.from([value * 3]),
    trainLoss: 1 / (epoch + 1),
    validationLoss: null,
    validationAccuracy: null,
  };
}

describe('history ring buffer (§6.6)', () => {
  it('stores and reads back in order', () => {
    const history = new HistoryBuffer(10);
    for (let i = 0; i < 4; i++) history.push(snapshot(i));
    expect(history.length).toBe(4);
    expect(history.at(0)?.epoch).toBe(0);
    expect(history.at(3)?.epoch).toBe(3);
    expect(history.newest?.epoch).toBe(3);
  });

  it('drops the oldest once full, keeping the most recent window', () => {
    const history = new HistoryBuffer(3);
    for (let i = 0; i < 7; i++) history.push(snapshot(i));
    expect(history.length).toBe(3);
    expect(history.full).toBe(true);
    expect(history.toArray().map((s) => s.epoch)).toEqual([4, 5, 6]);
  });

  it('COPIES parameters, so a caller reusing its buffer cannot corrupt history', () => {
    // The worker's parameter array is reused between messages; storing the
    // reference would make every snapshot show the latest weights.
    const history = new HistoryBuffer(4);
    const shared = Float64Array.from([1, 2]);
    history.push({ ...snapshot(0), parameters: shared });
    shared[0] = 99;
    expect(history.at(0)?.parameters[0]).toBe(1);
  });

  it('returns undefined outside its range', () => {
    const history = new HistoryBuffer(4);
    history.push(snapshot(0));
    expect(history.at(-1)).toBeUndefined();
    expect(history.at(1)).toBeUndefined();
  });

  it('clears completely', () => {
    const history = new HistoryBuffer(4);
    for (let i = 0; i < 3; i++) history.push(snapshot(i));
    history.clear();
    expect(history.length).toBe(0);
    expect(history.newest).toBeUndefined();
    expect(history.toArray()).toEqual([]);
  });

  it('rejects a non-positive capacity', () => {
    expect(() => new HistoryBuffer(0)).toThrowError(/capacity must be positive/);
  });

  it('survives many more pushes than its capacity', () => {
    // A few hundred epochs per second would grow an unbounded array into tens
    // of megabytes within a minute.
    const history = new HistoryBuffer(50);
    for (let i = 0; i < 5000; i++) history.push(snapshot(i));
    expect(history.length).toBe(50);
    expect(history.newest?.epoch).toBe(4999);
    expect(history.at(0)?.epoch).toBe(4950);
  });
});

describe('parameter delta (the A/B diff)', () => {
  it('subtracts elementwise', () => {
    const delta = parameterDelta(Float64Array.from([3, 1]), Float64Array.from([1, 4]));
    expect(Array.from(delta)).toEqual([2, -3]);
  });

  it('is zero against itself', () => {
    const values = Float64Array.from([0.5, -0.25]);
    expect(Array.from(parameterDelta(values, values))).toEqual([0, 0]);
  });

  it('refuses two differently shaped networks', () => {
    expect(() => parameterDelta(new Float64Array(3), new Float64Array(4))).toThrowError(
      /not the same shape/,
    );
  });
});
