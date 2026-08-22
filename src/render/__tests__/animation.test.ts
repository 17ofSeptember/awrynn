import { describe, expect, it } from 'vitest';
import {
  buildChoreography,
  Choreography,
  MAX_FRAME_DELTA_MS,
  clamp01,
  easeInOutCubic,
  easeOutCubic,
  PULSE_SPEED,
  SPEED_MAX,
  SPEED_MIN,
  stagger,
} from '../animation';

const OPTIONS = { layerCount: 2, longestEdgePerLayer: [320, 320] };

describe('choreography structure (§6.3)', () => {
  it('walks forward, computes the loss, then walks back right to left', () => {
    const beats = buildChoreography(OPTIONS);
    const kinds = beats.map((b) => `${b.kind}${b.layer >= 0 ? b.layer : ''}`);
    expect(kinds).toEqual([
      'input',
      'forward-pulse0',
      'forward-activate0',
      'forward-pulse1',
      'forward-activate1',
      'loss',
      'backward-delta1',
      'backward-grad1',
      'backward-update1',
      'backward-delta0',
      'backward-grad0',
      'backward-update0',
      'complete',
    ]);
  });

  it('keeps pulse speed constant, so a longer edge takes longer', () => {
    // §6.3 A.2: "pulse speed constant". A fixed beat duration would make pulses
    // on a long edge outrun pulses on a short one.
    const short = buildChoreography({ layerCount: 1, longestEdgePerLayer: [320] });
    const long = buildChoreography({ layerCount: 1, longestEdgePerLayer: [1280] });
    const pulseOf = (b: ReturnType<typeof buildChoreography>): number =>
      b.find((x) => x.kind === 'forward-pulse')!.durationMs;
    expect(pulseOf(long)).toBeCloseTo(pulseOf(short) * 4, 5);
    expect(pulseOf(short)).toBeCloseTo((320 / PULSE_SPEED) * 1000, 5);
  });

  it('labels every beat for the transport readout', () => {
    for (const beat of buildChoreography(OPTIONS)) {
      expect(beat.label.length).toBeGreaterThan(0);
    }
  });

  it('rejects a network with no layers', () => {
    expect(() => buildChoreography({ layerCount: 0, longestEdgePerLayer: [] })).toThrowError(
      /must be positive/,
    );
  });
});

describe('reduced motion (§6.3 — the pedagogy must survive with motion off)', () => {
  it('keeps every stage, on a uniform timer', () => {
    const normal = buildChoreography(OPTIONS);
    const reduced = buildChoreography({ ...OPTIONS, reducedMotion: true });
    // Same stages: nothing about the lesson is carried by the tweening.
    expect(reduced.map((b) => b.kind)).toEqual(normal.map((b) => b.kind));
    const durations = new Set(reduced.filter((b) => b.kind !== 'complete').map((b) => b.durationMs));
    expect(durations.size).toBe(1);
  });
});

describe('playback', () => {
  const make = (): Choreography => new Choreography(buildChoreography(OPTIONS));

  it('starts paused at the first beat', () => {
    const c = make();
    expect(c.isPaused).toBe(true);
    expect(c.state.beatIndex).toBe(0);
    expect(c.current.kind).toBe('input');
  });

  it('does not advance while paused', () => {
    const c = make();
    c.advance(1000, 1);
    expect(c.state.beatIndex).toBe(0);
  });

  it('advances through beats when playing', () => {
    const c = make();
    c.play();
    // Driven in realistic ~16ms frames rather than one giant delta, because
    // advance() deliberately clamps a single step to MAX_FRAME_DELTA_MS.
    const frames = Math.ceil(c.current.durationMs / 16) + 2;
    for (let i = 0; i < frames; i++) c.advance(16, 1);
    expect(c.state.beatIndex).toBe(1);
  });

  it('scales with the speed control', () => {
    const fast = make();
    const slow = make();
    fast.play();
    slow.play();
    fast.advance(100, 4);
    slow.advance(100, 1);
    expect(fast.state.progress).toBeGreaterThan(slow.state.progress);
  });

  it('clamps speed to the documented range', () => {
    const c = make();
    c.play();
    // 1e9× must not skip the whole choreography in one frame.
    c.advance(16, 1e9);
    const clamped = make();
    clamped.play();
    clamped.advance(16, SPEED_MAX);
    expect(c.state.beatIndex).toBe(clamped.state.beatIndex);

    const slow = make();
    slow.play();
    slow.advance(16, 0);
    const atMin = make();
    atMin.play();
    atMin.advance(16, SPEED_MIN);
    expect(slow.state.progress).toBeCloseTo(atMin.state.progress, 10);
  });

  it('survives a backgrounded tab without skipping the whole run', () => {
    // A multi-second delta after the tab wakes must not fast-forward past the
    // lesson the learner was watching.
    const c = make();
    c.play();
    c.advance(60_000, 1);
    expect(c.state.complete).toBe(false);
    // At most MAX_FRAME_DELTA_MS of choreography time was consumed.
    expect(c.state.beatIndex).toBe(0);
    expect(c.state.progress).toBeCloseTo(MAX_FRAME_DELTA_MS / c.current.durationMs, 6);
  });

  it('pauses itself on completion', () => {
    const c = make();
    c.play();
    for (let i = 0; i < 400; i++) c.advance(250, SPEED_MAX);
    expect(c.state.complete).toBe(true);
    expect(c.isPaused).toBe(true);
  });

  it('replays from the start when played after completing', () => {
    const c = make();
    c.play();
    for (let i = 0; i < 400; i++) c.advance(250, SPEED_MAX);
    expect(c.state.complete).toBe(true);
    c.play();
    expect(c.state.beatIndex).toBe(0);
    expect(c.isPaused).toBe(false);
  });
});

describe('single-stepping (§6.3 — space bar advances one beat)', () => {
  const make = (): Choreography => new Choreography(buildChoreography(OPTIONS));

  it('lands on the END of a beat, so the completed stage is visible', () => {
    const c = make();
    c.stepBeat();
    expect(c.state.beatIndex).toBe(1);
    expect(c.state.progress).toBe(1);
  });

  it('finishes the current beat first when stepping mid-beat', () => {
    const c = make();
    c.play();
    c.advance(c.current.durationMs / 2, 1);
    const before = c.state.beatIndex;
    c.stepBeat();
    // The first press completes what you were watching rather than skipping it.
    expect(c.state.beatIndex).toBe(before);
    expect(c.state.progress).toBe(1);
    c.stepBeat();
    expect(c.state.beatIndex).toBe(before + 1);
  });

  it('pauses playback', () => {
    const c = make();
    c.play();
    c.stepBeat();
    expect(c.isPaused).toBe(true);
  });

  it('steps backward and stops at the start', () => {
    const c = make();
    c.stepBeat();
    c.stepBeat();
    c.stepBack();
    expect(c.state.beatIndex).toBe(1);
    c.stepBack();
    c.stepBack();
    expect(c.state.beatIndex).toBe(0);
  });

  it('stops at the end rather than running off the array', () => {
    const c = make();
    for (let i = 0; i < 200; i++) c.stepBeat();
    expect(c.state.beatIndex).toBe(c.beatCount - 1);
    expect(c.current.kind).toBe('complete');
  });

  it('seeks to an arbitrary beat and clamps', () => {
    const c = make();
    c.seek(5);
    expect(c.state.beatIndex).toBe(5);
    c.seek(-10);
    expect(c.state.beatIndex).toBe(0);
    c.seek(9999);
    expect(c.state.beatIndex).toBe(c.beatCount - 1);
  });

  it('reports which stages are already complete', () => {
    const c = make();
    c.seek(4);
    expect(c.hasPassed(2)).toBe(true);
    expect(c.hasPassed(4)).toBe(false);
  });

  it('finds a beat by kind and layer', () => {
    const c = make();
    expect(c.findBeat('forward-activate', 1)).toBe(4);
    expect(c.findBeat('backward-update', 0)).toBe(11);
    expect(c.findBeat('forward-pulse', 99)).toBe(-1);
  });
});

describe('easing and staggering', () => {
  it('eases from 0 to 1', () => {
    expect(easeOutCubic(0)).toBe(0);
    expect(easeOutCubic(1)).toBe(1);
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(1)).toBe(1);
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5, 10);
  });

  it('clamps out-of-range input', () => {
    expect(easeOutCubic(-1)).toBe(0);
    expect(easeOutCubic(2)).toBe(1);
    expect(clamp01(-0.5)).toBe(0);
    expect(clamp01(1.5)).toBe(1);
  });

  it('starts terms in order, which is what makes a card assemble term by term', () => {
    const count = 4;
    // Early in the beat the first term is ahead of the last.
    const early = [0, 1, 2, 3].map((i) => stagger(0.3, i, count));
    for (let i = 1; i < count; i++) expect(early[i]!).toBeLessThanOrEqual(early[i - 1]!);
    // By the end of the beat every term has landed, so nothing arrives after
    // the card has resolved its z.
    for (let i = 0; i < count; i++) expect(stagger(1, i, count)).toBe(1);
  });

  it('treats a single term as ungrouped', () => {
    expect(stagger(0.4, 0, 1)).toBe(0.4);
  });
});
