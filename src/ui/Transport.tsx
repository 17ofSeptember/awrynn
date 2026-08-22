import { useEffect, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { SPEED_MAX, SPEED_MIN } from '../render/animation';

/*
 * Transport controls (§6.3).
 *
 * "Transport controls, always available: step-sample, step-batch, step-epoch,
 * run/pause, speed, reset-to-init (same seed), reseed."
 *
 * Phase 4 wires the dissection transport — run/pause, single-step, speed,
 * restart, and stepping the sample. Batch and epoch stepping arrive with the
 * training loop in a later phase; their buttons are not drawn yet rather than
 * drawn dead, because a control that does nothing is worse than an absent one.
 */

export function Transport(): JSX.Element {
  const dissectionEnabled = useAppStore((s) => s.dissectionEnabled);
  const toggleDissection = useAppStore((s) => s.toggleDissection);
  const sampleIndex = useAppStore((s) => s.sampleIndex);
  const setSampleIndex = useAppStore((s) => s.setSampleIndex);
  const dataset = useAppStore((s) => s.dataset);
  const transport = useAppStore((s) => s.transport);
  const speed = useAppStore((s) => s.speed);
  const setSpeed = useAppStore((s) => s.setSpeed);
  const status = useAppStore((s) => s.transportStatus);

  // The space bar advances one beat, per §6.3. Bound at the window so it works
  // wherever the learner's focus happens to be, except inside a text field.
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) return;
      if (event.code === 'Space') {
        event.preventDefault();
        transport('step');
      } else if (event.code === 'ArrowLeft') {
        event.preventDefault();
        transport('back');
      } else if (event.code === 'KeyK') {
        event.preventDefault();
        transport('toggle');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [transport]);

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-t border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)] px-3 py-2 sm:px-4">
      <button
        className="control"
        onClick={toggleDissection}
        aria-pressed={dissectionEnabled}
        style={dissectionEnabled ? { borderColor: 'var(--color-weight-positive)' } : undefined}
      >
        {dissectionEnabled ? 'Dissection: on' : 'Dissection: off'}
      </button>

      <div className="flex items-center gap-1">
        <button
          className="control"
          onClick={() => transport('restart')}
          disabled={!dissectionEnabled}
          aria-label="Restart the dissection"
          title="Restart"
        >
          ⏮
        </button>
        <button
          className="control"
          onClick={() => transport('back')}
          disabled={!dissectionEnabled}
          aria-label="Previous stage"
          title="Previous stage (←)"
        >
          ◀
        </button>
        <button
          className="control w-16"
          onClick={() => transport('toggle')}
          disabled={!dissectionEnabled}
          aria-label={status.playing ? 'Pause' : 'Play'}
          title="Play / pause (K)"
        >
          {status.playing ? 'Pause' : 'Play'}
        </button>
        <button
          className="control"
          onClick={() => transport('step')}
          disabled={!dissectionEnabled}
          aria-label="Next stage"
          title="Next stage (Space)"
        >
          ▶
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="label">speed</span>
        <input
          type="range"
          min={Math.log10(SPEED_MIN)}
          max={Math.log10(SPEED_MAX)}
          step={0.01}
          value={Math.log10(speed)}
          onChange={(e) => setSpeed(Math.pow(10, Number(e.target.value)))}
          className="w-20 accent-[var(--color-weight-positive)] sm:w-28"
          aria-label="Playback speed"
          disabled={!dissectionEnabled}
        />
        {/* Logarithmic, so 0.1x and 10x are equidistant from 1x. */}
        <span className="num w-12 text-[11px] text-[var(--color-text-mid)]">
          {speed.toFixed(speed < 1 ? 2 : 1)}×
        </span>
      </div>

      <div className="flex items-center gap-2">
        <span className="label">sample</span>
        <button
          className="control px-1.5 py-0"
          onClick={() => setSampleIndex(sampleIndex - 1)}
          aria-label="Previous sample"
        >
          −
        </button>
        <span className="num w-14 text-center text-[11px] text-[var(--color-text-mid)]">
          {sampleIndex + 1}/{dataset.x.rows}
        </span>
        <button
          className="control px-1.5 py-0"
          onClick={() => setSampleIndex(sampleIndex + 1)}
          aria-label="Next sample"
        >
          +
        </button>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <span className="num hidden text-[11px] text-[var(--color-text-lo)] sm:inline">
          {dissectionEnabled ? `stage ${status.beatIndex + 1}/${status.beatCount}` : 'idle'}
        </span>
        <span className="truncate text-[11px] text-[var(--color-text-mid)]">{status.label}</span>
      </div>
    </div>
  );
}

/** Re-renders on a timer only while the dissection is playing. */
export function useTransportTick(): void {
  const [, setTick] = useState(0);
  const playing = useAppStore((s) => s.transportStatus.playing);
  useEffect(() => {
    if (!playing) return;
    // 6Hz: enough for the stage readout to feel live, far below a frame rate.
    // The canvas is NOT driven from here — it has its own RAF loop (§6.1).
    const id = window.setInterval(() => setTick((t) => t + 1), 160);
    return () => window.clearInterval(id);
  }, [playing]);
}
