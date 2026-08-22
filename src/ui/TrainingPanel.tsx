import { useEffect, useRef } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { OPTIMIZER_NAMES } from '../engine/optimizers';
import type { OptimizerName } from '../engine/optimizers';
import { drawChart } from '../render/draw/curves';
import { COLORS, formatLoss, formatPercent, formatNorm } from '../render/theme';

/*
 * Training controls and the loss curve.
 *
 * The chart is a canvas rather than a charting library (§2), redrawn when the
 * metrics array changes. Metrics arrive in batches from the worker, a few times
 * a second at most, so this re-renders at a human rate rather than a frame rate.
 */

export function TrainingPanel(): JSX.Element {
  const training = useAppStore((s) => s.training);
  const setTraining = useAppStore((s) => s.setTraining);
  const status = useAppStore((s) => s.trainingStatus);
  const error = useAppStore((s) => s.trainingError);
  const epoch = useAppStore((s) => s.epoch);
  const latest = useAppStore((s) => s.latest);
  const onMainThread = useAppStore((s) => s.trainingOnMainThread);
  /*
   * Derived from the metric history rather than tracked in the store, so it
   * cannot drift from the curve the chart is drawing. The array is capped at
   * MAX_METRIC_HISTORY, and this runs once per epoch report rather than once
   * per frame.
   */
  const bestAccuracy = useAppStore((s) => {
    let best: number | null = null;
    for (const m of s.metrics) {
      if (m.validationAccuracy === null) continue;
      if (best === null || m.validationAccuracy > best) best = m.validationAccuracy;
    }
    return best;
  });

  const start = useAppStore((s) => s.startTraining);
  const pause = useAppStore((s) => s.pauseTraining);
  const resume = useAppStore((s) => s.resumeTraining);
  const stepEpoch = useAppStore((s) => s.stepEpoch);
  const resetToInit = useAppStore((s) => s.resetToInit);

  const running = status === 'running';

  return (
    <div className="px-4 py-3">
      <div className="mb-3 flex items-baseline justify-between">
        <p className="panel-title">Training</p>
        <span className="num text-[11px] text-[var(--color-text-lo)]">
          epoch {epoch}/{training.maxEpochs}
        </span>
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        <button
          className="control flex-1"
          onClick={running ? pause : status === 'paused' ? resume : start}
        >
          {running ? 'Pause' : status === 'paused' ? 'Resume' : 'Train'}
        </button>
        <button className="control" onClick={stepEpoch} title="Run a single epoch">
          +1 epoch
        </button>
        <button className="control" onClick={resetToInit} title="Back to the initial weights for this seed">
          Reset
        </button>
      </div>

      <StatusLine status={status} error={error} onMainThread={onMainThread} />

      <div className="mt-3 flex flex-col gap-1.5">
        <Field label="optimizer">
          <select
            className="control w-full"
            value={training.optimizer.name}
            onChange={(e) =>
              setTraining({ optimizer: { name: e.target.value as OptimizerName } })
            }
            aria-label="Optimizer"
          >
            {OPTIMIZER_NAMES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="learn rate">
          <input
            type="number"
            className="control num w-full"
            value={training.learningRate}
            step={0.001}
            min={0.0001}
            onChange={(e) => setTraining({ learningRate: Number(e.target.value) || 0.01 })}
            aria-label="Learning rate"
          />
        </Field>
        <Field label="batch">
          <input
            type="number"
            className="control num w-full"
            value={training.batchSize}
            min={1}
            onChange={(e) => setTraining({ batchSize: Math.max(1, Number(e.target.value) || 1) })}
            aria-label="Batch size"
          />
        </Field>
        <Field label="epochs">
          <input
            type="number"
            className="control num w-full"
            value={training.maxEpochs}
            min={1}
            onChange={(e) => setTraining({ maxEpochs: Math.max(1, Number(e.target.value) || 1) })}
            aria-label="Maximum epochs"
          />
        </Field>
        <Field label="dropout">
          <input
            type="number"
            className="control num w-full"
            value={training.dropout}
            step={0.05}
            min={0}
            max={0.9}
            onChange={(e) =>
              setTraining({ dropout: Math.min(0.9, Math.max(0, Number(e.target.value) || 0)) })
            }
            aria-label="Dropout probability"
          />
        </Field>
      </div>

      <LossChart />

      {latest !== null && (
        <div className="mt-3 flex flex-col gap-0.5">
          <Metric label="train loss" value={formatLoss(latest.trainLoss)} />
          {latest.validationLoss !== null && (
            <Metric label="val loss" value={formatLoss(latest.validationLoss)} />
          )}
          {latest.validationAccuracy !== null && (
            <Metric
              label="val acc"
              value={formatPercent(latest.validationAccuracy)}
              emphasis
              /*
               * The best epoch, whenever it is meaningfully above the latest.
               *
               * "val acc" is the CURRENT epoch, and on an oscillating run that
               * is a poor summary of what the network managed: a lesson judged
               * on its best epoch can show a green tick beside a mediocre
               * number, and the reader is left to guess which one is lying.
               * Neither is. Showing both says so without a paragraph.
               */
              note={
                bestAccuracy !== null && bestAccuracy > latest.validationAccuracy + 0.005
                  ? `best ${formatPercent(bestAccuracy)}`
                  : undefined
              }
            />
          )}
          <Metric label="grad norm" value={formatNorm(latest.gradientNorm)} />
          {latest.deadUnits > 0 && (
            <Metric label="dead units" value={String(latest.deadUnits)} warn />
          )}
        </div>
      )}
    </div>
  );
}

function StatusLine({
  status,
  error,
  onMainThread,
}: {
  status: string;
  error: string | null;
  onMainThread: boolean;
}): JSX.Element | null {
  if (status === 'diverged') {
    return (
      <div className="border-l-2 border-[var(--color-status-bad)] pl-3">
        {/* §9: errors give direction, not just an apology. */}
        <p className="text-[11px] leading-relaxed text-[var(--color-status-bad)]">
          Training diverged: the loss became NaN. Lower the learning rate, or set a
          gradient clip.
        </p>
      </div>
    );
  }
  if (status === 'error' && error !== null) {
    return (
      <div className="border-l-2 border-[var(--color-status-bad)] pl-3">
        <p className="text-[11px] leading-relaxed text-[var(--color-status-bad)]">{error}</p>
      </div>
    );
  }
  if (onMainThread) {
    return (
      <p className="text-[11px] leading-relaxed text-[var(--color-text-lo)]">
        Running on the main thread: no Web Worker is available, so the interface may
        stutter while training.
      </p>
    );
  }
  return null;
}

function LossChart(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const metrics = useAppStore((s) => s.metrics);
  const logScale = useAppStore((s) => s.lossLogScale);
  const toggleLog = useAppStore((s) => s.toggleLossLogScale);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const dpr = Math.max(1, window.devicePixelRatio);
    const width = canvas.clientWidth;
    const height = 128;
    if (width === 0) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bgChassis;
    ctx.fillRect(0, 0, width, height);

    const train = metrics.map((m) => m.trainLoss);
    const validation = metrics
      .map((m) => m.validationLoss)
      .filter((v): v is number => v !== null);

    drawChart(
      ctx as unknown as Parameters<typeof drawChart>[0],
      0,
      0,
      width,
      height,
      [
        { values: train, label: 'train', dashed: false, dim: false },
        { values: validation, label: 'validation', dashed: true, dim: true },
      ],
      { logScale, nonNegative: true, yLabel: logScale ? 'loss (log)' : 'loss' },
    );
  }, [metrics, logScale]);

  return (
    <div className="mt-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="label">loss</span>
        <button
          // Padded to a 24px target rather than left as bare text (WCAG 2.5.8).
          className="label -my-1 px-2 hover:text-[var(--color-text-mid)]"
          // Explicit, because the label face is 9px and padding alone does not
          // reach a 24px box (WCAG 2.5.8).
          style={{ minHeight: 24 }}
          onClick={toggleLog}
          aria-pressed={logScale}
          aria-label={`Loss axis scale: ${logScale ? 'logarithmic' : 'linear'}. Toggle.`}
        >
          {logScale ? 'log' : 'linear'}
        </button>
      </div>
      <canvas
        ref={canvasRef}
        className="block w-full border border-[var(--color-line-hair)]"
        style={{ height: 128 }}
        aria-label="Training and validation loss over epochs"
      />
    </div>
  );
}

function Field({ label, children }: { label: string; children: JSX.Element }): JSX.Element {
  return (
    <label className="flex items-center gap-2">
      <span className="label w-20 shrink-0">{label}</span>
      {children}
    </label>
  );
}

function Metric({
  label,
  value,
  emphasis = false,
  warn = false,
  note,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
  warn?: boolean;
  /** Secondary figure, shown recessive beside the label. */
  note?: string | undefined;
}): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[2px]">
      <span className="label">
        {label}
        {note !== undefined && <span className="ml-1.5 normal-case opacity-70">{note}</span>}
      </span>
      <span
        className="num text-[12px]"
        style={{
          color: warn
            ? 'var(--color-status-bad)'
            : emphasis
              ? 'var(--color-text-hi)'
              : 'var(--color-text-mid)',
        }}
      >
        {value}
      </span>
    </div>
  );
}
