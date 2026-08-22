import { useEffect, useMemo, useRef, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { drawChart, drawHistogram, histogram } from '../render/draw/curves';
import { drawConfusion } from '../render/draw/heatmap';
import { buildReport, predictedClass } from '../engine/trainer';
import { formatGradCheckResult, gradientCheck } from '../engine/gradcheck';
import { rowView } from '../engine/tensor';
import { COLORS, formatNorm, formatPercent } from '../render/theme';

/*
 * Diagnostics (§6.4).
 *
 * "per-layer gradient norm over time (log y-axis — this makes vanishing
 * gradients *undeniable*), weight histograms per layer, dead-unit count,
 * saturation percentage, confusion matrix."
 *
 * The log axis is not a preference here. On a linear axis a first layer
 * receiving 1e-6 and a last layer receiving 1e-1 both render as "on the
 * bottom"; the log axis is what turns that into five visible decades.
 */

export function DiagnosticsPanel(): JSX.Element {
  const metrics = useAppStore((s) => s.metrics);
  const latest = useAppStore((s) => s.latest);
  const network = useAppStore((s) => s.network);

  return (
    <div className="px-4 py-3">
      <p className="panel-title mb-3">Diagnostics</p>

      <GradientNormChart />

      <div className="mt-3 flex flex-col gap-1">
        <Stat
          label="dead units"
          value={latest === null ? '—' : String(latest.deadUnits)}
          warn={latest !== null && latest.deadUnits > 0}
        />
        <Stat
          label="saturation"
          value={latest === null ? '—' : formatPercent(latest.saturation)}
          warn={latest !== null && latest.saturation > 0.75}
        />
        <Stat label="grad norm" value={latest === null ? '—' : formatNorm(latest.gradientNorm)} />
        <Stat label="epochs" value={String(metrics.length)} />
      </div>

      {/* Directly under the readouts it verifies, rather than at the foot of
          the panel behind two charts. It is the control that backs every
          number in this app, and one nobody scrolls to is one nobody uses. */}

      <GradientCheck />
      <WeightHistograms />
      <ConfusionMatrix />

      {latest !== null && latest.deadUnits > 0 && (
        <p className="mt-2 text-[10px] leading-snug text-[var(--color-status-bad)]">
          {latest.deadUnits} unit{latest.deadUnits === 1 ? '' : 's'} produced zero output for the
          whole epoch. A ReLU that has been pushed fully negative cannot recover, because its
          gradient there is exactly zero. Lower the learning rate, or switch to leaky ReLU.
        </p>
      )}
      {network.layers.some((l) => l.activationName === 'sigmoid') &&
        latest !== null &&
        latest.saturation > 0.75 && (
          <p className="mt-2 text-[10px] leading-snug text-[var(--color-text-lo)]">
            Most units are pinned at the ends of their range, where the derivative is nearly zero.
            That is what starves the earlier layers of gradient.
          </p>
        )}
    </div>
  );
}

/**
 * The claim, checkable on your own network (§12).
 *
 * Everything else in this app asks you to believe that the numbers are real.
 * This differentiates the loss numerically, coordinate by coordinate, and
 * compares the result against the gradients backpropagation produced. If they
 * agree to nine or ten decimal places, the arithmetic behind every other panel
 * is right, and you did not have to take anyone's word for it.
 *
 * A SPOT CHECK rather than the whole network. The full check costs four forward
 * passes per parameter per step pair, which is 15ms for the default network and
 * six seconds for a wide one, and a button that freezes the tab for six seconds
 * is a button nobody presses twice. Four hundred coordinates spread evenly
 * across every layer is a fraction of a second and just as convincing: if that
 * many, drawn from everywhere, all agree, what the rest would tell you is
 * something about float64 rather than about the code. The count is reported, so
 * the claim being made is the one on screen.
 */
const SPOT_CHECK_COORDINATES = 400;
/** Enough rows for a real gradient, few enough to keep the check quick. */
const SPOT_CHECK_BATCH = 8;

function GradientCheck(): JSX.Element {
  const network = useAppStore((s) => s.network);
  const dataset = useAppStore((s) => s.dataset);
  const [report, setReport] = useState<{ text: string; passed: boolean } | null>(null);
  const [busy, setBusy] = useState(false);

  const ready = dataset.x.rows > 0 && dataset.x.cols === network.inputSize;

  const run = (): void => {
    setBusy(true);
    // Yield a frame so the button can paint its pending state before the main
    // thread goes away for up to half a second.
    requestAnimationFrame(() => {
      try {
        const rows = Math.min(SPOT_CHECK_BATCH, dataset.x.rows);
        const result = gradientCheck(network, rowView(dataset.x, rows), rowView(dataset.y, rows), {
          maxCoordinates: SPOT_CHECK_COORDINATES,
        });
        setReport({ text: formatGradCheckResult(result), passed: result.passed });
      } catch (error) {
        setReport({ text: error instanceof Error ? error.message : String(error), passed: false });
      } finally {
        setBusy(false);
      }
    });
  };

  return (
    <div className="mt-4 border-t border-[var(--color-line-hair)] pt-3">
      <p className="label mb-1.5">check gradients</p>
      <button className="control w-full" onClick={run} disabled={busy || !ready}>
        {busy ? 'Checking…' : 'Check gradients'}
      </button>

      {report !== null && (
        <p
          className="num mt-1.5 text-[10px] leading-snug"
          style={{
            color: report.passed ? 'var(--color-text-mid)' : 'var(--color-status-bad)',
          }}
        >
          {report.text}
        </p>
      )}

      <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
        {ready
          ? 'Nudges each parameter up and down, measures how the loss actually responds, and compares that against the gradient backpropagation computed. Agreement to 1e-7 or better is a pass.'
          : 'Needs a dataset whose width matches the network.'}
      </p>
    </div>
  );
}

function GradientNormChart(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const metrics = useAppStore((s) => s.metrics);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const dpr = Math.max(1, window.devicePixelRatio);
    const width = canvas.clientWidth;
    const height = 132;
    if (width === 0) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bgChassis;
    ctx.fillRect(0, 0, width, height);

    const layerCount = metrics[metrics.length - 1]?.gradientNorms.length ?? 0;
    const series = Array.from({ length: layerCount }, (_, layer) => ({
      values: metrics.map((m) => m.gradientNorms[layer] ?? 0),
      label: `L${layer + 1}`,
      dashed: false,
      dim: false,
      // Layer depth is ordinal, so the ramp runs from faint (first) to bright
      // (last). Categorical hues would be both wrong for ordered data and a
      // breach of the rule that nothing competes with the weight poles.
      tone: layerCount <= 1 ? 1 : layer / (layerCount - 1),
    }));

    drawChart(
      ctx as unknown as Parameters<typeof drawChart>[0],
      0,
      0,
      width,
      height,
      series,
      {
        logScale: true,
        // Window scales with the run, so a 50-epoch run is barely smoothed and
        // a 2000-epoch one is readable.
        smooth: Math.max(1, Math.round(metrics.length / 40)),
        yLabel: 'gradient norm (log, smoothed)',
      },
    );
  }, [metrics]);

  return (
    <div>
      <p className="label mb-1">per-layer gradient</p>
      <canvas
        ref={canvasRef}
        className="block w-full border border-[var(--color-line-hair)]"
        style={{ height: 132 }}
        aria-label="Per-layer gradient norm over epochs, log scale"
      />
      <p className="mt-1 text-[10px] leading-snug text-[var(--color-text-lo)]">
        Faint is the first layer, bright is the last. On a log axis, a gap of several decades
        between them is a vanishing gradient.
      </p>
    </div>
  );
}

function WeightHistograms(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const network = useAppStore((s) => s.network);
  const editRevision = useAppStore((s) => s.editRevision);
  const epoch = useAppStore((s) => s.epoch);
  const revision = useAppStore((s) => s.revision);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const perChart = 52;
    const count = network.layers.length;
    const dpr = Math.max(1, window.devicePixelRatio);
    const width = canvas.clientWidth;
    const height = count * perChart;
    if (width === 0) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bgChassis;
    ctx.fillRect(0, 0, width, height);

    network.layers.forEach((layer, i) => {
      drawHistogram(
        ctx as unknown as Parameters<typeof drawHistogram>[0],
        0,
        i * perChart,
        width,
        perChart - 4,
        histogram(layer.W.data),
        `L${i + 1} · ${layer.inputs}×${layer.units}`,
      );
    });
  }, [network, editRevision, epoch, revision]);

  return (
    <div className="mt-3">
      <p className="label mb-1">weight distribution</p>
      <canvas
        ref={canvasRef}
        className="block w-full"
        aria-label="Weight histogram per layer"
      />
    </div>
  );
}

function ConfusionMatrix(): JSX.Element | null {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const network = useAppStore((s) => s.network);
  const dataset = useAppStore((s) => s.dataset);
  const epoch = useAppStore((s) => s.epoch);
  const editRevision = useAppStore((s) => s.editRevision);

  const report = useMemo(() => {
    if (dataset.labels === null || dataset.x.rows === 0) return null;
    if (network.inputSize !== dataset.x.cols) return null;
    const predictions = network.forward(dataset.x, false);
    return buildReport(predictions, dataset.labels, Math.max(2, dataset.classCount));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [network, dataset, epoch, editRevision]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || report === null) return;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const dpr = Math.max(1, window.devicePixelRatio);
    const size = Math.min(canvas.clientWidth, 200);
    if (size === 0) return;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bgChassis;
    ctx.fillRect(0, 0, size, size);

    drawConfusion(
      ctx as unknown as Parameters<typeof drawConfusion>[0],
      0,
      0,
      size,
      report.confusion,
      dataset.classNames.length > 0
        ? dataset.classNames.map((n) => n.replace('class ', ''))
        : report.confusion.map((_, i) => String(i)),
    );
  }, [report, dataset]);

  if (report === null) return null;

  return (
    <div className="mt-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="label">confusion</span>
        <span className="num text-[10px] text-[var(--color-text-lo)]">
          {formatPercent(report.accuracy)}
        </span>
      </div>
      <canvas ref={canvasRef} className="block w-full" aria-label="Confusion matrix" />
      <p className="mt-1 text-[10px] leading-snug text-[var(--color-text-lo)]">
        Rows are the true class, columns the prediction. The diagonal is what it got right.
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  warn = false,
}: {
  label: string;
  value: string;
  warn?: boolean;
}): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[2px]">
      <span className="label">{label}</span>
      <span
        className="num text-[12px]"
        style={{ color: warn ? 'var(--color-status-bad)' : 'var(--color-text-mid)' }}
      >
        {value}
      </span>
    </div>
  );
}

export { predictedClass };
