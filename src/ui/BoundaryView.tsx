import { useCallback, useEffect, useRef, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { boundsFor, drawPoints, paintBoundary } from '../render/draw/heatmap';
import type { BoundaryData } from '../render/draw/heatmap';
import type { BoundaryRequest, BoundaryResponse } from '../worker/boundary.worker';
import { predictedClass } from '../engine/trainer';
import { createMatrix } from '../engine/tensor';
import { COLORS } from '../render/theme';

/*
 * The decision boundary (§6.4).
 *
 * The grid is evaluated in a worker and composited here. Recomputation is
 * throttled to ~20Hz and keyed on a nonce, so a reply that arrives after the
 * network has already moved on is discarded rather than painting a boundary
 * for weights that no longer exist.
 */

const RESOLUTION = 120;
const THROTTLE_MS = 50;

/*
 * Scrubbing computes the boundary SYNCHRONOUSLY so it tracks the hand with no
 * lag (§6.4). A worker round-trip cannot, because the reply lands a frame or
 * more after the pointer has moved on.
 *
 * But a full 120x120 pass is not affordable on the main thread. Measured:
 *
 *   2-6-4-1        14,400 rows    10 ms
 *   2-24-24-3      14,400 rows    60 ms
 *   2-32-32-32-1   14,400 rows   157 ms
 *
 * against the 8ms main-thread budget in §10. So the scrub resolution is chosen
 * from the MEASURED cost of the previous pass rather than fixed: a small
 * network scrubs at nearly full detail, a large one drops to a coarse grid that
 * still tracks the drag, and releasing the pointer asks the worker for the full
 * 120x120.
 */
/** §6.4 specifies 24x24 for the per-neuron activation maps. */
export const THUMBNAIL_RESOLUTION = 24;

const SCRUB_BUDGET_MS = 6;
const SCRUB_MIN_RESOLUTION = 24;
const SCRUB_MAX_RESOLUTION = 96;

export function BoundaryView(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dataset = useAppStore((s) => s.dataset);
  const network = useAppStore((s) => s.network);
  const revision = useAppStore((s) => s.revision);
  const epoch = useAppStore((s) => s.epoch);
  const [unsupported, setUnsupported] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const nonceRef = useRef(0);
  const dataRef = useRef<BoundaryData | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const lastRequestRef = useRef(0);
  /** Microseconds per grid row, measured and smoothed. */
  const rowCostRef = useRef(0.001);
  const scrubGridRef = useRef<{ grid: ReturnType<typeof createMatrix>; n: number } | null>(null);

  const is2d = dataset.featureCount === 2;
  const scrubbing = useAppStore((s) => s.scrubbing);
  const editRevision = useAppStore((s) => s.editRevision);
  const showThumbnails = useAppStore((s) => s.showThumbnails);
  const setThumbnails = useAppStore((s) => s.setThumbnails);

  /*
   * The worker's reply arrives well after the render that created its handler.
   * Reading `dataset` and `network` straight from the closure would therefore
   * paint the data as it was when the effect ran, so a boundary computed for a
   * new network would be composited under the OLD points. Refs are updated
   * every render and read at call time, so a callback always sees current state.
   */
  const latest = useRef({ dataset, network, showThumbnails });
  latest.current = { dataset, network, showThumbnails };


  /** Ask the worker for a fresh grid, at most every THROTTLE_MS. */
  const request = useCallback((): void => {
    const worker = workerRef.current;
    if (worker === null) return;
    const { dataset, network } = latest.current;
    if (dataset.featureCount !== 2) return;
    const now = performance.now();
    if (now - lastRequestRef.current < THROTTLE_MS) return;
    lastRequestRef.current = now;

    const bounds = boundsFor(dataset.x.data, dataset.x.rows);
    nonceRef.current += 1;
    const message: BoundaryRequest = {
      type: 'boundary',
      network: network.serialize(),
      resolution: RESOLUTION,
      xMin: bounds.xMin,
      xMax: bounds.xMax,
      yMin: bounds.yMin,
      yMax: bounds.yMax,
      thumbnails: latest.current.showThumbnails,
      thumbnailResolution: THUMBNAIL_RESOLUTION,
      nonce: nonceRef.current,
    };
    worker.postMessage(message);
  }, []);

  const paint = useCallback((): void => {
    const canvas = canvasRef.current;
    const { dataset, network } = latest.current;
    if (canvas === null || dataset.featureCount !== 2) return;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return;

    const dpr = Math.max(1, window.devicePixelRatio);
    const cssSize = canvas.clientWidth;
    if (cssSize === 0) return;
    if (canvas.width !== Math.round(cssSize * dpr)) {
      canvas.width = Math.round(cssSize * dpr);
      canvas.height = Math.round(cssSize * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = COLORS.bgCanvas;
    ctx.fillRect(0, 0, cssSize, cssSize);

    const data = dataRef.current;
    if (data !== null) {
      let offscreen = offscreenRef.current;
      if (offscreen === null || offscreen.width !== data.resolution) {
        offscreen = document.createElement('canvas');
        offscreen.width = data.resolution;
        offscreen.height = data.resolution;
        offscreenRef.current = offscreen;
      }
      const offCtx = offscreen.getContext('2d');
      if (offCtx !== null) {
        const image = offCtx.createImageData(data.resolution, data.resolution);
        paintBoundary(image, data);
        offCtx.putImageData(image, 0, 0);
        // The browser upscales with interpolation, which is both faster than
        // doing it per-pixel and what makes 120x120 read as a surface.
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(offscreen, 0, 0, cssSize, cssSize);
      }
    }

    const bounds = boundsFor(dataset.x.data, dataset.x.rows);
    if (dataset.labels !== null) {
      const predictions = new Int32Array(dataset.x.rows);
      const output = network.forward(dataset.x, false);
      for (let i = 0; i < dataset.x.rows; i++) predictions[i] = predictedClass(output, i);
      drawPoints(
        ctx as unknown as Parameters<typeof drawPoints>[0],
        dataset.x.data,
        dataset.labels,
        predictions,
        bounds,
        0,
        0,
        cssSize,
        cssSize,
        { radius: 2.6, showMisclassified: true },
      );
    }
  }, []);

  /**
   * Evaluate the grid on the main thread at whatever resolution fits the
   * budget, and store it as if it had come from the worker.
   */
  const computeSync = useCallback((): void => {
    const { dataset, network } = latest.current;
    if (dataset.featureCount !== 2 || network.inputSize !== 2) return;

    const rows = Math.max(1, Math.floor((SCRUB_BUDGET_MS * 1000) / rowCostRef.current));
    const n = Math.max(
      SCRUB_MIN_RESOLUTION,
      Math.min(SCRUB_MAX_RESOLUTION, Math.floor(Math.sqrt(rows))),
    );

    let cached = scrubGridRef.current;
    if (cached === null || cached.n !== n) {
      cached = { grid: createMatrix(n * n, 2), n };
      scrubGridRef.current = cached;
    }
    const bounds = boundsFor(dataset.x.data, dataset.x.rows);
    for (let row = 0; row < n; row++) {
      const y = bounds.yMax - ((bounds.yMax - bounds.yMin) * row) / (n - 1);
      for (let col = 0; col < n; col++) {
        const x = bounds.xMin + ((bounds.xMax - bounds.xMin) * col) / (n - 1);
        cached.grid.data[(row * n + col) * 2] = x;
        cached.grid.data[(row * n + col) * 2 + 1] = y;
      }
    }

    const started = performance.now();
    const predictions = network.forward(cached.grid, false);
    const elapsed = performance.now() - started;
    // Smoothed, so one slow frame does not collapse the resolution outright.
    const measured = (elapsed * 1000) / (n * n);
    rowCostRef.current = rowCostRef.current * 0.7 + measured * 0.3;

    const classes = predictions.cols;
    const values = new Float32Array(n * n);
    if (classes === 1) {
      for (let i = 0; i < values.length; i++) values[i] = predictions.data[i] as number;
    } else {
      for (let i = 0; i < values.length; i++) {
        let best = 0;
        let bestValue = -Infinity;
        for (let c = 0; c < classes; c++) {
          const v = predictions.data[i * classes + c] as number;
          if (v > bestValue) {
            bestValue = v;
            best = c;
          }
        }
        values[i] = best + Math.min(0.999, Math.max(0, bestValue));
      }
    }
    dataRef.current = { values, resolution: n, classes };
  }, []);

  useEffect(() => {
    if (!is2d) return;
    let worker: Worker;
    try {
      worker = new Worker(new URL('../worker/boundary.worker.ts', import.meta.url), {
        type: 'module',
      });
    } catch {
      setUnsupported(true);
      return;
    }
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<BoundaryResponse>): void => {
      const message = event.data;
      if (message.type !== 'boundary') return;
      // Stale replies are dropped: the weights they describe are gone.
      if (message.nonce !== nonceRef.current) return;
      dataRef.current = {
        values: message.values,
        resolution: message.resolution,
        classes: message.classes,
      };
      // Handed to the canvas, which draws each unit's map at its node (§6.4).
      setThumbnails(
        message.thumbnails === null
          ? null
          : {
              values: message.thumbnails,
              slots: message.thumbnailSlots,
              resolution: message.thumbnailResolution,
            },
      );
      paint();
    };
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [is2d, paint, setThumbnails]);

  // Recompute whenever the network changes: a new architecture, a reseed, or
  // another epoch of training.
  useEffect(() => {
    if (!is2d) return;
    if (scrubbing) {
      // Synchronous while the pointer is down, so the surface deforms under
      // the hand rather than trailing it.
      computeSync();
      paint();
      return;
    }
    request();
    paint();
  }, [revision, epoch, editRevision, scrubbing, is2d, request, paint, computeSync]);

  if (!is2d) {
    return (
      <div className="px-4 py-3">
        <p className="panel-title mb-2">Decision boundary</p>
        <p className="text-[11px] leading-relaxed text-[var(--color-text-lo)]">
          Only drawn for two-dimensional datasets. This one has{' '}
          <span className="num">{dataset.featureCount}</span> inputs, which cannot be
          shown as a plane.
        </p>
      </div>
    );
  }

  return (
    <div className="px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between">
        <p className="panel-title">Decision boundary</p>
        <span className="num text-[10px] text-[var(--color-text-lo)]">
          {RESOLUTION}×{RESOLUTION}
        </span>
      </div>
      {unsupported ? (
        <p className="text-[11px] text-[var(--color-status-bad)]">
          Web Workers are unavailable, so the boundary cannot be computed here.
        </p>
      ) : (
        <canvas
          ref={canvasRef}
          className="block aspect-square w-full border border-[var(--color-line-hair)]"
          aria-label="Decision boundary with training points"
        />
      )}
      <p className="mt-2 text-[11px] leading-relaxed text-[var(--color-text-lo)]">
        Colour is the predicted class, and it fades where the network is unsure. Ringed
        points are the ones it currently gets wrong.
      </p>
    </div>
  );
}
