import { useEffect, useRef } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { createFrameState, Scene } from '../render/scene';
import { NetworkView } from '../render/networkView';
import { fitToView, panBy, zoomAt } from '../render/layout';
import type { Viewport } from '../render/layout';
import { DissectionView } from '../render/dissectionView';
import { dissect } from '../render/dissection';
import { describeSelection, keyToNavigation, navigate } from '../render/navigation';

/*
 * The canvas host.
 *
 * Spec §6.1: React must NOT re-render on animation frames. This component
 * therefore renders exactly one <canvas> and never again — it subscribes to the
 * store imperatively and mutates the Scene's frame state in place. No store
 * value is read through a hook in a way that would re-render during animation.
 */

export function NetworkCanvas(): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveRef = useRef<HTMLParagraphElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (canvas === null || container === null) return;

    const store = useAppStore;
    const initial = store.getState();

    const frame = createFrameState(initial.layout);
    const view = new NetworkView(initial.network);
    const dissectionView = new DissectionView();
    const scene = new Scene(canvas, frame);

    // §6.3: with motion off, pulses become discrete stage highlights.
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /** Copy the store's slow-moving values into the frame state. */
    const sync = (): void => {
      const state = store.getState();
      frame.layout = state.layout;
      frame.viewport = state.viewport;
      frame.weightOf = view.weightOf;
      frame.normalizedActivation = view.normalizedActivation;
      frame.normalizedDelta = view.normalizedDelta;
      frame.isDead = view.isDead;
      frame.isFrozen = view.isFrozen;
      frame.isAblated = view.isAblated;
      frame.frozenLayers = view.frozenLayers();
      frame.captions = view.captions();
      frame.wRef = view.weightRef;
      state.setWRefDisplay(view.weightRef);
      frame.colorblindSafe = state.colorblindSafe;
      frame.hover = state.hover;
      frame.selection = state.selection;
      frame.dissection = state.dissectionEnabled ? dissectionView : null;
      frame.thumbnails = state.showThumbnails ? state.thumbnails : null;
      view.setDiffBase(state.diffBase);
      frame.diffing = view.diffing;
      frame.dirty = true;
    };

    /**
     * Rebuild the dissection from the engine.
     *
     * Every number the cards show comes from this call, which runs a real
     * forward and backward pass and reads the caches — the UI never recomputes
     * the network (§0.6, and the Phase 4 gate).
     */
    const rebuildDissection = (): void => {
      const state = store.getState();
      if (!state.dissectionEnabled) {
        dissectionView.clear();
        return;
      }
      if (state.dataset.x.rows === 0 || state.network.inputSize !== state.dataset.x.cols) return;
      const d = dissect(
        state.network,
        state.dataset.x,
        state.dataset.y,
        Math.min(state.sampleIndex, state.dataset.x.rows - 1),
        state.learningRate,
      );
      dissectionView.load(d, state.layout, state.network.layers.length, reducedMotion);
      dissectionView.speed = state.speed;
      state.setTransportStatus(dissectionView.status());
    };

    let lastRevision = -1;
    let lastLayout: unknown = null;
    /*
     * Whether the reader has positioned the view themselves.
     *
     * A resize should keep the network on screen, but must not yank a viewport
     * somebody deliberately panned. Opening the mobile drawer halves the canvas
     * height, and without this the network simply fell off the bottom: the fit
     * had been computed for the taller canvas and nothing recomputed it.
     */
    let userAdjusted = false;
    let lastNonce = -1;
    let lastSample = -1;
    let lastEnabled = false;
    let lastStatusBeat = -1;
    const onStoreChange = (): void => {
      /*
       * Subscribers run synchronously inside the store's set(), so anything
       * that throws here aborts the update that triggered it — which is how a
       * dataset change once silently prevented the matching architecture change
       * from ever being applied. The render layer is downstream of state and
       * must never be able to break it.
       */
      try {
        const state = store.getState();
        if (state.revision !== lastRevision) {
          lastRevision = state.revision;
          view.retarget(state.network);
          // Show the network responding to a real sample rather than sitting
          // blank: every number on screen still comes from the engine.
          if (state.dataset.x.rows > 0) view.captureSample(state.dataset.x, 0);
        }

        /*
         * Refit when the STRUCTURE changes, not on every update. Switching from
         * a 2-input network to a 35-input one grows the layout past the
         * viewport, and leaving the old transform in place pushes half the
         * network off-screen. Keyed on layout identity rather than on the
         * revision counter, so a dataset reseed that leaves the shape alone
         * does not yank a viewport the user has deliberately panned.
         */
        if (state.layout !== lastLayout) {
          lastLayout = state.layout;
          // A new structure supersedes any manual framing of the old one.
          userAdjusted = false;
          const bounds = container.getBoundingClientRect();
          if (bounds.width > 0 && bounds.height > 0) {
            state.setViewport(fitToView(state.layout, bounds.width, bounds.height));
          }
        }
        // Rebuild when what is being dissected changes — not every update.
        if (
          state.dissectionEnabled !== lastEnabled ||
          state.sampleIndex !== lastSample ||
          state.revision !== lastRevision
        ) {
          lastEnabled = state.dissectionEnabled;
          lastSample = state.sampleIndex;
          rebuildDissection();
        }
        dissectionView.speed = state.speed;

        if (state.pendingCommand !== null && state.transportNonce !== lastNonce) {
          lastNonce = state.transportNonce;
          switch (state.pendingCommand) {
            case 'toggle':
              if (dissectionView.status().playing) dissectionView.pause();
              else dissectionView.play();
              break;
            case 'step':
              dissectionView.stepBeat();
              break;
            case 'back':
              dissectionView.stepBack();
              break;
            case 'restart':
              dissectionView.restart();
              break;
          }
          state.setTransportStatus(dissectionView.status());
        }

        view.refreshWeightReference();
        sync();
      } catch (error) {
        // Draw whatever the last good frame was rather than tearing down.
        console.error('AwryNN: canvas failed to sync with the store.', error);
      }
    };

    const unsubscribe = store.subscribe(onStoreChange);
    onStoreChange();

    /*
     * Push the transport readout back to React on BEAT boundaries only.
     * Mirroring it every frame would re-render React 60 times a second, which
     * §6.1 explicitly forbids; a beat changes a few times a second at most.
     */
    const statusTimer = window.setInterval(() => {
      if (!store.getState().dissectionEnabled || !dissectionView.active) return;
      const status = dissectionView.status();
      if (status.beatIndex !== lastStatusBeat || status.playing !== store.getState().transportStatus.playing) {
        lastStatusBeat = status.beatIndex;
        store.getState().setTransportStatus(status);
      }
    }, 120);

    /* ---- sizing ---- */
    const resize = (): void => {
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      scene.resize(rect.width, rect.height, window.devicePixelRatio);
      if (!userAdjusted) {
        store.getState().setViewport(fitToView(store.getState().layout, rect.width, rect.height));
      }
      frame.dirty = true;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    // Fit once, after the first real size is known.
    const rect = container.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      store.getState().setViewport(fitToView(store.getState().layout, rect.width, rect.height));
    }

    /* ---- pointer ---- */
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    // Distinguishes a click from the end of a drag, so panning never selects.
    let dragDistance = 0;

    /*
     * Weight scrubbing (§6.5): drag horizontally on an edge to change it.
     *
     * Sensitivity scales with the network's weight range, so one screen-width
     * of travel spans a comparable amount whether weights sit near 0.5 or near
     * 50. The scale is captured ONCE at mousedown and held for the gesture: the
     * displayed reference is derived from the weights, so reading it live made
     * dragging a weight up increase the sensitivity, which made the drag
     * accelerate away under the hand.
     */
    let scrub: {
      index: number;
      startX: number;
      startValue: number;
      unitsPerPixel: number;
    } | null = null;
    const SCRUB_SENSITIVITY = 0.005;

    /*
     * Touch: pinch to zoom, one finger to pan or scrub.
     *
     * Pointer events give drag for free, but a trackpad's wheel-zoom has no
     * touch equivalent, so without this a phone can pan but never zoom, and the
     * network is stuck at whatever scale it was fitted to.
     */
    const activePointers = new Map<number, { x: number; y: number }>();
    let pinch: { distance: number; centreX: number; centreY: number } | null = null;

    const pinchState = (): { distance: number; centreX: number; centreY: number } | null => {
      if (activePointers.size < 2) return null;
      const [a, b] = [...activePointers.values()];
      if (a === undefined || b === undefined) return null;
      return {
        distance: Math.hypot(b.x - a.x, b.y - a.y),
        centreX: (a.x + b.x) / 2,
        centreY: (a.y + b.y) / 2,
      };
    };

    const localPoint = (event: PointerEvent): { x: number; y: number } => {
      const bounds = canvas.getBoundingClientRect();
      return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    };

    const onPointerMove = (event: PointerEvent): void => {
      const state = store.getState();
      const { x, y } = localPoint(event);

      if (activePointers.has(event.pointerId)) {
        activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      }
      if (pinch !== null) {
        const next = pinchState();
        if (next !== null && next.distance > 0 && pinch.distance > 0) {
          const bounds = canvas.getBoundingClientRect();
          userAdjusted = true;
          state.setViewport(
            zoomAt(
              state.viewport,
              next.centreX - bounds.left,
              next.centreY - bounds.top,
              next.distance / pinch.distance,
            ),
          );
          pinch = next;
        }
        return;
      }

      if (scrub !== null) {
        const travelled = event.clientX - scrub.startX;
        state.updateScrub(scrub.index, scrub.startValue + travelled * scrub.unitsPerPixel);
        view.refreshWeightReference();
        frame.dirty = true;
        return;
      }

      if (dragging) {
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        dragDistance += Math.abs(dx) + Math.abs(dy);
        lastX = event.clientX;
        lastY = event.clientY;
        userAdjusted = true;
        state.setViewport(panBy(state.viewport, dx, dy));
        return;
      }

      const hit = state.hitIndex.pickScreen(state.viewport, x, y);
      const current = state.hover;
      canvas.style.cursor =
        hit === null ? 'default' : hit.kind === 'edge' ? 'ew-resize' : 'pointer';
      // Only touch the store when the hit actually changed: a pointermove that
      // lands on the same edge must not push a state update.
      const changed =
        (hit === null) !== (current === null) ||
        (hit !== null && current !== null && (hit.kind !== current.kind || hit.index !== current.index));
      if (changed) state.setHover(hit);
    };

    const onPointerDown = (event: PointerEvent): void => {
      activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (activePointers.size === 2) {
        // A second finger cancels whatever the first was doing and becomes a
        // pinch, so a drag does not fight the zoom.
        pinch = pinchState();
        dragging = false;
        if (scrub !== null) {
          scrub = null;
          store.getState().endScrub();
        }
        return;
      }
      if (activePointers.size > 2) return;

      const state = store.getState();
      const { x, y } = localPoint(event);
      const hit = state.hitIndex.pickScreen(state.viewport, x, y);

      // Pressing on an edge starts a scrub; pressing anywhere else pans.
      if (hit !== null && hit.kind === 'edge') {
        const index = state.parameterIndex(
          hit.edge.layer,
          hit.edge.isBias ? 'b' : 'W',
          hit.edge.row,
          hit.edge.col,
        );
        if (index >= 0) {
          scrub = {
            index,
            startX: event.clientX,
            startValue: state.network.params[index] as number,
            unitsPerPixel: SCRUB_SENSITIVITY * Math.max(0.2, state.wRefDisplay),
          };
          state.beginScrub(index);
          state.setSelection(hit);
          canvas.setPointerCapture(event.pointerId);
          canvas.style.cursor = 'ew-resize';
          return;
        }
      }

      dragging = true;
      dragDistance = 0;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    };

    const onPointerUp = (event: PointerEvent): void => {
      activePointers.delete(event.pointerId);
      if (activePointers.size < 2) pinch = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (scrub !== null) {
        scrub = null;
        store.getState().endScrub();
        canvas.style.cursor = 'default';
        return;
      }
      const wasDragging = dragging;
      dragging = false;
      if (!wasDragging || dragDistance > 4) return;
      const state = store.getState();
      const { x, y } = localPoint(event);
      const hit = state.hitIndex.pickScreen(state.viewport, x, y);
      state.setSelection(hit);
      // Clicking a neuron moves the dissection's focus to it, which is how a
      // learner chooses whose card to read on a wide layer.
      if (hit !== null && hit.kind === 'node' && hit.node.kind !== 'bias' && hit.node.layer > 0) {
        dissectionView.focusUnit = hit.node.unit;
        frame.dirty = true;
      }
    };

    const onPointerLeave = (event: PointerEvent): void => {
      activePointers.delete(event.pointerId);
      if (activePointers.size < 2) pinch = null;
      store.getState().setHover(null);
      canvas.style.cursor = 'default';
    };

    const onWheel = (event: WheelEvent): void => {
      event.preventDefault();
      const state = store.getState();
      const bounds = canvas.getBoundingClientRect();
      // Exponential in the wheel delta so trackpad and mouse feel the same.
      const factor = Math.exp(-event.deltaY * 0.0015);
      userAdjusted = true;
      state.setViewport(
        zoomAt(state.viewport, event.clientX - bounds.left, event.clientY - bounds.top, factor),
      );
    };

    const onDoubleClick = (): void => {
      const bounds = container.getBoundingClientRect();
      const state = store.getState();
      // Fitting is how the reader says "stop remembering my framing".
      userAdjusted = false;
      state.setViewport(fitToView(state.layout, bounds.width, bounds.height));
    };

    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', onDoubleClick);

    /*
     * Keyboard navigation (§9).
     *
     * The canvas is a picture, so nothing inside it can take focus on its own.
     * The canvas takes focus and translates arrow keys into a traversal, and
     * every move is announced into a live region — otherwise the entire
     * interface is conveyed by colour and position, which a screen reader
     * cannot see.
     */
    const announce = (hit: ReturnType<typeof navigate>): void => {
      const live = liveRef.current;
      if (live === null) return;
      const state = store.getState();
      live.textContent = describeSelection(
        state.layout,
        hit,
        (target) => {
          if (target.kind === 'edge') {
            const index = state.parameterIndex(
              target.edge.layer,
              target.edge.isBias ? 'b' : 'W',
              target.edge.row,
              target.edge.col,
            );
            return index >= 0 ? (state.network.params[index] as number) : null;
          }
          if (target.node.layer === 0 || target.node.kind === 'bias') return null;
          const layer = state.network.layers[target.node.layer - 1];
          return layer === undefined ? null : (layer.b.data[target.node.unit] ?? null);
        },
        view.captions(),
      );
    };

    const onKeyDown = (event: KeyboardEvent): void => {
      const state = store.getState();

      if (event.key === 'Enter') {
        // Hand focus to the inspector, which is where a value is actually typed.
        event.preventDefault();
        const editor = document.querySelector<HTMLInputElement>('input[aria-label="value"]');
        editor?.focus();
        editor?.select();
        return;
      }

      // Nudge the selected weight without a mouse.
      if ((event.key === '+' || event.key === '=' || event.key === '-') && state.selection?.kind === 'edge') {
        event.preventDefault();
        const edge = state.selection.edge;
        const index = state.parameterIndex(edge.layer, edge.isBias ? 'b' : 'W', edge.row, edge.col);
        if (index >= 0) {
          const step = (event.key === '-' ? -1 : 1) * 0.05 * Math.max(0.2, state.wRefDisplay);
          state.setParameter(index, (state.network.params[index] as number) + step, 'Nudge weight');
          announce(state.selection);
        }
        return;
      }

      const command = keyToNavigation(event.key, event.shiftKey);
      if (command === null) return;
      event.preventDefault();
      const next = navigate(state.layout, state.selection, command);
      // A refused move leaves the selection where it was rather than clearing it.
      if (next === null && command !== 'clear') return;
      state.setSelection(next);
      if (next !== null && next.kind === 'node' && next.node.kind !== 'bias' && next.node.layer > 0) {
        dissectionView.focusUnit = next.node.unit;
      }
      announce(next);
      frame.dirty = true;
    };

    canvas.addEventListener('keydown', onKeyDown);

    scene.start();

    return () => {
      scene.stop();
      window.clearInterval(statusTimer);
      unsubscribe();
      observer.disconnect();
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDoubleClick);
      canvas.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <canvas
        ref={canvasRef}
        className="block h-full w-full touch-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[var(--color-focus)]"
        // Focusable so the arrow keys have somewhere to land, and described so
        // its controls are discoverable without seeing them.
        tabIndex={0}
        role="application"
        aria-label="Neural network diagram. Arrow keys move between units, shift with up or down walks a unit's connections, Enter edits the selected value, plus and minus nudge it."
      />
      {/* The canvas conveys everything through colour and position. This is the
          text equivalent, updated on every keyboard move. */}
      <p ref={liveRef} className="sr-only" aria-live="polite" role="status" />
    </div>
  );
}

export type { Viewport };
