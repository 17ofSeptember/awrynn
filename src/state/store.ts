/*
 * Application state.
 *
 * Spec §6.1: "The canvas subscribes to the Zustand store directly and reads
 * from a mutable frame-state object" — React must not re-render on animation
 * frames. So the split here is deliberate:
 *
 *   - Configuration and selection live in the store. Changing them is rare and
 *     a React re-render is the right response.
 *   - Per-frame values (activations, weight reference, hover position) live in
 *     the FrameState object the Scene owns. The canvas mutates and reads them
 *     without the store ever hearing about it.
 *
 * Anything that would tick at 60fps must not be put in this store.
 */

import { create } from 'zustand';
import type { Dataset, DatasetOptions } from '../engine/datasets/index';
import { generateDataset } from '../engine/datasets/index';
import type { LayerSpec } from '../engine/layers';
import { Network, validateConfig } from '../engine/network';
import { computeLayout } from '../render/layout';
import type { Layout, Viewport } from '../render/layout';
import { IDENTITY_VIEWPORT } from '../render/layout';
import { HitIndex } from '../render/hit';
import type { Hit } from '../render/hit';
import type { TransportStatus } from '../render/dissectionView';
import { SPEED_MAX, SPEED_MIN } from '../render/animation';
import type { EpochMetrics, TrainerConfig } from '../engine/trainer';
import { TrainingController } from './trainingController';
import { CommandStack, setFlagCommand, setParameterCommand } from './commands';
import { HistoryBuffer, parameterDelta } from './history';
import type { Snapshot } from './history';
import type { LessonPreset } from '../lessons/types';
import type { NetworkState, SerializedNetwork } from '../engine/network';
import { DEFAULT_TRAINING, toNetworkConfig } from './architecture';
import { shareUrl as buildShareUrl } from './shareLink';
import type { SharedState } from './shareLink';
import type { Architecture, TrainingSettings } from './architecture';

/*
 * Re-exported so that `Architecture` and `TrainingSettings` still read as
 * store concepts at every call site; architecture.ts exists to break an import
 * cycle, not to introduce a second place to look for them.
 */
export { DEFAULT_TRAINING, toNetworkConfig };
export type { Architecture, TrainingSettings };

/**
 * Adapt an architecture to a dataset, returning what had to change.
 *
 * A dataset carries a feature count, a class count and a loss it is built for,
 * and a network whose input width disagrees with its data is not a valid state
 * — the forward pass throws. So this reconciliation is a STATE INVARIANT, not a
 * UI concern, and it lives here rather than in the panel that happens to host
 * the dropdown.
 *
 * Hidden layers are preserved (§6.5: never silently reset the whole network on
 * an edit), and the changes are reported so the UI can say what happened.
 */
export function reconcileArchitecture(
  architecture: Architecture,
  dataset: Dataset,
): { architecture: Architecture; changes: string[] } {
  const changes: string[] = [];
  const outputUnits = dataset.classCount > 2 ? dataset.classCount : 1;
  const outputActivation: LayerSpec['activation'] =
    dataset.suggestedLoss === 'cce' ? 'softmax' : dataset.suggestedLoss === 'bce' ? 'sigmoid' : 'linear';

  const hidden = architecture.layers.slice(0, -1);
  const previousOutput = architecture.layers[architecture.layers.length - 1];

  if (architecture.inputSize !== dataset.featureCount) {
    changes.push(`Input layer resized ${architecture.inputSize} → ${dataset.featureCount}.`);
  }
  if (architecture.loss !== dataset.suggestedLoss) {
    changes.push(`Loss switched to ${dataset.suggestedLoss}.`);
  }
  if (previousOutput !== undefined && previousOutput.units !== outputUnits) {
    changes.push(`Output layer resized ${previousOutput.units} → ${outputUnits}.`);
  }
  if (previousOutput !== undefined && previousOutput.activation !== outputActivation) {
    changes.push(`Output activation switched to ${outputActivation}.`);
  }
  if (hidden.length > 0) {
    changes.push(`${hidden.length} hidden layer${hidden.length === 1 ? '' : 's'} preserved.`);
  }

  return {
    architecture: {
      ...architecture,
      inputSize: dataset.featureCount,
      loss: dataset.suggestedLoss,
      layers: [...hidden, { units: outputUnits, activation: outputActivation }],
    },
    changes,
  };
}

export interface AppState {
  /* ---- configuration ---- */
  architecture: Architecture;
  datasetOptions: DatasetOptions;

  /* ---- derived, rebuilt together so they can never disagree ---- */
  network: Network;
  dataset: Dataset;
  layout: Layout;
  hitIndex: HitIndex;
  /** Bumped whenever network/layout/dataset are replaced, so the canvas can rebind. */
  revision: number;
  /** Problems from validateConfig, surfaced rather than thrown (§4.4). */
  configErrors: readonly string[];
  /** What the last dataset change did to the architecture (§6.5). */
  lastReconciliation: readonly string[];

  /* ---- view ---- */
  viewport: Viewport;
  hover: Hit | null;
  selection: Hit | null;
  colorblindSafe: boolean;
  showBiases: boolean;
  /** Per-neuron activation maps drawn at each node (§6.4). */
  showThumbnails: boolean;
  thumbnails: ThumbnailData | null;

  /* ---- dissection (§6.3) ---- */
  dissectionEnabled: boolean;
  /** Which sample walks through the network. */
  sampleIndex: number;
  speed: number;
  learningRate: number;
  /**
   * Mirrored from the DissectionView so panels can render it.
   *
   * Pushed by the canvas at beat boundaries, NOT every frame — §6.1 forbids
   * React re-rendering on animation frames, and a per-frame store write would
   * do exactly that.
   */
  transportStatus: TransportStatus;
  /**
   * Transport commands are one-shot events, not state. They are delivered as a
   * (command, nonce) pair so the canvas can tell two identical presses apart —
   * pressing "step" twice must step twice, which a plain value would not convey.
   */
  pendingCommand: TransportCommand | null;
  transportNonce: number;

  /* ---- actions ---- */
  setArchitecture: (next: Partial<Architecture>) => void;
  setDatasetOptions: (next: Partial<DatasetOptions>) => void;
  setViewport: (viewport: Viewport) => void;
  setHover: (hit: Hit | null) => void;
  setSelection: (hit: Hit | null) => void;
  toggleColorblindSafe: () => void;
  toggleBiases: () => void;
  toggleThumbnails: () => void;
  setThumbnails: (data: ThumbnailData | null) => void;
  toggleDissection: () => void;
  setSampleIndex: (index: number) => void;
  setSpeed: (speed: number) => void;
  setLearningRate: (rate: number) => void;
  /** Transport verbs. The canvas subscribes and applies them to the timeline. */
  transport: (command: TransportCommand) => void;
  setTransportStatus: (status: TransportStatus) => void;

  /* ---- training ---- */
  training: TrainingSettings;
  trainingStatus: TrainingStatus;
  trainingError: string | null;
  epoch: number;
  metrics: readonly EpochMetrics[];
  latest: EpochMetrics | null;
  /** True when training is running on the main thread because no worker exists. */
  trainingOnMainThread: boolean;
  /** Log-scale toggle for the loss chart (§6.4). */
  lossLogScale: boolean;

  /* ---- history scrubber (§6.6) ---- */
  /** Snapshots taken as training progresses, oldest first. */
  history: readonly Snapshot[];
  /** Index being viewed, or null when showing the live network. */
  historyIndex: number | null;
  /** Snapshot pinned as A for the diff, or null. */
  pinnedIndex: number | null;
  /** Δw against the pinned snapshot, or null when not diffing. */
  diffBase: Float64Array | null;

  scrubHistory: (index: number | null) => void;
  pinSnapshot: (index: number | null) => void;

  /* ---- lessons (§7) ---- */
  /** Id of the lesson currently loaded, or null. */
  activeLessonId: string | null;
  /** Ids whose success predicate has been satisfied at least once. */
  completedLessons: readonly string[];
  applyLesson: (id: string, preset: LessonPreset) => void;
  markLessonComplete: (id: string) => void;
  clearLesson: () => void;

  /* ---- editing (§6.5) ---- */
  /** Bumped on any parameter edit, so views that read weights can repaint. */
  editRevision: number;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel: string | null;
  /** True while a weight is being dragged, so the boundary can go synchronous. */
  scrubbing: boolean;
  /**
   * The canvas's smoothed weight reference, mirrored here so scrub sensitivity
   * can scale with it: one screen-width of travel should span a comparable
   * range whether weights sit near 0.5 or near 50.
   */
  wRefDisplay: number;
  setWRefDisplay: (value: number) => void;

  /** Flat index of a parameter, from a layer and its position in W or b. */
  parameterIndex: (layer: number, kind: 'W' | 'b' | 'gamma', row: number, col: number) => number;
  setParameter: (index: number, value: number, label?: string) => void;
  beginScrub: (index: number) => void;
  updateScrub: (index: number, value: number) => void;
  endScrub: () => void;
  toggleFrozen: (layer: number) => void;
  toggleAblated: (layer: number) => void;
  randomizeLayer: (layer: number) => void;
  undo: () => void;
  redo: () => void;

  /* ---- persistence (§6.5) ---- */
  exportJson: () => string;
  importJson: (json: string) => string | null;
  saveLocal: (name: string) => void;
  loadLocal: (name: string) => string | null;
  listLocal: () => string[];
  deleteLocal: (name: string) => void;

  /** A full URL that reproduces the current state, given the page's own address. */
  shareUrlFor: (base: string) => string;
  /** Apply a decoded link. Returns an error sentence, or null on success. */
  applySharedState: (shared: SharedState) => string | null;

  setTraining: (next: Partial<TrainingSettings>) => void;
  startTraining: () => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stepEpoch: () => void;
  stopTraining: () => void;
  resetToInit: () => void;
  toggleLossLogScale: () => void;
  reseed: (seed: number) => void;
  rebuild: () => void;
}

export type TransportCommand = 'toggle' | 'step' | 'back' | 'restart';

/** Per-unit activation maps, packed end to end (§6.4). */
export interface ThumbnailData {
  readonly values: Float32Array;
  readonly slots: readonly { readonly layer: number; readonly unit: number; readonly offset: number }[];
  readonly resolution: number;
}

export type TrainingStatus = 'idle' | 'running' | 'paused' | 'done' | 'diverged' | 'error';

/**
 * Metrics kept for the charts.
 *
 * Capped because a long run at hundreds of epochs per second would otherwise
 * grow this array without bound. When the cap is hit the history is halved by
 * dropping every second entry, which preserves the SHAPE of the curve at lower
 * resolution rather than truncating its beginning, so the early steep descent
 * that carries most of the lesson stays visible.
 */
export const MAX_METRIC_HISTORY = 2000;

const IDLE_STATUS: TransportStatus = {
  label: 'Idle',
  beatIndex: 0,
  beatCount: 0,
  playing: false,
  complete: false,
  speed: 1,
};

export const DEFAULT_ARCHITECTURE: Architecture = {
  inputSize: 2,
  layers: [
    { units: 6, activation: 'tanh' },
    { units: 4, activation: 'tanh' },
    { units: 1, activation: 'sigmoid' },
  ],
  loss: 'bce',
  seed: 7,
  init: { kind: 'glorot_uniform' },
  l2: 0,
};

export const DEFAULT_DATASET: DatasetOptions = {
  name: 'moons',
  samples: 240,
  noise: 0.12,
  seed: 1,
  validationFraction: 0.2,
};

interface Derived {
  network: Network;
  dataset: Dataset;
  layout: Layout;
  hitIndex: HitIndex;
  configErrors: readonly string[];
}

/**
 * Rebuild everything downstream of the configuration in one step.
 *
 * Network, layout and hit index are derived together and replaced together.
 * Updating them independently is how a hit index ends up describing a layout
 * that no longer exists, which presents as clicks selecting the wrong thing.
 */
function derive(
  architecture: Architecture,
  datasetOptions: DatasetOptions,
  previous?: Derived,
): Derived {
  const config = toNetworkConfig(architecture);
  const errors = validateConfig(config);
  if (errors.length > 0 && previous !== undefined) {
    // Keep the last good network on screen and report the problem, rather than
    // blanking the canvas because someone is mid-edit (§9: errors give direction).
    return { ...previous, configErrors: errors };
  }
  if (errors.length > 0) {
    throw new Error(`Invalid initial configuration:\n  - ${errors.join('\n  - ')}`);
  }

  const network = new Network(config);
  const dataset = generateDataset(datasetOptions);
  const layout = computeLayout({
    sizes: [architecture.inputSize, ...architecture.layers.map((l) => l.units)],
    showBiases: true,
  });
  return { network, dataset, layout, hitIndex: new HitIndex(layout), configErrors: [] };
}

const initial = derive(DEFAULT_ARCHITECTURE, DEFAULT_DATASET);

export const useAppStore = create<AppState>((set, get) => ({
  architecture: DEFAULT_ARCHITECTURE,
  datasetOptions: DEFAULT_DATASET,
  ...initial,
  revision: 0,
  lastReconciliation: [],

  viewport: IDENTITY_VIEWPORT,
  hover: null,
  selection: null,
  colorblindSafe: false,
  showBiases: true,
  showThumbnails: false,
  thumbnails: null,

  dissectionEnabled: false,
  sampleIndex: 0,
  speed: 1,
  learningRate: 0.1,
  transportStatus: IDLE_STATUS,
  pendingCommand: null,
  transportNonce: 0,

  editRevision: 0,
  canUndo: false,
  canRedo: false,
  undoLabel: null,
  scrubbing: false,
  wRefDisplay: 1,

  training: DEFAULT_TRAINING,
  trainingStatus: 'idle',
  trainingError: null,
  epoch: 0,
  metrics: [],
  latest: null,
  trainingOnMainThread: false,
  lossLogScale: false,

  history: [],
  historyIndex: null,
  pinnedIndex: null,
  diffBase: null,

  activeLessonId: null,
  completedLessons: [],

  setArchitecture: (next) => {
    const state = get();
    const architecture = { ...state.architecture, ...next };
    const derived = derive(architecture, state.datasetOptions, state);
    // A structural change makes every recorded parameter index meaningless, so
    // the stack is cleared rather than left to apply indices into a different
    // network.
    commands.clear();
    set({
      architecture,
      ...derived,
      lastReconciliation: [],
      canUndo: false,
      canRedo: false,
      undoLabel: null,
      revision: state.revision + 1,
      // A changed architecture invalidates any selection: the indices refer to
      // a layout that no longer exists.
      hover: null,
      selection: derived.configErrors.length > 0 ? state.selection : null,
    });
  },

  /*
   * Dataset and network are updated in ONE set(), because a 35-feature dataset
   * paired with a 2-input network is not a state the rest of the app can
   * survive: the very next forward pass throws. Updating them in two steps left
   * that mismatch observable by subscribers in between.
   */
  setDatasetOptions: (next) => {
    const state = get();
    const datasetOptions = { ...state.datasetOptions, ...next };
    const dataset = generateDataset(datasetOptions);
    const { architecture, changes } = reconcileArchitecture(state.architecture, dataset);
    const derived = derive(architecture, datasetOptions, state);
    set({
      datasetOptions,
      architecture,
      ...derived,
      dataset,
      lastReconciliation: changes,
      revision: state.revision + 1,
      hover: null,
      selection: null,
    });
  },

  setViewport: (viewport) => set({ viewport }),
  setHover: (hover) => set({ hover }),
  setSelection: (selection) => set({ selection }),
  toggleColorblindSafe: () => set((s) => ({ colorblindSafe: !s.colorblindSafe })),

  toggleThumbnails: () =>
    set((s) => ({ showThumbnails: !s.showThumbnails, thumbnails: s.showThumbnails ? null : s.thumbnails })),

  setThumbnails: (thumbnails) => set({ thumbnails }),

  toggleDissection: () =>
    set((s) => ({
      dissectionEnabled: !s.dissectionEnabled,
      transportStatus: s.dissectionEnabled ? IDLE_STATUS : s.transportStatus,
    })),

  setSampleIndex: (index) => {
    const state = get();
    const count = state.dataset.x.rows;
    if (count === 0) return;
    // Wraps, so stepping past the end returns to the first sample rather than
    // silently doing nothing at a boundary.
    set({ sampleIndex: ((index % count) + count) % count });
  },

  setSpeed: (speed) =>
    set({ speed: Math.max(SPEED_MIN, Math.min(SPEED_MAX, speed)) }),

  setLearningRate: (rate) =>
    set({ learningRate: Number.isFinite(rate) && rate > 0 ? rate : 0.1 }),

  transport: (command) =>
    set((s) => ({ pendingCommand: command, transportNonce: s.transportNonce + 1 })),

  setTransportStatus: (transportStatus) => set({ transportStatus }),

  setWRefDisplay: (value) => {
    // Only written when it moves meaningfully, so a smoothly drifting reference
    // does not re-render React every frame.
    if (Math.abs(get().wRefDisplay - value) > 0.02) set({ wRefDisplay: value });
  },

  parameterIndex: (layer, kind, row, col) => {
    const state = get();
    let offset = 0;
    for (let i = 0; i < layer; i++) {
      const l = state.network.layers[i];
      if (l === undefined) return -1;
      offset += l.parameterCount;
    }
    const target = state.network.layers[layer];
    if (target === undefined) return -1;
    // Layout is W, then b, then γ within each layer (see ARCHITECTURE.md).
    const weights = target.inputs * target.units;
    if (kind === 'W') return offset + row * target.units + col;
    if (kind === 'b') return offset + weights + col;
    return target.batchNorm ? offset + weights + target.units + col : -1;
  },

  setParameter: (index, value, label = 'Edit parameter') => {
    const state = get();
    if (index < 0 || index >= state.network.params.length || !Number.isFinite(value)) return;
    const before = state.network.params[index] as number;
    if (before === value) return;
    commands.execute(
      setParameterCommand(label, state.network.params, index, before, value, notifyEdit),
    );
    syncUndoState();
  },

  /*
   * A drag is ONE undo entry, not one per pointer move.
   *
   * beginScrub records the value at mousedown; updateScrub mutates live so the
   * boundary tracks the hand; endScrub pushes a single command. Pushing per
   * move would bury the previous edit under a hundred entries.
   */
  beginScrub: (index) => {
    const state = get();
    scrubOrigin = { index, value: state.network.params[index] as number };
    set({ scrubbing: true });
  },

  updateScrub: (index, value) => {
    const state = get();
    if (scrubOrigin === null || scrubOrigin.index !== index || !Number.isFinite(value)) return;
    state.network.params[index] = value;
    set({ editRevision: state.editRevision + 1 });
  },

  endScrub: () => {
    const state = get();
    set({ scrubbing: false });
    if (scrubOrigin === null) return;
    const { index, value } = scrubOrigin;
    scrubOrigin = null;
    const after = state.network.params[index] as number;
    if (after === value) return;
    // Recorded rather than executed: the effect already happened live.
    commands.record(
      setParameterCommand('Scrub weight', state.network.params, index, value, after, notifyEdit),
    );
    syncUndoState();
  },

  toggleFrozen: (layer) => {
    const target = get().network.layers[layer];
    if (target === undefined) return;
    commands.execute(
      setFlagCommand(
        target.frozen ? 'Unfreeze layer' : 'Freeze layer',
        () => target.frozen,
        (v) => {
          target.frozen = v;
        },
        !target.frozen,
        notifyEdit,
      ),
    );
    syncUndoState();
  },

  toggleAblated: (layer) => {
    const target = get().network.layers[layer];
    if (target === undefined) return;
    commands.execute(
      setFlagCommand(
        target.ablated ? 'Restore layer' : 'Ablate layer',
        () => target.ablated,
        (v) => {
          target.ablated = v;
        },
        !target.ablated,
        notifyEdit,
      ),
    );
    syncUndoState();
  },

  /** Re-randomize one layer, leaving the rest of the network alone. */
  randomizeLayer: (layer) => {
    const state = get();
    const target = state.network.layers[layer];
    if (target === undefined) return;
    let offset = 0;
    for (let i = 0; i < layer; i++) offset += (state.network.layers[i]?.parameterCount ?? 0);
    const size = target.parameterCount;
    const before = Float64Array.from(state.network.params.subarray(offset, offset + size));

    const rng = state.network.rng.stream('init');
    const fanIn = target.inputs;
    const limit = Math.sqrt(6 / (fanIn + target.units));
    const after = new Float64Array(size);
    for (let i = 0; i < fanIn * target.units; i++) after[i] = rng.uniform(-limit, limit);

    const apply = (values: Float64Array): void => {
      state.network.params.set(values, offset);
      notifyEdit();
    };
    commands.execute({
      label: `Randomize layer ${layer + 1}`,
      apply: () => apply(after),
      revert: () => apply(before),
    });
    syncUndoState();
  },

  undo: () => {
    commands.undo();
    syncUndoState();
  },

  redo: () => {
    commands.redo();
    syncUndoState();
  },

  exportJson: () => JSON.stringify(get().network.serialize(), null, 2),

  importJson: (json) => {
    try {
      const data = JSON.parse(json) as SerializedNetwork;
      const network = Network.deserialize(data);
      const architecture: Architecture = {
        inputSize: network.inputSize,
        layers: network.layers.map((l) => l.spec()),
        loss: network.lossName,
        seed: network.seed,
        init: network.initScheme,
        l2: network.l2,
      };
      const state = get();
      const derived = derive(architecture, state.datasetOptions, state);
      derived.network.restoreState(network.captureState());
      commands.clear();
      set({
        architecture,
        ...derived,
        revision: state.revision + 1,
        editRevision: state.editRevision + 1,
        hover: null,
        selection: null,
        lastReconciliation: [],
      });
      syncUndoState();
      return null;
    } catch (error) {
      // Returned rather than thrown: a bad paste is a user error, not a crash.
      return error instanceof Error ? error.message : String(error);
    }
  },

  saveLocal: (name) => {
    try {
      const key = `${STORAGE_PREFIX}${name}`;
      window.localStorage.setItem(key, get().exportJson());
    } catch {
      /* Storage can be full or blocked; saving is best-effort. */
    }
  },

  loadLocal: (name) => {
    try {
      const json = window.localStorage.getItem(`${STORAGE_PREFIX}${name}`);
      if (json === null) return `No saved network called "${name}".`;
      return get().importJson(json);
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  },

  listLocal: () => {
    try {
      const names: string[] = [];
      for (let i = 0; i < window.localStorage.length; i++) {
        const key = window.localStorage.key(i);
        if (key !== null && key.startsWith(STORAGE_PREFIX)) names.push(key.slice(STORAGE_PREFIX.length));
      }
      return names.sort();
    } catch {
      return [];
    }
  },

  deleteLocal: (name) => {
    try {
      window.localStorage.removeItem(`${STORAGE_PREFIX}${name}`);
    } catch {
      /* best-effort */
    }
  },

  setTraining: (next) => {
    const state = get();
    // Changing a hyperparameter mid-run would produce a loss curve whose
    // segments were generated under different settings, which is unreadable.
    // Stop first; the learner restarts deliberately.
    if (state.trainingStatus === 'running' || state.trainingStatus === 'paused') {
      state.stopTraining();
    }
    set({ training: { ...state.training, ...next } });
  },

  startTraining: () => {
    const state = get();
    if (state.configErrors.length > 0) return;
    const controller = ensureController();
    set({
      trainingStatus: 'running',
      trainingError: null,
      epoch: 0,
      metrics: [],
      latest: null,
    });
    history.clear();
    liveState = null;
    set({ history: [], historyIndex: null, pinnedIndex: null, diffBase: null });
    controller.start(
      toTrainerConfig(state),
      state.datasetOptions,
      state.training.maxEpochs,
      reportEveryFor(state.training.maxEpochs),
    );
    set({ trainingOnMainThread: controller.isFallback });
  },

  pauseTraining: () => {
    controller?.pause();
    set({ trainingStatus: 'paused' });
  },

  resumeTraining: () => {
    controller?.resume();
    set({ trainingStatus: 'running' });
  },

  stepEpoch: () => {
    const state = get();
    if (state.trainingStatus === 'idle' || state.trainingStatus === 'done') {
      state.startTraining();
      // A fresh run starts playing; pause it so the step is a single epoch.
      controller?.pause();
      set({ trainingStatus: 'paused' });
    }
    controller?.stepEpoch();
  },

  stopTraining: () => {
    controller?.stop();
    set({ trainingStatus: 'idle' });
  },

  toggleLossLogScale: () => set((s) => ({ lossLogScale: !s.lossLogScale })),

  /**
   * Show a point in history, or return to the live network.
   *
   * The live parameters are stashed on the way in and restored on the way out,
   * so scrubbing is a VIEW rather than an edit: letting go of the scrubber must
   * not leave the network somewhere in its own past.
   */
  scrubHistory: (index) => {
    const state = get();
    if (index === null) {
      if (liveState !== null) {
        state.network.restoreState(liveState);
        liveState = null;
      }
      set({ historyIndex: null, editRevision: state.editRevision + 1 });
      return;
    }
    const snapshot = state.history[index];
    if (snapshot === undefined) return;
    if (liveState === null) liveState = state.network.captureState();
    state.network.restoreState(snapshot);
    set({ historyIndex: index, editRevision: state.editRevision + 1 });
  },

  /**
   * Load a lesson's preset: architecture, dataset and hyperparameters at once.
   *
   * Applied as ONE update so the reconciliation that normally follows a dataset
   * change cannot fight the architecture the lesson intends. A lesson that said
   * "eight sigmoid layers" and got its output layer rewritten underneath it
   * would be demonstrating something else.
   */
  applyLesson: (id, preset) => {
    const state = get();
    state.stopTraining();
    commands.clear();
    history.clear();
    liveState = null;

    const dataset = generateDataset(preset.dataset);
    const architecture: Architecture = { ...preset.architecture };
    const derived = derive(architecture, preset.dataset, state);
    set({
      activeLessonId: id,
      architecture,
      datasetOptions: preset.dataset,
      ...derived,
      dataset,
      training: { ...preset.training },
      revision: state.revision + 1,
      editRevision: state.editRevision + 1,
      epoch: 0,
      metrics: [],
      latest: null,
      history: [],
      historyIndex: null,
      pinnedIndex: null,
      diffBase: null,
      trainingStatus: 'idle',
      trainingError: null,
      lastReconciliation: [],
      hover: null,
      selection: null,
      canUndo: false,
      canRedo: false,
      undoLabel: null,
    });
  },

  /**
   * A URL that reproduces this state.
   *
   * The parameters handed over are the LIVE ones, not the ones a history scrub
   * is currently showing: `liveState` holds them whenever the scrubber has
   * temporarily written an old snapshot into the network. Sharing what the
   * scrubber happens to be displaying would hand over a network the sender does
   * not have.
   */
  shareUrlFor: (base) => {
    const state = get();
    return buildShareUrl(base, {
      architecture: state.architecture,
      datasetOptions: state.datasetOptions,
      training: state.training,
      lessonId: state.activeLessonId,
      parameters: liveState?.parameters ?? state.network.captureParameters(),
      buffers: liveState?.buffers ?? state.network.captureBuffers(),
    });
  },

  /**
   * Adopt a decoded link, wholesale.
   *
   * Deliberately the same shape as applyLesson: one `set`, everything derived
   * together, every trace of the previous session cleared. A link that restored
   * an architecture but left the old loss curve and undo stack in place would
   * be showing numbers from a network that no longer exists.
   *
   * The architecture is applied as sent, without dataset reconciliation. The
   * sender's pairing was valid when they made the link, and reconciling would
   * quietly rewrite the very thing they meant to show, exactly as it would for
   * a lesson preset.
   */
  applySharedState: (shared) => {
    const state = get();
    const errors = validateConfig(toNetworkConfig(shared.architecture));
    if (errors.length > 0) {
      return `This link describes a network that cannot be built: ${errors.join('; ')}`;
    }

    state.stopTraining();
    commands.clear();
    history.clear();
    liveState = null;

    let dataset: Dataset;
    let derived: Derived;
    try {
      dataset = generateDataset(shared.datasetOptions);
      derived = derive(shared.architecture, shared.datasetOptions, state);
    } catch (error) {
      return `This link's dataset could not be generated: ${error instanceof Error ? error.message : String(error)}`;
    }

    // Absent parameters are a claim that the seed reproduces them, and `derive`
    // has just done exactly that by constructing the network. The statistics
    // always arrive, since the decoder fills in the fresh ones when the link
    // does not carry any.
    try {
      if (shared.parameters !== null) derived.network.restoreParameters(shared.parameters);
      derived.network.restoreBuffers(shared.buffers);
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }

    set({
      activeLessonId: shared.lessonId,
      architecture: shared.architecture,
      datasetOptions: shared.datasetOptions,
      ...derived,
      dataset,
      training: shared.training,
      revision: state.revision + 1,
      editRevision: state.editRevision + 1,
      epoch: 0,
      metrics: [],
      latest: null,
      history: [],
      historyIndex: null,
      pinnedIndex: null,
      diffBase: null,
      trainingStatus: 'idle',
      trainingError: null,
      lastReconciliation: [],
      hover: null,
      selection: null,
      canUndo: false,
      canRedo: false,
      undoLabel: null,
    });
    return null;
  },

  markLessonComplete: (id) =>
    set((s) =>
      s.completedLessons.includes(id) ? {} : { completedLessons: [...s.completedLessons, id] },
    ),

  clearLesson: () => set({ activeLessonId: null }),

  pinSnapshot: (index) => {
    const state = get();
    if (index === null) {
      set({ pinnedIndex: null, diffBase: null, editRevision: state.editRevision + 1 });
      return;
    }
    const snapshot = state.history[index];
    if (snapshot === undefined) return;
    // Guard against a pin taken before an architecture change.
    if (snapshot.parameters.length !== state.network.params.length) return;
    set({
      pinnedIndex: index,
      diffBase: Float64Array.from(snapshot.parameters),
      editRevision: state.editRevision + 1,
    });
  },

  resetToInit: () => {
    const state = get();
    state.stopTraining();
    // Same seed, same initial parameters (§6.3 transport: reset-to-init).
    state.network.resetToInit();
    history.clear();
    liveState = null;
    set({
      epoch: 0,
      metrics: [],
      latest: null,
      history: [],
      historyIndex: null,
      pinnedIndex: null,
      diffBase: null,
      trainingStatus: 'idle',
      trainingError: null,
      revision: state.revision + 1,
    });
  },

  toggleBiases: () => {
    const state = get();
    const showBiases = !state.showBiases;
    const layout = computeLayout({
      sizes: [state.architecture.inputSize, ...state.architecture.layers.map((l) => l.units)],
      showBiases,
    });
    set({
      showBiases,
      layout,
      hitIndex: new HitIndex(layout),
      revision: state.revision + 1,
      hover: null,
      selection: null,
    });
  },

  reseed: (seed) => get().setArchitecture({ seed }),

  rebuild: () => {
    const state = get();
    set({
      ...derive(state.architecture, state.datasetOptions, state),
      revision: state.revision + 1,
      hover: null,
      selection: null,
    });
  },
}));


/* ------------------------------------------------------------------ *
 * Training controller wiring
 *
 * Created lazily and kept module-level: a worker is a side effect, and the
 * store should hold results rather than threads.
 * ------------------------------------------------------------------ */

export const STORAGE_PREFIX = 'awrynn:network:';

/** Shared across the store: one stack, so undo covers every kind of edit. */
const commands = new CommandStack();

/** Training snapshots for the scrubber (§6.6). */
const history = new HistoryBuffer();

/**
 * The live network, stashed while a past snapshot is being viewed.
 *
 * Both halves, not just the weights: eval-mode rendering reads the running
 * statistics, so releasing a scrub without putting them back would leave the
 * canvas drawing today's weights through an old epoch's statistics.
 */
let liveState: NetworkState | null = null;

/** The value a drag started from, so the whole gesture is one undo entry. */
let scrubOrigin: { index: number; value: number } | null = null;

function notifyEdit(): void {
  useAppStore.setState((s) => ({ editRevision: s.editRevision + 1 }));
}

function syncUndoState(): void {
  useAppStore.setState({
    canUndo: commands.canUndo,
    canRedo: commands.canRedo,
    undoLabel: commands.undoLabel,
  });
}

let controller: TrainingController | null = null;

function ensureController(): TrainingController {
  if (controller !== null) return controller;
  controller = new TrainingController({
    onReady: () => {
      /* The dataset summary is already in the store; nothing to reconcile. */
    },
    onProgress: (metrics, networkState, epoch) => {
      const state = useAppStore.getState();
      // The worker owns its own Network; this is how the main thread's copy,
      // which is what the canvas draws, learns what the worker computed.
      state.network.restoreState(networkState);

      const last = metrics[metrics.length - 1];
      if (last !== undefined) {
        // One snapshot per progress message. The worker already batches these,
        // so the cadence follows reportEvery and a long run stays inside the
        // ring rather than filling it in the first second.
        history.push({
          epoch,
          parameters: networkState.parameters,
          buffers: networkState.buffers,
          trainLoss: last.trainLoss,
          validationLoss: last.validationLoss,
          validationAccuracy: last.validationAccuracy,
        });
      }

      useAppStore.setState({
        epoch,
        metrics: appendMetrics(state.metrics, metrics),
        latest: last ?? state.latest,
        history: history.toArray(),
        // Training moves the live network on, so a paused scrub is released.
        historyIndex: null,
      });
    },
    onDone: (status, epoch, networkState) => {
      const state = useAppStore.getState();
      state.network.restoreState(networkState);
      useAppStore.setState({
        epoch,
        trainingStatus: status === 'diverged' ? 'diverged' : 'done',
      });
    },
    onError: (message) => {
      useAppStore.setState({ trainingStatus: 'error', trainingError: message });
    },
  });
  return controller;
}

function toTrainerConfig(state: AppState): TrainerConfig {
  return {
    network: toNetworkConfig(state.architecture),
    optimizer: state.training.optimizer,
    learningRate: state.training.learningRate,
    batchSize: state.training.batchSize,
    validationFraction: state.datasetOptions.validationFraction ?? 0.2,
    dropout: state.training.dropout,
    gradientClip: state.training.gradientClip,
    standardize: state.training.standardize,
  };
}

/**
 * How many epochs the worker runs between progress messages.
 *
 * A short run reports every epoch so the curve is smooth; a long one batches,
 * because 500 postMessages a second would flood the main thread with work the
 * chart cannot show anyway.
 */
function reportEveryFor(maxEpochs: number): number {
  return maxEpochs <= 200 ? 1 : Math.ceil(maxEpochs / 200);
}

function appendMetrics(
  existing: readonly EpochMetrics[],
  incoming: readonly EpochMetrics[],
): EpochMetrics[] {
  const merged = existing.concat(incoming);
  if (merged.length <= MAX_METRIC_HISTORY) return merged;
  // Halve by dropping every second entry, preserving the curve's shape at lower
  // resolution rather than losing its start.
  const thinned: EpochMetrics[] = [];
  for (let i = 0; i < merged.length; i += 2) thinned.push(merged[i] as EpochMetrics);
  return thinned;
}


/** Δw against the pinned snapshot, for the A/B diff (§6.6). */
export function currentDelta(): Float64Array | null {
  const state = useAppStore.getState();
  if (state.diffBase === null) return null;
  if (state.diffBase.length !== state.network.params.length) return null;
  return parameterDelta(state.network.params, state.diffBase);
}
