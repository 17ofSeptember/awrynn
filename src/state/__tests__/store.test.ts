import { beforeEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_ARCHITECTURE,
  DEFAULT_DATASET,
  DEFAULT_TRAINING,
  reconcileArchitecture,
  useAppStore,
} from '../store';
import { decodeShareLink } from '../shareLink';
import { generateDataset } from '../../engine/datasets/index';
import { NetworkView } from '../../render/networkView';

/*
 * The invariant these tests exist for, learned by running the app:
 *
 *   A dataset and a network that disagree about the input width is not a state
 *   the app can survive — the next forward pass throws. Selecting `glyphs`
 *   (35 features) while a 2-input network was live threw from inside a store
 *   subscriber, which aborted the very update that would have resized the
 *   network. The result was a dataset panel showing 35 features next to an
 *   unchanged 2-6-4-1 architecture.
 */

beforeEach(() => {
  useAppStore.setState({
    architecture: DEFAULT_ARCHITECTURE,
    datasetOptions: DEFAULT_DATASET,
  });
  useAppStore.getState().setDatasetOptions({ name: DEFAULT_DATASET.name });
});

describe('dataset and network never disagree', () => {
  it('resizes the input layer when the dataset changes width', () => {
    const store = useAppStore.getState();
    expect(store.architecture.inputSize).toBe(2);

    store.setDatasetOptions({ name: 'glyphs' });

    const next = useAppStore.getState();
    expect(next.dataset.featureCount).toBe(35);
    expect(next.architecture.inputSize).toBe(35);
    expect(next.network.inputSize).toBe(35);
  });

  it('holds the invariant across every dataset', () => {
    for (const name of ['xor', 'moons', 'spiral', 'glyphs', 'sine', 'circles'] as const) {
      useAppStore.getState().setDatasetOptions({ name });
      const state = useAppStore.getState();
      expect(state.network.inputSize, name).toBe(state.dataset.featureCount);
      expect(state.network.outputSize, name).toBe(state.dataset.y.cols);
      expect(state.network.lossName, name).toBe(state.dataset.suggestedLoss);
      // And the network can actually accept the data it is paired with.
      expect(() => state.network.forward(state.dataset.x, false), name).not.toThrow();
    }
  });

  it('keeps the layout and hit index in step with the network', () => {
    useAppStore.getState().setDatasetOptions({ name: 'glyphs' });
    const state = useAppStore.getState();
    const columns = [state.architecture.inputSize, ...state.architecture.layers.map((l) => l.units)];
    const unitNodes = state.layout.nodes.filter((n) => n.kind !== 'bias');
    expect(unitNodes.length).toBe(columns.reduce((a, b) => a + b, 0));
    // Picking the last node must resolve, which it cannot if the index is stale.
    const last = state.layout.nodes[state.layout.nodes.length - 1];
    expect(state.hitIndex.pick(last!.x, last!.y)?.kind).toBe('node');
  });

  it('reports what changed instead of silently resetting (§6.5)', () => {
    useAppStore.getState().setDatasetOptions({ name: 'glyphs' });
    const changes = useAppStore.getState().lastReconciliation;
    expect(changes.some((c) => /Input layer resized 2 → 35/.test(c))).toBe(true);
    expect(changes.some((c) => /hidden layers preserved/.test(c))).toBe(true);
  });

  it('preserves hidden layers across a dataset change', () => {
    const before = useAppStore.getState().architecture.layers.slice(0, -1);
    useAppStore.getState().setDatasetOptions({ name: 'glyphs' });
    const after = useAppStore.getState().architecture.layers.slice(0, -1);
    expect(after).toEqual(before);
  });

  it('clears the selection, whose indices refer to the old layout', () => {
    const state = useAppStore.getState();
    state.setSelection({ kind: 'node', index: 0, node: state.layout.nodes[0]!, distance: 0 });
    expect(useAppStore.getState().selection).not.toBeNull();
    useAppStore.getState().setDatasetOptions({ name: 'glyphs' });
    expect(useAppStore.getState().selection).toBeNull();
  });
});

describe('reconcileArchitecture', () => {
  it('switches to softmax and K units for a multi-class dataset', () => {
    const dataset = generateDataset({ name: 'spiral', samples: 90, seed: 1, classes: 3 });
    const { architecture } = reconcileArchitecture(DEFAULT_ARCHITECTURE, dataset);
    const output = architecture.layers[architecture.layers.length - 1];
    expect(architecture.loss).toBe('cce');
    expect(output?.activation).toBe('softmax');
    expect(output?.units).toBe(3);
  });

  it('switches to a single sigmoid unit for a binary dataset', () => {
    const dataset = generateDataset({ name: 'moons', samples: 40, seed: 1 });
    const { architecture } = reconcileArchitecture(DEFAULT_ARCHITECTURE, dataset);
    const output = architecture.layers[architecture.layers.length - 1];
    expect(architecture.loss).toBe('bce');
    expect(output?.activation).toBe('sigmoid');
    expect(output?.units).toBe(1);
  });

  it('switches to a linear output for regression', () => {
    const dataset = generateDataset({ name: 'sine', samples: 40, seed: 1 });
    const { architecture } = reconcileArchitecture(DEFAULT_ARCHITECTURE, dataset);
    const output = architecture.layers[architecture.layers.length - 1];
    expect(architecture.loss).toBe('mse');
    expect(output?.activation).toBe('linear');
    expect(architecture.inputSize).toBe(1);
  });

  it('is idempotent — reconciling twice changes nothing the second time', () => {
    const dataset = generateDataset({ name: 'glyphs', samples: 40, seed: 1, classes: 10 });
    const first = reconcileArchitecture(DEFAULT_ARCHITECTURE, dataset);
    const second = reconcileArchitecture(first.architecture, dataset);
    expect(second.architecture).toEqual(first.architecture);
    // Second pass reports only the preserved-layers note, no changes.
    expect(second.changes.filter((c) => /resized|switched/.test(c))).toEqual([]);
  });
});

describe('NetworkView refuses a mismatched sample rather than throwing', () => {
  it('returns false instead of letting forward() throw', () => {
    // The render layer runs inside store subscribers; an exception here would
    // abort whatever state update triggered it.
    const state = useAppStore.getState();
    const view = new NetworkView(state.network);
    const wrongWidth = generateDataset({ name: 'glyphs', samples: 4, seed: 1, classes: 10 });
    expect(state.network.inputSize).toBe(2);
    expect(() => view.captureSample(wrongWidth.x, 0)).not.toThrow();
    expect(view.captureSample(wrongWidth.x, 0)).toBe(false);
    // And accepts a matching one.
    expect(view.captureSample(state.dataset.x, 0)).toBe(true);
  });
});

/*
 * Share links, exercised through the store rather than the codec.
 *
 * shareLink.test.ts proves the encoding round-trips. What these prove is the
 * part a user actually experiences: that a link built from a live application
 * state rebuilds that state, and that a link which cannot be applied leaves the
 * app usable rather than half-updated.
 */
describe('share links through the store', () => {
  const HOME = 'https://awry.test/';

  it('rebuilds an edited network byte for byte', () => {
    const store = useAppStore.getState();
    store.setArchitecture({
      ...DEFAULT_ARCHITECTURE,
      seed: 21,
      layers: [
        { units: 5, activation: 'relu' },
        { units: 1, activation: 'sigmoid' },
      ],
    });
    // Stand in for training: parameters no seed will reproduce.
    const network = useAppStore.getState().network;
    const edited = network.captureParameters();
    for (let i = 0; i < edited.length; i++) edited[i] = Math.cos(i * 7.31) * 2.5;
    network.restoreParameters(edited);

    const url = useAppStore.getState().shareUrlFor(HOME);
    expect(url).toContain('&p=');

    // Wipe the app, then follow the link.
    useAppStore.getState().setArchitecture(DEFAULT_ARCHITECTURE);
    const decoded = decodeShareLink(url.slice(url.indexOf('#')));
    expect(decoded.ok).toBe(true);
    if (!decoded.ok) return;
    expect(useAppStore.getState().applySharedState(decoded.state)).toBeNull();

    const after = useAppStore.getState();
    expect(after.architecture.seed).toBe(21);
    expect(after.architecture.layers.map((l) => l.units)).toEqual([5, 1]);
    const restored = after.network.captureParameters();
    expect(Array.from(new Uint8Array(restored.buffer))).toEqual(
      Array.from(new Uint8Array(edited.buffer)),
    );
  });

  it('reproduces an untrained network from the seed alone, with no weights in the link', () => {
    const store = useAppStore.getState();
    store.setArchitecture({ ...DEFAULT_ARCHITECTURE, seed: 99 });
    const expected = useAppStore.getState().network.captureParameters();

    const url = useAppStore.getState().shareUrlFor(HOME);
    expect(url).not.toContain('&p=');

    useAppStore.getState().setArchitecture({ ...DEFAULT_ARCHITECTURE, seed: 1 });
    const decoded = decodeShareLink(url.slice(url.indexOf('#')));
    if (!decoded.ok) throw new Error(decoded.error);
    expect(decoded.state.parameters).toBeNull();
    expect(useAppStore.getState().applySharedState(decoded.state)).toBeNull();

    const restored = useAppStore.getState().network.captureParameters();
    expect(Array.from(new Uint8Array(restored.buffer))).toEqual(
      Array.from(new Uint8Array(expected.buffer)),
    );
  });

  it('carries the dataset and hyperparameters, not just the network', () => {
    const store = useAppStore.getState();
    store.setDatasetOptions({ name: 'spiral', samples: 300, noise: 0.05, classes: 3 });
    useAppStore.getState().setTraining({
      optimizer: { name: 'rmsprop' },
      learningRate: 0.007,
      batchSize: 8,
      maxEpochs: 42,
    });
    const url = useAppStore.getState().shareUrlFor(HOME);

    useAppStore.getState().setDatasetOptions({ name: 'moons' });
    useAppStore.getState().setTraining(DEFAULT_TRAINING);

    const decoded = decodeShareLink(url.slice(url.indexOf('#')));
    if (!decoded.ok) throw new Error(decoded.error);
    expect(useAppStore.getState().applySharedState(decoded.state)).toBeNull();

    const after = useAppStore.getState();
    expect(after.datasetOptions.name).toBe('spiral');
    expect(after.datasetOptions.classes).toBe(3);
    expect(after.dataset.x.rows).toBe(300);
    expect(after.training.optimizer.name).toBe('rmsprop');
    expect(after.training.learningRate).toBeCloseTo(0.007, 12);
    expect(after.training.maxEpochs).toBe(42);
  });

  it('clears the previous session rather than mixing it in', () => {
    const store = useAppStore.getState();
    store.setArchitecture({ ...DEFAULT_ARCHITECTURE, seed: 3 });
    useAppStore.setState({
      metrics: [
        {
          epoch: 1,
          trainLoss: 0.5,
          trainObjective: 0.5,
          validationLoss: 0.6,
          trainAccuracy: 0.5,
          validationAccuracy: 0.5,
          gradientNorm: 1,
          gradientNorms: [1],
          learningRate: 0.01,
          deadUnits: 0,
          saturation: 0,
          diverged: false,
        },
      ],
      epoch: 17,
    });

    const url = useAppStore.getState().shareUrlFor(HOME);
    const decoded = decodeShareLink(url.slice(url.indexOf('#')));
    if (!decoded.ok) throw new Error(decoded.error);
    useAppStore.getState().applySharedState(decoded.state);

    const after = useAppStore.getState();
    expect(after.metrics).toEqual([]);
    expect(after.epoch).toBe(0);
    expect(after.history).toEqual([]);
    expect(after.canUndo).toBe(false);
    expect(after.selection).toBeNull();
  });

  it('shares the live parameters, not the ones the history scrubber is showing', () => {
    // Scrubbing writes an old snapshot into the live network and remembers the
    // real ones. A link built during a scrub must carry what the sender has,
    // not the frame they happen to be looking at.
    const store = useAppStore.getState();
    store.setArchitecture(DEFAULT_ARCHITECTURE);
    const network = useAppStore.getState().network;

    const older = network.captureParameters();
    for (let i = 0; i < older.length; i++) older[i] = -1;
    useAppStore.setState({
      history: [
        {
          epoch: 0,
          parameters: older,
          buffers: new Float64Array(0),
          trainLoss: 1,
          validationLoss: 1,
          validationAccuracy: 0,
        },
      ],
    });

    const live = network.captureParameters();
    for (let i = 0; i < live.length; i++) live[i] = i / 10;
    network.restoreParameters(live);

    useAppStore.getState().scrubHistory(0);
    // The canvas is now showing the -1s...
    expect(useAppStore.getState().network.params[0]).toBe(-1);

    // ...but the link carries the live ones.
    const url = useAppStore.getState().shareUrlFor(HOME);
    const decoded = decodeShareLink(url.slice(url.indexOf('#')));
    if (!decoded.ok) throw new Error(decoded.error);
    expect(Array.from(decoded.state.parameters as Float64Array)).toEqual(Array.from(live));

    useAppStore.getState().scrubHistory(null);
  });

  it('reports a link whose architecture cannot be built, and changes nothing', () => {
    const store = useAppStore.getState();
    store.setArchitecture(DEFAULT_ARCHITECTURE);
    const before = useAppStore.getState().architecture;

    const error = useAppStore.getState().applySharedState({
      // Softmax with binary cross-entropy: rejected by validateConfig.
      architecture: { ...DEFAULT_ARCHITECTURE, layers: [{ units: 3, activation: 'softmax' }], loss: 'bce' },
      datasetOptions: DEFAULT_DATASET,
      training: DEFAULT_TRAINING,
      lessonId: null,
      parameters: null,
      buffers: new Float64Array(0),
    });

    expect(error).toMatch(/cannot be built/);
    expect(useAppStore.getState().architecture).toBe(before);
  });
});
