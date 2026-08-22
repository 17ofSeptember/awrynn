import { describe, expect, it } from 'vitest';
import {
  decodeShareLink,
  encodeShareLink,
  expectedParameterCount,
  fromBase64Url,
  LONG_LINK_THRESHOLD,
  parametersAreFresh,
  shareUrl,
  SHARE_FORMAT_VERSION,
  toBase64Url,
} from '../shareLink';
import type { SharedState } from '../shareLink';
import type { Architecture, TrainingSettings } from '../architecture';
import { DEFAULT_TRAINING, toNetworkConfig } from '../architecture';
import { Network } from '../../engine/network';
import type { DatasetOptions } from '../../engine/datasets/index';

const ARCHITECTURE: Architecture = {
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

const DATASET: DatasetOptions = {
  name: 'moons',
  samples: 240,
  noise: 0.12,
  seed: 1,
  validationFraction: 0.2,
};

function baseState(overrides: Partial<SharedState> = {}): SharedState {
  return {
    architecture: ARCHITECTURE,
    datasetOptions: DATASET,
    training: DEFAULT_TRAINING,
    lessonId: null,
    parameters: null,
    buffers: new Float64Array(0),
    ...overrides,
  };
}

function decoded(state: SharedState): SharedState {
  const result = decodeShareLink(encodeShareLink(state));
  if (!result.ok) throw new Error(`expected a valid link, got: ${result.error}`);
  return result.state;
}

describe('base64url', () => {
  it('round-trips every byte value', () => {
    const bytes = new Uint8Array(256);
    for (let i = 0; i < 256; i++) bytes[i] = i;
    expect(Array.from(fromBase64Url(toBase64Url(bytes)))).toEqual(Array.from(bytes));
  });

  it('round-trips every length remainder', () => {
    // 3-byte groups encode cleanly; the interesting cases are the tails.
    for (let length = 0; length <= 8; length++) {
      const bytes = new Uint8Array(length);
      for (let i = 0; i < length; i++) bytes[i] = (i * 37 + 11) & 0xff;
      const text = toBase64Url(bytes);
      expect(text.length % 4, `length ${length} produced an impossible tail`).not.toBe(1);
      expect(Array.from(fromBase64Url(text)), `length ${length}`).toEqual(Array.from(bytes));
    }
  });

  it('emits no padding and no URL-unsafe characters', () => {
    const bytes = new Uint8Array([251, 255, 190, 0, 1, 2, 3]);
    const text = toBase64Url(bytes);
    expect(text).not.toContain('=');
    expect(text).toMatch(/^[A-Za-z0-9_-]*$/);
  });

  it('rejects characters outside the alphabet', () => {
    expect(() => fromBase64Url('ab+d')).toThrow(/unexpected/);
    expect(() => fromBase64Url('ab/d')).toThrow(/unexpected/);
    // Above the reverse table, which is only 128 entries wide.
    expect(() => fromBase64Url('abcé')).toThrow(/unexpected/);
  });

  it('rejects a length that no byte sequence could have produced', () => {
    expect(() => fromBase64Url('AAAAA')).toThrow(/truncated/);
  });
});

describe('encodeShareLink', () => {
  it('round-trips architecture, dataset, training and lesson', () => {
    const state = baseState({ lessonId: 'vanishing-gradients' });
    const back = decoded(state);
    expect(back.architecture).toEqual(ARCHITECTURE);
    expect(back.datasetOptions).toEqual(DATASET);
    expect(back.training).toEqual(DEFAULT_TRAINING);
    expect(back.lessonId).toBe('vanishing-gradients');
  });

  it('omits the parameter block when the seed reproduces the parameters', () => {
    const fresh = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
    const fragment = encodeShareLink(baseState({ parameters: fresh }));
    expect(fragment).not.toContain('&p=');
    // And the receiving side rebuilds them from the seed alone.
    expect(decoded(baseState({ parameters: fresh })).parameters).toBeNull();
  });

  it('includes the parameter block once a single weight differs', () => {
    const edited = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
    edited[3] = (edited[3] as number) + 1e-12;
    const fragment = encodeShareLink(baseState({ parameters: edited }));
    expect(fragment).toContain('&p=');
  });

  it('carries parameters at full float64 precision', () => {
    const network = new Network(toNetworkConfig(ARCHITECTURE));
    const params = network.captureParameters();
    // Values chosen to be unrepresentable in float32, so any narrowing shows.
    params[0] = 0.1 + 0.2;
    params[1] = Math.PI;
    params[2] = -1 / 3;
    params[4] = Number.MIN_SAFE_INTEGER / 3;
    const back = decoded(baseState({ parameters: params }));
    expect(back.parameters).not.toBeNull();
    const bytesOut = new Uint8Array(params.buffer);
    const bytesIn = new Uint8Array((back.parameters as Float64Array).buffer);
    expect(Array.from(bytesIn)).toEqual(Array.from(bytesOut));
  });

  it('produces byte-identical parameters after a real training run', () => {
    // The property that matters: what arrives is what was sent, not a value
    // that merely prints the same.
    const network = new Network(toNetworkConfig(ARCHITECTURE));
    const params = network.captureParameters();
    for (let i = 0; i < params.length; i++) {
      params[i] = Math.sin(i * 12.9898) * 43758.5453;
    }
    const back = decoded(baseState({ parameters: params })).parameters as Float64Array;
    for (let i = 0; i < params.length; i++) {
      expect(Object.is(back[i], params[i]), `parameter ${i}`).toBe(true);
    }
  });

  it('keeps a setup link comfortably short', () => {
    const fragment = encodeShareLink(baseState());
    expect(fragment.length).toBeLessThan(LONG_LINK_THRESHOLD / 2);
  });

  it('replaces an existing fragment rather than appending to it', () => {
    const url = shareUrl('https://example.test/app#s=stale&p=stale', baseState());
    expect(url.indexOf('#')).toBe(url.lastIndexOf('#'));
    expect(url).not.toContain('stale');
    expect(url.startsWith('https://example.test/app#')).toBe(true);
  });
});

describe('expectedParameterCount', () => {
  it('agrees with the network it describes', () => {
    for (const architecture of [
      ARCHITECTURE,
      {
        ...ARCHITECTURE,
        inputSize: 14,
        layers: [{ units: 3, activation: 'softmax' as const }],
        loss: 'cce' as const,
      },
      {
        ...ARCHITECTURE,
        layers: [
          { units: 8, activation: 'relu' as const },
          { units: 8, activation: 'relu' as const },
          { units: 2, activation: 'softmax' as const },
        ],
        loss: 'cce' as const,
      },
    ]) {
      const network = new Network(toNetworkConfig(architecture));
      expect(expectedParameterCount(architecture)).toBe(network.parameterCount);
    }
  });
});

describe('parametersAreFresh', () => {
  it('is true for a newly built network and false after one change', () => {
    const params = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
    expect(parametersAreFresh(ARCHITECTURE, params)).toBe(true);
    params[0] = (params[0] as number) + 1e-15;
    expect(parametersAreFresh(ARCHITECTURE, params)).toBe(false);
  });

  it('is false when a different seed produced them', () => {
    const params = new Network(toNetworkConfig({ ...ARCHITECTURE, seed: 8 })).captureParameters();
    expect(parametersAreFresh(ARCHITECTURE, params)).toBe(false);
  });

  it('distinguishes -0 from 0, because the bytes differ', () => {
    const params = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
    // Biases initialize to +0; -0 compares equal with === but is a different
    // number, and a byte comparison is what catches it.
    const zeroAt = params.indexOf(0);
    expect(zeroAt).toBeGreaterThanOrEqual(0);
    params[zeroAt] = -0;
    expect(parametersAreFresh(ARCHITECTURE, params)).toBe(false);
  });
});

describe('decodeShareLink rejects bad input without throwing', () => {
  const reject = (fragment: string): string => {
    const result = decodeShareLink(fragment);
    expect(result.ok, `expected "${fragment.slice(0, 40)}" to be rejected`).toBe(false);
    return result.ok ? '' : result.error;
  };

  it('rejects an empty or structureless fragment', () => {
    expect(reject('')).toMatch(/no state/);
    expect(reject('#')).toMatch(/no state/);
    expect(reject('nonsense')).toMatch(/missing its state/);
  });

  it('rejects a payload that is not base64url', () => {
    expect(reject('s=@@@@')).toMatch(/Malformed base64url/);
  });

  it('reports a truncated payload in plain language, not the parser\'s', () => {
    const message = reject(`s=${toBase64Url(new TextEncoder().encode('{oops'))}`);
    expect(message).toMatch(/damaged, most likely truncated/);
    // The engine's own wording names a byte offset; it must not reach a reader.
    expect(message).not.toMatch(/JSON|token|position/i);
  });

  it('names the version when the format does not match', () => {
    const payload = toBase64Url(new TextEncoder().encode(JSON.stringify({ v: 99 })));
    expect(reject(`s=${payload}`)).toMatch(
      new RegExp(`format 99.*reads ${SHARE_FORMAT_VERSION}`),
    );
  });

  it('rejects unknown activations, losses, optimizers, datasets and init schemes', () => {
    const mutate = (fn: (payload: Record<string, unknown>) => void): string => {
      const payload = JSON.parse(
        new TextDecoder().decode(fromBase64Url(encodeShareLink(baseState()).slice(2))),
      ) as Record<string, unknown>;
      fn(payload);
      return reject(`s=${toBase64Url(new TextEncoder().encode(JSON.stringify(payload)))}`);
    };

    expect(
      mutate((p) => {
        (p['architecture'] as { layers: { activation: string }[] }).layers[0]!.activation = 'selu';
      }),
    ).toMatch(/Unknown layer 0 activation "selu"/);
    expect(mutate((p) => ((p['architecture'] as { loss: string }).loss = 'hinge'))).toMatch(
      /Unknown loss "hinge"/,
    );
    expect(
      mutate((p) => ((p['training'] as { optimizer: { name: string } }).optimizer.name = 'lion')),
    ).toMatch(/Unknown optimizer name "lion"/);
    expect(mutate((p) => ((p['dataset'] as { name: string }).name = 'mnist'))).toMatch(
      /Unknown dataset name "mnist"/,
    );
    expect(mutate((p) => ((p['architecture'] as { init: { kind: string } }).init.kind = 'magic'))).toMatch(
      /Unknown initialization scheme "magic"/,
    );
  });

  it('rejects sizes that would hang the tab before validateConfig saw them', () => {
    const mutate = (fn: (payload: Record<string, unknown>) => void): string => {
      const payload = JSON.parse(
        new TextDecoder().decode(fromBase64Url(encodeShareLink(baseState()).slice(2))),
      ) as Record<string, unknown>;
      fn(payload);
      return reject(`s=${toBase64Url(new TextEncoder().encode(JSON.stringify(payload)))}`);
    };

    expect(
      mutate((p) => {
        (p['architecture'] as { layers: { units: number }[] }).layers[0]!.units = 2 ** 31;
      }),
    ).toMatch(/whole number between 1 and 512/);
    expect(
      mutate((p) => {
        (p['architecture'] as { layers: unknown[] }).layers = new Array(64).fill({
          units: 2,
          activation: 'relu',
        });
      }),
    ).toMatch(/between 1 and 24 layers/);
    expect(mutate((p) => ((p['training'] as { batchSize: number }).batchSize = 0))).toMatch(
      /batchSize/,
    );
    expect(
      mutate((p) => ((p['dataset'] as { validationFraction: number }).validationFraction = 1)),
    ).toMatch(/Validation fraction/);
  });

  it('rejects a parameter block of the wrong size', () => {
    const network = new Network(toNetworkConfig(ARCHITECTURE));
    const params = network.captureParameters();
    params[0] = 1;
    const fragment = encodeShareLink(baseState({ parameters: params }));
    // Drop the final float, leaving a well-formed block of the wrong length.
    const [head, tail] = fragment.split('&p=') as [string, string];
    const bytes = fromBase64Url(tail).slice(0, -8);
    expect(reject(`${head}&p=${toBase64Url(bytes)}`)).toMatch(
      new RegExp(`carries ${network.parameterCount - 1} parameters but its architecture needs ${network.parameterCount}`),
    );
  });

  it('rejects a parameter block that is not a whole number of floats', () => {
    const params = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
    params[0] = 1;
    const [head, tail] = encodeShareLink(baseState({ parameters: params })).split('&p=') as [
      string,
      string,
    ];
    expect(reject(`${head}&p=${toBase64Url(fromBase64Url(tail).slice(0, -3))}`)).toMatch(
      /not a multiple of 8/,
    );
  });

  it('rejects NaN and Infinity in the parameter block', () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      const params = new Network(toNetworkConfig(ARCHITECTURE)).captureParameters();
      params[5] = bad;
      expect(reject(encodeShareLink(baseState({ parameters: params })))).toMatch(
        /not a finite number/,
      );
    }
  });

  it('rejects a training block with the wrong types', () => {
    const payload = JSON.parse(
      new TextDecoder().decode(fromBase64Url(encodeShareLink(baseState()).slice(2))),
    ) as { training: Record<string, unknown> };
    payload.training['standardize'] = 'yes';
    expect(
      reject(`s=${toBase64Url(new TextEncoder().encode(JSON.stringify(payload)))}`),
    ).toMatch(/standardize to be true or false/);
  });
});

describe('optional dataset and layer fields', () => {
  it('carries featureScale, classes and leakyAlpha when present', () => {
    const state = baseState({
      architecture: {
        ...ARCHITECTURE,
        layers: [
          { units: 6, activation: 'leaky_relu', leakyAlpha: 0.05 },
          { units: 3, activation: 'softmax' },
        ],
        loss: 'cce',
      },
      datasetOptions: { ...DATASET, name: 'spiral', classes: 3, featureScale: [1, 100] },
      training: { ...DEFAULT_TRAINING, optimizer: { name: 'adamw', weightDecay: 0.01 } },
    });
    const back = decoded(state);
    expect(back.architecture.layers[0]).toEqual({
      units: 6,
      activation: 'leaky_relu',
      leakyAlpha: 0.05,
    });
    expect(back.datasetOptions.classes).toBe(3);
    expect(back.datasetOptions.featureScale).toEqual([1, 100]);
    expect(back.training.optimizer).toEqual({ name: 'adamw', weightDecay: 0.01 });
  });

  it('does not invent fields the sender left out', () => {
    const sparse: DatasetOptions = { name: 'xor' };
    const back = decoded(baseState({ datasetOptions: sparse }));
    expect(back.datasetOptions).toEqual(sparse);
    expect(Object.keys(back.datasetOptions)).toEqual(['name']);
  });

  it('preserves a non-default training block exactly', () => {
    const training: TrainingSettings = {
      optimizer: { name: 'momentum', momentum: 0.85 },
      learningRate: 0.001,
      batchSize: 1,
      maxEpochs: 3,
      dropout: 0.35,
      gradientClip: 5,
      standardize: true,
    };
    expect(decoded(baseState({ training })).training).toEqual(training);
  });
});
