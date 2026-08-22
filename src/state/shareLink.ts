/*
 * Share links — the whole application state, encoded into a URL fragment.
 *
 * §2 forbids a backend, so there is nowhere to put a saved network and no id to
 * hand out. A URL is therefore not a convenience here, it is the only way one
 * person can give a network to another, and that makes it worth doing properly
 * rather than as a query string of loose numbers.
 *
 * Three decisions are load-bearing.
 *
 * THE FRAGMENT, NOT THE QUERY. Everything after `#` is stripped by the browser
 * before a request is sent and never reaches a server or an access log. Since
 * the app makes no network requests at all (§2), a query string would leak the
 * link's contents to the host serving the page for no benefit whatsoever.
 *
 * PARAMETERS ARE OMITTED WHEN THEY ARE REPRODUCIBLE. Initialization is seeded
 * and bitwise deterministic (§4.7), asserted over 100 steps by comparing raw
 * bytes. So a network that has not been trained or edited does not need its
 * weights transmitted: the seed and the architecture regenerate them exactly.
 * The encoder checks this rather than assuming it, comparing byte for byte
 * against a freshly constructed network, and includes the parameter block only
 * when they genuinely differ. A setup link is ~700 characters; a trained one
 * carries its weights and grows.
 *
 * PARAMETERS ARE EXACT WHEN PRESENT. Float64, little-endian, no quantization.
 * Halving the payload by shipping Float32 would mean the network you opened was
 * not the network that was sent, which is the one thing this project does not
 * do. A long URL is the honest cost.
 *
 * The payload is explicit and complete: no field is omitted because it happens
 * to equal a default. Default-omission would shorten the common link by a few
 * hundred characters and, the day a default changed, would silently redefine
 * every link ever made. `v` guards the format itself.
 */

import type { ActivationName } from '../engine/activations';
import { ACTIVATION_NAMES } from '../engine/activations';
import type { DatasetName, DatasetOptions } from '../engine/datasets/index';
import { DATASET_NAMES } from '../engine/datasets/index';
import type { InitScheme } from '../engine/init';
import type { LossName } from '../engine/losses';
import { LOSS_NAMES } from '../engine/losses';
import type { LayerSpec } from '../engine/layers';
import { Network } from '../engine/network';
import type { OptimizerConfig, OptimizerName } from '../engine/optimizers';
import { OPTIMIZER_NAMES } from '../engine/optimizers';
import type { Architecture, TrainingSettings } from './architecture';
import { toNetworkConfig } from './architecture';

/**
 * Bumped only if the payload shape changes incompatibly.
 *
 * 2 added batch normalization's running statistics. A version 1 reader given a
 * version 2 link would ignore the `r` block and rebuild a network with default
 * statistics: the weights would be right, nothing would look broken, and the
 * decision boundary would be wrong. That is exactly the kind of failure a
 * version number exists to turn into a sentence.
 */
export const SHARE_FORMAT_VERSION = 2;

/**
 * URLs longer than this are still opened correctly by every current browser,
 * but are truncated by some chat clients and mail gateways. Above it the UI
 * says so rather than handing over a link that silently arrives broken.
 */
export const LONG_LINK_THRESHOLD = 2000;

export interface SharedState {
  readonly architecture: Architecture;
  readonly datasetOptions: DatasetOptions;
  readonly training: TrainingSettings;
  /** Lesson the sender had open, if any. */
  readonly lessonId: string | null;
  /**
   * Flat parameters, or null to mean "whatever this seed initializes to".
   *
   * Null is not "no parameters"; it is a claim that the deterministic init
   * reproduces them, which `encodeShareLink` verifies before writing it.
   */
  readonly parameters: Float64Array | null;
  /**
   * Batch normalization's running statistics, empty without it.
   *
   * Carried whenever they are not all at their initial values, on the same
   * rule the parameters use. Omitting them would hand over a network whose
   * weights are exact and whose predictions are not.
   */
  readonly buffers: Float64Array;
}

/* ------------------------------------------------------------------ base64url */

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

/** Reverse table; -1 marks a character that is not in the alphabet. */
const REVERSE = ((): Int8Array => {
  const table = new Int8Array(128).fill(-1);
  for (let i = 0; i < ALPHABET.length; i++) table[ALPHABET.charCodeAt(i)] = i;
  return table;
})();

/**
 * Base64url without padding.
 *
 * Hand-rolled rather than routed through btoa, which speaks binary strings and
 * needs a latin1 dance to carry bytes above 0x7f, and rather than Buffer, which
 * does not exist in a browser. Thirty lines buys the same function everywhere.
 */
export function toBase64Url(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i] as number;
    const b1 = i + 1 < bytes.length ? (bytes[i + 1] as number) : 0;
    const b2 = i + 2 < bytes.length ? (bytes[i + 2] as number) : 0;
    out += ALPHABET[b0 >> 2] as string;
    out += ALPHABET[((b0 & 0b11) << 4) | (b1 >> 4)] as string;
    if (i + 1 < bytes.length) out += ALPHABET[((b1 & 0b1111) << 2) | (b2 >> 6)] as string;
    if (i + 2 < bytes.length) out += ALPHABET[b2 & 0b111111] as string;
  }
  return out;
}

export function fromBase64Url(text: string): Uint8Array {
  // A remainder of one character cannot have come from any byte sequence: 4
  // characters carry 3 bytes, 3 carry 2, 2 carry 1, and 1 carries none.
  if (text.length % 4 === 1) {
    throw new Error('Malformed base64url: truncated final group.');
  }
  const bytes = new Uint8Array(Math.floor((text.length * 3) / 4));
  let out = 0;
  for (let i = 0; i < text.length; i += 4) {
    const group = [0, 0, 0, 0];
    const have = Math.min(4, text.length - i);
    for (let j = 0; j < have; j++) {
      const code = text.charCodeAt(i + j);
      const value = code < 128 ? (REVERSE[code] as number) : -1;
      if (value < 0) throw new Error(`Malformed base64url: unexpected "${text[i + j] as string}".`);
      group[j] = value;
    }
    const [g0, g1, g2, g3] = group as [number, number, number, number];
    bytes[out++] = (g0 << 2) | (g1 >> 4);
    if (have > 2) bytes[out++] = ((g1 & 0b1111) << 4) | (g2 >> 2);
    if (have > 3) bytes[out++] = ((g2 & 0b11) << 6) | g3;
  }
  return bytes;
}

/*
 * Little-endian explicitly, via DataView, rather than a view onto the typed
 * array's own buffer. That view would use the host's byte order, which is
 * little-endian on every machine anyone will run this on and therefore a bug
 * that could never be reproduced if it were ever wrong.
 */
function packFloat64(values: Float64Array): Uint8Array {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat64(i * 8, values[i] as number, true);
  return bytes;
}

function unpackFloat64(bytes: Uint8Array): Float64Array {
  if (bytes.length % 8 !== 0) {
    throw new Error(`Parameter block is ${bytes.length} bytes, not a multiple of 8.`);
  }
  const values = new Float64Array(bytes.length / 8);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < values.length; i++) values[i] = view.getFloat64(i * 8, true);
  return values;
}

/* -------------------------------------------------------------------- encoding */

interface SharePayload {
  readonly v: number;
  readonly architecture: Architecture;
  readonly dataset: DatasetOptions;
  readonly training: TrainingSettings;
  readonly lesson: string | null;
}

/**
 * True when `parameters` is exactly what a fresh network of this architecture
 * initializes to, byte for byte.
 *
 * Exported because it is the claim the omitted parameter block makes, and a
 * claim of that kind should be checkable from a test rather than trusted.
 */
export function parametersAreFresh(
  architecture: Architecture,
  parameters: Float64Array,
): boolean {
  let fresh: Float64Array;
  try {
    fresh = new Network(toNetworkConfig(architecture)).captureParameters();
  } catch {
    // An architecture that will not build cannot be matched against; encode the
    // parameters and let the decoder report the real problem.
    return false;
  }
  if (fresh.length !== parameters.length) return false;
  const a = new Uint8Array(fresh.buffer, fresh.byteOffset, fresh.byteLength);
  const b = new Uint8Array(parameters.buffer, parameters.byteOffset, parameters.byteLength);
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * True when `buffers` is exactly what a fresh network of this architecture
 * starts with: μ̂ = 0 and σ̂² = 1 everywhere.
 *
 * The same rule the parameters follow, for the same reason. An untrained
 * network's statistics are reproducible from the architecture alone, so they do
 * not need to travel.
 */
export function buffersAreFresh(architecture: Architecture, buffers: Float64Array): boolean {
  let fresh: Float64Array;
  try {
    fresh = new Network(toNetworkConfig(architecture)).captureBuffers();
  } catch {
    return false;
  }
  if (fresh.length !== buffers.length) return false;
  for (let i = 0; i < fresh.length; i++) if (!Object.is(fresh[i], buffers[i])) return false;
  return true;
}

/**
 * The fragment for a shared state, without the leading `#`.
 *
 * Pass `parameters` as they stand; the encoder decides whether they need to
 * travel.
 */
export function encodeShareLink(state: SharedState): string {
  const payload: SharePayload = {
    v: SHARE_FORMAT_VERSION,
    architecture: state.architecture,
    dataset: state.datasetOptions,
    training: state.training,
    lesson: state.lessonId,
  };
  const json = new TextEncoder().encode(JSON.stringify(payload));
  let fragment = `s=${toBase64Url(json)}`;

  const params = state.parameters;
  if (params !== null && !parametersAreFresh(state.architecture, params)) {
    fragment += `&p=${toBase64Url(packFloat64(params))}`;
  }
  if (state.buffers.length > 0 && !buffersAreFresh(state.architecture, state.buffers)) {
    fragment += `&r=${toBase64Url(packFloat64(state.buffers))}`;
  }
  return fragment;
}

/** A full URL for `state`, built from `base` with its existing fragment replaced. */
export function shareUrl(base: string, state: SharedState): string {
  const hashAt = base.indexOf('#');
  const stem = hashAt < 0 ? base : base.slice(0, hashAt);
  return `${stem}#${encodeShareLink(state)}`;
}

/* -------------------------------------------------------------------- decoding */

export type DecodeResult =
  | { readonly ok: true; readonly state: SharedState }
  | { readonly ok: false; readonly error: string };

/**
 * A fragment is untrusted input. Every field is checked.
 *
 * The alternative — casting the parsed JSON to the state type and letting the
 * engine hit it — turns a mistyped link into a thrown exception from somewhere
 * deep in the forward pass, which reads to the person holding the link as "the
 * app is broken". Nothing here throws; a bad link produces a sentence.
 */
export function decodeShareLink(fragment: string): DecodeResult {
  const text = fragment.startsWith('#') ? fragment.slice(1) : fragment;
  if (text.length === 0) return { ok: false, error: 'The link carries no state.' };

  const parts = new Map<string, string>();
  for (const chunk of text.split('&')) {
    const eq = chunk.indexOf('=');
    if (eq > 0) parts.set(chunk.slice(0, eq), chunk.slice(eq + 1));
  }
  const encodedState = parts.get('s');
  if (encodedState === undefined) {
    return { ok: false, error: 'The link is missing its state section.' };
  }

  try {
    const json = new TextDecoder().decode(fromBase64Url(encodedState));
    let raw: unknown;
    try {
      raw = JSON.parse(json);
    } catch {
      // The engine's own message here names a byte offset in a string nobody
      // will ever look at. Every other failure in this function is a sentence;
      // this one should be too.
      return { ok: false, error: 'The link is damaged, most likely truncated in transit.' };
    }
    if (!isRecord(raw)) return { ok: false, error: 'The link does not contain an object.' };

    if (raw['v'] !== SHARE_FORMAT_VERSION) {
      return {
        ok: false,
        error: `This link was made by a different version of AwryNN (format ${String(raw['v'])}, this build reads ${SHARE_FORMAT_VERSION}).`,
      };
    }

    const architecture = parseArchitecture(raw['architecture']);
    const datasetOptions = parseDataset(raw['dataset']);
    const training = parseTraining(raw['training']);
    const lessonId = raw['lesson'] === null ? null : expectString(raw['lesson'], 'lesson');

    let parameters: Float64Array | null = null;
    const encodedParams = parts.get('p');
    if (encodedParams !== undefined) {
      parameters = unpackFloat64(fromBase64Url(encodedParams));
      const expected = expectedParameterCount(architecture);
      if (parameters.length !== expected) {
        return {
          ok: false,
          error: `The link carries ${parameters.length} parameters but its architecture needs ${expected}.`,
        };
      }
      for (let i = 0; i < parameters.length; i++) {
        if (!Number.isFinite(parameters[i] as number)) {
          return { ok: false, error: 'The link contains a parameter that is not a finite number.' };
        }
      }
    }

    const expectedBuffers = expectedBufferCount(architecture);
    let buffers: Float64Array = new Float64Array(expectedBuffers);
    // A fresh network's statistics are μ̂ = 0, σ̂² = 1, laid out mean-then-var
    // per normalizing layer.
    fillFreshBuffers(architecture, buffers);

    const encodedBuffers = parts.get('r');
    if (encodedBuffers !== undefined) {
      buffers = unpackFloat64(fromBase64Url(encodedBuffers));
      if (buffers.length !== expectedBuffers) {
        return {
          ok: false,
          error: `The link carries ${buffers.length} running statistics but its architecture needs ${expectedBuffers}.`,
        };
      }
      for (let i = 0; i < buffers.length; i++) {
        if (!Number.isFinite(buffers[i] as number)) {
          return {
            ok: false,
            error: 'The link contains a running statistic that is not a finite number.',
          };
        }
      }
    }

    return {
      ok: true,
      state: { architecture, datasetOptions, training, lessonId, parameters, buffers },
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * Weights, biases and γ for every layer, in the order `Network` lays them out.
 *
 * Duplicating the layout rather than building a Network to ask it is deliberate:
 * this runs on a link that has not been validated yet, and constructing a
 * network from unvalidated numbers is what the caps above exist to avoid.
 */
export function expectedParameterCount(architecture: Architecture): number {
  let total = 0;
  let inputs = architecture.inputSize;
  for (const layer of architecture.layers) {
    total += inputs * layer.units + layer.units;
    if (layer.batchNorm === true) total += layer.units;
    inputs = layer.units;
  }
  return total;
}

/** μ̂ and σ̂² for every normalizing layer. Zero when none of them normalize. */
export function expectedBufferCount(architecture: Architecture): number {
  let total = 0;
  for (const layer of architecture.layers) {
    if (layer.batchNorm === true) total += 2 * layer.units;
  }
  return total;
}

/** μ̂ = 0, σ̂² = 1, in the layout `Network` uses: mean then variance, per layer. */
function fillFreshBuffers(architecture: Architecture, out: Float64Array): void {
  let offset = 0;
  for (const layer of architecture.layers) {
    if (layer.batchNorm !== true) continue;
    offset += layer.units;
    out.fill(1, offset, offset + layer.units);
    offset += layer.units;
  }
}

/* ------------------------------------------------------------------ validation */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function expectString(value: unknown, field: string): string {
  if (typeof value !== 'string') throw new Error(`Expected ${field} to be text.`);
  return value;
}

function expectFinite(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Expected ${field} to be a number.`);
  }
  return value;
}

function expectInteger(value: unknown, field: string, min: number, max: number): number {
  const n = expectFinite(value, field);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new Error(`Expected ${field} to be a whole number between ${min} and ${max}, got ${n}.`);
  }
  return n;
}

function expectBoolean(value: unknown, field: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`Expected ${field} to be true or false.`);
  return value;
}

function expectMember<T extends string>(value: unknown, field: string, allowed: readonly T[]): T {
  const text = expectString(value, field);
  if (!(allowed as readonly string[]).includes(text)) {
    throw new Error(`Unknown ${field} "${text}".`);
  }
  return text as T;
}

function optionalFinite(value: unknown, field: string): number | undefined {
  return value === undefined ? undefined : expectFinite(value, field);
}

/*
 * Bounds here are sanity limits, not policy. They exist so a hand-edited link
 * cannot ask for 2^31 units and hang the tab before validateConfig sees it.
 */
const MAX_UNITS = 512;
const MAX_LAYERS = 24;
const MAX_SAMPLES = 20000;

function parseArchitecture(value: unknown): Architecture {
  if (!isRecord(value)) throw new Error('The link has no architecture.');
  const layersRaw = value['layers'];
  if (!Array.isArray(layersRaw)) throw new Error('Expected architecture.layers to be a list.');
  if (layersRaw.length < 1 || layersRaw.length > MAX_LAYERS) {
    throw new Error(`A network needs between 1 and ${MAX_LAYERS} layers, the link asks for ${layersRaw.length}.`);
  }
  const layers: LayerSpec[] = layersRaw.map((layer: unknown, i): LayerSpec => {
    if (!isRecord(layer)) throw new Error(`Layer ${i} is not an object.`);
    const spec: LayerSpec = {
      units: expectInteger(layer['units'], `layer ${i} units`, 1, MAX_UNITS),
      activation: expectMember<ActivationName>(
        layer['activation'],
        `layer ${i} activation`,
        ACTIVATION_NAMES,
      ),
    };
    const alpha = optionalFinite(layer['leakyAlpha'], `layer ${i} leakyAlpha`);
    const withAlpha = alpha === undefined ? spec : { ...spec, leakyAlpha: alpha };
    if (layer['batchNorm'] === undefined) return withAlpha;
    return { ...withAlpha, batchNorm: expectBoolean(layer['batchNorm'], `layer ${i} batchNorm`) };
  });

  return {
    inputSize: expectInteger(value['inputSize'], 'inputSize', 1, MAX_UNITS),
    layers,
    loss: expectMember<LossName>(value['loss'], 'loss', LOSS_NAMES),
    seed: expectInteger(value['seed'], 'seed', 0, Number.MAX_SAFE_INTEGER),
    init: parseInit(value['init']),
    l2: expectFinite(value['l2'], 'l2'),
  };
}

function parseInit(value: unknown): InitScheme {
  if (!isRecord(value)) throw new Error('The link has no initialization scheme.');
  const kind = expectString(value['kind'], 'init.kind');
  switch (kind) {
    case 'glorot_uniform':
    case 'he_normal':
    case 'lecun_normal':
    case 'zeros':
      return { kind };
    case 'normal':
      return { kind, std: expectFinite(value['std'], 'init.std') };
    case 'uniform':
      return {
        kind,
        min: expectFinite(value['min'], 'init.min'),
        max: expectFinite(value['max'], 'init.max'),
      };
    case 'constant':
      return { kind, value: expectFinite(value['value'], 'init.value') };
    default:
      throw new Error(`Unknown initialization scheme "${kind}".`);
  }
}

function parseDataset(value: unknown): DatasetOptions {
  if (!isRecord(value)) throw new Error('The link has no dataset.');
  const options: Record<string, unknown> = {
    name: expectMember<DatasetName>(value['name'], 'dataset name', DATASET_NAMES),
  };
  if (value['samples'] !== undefined) {
    options['samples'] = expectInteger(value['samples'], 'dataset samples', 4, MAX_SAMPLES);
  }
  if (value['noise'] !== undefined) options['noise'] = expectFinite(value['noise'], 'dataset noise');
  if (value['seed'] !== undefined) {
    options['seed'] = expectInteger(value['seed'], 'dataset seed', 0, Number.MAX_SAFE_INTEGER);
  }
  if (value['validationFraction'] !== undefined) {
    const fraction = expectFinite(value['validationFraction'], 'dataset validationFraction');
    if (fraction < 0 || fraction >= 1) {
      throw new Error(`Validation fraction must be in [0, 1), got ${fraction}.`);
    }
    options['validationFraction'] = fraction;
  }
  if (value['classes'] !== undefined) {
    options['classes'] = expectInteger(value['classes'], 'dataset classes', 2, 64);
  }
  if (value['featureScale'] !== undefined) {
    const scale = value['featureScale'];
    if (!Array.isArray(scale)) throw new Error('Expected dataset featureScale to be a list.');
    options['featureScale'] = scale.map((s: unknown, i) => expectFinite(s, `featureScale[${i}]`));
  }
  return options as unknown as DatasetOptions;
}

function parseTraining(value: unknown): TrainingSettings {
  if (!isRecord(value)) throw new Error('The link has no training settings.');
  return {
    optimizer: parseOptimizer(value['optimizer']),
    learningRate: expectFinite(value['learningRate'], 'learningRate'),
    batchSize: expectInteger(value['batchSize'], 'batchSize', 1, MAX_SAMPLES),
    maxEpochs: expectInteger(value['maxEpochs'], 'maxEpochs', 1, 100000),
    dropout: expectFinite(value['dropout'], 'dropout'),
    gradientClip: expectFinite(value['gradientClip'], 'gradientClip'),
    standardize: expectBoolean(value['standardize'], 'standardize'),
  };
}

function parseOptimizer(value: unknown): OptimizerConfig {
  if (!isRecord(value)) throw new Error('The link has no optimizer.');
  const config: Record<string, unknown> = {
    name: expectMember<OptimizerName>(value['name'], 'optimizer name', OPTIMIZER_NAMES),
  };
  for (const field of ['momentum', 'rho', 'beta1', 'beta2', 'epsilon', 'weightDecay'] as const) {
    if (value[field] !== undefined) {
      config[field] = expectFinite(value[field], `optimizer ${field}`);
    }
  }
  return config as unknown as OptimizerConfig;
}
