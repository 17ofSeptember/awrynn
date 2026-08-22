/*
 * Export a trained network as runnable code (§8).
 *
 * "The codegen matters pedagogically: the learner sees that the beautiful
 * animated thing on screen is thirty lines of matrix multiplication."
 *
 * So the generated code is written to be READ, not just run. It uses the same
 * symbols as docs/MATH.md, keeps the row-major batch convention, and inlines
 * the weights as literal numbers rather than hiding them behind a loader. A
 * reader should be able to put it next to the dissection view and see the same
 * arithmetic.
 *
 * Numbers are emitted at 17 significant digits, which is the shortest precision
 * that round-trips a float64 exactly. Anything less would make the generated
 * code disagree with the engine in the last decimal, and §10 requires them to
 * match to 1e-12.
 */

import type { ActivationName } from './activations';
import type { Network } from './network';
import { BATCH_NORM_EPSILON } from './layers';

/** Exact round-trip: 17 significant digits recovers any float64 bit for bit. */
function literal(value: number): string {
  if (Number.isNaN(value)) return 'float("nan")';
  if (!Number.isFinite(value)) return value > 0 ? 'float("inf")' : 'float("-inf")';
  return value.toPrecision(17);
}

function jsLiteral(value: number): string {
  if (Number.isNaN(value)) return 'NaN';
  if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
  return value.toPrecision(17);
}

function matrixRows(
  data: Float64Array,
  rows: number,
  cols: number,
  format: (v: number) => string,
  indent: string,
): string {
  const lines: string[] = [];
  for (let r = 0; r < rows; r++) {
    const values: string[] = [];
    for (let c = 0; c < cols; c++) values.push(format(data[r * cols + c] as number));
    lines.push(`${indent}[${values.join(', ')}]`);
  }
  return lines.join(',\n');
}

const NUMPY_ACTIVATIONS: Readonly<Record<ActivationName, string>> = {
  linear: 'z',
  relu: 'np.maximum(0.0, z)',
  leaky_relu: 'np.where(z > 0.0, z, LEAKY_ALPHA * z)',
  tanh: 'np.tanh(z)',
  // The same two-sided form the engine uses: neither branch ever exponentiates
  // a positive argument, so large |z| cannot overflow.
  sigmoid: 'np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))',
  softmax: 'softmax(z)',
};

const JS_ACTIVATIONS: Readonly<Record<ActivationName, string>> = {
  linear: 'z',
  relu: 'z > 0 ? z : 0',
  leaky_relu: 'z > 0 ? z : LEAKY_ALPHA * z',
  tanh: 'Math.tanh(z)',
  sigmoid: 'z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))',
  softmax: 'z',
};

/** True when any layer normalizes, so the export needs BN_EPS and the statistics. */
function usesBatchNorm(network: Network): boolean {
  return network.layers.some((l) => l.batchNorm);
}

/**
 * A line for the file header, or nothing.
 *
 * Exported code is INFERENCE, so it normalizes by the running estimates rather
 * than by the batch. Saying so matters: someone who fed a batch through this
 * and compared it against the training-time numbers would otherwise be chasing
 * a discrepancy that is not a bug.
 */
function batchNormNote(network: Network, prefix: string): string {
  if (!usesBatchNorm(network)) return '';
  return (
    `\n${prefix}\n${prefix}Batch normalisation runs in evaluation mode here: each normalised layer uses` +
    `\n${prefix}the running mean and variance measured during training, so a prediction does` +
    `\n${prefix}not depend on what else you pass in alongside it.`
  );
}

export function generateNumpy(network: Network): string {
  const sizes = [network.inputSize, ...network.layers.map((l) => l.units)];
  const alpha = network.layers.find((l) => l.leakyAlpha !== undefined)?.leakyAlpha ?? 0.01;
  const usesSoftmax = network.layers.some((l) => l.activationName === 'softmax');

  const parts: string[] = [];
  parts.push(`"""
AwryNN export: ${sizes.join('-')}, ${network.lossName} loss.

Row-major batch convention, matching docs/MATH.md:
    X        [B, ${network.inputSize}]   one row per sample
    W^l      [n_{l-1}, n_l]
    b^l      [1, n_l]                    broadcast down the batch

    Z^l = A^{l-1} @ W^l + b^l
    A^l = phi(Z^l)${
    usesBatchNorm(network)
      ? `

A normalised layer inserts a step, and its bias is the shift that follows:

    U^l = A^{l-1} @ W^l
    Z^l = gamma^l * (U^l - mean^l) / sqrt(var^l + BN_EPS) + b^l`
      : ''
  }

Weights are inlined at full float64 precision, so this reproduces the engine's
own output exactly. ${network.parameterCount} parameters in total.${batchNormNote(network, '')}
"""

import numpy as np

LEAKY_ALPHA = ${literal(alpha)}${usesBatchNorm(network) ? `\nBN_EPS = ${literal(BATCH_NORM_EPSILON)}` : ''}
`);

  if (usesSoftmax) {
    parts.push(`
def softmax(z):
    # The row max is subtracted before exponentiating. Algebraically a no-op,
    # but it caps the largest exponent at exp(0) so a big logit yields a
    # probability instead of inf/inf.
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)
`);
  }

  network.layers.forEach((layer, i) => {
    const n = i + 1;
    parts.push(`
# Layer ${n}: ${layer.inputs} -> ${layer.units}, ${layer.activationName}${layer.batchNorm ? ', batch-normalised' : ''}
W${n} = np.array([
${matrixRows(layer.W.data, layer.inputs, layer.units, literal, '    ')},
])
b${n} = np.array([${Array.from(layer.b.data).map(literal).join(', ')}])${
      layer.batchNorm
        ? `
gamma${n} = np.array([${Array.from(layer.gamma.data).map(literal).join(', ')}])
# Running estimates, measured during training. Inference uses these rather than
# the statistics of whatever batch you happen to pass in.
mean${n} = np.array([${Array.from(layer.runningMean).map(literal).join(', ')}])
var${n} = np.array([${Array.from(layer.runningVar).map(literal).join(', ')}])`
        : ''
    }
`);
  });

  const body = network.layers
    .map((layer, i) => {
      const n = i + 1;
      const expression = NUMPY_ACTIVATIONS[layer.activationName];
      if (!layer.batchNorm) return `    z = a @ W${n} + b${n}\n    a = ${expression}`;
      // Written out rather than folded into the weights. It could be folded —
      // at inference this is a fixed per-unit affine map, so scale = gamma /
      // sqrt(var + eps) and shift = b - mean * scale would give the same
      // answers with less arithmetic. Left explicit because the point of an
      // export from a teaching tool is to be read.
      return (
        `    u = a @ W${n}\n` +
        `    z = gamma${n} * (u - mean${n}) / np.sqrt(var${n} + BN_EPS) + b${n}\n` +
        `    a = ${expression}`
      );
    })
    .join('\n');

  parts.push(`
def forward(x):
    """x: array of shape [B, ${network.inputSize}]. Returns [B, ${network.outputSize}]."""
    a = np.asarray(x, dtype=np.float64)
${body}
    return a


if __name__ == "__main__":
    print(forward(np.zeros((1, ${network.inputSize}))))
`);

  return parts.join('');
}

export function generateJavaScript(network: Network): string {
  const sizes = [network.inputSize, ...network.layers.map((l) => l.units)];
  const alpha = network.layers.find((l) => l.leakyAlpha !== undefined)?.leakyAlpha ?? 0.01;
  const usesSoftmax = network.layers.some((l) => l.activationName === 'softmax');

  const parts: string[] = [];
  parts.push(`/*
 * AwryNN export: ${sizes.join('-')}, ${network.lossName} loss.
 *
 * Dependency-free inference. Row-major throughout, matching docs/MATH.md:
 *
 *     z^l = a^{l-1} · W^l + b^l
 *     a^l = phi(z^l)${
    usesBatchNorm(network)
      ? `
 *
 * A normalised layer inserts a step, and its bias is the shift that follows:
 *
 *     u^l = a^{l-1} · W^l
 *     z^l = gamma^l * (u^l - mean^l) / sqrt(var^l + BN_EPS) + b^l`
      : ''
  }
 *
 * Weights are inlined at full float64 precision, so this reproduces the
 * engine's own output exactly. ${network.parameterCount} parameters in total.${batchNormNote(network, ' * ')}
 */

const LEAKY_ALPHA = ${jsLiteral(alpha)};${usesBatchNorm(network) ? `\nconst BN_EPS = ${jsLiteral(BATCH_NORM_EPSILON)};` : ''}
`);

  network.layers.forEach((layer, i) => {
    const n = i + 1;
    parts.push(`
// Layer ${n}: ${layer.inputs} -> ${layer.units}, ${layer.activationName}${layer.batchNorm ? ', batch-normalised' : ''}
const W${n} = [
${matrixRows(layer.W.data, layer.inputs, layer.units, jsLiteral, '  ')},
];
const b${n} = [${Array.from(layer.b.data).map(jsLiteral).join(', ')}];${
      layer.batchNorm
        ? `
const gamma${n} = [${Array.from(layer.gamma.data).map(jsLiteral).join(', ')}];
// Running estimates, measured during training. Inference uses these rather than
// the statistics of whatever batch you happen to pass in.
const mean${n} = [${Array.from(layer.runningMean).map(jsLiteral).join(', ')}];
const var${n} = [${Array.from(layer.runningVar).map(jsLiteral).join(', ')}];`
        : ''
    }
`);
  });

  if (usesSoftmax) {
    parts.push(`
function softmax(values) {
  // Row max subtracted before exponentiating, so a large logit cannot overflow.
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  let total = 0;
  const out = values.map((v) => {
    const e = Math.exp(v - max);
    total += e;
    return e;
  });
  return out.map((e) => e / total);
}
`);
  }

  const body = network.layers
    .map((layer, i) => {
      const expression = JS_ACTIVATIONS[layer.activationName];
      const applied =
        layer.activationName === 'softmax'
          ? `  a = softmax(next);`
          : `  a = next.map((z) => ${expression});`;
      const n = i + 1;
      // With batch norm the sum starts at zero rather than at the bias: the
      // bias IS beta here, and it is added after the normalisation.
      const inner = layer.batchNorm
        ? `    let sum = 0;
    for (let i = 0; i < ${layer.inputs}; i++) sum += a[i] * W${n}[i][j];
    next[j] = (gamma${n}[j] * (sum - mean${n}[j])) / Math.sqrt(var${n}[j] + BN_EPS) + b${n}[j];`
        : `    let sum = b${n}[j];
    for (let i = 0; i < ${layer.inputs}; i++) sum += a[i] * W${n}[i][j];
    next[j] = sum;`;
      return `  // layer ${n}${layer.batchNorm ? ' (batch-normalised)' : ''}
  next = new Array(${layer.units}).fill(0);
  for (let j = 0; j < ${layer.units}; j++) {
${inner}
  }
${applied}`;
    })
    .join('\n');

  parts.push(`
/** One sample in, one prediction out. */
export function forward(input) {
  let a = Array.from(input, Number);
  let next;
${body}
  return a;
}
`);

  return parts.join('');
}

/**
 * Evaluate generated JavaScript in-process, for the round-trip test (§10).
 *
 * Uses `new Function` on code this module just produced, not on anything a user
 * supplied: the generated source never leaves the process, and the test needs
 * to run the real emitted text rather than a reimplementation of it.
 */
export function evaluateGeneratedJs(source: string): (input: readonly number[]) => number[] {
  const body = source.replace('export function forward', 'function forward');
  const factory = new Function(`${body}\nreturn forward;`) as () => (
    input: readonly number[],
  ) => number[];
  return factory();
}
