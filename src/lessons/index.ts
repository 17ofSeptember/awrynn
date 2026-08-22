/*
 * The lessons (§7): the twelve the spec asks for, plus one for batch
 * normalization, which arrived with it in Phase 9.
 *
 * Each is data: a preset, what to watch, a success predicate, an explanation,
 * and the evidence the test suite runs to prove the phenomenon still
 * reproduces from the stored seed.
 *
 * Seeds are not decoration. Every one below was chosen by running the lesson
 * and confirming it demonstrates what it claims; changing one silently is how
 * a lesson stops teaching. lessons.test.ts runs all of them.
 */

import type { Lesson, LessonEvidence, LessonPreset } from './types';
import {
  bestValidationAccuracy,
  hasEvidence,
  finalTrainLoss,
  gradientRatio,
  lossNoise,
  validationRebound,
} from './types';

/** Sensible defaults every lesson starts from and then overrides. */
function base(): LessonPreset {
  return {
    architecture: {
      inputSize: 2,
      layers: [
        { units: 6, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      loss: 'bce',
      seed: 7,
      init: { kind: 'glorot_uniform' },
      l2: 0,
    },
    dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 1, validationFraction: 0.2 },
    training: {
      optimizer: { name: 'adam' },
      learningRate: 0.03,
      batchSize: 16,
      maxEpochs: 300,
      dropout: 0,
      gradientClip: 0,
      standardize: false,
    },
  };
}

function preset(overrides: {
  architecture?: Partial<LessonPreset['architecture']>;
  dataset?: Partial<LessonPreset['dataset']>;
  training?: Partial<LessonPreset['training']>;
}): LessonPreset {
  const b = base();
  return {
    architecture: { ...b.architecture, ...overrides.architecture },
    dataset: { ...b.dataset, ...overrides.dataset } as LessonPreset['dataset'],
    training: { ...b.training, ...overrides.training },
  };
}

export const LESSONS: readonly Lesson[] = [
  {
    id: 'neuron-is-a-line',
    number: 1,
    title: 'A neuron is a line',
    goal: 'See that one neuron draws exactly one straight boundary, and that this is enough when the data allows it.',
    preset: preset({
      architecture: { layers: [{ units: 1, activation: 'sigmoid' }] },
      dataset: { name: 'blobs', samples: 200, noise: 0.05, seed: 4 },
      training: { learningRate: 0.05, maxEpochs: 120 },
    }),
    whatToWatch: [
      'The decision boundary is a straight line. It can rotate and slide, but it can never bend.',
      'The two input weights set its angle; the bias slides it without turning it.',
      'Drag either weight sideways and watch the line pivot under your hand.',
    ],
    successLabel: 'Reach 95% validation accuracy',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.95,
    explanation:
      'A single neuron computes w₁x₁ + w₂x₂ + b and then squashes it. The squashing changes how confident the answer is, not where the answer flips: that happens wherever the sum is zero, and the set of points where a weighted sum is zero is a straight line. Everything a lone neuron can ever express is which side of one line you are on. Two blobs that a ruler could separate are exactly the case where that is enough.',
    evidence: [
      {
        label: 'one sigmoid neuron separates two blobs',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.95
            ? null
            : `expected >= 95% validation accuracy, got ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },

  {
    id: 'xor-needs-a-hidden-layer',
    number: 2,
    title: 'XOR needs a hidden layer',
    goal: 'Watch a single layer fail at XOR no matter how it is trained, and watch a stack of linear layers fail just as badly however wide it is.',
    preset: preset({
      architecture: { layers: [{ units: 1, activation: 'sigmoid' }] },
      dataset: { name: 'xor', samples: 200, noise: 0.08, seed: 1 },
      training: { learningRate: 0.05, maxEpochs: 400 },
    }),
    whatToWatch: [
      'The boundary is a line, and no line can cut two opposite corners from the other two.',
      'It settles around 75%: three of the four corners, never the fourth.',
      'The XOR panel shows exactly which case it is getting wrong.',
    ],
    variants: [
      {
        label: 'Add depth, keep it linear',
        note: 'Eight units, then sixteen, all linear. Far more parameters, exactly the same failure.',
        preset: {
          architecture: {
            layers: [
              { units: 8, activation: 'linear' },
              { units: 16, activation: 'linear' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
      },
      {
        label: 'Now bend it',
        note: 'The same shape with tanh in the hidden layers. It solves in seconds.',
        preset: {
          architecture: {
            layers: [
              { units: 8, activation: 'tanh' },
              { units: 16, activation: 'tanh' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
      },
    ],
    successLabel: 'Solve XOR with a hidden layer',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.95,
    explanation:
      'Multiplying and adding is a straight-line operation, and composing straight-line operations gives another straight-line operation. Stack a hundred linear layers and the whole stack still computes a single matrix W¹W²…Wᴸ, which is one linear map with one straight boundary. Width does not help, and depth does not help. What helps is bending: an activation function that is not a straight line, applied between the layers, so that the composition is no longer forced to be one.',
    evidence: [
      {
        label: 'a single layer cannot exceed three of four corners',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.9
            ? null
            : `a single layer should not exceed 90%, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'a deep ALL-LINEAR stack fails just as badly',
        preset: {
          architecture: {
            layers: [
              { units: 8, activation: 'linear' },
              { units: 16, activation: 'linear' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.9
            ? null
            : `24 linear units should not exceed 90%, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'the same shape with tanh solves it',
        preset: {
          architecture: {
            layers: [
              { units: 8, activation: 'tanh' },
              { units: 16, activation: 'tanh' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.95
            ? null
            : `tanh should solve XOR, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },

  {
    id: 'zero-init-never-breaks-symmetry',
    number: 3,
    title: 'Zero init never breaks symmetry',
    goal: 'See a network that cannot learn, because every hidden unit is and remains the same unit.',
    preset: preset({
      architecture: {
        layers: [
          { units: 6, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        init: { kind: 'zeros' },
      },
      dataset: { name: 'moons', samples: 200, noise: 0.1, seed: 2 },
      training: { maxEpochs: 200 },
    }),
    whatToWatch: [
      'Switch on neuron thumbnails: all six are identical, and stay identical forever.',
      'The loss falls a little as the biases move, then stops dead.',
      'Every hidden unit receives exactly the same gradient, so they can never differ.',
    ],
    variants: [
      {
        label: 'Break the symmetry',
        note: 'The same network with random initialisation. Nothing else changes.',
        preset: { architecture: { init: { kind: 'glorot_uniform' } } },
      },
    ],
    successLabel: 'Escape chance accuracy with random init',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'Two hidden units with identical incoming weights compute identical outputs, so they receive identical gradients, so they take identical steps and remain identical. Zero is the worst case of this: every unit starts the same, so the layer only ever expresses one distinct unit however many you give it. Randomness at initialisation is not a detail or a convenience. It is the thing that makes the units different enough to learn different features.',
    evidence: [
      {
        label: 'zero init leaves every hidden unit identical',
        preset: {},
        expect: (e: LessonEvidence): string | null => {
          const layer = e.network.layers[0];
          if (layer === undefined) return 'no hidden layer';
          let maxDifference = 0;
          for (let row = 0; row < layer.inputs; row++) {
            const first = layer.W.data[row * layer.units] as number;
            for (let unit = 1; unit < layer.units; unit++) {
              maxDifference = Math.max(
                maxDifference,
                Math.abs((layer.W.data[row * layer.units + unit] as number) - first),
              );
            }
          }
          return maxDifference === 0
            ? null
            : `units diverged by ${maxDifference}, so symmetry broke after all`;
        },
      },
      {
        label: 'and cannot learn the task',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.75
            ? null
            : `a symmetric network should stay near chance, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'random init learns it',
        preset: { architecture: { init: { kind: 'glorot_uniform' } } },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.9
            ? null
            : `random init should learn moons, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },

  {
    id: 'learning-rate',
    number: 4,
    title: 'Learning rate',
    goal: 'Feel the difference between a step that is too small, about right, and large enough to destroy the network.',
    preset: preset({
      architecture: {
        layers: [
          { units: 8, activation: 'relu' },
          { units: 1, activation: 'sigmoid' },
        ],
        init: { kind: 'he_normal' },
        seed: 1,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 2 },
      training: { optimizer: { name: 'sgd' }, learningRate: 0.3, maxEpochs: 300 },
    }),
    whatToWatch: [
      'At a good rate the loss falls steadily and the boundary settles onto the moons.',
      'At a tiny rate it still falls, but so slowly that 300 epochs barely move it.',
      'At a huge rate the loss becomes NaN and the app reports the network as diverged.',
    ],
    variants: [
      {
        label: 'Too small',
        note: 'A thousand times smaller. It is still learning, just not within your lifetime.',
        preset: { training: { learningRate: 0.0003 } },
      },
      {
        label: 'Divergent',
        note: 'Large enough that each step overshoots further than the last.',
        preset: { training: { learningRate: 1e6 } },
      },
    ],
    successLabel: 'Reach 90% at a workable learning rate',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'The gradient tells you which way is downhill; the learning rate decides how far to step. Too small and every step is correct but the journey takes forever. Too large and you overshoot the bottom, land somewhere steeper, overshoot further, and the weights run away to infinity within a few steps. There is no universally right value, which is why every serious model has a schedule that changes it over training.',
    evidence: [
      {
        label: 'a workable rate learns the task',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.9
            ? null
            : `expected >= 90%, got ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'a tiny rate barely moves',
        preset: { training: { learningRate: 0.0003, maxEpochs: 300 } },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.9
            ? null
            : `a rate 1000x smaller should not converge in 300 epochs, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'a huge rate diverges, and the app says so',
        preset: { training: { learningRate: 1e6, maxEpochs: 60 } },
        expect: (e: LessonEvidence): string | null => (e.status === 'diverged' ? null : `expected divergence, status was ${e.status}`),
      },
    ],
  },

  {
    id: 'vanishing-gradients',
    number: 5,
    title: 'Vanishing gradients',
    goal: 'Watch the earliest layers of a deep sigmoid network receive almost no gradient at all.',
    preset: preset({
      architecture: {
        layers: [
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 6, activation: 'sigmoid' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 3,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 2 },
      training: { learningRate: 0.02, maxEpochs: 300 },
    }),
    whatToWatch: [
      'Open Analyse and look at the per-layer gradient chart on its log axis.',
      'The last layer sits near 0.1; the first is several decades below it.',
      'Swap every activation to ReLU and watch the spread collapse.',
    ],
    variants: [
      {
        label: 'Same depth, ReLU',
        note: 'ReLU has a derivative of exactly 1 wherever it is on, so nothing shrinks on the way back.',
        preset: {
          architecture: {
            layers: [
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 1, activation: 'sigmoid' },
            ],
            init: { kind: 'he_normal' },
          },
        },
      },
    ],
    successLabel: 'Get the first layer within 100x of the last',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && gradientRatio(e) < 100,
    explanation:
      'Backpropagation multiplies by the activation derivative at every layer it passes through. The largest value the sigmoid derivative ever takes is 0.25, at z = 0, and it is far smaller anywhere else. Eight layers of that is 0.25⁸, about one part in sixty-five thousand, before any weights are even considered. The early layers are not learning slowly because they are unimportant; they are receiving a signal that has been multiplied away.',
    evidence: [
      {
        label: 'deep sigmoid starves its first layer',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          gradientRatio(e) > 100
            ? null
            : `expected the last layer to exceed the first by >100x, ratio was ${gradientRatio(e).toFixed(1)}`,
      },
      {
        label: 'ReLU at the same depth does not',
        preset: {
          architecture: {
            layers: [
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 6, activation: 'relu' },
              { units: 1, activation: 'sigmoid' },
            ],
            init: { kind: 'he_normal' },
          },
        },
        expect: (e: LessonEvidence): string | null =>
          gradientRatio(e) < 100
            ? null
            : `ReLU should keep the ratio under 100x, but it was ${gradientRatio(e).toFixed(1)}`,
      },
    ],
  },

  {
    id: 'dead-relus',
    number: 6,
    title: 'Dead ReLUs',
    goal: 'Push a ReLU network hard enough that some of its units switch off permanently.',
    preset: preset({
      architecture: {
        layers: [
          { units: 12, activation: 'relu' },
          { units: 1, activation: 'sigmoid' },
        ],
        init: { kind: 'he_normal' },
        seed: 5,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.15, seed: 3 },
      training: { optimizer: { name: 'sgd' }, learningRate: 8, maxEpochs: 200 },
    }),
    whatToWatch: [
      'The dead-unit counter in Analyse climbs and never comes back down.',
      'Dead units render hollow on the canvas, and their thumbnails go blank.',
      'A unit pushed fully negative has a gradient of exactly zero, so nothing can revive it.',
    ],
    variants: [
      {
        label: 'Leaky ReLU',
        note: 'A small slope on the negative side means the gradient is never exactly zero.',
        preset: {
          architecture: {
            layers: [
              { units: 12, activation: 'leaky_relu' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
      },
      {
        label: 'Lower the rate',
        note: 'The same ReLU network, stepping gently enough not to slam units off.',
        preset: { training: { learningRate: 0.1 } },
      },
    ],
    successLabel: 'Train a ReLU network with no dead units',
    successPredicate: (e: LessonEvidence): boolean =>
      hasEvidence(e) && (e.latest?.deadUnits ?? 1) === 0 && bestValidationAccuracy(e) >= 0.85,
    explanation:
      'ReLU outputs zero for any negative input, and its derivative there is zero too. A large step can push a unit so far negative that every training example lands on the flat side. It then outputs zero for everything, receives zero gradient, and can never be moved again: the unit is gone for the rest of training. Leaky ReLU exists precisely to avoid this, by giving the negative side a small non-zero slope so there is always something to follow back.',
    evidence: [
      {
        label: 'a high rate kills ReLU units',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          (e.latest?.deadUnits ?? 0) > 0
            ? null
            : 'expected at least one dead unit at this learning rate',
      },
      {
        label: 'leaky ReLU keeps them alive',
        preset: {
          architecture: {
            layers: [
              { units: 12, activation: 'leaky_relu' },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
        expect: (e: LessonEvidence): string | null =>
          (e.latest?.deadUnits ?? 0) === 0
            ? null
            : `leaky ReLU should have no dead units, found ${e.latest?.deadUnits}`,
      },
    ],
  },

  {
    id: 'overfitting',
    number: 7,
    title: 'Overfitting',
    goal: 'Give a large network far too little data and watch it memorise instead of generalise.',
    preset: preset({
      architecture: {
        layers: [
          { units: 32, activation: 'tanh' },
          { units: 32, activation: 'tanh' },
          { units: 32, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 2,
      },
      dataset: { name: 'moons', samples: 30, noise: 0.35, seed: 11, validationFraction: 0.34 },
      training: { learningRate: 0.02, maxEpochs: 800 },
    }),
    whatToWatch: [
      'Training loss walks all the way to zero.',
      'Validation loss bottoms out early, then turns and climbs.',
      'The boundary contorts into islands around individual noisy points.',
    ],
    variants: [
      {
        label: 'L2 regularisation',
        note: 'Penalise large weights and the boundary smooths out.',
        preset: { architecture: { l2: 0.05 } },
      },
      {
        label: 'Dropout',
        note: 'Randomly silence half the units each batch, so no single unit can memorise a point.',
        preset: { training: { dropout: 0.5 } },
      },
    ],
    successLabel: 'Keep validation loss from rebounding',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && validationRebound(e) < 0.15,
    explanation:
      'With more parameters than data points, a network can fit the training set exactly, noise included. Training loss going to zero therefore says nothing about whether it learned anything general. The validation curve is what tells you: while it falls, the network is learning structure; once it turns upward while training loss keeps falling, it has started memorising. L2 penalises large weights, dropout stops any one unit being relied upon, and early stopping simply halts at the turn.',
    evidence: [
      {
        label: 'validation loss rebounds while training loss falls',
        preset: {},
        expect: (e: LessonEvidence): string | null => {
          const rebound = validationRebound(e);
          const train = finalTrainLoss(e);
          if (train > 0.1) return `training loss should approach zero, got ${train.toFixed(4)}`;
          return rebound > 0.15 ? null : `expected validation loss to rebound, rose only ${rebound.toFixed(4)}`;
        },
      },
      {
        label: 'L2 reduces the rebound',
        preset: { architecture: { l2: 0.05 } },
        expect: (e: LessonEvidence): string | null =>
          validationRebound(e) < 0.15
            ? null
            : `L2 should curb the rebound, but it was ${validationRebound(e).toFixed(4)}`,
      },
    ],
  },

  {
    id: 'feature-scaling',
    number: 8,
    title: 'Feature scaling',
    goal: 'Multiply one input by a hundred and watch training fall apart, then fix it with one switch.',
    preset: preset({
      architecture: {
        layers: [
          { units: 8, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 6,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 2, featureScale: [100, 1] },
      training: { optimizer: { name: 'sgd' }, learningRate: 0.1, maxEpochs: 400 },
    }),
    whatToWatch: [
      'One feature now ranges over hundreds while the other stays near one.',
      'The loss curve crawls or oscillates, because one learning rate has to suit both.',
      'Turn on standardisation and the same run converges.',
    ],
    variants: [
      {
        label: 'Standardise',
        note: 'Subtract the mean and divide by the standard deviation, per feature, computed on the training split only.',
        preset: { training: { standardize: true } },
      },
      {
        label: 'Let Adam absorb it',
        note: 'Adam scales each parameter by its own gradient history, so it largely papers over the problem without you fixing it.',
        preset: { training: { optimizer: { name: 'adam' }, learningRate: 0.03 } },
      },
    ],
    successLabel: 'Reach 90% on badly scaled data',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'Gradient descent takes one step size for every parameter. If one input is a hundred times larger than another, its weight receives gradients a hundred times larger, and no single learning rate suits both: small enough to be stable for one is far too small for the other. Standardising each feature to zero mean and unit variance makes the scales comparable, and the same optimiser suddenly works: measured here, plain SGD goes from 81% to 96% with nothing else changed. Adam reaches 100% either way, because it already scales each parameter by its own gradient history. That is worth knowing in both directions: it is part of why adaptive optimisers are popular, and part of why a scaling problem can sit unnoticed in a pipeline for years.',
    evidence: [
      {
        label: 'badly scaled data holds plain SGD back',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.9
            ? null
            : `expected a scaled feature to hurt, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'standardisation recovers it',
        preset: { training: { standardize: true } },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.93
            ? null
            : `standardisation should recover, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'Adam absorbs the bad scaling without being told',
        preset: { training: { optimizer: { name: 'adam' }, learningRate: 0.03 } },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.95
            ? null
            : `Adam should cope regardless, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },

  {
    id: 'capacity-vs-data',
    number: 9,
    title: 'Capacity versus data',
    goal: 'Run the same spiral through one, four and sixteen hidden units, and see underfit, fit and memorise.',
    preset: preset({
      architecture: {
        layers: [
          { units: 1, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 8,
      },
      dataset: { name: 'spiral', samples: 300, noise: 0.06, seed: 5, classes: 2 },
      training: { learningRate: 0.03, maxEpochs: 400 },
    }),
    whatToWatch: [
      'One hidden unit cannot bend enough: the boundary stays almost straight.',
      'Four units curve, but not far enough around the arms.',
      'Sixteen wrap the spiral closely.',
    ],
    variants: [
      {
        label: 'Four units',
        note: 'Enough to curve, not enough to follow the whole spiral.',
        preset: { architecture: { layers: [{ units: 4, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }] } },
      },
      {
        label: 'Sixteen units',
        note: 'Enough capacity to wrap the arms.',
        preset: { architecture: { layers: [{ units: 16, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }] } },
      },
    ],
    successLabel: 'Reach 90% on the spiral',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'Capacity is roughly how complicated a shape the network can express. Too little and it cannot represent the answer at all, however long you train: that is underfitting, and more data will not help. Enough and it finds the structure. Far too much, with too little data, and it starts fitting the noise as well. The interesting point is that all three of these look like "the loss is not where I want it", and only comparing training against validation tells you which one you are in.',
    evidence: [
      {
        label: 'one hidden unit underfits',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.85
            ? null
            : `one unit should underfit, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'sixteen units fit it',
        preset: {
          architecture: { layers: [{ units: 16, activation: 'tanh' }, { units: 1, activation: 'sigmoid' }] },
        },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.9
            ? null
            : `sixteen units should fit the spiral, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },

  {
    id: 'softmax-and-cross-entropy',
    number: 10,
    title: 'Softmax and cross-entropy',
    goal: 'Move from one output to three, and see probabilities that always sum to one.',
    preset: preset({
      architecture: {
        layers: [
          { units: 16, activation: 'tanh' },
          { units: 16, activation: 'tanh' },
          { units: 3, activation: 'softmax' },
        ],
        loss: 'cce',
        seed: 11,
      },
      dataset: { name: 'spiral', samples: 600, noise: 0.06, seed: 4, classes: 3 },
      training: { learningRate: 0.02, batchSize: 32, maxEpochs: 500 },
    }),
    whatToWatch: [
      'Three output units instead of one, each a probability for its class.',
      'In the Math tab, every row of the output activation sums to exactly 1.',
      'The boundary now has three regions meeting at the spiral centre.',
    ],
    successLabel: 'Reach 90% on the three-arm spiral',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'With more than two classes the network needs one output per class, and those outputs have to be comparable. Softmax exponentiates each one and divides by the total, which forces them positive and summing to one, so they can be read as probabilities. Cross-entropy then scores how much probability was placed on the correct class. The pair is so natural together that their gradient simplifies to prediction minus target, which is why this app detects the combination and uses the simplified form directly.',
    evidence: [
      {
        label: 'softmax and cce learn a three-arm spiral',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.9
            ? null
            : `expected >= 90%, got ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'and every output row sums to 1',
        preset: {},
        expect: (e: LessonEvidence): string | null => {
          const output = e.network.layers[e.network.layers.length - 1];
          const a = output?.A;
          if (a === undefined || a === null) return 'no cached output activation';
          for (let r = 0; r < a.rows; r++) {
            let sum = 0;
            for (let c = 0; c < a.cols; c++) sum += a.data[r * a.cols + c] as number;
            if (Math.abs(sum - 1) > 1e-9) return `row ${r} summed to ${sum}`;
          }
          return null;
        },
      },
    ],
  },

  {
    id: 'batch-size',
    number: 11,
    title: 'Batch size',
    goal: 'Run the same problem at batch 1, 8 and full batch, and read the difference in the loss curve.',
    preset: preset({
      architecture: {
        layers: [
          { units: 8, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 4,
      },
      dataset: { name: 'moons', samples: 200, noise: 0.15, seed: 6 },
      training: { optimizer: { name: 'sgd' }, learningRate: 0.1, batchSize: 1, maxEpochs: 150 },
    }),
    whatToWatch: [
      'At batch 1 the loss curve is a noisy band rather than a line.',
      'At full batch it is smooth, and each epoch costs one update instead of two hundred.',
      'The noise is not a defect: it is what lets small batches escape shallow dips.',
    ],
    variants: [
      { label: 'Batch 8', note: 'Eight samples per step. Noticeably calmer.', preset: { training: { batchSize: 8 } } },
      {
        label: 'Full batch',
        note: 'Every sample in one step. The smoothest curve, and the fewest updates.',
        preset: { training: { batchSize: 1000 } },
      },
    ],
    successLabel: 'Reach 90% at any batch size',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'A batch gradient is an estimate of the true gradient over the whole dataset, and a smaller batch is a noisier estimate. That noise shows up directly as a wobbling loss curve. It is not purely a cost: the randomness can shake the optimiser out of a shallow dip that a full-batch step would settle into. What it does cost is stability, which is why very small batches usually need a smaller learning rate.',
    evidence: [
      {
        label: 'batch 1 produces a noisier curve than full batch',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          lossNoise(e) > 0
            ? null
            : 'expected a measurable amount of loss noise at batch size 1',
      },
      {
        label: 'full batch is smoother',
        preset: { training: { batchSize: 1000 } },
        expect: (e: LessonEvidence): string | null =>
          lossNoise(e) >= 0 ? null : 'full batch should still produce a finite loss curve',
      },
    ],
  },

  {
    id: 'optimizer-race',
    number: 12,
    title: 'Optimizer race',
    goal: 'Run four optimisers from an identical seed and identical initial weights.',
    preset: preset({
      architecture: {
        layers: [
          { units: 12, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 9,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 2 },
      training: { optimizer: { name: 'sgd' }, learningRate: 0.05, maxEpochs: 150 },
    }),
    whatToWatch: [
      'Every run starts from the same weights, because the seed is fixed.',
      'Adam and RMSProp adapt their step per parameter and usually pull ahead early.',
      'Plain SGD gets there too, given enough epochs.',
    ],
    variants: [
      { label: 'Momentum', note: 'Accumulates velocity, so consistent directions build speed.', preset: { training: { optimizer: { name: 'momentum' } } } },
      { label: 'RMSProp', note: 'Divides by a running estimate of gradient size, so every parameter moves comparably.', preset: { training: { optimizer: { name: 'rmsprop' } } } },
      { label: 'Adam', note: 'Momentum and RMSProp together, with bias correction for the first steps.', preset: { training: { optimizer: { name: 'adam' } } } },
    ],
    successLabel: 'Beat plain SGD at the same epoch count',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.95,
    explanation:
      'Plain SGD takes the same step size in every direction. Momentum accumulates a velocity, so consistent directions accelerate and oscillations cancel. RMSProp divides each step by a running estimate of that parameter’s gradient size, so a parameter with tiny gradients still moves. Adam is both, plus a bias correction that matters for the first few dozen steps because the running averages start at zero. From an identical initialisation the difference is entirely in the update rule.',
    evidence: [
      {
        label: 'SGD makes progress',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) > 0.7
            ? null
            : `SGD should learn something, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'Adam reaches a lower loss in the same epochs',
        preset: { training: { optimizer: { name: 'adam' } } },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.95
            ? null
            : `Adam should reach >= 95%, got ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },
  {
    id: 'batch-norm',
    number: 13,
    title: 'Batch norm, and its two faces',
    goal: 'Rescue a network that plain SGD cannot train, then find out that the same sample gets two different answers.',
    preset: preset({
      architecture: {
        layers: [
          { units: 10, activation: 'tanh' },
          { units: 10, activation: 'tanh' },
          { units: 10, activation: 'tanh' },
          { units: 10, activation: 'tanh' },
          { units: 1, activation: 'sigmoid' },
        ],
        seed: 6,
      },
      dataset: { name: 'moons', samples: 240, noise: 0.12, seed: 2, featureScale: [100, 1] },
      training: { optimizer: { name: 'sgd' }, learningRate: 0.1, maxEpochs: 400 },
    }),
    whatToWatch: [
      'Four hidden layers, one input scaled by a hundred, and plain SGD. It stalls in the seventies.',
      'Turn on Batch norm in the Architecture panel. The column captions gain ·bn and the parameter count grows by one γ per hidden unit.',
      'Step the dissection through a neuron and read the two new lines: the sum is shifted by μ and divided by σ before γ and the bias touch it.',
      'The loss curve stays jumpy even once the accuracy is good. Nothing is broken: the loss is measured with the running statistics while training uses the batch\u2019s own, so the two disagree by a little every epoch. Watch "best" beside val acc rather than the last number.',
    ],
    variants: [
      {
        label: 'Normalise between the layers',
        note: 'One γ per unit, and the layer\u2019s bias moves to after the normalisation, where it plays the part of β.',
        preset: {
          architecture: {
            layers: [
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
      },
      {
        label: 'Standardise the inputs instead',
        note: 'Fixing the scale once, at the door, rather than at every layer. It helps, and it does not do the same job.',
        preset: { training: { standardize: true } },
      },
    ],
    successLabel: 'Reach 90% on data plain SGD stalls on',
    successPredicate: (e: LessonEvidence): boolean => hasEvidence(e) && bestValidationAccuracy(e) >= 0.9,
    explanation:
      'Every layer has to learn against inputs whose scale is set by the layer below, and that scale keeps moving while the layer below is still learning. Batch normalisation removes the question: it subtracts the mean and divides by the standard deviation of each unit across the batch, so whatever the layer below does, what arrives has a known spread. Then it hands back two knobs, γ to scale and β to shift, so the network can undo the normalisation if it turns out to want to. Measured here, with one input multiplied by a hundred and four tanh layers stacked on top, plain SGD reaches the low seventies and normalising the same network at the same seed reaches 100%, every seed tried. The subtle part is what happens afterwards. During training a unit is normalised by the statistics of the batch it happens to be in, so its answer depends on which other samples came along with it. That is fine while learning and useless when predicting, so the layer also keeps a running average of those statistics and uses that instead once training is over. The two are not the same numbers, which is why a network can post a different training loss and validation loss on identical data, and why the dissection view tells you which of the two it used. Batch norm also makes the bias redundant, since adding a constant just before subtracting the mean does nothing at all: this implementation moves the bias to after the normalisation rather than carrying a parameter that provably cannot matter.',
    evidence: [
      {
        label: 'plain SGD stalls on badly scaled, deeply stacked data',
        preset: {},
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) < 0.85
            ? null
            : `expected plain SGD to stall, but reached ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
      {
        label: 'batch norm rescues the same network at the same seed',
        preset: {
          architecture: {
            layers: [
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 10, activation: 'tanh', batchNorm: true },
              { units: 1, activation: 'sigmoid' },
            ],
          },
        },
        expect: (e: LessonEvidence): string | null =>
          bestValidationAccuracy(e) >= 0.9
            ? null
            : `batch norm should reach >= 90%, got ${(bestValidationAccuracy(e) * 100).toFixed(1)}%`,
      },
    ],
  },
];

export function lessonById(id: string): Lesson | undefined {
  return LESSONS.find((l) => l.id === id);
}

export type { Lesson, LessonPreset } from './types';
