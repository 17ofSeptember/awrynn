# LESSONS.md

Each lesson's pedagogy and success criteria.

> **Generated from `src/lessons/index.ts` by `npm run docs:lessons`.** Lessons are data, not prose, and a hand-written copy of them would drift from the code that runs. Edit the lessons and regenerate.

Every preset below stores an explicit seed. `src/lessons/__tests__/lessons.test.ts` trains each configuration headlessly and asserts the lesson still demonstrates what it claims, which is the Phase 7 gate in §11.

| # | Lesson | Success condition |
| --- | --- | --- |
| 1 | [A neuron is a line](#neuron-is-a-line) | Reach 95% validation accuracy |
| 2 | [XOR needs a hidden layer](#xor-needs-a-hidden-layer) | Solve XOR with a hidden layer |
| 3 | [Zero init never breaks symmetry](#zero-init-never-breaks-symmetry) | Escape chance accuracy with random init |
| 4 | [Learning rate](#learning-rate) | Reach 90% at a workable learning rate |
| 5 | [Vanishing gradients](#vanishing-gradients) | Get the first layer within 100x of the last |
| 6 | [Dead ReLUs](#dead-relus) | Train a ReLU network with no dead units |
| 7 | [Overfitting](#overfitting) | Keep validation loss from rebounding |
| 8 | [Feature scaling](#feature-scaling) | Reach 90% on badly scaled data |
| 9 | [Capacity versus data](#capacity-vs-data) | Reach 90% on the spiral |
| 10 | [Softmax and cross-entropy](#softmax-and-cross-entropy) | Reach 90% on the three-arm spiral |
| 11 | [Batch size](#batch-size) | Reach 90% at any batch size |
| 12 | [Optimizer race](#optimizer-race) | Beat plain SGD at the same epoch count |
| 13 | [Batch norm, and its two faces](#batch-norm) | Reach 90% on data plain SGD stalls on |

---

## 1. A neuron is a line

<a id="neuron-is-a-line"></a>

See that one neuron draws exactly one straight boundary, and that this is enough when the data allows it.

**Preset**

| | |
| --- | --- |
| architecture | `2-1` (sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `7` |
| dataset | `blobs`, 200 samples, noise 0.05, seed `4` |
| training | adam, lr 0.05, batch 16, 120 epochs |

**What to watch**

- The decision boundary is a straight line. It can rotate and slide, but it can never bend.
- The two input weights set its angle; the bias slides it without turning it.
- Drag either weight sideways and watch the line pivot under your hand.

**Success:** Reach 95% validation accuracy

**Explanation**

A single neuron computes w₁x₁ + w₂x₂ + b and then squashes it. The squashing changes how confident the answer is, not where the answer flips: that happens wherever the sum is zero, and the set of points where a weighted sum is zero is a straight line. Everything a lone neuron can ever express is which side of one line you are on. Two blobs that a ruler could separate are exactly the case where that is enough.

**Verified by**

- one sigmoid neuron separates two blobs

---

## 2. XOR needs a hidden layer

<a id="xor-needs-a-hidden-layer"></a>

Watch a single layer fail at XOR no matter how it is trained, and watch a stack of linear layers fail just as badly however wide it is.

**Preset**

| | |
| --- | --- |
| architecture | `2-1` (sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `7` |
| dataset | `xor`, 200 samples, noise 0.08, seed `1` |
| training | adam, lr 0.05, batch 16, 400 epochs |

**What to watch**

- The boundary is a line, and no line can cut two opposite corners from the other two.
- It settles around 75%: three of the four corners, never the fourth.
- The XOR panel shows exactly which case it is getting wrong.

**Variants**

- **Add depth, keep it linear.** Eight units, then sixteen, all linear. Far more parameters, exactly the same failure.
- **Now bend it.** The same shape with tanh in the hidden layers. It solves in seconds.

**Success:** Solve XOR with a hidden layer

**Explanation**

Multiplying and adding is a straight-line operation, and composing straight-line operations gives another straight-line operation. Stack a hundred linear layers and the whole stack still computes a single matrix W¹W²…Wᴸ, which is one linear map with one straight boundary. Width does not help, and depth does not help. What helps is bending: an activation function that is not a straight line, applied between the layers, so that the composition is no longer forced to be one.

**Verified by**

- a single layer cannot exceed three of four corners
- a deep ALL-LINEAR stack fails just as badly
- the same shape with tanh solves it

---

## 3. Zero init never breaks symmetry

<a id="zero-init-never-breaks-symmetry"></a>

See a network that cannot learn, because every hidden unit is and remains the same unit.

**Preset**

| | |
| --- | --- |
| architecture | `2-6-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `zeros` |
| network seed | `7` |
| dataset | `moons`, 200 samples, noise 0.1, seed `2` |
| training | adam, lr 0.03, batch 16, 200 epochs |

**What to watch**

- Switch on neuron thumbnails: all six are identical, and stay identical forever.
- The loss falls a little as the biases move, then stops dead.
- Every hidden unit receives exactly the same gradient, so they can never differ.

**Variants**

- **Break the symmetry.** The same network with random initialisation. Nothing else changes.

**Success:** Escape chance accuracy with random init

**Explanation**

Two hidden units with identical incoming weights compute identical outputs, so they receive identical gradients, so they take identical steps and remain identical. Zero is the worst case of this: every unit starts the same, so the layer only ever expresses one distinct unit however many you give it. Randomness at initialisation is not a detail or a convenience. It is the thing that makes the units different enough to learn different features.

**Verified by**

- zero init leaves every hidden unit identical
- and cannot learn the task
- random init learns it

---

## 4. Learning rate

<a id="learning-rate"></a>

Feel the difference between a step that is too small, about right, and large enough to destroy the network.

**Preset**

| | |
| --- | --- |
| architecture | `2-8-1` (relu, sigmoid) |
| loss | `bce` |
| init | `he_normal` |
| network seed | `1` |
| dataset | `moons`, 240 samples, noise 0.12, seed `2` |
| training | sgd, lr 0.3, batch 16, 300 epochs |

**What to watch**

- At a good rate the loss falls steadily and the boundary settles onto the moons.
- At a tiny rate it still falls, but so slowly that 300 epochs barely move it.
- At a huge rate the loss becomes NaN and the app reports the network as diverged.

**Variants**

- **Too small.** A thousand times smaller. It is still learning, just not within your lifetime.
- **Divergent.** Large enough that each step overshoots further than the last.

**Success:** Reach 90% at a workable learning rate

**Explanation**

The gradient tells you which way is downhill; the learning rate decides how far to step. Too small and every step is correct but the journey takes forever. Too large and you overshoot the bottom, land somewhere steeper, overshoot further, and the weights run away to infinity within a few steps. There is no universally right value, which is why every serious model has a schedule that changes it over training.

**Verified by**

- a workable rate learns the task
- a tiny rate barely moves
- a huge rate diverges, and the app says so

---

## 5. Vanishing gradients

<a id="vanishing-gradients"></a>

Watch the earliest layers of a deep sigmoid network receive almost no gradient at all.

**Preset**

| | |
| --- | --- |
| architecture | `2-6-6-6-6-6-6-6-1` (sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `3` |
| dataset | `moons`, 240 samples, noise 0.12, seed `2` |
| training | adam, lr 0.02, batch 16, 300 epochs |

**What to watch**

- Open Analyse and look at the per-layer gradient chart on its log axis.
- The last layer sits near 0.1; the first is several decades below it.
- Swap every activation to ReLU and watch the spread collapse.

**Variants**

- **Same depth, ReLU.** ReLU has a derivative of exactly 1 wherever it is on, so nothing shrinks on the way back.

**Success:** Get the first layer within 100x of the last

**Explanation**

Backpropagation multiplies by the activation derivative at every layer it passes through. The largest value the sigmoid derivative ever takes is 0.25, at z = 0, and it is far smaller anywhere else. Eight layers of that is 0.25⁸, about one part in sixty-five thousand, before any weights are even considered. The early layers are not learning slowly because they are unimportant; they are receiving a signal that has been multiplied away.

**Verified by**

- deep sigmoid starves its first layer
- ReLU at the same depth does not

---

## 6. Dead ReLUs

<a id="dead-relus"></a>

Push a ReLU network hard enough that some of its units switch off permanently.

**Preset**

| | |
| --- | --- |
| architecture | `2-12-1` (relu, sigmoid) |
| loss | `bce` |
| init | `he_normal` |
| network seed | `5` |
| dataset | `moons`, 240 samples, noise 0.15, seed `3` |
| training | sgd, lr 8, batch 16, 200 epochs |

**What to watch**

- The dead-unit counter in Analyse climbs and never comes back down.
- Dead units render hollow on the canvas, and their thumbnails go blank.
- A unit pushed fully negative has a gradient of exactly zero, so nothing can revive it.

**Variants**

- **Leaky ReLU.** A small slope on the negative side means the gradient is never exactly zero.
- **Lower the rate.** The same ReLU network, stepping gently enough not to slam units off.

**Success:** Train a ReLU network with no dead units

**Explanation**

ReLU outputs zero for any negative input, and its derivative there is zero too. A large step can push a unit so far negative that every training example lands on the flat side. It then outputs zero for everything, receives zero gradient, and can never be moved again: the unit is gone for the rest of training. Leaky ReLU exists precisely to avoid this, by giving the negative side a small non-zero slope so there is always something to follow back.

**Verified by**

- a high rate kills ReLU units
- leaky ReLU keeps them alive

---

## 7. Overfitting

<a id="overfitting"></a>

Give a large network far too little data and watch it memorise instead of generalise.

**Preset**

| | |
| --- | --- |
| architecture | `2-32-32-32-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `2` |
| dataset | `moons`, 30 samples, noise 0.35, seed `11` |
| training | adam, lr 0.02, batch 16, 800 epochs |

**What to watch**

- Training loss walks all the way to zero.
- Validation loss bottoms out early, then turns and climbs.
- The boundary contorts into islands around individual noisy points.

**Variants**

- **L2 regularisation.** Penalise large weights and the boundary smooths out.
- **Dropout.** Randomly silence half the units each batch, so no single unit can memorise a point.

**Success:** Keep validation loss from rebounding

**Explanation**

With more parameters than data points, a network can fit the training set exactly, noise included. Training loss going to zero therefore says nothing about whether it learned anything general. The validation curve is what tells you: while it falls, the network is learning structure; once it turns upward while training loss keeps falling, it has started memorising. L2 penalises large weights, dropout stops any one unit being relied upon, and early stopping simply halts at the turn.

**Verified by**

- validation loss rebounds while training loss falls
- L2 reduces the rebound

---

## 8. Feature scaling

<a id="feature-scaling"></a>

Multiply one input by a hundred and watch training fall apart, then fix it with one switch.

**Preset**

| | |
| --- | --- |
| architecture | `2-8-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `6` |
| dataset | `moons`, 240 samples, noise 0.12, seed `2` |
| training | sgd, lr 0.1, batch 16, 400 epochs |
| feature scale | `[100, 1]` |

**What to watch**

- One feature now ranges over hundreds while the other stays near one.
- The loss curve crawls or oscillates, because one learning rate has to suit both.
- Turn on standardisation and the same run converges.

**Variants**

- **Standardise.** Subtract the mean and divide by the standard deviation, per feature, computed on the training split only.
- **Let Adam absorb it.** Adam scales each parameter by its own gradient history, so it largely papers over the problem without you fixing it.

**Success:** Reach 90% on badly scaled data

**Explanation**

Gradient descent takes one step size for every parameter. If one input is a hundred times larger than another, its weight receives gradients a hundred times larger, and no single learning rate suits both: small enough to be stable for one is far too small for the other. Standardising each feature to zero mean and unit variance makes the scales comparable, and the same optimiser suddenly works: measured here, plain SGD goes from 81% to 96% with nothing else changed. Adam reaches 100% either way, because it already scales each parameter by its own gradient history. That is worth knowing in both directions: it is part of why adaptive optimisers are popular, and part of why a scaling problem can sit unnoticed in a pipeline for years.

**Verified by**

- badly scaled data holds plain SGD back
- standardisation recovers it
- Adam absorbs the bad scaling without being told

---

## 9. Capacity versus data

<a id="capacity-vs-data"></a>

Run the same spiral through one, four and sixteen hidden units, and see underfit, fit and memorise.

**Preset**

| | |
| --- | --- |
| architecture | `2-1-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `8` |
| dataset | `spiral`, 300 samples, noise 0.06, seed `5` |
| training | adam, lr 0.03, batch 16, 400 epochs |

**What to watch**

- One hidden unit cannot bend enough: the boundary stays almost straight.
- Four units curve, but not far enough around the arms.
- Sixteen wrap the spiral closely.

**Variants**

- **Four units.** Enough to curve, not enough to follow the whole spiral.
- **Sixteen units.** Enough capacity to wrap the arms.

**Success:** Reach 90% on the spiral

**Explanation**

Capacity is roughly how complicated a shape the network can express. Too little and it cannot represent the answer at all, however long you train: that is underfitting, and more data will not help. Enough and it finds the structure. Far too much, with too little data, and it starts fitting the noise as well. The interesting point is that all three of these look like "the loss is not where I want it", and only comparing training against validation tells you which one you are in.

**Verified by**

- one hidden unit underfits
- sixteen units fit it

---

## 10. Softmax and cross-entropy

<a id="softmax-and-cross-entropy"></a>

Move from one output to three, and see probabilities that always sum to one.

**Preset**

| | |
| --- | --- |
| architecture | `2-16-16-3` (tanh, softmax) |
| loss | `cce` |
| init | `glorot_uniform` |
| network seed | `11` |
| dataset | `spiral`, 600 samples, noise 0.06, seed `4` |
| training | adam, lr 0.02, batch 32, 500 epochs |

**What to watch**

- Three output units instead of one, each a probability for its class.
- In the Math tab, every row of the output activation sums to exactly 1.
- The boundary now has three regions meeting at the spiral centre.

**Success:** Reach 90% on the three-arm spiral

**Explanation**

With more than two classes the network needs one output per class, and those outputs have to be comparable. Softmax exponentiates each one and divides by the total, which forces them positive and summing to one, so they can be read as probabilities. Cross-entropy then scores how much probability was placed on the correct class. The pair is so natural together that their gradient simplifies to prediction minus target, which is why this app detects the combination and uses the simplified form directly.

**Verified by**

- softmax and cce learn a three-arm spiral
- and every output row sums to 1

---

## 11. Batch size

<a id="batch-size"></a>

Run the same problem at batch 1, 8 and full batch, and read the difference in the loss curve.

**Preset**

| | |
| --- | --- |
| architecture | `2-8-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `4` |
| dataset | `moons`, 200 samples, noise 0.15, seed `6` |
| training | sgd, lr 0.1, batch 1, 150 epochs |

**What to watch**

- At batch 1 the loss curve is a noisy band rather than a line.
- At full batch it is smooth, and each epoch costs one update instead of two hundred.
- The noise is not a defect: it is what lets small batches escape shallow dips.

**Variants**

- **Batch 8.** Eight samples per step. Noticeably calmer.
- **Full batch.** Every sample in one step. The smoothest curve, and the fewest updates.

**Success:** Reach 90% at any batch size

**Explanation**

A batch gradient is an estimate of the true gradient over the whole dataset, and a smaller batch is a noisier estimate. That noise shows up directly as a wobbling loss curve. It is not purely a cost: the randomness can shake the optimiser out of a shallow dip that a full-batch step would settle into. What it does cost is stability, which is why very small batches usually need a smaller learning rate.

**Verified by**

- batch 1 produces a noisier curve than full batch
- full batch is smoother

---

## 12. Optimizer race

<a id="optimizer-race"></a>

Run four optimisers from an identical seed and identical initial weights.

**Preset**

| | |
| --- | --- |
| architecture | `2-12-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `9` |
| dataset | `moons`, 240 samples, noise 0.12, seed `2` |
| training | sgd, lr 0.05, batch 16, 150 epochs |

**What to watch**

- Every run starts from the same weights, because the seed is fixed.
- Adam and RMSProp adapt their step per parameter and usually pull ahead early.
- Plain SGD gets there too, given enough epochs.

**Variants**

- **Momentum.** Accumulates velocity, so consistent directions build speed.
- **RMSProp.** Divides by a running estimate of gradient size, so every parameter moves comparably.
- **Adam.** Momentum and RMSProp together, with bias correction for the first steps.

**Success:** Beat plain SGD at the same epoch count

**Explanation**

Plain SGD takes the same step size in every direction. Momentum accumulates a velocity, so consistent directions accelerate and oscillations cancel. RMSProp divides each step by a running estimate of that parameter’s gradient size, so a parameter with tiny gradients still moves. Adam is both, plus a bias correction that matters for the first few dozen steps because the running averages start at zero. From an identical initialisation the difference is entirely in the update rule.

**Verified by**

- SGD makes progress
- Adam reaches a lower loss in the same epochs

---

## 13. Batch norm, and its two faces

<a id="batch-norm"></a>

Rescue a network that plain SGD cannot train, then find out that the same sample gets two different answers.

**Preset**

| | |
| --- | --- |
| architecture | `2-10-10-10-10-1` (tanh, sigmoid) |
| loss | `bce` |
| init | `glorot_uniform` |
| network seed | `6` |
| dataset | `moons`, 240 samples, noise 0.12, seed `2` |
| training | sgd, lr 0.1, batch 16, 400 epochs |
| feature scale | `[100, 1]` |

**What to watch**

- Four hidden layers, one input scaled by a hundred, and plain SGD. It stalls in the seventies.
- Turn on Batch norm in the Architecture panel. The column captions gain ·bn and the parameter count grows by one γ per hidden unit.
- Step the dissection through a neuron and read the two new lines: the sum is shifted by μ and divided by σ before γ and the bias touch it.
- The loss curve stays jumpy even once the accuracy is good. Nothing is broken: the loss is measured with the running statistics while training uses the batch’s own, so the two disagree by a little every epoch. Watch "best" beside val acc rather than the last number.

**Variants**

- **Normalise between the layers.** One γ per unit, and the layer’s bias moves to after the normalisation, where it plays the part of β.
- **Standardise the inputs instead.** Fixing the scale once, at the door, rather than at every layer. It helps, and it does not do the same job.

**Success:** Reach 90% on data plain SGD stalls on

**Explanation**

Every layer has to learn against inputs whose scale is set by the layer below, and that scale keeps moving while the layer below is still learning. Batch normalisation removes the question: it subtracts the mean and divides by the standard deviation of each unit across the batch, so whatever the layer below does, what arrives has a known spread. Then it hands back two knobs, γ to scale and β to shift, so the network can undo the normalisation if it turns out to want to. Measured here, with one input multiplied by a hundred and four tanh layers stacked on top, plain SGD reaches the low seventies and normalising the same network at the same seed reaches 100%, every seed tried. The subtle part is what happens afterwards. During training a unit is normalised by the statistics of the batch it happens to be in, so its answer depends on which other samples came along with it. That is fine while learning and useless when predicting, so the layer also keeps a running average of those statistics and uses that instead once training is over. The two are not the same numbers, which is why a network can post a different training loss and validation loss on identical data, and why the dissection view tells you which of the two it used. Batch norm also makes the bias redundant, since adding a constant just before subtracting the mean does nothing at all: this implementation moves the bias to after the normalisation rather than carrying a parameter that provably cannot matter.

**Verified by**

- plain SGD stalls on badly scaled, deeply stacked data
- batch norm rescues the same network at the same seed

