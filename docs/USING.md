# Using AwryNN

The [README](../README.md) says what this is. This says what to *do* with it.

It assumes nothing except that you have it running (`npm run dev`). It does not
assume you know what a gradient is. Where a section needs background, it says so
and tells you where to get it.

---

## Contents

1. [Your first ten minutes](#1-your-first-ten-minutes)
2. [The one thing most people skip](#2-the-one-thing-most-people-skip)
3. [A path through the thirteen lessons](#3-a-path-through-the-thirteen-lessons)
4. [How to actually read the screen](#4-how-to-actually-read-the-screen)
5. [Exercises that make it stick](#5-exercises-that-make-it-stick)
6. [Verifying it rather than trusting it](#6-verifying-it-rather-than-trusting-it)
7. [Teaching with it](#7-teaching-with-it)
8. [Habits that get the most out of it](#8-habits-that-get-the-most-out-of-it)
9. [When something looks wrong](#9-when-something-looks-wrong)

---

## 1. Your first ten minutes

Do these in order. Resist the urge to press Train first.

**Minute 1. Look at the shape.** Four columns of circles joined by lines. Each
circle is a neuron. Each line is a *weight*, one stored number. Orange is
positive, blue is negative, and thickness is size. That is the entire visual
language, and the legend at the bottom left of the canvas gives you the scale.

**Minute 2. Hover a line.** The Inspect tab reads out its exact value and its
address, `W[row, col]`. Hover a thick one, then a hairline one. Hairlines are
connections the network has decided not to use.

**Minute 3. Press Train.** Watch the lines thicken and change colour. Watch the
decision boundary in the Inspect tab carve itself around the two moons. Let it
finish. You now have a trained network.

**Minute 4. Press Reset, then Reseed, then Train again.** Same shape, same data,
completely different numbers, and a boundary that solves the problem a different
way. There is no single right answer inside a network. This surprises people and
it is worth sitting with.

**Minutes 5 to 8. Turn on Dissection and press Play.** This is the reason the
app exists. Read section 2.

**Minutes 9 and 10. Break something.** Set the learning rate to `5` and press
Train. Watch the loss explode and the network die. Set it to `0.0001` and watch
nothing happen at all. Put it back to `0.03`.

You now have the whole loop: build, train, look, break, understand.

---

## 2. The one thing most people skip

**Dissection mode.** Everything else in this app is available in some form
elsewhere. This is not.

Press **Dissection: off** to turn it on, then **Play**. One sample walks through
the network. Beside each neuron, a card assembles its arithmetic one term at a
time:

```
Z  LAYER 2 · UNIT 0
   (+0.73)(−0.12)     −0.085
 + (+0.65)(−0.16)     −0.106
 + (−0.52)(+0.42)     −0.216
 + (−0.65)(+0.04)     −0.029
 + (−0.60)(−0.07)     +0.044
 + (−0.10)(+0.12)     −0.012
 + bias               +0.000
z =                   −0.4043
```

Left bracket is a weight. Right bracket is the incoming activation. The right
column is their product. The bottom is the sum, and that sum is the number the
engine actually holds in memory.

**Do this once, properly:** pause, take the numbers off the screen, add them up
on paper or in a calculator, and confirm you get `z`. That five-minute exercise
converts "a neuron computes a weighted sum" from a sentence you have read into a
thing you have done.

Then keep stepping. Use **Next stage** (the ▶ beside Play) rather than letting it
run, and go through all eighteen stages:

| Stages | What is happening |
| --- | --- |
| Input | The sample's raw values enter the first column |
| Weighted sum | Each hidden neuron accumulates its terms, as above |
| Activation | `z` is squashed into `a` by tanh, relu or sigmoid |
| ... repeat per layer ... | |
| Prediction | The output neuron produces its answer |
| Loss | One number scoring how wrong that answer was |
| Backward pass | Right to left, `∂L/∂w = δ·a` for every connection |
| Update | `Δw = −η·∂L/∂w`, and the lines visibly re-weight |

The backward pass is the part almost nobody sees rendered. Watch it more than
once. Notice that the gradient arriving at a connection is the product of two
things you can already see on screen: how wrong the neuron above it was (`δ`) and
how active the neuron below it was (`a`).

**Use Next sample** to step the same machinery over a different input. The weights
do not change; only the activations do.

---

## 3. A path through the thirteen lessons

Open the **Learn** tab. Each lesson expands to a goal, then **Load this lesson**
sets up the architecture, dataset, seed and hyperparameters in one go.

You do not have to do them in order, but this order builds on itself.

### Group A: what a network *is* (lessons 1, 2, 3)

Start here even if you think you know this.

**1. A neuron is a line.** One neuron, one straight boundary. That is the whole
capability of a single unit, and everything else is combinations of it.

**2. XOR needs a hidden layer.** The famous one. Watch a single layer fail
forever. Then the important half: stack *linear* layers instead and watch it fail
just as badly however wide you make it. Depth without a nonlinearity buys nothing
at all, and this is the demonstration.

**3. Zero init never breaks symmetry.** Set every weight to zero and the network
cannot learn, because every hidden unit receives an identical gradient forever
and remains an identical unit. This is why initialization is random. Watch the
weight histogram in Analyse stay a single spike.

### Group B: making training work (4, 8, 11, 12)

**4. Learning rate.** Feel too small, about right, and large enough to destroy
the network. Do all three. The failure at high rates is not gradual.

**8. Feature scaling.** One input multiplied by a hundred, and plain SGD falls
apart. Then fix it with one switch. Also try the "let Adam absorb it" variant,
because it teaches something uncomfortable: adaptive optimizers paper over
scaling problems well enough that one can sit in a pipeline unnoticed for years.

**11. Batch size.** Batch 1, batch 8, full batch. Read the difference in the loss
curve rather than in the final number. Small batches are noisy and that noise is
not purely a cost.

**12. Optimizer race.** Four optimizers from an identical seed and identical
starting weights. Because the seed is fixed, the only variable is the optimizer.

### Group C: how it fails (5, 6, 7, 9)

This is the group that separates people who can debug a network from people who
cannot.

**5. Vanishing gradients.** Eight sigmoid layers. Open **Analyse** and look at the
per-layer gradient chart on its log axis: four decades between the last layer and
the first. Swap the activations to ReLU and watch the spread collapse.

**6. Dead ReLUs.** Push hard enough and units switch off permanently. They are
drawn hollow on the canvas. A ReLU pushed fully negative has a gradient of exactly
zero, so nothing can ever bring it back.

**7. Overfitting.** A large network and far too little data. Watch training loss
fall while validation loss turns around and climbs. That divergence *is* the
definition, and here you watch it happen rather than reading about it.

**9. Capacity versus data.** The same spiral through 1, 4 and 16 hidden units:
underfit, fit, memorise. Three runs, one idea.

### Group D: the machinery underneath (10, 13)

**10. Softmax and cross-entropy.** Move from one output to three and watch
probabilities that always sum to one. Use the **Math** tab here.

**13. Batch norm, and its two faces.** Do this one last. It rescues a network
plain SGD cannot train, and then it teaches the subtler thing: during training a
sample is normalized by the batch it is in, so its answer depends on which other
samples came along with it, while a prediction uses a running average instead.
The same sample gets two different answers. That is why a training loss and a
validation loss can disagree on identical data without anything being wrong.

---

## 4. How to actually read the screen

### The canvas

| What you see | What it means |
| --- | --- |
| Line thickness | Magnitude of the weight |
| Orange line | Positive weight |
| Blue line | Negative weight |
| Hairline | A weight near zero, a connection barely used |
| Filled circle | An active neuron; brightness follows its activation |
| Hollow circle | A dead unit, producing zero for the whole epoch |
| Small `b` satellite | That neuron's bias |
| `·bn` in a column caption | That layer normalizes across the batch |

Scroll to zoom, drag to pan, double-click to fit. If you lose the network
off-screen, double-click.

### The four tabs

**Learn** is the lessons. **Inspect** is the decision boundary, the inspector for
whatever you are hovering, and the editing controls. **Analyse** is the
diagnostics: per-layer gradients, weight histograms, the confusion matrix, and
the gradient check button. **Math** is the live matrices, `W`, `Z`, `A` and `dW`,
with bidirectional highlighting against the canvas.

### The numbers that matter most

- **Train loss versus val loss.** Both falling is learning. Train falling while
  val rises is memorising.
- **Val acc, and "best" beside it.** The big number is the current epoch. On a
  noisy run the best epoch can be far better, so both are shown.
- **Grad norm.** How big a step the network is about to take. Exploding means
  divergence, collapsing to near zero means it has stopped learning.
- **Dead units.** Any number above zero on a ReLU network is worth investigating.

---

## 5. Exercises that make it stick

Reading is not learning. Do these.

### Beginner

1. **Add up a neuron by hand.** Section 2. Do this one first, no exceptions.
2. **Find the most important connection.** Train on moons, then hover for the
   thickest line. Set it to `0` by typing in the inspector. What happens to the
   boundary? Undo, and try setting it to ten times its value instead.
3. **Kill a neuron.** Click a hidden unit, then tick **Ablate layer** in the
   Inspect tab. Its output is forced to zero. Does the network still work? Try it on a network with 8 hidden units, then on one with
   2. Redundancy is a real property and you can feel where it runs out.
4. **Make a straight line curve.** Set every activation to `linear` and train on
   `circles`. It cannot be done. Now change one layer to `tanh`. That is the
   entire argument for activation functions.

### Intermediate

5. **Build XOR by hand.** Load lesson 2, which is already on the `xor` dataset,
   then give yourself a hidden layer and place the weights yourself in the
   inspector. The XOR checker in the Inspect tab tells you when your network
   actually solves all four corners. Aim for weights you can explain out loud.
6. **Predict before you press.** Before each run, write down what you expect the
   loss curve to do. Then run it. The gap between your prediction and the result
   is exactly the thing you did not understand yet, and it is much easier to see
   when you have written it down.
7. **Make a network overfit on purpose, then rescue it.** Load lesson 7 and
   watch validation loss turn around. Now fix it three ways: raise **samples**
   in the Dataset panel, then use the lesson's own **Dropout** and **L2
   regularisation** buttons. Compare how each one shows up in the weight
   histogram in Analyse. L2 has no general slider; it is reached through those
   lesson variants or a saved file.
8. **Find the smallest network that solves the spiral.** Start at 2 hidden units
   and go up. Keep the seed fixed so you are comparing capacity rather than luck.
   Then change the seed and see how much of your answer was luck anyway.
9. **Break the same network four different ways.** Learning rate too high, zero
   init, no nonlinearity, and too little data. Four different failure signatures
   on the loss curve. Learn to tell them apart at a glance, because that is the
   actual skill.

### Advanced

10. **Verify the engine.** Section 6.
11. **Export and run it yourself.** Train something, export NumPy, run it in
    Python, and confirm you get the same prediction to sixteen decimal places.
    You now have a network you built, understand, and can run without this app.
12. **Reproduce a lesson from scratch.** Pick a lesson, note its setup, reset
    everything, and rebuild it by hand from the Architecture and Training panels.
    If you can rebuild it, you understand it.
13. **Use the history scrubber as a microscope.** Train, then drag the history
    slider back through the run. Pin an early epoch as **A** and watch `Δw`
    against it. Which layer moved most? Usually the last, and now you can see by
    how much.
14. **Make batch norm's two faces visible.** Load lesson 13 with normalization
    on, train it, then step the dissection through one neuron. The card names the
    statistics it used and says they came from the running estimate rather than a
    batch. Change the batch size and reason about what changes and what does not.

---

## 6. Verifying it rather than trusting it

This is a teaching tool making a strong claim: every number it shows is a number
its engine computed. You should not take that on faith, and you do not have to.

**Press Check gradients.** It is in the **Analyse** tab. It nudges each parameter
up and down, measures how the loss actually responds, and compares that against
the gradient backpropagation produced. Agreement to 1e-7 or better is a pass; you
will usually see 1e-10 or 1e-11.

That single button is the whole argument. If the numerically measured gradients
match the analytically derived ones to ten decimal places, then backpropagation
here is right, and everything downstream of it is too.

It works on *your* network, whatever you have built.

**Check the dissection yourself.** Add up the terms on a card and confirm you get
`z`. The app checks this too, on every unit, every time, and calls the difference
the *residual*.

**Check determinism.** Note the seed, train, note the final loss to four decimal
places. Press Reset, then Train again. Identical. Nothing here is random once the
seed is fixed.

**Run the test suite.** `npm test` runs 691 tests. `npm run gradcheck` runs just
the correctness gate: 60 network configurations, 2,379 parameters, worst relative
error 1.09e-9.

If you want the reasoning rather than the result, [`docs/MATH.md`](MATH.md) maps
every equation to the function that implements it.

---

## 7. Teaching with it

### A single 50-minute class

| Time | What |
| --- | --- |
| 0 to 5 | The shape on screen. Neurons, weights, colour, thickness |
| 5 to 15 | Dissection, one sample, forward only. Add up one neuron together |
| 15 to 25 | Loss, then the backward pass. `∂L/∂w = δ·a`, on screen |
| 25 to 35 | Train it. Watch the boundary form. Reseed and watch a different solution |
| 35 to 45 | Break it: lesson 4 at a high learning rate, then lesson 2 |
| 45 to 50 | Press Check gradients. Everything you saw was real |

### Setting work

Every network is a URL. **Inspect → Share → Copy link** gives you a link that
carries the architecture, the dataset, every hyperparameter and the weights.

- Set an exercise by sending a link to a *broken* network and asking what is
  wrong with it.
- Collect answers as links back.
- A link to an untrained network is short and carries no weights at all, because
  the seed regenerates them exactly.

Nothing is uploaded anywhere. The state travels in the part of the URL that
browsers never send to a server.

### For self-study

Work through section 3 in order, doing the section 5 exercises for each group
before moving on. Budget an hour per group. Group C is the one to spend longest
on, because recognising failure modes is the part that transfers to real work.

---

## 8. Habits that get the most out of it

**Change one thing at a time.** The seed is fixed for a reason. If you change the
architecture and the optimizer together, you have learned nothing about either.

**Watch the curve, not the number.** Final accuracy is a summary. The shape of
the loss curve tells you *why*, and shapes are what you learn to recognise.

**Use the log axis.** The loss chart has a linear/log toggle. Late training looks
flat on a linear axis and detailed on a log one. The per-layer gradient chart in
Analyse is log by default, because on a linear axis a layer receiving 1e-6 and one
receiving 1e-1 both render as "on the bottom".

**Say what you expect first.** Out loud, or written down. Every time.

**Pause and hover.** Training does not have to be running for you to inspect
things. Pause, then explore.

**When a run surprises you, scrub back.** The history slider replays the run.
Find the epoch where it went wrong rather than guessing from the end state.

**Read the explanation after, not before.** Every lesson's explanation is
available whenever you want it, but the phenomenon lands harder if you have
watched it happen first and formed your own theory.

---

## 9. When something looks wrong

| What you see | What is going on |
| --- | --- |
| Loss goes to a huge number or `NaN` | Learning rate too high. The app reports divergence as a state rather than hiding it |
| Loss flat from the start | Learning rate too low, zero init, or all activations dead |
| Boundary is a straight line whatever you do | Every activation is `linear`; depth cannot help |
| Val loss rises while train loss falls | Overfitting. More data, dropout, L2, or a smaller network |
| Hollow circles appear and stay | Dead ReLUs. Lower the rate or switch to leaky ReLU |
| Trained network, but the loss chart is empty | You opened a shared link. Weights travel in a link; loss curves do not |
| Loss stays jumpy with batch norm on | Expected. The loss is measured with running statistics while training uses the batch's own. Watch "best" beside val acc |
| Numbers you cannot explain | Press Check gradients, then read [`docs/MATH.md`](MATH.md) |

If something genuinely looks like a bug, it is worth reporting: the engine is
covered by 691 tests and a gradient check to 1e-9, so a real disagreement would
be interesting.

---

## Where next

- [Start here](../public/guide/start-here.html) explains the concepts from zero,
  in the browser, using the app's own numbers.
- [Handbook](../public/guide/handbook.html) covers every control, the conventions
  the maths uses, and how to check the arithmetic against your own.
- [`docs/MATH.md`](MATH.md) is the normative reference: every equation mapped to
  the function that implements it.
- [`docs/LESSONS.md`](LESSONS.md) is all thirteen lessons in full, generated from
  the same data the app runs.
