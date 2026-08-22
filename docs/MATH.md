# MATH.md

Every equation implemented by AwryNN, mapped to the function and file that
implements it. Spec §4 makes this document **normative**: if the code and this
file disagree, one of them is a bug.

**Status: Phase 2 complete.** §§1–12 are implemented and tested. Codegen (§8 of
the spec) arrives in Phase 7 and is listed at the end rather than described
speculatively.

---

## 1. Conventions (`tensor.ts`)

Row-major batch convention, fixed by spec §4.1 and not negotiable. Mixing
conventions is the single most common source of backprop bugs.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `X` | `[B, n₀]` | batch input, one **row** per sample |
| `W⁽ˡ⁾` | `[n_{l-1}, n_l]` | weights of layer *l* |
| `b⁽ˡ⁾` | `[1, n_l]` | biases, broadcast down the batch |
| `A⁽⁰⁾ = X` | `[B, n₀]` | input activations |
| `Z⁽ˡ⁾`, `A⁽ˡ⁾` | `[B, n_l]` | pre-activations and activations |
| `A⁽ᴸ⁾ = Ŷ` | `[B, n_L]` | predictions |

Every matrix is a `Float64Array` with explicit `rows`/`cols`
(`Matrix`, [`tensor.ts`](../src/engine/tensor.ts)). float64 throughout, because these
networks have tens of parameters, so there is no memory argument for float32,
and the gradient check needs the headroom.

Shape assertions are on by default and controlled by `setShapeAssertions()`.
They throw with **both** operand shapes in the message. The engine cannot read
`import.meta.env` or `process.env` (it compiles with `types: []` and must run in
bare Node), so the host decides rather than the bundler.

| Operation | Function |
| --- | --- |
| `C = A · B` | `matmul` |
| `C = Aᵀ · B` | `matmulAT` |
| `C = A · Bᵀ` | `matmulBT` |
| `Aᵀ` | `transpose` |
| `A + 1_B · v` | `addRowVector` |
| `colSum(A)` | `colSum` |
| `Σ aᵢⱼ²` | `sumSquares` |

`matmulAT` and `matmulBT` exist as separate entry points rather than
`matmul(transpose(a), b)` because the backward pass would otherwise allocate a
transposed matrix per layer per step purely to read it once.

---

## 2. Forward pass (`layers.ts`, `network.ts`)

```
Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ · W⁽ˡ⁾ + 1_B · b⁽ˡ⁾
A⁽ˡ⁾ = φ⁽ˡ⁾(Z⁽ˡ⁾)
```

`DenseLayer.forward(aPrev, training)`: `matmul` then `addRowVector` then
`applyActivation`. `Network.forward(x, training)` chains them.

`A⁽ˡ⁻¹⁾` and `Z⁽ˡ⁾` are cached per layer when `training` is true; backprop needs
both. Inference drops `A⁽ˡ⁻¹⁾` immediately.

---

## 3. Backward pass (`layers.ts`, `network.ts`)

Per-sample loss `ℓ_b`; batch loss `L = (1/B) Σ_b ℓ_b`.

```
dA⁽ᴸ⁾  = ∂ℓ/∂A⁽ᴸ⁾                  unaveraged, per-sample, [B, n_L]
dZ⁽ˡ⁾  = dA⁽ˡ⁾ ⊙ φ′⁽ˡ⁾(Z⁽ˡ⁾)        elementwise
dW⁽ˡ⁾  = (A⁽ˡ⁻¹⁾)ᵀ · dZ⁽ˡ⁾ / B      [n_{l-1}, n_l]
db⁽ˡ⁾  = colSum(dZ⁽ˡ⁾) / B          [1, n_l]
dA⁽ˡ⁻¹⁾ = dZ⁽ˡ⁾ · (W⁽ˡ⁾)ᵀ           [B, n_{l-1}]
```

**The divide-by-B convention.** Unaveraged per-sample gradients propagate
through the network; the division by `B` happens **exactly once**, when the
parameter gradients are formed. `dA⁽ˡ⁻¹⁾` is deliberately *not* divided. Divide
there too and every gradient picks up an extra `1/B` per layer, which presents
as "training is just slow" rather than as a bug.

| Equation | Function |
| --- | --- |
| `dZ = dA ⊙ φ′(Z)` then the tail | `DenseLayer.backwardFromDA` |
| the shared tail (`dW`, `db`, `dA⁽ˡ⁻¹⁾`) | `DenseLayer.backwardFromDZ` |
| output-layer dispatch and the loop | `Network.backward` |

Guarded by `network.test.ts` → "divides by B exactly once": a batch of two
identical rows must produce the same gradients as a batch of one. For any
network deeper than a single layer, that invariant fails if `dA⁽ˡ⁻¹⁾` is
averaged.

### 3.1 Fused output-layer gradients

```
dZ⁽ᴸ⁾ = Ŷ − Y
```

`Network.outputGradientMode` detects the two pairings explicitly and
`fusedOutputGradient` ([`losses.ts`](../src/engine/losses.ts)) applies them.

**sigmoid + bce:**
`∂ℓ/∂â = (â−y)/(â(1−â))` and `σ′(z) = â(1−â)`, so
`dZ = ∂ℓ/∂â · σ′(z) = â − y`.

**softmax + cce:**
`∂ℓ/∂â_k = −y_k/â_k` and `∂â_k/∂z_j = â_k(δ_kj − â_j)`, so
`dZ_j = Σ_k (−y_k/â_k)·â_k(δ_kj − â_j) = −y_j + â_j Σ_k y_k = â_j − y_j`,
using `Σ_k y_k = 1` for one-hot `Y`.

Both cancellations remove a division by a quantity that goes to zero exactly
when the network is confident, so the fused path is the numerically stable one
as well as the fast one.

`losses.test.ts` verifies **both** identities numerically, the softmax case
against the full dense Jacobian, rather than trusting the derivation.

---

## 4. Activations (`activations.ts`)

Implemented as `{ f(z), df(z, a) }` pairs; passing the computed activation `a`
into `df` lets sigmoid and tanh reuse it.

| Name | φ(z) | φ′(z) | Range |
| --- | --- | --- | --- |
| `linear` | `z` | `1` | unbounded |
| `relu` | `max(0, z)` | `z > 0 ? 1 : 0` (**0 at z = 0**) | unbounded |
| `leaky_relu` | `z > 0 ? z : αz`, α = 0.01 | `z > 0 ? 1 : α` (**α at z = 0**) | unbounded |
| `tanh` | `tanh(z)` | `1 − a²` | `[−1, 1]` |
| `sigmoid` | see below | `a(1 − a)` | `[0, 1]` |
| `softmax` | row-wise, see below | fused only | `[0, 1]` |

**Sigmoid stability (mandatory, §4.4).** `sigmoid()`:

```
z ≥ 0:  1 / (1 + exp(−z))
z < 0:  e = exp(z);  e / (1 + e)
```

Neither branch ever exponentiates a positive argument. The naive form evaluates
`exp(−z)` for large negative `z`, and `exp(800)` is `Infinity` in float64.

**Softmax stability (mandatory, §4.4).** `softmaxRows()`:

```
s_j = exp(z_j − max_k z_k) / Σ_k exp(z_k − max_k z_k)
```

Subtracting the row max is algebraically a no-op, since the shared factor cancels,
but it caps the largest exponent at `exp(0) = 1`, so a logit of 1000 yields a
probability instead of `Infinity/Infinity = NaN`.

Softmax is legal **only** as the final activation and **only** with `cce`.
`validateConfig()` rejects anything else with a written explanation; there is no
silent fallback.

---

## 5. Losses (`losses.ts`)

`loss()` returns the batch mean `L = (1/B) Σ_b ℓ_b`. `dA()` returns the
**unaveraged** per-sample `∂ℓ/∂â`.

| Name | `ℓ` (per sample) | `∂ℓ/∂â` | Valid output |
| --- | --- | --- | --- |
| `mse` | `½ Σ_j (â_j − y_j)²` | `â_j − y_j` | linear (or any) |
| `bce` | `−[y log â + (1−y) log(1−â)]` | fused `dZ = â − y` | sigmoid, 1 unit |
| `cce` | `−Σ_k y_k log â_k` | fused `dZ = â − y` | softmax, K units |

Every logarithm is clamped: `log(max(x, LOG_EPSILON))` with
`LOG_EPSILON = 1e-12`. An unclamped `log(0)` is `−Infinity`, which turns the
reported loss into `NaN` and destroys the loss curve even when the gradients are
fine. Labels for `cce` are one-hot.

The unfused `dA` for `bce` and `cce` is implemented even though the network
always takes the fused path, so the test suite can check the cancellation and
gradcheck can exercise the general route.

**Reported quantities**, kept distinct per §4.5:

| Quantity | Function |
| --- | --- |
| data loss `L` | `Network.dataLoss` |
| L2 term `(λ/2)·Σ‖W‖²` | `Network.l2Penalty` |
| total objective | `Network.objective` |

---

## 6. Initialization (`init.ts`)

`fan_in = n_{l-1}` (rows of `W`), `fan_out = n_l` (cols of `W`). Biases default
to zeros.

| Scheme | Distribution |
| --- | --- |
| `glorot_uniform` | `U(−limit, limit)`, `limit = sqrt(6 / (fan_in + fan_out))` |
| `he_normal` | `N(0, sqrt(2 / fan_in))`, the default for ReLU |
| `lecun_normal` | `N(0, sqrt(1 / fan_in))` |
| `normal(σ)` | `N(0, σ)` |
| `uniform(a, b)` | `U(a, b)` |
| `zeros` | `0`, present so lesson 3 can demonstrate unbroken symmetry |
| `constant(c)` | `c` |

Normals come from Box–Muller on the seeded `init` stream.

---

## 7. Determinism (`rng.ts`)

`mulberry32` seeded from a `splitmix32`-expanded user seed. `Math.random()` does
not appear anywhere in `src/engine/` and the ban is enforced three ways
(see [ARCHITECTURE.md](ARCHITECTURE.md)).

The named streams `init`, `shuffle`, `dropout` and `data` each get their own
`mulberry32` state derived from `splitmix32(seed ^ FNV1a(name))`. Streams are
therefore independent *and* order-insensitive: exercising `dropout` 500 times
does not shift what `init` produces, and requesting them in a different order
changes nothing. `rng.test.ts` asserts both.

**Why splitmix32 first:** a raw seed like `1` has almost no entropy in its high
bits, and feeding that straight to `mulberry32` makes nearby seeds produce
visibly correlated first draws. Learners type 1, 2, 3.

**Box–Muller discards its second normal** rather than caching it. Caching would
make stream state a pair `(counter, pending)`, and every `clone()`/serialize
path would have to carry that flag or silently desynchronise.

Same seed + same config + same steps ⇒ **bitwise identical** parameters, tested
over 100 steps in `network.test.ts` by comparing the raw bytes.

---

## 8. L2 regularization (`layers.ts`, `network.ts`)

```
objective = L + (λ/2)·Σ‖W‖²
dW ← dW + λ·W                    applied AFTER the batch average
```

`DenseLayer.applyL2(λ)` and `Network.l2Penalty()`. **Never applied to biases**,
`applyL2` only touches `W`, and `network.test.ts` asserts that `db` is
bit-identical with λ = 0 and λ = 0.5.

Spec §4.9 files L2 under training (Phase 2), but the Phase 1 gate requires the
gradient-check matrix to cover `{with, without L2}`, so the objective term is
implemented here.

---

## 12. Gradient checking (`gradcheck.ts`)

```
g_num[i] = (L(θ + ε·e_i) − L(θ − ε·e_i)) / (2ε)
relErr   = |g_num − g_ana| / max(1e-9, |g_num| + |g_ana|)
```

`gradientCheck(net, x, y, options)`. Three things cause false failures, and all
three are handled explicitly:

1. **A moving batch.** The comparison uses one fixed batch; resampling between
   the analytic and numerical passes compares two different functions.
2. **Dropout left on.** Dropout makes `L` stochastic, so `L(θ+ε)` and `L(θ−ε)`
   come from different masks and the difference is noise. Perturbation passes
   run in eval mode.
3. **The L2 term omitted from `L`.** With λ > 0 the analytic `dW` includes
   `λ·W`, so the numerical side must differentiate the **full** objective.
   `gradientCheck` calls `Network.objective`, not `dataLoss`.

### How the step size is chosen

Spec §4.11 names `ε = 1e-5`, exported as `SPEC_EPSILON` and honoured when passed
explicitly as `options.epsilon`. It is **not** the default.

A plain central difference carries two competing errors:

```
D(h) = (L(θ+h) − L(θ−h)) / 2h  =  g + c₂h² + c₄h⁴ + …    truncation, ∝ h²
                                    ± |L|·μ / 2h          roundoff,   ∝ 1/h
```

Shrinking `h` kills truncation and amplifies roundoff, so total error is
U-shaped and bottoms out at an `h` that depends on the ratio `|L| / |g|`, which
varies per coordinate and per network. No single constant is right everywhere.

Measured on the hardest case in the matrix (2-8-6-3, sigmoid, mse, seed 1056),
at the coordinate with the smallest gradient in the network
(`|g| = 1.03e-4` against `L = 0.74`):

| h | relative error |
| --- | --- |
| 1e-2 | 5.10e-5, truncation dominates |
| 1e-3 | 5.11e-7 |
| **1e-4** | **2.23e-9**, the floor for this coordinate |
| 1e-5 | 5.41e-8, the spec's ε; roundoff has already taken over |
| 1e-6 | 4.30e-7 |
| 1e-7 | 4.13e-6, roundoff dominates |

At the spec's ε the worst coordinate in the full matrix reaches **1.44e-7** and
fails a 1e-7 threshold, not because the gradient is wrong, but because float64
cannot represent the difference of two numbers near 0.74 finely enough to
resolve a gradient of 1e-4 at that step size.

**The fix is Richardson extrapolation, not a luckier constant.** Combining two
step sizes removes the `c₂h²` term outright:

```
R(h) = (4·D(h/2) − D(h)) / 3          error now O(h⁴)
```

With truncation gone, `h` can stay **large**, where roundoff is negligible,
instead of being driven down into the cancellation regime. Three pairs are
evaluated, `DEFAULT_STEP_PAIRS = [(1e-2, 5e-3), (1e-3, 5e-4), (1e-4, 5e-5)]`,
and every coordinate is scored on its best candidate, with the plain central
differences kept as fallbacks.

Measured worst case across the whole §4.11 matrix:

| estimator | worst relErr | headroom under the 1e-7 gate |
| --- | --- | --- |
| single central difference at the spec's 1e-5 | 1.44e-7 | **fails** |
| best of five plain central differences | 1.28e-8 | 7.8× |
| **Richardson + central fallbacks (implemented)** | **1.09e-9** | **92×** |

Taking the best candidate **cannot launder a wrong gradient**: an incorrect
analytic value disagrees with every candidate at every step size, since there is
no `h` at which a wrong number becomes right. `gradcheck.test.ts` proves this by
perturbing one gradient by 1% and asserting failure under every individual step
size, every Richardson pair in isolation, and the full default search.

The fallbacks matter for ReLU, which is piecewise linear: away from the kink a
plain central difference is already exact, and Richardson's smoothness
assumption buys nothing. The method self-selects: across the matrix Richardson
won 1610 coordinates and plain differences won 769, with ReLU networks
preferring the latter almost everywhere. `GradCheckResult.methodCounts` reports
the split.

`fine = coarse / 2` is validated at call time, because the 4/3 and −1/3 weights
are only correct at that ratio and a wrong one produces a plausible-looking but
biased estimate.

### ReLU kinks

ReLU and LeakyReLU are non-differentiable at zero, so a perturbation can
straddle the kink and produce a meaningless finite difference. `kinkSignature()`
records the sign pattern of every `Z` in kinked layers at both the `θ+h` and
`θ−h` passes; if any unit changed sides, that **step size** is discarded and the
others tried. A coordinate is skipped only when every step straddles, and the
skip count is reported. This is exact rather than a `|z| < threshold` heuristic,
and costs nothing because both passes already exist. The signature buffer is
reused across evaluations, so the check allocates nothing per coordinate.

### The matrix

`{linear, relu, leaky_relu, tanh, sigmoid} × {mse, bce, cce} × {with, without
L2} × {2-1, 2-4-1, 2-8-6-3}`, skipping invalid combinations: 60 cases, 2379
parameters checked, 1 skipped at a kink.

The output activation is dictated by the loss (§4.5), so the sweep covers both
fused paths and the general one. Worst relative error across the whole matrix:
**1.09e-9** against a **1e-7** gate, 92× headroom.

---

## 9. Optimizers (`optimizers.ts`)

All operate on the flat `Network.params` / `Network.grads` arrays. `t` is a
global step counter incremented **once per update**, before the parameter loop,
because Adam's bias correction reads it and expects `t = 1` on the first update.
The learning rate is passed per step rather than stored, since schedules change
it every epoch.

| Optimizer | Update |
| --- | --- |
| `sgd` | `θ ← θ − η·g` |
| `momentum` | `v ← μv + g` ; `θ ← θ − ηv` (μ = 0.9) |
| `nesterov` | `v ← μv + g` ; `θ ← θ − η(g + μv)` |
| `rmsprop` | `s ← ρs + (1−ρ)g²` ; `θ ← θ − ηg/(√s + ε)` (ρ = 0.9, ε = 1e-8) |
| `adam` | `m ← β₁m + (1−β₁)g` ; `v ← β₂v + (1−β₂)g²` ; `m̂ = m/(1−β₁ᵗ)` ; `v̂ = v/(1−β₂ᵗ)` ; `θ ← θ − ηm̂/(√v̂ + ε)` |
| `adamw` | as Adam, then `θ ← θ − η·λ·θ` (decoupled) |

**Momentum uses the accumulate-then-scale variant**, pinned by §4.8. The
alternative `v ← μv − ηg ; θ ← θ + v` folds η into the velocity, rescaling the
effective learning rate by `1/(1−μ)`; an LR that works for SGD would then
diverge. Lesson 12 races these at a shared LR, so the difference is visible in
the product.

**Bias correction is not optional.** `m` and `v` start at zero, so without the
`1/(1−βᵗ)` factors the first steps are biased toward zero. At `t = 1` the
corrected update reduces to exactly `−η·sign(g)`, independent of `|g|`, and that
clean cancellation is the signature of correct bias correction, and
`optimizers.test.ts` pins it there.

**Freezing uses a skip mask, not a zeroed gradient.** A momentum or Adam state
carries a parameter forward on accumulated velocity even when its gradient is
zero, so zeroing would let a "frozen" parameter drift. `Optimizer.step` takes a
`skip: Uint8Array` and leaves those coordinates completely untouched.

### Gradient clipping

`clipGradientsByNorm(grads, c)`: if `‖g‖ > c`, scale **all** gradients by
`c/‖g‖`. Global rather than per-coordinate, so the gradient's direction is
preserved; clipping each coordinate independently would bend it.

---

## 10. Regularization and training mechanics (`layers.ts`, `regularizers.ts`, `schedules.ts`, `trainer.ts`)

Each piece lives where it can least easily go wrong.

### Dropout (inverted, and owned by the layer)

```
train:  m ~ Bernoulli(1−p)/(1−p),  A_out = A ⊙ m,  dA ← dA ⊙ m  (before φ′)
eval:   identity
```

Three details, each a real bug if missed:

1. **The mask multiplies `dA` before `φ′`, not `dZ` afterwards.** Applying it
   after the backward pass would leave every upstream layer's gradient unmasked.
2. **The cached `A` stays pristine.** `df(z, a)` reads `a` for tanh and sigmoid,
   so multiplying the mask into `A` would corrupt `φ′`. The masked output goes
   to a separate buffer; `layer.A` is always `φ(Z)` and `layer.output` is what
   the next layer saw.
3. **"Inverted" means the `1/(1−p)` rescaling happens at training time.** That
   is what keeps inference a plain forward pass.

Dropout lives on `DenseLayer` rather than the trainer so forward and backward
share one mask buffer and cannot drift apart. Never applied to the output layer:
dropping outputs corrupts the loss itself, and with a fused softmax/CCE head it
would break the `Ŷ − Y` identity. `gradientCheck` disables it outright for the
duration of a check and restores it afterwards.

Ablation (§6.5) reuses the same machinery as a permanent all-zero mask.

### Batch normalization (and why there is no β)

```
train:  mu_j  = (1/B) Σ_i U_ij
        var_j = (1/B) Σ_i (U_ij − mu_j)²            biased: makes X̂ unit-variance
        X̂     = (U − mu) / sqrt(var + eps)
        Z     = gamma ⊙ X̂ + b

        mu_hat  ← (1−m)·mu_hat  + m·mu
        var_hat ← (1−m)·var_hat + m·(1/(B−1)) Σ_i (U_ij − mu_j)²   unbiased

eval:   X̂ = (U − mu_hat) / sqrt(var_hat + eps)
```

with `U = A^{l-1} · W` (no bias yet), `eps = 1e-5`, `m = 0.1`. `BATCH_NORM_EPSILON`
and `BATCH_NORM_MOMENTUM` in `layers.ts`.

**The two variances are different on purpose.** The one that normalizes the
batch is biased, dividing by `B`, because that is what makes `X̂` have unit
variance within the batch. The one fed to the running average is unbiased,
dividing by `B−1`, because there it is estimating a population rather than
describing a sample. PyTorch draws the same distinction.

**There is no β, because the layer's own bias is β.** Adding a constant just
before a step that subtracts the mean does nothing: shift every `u` in a column
by `δ` and `mu` shifts by `δ` too, so `U − mu` is unchanged and `∂L/∂b` is
identically zero. Rather than carry a parameter that provably cannot matter, the
existing bias MOVES, from before the activation to after the normalization,
which is the position β occupies. `Z` is still the true pre-activation, so
`φ′(Z, A)` is untouched and the dead-unit and saturation metrics are unaffected.

Backward, with `dX̂ = dZ ⊙ gamma` and `s = 1/sqrt(var + eps)`:

```
dgamma_j = (1/B) Σ_i dZ_ij · X̂_ij
db_j     = (1/B) Σ_i dZ_ij
dU_i     = (s/B) · ( B·dX̂_i − Σ_k dX̂_k − X̂_i · Σ_k dX̂_k·X̂_k )
```

The two sums are the point: every sample's `u` affects every other sample's `z`,
through `mu` and through `var`. The middle term is the mean's share, the last is
the variance's. Both vanish in the eval-mode case, where `mu_hat` and `var_hat`
are constants this batch had no part in, leaving `dU = dZ ⊙ gamma · s`.

The `1/B` in `dU` is the statistics' own, from `∂mu/∂u_i = 1/B`. It is NOT the
loss-averaging division, which still happens exactly once, later, at `dW`.

**A batch of one falls back to the running statistics**, in training too.
Normalizing a single sample by its own statistics maps every unit to exactly
zero and cuts the gradient at that layer, silently, and the last batch of an
epoch can easily hold one sample.

**Storage.** `gamma` joins the parameters, per layer `[W | b | gamma]`, with
`gamma` empty where a layer does not normalize; the running statistics are NOT
parameters and live in a parallel `Network.buffers`, since no optimizer, weight
decay or gradient clip should touch an estimate of a mean. Both must travel
together whenever a network moves: `captureState`/`restoreState`.

**Two hazards the gradient check hit**, both documented in `gradcheck.ts`:
perturbation passes must use the same `training` flag as the analytic pass,
because train and eval are genuinely different functions here; and relative
error is meaningless for the ~1e-8 gradients that `B = 2` produces, so a
coordinate failing the relative test is rescued when the two values agree to
within `ABSOLUTE_TOLERANCE = 1e-10`.

### Batching and splits

- Shuffled every epoch from the `shuffle` stream.
- **The final partial batch is kept** and averaged over its true size. The batch
  buffers are allocated once at full size and short batches are `rowView`s into
  them, so the `/B` in backprop divides by the real count.
- Splits are **stratified** for classification: each class is shuffled and split
  independently, so a small 3-class problem cannot produce a validation set
  missing a class.

### Standardization

`(x − μ)/σ`, with **μ and σ fitted on the training split only**. A constant
feature keeps `σ = 1` rather than dividing by zero. `leakStandardization` fits on
all data instead, the deliberate leak lesson 8 exposes.

### LR schedules

`learningRateAt(config, base, epoch)` is a **pure function of the epoch**, which
matters because the history scrubber replays by epoch index and a stateful
schedule would produce a different curve on replay.

| Schedule | Rate |
| --- | --- |
| `constant` | `η` |
| `step(drop, every)` | `η·drop^⌊epoch/every⌋` |
| `exponential(γ)` | `η·γ^epoch` |
| `cosine(T, min)` | `min + (η−min)·½(1 + cos(π·min(epoch/T, 1)))` |
| `warmup(n)` | multiplies the above by `(epoch+1)/n` while `epoch < n` |

Cosine holds at its floor past `T` rather than climbing back, because a restart should
be an explicit choice. Warmup starts at `η/n`, not 0, because a genuinely zero
first step is indistinguishable from a hung trainer.

### Early stopping

Patience on validation loss, restoring the best weights and recording the stop
epoch for the loss chart.

---

## 11. Metrics (`trainer.ts`)

| Metric | Notes |
| --- | --- |
| train/validation loss | data loss, reported separately from the objective |
| accuracy | argmax for K>1, threshold 0.5 for a single sigmoid |
| precision / recall / confusion | `buildReport`; a never-predicted class reports 0 rather than NaN |
| per-layer gradient L2 norm | `layerGradientNorms`, log-scaled, which makes vanishing gradients undeniable |
| dead units | zero activation across the **entire epoch**, counted only for ReLU-family activations (a tanh unit at 0 is centred, not dead) |
| saturation | within 1% of a bounded activation's range |

A divergence (`NaN`/`Infinity` in the predictions or the gradient norm) is
reported as a **state**, not thrown: `StopReason` becomes `'diverged'` and the
epoch loop stops. §7.4 requires the app to recover gracefully and say what to do.

One measured behaviour worth recording, because it is easy to rediscover the
hard way: **a `tanh` hidden layer with a fused sigmoid/BCE output cannot diverge
to NaN**, at any learning rate. `dZ = ŷ − y` is bounded in `[−1, 1]` and tanh
saturates, so the network stalls instead of exploding. Demonstrating a
divergence needs ReLU or an unbounded regression head.

---

## Not implemented

Everything §4 specifies is implemented and tested. Of §11 Phase 9's optional
extensions, batch normalization (above) and URL-encoded shareable state
(`state/shareLink.ts`) were built; a convolution demo, a recurrent network and
WebGPU acceleration were declined, with reasons in `README.md`.
