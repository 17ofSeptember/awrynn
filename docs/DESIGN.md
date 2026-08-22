# DESIGN.md

The token system, written before the CSS. Spec §9.

## The subject

Laboratory instrumentation: a signal analyzer, an oscilloscope, a bench meter.
Precise, dense, calm. Information hierarchy carried by **weight and spacing**,
not by boxes and shadows. The canvas is the instrument face; the panels are its
chassis.

Three looks were explicitly ruled out by §9 as defaults rather than decisions,
and none of them is used here: warm-cream + serif + terracotta; near-black +
acid-green; hairline-rule broadsheet.

---

## 1. The two poles

**The diverging weight scale is the visual core of the entire product**, so it
was chosen first and everything else was derived to sit quietly beneath it.

| Role | Token | Hex | Contrast on canvas |
| --- | --- | --- | --- |
| Negative weight | `weight-neg` | `#3EC5E8` | 9.54:1 |
| Positive weight | `weight-pos` | `#F2A33C` | 9.28:1 |

A cool cyan against a warm amber. The pairing is not decorative, it is the
oscilloscope/CRT phosphor register the whole app is dressed in, and it survives
the two constraints that matter:

**Legible at 0.5px stroke.** Both clear 9:1 against the canvas face, so a
hairline edge carrying a near-zero weight is still visible rather than guessed
at.

**Colour-vision safe on its own.** Measured OKLab separation between the poles,
under simulation:

| Vision | Negative | Positive | ΔE | Contrast vs canvas |
| --- | --- | --- | --- | --- |
| typical | `#3EC5E8` | `#F2A33C` | 0.261 | 9.54 / 9.28 |
| protanopia | `#787AE0` | `#D0CF55` | 0.359 | 5.21 / 11.73 |
| deuteranopia | `#7167DD` | `#D4DA5B` | 0.425 | 4.32 / 12.87 |
| tritanopia | `#45D9D7` | `#EE696D` | 0.312 | 11.20 / 6.33 |

Separation **increases** under all three deficiencies rather than collapsing,
which is why this pairing was chosen over the obvious blue/red. Red/green was
never a candidate. Colourblind-safe mode adds a dash pattern for negative
weights on top of this, per §6.2. Belt and braces, not a rescue.

`src/render/__tests__/theme.test.ts` recomputes every number in the two tables
above and fails if a palette edit erodes them.

### Encoding magnitude

Hue carries **sign**; alpha and stroke width carry **magnitude**.

```
strong negative ── cyan, opaque, thick
  weak negative ── cyan, faint, thin
          zero  ── invisible          <- pruning becomes visible (§6.2)
  weak positive ── amber, faint, thin
strong positive ── amber, opaque, thick
```

Both ramps are precomputed as 32 quantized `rgba()` strings at module load, so
drawing an edge is an array index rather than a string concatenation. §6.1
requires zero allocation in the hot path.

### Class identity is a different job

The decision boundary paints class **identity**, not polarity, so for three or
more classes it needs a categorical palette rather than a reading of the
diverging scale.

The first implementation interpolated between the two poles. It was measurably
wrong. Run through a palette validator, the midpoint landed at OKLCH chroma
**0.057**, which reads as gray and stops doing identity work at all, and sat
**ΔE 12.8** from the cool pole under *normal* vision, below the threshold at
which full-colour readers can distinguish a pair.

The replacement slots were searched against the validator rather than chosen by
eye. The decisive finding: **colour-vision deficiency destroys hue but preserves
lightness**, so every candidate placed at the poles' own lightness collapsed.
A violet that looked obviously distinct measured **ΔE 2.5** against cyan under
deuteranopia. The added slots therefore step *down* in lightness, which is what
makes them separable, and has the useful consequence that they can never
outshine the poles.

| Slot | Colour | Role |
| --- | --- | --- |
| 0 | `#3EC5E8` | the cool pole |
| 1 | `#F2A33C` | the warm pole |
| 2 | `#984260` | |
| 3 | `#3C6CDC` | |
| 4 | `#C86830` | |

Validated **all pairs**, not merely adjacent ones, because any two regions of a
decision boundary can touch. Worst-case separation against the canvas surface:

| Classes | CVD ΔE | Normal ΔE |
| --- | --- | --- |
| 2 | 21.4 | 26.1 |
| 3 | 21.4 | 26.1 |
| 4 | 20.3 | 23.4 |
| 5 | 15.9 | 16.0 |

against a target of 8 and a hard normal-vision floor of 15.

Slots 0 and 1 **are** the weight poles, so a binary boundary speaks the same
visual language as the weights, and only genuinely multi-class problems
introduce further hues.

### A documented deviation

The two poles sit at OKLCH lightness 0.766 and 0.776, above the 0.48–0.67 band a
general-purpose validator expects for dark mode. That band assumes a
conventional dark surface; this one is `#0A0E14`, far darker, and the poles
clear 9.5:1 against it. The band is a proxy for the properties that actually
matter, and those are measured directly above. The palette is not repainted to
satisfy a proxy.

---

## 2. Surfaces

Nothing in this column is allowed to compete with the poles: every surface is a
desaturated blue-slate, and none of them carries a hue you would name.

| Token | Hex | Use |
| --- | --- | --- |
| `bg-void` | `#07090D` | behind everything |
| `bg-canvas` | `#0A0E14` | the instrument face |
| `bg-chassis` | `#0E131A` | panel background |
| `bg-raised` | `#151C25` | inputs, raised wells |
| `line-grid` | `#161D27` | canvas grid, 1.14:1, deliberately subliminal |
| `line-hair` | `#1E2732` | panel dividers |
| `line-edge` | `#2A3542` | borders that need to be seen |

The canvas face is **darker than the chassis around it**, which is the opposite
of the usual card-on-page arrangement. An instrument's screen is a hole in its
housing, not a card floating above it.

---

## 3. Text

| Token | Hex | On chassis | Use |
| --- | --- | --- | --- |
| `text-hi` | `#E3E9F0` | 15.25:1 | values, headings |
| `text-mid` | `#94A3B4` | 7.24:1 | labels, prose |
| `text-lo` | `#647180` | 3.74:1 | units, axis ticks, disabled |

`text-lo` clears AA for large text only, so it is **restricted to non-essential
labels**: axis ticks, units, adornments. Anything a learner has to read to
operate the instrument uses `text-mid` or better.

---

## 4. Everything else, kept quiet

| Token | Hex | Use |
| --- | --- | --- |
| `focus` | `#EAF1F8` | selection ring, hover, keyboard focus. Near-white, deliberately hueless |
| `status-bad` | `#E4586B` | divergence and errors only, 5.24:1 on chassis |

Selection is a **neutral** rather than a third hue, because a coloured selection
ring on top of a coloured edge would be read as a weight value. There is no
"success green": success is communicated by a value being where it should be,
not by a colour.

**Series that need to be told apart**, such as train versus validation loss or
four optimizer curves, are distinguished by **lightness and dash pattern, not
hue**. Adding a
palette of series colours is the single easiest way to wreck the rule that
nothing competes with the poles.

---

## 5. Type

Three faces, all self-hosted (§2 forbids runtime network requests), Latin
subsets, ~112 KB total.

**Display: Space Grotesk.** A technical grotesk drawn from Space Mono's
skeleton: angular `G`, flat-sided `O`, a `g` with an odd tail. It reads as
*engineered* rather than neutral, which is exactly the register of a bench
instrument's silkscreen. Used with restraint: the wordmark, panel headers,
lesson titles. Never for prose, never below 12px.

**Body: IBM Plex Sans.** Commissioned as the typeface for a technology
company's technical documentation, so being quiet is its design brief. Holds up
at 12–13px, which is where most of this interface lives.

**Data: IBM Plex Mono.** Every number in the app. Monospace, so figures are
tabular by construction; `font-variant-numeric: tabular-nums` is set anyway so a
fallback face cannot reintroduce jitter.

The Plex pair share a skeleton, which is the actual reason for choosing them
together: in a dense panel a label and its value are always adjacent, and
`learning rate` next to `0.0300` should look like one object, not two.

Inter was rejected on the spec's own terms: it is the generic default, not a
decision.

### Numerics are non-negotiable

Every displayed number uses the mono face, `tabular-nums`, and a **fixed decimal
precision per quantity** so values never change width. Jittering digits make the
whole app feel unstable, and this app changes most of its numbers every frame.

| Quantity | Format |
| --- | --- |
| weights, biases, activations, gradients | `+0.000` / `−0.000`, sign always shown |
| loss | `0.0000` |
| accuracy | `00.0%` |
| learning rate | `0.0000` or exponential below 1e-4 |
| epoch, counts | integer, right-aligned |
| gradient norms | exponential, `0.00e+0` |

Signs are always rendered, including on positives, so a column of values does
not shift by one character when a sign appears.

---

## 6. Space and weight

A 4px base grid. Density is the point, because this is an instrument rather
than a landing page, so the scale is tight and stops early.

```
space: 2 4 6 8 12 16 24 32 48
```

Hierarchy comes from type weight and spacing alone. **No shadows. No rounded
cards.** Borders are 1px `line-hair`; a panel is separated from its neighbour by
a rule and 16px, not by elevation. The only radius in the system is 2px, on
inputs and buttons, which reads as a machined edge rather than a soft one.

---

## 7. Motion

Continuous motion is reserved for the signature element, the dissection view's
pulses and the formula card assembling term by term. Everything else moves only
to acknowledge input: 120ms on hover, 160ms on selection, both `ease-out`.

Under `prefers-reduced-motion` the pulses become discrete stage highlights that
advance on a timer. The pedagogy has to survive with motion off, so the
choreography is a sequence of *states*, not an animation that happens to pause.
