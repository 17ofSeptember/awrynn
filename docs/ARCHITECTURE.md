# ARCHITECTURE.md

Data flow, threading model, and render pipeline.

> **Status: partial (Phase 0).** Only the decisions actually made during the
> scaffold are recorded. The threading model and render pipeline are written in
> Phase 2 and Phase 3 respectively, once they exist.

## Layering, and the one rule that matters

```
src/engine/   pure TypeScript over Float64Array. No DOM, no React, no store.
src/worker/   owns engine instances off the main thread.
src/render/   canvas drawing. Reads a mutable frame-state object.
src/state/    Zustand store. The seam between React and the canvas.
src/ui/       React panels. Never re-renders on an animation frame.
src/lessons/  data, not code paths.
```

Dependencies point **downward only**. `engine` is the bottom and imports
nothing from the layers above it.

### How engine isolation is enforced

Not by convention. By three mechanisms that fail loudly:

1. **`tsconfig.engine.json`** compiles `src/engine/**` with `lib: ["ES2022"]`
   and `types: []`. No `DOM` lib, no ambient type packages. Touching `window`,
   `document`, or `console` is a compile error. Run by `npm run typecheck:engine`,
   which `npm run build` depends on.
2. **`src/engine/__tests__/purity.test.ts`** scans the engine source tree for
   `Math.random`, forbidden import specifiers, and browser globals. Required by
   spec §10.
3. **ESLint** (`no-restricted-properties`, `no-restricted-globals`) scoped to
   `src/engine/**`.

The reason for the redundancy: this is the property that makes the engine
testable and trustworthy, and it is the property most likely to be eroded by a
single convenient import six phases from now.

## TypeScript configuration

`tsconfig.base.json` holds the shared strictness; the three leaf configs differ
only in `lib`/`types`/`include`.

Beyond the two flags required by spec §0.3 (`strict`, `noUncheckedIndexedAccess`):

| Flag | Why |
| --- | --- |
| `exactOptionalPropertyTypes` | Network/optimizer configs are option bags; `undefined` vs absent is a real distinction for hyperparameters. |
| `verbatimModuleSyntax` | Type-only imports stay erasable, which matters for worker bundles. |
| `noPropertyAccessFromIndexSignature` | Metric and config records are keyed dynamically; forces the unsafe access to be visible. |
| `noImplicitReturns`, `noFallthroughCasesInSwitch` | Activation and optimizer dispatch are switch-heavy. |
| `noUnusedLocals`, `noUnusedParameters` | Cheap, and catches half-finished refactors of the backward pass. |

## Testing

Vitest, `environment: 'node'` by default and `globals: false`. Engine tests must
run without a browser-like environment, so making jsdom the default would hide
exactly the violation the architecture is designed to prevent. UI tests opt into
jsdom per-file with `// @vitest-environment jsdom`.

`npm run gradcheck` targets `src/engine/__tests__/gradcheck.test.ts`, the
correctness backbone, per spec §0.8 and §4.11.

## Parameter storage

Every parameter in a network lives in **one contiguous `Float64Array`**
(`Network.params`), with each layer's `W` and `b` as `subarray` views into it,
and the gradients laid out identically in `Network.grads`. Layout is layer
order, `W` then `b`, matching `parameterHandles()` in `gradcheck.ts`.

This is what lets optimizers take the flat arrays §4.8 describes with no gather
or scatter per step, makes gradient clipping by global norm a single pass, and
reduces a history snapshot (§6.6) to a memcpy.

The consequence to remember: **never reassign a layer's `W.data`**. `setWeights`
copies in place and `bindStorage` rebinds deliberately; anything else silently
detaches the layer from the optimizer. `DenseLayer` documents this at the field.

## Batch normalization, and the buffers array

`Network` holds two contiguous arrays, not one.

| | Written by | Travels with | Reached by optimizers |
| --- | --- | --- | --- |
| `params` | the optimizer | everything | yes |
| `buffers` | the forward pass | everything | never |

`params` is per layer `[W \| b \| γ]`, with γ empty unless that layer normalizes,
so a network without batch norm has byte-for-byte the layout it had before batch
norm existed and every share link and save file made until then still restores.
`buffers` holds μ̂ and σ̂² per normalizing layer, and is empty otherwise.

They are separate because an optimizer must never reach the statistics: AdamW
would decay a running mean toward zero and the global-norm clipper would scale
it, and the parameter count on screen should keep counting parameters.

They are inseparable in every other respect. `captureState`/`restoreState` move
the pair together, and the boundaries take that object rather than a bare
`Float64Array`, because a positional `(parameters, buffers)` argument list is an
invitation to pass one and forget the other. Everything that MOVES a network
carries both: worker messages, history snapshots, early stopping's best-weight
restore, save files and share links. Restoring weights without their statistics
gives a network whose eval-mode predictions are wrong with nothing on screen to
explain it.

### `Network.inspect()`

Displaying a network must not change it. Two panels need a training-mode forward
for one reason only, that it retains `A^{l-1}` for the backward pass, and get two
side effects they never asked for: dropout masks that make the arithmetic on
screen unreproducible, and a nudge to the running statistics that moves the
decision boundary on every re-render. `inspect()` silences both and restores the
caller's configuration. `gradcheck` uses it for the same two reasons.

## Threading model

```
main thread                          trainer.worker.ts
-----------                          -----------------
start ──────────────────────────────▶ construct Trainer, generate dataset
                        ◀─────────── ready { datasetSummary }
                        ◀─────────── progress { metrics[], parameters, epoch }
pause / resume ─────────────────────▶ (loop yields between chunks)
step-epoch ─────────────────────────▶ exactly one epoch
                        ◀─────────── snapshot { epoch, parameters }
                        ◀─────────── done { status, epoch, parameters }
```

Raw `postMessage` with an explicit typed protocol (`src/worker/protocol.ts`)
rather than comlink. The protocol is small, and keeping it explicit means the
worker boundary is visible in the type system.

- **The loop yields between chunks** (`setTimeout(tick, 0)`) rather than running
  to completion, so `pause`/`stop`/`step-epoch` are actually delivered instead
  of queueing behind a synchronous run.
- **`reportEvery` batches metrics** so a fast run does not flood the main thread
  with one message per epoch. The UI picks it from the animation mode (§6.3).
- **Parameters are copied, not transferred.** Transferring would detach the
  worker's own buffer, and the worker needs to keep training from it.
- **Errors and divergences are different things.** A divergence is a `done`
  message with `status: 'diverged'`; only a thrown exception produces `error`.

`trainer.worker.ts` is tested by driving the module through a stand-in `self`
(`src/worker/__tests__/`), which exercises the protocol and the chunked loop
rather than the platform's ability to spawn a thread.

## Render pipeline

One `requestAnimationFrame` loop in `scene.ts`, dirty-tracked: an idle scene
costs one comparison per frame and draws nothing, which is what makes "60fps
idle" mean an idle CPU rather than a busy one keeping up.

**React never re-renders on an animation frame.** The canvas subscribes to the
store imperatively and mutates a `FrameState` object in place. Values that tick
at frame rate (activations, the smoothed weight reference, pulse positions)
live in that object and never enter the store. Values that change a few times a
second, such as the transport's current beat and the training metrics, are mirrored back
on a timer, not per frame.

Fixed draw order:

```
grid → edges → pulses → nodes → thumbnails → labels → dissection → legend
```

Everything is drawn in SCREEN coordinates, converting world positions
explicitly rather than applying a canvas transform. That keeps a 0.5px hairline
a real 0.5px at every zoom. Node radii DO scale with zoom, because a node is an
object in the scene; stroke width does not, because it is an encoding that must
mean the same thing however far you have zoomed.

Allocation-free hot path: both weight ramps are precomputed as quantized `rgba`
strings, pulses come from a fixed pool, and the thumbnails are baked into one
sprite sheet and blitted per node.

## Keyboard navigation

`render/navigation.ts` is pure traversal logic over a `Layout`, so it is tested
without a browser. The canvas itself takes focus, since nothing inside a picture
can, and every move is announced into an `aria-live` region, because the interface
otherwise conveys everything through colour and position.

```
← →           move between columns
↑ ↓           move between units
shift + ↑ ↓   walk the connections arriving at a unit
enter         focus the inspector's value field
+ / −         nudge the selected weight
escape        clear the selection
```

Column-then-unit rather than one flat order, because the network IS a grid and a
flat traversal of several hundred edges is a list, not navigation.

## Responsive layout

Two breakpoints, both chosen from content rather than device names:

| Width | Layout |
| --- | --- |
| < 1024 | everything in a bottom-sheet drawer |
| 1024–1279 | build panel as a sidebar, analysis in the drawer |
| ≥ 1280 | both sidebars, no drawer |

The panels are the same components in both places rather than two copies that
would drift. The drawer overlays the canvas rather than pushing it, so opening
it does not resize the canvas under the reader. A resize DOES refit the
viewport, unless the reader has panned or zoomed themselves.

## Share links

`state/shareLink.ts` is a pure codec: `SharedState` (architecture, dataset
options, training settings, lesson id, parameters) to a URL fragment and back.
It knows nothing about React or the store, which is what lets its 29 tests run
in Node.

Two structural notes.

**It lives below the store, not inside it.** `Architecture` and
`TrainingSettings` were moved out of `store.ts` into `state/architecture.ts` so
the codec can name them without the store importing the codec importing the
store. This is the second time that cycle has appeared here (the first produced
`engine/datasets/types.ts`), and the remedy is the same both times: the shared
types move down a level rather than the import being allowed to loop.

The fragment carries three blocks: `s` for the configuration, `p` for the
parameters, `r` for the running statistics. `p` and `r` are each omitted when a
fresh network of the same architecture reproduces them exactly, checked rather
than assumed.

**Decoding is total.** `decodeShareLink` never throws. A fragment is untrusted
input, and every field is checked for type, membership and range before it
reaches the engine, including sanity caps on layer count and units, so a
hand-edited link cannot ask for 2³¹ neurons and hang the tab before
`validateConfig` gets a chance to object. Failures return a sentence a reader
can act on; the app shows it rather than silently presenting a default network
as though it were the one that was sent.

The parameter block is omitted when a fresh network built from the same seed
produces byte-identical parameters. This is a load-bearing use of the
determinism guarantee rather than a size optimization, and `parametersAreFresh`
is exported so a test can assert the claim the omission makes.

`ui/useSharedLink.ts` applies the fragment on mount and on `hashchange`, the
latter so that pasting a link into an open tab works: a fragment change does not
reload the page.

## Not yet decided

- **`comlink`**: approved by §2 but not installed. Raw `postMessage` with a
  typed protocol proved perfectly ergonomic; revisit only if the protocol grows.
- **`OffscreenCanvas`**: allowed by §2. The main-thread render is comfortably
  inside budget (see `render/__tests__/performance.test.ts`), so moving it would
  add a threading boundary to solve a problem that does not exist yet.
