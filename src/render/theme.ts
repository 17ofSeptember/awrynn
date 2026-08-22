/*
 * Design tokens — the single source of truth for colour (§3 layout, §9 design).
 *
 * This is a TypeScript module rather than a CSS file because the canvas needs
 * NUMERIC colour values at 60fps. Reading CSS custom properties per frame with
 * getComputedStyle forces style resolution and allocates a string every call,
 * which is the wrong direction entirely. Instead the tokens live here and
 * `cssVariables()` pushes them out to :root once at boot, so the panels and the
 * instrument face are provably the same palette.
 *
 * The rationale for every value is in docs/DESIGN.md, and theme.test.ts
 * recomputes the contrast and colour-vision figures quoted there.
 */

export interface Rgb {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

/* ------------------------------------------------------------------ *
 * The two poles (§9: chosen first, everything else derived beneath)
 * ------------------------------------------------------------------ */

/** Negative weights. Cool. */
export const WEIGHT_NEGATIVE = '#3EC5E8';
/** Positive weights. Warm. */
export const WEIGHT_POSITIVE = '#F2A33C';

export const COLORS = {
  /* Surfaces — desaturated blue-slate, none of them a hue you would name. */
  bgVoid: '#07090D',
  /** The instrument face. Darker than the chassis: a screen is a hole, not a card. */
  bgCanvas: '#0A0E14',
  bgChassis: '#0E131A',
  bgRaised: '#151C25',

  lineGrid: '#161D27',
  lineHair: '#1E2732',
  lineEdge: '#2A3542',

  /* Text. `textLo` is AA-large only — non-essential labels exclusively. */
  textHi: '#E3E9F0',
  textMid: '#94A3B4',
  textLo: '#647180',

  /* Deliberately hueless: a coloured selection ring on a coloured edge would
   * be misread as a weight value. */
  focus: '#EAF1F8',
  /** Divergence and errors only. */
  statusBad: '#E4586B',

  weightNegative: WEIGHT_NEGATIVE,
  weightPositive: WEIGHT_POSITIVE,
} as const;

export type ColorToken = keyof typeof COLORS;

/* ------------------------------------------------------------------ *
 * Categorical class colours
 *
 * The decision boundary paints class IDENTITY, which is a different encoding
 * job from weight polarity, so it needs its own palette rather than an
 * interpolation between the poles.
 *
 * The first attempt did interpolate, and it was measurably wrong: the midpoint
 * landed at OKLCH chroma 0.057, which reads as gray and stops doing identity
 * work, and sat ΔE 12.8 from the cool pole under NORMAL vision, below the
 * threshold at which full-colour readers can tell a pair apart.
 *
 * These slots were then searched against the validator rather than chosen by
 * eye. The decisive constraint is that colour-vision deficiency destroys hue
 * but preserves LIGHTNESS: every candidate at the poles' own lightness
 * collapsed under deuteranopia (violet against cyan measured ΔE 2.5). So the
 * added slots step down in lightness, which is what makes them separable, and
 * has the useful side effect that they cannot outshine the poles.
 *
 * Validated all-pairs, not merely adjacent pairs, because any two regions of a
 * decision boundary can touch. Worst-case separation, against a #0A0E14
 * surface:
 *
 *   K=2   CVD ΔE 21.4   normal ΔE 26.1
 *   K=3   CVD ΔE 21.4   normal ΔE 26.1
 *   K=4   CVD ΔE 20.3   normal ΔE 23.4
 *   K=5   CVD ΔE 15.9   normal ΔE 16.0
 *
 * against a target of 8 and a hard normal-vision floor of 15.
 */
export const CLASS_COLORS: readonly string[] = [
  WEIGHT_NEGATIVE,
  WEIGHT_POSITIVE,
  '#984260',
  '#3C6CDC',
  '#C86830',
];

/**
 * Colour for class `k`.
 *
 * Assigned in fixed order and never cycled, so a class keeps its colour when
 * the class count changes. Beyond the defined slots the colour repeats and the
 * caller must add a second encoding; in practice the boundary view is capped
 * below that, and datasets with more classes have too many input features to
 * be drawn as a plane at all.
 */
export function classColor(k: number): string {
  return CLASS_COLORS[k % CLASS_COLORS.length] as string;
}

/* ------------------------------------------------------------------ *
 * Space, type and motion
 * ------------------------------------------------------------------ */

/** 4px base grid. Tight and it stops early — this is an instrument. */
export const SPACE = [2, 4, 6, 8, 12, 16, 24, 32, 48] as const;

export const FONTS = {
  /** Space Grotesk. Restraint: wordmark, panel headers, never below 12px. */
  display: "'Space Grotesk', 'Helvetica Neue', Arial, sans-serif",
  /** IBM Plex Sans. Quiet by design brief. */
  body: "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif",
  /** IBM Plex Mono. Every number in the app. */
  mono: "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
} as const;

export const MOTION = {
  hoverMs: 120,
  selectMs: 160,
  easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
} as const;

/* ------------------------------------------------------------------ *
 * Colour utilities
 * ------------------------------------------------------------------ */

export function parseHex(hex: string): Rgb {
  const value = hex.startsWith('#') ? hex.slice(1) : hex;
  if (value.length !== 6) {
    throw new Error(`theme.parseHex: expected a 6-digit hex colour, got ${JSON.stringify(hex)}.`);
  }
  const n = Number.parseInt(value, 16);
  if (Number.isNaN(n)) {
    throw new Error(`theme.parseHex: ${JSON.stringify(hex)} is not a hex colour.`);
  }
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function channelLuminance(c: number): number {
  const s = c / 255;
  return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

export function relativeLuminance(color: Rgb): number {
  return (
    0.2126 * channelLuminance(color.r) +
    0.7152 * channelLuminance(color.g) +
    0.0722 * channelLuminance(color.b)
  );
}

/** WCAG contrast ratio, 1:1 to 21:1. */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(parseHex(a));
  const lb = relativeLuminance(parseHex(b));
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}

/** OKLab, for perceptual separation checks. */
export function oklab(hex: string): readonly [number, number, number] {
  const { r, g, b } = parseHex(hex);
  const lr = channelLuminance(r);
  const lg = channelLuminance(g);
  const lb = channelLuminance(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function perceptualDistance(a: string, b: string): number {
  const [l1, a1, b1] = oklab(a);
  const [l2, a2, b2] = oklab(b);
  return Math.hypot(l1 - l2, a1 - a2, b1 - b2);
}

export type ColorVisionDeficiency = 'protanopia' | 'deuteranopia' | 'tritanopia';

const CVD_MATRICES: Readonly<Record<ColorVisionDeficiency, readonly number[]>> = {
  protanopia: [0.567, 0.433, 0, 0.558, 0.442, 0, 0, 0.242, 0.758],
  deuteranopia: [0.625, 0.375, 0, 0.7, 0.3, 0, 0, 0.3, 0.7],
  tritanopia: [0.95, 0.05, 0, 0, 0.433, 0.567, 0, 0.475, 0.525],
};

/** Simulate a colour under a vision deficiency, for the palette tests. */
export function simulateColorVision(hex: string, kind: ColorVisionDeficiency): string {
  const { r, g, b } = parseHex(hex);
  const m = CVD_MATRICES[kind];
  const out = [
    m[0]! * r + m[1]! * g + m[2]! * b,
    m[3]! * r + m[4]! * g + m[5]! * b,
    m[6]! * r + m[7]! * g + m[8]! * b,
  ].map((v) => Math.max(0, Math.min(255, Math.round(v))));
  return `#${out.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

/* ------------------------------------------------------------------ *
 * Precomputed weight ramps
 *
 * §6.1 requires zero allocation in the hot path, and an edge draw happens
 * hundreds of times per frame. Building `rgba(...)` strings per edge would
 * allocate on every one, so both ramps are quantized to fixed steps and built
 * once at module load. Drawing an edge is then an array index.
 * ------------------------------------------------------------------ */

export const WEIGHT_RAMP_STEPS = 32;

/** Faintest step. Near-zero weights fade almost out, so pruning is visible. */
const MIN_ALPHA = 0.06;
const MAX_ALPHA = 1;

function buildRamp(hex: string): readonly string[] {
  const { r, g, b } = parseHex(hex);
  const ramp: string[] = [];
  for (let i = 0; i < WEIGHT_RAMP_STEPS; i++) {
    const t = i / (WEIGHT_RAMP_STEPS - 1);
    const alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * t;
    ramp.push(`rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`);
  }
  return ramp;
}

const NEGATIVE_RAMP = buildRamp(WEIGHT_NEGATIVE);
const POSITIVE_RAMP = buildRamp(WEIGHT_POSITIVE);

/**
 * Stroke colour for a weight.
 *
 * `wRef` is the 95th percentile of |w| across the network, smoothed over time
 * so the picture does not strobe during training (§6.2). Weights beyond it
 * clamp to the strongest step rather than saturating invisibly.
 */
export function weightStroke(w: number, wRef: number): string {
  const ramp = w < 0 ? NEGATIVE_RAMP : POSITIVE_RAMP;
  const magnitude = wRef > 0 ? Math.min(Math.abs(w) / wRef, 1) : 0;
  const index = Math.min(WEIGHT_RAMP_STEPS - 1, Math.round(magnitude * (WEIGHT_RAMP_STEPS - 1)));
  return ramp[index] as string;
}

/**
 * Stroke width for a weight (§6.2):
 *   width = clamp(0.5 + 5.5·|w|/wRef, 0.5, 7)
 */
export const EDGE_WIDTH_MIN = 0.5;
export const EDGE_WIDTH_MAX = 7;

export function weightWidth(w: number, wRef: number): number {
  const magnitude = wRef > 0 ? Math.abs(w) / wRef : 0;
  return Math.max(EDGE_WIDTH_MIN, Math.min(EDGE_WIDTH_MAX, 0.5 + 5.5 * magnitude));
}

/**
 * Dash pattern for negative weights in colourblind-safe mode (§6.2).
 *
 * Additive: the hues already separate under simulation, so this is redundancy
 * rather than a rescue. Positive weights stay solid.
 */
export const COLORBLIND_NEGATIVE_DASH: readonly number[] = [4, 3];

/* ------------------------------------------------------------------ *
 * Node fill ramp
 * ------------------------------------------------------------------ */

const NODE_RAMP_STEPS = 24;

function buildNodeRamp(hex: string): readonly string[] {
  const { r, g, b } = parseHex(hex);
  const ramp: string[] = [];
  for (let i = 0; i < NODE_RAMP_STEPS; i++) {
    const t = i / (NODE_RAMP_STEPS - 1);
    ramp.push(`rgba(${r}, ${g}, ${b}, ${(0.08 + 0.84 * t).toFixed(3)})`);
  }
  return ramp;
}

const NODE_POSITIVE_RAMP = buildNodeRamp(WEIGHT_POSITIVE);
const NODE_NEGATIVE_RAMP = buildNodeRamp(WEIGHT_NEGATIVE);

/**
 * Node fill for an activation, normalized to [-1, 1].
 *
 * Bounded activations use their true range; ReLU and linear normalize against a
 * smoothed per-layer running max, and the layer displays that max (§6.2) — the
 * caller does the normalizing, because only it knows the layer.
 */
export function activationFill(normalized: number): string {
  const ramp = normalized < 0 ? NODE_NEGATIVE_RAMP : NODE_POSITIVE_RAMP;
  const magnitude = Math.min(Math.abs(normalized), 1);
  const index = Math.min(NODE_RAMP_STEPS - 1, Math.round(magnitude * (NODE_RAMP_STEPS - 1)));
  return ramp[index] as string;
}

/* ------------------------------------------------------------------ *
 * Numeric formatting (§9: fixed precision per quantity, never jittering)
 * ------------------------------------------------------------------ */

/** Signed, fixed width: weights, biases, activations, gradients. */
export function formatSigned(value: number, decimals = 3): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '+∞' : '−∞';
  // The sign is ALWAYS rendered, including on positives, so a column of values
  // does not shift by a character when one turns negative.
  const sign = value < 0 ? '−' : '+';
  return sign + Math.abs(value).toFixed(decimals);
}

export function formatLoss(value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : '∞';
  return value.toFixed(4);
}

export function formatPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(1)}%`;
}

export function formatLearningRate(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return value < 1e-4 && value > 0 ? value.toExponential(2) : value.toFixed(4);
}

export function formatNorm(value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : '∞';
  if (value === 0) return '0.00e+0';
  return value.toExponential(2);
}

/* ------------------------------------------------------------------ *
 * CSS bridge
 * ------------------------------------------------------------------ */

function kebab(name: string): string {
  return name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
}

/** Every token as CSS custom property declarations, for :root. */
export function cssVariables(): Record<string, string> {
  const vars: Record<string, string> = {};
  for (const [name, value] of Object.entries(COLORS)) {
    vars[`--color-${kebab(name)}`] = value;
  }
  SPACE.forEach((value, i) => {
    vars[`--space-${i}`] = `${value}px`;
  });
  vars['--font-display'] = FONTS.display;
  vars['--font-body'] = FONTS.body;
  vars['--font-mono'] = FONTS.mono;
  vars['--motion-hover'] = `${MOTION.hoverMs}ms`;
  vars['--motion-select'] = `${MOTION.selectMs}ms`;
  vars['--motion-easing'] = MOTION.easing;
  return vars;
}

/** Push the tokens onto :root once at boot. Idempotent. */
export function installTheme(root: { style: { setProperty(k: string, v: string): void } }): void {
  for (const [name, value] of Object.entries(cssVariables())) {
    root.style.setProperty(name, value);
  }
}
