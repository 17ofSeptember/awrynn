/*
 * Formula cards — the signature element (§9).
 *
 * Spec §6.3 A.3–A.5: a card materialises beside the neuron and assembles term
 * by term as each pulse arrives, with a running subtotal that ticks, then the
 * resolved z; a mini activation plot shows the φ curve with a dot travelling
 * from z on the x-axis to a on the y-axis; then the card collapses to a compact
 * z / a chip.
 *
 * Every number drawn here comes from a NeuronDissection, which reads the
 * engine's caches. Nothing on a card is recomputed from the picture.
 */

import type { NeuronDissection, EdgeGradient, OutputDissection } from '../dissection';
import { COLORS, FONTS, formatSigned, formatLoss } from '../theme';
import { clamp01, easeOutCubic, stagger } from '../animation';
import type { Ctx2D } from './context';
import { TAU } from './context';

export const CARD_PADDING = 10;
const LINE_HEIGHT = 16;
const TERM_FONT = `400 11px ${FONTS.mono}`;
const LABEL_FONT = `500 9px ${FONTS.mono}`;
const RESULT_FONT = `500 12px ${FONTS.mono}`;

export interface CardPlacement {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/**
 * How many terms a card shows before summarising the rest.
 *
 * A 35-input layer would produce a card taller than the viewport, and a card
 * nobody can read teaches nothing. The remainder is shown as a single summed
 * line so the arithmetic still balances on screen.
 */
export const MAX_VISIBLE_TERMS = 6;

export interface FormulaCardState {
  readonly neuron: NeuronDissection;
  /** 0..1 through the assembling beat. */
  readonly assembleProgress: number;
  /** 0..1 through the activation beat; 0 before it starts. */
  readonly activateProgress: number;
  /** Collapsed to a z / a chip once the layer is done. */
  readonly collapsed: boolean;
}

export function measureFormulaCard(state: FormulaCardState): { width: number; height: number } {
  if (state.collapsed) return { width: 104, height: 24 };
  const visible = Math.min(state.neuron.terms.length, MAX_VISIBLE_TERMS);
  const hasRemainder = state.neuron.terms.length > MAX_VISIBLE_TERMS;
  // Normalizing adds two rows: subtracting μ and dividing by σ, then scaling
  // by γ. Two rather than one because they are two different ideas, and folding
  // them together is what makes batch norm look like a magic constant.
  const normalizeRows = state.neuron.normalization === null ? 0 : 2;
  const rows = visible + (hasRemainder ? 1 : 0) + normalizeRows + 1 /* bias */ + 1 /* z */;
  return { width: 208, height: CARD_PADDING * 2 + 14 + rows * LINE_HEIGHT + 8 };
}

/**
 * Draw one neuron's card.
 *
 * Terms fade and slide in on a stagger, so the card is seen ASSEMBLING rather
 * than appearing — the running subtotal is what makes a dot product feel like
 * an accumulation instead of a formula.
 */
export function drawFormulaCard(ctx: Ctx2D, state: FormulaCardState, at: CardPlacement): void {
  const { neuron } = state;
  ctx.setLineDash([]);

  if (state.collapsed) {
    drawChip(ctx, neuron, at);
    return;
  }

  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(at.x, at.y, at.width, at.height);
  ctx.strokeStyle = COLORS.lineEdge;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(at.x + 0.5, at.y + 0.5, at.width - 1, at.height - 1);
  ctx.stroke();

  const left = at.x + CARD_PADDING;
  let y = at.y + CARD_PADDING + 8;

  ctx.font = LABEL_FONT;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText(`Z  LAYER ${neuron.layer + 1} · UNIT ${neuron.unit}`, left, y);
  y += 14;

  const visible = Math.min(neuron.terms.length, MAX_VISIBLE_TERMS);
  const hasRemainder = neuron.terms.length > MAX_VISIBLE_TERMS;
  const norm = neuron.normalization;
  const stepCount = visible + (hasRemainder ? 1 : 0) + (norm === null ? 0 : 2) + 1;

  let subtotal = 0;
  ctx.font = TERM_FONT;

  for (let i = 0; i < visible; i++) {
    const term = neuron.terms[i] as NeuronDissection['terms'][number];
    const appear = easeOutCubic(stagger(state.assembleProgress, i, stepCount));
    if (appear <= 0) {
      y += LINE_HEIGHT;
      continue;
    }
    subtotal += term.contribution * appear;
    ctx.globalAlpha = appear;
    ctx.fillStyle = term.contribution < 0 ? COLORS.weightNegative : COLORS.weightPositive;
    // The displayed product, from the engine's own w and a.
    ctx.fillText(
      `${i === 0 ? ' ' : '+'} (${formatSigned(term.weight, 2)})(${formatSigned(term.input, 2)})`,
      left,
      y,
    );
    ctx.fillStyle = COLORS.textMid;
    ctx.textAlign = 'right';
    ctx.fillText(formatSigned(term.contribution, 3), at.x + at.width - CARD_PADDING, y);
    ctx.textAlign = 'left';
    ctx.globalAlpha = 1;
    y += LINE_HEIGHT;
  }

  if (hasRemainder) {
    const rest = neuron.terms.slice(MAX_VISIBLE_TERMS);
    const restSum = rest.reduce((total, t) => total + t.contribution, 0);
    const appear = easeOutCubic(stagger(state.assembleProgress, visible, stepCount));
    if (appear > 0) {
      subtotal += restSum * appear;
      ctx.globalAlpha = appear;
      ctx.fillStyle = COLORS.textLo;
      // Summarised, not omitted: the arithmetic on screen still balances.
      ctx.fillText(`+ ${rest.length} more terms`, left, y);
      ctx.textAlign = 'right';
      ctx.fillText(formatSigned(restSum, 3), at.x + at.width - CARD_PADDING, y);
      ctx.textAlign = 'left';
      ctx.globalAlpha = 1;
    }
    y += LINE_HEIGHT;
  }

  /*
   * The normalization, in the order the engine applied it.
   *
   * The subtotal stops being an accumulation here and becomes a transformation,
   * which is exactly what is worth seeing: everything above this line is a
   * weighted sum, and then the sum is re-expressed in units of its own spread
   * before anything else happens to it.
   */
  if (norm !== null) {
    const shiftStep = visible + (hasRemainder ? 1 : 0);
    const shiftAppear = easeOutCubic(stagger(state.assembleProgress, shiftStep, stepCount));
    if (shiftAppear > 0) {
      subtotal = subtotal + (norm.normalized - subtotal) * shiftAppear;
      ctx.globalAlpha = shiftAppear;
      ctx.fillStyle = COLORS.textLo;
      ctx.fillText(`− ${formatSigned(norm.mean, 2)}  ÷ ${norm.sigma.toFixed(2)}`, left, y);
      ctx.fillStyle = COLORS.textMid;
      ctx.textAlign = 'right';
      ctx.fillText(formatSigned(norm.normalized, 3), at.x + at.width - CARD_PADDING, y);
      ctx.textAlign = 'left';
      ctx.globalAlpha = 1;
    }
    y += LINE_HEIGHT;

    const scaleAppear = easeOutCubic(stagger(state.assembleProgress, shiftStep + 1, stepCount));
    if (scaleAppear > 0) {
      const scaled = norm.gamma * norm.normalized;
      subtotal = subtotal + (scaled - subtotal) * scaleAppear;
      ctx.globalAlpha = scaleAppear;
      ctx.fillStyle = COLORS.textLo;
      ctx.fillText(`× γ ${formatSigned(norm.gamma, 2)}`, left, y);
      ctx.fillStyle = COLORS.textMid;
      ctx.textAlign = 'right';
      ctx.fillText(formatSigned(scaled, 3), at.x + at.width - CARD_PADDING, y);
      ctx.textAlign = 'left';
      ctx.globalAlpha = 1;
    }
    y += LINE_HEIGHT;
  }

  const biasAppear = easeOutCubic(stagger(state.assembleProgress, stepCount - 1, stepCount));
  if (biasAppear > 0) {
    subtotal += neuron.bias * biasAppear;
    ctx.globalAlpha = biasAppear;
    ctx.fillStyle = COLORS.textMid;
    ctx.fillText('+ bias', left, y);
    ctx.textAlign = 'right';
    ctx.fillText(formatSigned(neuron.bias, 3), at.x + at.width - CARD_PADDING, y);
    ctx.textAlign = 'left';
    ctx.globalAlpha = 1;
  }
  y += LINE_HEIGHT;

  ctx.strokeStyle = COLORS.lineHair;
  ctx.beginPath();
  ctx.moveTo(left, y - 10.5);
  ctx.lineTo(at.x + at.width - CARD_PADDING, y - 10.5);
  ctx.stroke();

  // The running subtotal ticks while assembling, then snaps to the engine's
  // cached z. Showing the tally rather than the answer is the point.
  const settled = state.assembleProgress >= 1;
  ctx.font = RESULT_FONT;
  ctx.fillStyle = settled ? COLORS.textHi : COLORS.textMid;
  ctx.fillText('z =', left, y);
  ctx.textAlign = 'right';
  ctx.fillText(formatSigned(settled ? neuron.z : subtotal, 4), at.x + at.width - CARD_PADDING, y);
  ctx.textAlign = 'left';
}

function drawChip(ctx: Ctx2D, neuron: NeuronDissection, at: CardPlacement): void {
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(at.x, at.y, at.width, at.height);
  ctx.strokeStyle = COLORS.lineHair;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(at.x + 0.5, at.y + 0.5, at.width - 1, at.height - 1);
  ctx.stroke();

  ctx.font = `500 10px ${FONTS.mono}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText('z', at.x + 8, at.y + at.height / 2);
  ctx.fillStyle = COLORS.textMid;
  ctx.fillText(formatSigned(neuron.z, 2), at.x + 18, at.y + at.height / 2);
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText('a', at.x + 62, at.y + at.height / 2);
  ctx.fillStyle = COLORS.textHi;
  ctx.fillText(formatSigned(neuron.a, 2), at.x + 72, at.y + at.height / 2);
  ctx.textBaseline = 'alphabetic';
}

/* ------------------------------------------------------------------ *
 * Mini activation plot (§6.3 A.4)
 * ------------------------------------------------------------------ */

export const MINI_PLOT_SIZE = 74;

/**
 * The φ curve with a dot travelling from z on the x-axis to a on the y-axis.
 *
 * This is the moment the activation function stops being a name in a dropdown:
 * the learner watches z enter the curve and a come out.
 */
export function drawActivationPlot(
  ctx: Ctx2D,
  x: number,
  y: number,
  neuron: NeuronDissection,
  phi: (z: number) => number,
  range: readonly [number, number] | null,
  progress: number,
): void {
  const size = MINI_PLOT_SIZE;
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = COLORS.lineHair;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x + 0.5, y + 0.5, size - 1, size - 1);
  ctx.stroke();

  // Domain always contains z, so the dot is never off the plot.
  const zSpan = Math.max(3, Math.abs(neuron.z) * 1.4);
  const yLo = range === null ? -Math.max(1, Math.abs(neuron.a) * 1.4) : range[0];
  const yHi = range === null ? Math.max(1, Math.abs(neuron.a) * 1.4) : range[1];

  const px = (zv: number): number => x + ((zv + zSpan) / (2 * zSpan)) * size;
  const py = (av: number): number => y + size - ((av - yLo) / (yHi - yLo)) * size;

  // Axes.
  ctx.strokeStyle = COLORS.lineGrid;
  ctx.beginPath();
  ctx.moveTo(x, py(0));
  ctx.lineTo(x + size, py(0));
  ctx.moveTo(px(0), y);
  ctx.lineTo(px(0), y + size);
  ctx.stroke();

  // The curve.
  ctx.strokeStyle = COLORS.textLo;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const zv = -zSpan + (2 * zSpan * i) / steps;
    const av = phi(zv);
    const sx = px(zv);
    const sy = py(av);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();

  // Travel: along the x-axis to z, then up the curve to a.
  const t = clamp01(progress);
  const lead = easeOutCubic(Math.min(1, t * 2));
  const rise = easeOutCubic(Math.max(0, t * 2 - 1));

  ctx.strokeStyle = COLORS.focus;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px(0), py(0));
  ctx.lineTo(px(neuron.z * lead), py(0));
  if (rise > 0) {
    ctx.moveTo(px(neuron.z), py(0));
    ctx.lineTo(px(neuron.z), py(neuron.a * rise));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  const dotX = rise > 0 ? px(neuron.z) : px(neuron.z * lead);
  const dotY = rise > 0 ? py(neuron.a * rise) : py(0);
  ctx.beginPath();
  ctx.arc(dotX, dotY, 3, 0, TAU);
  ctx.fillStyle = COLORS.focus;
  ctx.fill();

  ctx.font = LABEL_FONT;
  ctx.fillStyle = COLORS.textLo;
  ctx.textAlign = 'left';
  ctx.fillText(neuron.activation.toUpperCase(), x + 5, y + 11);
}

/* ------------------------------------------------------------------ *
 * Loss card (§6.3 A) and gradient card (§6.3 backward)
 * ------------------------------------------------------------------ */

export const LOSS_CARD_WIDTH = 200;
export const LOSS_CARD_HEIGHT = 92;

export function drawLossCard(
  ctx: Ctx2D,
  x: number,
  y: number,
  output: OutputDissection,
  progress: number,
): void {
  ctx.setLineDash([]);
  ctx.globalAlpha = easeOutCubic(clamp01(progress));
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(x, y, LOSS_CARD_WIDTH, LOSS_CARD_HEIGHT);
  ctx.strokeStyle = COLORS.lineEdge;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x + 0.5, y + 0.5, LOSS_CARD_WIDTH - 1, LOSS_CARD_HEIGHT - 1);
  ctx.stroke();

  const left = x + CARD_PADDING;
  let cy = y + CARD_PADDING + 8;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.font = LABEL_FONT;
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText(`LOSS · ${output.lossName.toUpperCase()}`, left, cy);
  cy += 18;

  ctx.font = TERM_FONT;
  const row = (label: string, value: string, emphasis = false): void => {
    ctx.fillStyle = COLORS.textLo;
    ctx.fillText(label, left, cy);
    ctx.textAlign = 'right';
    ctx.fillStyle = emphasis ? COLORS.textHi : COLORS.textMid;
    ctx.fillText(value, x + LOSS_CARD_WIDTH - CARD_PADDING, cy);
    ctx.textAlign = 'left';
    cy += LINE_HEIGHT;
  };
  row('prediction', formatSigned(output.prediction, 4));
  row('target', formatSigned(output.target, 4));
  ctx.strokeStyle = COLORS.lineHair;
  ctx.beginPath();
  ctx.moveTo(left, cy - 10.5);
  ctx.lineTo(x + LOSS_CARD_WIDTH - CARD_PADDING, cy - 10.5);
  ctx.stroke();
  ctx.font = RESULT_FONT;
  row('loss', formatLoss(output.loss), true);

  ctx.globalAlpha = 1;
}

export const GRADIENT_CARD_WIDTH = 214;
export const GRADIENT_CARD_HEIGHT = 78;

/**
 * The card on an edge during the backward pass.
 *
 * Spec §6.3: "each edge showing ∂L/∂w = δ·a and Δw = −η · ∂L/∂w with real
 * numbers". Both are shown as equations with their operands, not as bare
 * results, because the operands are the lesson.
 */
export function drawGradientCard(
  ctx: Ctx2D,
  x: number,
  y: number,
  gradient: EdgeGradient,
  learningRate: number,
  progress: number,
): void {
  ctx.setLineDash([]);
  ctx.globalAlpha = easeOutCubic(clamp01(progress));
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(x, y, GRADIENT_CARD_WIDTH, GRADIENT_CARD_HEIGHT);
  ctx.strokeStyle = COLORS.lineEdge;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x + 0.5, y + 0.5, GRADIENT_CARD_WIDTH - 1, GRADIENT_CARD_HEIGHT - 1);
  ctx.stroke();

  const left = x + CARD_PADDING;
  let cy = y + CARD_PADDING + 8;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.font = LABEL_FONT;
  ctx.fillStyle = COLORS.textLo;
  ctx.fillText(`W[${gradient.row}, ${gradient.col}]`, left, cy);
  cy += 16;

  ctx.font = TERM_FONT;
  ctx.fillStyle = COLORS.textMid;
  ctx.fillText(
    `∂L/∂w = (${formatSigned(gradient.delta, 3)})(${formatSigned(gradient.input, 3)})`,
    left,
    cy,
  );
  cy += LINE_HEIGHT;
  ctx.fillStyle = gradient.gradient < 0 ? COLORS.weightNegative : COLORS.weightPositive;
  ctx.fillText(`      = ${formatSigned(gradient.gradient, 4)}`, left, cy);
  cy += LINE_HEIGHT;
  ctx.fillStyle = COLORS.textHi;
  ctx.fillText(
    `Δw = −${learningRate.toFixed(3)} × ∂L/∂w = ${formatSigned(gradient.step, 4)}`,
    left,
    cy,
  );

  ctx.globalAlpha = 1;
}
