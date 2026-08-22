/*
 * Legend for the weight scale.
 *
 * Spec §6.2: "Show wRef in the legend so the scale is never a mystery." An
 * encoding the viewer cannot decode is decoration, so the reference value is
 * printed as a number, not implied by a gradient swatch alone.
 */

import { COLORS, FONTS, formatSigned, weightStroke, weightWidth } from '../theme';
import type { Ctx2D } from './context';

export const LEGEND_WIDTH = 168;
export const LEGEND_HEIGHT = 58;

export function drawLegend(
  ctx: Ctx2D,
  x: number,
  y: number,
  wRef: number,
  colorblindSafe: boolean,
  /** True while edges encode Δw against a pinned snapshot (§6.6). */
  diffing = false,
): void {
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.bgChassis;
  ctx.fillRect(x, y, LEGEND_WIDTH, LEGEND_HEIGHT);
  ctx.strokeStyle = COLORS.lineHair;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x + 0.5, y + 0.5, LEGEND_WIDTH - 1, LEGEND_HEIGHT - 1);
  ctx.stroke();

  ctx.font = `500 9px ${FONTS.mono}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = COLORS.textLo;
  // The label changes with the encoding: a scale that silently switches from
  // "weight" to "change in weight" would be read as the wrong quantity.
  ctx.fillText(diffing ? 'Δ WEIGHT vs A' : 'WEIGHT', x + 10, y + 8);

  // A swatch strip running from −wRef through 0 to +wRef, sampled at the same
  // ramp the edges use, so the legend cannot drift from the encoding.
  const stripY = y + 26;
  const stripX = x + 10;
  const stripW = LEGEND_WIDTH - 20;
  const samples = 24;
  ctx.lineCap = 'butt';
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    const w = (t * 2 - 1) * wRef;
    ctx.strokeStyle = weightStroke(w, wRef);
    ctx.lineWidth = weightWidth(w, wRef);
    if (colorblindSafe && w < 0) ctx.setLineDash([2, 2]);
    else ctx.setLineDash([]);
    const sx = stripX + (stripW * i) / samples;
    ctx.beginPath();
    ctx.moveTo(sx, stripY);
    ctx.lineTo(sx + stripW / samples, stripY);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.lineCap = 'round';

  ctx.font = `400 9px ${FONTS.mono}`;
  ctx.fillStyle = COLORS.textLo;
  ctx.textAlign = 'left';
  ctx.fillText(formatSigned(-wRef, 2), stripX, stripY + 8);
  ctx.textAlign = 'right';
  ctx.fillText(formatSigned(wRef, 2), stripX + stripW, stripY + 8);
  ctx.textAlign = 'center';
  ctx.fillText('0', stripX + stripW / 2, stripY + 8);
  ctx.textAlign = 'left';
}
