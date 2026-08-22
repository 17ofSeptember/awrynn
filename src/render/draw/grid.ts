/*
 * Background grid — the instrument face.
 *
 * Deliberately subliminal (1.14:1 against the canvas, see docs/DESIGN.md): it
 * gives the scene a sense of scale under pan and zoom without ever competing
 * with the network drawn on top of it.
 */

import { COLORS } from '../theme';
import type { Viewport } from '../layout';
import type { Ctx2D } from './context';

/** World-space grid pitch. */
export const GRID_PITCH = 40;
/** Below this screen pitch the grid is dropped rather than drawn as mush. */
const MIN_SCREEN_PITCH = 12;

export function drawGrid(
  ctx: Ctx2D,
  viewport: Viewport,
  width: number,
  height: number,
): void {
  ctx.fillStyle = COLORS.bgCanvas;
  ctx.fillRect(0, 0, width, height);

  const pitch = GRID_PITCH * viewport.scale;
  if (pitch < MIN_SCREEN_PITCH) return;

  ctx.strokeStyle = COLORS.lineGrid;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();

  // Start at the first gridline at or before the left edge, so the grid stays
  // anchored to world space and slides with the content under a pan.
  const firstX = Math.floor(-viewport.offsetX / pitch) * pitch + viewport.offsetX;
  for (let x = firstX; x <= width; x += pitch) {
    // +0.5 puts a 1px line on a pixel centre instead of straddling two.
    const px = Math.round(x) + 0.5;
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
  }

  const firstY = Math.floor(-viewport.offsetY / pitch) * pitch + viewport.offsetY;
  for (let y = firstY; y <= height; y += pitch) {
    const py = Math.round(y) + 0.5;
    ctx.moveTo(0, py);
    ctx.lineTo(width, py);
  }

  ctx.stroke();
}
