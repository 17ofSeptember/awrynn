/*
 * The subset of Canvas 2D this app actually uses.
 *
 * Declaring it explicitly does two things: it keeps the render layer honest
 * about its surface area, and it makes every draw module testable with a
 * recording stub instead of a real canvas — which is how draw ORDER (§6.1) is
 * verified without pixel snapshots.
 */

export interface Ctx2D {
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
  lineCap: CanvasLineCap;
  lineJoin: CanvasLineJoin;
  font: string;
  textAlign: CanvasTextAlign;
  textBaseline: CanvasTextBaseline;
  globalAlpha: number;

  save(): void;
  restore(): void;
  beginPath(): void;
  closePath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, start: number, end: number): void;
  rect(x: number, y: number, w: number, h: number): void;
  fill(): void;
  stroke(): void;
  fillRect(x: number, y: number, w: number, h: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  fillText(text: string, x: number, y: number): void;
  measureText(text: string): { width: number };
  setLineDash(segments: number[]): void;
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;
}

export const TAU = Math.PI * 2;
