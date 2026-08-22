import { describe, expect, it } from 'vitest';
import {
  activationFill,
  COLORS,
  COLORBLIND_NEGATIVE_DASH,
  contrastRatio,
  cssVariables,
  EDGE_WIDTH_MAX,
  EDGE_WIDTH_MIN,
  formatLearningRate,
  formatLoss,
  formatNorm,
  formatPercent,
  formatSigned,
  installTheme,
  parseHex,
  perceptualDistance,
  simulateColorVision,
  WEIGHT_NEGATIVE,
  WEIGHT_POSITIVE,
  WEIGHT_RAMP_STEPS,
  weightStroke,
  weightWidth,
} from '../theme';
import type { ColorVisionDeficiency } from '../theme';

/*
 * These tests recompute every figure quoted in docs/DESIGN.md. A palette edit
 * that erodes contrast or collapses the two poles under colour-vision
 * deficiency fails here rather than being noticed by a user.
 */

describe('the two poles (§9 — the visual core)', () => {
  it('are both legible at 0.5px stroke on the canvas face', () => {
    // A hairline edge carrying a near-zero weight must still be visible.
    expect(contrastRatio(WEIGHT_NEGATIVE, COLORS.bgCanvas)).toBeGreaterThan(7);
    expect(contrastRatio(WEIGHT_POSITIVE, COLORS.bgCanvas)).toBeGreaterThan(7);
  });

  it('match the contrast figures documented in DESIGN.md', () => {
    expect(contrastRatio(WEIGHT_NEGATIVE, COLORS.bgCanvas)).toBeCloseTo(9.54, 1);
    expect(contrastRatio(WEIGHT_POSITIVE, COLORS.bgCanvas)).toBeCloseTo(9.28, 1);
  });

  it('stay distinguishable under every colour-vision deficiency', () => {
    // Separation must not merely survive — red/green would collapse here.
    const baseline = perceptualDistance(WEIGHT_NEGATIVE, WEIGHT_POSITIVE);
    expect(baseline).toBeGreaterThan(0.2);

    const kinds: ColorVisionDeficiency[] = ['protanopia', 'deuteranopia', 'tritanopia'];
    for (const kind of kinds) {
      const neg = simulateColorVision(WEIGHT_NEGATIVE, kind);
      const pos = simulateColorVision(WEIGHT_POSITIVE, kind);
      expect(perceptualDistance(neg, pos), kind).toBeGreaterThan(0.2);
      // And both remain visible against the canvas.
      expect(contrastRatio(neg, COLORS.bgCanvas), kind).toBeGreaterThan(3);
      expect(contrastRatio(pos, COLORS.bgCanvas), kind).toBeGreaterThan(3);
    }
  });

  it('offers a dash pattern as additional redundancy for negative weights', () => {
    expect(COLORBLIND_NEGATIVE_DASH.length).toBeGreaterThan(0);
  });
});

describe('text contrast (§9 quality floor)', () => {
  const surfaces = [COLORS.bgVoid, COLORS.bgCanvas, COLORS.bgChassis, COLORS.bgRaised];

  it('textHi and textMid clear WCAG AA on every surface', () => {
    for (const surface of surfaces) {
      expect(contrastRatio(COLORS.textHi, surface)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(COLORS.textMid, surface)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it('textLo clears AA-large only — it is restricted to non-essential labels', () => {
    for (const surface of surfaces) {
      const ratio = contrastRatio(COLORS.textLo, surface);
      expect(ratio).toBeGreaterThanOrEqual(3);
      // Documented as AA-large; if it ever clears 4.5 the restriction in
      // DESIGN.md should be relaxed deliberately rather than drift.
      expect(ratio).toBeLessThan(4.5);
    }
  });

  it('the error colour is readable on the chassis', () => {
    expect(contrastRatio(COLORS.statusBad, COLORS.bgChassis)).toBeGreaterThanOrEqual(4.5);
  });

  it('the grid is deliberately subliminal', () => {
    // Present but never competing with the network drawn on top of it.
    const ratio = contrastRatio(COLORS.lineGrid, COLORS.bgCanvas);
    expect(ratio).toBeGreaterThan(1);
    expect(ratio).toBeLessThan(1.5);
  });

  it('the canvas face is darker than the chassis around it', () => {
    // An instrument's screen is a hole in its housing, not a card above it.
    expect(contrastRatio(COLORS.bgCanvas, '#000000')).toBeLessThan(
      contrastRatio(COLORS.bgChassis, '#000000'),
    );
  });
});

describe('weight encoding (§6.2)', () => {
  it('maps sign to hue and magnitude to alpha', () => {
    expect(weightStroke(-1, 1)).toContain('62, 197, 232'); // cool
    expect(weightStroke(1, 1)).toContain('242, 163, 60'); // warm
  });

  it('fades near-zero weights almost out, so pruning is visible', () => {
    const faint = weightStroke(0.001, 1);
    const strong = weightStroke(1, 1);
    const alphaOf = (s: string): number => Number(s.slice(s.lastIndexOf(',') + 1, -1));
    expect(alphaOf(faint)).toBeLessThan(0.1);
    expect(alphaOf(strong)).toBeCloseTo(1, 2);
  });

  it('clamps beyond wRef instead of saturating invisibly', () => {
    expect(weightStroke(5, 1)).toBe(weightStroke(1, 1));
    expect(weightWidth(100, 1)).toBe(EDGE_WIDTH_MAX);
  });

  it('implements the documented width formula', () => {
    // width = clamp(0.5 + 5.5·|w|/wRef, 0.5, 7)
    expect(weightWidth(0, 1)).toBe(EDGE_WIDTH_MIN);
    expect(weightWidth(0.5, 1)).toBeCloseTo(0.5 + 5.5 * 0.5, 12);
    expect(weightWidth(1, 1)).toBeCloseTo(6, 12);
    expect(weightWidth(-1, 1)).toBeCloseTo(6, 12);
  });

  it('survives a zero reference without producing NaN', () => {
    // Every weight is zero — lesson 3's starting state.
    expect(weightWidth(0, 0)).toBe(EDGE_WIDTH_MIN);
    expect(weightStroke(0, 0)).toBe(weightStroke(0, 1));
  });

  it('allocates nothing per draw — the ramp is precomputed', () => {
    // §6.1: zero allocation in the hot path. Identical inputs must return the
    // identical string instance, which is only true if it came from the ramp.
    expect(weightStroke(0.5, 1)).toBe(weightStroke(0.5, 1));
    const distinct = new Set<string>();
    for (let i = 0; i <= 200; i++) distinct.add(weightStroke(i / 200, 1));
    expect(distinct.size).toBeLessThanOrEqual(WEIGHT_RAMP_STEPS);
  });

  it('is monotonic in magnitude', () => {
    let previous = -1;
    for (let i = 0; i <= 20; i++) {
      const width = weightWidth(i / 20, 1);
      expect(width).toBeGreaterThanOrEqual(previous);
      previous = width;
    }
  });
});

describe('activation fill', () => {
  it('uses the warm pole for positive and the cool pole for negative', () => {
    expect(activationFill(1)).toContain('242, 163, 60');
    expect(activationFill(-1)).toContain('62, 197, 232');
  });

  it('clamps out-of-range input', () => {
    expect(activationFill(5)).toBe(activationFill(1));
    expect(activationFill(-5)).toBe(activationFill(-1));
  });
});

describe('numeric formatting (§9 — values must never jitter in width)', () => {
  it('always renders a sign, so a column never shifts by a character', () => {
    expect(formatSigned(0.5)).toBe('+0.500');
    expect(formatSigned(-0.5)).toBe('−0.500');
    expect(formatSigned(0)).toBe('+0.000');
  });

  it('keeps signed values a constant width', () => {
    const widths = [-1.5, -0.25, 0, 0.25, 1.5, 9.999].map((v) => formatSigned(v).length);
    expect(new Set(widths).size).toBe(1);
  });

  it('keeps loss a constant width across its useful range', () => {
    const widths = [0, 0.0001, 0.5, 9.9999].map((v) => formatLoss(v).length);
    expect(new Set(widths).size).toBe(1);
  });

  it('handles non-finite values rather than printing garbage', () => {
    expect(formatLoss(NaN)).toBe('NaN');
    expect(formatLoss(Infinity)).toBe('∞');
    expect(formatSigned(NaN)).toBe('NaN');
    expect(formatSigned(-Infinity)).toBe('−∞');
    expect(formatNorm(NaN)).toBe('NaN');
    expect(formatPercent(NaN)).toBe('—');
    expect(formatLearningRate(NaN)).toBe('—');
  });

  it('formats percentages and learning rates', () => {
    expect(formatPercent(0.9512)).toBe('95.1%');
    expect(formatLearningRate(0.03)).toBe('0.0300');
    expect(formatLearningRate(1e-6)).toBe('1.00e-6');
  });

  it('formats gradient norms exponentially, including zero', () => {
    expect(formatNorm(0)).toBe('0.00e+0');
    expect(formatNorm(0.00123)).toBe('1.23e-3');
  });
});

describe('the CSS bridge', () => {
  it('exports every colour token as a custom property', () => {
    const vars = cssVariables();
    for (const name of Object.keys(COLORS)) {
      const kebab = name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
      expect(vars[`--color-${kebab}`]).toBeDefined();
    }
    expect(vars['--color-weight-negative']).toBe(WEIGHT_NEGATIVE);
    expect(vars['--font-mono']).toContain('IBM Plex Mono');
  });

  it('installs onto a root element', () => {
    const applied: Record<string, string> = {};
    installTheme({ style: { setProperty: (k, v) => (applied[k] = v) } });
    expect(applied['--color-bg-canvas']).toBe(COLORS.bgCanvas);
    expect(applied['--space-4']).toBe('12px');
  });
});

describe('parseHex', () => {
  it('parses a colour', () => {
    expect(parseHex('#FF8040')).toEqual({ r: 255, g: 128, b: 64 });
    expect(parseHex('3EC5E8')).toEqual({ r: 62, g: 197, b: 232 });
  });

  it('rejects malformed input', () => {
    expect(() => parseHex('#FFF')).toThrowError(/6-digit/);
    expect(() => parseHex('#GGGGGG')).toThrowError(/not a hex colour/);
  });
});
