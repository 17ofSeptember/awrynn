import { describe, expect, it } from 'vitest';
import {
  addRowVector,
  addScaledInPlace,
  at,
  colSum,
  createMatrix,
  fromRows,
  matmul,
  matmulAT,
  matmulBT,
  setShapeAssertions,
  sub,
  sumSquares,
  toRows,
  transpose,
} from '../tensor';

/*
 * Spec §10: "matmul, transpose, broadcast-add, colSum against tiny hand-written
 * cases with asymmetric shapes (a symmetric test case will happily pass with
 * transposed code)."
 *
 * Every case below therefore uses distinct row and column counts, and no matrix
 * is symmetric. Expected values are computed by hand in the comments.
 */

describe('matmul — C = A · B', () => {
  it('multiplies [2,3] · [3,4] -> [2,4]', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = fromRows([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    // row0: 1*1+2*5+3*9=38, 1*2+2*6+3*10=44, 1*3+2*7+3*11=50, 1*4+2*8+3*12=56
    // row1: 4*1+5*5+6*9=83, 4*2+5*6+6*10=98, 4*3+5*7+6*11=113, 4*4+5*8+6*12=128
    const c = matmul(a, b);
    expect(c.rows).toBe(2);
    expect(c.cols).toBe(4);
    expect(toRows(c)).toEqual([
      [38, 44, 50, 56],
      [83, 98, 113, 128],
    ]);
  });

  it('rejects disagreeing inner dimensions with both shapes in the message', () => {
    const a = createMatrix(2, 3);
    const b = createMatrix(4, 5);
    expect(() => matmul(a, b)).toThrowError(/\[2, 3\].*\[4, 5\]/);
  });
});

describe('matmulAT — C = Aᵀ · B (the dW product)', () => {
  it('multiplies [3,2]ᵀ · [3,4] -> [2,4]', () => {
    const a = fromRows([
      [1, 4],
      [2, 5],
      [3, 6],
    ]); // Aᵀ = [[1,2,3],[4,5,6]]
    const b = fromRows([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    // identical to the matmul case above, since Aᵀ is that same [2,3]
    expect(toRows(matmulAT(a, b))).toEqual([
      [38, 44, 50, 56],
      [83, 98, 113, 128],
    ]);
  });

  it('agrees with materialising the transpose', () => {
    const a = fromRows([
      [2, -1, 0.5],
      [1, 3, -2],
      [0, 4, 1],
      [-3, 2, 7],
    ]); // [4,3]
    const b = fromRows([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]); // [4,2]
    expect(toRows(matmulAT(a, b))).toEqual(toRows(matmul(transpose(a), b)));
  });
});

describe('matmulBT — C = A · Bᵀ (the dA_prev product)', () => {
  it('multiplies [2,3] · [4,3]ᵀ -> [2,4]', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = fromRows([
      [1, 5, 9],
      [2, 6, 10],
      [3, 7, 11],
      [4, 8, 12],
    ]); // Bᵀ is the [3,4] from the matmul case
    expect(toRows(matmulBT(a, b))).toEqual([
      [38, 44, 50, 56],
      [83, 98, 113, 128],
    ]);
  });

  it('agrees with materialising the transpose', () => {
    const a = fromRows([
      [2, -1, 0.5, 4],
      [1, 3, -2, 0],
      [8, -5, 1, 2],
    ]); // [3,4]
    const b = fromRows([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]); // [2,4]
    expect(toRows(matmulBT(a, b))).toEqual(toRows(matmul(a, transpose(b))));
  });
});

describe('transpose', () => {
  it('flips a [2,3] into a [3,2]', () => {
    const m = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const t = transpose(m);
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(toRows(t)).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it('is an involution', () => {
    const m = fromRows([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    expect(toRows(transpose(transpose(m)))).toEqual(toRows(m));
  });
});

describe('addRowVector — the 1_B · b broadcast', () => {
  it('adds a [1,3] down all rows of a [4,3]', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const v = fromRows([[100, 200, 300]]);
    expect(toRows(addRowVector(a, v))).toEqual([
      [101, 202, 303],
      [104, 205, 306],
      [107, 208, 309],
      [110, 211, 312],
    ]);
  });

  it('rejects a column vector — the classic transposed-bias bug', () => {
    const a = createMatrix(4, 3);
    const v = createMatrix(3, 1);
    expect(() => addRowVector(a, v)).toThrowError(/\[3, 1\].*\[4, 3\].*\[1, 3\]/);
  });
});

describe('colSum — the db reduction', () => {
  it('sums down a [4,3] into a [1,3]', () => {
    const a = fromRows([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const s = colSum(a);
    expect(s.rows).toBe(1);
    expect(s.cols).toBe(3);
    // columns: 1+4+7+10=22, 2+5+8+11=26, 3+6+9+12=30
    expect(toRows(s)).toEqual([[22, 26, 30]]);
  });

  it('sums columns, not rows (a [1,4] result would mean rowSum)', () => {
    const a = fromRows([
      [1, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 3, 0],
    ]);
    expect(toRows(colSum(a))).toEqual([[1, 2, 3, 0]]);
  });
});

describe('elementwise helpers', () => {
  it('sub subtracts', () => {
    const a = fromRows([
      [5, 6, 7],
      [8, 9, 10],
    ]);
    const b = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(toRows(sub(a, b))).toEqual([
      [4, 4, 4],
      [4, 4, 4],
    ]);
  });

  it('addScaledInPlace computes a += b * s', () => {
    const a = fromRows([[1, 2, 3]]);
    const b = fromRows([[10, 20, 30]]);
    addScaledInPlace(a, b, -0.5);
    expect(toRows(a)).toEqual([[-4, -8, -12]]);
  });

  it('sumSquares totals every entry squared', () => {
    const m = fromRows([
      [1, -2],
      [3, -4],
      [0.5, 0],
    ]);
    expect(sumSquares(m)).toBeCloseTo(1 + 4 + 9 + 16 + 0.25, 12);
  });

  it('at reads row-major', () => {
    const m = fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(at(m, 0, 2)).toBe(3);
    expect(at(m, 1, 0)).toBe(4);
  });
});

describe('shape assertions', () => {
  it('can be disabled, and then no longer guard', () => {
    setShapeAssertions(false);
    try {
      const a = createMatrix(2, 3);
      const b = createMatrix(4, 5);
      // With assertions off the op runs on garbage rather than throwing; the
      // point of the flag is that the host chooses, so verify it really is off.
      expect(() => matmul(a, b)).not.toThrowError(/inner dimensions/);
    } finally {
      setShapeAssertions(true);
    }
    const a = createMatrix(2, 3);
    const b = createMatrix(4, 5);
    expect(() => matmul(a, b)).toThrowError(/inner dimensions/);
  });
});
