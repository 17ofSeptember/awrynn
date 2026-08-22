/*
 * Matrix type and operations, over Float64Array.
 *
 * Spec §4.1. The row-major batch convention is fixed and not negotiable:
 *
 *   X   (batch input)      [B, n0]           one ROW per sample
 *   W^l (layer weights)    [n_{l-1}, n_l]
 *   b^l (layer biases)     [1, n_l]          broadcast down the batch
 *   A^l, Z^l               [B, n_l]
 *
 * float64 throughout, deliberately: these networks have tens of parameters, so
 * there is no memory argument for float32, and the gradient check in §4.11
 * needs the headroom to hit relErr < 1e-7.
 *
 * CONVENTION — typed-array reads use `!`. `noUncheckedIndexedAccess` is on
 * (spec §0.3), so `data[i]` types as `number | undefined`. Every read below is
 * inside a loop bounded by `rows * cols`, i.e. provably in range, so the
 * assertion states a fact the compiler cannot see rather than papering over an
 * unknown. Plain arrays elsewhere in the engine still handle undefined honestly.
 */

export interface Matrix {
  readonly rows: number;
  readonly cols: number;
  /** Row-major, length rows * cols. */
  readonly data: Float64Array;
}

/* ------------------------------------------------------------------ *
 * Shape assertions
 * ------------------------------------------------------------------ */

/*
 * Spec §4.1 asks for shape assertions "in development builds". The engine has
 * `types: []` and must run in bare Node (§0.5), so it cannot read
 * import.meta.env or process.env to find out what kind of build it is in.
 * Instead the flag lives here and the host decides. Default ON: a wrong shape
 * is the single most common backprop bug, and the cost is a few comparisons
 * against matmuls of tens of elements.
 */
let assertionsEnabled = true;

export function setShapeAssertions(enabled: boolean): void {
  assertionsEnabled = enabled;
}

export function shapeAssertionsEnabled(): boolean {
  return assertionsEnabled;
}

function shape(m: Matrix): string {
  return `[${m.rows}, ${m.cols}]`;
}

function fail(op: string, message: string): never {
  throw new Error(`tensor.${op}: ${message}`);
}

/* ------------------------------------------------------------------ *
 * Construction
 * ------------------------------------------------------------------ */

export function createMatrix(rows: number, cols: number): Matrix {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
    fail('createMatrix', `rows and cols must be non-negative integers, got [${rows}, ${cols}].`);
  }
  return { rows, cols, data: new Float64Array(rows * cols) };
}

export function fromRows(values: readonly (readonly number[])[]): Matrix {
  const rows = values.length;
  const cols = rows === 0 ? 0 : (values[0] as readonly number[]).length;
  const out = createMatrix(rows, cols);
  for (let r = 0; r < rows; r++) {
    const row = values[r] as readonly number[];
    if (row.length !== cols) {
      fail('fromRows', `row ${r} has length ${row.length}, expected ${cols}.`);
    }
    for (let c = 0; c < cols; c++) {
      out.data[r * cols + c] = row[c] as number;
    }
  }
  return out;
}

export function fromArray(rows: number, cols: number, values: readonly number[]): Matrix {
  if (values.length !== rows * cols) {
    fail('fromArray', `expected ${rows * cols} values for [${rows}, ${cols}], got ${values.length}.`);
  }
  const out = createMatrix(rows, cols);
  out.data.set(values);
  return out;
}

export function cloneMatrix(m: Matrix): Matrix {
  return { rows: m.rows, cols: m.cols, data: Float64Array.from(m.data) };
}

export function toRows(m: Matrix): number[][] {
  const out: number[][] = [];
  for (let r = 0; r < m.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < m.cols; c++) row.push(m.data[r * m.cols + c]!);
    out.push(row);
  }
  return out;
}

export function at(m: Matrix, row: number, col: number): number {
  if (assertionsEnabled && (row < 0 || row >= m.rows || col < 0 || col >= m.cols)) {
    fail('at', `index (${row}, ${col}) out of bounds for ${shape(m)}.`);
  }
  return m.data[row * m.cols + col]!;
}

export function setAt(m: Matrix, row: number, col: number, value: number): void {
  if (assertionsEnabled && (row < 0 || row >= m.rows || col < 0 || col >= m.cols)) {
    fail('setAt', `index (${row}, ${col}) out of bounds for ${shape(m)}.`);
  }
  m.data[row * m.cols + col] = value;
}

/**
 * Return `existing` if it already has this shape, otherwise a fresh matrix.
 * Layers hold their caches through this so that a steady batch size allocates
 * once and a changed batch size reallocates exactly once.
 */
export function ensureShape(existing: Matrix | null, rows: number, cols: number): Matrix {
  if (existing !== null && existing.rows === rows && existing.cols === cols) return existing;
  return createMatrix(rows, cols);
}

/* ------------------------------------------------------------------ *
 * Products
 *
 * Three separate entry points rather than one matmul plus transposes. The
 * backward pass needs A^T·B and A·B^T, and materialising the transpose would
 * allocate a matrix per layer per step purely to be read once.
 * ------------------------------------------------------------------ */

/** C = A · B — [m, k] · [k, n] -> [m, n] */
export function matmul(a: Matrix, b: Matrix, out: Matrix | null = null): Matrix {
  if (assertionsEnabled && a.cols !== b.rows) {
    fail('matmul', `inner dimensions disagree: ${shape(a)} · ${shape(b)}.`);
  }
  const m = a.rows;
  const k = a.cols;
  const n = b.cols;
  const c = ensureShape(out, m, n);
  if (assertionsEnabled && (c.rows !== m || c.cols !== n)) {
    fail('matmul', `output ${shape(c)} cannot hold ${shape(a)} · ${shape(b)}.`);
  }
  c.data.fill(0);
  // i-p-j order: walks both B and C along rows, which is the cache-friendly
  // direction for row-major storage.
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const cRow = i * n;
    for (let p = 0; p < k; p++) {
      const aVal = a.data[aRow + p]!;
      if (aVal === 0) continue;
      const bRow = p * n;
      for (let j = 0; j < n; j++) {
        c.data[cRow + j] = c.data[cRow + j]! + aVal * b.data[bRow + j]!;
      }
    }
  }
  return c;
}

/** C = Aᵀ · B — [k, m]ᵀ · [k, n] -> [m, n]. Used for dW = (A^{l-1})ᵀ · dZ. */
export function matmulAT(a: Matrix, b: Matrix, out: Matrix | null = null): Matrix {
  if (assertionsEnabled && a.rows !== b.rows) {
    fail('matmulAT', `row counts disagree for Aᵀ·B: ${shape(a)}ᵀ · ${shape(b)}.`);
  }
  const k = a.rows;
  const m = a.cols;
  const n = b.cols;
  const c = ensureShape(out, m, n);
  if (assertionsEnabled && (c.rows !== m || c.cols !== n)) {
    fail('matmulAT', `output ${shape(c)} cannot hold ${shape(a)}ᵀ · ${shape(b)}.`);
  }
  c.data.fill(0);
  for (let p = 0; p < k; p++) {
    const aRow = p * m;
    const bRow = p * n;
    for (let i = 0; i < m; i++) {
      const aVal = a.data[aRow + i]!;
      if (aVal === 0) continue;
      const cRow = i * n;
      for (let j = 0; j < n; j++) {
        c.data[cRow + j] = c.data[cRow + j]! + aVal * b.data[bRow + j]!;
      }
    }
  }
  return c;
}

/** C = A · Bᵀ — [m, k] · [n, k]ᵀ -> [m, n]. Used for dA^{l-1} = dZ · (W^l)ᵀ. */
export function matmulBT(a: Matrix, b: Matrix, out: Matrix | null = null): Matrix {
  if (assertionsEnabled && a.cols !== b.cols) {
    fail('matmulBT', `column counts disagree for A·Bᵀ: ${shape(a)} · ${shape(b)}ᵀ.`);
  }
  const m = a.rows;
  const k = a.cols;
  const n = b.rows;
  const c = ensureShape(out, m, n);
  if (assertionsEnabled && (c.rows !== m || c.cols !== n)) {
    fail('matmulBT', `output ${shape(c)} cannot hold ${shape(a)} · ${shape(b)}ᵀ.`);
  }
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const cRow = i * n;
    for (let j = 0; j < n; j++) {
      const bRow = j * k;
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a.data[aRow + p]! * b.data[bRow + p]!;
      c.data[cRow + j] = sum;
    }
  }
  return c;
}

export function transpose(m: Matrix, out: Matrix | null = null): Matrix {
  const t = ensureShape(out, m.cols, m.rows);
  if (assertionsEnabled && (t.rows !== m.cols || t.cols !== m.rows)) {
    fail('transpose', `output ${shape(t)} cannot hold ${shape(m)}ᵀ.`);
  }
  for (let r = 0; r < m.rows; r++) {
    for (let c = 0; c < m.cols; c++) {
      t.data[c * m.rows + r] = m.data[r * m.cols + c]!;
    }
  }
  return t;
}

/* ------------------------------------------------------------------ *
 * Broadcast and reduction
 * ------------------------------------------------------------------ */

/** out[r, c] = a[r, c] + v[0, c] — the `1_B · b` term of Z = A·W + 1_B·b. */
export function addRowVector(a: Matrix, v: Matrix, out: Matrix | null = null): Matrix {
  if (assertionsEnabled && (v.rows !== 1 || v.cols !== a.cols)) {
    fail('addRowVector', `cannot broadcast ${shape(v)} across ${shape(a)}; expected [1, ${a.cols}].`);
  }
  const c = ensureShape(out, a.rows, a.cols);
  for (let r = 0; r < a.rows; r++) {
    const row = r * a.cols;
    for (let j = 0; j < a.cols; j++) {
      c.data[row + j] = a.data[row + j]! + v.data[j]!;
    }
  }
  return c;
}

/** Column sums as a [1, cols] row vector — db = colSum(dZ) before the /B. */
export function colSum(a: Matrix, out: Matrix | null = null): Matrix {
  const c = ensureShape(out, 1, a.cols);
  if (assertionsEnabled && (c.rows !== 1 || c.cols !== a.cols)) {
    fail('colSum', `output ${shape(c)} cannot hold colSum of ${shape(a)}.`);
  }
  c.data.fill(0);
  for (let r = 0; r < a.rows; r++) {
    const row = r * a.cols;
    for (let j = 0; j < a.cols; j++) c.data[j] = c.data[j]! + a.data[row + j]!;
  }
  return c;
}

/* ------------------------------------------------------------------ *
 * Elementwise
 * ------------------------------------------------------------------ */

function assertSameShape(op: string, a: Matrix, b: Matrix): void {
  if (assertionsEnabled && (a.rows !== b.rows || a.cols !== b.cols)) {
    fail(op, `shapes disagree: ${shape(a)} vs ${shape(b)}.`);
  }
}

/** out = a ⊙ b */
export function mul(a: Matrix, b: Matrix, out: Matrix | null = null): Matrix {
  assertSameShape('mul', a, b);
  const c = ensureShape(out, a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i]! * b.data[i]!;
  return c;
}

/** a ⊙= b, in place. */
export function mulInPlace(a: Matrix, b: Matrix): Matrix {
  assertSameShape('mulInPlace', a, b);
  for (let i = 0; i < a.data.length; i++) a.data[i] = a.data[i]! * b.data[i]!;
  return a;
}

/** out = a − b */
export function sub(a: Matrix, b: Matrix, out: Matrix | null = null): Matrix {
  assertSameShape('sub', a, b);
  const c = ensureShape(out, a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i]! - b.data[i]!;
  return c;
}

/** a += b · scale, in place. */
export function addScaledInPlace(a: Matrix, b: Matrix, scale: number): Matrix {
  assertSameShape('addScaledInPlace', a, b);
  for (let i = 0; i < a.data.length; i++) a.data[i] = a.data[i]! + b.data[i]! * scale;
  return a;
}

/** a *= s, in place. */
export function scaleInPlace(a: Matrix, s: number): Matrix {
  for (let i = 0; i < a.data.length; i++) a.data[i] = a.data[i]! * s;
  return a;
}

export function fill(a: Matrix, value: number): Matrix {
  a.data.fill(value);
  return a;
}

/** Σ a_ij² — the L2 term needs this per weight matrix. */
export function sumSquares(a: Matrix): number {
  let sum = 0;
  for (let i = 0; i < a.data.length; i++) {
    const v = a.data[i]!;
    sum += v * v;
  }
  return sum;
}

/**
 * A view of the first `rows` rows of `m`, sharing storage.
 *
 * The trainer uses this for the final partial batch: the batch buffers are
 * allocated once at full size, and the short batch is a view rather than a
 * reallocation. Because it is a view, the /B in backprop divides by the TRUE
 * batch size, which is what §4.9 requires of the partial batch.
 */
export function rowView(m: Matrix, rows: number): Matrix {
  if (rows < 0 || rows > m.rows) {
    fail('rowView', `cannot take ${rows} rows from ${shape(m)}.`);
  }
  if (rows === m.rows) return m;
  return { rows, cols: m.cols, data: m.data.subarray(0, rows * m.cols) };
}

export function sameShape(a: Matrix, b: Matrix): boolean {
  return a.rows === b.rows && a.cols === b.cols;
}

/** True when every entry is finite — the divergence detector (§7.4). */
export function isFinite(a: Matrix): boolean {
  for (let i = 0; i < a.data.length; i++) {
    if (!Number.isFinite(a.data[i]!)) return false;
  }
  return true;
}
