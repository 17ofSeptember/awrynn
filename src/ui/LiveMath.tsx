import { useMemo, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import type { Matrix } from '../engine/tensor';

/*
 * The live math tab (§6.4).
 *
 * "for the current batch, render the actual matrices A⁽ˡ⁻¹⁾, W⁽ˡ⁾, Z⁽ˡ⁾, A⁽ˡ⁾,
 * dZ⁽ˡ⁾, dW⁽ˡ⁾ as labeled numeric grids with their shapes. Clicking a cell
 * highlights the corresponding node or edge on the canvas, and vice versa."
 *
 * The matrices are read from the engine's caches, exactly as the dissection
 * cards are: this tab and the canvas are two views of one set of numbers, which
 * is the entire point of putting them side by side.
 *
 * A small batch is used deliberately. §6.4 says "small networks, so they fit",
 * and a 240-row A matrix does not fit on any screen; showing the first few rows
 * of a real batch is honest and legible, where a scrollable wall of digits is
 * neither.
 */

const PREVIEW_ROWS = 4;
const PREVIEW_COLS = 8;

export function LiveMath(): JSX.Element {
  const network = useAppStore((s) => s.network);
  const dataset = useAppStore((s) => s.dataset);
  const epoch = useAppStore((s) => s.epoch);
  const editRevision = useAppStore((s) => s.editRevision);
  const selection = useAppStore((s) => s.selection);
  const setSelection = useAppStore((s) => s.setSelection);
  const layout = useAppStore((s) => s.layout);
  const [layer, setLayer] = useState(0);

  const ready = useMemo(() => {
    if (dataset.x.rows === 0 || network.inputSize !== dataset.x.cols) return false;
    // A real forward and backward pass on a small slice of the actual batch,
    // so every number below is one the engine produced.
    const rows = Math.min(PREVIEW_ROWS, dataset.x.rows);
    const x: Matrix = { rows, cols: dataset.x.cols, data: dataset.x.data.subarray(0, rows * dataset.x.cols) };
    const y: Matrix = { rows, cols: dataset.y.cols, data: dataset.y.data.subarray(0, rows * dataset.y.cols) };
    // inspect(): a panel re-rendering must not train the network. Without it,
    // opening this tab would nudge batch norm's running statistics and move the
    // decision boundary.
    network.inspect(() => {
      network.forward(x, true);
      network.backward(y);
    });
    return true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [network, dataset, epoch, editRevision]);

  const index = Math.min(layer, network.layers.length - 1);
  const dense = network.layers[index];

  if (!ready || dense === undefined) {
    return (
      <div className="px-4 py-3">
        <p className="panel-title mb-2">Live math</p>
        <p className="text-[11px] leading-relaxed text-[var(--color-text-lo)]">
          Nothing to show: the network and the dataset do not currently match.
        </p>
      </div>
    );
  }

  /** Selecting a cell of W selects the matching edge on the canvas. */
  const selectWeight = (row: number, col: number): void => {
    const edgeIndex = layout.edges.findIndex(
      (e) => e.layer === index && e.row === row && e.col === col && !e.isBias,
    );
    const edge = layout.edges[edgeIndex];
    if (edge === undefined) return;
    setSelection({ kind: 'edge', index: edgeIndex, edge, distance: 0 });
  };

  const selectedCell =
    selection !== null && selection.kind === 'edge' && selection.edge.layer === index && !selection.edge.isBias
      ? { row: selection.edge.row, col: selection.edge.col }
      : null;

  return (
    <div className="px-4 py-3">
      <div className="mb-3 flex items-baseline justify-between">
        <p className="panel-title">Live math</p>
        <select
          className="control"
          value={index}
          onChange={(e) => setLayer(Number(e.target.value))}
          aria-label="Layer"
        >
          {network.layers.map((l, i) => (
            <option key={i} value={i}>
              layer {i + 1} · {l.activationName}
            </option>
          ))}
        </select>
      </div>

      <p className="mb-3 text-[10px] leading-snug text-[var(--color-text-lo)]">
        Z = A·W + b, then A = φ(Z). Every number below is read from the engine after a real
        forward and backward pass over the first {PREVIEW_ROWS} samples.
      </p>

      <Grid title="A⁽ˡ⁻¹⁾" subtitle="input activations" matrix={dense.inputActivations} />
      <Grid
        title="W⁽ˡ⁾"
        subtitle="weights"
        matrix={dense.W}
        onCell={selectWeight}
        selected={selectedCell}
      />
      <Grid title="Z⁽ˡ⁾" subtitle="pre-activations" matrix={dense.Z} />
      <Grid title="A⁽ˡ⁾" subtitle="activations" matrix={dense.A} />
      <Grid title="dZ⁽ˡ⁾" subtitle="δ, from backprop" matrix={dense.dZ} />
      <Grid title="dW⁽ˡ⁾" subtitle="gradients, already ÷ B" matrix={dense.dW} />
    </div>
  );
}

function Grid({
  title,
  subtitle,
  matrix,
  onCell,
  selected,
}: {
  title: string;
  subtitle: string;
  matrix: Matrix | null;
  onCell?: (row: number, col: number) => void;
  selected?: { row: number; col: number } | null;
}): JSX.Element {
  if (matrix === null) {
    return (
      <div className="mb-3">
        <Header title={title} subtitle={subtitle} shape="not cached" />
      </div>
    );
  }

  const rows = Math.min(matrix.rows, PREVIEW_ROWS);
  const cols = Math.min(matrix.cols, PREVIEW_COLS);
  const truncated = rows < matrix.rows || cols < matrix.cols;

  return (
    <div className="mb-3">
      <Header title={title} subtitle={subtitle} shape={`[${matrix.rows}, ${matrix.cols}]`} />
      <div className="overflow-x-auto">
        <table className="num border-collapse text-[10px]">
          <tbody>
            {Array.from({ length: rows }, (_, r) => (
              <tr key={r}>
                {Array.from({ length: cols }, (_, c) => {
                  const value = matrix.data[r * matrix.cols + c] ?? 0;
                  const isSelected = selected?.row === r && selected?.col === c;
                  return (
                    <td
                      key={c}
                      onClick={onCell === undefined ? undefined : () => onCell(r, c)}
                      className="border border-[var(--color-line-hair)] px-1 py-[1px] text-right"
                      style={{
                        cursor: onCell === undefined ? 'default' : 'pointer',
                        background: isSelected ? 'var(--color-line-edge)' : 'transparent',
                        // Sign uses the weight poles, so a cell and its edge on
                        // the canvas read the same way.
                        color:
                          value < 0
                            ? 'var(--color-weight-negative)'
                            : value > 0
                              ? 'var(--color-weight-positive)'
                              : 'var(--color-text-lo)',
                      }}
                    >
                      {value.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated && (
        <p className="mt-0.5 text-[9px] text-[var(--color-text-lo)]">
          showing {rows}×{cols} of {matrix.rows}×{matrix.cols}
        </p>
      )}
    </div>
  );
}

function Header({
  title,
  subtitle,
  shape,
}: {
  title: string;
  subtitle: string;
  shape: string;
}): JSX.Element {
  return (
    <div className="mb-1 flex items-baseline gap-2">
      <span className="num text-[12px] text-[var(--color-text-hi)]">{title}</span>
      <span className="text-[10px] text-[var(--color-text-lo)]">{subtitle}</span>
      <span className="num ml-auto text-[10px] text-[var(--color-text-lo)]">{shape}</span>
    </div>
  );
}
