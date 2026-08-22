import { useEffect, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { formatSigned } from '../render/theme';
import type { DenseLayer } from '../engine/layers';

/*
 * The inspector, and the edit surface for whatever is selected.
 *
 * Spec §6.2: "Every colour-encoded quantity is also readable as a number in the
 * inspector." §6.5 adds editing: type an exact weight, set a bias, freeze or
 * ablate a layer.
 *
 * Typing is deliberately committed on blur or Enter rather than per keystroke.
 * Committing per keystroke would push an undo entry for every character and
 * make "0.5" pass through "0" and "0." on the way, each of which is a different
 * network.
 */

export function Inspector(): JSX.Element {
  const selection = useAppStore((s) => s.selection);
  const hover = useAppStore((s) => s.hover);
  const layout = useAppStore((s) => s.layout);
  const network = useAppStore((s) => s.network);
  const editRevision = useAppStore((s) => s.editRevision);
  const parameterIndex = useAppStore((s) => s.parameterIndex);
  const setParameter = useAppStore((s) => s.setParameter);
  const toggleFrozen = useAppStore((s) => s.toggleFrozen);
  const toggleAblated = useAppStore((s) => s.toggleAblated);
  const randomizeLayer = useAppStore((s) => s.randomizeLayer);

  // Hover is a preview; a click holds it so the value can be edited without
  // the pointer having to stay on a 1px line.
  const target = selection ?? hover;
  const editable = selection !== null;

  if (target === null) {
    return (
      <div className="px-4 py-3">
        <p className="panel-title mb-2">Inspector</p>
        <p className="text-[12px] leading-relaxed text-[var(--color-text-lo)]">
          Hover a connection to read its weight. Click to hold it, then type an exact
          value or drag the connection sideways to scrub it.
        </p>
      </div>
    );
  }

  if (target.kind === 'edge') {
    const edge = target.edge;
    const layer = network.layers[edge.layer];
    const index = parameterIndex(edge.layer, edge.isBias ? 'b' : 'W', edge.row, edge.col);
    const value = index >= 0 ? (network.params[index] as number) : 0;

    return (
      <div className="px-4 py-3">
        <p className="panel-title mb-3">{edge.isBias ? 'Bias' : 'Weight'}</p>
        <Row label="address" value={edge.isBias ? `b[${edge.col}]` : `W[${edge.row}, ${edge.col}]`} />
        <Row label="layer" value={`${edge.layer + 1} of ${network.layers.length}`} />

        {editable ? (
          <NumericEditor
            key={`${index}:${editRevision}`}
            label="value"
            value={value}
            onCommit={(next) => setParameter(index, next, 'Set weight')}
          />
        ) : (
          <Row label="value" value={formatSigned(value, 4)} emphasis />
        )}

        {editable && (
          <>
            <input
              type="range"
              className="mt-2 w-full accent-[var(--color-weight-positive)]"
              min={-5}
              max={5}
              step={0.01}
              value={Math.max(-5, Math.min(5, value))}
              onChange={(e) => setParameter(index, Number(e.target.value), 'Set weight')}
              aria-label="Weight"
            />
            <div className="flex justify-between">
              <span className="label">−5</span>
              <span className="label">+5</span>
            </div>
          </>
        )}

        {layer !== undefined && (
          <p className="mt-3 text-[11px] leading-relaxed text-[var(--color-text-lo)]">
            {edge.isBias
              ? 'Added to this neuron regardless of its input.'
              : `Multiplies unit ${edge.row} of the previous column on its way into unit ${edge.col}.`}
          </p>
        )}
      </div>
    );
  }

  const node = layout.nodes[target.index];
  if (node === undefined) return <div className="px-4 py-3" />;
  const denseIndex = node.layer - 1;
  const dense = network.layers[denseIndex];
  const biasIndex = dense === undefined ? -1 : parameterIndex(denseIndex, 'b', 0, node.unit);

  return (
    <div className="px-4 py-3">
      <p className="panel-title mb-3">{node.kind === 'bias' ? 'Bias unit' : 'Neuron'}</p>
      <Row label="column" value={node.layer === 0 ? 'input' : `layer ${node.layer}`} />
      <Row label="unit" value={String(node.unit)} />

      {dense !== undefined && (
        <>
          <Row label="activation" value={dense.activationName} />
          <Row label="fan in" value={String(dense.inputs)} />
          {editable && biasIndex >= 0 ? (
            <NumericEditor
              key={`b${biasIndex}:${editRevision}`}
              label="bias"
              value={network.params[biasIndex] as number}
              onCommit={(next) => setParameter(biasIndex, next, 'Set bias')}
            />
          ) : (
            <Row label="bias" value={formatSigned(dense.b.data[node.unit] ?? 0, 4)} />
          )}

          {dense.batchNorm && <BatchNormRows layer={dense} unit={node.unit} index={denseIndex} />}

          {editable && node.kind !== 'bias' && (
            <div className="mt-3 flex flex-col gap-1.5">
              <Toggle
                label="Freeze layer"
                hint="Still forward-propagates, but the optimizer leaves it alone."
                checked={dense.frozen}
                onChange={() => toggleFrozen(denseIndex)}
              />
              <Toggle
                label="Ablate layer"
                hint="Output forced to zero, to see what the rest of the network can do without it."
                checked={dense.ablated}
                onChange={() => toggleAblated(denseIndex)}
              />
              <button className="control mt-1" onClick={() => randomizeLayer(denseIndex)}>
                Randomize this layer
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/**
 * What batch normalization did to this unit, in the order it did it.
 *
 * γ is editable because it is a parameter like any other and scrubbing it is
 * instructive: it is the only dial that says how much of the normalized signal
 * survives. μ̂ and σ̂² are read-only because they are MEASURED, not learned, and
 * a field that let you type over a measurement would be inviting a lie.
 *
 * The statistics shown are the running ones, which is what a prediction uses.
 * During training the layer normalizes by the batch instead, and those numbers
 * live in the dissection view where a specific sample is on screen to attach
 * them to.
 */
function BatchNormRows({
  layer,
  unit,
  index,
}: {
  layer: DenseLayer;
  unit: number;
  index: number;
}): JSX.Element {
  const parameterIndex = useAppStore((s) => s.parameterIndex);
  const setParameter = useAppStore((s) => s.setParameter);
  const selection = useAppStore((s) => s.selection);
  const editRevision = useAppStore((s) => s.editRevision);
  const network = useAppStore((s) => s.network);

  const gammaIndex = parameterIndex(index, 'gamma', 0, unit);
  const variance = layer.runningVar[unit] ?? 1;

  return (
    <>
      <div className="mt-3 border-t border-[var(--color-line-hair)] pt-2">
        <p className="label mb-1.5">batch norm</p>
      </div>
      {selection !== null && gammaIndex >= 0 ? (
        <NumericEditor
          key={`g${gammaIndex}:${editRevision}`}
          label="γ scale"
          value={network.params[gammaIndex] as number}
          onCommit={(next) => setParameter(gammaIndex, next, 'Set γ')}
        />
      ) : (
        <Row label="γ scale" value={formatSigned(layer.gamma.data[unit] ?? 1, 4)} />
      )}
      <Row label="μ̂ running" value={formatSigned(layer.runningMean[unit] ?? 0, 4)} />
      <Row label="σ̂ running" value={formatSigned(Math.sqrt(variance), 4)} />
      <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
        Predictions divide by σ̂ and subtract μ̂, measured over training. While
        training, this unit normalizes by its own batch instead.
      </p>
    </>
  );
}

/**
 * A number field that commits on Enter or blur.
 *
 * Local state while typing so a partially typed value like "-" or "0." never
 * reaches the network, and one edit produces one undo entry.
 */
function NumericEditor({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: number;
  onCommit: (value: number) => void;
}): JSX.Element {
  const [draft, setDraft] = useState(() => value.toFixed(4));
  useEffect(() => {
    setDraft(value.toFixed(4));
  }, [value]);

  const commit = (): void => {
    const next = Number(draft);
    if (Number.isFinite(next)) onCommit(next);
    else setDraft(value.toFixed(4));
  };

  return (
    <label className="mt-1 flex items-center justify-between gap-3 py-[3px]">
      <span className="label">{label}</span>
      <input
        className="control num w-28 text-right"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit();
            (e.target as HTMLInputElement).blur();
          }
          if (e.key === 'Escape') setDraft(value.toFixed(4));
        }}
        aria-label={label}
      />
    </label>
  );
}

function Row({
  label,
  value,
  emphasis = false,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[3px]">
      <span className="label">{label}</span>
      <span
        className={`num text-[12px] ${
          emphasis ? 'text-[var(--color-text-hi)]' : 'text-[var(--color-text-mid)]'
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: () => void;
}): JSX.Element {
  return (
    <div>
      <label className="flex cursor-pointer items-center justify-between gap-3">
        <span className="text-[12px] text-[var(--color-text-mid)]">{label}</span>
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          className="h-3.5 w-3.5 accent-[var(--color-weight-positive)]"
        />
      </label>
      <p className="text-[10px] leading-snug text-[var(--color-text-lo)]">{hint}</p>
    </div>
  );
}
