import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { ACTIVATION_NAMES } from '../engine/activations';
import type { ActivationName } from '../engine/activations';

/*
 * Architecture controls.
 *
 * Spec §6.5: "Add/remove layers; add/remove neurons per layer (+/− on the layer
 * header)." Full editing semantics — preserving existing parameters across an
 * edit — belong to Phase 5; this panel is the Phase 3 surface for proving that
 * an arbitrary architecture renders.
 */

export function ArchitecturePanel(): JSX.Element {
  const architecture = useAppStore((s) => s.architecture);
  const setArchitecture = useAppStore((s) => s.setArchitecture);
  const configErrors = useAppStore((s) => s.configErrors);

  const setUnits = (index: number, delta: number): void => {
    const layers = architecture.layers.map((layer, i) =>
      i === index ? { ...layer, units: Math.max(1, layer.units + delta) } : layer,
    );
    setArchitecture({ layers });
  };

  const setActivation = (index: number, activation: ActivationName): void => {
    const layers = architecture.layers.map((layer, i) =>
      i === index ? { ...layer, activation } : layer,
    );
    setArchitecture({ layers });
  };

  const addLayer = (): void => {
    const layers = [
      ...architecture.layers.slice(0, -1),
      { units: 4, activation: 'tanh' as ActivationName },
      ...architecture.layers.slice(-1),
    ];
    setArchitecture({ layers });
  };

  const removeLayer = (index: number): void => {
    if (architecture.layers.length <= 1) return;
    setArchitecture({ layers: architecture.layers.filter((_, i) => i !== index) });
  };

  /*
   * Batch norm on the hidden layers, all together, the same granularity dropout
   * uses and for the same reason: normalizing the output layer is a research
   * knob rather than a teaching one. The engine supports it per layer, and the
   * gradient check covers a normalized softmax output.
   *
   * Rebuilding the network is not a loss here. γ is filled with ones rather
   * than sampled, so it never touches the init stream, and W and b come back
   * bit-identical at the same seed. Flipping this switch changes exactly one
   * thing, which is what makes comparing the two sides of it worth anything.
   */
  const hidden = architecture.layers.slice(0, -1);
  const normalized = hidden.filter((l) => l.batchNorm === true).length;
  const toggleBatchNorm = (): void => {
    const enable = normalized < hidden.length;
    const layers = architecture.layers.map((layer, i) => {
      if (i === architecture.layers.length - 1) return layer;
      if (!enable) {
        // Omitted rather than set to false, so a spec that never used batch
        // norm keeps encoding exactly as it did before batch norm existed.
        const rest = { ...layer };
        delete (rest as { batchNorm?: boolean }).batchNorm;
        return rest;
      }
      return { ...layer, batchNorm: true };
    });
    setArchitecture({ layers });
  };

  return (
    <div className="px-4 py-3">
      <div className="mb-3 flex items-baseline justify-between">
        <p className="panel-title">Architecture</p>
        <span className="num text-[11px] text-[var(--color-text-lo)]">
          {[architecture.inputSize, ...architecture.layers.map((l) => l.units)].join('-')}
        </span>
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="label">input</span>
          <span className="num text-[12px] text-[var(--color-text-mid)]">
            {architecture.inputSize}
          </span>
        </div>

        {architecture.layers.map((layer, i) => {
          const isOutput = i === architecture.layers.length - 1;
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="label w-12 shrink-0">{isOutput ? 'out' : `h${i + 1}`}</span>
              <button
                className="control px-1.5 py-0"
                onClick={() => setUnits(i, -1)}
                disabled={layer.units <= 1}
                aria-label={`Remove a unit from layer ${i + 1}`}
              >
                −
              </button>
              <span className="num w-6 text-center text-[12px]">{layer.units}</span>
              <button
                className="control px-1.5 py-0"
                onClick={() => setUnits(i, 1)}
                aria-label={`Add a unit to layer ${i + 1}`}
              >
                +
              </button>
              <select
                className="control min-w-0 flex-1"
                value={layer.activation}
                onChange={(e) => setActivation(i, e.target.value as ActivationName)}
                aria-label={`Activation for layer ${i + 1}`}
              >
                {ACTIVATION_NAMES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              {!isOutput && (
                <button
                  className="control px-1.5 py-0"
                  onClick={() => removeLayer(i)}
                  aria-label={`Remove layer ${i + 1}`}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>

      <button className="control mt-3 w-full" onClick={addLayer}>
        Add hidden layer
      </button>

      {hidden.length > 0 && (
        <label className="mt-2 flex cursor-pointer items-center justify-between gap-4 py-1">
          <span className="text-[12px] text-[var(--color-text-mid)]">
            Batch norm
            <span className="label ml-2">{normalized > 0 ? 'hidden layers' : 'off'}</span>
          </span>
          <input
            type="checkbox"
            checked={normalized > 0}
            onChange={toggleBatchNorm}
            className="h-3.5 w-3.5 accent-[var(--color-weight-positive)]"
            aria-label="Batch normalization on the hidden layers"
          />
        </label>
      )}

      {configErrors.length > 0 && (
        <div className="mt-3 border-l-2 border-[var(--color-status-bad)] pl-3">
          {configErrors.map((error) => (
            <p key={error} className="text-[11px] leading-relaxed text-[var(--color-status-bad)]">
              {error}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
