import { useEffect, useMemo, useRef, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { LONG_LINK_THRESHOLD } from '../state/shareLink';
import { checkXor, CONVINCING_MARGIN, XOR_SOLUTION } from '../engine/xor';
import { fromRows } from '../engine/tensor';
import { generateJavaScript, generateNumpy } from '../engine/codegen';

/*
 * Editing, persistence and the XOR challenge (§6.5).
 *
 * Undo/redo is bound at the window rather than to a focused element, because
 * most edits here happen on a canvas that never takes focus.
 */

export function EditPanel(): JSX.Element {
  const undo = useAppStore((s) => s.undo);
  const redo = useAppStore((s) => s.redo);
  const canUndo = useAppStore((s) => s.canUndo);
  const canRedo = useAppStore((s) => s.canRedo);
  const undoLabel = useAppStore((s) => s.undoLabel);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target;
      // Never steal undo from a text field the learner is typing in.
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
      const mod = event.metaKey || event.ctrlKey;
      if (!mod || event.key.toLowerCase() !== 'z') return;
      event.preventDefault();
      if (event.shiftKey) redo();
      else undo();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo]);

  return (
    <div className="px-4 py-3">
      <p className="panel-title mb-3">Edit</p>
      <div className="flex gap-1.5">
        <button className="control flex-1" onClick={undo} disabled={!canUndo} title="⌘Z">
          Undo
        </button>
        <button className="control flex-1" onClick={redo} disabled={!canRedo} title="⇧⌘Z">
          Redo
        </button>
      </div>
      <p className="mt-1.5 h-4 text-[10px] text-[var(--color-text-lo)]">
        {undoLabel === null ? 'Nothing to undo' : `Undo: ${undoLabel}`}
      </p>

      <XorChallenge />
      <Persistence />
      <ShareLink />
      <CodeExport />
    </div>
  );
}

function XorChallenge(): JSX.Element | null {
  const network = useAppStore((s) => s.network);
  const editRevision = useAppStore((s) => s.editRevision);
  const revision = useAppStore((s) => s.revision);
  const epoch = useAppStore((s) => s.epoch);
  const datasetName = useAppStore((s) => s.datasetOptions.name);

  // Recomputed whenever anything that could change the answer changes.
  const report = useMemo(
    () => checkXor(network),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [network, editRevision, revision, epoch],
  );

  const applySolution = (): void => {
    const state = useAppStore.getState();
    const hidden = state.network.layers[0];
    const output = state.network.layers[state.network.layers.length - 1];
    if (hidden === undefined || output === undefined) return;
    if (hidden.inputs !== 2 || hidden.units !== 2 || output.units !== 1) return;
    hidden.setWeights(
      fromRows(XOR_SOLUTION.hidden.weights.map((r) => [...r])),
      fromRows([[...XOR_SOLUTION.hidden.biases]]),
    );
    output.setWeights(
      fromRows(XOR_SOLUTION.output.weights.map((r) => [...r])),
      fromRows([[...XOR_SOLUTION.output.biases]]),
    );
    useAppStore.setState((s) => ({ editRevision: s.editRevision + 1 }));
  };

  if (datasetName !== 'xor') return null;

  if (report.problem !== null) {
    return (
      <div className="mt-4">
        <p className="label mb-1">XOR challenge</p>
        <p className="text-[11px] leading-relaxed text-[var(--color-text-lo)]">{report.problem}</p>
      </div>
    );
  }

  const canShowSolution =
    network.layers.length === 2 &&
    network.layers[0]?.units === 2 &&
    network.layers[0]?.inputs === 2;

  /*
   * The unsolved message has to say the RIGHT thing.
   *
   * "A single layer can only get three of four" is a true and important claim
   * about a network with no hidden layer, and simply false about one whose
   * hidden layer is merely too small. Saying it in both cases would teach the
   * wrong lesson to whoever is looking at the smaller case.
   */
  const hiddenUnits = network.layers.length > 1 ? (network.layers[0]?.units ?? 0) : 0;
  const hint =
    network.layers.length === 1
      ? 'No hidden layer, so this is one straight line. A line can separate at most three of the four corners; the fourth is always on the wrong side.'
      : hiddenUnits < 2
        ? `One hidden unit is still one line. XOR needs at least two, so their outputs can be combined.`
        : `${report.solvedCount} of 4. Two hidden units are enough: try making one behave like OR and the other like NAND.`;

  return (
    <div className="mt-4">
      <div className="mb-1 flex items-baseline justify-between">
        <p className="label">XOR challenge</p>
        <span className="num text-[11px] text-[var(--color-text-mid)]">
          {report.solvedCount}/4
        </span>
      </div>

      <div className="flex flex-col gap-0.5">
        {report.cases.map((c) => (
          <div
            key={c.inputs.join(',')}
            className="flex items-baseline justify-between gap-2 py-[1px]"
          >
            <span className="num text-[11px] text-[var(--color-text-lo)]">
              {c.inputs[0]}, {c.inputs[1]} → {c.target}
            </span>
            <span className="num text-[11px] text-[var(--color-text-mid)]">
              {c.output.toFixed(3)}
            </span>
            <span
              className="num w-4 text-[11px]"
              style={{ color: c.correct ? 'var(--color-text-hi)' : 'var(--color-status-bad)' }}
            >
              {c.correct ? '✓' : '✗'}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
        {report.solved
          ? report.worstMargin > CONVINCING_MARGIN
            ? 'Solved, and firmly: every corner sits well clear of the 0.5 threshold.'
            : `Solved, but only just. The closest corner is ${report.worstMargin.toFixed(3)} from the threshold.`
          : hint}
      </p>

      {canShowSolution && (
        <button className="control mt-2 w-full" onClick={applySolution}>
          Show me a solution
        </button>
      )}
    </div>
  );
}

/*
 * Code export (§8).
 *
 * "The codegen matters pedagogically: the learner sees that the beautiful
 * animated thing on screen is thirty lines of matrix multiplication."
 *
 * A PNG of the canvas is offered alongside, because a snapshot of the picture
 * is what people actually want to keep.
 */
function CodeExport(): JSX.Element {
  const network = useAppStore((s) => s.network);
  const [copied, setCopied] = useState<string | null>(null);

  const download = (name: string, text: string, type: string): void => {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const copy = (label: string, text: string): void => {
    void navigator.clipboard?.writeText(text).then(
      () => setCopied(label),
      // Clipboard access can be denied; falling back to a download is better
      // than a button that silently does nothing.
      () => download(`awrynn-${label}.txt`, text, 'text/plain'),
    );
  };

  const shape = [network.inputSize, ...network.layers.map((l) => l.units)].join('-');

  const savePng = (): void => {
    const canvas = document.querySelector<HTMLCanvasElement>('canvas[aria-label*="Neural network"]');
    if (canvas === null) return;
    canvas.toBlob((blob) => {
      if (blob === null) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `awrynn-${shape}.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    });
  };

  return (
    <div className="mt-4">
      <p className="label mb-1.5">Export code</p>
      <div className="flex flex-wrap gap-1.5">
        <button
          className="control flex-1"
          onClick={() => download(`awrynn-${shape}.py`, generateNumpy(network), 'text/x-python')}
        >
          NumPy
        </button>
        <button
          className="control flex-1"
          onClick={() =>
            download(`awrynn-${shape}.js`, generateJavaScript(network), 'text/javascript')
          }
        >
          JavaScript
        </button>
        <button className="control flex-1" onClick={savePng}>
          PNG
        </button>
      </div>
      <div className="mt-1.5 flex gap-1.5">
        <button className="control flex-1" onClick={() => copy('numpy', generateNumpy(network))}>
          Copy NumPy
        </button>
        <button className="control flex-1" onClick={() => copy('js', generateJavaScript(network))}>
          Copy JS
        </button>
      </div>
      <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
        {copied === null
          ? 'Weights inlined at full precision, so the exported code reproduces this network exactly.'
          : `Copied the ${copied} export to the clipboard.`}
      </p>
    </div>
  );
}

/**
 * The share link.
 *
 * The URL is shown as well as copied, for two reasons. The clipboard API needs
 * a secure context and a user gesture and can still be refused, so a button
 * whose only feedback is "Copied" sometimes lies; and a link this long is worth
 * seeing before you send it, which is why its length is reported next to it.
 */
function ShareLink(): JSX.Element {
  const shareUrlFor = useAppStore((s) => s.shareUrlFor);
  const [url, setUrl] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const build = (): void => {
    const next = shareUrlFor(window.location.href);
    setUrl(next);
    // Whether the weights had to travel is the interesting part: it is the
    // difference between "here is my setup" and "here is my trained network".
    const carriesWeights = next.includes('&p=');
    const shape = carriesWeights
      ? 'Includes the exact weights.'
      : 'The seed rebuilds these weights, so they are not in the link.';
    const length =
      next.length > LONG_LINK_THRESHOLD
        ? ` ${next.length} characters, which some chat apps will truncate.`
        : ` ${next.length} characters.`;

    const clipboard = navigator.clipboard;
    if (clipboard === undefined) {
      setNote(`Copy it from the box below. ${shape}${length}`);
      return;
    }
    clipboard.writeText(next).then(
      () => setNote(`Copied. ${shape}${length}`),
      () => setNote(`Could not reach the clipboard, so copy it from the box below. ${shape}${length}`),
    );
  };

  return (
    <div className="mt-4">
      <p className="label mb-1.5">Share</p>
      <button className="control w-full" onClick={build}>
        Copy link
      </button>
      {url !== null && (
        <input
          className="control num mt-1.5 w-full"
          value={url}
          readOnly
          onFocus={(e) => e.currentTarget.select()}
          aria-label="Share link"
        />
      )}
      <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
        {note ??
          'A link carrying the architecture, dataset, hyperparameters and weights. It travels in the part of the URL that never reaches a server.'}
      </p>
    </div>
  );
}

function Persistence(): JSX.Element {
  const exportJson = useAppStore((s) => s.exportJson);
  const importJson = useAppStore((s) => s.importJson);
  const saveLocal = useAppStore((s) => s.saveLocal);
  const loadLocal = useAppStore((s) => s.loadLocal);
  const listLocal = useAppStore((s) => s.listLocal);
  const deleteLocal = useAppStore((s) => s.deleteLocal);

  const [name, setName] = useState('my network');
  const [saved, setSaved] = useState<string[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const refresh = (): void => setSaved(listLocal());
  useEffect(refresh, [listLocal]);

  const download = (): void => {
    const blob = new Blob([exportJson()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${name.replace(/[^\w-]+/g, '-') || 'network'}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const upload = (file: File): void => {
    void file.text().then((text) => {
      const error = importJson(text);
      setMessage(error === null ? 'Loaded.' : error);
    });
  };

  return (
    <div className="mt-4">
      <p className="label mb-1.5">Save and load</p>
      <input
        className="control num mb-1.5 w-full"
        value={name}
        onChange={(e) => setName(e.target.value)}
        aria-label="Network name"
      />
      <div className="flex flex-wrap gap-1.5">
        <button
          className="control flex-1"
          onClick={() => {
            saveLocal(name);
            refresh();
            setMessage(`Saved as “${name}”.`);
          }}
        >
          Save
        </button>
        <button className="control flex-1" onClick={download}>
          Export
        </button>
        <button className="control flex-1" onClick={() => fileRef.current?.click()}>
          Import
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file !== undefined) upload(file);
            e.target.value = '';
          }}
        />
      </div>

      {saved.length > 0 && (
        <div className="mt-2 flex flex-col gap-0.5">
          {saved.map((entry) => (
            <div key={entry} className="flex items-center gap-1.5">
              <button
                className="control num flex-1 truncate text-left"
                onClick={() => {
                  const error = loadLocal(entry);
                  setMessage(error === null ? `Loaded “${entry}”.` : error);
                }}
              >
                {entry}
              </button>
              <button
                className="control px-1.5 py-0"
                onClick={() => {
                  deleteLocal(entry);
                  refresh();
                }}
                aria-label={`Delete ${entry}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {message !== null && (
        <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-text-lo)]">{message}</p>
      )}
    </div>
  );
}
