import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { formatLoss, formatPercent } from '../render/theme';

/*
 * The training history scrubber (§6.6).
 *
 * "A timeline scrubber replays training: drag it and the edges, boundary, and
 * thumbnails animate through their real history. Pin any snapshot as A and diff
 * against the live network — edges then color by Δw instead of w."
 *
 * Scrubbing is a VIEW, not an edit: the live parameters are stashed on the way
 * in and restored on release, so letting go never leaves the network stranded
 * in its own past.
 */

export function HistoryScrubber(): JSX.Element | null {
  const history = useAppStore((s) => s.history);
  const index = useAppStore((s) => s.historyIndex);
  const pinned = useAppStore((s) => s.pinnedIndex);
  const scrub = useAppStore((s) => s.scrubHistory);
  const pin = useAppStore((s) => s.pinSnapshot);

  if (history.length < 2) return null;

  const position = index ?? history.length - 1;
  const viewing = history[position];
  const live = index === null;

  return (
    <div className="flex shrink-0 items-center gap-2 border-t border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)] px-3 py-2 sm:gap-3 sm:px-4">
      <span className="label shrink-0">history</span>

      <input
        type="range"
        min={0}
        max={history.length - 1}
        step={1}
        value={position}
        onChange={(e) => scrub(Number(e.target.value))}
        // Releasing returns to the live network, so the scrubber behaves like a
        // preview rather than a destructive seek.
        onPointerUp={() => scrub(null)}
        onKeyUp={(e) => {
          if (e.key === 'Escape') scrub(null);
        }}
        className="min-w-0 flex-1 accent-[var(--color-weight-positive)]"
        aria-label="Training history"
      />

      <span className="num shrink-0 text-[11px] text-[var(--color-text-mid)]">
        {viewing === undefined ? '—' : `epoch ${viewing.epoch}`}
      </span>
      <span className="num hidden shrink-0 text-[11px] text-[var(--color-text-lo)] sm:inline">
        {viewing === undefined
          ? ''
          : viewing.validationAccuracy !== null
            ? formatPercent(viewing.validationAccuracy)
            : formatLoss(viewing.trainLoss)}
      </span>

      <button
        className="control shrink-0"
        onClick={() => scrub(live ? history.length - 1 : null)}
        title="Return to the live network"
        disabled={live}
      >
        Live
      </button>

      <button
        className="control shrink-0"
        onClick={() => pin(pinned === null ? position : null)}
        title="Pin this snapshot and colour edges by how much they have changed since"
        style={pinned !== null ? { borderColor: 'var(--color-weight-positive)' } : undefined}
      >
        {pinned === null ? 'Pin as A' : `Diff vs ${history[pinned]?.epoch ?? '?'}`}
      </button>
    </div>
  );
}
