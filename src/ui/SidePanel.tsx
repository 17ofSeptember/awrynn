import { useState } from 'react';
import type { JSX } from 'react';
import { BoundaryView } from './BoundaryView';
import { Inspector } from './Inspector';
import { EditPanel } from './EditPanel';
import { DiagnosticsPanel } from './DiagnosticsPanel';
import { LiveMath } from './LiveMath';
import { LessonsPanel } from './LessonsPanel';
import { useAppStore } from '../state/store';

/*
 * The right-hand chassis, as tabs.
 *
 * §6.4 calls for a "Live math tab", and stacking every analysis surface into
 * one column had already produced a sidebar you had to scroll 900px through to
 * reach the gradient chart. An instrument does not make you scroll past the
 * oscilloscope to find the meter.
 *
 * Tab state is local rather than in the store: which panel is showing is a
 * property of this view, not of the network, and nothing else needs to know.
 */

type TabId = 'learn' | 'inspect' | 'analyse' | 'math';

const TABS: readonly { readonly id: TabId; readonly label: string }[] = [
  { id: 'learn', label: 'Learn' },
  { id: 'inspect', label: 'Inspect' },
  { id: 'analyse', label: 'Analyse' },
  { id: 'math', label: 'Math' },
];

export function SidePanel(): JSX.Element {
  return (
    <aside className="flex w-[320px] shrink-0 flex-col border-l border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)]">
      <AnalysisPanels />
    </aside>
  );
}

/**
 * The tabbed analysis surfaces, without a container.
 *
 * Extracted so the desktop sidebar and the mobile drawer render the SAME
 * components rather than two drifting copies of them.
 */
export function AnalysisPanels(): JSX.Element {
  // Opens on the lessons, because that is what the app is for (§12).
  const [tab, setTab] = useState<TabId>('learn');
  const network = useAppStore((s) => s.network);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className="flex shrink-0 border-b border-[var(--color-line-hair)]"
        role="tablist"
        aria-label="Analysis panels"
      >
        {TABS.map((entry) => {
          const active = entry.id === tab;
          return (
            <button
              key={entry.id}
              role="tab"
              aria-selected={active}
              onClick={() => setTab(entry.id)}
              className="flex-1 px-2 py-2 text-[11px] transition-colors sm:px-3"
              style={{
                color: active ? 'var(--color-text-hi)' : 'var(--color-text-lo)',
                // A 1px underline rather than a filled pill: hierarchy by
                // weight and spacing, not by boxes (§9).
                boxShadow: active ? 'inset 0 -1px 0 0 var(--color-weight-positive)' : 'none',
                fontFamily: 'var(--font-display)',
                fontWeight: active ? 600 : 400,
              }}
            >
              {entry.label}
            </button>
          );
        })}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === 'learn' && <LessonsPanel />}
        {tab === 'inspect' && (
          <>
            <BoundaryView />
            <hr className="rule" />
            <Inspector />
            <hr className="rule" />
            <EditPanel />
          </>
        )}
        {tab === 'analyse' && <DiagnosticsPanel />}
        {tab === 'math' && <LiveMath />}
      </div>

      <div className="shrink-0 border-t border-[var(--color-line-hair)] px-4 py-2">
        <div className="flex items-baseline justify-between">
          <span className="label">network</span>
          <span className="num text-[11px] text-[var(--color-text-mid)]">
            {network.parameterCount} params · {network.layers.length} layers · {network.lossName}
          </span>
        </div>
      </div>
    </div>
  );
}
