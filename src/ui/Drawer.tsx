import { useState } from 'react';
import type { JSX } from 'react';
import { BuildPanels } from './App';
import { AnalysisPanels } from './SidePanel';

/*
 * The mobile fallback (§9 quality floor).
 *
 * "responsive down to tablet with a sensible mobile fallback (canvas stays,
 * panels collapse into a drawer)."
 *
 * A bottom sheet rather than a side drawer: the canvas is wider than it is
 * tall on a phone held upright, so taking height costs less of the picture than
 * taking width would. Collapsed it is a single bar, and the network stays fully
 * visible above it.
 *
 * The sheet OVERLAYS the canvas rather than pushing it, so opening the drawer
 * does not resize the canvas and re-fit the viewport under the reader.
 */

/**
 * Height of the collapsed bar.
 *
 * Exported because the app column reserves exactly this much padding: the
 * drawer is fixed, so anything it does not account for gets covered.
 */
export const COLLAPSED_DRAWER_HEIGHT = 46;

type Section = 'build' | 'analyse';

const SECTIONS: readonly { readonly id: Section; readonly label: string }[] = [
  { id: 'build', label: 'Build & train' },
  { id: 'analyse', label: 'Analyse' },
];

/**
 * @param includeBuild false once the build panel has its own sidebar, so the
 * drawer does not offer a second copy of controls already on screen.
 */
export function Drawer({ includeBuild = true }: { includeBuild?: boolean }): JSX.Element {
  const [open, setOpen] = useState(false);
  const [section, setSection] = useState<Section>(includeBuild ? 'build' : 'analyse');
  const sections = includeBuild ? SECTIONS : SECTIONS.filter((s) => s.id === 'analyse');

  return (
    <>
      {/* A scrim so a tap outside closes the sheet, and the canvas behind it
          does not receive stray taps meant for the drawer. */}
      {open && (
        <button
          className="fixed inset-0 z-20 bg-black/50"
          onClick={() => setOpen(false)}
          aria-label="Close panel"
          tabIndex={-1}
        />
      )}

      <div
        className="fixed inset-x-0 bottom-0 z-30 flex flex-col border-t border-[var(--color-line-edge)] bg-[var(--color-bg-chassis)]"
        style={{
          // Capped so the canvas is never fully hidden; the point of the app is
          // the picture, and a sheet that covers it defeats the exercise.
          height: open ? 'min(58vh, 520px)' : 'auto',
          transition: 'height var(--motion-select) var(--motion-easing)',
          // Clears the home indicator on a phone without a notch assumption.
          paddingBottom: 'env(safe-area-inset-bottom)',
        }}
      >
        <div className="flex shrink-0 items-stretch">
          {sections.map((entry) => {
            const active = open && entry.id === section;
            return (
              <button
                key={entry.id}
                onClick={() => {
                  if (open && section === entry.id) setOpen(false);
                  else {
                    setSection(entry.id);
                    setOpen(true);
                  }
                }}
                aria-expanded={active}
                className="flex-1 px-3 py-3 text-[12px]"
                style={{
                  color: active ? 'var(--color-text-hi)' : 'var(--color-text-mid)',
                  boxShadow: active ? 'inset 0 2px 0 0 var(--color-weight-positive)' : 'none',
                  fontFamily: 'var(--font-display)',
                  fontWeight: active ? 600 : 400,
                  // 44px minimum touch target.
                  minHeight: 44,
                }}
              >
                {entry.label}
              </button>
            );
          })}
          <button
            onClick={() => setOpen(!open)}
            aria-label={open ? 'Collapse panel' : 'Expand panel'}
            className="px-4"
            style={{ color: 'var(--color-text-lo)', minHeight: 44, minWidth: 44 }}
          >
            {open ? '▾' : '▴'}
          </button>
        </div>

        {open && (
          <div className="min-h-0 flex-1 overflow-y-auto border-t border-[var(--color-line-hair)]">
            {section === 'build' ? <BuildPanels /> : <AnalysisPanels />}
          </div>
        )}
      </div>
    </>
  );
}
