import { useEffect } from 'react';
import type { JSX } from 'react';
import { NetworkCanvas } from './NetworkCanvas';
import { ArchitecturePanel } from './ArchitecturePanel';
import { SidePanel } from './SidePanel';
import { Transport, useTransportTick } from './Transport';
import { HistoryScrubber } from './HistoryScrubber';
import { COLLAPSED_DRAWER_HEIGHT, Drawer } from './Drawer';
import { ANALYSIS_SIDEBAR_QUERY, BUILD_SIDEBAR_QUERY, useMediaQuery } from './useMediaQuery';
import { TrainingPanel } from './TrainingPanel';
import { useAppStore } from '../state/store';
import { useSharedLink } from './useSharedLink';
import type { SharedLinkNotice } from './useSharedLink';
import { installTheme } from '../render/theme';
import { DATASET_NAMES } from '../engine/datasets/index';
import type { DatasetName } from '../engine/datasets/index';

/*
 * The chassis.
 *
 * §9: the canvas is the instrument face, the panels are its housing. Hierarchy
 * comes from weight and spacing — the only separators here are 1px rules, and
 * nothing is elevated on a shadow.
 */

/**
 * The build controls, without a container.
 *
 * Shared verbatim between the desktop sidebar and the mobile drawer, so the two
 * cannot drift apart.
 */
export function BuildPanels(): JSX.Element {
  return (
    <>
      <ArchitecturePanel />
      <hr className="rule" />
      <TrainingPanel />
      <hr className="rule" />
      <DatasetPanel />
      <hr className="rule" />
      <DisplayPanel />
    </>
  );
}

export function App(): JSX.Element {
  useTransportTick();
  const sharedLink = useSharedLink();
  const buildSidebar = useMediaQuery(BUILD_SIDEBAR_QUERY);
  const analysisSidebar = useMediaQuery(ANALYSIS_SIDEBAR_QUERY);
  // The drawer carries whatever the sidebars are not showing.
  const compact = !analysisSidebar;
  useEffect(() => {
    // The token system is defined in theme.ts; this is what makes the CSS and
    // the canvas provably the same palette.
    installTheme(document.documentElement);
  }, []);

  return (
    <div
      className="flex h-full flex-col bg-[var(--color-bg-void)]"
      style={
        compact
          ? {
              // The drawer is fixed to the bottom, so the column has to reserve
              // its collapsed height or the transport controls sit underneath it.
              paddingBottom: `calc(${COLLAPSED_DRAWER_HEIGHT}px + env(safe-area-inset-bottom))`,
            }
          : undefined
      }
    >
      <Header />
      {sharedLink.notice !== null && (
        <LinkNotice notice={sharedLink.notice} onDismiss={sharedLink.dismiss} />
      )}
      <div className="flex min-h-0 flex-1">
        {buildSidebar && (
          <aside className="flex w-[300px] shrink-0 flex-col overflow-y-auto border-r border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)]">
            <BuildPanels />
          </aside>
        )}

        <main className="flex min-w-0 flex-1 flex-col bg-[var(--color-bg-canvas)]">
          <div className="min-h-0 flex-1">
            <NetworkCanvas />
          </div>
          <HistoryScrubber />
          <Transport />
        </main>

        {analysisSidebar && <SidePanel />}
      </div>
      {compact && <Drawer includeBuild={!buildSidebar} />}
    </div>
  );
}

/**
 * What happened to the link in the address bar.
 *
 * Sits between the header and the canvas rather than floating over it. Someone
 * who followed a broken link needs to know the network in front of them is not
 * the one they were sent, and a toast that fades takes that away again.
 */
function LinkNotice({
  notice,
  onDismiss,
}: {
  notice: SharedLinkNotice;
  onDismiss: () => void;
}): JSX.Element {
  const failed = notice.kind === 'error';
  return (
    <div
      role={failed ? 'alert' : 'status'}
      className="flex shrink-0 items-start gap-3 border-b border-l-2 border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)] px-3 py-2 sm:px-4"
      style={{
        // The same left rule the config-error and training-error panels use, in
        // the error tone for a failure and the negative-weight blue for news.
        // Colour is the second signal here; the sentence is the first.
        borderLeftColor: failed ? 'var(--color-status-bad)' : 'var(--color-weight-negative)',
      }}
    >
      <p className="min-w-0 flex-1 text-[12px] leading-snug text-[var(--color-text-mid)]">
        {failed && (
          <span className="text-[var(--color-text-hi)]">This link could not be opened. </span>
        )}
        {notice.message}
        {failed && ' You are looking at the default network instead.'}
      </p>
      <button className="control shrink-0" onClick={onDismiss}>
        Dismiss
      </button>
    </div>
  );
}

function Header(): JSX.Element {
  const architecture = useAppStore((s) => s.architecture);
  const reseed = useAppStore((s) => s.reseed);

  return (
    <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-[var(--color-line-hair)] bg-[var(--color-bg-chassis)] px-3 py-2 sm:px-4 sm:py-2.5">
      <h1
        className="text-[15px] font-semibold tracking-tight text-[var(--color-text-hi)]"
        style={{ fontFamily: 'var(--font-display)' }}
      >
        AwryNN
      </h1>
      <span className="label hidden md:inline">neural network laboratory</span>

      <div className="ml-auto flex items-center gap-2 sm:gap-3">
        <GuideLinks />
        <span className="label hidden sm:inline">seed</span>
        <input
          type="number"
          className="control num w-14 sm:w-20"
          value={architecture.seed}
          onChange={(e) => reseed(Number(e.target.value) || 0)}
          aria-label="Random seed"
        />
        <button className="control" onClick={() => reseed(architecture.seed + 1)}>
          Reseed
        </button>
      </div>
    </header>
  );
}

/**
 * The two guides.
 *
 * Built from BASE_URL rather than written as "/guide/...", so the links still
 * resolve when the app is served from a subpath such as a GitHub Pages project
 * site. Vite rewrites asset references at build time but not string literals
 * like these.
 *
 * Plain links to standalone pages rather than an in-app modal: they are
 * documents, they should open instantly in their own tab, and they must keep
 * working if the app itself is broken — which is exactly when someone reaches
 * for the manual.
 */
function GuideLinks(): JSX.Element {
  return (
    <div className="flex items-center gap-1.5">
      <a
        className="control no-underline"
        href={`${import.meta.env.BASE_URL}guide/start-here.html`}
        target="_blank"
        rel="noopener"
        title="A complete introduction, assuming no background"
      >
        Start here
      </a>
      <a
        className="control no-underline"
        href={`${import.meta.env.BASE_URL}guide/handbook.html`}
        target="_blank"
        rel="noopener"
        title="Conventions, controls, and how to verify the engine"
      >
        Handbook
      </a>
    </div>
  );
}

function DatasetPanel(): JSX.Element {
  const datasetOptions = useAppStore((s) => s.datasetOptions);
  const dataset = useAppStore((s) => s.dataset);
  const setDatasetOptions = useAppStore((s) => s.setDatasetOptions);
  const reconciliation = useAppStore((s) => s.lastReconciliation);

  // Only spiral and glyphs have a meaningful class count; the rest are fixed
  // by their geometry, and offering the control would imply otherwise.
  const supportsClasses = dataset.name === 'spiral' || dataset.name === 'glyphs';

  return (
    <div className="px-4 py-3">
      <p className="panel-title mb-3">Dataset</p>
      <select
        className="control w-full"
        value={datasetOptions.name}
        onChange={(e) => setDatasetOptions({ name: e.target.value as DatasetName })}
        aria-label="Dataset"
      >
        {DATASET_NAMES.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>

      {/* §5: every dataset exposes sample count, noise level, class balance,
          train/val fraction and seed. Changing any of them regenerates
          deterministically. */}
      <div className="mt-3 flex flex-col gap-1.5">
        <NumberField
          label="samples"
          value={datasetOptions.samples ?? dataset.x.rows}
          min={4}
          step={20}
          onChange={(samples) => setDatasetOptions({ samples })}
        />
        <NumberField
          label="noise"
          value={datasetOptions.noise ?? 0.1}
          min={0}
          step={0.02}
          onChange={(noise) => setDatasetOptions({ noise })}
        />
        {supportsClasses && (
          <NumberField
            label="classes"
            value={datasetOptions.classes ?? dataset.classCount}
            min={2}
            max={dataset.name === 'glyphs' ? 14 : 5}
            step={1}
            onChange={(classes) => setDatasetOptions({ classes })}
          />
        )}
        <NumberField
          label="val split"
          value={datasetOptions.validationFraction ?? 0.2}
          min={0}
          max={0.5}
          step={0.05}
          onChange={(validationFraction) => setDatasetOptions({ validationFraction })}
        />
        <NumberField
          label="data seed"
          value={datasetOptions.seed ?? 0}
          min={0}
          step={1}
          onChange={(seed) => setDatasetOptions({ seed })}
        />
      </div>

      <div className="mt-3 flex flex-col gap-1">
        <Stat label="generated" value={String(dataset.x.rows)} />
        <Stat label="features" value={String(dataset.featureCount)} />
        <Stat label="classes" value={dataset.classCount === 0 ? 'regression' : String(dataset.classCount)} />
        <Stat label="loss" value={dataset.suggestedLoss} />
      </div>

      {reconciliation.length > 0 && (
        <div className="mt-3 border-l-2 border-[var(--color-line-edge)] pl-3">
          {/* §6.5: never silently reset the network — say what was preserved. */}
          {reconciliation.map((line) => (
            <p key={line} className="text-[11px] leading-relaxed text-[var(--color-text-lo)]">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function DisplayPanel(): JSX.Element {
  const colorblindSafe = useAppStore((s) => s.colorblindSafe);
  const showBiases = useAppStore((s) => s.showBiases);
  const showThumbnails = useAppStore((s) => s.showThumbnails);
  const toggleThumbnails = useAppStore((s) => s.toggleThumbnails);
  const toggleColorblindSafe = useAppStore((s) => s.toggleColorblindSafe);
  const toggleBiases = useAppStore((s) => s.toggleBiases);

  return (
    <div className="px-4 py-3">
      <p className="panel-title mb-3">Display</p>
      <Toggle label="Bias satellites" checked={showBiases} onChange={toggleBiases} />
      <Toggle
        label="Neuron thumbnails"
        checked={showThumbnails}
        onChange={toggleThumbnails}
      />
      <Toggle
        label="Colourblind-safe"
        checked={colorblindSafe}
        onChange={toggleColorblindSafe}
      />
      <p className="mt-3 text-[11px] leading-relaxed text-[var(--color-text-lo)]">
        Scroll to zoom, drag to pan, double-click to fit.
      </p>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}): JSX.Element {
  return (
    <label className="flex items-center gap-2">
      <span className="label w-20 shrink-0">{label}</span>
      <input
        type="number"
        className="control num w-full"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = Number(e.target.value);
          if (!Number.isFinite(next)) return;
          const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, next));
          onChange(clamped);
        }}
        aria-label={label}
      />
    </label>
  );
}

function Stat({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[3px]">
      <span className="label">{label}</span>
      <span className="num text-[12px] text-[var(--color-text-mid)]">{value}</span>
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
}): JSX.Element {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 py-1">
      <span className="text-[12px] text-[var(--color-text-mid)]">{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-3.5 w-3.5 accent-[var(--color-weight-positive)]"
      />
    </label>
  );
}
