import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/*
 * Enforcement of spec §0.5 ("the engine is sacred") and §4.7 ("Math.random()
 * must not appear anywhere in src/engine/"). §10 requires the Math.random rule
 * to be enforced by a test specifically, so it lives here rather than only in
 * the ESLint config.
 *
 * These scans are vacuous while src/engine/ is empty (Phase 0). They start
 * biting the moment Phase 1 lands, which is the point of writing them now.
 */

const ENGINE_ROOT = fileURLToPath(new URL('..', import.meta.url));

function engineSourceFiles(dir: string = ENGINE_ROOT): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === '__tests__') continue;
      found.push(...engineSourceFiles(full));
    } else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.d.ts')) {
      found.push(full);
    }
  }
  return found;
}

interface EngineFile {
  readonly path: string;
  readonly source: string;
}

/*
 * Comments are stripped before scanning. Engine files legitimately *discuss*
 * the banned APIs — `rng.ts` has to explain that Math.random() is forbidden —
 * and flagging that prose would train us to ignore this test, which is worse
 * than the narrow evasion it leaves open (a banned identifier hidden inside a
 * string literal, which nothing in this codebase has reason to do).
 */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

const FILES: readonly EngineFile[] = engineSourceFiles().map((path) => ({
  path: relative(ENGINE_ROOT, path),
  source: stripComments(readFileSync(path, 'utf8')),
}));

/* Imports the engine is forbidden from making (spec §0.5). */
const FORBIDDEN_IMPORTS: readonly RegExp[] = [
  /\breact\b/,
  /\breact-dom\b/,
  /\bzustand\b/,
  /\/state\//,
  /\/ui\//,
  /\/render\//,
  /\/worker\//,
];

/* Globals that would prove the engine is not runnable in bare Node. */
const FORBIDDEN_GLOBALS: readonly string[] = [
  'window',
  'document',
  'navigator',
  'localStorage',
  'requestAnimationFrame',
  'OffscreenCanvas',
  'ImageData',
];

function importSpecifiers(source: string): string[] {
  const specifiers: string[] = [];
  const pattern = /(?:from|import)\s*['"]([^'"]+)['"]/g;
  let match = pattern.exec(source);
  while (match !== null) {
    const spec = match[1];
    if (spec !== undefined) specifiers.push(spec);
    match = pattern.exec(source);
  }
  return specifiers;
}

describe('engine purity', () => {
  it('contains no Math.random (spec §4.7 — all randomness must be seeded)', () => {
    const offenders = FILES.filter((f) => /Math\s*\.\s*random/.test(f.source)).map((f) => f.path);
    expect(offenders).toEqual([]);
  });

  it('imports nothing from React, the DOM, the store, or any UI code (spec §0.5)', () => {
    const offenders: string[] = [];
    for (const file of FILES) {
      for (const spec of importSpecifiers(file.source)) {
        if (FORBIDDEN_IMPORTS.some((pattern) => pattern.test(spec))) {
          offenders.push(`${file.path} → ${spec}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });

  it('references no browser globals (spec §0.5 — must run in Node with zero shims)', () => {
    const offenders: string[] = [];
    for (const file of FILES) {
      for (const name of FORBIDDEN_GLOBALS) {
        if (new RegExp(`\\b${name}\\b`).test(file.source)) {
          offenders.push(`${file.path} → ${name}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});
