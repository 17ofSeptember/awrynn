import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    /*
     * `node` by default, deliberately. The engine must be runnable in Node with
     * zero shims (spec §0.5), and engine tests running in a browser-like
     * environment would hide a violation of that. UI tests opt into jsdom
     * per-file with `// @vitest-environment jsdom`.
     */
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    // Explicit imports from 'vitest' rather than ambient globals.
    globals: false,
  },
});
