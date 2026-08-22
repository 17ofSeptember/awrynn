import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'coverage'] },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  {
    files: ['**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    rules: {
      // Spec §0.3: no `any`, no `@ts-ignore`.
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/ban-ts-comment': [
        'error',
        { 'ts-ignore': true, 'ts-expect-error': 'allow-with-description', 'ts-nocheck': true },
      ],
      // Spec §0.3: explicit return types on exported functions.
      '@typescript-eslint/explicit-module-boundary-types': 'error',
      '@typescript-eslint/consistent-type-imports': 'error',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      // Stale closures in effects are a real class of bug here: worker replies
      // arrive after the render that created their handler.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },

  {
    /*
     * Spec §4.7: Math.random() must not appear anywhere in src/engine/.
     * Belt and braces — src/engine/__tests__/purity.test.ts enforces the same
     * rule at test time, because §10 requires it to be enforced by a test.
     */
    files: ['src/engine/**/*.ts'],
    rules: {
      'no-restricted-properties': [
        'error',
        {
          object: 'Math',
          property: 'random',
          message:
            'Math.random() is banned in src/engine/. All randomness must draw from a named, seeded stream (spec §4.7).',
        },
      ],
      'no-restricted-globals': [
        'error',
        { name: 'window', message: 'The engine must not touch the DOM (spec §0.5).' },
        { name: 'document', message: 'The engine must not touch the DOM (spec §0.5).' },
      ],
    },
  },

  {
    files: ['**/*.test.ts', '**/*.test.tsx', 'vite.config.ts', 'vitest.config.ts'],
    languageOptions: { globals: globals.node },
    rules: { 'no-console': 'off' },
  },

  {
    // Dev tooling: runs in Node, prints to stdout by design.
    files: ['scripts/**/*.{ts,mjs,js}'],
    languageOptions: { globals: { ...globals.node, WebSocket: 'readonly' } },
    rules: { 'no-console': 'off' },
  },
);
