import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  /*
   * Relative asset URLs, so the built site works from any path.
   *
   * The default is '/', which only works when the app is served from a domain
   * root. A GitHub Pages project site lives at /<repo>/, and every asset
   * reference, the bundle, the stylesheet, the fonts and both guide pages,
   * would 404 there with nothing but a blank page to show for it.
   */
  base: './',

  plugins: [react(), tailwindcss()],
  // Workers are ES modules (spec §2 allows Web Workers + OffscreenCanvas).
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
