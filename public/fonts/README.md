# Bundled fonts

Self-hosted, not loaded from a CDN: spec §2 forbids network requests at runtime,
so every font ships with the app.

| File | Family | Licence |
| --- | --- | --- |
| `space-grotesk-var.woff2` | Space Grotesk (variable, 300–700) | SIL OFL 1.1 |
| `ibm-plex-sans-var.woff2` | IBM Plex Sans (variable) | SIL OFL 1.1 |
| `ibm-plex-mono-400.woff2` | IBM Plex Mono Regular | SIL OFL 1.1 |
| `ibm-plex-mono-500.woff2` | IBM Plex Mono Medium | SIL OFL 1.1 |
| `ibm-plex-mono-600.woff2` | IBM Plex Mono SemiBold | SIL OFL 1.1 |

All are Latin subsets (U+0000–00FF plus common punctuation), ~100 KB total.
See `OFL.txt` for the licence text. Rationale for the three faces is in
[`docs/DESIGN.md`](../../docs/DESIGN.md).
