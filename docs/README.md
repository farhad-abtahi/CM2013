# docs/

Published to GitHub Pages at https://farhad-abtahi.github.io/CM2013/.

**Committed here (source):**
- `index.html` — the notebook gallery page: searchable/filterable cards for every
  demo, lab, and capstone-track notebook, each with Colab/View/JupyterLite/
  download links.
- `notebooks.json` — the manifest `index.html` reads to render those cards (title,
  question, concepts, and JupyterLite-viability per notebook).

**Built by CI, not committed** (see `.github/workflows/deploy.yml`, gitignored):
- `nb/` — every notebook exported to static HTML (demo/track notebooks with fresh
  executed outputs; labs as-is, unsolved).
- `lite/` — a JupyterLite site so every notebook also runs in-browser, no install.

To add or rename a notebook, edit `notebooks.json` directly — add an entry with
`num`, `kind` (`chapter`/`lab`/`clinic`/`track`), `slug`, `file`, `title`,
`question`, `concepts`, `lite`, `ml`, `pilot`.
