# Documentation Authoring Contract

Requirements for Markdown documents so they render identically in the built-in
documentation viewer you can start through links in "?"-button → About window (implemented in `tcm_gui/browser/`) and on
GitHub. The viewer is fully offline: marked (GFM CommonMark) + MathJax 4 +
highlight.js are vendored (generated into `_build/browser-runtime` by
`browser/vendor.mjs`); nothing is fetched from the Internet.

## Math

| Form | Delimiters | Use |
|---|---|---|
| Inline | `\( ... \)` | math inside prose: `\(E = mc^2\)` |
| Display | `$$ ... $$` or `\[ ... \]` | standalone equation blocks (blank line before/after) |

- Single dollar `$...$` is **deliberately disabled** (dollar signs are common in
  plain text) — it renders literally.
- **Never** leave ASCII math in prose: `h = A^-1 (F u) + b`, `S^2`, `sum_i`
  render verbatim. Convert to TeX: `\(h = A^{-1} (F u) + b\)`.
- TeX inside code spans/fences stays literal (the viewer shields code first) —
  use a code span when the literal characters are the point.
- `\(...\)` and `\[...\]` are safe in prose: the viewer shields math spans from
  CommonMark's backslash-escape stripping before parsing.
- Prefer `\text{}` for words inside formulas; `\mathbb{R}`, `\lfloor\rfloor`,
  `\begin{cases}` are supported (full TeX via MathJax).

## Code

- Fenced blocks **always with a language tag** — the tag drives highlight.js:

  ````markdown
  ```python
  def f(x): ...
  ```
  ````

  Supported tags include: `python`, `bash`, `powershell`, `dos`, `yaml`, `json`,
  `toml`, `ini`, `text` (no highlighting — alignment-safe for ASCII diagrams,
  trees, tabular traces).
- **No indented (4-space) code blocks**: CommonMark renders them, but without
  highlighting, and they are ambiguous with list continuation. They are also the
  classic trap for pasting math — an equation indented 4 spaces silently becomes
  a monospace block instead of a display formula. Use fences or `$$`.
- Inline identifiers in backticks: `solve_optimal_weights`, `input.path`.

## Links

- **Relative to the document's own directory** — the viewer resolves against the
  file's location, not the server root. From `docs/reference/`:

  | Target | Link |
  |---|---|
  | sibling doc | `config_tuning.md` |
  | other section | `../user_guide/processing.md` |
  | package source | `../../src/tcm/format.py` |
  | build scripts | `../../scripts/tcm_proc.py` |

  Source lives under `src/tcm/` — `../../tcm/x.py` does not exist (a historical
  batch of such links 400'd in the viewer).
- **Heading anchors**: GitHub slugs — lowercase, punctuation dropped, each space
  → `-`; or explicit `{#my-id}` suffix on the heading. Cross-file:
  `io_formats.md#file-name-parsing`.
- **Source line anchors**: `file.py#L42` scrolls the source view to line 42.
- **Display text**:
  - Content lists (navigation enumerating documents): text = the target's H1,
    verbatim; an em-dash topic list after the link only when the target covers
    many topics (`[GUI Internals](GUI/_index.md) — module map, data flow, …`),
    otherwise absorb the description into the target's H1 and leave the link
    bare (`[GUI Key Decisions with Rationale and Regression Notes](GUI_decisions.md)`).
  - File name as text is discouraged — acceptable only when the name is itself
    the identifier the reader meets in code (e.g. `config_reference.md`, parsed
    by `_help.py`).
  - Elsewhere: text clarifies what the reader finds in the target in the
    context of the link — free wording, no requirement to match any existing
    text exactly.


## Images

Relative path to any configured image type (png, jpg, gif, webp, svg, bmp,
avif): `![calibration flow](images/flow.png)`. Kept next to the doc (or in a
sibling `images/` dir) so GitHub and the viewer resolve them identically.

## Browsable file classes

| Class | Extensions |
|---|---|
| markdown | `.md`, `.markdown` |
| source/text | `.py`, `.js`, `.ts`, `.yaml`, `.toml`, `.ini`, `.txt`, `.log`, `.sh`, `.ps1`, `.bat` … |
| images | see above |

Anything else is not served — link only to these classes.

## Self-check

Open the document through the viewer (About → Documentation tree, or any doc
link in it) and verify: every formula is typeset (no literal `\(` / `^` in
prose), every code block highlighted, every local link navigates without a
400/404, back button returns.


# config_reference.md for programmatic display in status and tooltips

General rules:

- Do not refer to a section's position in the document ("see details below")
  — a section body may be displayed programmatically in an arbitrary place.
- `###`/`####` bodies are interface-neutral: no GUI widget names, no
  CLI-argument framing — the consumer context is expressed only by `<mode>` tags.
- The Tk dwell tooltip renders plain text only — display math (`$$…$$`) never
  goes into `###`/`####` bodies. Put formulas on a methodology page and link,
  e.g. `[§Pressure computation](../methodology/pressure.md)`; the link opens
  the doc browser where MathJax typesets it.

## Structure

| Doc element | GUI consumer |
|---|---|
| Table cell — one concise sentence | status bar message |
| `### \`field.path\`` — modeless, single-context detail | browser/doc view; its `#### Detailed` body arms the dwell tooltip |
| `### \`field.path\` <mode>value</mode>` — one section per consumer context | consumer selects by mode (e.g. `coefs_path` dir/file status hints) |

- The `<mode>` tag is **optional**: use it only when the field's meaning
  depends on the consumer context; otherwise write a modeless section.
- Heading syntax: `` ### `dotted.field.path` `` optionally followed by
  `<mode>value</mode>` (`</>` shorthand accepted) and an explicit `{#id}`.
- `#### <Tag>` blocks nest inside the active `###` section; `#### Detailed`
  is the only body that arms the dwell tooltip.
- Parser: a `###` section does not close the parent `##` field section; the
  next `###`/`##` closes the accumulation; any `[a-z_]+` mode value is
  recognized, no code changes needed in `_help.py`.

| Mode | Content regime | Example consumer |
|-------|---------|-------------------|
| `<mode>probe</mode>` | Per-probe processing meaning — what the field does, how it affects the result | GUI coef hover, popup |
| `<mode>search</mode>` | Input specification patterns — glob, regex, directory, YAML | GUI path field, CLI help |

Parser internals: [§Field detail sections](GUI_help_system.md#field-detail-sections-in-config_referencemd).
