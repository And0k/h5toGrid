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
  | sibling project | `../../../meta_finder/docs/reference/io_formats.md` |

  Source lives under `src/tcm/` — `../../tcm/x.py` does not exist (a historical
  batch of such links 400'd in the viewer).  Sibling-project links (e.g.
  `meta_finder/`) traverse up to `REPO_ROOT` — allowed because
  `DocumentationBrowser` includes both `resource_root()` and `REPO_ROOT` in
  its default `allowed_roots`.
- **Heading anchors**: GitHub slugs — lowercase, punctuation dropped, each space
  → `-`; or explicit `{#my-id}` suffix on the heading. Cross-file:
  `io_formats.md#file-name-parsing`.
- **Source line anchors**: `file.py#L42` scrolls the source view to line 42.
- **Display text**:
  - Content lists (navigation enumerating documents): text = the target's H1,
    verbatim; an em-dash topic list after the link only when the target covers
    many topics (`[GUI Internals](GUI/_index.md) — module map, data flow, …`),
    otherwise absorb the description into the target's H1 and leave the link
    bare (`[GUI Key Decisions with Rationale and Regression Notes](GUI/decisions.md)`).
  - File name as text is discouraged — acceptable only when the name is itself
    the identifier the reader meets in code (e.g. `config_reference.md`, parsed
    by `_help.py`).
  - Elsewhere: text clarifies what the reader finds in the target in the
    context of the link — free wording, no requirement to match any existing
    text exactly.
- **Bare filesystem paths auto-link in the Tk GUI** (not in the doc viewer):
  absolute paths with a 2–4 letter extension under the directory last entered
  in the path field render as file-name links that open with the
  OS-associated application. No authored markup needed — explicit
  `[text](url)` links behave as before.


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
| Paragraph between a `## ` heading and its table | status bar message (falls back to the `## ` subtitle when absent) |
| `### Detailed` — bare heading (no backticks) under a `## ` section | parent section's dwell tooltip; does not close the section |
| `### \`field.path\`` — modeless, single-context detail | browser/doc view; its `#### Detailed` body arms the dwell tooltip |
| `### \`field.path\` <mode>value</mode>` — one section per consumer context | consumer selects by mode (e.g. hypothetical context-dependent field); its `#### Detailed` arms dwell |
| `#### <mode>value</mode>` nested under `### \`field.path\`` — inherits parent field path | same as `### \`field.path\` <mode>value</mode>` but keeps general description; its `##### Detailed` arms dwell |

- The `<mode>` tag is **optional**: use it only when the field's meaning
  depends on the consumer context; otherwise write a modeless section. A
  modeless body is also the fallback for any requested mode that has no
  tagged section.
- Heading syntax: `` ### `dotted.field.path` `` optionally followed by
  `<mode>value</mode>` (`</>` shorthand accepted) and an explicit `{#id}`.
- `##` section headings accept plain titles (backticks optional): a heading
  registers a field section iff its name is a known config group or
  `metadata` / `path_field`; general prose headings (CLI keys, "See also")
  are free-form.
- A `` ### `path_field` `` heading alone opens its section — no `##` heading
  or table row needed (its `###` bodies attach to the auto-created entry).
- `#### <Tag>`` / `##### <Tag>` blocks nest inside the active section; `#### Detailed`
  (child of `###`) and `##### <Tag>` (child of `#### <mode>`) are the only bodies
  that arm the dwell tooltip.  Heading level = parent level + 1, so a `#### <mode>`
  detail must be `#####`.
- **Bare `### Detailed`** (no backticks) under a `## ` section is the parent
  section's tooltip: the parser starts a mode for the current section with tag
  `"Detailed"` and stores the content as the mode body's short.  Unlike a
  `` ### `path` `` heading it does **not** close the section, so subsequent
  `` ### `` path blocks (e.g. `` ### `metadata.path` ``) still parse.  Use this
  when the tooltip text should not itself be a navigable field row.
- **Post-heading paragraph**: the paragraph between a `## ` heading and its
  table becomes the section's status-bar short (the `## ` subtitle is the
  fallback when the paragraph is absent).  Keep it to one concise sentence.
  Headings are labels, not prose: an intermediate heading (e.g. `#### Detailed`,
  a table title) keeps the section open but its text never enters the short.
  A sibling non-mode heading at/below an open mode's level (e.g. a table title
  after bare `### Detailed`) closes that mode, so the table below parses as
  field rows instead of tooltip text.  Opening a mode finalizes the short
  capture: earlier paragraph stays the status short, later prose arms the
  dwell body — the two never share text, so the dwell firing stays visible.
- **Citation blockquote** (`>`): a line starting with `>` finalizes the current
  section (post-heading paragraph or mode body) and discards the citation line
  itself.  Subsequent headings still start new sections.  Use it to exclude
  supplementary asides from GUI status/tooltip text.
- **New**: `#### <mode>value</mode>` under a `### \`field.path\`` heading
  inherits the parent field path — equivalent to a separate
  `### \`field.path\` <mode>value</mode>` heading but nests the mode detail
  under the general field description.  This lets the general description
  (stored under the modeless key; its `#### Important` sub-block is the error hint)
  be shown on field-associated errors (e.g. ``FileNotFoundError``) while the
  mode-specific bodies feed consumer statuses.
- **When mode differences belong in STR vs markdown ``<mode>`` tags** — the
  split follows the interface-neutrality rule above.  When the field's
  **meaning itself** depends on the consumer context (one interpretation per
  context), each meaning lives in its own ``<mode>`` section in markdown —
  the doc captures the semantic difference.  But when the meaning is
  mode-neutral and only the **GUI action** differs (e.g. "browse directory"
  vs "browse file" — same field, same meaning, just a different dialog on
  click), the mode-specific verb is an interface concern that belongs in
  ``str.yaml``: write a single modeless ``### `` section for the field, then
  augment it at runtime with ``STR["{field}.status.{mode}"]`` (e.g.
  ``path_field.status.dirs``, ``input.coefs.path.status.dir``).  The doc stays
  clean; the GUI action hint is localized like any other chrome string.  The
  same pattern applies to ``time_ranges.hover.*`` on ``metadata.time_range``.
- Parser: a `###` section does not close the parent `##` field section; the
  next `###`/`##` (or a `####` sibling of a `#### <mode>` section) closes the
  accumulation; any `[a-z_]+` mode value is recognized, no code changes needed
  in `_help.py`.
- A bare `### Detailed` heading (no backticks) starts a mode for the current
  section with tag `"Detailed"` — its content becomes the section's tooltip.
  It does not close the section, so subsequent `###` path blocks still parse.
- The paragraph between a `##` heading and its table is captured as the
  section's short (falls back to the subtitle).  A `>` citation line
  finalizes the current accumulation and is itself discarded.

| Mode | Content regime | Example consumer |
|-------|---------|-------------------|
| `<mode>dirs</mode>` (STR `path_field.status.dirs`) | Input data directory + GUI browse-dir hint | GUI path field (default) — md general + STR suffix |
| `<mode>files</mode>` (STR `path_field.status.files`) | Individual file selection — data files or their configs + GUI browse-files hint | GUI path field with Shift — md general + STR suffix |
| `<mode>dir</mode>` / `<mode>file</mode>` (STR `input.coefs.path.status.*`) | Coefficient source: directory vs single file | `input.coefs.path` status hints (md general + STR suffix) |
| - | Input specification patterns — glob, regex, directory, YAML | General field information, error message, CLI help |

Parser internals: [§Field detail sections](GUI/help_system.md#field-detail-sections-in-config_referencemd).
