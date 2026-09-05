# Documentation Authoring Contract

meta_finder follows the same documentation authoring rules as the `tcm` module. The complete contract is maintained in the tcm project:

- [tcm Documentation Authoring Contract](../../tcm/docs/project_developer_guide/doc_authoring.md)

Key points:

- **Math**: Use `\(...\)` for inline math, `$$...$$` for display math. Single dollar `$...$` is disabled.
- **Code**: Fenced blocks **always** with a language tag. No indented (4-space) code blocks.
- **Links**: Relative to the document's own directory. From `docs/user_guide/`:
  - sibling doc: `configuration.md`
  - other section: `../reference/io_formats.md`
  - package source: `../../src/meta_finder/file_finder.py`
- **Heading anchors**: GitHub slugs — lowercase, punctuation dropped, each space → `-`; or explicit `{#my-id}` suffix.
- **Images**: Relative path, kept next to the doc.

## meta_finder-specific notes

- Russian docs use `_Ru` suffix (e.g. `getting_started_Ru.md`); Russian root readme is `readme_Ru.md`.
- **Linking rule**: from Russian docs, link **only** to `_Ru` versions when they exist. When no `_Ru` version exists, link to the English doc **without** any comment that it is English.
- From English docs, link to the best existing version. Do not add language annotations to links.
- The main entry point is `readme.md`; Russian equivalent is `readme_Ru.md`.
- All docs are under `docs/` with subdirectories: `user_guide/`, `reference/`, `methodology/`, `python_developer_guide/`, `project_developer_guide/`.
- **Empty sections**: when a documentation section exists but contains no documents, the readme must explicitly state that no documents are available yet, rather than leaving an empty placeholder. If a section (e.g. `methodology/`) is not relevant to the project, remove it entirely from the readme instead of creating an empty placeholder.

## Self-check

Open each document through the viewer and verify: every formula is typeset, every code block highlighted, every local link navigates without 400/404, back button returns.
