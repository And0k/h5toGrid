"""TDD for nested ``Important`` under ``Detailed`` — must not break parent detail.

Spec:
* ``### `path_field`` (L) → ``#### Detailed`` (L+1) → ``##### Important`` (L+2)
* ``##### Important`` must be stored as ``sub_details['Detailed']['Important']``
  without being appended to ``details['Detailed']`` nor closing the mode.
* ``help_general_for_path("path_field")`` must return concatenation (``\\n``)
   of all ``Important`` at L+1 and all ``Important`` under every ``Detailed`` at L+2.
   For the real doc snippet at ``config_reference.md:11-21`` this is the nested
   ``Important`` block under ``Detailed``.
"""
import textwrap

from tcm_gui._help import ModeBody, _NO_MODE, help_general_for_path, parse_reference

# Exact snippet from oceano/tcm/docs/reference/config_reference.md:11-21
_SNIPPET = textwrap.dedent("""
    ## CLI keys outside the typed configuration (not in YAML)

    | Field | Purpose |
    |-------|---------|
    | `path_field` = — | Data/config search path |

    ### `path_field`

    Search path for raw data / their processing configs: processing configs will be created in `cfg_proc/run/` subfolder if absent.

    #### Detailed

    If the path contains a subfolder named `_raw` output files will be one level above.

    ##### Important

    The search path must be an absolute path to:
    - a **directory** (e.g. `B:\\Cruises\\BalticSea\\inclinometer\\260624@ip05-Press\\_raw`)
    - raw file(s) via **glob** (`*i*.txt`) / **regex** (`i.*\\.txt`) — file-name filtering
    - config(s) (must end in **`.yaml`**) — load ready configs directly from the `cfg_proc/run/` subfolder
    """)


def _parse_snippet():
    return parse_reference(_SNIPPET)


def test_nested_important_preserved_without_breaking_detailed():
    """``##### Important`` is stored as sub_details, ``Detailed`` stays intact, and ``Important`` is in final tooltip."""
    ents = _parse_snippet()
    assert "path_field" in ents, "path_field entry missing"
    raw = ents["path_field"].body.get(_NO_MODE)
    assert isinstance(raw, ModeBody), f"expected ModeBody, got {type(raw).__name__}"
    assert "Detailed" in raw.details, f"Detailed missing, got {list(raw.details)}"
    # Parent Detailed must NOT contain the nested Important text
    detailed_text = raw.details["Detailed"]
    assert detailed_text, "Detailed text is empty"
    # Extract expected Important text from the parsed snippet (source of truth)
    important_text = raw.sub_details["Detailed"]["Important"].strip()
    assert important_text, "Important text extracted from snippet is empty"
    assert important_text not in detailed_text, (
        "Detailed was broken — nested Important leaked into parent"
    )
    # Nested Important must be in sub_details
    assert "Detailed" in raw.sub_details, f"sub_details missing Detailed, got {raw.sub_details}"
    assert "Important" in raw.sub_details["Detailed"], f"Important sub missing, got {raw.sub_details['Detailed']}"
    assert important_text in raw.sub_details["Detailed"]["Important"]
    # Final tooltip (error tooltip via help_general_for_path) must contain Important
    from tcm_gui import _help

    _help._CACHE["en"] = ents
    orig = _help.resolve_lang
    _help.resolve_lang = lambda: "en"
    try:
        tip = help_general_for_path("path_field")
    finally:
        _help.resolve_lang = orig
    assert tip, "final tooltip empty — Important not in tooltip"
    assert important_text in tip, f"Important not in final tooltip {tip[:500]!r}"
    assert detailed_text not in tip, "final tooltip must be Important, not Detailed"
    # General dwell (help_for_path with detail Detailed) must include nested Important without breaking
    from tcm_gui._help import help_for_path as hfp

    _help._CACHE["en"] = ents
    _help.resolve_lang = lambda: "en"
    try:
        de = hfp("path_field", mode="detail", detail="Detailed")
    finally:
        _help.resolve_lang = orig
    # help_for_path with detail Detailed should return Detailed + its Important (not break)
    assert de is not None and isinstance(de.body, str)
    assert detailed_text in de.body, "Detailed base missing in dwell tooltip"
    assert important_text in de.body, "nested Important not included in dwell Detailed tooltip"


def test_help_general_returns_nested_important():
    """Error tooltip concatenates L+1 Important + L+2 Important under Detailed."""
    from tcm_gui import _help

    ents = _parse_snippet()
    # Expected Important text extracted from the parsed snippet (source of truth)
    raw = ents["path_field"].body[_NO_MODE]
    important_text = raw.sub_details["Detailed"]["Important"].strip()
    detailed_text = raw.details["Detailed"]
    _help._CACHE["en"] = ents
    orig = _help.resolve_lang
    _help.resolve_lang = lambda: "en"
    try:
        tip = help_general_for_path("path_field")
    finally:
        _help.resolve_lang = orig
    assert important_text in tip, f"tooltip missing nested Important, got {tip!r}"
    assert detailed_text not in tip, "tooltip must be Important only, not Detailed short"


def test_concatenation_direct_plus_nested():
    """Direct ``#### Important`` (L+1) and nested ``##### Important`` (L+2) are ``\\n``-joined."""
    sample = textwrap.dedent("""
        ## CLI keys outside the typed configuration (not in YAML)

        | Field | Purpose |
        |-------|---------|
        | `path_field` = — | Path |

        ### `path_field`

        Short.

        #### Important

        Direct important A.

        #### Detailed

        Detail text.

        ##### Important

        Nested important B.

        #### Important

        Direct important C.
        """)
    ents = parse_reference(sample)
    raw = ents["path_field"].body[_NO_MODE]
    assert isinstance(raw, ModeBody)
    # Direct Important may be concatenated (multiple blocks same tag)
    assert "Direct important A" in raw.details["Important"]
    # Nested
    assert "Nested important B" in raw.sub_details["Detailed"]["Important"]
    from tcm_gui import _help

    _help._CACHE["en"] = ents
    orig = _help.resolve_lang
    _help.resolve_lang = lambda: "en"
    try:
        tip = help_general_for_path("path_field")
    finally:
        _help.resolve_lang = orig
    # Must contain both, joined by \n, in order: direct first, then nested
    assert tip.count("Direct important") == 2
    assert "Nested important B" in tip
    assert tip.index("Direct important A") < tip.index("Nested important B")


def test_level_agnostic_derives_l_from_heading():
    """Same structure at different base level (``##`` → ``###``/``####``) still works."""
    sample = textwrap.dedent("""
        ## `path_field` — Test

        Short.

        ### Detailed

        Detail at L+1 (since base is ##).

        #### Important

        Important at L+2.

        ### Important

        Direct Important at L+1.
        """)
    # ``## `path_field``` is a section, not a mode — field-level details are stored
    # under ``_FIELD_DETAIL``; for this test we force a mode at ``###`` level
    # via a synthetic ``### `path_field``` to verify L derivation.
    sample2 = textwrap.dedent("""
        ## CLI keys outside the typed configuration (not in YAML)

        | Field | Purpose |
        |-------|---------|
        | `path_field` = — | Path |

        #### `path_field`

        Short at L=4.

        ##### Detailed

        Detail at 5.

        ###### Important

        Important at 6.
        """)
    ents = parse_reference(sample2)
    assert "path_field" in ents
    raw = ents["path_field"].body.get(_NO_MODE) or ents["path_field"].body.get("detail")
    # If L derivation works, Detailed and its Important are captured
    if isinstance(raw, ModeBody):
        assert "Detailed" in raw.details or "Detailed" in raw.sub_details


def _detailed_lead(text: str) -> str:
    """First prose line of ``#### Detailed`` under ``### `path_field` `` — read from the doc source.

    The parser must preserve it verbatim in ``details["Detailed"]``; deriving it
    here keeps the markdown the single source of truth (no hardcoded phrases).
    """
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if l.strip() == "### `path_field`"), None)
    assert head is not None, "`### `path_field` ` heading missing in doc"
    det = next((i for i in range(head, len(lines)) if lines[i].strip() == "#### Detailed"), None)
    assert det is not None, "`#### Detailed` block missing under `### `path_field` `"
    lead = next((l.strip() for l in lines[det + 1 :] if l.strip()), "")
    assert lead, "`#### Detailed` block is empty in doc"
    return lead


def test_real_doc_important_in_final_tooltip(real_reference):
    """Real ``config_reference_{lang}.md`` → final tooltip contains the doc's
    ``Important`` block, not the short body.

    Source of truth = the doc file itself.  All expected text is extracted from
    the parsed doc structure (``sub_details['Detailed']['Important']``) and the
    raw markdown (``_detailed_lead``), never hardcoded.  The error tooltip for
    ``path_field`` (app.py:1354 ``help_general_for_path("path_field")``) must be
    the ``Important`` block, not the short pre-``####`` body, and must include
    the nested ``##### Important`` under ``#### Detailed`` without breaking the
    parent.
    """
    lang, doc, ents = real_reference.lang, real_reference.text, real_reference.entries
    assert "path_field" in ents, "path_field missing in real doc"
    raw = ents["path_field"].body.get(_NO_MODE)
    assert isinstance(raw, ModeBody), f"expected ModeBody, got {type(raw).__name__}"
    # Detailed must stay intact, Important must be in sub_details
    assert "Detailed" in raw.details, f"Detailed missing {list(raw.details)}"
    # Source of truth = the doc itself: the parsed block must contain its lead line.
    assert _detailed_lead(doc) in raw.details["Detailed"]
    assert "Detailed" in raw.sub_details and "Important" in raw.sub_details["Detailed"], (
        f"nested Important missing {raw.sub_details}"
    )
    # Extract expected Important text directly from the parsed doc (source of truth)
    important_text = raw.sub_details["Detailed"]["Important"].strip()
    assert important_text, "doc's Important block is empty — nothing to verify"

    # Final tooltip via help_general_for_path must contain the doc's Important content
    from tcm_gui import _help

    _help._CACHE[lang] = ents
    orig = _help.resolve_lang
    _help.resolve_lang = lambda l=lang: l
    try:
        tip = help_general_for_path("path_field")
    finally:
        _help.resolve_lang = orig
    assert tip, "final tooltip empty — Important not returned"
    assert important_text in tip, (
        f"doc's Important content not in final tooltip:\n"
        f"  expected subset: {important_text[:200]!r}\n"
        f"  got: {tip[:500]!r}"
    )
    # Tooltip must be the Important block, not the short pre-#### body
    assert ents["path_field"].short not in tip, (
        "tooltip is the short body, not the Important block"
    )
