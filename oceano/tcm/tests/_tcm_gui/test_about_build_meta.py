"""Tests for build metadata (version_meta.json) and About dialog doc discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Build metadata (generate_version_info.py) ───────────────────────────────


class TestVersionMeta:
    """JSON meta write/read round-trip."""

    def test_write_load_roundtrip(self, tmp_path: Path) -> None:
        """write_meta creates JSON in out_dir; load_meta reads it back."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            from generate_version_info import load_meta, write_meta

            meta = write_meta(
                "2026.08",
                product="tcm_gui",
                description="Test description",
                internal_name="tcm_gui.exe",
                original_filename="tcm\\scripts\\tcm_gui.py",
                company_name="AB SIO RAS",
                legal_copyright="© Test",
                product_name="TCM calculations",
                suffixes=["test-env"],
                project_root=tmp_path,
                out_dir=tmp_path,
            )

            assert meta["name"] == "TCM"
            assert meta["product"] == "tcm_gui"
            assert meta["version"] == "2026.08+test-env"
            assert meta["filevers"] == [2026, 8, 0, 0]
            assert meta["description"] == "Test description"
            assert meta["company_name"] == "AB SIO RAS"
            assert meta["legal_copyright"] == "© Test"
            assert meta["product_name"] == "TCM calculations"
            assert meta["internal_name"] == "tcm_gui.exe"

            # JSON was written to out_dir (never the real scripts/build dir)
            assert (tmp_path / "version_meta.json").is_file()
            assert load_meta(tmp_path)["version"] == "2026.08+test-env"
        finally:
            sys.path.pop(0)

    def test_load_meta_from_spec_dir(self, tmp_path: Path) -> None:
        """load_meta reads from specified spec_dir."""
        meta_content = {
            "name": "TCM",
            "product": "tcm_proc",
            "version": "2025.12",
            "filevers": [2025, 12, 0, 0],
            "description": "CLI",
            "internal_name": "tcm_proc.exe",
            "original_filename": "tcm_proc.py",
            "repo_url": None,
            "docs_url": None,
        }
        json_path = tmp_path / "version_meta.json"
        json_path.write_text(json.dumps(meta_content), encoding="utf-8")

        # Import after path setup
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            from generate_version_info import load_meta

            loaded = load_meta(tmp_path)
            assert loaded["product"] == "tcm_proc"
            assert loaded["version"] == "2025.12"
        finally:
            sys.path.pop(0)


class TestRepoUrl:
    """Git remote URL normalization."""

    def test_https_url_unchanged(self) -> None:
        """HTTPS URLs are returned as-is (minus .git suffix)."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            # Mock subprocess to return a known URL
            from generate_version_info import repo_url

            def mock_run(cmd, *args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "https://github.com/User/Repo.git\n"
                    stderr = ""

                return Result()

            import generate_version_info

            generate_version_info.subprocess.run = mock_run

            result = repo_url(Path("/fake"))
            assert result == "https://github.com/User/Repo"
        finally:
            sys.path.pop(0)

    def test_ssh_url_normalized(self) -> None:
        """SSH URLs are converted to HTTPS."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            import generate_version_info

            def mock_run(cmd, *args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "git@github.com:User/Repo.git\n"
                    stderr = ""

                return Result()

            generate_version_info.subprocess.run = mock_run
            result = generate_version_info.repo_url(Path("/fake"))
            assert result == "https://github.com/User/Repo"
        finally:
            sys.path.pop(0)

    def test_no_git_returns_none(self) -> None:
        """Missing git or remote returns None."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            import generate_version_info

            def mock_run(cmd, *args, **kwargs):
                class Result:
                    returncode = 1
                    stdout = ""
                    stderr = "error"

                return Result()

            generate_version_info.subprocess.run = mock_run
            result = generate_version_info.repo_url(Path("/fake"))
            assert result is None
        finally:
            sys.path.pop(0)

    def test_docs_url_points_to_docs_subfolder(self) -> None:
        """docs_url resolves to oceano/tcm/docs subfolder of the repo on the given branch."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build"))
        try:
            from generate_version_info import docs_url

            repo = "https://github.com/User/Repo"
            assert docs_url(repo, "main") == "https://github.com/User/Repo/tree/main/oceano/tcm/docs"
            assert docs_url(repo, "dev") == "https://github.com/User/Repo/tree/dev/oceano/tcm/docs"
            assert (
                docs_url(repo) == "https://github.com/User/Repo/tree/main/oceano/tcm/docs"
            )  # default branch
            assert docs_url(None) is None
        finally:
            sys.path.pop(0)


# ── About dialog doc discovery ──────────────────────────────────────────────


class TestDocDiscovery:
    """Doc title extraction and discovery."""

    def test_extract_title_from_heading(self, tmp_path: Path) -> None:
        """First # heading becomes the title."""
        md = tmp_path / "test.md"
        md.write_text("# My Document Title\n\nSome content here.", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _extract_title

            title = _extract_title(md)
            assert title == "My Document Title"
        finally:
            sys.path.pop(0)

    def test_extract_title_no_heading(self, tmp_path: Path) -> None:
        """File without # heading returns None."""
        md = tmp_path / "no_heading.md"
        md.write_text("## Subheading only\n\nNo top-level heading.", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _extract_title

            title = _extract_title(md)
            assert title is None
        finally:
            sys.path.pop(0)

    def test_discover_docs_excludes_todo(self, tmp_path: Path) -> None:
        """discover_docs excludes todo/ directory and reports folder names."""
        # Create test structure
        (tmp_path / "python_developer_guide").mkdir()
        (tmp_path / "todo").mkdir()
        (tmp_path / "python_developer_guide" / "readme.md").write_text("# CLI Docs", encoding="utf-8")
        (tmp_path / "todo" / "notes.md").write_text("# TODO Notes", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import discover_docs

            docs = discover_docs(tmp_path, lang="en")
            assert [(folder, title) for folder, title, _ in docs] == [("python_developer_guide", "CLI Docs")]
        finally:
            sys.path.pop(0)

    def test_discover_docs_sorted(self, tmp_path: Path) -> None:
        """discover_docs returns sorted results; top-level files have empty folder."""
        (tmp_path / "b.md").write_text("# Beta", encoding="utf-8")
        (tmp_path / "a.md").write_text("# Alpha", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import discover_docs

            docs = discover_docs(tmp_path, lang="en")
            assert [(folder, p.name) for folder, _, p in docs] == [("", "a.md"), ("", "b.md")]
        finally:
            sys.path.pop(0)

    def test_docs_tree_groups_by_folder(self, tmp_path: Path) -> None:
        """_docs_tree nests flat docs by directory, sorted by folder name."""
        (tmp_path / "reference").mkdir()
        (tmp_path / "python_developer_guide").mkdir()
        (tmp_path / "reference" / "a.md").write_text("# GUI", encoding="utf-8")
        (tmp_path / "python_developer_guide" / "b.md").write_text("# CLI", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _docs_tree, discover_docs

            tree = _docs_tree(discover_docs(tmp_path, lang="en"), tmp_path)
            assert list(tree["sub"]) == ["python_developer_guide", "reference"], "top-level dirs mismatch"
            assert [t for t, _ in tree["sub"]["python_developer_guide"]["files"]] == ["CLI"], (
                "python_developer_guide files mismatch"
            )
            assert [t for t, _ in tree["sub"]["reference"]["files"]] == ["GUI"], "reference files mismatch"
        finally:
            sys.path.pop(0)

    def test_docs_tree_follows_readme_order(self, tmp_path: Path) -> None:
        """Folders sort by readme.md link appearance, extras after."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "reference").mkdir()
        (tmp_path / "docs" / "user_guide").mkdir()
        (tmp_path / "docs" / "images").mkdir()
        (tmp_path / "docs" / "reference" / "z.md").write_text("# Z", encoding="utf-8")
        (tmp_path / "docs" / "user_guide" / "a.md").write_text("# A", encoding="utf-8")
        (tmp_path / "readme.md").write_text(
            'See <img src="docs/images/logo.png"> [guide](docs/user_guide/a.md)'
            " and [ref](docs/reference/z.md).",
            encoding="utf-8",
        )

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _docs_tree, discover_docs

            tree = _docs_tree(discover_docs(tmp_path / "docs", lang="en"), tmp_path / "docs")
            # image src must not count as a folder; link order wins
            assert list(tree["sub"]) == ["user_guide", "reference"], "readme order mismatch"
        finally:
            sys.path.pop(0)

    @pytest.mark.parametrize(
        "case,files,readme,expected_sub,expected_gui_files",
        [
            (
                "nested-gui-index-linked",
                [
                    ("project_developer_guide/CLI.md", "# CLI Internals"),
                    ("project_developer_guide/GUI/_index.md", "# GUI Internals"),
                    ("project_developer_guide/GUI/architecture.md", "# GUI Architecture"),
                    ("project_developer_guide/doc_authoring.md", "# Contract"),
                ],
                "- [CLI](docs/project_developer_guide/CLI.md)\n"
                "#### [GUI Internals](docs/project_developer_guide/GUI/_index.md)\n"
                "- [Arch](docs/project_developer_guide/GUI/architecture.md)\n"
                "- [Contract](docs/project_developer_guide/doc_authoring.md)\n",
                ["project_developer_guide"],
                ["GUI Architecture"],
            ),
        ],
        ids=["nested-gui-index-linked"],
    )
    def test_docs_tree_nested_index(
        self,
        tmp_path: Path,
        case: str,
        files: list,
        readme: str,
        expected_sub: list,
        expected_gui_files: list,
    ) -> None:
        """Parent dir links to _index.md; no separate _index leaf; readme orders siblings."""
        docs = tmp_path / "docs"
        for rel, body in files:
            p = docs / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{body}\n", encoding="utf-8")
        (tmp_path / "readme.md").write_text(readme, encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _docs_tree, _readme_doc_order, discover_docs

            order = _readme_doc_order(docs)
            assert order.get("project_developer_guide/GUI/_index.md") is not None, (
                f"{case}: _index.md link missing from readme order"
            )
            tree = _docs_tree(discover_docs(docs, lang="en"), docs)
            assert list(tree["sub"]) == expected_sub, f"{case}: top-level mismatch"
            pdg = tree["sub"]["project_developer_guide"]
            assert pdg["index"] is None, f"{case}: project_developer_guide must have no index"
            assert "GUI" in pdg["sub"], f"{case}: GUI subdir missing"
            gui = pdg["sub"]["GUI"]
            assert gui["index"] is not None, f"{case}: GUI index missing"
            assert gui["index"][0] == "GUI Internals", f"{case}: GUI index title mismatch"
            assert [t for t, _ in gui["files"]] == expected_gui_files, (
                f"{case}: GUI files mismatch — _index.md must not appear as leaf"
            )
            # sibling order: CLI, GUI, Contract (readme appearance)
            names = [t for t, _ in pdg["files"]] + ["GUI"]
            assert names[0] == "CLI Internals", f"{case}: CLI must precede GUI"
        finally:
            sys.path.pop(0)


class TestTreeClickTarget:
    """``_click_target`` separates the disclosure indicator from row text."""

    @pytest.mark.parametrize(
        "case,element,item,expected",
        [
            ("indicator-keeps-toggle", "Treeitem.indicator", "parent", None),
            ("text-opens-linked-parent", "Treeitem.text", "parent", "/docs/gui/_index.md"),
            ("leaf-text-opens", "Treeitem.text", "leaf", "/docs/a.md"),
            ("no-row-opens-nothing", "Treeitem.text", "", None),
        ],
        ids=["indicator-keeps-toggle", "text-opens-linked-parent", "leaf-text-opens", "no-row-opens-nothing"],
    )
    def test_click_target(self, case: str, element: str, item: str, expected: str | None) -> None:
        """Indicator clicks never open; text clicks resolve via ``_iid_path``."""

        class _FakeTree:
            def identify_row(self, _y: int) -> str:
                return item

            def identify_element(self, _x: int, _y: int) -> str:
                return element

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _click_target

            got = _click_target(_FakeTree(), 10, 20, {"parent": "/docs/gui/_index.md", "leaf": "/docs/a.md"})
            assert got == expected, f"{case}: expected {expected!r}, got {got!r}"
        finally:
            sys.path.pop(0)

    @pytest.mark.parametrize(
        "case,element,item,expected",
        [
            ("indicator-normal-cursor", "Treeitem.indicator", "parent", ""),
            ("text-hand-cursor", "Treeitem.text", "parent", "/docs/gui/_index.md"),
            ("empty-area-normal", "Treeitem.text", "", ""),
        ],
        ids=["indicator-normal-cursor", "text-hand-cursor", "empty-area-normal"],
    )
    def test_hover_path(self, case: str, element: str, item: str, expected: str) -> None:
        """Indicator and empty areas resolve to ``""`` (normal cursor); text resolves link."""

        class _FakeHoverTree:
            def identify_row(self, _y: int) -> str:
                return item

            def identify_element(self, _x: int, _y: int) -> str:
                return element

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _hover_path

            got = _hover_path(_FakeHoverTree(), 10, 20, {"parent": "/docs/gui/_index.md"})
            assert got == expected, f"{case}: expected {expected!r}, got {got!r}"
        finally:
            sys.path.pop(0)


class TestDocLangFilter:
    """Language-based doc filtering (``_lang_parts`` / ``_lang_filter``)."""

    @pytest.fixture()
    def about(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui import _about

            yield _about
        finally:
            sys.path.pop(0)

    def test_lang_parts_splits_suffix(self, about) -> None:
        assert about._lang_parts("readme_noh5_Ru") == ("readme_noh5", "ru")
        assert about._lang_parts("config_reference_ru") == ("config_reference", "ru")
        assert about._lang_parts("calibration_wiki") == ("calibration_wiki", None)
        assert about._lang_parts("README") == ("README", None)

    def test_en_excludes_suffixed(self, tmp_path: Path, about) -> None:
        """English keeps only unsuffixed docs (translations excluded)."""
        for name in ("a.md", "b_ru.md", "c_Ru.md"):
            (tmp_path / name).write_text(f"# {name}", encoding="utf-8")
        docs = about.discover_docs(tmp_path, lang="en")
        assert [p.name for _, _, p in docs] == ["a.md"]

    def test_ru_prefers_localized_over_plain(self, tmp_path: Path, about) -> None:
        """Current lang: ``_ru`` version wins; plain kept when no translation."""
        for name in ("a.md", "a_ru.md", "b.md"):
            (tmp_path / name).write_text(f"# {name}", encoding="utf-8")
        docs = about.discover_docs(tmp_path, lang="ru")
        assert [p.name for _, _, p in docs] == ["a_ru.md", "b.md"]

    def test_ru_fallback_keeps_only_translation(self, tmp_path: Path, about) -> None:
        """No ru/plain version of a base → keep whatever translation exists."""
        (tmp_path / "c_en.md").write_text("# C", encoding="utf-8")
        docs = about.discover_docs(tmp_path, lang="ru")
        assert [p.name for _, _, p in docs] == ["c_en.md"]

    def test_ru_excludes_other_lang_when_localized_exists(self, tmp_path: Path, about) -> None:
        """``_en`` sibling dropped when the ``_ru`` version of the same base exists."""
        for name in ("d.md", "d_en.md", "d_ru.md"):
            (tmp_path / name).write_text(f"# {name}", encoding="utf-8")
        docs = about.discover_docs(tmp_path, lang="ru")
        assert [p.name for _, _, p in docs] == ["d_ru.md"]


class TestLocalReadme:
    """bundled readme chosen by the resolved language, any language."""

    def test_picks_localized_readme_of_resolved_lang(self, tmp_path: Path, monkeypatch) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            import tcm_gui._about as about

            readmes = tmp_path
            (readmes / "readme.md").write_text("# Base", encoding="utf-8")
            (readmes / "readme_Ru.md").write_text("# RU", encoding="utf-8")
            monkeypatch.setattr(about, "DOC_DIR", readmes / "docs")
            monkeypatch.setattr(about, "resolve_lang", lambda: "ru")
            assert about.local_readme() == readmes / "readme_Ru.md"
            monkeypatch.setattr(about, "resolve_lang", lambda: "en")
            assert about.local_readme() == readmes / "readme.md"
            monkeypatch.setattr(about, "resolve_lang", lambda: "fr")  # no _fr file → base
            assert about.local_readme() == readmes / "readme.md"
        finally:
            sys.path.pop(0)


# ── Runtime version_meta loader ─────────────────────────────────────────────


class TestVersionMetaLoader:
    """tcm._constants.version_meta() fallback behavior."""

    def test_missing_file_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """version_meta() returns {} when no JSON exists."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

        from tcm import _constants

        # Clear cache and point to non-existent location
        _constants.version_meta.cache_clear()
        monkeypatch.setattr(_constants, "resource_root", lambda: Path("/nonexistent"))

        meta = _constants.version_meta()
        assert meta == {}

    def test_reads_from_dev_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """version_meta() reads from dev location when present."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

        from tcm import _constants

        # Create a test JSON at the dev layout: resource_root()/scripts/build/
        build_dir = tmp_path / "scripts" / "build"
        build_dir.mkdir(parents=True)
        meta_content = {"name": "TCM", "version": "2026.08", "product": "tcm_gui"}
        (build_dir / "version_meta.json").write_text(json.dumps(meta_content), encoding="utf-8")

        _constants.version_meta.cache_clear()
        monkeypatch.setattr(_constants, "resource_root", lambda: tmp_path)

        meta = _constants.version_meta()
        assert meta["name"] == "TCM"
        assert meta["version"] == "2026.08"
