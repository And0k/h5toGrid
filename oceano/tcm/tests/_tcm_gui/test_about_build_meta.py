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
        """_docs_tree groups flat docs by folder, sorted by folder name."""
        (tmp_path / "reference").mkdir()
        (tmp_path / "python_developer_guide").mkdir()
        (tmp_path / "reference" / "a.md").write_text("# GUI", encoding="utf-8")
        (tmp_path / "python_developer_guide" / "b.md").write_text("# CLI", encoding="utf-8")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from tcm_gui._about import _docs_tree, discover_docs

            tree = _docs_tree(discover_docs(tmp_path, lang="en"), tmp_path)
            assert list(tree) == ["python_developer_guide", "reference"]
            assert [title for title, _ in tree["python_developer_guide"]] == ["CLI"]
            assert [title for title, _ in tree["reference"]] == ["GUI"]
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
            assert list(tree) == ["user_guide", "reference"]
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
            assert about.AboutDialog._local_readme() == readmes / "readme_Ru.md"
            monkeypatch.setattr(about, "resolve_lang", lambda: "en")
            assert about.AboutDialog._local_readme() == readmes / "readme.md"
            monkeypatch.setattr(about, "resolve_lang", lambda: "fr")  # no _fr file → base
            assert about.AboutDialog._local_readme() == readmes / "readme.md"
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
