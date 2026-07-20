"""Extract device_dir paths from all *_files_TCM.tsv and generate path availability report."""

import argparse
import logging
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set, Tuple

from meta_finder.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Import matcher  ("C:/Work/Python/AB_SIO_RAS/cruises_organizer/match_dirs/src")
if (MATCH_DIRS_PATH := PROJECT_ROOT.parent / "match_dirs" / "src") not in sys.path:
    sys.path.insert(0, str(MATCH_DIRS_PATH))
import matcher
    
FILE_EXTENSIONS: Set[str] = {
    ".csv",
    ".txt",
    ".tsv",
    ".h5",
    ".hdf5",
    ".mat",
    ".zip",
    ".7z",
    ".rar",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
}

MATCH_CUTOFF: float = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check availability of device directories from *_files_TCM.tsv files"
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=PROJECT_ROOT / "meta" / "collection",
        help="Directory containing *_files_TCM.tsv files",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=MATCH_CUTOFF,
        help=f"Minimum similarity threshold (default: {MATCH_CUTOFF})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output report file path"
    )
    parser.add_argument(
        "symlinks_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Create NTFS symlinks to all device directories in this directory",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only process the latest *_files_TCM.tsv file",
    )
    return parser.parse_args()


def is_device_dir_line(line: str) -> bool:
    for part in PurePosixPath(line).parts:
        if PurePosixPath(part).suffix.lower() in FILE_EXTENSIONS:
            return False
    return True


def extract_device_dirs(files_tsv: Path) -> List[str]:
    device_dirs: List[str] = []
    with open(files_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and is_device_dir_line(line):
                device_dirs.append(line)
    return device_dirs


def _longest_common_parent(paths: List[str]) -> Path:
    if not paths:
        raise ValueError("No common parent directory found among device dirs")
    parsed = [Path(p).resolve().parents for p in paths]
    common = list(parsed[0])
    for parents in parsed[1:]:
        common = [d for d in common if d in parents]
    if not common:
        raise ValueError("No common parent directory found among device dirs")
    return common[0]


def _filter_nested(
    target_map: Dict[str, Path], logger: logging.Logger
) -> Dict[str, Path]:
    items = sorted(target_map.items(), key=lambda kv: str(kv[1]))
    filtered: Dict[str, Path] = {}
    for dd, target in items:
        target_resolved = target.resolve()
        is_nested = False
        for prev_dd, prev_target in filtered.items():
            prev_resolved = prev_target.resolve()
            try:
                target_resolved.relative_to(prev_resolved)
                logger.warning(
                    f"Skipping nested device dir: {dd} ({target_resolved} "
                    f"is inside {prev_resolved} from {prev_dd})"
                )
                is_nested = True
                break
            except ValueError:
                pass
        if not is_nested:
            filtered[dd] = target
    return filtered


def main() -> int:
    args = parse_args()
    sys.argv = [sys.argv[0]]

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)

    def find_similar_in_parent(
        old_path: str, cutoff: float = MATCH_CUTOFF
    ) -> Tuple[Optional[str], float]:
        old_path_obj = Path(old_path)
        parent = old_path_obj.parent
        old_name = old_path_obj.name

        if not parent.exists():
            return None, 0.0

        candidate_dirs: List[str] = []
        try:
            for item in parent.iterdir():
                if item.is_dir():
                    candidate_dirs.append(str(item))
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot access {parent}: {e}")
            return None, 0.0

        if not candidate_dirs:
            return None, 0.0

        scores: List[Tuple[str, float]] = []
        for candidate in candidate_dirs:
            sim = matcher.hierarchical_weighed_similarity(
                old_name, Path(candidate).name
            )
            if sim >= cutoff:
                scores.append((candidate, sim))

        if not scores:
            return None, 0.0
        return max(scores, key=lambda x: x[1])

    def check_device_dir(
        device_dir: str, cutoff: float = MATCH_CUTOFF
    ) -> Tuple[str, Optional[str], float]:
        if Path(device_dir).exists():
            return "OK", None, 0.0

        similar_path, sim_score = find_similar_in_parent(device_dir, cutoff)
        if similar_path:
            if sim_score >= matcher.HIGH_CONFIDENCE_THRESHOLD:
                return "RENAMED", similar_path, sim_score
            return "UNCERTAIN", similar_path, sim_score
        return "MISSING", None, 0.0

    collection_dir = args.collection_dir
    if not collection_dir.exists():
        logger.error(f"Collection directory not found: {collection_dir}")
        return 1

    tsv_files = sorted(collection_dir.glob("*_files_TCM.tsv"))
    if not tsv_files:
        logger.warning(f"No *_files_TCM.tsv files found in {collection_dir}")
        return 0

    if args.latest_only:
        tsv_files = [tsv_files[-1]]
        logger.info(f"Processing latest file only: {tsv_files[0].name}")

    logger.info(f"Found {len(tsv_files)} *_files_TCM.tsv file(s)")

    all_device_dirs: Dict[str, List[str]] = {}
    for tsv_file in tsv_files:
        device_dirs = extract_device_dirs(tsv_file)
        for dd in device_dirs:
            if dd not in all_device_dirs:
                all_device_dirs[dd] = []
            all_device_dirs[dd].append(tsv_file.name)

    logger.info(f"Found {len(all_device_dirs)} unique device directories")

    results: List[Tuple[str, str, Optional[str], float, List[str]]] = []
    counts = {"OK": 0, "RENAMED": 0, "UNCERTAIN": 0, "MISSING": 0}

    for device_dir in sorted(all_device_dirs):
        status, similar, score = check_device_dir(device_dir, args.cutoff)
        sources = all_device_dirs[device_dir]
        results.append((status, device_dir, similar, score, sources))
        counts[status] += 1

    header_comment = (
        "# similar_path: suggested renamed path (when device_dir not found)\n"
        "# score: similarity score between original dir name and similar_path (0.0-1.0)\n"
        "# sources: *_files_TCM.tsv files where this device_dir appears"
    )

    lines: List[str] = [header_comment]
    lines.append("status\tdevice_dir\tsimilar_path\tscore\tsources")
    for status, device_dir, similar, score, sources in results:
        similar_str = similar or ""
        score_str = f"{score:.2f}" if similar else ""
        sources_str = "; ".join(sources)
        lines.append(
            f"{status}\t{device_dir}\t{similar_str}\t{score_str}\t{sources_str}"
        )

    lines.append("")
    lines.append(f"Total: {len(all_device_dirs)}")
    lines.append(f"  OK: {counts['OK']}")
    lines.append(f"  Renamed: {counts['RENAMED']}")
    lines.append(f"  Uncertain: {counts['UNCERTAIN']}")
    lines.append(f"  Missing: {counts['MISSING']}")

    report = "\n".join(lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        logger.info(f"Report written to {args.output}")
    else:
        print(report)

    if args.symlinks_dir:
        import os

        symlinks_dir: Path = args.symlinks_dir

        target_map: Dict[str, Path] = {}
        for status, device_dir, similar, score, sources in results:
            target = (
                Path(str(similar))
                if status in ("RENAMED", "UNCERTAIN")
                else Path(device_dir)
            )
            if not target.exists():
                logger.warning(f"Symlink target does not exist, skipping: {target}")
                continue
            target_map[device_dir] = target

        filtered_map = _filter_nested(target_map, logger)

        if len(filtered_map) < 2:
            logger.warning(
                "Fewer than 2 non-nested device dirs; cannot determine common parent, "
                "creating flat symlinks"
            )
            symlinks_dir.mkdir(parents=True, exist_ok=True)
            created, skipped, failed = 0, 0, 0
            for dd, target in filtered_map.items():
                target_resolved = target.resolve()
                link_path = symlinks_dir / target_resolved.name
                if link_path.exists() or link_path.is_symlink():
                    logger.debug(f"Symlink already exists: {link_path}")
                    skipped += 1
                    continue
                try:
                    os.symlink(
                        str(target_resolved), str(link_path), target_is_directory=True
                    )
                    created += 1
                except OSError as e:
                    logger.error(
                        f"Failed to create symlink {link_path} -> {target_resolved}: {e}"
                    )
                    failed += 1
            logger.info(
                f"Symlinks in {symlinks_dir}: {created} created, "
                f"{skipped} skipped, {failed} failed"
            )
        else:
            common_parent = _longest_common_parent(
                [str(t) for t in filtered_map.values()]
            )
            logger.info(f"Common parent directory: {common_parent}")

            created, skipped, failed = 0, 0, 0

            for device_dir, target in sorted(filtered_map.items()):
                target_resolved = target.resolve()
                try:
                    rel = target_resolved.parent.relative_to(common_parent)
                except ValueError:
                    logger.warning(
                        f"Target parent {target_resolved.parent} is not under "
                        f"common parent {common_parent}, skipping: {device_dir}"
                    )
                    skipped += 1
                    continue

                link_parent = symlinks_dir / rel
                link_parent.mkdir(parents=True, exist_ok=True)
                link_path = link_parent / target_resolved.name

                if link_path.exists() or link_path.is_symlink():
                    logger.debug(f"Symlink already exists: {link_path}")
                    skipped += 1
                    continue

                try:
                    os.symlink(
                        str(target_resolved), str(link_path), target_is_directory=True
                    )
                    created += 1
                except OSError as e:
                    logger.error(
                        f"Failed to create symlink {link_path} -> {target_resolved}: {e}"
                    )
                    failed += 1

            logger.info(
                f"Symlinks in {symlinks_dir} (hierarchy from {common_parent}): "
                f"{created} created, {skipped} skipped, {failed} failed"
            )

    return 0 if counts["MISSING"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
