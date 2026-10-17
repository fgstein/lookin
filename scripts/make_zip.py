"""Zip the whole repository, except .git, .venv, .claude/worktrees and
src/lookin/data/example.

The archive is named lookinX.Y.zip, X.Y being `__version__` from
src/lookin/__init__.py (the same place hatch reads it from).

The archive's entries live under a top-level `lookin/` folder, so unzipping
it gives one directory rather than spilling files into the current one.

Usage:
    uv run python scripts/make_zip.py [--out ../lookinX.Y.zip]
"""
from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Skipped wherever they appear (nested worktrees carry their own).
EXCLUDED_NAMES = {".git", ".venv"}
# Paths relative to the repository root; a directory excludes everything in it.
EXCLUDED_PATHS = [Path("src/lookin/data/example"), Path(".claude/worktrees")]


def read_version() -> str:
    # Parsed rather than imported, so the script needs none of Lookin's dependencies.
    tree = ast.parse((ROOT / "src/lookin/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    raise SystemExit("No __version__ found in src/lookin/__init__.py")


def is_excluded(rel: Path) -> bool:
    if EXCLUDED_NAMES.intersection(rel.parts):
        return True
    return any(rel == ex or ex in rel.parents for ex in EXCLUDED_PATHS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip the repository for distribution.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="archive to write (default: lookinX.Y.zip next to the repository)",
    )
    args = parser.parse_args()
    out = (args.out or ROOT.parent / f"lookin{read_version()}.zip").resolve()

    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if is_excluded(rel) or path == out or not path.is_file():
                continue
            zf.write(path, Path(ROOT.name) / rel)
            count += 1

    print(f"Wrote {count} files to {out}")


if __name__ == "__main__":
    main()
