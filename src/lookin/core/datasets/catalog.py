"""The CSVs of the shared folders, and which one is derived from which.

A main dataset sits in a folder; the datasets derived from it by scripts
(a cleaned copy, a subset) sit in a folder named after it:

    shared/
      dataset_A.csv
      dataset_A/
        dataset_A_clean.csv
        dataset_A_subset.csv

So a CSV whose enclosing folder (or any folder above it, up to the shared
root) is named like a CSV beside that folder is a derivative of that CSV --
the nearest such folder wins. Nothing else is inferred: a CSV elsewhere is
a main dataset of its own.

Only the file system is read here: names, sizes, modification times. What
is *in* a file is profile.py's business, on demand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAX_SCAN_FILES = 5000
MAX_SCAN_DEPTH = 4
# Files that sit beside datasets and are not datasets: results CSVs are
# experiments, and the review sidecars belong to the dataset next to them.
NOT_A_DATASET_SUFFIXES = ("_results.csv", "_review.csv", "_review_versions.csv")


@dataclass
class DatasetFile:
    path: str                       # absolute
    name: str                       # file name, e.g. "dataset_A.csv"
    root: str                       # the shared folder it was found under
    size: int = 0
    mtime: float = 0.0
    mtime_ns: int = 0
    parent: Optional[str] = None    # path of the main dataset it derives from
    children: list = field(default_factory=list)    # paths of its derivatives

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    def relative(self) -> str:
        """The path from the shared folder's own name down, e.g.
        "datasets/A/A_clean.csv" -- unique across folders, which a path
        relative to the folder is not (two folders each with a dataset.csv)."""
        try:
            return str(Path(Path(self.root).name) / Path(self.path).relative_to(self.root))
        except ValueError:
            return self.path

    def stamp(self) -> tuple:
        """What says the file changed: size and modification time."""
        return (self.size, self.mtime_ns)


def is_dataset_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".csv") and not lowered.endswith(NOT_A_DATASET_SUFFIXES) and not name.startswith(".")


def _scan(root: Path) -> list:
    out = []
    stack = [(root, 0)]
    while stack and len(out) < MAX_SCAN_FILES:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir(follow_symlinks=True):
                    if depth < MAX_SCAN_DEPTH:
                        stack.append((Path(e.path), depth + 1))
                elif is_dataset_file(e.name):
                    st = e.stat()
                    out.append(DatasetFile(path=os.path.abspath(e.path), name=e.name, root=str(root),
                                           size=int(st.st_size), mtime=st.st_mtime, mtime_ns=st.st_mtime_ns))
            except OSError:
                continue
    return out


def _parent_of(f: DatasetFile, by_path: dict) -> Optional[str]:
    root = Path(f.root)
    folder = Path(f.path).parent
    while folder != root and root in folder.parents:
        candidate = str(folder.parent / (folder.name + ".csv"))
        if candidate in by_path and candidate != f.path:
            return candidate
        folder = folder.parent
    return None


def scan(folders: list) -> list:
    """Every dataset CSV under `folders`, mains first then their
    derivatives, each group by name. A folder that cannot be read gives
    nothing rather than an error: a shared mount that is down must not take
    the page with it (the page says which folders are missing)."""
    files: dict = {}
    for folder in folders:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        for f in _scan(root.resolve()):
            files.setdefault(f.path, f)
    for f in files.values():
        f.parent = _parent_of(f, files)
    for f in files.values():
        if f.parent:
            files[f.parent].children.append(f.path)
    for f in files.values():
        f.children.sort(key=lambda p: files[p].name.lower())
    return sorted(files.values(), key=lambda f: (f.name.lower(), f.path))


def missing_folders(folders: list) -> list:
    return [f for f in folders if not Path(f).expanduser().is_dir()]
