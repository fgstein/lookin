"""Column roles and image root of dataset CSVs, read from a lookin.toml
sitting next to them.

Guessing the id, image and label columns from the values works for most
files, but a folder of datasets prepared by someone else usually comes with
the answer already known. Written once beside the CSVs, it fills the
Sources tab in for every one of them:

    # lookin.toml -- in the folder of the CSVs it describes
    [defaults]              # optional: every CSV of this folder
    root = "images"         # relative to this file's folder

    ["train.csv"]           # one table per CSV, by file name
    id = "uuid"
    image = "crop"
    label = "label"

    ["*_test.csv"]          # or by pattern; an exact name wins over one
    label = "true_class"
    root = "/data/test_images"

Keys: `id`, `image`, `label` (column names; `id = ""` says the file has no
id column) and `root` (the folder crop paths are relative to). A table
names only the keys it knows; the others come from [defaults], then from
the guess. Only this one file name is read, so the user always knows where
the answer came from.
"""
from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONFIG_NAME = "lookin.toml"
DEFAULTS_TABLE = "defaults"
KEYS = ("id", "image", "label", "root")


@dataclass
class DatasetConfig:
    """What lookin.toml says about one CSV. None: the file does not say."""
    source: str = ""                  # the lookin.toml it came from
    id: Optional[str] = None          # "" = no id column
    image: Optional[str] = None
    label: Optional[str] = None
    root: Optional[str] = None        # absolute
    error: str = ""                   # the file is there but unusable

    def any(self) -> bool:
        return any(getattr(self, k) is not None for k in KEYS)


def _table_values(table: dict, base: Path) -> dict:
    out = {}
    for key in KEYS:
        value = table.get(key)
        if not isinstance(value, str):
            continue
        if key == "root":
            value = value.strip()
            if not value:
                continue
            path = Path(value).expanduser()
            value = str(path if path.is_absolute() else (base / path))
        out[key] = value
    return out


def config_for_csv(csv_path) -> Optional[DatasetConfig]:
    """The lookin.toml entry for `csv_path`, or None when there is no such
    file or it says nothing about this CSV. A file that cannot be read
    comes back as a DatasetConfig with only `error` set: the user wrote it
    on purpose and should hear why it is ignored."""
    try:
        csv = Path(str(csv_path or "")).expanduser()
        config = csv.parent / CONFIG_NAME
        if not csv.name or not config.is_file():
            return None
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as ex:
        return DatasetConfig(source=str(csv.parent / CONFIG_NAME), error=f"could not be read: {ex}")
    except tomllib.TOMLDecodeError as ex:
        return DatasetConfig(source=str(config), error=f"is not valid TOML: {ex}")
    base = config.parent
    values: dict = {}
    defaults = raw.get(DEFAULTS_TABLE)
    if isinstance(defaults, dict):
        values.update(_table_values(defaults, base))
    tables = {k: v for k, v in raw.items() if k != DEFAULTS_TABLE and isinstance(v, dict)}
    if csv.name in tables:
        values.update(_table_values(tables[csv.name], base))
    else:
        # The most specific pattern: the longest one that matches.
        patterns = sorted((k for k in tables if any(c in k for c in "*?[") and fnmatch.fnmatchcase(csv.name, k)),
                          key=len, reverse=True)
        if patterns:
            values.update(_table_values(tables[patterns[0]], base))
    found = DatasetConfig(source=str(config), **values)
    return found if found.any() else None
