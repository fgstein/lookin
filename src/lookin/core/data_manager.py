"""In-memory data model for the tool: the crop/label dataset, one or more
loaded "experiment" result CSVs, and the joins/derived views built from them.

There is no database -- everything lives in pandas DataFrames in memory,
matching the "load a CSV" workflow described for the tool (a future version
will load the same shapes from MLFlow instead of CSV files).

Sized for up to ~20 experiments of ~500k rows each: CSVs are read in chunks
(with progress callbacks so the UI can show a real progress bar), numeric
columns are downcast to float32/int32 to roughly halve memory, and the
dataset<->experiment join is cached (a single slot, since only one
experiment's merge is ever needed at a time by any one view) so repeated
UI interactions -- toggling filters, switching tabs -- don't repeatedly
re-merge hundreds of thousands of rows.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

BINARY_MAX_UNIQUE = 2
CATEGORICAL_MAX_UNIQUE = 20
# Every categorical value becomes one UI control (a checkbox in the Gallery
# filter panel, a bar in its Statistics histogram) -- fine for a genuine
# category column, but a real-world "filename", "notes" or "session_id"
# metadata column can easily have as many unique values as there are rows.
# Without this cap, such a column silently generates hundreds of thousands
# of Flet controls and looks exactly like a hang: no error, no progress,
# just a browser/server that never comes back.
HIGH_CARDINALITY_MAX_UNIQUE = 200
# The same guard for the label column, set higher: real datasets have
# thousands of classes (fine-grained species, product SKUs), and the views
# that used to build one control per class pair now show a focus set of at
# most class_focus.FOCUS_MAX classes instead. This only has to catch a
# wrong column -- a uuid or free-text field has about as many unique
# values as rows.
LABEL_MAX_CLASSES = 5_000
CHUNK_ROWS = 50_000

# What pool is currently loaded. Keys, compared in code (see
# ui/cleaning_view.py): predictions on a train pool are in-sample, which
# changes what a cleaning diagnostic is allowed to claim and gates the bulk
# actions -- excluding hundreds of test crops in one click is how a metric
# gets faked, so that only exists for train.
SPLIT_TEST = "test"
SPLIT_TRAIN = "train"
SPLIT_OTHER = "other"
SPLIT_KINDS = (SPLIT_TEST, SPLIT_TRAIN, SPLIT_OTHER)

ProgressCallback = Callable[[int, int], None]  # (rows_done, rows_total)


def _log(msg: str) -> None:
    """Every load phase prints here so a stuck/slow load is visible in the
    terminal running `lookin ui` -- the UI only ever shows the last status
    line, which isn't enough to tell a hang from a merely-slow phase, or to
    tell which phase (row counting, CSV parsing, dtype downcasting, preview
    building...) is the one actually taking the time."""
    print(f"[lookin] {msg}", flush=True)


def _count_data_rows(path: str) -> int:
    """Fast line count (no parsing) used to size the progress bar. This is a
    full read of the file (just not a parse), so for a very large CSV it can
    itself take a noticeable amount of time -- logged separately from the
    parse below so that's visible rather than looking like part of it."""
    t0 = time.monotonic()
    with open(path, "rb") as f:
        count = sum(1 for _ in f)
    _log(f"counted {max(0, count - 1):,} data rows in {path} ({time.monotonic() - t0:.1f}s)")
    return max(0, count - 1)  # minus header row


def _read_csv_with_progress(path: str, progress_cb: Optional[ProgressCallback] = None,
                            dtype: Optional[dict] = None) -> pd.DataFrame:
    _log(f"reading {path} ...")
    t0 = time.monotonic()
    if progress_cb is None:
        df = pd.read_csv(path, dtype=dtype)
        _log(f"read {len(df):,} rows from {path} in {time.monotonic() - t0:.1f}s")
        return df

    total = _count_data_rows(path)
    if total <= CHUNK_ROWS:
        df = pd.read_csv(path, dtype=dtype)
        progress_cb(len(df), max(total, len(df), 1))
        _log(f"read {len(df):,} rows from {path} in {time.monotonic() - t0:.1f}s")
        return df

    chunks = []
    rows_read = 0
    n_chunks = -(-total // CHUNK_ROWS)  # ceil
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_ROWS, dtype=dtype), start=1):
        chunks.append(chunk)
        rows_read += len(chunk)
        progress_cb(min(rows_read, total), total)
        _log(f"  chunk {i}/{n_chunks}: {rows_read:,}/{total:,} rows ({time.monotonic() - t0:.1f}s elapsed)")
    _log(f"concatenating {len(chunks)} chunks ({rows_read:,} rows)...")
    df = pd.concat(chunks, ignore_index=True)
    _log(f"read {len(df):,} rows from {path} in {time.monotonic() - t0:.1f}s total")
    return df


def _downcast_numeric(df: pd.DataFrame, exclude: set) -> None:
    """In place: float64 -> float32, int64 -> smallest safe int dtype.
    Cuts memory roughly in half for wide numeric/score tables at scale,
    at a precision cost that's irrelevant for metadata/probabilities."""
    t0 = time.monotonic()
    for col in df.columns:
        if col in exclude:
            continue
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    _log(f"downcast {len(df.columns)} columns in {time.monotonic() - t0:.1f}s")


def _labels_as_str(series: pd.Series) -> pd.Series:
    """A label column as strings, missing values left missing.

    An experiment's class names are its score-column *headers*, which are
    always strings -- so a dataset whose labels read as numbers (0/1) would
    never match a single prediction, and sklearn refuses the mix outright.
    A column of whole numbers with a gap in it reads as float, so it goes
    through Int64 first: the header is "1", not "1.0".
    """
    values = series
    if pd.api.types.is_float_dtype(values):
        present = values.dropna()
        if (present == present.round()).all():
            values = values.astype("Int64")
    return values.astype(str).astype(object).where(values.notna(), np.nan)


def _load_npz_no_pickle(path: str) -> dict:
    """Read an .npz into a plain {name: array} dict, refusing pickle.

    Refusing pickle is deliberate: these sidecars are exactly the kind of
    file that gets copied around between machines/users, and allowing
    pickle would mean loading one can execute arbitrary code. The trade-off
    is that the single most common way to build a string array
    (np.array(a_list_of_python_str) via pandas' .to_numpy(), or an explicit
    dtype=object) produces a pickled object array that this then refuses --
    a numpy internals traceback about 'Object arrays cannot be loaded' is
    not something a user can act on, so that specific failure is translated
    into the actual fix here.

    Every member is materialized inside the try, and that is the point:
    np.load on an .npz returns a *lazy* NpzFile, so the refusal is raised
    when a member is first read, not by np.load itself. Handing that lazy
    object back would put the failure outside this handler and give the
    user the raw numpy message -- which is what used to happen for every
    sidecar, since .npz is the only thing this is ever called on.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except ValueError as ex:
        if "Object arrays cannot be loaded" in str(ex):
            raise ValueError(
                f"Could not read {path}: it contains a pickled object array, which is refused for "
                f"safety (a sidecar file can come from anywhere). This almost always means a string "
                f"array (uuid/labels) was saved with dtype=object -- save it as a fixed-width string "
                f"array instead, e.g. np.array(the_list, dtype='<U64')."
            ) from ex
        raise


def sidecar_paths_for_csv(csv_path) -> tuple:
    """The by-convention embeddings/prototypes sidecar paths for a results
    CSV: same directory, same base name with the trailing '_results' (if
    any) swapped for '_embeddings'/'_prototypes' -- 'baseline_cnn_results.csv'
    pairs with 'baseline_cnn_embeddings.npz' and
    'baseline_cnn_prototypes.npz'. A stem that doesn't end in '_results'
    (e.g. a CSV the user just named 'baseline_cnn.csv') is used as-is, so
    this always degrades to "same base name, .npz" rather than failing to
    guess anything. Doesn't check existence -- callers do that themselves,
    since "does it exist" means different things for a local path
    (Path.exists()) vs an MLflow artifact path (present in that run's
    artifact listing).

    Works equally on a real filesystem path and on an MLflow
    artifact-relative path (e.g. "some/dir/baseline_cnn_results.csv") --
    Path.with_name only touches the final path component either way.
    """
    p = Path(csv_path)
    stem = p.stem
    if stem.endswith("_results"):
        stem = stem[: -len("_results")]
    return p.with_name(f"{stem}_embeddings.npz"), p.with_name(f"{stem}_prototypes.npz")


@dataclass
class ColumnConfig:
    """Editable mapping from the tool's logical roles to actual CSV column names.

    `uuid_col` may be "" -- a dataset with no id column at all. Such a
    dataset is only ever looked at: the crops, their metadata and its
    statistics. Every join with an experiment goes through the id, so no
    results CSV can be scored against it (see AppState.add_experiment), and
    neither can the review store, whose decisions are keyed by it.
    """
    uuid_col: str = "uuid"
    crop_col: str = "crop"
    label_col: str = "label"

    def has_ids(self) -> bool:
        return bool(self.uuid_col)


@dataclass
class ColumnKinds:
    numeric: list = field(default_factory=list)
    binary: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    # Non-numeric columns with more unique values than HIGH_CARDINALITY_MAX_UNIQUE
    # -- excluded from Statistics charts and Gallery filter checkboxes (see
    # the constant's comment for why), but still shown in the dataset preview.
    high_cardinality: list = field(default_factory=list)


@dataclass
class Dataset:
    df: pd.DataFrame
    columns: ColumnConfig
    base_dir: Path
    # Explicit override for where relative crop paths resolve from, set via
    # the Data tab's "Image root path" field -- lets the CSV live somewhere
    # other than the images themselves (e.g. CSV checked into a repo, images
    # on a separate mounted volume). Falls back to base_dir (the CSV's own
    # directory) when not set, which was the only behavior before.
    image_root: Optional[Path] = None
    # The CSV this was read from, kept because two things need the *file*
    # rather than the loaded frame: the review sidecars are named after it
    # (see core/review.py review_paths_for_csv), and exporting for training
    # has to re-read it in full -- `df` here may be a single split, and an
    # export that silently dropped every other split would be worse than
    # useless.
    csv_path: Optional[Path] = None
    # The (column, value) row filter this was loaded through, if any --
    # remembered for the session history and shown next to the split kind.
    row_filter: Optional[tuple] = None

    def metadata_columns(self) -> list:
        reserved = {self.columns.uuid_col, self.columns.crop_col, self.columns.label_col}
        return [c for c in self.df.columns if c not in reserved]

    def column_kinds(self) -> ColumnKinds:
        kinds = ColumnKinds()
        for col in self.metadata_columns():
            series = self.df[col]
            nunique = series.nunique(dropna=True)
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if nunique <= BINARY_MAX_UNIQUE:
                kinds.binary.append(col)
            elif is_numeric and nunique > CATEGORICAL_MAX_UNIQUE:
                kinds.numeric.append(col)
            elif not is_numeric and nunique > HIGH_CARDINALITY_MAX_UNIQUE:
                kinds.high_cardinality.append(col)
            else:
                kinds.categorical.append(col)
        return kinds

    def resolve_crop_path(self, relative_or_abs: str) -> Path:
        p = Path(relative_or_abs)
        if p.is_absolute():
            return p
        return (self.image_root or self.base_dir) / p

    def labels(self) -> list:
        return sorted(self.df[self.columns.label_col].dropna().unique().tolist())

    def apply_label_remap(self, mapping: dict) -> None:
        """Rename raw label values in place, e.g. {"0": "cat", "1": "dog"}."""
        if not mapping:
            return
        col = self.columns.label_col
        self.df[col] = self.df[col].replace(mapping)


# The legend/group name the *dataset* pool takes when points are colored by
# where they come from (see the Embeddings tab's "Source" coloring). A
# display string, but also the name that coloring groups on, so an
# inference set is not allowed to claim it -- two sources sharing a name
# would merge into one legend entry and one color.
DATASET_SOURCE_GROUP = "dataset"


def read_embeddings_npz(path: str) -> tuple:
    """(uuids (N,) str, embeddings (N, D) float32, L2-normalized) from an
    .npz holding 'uuid' and 'embeddings' arrays.

    Shared by the two things that read that format: an experiment's
    embeddings sidecar, and a simple inference's embeddings (see
    InferenceSet) -- the file is the same shape in both cases, and so is
    every way it can be wrong.
    """
    from lookin.core.embedding_analysis import l2_normalize

    data = _load_npz_no_pickle(path)
    if "uuid" not in data or "embeddings" not in data:
        raise ValueError(f"Embeddings file must contain 'uuid' and 'embeddings' arrays, found: {list(data)}")
    uuids = data["uuid"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    if len(uuids) != len(embeddings):
        raise ValueError(f"'uuid' ({len(uuids)}) and 'embeddings' ({len(embeddings)}) length mismatch")
    return uuids, l2_normalize(embeddings)


@dataclass
class InferenceSet:
    """Crops that are *not* in the dataset, carried only far enough to be
    looked at: a CSV of paths plus one embedding per row.

    What this is for
    ----------------
    "Here are images the model has never been scored on -- where do they
    land?" Until now the only way onto the projection was to be a row of
    the dataset CSV *and* have a score in a results CSV, because every
    point comes from AppState.merged. A batch of new acquisitions, a
    competitor's samples, yesterday's production images: none of those have
    a label or a place in the dataset, and none of them could be shown next
    to it.

    Why it is deliberately not a Dataset and not an Experiment
    ----------------------------------------------------------
    It carries no label, so nothing here can be right or wrong -- and a
    pool with no ground truth must never reach a metric. Rather than a flag
    every future caller has to remember to test, these live in their own
    slot: AppState.merged does not know they exist, so Metrics, Cleaning,
    Compare, Calibration, the slice breakdown and the review store cannot
    accidentally count them. What they do get is the Embeddings tab (the
    map, the thumbnails, the carousel, "find similar"), which is what they
    were asked for.

    Whatever other columns the CSV happens to carry -- a confidence, a
    sensor, even per-class scores -- are kept and offered as colorings, and
    that is *all* they are used for.

    Rows and embeddings, aligned once
    ---------------------------------
    df row i is embeddings row i, always: the join with the .npz happens at
    load time and rows without an embedding are dropped there (they could
    not be plotted anyway). Downstream code then never has to re-align
    anything -- unlike an experiment, whose sidecar covers an arbitrary
    subset of the merged frame and is re-aligned on every use.
    """
    name: str
    df: pd.DataFrame                 # one row per crop that has an embedding
    uuid_col: str
    crop_col: str
    embeddings: np.ndarray           # (N, D) L2-normalized, row-aligned with df
    csv_path: Optional[Path] = None
    embeddings_path: Optional[str] = None
    # Where relative crop paths were resolved from. Kept for the status
    # line only: the paths in df are already absolute (see load_inference).
    image_root: Optional[Path] = None
    # What the two input files held, for the "kept 240 of 300 rows" note --
    # partial coverage is normal here for the same reason it is for an
    # experiment's sidecar.
    csv_rows: int = 0
    npz_rows: int = 0

    def metadata_columns(self) -> list:
        reserved = {self.uuid_col, self.crop_col}
        return [c for c in self.df.columns if c not in reserved]

    def uuids(self) -> np.ndarray:
        return self.df[self.uuid_col].astype(str).to_numpy()

    def vector_for(self, uuid: str):
        """This set's embedding for one uuid, or None. The lookup behind
        "find similar" on an inference crop: the query vector comes from
        here, the neighbours from the experiment being viewed."""
        matches = np.flatnonzero(self.uuids() == str(uuid))
        return None if not len(matches) else self.embeddings[matches[0]]

    def display_records(self, uuid_col: str, crop_col: str, rows=None) -> list:
        """This set's rows as dicts keyed the way the shared crop widgets
        expect -- the *dataset's* uuid/crop column names.

        The carousel and the map read a record through one pair of column
        names for every point they are given, and an inference CSV is free
        to name those columns anything. Renaming here (rather than teaching
        every widget about a second pair of names) is what lets an
        inference crop hover, open and page exactly like a dataset crop.
        The original values are not lost: only the *keys* move, and the two
        source columns are dropped from their old names so a record never
        carries the same value twice under two spellings.

        `rows` restricts this to a subset of the frame (the rows a map
        actually drew), so a set of a hundred thousand crops does not
        materialize a hundred thousand dicts to show a few thousand.
        """
        frame = self.df if rows is None else self.df.iloc[rows]
        records = []
        for row in frame.to_dict("records"):
            uuid = row.pop(self.uuid_col, None)
            crop = row.pop(self.crop_col, None)
            row[uuid_col] = uuid
            row[crop_col] = crop
            records.append(row)
        return records


@dataclass
class Experiment:
    """One loaded results CSV: per-uuid class scores for a named model run.

    class_labels is this experiment's own class_i -> label mapping (index
    aligned with score_cols) -- kept per-experiment since different model
    runs can use different class orderings.
    """
    name: str
    df: pd.DataFrame
    uuid_col: str
    score_cols: list
    class_labels: list = field(default_factory=list)

    # -- optional embedding-space sidecars (see core/embedding_analysis.py) --
    # embeddings are the backbone's raw feature vectors for this run, keyed
    # by uuid rather than aligned to `df`'s row order (a sidecar covering
    # only some of df's rows -- e.g. a subsampled extraction -- is normal,
    # not an error; see aligned_embeddings()).
    embeddings: Optional[np.ndarray] = field(default=None, repr=False)  # (M, D), L2-normalized
    embeddings_uuids: Optional[np.ndarray] = field(default=None, repr=False)  # (M,) str
    embeddings_path: Optional[str] = None
    # Real, imported prototypes only -- see embedding_analysis.PrototypeSet;
    # a kmeans-estimated PrototypeSet is never stored here, it's computed on
    # demand by the UI and kept out of this persistent, "trustworthy" slot
    # specifically so no scored diagnostic can accidentally pick it up.
    model_prototypes: Optional[object] = field(default=None, repr=False)
    model_prototypes_path: Optional[str] = None

    def n_classes(self) -> int:
        return len(self.score_cols)

    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    def load_embeddings(self, path: str) -> int:
        """Load an .npz sidecar with 'uuid' (N,) and 'embeddings' (N, D)
        arrays. L2-normalizes on load regardless of whether the exporter
        already did -- cosine similarity assumes unit vectors, and
        re-normalizing is cheap insurance against a sidecar that forgot to.
        Returns the number of embeddings loaded."""
        uuids, embeddings = read_embeddings_npz(path)
        self.embeddings = embeddings
        self.embeddings_uuids = uuids
        self.embeddings_path = path
        return len(uuids)

    def load_model_prototypes(self, path: str):
        """Load an .npz sidecar with 'labels' (C,) and 'prototypes'
        (C, K, D) -- or (C, D) for K=1 -- arrays: the model's own,
        architecturally-real prototypes (e.g. an ArcFace head's
        sub-centers). NaN rows (a class absent from whatever pool the
        prototypes were built from) are preserved and skipped by every
        downstream computation. Returns the built PrototypeSet."""
        from lookin.core.embedding_analysis import MODEL_SOURCE, PrototypeSet, l2_normalize

        data = _load_npz_no_pickle(path)
        if "labels" not in data or "prototypes" not in data:
            raise ValueError(f"Prototypes file must contain 'labels' and 'prototypes' arrays, found: {list(data)}")
        labels = data["labels"].astype(str).tolist()
        vectors = data["prototypes"].astype(np.float32)
        if vectors.ndim == 2:
            vectors = vectors[:, None, :]
        if len(labels) != vectors.shape[0]:
            raise ValueError(f"'labels' ({len(labels)}) and 'prototypes' first dim ({vectors.shape[0]}) mismatch")
        nan_mask = np.isnan(vectors).any(axis=-1, keepdims=True)
        vectors = np.where(nan_mask, np.nan, l2_normalize(np.nan_to_num(vectors)))
        proto_set = PrototypeSet(vectors=vectors.astype(np.float32), class_labels=labels, source=MODEL_SOURCE, k=vectors.shape[1])
        self.model_prototypes = proto_set
        self.model_prototypes_path = path
        return proto_set

    def aligned_embeddings(self, merged: pd.DataFrame, uuid_col: str):
        """This experiment's embeddings reindexed to match `merged`'s row
        order. Returns (embeddings, mask): mask marks which of merged's
        rows actually had an embedding in the sidecar -- a sidecar covering
        fewer uuids than the score CSV is expected, not an error.

        The explicit int64 dtype matters for the degenerate case: `merged`
        is empty when the experiment's uuids don't overlap the dataset's at
        all (a results CSV loaded against the wrong dataset -- easy to do
        from the MLflow browser), and np.array([]) would default to float64,
        which numpy then refuses as an index array."""
        if self.embeddings is None:
            return None, None
        uuid_to_row = {u: i for i, u in enumerate(self.embeddings_uuids)}
        merged_uuids = merged[uuid_col].astype(str).to_numpy()
        idx = np.array([uuid_to_row.get(u, -1) for u in merged_uuids], dtype=np.int64)
        mask = idx >= 0
        out = np.zeros((len(merged), self.embeddings.shape[1]), dtype=np.float32)
        out[mask] = self.embeddings[idx[mask]]
        return out, mask


@dataclass
class UuidOverlap:
    """Cheap (set-based, no join) comparison between a dataset and one
    experiment's uuid columns -- used to warn about mismatches (wrong uuid
    column picked, formatting differences, ids that just don't overlap)
    without paying for a full merge, and to catch duplicate-uuid inputs
    that would otherwise blow up an inner join into a much bigger-than-
    expected result."""
    matched: int
    dataset_only: int
    experiment_only: int
    dataset_dupes: int
    experiment_dupes: int


@dataclass
class AppState:
    dataset: Optional[Dataset] = None
    experiments: dict = field(default_factory=dict)  # name -> Experiment
    # name -> InferenceSet: crops with embeddings but no place in the
    # dataset. Kept apart from `dataset`/`experiments` on purpose -- see
    # InferenceSet for why nothing that computes a metric may see them.
    inferences: dict = field(default_factory=dict)
    # Annotation-review state for the loaded dataset (core.review.ReviewStore).
    # Attached on load_dataset() from the CSV's sidecars, so it is always
    # the store belonging to the data currently on screen.
    review: Optional[object] = None
    # Which pool is loaded -- see SPLIT_KINDS.
    split_kind: str = SPLIT_TEST
    # When True, merged() resolves labels through the review overlay and
    # drops excluded rows, so every metric in the app answers "what would
    # this be on the cleaned pool". Off by default, and the views that
    # honor it show a banner while it's on -- a silently cleaned metric is
    # the easiest way to fool yourself with this tool.
    use_reviewed_labels: bool = False
    # Which review version the overlay resolves through. None means "the
    # store's active version", which is what you want while reviewing.
    # Pinning one is what lets a v1 model and a v3 model be scored against
    # the same labels -- comparing an experiment against two different
    # cleaning passes and calling the difference a model improvement is the
    # mistake this exists to make impossible.
    review_version_id: Optional[str] = None
    _merged_cache_name: Optional[str] = field(default=None, repr=False, compare=False)
    _merged_cache_df: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)
    # (use_reviewed_labels, review revision) at the time the cache was
    # filled. A decision must not by itself invalidate a merge of hundreds
    # of thousands of rows -- but a merge computed *through* the overlay
    # has to notice the overlay moved, and this is what tells the two apart.
    _merged_cache_stamp: Optional[tuple] = field(default=None, repr=False, compare=False)

    def load_dataset(
        self, csv_path: str, columns: ColumnConfig, progress_cb: Optional[ProgressCallback] = None,
        row_filter: Optional[tuple] = None, image_root: Optional[str] = None,
    ) -> Dataset:
        df = _read_csv_with_progress(csv_path, progress_cb)
        required_cols = [columns.crop_col, columns.label_col]
        if columns.has_ids():
            required_cols.insert(0, columns.uuid_col)
        for required in required_cols:
            if required not in df.columns:
                raise ValueError(f"Column '{required}' not found in dataset CSV. Available: {list(df.columns)}")

        if row_filter is not None:
            filter_col, filter_value = row_filter
            if filter_col not in df.columns:
                # Silently ignoring a typo'd filter column would be worse
                # than doing nothing -- the user would see an unfiltered
                # dataset and not know why the filter didn't apply.
                _log(f"row filter column '{filter_col}' not found in dataset CSV -- skipping row filter "
                     f"(available columns: {list(df.columns)})")
            else:
                before = len(df)
                df = df[df[filter_col].astype(str) == str(filter_value)].reset_index(drop=True)
                _log(f"row filter {filter_col}={filter_value!r}: kept {len(df):,}/{before:,} rows")

        df[columns.label_col] = _labels_as_str(df[columns.label_col])

        # The label column drives per-value UI everywhere (confusion matrix
        # rows/cols, relabeling fields, filter checkboxes) -- if it's not
        # actually a small set of classes (wrong column name pointed at
        # something else, e.g. a uuid or free-text field), those all try to
        # build one control per unique value and hang exactly like the
        # HIGH_CARDINALITY_MAX_UNIQUE metadata-column case. Catching it here
        # gives a clear error instead of a silent, unexplained freeze.
        n_labels = df[columns.label_col].nunique(dropna=True)
        if n_labels > LABEL_MAX_CLASSES:
            raise ValueError(
                f"Column '{columns.label_col}' has {n_labels:,} unique values -- too many to be a "
                f"classification label column (limit {LABEL_MAX_CLASSES:,}). "
                f"Is '{columns.label_col}' the right column name?"
            )
        _downcast_numeric(df, exclude={columns.uuid_col, columns.crop_col})
        resolved_root = Path(image_root).expanduser().resolve() if image_root and image_root.strip() else None
        dataset = Dataset(
            df=df, columns=columns, base_dir=Path(csv_path).resolve().parent, image_root=resolved_root,
            csv_path=Path(csv_path).resolve(), row_filter=row_filter,
        )
        self.dataset = dataset
        if not columns.has_ids() and self.experiments:
            # Loaded against the previous dataset; with no id to join on,
            # every view would fail on them rather than show nothing.
            _log(f"dataset has no uuid column: unloading {len(self.experiments)} experiment(s)")
            self.experiments.clear()
        self.attach_review_store(dataset)
        self._invalidate_merged_cache()
        return dataset

    def attach_review_store(self, dataset: Dataset, author: str = "") -> object:
        """Load (or start) the review store belonging to a dataset CSV, and
        adopt any `review_status_*` columns the CSV itself carries.

        That adoption is what closes the loop: export an enriched CSV, train
        on it, load *that* file back here, and the versions it encodes are
        picked back up -- the sidecars are a convenience, not the only place
        the review state can live.
        """
        from lookin.core.review import ReviewStore, review_paths_for_csv

        if not dataset.columns.has_ids():
            # A decision is keyed by the crop's id; without one there is
            # nothing to key it by, and a store keyed by row position would
            # re-attach every decision to the wrong crop after one edit.
            self.review = None
            return None
        decisions_path, versions_path = review_paths_for_csv(dataset.csv_path)
        store = ReviewStore.load(decisions_path, versions_path, author=author)
        adopted = store.adopt_from_dataframe(dataset.df, dataset.columns.uuid_col, dataset.columns.label_col)
        if adopted:
            store.active_version_id = store.resume_version_id()
        self.review = store
        return store

    def add_experiment(self, csv_path: str, name: str, uuid_col: str = "uuid", progress_cb: Optional[ProgressCallback] = None) -> Experiment:
        if self.dataset is not None and not self.dataset.columns.has_ids():
            raise ValueError(
                "The dataset was loaded with no uuid column, so no results CSV can be matched to its "
                "crops. Pick its uuid column in the Data tab and load the dataset again."
            )
        # Score columns parsed straight to float32, which is what they are
        # downcast to anyway: parsed as float64 first, a results file of two
        # thousand classes held twice its final size in memory while loading
        # (and once more while its chunks were concatenated).
        header = pd.read_csv(csv_path, nrows=0).columns
        try:
            df = _read_csv_with_progress(csv_path, progress_cb, dtype={
                c: np.float32 for c in header if c not in (uuid_col, "label")})
        except ValueError as ex:
            raise ValueError(
                f"Every column of a results CSV other than '{uuid_col}' (and an optional 'label') "
                f"is read as one class's score, so it must be numeric -- {ex}"
            ) from ex
        if uuid_col not in df.columns:
            raise ValueError(f"Column '{uuid_col}' not found in results CSV. Available: {list(df.columns)}")
        # Score columns are every column except uuid and an optional
        # pre-computed 'label' column (the already-argmax'd prediction, not
        # a class score, so it's not usable as one and must be excluded).
        # The column *names* are the class names directly -- "cat", "dog",
        # "airplane", ... -- there's nothing to guess: a previous version of
        # this tried to guess class names from the dataset's own label
        # values, which had no real connection to which score column meant
        # which class.
        non_score = {uuid_col, "label"}
        score_cols = [c for c in df.columns if c not in non_score]
        if not score_cols:
            raise ValueError("No class score columns found in results CSV.")
        _downcast_numeric(df, exclude={uuid_col})
        experiment = Experiment(
            name=name, df=df, uuid_col=uuid_col, score_cols=score_cols,
            class_labels=list(score_cols),
        )
        self.experiments[name] = experiment
        self._invalidate_merged_cache(name)
        return experiment

    def remove_experiment(self, name: str) -> None:
        self.experiments.pop(name, None)
        self._invalidate_merged_cache(name)

    def load_inference(
        self, name: str, csv_path: str, embeddings_path: str, uuid_col: str = "uuid",
        crop_col: str = "crop", image_root: Optional[str] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> InferenceSet:
        """Load a simple inference: a CSV of crops plus its embeddings .npz.

        Only two columns are required -- an id and a path to the image.
        Everything else the file happens to carry is kept as metadata (see
        InferenceSet for what that is and is not allowed to be used for),
        and no label is expected: these crops are not in the dataset and
        nothing here knows what they are.

        The .npz is not optional. Without it there is no position to draw
        the crops at, and the *only* place they are shown is the projection
        -- so a CSV on its own would load successfully and then be invisible
        everywhere, which is worse than refusing it.

        Crop paths are resolved to absolute here, once, against
        `image_root` (or the CSV's own directory). That is what makes an
        inference crop displayable by the shared widgets: they all resolve
        a path through the *dataset* (Dataset.resolve_crop_path), which
        returns an absolute path untouched -- so these images come out
        right without every one of them learning about a second root.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("A simple inference needs a name.")
        if name == DATASET_SOURCE_GROUP:
            raise ValueError(
                f"'{DATASET_SOURCE_GROUP}' is reserved: it is the name the dataset's own points take "
                f"when the projection is colored by source, and two sources cannot share one name."
            )
        df = _read_csv_with_progress(csv_path, progress_cb)
        for required, role in ((uuid_col, "id"), (crop_col, "crop path")):
            if required not in df.columns:
                raise ValueError(
                    f"Column '{required}' ({role}) not found in '{name}'. Available: {list(df.columns)}"
                )
        csv_rows = len(df)

        uuids, embeddings = read_embeddings_npz(embeddings_path)
        npz_rows = len(uuids)
        row_of_uuid = {u: i for i, u in enumerate(uuids)}
        emb_idx = np.array([row_of_uuid.get(u, -1) for u in df[uuid_col].astype(str)], dtype=np.int64)
        keep = emb_idx >= 0
        if not keep.any():
            raise ValueError(
                f"None of the {csv_rows:,} rows of '{name}' has an embedding in {Path(embeddings_path).name} "
                f"({npz_rows:,} vectors). Are the two files from the same run, and do their uuids match?"
            )
        # Dropped rather than kept-and-hidden: a row with no embedding has
        # no position, and the map is the only place these are shown.
        df = df[keep].reset_index(drop=True)
        embeddings = embeddings[emb_idx[keep]]
        _log(f"inference '{name}': {len(df):,} of {csv_rows:,} rows matched an embedding "
             f"({npz_rows:,} in the sidecar)")

        root = Path(image_root).expanduser().resolve() if image_root and image_root.strip() \
            else Path(csv_path).resolve().parent
        df[crop_col] = [
            str(p if p.is_absolute() else root / p)
            for p in (Path(str(value)) for value in df[crop_col])
        ]
        _downcast_numeric(df, exclude={uuid_col, crop_col})

        inference = InferenceSet(
            name=name, df=df, uuid_col=uuid_col, crop_col=crop_col, embeddings=embeddings,
            csv_path=Path(csv_path).resolve(), embeddings_path=embeddings_path, image_root=root,
            csv_rows=csv_rows, npz_rows=npz_rows,
        )
        self.inferences[name] = inference
        return inference

    def remove_inference(self, name: str) -> None:
        self.inferences.pop(name, None)

    def inference_for_uuid(self, uuid: str):
        """The inference set holding this uuid, or None -- for the widgets
        that are handed a uuid and have to work out where it came from
        ("find similar" on a crop that may be from either pool). The
        dataset is checked first by the callers, so a uuid that happens to
        exist in both belongs to the dataset here."""
        for inference in self.inferences.values():
            if inference.vector_for(uuid) is not None:
                return inference
        return None

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Change an experiment's display name in place, keeping its
        position in `experiments` and -- crucially -- its already-parsed
        DataFrame, embeddings and prototypes. A name is only ever a label
        here (nothing about the loaded data depends on it), so renaming
        must never mean re-reading the results CSV: at a few hundred
        thousand rows that read is the single most expensive thing the app
        does, and paying it just to fix a typo is what this avoids.

        Views hold their own name-keyed caches and dropdown values, so the
        rename has to be announced to them too -- see
        AppContext.notify_rename(), which every view mirrors into its own
        state instead of recomputing.
        """
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if old_name not in self.experiments:
            raise KeyError(f"No experiment named '{old_name}' is loaded.")
        if new_name in self.experiments:
            raise ValueError(f"An experiment named '{new_name}' is already loaded.")
        # Rebuilt rather than popped-and-reinserted: dicts preserve
        # insertion order, and that order is the user-visible experiment
        # order everywhere (see reorder_experiments), so a rename must not
        # quietly move the experiment to the end of every table and legend.
        self.experiments = {
            (new_name if name == old_name else name): exp
            for name, exp in self.experiments.items()
        }
        self.experiments[new_name].name = new_name
        if self._merged_cache_name == old_name:
            self._merged_cache_name = new_name

    def reorder_experiments(self, names: list) -> None:
        """Set the experiment order to `names`. That order is what every
        comparison view reads: table rows, chart legends, bar groups, the
        per-experiment color assignment -- so this is the one place the
        user's preferred reading order is stored.

        Names not currently loaded are ignored and loaded experiments
        missing from `names` keep their relative order at the end, so a
        list built from a stale UI snapshot reorders what it can instead
        of dropping experiments.
        """
        ordered = [n for n in names if n in self.experiments]
        remaining = [n for n in self.experiments if n not in set(ordered)]
        self.experiments = {n: self.experiments[n] for n in ordered + remaining}

    def clear_all(self) -> None:
        self.dataset = None
        self.experiments.clear()
        self.inferences.clear()
        # Decisions are already on disk (the review store saves after every
        # mutation), so dropping the in-memory store here loses nothing --
        # and keeping a store bound to a dataset that is no longer loaded
        # would let a decision be written against the wrong CSV.
        self.review = None
        self.use_reviewed_labels = False
        self.review_version_id = None
        self._merged_cache_name = None
        self._merged_cache_df = None
        self._merged_cache_stamp = None

    def set_class_label(self, experiment_name: str, index: int, label: str) -> None:
        exp = self.experiments.get(experiment_name)
        if exp and 0 <= index < len(exp.class_labels):
            exp.class_labels[index] = label
            self._invalidate_merged_cache(experiment_name)

    def apply_label_remap(self, mapping: dict) -> None:
        """Rename dataset label values and cascade the rename into every
        loaded experiment's class_labels, so predictions keep matching up."""
        if self.dataset is None or not mapping:
            return
        self.dataset.apply_label_remap(mapping)
        for exp in self.experiments.values():
            exp.class_labels = [mapping.get(l, l) for l in exp.class_labels]
        self._invalidate_merged_cache()

    def _invalidate_merged_cache(self, name: Optional[str] = None) -> None:
        if name is None or self._merged_cache_name == name:
            self._merged_cache_name = None
            self._merged_cache_df = None
            self._merged_cache_stamp = None

    def overlay_version_id(self) -> Optional[str]:
        """The version the overlay actually resolves through: the pinned one
        when it still exists, otherwise the store's active version."""
        if self.review is None:
            return None
        if self.review_version_id and self.review_version_id in self.review.versions:
            return self.review_version_id
        return self.review.active_version_id

    def _review_stamp(self) -> tuple:
        revision = getattr(self.review, "revision", -1) if self.review is not None else -1
        version_id = self.overlay_version_id()
        # The revision only matters while the overlay is actually applied:
        # with the toggle off, a decision changes nothing about the merge,
        # so it must not throw away a cached one.
        return (self.use_reviewed_labels, revision, version_id) if self.use_reviewed_labels else (False,)

    def uuid_overlap(self, experiment_name: str) -> UuidOverlap:
        dataset_uuids = self.dataset.df[self.dataset.columns.uuid_col]
        experiment = self.experiments[experiment_name]
        experiment_uuids = experiment.df[experiment.uuid_col]
        dataset_set = set(dataset_uuids)
        experiment_set = set(experiment_uuids)
        return UuidOverlap(
            matched=len(dataset_set & experiment_set),
            dataset_only=len(dataset_set - experiment_set),
            experiment_only=len(experiment_set - dataset_set),
            dataset_dupes=int(dataset_uuids.duplicated().sum()),
            experiment_dupes=int(experiment_uuids.duplicated().sum()),
        )

    def merged(self, experiment_name: str) -> pd.DataFrame:
        """Dataset joined with one experiment's scores, plus derived columns:
        predicted_label, predicted_score (confidence), correct.

        Cached for the single most-recently-requested experiment -- cheap to
        keep (only one experiment's merge is ever in view at once, in either
        the Gallery or a single Metrics computation pass) and avoids
        re-merging hundreds of thousands of rows on every filter tweak.
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded.")
        if not self.dataset.columns.has_ids():
            raise ValueError("The dataset has no uuid column, so it cannot be joined with an experiment.")
        stamp = self._review_stamp()
        if (self._merged_cache_name == experiment_name and self._merged_cache_df is not None
                and self._merged_cache_stamp == stamp):
            return self._merged_cache_df

        experiment = self.experiments[experiment_name]
        cols = self.dataset.columns
        overlap = self.uuid_overlap(experiment_name)
        _log(f"merging dataset ({len(self.dataset.df):,} rows) with experiment '{experiment_name}' "
             f"({len(experiment.df):,} rows): {overlap.matched:,} matching uuids, "
             f"{overlap.dataset_only:,} dataset-only, {overlap.experiment_only:,} experiment-only")
        if overlap.dataset_dupes or overlap.experiment_dupes:
            # An inner join is many-to-many for any duplicated key, so
            # duplicate uuids on both sides multiply out -- flagging this
            # up front explains an unexpectedly huge/slow merge instead of
            # it just looking like a hang.
            _log(f"warning: duplicate uuids found (dataset: {overlap.dataset_dupes:,}, "
                 f"experiment: {overlap.experiment_dupes:,}) -- the join below may produce more "
                 f"rows than either input if the same uuid repeats on both sides")
        t0 = time.monotonic()
        merged = self.dataset.df.merge(
            experiment.df, left_on=cols.uuid_col, right_on=experiment.uuid_col, how="inner",
            suffixes=("", "_result"),
        )
        _log(f"merge produced {len(merged):,} rows in {time.monotonic() - t0:.1f}s")
        merged = self._apply_review_overlay(merged, cols)
        score_cols = experiment.score_cols
        labels_for_cols = experiment.class_labels[: len(score_cols)]
        scores = merged[score_cols].to_numpy()
        pred_idx = scores.argmax(axis=1)
        # pandas warns that inserting into a frame of thousands of column
        # blocks is slow, and suggests concatenating instead -- which would
        # copy every score column to add three. Three inserts are cheap.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            merged["predicted_label"] = [labels_for_cols[i] if i < len(labels_for_cols) else None for i in pred_idx]
            merged["predicted_score"] = scores.max(axis=1)
            merged["correct"] = merged["predicted_label"] == merged[cols.label_col]

        self._merged_cache_name = experiment_name
        self._merged_cache_df = merged
        self._merged_cache_stamp = stamp
        return merged

    def _apply_review_overlay(self, merged: pd.DataFrame, cols: ColumnConfig) -> pd.DataFrame:
        """With use_reviewed_labels on: replace the label column by the
        reviewed label and drop excluded rows, *before* correctness is
        computed, so every downstream number (confusion matrix, per-class
        report, curves) answers the same "on the cleaned pool" question.

        Excluded crops are dropped rather than kept-and-ignored on purpose:
        a crop judged unusable isn't a sample the model got wrong, it's a
        sample that should never have been in the pool.
        """
        from lookin.core.review import STATUS_EXCLUDE

        if not self.use_reviewed_labels or self.review is None or not self.review.versions:
            return merged
        version_id = self.overlay_version_id()
        labels, statuses = self.review.overlay(
            merged[cols.uuid_col].astype(str).to_numpy(), merged[cols.label_col].to_numpy(),
            version_id,
        )
        merged = merged.copy()
        merged[cols.label_col] = labels
        merged["review_status"] = statuses
        before = len(merged)
        merged = merged[merged["review_status"] != STATUS_EXCLUDE].reset_index(drop=True)
        _log(f"review overlay: {before - len(merged):,} excluded row(s) dropped, "
             f"{len(merged):,} kept (version '{version_id}')")
        return merged
