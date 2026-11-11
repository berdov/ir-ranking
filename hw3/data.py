"""Input validation and query-level splitting for the HW3 feature tables.

Identifiers are strings throughout: they are grouping/output keys, never model
features. Missing numeric cells are represented by NaN; infinite values are
rejected. Both CatBoost and LightGBM can handle numeric NaNs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


RESERVED_COLUMNS = frozenset({"ObjectId", "QueryId", "Label"})
MISSING_NUMERIC = ("", "NaN", "nan", "NA", "N/A", "NULL", "null", "None", "none")


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    query_ids: np.ndarray
    labels: np.ndarray | None
    object_ids: np.ndarray | None
    feature_names: tuple[str, ...]

    def subset(self, indices: np.ndarray) -> "Dataset":
        """Select rows without changing their identity or feature order."""
        return Dataset(
            self.X[indices], self.query_ids[indices],
            None if self.labels is None else self.labels[indices],
            None if self.object_ids is None else self.object_ids[indices],
            self.feature_names,
        )


def _check_csv(path: Path) -> tuple[list[str], int]:
    """Validate the original header and row widths before pandas normalizes them.

    This streaming pass uses O(number of columns) extra memory. In particular,
    it prevents duplicate headers being silently renamed and malformed rows
    being interpreted as an implicit dataframe index.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{path.name}: empty CSV")
            if any(not name or name != name.strip() for name in header):
                raise ValueError(f"{path.name}: blank or whitespace-padded column name")
            if len(set(header)) != len(header):
                raise ValueError(f"{path.name}: duplicate column names")
            rows = 0
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        f"{path.name}: line {reader.line_num} has {len(row)} fields; "
                        f"expected {len(header)}"
                    )
                rows += 1
            if rows == 0:
                raise ValueError(f"{path.name}: no data rows")
            return header, rows
    except csv.Error as exc:
        raise ValueError(f"{path.name}: malformed CSV: {exc}") from exc


def _identifiers(frame: pd.DataFrame, column: str, *, unique: bool = False) -> np.ndarray:
    values = frame[column].to_numpy(dtype=str)
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{column}: identifiers must be nonempty and have no outer whitespace")
    if unique and len(np.unique(values)) != len(values):
        raise ValueError(f"{column}: duplicate identifiers")
    return values


def _load(path: str | Path, *, train: bool,
          expected_features: tuple[str, ...] | None = None) -> Dataset:
    path = Path(path)
    header, row_count = _check_csv(path)
    required = {"QueryId", "Label"} if train else {"ObjectId", "QueryId"}
    actual_reserved = set(header) & RESERVED_COLUMNS
    if actual_reserved != required:
        raise ValueError(
            f"{path.name}: expected identifier/label columns {sorted(required)}, "
            f"got {sorted(actual_reserved)}"
        )
    features = tuple(name for name in header if name not in RESERVED_COLUMNS)
    if not features:
        raise ValueError(f"{path.name}: no feature columns")
    if expected_features is not None and features != tuple(expected_features):
        raise ValueError(f"{path.name}: feature names/order differ from the trained model")
    dtype = {name: np.float32 for name in features}
    dtype["QueryId"] = str
    if train:
        dtype["Label"] = np.float64
    else:
        dtype["ObjectId"] = str
    numeric = features + (("Label",) if train else ())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            warnings.simplefilter("error", RuntimeWarning)
            frame = pd.read_csv(
                path, dtype=dtype, encoding="utf-8-sig", keep_default_na=False,
                na_values={name: MISSING_NUMERIC for name in numeric},
                index_col=False, on_bad_lines="error",
            )
    except (ValueError, pd.errors.ParserError, pd.errors.ParserWarning, RuntimeWarning) as exc:
        raise ValueError(f"{path.name}: invalid numeric CSV: {exc}") from exc
    if len(frame) != row_count:
        raise ValueError(f"{path.name}: CSV parser changed the number of rows")
    X = np.ascontiguousarray(frame.loc[:, list(features)].to_numpy(dtype=np.float32))
    if np.isinf(X).any():
        raise ValueError(f"{path.name}: infinite or float32-overflowing features")
    query_ids = _identifiers(frame, "QueryId")
    labels = None
    object_ids = None
    if train:
        values = frame["Label"].to_numpy(dtype=np.float64)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > 3) or np.any(values != np.floor(values))):
            raise ValueError(f"{path.name}: Label must be an integer in [0, 3]")
        labels = values.astype(np.int32)
    else:
        object_ids = _identifiers(frame, "ObjectId", unique=True)
    return Dataset(X, query_ids, labels, object_ids, features)


def load_train(path: str | Path) -> Dataset:
    """Read only the specified training CSV; do not inspect any test files."""
    return _load(path, train=True)


def load_test(path: str | Path, feature_names: tuple[str, ...]) -> Dataset:
    """Read only test.csv using the model's recorded feature schema."""
    return _load(path, train=False, expected_features=tuple(feature_names))


def grouped(dataset: Dataset) -> tuple[Dataset, np.ndarray]:
    """Group query rows with a stable lexicographic sort.

    permutation[new_position] is the original row position. To undo the sort,
    allocate restored and assign restored[permutation] = sorted_predictions.
    """
    order = np.argsort(dataset.query_ids, kind="stable")
    return dataset.subset(order), order


def group_sizes(query_ids: np.ndarray) -> np.ndarray:
    """Lengths of contiguous groups; reject a query appearing in two blocks."""
    query_ids = np.asarray(query_ids)
    if query_ids.ndim != 1:
        raise ValueError("query_ids must be one-dimensional")
    if not len(query_ids):
        return np.empty(0, dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(query_ids[1:] != query_ids[:-1]) + 1]
    if len(np.unique(query_ids[starts])) != len(starts):
        raise ValueError("query rows are not contiguous; call grouped() first")
    return np.diff(np.r_[starts, len(query_ids)]).astype(np.int32)


def split_by_query(query_ids: np.ndarray, validation_fraction: float = 0.2,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic holdout with complete queries, independent of labels.

    The permutation is applied to sorted unique IDs, making membership invariant
    to the input row order. Returned row indices retain the original CSV order.
    """
    query_ids = np.asarray(query_ids)
    if query_ids.ndim != 1:
        raise ValueError("query_ids must be one-dimensional")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    unique = np.unique(query_ids)
    if len(unique) < 2:
        raise ValueError("at least two queries are needed for a holdout")
    n_validation = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation_queries = np.random.default_rng(seed).permutation(unique)[:n_validation]
    is_validation = np.isin(query_ids, validation_queries)
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)
