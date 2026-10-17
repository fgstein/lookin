"""Removing rows from the loaded dataset: conditions on its columns, and a
query written as a formula.

What a filter is
----------------
A filter is a plain dict -- it is stored in workspaces and snapshots, so its
shape is a file format and its keys are never reworded:

    {
      "conditions": [
        {"column": "split", "op": "in", "values": ["test"], "on": true},
        {"column": "width", "op": "range", "min": 32, "max": null, "on": true},
      ],
      "query": "sensor == 'A' and not path.str.contains('blurry')",
      "query_on": true,
    }

A row is kept when it satisfies every condition that is on *and* the query
(when there is one and it is on). Conditions are what the Filter tab builds
by clicking; the query is for everything the conditions cannot say -- an
`or`, arithmetic between two columns, a regular expression.

Why the query is not `DataFrame.query`
--------------------------------------
pandas' `query`/`eval` runs Python: an expression can reach attributes and
call functions it was never meant to, and the text comes from a browser
form (and from snapshots other people hand around). So the query is parsed
here with `ast` and walked node by node against a whitelist: column names,
literals, comparisons, `and`/`or`/`not` (and `&`, `|`, `~`), arithmetic, and
a short list of functions and methods (`isin`, `between`, `str.contains`,
...). Anything else is refused with a message saying what was found and
what to write instead. The syntax is pandas' own for everything it covers,
so a query copied from a notebook usually works unchanged.

Values are compared as text
---------------------------
A "pick values" condition stores what it matched as text, and matches a
row by the text of its value. A CSV has no types: the same column reads as
integers in one file and as strings in the next (a single "n/a" is enough),
and a condition saved against `3` must still match `"3"` after that. The
query does the same where a comparison mixes a number with a text column:
the column is read as numbers, and a value that is not one does not match.
"""
from __future__ import annotations

import ast
import difflib
import operator
import re
from typing import Optional

import numpy as np
import pandas as pd

# ---- keys (stored in workspaces and snapshots: never reworded) -------------

CONDITIONS = "conditions"
QUERY = "query"
QUERY_ON = "query_on"

COLUMN = "column"
OP = "op"
VALUES = "values"
MIN = "min"
MAX = "max"
TEXT = "text"
ON = "on"

OP_IN = "in"
OP_NOT_IN = "not_in"
OP_RANGE = "range"
OP_CONTAINS = "contains"
OP_NOT_CONTAINS = "not_contains"
OP_REGEX = "regex"
OP_MISSING = "missing"
OP_PRESENT = "present"

OPS = (OP_IN, OP_NOT_IN, OP_RANGE, OP_CONTAINS, OP_NOT_CONTAINS, OP_REGEX, OP_MISSING, OP_PRESENT)
VALUE_OPS = (OP_IN, OP_NOT_IN)
TEXT_OPS = (OP_CONTAINS, OP_NOT_CONTAINS, OP_REGEX)

# A query longer than this is not something anyone typed; refusing it keeps
# a pasted file from being parsed and walked.
QUERY_MAX_CHARS = 4000
QUERY_MAX_NODES = 600
# Values a "pick values" condition may hold -- past this, a query with
# `isin` or `str.contains` is the right tool.
VALUES_MAX = 5000


class FilterError(ValueError):
    """A condition or a query that cannot be applied, with a message meant
    for the user."""


# ---- the spec --------------------------------------------------------------

def empty_spec() -> dict:
    return {CONDITIONS: [], QUERY: "", QUERY_ON: True}


def _number(value) -> Optional[float]:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def normalize_condition(raw) -> Optional[dict]:
    """One condition, cleaned of anything a hand-edited or older file may
    carry; None when it cannot be one."""
    if not isinstance(raw, dict):
        return None
    column = raw.get(COLUMN)
    op = raw.get(OP)
    if not isinstance(column, str) or not column or op not in OPS:
        return None
    cond = {COLUMN: column, OP: op, ON: bool(raw.get(ON, True))}
    if op in VALUE_OPS:
        values = raw.get(VALUES) or []
        if not isinstance(values, (list, tuple)):
            values = []
        seen, clean = set(), []
        for v in values[:VALUES_MAX]:
            text = "" if v is None else str(v)
            if text not in seen:
                seen.add(text)
                clean.append(text)
        cond[VALUES] = clean
    elif op == OP_RANGE:
        cond[MIN] = _number(raw.get(MIN))
        cond[MAX] = _number(raw.get(MAX))
    elif op in TEXT_OPS:
        text = raw.get(TEXT)
        cond[TEXT] = "" if text is None else str(text)
    return cond


def normalize_spec(raw) -> dict:
    """A filter as read back from a file: anything malformed is dropped
    rather than raised, the same rule as every other piece of view state."""
    spec = empty_spec()
    if not isinstance(raw, dict):
        return spec
    conds = raw.get(CONDITIONS) or []
    if isinstance(conds, list):
        spec[CONDITIONS] = [c for c in (normalize_condition(r) for r in conds) if c is not None]
    query = raw.get(QUERY)
    spec[QUERY] = query.strip() if isinstance(query, str) else ""
    spec[QUERY_ON] = bool(raw.get(QUERY_ON, True))
    return spec


def condition_is_active(cond: dict) -> bool:
    """A condition that is off, or has nothing to act on yet (no value
    picked, no bound typed), keeps every row and is not counted as a
    filter -- a half-built condition must not empty the dataset."""
    if not cond.get(ON, True):
        return False
    op = cond.get(OP)
    if op in VALUE_OPS:
        return bool(cond.get(VALUES))
    if op == OP_RANGE:
        return cond.get(MIN) is not None or cond.get(MAX) is not None
    if op in TEXT_OPS:
        return bool(cond.get(TEXT))
    return op in (OP_MISSING, OP_PRESENT)


def query_is_active(spec: dict) -> bool:
    return bool(spec.get(QUERY_ON, True) and (spec.get(QUERY) or "").strip())


def is_active(spec) -> bool:
    if not spec:
        return False
    return any(condition_is_active(c) for c in spec.get(CONDITIONS, [])) or query_is_active(spec)


def columns_used(spec: dict) -> list:
    """The columns the conditions name (the query's are found when it is
    parsed: see query_columns)."""
    out = []
    for c in spec.get(CONDITIONS, []):
        if condition_is_active(c) and c[COLUMN] not in out:
            out.append(c[COLUMN])
    return out


def _fmt_number(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def describe_condition(cond: dict) -> str:
    col, op = cond[COLUMN], cond[OP]
    if op in VALUE_OPS:
        values = cond.get(VALUES) or []
        shown = ", ".join(repr(v) for v in values[:4]) + (f", ... ({len(values)} values)" if len(values) > 4 else "")
        return f"{col} {'is' if op == OP_IN else 'is not'} {'one of ' if len(values) > 1 else ''}{shown}"
    if op == OP_RANGE:
        lo, hi = cond.get(MIN), cond.get(MAX)
        if lo is not None and hi is not None:
            return f"{_fmt_number(lo)} <= {col} <= {_fmt_number(hi)}"
        if lo is not None:
            return f"{col} >= {_fmt_number(lo)}"
        return f"{col} <= {_fmt_number(hi)}"
    if op == OP_CONTAINS:
        return f"{col} contains {cond.get(TEXT)!r}"
    if op == OP_NOT_CONTAINS:
        return f"{col} does not contain {cond.get(TEXT)!r}"
    if op == OP_REGEX:
        return f"{col} matches /{cond.get(TEXT)}/"
    if op == OP_MISSING:
        return f"{col} is empty"
    return f"{col} is not empty"


def describe(spec) -> list:
    """One line per active part of the filter, for the app bar's tooltip,
    reports and the session history."""
    if not spec:
        return []
    lines = [describe_condition(c) for c in spec.get(CONDITIONS, []) if condition_is_active(c)]
    if query_is_active(spec):
        lines.append(f"query: {spec[QUERY]}")
    return lines


# ---- applying it ------------------------------------------------------------

def _unknown_column(name: str, columns) -> FilterError:
    names = [str(c) for c in columns]
    close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return FilterError(f"No column '{name}' in this dataset.{hint}")


def _as_text(series: pd.Series) -> pd.Series:
    """The text a "pick values" condition matches against: what the value
    reads as, with whole floats written as integers -- a column that went
    from int to float because one row was empty must still match "3"."""
    if pd.api.types.is_float_dtype(series):
        values = series.to_numpy()
        whole = np.isfinite(values) & (np.mod(values, 1) == 0) & (np.abs(values) < 2 ** 53)
        out = series.astype(str)
        if whole.any():
            out = out.where(~whole, pd.Series(values, index=series.index).where(whole).fillna(0).astype(np.int64).astype(str))
        return out
    if pd.api.types.is_bool_dtype(series):
        return series.astype(str)
    return series.astype(str)


def value_text(value) -> str:
    """The text one value is matched by -- the scalar twin of _as_text."""
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer() \
            and abs(value) < 2 ** 53:
        return str(int(value))
    return str(value)


def condition_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
    """The rows one condition keeps. A condition that is not active keeps
    every row."""
    if not condition_is_active(cond):
        return pd.Series(True, index=df.index)
    col = cond[COLUMN]
    if col not in df.columns:
        raise _unknown_column(col, df.columns)
    series = df[col]
    op = cond[OP]
    present = series.notna()
    if op == OP_MISSING:
        return ~present | (series.astype(str).str.strip() == "") if series.dtype == object else ~present
    if op == OP_PRESENT:
        return present & (series.astype(str).str.strip() != "") if series.dtype == object else present
    if op in VALUE_OPS:
        hit = present & _as_text(series).isin(set(cond[VALUES]))
        return hit if op == OP_IN else ~hit
    if op == OP_RANGE:
        numbers = series if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) \
            else pd.to_numeric(series, errors="coerce")
        keep = numbers.notna()
        if cond.get(MIN) is not None:
            keep &= numbers >= cond[MIN]
        if cond.get(MAX) is not None:
            keep &= numbers <= cond[MAX]
        return keep.fillna(False).astype(bool)
    text = cond.get(TEXT) or ""
    strings = series.astype(str).where(present, "")
    if op == OP_REGEX:
        try:
            re.compile(text)
        except re.error as ex:
            raise FilterError(f"'{text}' is not a valid regular expression ({ex}).") from ex
        return strings.str.contains(text, regex=True, na=False).astype(bool)
    hit = strings.str.contains(text, case=False, regex=False, na=False).astype(bool)
    return hit if op == OP_CONTAINS else ~hit


def apply_spec(df: pd.DataFrame, spec) -> tuple:
    """(mask, steps): the rows the whole filter keeps, and for each active
    part of it, in order, (description, rows that part alone removes).

    Raises FilterError naming the first part that cannot be applied -- a
    filter is applied whole or not at all: dropping the part that failed
    would show a dataset filtered differently from what the tab says.
    """
    mask = pd.Series(True, index=df.index)
    steps = []
    if not spec:
        return mask, steps
    for cond in spec.get(CONDITIONS, []):
        if not condition_is_active(cond):
            continue
        part = condition_mask(df, cond)
        steps.append((describe_condition(cond), int((~part).sum())))
        mask &= part
    if query_is_active(spec):
        part = query_mask(df, spec[QUERY])
        steps.append((f"query: {spec[QUERY]}", int((~part).sum())))
        mask &= part
    return mask.astype(bool), steps


# ---- the query language ------------------------------------------------------

_BACKTICK = re.compile(r"`([^`]*)`")
_PLACEHOLDER = "__lk_col_{}__"


class QueryError(FilterError):
    pass


class _Str:
    """`column.str` -- only ever followed by one of _STR_METHODS."""

    def __init__(self, series: pd.Series):
        self.series = series


def _text(series: pd.Series) -> pd.Series:
    """A column as text, missing values staying missing (so they match
    nothing rather than the string "nan")."""
    return series.astype(str).where(series.notna())


def _need_series(value, what: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise QueryError(f"{what} needs a column, e.g. label.{what}(...) or {what}(label, ...).")
    return value


def _need_str(value, what: str) -> str:
    if not isinstance(value, str):
        raise QueryError(f"{what} needs a text argument in quotes, e.g. {what}('abc').")
    return value


def _regex(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as ex:
        raise QueryError(f"'{pattern}' is not a valid regular expression ({ex}).") from ex
    return pattern


def _str_contains(s, pat, case=True, regex=True):
    pat = _need_str(pat, "contains")
    if regex:
        _regex(pat)
    return _text(s).str.contains(pat, case=bool(case), regex=bool(regex), na=False)


def _str_method(name: str, s: pd.Series, args: list, kwargs: dict):
    t = _text(s)
    if name == "contains":
        return _str_contains(s, *args, **kwargs)
    if name in ("startswith", "endswith"):
        if len(args) != 1 or kwargs:
            raise QueryError(f"str.{name} takes one text argument.")
        arg = args[0]
        if isinstance(arg, list):
            arg = tuple(_need_str(a, name) for a in arg)
        else:
            arg = _need_str(arg, name)
        return getattr(t.str, name)(arg, na=False)
    if name in ("match", "fullmatch"):
        if len(args) != 1:
            raise QueryError(f"str.{name} takes one pattern.")
        case = bool(kwargs.pop("case", True))
        if kwargs:
            raise QueryError(f"str.{name} only takes case=...")
        return getattr(t.str, name)(_regex(_need_str(args[0], name)), case=case, na=False)
    if name in ("lower", "upper", "strip", "len"):
        if args or kwargs:
            raise QueryError(f"str.{name}() takes no argument.")
        return getattr(t.str, name)()
    raise QueryError(f"'.str.{name}' is not available. Available: contains, startswith, endswith, "
                     f"match, fullmatch, lower, upper, strip, len.")


_STR_METHODS = ("contains", "startswith", "endswith", "match", "fullmatch", "lower", "upper", "strip", "len")


def _series_method(name: str, s: pd.Series, args: list, kwargs: dict):
    if name in ("isna", "isnull", "notna", "notnull"):
        if args or kwargs:
            raise QueryError(f"{name}() takes no argument.")
        return s.isna() if name in ("isna", "isnull") else s.notna()
    if name == "isin":
        if len(args) != 1 or not isinstance(args[0], list) or kwargs:
            raise QueryError("isin takes one list, e.g. label.isin(['cat', 'dog']).")
        return _isin(s, args[0])
    if name == "between":
        if len(args) != 2 or kwargs:
            raise QueryError("between takes two bounds, e.g. width.between(32, 64).")
        lo, hi = args
        return _compare(ast.GtE(), s, lo) & _compare(ast.LtE(), s, hi)
    if name == "abs":
        return _numeric(s).abs()
    if name == "round":
        digits = args[0] if args else 0
        if not isinstance(digits, int):
            raise QueryError("round takes a whole number of digits.")
        return _numeric(s).round(digits)
    raise QueryError(f"'.{name}()' is not available. Available: isna, notna, isin, between, abs, round, "
                     f"and str.contains / startswith / endswith / match / lower / upper / len.")


_SERIES_METHODS = ("isna", "isnull", "notna", "notnull", "isin", "between", "abs", "round")


def _numeric(value):
    if isinstance(value, pd.Series):
        if pd.api.types.is_bool_dtype(value):
            return value.astype(np.int64)
        if pd.api.types.is_numeric_dtype(value):
            return value
        return pd.to_numeric(value, errors="coerce")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.number)):
        return value
    raise QueryError(f"{value!r} is not a number.")


def _isin(s: pd.Series, values: list) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numbers = [_number(v) for v in values]
        if all(n is not None for n in numbers):
            return s.isin(numbers)
    return s.notna() & _as_text(s).isin({value_text(v) for v in values})


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_CMP_TEXT = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def _compare(op, left, right):
    if isinstance(op, (ast.In, ast.NotIn)):
        if isinstance(right, list):
            if isinstance(left, pd.Series):
                hit = _isin(left, right)
            else:
                hit = value_text(left) in {value_text(v) for v in right}
        elif isinstance(left, str) and isinstance(right, pd.Series):
            # "dog" in path: a substring, as Python reads it for text.
            hit = _text(right).str.contains(left, regex=False, na=False)
        else:
            raise QueryError("'in' needs a list on its right, e.g. label in ['cat', 'dog'], "
                             "or a text on its left, e.g. 'blurry' in path.")
        return ~hit if isinstance(op, ast.NotIn) else hit
    if isinstance(op, (ast.Is, ast.IsNot)):
        if right is not None:
            raise QueryError("'is' only compares with None: write 'x is None' or 'x is not None', and '==' otherwise.")
        missing = left.isna() if isinstance(left, pd.Series) else left is None
        return ~missing if isinstance(op, ast.IsNot) else missing
    fn = _CMP.get(type(op))
    if fn is None:
        raise QueryError("Unsupported comparison.")
    left, right = _align_types(left, right, type(op))
    try:
        return fn(left, right)
    except TypeError as ex:
        raise QueryError(f"Cannot compare these with {_CMP_TEXT[type(op)]}: {ex}") from ex


def _align_types(left, right, op_type):
    """A number against a text column reads the column as numbers; a text
    against a numeric column compares the column's text -- see the module
    docstring."""
    for a, b, swap in ((left, right, False), (right, left, True)):
        if not isinstance(a, pd.Series) or isinstance(b, pd.Series):
            continue
        numeric_col = pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a)
        if _is_number(b) and not numeric_col and not pd.api.types.is_bool_dtype(a):
            a = pd.to_numeric(a, errors="coerce")
        elif isinstance(b, str) and numeric_col:
            n = _number(b)
            if n is not None:
                b = n
            elif op_type in (ast.Eq, ast.NotEq):
                a = _as_text(a)
            else:
                raise QueryError(f"A numeric column is compared with the text {b!r}.")
        elif isinstance(b, str) and pd.api.types.is_bool_dtype(a):
            a = a.astype(str)
        return (b, a) if swap else (a, b)
    return left, right


def _logical(value, what: str):
    if isinstance(value, pd.Series):
        if pd.api.types.is_bool_dtype(value):
            return value
        if value.dtype == object and value.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            return value.fillna(False).astype(bool)
        raise QueryError(f"'{what}' combines conditions (true or false for each row), and one side is a "
                         f"column of values: compare it first, e.g. width > 10.")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise QueryError(f"'{what}' combines conditions, and one side is {value!r}.")


def _arith(op, left, right):
    if isinstance(op, ast.Add) and isinstance(left, str) and isinstance(right, str):
        return left + right
    if isinstance(op, ast.Add) and (isinstance(left, str) or isinstance(right, str)):
        a = _text(left) if isinstance(left, pd.Series) else left
        b = _text(right) if isinstance(right, pd.Series) else right
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            return a + b
    a, b = _numeric(left), _numeric(right)
    if isinstance(op, ast.Pow) and _is_number(b) and abs(b) > 64:
        raise QueryError("Exponents above 64 are not accepted.")
    fns = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
    fn = fns.get(type(op))
    if fn is None:
        raise QueryError("Unsupported operator.")
    if not isinstance(a, pd.Series) and not isinstance(b, pd.Series):
        a, b = float(a), float(b)
    with np.errstate(all="ignore"):
        try:
            return fn(a, b)
        except ZeroDivisionError:
            return float("nan")


def _func_contains(x, pat, case=False):
    return _str_contains(_need_series(x, "contains"), pat, case=case, regex=False)


def _func_matches(x, pat, case=True):
    return _str_contains(_need_series(x, "matches"), pat, case=case, regex=True)


def _func_between(x, lo, hi):
    return _series_method("between", _need_series(x, "between"), [lo, hi], {})


# Top-level functions, for those who prefer them to methods (and for the
# FiftyOne habit of F("column")).
_FUNCTIONS = {
    "abs": lambda x: _numeric(x).abs() if isinstance(x, pd.Series) else abs(_numeric(x)),
    "len": lambda x: _text(x).str.len() if isinstance(x, pd.Series) else len(str(x)),
    "lower": lambda x: _text(_need_series(x, "lower")).str.lower(),
    "upper": lambda x: _text(_need_series(x, "upper")).str.upper(),
    "num": lambda x: _numeric(_need_series(x, "num")),
    "isna": lambda x: _need_series(x, "isna").isna(),
    "notna": lambda x: _need_series(x, "notna").notna(),
    "contains": _func_contains,
    "matches": _func_matches,
    "startswith": lambda x, s: _str_method("startswith", _need_series(x, "startswith"), [s], {}),
    "endswith": lambda x, s: _str_method("endswith", _need_series(x, "endswith"), [s], {}),
    "between": _func_between,
}
_COLUMN_FUNCTIONS = ("col", "F")


class _Evaluator:
    def __init__(self, df: pd.DataFrame, names: dict):
        self.df = df
        self.names = names          # placeholder -> real column name
        self.used: list = []

    def column(self, name: str) -> pd.Series:
        if name not in self.df.columns:
            raise _query_unknown(name, self.df.columns)
        if name not in self.used:
            self.used.append(name)
        return self.df[name]

    def eval(self, node):
        method = getattr(self, f"_{type(node).__name__}", None)
        if method is None:
            raise QueryError(_unsupported(node))
        return method(node)

    def _Expression(self, node):
        return self.eval(node.body)

    def _Constant(self, node):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise QueryError(_unsupported(node))

    def _Name(self, node):
        name = self.names.get(node.id, node.id)
        if name in ("nan", "NaN"):
            return float("nan")
        return self.column(name)

    def _List(self, node):
        out = []
        for elt in node.elts:
            value = self.eval(elt)
            if isinstance(value, pd.Series):
                raise QueryError("A list holds values, not columns: e.g. ['cat', 'dog'].")
            out.append(value)
        return out

    _Tuple = _List
    _Set = _List

    def _BoolOp(self, node):
        word = "and" if isinstance(node.op, ast.And) else "or"
        values = [_logical(self.eval(v), word) for v in node.values]
        out = values[0]
        for v in values[1:]:
            out = (out & v) if word == "and" else (out | v)
        return out

    def _UnaryOp(self, node):
        value = self.eval(node.operand)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            value = _logical(value, "not")
            return ~value if isinstance(value, pd.Series) else not value
        value = _numeric(value)
        return -value if isinstance(node.op, ast.USub) else value

    def _BinOp(self, node):
        left, right = self.eval(node.left), self.eval(node.right)
        if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            word = "&" if isinstance(node.op, ast.BitAnd) else "|"
            a, b = _logical(left, word), _logical(right, word)
            return (a & b) if word == "&" else (a | b)
        return _arith(node.op, left, right)

    def _Compare(self, node):
        left = self.eval(node.left)
        out = None
        for op, right_node in zip(node.ops, node.comparators):
            right = self.eval(right_node)
            part = _compare(op, left, right)
            out = part if out is None else (out & part)
            left = right
        return out

    def _Attribute(self, node):
        if node.attr == "str":
            return _Str(_need_series(self.eval(node.value), ".str"))
        raise QueryError(f"'.{node.attr}' is not available here. Column names with dots or spaces go "
                         f"between backticks: `{_source_name(node)}`.")

    def _Call(self, node):
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None or kw.arg not in ("case", "regex"):
                raise QueryError(f"Unknown argument '{kw.arg}': only case=... and regex=... are accepted.")
            kwargs[kw.arg] = self.eval(kw.value)
            if not isinstance(kwargs[kw.arg], bool):
                raise QueryError(f"{kw.arg}= takes True or False.")
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in _COLUMN_FUNCTIONS:
                if len(node.args) != 1 or kwargs:
                    raise QueryError(f"{name}() takes one column name, e.g. {name}('my column').")
                arg = self.eval(node.args[0])
                if not isinstance(arg, str):
                    raise QueryError(f"{name}() takes a column name in quotes, e.g. {name}('my column').")
                return self.column(arg)
            fn = _FUNCTIONS.get(name)
            if fn is None:
                raise QueryError(f"Unknown function '{name}'. Available: col, "
                                 f"{', '.join(sorted(_FUNCTIONS))}.")
            args = [self.eval(a) for a in node.args]
            try:
                return fn(*args, **kwargs)
            except TypeError as ex:
                raise QueryError(f"{name}(): {ex}") from ex
        if isinstance(func, ast.Attribute):
            owner = self.eval(func.value) if not (isinstance(func.value, ast.Attribute) and func.value.attr == "str") \
                else self._Attribute(func.value)
            args = [self.eval(a) for a in node.args]
            if isinstance(owner, _Str):
                if func.attr not in _STR_METHODS:
                    raise QueryError(f"'.str.{func.attr}' is not available. Available: {', '.join(_STR_METHODS)}.")
                try:
                    return _str_method(func.attr, owner.series, args, dict(kwargs))
                except TypeError as ex:
                    raise QueryError(f"str.{func.attr}(): {ex}") from ex
            if isinstance(owner, pd.Series):
                if func.attr not in _SERIES_METHODS:
                    raise QueryError(f"'.{func.attr}()' is not available. Available: {', '.join(_SERIES_METHODS)}, "
                                     f"and .str.<{'|'.join(_STR_METHODS)}>.")
                return _series_method(func.attr, owner, args, dict(kwargs))
            raise QueryError(f"'.{func.attr}()' needs a column on its left.")
        raise QueryError("Only a function or a column method can be called here.")


def _source_name(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _unsupported(node) -> str:
    kind = type(node).__name__
    friendly = {
        "Lambda": "a lambda", "IfExp": "an if/else expression", "Subscript": "indexing with [...]",
        "Dict": "a dictionary", "ListComp": "a comprehension", "GeneratorExp": "a comprehension",
        "SetComp": "a comprehension", "DictComp": "a comprehension", "NamedExpr": "':='",
        "Await": "await", "Starred": "'*'", "JoinedStr": "an f-string",
    }.get(kind, kind)
    return f"{friendly} is not accepted in a filter query."


def _query_unknown(name: str, columns) -> QueryError:
    message = str(_unknown_column(name, columns))
    if "Did you mean" not in message:
        names = [str(c) for c in columns]
        shown = ", ".join(quote_column(c) for c in names[:12]) + (", ..." if len(names) > 12 else "")
        message += f" Columns: {shown}."
    return QueryError(message)


def _prepare(query: str) -> tuple:
    """(source, names): backticked column names replaced by identifiers."""
    names = {}

    def sub(match):
        key = _PLACEHOLDER.format(len(names))
        names[key] = match.group(1)
        return key

    return _BACKTICK.sub(sub, query), names


def _syntax_hint(query: str) -> str:
    if re.search(r"(?<![=!<>])=(?!=)", query):
        return " To test equality write '==' (one '=' assigns)."
    if "&&" in query or "||" in query:
        return " Write 'and' / 'or' (or '&' / '|')."
    if query.count("'") % 2 or query.count('"') % 2:
        return " A quote is not closed."
    if query.count("(") != query.count(")"):
        return " The parentheses do not match."
    return ""


def parse_query(query: str) -> tuple:
    """(tree, names), or QueryError. Parsing only: cheap enough to run on
    every keystroke."""
    query = (query or "").strip()
    if not query:
        raise QueryError("The query is empty.")
    if len(query) > QUERY_MAX_CHARS:
        raise QueryError(f"The query is longer than {QUERY_MAX_CHARS:,} characters.")
    source, names = _prepare(query.replace("\n", " "))
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as ex:
        where = f" (at character {ex.offset})" if ex.offset else ""
        raise QueryError(f"This is not a valid expression{where}.{_syntax_hint(query)}") from None
    if sum(1 for _ in ast.walk(tree)) > QUERY_MAX_NODES:
        raise QueryError("The query is too long to be applied.")
    return tree, names


def query_mask(df: pd.DataFrame, query: str) -> pd.Series:
    """The rows a query keeps. Raises QueryError with a message for the user."""
    tree, names = parse_query(query)
    evaluator = _Evaluator(df, names)
    try:
        result = evaluator.eval(tree)
    except QueryError:
        raise
    except (ValueError, TypeError, re.error) as ex:
        raise QueryError(f"The query could not be applied: {ex}") from ex
    if isinstance(result, (bool, np.bool_)):
        return pd.Series(bool(result), index=df.index)
    if not isinstance(result, pd.Series):
        raise QueryError("The query must be a condition, true or false for each row -- e.g. width > 32.")
    if not pd.api.types.is_bool_dtype(result):
        try:
            result = _logical(result, "query")
        except QueryError:
            raise QueryError("The query must be a condition, true or false for each row -- it gives a "
                             "column of values. Compare it with something, e.g. width > 32.") from None
    return result.fillna(False).astype(bool)


def query_columns(df: pd.DataFrame, query: str) -> list:
    """The columns a query reads (empty when it does not apply)."""
    try:
        tree, names = parse_query(query)
        evaluator = _Evaluator(df, names)
        evaluator.eval(tree)
    except FilterError:
        return []
    return evaluator.used


def quote_column(name: str) -> str:
    """How a column is written in a query: bare when it can be, between
    backticks otherwise."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name not in _FUNCTIONS \
            and name not in _COLUMN_FUNCTIONS and name not in ("and", "or", "not", "in", "is", "None",
                                                                 "True", "False", "nan", "NaN"):
        return name
    return f"`{name}`"


def query_from_conditions(spec: dict) -> str:
    """The active conditions written as one query -- what "Convert to
    query" puts in the query field, so a filter built by clicking can be
    taken further by hand."""
    parts = []
    for cond in spec.get(CONDITIONS, []):
        if not condition_is_active(cond):
            continue
        col, op = quote_column(cond[COLUMN]), cond[OP]
        if op in VALUE_OPS:
            values = ", ".join(repr(v) for v in cond[VALUES])
            parts.append(f"{col} {'in' if op == OP_IN else 'not in'} [{values}]")
        elif op == OP_RANGE:
            lo, hi = cond.get(MIN), cond.get(MAX)
            if lo is not None and hi is not None:
                parts.append(f"{col}.between({_fmt_number(lo)}, {_fmt_number(hi)})")
            elif lo is not None:
                parts.append(f"{col} >= {_fmt_number(lo)}")
            else:
                parts.append(f"{col} <= {_fmt_number(hi)}")
        elif op == OP_CONTAINS:
            parts.append(f"contains({col}, {cond[TEXT]!r})")
        elif op == OP_NOT_CONTAINS:
            parts.append(f"not contains({col}, {cond[TEXT]!r})")
        elif op == OP_REGEX:
            parts.append(f"{col}.str.contains({cond[TEXT]!r})")
        elif op == OP_MISSING:
            parts.append(f"{col}.isna()")
        else:
            parts.append(f"{col}.notna()")
    if query_is_active(spec):
        parts.append(f"({spec[QUERY]})" if parts else spec[QUERY])
    return " and ".join(parts)
