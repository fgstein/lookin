"""The Metrics tab says each kind of warning once, whatever the number of
experiments it applies to."""
from __future__ import annotations

from types import SimpleNamespace

from lookin.web.views.common import grouped_warnings, overlap_warnings


def _overlap(dataset_only=0, dupes=0):
    return SimpleNamespace(dataset_only=dataset_only, experiment_only=0, dataset_dupes=dupes, experiment_dupes=0)


def test_the_same_warning_on_several_experiments_is_one_line():
    by = {"a": overlap_warnings(_overlap(50, 3)), "b": overlap_warnings(_overlap(50, 3)),
          "c": overlap_warnings(_overlap(7, 3))}
    folded = grouped_warnings(by, False, "")
    assert folded.count("Dataset rows with no matching prediction") == 1
    assert "all 3 experiments" in folded           # the duplicates: in every one
    assert "50 rows" not in folded                 # details only on request
    shown = grouped_warnings(by, True, "")
    assert "a, b" in shown and "50 rows" in shown and "7 rows" in shown


def test_no_warning_no_banner():
    assert grouped_warnings({"a": overlap_warnings(_overlap())}, False, "") == ""
