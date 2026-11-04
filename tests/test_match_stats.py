"""Win / draw / loss bookkeeping of eval_matches.py (match_stats.py). Pure Python, runs anywhere."""
import json
import os

import pytest

from match_stats import add_matches, finalize, new_tally, side_view

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Arr(list):
    """Tiny stand-in for a 1-D tensor / array: elementwise comparisons and .sum()."""

    def _cmp(self, other, op):
        return Arr(int(op(x, y)) for x, y in zip(self, other))

    def __gt__(self, o):
        return self._cmp(o, lambda x, y: x > y)

    def __lt__(self, o):
        return self._cmp(o, lambda x, y: x < y)

    def __eq__(self, o):  # noqa: D105 - elementwise like numpy / torch
        return self._cmp(o, lambda x, y: x == y)

    __hash__ = None

    def sum(self):
        return sum(self)


class Arr2(list):
    """Tiny stand-in for a 2-D tensor: [:, j] column, [:, a:b] column block, .sum(1) row sums."""

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            assert rows == slice(None)
            if isinstance(cols, slice):
                return Arr2([r[cols] for r in self])
            return Arr(r[cols] for r in self)
        return list.__getitem__(self, key)

    def sum(self, dim):
        assert dim == 1
        return Arr(sum(r) for r in self)


def test_counts_wins_draws_losses_goals_falls():
    tot = new_tally("x", "y")
    add_matches(tot, Arr([2, 0, 1, 0]), Arr([0, 0, 3, 1]), Arr([1, 0, 2, 0]), Arr([0, 4, 0, 1]))
    assert (tot["matches"], tot["a_wins"], tot["draws"], tot["b_wins"]) == (4, 1, 1, 2)
    assert (tot["a_goals"], tot["b_goals"], tot["a_falls"], tot["b_falls"]) == (3, 4, 3, 5)


def test_side_view_swaps_teams_when_a_plays_red():
    # two matches; score = [blue, red]; falls per robot 0..3 (0, 1 blue / 2, 3 red)
    score = Arr2([[3, 1], [0, 2]])
    falls = Arr2([[1, 2, 10, 20], [0, 1, 30, 0]])
    ga, gb, fa, fb = side_view(score, falls, 0)
    assert (list(ga), list(gb), list(fa), list(fb)) == ([3, 0], [1, 2], [3, 1], [30, 30])
    ga, gb, fa, fb = side_view(score, falls, 1)
    assert (list(ga), list(gb), list(fa), list(fb)) == ([1, 2], [3, 0], [30, 30], [3, 1])
    with pytest.raises(ValueError):
        side_view(score, falls, 2)


def test_both_sides_of_a_symmetric_pairing_cancel():
    """Same raw scores seen from both sides: wins and losses swap, totals are symmetric."""
    score = Arr2([[1, 0], [0, 0], [0, 2]])
    falls = Arr2([[1, 0, 0, 1], [0, 0, 0, 0], [2, 0, 0, 0]])
    tot = new_tally("t", "t")
    for side in (0, 1):
        add_matches(tot, *side_view(score, falls, side))
    finalize(tot)
    assert tot["matches"] == 6
    assert tot["a_wins"] == tot["b_wins"] == 2 and tot["draws"] == 2
    assert tot["a_goals"] == tot["b_goals"] == 3
    assert tot["a_falls"] == tot["b_falls"] == 4
    assert tot["a_points_rate"] == 0.5


def test_finalize_reproduces_derived_fields_of_stored_eval_json():
    """Recomputing the rates from the stored counts gives exactly the stored values (results/eval.json)."""
    with open(os.path.join(ROOT, "results", "eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    keys = ["a", "b", "matches", "a_wins", "draws", "b_wins", "a_goals", "b_goals", "a_falls", "b_falls"]
    for r in ev["results"]:
        tot = finalize({k: r[k] for k in keys})
        for k in ("a_points_rate", "a_goals_per_match", "b_goals_per_match"):
            assert tot[k] == r[k], (r["a"], r["b"], k)
        # the key order written by eval_matches.py is unchanged
        assert list(tot) + ["wall_s"] == list(r)


def test_finalize_points_rate_counts_draw_as_half():
    tot = finalize(dict(new_tally("a", "b"), matches=4, a_wins=1, draws=2, b_wins=1, a_goals=3, b_goals=1))
    assert tot["a_points_rate"] == 0.5
    assert tot["a_goals_per_match"] == 0.75
    assert tot["b_goals_per_match"] == 0.25


def test_works_with_numpy_arrays():
    numpy = pytest.importorskip("numpy")
    score = numpy.array([[2, 1], [1, 1], [0, 1]])
    falls = numpy.array([[1, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])
    tot = new_tally("a", "b")
    add_matches(tot, *side_view(score, falls, 1))
    assert (tot["a_wins"], tot["draws"], tot["b_wins"]) == (1, 1, 1)
    assert (tot["a_goals"], tot["b_goals"], tot["a_falls"], tot["b_falls"]) == (3, 3, 2, 2)
    assert all(type(tot[k]) is int for k in ("a_wins", "a_goals", "a_falls"))
