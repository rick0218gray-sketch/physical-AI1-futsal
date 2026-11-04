"""Win / draw / loss bookkeeping used by eval_matches.py.

Pure Python: works on torch tensors or numpy arrays and needs neither the simulator nor a GPU,
so it can be unit-tested anywhere (tests/test_match_stats.py).

Conventions of FutsalSim
  score (n, 2) : goals of [blue = team 0 (attacks +x), red = team 1]
  falls (n, 4) : per robot; robots 0, 1 are blue (keeper, player), robots 2, 3 are red
"""


def new_tally(name_a, name_b):
    return dict(a=name_a, b=name_b, matches=0, a_wins=0, draws=0, b_wins=0, a_goals=0, b_goals=0, a_falls=0,
                b_falls=0)


def side_view(score, falls, side):
    """Goals and falls of team A and team B when A played blue (side 0) or red (side 1).

    Returns (goals_a, goals_b, falls_a, falls_b), each with one entry per match.
    """
    if side == 0:
        return score[:, 0], score[:, 1], falls[:, :2].sum(1), falls[:, 2:].sum(1)
    if side == 1:
        return score[:, 1], score[:, 0], falls[:, 2:].sum(1), falls[:, :2].sum(1)
    raise ValueError(f"side must be 0 (A is blue) or 1 (A is red), got {side!r}")


def add_matches(tot, ga, gb, fa, fb):
    """Accumulate one batch of finished matches (per-match goals and falls of A and B) into `tot`."""
    tot["matches"] += len(ga)
    tot["a_wins"] += int((ga > gb).sum())
    tot["draws"] += int((ga == gb).sum())
    tot["b_wins"] += int((ga < gb).sum())
    tot["a_goals"] += int(ga.sum())
    tot["b_goals"] += int(gb.sum())
    tot["a_falls"] += int(fa.sum())
    tot["b_falls"] += int(fb.sum())
    return tot


def finalize(tot):
    """Add the derived rates (a draw counts as half a point)."""
    m = tot["matches"]
    tot["a_points_rate"] = round((tot["a_wins"] + 0.5 * tot["draws"]) / m, 3)
    tot["a_goals_per_match"] = round(tot["a_goals"] / m, 3)
    tot["b_goals_per_match"] = round(tot["b_goals"] / m, 3)
    return tot
