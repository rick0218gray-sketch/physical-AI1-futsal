"""Step 4: head-to-head evaluation. Plays N parallel matches per pairing, each pairing on both sides.

usage: python eval_matches.py --ckpt logs/go2-soccer/model_1398.pt --n 128 --match-s 60 \
         --pair "rl=logs/futsal-rl/policy_600.pt:teacher=teacher" --pair "bc=logs/bc/bc_policy.pt:teacher=teacher" ...
Writes results/eval_<name>.json with wins/draws/losses, goals, falls.
"""
import argparse
import json
import os
import time

import torch

import genesis as gs

from collect_bc import load_tactics, sample_tactics
from futsal_env import FutsalSim
from players import Team, match_commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--match-s", type=float, default=60)
    ap.add_argument("--pair", action="append", required=True, help="nameA=kindA:nameB=kindB")
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=7, performance_mode=True)
    sim = FutsalSim(a.n, a.ckpt, match_s=a.match_s)
    dev = sim.dev
    tv = load_tactics("tactics.json", dev)
    results = []
    for spec in a.pair:
        (na, ka), (nb, kb) = [s.split("=", 1) for s in spec.split(":")]
        A, B = Team(ka, dev, na), Team(kb, dev, nb)
        tot = dict(a=na, b=nb, matches=0, a_wins=0, draws=0, b_wins=0, a_goals=0, b_goals=0, a_falls=0, b_falls=0)
        t0 = time.time()
        for side in (0, 1):  # A plays blue (side 0) then red (side 1)
            teams = [A, B] if side == 0 else [B, A]
            torch.manual_seed(100 + side)
            sim.reset_envs(torch.arange(sim.n, device=dev))
            tactics = sample_tactics(tv, sim.n, dev)
            for _ in range(sim.max_hl_steps):
                sim.hl_step(match_commands(sim, teams, tactics))
            ga, gb = (sim.score[:, 0], sim.score[:, 1]) if side == 0 else (sim.score[:, 1], sim.score[:, 0])
            fa = sim.falls[:, :2].sum(1) if side == 0 else sim.falls[:, 2:].sum(1)
            fb = sim.falls[:, 2:].sum(1) if side == 0 else sim.falls[:, :2].sum(1)
            tot["matches"] += sim.n
            tot["a_wins"] += int((ga > gb).sum())
            tot["draws"] += int((ga == gb).sum())
            tot["b_wins"] += int((ga < gb).sum())
            tot["a_goals"] += int(ga.sum())
            tot["b_goals"] += int(gb.sum())
            tot["a_falls"] += int(fa.sum())
            tot["b_falls"] += int(fb.sum())
        m = tot["matches"]
        tot["a_points_rate"] = round((tot["a_wins"] + 0.5 * tot["draws"]) / m, 3)
        tot["a_goals_per_match"] = round(tot["a_goals"] / m, 3)
        tot["b_goals_per_match"] = round(tot["b_goals"] / m, 3)
        tot["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(tot, ensure_ascii=False), flush=True)
        results.append(tot)
    json.dump(dict(match_s=a.match_s, n_per_side=a.n, results=results), open(a.out, "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
