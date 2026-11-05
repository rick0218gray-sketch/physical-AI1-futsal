"""Step 4: head-to-head evaluation. Plays N parallel matches per pairing, each pairing on both sides.

usage (run inside futsal/, needs Genesis + a GPU):
  python eval_matches.py --ckpt ../models/go2-soccer/model_1398.pt --n 128 --match-s 60 \
         --pair "rl=../models/futsal-rl/policy_500.pt:teacher=teacher" \
         --pair "bc=../models/bc/bc_policy.pt:teacher=teacher" ...
Writes --out (default: results/eval.json, relative to the current directory) with wins/draws/losses, goals and falls
per pairing. The bookkeeping itself lives in match_stats.py.
"""
import argparse
import json
import os
import time

import torch

import genesis as gs

from collect_bc import load_tactics, sample_tactics
from futsal_env import FutsalSim
from match_stats import add_matches, finalize, new_tally, side_view
from players import Team, match_commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--match-s", type=float, default=60)
    ap.add_argument("--pair", action="append", required=True, help="nameA=kindA:nameB=kindB")
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=7, performance_mode=True)
    sim = FutsalSim(a.n, a.ckpt, match_s=a.match_s)
    dev = sim.dev
    tv = load_tactics("tactics.json", dev)
    results = []
    for spec in a.pair:
        (na, ka), (nb, kb) = [s.split("=", 1) for s in spec.split(":")]
        A, B = Team(ka, dev, na), Team(kb, dev, nb)
        tot = new_tally(na, nb)
        t0 = time.time()
        for side in (0, 1):  # A plays blue (side 0) then red (side 1)
            teams = [A, B] if side == 0 else [B, A]
            torch.manual_seed(100 + side)
            sim.reset_envs(torch.arange(sim.n, device=dev))
            tactics = sample_tactics(tv, sim.n, dev)
            for _ in range(sim.max_hl_steps):
                sim.hl_step(match_commands(sim, teams, tactics))
            add_matches(tot, *side_view(sim.score, sim.falls, side))
        finalize(tot)
        tot["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(tot, ensure_ascii=False), flush=True)
        results.append(tot)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(match_s=a.match_s, n_per_side=a.n, results=results), f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
