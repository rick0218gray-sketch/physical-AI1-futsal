"""Play one match headless and store the full state per high-level step (for rendering with render_replay.py).

usage: python record_match.py --ckpt logs/go2-soccer/model_1398.pt --blue rl=logs/futsal-rl/policy_600.pt \
          --red teacher=teacher --coach-blue coach_blue.json --coach-red coach_red.json --seconds 60 --out rec/m1.pt
coach_*.json: list of {"t": seconds, "instruction": ..., "attack": .., "keeper_depth": .., "press": ..}
"""
import argparse
import json
import os

import torch

import genesis as gs

from futsal_env import FutsalSim
from players import Team, match_commands


def tactic_at(plan, t):
    cur = plan[0]
    for p in plan:
        if p["t"] <= t:
            cur = p
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--blue", required=True)
    ap.add_argument("--red", required=True)
    ap.add_argument("--coach-blue")
    ap.add_argument("--coach-red")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=a.seed, performance_mode=True)
    torch.manual_seed(a.seed)
    sim = FutsalSim(1, a.ckpt, match_s=a.seconds + 1)
    dev = sim.dev
    (nb, kb), (nr, kr) = a.blue.split("=", 1), a.red.split("=", 1)
    teams = [Team(kb, dev, nb), Team(kr, dev, nr)]
    default = [{"t": 0, "instruction": "", "attack": 0.5, "keeper_depth": 0.5, "press": 0.5}]
    plans = [json.load(open(p, encoding="utf-8")) if p else default for p in (a.coach_blue, a.coach_red)]
    frames = []
    steps = int(a.seconds / (sim.dt * sim.decimation))
    for k in range(steps):
        t = k * sim.dt * sim.decimation
        cur = [tactic_at(p, t) for p in plans]
        tac = torch.tensor([[[c["attack"], c["keeper_depth"], c["press"]] for c in cur]], device=dev)
        ga, gb, fell, _ = sim.hl_step(match_commands(sim, teams, tac))
        frames.append(dict(
            t=t + sim.dt * sim.decimation,
            robots=torch.stack([r.get_qpos()[0] for r in sim.robots]).cpu(),
            ball=sim.ball.get_qpos()[0].cpu(),
            score=sim.score[0].tolist(), goal=[bool(ga[0]), bool(gb[0])], fell=fell[0].tolist(),
            coach=[c["instruction"] for c in cur],
        ))
    torch.save(dict(frames=frames, names=[nb, nr], dt=sim.dt * sim.decimation), a.out)
    print("final score", sim.score[0].tolist(), "falls", sim.falls[0].tolist(), "->", a.out)


if __name__ == "__main__":
    main()
