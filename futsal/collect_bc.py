"""Step 1+2: teacher matches under coach (LLM) tactics -> imitation learning of the high-level player policy.

Teacher = rule-based tactics conditioned on a tactic vector produced by the coach LLM (tactics.json).
During collection a fraction of robots execute noisy teacher commands (DART-style) so the dataset also covers
recovery situations; the label is always the clean teacher command.

usage (inside futsal/, needs Genesis + a GPU):
  python collect_bc.py --ckpt ../models/go2-soccer/model_1398.pt --n 256 --seconds 240 [--out-dir logs/bc]
  (then trains the BC policy and writes <out-dir>/bc_policy.pt and <out-dir>/bc_stats.json)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

import genesis as gs

from futsal_env import OBS_DIM, FutsalSim
from futsal_teacher import teacher_commands

HIDDEN = (256, 128)


def make_mlp(dev):
    layers, d = [], OBS_DIM
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, 3))
    return nn.Sequential(*layers).to(dev)


def load_tactics(path, dev):
    tl = json.load(open(path, encoding="utf-8"))
    return torch.tensor([[t["attack"], t["keeper_depth"], t["press"]] for t in tl], device=dev)


def sample_tactics(tv, n, dev, jitter=0.1):
    t = tv[torch.randint(len(tv), (n, 2), device=dev)]
    return (t + (torch.rand_like(t) * 2 - 1) * jitter).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=240, help="simulated seconds per env")
    ap.add_argument("--match-s", type=float, default=60)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-dir", default="logs/bc")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1, performance_mode=True)
    sim = FutsalSim(a.n, a.ckpt, match_s=a.match_s)
    dev = sim.dev
    tv = load_tactics("tactics.json", dev)
    tactics = sample_tactics(tv, a.n, dev)
    steps = int(a.seconds / (sim.dt * sim.decimation))
    X, Y = [], []
    goals = torch.zeros(2, device=dev)
    falls = torch.zeros(4, device=dev)
    matches = 0
    t0 = time.time()
    for k in range(steps):
        obs = sim.observations(tactics)
        lab = teacher_commands(sim, tactics)
        X.append(obs.reshape(-1, OBS_DIM).cpu())
        Y.append(lab.reshape(-1, 3).cpu())
        noisy = torch.rand((a.n, 4, 1), device=dev) < 0.3
        act = torch.where(noisy, lab + torch.randn_like(lab) * a.noise, lab)
        ga, gb, fell, done = sim.hl_step(act)
        goals += torch.stack((ga.sum(), gb.sum())).float()
        falls += fell.float().sum(0)
        if done.any():
            idx = done.nonzero().flatten()
            matches += idx.numel()
            sim.reset_envs(idx)
            tactics[idx] = sample_tactics(tv, idx.numel(), dev)
        if k % 200 == 0:
            print(f"step {k}/{steps}  wall {time.time() - t0:.0f}s  goals {goals.tolist()}  falls {falls.tolist()}",
                  flush=True)
    wall = time.time() - t0
    X, Y = torch.cat(X), torch.cat(Y)
    sim_s = a.n * a.seconds
    stats = dict(samples=len(X), envs=a.n, sim_seconds=sim_s, wall_s=round(wall, 1), matches_finished=matches,
                 goals_A=goals[0].item(), goals_B=goals[1].item(), goals_per_min=(goals.sum() / (sim_s / 60)).item(),
                 falls_per_robot_min=(falls / (sim_s / 60)).tolist())
    print(json.dumps(stats), flush=True)

    # ---- behaviour cloning
    net = make_mlp(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xd, Yd = X.to(dev), Y.to(dev)
    n_val = len(Xd) // 20
    perm = torch.randperm(len(Xd), device=dev)
    val, tr = perm[:n_val], perm[n_val:]
    hist = []
    for ep in range(a.epochs):
        net.train()
        p = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(p), 4096):
            b = p[i:i + 4096]
            loss = ((net(Xd[b]) - Yd[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xd[val]) - Yd[val]) ** 2).mean().item()
        hist.append(vl)
        print(f"bc epoch {ep} val_mse {vl:.4f}", flush=True)
    sd = {f"mlp.{k}": v.cpu() for k, v in net.state_dict().items()}
    torch.save({"actor_mlp": sd, "hidden": HIDDEN, "stats": stats, "val_mse": hist}, f"{a.out_dir}/bc_policy.pt")
    json.dump(dict(stats, val_mse=hist), open(f"{a.out_dir}/bc_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
