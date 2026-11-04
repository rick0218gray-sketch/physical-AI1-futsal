"""Step 3: players-only reinforcement learning (PPO) of the high-level policy, initialised from imitation learning.

The learning team (A: keeper + player) shares one policy (role is part of the observation).
Opponent team (B) per match: a frozen snapshot of the learner (self-play) or the teacher (rule + coach tactics),
refreshed every `--snapshot-every` iterations. No coach and no rules act for the learner.

reward (shared by both learners, per 0.1 s step)
  +10 goal scored, -10 goal conceded, +2 * ball progress toward the opponent goal [m], -1 per own fall

usage (inside futsal/, needs Genesis + rsl-rl-lib + a GPU; the submitted run used --iters 500):
  python rl_train.py --ckpt ../models/go2-soccer/model_1398.pt --bc ../models/bc/bc_policy.pt -n 192 --iters 500
"""
import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn as nn
from tensordict import TensorDict

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from collect_bc import HIDDEN, load_tactics, make_mlp, sample_tactics
from futsal_env import OBS_DIM, FutsalSim
from futsal_teacher import teacher_commands


class FutsalRLEnv:
    """rsl_rl-style vectorised env: one row per learning robot (2 per match)."""

    def __init__(self, sim, tactics_pool, teacher_prob=0.5):
        self.sim = sim
        self.n = sim.n
        self.num_envs = 2 * sim.n
        self.num_actions = 3
        self.device = sim.dev
        self.max_episode_length = sim.max_hl_steps
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {}
        self.tv = tactics_pool
        self.teacher_prob = teacher_prob
        self.opponent = None  # nn.Module snapshot
        self.tactics = sample_tactics(self.tv, self.n, self.device)
        self.opp_teacher = torch.rand(self.n, device=self.device) < teacher_prob
        self.prev_ball_x = sim.ball_pos[:, 0].clone()
        self.stats = dict(goals_for=0.0, goals_against=0.0, gf_vs_teacher=0.0, ga_vs_teacher=0.0,
                          gf_vs_self=0.0, ga_vs_self=0.0, falls=0.0, steps=0)
        self._obs = None
        self._compute_obs()

    def _compute_obs(self):
        self._full_obs = self.sim.observations(self.tactics)  # (n,4,D)
        self._obs = self._full_obs[:, :2].reshape(-1, OBS_DIM)

    def get_observations(self):
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs])

    @torch.no_grad()
    def _opponent_cmd(self):
        t = teacher_commands(self.sim, self.tactics)[:, 2:]
        if self.opponent is None:
            return t
        o = self.opponent(self._full_obs[:, 2:].reshape(-1, OBS_DIM)).reshape(self.n, 2, 3).clamp(-1, 1)
        return torch.where(self.opp_teacher[:, None, None], t, o)

    def step(self, actions):
        a = actions.reshape(self.n, 2, 3).clamp(-1, 1)
        cmd = torch.cat((a, self._opponent_cmd()), dim=1)
        bx0 = self.sim.ball_pos[:, 0].clone()
        ga, gb, fell, done = self.sim.hl_step(cmd)
        progress = (self.sim.ball_pos[:, 0] - bx0)
        progress = torch.where(ga | gb, torch.zeros_like(progress), progress)  # kick-off teleports the ball
        r = 10.0 * ga.float() - 10.0 * gb.float() + 2.0 * progress - 1.0 * fell[:, :2].float().sum(-1)
        st = self.stats
        st["goals_for"] += ga.sum().item()
        st["goals_against"] += gb.sum().item()
        st["gf_vs_teacher"] += (ga & self.opp_teacher).sum().item()
        st["ga_vs_teacher"] += (gb & self.opp_teacher).sum().item()
        st["gf_vs_self"] += (ga & ~self.opp_teacher).sum().item()
        st["ga_vs_self"] += (gb & ~self.opp_teacher).sum().item()
        st["falls"] += fell[:, :2].sum().item()
        st["steps"] += self.n
        if done.any():
            idx = done.nonzero().flatten()
            self.sim.reset_envs(idx)
            self.tactics[idx] = sample_tactics(self.tv, idx.numel(), self.device)
            self.opp_teacher[idx] = torch.rand(idx.numel(), device=self.device) < self.teacher_prob
        self._compute_obs()
        rew = r.repeat_interleave(2)
        dones = done.repeat_interleave(2)
        self.episode_length_buf = self.sim.hl_step_count.repeat_interleave(2)
        extras = {"time_outs": dones.float(), "episode": {}}
        return self.get_observations(), rew, dones, extras

    def reset(self):
        return self.get_observations()


def train_cfg(exp, init_std):
    return {
        "algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.002,
                      "gamma": 0.99, "lam": 0.95, "learning_rate": 3e-4, "max_grad_norm": 1.0,
                      "num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "adaptive",
                      "use_clipped_value_loss": True, "value_loss_coef": 1.0},
        "actor": {"class_name": "MLPModel", "hidden_dims": list(HIDDEN), "activation": "elu",
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_std,
                                       "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 32,
        "save_interval": 50,
        "run_name": exp,
        "logger": "tensorboard",
    }


def snapshot(actor_mlp):
    m = copy.deepcopy(actor_mlp).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bc", required=True)
    ap.add_argument("-e", "--exp", default="futsal-rl")
    ap.add_argument("-n", type=int, default=192)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=20, help="critic-only iterations")
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--match-s", type=float, default=40)
    ap.add_argument("--init-std", type=float, default=0.25)
    ap.add_argument("--resume", help="runner checkpoint (actor+critic+optimizer) to continue from")
    ap.add_argument("--start-iter", type=int, default=0, help="iteration count already done (for --resume)")
    ap.add_argument("--opponent", help="policy file used as the first self-play opponent (default: BC)")
    a = ap.parse_args()
    log_dir = f"logs/{a.exp}"
    os.makedirs(log_dir, exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=3, performance_mode=True)
    sim = FutsalSim(a.n, a.ckpt, match_s=a.match_s)
    env = FutsalRLEnv(sim, load_tactics("tactics.json", sim.dev))
    cfg = train_cfg(a.exp, a.init_std)
    json.dump(vars(a) | {"train_cfg": cfg}, open(f"{log_dir}/config.json", "w"), indent=1)
    runner = OnPolicyRunner(env, cfg, log_dir, device=sim.dev)
    actor = runner.alg.actor
    bc = torch.load(a.bc, weights_only=False)["actor_mlp"]
    missing, unexpected = actor.load_state_dict(bc, strict=False)
    print("BC loaded; missing:", missing, "unexpected:", unexpected, flush=True)
    if a.resume:
        runner.load(a.resume)
        print("resumed from", a.resume, "at iteration", a.start_iter, flush=True)
    opp = a.opponent or a.bc
    opp_sd = torch.load(opp, weights_only=False)["actor_mlp"]
    opp_net = make_mlp(sim.dev)
    opp_net.load_state_dict({k[4:]: v for k, v in opp_sd.items()})
    env.opponent = snapshot(opp_net)  # self-play starts against the imitation policy (or the given snapshot)

    resumed = a.resume is not None
    log = open(f"{log_dir}/progress.csv", "a" if resumed else "w", newline="")
    w = csv.writer(log)
    if not resumed:
        w.writerow(["iter", "wall_s", "gf_per_min", "ga_per_min", "gf_vs_teacher", "ga_vs_teacher", "gf_vs_self",
                    "ga_vs_self", "falls_per_min"])
    t0 = time.time()
    done_iters = a.start_iter
    chunk = 10
    for p in actor.parameters():
        p.requires_grad_(False)
    while done_iters < a.iters:
        if done_iters >= a.warmup or resumed:
            for p in actor.parameters():
                p.requires_grad_(True)
        for k in env.stats:
            env.stats[k] = 0.0
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=False)
        done_iters += chunk
        st = env.stats
        mins = st["steps"] * sim.dt * sim.decimation / 60.0
        w.writerow([done_iters, round(time.time() - t0), st["goals_for"] / mins, st["goals_against"] / mins,
                    st["gf_vs_teacher"], st["ga_vs_teacher"], st["gf_vs_self"], st["ga_vs_self"],
                    st["falls"] / mins])
        log.flush()
        print(f"[futsal] iter {done_iters} wall {time.time() - t0:.0f}s GF/min {st['goals_for'] / mins:.3f} "
              f"GA/min {st['goals_against'] / mins:.3f} vsT {st['gf_vs_teacher']:.0f}-{st['ga_vs_teacher']:.0f} "
              f"vsSelf {st['gf_vs_self']:.0f}-{st['ga_vs_self']:.0f}", flush=True)
        if done_iters % a.snapshot_every == 0:
            mlp = nn.Sequential(*[m for m in actor.mlp]) if hasattr(actor, "mlp") else actor
            env.opponent = snapshot(mlp)
            torch.save({"actor_mlp": {f"mlp.{k}": v.cpu() for k, v in mlp.state_dict().items()},
                        "iter": done_iters}, f"{log_dir}/policy_{done_iters}.pt")
    runner.save(f"{log_dir}/model_final.pt")


if __name__ == "__main__":
    main()
