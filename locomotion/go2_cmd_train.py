"""Go2 velocity-command tracking training (extension of the Genesis go2_train.py example).

Differences from the original example (forward 0.5 m/s only):
  * commands are sampled from vx in [-0.8, 1.0] m/s, vy in [-0.3, 0.3] m/s, yaw rate in [-1.0, 1.0] rad/s
  * 15% of resampled commands are "stand still" (all zero), so the policy also learns to stop
  * the yaw-tracking reward weight is raised (0.2 -> 0.5) so that turning is actually learned
  * commands are re-sampled every 4 s inside an episode, so the policy must follow command switches

usage: python go2_cmd_train.py -e go2-cmd -b 1024 --max-iterations 800
"""
import argparse
import copy
import os
import pickle
import shutil

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from go2_train import get_cfgs, get_train_cfg


class Go2CmdEnv(Go2Env):
    """Go2Env with a fraction of zero (stand) commands."""

    def __init__(self, *args, zero_cmd_prob=0.15, **kwargs):
        self.zero_cmd_prob = zero_cmd_prob
        super().__init__(*args, **kwargs)

    def _resample_commands(self, envs_idx):
        super()._resample_commands(envs_idx)
        if self.zero_cmd_prob <= 0:
            return
        zero = torch.rand((self.num_envs,), device=gs.device) < self.zero_cmd_prob
        if envs_idx is not None:
            zero &= envs_idx
        self.commands.masked_fill_(zero[:, None], 0.0)


def get_cmd_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.8, 1.0],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-1.0, 1.0],
    }
    reward_cfg = copy.deepcopy(reward_cfg)
    reward_cfg["reward_scales"]["tracking_ang_vel"] = 0.5
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-cmd")
    parser.add_argument("-b", "--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=800)
    parser.add_argument("--zero-cmd-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cmd_cfgs()
    env_cfg["zero_cmd_prob"] = args.zero_cmd_prob
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = Go2CmdEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        zero_cmd_prob=args.zero_cmd_prob,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
