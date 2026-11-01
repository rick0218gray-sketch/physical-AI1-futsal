"""Low-level locomotion for futsal players: fine-tune the go2-cmd policy for wider lateral/turn commands
and robustness to body contact (random pushes).

Differences from go2_cmd_train.py:
  * vx in [-0.8, 1.2] m/s, vy in [-0.6, 0.6] m/s (keeper side-steps), yaw rate in [-1.5, 1.5] rad/s
  * every `push_interval_s` a random half of the robots gets a horizontal base-velocity kick (up to push_max m/s)
  * roll/pitch termination relaxed from 10 to 20 deg (players get bumped)
  * starts from an existing checkpoint (default: go2-cmd model_799)

usage: python go2_soccer_loco_train.py -e go2-soccer --init ../c_genesis/locomotion/logs/go2-cmd/model_799.pt
"""
import argparse
import os
import pickle
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "locomotion"))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import genesis as gs  # noqa: E402

from go2_cmd_train import Go2CmdEnv, get_cmd_cfgs  # noqa: E402
from go2_train import get_train_cfg  # noqa: E402


class Go2PushEnv(Go2CmdEnv):
    def __init__(self, *args, push_interval_s=3.0, push_max=0.6, **kwargs):
        self.push_every = int(push_interval_s / 0.02)
        self.push_max = push_max
        super().__init__(*args, **kwargs)

    def step(self, actions):
        out = super().step(actions)
        if int(self.episode_length_buf[0]) % self.push_every == 0:
            vel = self.robot.get_dofs_velocity([0, 1])
            hit = (torch.rand((self.num_envs, 1), device=gs.device) < 0.5).float()
            kick = (torch.rand((self.num_envs, 2), device=gs.device) * 2 - 1) * self.push_max
            self.robot.set_dofs_velocity(vel + hit * kick, [0, 1])
        return out


def get_soccer_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cmd_cfgs()
    command_cfg["lin_vel_x_range"] = [-0.8, 1.2]
    command_cfg["lin_vel_y_range"] = [-0.6, 0.6]
    command_cfg["ang_vel_range"] = [-1.5, 1.5]
    env_cfg["termination_if_roll_greater_than"] = 20
    env_cfg["termination_if_pitch_greater_than"] = 20
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="go2-soccer")
    p.add_argument("-b", "--num-envs", type=int, default=1024)
    p.add_argument("--max-iterations", type=int, default=600)
    p.add_argument("--init", default=os.path.join(HERE, "..", "models", "go2-cmd", "model_799.pt"))
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    log_dir = os.path.join(HERE, "logs", args.exp_name)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_soccer_cfgs()
    train_cfg = get_train_cfg(args.exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = Go2PushEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                     command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.init)  # continues the iteration count of the init checkpoint (799 -> 800...)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
