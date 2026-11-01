"""2-vs-2 futsal (keeper + field player per team) with four Unitree Go2 in Genesis.

Layers
  low level : a trained velocity-command locomotion policy (go2-soccer / go2-cmd, frozen) runs every robot at 50 Hz
  high level: (vx, vy, wz) commands for each robot every `decimation` physics steps (default 0.1 s)
              -> rule-based teacher (futsal_teacher.py) or a learned policy

Everything that a high-level policy sees is expressed in a *team frame*: team A attacks +x, team B is mirrored
(x, y) -> (-x, -y), so one policy can play on either side.

Robots: 0 = A keeper, 1 = A player, 2 = B keeper, 3 = B player.
"""
import math
import os
import pickle
import sys

import torch
import torch.nn as nn

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

HERE = os.path.dirname(os.path.abspath(__file__))
LOCO_DIR = os.path.join(HERE, "..", "locomotion")
sys.path.insert(0, LOCO_DIR)

HALF_L, HALF_W = 3.0, 2.0  # pitch 6 m x 4 m
GOAL_W, GOAL_D = 1.4, 0.5
BALL_R = 0.1
WALL_H, WALL_T = 0.3, 0.1
TEAM_COLOR = [(0.15, 0.35, 0.95), (0.15, 0.35, 0.95), (0.95, 0.2, 0.15), (0.95, 0.2, 0.15)]
ROLE = [0, 1, 0, 1]  # 0 keeper, 1 player
TEAM = [0, 0, 1, 1]
# home positions in the team frame (attacking +x)
HOME = {0: (-HALF_L + 0.35, 0.0), 1: (-0.9, 0.0)}
CMD_LO = (-0.8, -0.6, -1.5)
CMD_HI = (1.2, 0.6, 1.5)
OBS_DIM = 23


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_loco(ckpt, device):
    """The low-level actor is a plain MLP 45-512-256-128-12 (ELU); load its weights without rsl_rl."""
    sd = torch.load(ckpt, weights_only=False, map_location=device)["actor_state_dict"]
    mlp = nn.Sequential(nn.Linear(45, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12)).to(device)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    with open(os.path.join(os.path.dirname(ckpt), "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, *_ = pickle.load(f)
    return mlp, env_cfg, obs_cfg


class FutsalSim:
    def __init__(self, n_envs, loco_ckpt, decimation=5, match_s=60.0, camera_res=None, randomize=True,
                 ball_decay=0.985):
        self.n = n_envs
        self.dev = gs.device
        self.dt = 0.02
        self.decimation = decimation
        self.max_hl_steps = int(match_s / (self.dt * decimation))
        self.randomize = randomize
        self.ball_decay = ball_decay
        self.mlp, self.env_cfg, obs_cfg = load_loco(loco_ckpt, self.dev)
        sc = obs_cfg["obs_scales"]
        self.s_ang, self.s_dofv = sc["ang_vel"], sc["dof_vel"]
        self.cmd_scale = torch.tensor([sc["lin_vel"], sc["lin_vel"], sc["ang_vel"]], device=self.dev)
        self.cmd_lo = torch.tensor(CMD_LO, device=self.dev)
        self.cmd_hi = torch.tensor(CMD_HI, device=self.dev)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, max_collision_pairs=120,
                                                  tolerance=1e-5),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robots = []
        for i in range(4):
            x, y, yaw = self._home_world(i)
            self.robots.append(self.scene.add_entity(
                gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(x, y, 0.42),
                               quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                surface=gs.surfaces.Default(color=TEAM_COLOR[i]),
            ))
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_R, pos=(0.0, 0.0, BALL_R)),
            material=gs.materials.Rigid(rho=70.0),
            surface=gs.surfaces.Default(color=(1.0, 0.85, 0.1)),
        )
        self._add_walls()
        self.cam = None
        if camera_res:
            self.cam = self.scene.add_camera(res=camera_res, pos=(0.0, -5.0, 4.4), lookat=(0.0, -0.2, 0.0),
                                             fov=42, GUI=False)
        self.scene.build(n_envs=n_envs)

        r0 = self.robots[0]
        names = self.env_cfg["joint_names"]
        self.motors = torch.tensor([r0.get_joint(nm).dof_start for nm in names], dtype=gs.tc_int, device=self.dev)
        self.act_order = torch.argsort(self.motors)
        for r in self.robots:
            r.set_dofs_kp([self.env_cfg["kp"]] * 12, self.motors)
            r.set_dofs_kv([self.env_cfg["kd"]] * 12, self.motors)
        self.default_dof = torch.tensor([self.env_cfg["default_joint_angles"][nm] for nm in names],
                                        dtype=gs.tc_float, device=self.dev)
        self.init_dof_qpos = torch.tensor([self.env_cfg["default_joint_angles"][j.name] for j in r0.joints[1:]],
                                          dtype=gs.tc_float, device=self.dev)
        self.gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.dev)

        n = n_envs
        self.actions = torch.zeros((n, 4, 12), device=self.dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.cmd = torch.zeros((n, 4, 3), device=self.dev)
        self.pos = torch.zeros((n, 4, 3), device=self.dev)
        self.yaw = torch.zeros((n, 4), device=self.dev)
        self.vel = torch.zeros((n, 4, 3), device=self.dev)
        self.up = torch.zeros((n, 4), device=self.dev)
        self.ball_pos = torch.zeros((n, 3), device=self.dev)
        self.ball_vel = torch.zeros((n, 3), device=self.dev)
        self.score = torch.zeros((n, 2), dtype=torch.long, device=self.dev)
        self.hl_step_count = torch.zeros((n,), dtype=torch.long, device=self.dev)
        self.ball_still = torch.zeros((n,), device=self.dev)
        self.falls = torch.zeros((n, 4), dtype=torch.long, device=self.dev)
        self.reset_all()

    # ------------------------------------------------------------------ setup
    def _home_world(self, i, jitter=0.0):
        hx, hy = HOME[ROLE[i]]
        hx += (torch.rand(1).item() * 2 - 1) * jitter
        hy += (torch.rand(1).item() * 2 - 1) * jitter * (2.0 if ROLE[i] == 1 else 0.8)
        if TEAM[i] == 0:
            return hx, hy, 0.0
        return -hx, -hy, math.pi

    def _add_walls(self):
        def box(size, pos):
            self.scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True),
                                  surface=gs.surfaces.Default(color=(0.85, 0.85, 0.85)))
        L, Wd = 2 * HALF_L, 2 * HALF_W
        for s in (-1, 1):
            box((L + 2 * WALL_T, WALL_T, WALL_H), (0.0, s * (HALF_W + WALL_T / 2), WALL_H / 2))  # side boards
            seg = (Wd - GOAL_W) / 2
            for t in (-1, 1):  # end boards beside the goal mouth
                box((WALL_T, seg, WALL_H), (s * (HALF_L + WALL_T / 2), t * (GOAL_W / 2 + seg / 2), WALL_H / 2))
            # goal: back + two sides (ball stays in once it is in)
            box((WALL_T, GOAL_W + 2 * WALL_T, WALL_H), (s * (HALF_L + GOAL_D + WALL_T / 2), 0.0, WALL_H / 2))
            for t in (-1, 1):
                box((GOAL_D, WALL_T, WALL_H), (s * (HALF_L + GOAL_D / 2), t * (GOAL_W / 2 + WALL_T / 2), WALL_H / 2))

    # ------------------------------------------------------------------ resets
    def _robot_qpos(self, i, k, jitter):
        q = torch.zeros((k, 19), device=self.dev)
        for j in range(k):
            x, y, yaw = self._home_world(i, jitter)
            if jitter > 0:
                yaw += (torch.rand(1).item() * 2 - 1) * 0.4
            q[j, :3] = torch.tensor([x, y, 0.42])
            q[j, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        q[:, 7:] = self.init_dof_qpos
        return q

    def _reset_robot(self, i, idx, jitter):
        if idx.numel() == 0:
            return
        self.robots[i].set_qpos(self._robot_qpos(i, idx.numel(), jitter), envs_idx=idx, zero_velocity=True)
        self.actions[idx, i] = 0.0
        self.last_actions[idx, i] = 0.0

    def _reset_ball(self, idx, spread):
        if idx.numel() == 0:
            return
        q = torch.zeros((idx.numel(), 7), device=self.dev)
        q[:, 0] = (torch.rand(idx.numel(), device=self.dev) * 2 - 1) * spread
        q[:, 1] = (torch.rand(idx.numel(), device=self.dev) * 2 - 1) * spread * 1.5
        q[:, 2] = BALL_R
        q[:, 3] = 1.0
        self.ball.set_qpos(q, envs_idx=idx, zero_velocity=True)
        self.ball_still[idx] = 0.0

    def kickoff(self, idx):
        j = 0.5 if self.randomize else 0.0
        for i in range(4):
            self._reset_robot(i, idx, j)
        self._reset_ball(idx, 0.6 if self.randomize else 0.0)

    def reset_envs(self, idx):
        if idx.numel() == 0:
            return
        self.kickoff(idx)
        self.score[idx] = 0
        self.hl_step_count[idx] = 0
        self.falls[idx] = 0
        self.cmd[idx] = 0.0

    def reset_all(self):
        self.reset_envs(torch.arange(self.n, device=self.dev))
        self._read_state()

    # ------------------------------------------------------------------ state
    def _read_state(self):
        for i, r in enumerate(self.robots):
            self.pos[:, i] = r.get_pos()
            q = r.get_quat()
            self.yaw[:, i] = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                         1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            self.vel[:, i] = r.get_vel()
            self.up[:, i] = -transform_by_quat(self.gravity.expand(self.n, 3), inv_quat(q))[:, 2]
        self.ball_pos[:] = self.ball.get_pos()
        self.ball_vel[:] = self.ball.get_vel()

    def _loco_obs(self):
        obs = []
        for i, r in enumerate(self.robots):
            q = r.get_quat()
            iq = inv_quat(q)
            ang = transform_by_quat(r.get_ang(), iq)
            grav = transform_by_quat(self.gravity.expand(self.n, 3), iq)
            dpos = r.get_dofs_position(self.motors) - self.default_dof
            dvel = r.get_dofs_velocity(self.motors)
            obs.append(torch.cat((ang * self.s_ang, grav, self.cmd[:, i] * self.cmd_scale, dpos,
                                  dvel * self.s_dofv, self.actions[:, i]), dim=-1))
        return torch.stack(obs, dim=1)

    # ------------------------------------------------------------------ stepping
    def set_commands(self, cmd_norm):
        """cmd_norm: (n, 4, 3) in [-1, 1] -> body-frame (vx, vy, wz)."""
        c = cmd_norm.clamp(-1, 1)
        self.cmd = self.cmd_lo + (c + 1) * 0.5 * (self.cmd_hi - self.cmd_lo)

    @torch.no_grad()
    def physics_step(self):
        a = self.mlp(self._loco_obs().reshape(-1, 45)).reshape(self.n, 4, 12)
        self.last_actions, self.actions = self.actions, a  # 1-step action latency like the training env
        for i, r in enumerate(self.robots):
            tgt = self.last_actions[:, i] * self.env_cfg["action_scale"] + self.default_dof
            r.control_dofs_position(tgt[:, self.act_order], slice(6, 18))
        self.scene.step()
        bv = self.ball.get_dofs_velocity()
        self.ball.set_dofs_velocity(bv * self.ball_decay)

    @torch.no_grad()
    def hl_step(self, cmd_norm):
        """Run one high-level step. Returns per-env event tensors (goals_A, goals_B, falls (n,4), done)."""
        self.set_commands(cmd_norm)
        goal_a = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        goal_b = torch.zeros_like(goal_a)
        fell = torch.zeros((self.n, 4), dtype=torch.bool, device=self.dev)
        for _ in range(self.decimation):
            self.physics_step()
            bp = self.ball.get_pos()
            in_mouth = bp[:, 1].abs() < GOAL_W / 2
            ga = (bp[:, 0] > HALF_L + BALL_R) & in_mouth
            gb = (bp[:, 0] < -HALF_L - BALL_R) & in_mouth
            out = (bp[:, 1].abs() > HALF_W + 0.3) | (bp[:, 0].abs() > HALF_L + GOAL_D + 0.3) | (bp[:, 2] < -0.5)
            goal_a |= ga
            goal_b |= gb
            if (ga | gb).any():
                self.kickoff((ga | gb).nonzero().flatten())
            if out.any():
                self._reset_ball(out.nonzero().flatten(), 0.3)
            for i, r in enumerate(self.robots):
                q = r.get_quat()
                upz = -transform_by_quat(self.gravity.expand(self.n, 3), inv_quat(q))[:, 2]
                f = upz < 0.5  # tilted more than ~60 deg -> fallen, respawn at home
                if f.any():
                    idx = f.nonzero().flatten()
                    self._reset_robot(i, idx, 0.2)
                    fell[:, i] |= f
        self._read_state()
        # dead ball: nobody touched it for 8 s -> drop ball near the centre
        moving = self.ball_vel[:, :2].norm(dim=-1) > 0.05
        self.ball_still = torch.where(moving, torch.zeros_like(self.ball_still),
                                      self.ball_still + self.dt * self.decimation)
        stuck = self.ball_still > 8.0
        if stuck.any():
            self._reset_ball(stuck.nonzero().flatten(), 0.3)
        self.score[:, 0] += goal_a.long()
        self.score[:, 1] += goal_b.long()
        self.falls += fell.long()
        self.hl_step_count += 1
        done = self.hl_step_count >= self.max_hl_steps
        return goal_a, goal_b, fell, done

    # ------------------------------------------------------------------ team-frame views
    def team_view(self):
        """Positions/velocities/yaws of robots and ball in each robot's team frame.

        Returns dict of tensors with a leading (n, 4) 'viewer' axis:
          pos (n,4,4,2), yaw (n,4,4), vel (n,4,4,2), ball (n,4,2), ball_vel (n,4,2)
        """
        s = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.dev)  # mirror for team B viewers
        off = torch.tensor([0.0, 0.0, math.pi, math.pi], device=self.dev)
        pos = self.pos[:, None, :, :2] * s[None, :, None, None]
        vel = self.vel[:, None, :, :2] * s[None, :, None, None]
        yaw = wrap_t(self.yaw[:, None, :] + off[None, :, None])
        ball = self.ball_pos[:, None, :2] * s[None, :, None]
        bvel = self.ball_vel[:, None, :2] * s[None, :, None]
        return dict(pos=pos, vel=vel, yaw=yaw, ball=ball, ball_vel=bvel)

    def observations(self, tactics):
        """High-level observation (n, 4, OBS_DIM) for every robot. tactics: (n, 2, 3) per team."""
        v = self.team_view()
        ar = torch.arange(4, device=self.dev)
        own_p = v["pos"][:, ar, ar]  # (n,4,2)
        own_y = v["yaw"][:, ar, ar]
        own_v = v["vel"][:, ar, ar]
        c, s_ = torch.cos(own_y), torch.sin(own_y)

        def to_body(p):  # p (n,4,2) team-frame point -> body frame of each viewer
            d = p - own_p
            return torch.stack((c * d[..., 0] + s_ * d[..., 1], -s_ * d[..., 0] + c * d[..., 1]), dim=-1)

        mate = torch.tensor([1, 0, 3, 2], device=self.dev)
        opp_k = torch.tensor([2, 2, 0, 0], device=self.dev)
        opp_p = torch.tensor([3, 3, 1, 1], device=self.dev)
        scale_p = torch.tensor([HALF_L, HALF_W], device=self.dev)
        role = torch.nn.functional.one_hot(torch.tensor(ROLE, device=self.dev), 2).float().expand(self.n, 4, 2)
        tac = tactics[:, torch.tensor(TEAM, device=self.dev)]
        obs = torch.cat((
            own_p / scale_p,
            torch.stack((c, s_), dim=-1),
            own_v,
            to_body(v["ball"]) / 3.0,
            v["ball"] / scale_p,
            v["ball_vel"],
            to_body(v["pos"][:, ar, mate]) / 3.0,
            to_body(v["pos"][:, ar, opp_k]) / 3.0,
            to_body(v["pos"][:, ar, opp_p]) / 3.0,
            role,
            tac,
        ), dim=-1)
        return obs


def wrap_t(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
