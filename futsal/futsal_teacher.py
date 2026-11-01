"""Rule-based futsal tactics (the teacher), conditioned on the coach's tactic vector.

tactic per team (3 values in [0, 1]):
  attack       : 0 = player drops back when the ball is in the opponent half, 1 = always chase the ball
  keeper_depth : 0 = keeper stays on the goal line, 1 = keeper comes ~0.9 m off the line
  press        : speed of the field player (0 -> 0.6 m/s, 1 -> 1.2 m/s)

Returns normalized commands (n, 4, 3) in [-1, 1] as expected by FutsalSim.set_commands.
"""
import math

import torch

from futsal_env import CMD_HI, CMD_LO, GOAL_W, HALF_L, ROLE, TEAM, wrap_t


def _norm(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def to_normalized(cmd, lo, hi):
    return (2 * (cmd - lo) / (hi - lo) - 1).clamp(-1, 1)


@torch.no_grad()
def teacher_commands(sim, tactics):
    dev = sim.dev
    v = sim.team_view()
    ar = torch.arange(4, device=dev)
    p = v["pos"][:, ar, ar]          # (n,4,2) own position, team frame
    yaw = v["yaw"][:, ar, ar]
    ball = v["ball"]                  # (n,4,2)
    bvel = v["ball_vel"]
    tac = tactics[:, torch.tensor(TEAM, device=dev)]  # (n,4,3)
    attack, depth, press = tac[..., 0], tac[..., 1], tac[..., 2]
    is_keeper = (torch.tensor(ROLE, device=dev) == 0).expand_as(yaw)

    # ---- field player: get behind the ball on the line ball->goal, then dribble through it
    goal = torch.tensor([HALF_L + 0.3, 0.0], device=dev)
    d = _norm(goal - ball)
    perp = torch.stack((-d[..., 1], d[..., 0]), dim=-1)
    rel = p - ball
    ahead = (rel * d).sum(-1) > -0.15          # robot is level with / in front of the ball
    side = torch.sign((rel * perp).sum(-1))
    side = torch.where(side == 0, torch.ones_like(side), side)
    around = ball - d * 0.35 + perp * side[..., None] * 0.55
    behind = ball - d * 0.45
    near_behind = (p - behind).norm(dim=-1) < 0.3
    push_tgt = ball + d * 0.8
    tgt = torch.where(ahead[..., None], around, torch.where(near_behind[..., None], push_tgt, behind))
    pushing = (~ahead) & near_behind
    speed = 0.6 + 0.6 * press

    # defensive drop: with a low attack value, wait in the own half while the ball is in the opponent half
    drop = (attack < 0.4) & (ball[..., 0] > 0.3)
    drop_tgt = torch.stack((torch.full_like(ball[..., 0], -0.8), ball[..., 1] * 0.5), dim=-1)
    tgt = torch.where(drop[..., None], drop_tgt, tgt)

    # ---- keeper: stand on a line in front of the goal, follow the ball laterally; clear when it is close
    gx = -HALF_L + 0.3 + 0.9 * depth
    ty = (ball[..., 1] * 0.8 + bvel[..., 1] * 0.3).clamp(-GOAL_W / 2, GOAL_W / 2)
    k_tgt = torch.stack((gx, ty), dim=-1)
    danger = ((p - ball).norm(dim=-1) < 0.9) & (ball[..., 0] < -HALF_L + 1.6)
    clear_d = _norm(torch.stack((torch.ones_like(ball[..., 0]), -ball[..., 1] * 0.5), dim=-1))
    k_tgt = torch.where(danger[..., None], ball + clear_d * 0.8, k_tgt)

    tgt = torch.where(is_keeper[..., None], k_tgt, tgt)
    dvec = tgt - p
    dist = dvec.norm(dim=-1)
    head_move = torch.atan2(dvec[..., 1], dvec[..., 0])
    head_ball = torch.atan2((ball - p)[..., 1], (ball - p)[..., 0])
    head_push = torch.atan2(d[..., 1], d[..., 0])
    head = torch.where(pushing, head_push, torch.where(dist > 0.5, head_move, head_ball))
    head = torch.where(drop, head_ball, head)
    k_head = torch.where(danger, head_ball, torch.zeros_like(head_ball))
    head = torch.where(is_keeper, k_head, head)
    max_v = torch.where(is_keeper, torch.full_like(speed, 1.0), speed)

    # ---- go-to controller in the body frame
    c, s = torch.cos(yaw), torch.sin(yaw)
    bx = c * dvec[..., 0] + s * dvec[..., 1]
    by = -s * dvec[..., 0] + c * dvec[..., 1]
    e = wrap_t(head - yaw)
    vx = (2.0 * bx).clamp(-0.8, 1.2)
    vx = torch.minimum(vx, max_v) * torch.cos(e).clamp(0.2, 1.0)
    vx = torch.where(pushing & ~is_keeper, speed * torch.cos(e).clamp(0.3, 1.0), vx)
    vy = (2.0 * by).clamp(-0.6, 0.6)
    wz = (2.5 * e).clamp(-1.5, 1.5)
    cmd = torch.stack((vx, vy, wz), dim=-1)
    lo = torch.tensor(CMD_LO, device=dev)
    hi = torch.tensor(CMD_HI, device=dev)
    return to_normalized(cmd, lo, hi)


def random_tactics(n, device):
    return torch.rand((n, 2, 3), device=device)
