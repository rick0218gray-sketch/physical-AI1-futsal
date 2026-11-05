"""Pure tensor logic of the futsal environment and the rule-based teacher (needs torch, not Genesis or a GPU).

Skipped automatically when torch is not installed (as in CI).
"""
import math
import os
import types

import pytest

torch = pytest.importorskip("torch")

import futsal_env as fe  # noqa: E402
from futsal_teacher import teacher_commands, to_normalized  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SWAP = [2, 3, 0, 1]  # robot i of the mirrored world is robot SWAP[i] of the original one


def fake_sim(n=64, seed=0):
    """Object with the state tensors FutsalSim exposes, plus its real team_view / observations methods."""
    g = torch.Generator().manual_seed(seed)

    def u(*shape, lo=-1.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=g)

    s = types.SimpleNamespace(n=n, dev="cpu")
    s.pos = torch.stack((u(n, 4, lo=-fe.HALF_L, hi=fe.HALF_L), u(n, 4, lo=-fe.HALF_W, hi=fe.HALF_W),
                         u(n, 4, lo=0.25, hi=0.35)), dim=-1)
    s.yaw = u(n, 4, lo=-math.pi, hi=math.pi)
    s.vel = u(n, 4, 3)
    s.ball_pos = torch.stack((u(n, lo=-fe.HALF_L, hi=fe.HALF_L), u(n, lo=-fe.HALF_W, hi=fe.HALF_W),
                              torch.full((n,), fe.BALL_R)), dim=-1)
    s.ball_vel = u(n, 3)
    return bind(s)


def bind(s):
    s.team_view = types.MethodType(fe.FutsalSim.team_view, s)
    s.observations = types.MethodType(fe.FutsalSim.observations, s)
    return s


def mirrored(s):
    """The same situation with the teams swapped: rotate the pitch by 180 deg and relabel blue <-> red."""
    m = types.SimpleNamespace(n=s.n, dev=s.dev)
    flip = torch.tensor([-1.0, -1.0, 1.0])
    m.pos = s.pos[:, SWAP] * flip
    m.vel = s.vel[:, SWAP] * flip
    m.yaw = fe.wrap_t(s.yaw[:, SWAP] + math.pi)
    m.ball_pos = s.ball_pos * flip
    m.ball_vel = s.ball_vel * flip
    return bind(m)


def tactics(n, seed=1):
    return torch.rand((n, 2, 3), generator=torch.Generator().manual_seed(seed))


def test_wrap_and_wrap_t_agree_and_stay_in_range():
    xs = [-10.0, -math.pi, -1.0, 0.0, 1.0, math.pi - 1e-6, 7.5, 100.0]
    t = fe.wrap_t(torch.tensor(xs, dtype=torch.float64))
    for x, y in zip(xs, t.tolist()):
        assert -math.pi <= fe.wrap(x) < math.pi
        assert y == pytest.approx(fe.wrap(x), abs=1e-9)
        assert math.cos(y) == pytest.approx(math.cos(x), abs=1e-9)


def test_command_scaling_round_trip():
    """FutsalSim.set_commands maps [-1, 1] to the body-frame command range; the teacher's to_normalized inverts it."""
    sim = types.SimpleNamespace(cmd_lo=torch.tensor(fe.CMD_LO), cmd_hi=torch.tensor(fe.CMD_HI))
    c = torch.rand((5, 4, 3)) * 2 - 1
    fe.FutsalSim.set_commands(sim, c)
    assert torch.allclose(to_normalized(sim.cmd, sim.cmd_lo, sim.cmd_hi), c, atol=1e-6)
    fe.FutsalSim.set_commands(sim, torch.full((1, 4, 3), 5.0))  # out of range -> clamped
    assert torch.allclose(sim.cmd[0, 0], torch.tensor(fe.CMD_HI))
    fe.FutsalSim.set_commands(sim, torch.zeros((1, 4, 3)))
    assert torch.allclose(sim.cmd[0, 0], (torch.tensor(fe.CMD_LO) + torch.tensor(fe.CMD_HI)) / 2)


def test_observation_shape_and_layout():
    s = fake_sim(n=3)
    s.pos[0, 0, :2] = torch.tensor([0.0, 0.0])
    s.yaw[0, 0] = 0.0
    s.ball_pos[0, :2] = torch.tensor([1.5, -0.6])
    obs = s.observations(tactics(3))
    assert obs.shape == (3, 4, fe.OBS_DIM)
    o = obs[0, 0]
    assert torch.allclose(o[2:4], torch.tensor([1.0, 0.0]))                    # cos, sin of own yaw
    assert torch.allclose(o[6:8], torch.tensor([1.5, -0.6]) / 3.0, atol=1e-6)  # ball in the body frame
    assert torch.allclose(o[8:10], torch.tensor([1.5 / fe.HALF_L, -0.6 / fe.HALF_W]), atol=1e-6)
    roles = obs[0, :, 18:20]
    assert roles.tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]   # keeper / player one-hot


def test_observations_are_side_symmetric():
    """README: 'one policy plays either side'. Swapping teams and rotating the pitch must give identical inputs."""
    s = fake_sim()
    tac = tactics(s.n)
    obs = s.observations(tac)
    obs_m = mirrored(s).observations(tac[:, [1, 0]])
    assert torch.allclose(obs_m, obs[:, SWAP], atol=1e-5)


def test_teacher_is_side_symmetric_and_in_range():
    s = fake_sim(seed=3)
    tac = tactics(s.n, seed=4)
    cmd = teacher_commands(s, tac)
    assert cmd.shape == (s.n, 4, 3)
    assert cmd.abs().max() <= 1.0
    cmd_m = teacher_commands(mirrored(s), tac[:, [1, 0]])
    assert torch.allclose(cmd_m, cmd[:, SWAP], atol=1e-4)


def test_teacher_keeper_stays_near_goal_without_danger():
    """Keeper (robot 0) at its target with the ball far away -> (almost) no translation command."""
    s = fake_sim(n=1)
    tac = torch.tensor([[[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]])  # keeper_depth 0 -> target x = -HALF_L + 0.3
    s.ball_pos[0] = torch.tensor([2.0, 0.0, fe.BALL_R])
    s.ball_vel[0] = 0.0
    s.pos[0, 0] = torch.tensor([-fe.HALF_L + 0.3, 0.0, 0.3])
    s.yaw[0, 0] = 0.0
    vx, vy, wz = to_physical(teacher_commands(s, tac))[0, 0].tolist()
    assert abs(vx) < 1e-5 and abs(vy) < 1e-5 and abs(wz) < 1e-5


def to_physical(c):
    lo, hi = torch.tensor(fe.CMD_LO), torch.tensor(fe.CMD_HI)
    return lo + (c + 1) * 0.5 * (hi - lo)


@pytest.mark.parametrize("path", ["models/bc/bc_policy.pt", "models/futsal-rl/policy_500.pt"])
def test_shipped_high_level_policies_load_and_run_on_cpu(path):
    from players import load_policy

    net = load_policy(os.path.join(ROOT, path), "cpu")
    s = fake_sim(n=8)
    obs = s.observations(tactics(8))
    with torch.no_grad():
        out = net(obs.reshape(-1, fe.OBS_DIM))
    assert out.shape == (32, 3)
    assert torch.isfinite(out).all()


def test_shipped_locomotion_policy_loads_on_cpu():
    mlp, env_cfg, obs_cfg = fe.load_loco(os.path.join(ROOT, "models", "go2-soccer", "model_1398.pt"), "cpu")
    assert len(env_cfg["joint_names"]) == 12
    assert set(obs_cfg["obs_scales"]) >= {"ang_vel", "dof_vel", "lin_vel"}
    with torch.no_grad():
        assert mlp(torch.zeros(4, 45)).shape == (4, 12)


def test_reset_envs_refreshes_cached_state():
    """reset_envs() teleports robots and ball between hl_steps; the cached state (pos / yaw / up / ball_pos ...)
    must follow immediately, otherwise the first observation and ball-progress reward of the new episode use
    the previous episode's final state (fixed after submission, see README)."""
    class Body:
        def __init__(self):
            self.qpos = None

        def set_qpos(self, q, envs_idx=None, zero_velocity=True):
            self.qpos = q

    n = 2
    s = fake_sim(n=n)
    s.randomize = False
    s.robots = [Body() for _ in range(4)]
    s.ball = Body()
    s.init_dof_qpos = torch.zeros(12)
    s.actions = torch.zeros((n, 4, 12))
    s.last_actions = torch.zeros((n, 4, 12))
    s.cmd = torch.zeros((n, 4, 3))
    s.ball_still = torch.zeros(n)
    s.ball_reset = torch.zeros(n, dtype=torch.bool)
    s.up = torch.zeros((n, 4))
    s.score = torch.ones((n, 2), dtype=torch.long)
    s.hl_step_count = torch.ones(n, dtype=torch.long)
    s.falls = torch.ones((n, 4), dtype=torch.long)
    for name in ("_home_world", "_robot_qpos", "_reset_robot", "_reset_ball", "kickoff", "reset_envs"):
        setattr(s, name, types.MethodType(getattr(fe.FutsalSim, name), s))
    s.ball_pos[:, :2] = torch.tensor([2.5, 1.0])

    s.reset_envs(torch.arange(n))
    assert s.score.sum() == 0 and s.hl_step_count.sum() == 0 and s.falls.sum() == 0  # bookkeeping is reset
    assert torch.allclose(s.ball.qpos[:, :2], torch.zeros(n, 2))                      # the ball was re-placed ...
    assert torch.allclose(s.ball_pos[:, :2], s.ball.qpos[:, :2])                      # ... and the cache follows
    assert torch.all(s.ball_vel == 0) and s.ball_reset.all()
    for i in range(4):
        assert torch.allclose(s.pos[:, i], s.robots[i].qpos[:, :3])
        q = s.robots[i].qpos
        assert torch.allclose(s.yaw[:, i], 2 * torch.atan2(q[:, 6], q[:, 3]))
    assert torch.all(s.vel == 0) and torch.all(s.up == 1)
