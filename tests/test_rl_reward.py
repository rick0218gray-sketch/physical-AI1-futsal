"""Reward and goal bookkeeping of the self-play RL env (rl_train.FutsalRLEnv) with a scripted fake simulator.

Needs torch; tensordict / rsl_rl / Genesis are replaced by stubs when they are not installed. Skipped without torch.
"""
import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _stub_missing(monkeypatch):
    try:
        importlib.import_module("tensordict")
    except ImportError:
        td = types.ModuleType("tensordict")

        class TensorDict(dict):
            def __init__(self, data, batch_size=None):
                super().__init__(data)
                self.batch_size = batch_size

        td.TensorDict = TensorDict
        monkeypatch.setitem(sys.modules, "tensordict", td)
    try:
        importlib.import_module("rsl_rl.runners")
    except ImportError:
        rsl = types.ModuleType("rsl_rl")
        runners = types.ModuleType("rsl_rl.runners")
        runners.OnPolicyRunner = object
        rsl.runners = runners
        monkeypatch.setitem(sys.modules, "rsl_rl", rsl)
        monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)


class ScriptedSim:
    """Plays back scripted events instead of physics. Same attributes FutsalRLEnv reads from FutsalSim."""

    dt, decimation = 0.02, 5

    def __init__(self, n, max_hl_steps=3):
        self.n, self.dev, self.max_hl_steps = n, "cpu", max_hl_steps
        self.ball_pos = torch.zeros((n, 3))
        self.hl_step_count = torch.zeros(n, dtype=torch.long)
        self.script = []
        self.resets = []

    def observations(self, tactics):
        return torch.zeros((self.n, 4, 23))

    def hl_step(self, cmd):
        assert cmd.shape == (self.n, 4, 3)
        ga, gb, fell, dx = self.script.pop(0)
        self.ball_pos[:, 0] += torch.tensor(dx)
        self.hl_step_count += 1
        done = self.hl_step_count >= self.max_hl_steps
        return torch.tensor(ga), torch.tensor(gb), torch.tensor(fell), done

    def reset_envs(self, idx):
        self.resets.append(idx.tolist())
        self.hl_step_count[idx] = 0


@pytest.fixture
def env(monkeypatch):
    _stub_missing(monkeypatch)
    rl_train = importlib.import_module("rl_train")
    monkeypatch.setattr(rl_train, "teacher_commands", lambda sim, tac: torch.zeros((sim.n, 4, 3)))
    sim = ScriptedSim(n=4)
    tv = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.2]])
    e = rl_train.FutsalRLEnv(sim, tv, teacher_prob=0.5)
    e.opp_teacher = torch.tensor([True, False, True, False])
    return e


F = [False] * 4


def test_reward_terms_and_row_layout(env):
    sim = env.sim
    sim.script.append((
        [True, False, False, False],                     # env 0: goal for (+10, ball progress ignored)
        [False, True, False, False],                     # env 1: goal against (-10)
        [[False] * 4, [False] * 4, [False] * 4,
         [True, True, True, False]],                     # env 3: both learners fell (-2); opponent falls ignored
        [3.0, -3.0, 0.25, 0.0],                          # env 2: ball moved 0.25 m toward the opponent goal (+0.5)
    ))
    obs, rew, dones, extras = env.step(torch.zeros(8, 3))
    assert rew.tolist() == pytest.approx([10, 10, -10, -10, 0.5, 0.5, -2, -2])
    assert obs["policy"].shape == (8, 23)
    assert not dones.any()
    st = env.stats
    assert (st["goals_for"], st["goals_against"]) == (1, 1)
    assert (st["gf_vs_teacher"], st["ga_vs_teacher"], st["gf_vs_self"], st["ga_vs_self"]) == (1, 0, 0, 1)
    assert st["falls"] == 2 and st["steps"] == 4


def test_episode_end_resets_and_reports_time_outs(env):
    sim = env.sim
    for _ in range(3):
        sim.script.append((F, F, [[False] * 4] * 4, [0.0] * 4))
    for k in range(3):
        _, _, dones, extras = env.step(torch.zeros(8, 3))
    assert dones.all() and extras["time_outs"].tolist() == [1.0] * 8
    assert sim.resets == [[0, 1, 2, 3]]
    assert env.episode_length_buf.tolist() == [0] * 8
