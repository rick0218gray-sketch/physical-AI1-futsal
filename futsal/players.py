"""Team controllers: rule-based teacher (with coach tactics) or a learned high-level policy."""
import torch

from collect_bc import make_mlp
from futsal_env import OBS_DIM
from futsal_teacher import teacher_commands


def load_policy(path, dev):
    net = make_mlp(dev)
    sd = torch.load(path, weights_only=False, map_location=dev)["actor_mlp"]
    net.load_state_dict({k[4:]: v for k, v in sd.items()})
    return net.eval()


class Team:
    """kind: 'teacher' or a path to a policy file (bc_policy.pt / policy_<iter>.pt)."""

    def __init__(self, kind, dev, label=None):
        self.kind = kind
        self.net = None if kind == "teacher" else load_policy(kind, dev)
        self.label = label or kind

    @torch.no_grad()
    def commands(self, sim, tactics, team, obs=None):
        sl = slice(2 * team, 2 * team + 2)
        if self.net is None:
            return teacher_commands(sim, tactics)[:, sl]
        if obs is None:
            obs = sim.observations(tactics)
        return self.net(obs[:, sl].reshape(-1, OBS_DIM)).reshape(sim.n, 2, 3).clamp(-1, 1)


def match_commands(sim, teams, tactics):
    obs = sim.observations(tactics)
    return torch.cat([teams[t].commands(sim, tactics, t, obs) for t in (0, 1)], dim=1)
