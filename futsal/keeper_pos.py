"""Where do the keepers stand? Mean / share of time near the own goal, in the team frame.

usage: python keeper_pos.py rec/x.pt [...]
"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, weights_only=False)
    for name, i, s in ((d["names"][0], 0, 1.0), (d["names"][1], 2, -1.0)):
        xy = torch.stack([f["robots"][i][:2] for f in d["frames"]]) * s  # team frame: own goal at x = -3
        near = ((xy[:, 0] < -2.0) & (xy[:, 1].abs() < 1.0)).float().mean().item()
        print(f"{path} {name} keeper: mean x {xy[:, 0].mean():.2f} y {xy[:, 1].mean():.2f}  "
              f"time in front of own goal {near * 100:.0f}%")
