"""List goal times in a recorded match: python goals.py rec/x.pt"""
import sys

import torch

d = torch.load(sys.argv[1], weights_only=False)
print(d["names"], "final", d["frames"][-1]["score"])
print([(round(f["t"], 1), f["score"]) for f in d["frames"] if any(f["goal"])])
