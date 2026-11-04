"""Test setup: make futsal/ importable and, when Genesis is not installed, register a minimal stub.

The stub only lets modules that do `import genesis as gs` at the top be imported so that their pure tensor logic
(observations, teacher rules, command scaling) can be tested on CPU. Anything that really needs the simulator
(gs.Scene, FutsalSim.__init__, physics) is never called by the tests.
"""
import os
import sys
import types

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("futsal", "locomotion"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import genesis  # noqa: F401
except ImportError:
    def _needs_simulator(*_args, **_kwargs):
        raise RuntimeError("Genesis is not installed; this code path needs the simulator")

    gs = types.ModuleType("genesis")
    gs.__dict__.update(device="cpu", gpu="gpu", cpu="cpu", init=_needs_simulator, IS_STUB=True)
    utils = types.ModuleType("genesis.utils")
    geom = types.ModuleType("genesis.utils.geom")
    geom.inv_quat = _needs_simulator
    geom.transform_by_quat = _needs_simulator
    gs.utils = utils
    utils.geom = geom
    sys.modules.update({"genesis": gs, "genesis.utils": utils, "genesis.utils.geom": geom})
