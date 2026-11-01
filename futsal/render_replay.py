"""Render a recorded match (record_match.py) to mp4. Physics is not stepped: each frame's qpos is set and rendered.

usage: python render_replay.py rec/m1.pt out.mp4 [--res 960x540] [--every 1] [--start 0] [--end 999]
"""
import argparse

import torch

import genesis as gs

from futsal_env import FutsalSim
from futsal_video import Recorder

TEAM_JP = {"teacher": "お手本（ルール＋監督）", "bc": "模倣学習", "rl": "強化学習"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("out")
    ap.add_argument("--ckpt", default="logs/go2-soccer/model_1398.pt")
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    res = tuple(int(x) for x in a.res.split("x"))
    d = torch.load(a.rec, weights_only=False)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0, performance_mode=True)
    sim = FutsalSim(1, a.ckpt, camera_res=res, randomize=False)
    fps = round(1 / (d["dt"] * a.every))
    rec = Recorder(a.out, res, fps=fps)
    names = [TEAM_JP.get(n, n) for n in d["names"]]
    goal_flash = 0
    sel = [f for f in d["frames"] if a.start <= f["t"] <= a.end][:: a.every]
    for i, f in enumerate(sel):
        for r, q in zip(sim.robots, f["robots"]):
            r.set_qpos(q[None].to(sim.dev), envs_idx=torch.tensor([0], device=sim.dev), zero_velocity=True)
        sim.ball.set_qpos(f["ball"][None].to(sim.dev), envs_idx=torch.tensor([0], device=sim.dev), zero_velocity=True)
        sim.scene.visualizer.update_visual_states(force_render=True)  # qpos was set without stepping physics
        rgb = sim.cam.render(rgb=True, force_render=True)[0]
        if any(f["goal"]):
            goal_flash = int(1.5 / (d["dt"] * a.every))
        top = [a.title] if a.title else []
        if goal_flash > 0:
            top.append("ゴール！")
            goal_flash -= 1
        bottom = []
        if f["coach"][0]:
            bottom.append(f"監督（青）：{f['coach'][0]}")
        if f["coach"][1]:
            bottom.append(f"監督（赤）：{f['coach'][1]}")
        bottom.append(f"青：{names[0]}　赤：{names[1]}　{f['t']:5.1f} 秒（等速）")
        rec.add(rgb, lines_top=top, lines_bottom=bottom, score=f["score"], names=("青", "赤"))
        if i % 50 == 0:
            print(f"frame {i}/{len(sel)}", flush=True)
    rec.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
