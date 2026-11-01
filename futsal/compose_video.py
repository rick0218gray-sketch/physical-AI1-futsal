"""Compose the submission video from rendered match clips and text cards (PyAV + PIL only).

usage: python compose_video.py plan.json out.mp4
plan.json: {"res": [1280, 720], "fps": 10, "parts": [
   {"card": ["title", "subtitle", ...], "seconds": 4},
   {"clip": "rec/teacher_demo.mp4", "start": 0, "end": 40}, ...]}
Adds a silent AAC track so that upload sites treat it as a normal video.
"""
import json
import sys
from fractions import Fraction

import av
import numpy as np
from PIL import Image, ImageDraw

from futsal_video import font

BG = (27, 36, 51)


def card_frame(lines, res):
    img = Image.new("RGB", res, BG)
    d = ImageDraw.Draw(img)
    W, H = res
    k = W / 1280
    sizes = [int(56 * k)] + [int(30 * k)] * (len(lines) - 1)
    colors = [(255, 255, 255)] + [(201, 210, 224)] * (len(lines) - 1)
    total = sum(s * 1.6 for s in sizes)
    y = (H - total) / 2
    for t, s, c in zip(lines, sizes, colors):
        if t.startswith("!"):
            t, c = t[1:], (242, 166, 107)
        f = font(s)
        d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=c)
        y += s * 1.6
    return np.asarray(img)


def main():
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    res, fps = tuple(plan["res"]), plan["fps"]
    out = av.open(sys.argv[2], mode="w")
    vs = out.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = res[0], res[1], "yuv420p"
    vs.options = {"crf": "18"}
    as_ = out.add_stream("aac", rate=48000)
    as_.layout = "stereo"
    n_frames = 0

    def put(arr):
        nonlocal n_frames
        fr = av.VideoFrame.from_ndarray(arr, format="rgb24")
        fr.pts = n_frames
        fr.time_base = Fraction(1, fps)
        for p in vs.encode(fr):
            out.mux(p)
        n_frames += 1

    for part in plan["parts"]:
        if "card" in part:
            arr = card_frame(part["card"], res)
            for _ in range(int(part["seconds"] * fps)):
                put(arr)
        else:
            src = av.open(part["clip"])
            rate = float(src.streams.video[0].average_rate)
            for i, fr in enumerate(src.decode(video=0)):
                t = i / rate
                if t < part.get("start", 0):
                    continue
                if t > part.get("end", 1e9):
                    break
                put(np.asarray(fr.to_image().resize(res, Image.LANCZOS)))
            src.close()
    for p in vs.encode():
        out.mux(p)
    # silent audio for the whole duration
    samples = int(n_frames / fps * 48000)
    chunk = 1024
    for s in range(0, samples, chunk):
        n = min(chunk, samples - s)
        af = av.AudioFrame.from_ndarray(np.zeros((1, n * 2), dtype=np.float32), format="flt", layout="stereo")
        af.sample_rate = 48000
        af.pts = s
        for p in as_.encode(af):
            out.mux(p)
    for p in as_.encode():
        out.mux(p)
    out.close()
    print(f"-> {sys.argv[2]}  {n_frames / fps:.1f}s")


if __name__ == "__main__":
    main()
