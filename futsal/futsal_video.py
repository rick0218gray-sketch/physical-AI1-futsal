"""Render one match (env 0) to mp4 with a scoreboard overlay."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/mnt/c/Windows/Fonts/meiryob.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/YuGothB.ttc"]
_cache = {}


def font(size):
    if size not in _cache:
        for p in FONTS:
            try:
                _cache[size] = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
        else:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


class Recorder:
    def __init__(self, path, res, fps=25):
        self.c = av.open(path, mode="w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height = res
        self.s.pix_fmt = "yuv420p"
        self.s.options = {"crf": "20"}

    def add(self, rgb, lines_top=(), lines_bottom=(), score=None, names=("青", "赤")):
        img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        W, H = img.size
        k = W / 1280  # scale text with the frame width
        if score is not None:
            txt = f"{names[0]}  {score[0]} - {score[1]}  {names[1]}"
            f = font(int(36 * k))
            tw = d.textlength(txt, font=f)
            d.rectangle([(W - tw) / 2 - 16 * k, 6 * k, (W + tw) / 2 + 16 * k, 56 * k], fill=(20, 26, 40, 200))
            d.text(((W - tw) / 2, 10 * k), txt, font=f, fill=(255, 255, 255))
        y = 66 * k
        for t in lines_top:
            f = font(int(30 * k))
            tw = d.textlength(t, font=f)
            d.rectangle([(W - tw) / 2 - 12 * k, y - 4 * k, (W + tw) / 2 + 12 * k, y + 40 * k], fill=(20, 26, 40, 190))
            d.text(((W - tw) / 2, y), t, font=f, fill=(255, 220, 140))
            y += 48 * k
        step = 34 * k
        y = H - step * len(lines_bottom) - 6 * k
        for t in lines_bottom:
            f = font(int(20 * k))
            d.rectangle([8 * k, y - 3 * k, 18 * k + d.textlength(t, font=f), y + 27 * k], fill=(20, 26, 40, 170))
            d.text((13 * k, y), t, font=f, fill=(235, 235, 235))
            y += step
        frame = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
        for pkt in self.s.encode(frame):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()
