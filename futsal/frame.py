"""Save the frame at time t (seconds) of an mp4 as png: python frame.py in.mp4 t out.png"""
import sys

import av

src, t, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
c = av.open(src)
rate = float(c.streams.video[0].average_rate)
for i, f in enumerate(c.decode(video=0)):
    if i / rate >= t:
        f.to_image().save(out)
        print("->", out, f"{i / rate:.1f}s")
        break
