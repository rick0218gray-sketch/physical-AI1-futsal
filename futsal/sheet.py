"""Contact sheet of an mp4: python sheet.py in.mp4 out.png [every] [cols]"""
import sys

import av
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 3
imgs = [f.to_image() for i, f in enumerate(av.open(src).decode(video=0)) if i % every == 0][:cols * 3]
w, h = imgs[0].size
sheet = Image.new("RGB", (w * cols, h * ((len(imgs) + cols - 1) // cols)))
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out)
print(len(imgs), sheet.size)
