"""Learning curve of the self-play RL run: goals for / against per match-minute, vs teacher and vs self.

usage: python plot_progress.py logs/futsal-rl/progress.csv results/rl_progress.png
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Japanese font: Windows fonts under WSL, else Noto CJK (Debian/Ubuntu: apt install fonts-noto-cjk)
for p in ("/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/YuGothM.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    try:
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break
    except OSError:
        continue

df = pd.read_csv(sys.argv[1]).drop_duplicates("iter", keep="last").sort_values("iter")  # resumed run overlaps
roll = 5
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(df["iter"], df["gf_per_min"].rolling(roll, min_periods=1).mean(), label="得点 / 分", color="#2F6FDE")
ax[0].plot(df["iter"], df["ga_per_min"].rolling(roll, min_periods=1).mean(), label="失点 / 分", color="#D9662B")
ax[0].set_title("学習チームの得失点（全相手、5点移動平均）")
ax[0].set_xlabel("PPO反復")
ax[0].legend()
gd_t = (df["gf_vs_teacher"] - df["ga_vs_teacher"]).rolling(roll, min_periods=1).mean()
gd_s = (df["gf_vs_self"] - df["ga_vs_self"]).rolling(roll, min_periods=1).mean()
ax[1].plot(df["iter"], gd_t, label="対お手本（ルール＋監督）", color="#2F6FDE")
ax[1].plot(df["iter"], gd_s, label="対自分の過去のコピー", color="#7A869A")
ax[1].axhline(0, color="#999", lw=0.8)
ax[1].set_title("10反復ごとの得失点差（5点移動平均）")
ax[1].set_xlabel("PPO反復")
ax[1].legend()
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print("->", sys.argv[2])
