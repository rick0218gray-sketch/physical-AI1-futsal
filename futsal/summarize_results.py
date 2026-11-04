"""Print the result tables from the files stored in the repository (no simulator, GPU or third-party package needed).

  results/eval.json            -> win / draw / loss table (same layout as the README)
  models/bc/bc_stats.json      -> imitation-learning dataset and validation MSE
  models/futsal-rl/progress.csv -> last logged iteration of the self-play RL run

usage: python futsal/summarize_results.py [--eval results/eval.json] [--bc models/bc/bc_stats.json]
                                          [--progress models/futsal-rl/progress.csv]
"""
import argparse
import csv
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# labels used in the README table
TEAM_JP = {"rl": "強化学習後", "bc": "模倣学習のみ", "teacher": "お手本"}


def pairing_label(a, b):
    left = TEAM_JP.get(a, a)
    right = TEAM_JP.get(b, b)
    if a == "rl" and b == "teacher":
        right += "（ルール＋監督）"
    if a == b == "teacher":
        right += "（確認用）"
    return f"{left} 対 {right}"


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_rows(ev):
    """One row per pairing: (label, wins, draws, losses, goals_for, goals_against, falls_own, falls_opp)."""
    return [(pairing_label(r["a"], r["b"]), r["a_wins"], r["draws"], r["b_wins"], r["a_goals"], r["b_goals"],
             r["a_falls"], r["b_falls"]) for r in ev["results"]]


def eval_table(ev):
    lines = ["| 対戦 | 勝ち | 引き分け | 負け | 得点 − 失点 | 転倒（自 / 相手） |",
             "|---|---|---|---|---|---|"]
    for lab, w, d, l, gf, ga, fa, fb in eval_rows(ev):
        lines.append(f"| {lab} | {w} | {d} | {l} | {gf} − {ga} | {fa} / {fb} |")
    return "\n".join(lines)


def check_eval(ev):
    """Internal consistency of eval.json. Returns a list of problems (empty = OK)."""
    bad = []
    for r in ev["results"]:
        tag = f"{r['a']} vs {r['b']}"
        m = r["matches"]
        if r["a_wins"] + r["draws"] + r["b_wins"] != m:
            bad.append(f"{tag}: wins + draws + losses != matches")
        if m != 2 * ev["n_per_side"]:
            bad.append(f"{tag}: matches != 2 * n_per_side")
        if round((r["a_wins"] + 0.5 * r["draws"]) / m, 3) != r["a_points_rate"]:
            bad.append(f"{tag}: a_points_rate does not match the counts")
        if round(r["a_goals"] / m, 3) != r["a_goals_per_match"] or round(r["b_goals"] / m, 3) != r["b_goals_per_match"]:
            bad.append(f"{tag}: goals_per_match does not match the counts")
        # a team needs at least one goal per win
        if r["a_goals"] < r["a_wins"] or r["b_goals"] < r["b_wins"]:
            bad.append(f"{tag}: fewer goals than wins")
    return bad


def read_progress(path):
    """progress.csv of rl_train.py; a resumed run re-logs some iterations, keep the last row per iteration."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_iter = {}
    for r in rows:
        by_iter[int(r["iter"])] = r
    return [by_iter[k] for k in sorted(by_iter)], len(rows) - len(by_iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(ROOT, "results", "eval.json"))
    ap.add_argument("--bc", default=os.path.join(ROOT, "models", "bc", "bc_stats.json"))
    ap.add_argument("--progress", default=os.path.join(ROOT, "models", "futsal-rl", "progress.csv"))
    a = ap.parse_args()

    ev = load_eval(a.eval)
    print(f"## 評価（{os.path.relpath(a.eval, ROOT)}：1試合{ev['match_s']:g}秒、サイドごとに{ev['n_per_side']}試合）\n")
    print(eval_table(ev))
    for r in ev["results"]:
        print(f"- {r['a']} vs {r['b']}: 勝ち点率 {r['a_points_rate']}（引き分け=0.5）, "
              f"1試合あたり得点 {r['a_goals_per_match']} / 失点 {r['b_goals_per_match']}")
    bad = check_eval(ev)
    print("\n整合性チェック:", "OK" if not bad else "; ".join(bad))

    if os.path.exists(a.bc):
        with open(a.bc, encoding="utf-8") as f:
            bc = json.load(f)
        mse = bc["val_mse"]
        print(f"\n## 模倣学習（{os.path.relpath(a.bc, ROOT)}）\n")
        print(f"- 学習データ {bc['samples']:,} 組、終了した試合 {bc['matches_finished']:,}、"
              f"シミュレーション時間 {bc['sim_seconds'] / 3600:.1f} 時間、実時間 {bc['wall_s'] / 60:.1f} 分")
        print(f"- 検証MSE {mse[0]:.3f} → {mse[-1]:.3f}（{len(mse)} エポック）")

    if os.path.exists(a.progress):
        rows, dup = read_progress(a.progress)
        last = rows[-1]
        print(f"\n## 強化学習の記録（{os.path.relpath(a.progress, ROOT)}）\n")
        print(f"- 記録 {len(rows)} 点（再開で重複した {dup} 行は後の値を採用）、最終反復 {last['iter']}")
        print(f"- 最終10反復：得点/分 {float(last['gf_per_min']):.3f}、失点/分 {float(last['ga_per_min']):.3f}、"
              f"対お手本 {float(last['gf_vs_teacher']):.0f}−{float(last['ga_vs_teacher']):.0f}、"
              f"対自分 {float(last['gf_vs_self']):.0f}−{float(last['ga_vs_self']):.0f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
