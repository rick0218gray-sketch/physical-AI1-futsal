"""The numbers written in README.md / docs/report.md must match the stored result files exactly."""
import json
import os
import re
import subprocess
import sys

import pytest

import summarize_results as sr

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def ev():
    return sr.load_eval(os.path.join(ROOT, "results", "eval.json"))


def table_rows(md, first_cell_contains):
    """Rows of the Markdown table whose first cell contains `first_cell_contains`, as lists of integers."""
    out = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        if first_cell_contains in cells[0] and " 対 " in cells[0]:
            out.append((cells[0], [int(x) for c in cells[1:] for x in re.findall(r"\d+", c)]))
    return out


def expected(ev, with_falls):
    rows = []
    for lab, w, d, l, gf, ga, fa, fb in sr.eval_rows(ev):
        rows.append((lab, [w, d, l, gf, ga] + ([fa, fb] if with_falls else [])))
    return rows


def test_eval_json_is_internally_consistent(ev):
    assert sr.check_eval(ev) == []
    assert ev["match_s"] == 60.0 and ev["n_per_side"] == 128


def test_readme_result_table_matches_eval_json(ev):
    rows = table_rows(read("README.md"), "")
    assert rows == expected(ev, with_falls=True)


def test_report_result_table_matches_eval_json(ev):
    rows = table_rows(read("docs/report.md"), "")
    assert [r[1] for r in rows] == [r[1] for r in expected(ev, with_falls=False)]


def test_readme_derived_claims(ev):
    readme = read("README.md")
    rl_t = ev["results"][0]
    assert (rl_t["a"], rl_t["b"]) == ("rl", "teacher")
    ratio = rl_t["a_goals"] / rl_t["b_goals"]
    assert f"{ratio:.1f}倍" in readme  # 140 / 21
    tt = ev["results"][3]
    assert f"{tt['a_wins']}勝{tt['b_wins']}敗" in readme
    assert f"{rl_t['a_wins']}勝{rl_t['b_wins']}敗" in readme
    assert "256試合" in readme and "128試合" in readme


def test_readme_bc_numbers_match_bc_stats():
    bc = json.loads(read("models/bc/bc_stats.json"))
    readme = read("README.md")
    assert f"検証MSE {bc['val_mse'][0]:.3f} → {bc['val_mse'][-1]:.3f}" in readme
    assert f"{bc['samples'] / 1e4:.0f}万組" in readme  # 2,457,600 -> 246万
    assert f"{bc['matches_finished']:,}試合" in readme
    assert f"{bc['envs']}試合を並列" in readme


def test_rl_progress_and_config_match_readme():
    rows, dup = sr.read_progress(os.path.join(ROOT, "models", "futsal-rl", "progress.csv"))
    cfg = json.loads(read("models/futsal-rl/config.json"))
    readme = read("README.md")
    assert int(rows[-1]["iter"]) == cfg["iters"] == 500
    assert [int(r["iter"]) for r in rows] == list(range(10, 501, 10))
    assert dup == 2  # iterations 200 and 210 were logged twice (crash at 210, resumed from 189)
    assert cfg["start_iter"] == 190 and cfg["resume"].endswith("model_189.pt")
    assert f"{cfg['n']}試合並列、{cfg['iters']}反復" in readme
    assert os.path.exists(os.path.join(ROOT, "models", "futsal-rl", f"policy_{cfg['iters']}.pt"))


def test_readme_links_point_to_existing_files():
    readme = read("README.md")
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert os.path.exists(os.path.join(ROOT, target)), target
    assert "https://youtu.be/w7fTLleIA-0" in readme


def test_summary_script_runs_and_prints_readme_table(ev):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "futsal", "summarize_results.py")],
                         capture_output=True, text=True, check=True).stdout
    assert sr.eval_table(ev) in out
    assert "整合性チェック: OK" in out
    readme_table = table_rows(read("README.md"), "")
    assert readme_table == table_rows(out, "")
