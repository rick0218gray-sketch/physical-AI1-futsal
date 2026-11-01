"""Coach: Japanese tactical instruction -> tactic vector via a local LLM (Ollama, JSON-schema constrained).

  attack       0..1  0 = player drops back when the ball is in the opponent half, 1 = always chase
  keeper_depth 0..1  0 = keeper stays on the goal line, 1 = keeper comes ~0.9 m off the line
  press        0..1  field player speed (0 -> 0.6 m/s, 1 -> 1.2 m/s)

usage:
  python coach.py --build tactics.json            # convert the instruction list below (cached for training)
  python coach.py "守りを固めて、カウンターを狙え"    # one instruction (used at half time in the demo match)
"""
import argparse
import json
import os
import time
import urllib.request

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL = os.environ.get("COACH_MODEL", "qwen2.5:7b-instruct")

SYSTEM = """あなたは2対2のロボットフットサル（各チーム キーパー1体＋フィールドプレーヤー1体）の監督です。
監督の日本語の指示を、次の3つの作戦パラメータ（0.0〜1.0の実数）に変換してください。
- attack: フィールドプレーヤーの攻撃性。0=ボールが相手陣にあるときは自陣に下がって守る、1=常にボールを追って攻める
- keeper_depth: キーパーの位置。0=ゴールライン上に留まる、1=ゴールから約0.9m前に出る
- press: フィールドプレーヤーの速さと寄せの強さ。0=ゆっくり（0.6m/s）、1=全速（1.2m/s）
指示に書かれていない項目は0.5にしてください。summaryには解釈を日本語で短く書いてください。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "attack": {"type": "number"},
        "keeper_depth": {"type": "number"},
        "press": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["attack", "keeper_depth", "press", "summary"],
}

INSTRUCTIONS = [
    "全員で攻めろ。とにかく点を取りに行け",
    "守りを固めて、カウンターを狙え",
    "キーパーは前に出てシュートコースを狭めろ",
    "キーパーはゴールラインから動くな",
    "相手に激しくプレッシャーをかけろ",
    "落ち着いて、ゆっくりボールを運べ",
    "リードしているので時間を使って守り切れ",
    "負けているから前に出て攻め続けろ",
    "バランスよく戦え",
    "前半は様子見で、無理に攻めるな",
    "ボールを奪ったら全速で相手ゴールへ",
    "キーパーも積極的に飛び出してボールを処理しろ",
    "自陣に引いて、相手のミスを待て",
    "ハイプレスで相手陣内でボールを奪え",
    "キーパーは少しだけ前に出て、プレーヤーは攻撃的に",
    "体力を温存して、ほどほどに攻めろ",
    "とにかく失点しないことを最優先にしろ",
    "一点取りに行くぞ、リスクを取れ",
    "相手のキーパーが前に出ているから、どんどん攻めろ",
    "慎重に守りながら、チャンスがあれば攻めろ",
    "全速力でボールを追いかけろ",
    "ゴール前を固めて、キーパーは下がれ",
    "攻撃も守備も中くらいで",
    "最後の1分だ、全員攻撃",
]


def ask(instruction, timeout=120):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(json.loads(r.read())["message"]["content"])
    for k in ("attack", "keeper_depth", "press"):
        out[k] = min(1.0, max(0.0, float(out[k])))
    out["instruction"] = instruction
    out["latency_s"] = round(time.time() - t0, 2)
    out["model"] = MODEL
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="?")
    ap.add_argument("--build")
    a = ap.parse_args()
    if a.build:
        res = []
        for ins in INSTRUCTIONS:
            r = ask(ins)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            res.append(r)
        with open(a.build, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    else:
        print(json.dumps(ask(a.instruction), ensure_ascii=False))


if __name__ == "__main__":
    main()
