"""리그전 기록에서 선(첫 총독, 처음 역할을 고르는 쪽)일 때와 후일 때의 승률을 나눠서 봅니다.

좌석 번호(0, 1)가 아니라, 실제로 첫 역할을 고른 쪽이 누구인지로 나눕니다.

사용법:
    python examples/study/seat_split.py results/ladder/finalA3/games --agent E6_60
"""
import argparse
import glob
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from puerto_rico.records import load_record, replay_record  # noqa: E402


def first_mover(rec):
    seat = {}

    def on_step(i, s, action, env):
        if i == 0:
            seat["first"] = s

    replay_record(rec, on_step=on_step, stop_after=1)
    return seat["first"]


def ci(p, n):
    h = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0
    return max(0.0, p - h), min(1.0, p + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--agent", required=True)
    args = ap.parse_args()

    # (상대, 선/후) -> [승점, 판수]
    stats = defaultdict(lambda: [0.0, 0])
    for path in sorted(glob.glob(os.path.join(args.folder, "*.json"))):
        rec = load_record(path)
        labels = rec["player_labels"]
        if args.agent not in labels:
            continue
        me = labels.index(args.agent)
        opp = labels[1 - me]
        winners = rec["winners"]
        pts = (1.0 if len(winners) == 1 else 0.5) if me in winners else 0.0
        side = "선" if first_mover(rec) == me else "후"
        for key in ((opp, side), ("(전체)", side)):
            stats[key][0] += pts
            stats[key][1] += 1

    opps = sorted({o for o, _ in stats if o != "(전체)"}) + ["(전체)"]
    print(f"{args.agent}의 선/후별 승률 (95% 범위)")
    for opp in opps:
        line = f"  {opp:14s}"
        for side in ("선", "후"):
            w, n = stats[(opp, side)]
            if n:
                p = w / n
                lo, hi = ci(p, n)
                line += f"  {side} {p:6.1%} ({lo:.1%}~{hi:.1%}, {n}판)"
        print(line)


if __name__ == "__main__":
    main()