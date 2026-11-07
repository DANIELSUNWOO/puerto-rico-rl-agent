"""리그전(course_ladder.py) 기록에서 참가자끼리의 맞대결 표를 만듭니다.

각 칸은 "행의 에이전트가 열의 에이전트를 상대로 거둔 승수 / 판수" 입니다 (무승부는 0.5승).

사용법:
    python examples/study/head_to_head.py results/ladder/2026-11-07/games
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict


def main(folder):
    pts = defaultdict(float)      # (a, b) -> a가 b에게 얻은 승점
    games = defaultdict(int)      # (a, b) -> 판수
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        a, b = rec["player_labels"]
        winners = rec["winners"]
        for seat, me, opp in ((0, a, b), (1, b, a)):
            games[(me, opp)] += 1
            if seat in winners:
                pts[(me, opp)] += 1.0 / len(winners)

    names = sorted({k[0] for k in games})
    print("행이 열을 상대로 거둔 승률 (판수)\n")
    print(f"{'':10s}" + "".join(f"{n:>14s}" for n in names))
    for a in names:
        row = f"{a:10s}"
        for b in names:
            n = games.get((a, b), 0)
            row += f"{'-':>14s}" if n == 0 else f"{pts[(a, b)] / n:>9.1%} ({n:3d})"
        print(row)

    print("\n원본(Orig) 상대 승률과 95% 범위")
    for a in names:
        n = games.get((a, "Orig"), 0)
        if n:
            p = pts[(a, "Orig")] / n
            half = 1.96 * math.sqrt(p * (1 - p) / n)
            print(f"  {a:10s} {p:6.1%}  ({p - half:5.1%} ~ {p + half:5.1%}, {n}판)")


if __name__ == "__main__":
    main(sys.argv[1])