"""두 리그전 결과 폴더에서 같은 판(같은 시드, 같은 좌석, 같은 상대)의 승패가 같은지 비교합니다.

같은 에이전트, 같은 시드인데 결과가 다르면 어느 한쪽이 다른 조건에서 둔 것입니다.
(예: 컴퓨터가 바빠서 탐색 봇이 시간 제한에 걸려 덜 계산한 경우)

사용법:
    python examples/study/compare_runs.py results/ladder/finalC3_1/games results/ladder/finalC3_check/games
"""
import glob
import json
import os
import sys


def load(folder):
    games = {}
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        key = (rec["seed"], tuple(rec["player_labels"]))
        games[key] = sorted(rec["winners"])
    return games


def main(a, b):
    ga, gb = load(a), load(b)
    common = sorted(set(ga) & set(gb))
    if not common:
        sys.exit("겹치는 판이 없습니다. 시드와 참가자를 확인해 주세요.")
    diff = [k for k in common if ga[k] != gb[k]]
    print(f"겹치는 판: {len(common)}판")
    print(f"결과가 같은 판: {len(common) - len(diff)}판")
    print(f"결과가 다른 판: {len(diff)}판")
    for seed, labels in diff[:20]:
        print(f"  시드 {seed}, 좌석 {labels}: {ga[(seed, labels)]} vs {gb[(seed, labels)]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])