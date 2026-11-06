"""기록된 게임들을 다시 재생해서, 이긴 판과 진 판의 점수 구성과 역할 선택을 비교합니다.

사용법 (먼저 play.py --record 로 게임을 기록한 뒤):
    python examples/study/analyze_losses.py results/games --agent Moka
"""
import argparse
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from puerto_rico.records import load_record, replay_record  # noqa: E402

ROLES = ["개척자", "모집관", "건축가", "생산자", "상인", "선장", "광부", "광부2"]


def analyze(path, agent):
    rec = load_record(path)
    me = rec["player_labels"].index(agent)
    opp = 1 - me
    my_roles = Counter()

    def on_step(i, seat, action, env):
        if seat == me and action <= 7:          # 0~7번은 역할 선택
            my_roles[ROLES[action]] += 1

    env = replay_record(rec, on_step=on_step)
    # (총점, 동점처리값, 선적, 건물, 보너스)
    scores = env.unwrapped.game.get_scores()
    rounds = sum(1 for _, a in rec["actions"] if a <= 7) / 6   # 2인: 라운드당 역할 선택 6번
    return {
        "opp_name": rec["player_labels"][opp],
        "won": me in rec["winners"] and len(rec["winners"]) == 1,
        "me": scores[me], "opp": scores[opp],
        "roles": my_roles, "rounds": rounds,
        "file": os.path.basename(path),
    }


def summarize(games, title):
    n = len(games)
    if n == 0:
        return
    avg = lambda f: sum(f(g) for g in games) / n
    print(f"\n[{title}] {n}판")
    print(f"  라운드 수 평균: {avg(lambda g: g['rounds']):.1f}")
    print(f"  {'':10s}{'총점':>6s}{'선적':>6s}{'건물':>6s}{'보너스':>7s}")
    for who in ("me", "opp"):
        label = "나" if who == "me" else "상대"
        print(f"  {label:10s}" + "".join(f"{avg(lambda g, k=k: g[who][k]):6.1f}" for k in (0, 2, 3, 4)))
    total = Counter()
    for g in games:
        total.update(g["roles"])
    print("  내가 고른 역할 (한 판 평균): " +
          ", ".join(f"{r} {total[r] / n:.1f}" for r in ROLES[:7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--agent", default="Moka")
    args = ap.parse_args()

    results = [analyze(p, args.agent) for p in sorted(glob.glob(os.path.join(args.folder, "*.json")))]
    for opp in sorted({r["opp_name"] for r in results}):
        rs = [r for r in results if r["opp_name"] == opp]
        print(f"\n===== 상대: {opp} =====")
        summarize([r for r in rs if r["won"]], "이긴 판")
        summarize([r for r in rs if not r["won"]], "진 판")
        print("  진 판 파일:", ", ".join(r["file"] for r in rs if not r["won"])[:300])


if __name__ == "__main__":
    main()