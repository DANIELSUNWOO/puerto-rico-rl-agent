"""리그전 기록에서 한 에이전트의 습관을 상대별로 정리합니다.

1. 선(첫 역할을 고르는 쪽)일 때와 후일 때의 승률
2. 역할 선택: 한 판에 나와 상대가 각 역할을 몇 번 골랐는지, 건축가를 누가 가져갔는지
3. 점수 구성: 이긴 판과 진 판에서 선적, 건물, 보너스 점수
4. 선/후별 역할: 선일 때와 후일 때 고른 역할, 첫 라운드에 처음 고른 역할, 선일 때 진 판

여러 폴더를 한 번에 받고, 하위 폴더의 games/*.json까지 모두 찾습니다.
같은 판(같은 시드, 같은 좌석, 같은 참가자)이 여러 리그전에 겹쳐 있으면 한 번만 셉니다.
1, 2는 기록 파일만 읽어서 빠릅니다. 3은 기록에 점수 구성이 없어서 게임을 다시 둬야
하므로 --scores를 줄 때만 계산합니다 (--vs로 상대를 좁혀서 쓰는 걸 권함).

사용법:
    python examples/study/habits.py results/ladder --agent X10_60 --out habits_x10.txt
    python examples/study/habits.py results/ladder --agent X10_60 --vs Search --scores --out habits_search.txt
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROLES = ["개척자", "모집관", "건축가", "생산자", "상인", "선장", "광부", "광부2"]
SHOW = [0, 1, 2, 3, 4, 5, 6]          # 광부2는 광부와 합쳐서 보여 줌


def find_games(folders, agents):
    """기록을 읽어서, 찾는 에이전트가 둔 판만 작게 요약해 둠.

    기록 전체(수만 판의 수 목록)를 메모리에 올리면 느려지므로, 역할 선택 횟수와
    승패만 남기고 수 목록은 버림. 점수 구성이 필요하면 나중에 파일을 다시 읽음.
    """
    seen = {}
    needles = [f'"{a}"' for a in agents]
    paths = []
    for folder in folders:
        paths += glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    print(f"기록 파일 {len(paths)}개를 읽습니다", flush=True)
    for i, path in enumerate(paths, 1):
        if i % 5000 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        if not any(n in text for n in needles):        # 찾는 에이전트가 없는 판은 바로 건너뜀
            continue
        try:
            rec = json.loads(text)
        except ValueError:
            continue
        if not isinstance(rec, dict) or "actions" not in rec or "player_labels" not in rec:
            continue
        key = (rec["seed"], tuple(rec["player_labels"]))
        if key in seen:
            continue
        roles = [Counter(), Counter()]
        n_roles = 0
        first_roles = [None, None]               # 각 좌석이 첫 라운드에 처음 고른 역할
        for st, a in rec["actions"]:
            if a <= 7:
                roles[st][6 if a == 7 else a] += 1
                n_roles += 1
                if first_roles[st] is None and n_roles <= 6:
                    first_roles[st] = 6 if a == 7 else a
        seen[key] = {
            "seed": rec["seed"], "player_labels": rec["player_labels"], "winners": rec["winners"],
            "first": rec["actions"][0][0], "roles": roles, "rounds": n_roles / 6, "_path": path,
            "first_roles": first_roles,
        }
    return list(seen.values())


def ci(p, n):
    h = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0
    return max(0.0, p - h), min(1.0, p + h)


def role_counts(rec, seat):
    return rec["roles"][seat]


_SCORES = {}


def full_scores(rec):
    """(총점, 동점처리, 선적, 건물, 보너스). 기록에 없으면 게임을 다시 둬서 구함."""
    key = (rec["seed"], tuple(rec["player_labels"]))
    if key not in _SCORES:
        from puerto_rico.records import load_record, replay_record
        full = load_record(rec["_path"])
        if all(len(x) >= 5 for x in full["scores"]):
            _SCORES[key] = full["scores"]
        else:
            env = replay_record(full)
            _SCORES[key] = [list(x) for x in env.unwrapped.game.get_scores()]
    return _SCORES[key]


def points(rec, me):
    w = rec["winners"]
    if me not in w:
        return 0.0
    return 1.0 if len(w) == 1 else 0.5


def report(agent, games, vs, scores):
    by_opp = defaultdict(list)
    for rec in games:
        labels = rec["player_labels"]
        if agent not in labels or len(labels) != 2:
            continue
        me = labels.index(agent)
        opp = labels[1 - me]
        if vs and opp not in vs:
            continue
        by_opp[opp].append((rec, me))

    if not by_opp:
        print(f"\n{agent}: 기록을 찾지 못했습니다. 이름(대소문자)을 확인해 주세요.")
        return

    print(f"\n{'=' * 70}\n{agent}\n{'=' * 70}")
    for opp in sorted(by_opp):
        rows = by_opp[opp]
        n = len(rows)
        pts = sum(points(r, me) for r, me in rows)
        print(f"\n[상대 {opp}] {n}판, 승률 {pts / n:.1%}")

        # 1. 선/후
        for name, first in (("선", True), ("후", False)):
            sub = [(r, me) for r, me in rows if (r["first"] == me) == first]
            if sub:
                p = sum(points(r, me) for r, me in sub) / len(sub)
                lo, hi = ci(p, len(sub))
                print(f"  {name}: {p:6.1%} ({lo:.1%} ~ {hi:.1%}, {len(sub)}판)")

        # 2. 역할 선택
        mine, theirs = Counter(), Counter()
        for r, me in rows:
            mine.update(role_counts(r, me))
            theirs.update(role_counts(r, 1 - me))
        print("  역할 (한 판 평균)   " + "".join(f"{ROLES[i]:>6s}" for i in SHOW))
        print("    나               " + "".join(f"{mine[i] / n:6.1f}" for i in SHOW))
        print("    상대             " + "".join(f"{theirs[i] / n:6.1f}" for i in SHOW))
        total_b = mine[2] + theirs[2]
        if total_b:
            print(f"  건축가 중 내가 가져간 비율: {mine[2] / total_b:.1%} (나 {mine[2] / n:.1f}번, 상대 {theirs[2] / n:.1f}번)")

        # 3. 이긴 판 / 진 판 점수와 역할
        losses = [r for r, me in rows if points(r, me) < 1.0]
        if losses:
            print(f"  이기지 못한 판 {len(losses)}개. 처음 3개 (game_timeline.py로 열어 볼 수 있음):")
            for r in sorted(losses, key=lambda r: r["seed"])[:3]:
                print(f"    {r['_path']}")
        for name, won in (("이긴 판", True), ("진 판", False)):
            sub = [(r, me) for r, me in rows if (points(r, me) == 1.0) == won]
            if not sub:
                continue
            k = len(sub)
            rounds = sum(r["rounds"] for r, _ in sub) / k
            mb = sum(role_counts(r, me)[2] for r, me in sub) / k
            ob = sum(role_counts(r, 1 - me)[2] for r, me in sub) / k
            line = f"  {name} {k}판: 라운드 {rounds:.1f}, 건축가 나 {mb:.1f} 상대 {ob:.1f}"
            if scores:
                avg = lambda who, idx: sum(full_scores(r)[who(me)][idx] for r, me in sub) / k
                my = [avg(lambda me: me, i) for i in (0, 2, 3, 4)]
                op = [avg(lambda me: 1 - me, i) for i in (0, 2, 3, 4)]
                line += (f"\n      총점/선적/건물/보너스  나 {my[0]:.1f}/{my[1]:.1f}/{my[2]:.1f}/{my[3]:.1f}"
                         f"  상대 {op[0]:.1f}/{op[1]:.1f}/{op[2]:.1f}/{op[3]:.1f}")
            print(line)
        seat_report(rows)


def seat_report(rows):
    """4. 선/후별 역할 선택과 첫 역할, 선일 때 진 판"""
    print("  [선/후별]")
    for name, first in (("선", True), ("후", False)):
        sub = [(r, me) for r, me in rows if (r["first"] == me) == first]
        if not sub:
            continue
        k = len(sub)
        mine = Counter()
        for r, me in sub:
            mine.update(r["roles"][me])
        print(f"    {name} {k}판, 역할 평균  " + " ".join(f"{ROLES[i]} {mine[i] / k:.1f}" for i in SHOW))
        for label, won in (("이긴 판", True), ("진 판", False)):
            part = [(r, me) for r, me in sub if (points(r, me) == 1.0) == won]
            if not part:
                continue
            fr = Counter(r["first_roles"][me] for r, me in part)
            top = ", ".join(f"{ROLES[i]} {c / len(part):.0%}" for i, c in fr.most_common(4) if i is not None)
            print(f"      {label} {len(part)}판, 첫 라운드에 처음 고른 역할: {top}")
    lost_first = sorted((r for r, me in rows if r["first"] == me and points(r, me) < 1.0), key=lambda r: r["seed"])
    if lost_first:
        print("    선일 때 진 판 (처음 3개):")
        for r in lost_first[:3]:
            print(f"      {r['_path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--agent", action="append", required=True)
    ap.add_argument("--vs", action="append", default=None, help="이 상대만 보기 (여러 번 가능)")
    ap.add_argument("--scores", action="store_true", help="점수 구성까지 (게임을 다시 둬서 느림)")
    ap.add_argument("--out", default=None, help="결과를 이 파일에 저장 (진행 상황은 화면에)")
    args = ap.parse_args()

    games = find_games(args.folders, args.agent)
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    if out:
        sys.stdout, console = out, sys.stdout
    print(f"찾는 에이전트가 둔 판 {len(games)}판 (겹치는 판은 한 번만)")
    for agent in args.agent:
        report(agent, games, args.vs, args.scores)
    if out:
        sys.stdout = console
        out.close()
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()