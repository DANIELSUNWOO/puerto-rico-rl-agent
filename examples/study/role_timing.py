"""역할을 '헛되이' 고른 횟수를 셉니다. 리그전 기록을 다시 둬서 계산합니다.

- 헛된 생산자: 생산자를 골랐는데 내 상품이 하나도 늘지 않음 (일꾼이 없는 농장 등)
- 헛된 선장: 선장을 골랐는데 내가 실을 상품이 하나도 없음
- 생산자, 선장을 고른 단계에서 나와 상대가 얻은 상품 / 선적 점수도 같이 봄

선/후, 이긴 판/진 판으로 나누고, 첫 3라운드와 전체를 따로 보여 줍니다.

사용법:
    python examples/study/role_timing.py results/ladder/exp12 --agent E12b_60 --vs P1
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from puerto_rico.records import replay_record  # noqa: E402

CRAFTSMAN, CAPTAIN = 3, 5


def goods_total(game, seat):
    return sum(int(n) for n in game.players[seat].goods.values())


def analyze(rec, me):
    """한 판에서 me가 고른 생산자, 선장의 결과를 모음."""
    opp = 1 - me
    out = defaultdict(float)
    pending = {}          # 생산자/선장을 고른 직후 상태 (다음 역할 선택 때 비교)
    state = {"picks": 0}

    def close(game):
        if not pending:
            return
        kind, rnd, g_me, g_op, vp_me, vp_op = (pending[k] for k in
                                               ("kind", "round", "g_me", "g_op", "vp_me", "vp_op"))
        early = rnd <= 3
        tag = "early_" if early else ""
        if kind == CRAFTSMAN:
            gain_me = goods_total(game, me) - g_me
            gain_op = goods_total(game, opp) - g_op
            for t in ({tag, ""} if early else {""}):
                out[t + "craft"] += 1
                out[t + "craft_gain_me"] += gain_me
                out[t + "craft_gain_op"] += gain_op
                if gain_me <= 0:
                    out[t + "craft_wasted"] += 1
        else:
            gain_me = game.players[me].vp_chips - vp_me
            gain_op = game.players[opp].vp_chips - vp_op
            for t in ({tag, ""} if early else {""}):
                out[t + "cap"] += 1
                out[t + "cap_vp_me"] += gain_me
                out[t + "cap_vp_op"] += gain_op
                if pending["empty"]:
                    out[t + "cap_wasted"] += 1
        pending.clear()

    def on_step(i, seat, action, env):
        game = env.unwrapped.game
        if action <= 7:
            close(game)
            rnd = state["picks"] // 6 + 1
            state["picks"] += 1
            if seat == me and action in (CRAFTSMAN, CAPTAIN):
                pending.update(kind=action, round=rnd,
                               g_me=goods_total(game, me), g_op=goods_total(game, opp),
                               vp_me=game.players[me].vp_chips, vp_op=game.players[opp].vp_chips,
                               empty=goods_total(game, me) == 0)

    env = replay_record(rec, on_step=on_step)
    close(env.unwrapped.game)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--vs", action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seen = {}
    paths = []
    for folder in args.folders:
        paths += glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    needle = f'"{args.agent}"'
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        if needle not in text:
            continue
        rec = json.loads(text)
        labels = rec.get("player_labels", [])
        if args.agent not in labels or len(labels) != 2:
            continue
        opp = labels[1 - labels.index(args.agent)]
        if args.vs and opp not in args.vs:
            continue
        seen.setdefault((rec["seed"], tuple(labels)), rec)
    print(f"{args.agent}: {len(seen)}판을 다시 둡니다", flush=True)

    groups = defaultdict(lambda: defaultdict(float))
    for n, rec in enumerate(seen.values(), 1):
        if n % 500 == 0:
            print(f"  {n}/{len(seen)}", flush=True)
        labels = rec["player_labels"]
        me = labels.index(args.agent)
        opp = labels[1 - me]
        first = "선" if rec["actions"][0][0] == me else "후"
        won = "이긴 판" if rec["winners"] == [me] else "진 판"
        res = analyze(rec, me)
        for key in ((opp, first, won), (opp, first, "전체")):
            g = groups[key]
            g["games"] += 1
            for k, v in res.items():
                g[k] += v

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    print(f"{args.agent}: 생산자, 선장을 고른 결과 (한 판 평균)", file=out)
    for opp in sorted({k[0] for k in groups}):
        print(f"\n[상대 {opp}]", file=out)
        for first in ("선", "후"):
            for won in ("전체", "이긴 판", "진 판"):
                g = groups.get((opp, first, won))
                if not g:
                    continue
                n = g["games"]
                a = lambda k: g[k] / n  # noqa: E731
                print(f"  {first} {won} {int(n)}판", file=out)
                print(f"    첫 3라운드: 생산자 {a('early_craft'):.2f}번 (헛된 것 {a('early_craft_wasted'):.2f}, "
                      f"내 상품 +{a('early_craft_gain_me'):.1f}, 상대 +{a('early_craft_gain_op'):.1f}) / "
                      f"선장 {a('early_cap'):.2f}번 (헛된 것 {a('early_cap_wasted'):.2f}, "
                      f"내 선적 +{a('early_cap_vp_me'):.1f}, 상대 +{a('early_cap_vp_op'):.1f})", file=out)
                print(f"    전체:       생산자 {a('craft'):.2f}번 (헛된 것 {a('craft_wasted'):.2f}, "
                      f"내 상품 +{a('craft_gain_me'):.1f}, 상대 +{a('craft_gain_op'):.1f}) / "
                      f"선장 {a('cap'):.2f}번 (헛된 것 {a('cap_wasted'):.2f}, "
                      f"내 선적 +{a('cap_vp_me'):.1f}, 상대 +{a('cap_vp_op'):.1f})", file=out)
    if args.out:
        out.close()
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()