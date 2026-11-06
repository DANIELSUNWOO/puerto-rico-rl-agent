"""기록된 게임 한 판을 라운드 단위로 요약합니다.

라운드마다 누가 어떤 역할을 골랐는지, 누가 무슨 건물을 지었는지,
라운드가 끝났을 때 각자의 선적 점수, 돈, 상품, 건물 수가 어떤지 보여 줍니다.

사용법:
    python examples/study/game_timeline.py results/games/<파일>.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from puerto_rico.constants import BUILDING_DATA, BuildingType  # noqa: E402
from puerto_rico.records import load_record, replay_record  # noqa: E402

ROLES = ["개척자", "모집관", "건축가", "생산자", "상인", "선장", "광부", "광부2"]
GOODS = ["커피", "담배", "옥수수", "설탕", "인디고"]


def building_name(bt):
    name = BuildingType(bt).name
    return f"{name}(대형)" if BUILDING_DATA[BuildingType(bt)][4] else name


def player_state(game, seat):
    p = game.players[seat]
    goods = ", ".join(f"{GOODS[int(g)]}{n}" for g, n in p.goods.items() if n > 0) or "없음"
    real = [b for b in p.city_board if b.building_type != BuildingType.OCCUPIED_SPACE]
    staffed_large = sum(1 for b in real if BUILDING_DATA[b.building_type][4] and b.colonists > 0)
    return (f"선적점수 {p.vp_chips:2d} | 돈 {p.doubloons:2d} | 상품 {goods} | "
            f"건물 {len(real)}개 (대형 중 일꾼 있는 것 {staffed_large})")


def main(path):
    rec = load_record(path)
    names = rec["player_labels"]
    print(f"{os.path.basename(path)}  seed {rec['seed']}")
    print(f"  좌석 0: {names[0]}   좌석 1: {names[1]}   승자: {', '.join(names[w] for w in rec['winners'])}")

    state = {"picks": [], "builds": [], "round": 1}

    def print_round(env):
        game = env.unwrapped.game
        print(f"\n[라운드 {state['round']}]")
        print("  역할: " + " → ".join(f"{names[s]}:{r}" for s, r in state["picks"]))
        if state["builds"]:
            print("  건설: " + ", ".join(f"{names[s]} {b}" for s, b in state["builds"]))
        for seat in (0, 1):
            print(f"  {names[seat]:14s} {player_state(game, seat)}")
        state["picks"], state["builds"] = [], []
        state["round"] += 1

    def on_step(i, seat, action, env):
        if action <= 7:                                   # 역할 선택
            if len(state["picks"]) == 6:                  # 6번 고르면 라운드 끝
                print_round(env)
            state["picks"].append((seat, ROLES[action]))
        elif 16 <= action <= 38:                          # 건설
            state["builds"].append((seat, building_name(action - 16)))

    env = replay_record(rec, on_step=on_step)
    print_round(env)

    print("\n[최종 점수] (총점, 동점처리, 선적, 건물, 보너스)")
    for seat, s in enumerate(env.unwrapped.game.get_scores()):
        print(f"  {names[seat]:14s} {tuple(int(x) for x in s)}")


if __name__ == "__main__":
    main(sys.argv[1])