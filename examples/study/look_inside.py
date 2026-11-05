"""한 게임의 처음 몇 결정에서 에이전트가 받는 관측과 가능한 행동을 출력합니다."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from puerto_rico import make_env, flatten_observation, describe_action

ROLES = ["Settler", "Mayor", "Builder", "Craftsman", "Trader", "Captain", "Prospector-1", "Prospector-2"]
GOODS = ["Coffee", "Tobacco", "Corn", "Sugar", "Indigo"]


def goods_str(v):
    return ", ".join(f"{g} {int(x)}" for g, x in zip(GOODS, v))


env = make_env(seed=0, num_players=2)
rng = np.random.default_rng(0)
for step in range(6):
    raw = env.observe(env.agent_selection)
    obs = flatten_observation(raw["observation"])
    mask = np.asarray(raw["action_mask"])
    me = int(obs[36])
    print(f"\n=== 결정 {step + 1}: player {me} 차례 ===")
    print("남은 역할:", [ROLES[i] for i in range(8) if obs[64 + i] > 0.5])
    for i in range(2):
        b = 79 + 73 * i
        print(f"  player {i}{' (나)' if i == me else ''}: 돈 {int(obs[b + 23])}, "
              f"상품 {goods_str(obs[b + 25:b + 30])}, VP {int(obs[b + 72])}")
    legal = np.where(mask > 0.5)[0]
    print("가능한 행동:")
    for a in legal:
        print(f"  {a:3d}: {describe_action(me, int(a))}")
    a = int(rng.choice(legal))
    print("-> 고른 행동:", a)
    env.step(a)