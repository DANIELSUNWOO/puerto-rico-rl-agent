"""
Moka: 2p 트랙 에이전트 (전남대학교 강화학습 기말 과제).

조교님이 제공한 PPO 체크포인트(training/checkpoints/ppo_baseline_2p.pt)에서
출발해, training/train_moka.py로 여러 상대(자기 자신, 과거 버전, 학생 대역
에이전트, 규칙 봇, SearchLite)를 섞어 이어서 학습한 정책 신경망을 씁니다.
학습 과정과 측정 결과는 docs/experiments/ 에 있습니다.

두는 방식: 매 수마다 정책 신경망이 고른, 가장 확률이 높은 합법 수를 둡니다.
탐색은 하지 않습니다 (한 수 평균 약 1 ms).

규칙 확인:
- weights.pt 하나를 이 파일 위치 기준으로, 처음 만들 때 한 번만 읽습니다.
- 모듈 변수에 두는 것은 그 신경망뿐이고, 게임 중에 바뀌지 않습니다.
  게임 사이에 넘기는 정보는 없습니다.
- 네트워크, 하위 프로세스, 파일 쓰기, 스레드 수 변경을 쓰지 않습니다.
"""
import os

import numpy as np
import torch

from agents.base import Agent
from agents.ppo_agent import PpoNetwork
from puerto_rico.observation import egocentric_view

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.pt")    # from __file__, never from the cwd

# The harness builds a new agent every game, so the file is read once per
# process and the network kept here.
_NET = None


def _net():
    global _NET
    if _NET is None:
        ckpt = torch.load(WEIGHTS, map_location="cpu")   # 조교 공지: 기본 설정으로 불러오기
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        net = PpoNetwork(obs_dim=int(state["embed.0.weight"].shape[1]))
        net.load_state_dict(state)
        net.eval()
        # [Moka] 예열: 첫 계산은 준비 시간 때문에 느릴 수 있어서, 게임 시작 전에 한 번 미리 계산해 둠
        with torch.no_grad():
            net.actor_logits(torch.zeros(1, net.embed[0].in_features))
        _NET = net
    return _NET


class Moka(Agent):
    name = "Moka"     # shown on the board; it may not be a baseline's name

    def __init__(self):
        super().__init__()
        self.net = _net()     # at construction, never inside act()

    def act(self, observation, action_mask):
        # train_ppo.py trains on the egocentric view (block 0 = the acting
        # player), so the same rotation must be applied here.
        obs = torch.as_tensor(egocentric_view(observation), dtype=torch.float32)
        with torch.no_grad():
            logits = self.net.actor_logits(obs.unsqueeze(0))[0].numpy()
        logits = np.where(action_mask > 0.5, logits, -np.inf)
        return int(np.argmax(logits))