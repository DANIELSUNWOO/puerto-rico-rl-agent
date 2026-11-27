"""
Moka: 2p 트랙 에이전트 (전남대학교 강화학습 기말 과제).

조교님이 제공한 PPO 체크포인트(training/checkpoints/ppo_baseline_2p.pt)에서 출발해,
training/train_moka.py로 여러 상대(자기 자신, 과거 버전, 학생 대역 에이전트, 규칙 봇,
SearchLite)를 섞어 이어서 학습한 정책·가치 신경망(Moka v8)을 씁니다.
학습 과정과 측정 결과는 docs/experiments/ 에 있습니다.

두는 방식: 학습한 신경망을 짧은 탐색 안에서 씁니다.
- 내 차례 마디는 정책 확률 상위 3개, 상대 차례 마디는 (상대 입장에서 본) 정책 상위 3개만 펼칩니다.
- 마디 값은 가치 신경망(내 입장). 내 차례는 가장 좋은 값, 상대 차례는 가장 나쁜 값을 고릅니다.
  상대가 우리 정책대로 둔다고 가정하지 않고, 그럴듯한 수 중 가장 나쁜 경우를 봅니다.
- 3수 앞까지 한 층씩 묶어서 계산하고, 시간(30 ms)이 넘치면 끝까지 계산한 층의 결론을 씁니다.
- 탐색 결과가 정책 1순위보다 0.05 이상 좋을 때만 수를 바꿉니다. 1순위 확률이 0.95 이상이면 탐색하지 않습니다.

규칙 확인:
- 시뮬레이션은 forward model의 clone()에서만 합니다. clone()이 숨겨진 농장 순서를 섞어 주므로
  실제 순서를 읽지 않고, 실제 게임 상태도 바꾸지 않습니다.
- weights.pt 하나를 이 파일 위치 기준으로, 처음 만들 때 한 번만 읽습니다. 모듈 변수에 두는 것은
  그 신경망뿐이고 게임 중에 바뀌지 않습니다. 탐색 나무는 수마다 버리며, 게임 사이에 넘기는 정보는 없습니다.
- 탐색하는 동안만 Python의 순환 참조 정리(gc)를 멈추고 끝나면 바로 되돌립니다 (복사본을 많이 만들 때
  가끔 생기는 긴 정리 때문에 시간 제한을 넘기지 않도록). 스레드 수는 바꾸지 않습니다.
- 네트워크, 하위 프로세스, 파일 쓰기를 쓰지 않습니다.
"""
import gc
import os
import time

import numpy as np
import torch

from agents.base import Agent
from agents.ppo_agent import PpoNetwork
from puerto_rico.observation import egocentric_view

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.pt")

_NET = None


def _net():
    global _NET
    if _NET is None:
        ckpt = torch.load(WEIGHTS, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        net = PpoNetwork(obs_dim=int(state["embed.0.weight"].shape[1]))
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            f = net._shared_features(torch.zeros(4, net.embed[0].in_features))
            net.actor_head(f)
            net.critic_head(f)
        _NET = net
    return _NET


class _TimeUp(Exception):
    pass


class Moka(Agent):
    name = "Moka"

    K_ME = 3
    K_OPP = 3
    DEPTH = 3          # 결정 수 (내 것과 상대 것을 합쳐서)
    BUDGET_S = 0.030
    MARGIN = 0.05
    SURE = 0.95        # 정책 1순위 확률이 이보다 크면 탐색하지 않음

    def __init__(self):
        super().__init__()
        self.net = _net()
        self.fm = None

    def on_game_start(self, forward_model=None):
        self.fm = forward_model

    def _eval(self, obs_list):
        x = torch.as_tensor(np.stack(obs_list), dtype=torch.float32)
        with torch.no_grad():
            f = self.net._shared_features(x)
            logits = self.net.actor_head(f).numpy()
            values = self.net.critic_head(f)[:, 0].numpy()
        return logits, values

    def act(self, observation, action_mask):
        mask = np.asarray(action_mask) > 0.5
        legal = np.flatnonzero(mask)
        obs = egocentric_view(observation)
        logits, _ = self._eval([obs])
        lg = np.where(mask, logits[0], -np.inf)
        greedy = int(np.argmax(lg))
        if len(legal) <= 1 or self.fm is None:
            return greedy
        p = np.exp(lg - lg.max())
        p /= p.sum()
        if p[greedy] >= self.SURE:
            return greedy
        # 탐색 중에는 순환 참조 정리(gc)를 멈춤: 복사본을 많이 만들 때 가끔 수십 ms 걸리는 정리가
        # 탐색 한가운데에서 일어나 시간 제한을 넘기는 것을 막기 위함. 끝나면 바로 되돌림.
        was = gc.isenabled()
        gc.disable()
        try:
            return self._search(lg, greedy)
        except _TimeUp:
            return greedy
        finally:
            if was:
                gc.enable()

    def _top(self, lg, k):
        idx = np.argsort(-lg)[:k]
        return [int(a) for a in idx if np.isfinite(lg[a])]

    def _search(self, root_lg, greedy):
        """한 층씩 깊게 보고, 끝까지 계산한 가장 깊은 층의 결과를 씀 (시간이 넘치면 그 결과로)."""
        deadline = time.perf_counter() + self.BUDGET_S
        me = self.fm.current_player()
        root = self.fm.clone()
        cands = self._top(root_lg, self.K_ME)
        if greedy not in cands:
            cands.append(greedy)

        def check():
            if time.perf_counter() > deadline:
                raise _TimeUp

        def backup(n):
            if not n["kids"]:
                return n["v"]
            vals = [backup(c) for c in n["kids"]]
            return max(vals) if n["max"] else min(vals)

        def decide():
            scores = {n["a"]: backup(n) for n in nodes}
            best = max(scores, key=scores.get)
            if best != greedy and scores[best] - scores[greedy] > self.MARGIN:
                return best
            return greedy

        nodes, frontier = [], []
        for a in cands:
            check()
            m = root.clone()
            m.step(a)
            node = {"m": m, "kids": [], "v": None, "a": a, "max": True}
            nodes.append(node)
            frontier.append(node)

        choice = greedy
        for depth in range(1, self.DEPTH + 1):
            # 1) 이번 층 마디들의 값 계산 (묶음으로)
            live = []
            try:
                for n in frontier:
                    m = n["m"]
                    if m.is_terminal():
                        n["v"] = 1.0 if me in m.winners() else -1.0
                    else:
                        live.append(n)
                obs = []
                for n in live:
                    check()
                    obs.append(egocentric_view(n["m"].observation()))
                if live:
                    logits, values = self._eval(obs)
            except _TimeUp:
                return choice
            for i, n in enumerate(live):
                cur = n["m"].current_player()
                v = float(values[i])
                n["v"] = v if cur == me else -v
                n["max"] = cur == me
                n["lg"] = logits[i]
            # 이번 층까지 계산이 끝났으니 지금까지의 결론을 저장
            choice = decide()
            if depth == self.DEPTH or not live:
                break
            # 2) 다음 층 펼치기. 도중에 시간이 넘치면 지금 층의 결론을 씀
            nxt = []
            try:
                for n in live:
                    msk = n["m"].action_mask() > 0
                    lg = np.where(msk, n["lg"], -np.inf)
                    k = self.K_ME if n["max"] else self.K_OPP
                    kids = []
                    for a in self._top(lg, k):
                        check()
                        c = n["m"].clone()
                        c.step(a)
                        kids.append({"m": c, "kids": [], "v": None, "a": a, "max": True})
                    n["new"] = kids
                    nxt.extend(kids)
                check()
            except _TimeUp:
                return choice
            for n in live:
                n["kids"] = n.pop("new", [])
            frontier = nxt
        return choice
