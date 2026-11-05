# 실험 기록

## 실험 0: 기준선 (2026-11-05)

- 에이전트: 조교 제공 PPO (examples/ppo_entry, ppo_baseline_2p.pt)
- 측정: play.py, 상대 5종 x 50판, 0.1초 제한

| 상대 | 승률 | 승-패 | 점수 차 |
|---|---|---|---|
| Random | 100% | 50-0 | +31.9 |
| Factory | 98% | 49-1 | +27.4 |
| ShippingRush | 100% | 50-0 | +26.5 |
| TradeBuilding | 80% | 40-10 | +9.3 |
| ActionValue | 90% | 45-5 | +11.8 |
| 전체 | 94% | 234-16 | |

- 해석: Random, Factory, ShippingRush에는 거의 100% 승리함. 약한 상대는 TradeBuilding(80%)과 ActionValue(90%)이며, 둘 다 건물 위주 전략임.
- 가설: 제공된 PPO는 자기 자신과만 대전하며 학습했기 때문에, 건물 위주로 두는 상대를 학습 중에 거의 만나지 못했을 수 있음.
- 다음에 할 것:
  1. TradeBuilding에게 진 판을 열어서 어디서 밀리는지 확인
  2. 학습 상대에 TradeBuilding과 ActionValue를 섞어서 약점이 줄어드는지 실험