# Moka — a Puerto Rico agent trained with PPO

Moka is my entry for the final project of the Reinforcement Learning course at Chonnam National University (fall 2026): a 2-player agent for the board game *Puerto Rico*. Everyone in the class submits an agent, and the agents play each other to decide the ranking.

> **Status (Nov 23, 2026):** the current version is **Moka v7**. Training experiments end with experiment 14; a v8 is adopted only if it clearly beats v7. Class ladders run on Nov 19, 24, 26 and 28, and the final code is due Nov 29. In the first class ladder (Nov 19, Moka v6) Moka placed 3rd of 17.

The game engine, the competition rules, the baseline bots and a starting PPO checkpoint come from the TA's repository (the original README is kept below). What I did on top of it: designed the training, changed the training script, and ran and measured the experiments. Every experiment has its own note in [docs/experiments](docs/experiments/README.md) (in Korean), and [docs/ai-usage-log.md](docs/ai-usage-log.md) (in Korean) describes how I worked with an AI assistant (Claude) along the way.

## How it plays

At every decision Moka feeds the (egocentric) observation to its policy network and plays the most probable legal move. There is no search inside `act()`; a move takes about 1 ms on one CPU thread (limit: 100 ms). Everything Moka knows comes from training.

## Results

Win rates of the TA's starting checkpoint and of Moka v7, on seeds that were never used for training or for picking the candidate (1,000 games per opponent unless noted). Every pairing plays each seed twice with the seats swapped.

| Opponent | TA checkpoint | Moka v7 |
|---|---:|---:|
| TradeBuilding | 88.0% | 99.5% |
| ActionValue | 90.0% | 98.9% |
| SearchLite | 40.3% | 96.2% |
| MCTS | 91.5% (200) | 98.2% |
| Search | 32.5% (200) | 92.0% |
| Random / Factory / ShippingRush | – | 99.8–100% |
| Student proxy P5 (never trained against) | – | 69.5% |

No timeouts and no illegal moves (limit: 0.1 s per move on one CPU thread).

The "student proxies" are agents I trained from the TA checkpoint the way another student might (self-play only, rule bots, SearchLite, longer runs), to stand in for the real opponents I cannot see. The strongest one, P5 (self-play for 12M steps), is Moka's weakest matchup.

## Versions

A new version only replaces the old one if it beats it in 1,000 games on unseen seeds with the 95% interval above 50%, and does not get worse against the strong opponents.

| Version | Main change | vs previous version | Weakest opponent |
|---|---|---|---:|
| v1 | Mixed opponents instead of pure self-play, lr 1e-5 | 77.3% vs the TA checkpoint | Search 44.5% |
| v2 | Past checkpoints in the pool | 65.3% | Search 57.5% |
| v3 | SearchLite in the pool | 71.4% | Search 82.2% |
| v4 | 6M more steps | 61.8% | Search 91.65% |
| v5 | Exploiter: trained only against a frozen copy of the best Moka | 77.2% | Search 93.0% |
| v6 | Student proxies in the pool instead of our own past versions | 62.1% | Proxy P4 85.1% |
| v7 | v6's pool combined with more self-play | 63.4% | Proxy P5 69.5% |

## What I changed, and what I learned

**Fine-tuning from the TA checkpoint went wrong first.** I resumed training at the original learning rate (3e-4) and the policy fell apart: KL jumped to 0.72 and the win rate against the starting checkpoint went *down*. The original run had already decayed its learning rate to zero, so restarting at 3e-4 was far too large a step. 1e-5 brought KL back to about 0.013.

**Mixed opponents instead of pure self-play.** Each game draws an opponent from a weighted pool: the current policy, the TA checkpoint, older Moka checkpoints, and the rule-based bots. Against SearchLite, the weakest matchup at the time, 6M more steps without it reached only 66.3%; with SearchLite in 10% of the games, 89.5%. Search and MCTS improved too although they were never in the pool.

**You get strongest against what you practice with.** An exploiter trained only against a frozen Moka v3 beat it 84%, while an agent of the same general strength beat it only 58.5%: v3 had a hole that a targeted opponent could find. Putting the exploiter into the pool closed most of it. But agents that kept practicing against our own lineage beat outside agents less than expected (Moka v5 beat the student proxies only 76–81%). A controlled experiment (same start, one variable) showed each variant was best against the kind of opponent it trained with, so I moved the pool toward student proxies and tested on a proxy that no version had seen.

**Search inside `act()` did not survive outside opponents.** Moka uses about 1% of its time budget, so I tried policy rollouts at decision time. Against the same network they won 80%, but against ActionValue the win rate collapsed from 92% to 51%: the rollouts assumed the opponent plays like our own policy. I did not adopt it.

**Measuring was harder than training.** My first measurements used 50 games and I read a ±10-point swing as a real weakness. Two copies of the same weights scored 43% against each other, which turned out to be seat luck in the test script. Since then: a seat-swapped ladder, separate seed ranges for experiments and for final checks, and 1,000 games with a 95% interval before any decision. A 200-game result (57% for one candidate against P5) once looked like a weakness that 1,000 games did not confirm (69%). The game itself is not symmetric either: the first player starts with indigo and the second with corn, and almost every agent in the class ladder did 6–18 points worse as the first player.

**The first move timed out.** On a single thread the first forward pass took 1.8 s. The agent now runs one dummy pass while loading, before the game starts.

## Files I wrote or changed

- `my_agent/my_agent.py` — the submitted agent (`Moka`)
- `training/train_moka.py` — copy of the TA's `train_ppo.py` with an opponent pool, past-checkpoint opponents, Search/MCTS opponents and `--init_ckpt` (changes marked `[Moka]`)
- `training/export_entry.py` — turns a checkpoint into a ladder entry folder (weights saved as a `state_dict`)
- `examples/study/` — scripts to replay and analyze recorded games (`habits.py` for seat, role and score patterns over many games, `role_timing.py` for wasted roles)
- `docs/experiments/` — one note per experiment

## Reproducing Moka

The trained weights are not in this repository until the course is over, so that other participants cannot train against Moka. Each version is one run of `training/train_moka.py` started from the previous version; the exact opponent pool, steps and learning rate of every run are in its experiment note.

| Version | Experiment | Starts from | Steps |
|---|---|---|---:|
| v1 | [03](docs/experiments/2026-11-07_03_mixed-opponents.md) | TA checkpoint | 3M |
| v2 | [04](docs/experiments/2026-11-08_04_longer-and-past-self.md) | v1 | 3M |
| v3 | [06](docs/experiments/2026-11-10_06_searchlite.md) | v2 | 6M |
| v4 | [08](docs/experiments/2026-11-11_08_longer.md) | v3 | 6M |
| v5 | [09](docs/experiments/2026-11-13_09_new_opponent_pool.md), [10](docs/experiments/2026-11-14_10_exploiter_round2.md) | v4 → E9a_60 → exploiter | 6M + 6M |
| v6 | [11](docs/experiments/2026-11-17_11_student-proxies.md), [12](docs/experiments/2026-11-18_12_proxies-vs-selfplay.md) | v5 (proxies P1–P4 from the TA checkpoint) | 6M |
| v7 | [13](docs/experiments/2026-11-21_13_combined-and-searchbots.md) | v6 | 6M |

All runs use `--lr 1e-5`, which decays linearly to 0. A checkpoint becomes an entry with:

```bash
pip install -e ".[rl]"
python training/export_entry.py <run>/checkpoints/ckpt_final.pt <entry_folder> Moka
python tools/validate_submission.py <entry_folder>/my_agent.py:Moka --track 2p --time-limit 0.1
```

## What I would try next

- Search inside `act()` that does not assume the opponent plays like Moka: sample opponent moves from several networks (student proxies) and rule bots, and measure it only against opponents outside our own lineage.
- A higher learning rate for long runs (the strongest proxy used twice Moka's rate), and comparing one long run with several restarted ones.
- The first-player seat: most of the recent improvement came from the second seat.

---

*Below is the original README of the competition repository by the TA.*

# Puerto Rico AI Competition

[![tests](https://github.com/dae-hany/PuertoRico_AI_Competition/actions/workflows/tests.yml/badge.svg)](https://github.com/dae-hany/PuertoRico_AI_Competition/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dae-hany/PuertoRico_AI_Competition/blob/main/examples/first_agent.ipynb)

An AI competition built on **Puerto Rico**, an economic strategy board game
proposed as a reinforcement-learning / game-AI benchmark. You write an agent,
it plays full games against other agents, and a tournament ranks everyone.

The competition runs in **two independent tracks**, each with its own leaderboard:

- **2p track** — 1-vs-1 (2-player) Puerto Rico.
- **3p track** — 3-player Puerto Rico.

You may enter **either track or both**; each track is a **separate submission**
(among other things, the observation length differs — 225 in 2p, 298 in 3p). The
rules, engine, and action space are otherwise shared.

The competition **running now** is the final project of the **Reinforcement
Learning** course at **Chonnam National University** (2p track, autumn 2026) —
see [docs/COURSE_CNU_2026.md](docs/COURSE_CNU_2026.md) if you are taking it
([한국어](docs/COURSE_CNU_2026.ko.md)). A
public **IEEE CoG 2027** edition is planned on the same engine and the same
technical rules; its dates will be announced here. Everything — the
engine, the agents, the docs — is in English so anyone can take part. The engine
is a standard [PettingZoo](https://pettingzoo.farama.org/) AEC environment
(`puerto_rico/env.py`), so multi-agent RL tooling can drive it directly.

**New here?** The fastest start is the notebook
[`examples/first_agent.ipynb`](examples/first_agent.ipynb) (open it in Colab
with the badge above): write an agent, play it against the baselines, and
replay a game, in about ten minutes and with nothing installed.

> Why a competition? Puerto Rico is easy to simulate but hard to master:
> hand-written heuristics, tree search and self-play reinforcement learning all
> play it, at the strengths measured in [docs/BASELINES.md](docs/BASELINES.md).
> Building something that beats them is the challenge.

![The web UI: two baseline agents playing a 2-player game](docs/img/webui.png)

## Quickstart

```bash
git clone https://github.com/dae-hany/PuertoRico_AI_Competition.git
cd PuertoRico_AI_Competition
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                        # installs deps + makes the packages importable
pip install -e ".[webui]"               # optional: flask, for the browser UI
pip install -e ".[rl]"                  # optional: PyTorch, for the PPO baseline

python examples/play_one_game.py        # one baseline game per track (2p and 3p)
python tools/play.py --agent submissions/example_agent.py:ExampleAgent   # an agent vs the baselines
python tools/validate_submission.py submissions/example_agent.py         # the official sandbox check
python examples/run_tournament.py       # a round-robin + leaderboard for each track
python webui/server.py                  # browser UI: play / watch / debug (webui extra)
# then open http://127.0.0.1:5000/?players=2&seats=actionvalue,trade to watch two bots play
```

`pip install -e .` is recommended (it makes `agents`, `tournament`, `puerto_rico`
importable from anywhere). `pip install -r requirements.txt` also works — the
example scripts add the repo root to the path themselves, but you should then run
them from the repo root.

## Write an agent

Subclass `Agent` and implement one method:

```python
import numpy as np
from agents.base import Agent

class MyAgent(Agent):
    name = "MyAgent"

    def act(self, observation, action_mask):
        # observation: float32[225] in 2p / [298] in 3p  |  action_mask: int[200] (1 = legal)
        legal = np.where(action_mask > 0.5)[0]
        return int(legal[0])            # replace with your strategy
```

The action space is identical in both tracks; only the observation length differs
(`len(observation)` tells you which track you are in). Agents that read the mask
or the `forward_model` — like every baseline below — work in both. (PPO reads
neither; it ships one checkpoint per track instead.) Besides the moves of each
role, your agent decides whether to take each optional benefit (the Mayor's and
the Captain's privileges, the Hospice and University colonists, the Harbor VP):
then only `170` (accept) and `171` (decline) are legal. See
[Observation & action encoding](docs/OBSERVATION_AND_ACTIONS.md).

Copy [`submission_template/`](submission_template/) to get started, then read the
[Submission guide](docs/SUBMISSION_GUIDE.md).

## Test and validate your agent

```bash
python tools/play.py --agent my_agent.py:MyAgent                        # vs ActionValue and TradeBuilding
python tools/play.py --agent my_agent.py:MyAgent --vs Search --games 20 # win rate, VP margin, ms per move
python tools/play.py --agent my_agent.py:MyAgent --players 3 --vs all --record
python tools/validate_submission.py my_agent.py:MyAgent                 # plays inside the official sandbox
```

`play.py` plays seat-rotated games under the competition rules and prints, per
opponent, your win rate, mean VP margin, and per-move time, so you can see how
close to the 0.1 s budget you are. `validate_submission.py` runs your file the way
the official run does — its own process, killed at the deadline — and prints a
PASS / WARN / FAIL checklist (imports, name, timeouts, illegal moves) with exit
status 0 when the entry is ready. Any recorded game (`--record`, or every game
played in the web UI) replays move by move with `tools/replay_game.py`.

## Baselines

| Agent | Type | Track |
|---|---|---|
| `PpoAgent` | PPO self-play (RL), greedy | both — one checkpoint per track |
| `SearchAgent` | alpha‑beta search, 1500‑node budget | 2p (reactive fallback in 3p) |
| `SearchLiteAgent` | the same search, 250‑node budget | 2p (reactive fallback in 3p) |
| `MctsAgent` | Max^N UCT tree search, 60 simulations | both |
| `ActionValueAgent` | greedy heuristic — scores each legal action, plays the best | both |
| `ShippingRushAgent` | shipping-focused heuristic | both |
| `TradeBuildingAgent` | trade → building heuristic | both |
| `FactoryAgent` | Factory-engine heuristic | both |
| `RandomAgent` | uniform random legal move | both |

**How strong is each one? → [`docs/BASELINES.md`](docs/BASELINES.md)**, measured
rather than asserted, and regenerated by `python tools/measure_baselines.py`
whenever a baseline or a rule changes; the numbers are kept there and nowhere
else. It has, per track, the round-robin of the heuristics, each planner (and
PPO) against each heuristic, and in 2p PPO against each search baseline. Check
the track you are entering: the two tracks need not rank the agents alike.

A search agent's strength is only defined at a **stated compute budget**, so the
budgets are quoted with the numbers; `SearchAgent`'s is a node count rather than
a clock, so it plays the same move on any hardware. The numbers hold under the
**0.1 s rule**: it applies to entries, while the bundled baselines keep their own
1 s budget, and none of them comes near it (`Search`'s slowest move measured
≈ 0.3 s), so these are the opponents your entry meets.

The heuristic baselines are player-count-agnostic, and so is `MctsAgent` (its
Max^N value vectors are sized from the game it is handed). `SearchAgent` /
`SearchLiteAgent` search the 1‑vs‑1 game and fall back to a reactive heuristic in
a 3p game. `PpoAgent` plays both tracks with one checkpoint each, both
trained with the same command and default settings (5M steps of self-play
each, about 2 h each on 8 CPU threads); see [`training/README.md`](training/README.md), and
[`examples/ppo_entry/`](examples/ppo_entry/) for an entry built on the 2p one.

### Recommended approach for the 2p track

Puerto Rico 1‑vs‑1 is a two-player, winner-takes-all (≈ zero-sum),
**near‑perfect‑information** game (the only hidden state is the face-down
plantation draw order), so both routes work: a **learned** policy (the `PPO`
baseline) and **adversarial search**, the classic tool for this kind of game
([BASELINES.md](docs/BASELINES.md) has the two against each other). They also
combine: a policy or value network can guide or cut off a search (see
[`examples/ppo_entry/`](examples/ppo_entry/) for the learned starting point).
`SearchAgent` is a clean, readable alpha‑beta reference you can **play against,
study, and improve**. Its strength is set by a **node budget** (not wall-clock),
so it plays the *same* move on any hardware — fair to compare against. (At 1500
nodes it needs more than an entrant's 0.1 s per decision; as a bundled baseline
it keeps its own 1 s budget, but an entry built on it must use fewer nodes.) See
**[docs/SEARCH_BASELINE_2P.md](docs/SEARCH_BASELINE_2P.md)** for how it works and
a ranked list of concrete ways to make it stronger (your opportunity as an
entrant).

## Repository layout

```
puerto_rico/        core game engine + environment + forward model
agents/             the Agent interface and all baseline agents
tournament/         single-match harness, round-robin runner, rankers, leaderboard
                    (sandbox.py runs each entrant in its own process — the
                     official run's time limit and anti-tampering guarantees)
training/           optional PPO self-play trainer + the bundled RL checkpoints (2p, 3p)
webui/              browser UI to play, watch, and debug agents
examples/           first_agent.ipynb (Colab), play_one_game.py, run_tournament.py,
                    ppo_entry/ (a learned 2p entry with its weight file)
tools/              play.py (your agent vs the baselines), validate_submission.py
                    (the official sandbox check), replay_game.py, run_ladder.py
                    (organizer: one ladder round), course_ladder.py (organizer:
                    a course cohort — zip intake, fixed seed list, parallel
                    games, wins-and-rank board), make_private_seeds.py,
                    bench2p.py, bench_engine.py, measure_baselines.py
leaderboard/        ladder standings and replays, published during the window
submission_template/ copy this to build your competition entry
submissions/        drop an agent here to debug it in the web UI
docs/               rules, observation/action encoding, ranking, submission guide
tests/              pytest suite
```

## The competition

- **Format.** Two tracks, 2p and 3p, one agent file per track (plus at most
  one weight file of up to 50 MB for a learned agent), **0.1 s per
  decision**. Each track is a seat-balanced round-robin of every entry against
  every other entry and the bundled baselines, ranked by Elo (2p) or TrueSkill
  (3p); the rules are in [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md).
- **Running now: the course edition.** The 2p track is the final project of the
  Reinforcement Learning course at Chonnam National University (autumn 2026).
  Its extra rules — ranking by wins and rank in games between students, a code
  zip and a separate report and presentation zip, and four ladder rounds that
  publish only wins and rank — are
  in [docs/COURSE_CNU_2026.md](docs/COURSE_CNU_2026.md), which takes precedence
  for students of that course. The course is graded with Python 3.12 and
  CPU-only PyTorch 2.15, one thread per agent, 32 games at a time; the machine
  and the details are under
  [Grading environment](docs/COURSE_CNU_2026.md#grading-environment).
- **Window.** The competition runs for about three months. The dates of the
  IEEE CoG 2027 edition will be announced here.
- **Weekly ladder.** During the window the organizer runs every entry received
  so far (`tools/run_ladder.py`) and publishes the standings **and every game's
  replay** under [`leaderboard/`](leaderboard/). Submit early, watch your rank,
  and resubmit as often as you like; the last file received before the deadline
  is the one that counts.
- **How to submit.** To be announced with the dates (CNU course students: the
  LMS, as described on the course page). Until then, questions and
  bug reports go to the
  [issue tracker](https://github.com/dae-hany/PuertoRico_AI_Competition/issues).

## How ranking works

**Each track is ranked separately** (its own round-robin and leaderboard). The
official metric is a **skill rating matched to the track** — **Elo** in the 2p
(1‑vs‑1) track, **TrueSkill** in the 3p track — computed over a seat-balanced
round-robin, with **win rate** (Wilson 95% CI) shown alongside. (α‑Rank is
available as opt‑in analysis, not part of the standings.) See
[Ranking](docs/RANKING.md). [`docs/BASELINES.md`](docs/BASELINES.md) shows both
boards over the bundled heuristics, Elo in 2p and TrueSkill in 3p, with win
rates alongside.

## Documentation

- [Game rules](docs/GAME_RULES.md)
- [Baseline strengths](docs/BASELINES.md) — measured, per track
- [Observation & action encoding](docs/OBSERVATION_AND_ACTIONS.md)
- [Competition rules](docs/COMPETITION_RULES.md)
- [Submission guide](docs/SUBMISSION_GUIDE.md)
- [Ranking](docs/RANKING.md)
- [Changes that affect results](CHANGES.md)

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## License

Code is released under the [MIT License](LICENSE). The board game *Puerto Rico*
(designer Andreas Seyfarth) is the intellectual property of its rights holders;
this is an independent, non-commercial re-implementation for education and
research, with no original artwork. See [LICENSE](LICENSE) for the full notice.
