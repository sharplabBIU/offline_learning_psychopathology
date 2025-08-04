
# Offline Learning Models for Psychopathology

This repository collects **computational simulations of two offline-learning agents** that have been used to study planning, replay, and their alterations in anxiety and related psychopathology.

| Directory | Model | Original paper |
|-----------|-------|----------------|
| `initial_dyna_sims/` | **SR-DYNA** – successor-representation learner with DYNA-style offline replay | Russek et al., 2017, *PLoS Comp Biol* |
| `Agrawal_simulations/` | **Replay-with-Time-Cost** – rational replay that weighs benefit vs. opportunity-cost | Agrawal et al., 2021, *Psychological Review* |

---

## 1  Successor-Representation DYNA (`initial_dyna_sims/`)

### Algorithm at a glance

1. **Online SR update** after every real step

$H_{sa} \leftarrow H_{sa} + \alpha_{SR}(\mathbf{1_{sa}} + \gamma H_{s'a'} - H_{sa})$

2. **Reward-weight TD update**:
- learning

$PE = r + \gamma Q_{s'a'} - Q_{sa}$

$W \leftarrow W + \alpha_{TD}  \text{PE} H_{sa,i} \ \ \forall i$

- decision-making (value computation)

$Q_{s,a} = W_{sa} \cdot H_{sa}$



3. **Offline DYNA replay** (when a “rest” episode or specific trial is flagged):  
   draw **k** past transitions from memory (biased toward recency) and repeat step&nbsp;1 using the greedy action \(a^* = \arg\max_a Q(s',a)\).

The SR encodes long‑run state–action occupancies; coupled with learned reward weights this allows the agent to behave almost model‑based, yet learnable with TD.

### Major code switches

| Argument (constructor) | Typical value in demo | Role |
|---|---|---|
| `maze_size` | 6 | Side of square grid world (\#states = `maze_size²`) |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.05 | Epsilon‑greedy exploration when choosing actions |
| `alpha_sr` | 0.2 | Learning rate for SR updates (online + offline unless `dynamic_lr=True`) |
| `alpha_td` | 0.2 | Learning rate for reward weights `W` |
| `k` | 1000 | **Number of memory samples** processed per offline burst (`update_sr_offline`) |
| `dynamic_lr` | `False` | If `True`, SR LR becomes 1/visit‑count for each state‑action |
| `reward_location`, `punishment_location` | `(3,1)` / `(1,2)` | Terminal goal / penalty coordinates (row, col) |
| `reward_value`, `punishment_value` | 100 / −1 | Outcome magnitudes |
| `obstacles` | `[]` | List of blocked cells |
| `rest_episode` | `5` or `-1` | Engage offline replay only in these episode indices (`-1` → every episode) |
| `specific_trial` | `2` | Trial # inside an episode at which replay occurs |

**Outputs of interest**

* `H` – successor matrix (|SA| × |SA|)
* `W` – learned reward weights
* `Q` – planned action‑values (`H·W`)
* `episode_goal_counts` – goal hits per episode (for plotting learning curves)

---

## 2  Replay-with-Time-Cost (`Agrawal_simulations/`)

Agrawal et al. formalised a rational principle for deciding **when** to engage replay: the agent fires a burst only if the expected improvement in Q outweighs the *opportunity cost* of the time the burst would consume.

```text
for each step:
    take action → observe (s, a, r, sʹ)
    store transition
    update Q online (TD)
    while burst_len < BURST_MAX and max(EVB_C) > 0:
        pick memory with highest EVB_C = Gain × Need − c_time
        perform replay update (α_replay · TD)
```

### Key parameters in the demo

| Parameter | Meaning |
|---|---|
| `gamma_learn` / `gamma_replay` | Discount in real vs. imagined updates (lower `gamma_replay` models anxiety) |
| `burst_max` | Hard cap on replays per burst |
| `beta` | Softmax inverse‑temperature |
| `td_min` | TD‑error convergence threshold |

Lowering `gamma_replay` – the **anxious** setting – makes imagined future rewards smaller, so the agent initiates fewer replays, mirroring empirical findings.

### Key code knobs

| Parameter | Role |
|-----------|------|
| `GRID_SIZE` | world dimensions of the grid‑world environment |
| `REWARD_ST` / `PUNISH_ST` | coordinates of terminal reward (+) and punishment (−) states |
| `REWARD_SCALE`, `PROBABILITY_SCALE` | magnitude / likelihood of outcome states |
| `TOTAL_EXPERIENCES_OFFLINE_TRAINING` | number of replay updates after each real step (controls the DYNA loop length) |

## Quick start

```bash
conda create -n offline_learning python=3.11 numpy pandas matplotlib jupyter -y
conda activate offline_learning

# SR‑DYNA demo
python initial_dyna_sims/run_srdyna_demo.py

# Agrawal replay demo
python Agrawal_simulations/run_replay_demo.py
```

---

## Citing

```bibtex
@article{Russek2017SRDyna,
  title  = {Predictive representations can link model-based reinforcement learning to model-free mechanisms},
  author = {Russek, Erin M. and Momennejad, Ida and Botvinick, Matthew M. and Gershman, Samuel J. and Daw, Nathaniel D.},
  journal= {PLoS Computational Biology},
  year   = 2017
}

@article{Agrawal2021ReplayCost,
  title  = {The Temporal Dynamics of Opportunity Costs: A Normative Account of Cognitive Fatigue and Boredom},
  author = {Agrawal, Manaswi and Mattar, A. M. Grant and Cohen, Jonathan D. and Daw, Nathaniel D.},
  journal= {Psychological Review},
  year   = 2021
}
```

---


