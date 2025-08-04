
# Offline Learning Models for Psychopathology

This repository collects **computational simulations of two offline-learning agents** that have been used to study planning, replay, and their alterations in anxiety and related psychopathology.

| Directory | Model | Original paper |
|-----------|-------|----------------|
| `Initial_dyna_sims/` | **SR-DYNA agent** – successor-representation (SR) updated with DYNA-style offline replay | Russek et al., 2017 (PLOS Comp. Biol.) |
| `Agrawal_simulations/` | **Replay‑with‑time‑cost agent** – prioritised replay that trades off improvement in future reward against opportunity cost of time | Agrawal et al., 2021 (Psych. Review) |

---

## 1  Successor‑Representation DYNA (`Initial_dyna_sims/`)

The SR‑DYNA agent combines:

1. **Online SR learning** – after every real state–action transition the agent performs TD(0) updates on the successor matrix \(M\).
2. **Offline replay** – after each episode it samples past experience from a priority queue and performs additional SR updates (`TOTAL_EXPERIENCES_OFFLINE_TRAINING` in the code).
3. **Q‑value computation** – action values are obtained on‑demand by the dot‑product \(Q = M \cdot R\), giving near‑model‑based flexibility at a lower computational cost.

**Key code knobs**

| Parameter | Role |
|-----------|------|
| `GRID_SIZE` | world dimensions of the grid‑world environment |
| `REWARD_ST` / `PUNISH_ST` | coordinates of terminal reward (+) and punishment (−) states |
| `REWARD_SCALE`, `PROBABILITY_SCALE` | magnitude / likelihood of outcome states |
| `TOTAL_EXPERIENCES_OFFLINE_TRAINING` | number of replay updates after each real step (controls the DYNA loop length) |

Setting **no replay** (`TOTAL_EXPERIENCES_OFFLINE_TRAINING = 0`) yields a vanilla SR learner; extensive replay approximates a fully model‑based planner.

---

## 2  Replay‑with‑Time‑Cost (`Agrawal_simulations/`)

Agrawal et al. formalised a rational principle for deciding **_when_** to engage replay: fire a burst only if the expected value of perfecting Q outweighs the *opportunity cost* of the time it would take.

```text
for each step:
    take action → observe (s, a, r, sʹ)
    store transition
    update Q online (TD)
    while burst_len < BURST_MAX and max(EVB_C) > 0:
        pick memory with highest EVB_C
        perform replay update (α_replay ⋅ TD)
```

**EVB\_C** = _Gain × Need – time‑cost_.

### Healthy vs Anxious manipulation

* `gamma_learn` – discount factor for *online* learning.  
* `gamma_replay` – discount when evaluating replay.

```python
if AGENT_TYPE == "healthy":
    gamma_replay = gamma_learn
else:  # anxious
    gamma_replay = gamma_learn - 0.2
```

Lower \(\gamma_{\text{replay}}\) makes imagined future rewards smaller, so the **anxious** agent triggers fewer replay events and learns less offline.

**Other flags**

| Parameter | Meaning |
|-----------|---------|
| `burst_max` | max memories updated per burst |
| `td_min` | TD‑error convergence threshold |
| `beta` | softmax inverse‑temperature |

---

## Quick start

```bash
# create env
conda create -n offline_learning python=3.11 numpy pandas matplotlib jupyter -y
conda activate offline_learning

# run notebooks
jupyter notebook Initial_dyna_sims/SR_dyna_demo.ipynb
jupyter notebook Agrawal_simulations/replay_cost_demo.ipynb
```

---

## Citing

```bibtex
@article{Russek2017SRDyna,
  title  = {Predictive representations can link model-based reinforcement learning to model-free mechanisms},
  author = {Russek, E.M. and Momennejad, I. and Botvinick, M. and Gershman, S. and Daw, N.},
  journal= {PLoS Computational Biology},
  year   = 2017
}

@article{Agrawal2021ReplayCost,
  title  = {The Temporal Dynamics of Opportunity Costs: A Normative Account of Cognitive Fatigue and Boredom},
  author = {Agrawal, M. and Mattar, M. and Cohen, J. and Daw, N.},
  journal= {Psychological Review},
  year   = 2021
}
```

---

## License

MIT – see `LICENSE`.
