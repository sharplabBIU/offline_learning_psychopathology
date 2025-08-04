import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 1.  Environment
# ------------------------------------------------------------
GRID_SIZE   = 3
ACTIONS     = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # U, R, D, L
N_ACTIONS   = 4
REWARD_ST   = (0, 0)
PUNISH_ST   = (2, 2)
REWARD_SCALE = 20
PROBABILITY_SCALE=0.5
TRAVEL_TO_REWARD_TRIAL=1
CHANGE_REWARD_INTERVAL=20
TOTAL_EXPERIENCES_ONLINE=100
TOTAL_EXPERIENCES_OFFLINE_TRAINING=1000
# ----- helpers ------------------------------------------------

def clip(z: int) -> int:
    """Keep coordinate inside the grid."""
    return max(0, min(z, GRID_SIZE - 1))


def step(state, a,t):
    
    dx, dy = ACTIONS[a]
    nxt    = (clip(state[0] + dx), clip(state[1] + dy))

    
    # current_outcome=random.random() < PROBABILITY_SCALE
    current_outcome=1

    r      = REWARD_SCALE*current_outcome if nxt == REWARD_ST else \
             -REWARD_SCALE*current_outcome if nxt == PUNISH_ST else 0
        
    return nxt, r


def step_explore(state, a):
    """Like `step`, but with zero reward (used for SR pre‑training)."""
    dx, dy = ACTIONS[a]
    nxt    = (clip(state[0] + dx), clip(state[1] + dy))
    return nxt, 0


def softmax(q, beta=5.0):
    q = np.array(q) - np.max(q)
    p = np.exp(beta * q)
    return p / p.sum()

# ---------- random reward & punishment locations -------------
def sample_valence_states(grid_size=GRID_SIZE):
    """Return two distinct (x, y) tuples drawn uniformly at random."""
    all_states = list(product(range(grid_size), repeat=2))
    reward_st, punish_st = random.sample(all_states, k=2)  # no replacement
    return reward_st, punish_st

# ----- state ↔ index maps (for SR) ---------------------------
state2idx = {(x, y): y + GRID_SIZE * x for x in range(GRID_SIZE) for y in range(GRID_SIZE)}
idx2state = {v: k for k, v in state2idx.items()}

# ------------------------------------------------------------
# 2.  Simulation
# ------------------------------------------------------------

def simulate_condition(agent_type: str,
                       first_outcome: str,
                       tau: float,
                       n_steps: int = TOTAL_EXPERIENCES_ONLINE,
                       n_steps_explore: int = TOTAL_EXPERIENCES_OFFLINE_TRAINING,
                       alpha_replay=0.5,
                       gamma_pos: float = 0.05,
                       gamma_neg: float = 0.95,
                       gamma_learn: float = 0.99,
                       trial_rwd: int = TRAVEL_TO_REWARD_TRIAL,
                       change_rwd: int = CHANGE_REWARD_INTERVAL,
                       beta: float = 5,
                       td_min: float = 1e-8,
                       burst_max: int = 200):
    """Run one episode and return (ΔV(s₀), total_replay_backups)."""

    # ---------- initialise value and SR ----------------------
    Q  = np.zeros((GRID_SIZE, GRID_SIZE, N_ACTIONS))
    SR = np.zeros((GRID_SIZE ** 2, GRID_SIZE ** 2))+1/9

    # ---------- counters for dynamic learning‑rates ----------
    # 1/n where n is *total* update count (real + replay) for that state
    update_counts_SR = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    update_counts_RWD = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    def alpha_SR(st):  # learning‑rate helper
        n = update_counts_SR[st]
        return 1.0 if n == 0 else 1.0 / n
    
    def alpha_RWD(st):  # learning‑rate helper
        n = update_counts_RWD[st]
        return 1.0 if n == 0 else 1.0 / n
    
    def legal_actions(state):
        """Return a list of actions whose next-state ≠ current state."""
        return [a for a in range(N_ACTIONS) if step(state, a,1)[0] != state]

    def pick_action(state, Q, beta=5.0):
        """Soft-max policy over legal actions only."""
        acts = legal_actions(state)
        if not acts:                       # fallback (shouldn’t happen in open grid)
            acts = list(range(N_ACTIONS))
        q_subset = np.array([Q[state + (a,)] for a in acts])
        p_subset = softmax(q_subset, beta)
        return np.random.choice(acts, p=p_subset)

    # ---------- memory & bookkeeping -------------------------
    memory       = []     # list of (s, a, r, s')
    total_replay = 0
    gamma_curr   = gamma_pos
    V_start      = 0
    gam_pos=gamma_pos
    gam_neg=gamma_neg

    # ---------- nested: replay burst -------------------------
    def replay_burst(curr_state):
        nonlocal total_replay,gamma_curr,gam_neg,gam_pos,Q
        n_burst = 0
        V_cost  = max(np.dot(softmax(Q[curr_state], beta), Q[curr_state]), 0.0)
        old_policy=softmax(Q[curr_state], beta)

        while n_burst < burst_max:
            best_evb_c, best_mem = -np.inf, None

            for (s_mem, a_mem, r_mem, s2_mem) in memory:
                need   = SR[state2idx[curr_state], state2idx[s_mem]]
                q_old  = Q[s_mem + (a_mem,)]
                td     = r_mem + gamma_learn * np.max(Q[s2_mem]) - q_old
                if abs(td) < td_min:
                    continue  # converged

                # dynamic α for this state (increment count *before* use)
                # update_counts_RWD[s_mem] += 1
                a_mem_lr = alpha_replay
                q_new    = q_old + a_mem_lr * td

                # gain of value after hypothetical update
                q_tmp          = Q[s_mem].copy()
                q_tmp[a_mem] = q_new
                new_policy=softmax(q_tmp[a_mem], beta)
                gain           = (np.dot(softmax(q_tmp, beta), q_tmp) -
                                   np.dot(softmax(Q[s_mem], beta), q_tmp))

                if (agent_type == "Anxious" and r_mem <= 0):
                    g_now=gam_neg
                else:
                    g_now=gam_pos

                evb   = need * gain
                
                evb_c = (g_now ** tau) * evb - (1 - g_now ** tau) * V_cost
               
                if evb_c > best_evb_c: #retain just the best backup
                    best_evb_c, best_mem = evb_c, (s_mem, a_mem, q_new, r_mem)

            if best_evb_c <= 0:
                break  # nothing worth replaying
                

            # ---------- execute best backup ------------------
            s_b, a_b, q_new_b, r_b = best_mem
            Q[s_b + (a_b,)] = q_new_b
            
            total_replay   += 1
            n_burst        += 1

            
    # ---------- helper: perform one exploratory SR step ------
    def sr_train_step(s_t, a_t):
        
        # 1. increment count & compute α
        update_counts_SR[s_t] += 1
        a_lr = alpha_SR(s_t)

        # 2. execute exploratory move (reward‑free)
        s_next, r_t = step_explore(s_t, a_t)
        if s_next==s_t:
            a_t      = pick_action(s_t, Q, beta)   # ← new call
            s_next, r_t = step_explore(s_t, a_t)

        
        if (s_t, a_t, r_t, s_next) not in  memory:
            memory.append((s_t, a_t, r_t, s_next))

        # 3. Q update
        td = r_t + gamma_curr * np.max(Q[s_next]) - Q[s_t + (a_t,)]
        Q[s_t + (a_t,)] += a_lr * td

        # 4. SR update for current row
        idx_s, idx_snxt = state2idx[s_t], state2idx[s_next]
        SR[idx_s, idx_s] += a_lr * (1 - SR[idx_s, idx_s])
        SR[idx_s, :]     += a_lr * (gamma_curr * SR[idx_snxt, :] - SR[idx_s, :])

        # 5. policy‑guided next action / state
        a_next = np.random.choice(N_ACTIONS, p=softmax(Q[s_next], beta))
        return s_next, a_next

    # --------------------------------------------------------
    # 2a.  SR pre‑training (free exploration)
    # --------------------------------------------------------
    s_t = random.choice(list(state2idx))
    a_t = np.random.choice(N_ACTIONS, p=softmax(Q[s_t], beta))

    for _ in range(n_steps_explore):
        s_t, a_t = sr_train_step(s_t, a_t)

    # --------------------------------------------------------
    # 2b.  Main episode with rewards & replay
    # --------------------------------------------------------
    # teleport to specific valence‑defining starting position
    

    for t in range(n_steps):
        if first_outcome == "Positive":
            if t==trial_rwd:
                s_t, a_t = (0, 1), 3  # move left into +reward
        if first_outcome == "Negative":
            if t==trial_rwd:
                s_t, a_t = (1, 2), 2  # move left into –punishment
        
        if t+1%change_rwd==0:
            REWARD_ST, PUNISH_ST = sample_valence_states()
            # print("New reward @", REWARD_ST, "| punishment @", PUNISH_ST)

        # ----- count, α, and environment step --------------
        update_counts_RWD[s_t] += 1
        # a_lr = alpha_RWD(s_t) #if learning rate declines
        a_lr=alpha_replay+0.1
        s_next, r_t = step(s_t, a_t,t)
        
        if s_next==s_t:
            a_t = pick_action(s_t, Q, beta)   # ← new call
            s_next, r_t = step(s_t, a_t,t)
        

        if (s_t, a_t, r_t, s_next) not in  memory:
            memory.append((s_t, a_t, r_t, s_next))

        # ----- Q update -----------------------------------
        td = r_t + gamma_curr * np.max(Q[s_next]) - Q[s_t + (a_t,)]
        Q[s_t + (a_t,)] += a_lr * td

    


        # ----- SR update ----------------------------------
        idx_s, idx_snxt = state2idx[s_t], state2idx[s_next]
        SR[idx_s, idx_s] += a_lr * (1 - SR[idx_s, idx_s])
        SR[idx_s, :]     += a_lr * (gamma_curr * SR[idx_snxt, :] - SR[idx_s, :])

        # capture starting value (before any replay)

        # ----- replay burst --------------------------------
        replay_burst(s_t)

        # discount factor adaptation for anxious agents
        if agent_type == "Anxious":
            gamma_curr = gamma_neg if r_t < 0 else gamma_pos

        # advance to next real state & choose action
        if r_t!=0:
            s_next = (GRID_SIZE // 2, GRID_SIZE // 2)
        a_t = np.random.choice(N_ACTIONS, p=softmax(Q[s_next], beta))
        s_t = s_next

    # --------------------------------------------------------
    delta_V = np.max(Q) - V_start
    return delta_V, total_replay


# ------------------------------------------------------------
# 3.  Parameter sweep – MANY RUNS
# ------------------------------------------------------------
n_runs   = 100        # ← how many repetitions per (Agent, FirstExperience, τ)
tau_vals = [0.01, 0.03]
scenarios = [
    ("Healthy", "Positive"), ("Healthy", "Negative"),
    ("Anxious", "Positive"), ("Anxious", "Negative")
]

records = []
for agent, first_exp in scenarios:
    for tau in tau_vals:
        for _ in range(n_runs):
            dV, nrep = simulate_condition(agent, first_exp, tau)
            records.append(dict(Agent=agent,
                                FirstExperience=first_exp,
                                Tau=tau,
                                Value_Improvement=dV,
                                N_Replay=nrep))
results = pd.DataFrame(records)

# ------------------------------------------------------------
# 4. Aggregate: mean and SE (σ / √n)
# ------------------------------------------------------------
agg = (
    results
    .groupby(["Agent", "FirstExperience", "Tau"])
    .agg(mean_V   = ("Value_Improvement", "mean"),
         se_V     = ("Value_Improvement", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
         mean_R   = ("N_Replay",          "mean"),
         se_R     = ("N_Replay",          lambda x: x.std(ddof=1) / np.sqrt(len(x))))
    .reset_index()
)

# ------------------------------------------------------------
# 5.  Plotting: mean ± SE error bars
# ------------------------------------------------------------
palette = {"Healthy": "C0", "Anxious": "C1"}
style   = {"Positive": "-", "Negative": "--"}

# same tiny horizontal offsets so points don’t sit on top of each other
jitter = {("Healthy", "Positive") : -0.002,
          ("Healthy", "Negative") : -0.001,
          ("Anxious", "Positive") :  0.001,
          ("Anxious", "Negative") :  0.002}

# ---- (a) Value improvement ---------------------------------
plt.figure()
for (agent, first), grp in agg.groupby(["Agent", "FirstExperience"]):
    offs = jitter[(agent, first)]
    plt.errorbar(grp["Tau"], grp["mean_V"],
                 yerr=grp["se_V"], fmt="o",
                 color=palette[agent], linestyle=style[first],
                 capsize=3, label=f"{agent} – {first}")
plt.xlabel("τ  (replay / action time ratio)")
plt.ylabel("Δ Value(s₀)  after episode   (mean ± SE)")
plt.title(f"State-Value Improvement  (n = {n_runs} runs)")
plt.legend(); plt.grid(True)
plt.show()
# ---- (b) Total replay backups ------------------------------
plt.figure()
for (agent, first), grp in agg.groupby(["Agent", "FirstExperience"]):
    offs = jitter[(agent, first)]
    plt.errorbar(grp["Tau"], grp["mean_R"],
                 yerr=grp["se_R"], fmt="o",
                 color=palette[agent], linestyle=style[first],
                 capsize=3, label=f"{agent} – {first}")
plt.xlabel("τ  (replay / action time ratio)")
plt.ylabel("Total replay backups   (mean ± SE)")
plt.title(f"Replay Volume  (n = {n_runs} runs)")
plt.legend(); plt.grid(True)

plt.show()

