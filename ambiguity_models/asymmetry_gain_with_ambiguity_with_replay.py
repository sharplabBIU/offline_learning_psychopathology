import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(q, beta=1.0):
    q = np.array(q)
    max_q = np.max(q)
    exp_q = np.exp(beta * (q - max_q))
    return exp_q / np.sum(exp_q)

def calculate_gain(q_current, action_idx, stored_reward, alpha, beta):
    """
    Calculates Gain for a specific memory transition (action, reward).
    Gain = EV(new_policy | new_Q) - EV(old_policy | new_Q)
    """
    # 1. Hypothetical Update (Mental Simulation)
    q_new = q_current.copy()
    # Standard Q-learning update logic for the replay
    q_new[action_idx] += alpha * (stored_reward - q_new[action_idx])
    
    # 2. Policies
    pi_old = softmax(q_current, beta)
    pi_new = softmax(q_new, beta)
    
    # 3. Calculate Expected Values using the NEW Q-values as the "Truth"
    # (The agent assumes the updated Q is the better estimate of reality)
    ev_new = np.sum(q_new * pi_new)
    ev_old = np.sum(q_new * pi_old)
    
    return ev_new - ev_old

class MemoryBuffer:
    def __init__(self, decay_rate=0.1):
        self.actions = []
        self.rewards = []
        self.timestamps = []
        self.decay = decay_rate
        
    def add(self, action, reward, timestamp):
        self.actions.append(action)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        
    def sample(self, current_time):
        if not self.actions:
            return None
        
        # Recency Weighting
        # weight = exp(-decay * age)
        ages = current_time - np.array(self.timestamps)
        weights = np.exp(-self.decay * ages)
        probs = weights / np.sum(weights)
        
        # Sample one index
        idx = np.random.choice(len(self.actions), p=probs)
        return idx

def run_simulation_with_replay(mean_val, noise_std, n_trials=60, alpha=0.1, beta=1.5, replay_steps=10):
    n_actions = 3
    q_values = np.zeros(n_actions)
    memory = MemoryBuffer(decay_rate=0.2) # Moderate recency bias
    
    target_idx = 0
    replay_gain_history = []
    
    for t in range(n_trials):
        # --- 1. Agent Acts ---
        pi = softmax(q_values, beta)
        action = np.random.choice(n_actions, p=pi)
        
        if action == target_idx:
            reward = np.random.normal(mean_val, noise_std)
        else:
            reward = 0.0
            
        # Store in Memory
        memory.add(action, reward, t)
        
        # Direct Online Update
        q_values[action] += alpha * (reward - q_values[action])
        
        # --- 2. Replay Phase ---
        total_gain_this_trial = 0
        
        for _ in range(replay_steps):
            # Sample a memory
            idx = memory.sample(t)
            if idx is None:
                break
                
            past_a = memory.actions[idx]
            past_r = memory.rewards[idx]
            
            # Check Gain
            g = calculate_gain(q_values, past_a, past_r, alpha, beta)
            
            if g > 0:
                # Execute Replay (Update Q)
                q_values[past_a] += alpha * (past_r - q_values[past_a])
                total_gain_this_trial += g
            else:
                # Stop replaying if gain is non-positive (per user instruction)
                break
                
        replay_gain_history.append(total_gain_this_trial)
        
    return replay_gain_history

# --- Run Experiment ---
n_runs = 200
n_trials = 60
alpha = 0.1
beta = 2.0  # Higher beta to make policies distinct

data = []
# Scenarios: Reward (+5) vs Punishment (-5)
# Noise: 0 vs 20
scenarios = [
    ('Reward (+5)', 5.0, [0.0, 20.0]),
    ('Punishment (-5)', -5.0, [0.0, 20.0])
]

for scenario_name, mean_val, noise_levels in scenarios:
    for noise in noise_levels:
        label = f"Noise SD={int(noise)}"
        for r in range(n_runs):
            trace = run_simulation_with_replay(mean_val, noise, n_trials, alpha, beta)
            for t, g in enumerate(trace):
                data.append({
                    'Trial': t,
                    'Total Replay Gain': g,
                    'Noise': label,
                    'Scenario': scenario_name,
                    'Run': r
                })

df = pd.DataFrame(data)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Reward
sns.lineplot(data=df[df['Scenario']=='Reward (+5)'], x='Trial', y='Total Replay Gain', hue='Noise',
             palette=['green', 'darkgreen'], linewidth=2.5, errorbar='se', ax=axes[0])
axes[0].set_title('REWARD (+5) with Replay Loop', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Total Gain Generated (Sum per Trial)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Punishment
sns.lineplot(data=df[df['Scenario']=='Punishment (-5)'], x='Trial', y='Total Replay Gain', hue='Noise',
             palette=['red', 'darkred'], linewidth=2.5, errorbar='se', ax=axes[1])
axes[1].set_title('PUNISHMENT (-5) with Replay Loop', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Replay Dynamics: Recency Sampling & Gain Thresholding', fontsize=16, y=1.02)
plt.savefig('replay_loop_gain.png')
plt.tight_layout()
