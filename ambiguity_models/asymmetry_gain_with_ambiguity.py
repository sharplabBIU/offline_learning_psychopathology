import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(q, beta=1.0):
    q = np.array(q)
    max_q = np.max(q)
    exp_q = np.exp(beta * (q - max_q))
    return exp_q / np.sum(exp_q)

def get_gain(q_values, action_idx, true_mean_value, beta):
    """
    Calculates Gain assuming Replay reveals the True Mean Value.
    """
    # 1. Current Policy
    pi_old = softmax(q_values, beta)
    
    # 2. Hypothetical Oracle State (Replay restores truth)
    q_oracle = q_values.copy()
    q_oracle[action_idx] = true_mean_value
    
    # 3. New Policy
    pi_new = softmax(q_oracle, beta)
    
    # 4. Evaluate using True Mean (Oracle perspective)
    eval_q = q_values.copy()
    eval_q[action_idx] = true_mean_value
    
    ev_new = np.sum(eval_q * pi_new)
    ev_old = np.sum(eval_q * pi_old)
    
    return ev_new - ev_old

def run_noisy_simulation(mean_val, noise_std, n_trials=50, alpha=0.1, beta=1.5):
    n_actions = 3
    q_values = np.zeros(n_actions)
    
    target_idx = 0
    true_mean_vec = np.array([0.0, 0.0, 0.0])
    true_mean_vec[target_idx] = mean_val
    
    gain_trace = []
    
    for t in range(n_trials):
        # 1. Calculate Gain (Value of restoring Q to True Mean)
        # We use the Mean as the "True" value the agent *would* learn from an infinite replay
        g = get_gain(q_values, target_idx, mean_val, beta)
        gain_trace.append(g)
        
        # 2. Act
        pi = softmax(q_values, beta)
        action = np.random.choice(n_actions, p=pi)
        
        # 3. Observe Noisy Reward
        # If the chosen action is the target, it has noise. Others are 0 (safe).
        if action == target_idx:
            reward = np.random.normal(mean_val, noise_std)
        else:
            reward = 0.0 # Safe actions assumed deterministic 0 for simplicity
            
        # 4. Update
        q_values[action] += alpha * (reward - q_values[action])
        
    return gain_trace

# Parameters
n_runs = 200
n_trials = 100
alpha = 0.05
beta = 1.5 # Slightly higher beta to emphasize selection effects

# Simulation Loop
data = []
noise_levels = [0.0, 20.0]
noise_labels = ['Low (SD=0)', 'High (SD=20)', ]

# We will run this for BOTH Reward and Punishment worlds to show the effect
scenarios = [
    ('Reward (+5)', 5.0),
    ('Punishment (-5)', -5.0)
]

for scenario_name, mean_val in scenarios:
    for noise_std, noise_label in zip(noise_levels, noise_labels):
        for r in range(n_runs):
            trace = run_noisy_simulation(mean_val, noise_std, n_trials, alpha, beta)
            for t, g in enumerate(trace):
                data.append({
                    'Trial': t,
                    'Gain': g,
                    'Noise': noise_label,
                    'Scenario': scenario_name,
                    'Run': r
                })

df = pd.DataFrame(data)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Reward
sns.lineplot(data=df[df['Scenario']=='Reward (+5)'], x='Trial', y='Gain', hue='Noise', 
             palette='Greens', linewidth=2, ax=axes[0])
axes[0].set_ylabel('Gain (Value of Replay)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot Punishment
sns.lineplot(data=df[df['Scenario']=='Punishment (-5)'], x='Trial', y='Gain', hue='Noise', 
             palette='Reds', linewidth=2, ax=axes[1])
axes[1].set_title('Impact of Noise on PUNISHMENT Replay', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Ambiguity (Reward Noise) Sustains the Demand for Replay', fontsize=16, y=1.02)
plt.show()
plt.savefig('noise_impact_gain.png',dpi=300,bbox_inches='tight')