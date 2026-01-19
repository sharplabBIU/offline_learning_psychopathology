import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(q, beta=1.0):
    q = np.array(q)
    max_q = np.max(q)
    exp_q = np.exp(beta * (q - max_q))
    return exp_q / np.sum(exp_q)

def get_gain(q_values, action_idx, true_q_value, beta):
    # 1. Current Policy (Old)
    pi_old = softmax(q_values, beta)
    
    # 2. Hypothetical New Knowledge State
    q_knowledge = q_values.copy()
    q_knowledge[action_idx] = true_q_value
    
    # 3. New Policy (after replay)
    pi_new = softmax(q_knowledge, beta)
    
    # 4. Evaluation Vector
    eval_q = q_values.copy() 
    eval_q[action_idx] = true_q_value 
    
    ev_new = np.sum(eval_q * pi_new)
    ev_old = np.sum(eval_q * pi_old)
    
    return ev_new - ev_old

def run_simulation(world_type, n_trials=50, alpha=0.1, beta=1.5):
    n_actions = 3
    q_values = np.zeros(n_actions) # Initial beliefs
    
    if world_type == 'positive':
        true_values = np.array([5.0, 0.0, 0.0,0.0,0.0,0.0,0.0]) # Action 0 is Good
        target_idx = 0
    else: # negative
        true_values = np.array([-5.0, 0.0, 0.0,0.0,0.0,0.0,0.0]) # Action 0 is Bad
        target_idx = 0
        
    gain_trace = []
    
    for t in range(n_trials):
        # 1. Record Gain
        g = get_gain(q_values, target_idx, true_values[target_idx], beta)
        gain_trace.append(g)
        
        # 2. Agent Acts
        pi = softmax(q_values, beta)
        action = np.random.choice(n_actions, p=pi)
        
        # 3. Learn
        reward = true_values[action]
        q_values[action] += alpha * (reward - q_values[action])
        
    return gain_trace

# Parameters
n_runs = 200 
n_trials = 50
alpha = 0.1
beta = 1.5

# Data Collection
data = []

for r in range(n_runs):
    # Positive World
    trace_pos = run_simulation('positive', n_trials, alpha, beta)
    for t, g in enumerate(trace_pos):
        data.append({
            'Trial': t,
            'Gain': g,
            'World': 'Reward (+5)',
            'Run': r
        })
        
    # Negative World
    trace_neg = run_simulation('negative', n_trials, alpha, beta)
    for t, g in enumerate(trace_neg):
        data.append({
            'Trial': t,
            'Gain': g,
            'World': 'Punishment (-5)',
            'Run': r
        })

# Create DataFrame
df = pd.DataFrame(data)

# Plotting with Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Trial', y='Gain', hue='World', 
             palette={'Reward (+5)': 'green', 'Punishment (-5)': 'red'},
             linewidth=2, errorbar='se')

plt.title('Evolution of Gain: Punishment vs Reward\n(Mean ± SE over 200 runs)', fontsize=14)
plt.ylabel('Gain (Expected Value of Replay)', fontsize=12)
plt.xlabel('Trial', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()