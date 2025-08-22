import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import gymnasium as gym  
from common.layers import EnsembleStochasticLinear

# ========== Setup MuJoCo Humanoid Env ==========
env = gym.make('Humanoid-v5')
env.reset(seed=0)
obs_dim = env.observation_space.shape[0]  # ~376
act_dim = env.action_space.shape[0]       # 17

# ========== Training Parameters ==========
torch.manual_seed(42)
np.random.seed(0)
steps = 20000
samples_per_step = 100
ensemble_size = 100
num_candidates = 1

# ========== Model ==========
model = EnsembleStochasticLinear(obs_dim + act_dim, 64, obs_dim, ensemble_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== Storage ==========
epistemic_max_over_time = []
epistemic_mean_over_time = []
timestep = 0

# ========== Training Loop ==========
for step in range(steps):
    all_data = []
    obs , _ = env.reset(seed=0)

    for i in range(samples_per_step):
        action_candidates = np.zeros((num_candidates, act_dim))
        ep_candidates = torch.zeros(num_candidates)

        for j in range(num_candidates):
            action = env.action_space.sample()
            input_tensor = torch.tensor(np.concatenate([obs, action]), dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.repeat(ensemble_size, 1, 1)
            _, _, ep, _ = model(input_tensor)
            ep_candidates[j] = ep
            action_candidates[j] = action

        # Pick action with highest epistemic uncertainty
        max_ep = ep_candidates.max()
        mean_ep = ep_candidates.mean()

        action = action_candidates[ep_candidates.argmax().item()]
        next_obs, _, _, _, _ = env.step(action)

        epistemic_max_over_time.append(max_ep.item())
        epistemic_mean_over_time.append(mean_ep.item())

        print(f"[Step {timestep}] Max Epi: {max_ep:.4f}, Mean Epi: {mean_ep:.4f}")
        timestep += 1

        all_data.append((obs, action, next_obs))
        obs = next_obs

    # === Training step ===
    X = torch.tensor([np.concatenate([s, a]) for s, a, ns in all_data], dtype=torch.float32)
    Y = torch.tensor([ns for s, a, ns in all_data], dtype=torch.float32)

    X_in = X.unsqueeze(0).repeat(ensemble_size, 1, 1)
    Y_target = Y.unsqueeze(0).repeat(ensemble_size, 1, 1)

    model.train()
    mu, std, ep, _ = model(X_in)
    loss = ((mu - Y_target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ========== Plot ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Plot Max Epistemic
axes[0].plot(range(1, len(epistemic_max_over_time) + 1), epistemic_max_over_time, marker='o', color='red')
axes[0].set_title("Max Epistemic Uncertainty per Step")
axes[0].set_xlabel("Environment Step")
axes[0].set_ylabel("Epistemic Uncertainty")
axes[0].grid(True)

# Plot Mean Epistemic
axes[1].plot(range(1, len(epistemic_mean_over_time) + 1), epistemic_mean_over_time, marker='x', color='blue')
axes[1].set_title("Mean Epistemic Uncertainty per Step")
axes[1].set_xlabel("Environment Step")
axes[1].grid(True)

plt.suptitle("Epistemic Uncertainty vs Time (MuJoCo Humanoid-v4)")
plt.tight_layout()
plt.savefig("epistemic_max_vs_mean_humanoid.png", dpi=300, bbox_inches='tight')
plt.show()
