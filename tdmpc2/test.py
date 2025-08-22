import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from common.layers import EnsembleStochasticLinear
# ========== Gridworld Setup ==========
GRID_SIZE = 40
ACTIONS = [0, 1, 2, 3]  # up, down, left, right
ACTION_TO_DELTA = {
    0: np.array([0, 1]),   # up
    1: np.array([0, -1]),  # down
    2: np.array([-1, 0]),  # left
    3: np.array([1, 0]),   # right
}

def true_dynamics(state, action):
    delta = ACTION_TO_DELTA[action]
    return np.clip(state + delta, 0, GRID_SIZE - 1)

def collect_data(n_samples):
    data = []
    for _ in range(n_samples):
        s = np.random.randint(0, GRID_SIZE, size=(2,))
        a = np.random.choice(ACTIONS)
        ns = true_dynamics(s, a)
        data.append((s, a, ns))
    return data


# ========== Training + Uncertainty Over Time ==========
torch.manual_seed(42)
steps = 200
samples_per_step = 20
ensemble_size = 5

model = EnsembleStochasticLinear(3, 64, 2, ensemble_size=ensemble_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

all_data = []
epistemic_over_time = []

for step in range(steps):
    new_data = collect_data(samples_per_step)
    all_data.extend(new_data)

    X = torch.tensor([np.concatenate([s, [a]]) for s, a, ns in all_data], dtype=torch.float32)
    Y = torch.tensor([ns for s, a, ns in all_data], dtype=torch.float32)

    X_in = X.unsqueeze(0).repeat(ensemble_size, 1, 1)  # [E, B, D]
    Y_target = Y.unsqueeze(0).repeat(ensemble_size, 1, 1)  # [E, B, D]

    # Train
    for epoch in range(100):
        model.train()
        mu, _, _, _ = model(X_in)
        loss = ((mu - Y_target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate epistemic uncertainty over entire grid
    model.eval()
    grid_inputs = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for a in ACTIONS:
                grid_inputs.append([x, y, a])
    grid_tensor = torch.tensor(grid_inputs, dtype=torch.float32)
    grid_tensor = grid_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1)

    with torch.no_grad():
        _, _, epistemic, _ = model(grid_tensor)
        avg_epistemic = epistemic.mean().item()
        epistemic_over_time.append(avg_epistemic)

    print(f"[Step {step+1}] Avg Epistemic: {avg_epistemic:.4f}")

# ========== Plot ==========
plt.figure(figsize=(8, 5))
plt.plot(range(1, steps + 1), epistemic_over_time, marker='o')
plt.xlabel("Data Collection Step")
plt.ylabel("Average Epistemic Uncertainty")
plt.title("Epistemic Uncertainty vs Time")
plt.grid(True)
plt.savefig("epistemic_uncertainty_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()
