import numpy as np
import torch
import matplotlib.pyplot as plt
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from common.layers import EnsembleStochasticLinear
from common.loss import gaussian_nll_loss
import random
SEED = 0

# Python RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# PyTorch RNG
torch.manual_seed(SEED)

# CuDNN Determinism (only applies if running on GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Enforce full determinism (may throw if unsupported ops are used)
torch.use_deterministic_algorithms(True)


#========== AUSE Calculation ==========

def AUSE(abs_error, y_var):
    idx_errors = np.argsort(abs_error)[::-1]
    idx_variances = np.argsort(y_var)[::-1]
    scurve_errors = np.array([])
    scurve_variances = np.array([])
    abs_error_sorted = abs_error[idx_errors]
    variances_sorted = abs_error[idx_variances]
    for _ in range(len(abs_error)-1):
        abs_error_sorted = np.delete(abs_error_sorted, 0)
        scurve_errors = np.append(scurve_errors,np.mean(abs_error_sorted))
        variances_sorted = np.delete(variances_sorted,0)
        scurve_variances = np.append(scurve_variances, np.mean(variances_sorted))
    integral_errors = np.trapezoid(scurve_errors)
    integral_variances = np.trapezoid(scurve_variances)
    ause = np.abs(integral_errors - integral_variances)
    return scurve_errors, scurve_variances, ause

def PlotAuse(method, mu, var, y_test, ax=None):
    color = 'tab:green'

    y_test, mu, var = y_test, mu, var
    ae = np.abs(mu - y_test)
    scurve_errors, scurve_variances, ause = AUSE(ae, var)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(10, 5))

    ax.scatter(range(len(scurve_errors)), scurve_errors, color='blue', label='oracle')
    ax.scatter(range(len(scurve_errors)), scurve_variances, color='black', label='model')
    ax.fill_between(range(len(scurve_errors)), scurve_errors, scurve_variances,
                    alpha=0.2, color=color, label='AUSE')
    ax.legend(prop=dict(size=14))
    ax.set_title(f"{method} AUSE")
    ax.set_ylabel('MAE')
    ax.set_xlabel('Samples Removed')
    ax.grid(True)

    if ax is None:
        plt.tight_layout()
        plt.savefig(f"{method}_AUSE.png", dpi=300, bbox_inches='tight')
        plt.show()
    #return scurve_errors, scurve_variances, ause


# ========== Setup MetaWorld Env ==========
env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v3-goal-observable']()
env.seed(SEED)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# ========== Training Parameters ==========

steps = 1000
samples_per_step = 100
ensemble_size = 5
num_candidates = 10
ause_log_interval = 10  # Log AUSE every 50 steps

# ========== Model ==========
model = EnsembleStochasticLinear(obs_dim + act_dim, 64, obs_dim, ensemble_size , uncertainity="BC")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== Storage ==========
epistemic_max_over_time = []
epistemic_mean_over_time = []
loss_over_time = []
ause_over_time = []
ause_steps = []
timestep = 0


# ========== Training Loop ==========
for step in range(steps):
    all_data = []
    obs, _ = env.reset(seed=0)

    for i in range(samples_per_step):
        action_candidates = np.zeros((num_candidates, act_dim))
        ep_candidates = torch.zeros(num_candidates)
        for j in range(num_candidates):
            action = env.action_space.sample()
            input_tensor = torch.tensor(np.concatenate([obs, action]), dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.repeat(ensemble_size, 1, 1)
            _, ep, _ = model(input_tensor)
            ep_candidates[j] = ep
            action_candidates[j] = action

        # Pick action with highest epistemic uncertainty
        max_ep = ep_candidates.max()
        mean_ep = ep_candidates.mean()

        #action = action_candidates[ep_candidates.argmax().item()]
        next_obs = env.step(action)[0] 
        epistemic_max_over_time.append(max_ep.item())
        epistemic_mean_over_time.append(mean_ep.item())

        #print(f"[Step {timestep}] Max Epi: {max_ep:.4f}, Mean Epi: {mean_ep:.4f}")
        timestep += 1

        all_data.append((obs, action, next_obs))
        obs = next_obs

    # === Training step ===
    X = torch.tensor([np.concatenate([s, a]) for s, a, ns in all_data], dtype=torch.float32)
    Y = torch.tensor([ns for s, a, ns in all_data], dtype=torch.float32)

    X_in = X.unsqueeze(0).repeat(ensemble_size, 1, 1)
    Y_target = Y.unsqueeze(0).repeat(ensemble_size, 1, 1)
    

    #train the model independently for each ensemble member
    consistency_loss = 0.0
    for member in range(ensemble_size):
        model.train()
        (mu, var) = model.single_forward(X_in[member], member)
        loss =  gaussian_nll_loss(mu, Y_target[member], var)
        consistency_loss = consistency_loss + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_over_time.append(consistency_loss.detach().numpy())  # Store the loss value
    print(f"[Step {step}] loss: {consistency_loss:.4f}")

    # === Compute & log AUSE every 'ause_log_interval' steps ===
    if (step + 1) % ause_log_interval == 0:
        with torch.no_grad():
            mu, epistemic, _ = model(X_in)
            y_pred = mu.mean(dim=0).cpu()
            y_true = Y.cpu()
            epistemic_np = epistemic.squeeze().cpu().numpy()

            abs_error_np = np.abs(y_pred.numpy() - y_true.numpy())
            scurve_errors, scurve_variances, ause = AUSE(abs_error_np, epistemic_np)
            ause_over_time.append(ause)
            ause_steps.append(step + 1)

            print(f"[Step {step + 1}] AUSE: {ause:.4f}")

# ========== Plot Final AUSE Curve on last batch ==========
ause_fig = PlotAuse("BC_Random", scurve_errors, scurve_variances, ause)

# ========== Plot Epistemic Uncertainty, Loss, and AUSE over Time ==========
fig, axes = plt.subplots(1, 5, figsize=(30, 5))  # Increased width for AUSE plot

# Max Epistemic
axes[0].plot(range(1, len(epistemic_max_over_time) + 1), epistemic_max_over_time, color='red')
axes[0].set_title("Max Epistemic Uncertainty per Step")
axes[0].set_xlabel("Environment Step")
axes[0].set_ylabel("Epistemic Uncertainty")
axes[0].grid(True)

# Mean Epistemic
axes[1].plot(range(1, len(epistemic_mean_over_time) + 1), epistemic_mean_over_time, color='blue')
axes[1].set_title("Mean Epistemic Uncertainty per Step")
axes[1].set_xlabel("Environment Step")
axes[1].set_ylabel("Epistemic Uncertainty")
axes[1].grid(True)

# Training Loss
axes[2].plot(range(len(loss_over_time)), loss_over_time, color='green')
axes[2].set_title("Training Loss over Time")
axes[2].set_xlabel("Training Step")
axes[2].set_ylabel("Loss")
axes[2].grid(True)

# AUSE over Time
axes[3].plot(ause_steps, ause_over_time, marker='o', color='purple')
axes[3].set_title("AUSE over Time")
axes[3].set_xlabel("Training Step")
axes[3].set_ylabel("AUSE")
axes[3].grid(True)

# AUSE Final Curve
PlotAuse("BC", scurve_errors, scurve_variances, ause, ax=axes[4])

plt.suptitle("Training Metrics (MetaWorld Reach-v3)")
plt.tight_layout()
plt.savefig("BC_Random_Action.png", dpi=300, bbox_inches='tight')
plt.show()