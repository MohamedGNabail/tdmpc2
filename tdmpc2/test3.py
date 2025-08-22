import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from common.layers import EnsembleStochasticLinear
import hydra
from common import layers
from types import SimpleNamespace
from common.parser import parse_cfg
from common.seed import set_seed



@hydra.main(config_path='.', config_name='config')
def main(cfg):
    def encode(obs):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([_encoder[cfg.obs](o) for o in obs])
        return _encoder[cfg.obs](obs)
    
    cfg = parse_cfg(cfg)
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v3-goal-observable']()
    env.seed(0)
    torch.manual_seed(42)
    np.random.seed(0)
    try: # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except: # Box
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
          
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    latent_dim = 512
    ensemble_size = 5
    steps =  2000
    samples_per_step = 100
    _encoder = layers.enc(cfg)
    

    model = EnsembleStochasticLinear(latent_dim + act_dim, 64, latent_dim, ensemble_size)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(_encoder.parameters()), lr=1e-3)


    epistemic_over_time = []
    timestep = 0

    for step in range(steps):
        obs, _ = env.reset(seed=0)
        obs_z = encode(torch.tensor(obs, dtype = torch.float32))
        all_data = []
        for i in range(samples_per_step):
            action = env.action_space.sample()
            next_obs, _, _, _, _ = env.step(action)
            next_obs_z = encode(torch.tensor(next_obs, dtype = torch.float32))
            with torch.no_grad():
                _, _, ep, _ = model(torch.cat([obs_z, torch.tensor(action, dtype=torch.float32)],dim=-1).unsqueeze(0).repeat(ensemble_size, 1, 1))
            # _, _, ep, _ = model(torch.cat([obs_z, torch.tensor(action, dtype = torch.float32)], dim=-1).unsqueeze(0).repeat(ensemble_size, 1, 1))
            avg_epistemic = ep.item()
            epistemic_over_time.append(avg_epistemic)
            print(f"[Step {timestep}] Avg Epistemic Uncertainty: {avg_epistemic:.4f}")
            timestep += 1
            all_data.append((obs_z, torch.tensor(action , dtype = torch.float32), next_obs_z))
            obs_z = next_obs_z

        X = torch.stack([torch.cat([s, torch.tensor(a, dtype=torch.float32)], dim=-1) for s, a, ns in all_data])
        Y = torch.stack([ns for s, a, ns in all_data])
        X_in = X.unsqueeze(0).repeat(ensemble_size, 1, 1)
        Y_target = Y.unsqueeze(0).repeat(ensemble_size, 1, 1)

        model.train()
        mu, std, ep, _ = model(X_in)
        loss = ((mu - Y_target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epistemic_over_time) + 1), epistemic_over_time, marker='o')
    plt.xlabel("Environment Step")
    plt.ylabel("Average Epistemic Uncertainty")
    plt.title("Epistemic Uncertainty vs Time (MetaWorld reach-v2)")
    plt.grid(True)
    plt.savefig("epistemic_uncertainty_metaworld5.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()