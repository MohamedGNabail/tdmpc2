import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    device = torch.device(cfg.cuda_device if hasattr(cfg, 'cuda_device') else "cuda")

    eval_steps = np.arange(0, 280000 + 1, 5000)
    seeds = [1, 2, 3]

    print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
    epi_uncertainty_summary = dict()

    # Example actions
    example_actions = np.array([
        [ 0.7527783,   0.78921333, -0.82991158, -0.92189043],
        [-0.66033916,  0.75628501, -0.80330633, -0.15778475],
        [ 0.91577906,  0.06633057,  0.38375423, -0.36896874],
        [ 0.37300186,  0.66925134, -0.96342345,  0.50028863],
        [ 0.97772218,  0.49633131, -0.43911202,  0.57855866],
        [-0.79354799, -0.10421295,  0.81719101, -0.4127717 ],
        [-0.42444932, -0.73994286, -0.96126608,  0.35767107],
        [-0.57674377, -0.46890668, -0.01685368, -0.89327491],
        [ 0.14823521, -0.70654285,  0.17861107,  0.39951672],
        [-0.79533114, -0.17188802,  0.38880032, -0.17164146]
    ])
    example_actions_torch = torch.from_numpy(example_actions).float().to(device)
    cfg = parse_cfg(cfg)
    checkpoint_path = cfg.checkpoint
    for step in eval_steps:
        total_uncertainty = 0
        print(f"Step: {step}")

        for seed in seeds:
            cfg.seed = seed
            set_seed(cfg.seed)

            env = make_env(cfg)
            agent = TDMPC2(cfg)

            cfg.checkpoint = f"{checkpoint_path}/reward_driven{cfg.seed}-{cfg.uncertainity_rep}-{step}.pt"
            print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
            assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
            agent.load(cfg.checkpoint)

            obs = env.reset()
            low = env.observation_space.low[:3]
            voxel_size = 0.01
            dims = (105, 68, 76)  # from grid.shape

            x = low[0] + (np.arange(dims[0]) + 0.5) * voxel_size
            y = low[1] + (np.arange(dims[1]) + 0.5) * voxel_size
            z = low[2] + (np.arange(dims[2]) + 0.5) * voxel_size

            centers = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T
            batch_size = 1000
            num_batches = int(np.ceil(len(centers) / batch_size))

            for i in range(num_batches):
                batch_centers = centers[i * batch_size : (i + 1) * batch_size]
                obs_batch = np.tile(obs, (len(batch_centers), 1))
                obs_batch[:, :3] = batch_centers

                # Move obs_batch to GPU
                obs_batch_torch = torch.from_numpy(obs_batch).float().to(device)

                # Encode
                z = agent.model.encode(obs_batch_torch, cfg.task)

                for action_t in example_actions_torch:
                    action_batch = action_t.unsqueeze(0).repeat(len(batch_centers), 1)
                    reward, reward_epi_uncer, reward_aleatoric_uncer = agent.model.reward(z, action_batch, cfg.task)
                    total_uncertainty += reward_epi_uncer.sum().item()

                # Free GPU memory
                del obs_batch_torch, z, action_batch
                torch.cuda.empty_cache()

        epi_uncertainty_summary[step] = total_uncertainty
        print(total_uncertainty)
    print(epi_uncertainty_summary)


if __name__ == '__main__':
    evaluate()
