import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_nll_loss(mu, target, var):
    # mu [Batch Size, latent dimension ], var [Batch Size, latent dimension ]
    # Custom Gaussian Negative Log Likelihood Loss
    loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var) #[B, D]
    return torch.mean(loss)


