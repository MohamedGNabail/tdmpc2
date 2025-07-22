import torch

class BhattacharyyaOverlap:
    def __init__(self, states_mean, states_std):
        """
        states_mean: Tensor of shape [ensemble_size, batch_size, latent_dim]
        states_std:  Tensor of shape [ensemble_size, batch_size, latent_dim]
        """
        self.mu = states_mean
        self.std = states_std

    def _bhattacharyya_coefficient_diag(self, mu1, std1, mu2, std2, eps=1e-8):
        """
        Compute BC between two diagonal Gaussian distributions per sample:
        mu1, std1, mu2, std2: [num_pairs, batch_size, latent_dim]
        Returns: [num_pairs, batch_size] with BC scores in [0,1]
        """
        var1 = std1 ** 2
        var2 = std2 ** 2
        var_sum = var1 + var2 + eps  # [N, B, D]
        sqrt_term = torch.sqrt(2 * std1 * std2 / var_sum)  # [N, B, D]
        exp_term = torch.exp(- (mu1 - mu2) ** 2 / (4 * var_sum))  # [N, B, D]

        bc_per_dim = sqrt_term * exp_term  # [N, B, D]
        bc = bc_per_dim.mean(dim=-1)  # average over latent_dim → [N, B]
        return bc

    def compute_measure(self):
        """
        Returns:
            epistemic_uncertainty: [batch_size] — higher means more disagreement
            mean_bc_per_member: [ensemble_size, batch_size]
        """
        M, B, D = self.mu.shape
        mean_bc_per_member = torch.zeros(M, B, device=self.mu.device)

        for i in range(M):
            # Collect all other members
            mask = torch.ones(M, dtype=torch.bool, device=self.mu.device)
            mask[i] = False
            mu_i = self.mu[i].unsqueeze(0).repeat(M - 1, 1, 1)     # [M-1, B, D]
            std_i = self.std[i].unsqueeze(0).repeat(M - 1, 1, 1)   # [M-1, B, D]
            mu_others = self.mu[mask]                              # [M-1, B, D]
            std_others = self.std[mask]                            # [M-1, B, D]

            bc_ij = self._bhattacharyya_coefficient_diag(mu_i, std_i, mu_others, std_others)  # [M-1, B]
            mean_bc_per_member[i] = bc_ij.mean(dim=0)  # [B]

        mean_bc = mean_bc_per_member.mean(dim=0)  # [B]
        epistemic_uncertainty = 1.0 - mean_bc     # [B]
        return epistemic_uncertainty, mean_bc_per_member
