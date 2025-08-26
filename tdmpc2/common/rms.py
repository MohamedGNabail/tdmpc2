import torch

class RunningMeanStd:
    """Keeps running estimates of mean and variance (Welford’s algorithm)."""
    def __init__(self, epsilon=1e-4, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, x: torch.Tensor):
        x = x.to(self.mean.device)   # ensure same device
        x = x.view(-1).float()       # flatten to 1D
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
