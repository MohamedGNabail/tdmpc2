import torch
import numpy as np


class PrefBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.capacity = cfg.max_pref_feedback
        self.buffer = []
        self.pos = 0

    def add(self, z1, z2, a1, a2, labels):
        # Flatten to trajectory-level storage
        batch_size = labels.shape[0]
        for i in range(batch_size):
            data = (
                z1[:,i,:], z2[:,i,:], a1[:,i,:], a2[:,i,:], labels[i]
            )
            if len(self.buffer) < self.capacity:
                self.buffer.append(data)
            else:
                self.buffer[self.pos] = data
                self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        idx = np.random.choice(len(self.buffer), self.cfg.num_pref_sampled, replace=False)
        batch = [self.buffer[i] for i in idx]
        z1, z2, a1, a2, labels = zip(*batch)
        z1 = torch.stack(z1).permute(1, 0, 2)  # [T, N, D]
        z2 = torch.stack(z2).permute(1, 0, 2)
        a1 = torch.stack(a1).permute(1, 0, 2)
        a2 = torch.stack(a2).permute(1, 0, 2)
        labels = torch.stack(labels)  
        return z1, z2, a1, a2, labels
    def __len__(self):
        return len(self.buffer)