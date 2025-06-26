import numpy as np
class VoxelGrid:
	def __init__(self, obs_space_low, obs_space_high, voxel_size=0.01):
		"""
		Initialize a voxel grid for discretizing 3D space.
		
		Args:
			obs_space_low (np.ndarray): Lower bounds of observation space (shape [3])
			obs_space_high (np.ndarray): Upper bounds of observation space (shape [3])
			voxel_size (float): Size of each voxel (e.g., 0.01 for 1cm)
		"""
		self.obs_space_low = np.array(obs_space_low)
		self.obs_space_high = np.array(obs_space_high)
		self.voxel_size = voxel_size

		self.grid_shape = np.ceil((self.obs_space_high - self.obs_space_low) / voxel_size).astype(int)
		self.voxel_counts = np.zeros(self.grid_shape, dtype=np.int32)

	def update(self, obs_pointcloud):
		"""
		Update voxel visitation counts based on new observations.

		Args:
			obs_pointcloud (np.ndarray): Array of shape [N, 3] or [N, 6] where [:, :3] are positions
		"""
		positions = obs_pointcloud[:, :3]
		indices = np.floor((positions - self.obs_space_low) / self.voxel_size).astype(int)

		# Filter valid indices
		valid_mask = np.all((indices >= 0) & (indices < self.grid_shape), axis=1)
		indices = indices[valid_mask]

		# Increment counts
		np.add.at(self.voxel_counts, (indices[:, 0], indices[:, 1], indices[:, 2]), 1)

	def compute_entropy(self):
		"""
		Compute entropy of the current voxel visitation distribution.

		Returns:
			entropy_bits (float): Entropy of visitation in bits
			visited_voxel_ratio (float): Fraction of voxels that were visited
		"""
		counts = self.voxel_counts.flatten()
		counts = counts[counts > 0]

		if len(counts) == 0:
			return 0.0, 0.0

		probs = counts / counts.sum()
		entropy_nats = -np.sum(probs * np.log(probs + 1e-8))
		entropy_bits = entropy_nats / np.log(2)
		visited_voxel_ratio = len(counts) / self.voxel_counts.size

		return entropy_bits, visited_voxel_ratio
