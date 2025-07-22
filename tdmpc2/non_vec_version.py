
	#non vectorized version
	def _compute_reward_loss(self, _zs, action, rewards, member ,task):
		"""
		_zs: [3, 256 , 512] latent observation for 3 time steps, 256 samples (from batch size) and 512 latent dimension
		action: [3 , 256 , 4] actions for 3 time steps, 256 samples (from batch size) and 512 action dimenion  
		rewards : [3 , 256 , 1] actual rewards for 3 timesteps for 256 samples (from batch size) and 1 scaler reward

		divide the 256 samples into preferences 
		"""
		
		T, N, _ = _zs.shape

		# Discount vector
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		# Discounted sum of true rewards: [N, 1]
		discounted_true_rewards = torch.sum(rewards * discount_factors, dim=0).squeeze(-1)  # [N]
		#[0.212 , 0.143 , 0.4191 , 0.0378]
		# Generate all unique unordered pairs
		pair_indices = list(combinations(range(N), 2))  # [(i, j)]
		
		# Step 3: Compare rewards and filter
		filtered_pairs = []
		chosen_labels = []
		rejected_label = []

		for i, j in pair_indices:
			r_i = discounted_true_rewards[i]
			r_j = discounted_true_rewards[j]

			if r_i > r_j:
				chosen_labels.append(i)
				rejected_label.append(j)
			elif r_i < r_j:
				chosen_labels.append(j)
				rejected_label.append(i)

		# Convert to tensors [M] where M is the number of valid preference pairs
		chosen_indices = torch.tensor(chosen_labels, dtype=torch.long, device=_zs.device)
		rejected_indices = torch.tensor(rejected_label, dtype=torch.long, device=_zs.device)

		# Discounted sum of predicted rewards
		# Predict reward mean and var per time step
		means = []
		variances = []
		for t in range(T):
			z_t = _zs[t]           # [latent_dim]
			a_t = action[t]        # [action_dim]

			mu_t, var_t = self.model.reward_single_member(z_t, a_t, index=member, task=task)  # each [N]

			means.append(mu_t * discount_factors[t])                  # discount the mean
			variances.append(var_t * (discount_factors[t] ** 2))      # discount the variance properly
		means = torch.stack(means).unsqueeze(dim=-1)
		vars = torch.stack(variances).unsqueeze(dim=-1)
		total_means = means.sum(dim=0)        # [N]
		total_vars = vars.sum(dim=0)     # [N]

		#total means [-0.2419, -0.2370, -0.2330, -0.2453]
		#total var [0.7326, 0.7262, 0.7268, 0.7351]

		# Get chosen vs rejected
		mu_c = total_means[chosen_indices]     # [M]
		mu_r = total_means[rejected_indices]   # [M]
		var_c = total_vars[chosen_indices]     # [M]
		var_r = total_vars[rejected_indices]   # [M]

		#mu_c: [[-0.2419, -0.2330, -0.2419, -0.2330, -0.2370, -0.2330]
		#mu_r: [[-0.2370, -0.2419, -0.2453, -0.2370, -0.2453, -0.2453]]
		
		mean_z = mu_c - mu_r #mean_rejected - mean_chosen
		var_z = torch.sqrt(var_c**2 + var_r**2)
        
        # MC loss estimation
		num_sample = 1000
		z_samples = torch.randn(num_sample, var_z.shape[0], device=var_z.device, dtype=torch.float32)
		z_samples = z_samples * var_z.unsqueeze(0) + mean_z.unsqueeze(0)

		loss = -torch.nn.functional.logsigmoid(z_samples).mean()
		return loss

	def _compute_reward_loss(self, _zs, action, rewards , task):
		T, N, _ = _zs.shape
		
		# Discount vector
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		# Discounted sum of true rewards: [N, 1]
		discounted_true_rewards = torch.sum(rewards * discount_factors, dim=0).squeeze(-1)  # [N]

		# Discounted sum of predicted rewards
		discounted_pred_rewards = torch.zeros((self.cfg.num_r_d, N), device=self.cfg.cuda_device)  # [reward ensemble size, N]
		for t in range(T):
			discounted_pred_rewards =  discounted_pred_rewards + (self.model.reward(_zs[t] , action[t] , task)).squeeze(-1)  * self.cfg.rho**t
			
		# Generate all unique unordered pairs
		pair_indices = list(combinations(range(N), 2))  # [(i, j)]
		random.shuffle(pair_indices)
		pair_indices = torch.tensor(pair_indices, device=self.cfg.cuda_device)  # [num_pairs, 2]
		num_pairs = pair_indices.shape[0]
		chunk_size = num_pairs // self.cfg.num_r_d
		ensemble_pairs = pair_indices.view(self.cfg.num_r_d, chunk_size, 2).view(-1, 2)  # [num_r_d * chunk_size, 2]

		# True label
		r_ij_flat = discounted_true_rewards[ensemble_pairs]  # [num_r_d * chunk_size, 2]
		true_label = torch.full((r_ij_flat.shape[0],), -1, dtype=torch.int64, device=self.cfg.cuda_device)
		true_label[r_ij_flat[:, 0] > r_ij_flat[:, 1]] = 0
		true_label[r_ij_flat[:, 0] < r_ij_flat[:, 1]] = 1

		# Predicted label
		ensemble_pairs = pair_indices.view(self.cfg.num_r_d, chunk_size, 2)  # [num_r_d, chunk_size, 2]
		i_idx = ensemble_pairs[:, :, 0]  
		j_idx = ensemble_pairs[:, :, 1]
		pred_r_i = discounted_pred_rewards.gather(dim=1, index=i_idx)  # [num_r_d, chunk_size]
		pred_r_j = discounted_pred_rewards.gather(dim=1, index=j_idx)  # [num_r_d, chunk_size]
		logits = torch.stack([pred_r_i, pred_r_j], dim=-1).view(-1, 2)  # [num_r_d * chunk_size, 2]

		reward_loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, true_label)
		return reward_loss
	
	#computing reward and disagreement on preferences 
	def _compute_reward(self, z, actions, task):
		pred_reward_ens = self.model.reward(z, actions, task)[0].squeeze(-1)  # (Ensemble size x 512)
		r_i = pred_reward_ens.unsqueeze(2)  							   # [M, N, 1]
		r_j = pred_reward_ens.unsqueeze(1)  							   # [M, 1, N]

		# Assign labels as follows: r_i > r_j -> 0, r_i < r_j -> 1, r_i == r_j -> 0.5, this is only used for disagreement in preferences not actual labels for update
		preference = torch.where(
			r_i > r_j, 0.0,
			torch.where(r_i < r_j, 1.0, 0.5)
		)  # shape: [M, N, N]
		
		# Compute std over ensemble members, ignoring NaNs
		disagreement_std = torch.std(preference, dim=0 , unbiased=False)  # shape: [N, N]
		
		# Remove self-comparisons
		disagreement_std.fill_diagonal_(0)

		# Aggregate: mean disagreement for each sample across others
		disagreement = disagreement_std.mean(dim=1)  # shape: [N]
		# Just to to plot: to be removed
		# Normalized Ensemble Standard Deviation
		# normalize each model's predictions to zero mean and unit variance
		model_mean = pred_reward_ens.mean(dim=1, keepdim=True)  # [M, 1]
		model_std = pred_reward_ens.std(dim=1, keepdim=True) + 1e-8  # [M, 1]
		pred_reward_normalized = (pred_reward_ens - model_mean) / model_std  # [M, N]
		# Step 2: compute std across ensemble members 
		normalized_std = pred_reward_normalized.std(dim=0, unbiased=False, keepdim=True).T  # [N, 1]
		return pred_reward_ens.unsqueeze(-1) , disagreement.unsqueeze(-1) , normalized_std  # shape: [N,1]
	