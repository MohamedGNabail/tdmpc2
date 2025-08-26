import torch
import torch.nn.functional as F
import torch.nn as nn
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from common.loss import gaussian_nll_loss
from common.rms import RunningMeanStd
from tensordict import TensorDict

from itertools import combinations
import random

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device(self.cfg.cuda_device)
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else [] , 'lr': self.cfg.lr / self.cfg.num_r_d},
			{'params': self.model._Qs.parameters() , 'lr': self.cfg.lr / self.cfg.num_r_d},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.rnd_optim = torch.optim.Adam(self.model._rnd_predictor.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.cfg.cuda_device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
		self.reward_int_rms = RunningMeanStd(shape=(), device=self.device)
	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def rand_act(self, obs, env, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs (torch.Tensor): Observation from environment.
			env: Environment instance.
			eval_mode (bool): If True, turn off uncertainty bonuses.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			action (torch.Tensor): Action to take (shape: [action_dim]).
			info (dict): Dictionary of extra outputs like reward and uncertainty terms.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)  # [1, obs_dim]
		action = env.rand_act()
		action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, act_dim]

		if task is not None:
			task = torch.tensor([task], device=self.device)

		# Encode observation
		z = self.model.encode(obs, task)  # [1, latent_dim]

		# Reward prediction
		reward, reward_epi_uncer, reward_aleatoric_uncer = self.model.reward(z, action, task)

		# Transition prediction
		z, dyn_epi_uncer = self.model.next(z, action, task)

		# Uncertainty weighting (disabled during eval)
		dyn_beta = 0 if eval_mode else self.cfg.dyn_uncer_beta_coef
		rew_alpha = 0 if eval_mode else self.cfg.rew_uncer_alpha_coef

		adjusted_reward = reward + (rew_alpha * reward_epi_uncer) + (dyn_beta * dyn_epi_uncer)

		# Collect info
		info = {
			"value" : 0 ,
			"reward": reward.squeeze(0),                     # scalar
			"reward_epistemic": reward_epi_uncer.squeeze(0), # scalar
			"reward_aleatoric": reward_aleatoric_uncer.squeeze(0), # scalar
			"dyn_epistemic": dyn_epi_uncer.squeeze(0),       # scalar
			"ubp_reward": adjusted_reward.squeeze(0),   # scalar
		}
		#TODO: Nitpicking value is not calculated for rand action, and since ubp does not use value, it is not needed now but it would be a nice plot to have 
		return action.clamp(-1, 1).squeeze(0), info

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions.
		eval: N = number of samples 512, num_ensemble = 5 , D = 512
			z = [512,512] [N , D]
			actions = [3,512,4] [T, N , A]
			eval_mode = True
		"""

		# Accumulators for summing over time
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		G          = torch.zeros_like(z[:, :1])      # [N,1]
		R          = torch.zeros_like(z[:, :1])
		discount   = torch.ones(1, device=z.device)  # scalar tensor
		ubp_reward = torch.zeros_like(z[:, :1])
		epi_rew    = torch.zeros_like(z[:, :1])
		alea_rew   = torch.zeros_like(z[:, :1])
		epi_dyn    = torch.zeros_like(z[:, :1])

		for t in range(self.cfg.horizon):
			# reward_ens = math.two_hot_inv(reward_ens, self.cfg) removed because the preference model is deterministic
			# Reward prediction: reward =  [N , 1] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			reward, reward_epi_uncer, reward_aleatoric_uncer = self.model.reward(z, actions[t], task)
			
			# Dynamics prediction
			# Next State prediction: reward =  [N , D] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			z, dyn_epi_uncer = self.model.next(z, actions[t], task)

			dyn_beta = 0 if eval_mode else self.cfg.dyn_uncer_beta_coef
			rew_alpha = 0 if eval_mode else self.cfg.rew_uncer_alpha_coef

			# Adjusted reward (reward bonus shaping)
			# Get mean magnitude of reward predictions for computing penalty, reward should not be negative in metaworld tasks but it is safer not to confuse the results by negating the signal of the alpha coefficient
			# adjusted_reward = reward + (rew_alpha * reward_epi_uncer) + (dyn_beta * dyn_epi_uncer)
			with torch.no_grad():
				latent_rnd = self.model.rnd_predict(z, actions[t], task)
				latent_rnd_target = self.model.rnd_target(z, actions[t], task)
				raw_int_reward = F.mse_loss(latent_rnd_target, latent_rnd, reduction='none').mean(dim=-1 , keepdim=True)
		 
	
			# Normalize intrinsic reward , just by variance, don't adjust the mean
			self.reward_int_rms.update(raw_int_reward)
			int_reward = raw_int_reward / torch.sqrt(self.reward_int_rms.var + 1e-8)

			adjusted_reward = reward + (rew_alpha * int_reward)
			G = G + discount * (1 - termination) * adjusted_reward

			# Discount update
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update

			# Termination update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)

			# Sum all quantities over time [N,1]
			R = R + reward
			ubp_reward += adjusted_reward
			epi_rew += reward_epi_uncer
			alea_rew += reward_aleatoric_uncer
			epi_dyn += dyn_epi_uncer

		# Bootstrap value from final state
		action, _ = self.model.pi(z, task)
		value = G + discount * (1 - termination) * self.model.Q(z, action, task, return_type='avg')
		# Info for logging
		info = {
			"reward" : R,
			"reward_epistemic": int_reward,
			"reward_aleatoric": alea_rew,
			"dynamics_epistemic": epi_dyn,
		}
		value = value.nan_to_num(0)
		ubp_reward = ubp_reward.nan_to_num(0)
		return value, ubp_reward, info


	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs(torch.Tensor): state from which to plan. [1,39]
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the uncertainties in reward and dynamics in planning, if true , uncertainty is not used
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Latent Space encoding for the current observation [1,Latent Dimension D]
		z = self.model.encode(obs, task)

		# Sample policy trajectories [24]
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			# Repeated State [self.cfg.num_pi_trajs, D]
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			# Actions sampled from policy [T,self.cfg.num_pi_trajs,A]
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z , _ = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		# Repeated State [N, D]
		z = z.repeat(self.cfg.num_samples, 1)
		#Mean for action [T, A]
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		#std for action [T, A]
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		
		#Actions  are [T, N , A] filling the first self.cfg.num_pi_trajs number of them with [T, self.cfg.num_pi_trajs , A]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions for non policy sampled actions (empty ones)
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute value, reward, and associated uncertainty info 
			# value : [N,1] , ubp_reward[N,1] , infos [N, 1]
			value, ubp_reward, info = self._estimate_value(z, actions, task, eval_mode)
			

			# Define metrics with default fallback to "ubp_reward"
			planning_metric_map = {
				"total_value": value.squeeze(1),
				"ubp_reward": ubp_reward.squeeze(1),
			}
			metric_key = getattr(self.cfg, "planning_criteria", "ubp_reward")
			metric_values = planning_metric_map.get(metric_key, planning_metric_map["ubp_reward"]) #the ubp reward [N,1]

			# Top-k selection [64]
			elite_idxs = torch.topk(metric_values, self.cfg.num_elites, dim=0).indices  # [num_elites]

			# Extract elite values and actions
			elite_value = value[elite_idxs]                   # [num_elites, 1]
			elite_ubp_reward = ubp_reward[elite_idxs]         # [num_elites, 1]
			elite_actions = actions[:, elite_idxs]            # [horizon, num_elites, action_dim]

			# Extract elite info tensors
			elite_info = {k: v[elite_idxs] for k, v in info.items()}  #[ num elites, 1]

			# Elite scoring
			elite_metric = metric_values[elite_idxs].unsqueeze(1)        # [num_elites, 1]
			max_metric = elite_metric.max(0).values
			score = torch.exp(self.cfg.temperature * (elite_metric - max_metric))  # [num_elites, 1]
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9) #[1, num elite, 1] * [T,num elite,A] = weighted actinos by score [T, num elite, A] , sum dim 1 : [T,A] , divide by scaler , mean is T,A
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std) 							#[T,A]
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1)) #scaler
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1) #[T,A]
		# Final info dict
		info = {
			"value":  elite_value[rand_idx].squeeze(0), #value of the random action chosen from the elite actions, scaler 
			"reward": elite_info["reward"][rand_idx].squeeze(0), #reward of the random action chosen from the elite actions, scaler 
			"reward_epistemic": elite_info["reward_epistemic"][rand_idx].squeeze(0), #reward epi uncertainty of the random action chosen from the elite actions, scaler
			"reward_aleatoric":  elite_info["reward_aleatoric"][rand_idx].squeeze(0),#reward  aleatoric of the random action chosen from the elite actions, scaler
			"dyn_epistemic": elite_info["dynamics_epistemic"][rand_idx].squeeze(0), #dyn epi uncertainty of the random action chosen from the elite actions, scaler
			"ubp_reward":  elite_ubp_reward[rand_idx].squeeze(0)  #ubp reward of the random action chosen from the elite actions, scaler 
		}
		a, std= actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1), info
	
	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	def update_rnd(self, _zs, action, task):
		"""
		Update rnd predictor network using a the first 3 timesteps of latent states for an observation _zs.

		Args:
			_zs (torch.Tensor): Sequence of latent states.
			action (torch.Tensor): Sequence of actions.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the rnd update.
		"""
		rnd_loss = torch.tensor(0.0, device=_zs.device)
		for t in range(self.cfg.horizon):
			latent_rnd = self.model.rnd_predict(_zs[t], action[t], task)
			latent_rnd_target = self.model.rnd_target(_zs[t], action[t], task).detach()
			rnd_loss = rnd_loss + F.mse_loss(latent_rnd_target, latent_rnd).mean()
			
		rnd_loss = rnd_loss / self.cfg.horizon

        # Backpropagation and weight update
		rnd_loss.backward()
		self.rnd_optim.step()
		self.rnd_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"rnd_loss": rnd_loss,
		})
		return info
















	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _compute_reward_loss(self, _zs, action, rewards, member ,task):
		"""
		_zs: [3, 256 , 512] latent observation for 3 time steps, 256 samples (from batch size) and 512 latent dimension
		action: [3 , 256 , 4] actions for 3 time steps, 256 samples (from batch size) and 512 action dimenion  
		rewards : [3 , 256 , 1] actual rewards for 3 timesteps for 256 samples (from batch size) and 1 scaler reward

		returns reward loss for preferences using sigmoid loss 
		"""
		T, N, _ = _zs.shape

		# Discount vector
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		# Discounted sum of true rewards: [N, 1]
		discounted_true_rewards = (rewards * discount_factors).sum(dim=0).view(N) #[N]
		pair_indices = list(combinations(range(N), 2))  # [(i, j)]
		# Step 1: Convert to index tensors
		pair_indices = torch.tensor(pair_indices, device=_zs.device)  # [M, 2]
		i_idx = pair_indices[:, 0]  # [M]
		j_idx = pair_indices[:, 1]  # [M]

		# Step 2: Get rewards for both sides
		r_i = discounted_true_rewards[i_idx]  # [M]
		r_j = discounted_true_rewards[j_idx]  # [M]

		# Step 3: Compare and filter
		# Mask where rewards are not equal
		not_equal = r_i != r_j  # [M]
		greater = r_i > r_j     # [M]

		# Filter only non-equal pairs
		i_idx = i_idx[not_equal]
		j_idx = j_idx[not_equal]
		greater = greater[not_equal]

		# Step 4: Assign chosen/rejected based on comparison
		chosen_indices = torch.where(greater, i_idx, j_idx)     # [M']
		rejected_indices = torch.where(greater, j_idx, i_idx)   # [M']
		

		# Flatten from [T, N, ...] to [T*N, ...]
		zs_flat = _zs.reshape(T * N, -1)            # [T*N, latent_dim]
		action_flat = action.reshape(T * N, -1)     # [T*N, action_dim]

		# Call model on flattened input
		mu_flat, var_flat = self.model.reward_single_member(zs_flat, action_flat, index=member, task=task)  # [T*N]

		# Reshape back to [T, N]
		mu = mu_flat.view(T, N)
		var = var_flat.view(T, N)

		# Sum over time to get trajectory-level mean/var
		total_means = (mu * discount_factors.view(T, 1)).sum(dim=0)        # [N]
		total_vars = (var * (discount_factors.view(T, 1) ** 2)).sum(dim=0)     # [N]

		# Get chosen vs rejected
		var_c = total_vars[chosen_indices]     # [M]
		var_r = total_vars[rejected_indices]   # [M]

		mean_z = total_means[chosen_indices]  - total_means[rejected_indices]
		var_z = torch.sqrt(var_c**2 + var_r**2 + 1e-8)

		# MC loss estimation
		num_sample = 1000
		z_samples = torch.randn(num_sample, var_z.size(0), device=var_z.device, dtype=torch.float32)
		z_samples = z_samples * var_z.unsqueeze(0) + mean_z.unsqueeze(0)
		loss = -torch.nn.functional.logsigmoid(z_samples).mean()
		return loss

	
	

	def _update_independant(self, obs, action, reward, terminated, task=None):
		"""
		# Updating the loss is not dependant on episetmic uncertainty, the variance of each single member is only used for its loss computation only
		obs: A batch of [batch size], each element in the batch is four conseqtive observations (4 because T = 3 +1), each observation is obs dimension [T+1,B,39]
		action: A batch of [batch size], each element in the batch is three conseqtive actions (because T = 3), each action is action dimension [T,B,4]
		reward: A batch of [batch size], each element in the batch is three conseqtive rewards (because T = 3), each reward is scaler [T,B,1]
		terminated: A batch of [batch size], each element in the batch is three conseqtive terminated status (because T = 3), each reward is scaler [T,B,1]
		"""
		consistency_loss_all, reward_loss_all, value_loss_all, termination_loss_all, total_loss_all, grad_norm_all = 0,0,0,0,0,0
		for member in range(self.cfg.num_r_d):
			#Compute Targets using the true rewards and true next states (encoded) to be passed to the Q model. 
			# True input used : rewards, terminated status, next observed stated. 
			# World Model prediction is used in encoded next states, Q value estimate given the encoded next states and predicted action from policy
			# #TODO MN: Why were true actions not used even though they are available from the buffer
			with torch.no_grad():
				next_z = self.model.encode(obs[1:], task) # [T,B, D] the latent dimension of the next states of the current observation in the selected sequence [observation at T=2,3,4] ie target next states for the current state
				td_targets = self._td_target(next_z, reward, terminated, task) #[T,B,1] target expected return starting from current state

			# Prepare for update
			self.model.train()

			# Latent rollout
			# zs is the Predicted states, except for the first observation in the sequence because it is the first one, can not be predicetd given something. [4 (T+1), Batch Size , Latent Dimension]
			zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
			# encode the first observation of each sequence [B, D]
			z = self.model.encode(obs[0], task)
			# fill the latent state of the non predictable first observation
			zs[0] = z
			consistency_loss = 0
			for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
				z = self.model.next_single_member(z, _action, member, task)  # mu shape  is [batch_size, latent_dim] , var shape is [batch size,latent dim]
				consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
				zs[t+1] = z #fill each predicted latent observation in zs

			# Predictions
			# zs are the Predicted states , since no action is associated with the last state, no reward , no terminated signal. It is not usable, hence exclude the last predicted state. 
			_zs = zs[:-1] #[T,B , D]
			qs = self.model.Q(_zs, action, task, return_type='all')

			if self.cfg.pref_learn:
				reward_loss = self._compute_reward_loss(_zs, action, reward, member ,task)
			else:
				# Compute losses
				reward_loss = 0
				for t in range(self.cfg.horizon):
					reward_pred_mean, reward_pred_var = self.model.reward_single_member(_zs[t], action[t], index=member, task=task)
					reward_loss = reward_loss + F.mse_loss(reward_pred_mean, reward[t]).mean() * self.cfg.rho**t
				
			if self.cfg.episodic:
				termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

			# Compute losses
			value_loss = 0
			for t, (td_targets_unbind, qs_unbind) in enumerate(zip(td_targets.unbind(0), qs.unbind(1))):
				for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
					value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

			consistency_loss = consistency_loss / self.cfg.horizon
			reward_loss = reward_loss / self.cfg.horizon
			if self.cfg.episodic:
				termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
			else:
				termination_loss = 0.
			value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.termination_coef * termination_loss +
				self.cfg.value_coef * value_loss
			)

			# Update model
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			self.optim.step()
			self.optim.zero_grad(set_to_none=True)

			# Update policy
			pi_info = self.update_pi(zs.detach(), task)

			#Update Predictor RND network
			rnd_info = self.update_rnd (_zs.detach(), action, task)

			# Update target Q-functions
			self.model.soft_update_target_Q()
			consistency_loss_all =consistency_loss_all + consistency_loss 
			reward_loss_all = reward_loss_all + reward_loss 
			value_loss_all = value_loss_all + value_loss
			termination_loss_all = termination_loss_all + termination_loss 
			total_loss_all = total_loss_all + total_loss
			grad_norm_all = grad_norm_all + grad_norm 

		
		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss_all / self.cfg.num_r_d,
			"reward_loss": reward_loss_all / self.cfg.num_r_d,
			"value_loss": value_loss_all / self.cfg.num_r_d,
			"termination_loss": termination_loss_all / self.cfg.num_r_d,
			"total_loss": total_loss_all / self.cfg.num_r_d,
			"grad_norm": grad_norm_all / self.cfg.num_r_d,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		info.update(rnd_info)
		return info.detach().mean()

	def update(self, buffer, step):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update_independant(obs, action, reward, terminated, **kwargs)
