import torch
import torch.nn.functional as F
import torch.nn as nn
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
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
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.cfg.cuda_device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self.training_step = 0
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

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
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):

			# reward_ens = math.two_hot_inv(reward_ens, self.cfg) removed because the preference model is deterministic
			reward_ens, reward_disagreement, reward_numeric_uncer = self._compute_reward(z, actions[t], task)
			reward = reward_ens.mean(dim=0)
			z_ens , dyn_disagreement = self.model.next(z, actions[t], task)                    #(5 x 512 x 512) num_ensemble x batch_size x latent_dim , (512 x 1) batch_size x 1
			z = z_ens.mean(dim=0)                                                              #(512 x 512) batch_size x latent_dim
			dyn_uncer_beta_coef = 0 if eval_mode else self.cfg.dyn_uncer_beta_coef
			rew_uncer_alpha_coef = 0 if eval_mode else self.cfg.rew_uncer_alpha_coef
			dyn_uncer = dyn_uncer_beta_coef * dyn_disagreement                                #(512 x 1) batch_size x 1
			reward_uncer = rew_uncer_alpha_coef * reward_disagreement                           #(512 x 1) batch_size x 1
			reward_num_uncer = rew_uncer_alpha_coef * reward_numeric_uncer                           #(512 x 1) batch_size x 1
			adjusted_reward = reward + reward_uncer + dyn_uncer
			G = G + discount * (1-termination) * adjusted_reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		value  = G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')
		if eval_mode:
			assert (reward_uncer == 0).all(), f"Non-zero reward_uncer in eval_mode: {reward_uncer}"
			assert (dyn_uncer == 0).all(), f"Non-zero dyn_uncer in eval_mode: {dyn_uncer}"
		else:
			assert (reward_uncer >= 0).all(), f"Negative reward_uncer: {reward_uncer[reward_uncer < 0]}"
			assert (dyn_uncer >= 0).all(), f"Negative dyn_uncer: {dyn_uncer[dyn_uncer < 0]}"
		return value, reward, reward_uncer, dyn_uncer , adjusted_reward, reward_num_uncer

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				z_ens , _ = self.model.next(_z, pi_actions[t], task)
				_z = z_ens.mean(dim=0)  # Use mean of ensemble predictions
				pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value, pred_reward, reward_uncer, dyn_uncer , adjusted_pred_reward , reward_num_uncer = self._estimate_value(z, actions, task, eval_mode)
			value = value.nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions , elite_pred_rewards , elite_rew_uncer , elite_dyn_uncer , elite_adjusted_reward , elite_num_rew_uncer \
			= value[elite_idxs], actions[:, elite_idxs] , pred_reward[elite_idxs], reward_uncer[elite_idxs], dyn_uncer[elite_idxs] , adjusted_pred_reward[elite_idxs] ,  reward_num_uncer[elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		pred_values = torch.index_select(elite_value, 0, rand_idx).squeeze(1)
		reward_uncer = torch.index_select(elite_rew_uncer, 0, rand_idx).squeeze(1)
		reward_num_uncer = torch.index_select(elite_num_rew_uncer, 0, rand_idx).squeeze(1)
		dyn_uncer = torch.index_select(elite_dyn_uncer, 0, rand_idx).squeeze(1)
		adjusted_pred_reward = torch.index_select(elite_adjusted_reward, 0, rand_idx).squeeze(1)
		pred_reward = torch.index_select(elite_pred_rewards, 0, rand_idx)
		a, std= actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1), pred_values[0] , pred_reward[0], reward_uncer[0], dyn_uncer[0], adjusted_pred_reward[0], reward_num_uncer[0]

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


	def _compute_dynamics_loss(self, obs, action, next_z, task):
		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z

		num_ensemble = self.cfg.num_d                # 5
		batch_size = z.shape[0]                      # 256
		split_sizes = [batch_size // num_ensemble] * num_ensemble
		split_sizes[-1] += batch_size % num_ensemble  # Add remainder to last chunk

		# Create data partition indices
		indices = torch.arange(batch_size, device=z.device)
		split_indices = torch.split(indices, split_sizes)
		consistency_loss = 0.0
		for member, idx in enumerate(split_indices):
			for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
				# Predict next latent state for all ensemble members, then select this member's
				z_ens , _ = self.model.next(z[idx], _action[idx], task)  # shape [ensemble_size, chunk_size, latent_dim]
				z[idx] = z_ens[member]  # shape [chunk_size, latent_dim]

				# Compute loss and update latent
				consistency_loss = consistency_loss + F.mse_loss(z[idx], _next_z[idx]) * self.cfg.rho**t
				zs[t + 1][idx] = z[idx]
		return consistency_loss, zs  

	@torch._dynamo.disable
	def _compute_reward_loss(self, _zs, action, rewards , task):
		T, N, _ = _zs.shape
		
		# Discount vector
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		# Discounted sum of true rewards: [N, 1]
		discounted_true_rewards = torch.sum(rewards * discount_factors, dim=0).squeeze(-1)  # [N]

		# Discounted sum of predicted rewards
		discounted_pred_rewards = torch.zeros((self.cfg.num_r, N), device=self.cfg.cuda_device)  # [reward ensemble size, N]
		for t in range(T):
			discounted_pred_rewards =  discounted_pred_rewards + (self.model.reward(_zs[t] , action[t] , task)).squeeze(-1)  * self.cfg.rho**t

		# Generate all unique unordered pairs
		pair_indices = list(combinations(range(N), 2))  # [(i, j)]
		random.shuffle(pair_indices)
		pair_indices = torch.tensor(pair_indices, device=self.cfg.cuda_device)  # [num_pairs, 2]
		num_pairs = pair_indices.shape[0]
		chunk_size = num_pairs // self.cfg.num_r
		ensemble_pairs = pair_indices.view(self.cfg.num_r, chunk_size, 2).view(-1, 2)  # [num_r * chunk_size, 2]

		# True label
		r_ij_flat = discounted_true_rewards[ensemble_pairs]  # [num_r * chunk_size, 2]
		true_label = torch.full((r_ij_flat.shape[0],), -1, dtype=torch.int64, device=self.cfg.cuda_device)
		true_label[r_ij_flat[:, 0] > r_ij_flat[:, 1]] = 0
		true_label[r_ij_flat[:, 0] < r_ij_flat[:, 1]] = 1

		# Predicted label
		ensemble_pairs = pair_indices.view(self.cfg.num_r, chunk_size, 2)  # [num_r, chunk_size, 2]
		i_idx = ensemble_pairs[:, :, 0]  
		j_idx = ensemble_pairs[:, :, 1]
		pred_r_i = discounted_pred_rewards.gather(dim=1, index=i_idx)  # [num_r, chunk_size]
		pred_r_j = discounted_pred_rewards.gather(dim=1, index=j_idx)  # [num_r, chunk_size]
		logits = torch.stack([pred_r_i, pred_r_j], dim=-1).view(-1, 2)  # [num_r * chunk_size, 2]

		reward_loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, true_label)
		return reward_loss

	def _compute_reward(self, z, actions, task):
		pred_reward_ens = self.model.reward(z, actions, task).squeeze(-1)  # (5 x 512)
		r_i = pred_reward_ens.unsqueeze(2)  # [M, N, 1]
		r_j = pred_reward_ens.unsqueeze(1)  # [M, 1, N]

		# Assign flipped labels as specified
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
		
	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			ensemble_pred , _ = self.model.next(z, _action, task)  # shape [batch_size, latent_dim]
			z = ensemble_pred.mean(dim=0) 
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')

		reward_loss = self._compute_reward_loss(_zs, action, reward, task)

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

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
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
		self.training_step = step
		return self._update(obs, action, reward, terminated, **kwargs)
