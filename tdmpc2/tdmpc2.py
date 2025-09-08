import torch
import torch.nn.functional as F
import torch.nn as nn
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from common.pref_buffer import PrefBuffer

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
		self.log_rew_uncer_autotune = torch.nn.Parameter(torch.zeros(1 , device = self.device)) # initialize log(lambda)
		self.log_rew_uncer_autotune_optimizer = torch.optim.Adam([self.log_rew_uncer_autotune], lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.log_dyn_uncer_autotune = torch.nn.Parameter(torch.zeros(1 , device = self.device))  # initialize log(lambda)
		self.log_dyn_uncer_autotune_optimizer = torch.optim.Adam([self.log_dyn_uncer_autotune], lr=self.cfg.lr, eps=1e-5, capturable=True)
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
		if self.cfg.pref_learn:
			self.total_pref_feedback = 0
			self.pref_buffer = PrefBuffer(cfg)
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

		# Compute λ from log_lambda
		dyn_uncer_autotune = torch.exp(self.log_dyn_uncer_autotune) 
		rew_uncer_autotune = torch.exp(self.log_rew_uncer_autotune)  

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
			"rew_uncer_autotune": rew_uncer_autotune , #scalar
			"dyn_uncer_autotune": dyn_uncer_autotune , #scalar
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
		rew_uncer_lambda = torch.zeros_like(z[:, :1])
		dyn_uncer_lambda = torch.zeros_like(z[:, :1])
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

			# Compute λ from log_lambda
			dyn_uncer_autotune = torch.exp(self.log_dyn_uncer_autotune) 
			rew_uncer_autotune = torch.exp(self.log_rew_uncer_autotune)  

			# Adjusted reward (reward bonus shaping)
			adjusted_reward = reward + (rew_alpha * rew_uncer_autotune * reward_epi_uncer) + (dyn_beta * dyn_uncer_autotune * dyn_epi_uncer)
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
			rew_uncer_lambda += rew_uncer_autotune
			dyn_uncer_lambda += dyn_uncer_autotune
			epi_dyn += dyn_epi_uncer

		# Bootstrap value from final state
		action, _ = self.model.pi(z, task)
		value = G + discount * (1 - termination) * self.model.Q(z, action, task, return_type='avg')
		# Info for logging
		info = {
			"reward" : R,
			"reward_epistemic": epi_rew,
			"reward_aleatoric": alea_rew,
			"rew_uncer_autotune": rew_uncer_lambda / self.cfg.horizon,
			"dyn_uncer_autotune": dyn_uncer_lambda / self.cfg.horizon,
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
			"rew_uncer_autotune": elite_info["rew_uncer_autotune"][rand_idx].squeeze(0), #reward epi uncertainty of the random action chosen from the elite actions, scaler
			"dyn_uncer_autotune":  elite_info["dyn_uncer_autotune"][rand_idx].squeeze(0),#reward  aleatoric of the random action chosen from the elite actions, scaler
			"dyn_epistemic": elite_info["dynamics_epistemic"][rand_idx].squeeze(0), #dyn epi uncertainty of the random action chosen from the elite actions, scaler
			"ubp_reward":  elite_ubp_reward[rand_idx].squeeze(0)  #ubp reward of the random action chosen from the elite actions, scaler 
		}
		a, std= actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1), info


	def update_autotune(self , zs, task):
		action, _ = self.model.pi(zs, task)
		action_bar = self.model.pi_bar(zs, task)
		zs_stacked = zs.reshape(-1, zs.shape[-1])
		action_stacked = action.reshape(-1, action.shape[-1])
		action_bar_stacked = action_bar.reshape(-1, action_bar.shape[-1])

		# Compute Dynamics uncertainty
		_, dyn_epi_uncer = self.model.next(zs_stacked, action_stacked, task)
		_, dyn_epi_uncer_bar = self.model.next(zs_stacked, action_bar_stacked, task)

		#dyn uncertainty autotune update
		log_dyn_uncer_autotune_loss = (self.log_dyn_uncer_autotune * (dyn_epi_uncer - dyn_epi_uncer_bar).detach()).mean()  # detach diff so that gradients don't flow into pi
		# Backprop and optimizer step for dyn uncertainty autotune
		log_dyn_uncer_autotune_loss.backward()
		self.log_dyn_uncer_autotune_optimizer.step()
		self.log_dyn_uncer_autotune_optimizer.zero_grad()



		# Compute Reward uncertainty
		_ , reward_epi_uncer, _ = self.model.reward(zs_stacked, action_stacked, task)
		_ , reward_epi_uncer_bar, _ = self.model.reward(zs_stacked, action_bar_stacked, task)

		#reward uncertainty autotune update
		log_rew_uncer_autotune_loss = (self.log_rew_uncer_autotune * (reward_epi_uncer - reward_epi_uncer_bar).detach()).mean()  # detach diff so that gradients don't flow into pi
		# Backprop and optimizer step for reward uncertainty autotune
		log_rew_uncer_autotune_loss.backward()
		self.log_rew_uncer_autotune_optimizer.step()
		self.log_rew_uncer_autotune_optimizer.zero_grad()


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
			reward_loss = 0
			if self.cfg.pref_learn and self.total_pref_feedback >0:
				pref_z1, pref_z2, pref_a1, pref_a2, labels = self.pref_buffer.sample()
				# get logits
				rhat_1 = torch.zeros(self.cfg.num_pref_sampled , device=pref_z1.device)
				rhat_2 = torch.zeros(self.cfg.num_pref_sampled , device=pref_z2.device)
				for t in range(self.cfg.horizon):
					rhat_1 = rhat_1 + (self.model.reward_single_member(pref_z1[t], pref_a1[t], index=member, task=None)[0] * self.cfg.rho**t).squeeze(-1)
					rhat_2 = rhat_2 + (self.model.reward_single_member(pref_z2[t], pref_a2[t], index=member, task=None)[0] * self.cfg.rho**t).squeeze(-1)
				r_hat = torch.stack([rhat_1, rhat_2], dim=-1)  # shape: [batch_size, 2]
				reward_loss = nn.CrossEntropyLoss(ignore_index=-1)(r_hat, labels)
			else:
				# Compute losses
				
				for t in range(self.cfg.horizon):
					reward_pred_mean, reward_pred_var = self.model.reward_single_member(_zs[t], action[t], index=member, task=task)
					reward_loss = reward_loss + F.mse_loss(reward_pred_mean, reward[t]).mean() * self.cfg.rho**t
				reward_loss = reward_loss / self.cfg.horizon

			if self.cfg.episodic:
				termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

			# Compute losses
			value_loss = 0
			for t, (td_targets_unbind, qs_unbind) in enumerate(zip(td_targets.unbind(0), qs.unbind(1))):
				for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
					value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

			consistency_loss = consistency_loss / self.cfg.horizon
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

			# Update target policy
			with torch.no_grad():
				for param, target_param in zip(self.model._pi.parameters(), self.model._pi_bar.parameters()):
					target_param.data.mul_(1 - self.cfg.polyak_tau)
					target_param.data.add_(self.cfg.polyak_tau * param.data)

			# Update autotunning param
			self.uncer_autotune = self.update_autotune(zs.detach(), task)

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
		return info.detach().mean()


	def update(self, buffer, add_pref):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		if self.cfg.pref_learn and add_pref and self.total_pref_feedback < self.cfg.max_pref_feedback:
			pref_obs, pref_action, pref_reward, pref_terminated, pref_task = buffer.last_K()
			self._add_reward_pref(pref_obs, pref_action, pref_reward, pref_task)
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update_independant(obs, action, reward, terminated, **kwargs)

	# Helper functions for preference learning
	def _add_reward_pref(self, obs, action, reward, task):
		"""
		obs: [4, 256 , 512] latent observation for 4 time steps, 256 samples (from batch size) and 512 latent dimension
		action: [3 , 256 , 4] actions for 3 time steps, 256 samples (from batch size) and 512 action dimenion  
		rewards : [3 , 256 , 1] actual rewards for 3 timesteps for 256 samples (from batch size) and 1 scaler reward

		adds preference labels to buffer 
		"""
		with torch.no_grad():
			z = self.model.encode(obs[:-1], task) # [T,B, D] the latent dimension of the next states of the current observation in the selected sequence [observation at T=1,2,3] ie target latent states for the current obs
		
		z1, z2, a1, a2, tr1, tr2 = self.pref_create_pairs(z , action, reward, task)
		if self.cfg.optimistic_pref_sampling:
			optimistic_pref = self.pref_ranking_prob(z1, z2, a1, a2)
			top_opt_pref_index = (-optimistic_pref).argsort()[:self.cfg.num_pref_sampled]
		else:
			top_opt_pref_index = torch.randperm(z1.shape[1], device=z1.device)[:self.cfg.num_pref_sampled]

		# get labels
		labels = self.pref_label(z1[:,top_opt_pref_index,:],
								z2[:,top_opt_pref_index,:],
								a1[:,top_opt_pref_index,:],
								a2[:,top_opt_pref_index,:],
								tr1[:,top_opt_pref_index,:], 
								tr2[:,top_opt_pref_index,:])     
		self.pref_buffer.add(
			z1[:, top_opt_pref_index, :],
			z2[:, top_opt_pref_index, :],
			a1[:, top_opt_pref_index, :],
			a2[:, top_opt_pref_index, :],
			labels
		)
		self.total_pref_feedback = self.total_pref_feedback + self.cfg.num_pref_sampled


	def pref_ranking_prob(self, z1, z2, a1, a2):
		T, num_pairs, _ = z1.shape
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		probs = torch.zeros(self.cfg.num_r_d, num_pairs, device=z1.device)  # to store the probabilities from each member [num members, num pairs]
		for member in range(self.cfg.num_r_d):
			# Flatten from [T, N, ...] to [T*N, ...], Call model on flattened input, revert model output back to [T, N]
			r_hat_t1= self.model.reward_single_member(z1.reshape(T * num_pairs, -1), a1.reshape(T * num_pairs, -1), index=member, task=None)[0].view(T, num_pairs)  # only return mean reward [num pairs = N/2]
			r_hat_t2= self.model.reward_single_member(z2.reshape(T * num_pairs, -1),  a2.reshape(T * num_pairs, -1), index=member, task=None)[0].view(T, num_pairs)  # only return mean reward [num pairs = N/2]

			# Sum over time to get trajectory-level mean
			r1 = (r_hat_t1 * discount_factors.view(T, 1)).sum(dim=0)        # [num pairs = N/2]
			r2 = (r_hat_t2 * discount_factors.view(T, 1)).sum(dim=0)        # [num pairs = N/2]

			# Concatenate and softmax
			r_hat = torch.stack([r1, r2], dim=-1)        		  # [B//2, 2]
			probs[member] = F.softmax(r_hat, dim=-1)[:, 0]        # prob(traj1 > traj2)
			
		# compute mean and std along the "ensemble dimension"
		optimistic_pref = probs.mean(dim=0) + probs.std(dim=0) 
		return optimistic_pref

	def pref_label(self, z1, z2, a1, a2, tr1, tr2):
		T, num_pref, _ = z1.shape
		discount_factors = (self.cfg.rho ** torch.arange(T, device=self.cfg.cuda_device).float()).view(T, 1, 1)
		
		# Discounted sum of true rewards: [N, 1]
		disc_tr1 = (tr1 * discount_factors).sum(dim=0).view(num_pref) #[N]
		disc_tr2 = (tr2 * discount_factors).sum(dim=0).view(num_pref) #[N]

		# equally preferable
		margin_index = (torch.abs(disc_tr1 - disc_tr2) == 0).reshape(-1)
		#label 1 if trajectory 2 is better, 0 if trajectory 1 is better.
		labels = 1*(disc_tr1 < disc_tr2)
		# equally preferable
		labels[margin_index] = -1 		
		return labels

	def pref_create_pairs(self, z, action, reward, task):
		pref_batch_size = z.shape[1]
		perm = torch.randperm(pref_batch_size, device=z.device)  # random permutation of indices
		idx1 = perm[:(pref_batch_size//2)]   # first random half
		idx2 = perm[(pref_batch_size//2):]   # second random half

		z1, z2 = z[:, idx1], z[:, idx2]
		a1, a2 = action[:, idx1], action[:, idx2] 
		tr1, tr2 = reward[:, idx1], reward[:, idx2]
		return z1, z2, a1, a2, tr1, tr2
	