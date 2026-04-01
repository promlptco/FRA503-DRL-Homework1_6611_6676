"""
RL_Algorithm/Function_based/AC.py

Actor-Critic (AC) — episodic Monte Carlo variant.

Classification:
  - Type:        Actor-Critic
  - Policy:      Stochastic (Gaussian for continuous action)
  - On/Off:      On-policy
  - Action space: Continuous
  - Exploration: Stochastic policy (learned mean + std)

BUG FIXED (OOM kill):
  _init_storage() was called every single episode inside learn(), allocating
  a new RolloutBuffer on GPU each time without freeing the old one.
  Fix: allocate once in __init__ at max_episode_steps capacity, then
  call storage.clear() at the start of each episode.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import os

from RL_Algorithm.storage.on_policy import OnPolicyAlgorithm
from RL_Algorithm.network.mlp import MLP


class AC(OnPolicyAlgorithm):
    def __init__(
        self,
        device             = None,
        num_of_action: int = 1,
        action_range: list = [-10.0, 10.0],
        n_observations: int = 4,
        hidden_dim: int    = 256,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        entropy_coef: float  = 0.01,
        max_episode_steps: int = 500,  # pre-allocate buffer at this capacity
    ):
        self.device = device if device is not None else torch.device('cpu')

        self.actor_base = MLP(n_observations, hidden_dim, [hidden_dim],
                              activation='tanh').to(self.device)
        self.mu_head    = nn.Linear(hidden_dim, num_of_action).to(self.device)
        self.log_std    = nn.Parameter(torch.zeros(num_of_action, device=self.device))
        self.critic     = MLP(n_observations, 1, [hidden_dim, hidden_dim],
                              activation='tanh').to(self.device)

        actor_params = (list(self.actor_base.parameters()) +
                        list(self.mu_head.parameters()) + [self.log_std])
        self.actor_optimizer  = optim.Adam(actor_params,             lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.entropy_coef      = entropy_coef
        self.n_observations    = n_observations
        self.action_output_dim = num_of_action
        self.max_episode_steps = max_episode_steps

        self.episode_rewards = []
        self.episode_lengths = []

        super().__init__(
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            discount_factor = discount_factor,
        )

        # FIX: allocate once — reused every episode via storage.clear()
        self._init_storage(
            num_envs                = 1,
            num_transitions_per_env = max_episode_steps,
            obs_shape               = (n_observations,),
            actions_shape           = (num_of_action,),
        )

    def _get_distribution(self, obs_tensor):
        feat = self.actor_base(obs_tensor)
        mu   = self.mu_head(feat)
        std  = self.log_std.exp().clamp(1e-4, 1.0)
        return D.Normal(mu, std)

    def _scale_raw(self, raw):
        action_min, action_max = self.action_range
        return action_min + (torch.tanh(raw) + 1.0) * 0.5 * (action_max - action_min)

    def act_inference(self, obs):
        """
        Deterministic action for deployment (play.py).

        Uses tanh(mu) — the mean of the Gaussian policy — instead of sampling.
        No exploration noise. This is what the HW calls for in deployment evaluation.

        Args:
            obs: State observation (dict or tensor).

        Returns:
            torch.Tensor: Scaled deterministic action.
        """
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            feat = self.actor_base(state)
            mu   = self.mu_head(feat)          # mean of the Gaussian
        return self._scale_raw(mu)             # tanh(mu) scaled to action range

    def act(self, obs):
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Use no_grad for data collection — buffer stores plain tensors,
        # NOT computation graph nodes.  Gradients are recomputed fresh
        # inside update() when we call _get_distribution(obs) again.
        with torch.no_grad():
            dist     = self._get_distribution(state)
            raw      = dist.rsample()
            log_prob = dist.log_prob(raw).sum(-1)
            value    = self.critic(state).squeeze(-1)
            scaled   = self._scale_raw(raw)

        self.transition.observations     = state.detach()
        self.transition.actions          = raw.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.values           = value.detach()
        self.transition.mu               = dist.loc.detach()
        self.transition.sigma            = dist.scale.detach()
        return scaled

    def process_env_step(self, rewards, dones):
        if rewards.dim() == 0: rewards = rewards.unsqueeze(0)
        if dones.dim()   == 0: dones   = dones.unsqueeze(0)
        self.transition.rewards = rewards.to(self.device).float()
        self.transition.dones   = dones.to(self.device).float()
        self.add_transition()

    def compute_returns(self, last_obs):
        T = self.storage.step
        if T == 0:
            return
        rewards = self.storage.rewards[:T]
        dones   = self.storage.dones[:T]
        values  = self.storage.values[:T]

        returns = torch.zeros_like(rewards)
        G = torch.zeros(1, device=self.device)
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
            returns[t] = G.detach()  # detach to prevent building a 500-deep graph chain
        G = None  # free immediately

        advantages = returns - values
        adv_flat   = advantages.view(-1)
        if adv_flat.numel() > 1 and adv_flat.std() > 1e-8:
            advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        self.storage.returns[:T]    = returns
        self.storage.advantages[:T] = advantages

    def update(self):
        T = self.storage.step
        if T == 0:
            self.storage.clear()
            return None, None

        obs        = self.storage.observations[:T].view(-1, self.n_observations).detach()
        actions    = self.storage.actions[:T].view(-1, self.action_output_dim).detach()
        returns    = self.storage.returns[:T].view(-1).detach()
        advantages = self.storage.advantages[:T].view(-1).detach()

        values      = self.critic(obs).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        dist       = self._get_distribution(obs)
        log_prob   = dist.log_prob(actions).sum(-1)
        entropy    = dist.entropy().sum(-1).mean()
        actor_loss = -(log_prob * advantages.detach()).mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor_base.parameters()) + list(self.mu_head.parameters()), 0.5)
        self.actor_optimizer.step()

        self.storage.clear()
        return actor_loss.item(), critic_loss.item()

    def learn(self, env, max_steps: int = 500):
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        # FIX: clear the pre-allocated buffer — no re-allocation
        self.storage.clear()

        cumulative_reward = 0.0
        steps = 0
        done  = False

        while not done and steps < max_steps:
            if self.storage.step >= self.storage.T:
                break

            scaled_action = self.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r     = reward[0].unsqueeze(0)     if reward.dim()    > 0 else reward.unsqueeze(0)
            term  = terminated[0].unsqueeze(0) if terminated.dim() > 0 else terminated.unsqueeze(0)
            trunc = truncated[0].item()        if truncated.dim() > 0 else truncated.item()
            done  = bool(term.item()) or bool(trunc)

            self.process_env_step(r, term.float())
            cumulative_reward += r.item()
            steps += 1
            obs    = next_obs

        self.compute_returns(obs)
        self.update()

        self.episode_rewards.append(cumulative_reward)
        self.episode_lengths.append(steps)
        return cumulative_reward, steps

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor_base': self.actor_base.state_dict(),
            'mu_head':    self.mu_head.state_dict(),
            'log_std':    self.log_std.data,
            'critic':     self.critic.state_dict(),
        }, os.path.join(path, filename))

    def load_model(self, path, filename):
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor_base.load_state_dict(ckpt['actor_base'])
        self.mu_head.load_state_dict(ckpt['mu_head'])
        self.log_std.data.copy_(ckpt['log_std'])
        self.critic.load_state_dict(ckpt['critic'])