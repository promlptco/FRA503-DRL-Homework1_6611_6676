"""
RL_Algorithm/Function_based/PPO.py

Proximal Policy Optimization (PPO) — clipped surrogate + GAE.

Classification:
  - Type:        Actor-Critic
  - Policy:      Stochastic (Gaussian)
  - On/Off:      On-policy (multiple gradient steps over same rollout)
  - Action space: Continuous
  - Exploration: Stochastic policy + entropy bonus

All bugs fixed:
  1. T==0 guard — no NaN when rollout is empty
  2. gae.detach() — no graph chain accumulation
  3. returns.detach() — plain tensors in storage
  4. numel()>1 std guard — no warning on single-step rollout
  5. act() in no_grad + detach — no double-backward crash
  6. learn() uses all N parallel envs
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


class PPO(OnPolicyAlgorithm):
    def __init__(
        self,
        device                   = None,
        num_of_action:     int   = 1,
        action_range:      list  = [-10.0, 10.0],
        n_observations:    int   = 4,
        hidden_dim:        int   = 256,
        learning_rate:     float = 3e-4,
        discount_factor:   float = 0.99,
        gae_lambda:        float = 0.95,
        clip_param:        float = 0.2,
        value_loss_coef:   float = 0.5,
        entropy_coef:      float = 0.01,
        max_grad_norm:     float = 0.5,
        num_learning_epochs:  int = 4,
        num_mini_batches:     int = 4,
        num_transitions_per_env: int = 64,
        num_envs:          int   = 1,
    ):
        self.device = device if device is not None else torch.device('cpu')

        self.actor_base  = MLP(n_observations, hidden_dim, [hidden_dim, hidden_dim],
                               activation='elu').to(self.device)
        self.mu_head     = nn.Linear(hidden_dim, num_of_action).to(self.device)
        self.log_std     = nn.Parameter(torch.zeros(num_of_action, device=self.device))
        self.critic      = MLP(n_observations, 1, [hidden_dim, hidden_dim],
                               activation='elu').to(self.device)

        actor_params = (list(self.actor_base.parameters()) +
                        list(self.mu_head.parameters()) + [self.log_std])
        self.actor_optimizer  = optim.Adam(actor_params,             lr=learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)

        self.clip_param          = clip_param
        self.value_loss_coef     = value_loss_coef
        self.entropy_coef        = entropy_coef
        self.max_grad_norm       = max_grad_norm
        self.gae_lambda          = gae_lambda
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches    = num_mini_batches
        self.n_observations      = n_observations
        self.action_output_dim   = num_of_action
        self.num_transitions     = num_transitions_per_env
        self.num_envs            = num_envs

        self.episode_rewards = []
        self.episode_lengths = []

        super().__init__(
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            discount_factor = discount_factor,
        )
        self._init_storage(num_envs, num_transitions_per_env,
                           (n_observations,), (num_of_action,))

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _get_dist(self, obs):
        feat = self.actor_base(obs)
        mu   = self.mu_head(feat)
        std  = self.log_std.exp().clamp(1e-4, 1.0)
        return D.Normal(mu, std)

    def _scale_raw(self, raw):
        action_min, action_max = self.action_range
        return action_min + (torch.tanh(raw) + 1.0) * 0.5 * (action_max - action_min)

    # ------------------------------------------------------------------
    # Inference (deployment) — deterministic mean action
    # ------------------------------------------------------------------

    def act_inference(self, obs):
        """
        Deterministic action for deployment/play.py.
        Uses tanh(mu) — no sampling, no exploration noise.
        """
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            dist = self._get_dist(state)
        return self._scale_raw(dist.loc)   # mean of Gaussian, not a sample

    # ------------------------------------------------------------------
    # OnPolicyAlgorithm interface
    # ------------------------------------------------------------------

    def act(self, obs):
        """
        Sample action for rollout collection.
        Wrapped in no_grad — buffer stores plain detached tensors.
        Gradients are recomputed fresh inside update().
        """
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            dist  = self._get_dist(state)
            raw   = dist.rsample()
            log_p = dist.log_prob(raw).sum(-1)
            value = self.critic(state).squeeze(-1)

        self.transition.observations     = state.detach()
        self.transition.actions          = raw.detach()
        self.transition.actions_log_prob = log_p.detach()
        self.transition.values           = value.detach()
        self.transition.mu               = dist.loc.detach()
        self.transition.sigma            = dist.scale.detach()
        return self._scale_raw(raw)

    def process_env_step(self, rewards, dones):
        if rewards.dim() == 0: rewards = rewards.unsqueeze(0)
        if dones.dim()   == 0: dones   = dones.unsqueeze(0)
        self.transition.rewards = rewards.to(self.device).float()
        self.transition.dones   = dones.to(self.device).float()
        self.add_transition()

    def compute_returns(self, last_obs):
        """GAE-λ advantage and return computation. All results detached."""
        T = self.storage.step
        if T == 0:
            return

        if isinstance(last_obs, dict):
            last_state = last_obs['policy'].to(self.device).float()
        else:
            last_state = last_obs.to(self.device).float()
        if last_state.dim() == 1:
            last_state = last_state.unsqueeze(0)

        with torch.no_grad():
            last_value = self.critic(last_state).squeeze(-1)

        values  = torch.cat([self.storage.values[:T], last_value.unsqueeze(0)], dim=0)
        rewards = self.storage.rewards[:T]
        dones   = self.storage.dones[:T]

        advantages = torch.zeros_like(rewards)
        gae        = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            delta = (rewards[t] + self.discount_factor * values[t+1] *
                     (1.0 - dones[t]) - values[t])
            gae = (delta + self.discount_factor * self.gae_lambda *
                   (1.0 - dones[t]) * gae).detach()  # break graph chain
            advantages[t] = gae

        returns    = (advantages + self.storage.values[:T]).detach()
        advantages = advantages.detach()

        adv_flat = advantages.view(-1)
        if adv_flat.numel() > 1 and adv_flat.std() > 1e-8:
            advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        self.storage.advantages[:T] = advantages
        self.storage.returns[:T]    = returns

    def update(self):
        """PPO clipped surrogate over multiple epochs and mini-batches."""
        T = self.storage.step
        if T == 0:
            self.storage.clear()
            return None, None

        actor_losses, critic_losses = [], []

        for obs_b, acts_b, ret_b, adv_b, _, old_logp_b, _, _ in \
                self.storage.mini_batch_generator(self.num_mini_batches,
                                                  self.num_learning_epochs):

            dist     = self._get_dist(obs_b)
            log_prob = dist.log_prob(acts_b).sum(-1)
            entropy  = dist.entropy().sum(-1).mean()

            ratio = torch.exp(log_prob - old_logp_b.detach())
            surr1 = ratio * adv_b.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_b.detach()
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_base.parameters()) + list(self.mu_head.parameters()),
                self.max_grad_norm)
            self.actor_optimizer.step()

            values      = self.critic(obs_b).squeeze(-1)
            critic_loss = F.mse_loss(values, ret_b.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.storage.clear()
        return (sum(actor_losses) / len(actor_losses),
                sum(critic_losses) / len(critic_losses))

    # ------------------------------------------------------------------
    # Training loop — uses all N parallel envs
    # ------------------------------------------------------------------

    def learn(self, env, max_steps: int = 500):
        """
        Collect num_transitions steps across ALL parallel envs, then update.

        Returns:
            Tuple[float, int]: (mean reward across envs, total steps collected)
        """
        obs, _ = env.reset()
        N = obs['policy'].shape[0]

        if N != self.num_envs:
            self._init_storage(N, self.num_transitions,
                               (self.n_observations,), (self.action_output_dim,))
            self.num_envs = N

        cumulative_rewards = torch.zeros(N, device=self.device)
        active = torch.ones(N, dtype=torch.bool, device=self.device)
        steps  = 0

        while steps < max_steps:
            for _ in range(self.num_transitions):
                if steps >= max_steps:
                    break

                scaled  = self.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(scaled)

                r    = reward.to(self.device).float()
                term = terminated.to(self.device).float()

                self.process_env_step(r, term)
                cumulative_rewards += r * active.float()
                active = active & ~(terminated.bool() | truncated.bool()).to(self.device)

                steps += 1
                obs    = next_obs

            self.compute_returns(obs)
            self.update()

            if not active.any():
                break

        mean_reward = cumulative_rewards.mean().item()
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(steps)
        return mean_reward, steps

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor_base': self.actor_base.state_dict(),
            'mu_head':    self.mu_head.state_dict(),
            'log_std':    self.log_std.data,
            'critic':     self.critic.state_dict(),
        }, os.path.join(path, filename))

    def load_model(self, path, filename):
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device,
                          weights_only=True)
        self.actor_base.load_state_dict(ckpt['actor_base'])
        self.mu_head.load_state_dict(ckpt['mu_head'])
        self.log_std.data.copy_(ckpt['log_std'])
        self.critic.load_state_dict(ckpt['critic'])