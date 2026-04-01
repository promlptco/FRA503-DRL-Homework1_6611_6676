"""
RL_Algorithm/Function_based/A2C.py

Advantage Actor-Critic (A2C) — synchronous, TD-based (GAE) advantage estimation.

Classification:
  - Type:        Actor-Critic
  - Policy:      Stochastic (Gaussian)
  - On/Off:      On-policy
  - Action space: Continuous
  - Exploration: Stochastic policy + entropy bonus

All bugs fixed:
  1. T==0 guard — no NaN when rollout is empty
  2. gae.detach() — no 64-deep computation graph chain
  3. returns.detach() — plain tensors in storage
  4. numel()>1 std guard — no warning on single-step rollout
  5. act() wrapped in no_grad + detach — no double-backward
  6. learn() uses all N parallel envs — linear speedup with --num_envs
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


class A2C(OnPolicyAlgorithm):
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
        value_loss_coef:   float = 0.5,
        entropy_coef:      float = 0.01,
        num_transitions_per_env: int = 64,
        num_envs:          int   = 1,
    ):
        self.device = device if device is not None else torch.device('cpu')

        # Shared backbone + actor/critic heads
        self.backbone   = MLP(n_observations, hidden_dim, [hidden_dim],
                              activation='tanh').to(self.device)
        self.mu_head    = nn.Linear(hidden_dim, num_of_action).to(self.device)
        self.log_std    = nn.Parameter(torch.zeros(num_of_action, device=self.device))
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

        all_params = (list(self.backbone.parameters()) +
                      list(self.mu_head.parameters()) +
                      [self.log_std] +
                      list(self.value_head.parameters()))
        self.optimizer = optim.Adam(all_params, lr=learning_rate)

        self.gae_lambda        = gae_lambda
        self.value_loss_coef   = value_loss_coef
        self.entropy_coef      = entropy_coef
        self.n_observations    = n_observations
        self.action_output_dim = num_of_action
        self.num_transitions   = num_transitions_per_env
        self.num_envs          = num_envs

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

    def _forward(self, obs):
        """Shared forward: returns (mu, std, value)."""
        feat  = self.backbone(obs)
        mu    = self.mu_head(feat)
        std   = self.log_std.exp().clamp(1e-4, 1.0)
        value = self.value_head(feat).squeeze(-1)
        return mu, std, value

    def _dist(self, obs):
        mu, std, value = self._forward(obs)
        return D.Normal(mu, std), value

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
            mu, _, _ = self._forward(state)
        return self._scale_raw(mu)

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
            dist, value = self._dist(state)
            raw      = dist.rsample()
            log_prob = dist.log_prob(raw).sum(-1)

        self.transition.observations     = state.detach()
        self.transition.actions          = raw.detach()
        self.transition.actions_log_prob = log_prob.detach()
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
            _, last_value = self._dist(last_state)

        values  = torch.cat([self.storage.values[:T],
                             last_value.unsqueeze(0)], dim=0)
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
        """One gradient step over the complete rollout."""
        T = self.storage.step
        if T == 0:
            self.storage.clear()
            return None, None

        obs  = self.storage.observations[:T].view(-1, self.n_observations).detach()
        acts = self.storage.actions[:T].view(-1, self.action_output_dim).detach()
        rets = self.storage.returns[:T].view(-1).detach()
        advs = self.storage.advantages[:T].view(-1).detach()

        dist, values = self._dist(obs)
        log_prob = dist.log_prob(acts).sum(-1)
        entropy  = dist.entropy().sum(-1).mean()

        actor_loss  = -(log_prob * advs).mean()
        critic_loss = F.mse_loss(values, rets)
        loss        = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.mu_head.parameters()) +
            [self.log_std] + list(self.value_head.parameters()), 0.5)
        self.optimizer.step()
        self.storage.clear()
        return actor_loss.item(), critic_loss.item()

    # ------------------------------------------------------------------
    # Training loop — uses all N parallel envs
    # ------------------------------------------------------------------

    def learn(self, env, max_steps: int = 500):
        """
        Collect num_transitions steps across ALL parallel envs, then update.
        With N envs: N × num_transitions experiences per update → linear speedup.

        Returns:
            Tuple[float, int]: (mean reward across envs, total steps collected)
        """
        obs, _ = env.reset()
        N = obs['policy'].shape[0]

        # Re-init storage if env count changed (e.g. first call with --num_envs 128)
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

                scaled = self.act(obs)
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
            'backbone':   self.backbone.state_dict(),
            'mu_head':    self.mu_head.state_dict(),
            'log_std':    self.log_std.data,
            'value_head': self.value_head.state_dict(),
        }, os.path.join(path, filename))

    def load_model(self, path, filename):
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device,
                          weights_only=True)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.mu_head.load_state_dict(ckpt['mu_head'])
        self.log_std.data.copy_(ckpt['log_std'])
        self.value_head.load_state_dict(ckpt['value_head'])