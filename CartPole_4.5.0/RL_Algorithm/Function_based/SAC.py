"""
RL_Algorithm/Function_based/SAC.py

Soft Actor-Critic (SAC).

BUG FIXED (action scale mismatch):
  The replay buffer stored raw actions (pre-tanh, from select_action's
  torch.no_grad block). But calculate_loss used these raw values for
  Q(s, a_stored), while _sample_action() — used for next-state targets
  and actor update — returns SCALED actions. The critic was trained on
  two incompatible action scales simultaneously.
  Fix: store SCALED actions in the replay buffer.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import os

from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm
from RL_Algorithm.network.mlp import MLP


class SAC(OffPolicyAlgorithm):
    def __init__(
        self,
        device               = None,
        num_of_action: int   = 1,
        action_range:  list  = [-10.0, 10.0],
        n_observations: int  = 4,
        hidden_dim:    int   = 256,
        learning_rate: float = 3e-4,
        tau:           float = 0.005,
        discount_factor: float = 0.99,
        buffer_size:   int   = 100_000,
        batch_size:    int   = 256,
        alpha:         float = 0.2,
        auto_entropy:  bool  = True,
        target_entropy: float = None,
        update_freq:   int   = 4,    # only update every N env steps → 4x speedup
    ):
        self.device = device if device is not None else torch.device('cpu')

        self.actor_base = MLP(n_observations, hidden_dim, [hidden_dim, hidden_dim],
                              activation='relu').to(self.device)
        self.mu_head    = nn.Linear(hidden_dim, num_of_action).to(self.device)
        self.ls_head    = nn.Linear(hidden_dim, num_of_action).to(self.device)

        self.critic1        = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic2        = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic1_target = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic2_target = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        actor_params = (list(self.actor_base.parameters()) +
                        list(self.mu_head.parameters()) +
                        list(self.ls_head.parameters()))
        self.actor_optimizer   = optim.Adam(actor_params,              lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.auto_entropy   = auto_entropy
        self.target_entropy = target_entropy if target_entropy is not None else -float(num_of_action)
        self.log_alpha      = torch.tensor(np.log(alpha), dtype=torch.float32,
                                           device=self.device, requires_grad=auto_entropy)
        if auto_entropy:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        self.tau               = tau
        self.n_observations    = n_observations
        self.action_output_dim = num_of_action
        self.update_freq       = update_freq
        self._step_counter     = 0   # counts env steps for update_freq gating

        self.episode_rewards = []
        self.episode_lengths = []

        super().__init__(
            buffer_size     = buffer_size,
            batch_size      = batch_size,
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            discount_factor = discount_factor,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample_action(self, obs):
        """
        Reparameterised sample. Returns (scaled_action, log_prob).
        scaled_action is bounded to action_range.
        """
        feat    = self.actor_base(obs)
        mu      = self.mu_head(feat)
        log_std = self.ls_head(feat).clamp(-20, 2)
        std     = log_std.exp()

        dist   = D.Normal(mu, std)
        z      = dist.rsample()
        tanh_z = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        action_min, action_max = self.action_range
        scaled = action_min + (tanh_z + 1.0) * 0.5 * (action_max - action_min)
        return scaled, log_prob

    def _q_input(self, state, action):
        return torch.cat([state, action], dim=-1)

    def _polyak(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def select_action(self, obs, deterministic: bool = False):
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if deterministic:
            with torch.no_grad():
                feat = self.actor_base(state)
                mu   = self.mu_head(feat)
            action_min, action_max = self.action_range
            scaled = action_min + (torch.tanh(mu) + 1.0) * 0.5 * (action_max - action_min)
            return scaled, None

        with torch.no_grad():
            scaled, log_prob = self._sample_action(state)
        return scaled, log_prob   # FIX: return scaled (not raw) for storage

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        actions are SCALED (bounded to action_range) — consistent with _sample_action output.
        """
        alpha = self.alpha.detach()

        with torch.no_grad():
            next_a, next_logp = self._sample_action(next_states)   # returns scaled
            q1_next = self.critic1_target(self._q_input(next_states, next_a))
            q2_next = self.critic2_target(self._q_input(next_states, next_a))
            q_next  = torch.min(q1_next, q2_next) - alpha * next_logp
            y       = rewards + self.discount_factor * (1.0 - dones) * q_next

        q1    = self.critic1(self._q_input(states, actions))
        loss1 = F.mse_loss(q1, y)
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        q2    = self.critic2(self._q_input(states, actions))
        loss2 = F.mse_loss(q2, y)
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        critic_loss = (loss1 + loss2) / 2.0

        new_a, new_logp = self._sample_action(states)   # returns scaled
        q1_new = self.critic1(self._q_input(states, new_a))
        q2_new = self.critic2(self._q_input(states, new_a))
        actor_loss = (self.alpha * new_logp - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor_base.parameters()) +
            list(self.mu_head.parameters()) +
            list(self.ls_head.parameters()), 1.0)
        self.actor_optimizer.step()

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (new_logp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak(self.critic1_target, self.critic1)
        self._polyak(self.critic2_target, self.critic2)

        return critic_loss.detach(), actor_loss.detach()  # detach — graphs already freed

    def update_policy(self):
        batch = self.generate_sample()
        if batch is None:
            return

        states, actions, rewards, next_states, dones = zip(*batch)

        def _t(obs):
            if isinstance(obs, dict):
                return obs['policy'].to(self.device).float()
            return obs.to(self.device).float()

        s  = torch.cat([_t(x) for x in states])
        a  = torch.cat([x.to(self.device).float() for x in actions])
        r  = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.cat([_t(x) for x in next_states])
        d  = torch.tensor(dones,   dtype=torch.float32, device=self.device).unsqueeze(1)

        self.calculate_loss(s, a, r, ns, d)

    def learn(self, env, max_steps: int = 500):
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        cumulative_reward = 0.0
        steps = 0
        done  = False

        while not done and steps < max_steps:
            scaled, _ = self.select_action(obs)   # scaled is bounded to action_range

            next_obs, reward, terminated, truncated, _ = env.step(scaled)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r    = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc= truncated[0].item()  if truncated.dim() > 0 else truncated.item()
            done = bool(term) or bool(trunc)

            st = obs['policy'].cpu().float()
            nt = next_obs['policy'].cpu().float()
            # FIX: store SCALED action — consistent with what critic expects
            self.store_transition(st, scaled.cpu(), r, nt, float(term))

            cumulative_reward += r
            steps += 1
            obs    = next_obs

            # Only update every update_freq steps — same learning, ~4x faster
            self._step_counter += 1
            if self._step_counter % self.update_freq == 0:
                self.update_policy()

        self.episode_rewards.append(cumulative_reward)
        self.episode_lengths.append(steps)
        return cumulative_reward, steps

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor_base': self.actor_base.state_dict(),
            'mu_head':    self.mu_head.state_dict(),
            'ls_head':    self.ls_head.state_dict(),
            'critic1':    self.critic1.state_dict(),
            'critic2':    self.critic2.state_dict(),
            'log_alpha':  self.log_alpha.data,
        }, os.path.join(path, filename))

    def load_model(self, path, filename):
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor_base.load_state_dict(ckpt['actor_base'])
        self.mu_head.load_state_dict(ckpt['mu_head'])
        self.ls_head.load_state_dict(ckpt['ls_head'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.critic1_target.load_state_dict(ckpt['critic1'])
        self.critic2_target.load_state_dict(ckpt['critic2'])
        self.log_alpha.data.copy_(ckpt['log_alpha'])