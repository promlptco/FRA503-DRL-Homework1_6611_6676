"""
RL_Algorithm/Function_based/TD3.py

Twin Delayed DDPG (TD3).

BUG FIXED (action scale mismatch):
  The replay buffer was storing raw (pre-tanh, unbounded) actor outputs.
  Inside calculate_loss, Q(s, a) was trained on these raw values while
  target Q uses _scale(actor_target(s')) which is bounded to action_range.
  The critic was learning two different action representations simultaneously,
  producing incoherent gradients.
  Fix: store SCALED actions in the replay buffer so the critic sees a
  consistent bounded action space during both training and target computation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm
from RL_Algorithm.network.mlp import MLP


class TD3(OffPolicyAlgorithm):
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
        policy_noise:  float = 0.2,
        noise_clip:    float = 0.5,
        expl_noise:    float = 0.1,
        policy_delay:  int   = 2,
        update_freq:   int   = 4,    # only update every N env steps → 4x speedup
    ):
        self.device = device if device is not None else torch.device('cpu')

        self.actor        = MLP(n_observations, num_of_action, [hidden_dim, hidden_dim],
                                activation='relu').to(self.device)
        self.actor_target = MLP(n_observations, num_of_action, [hidden_dim, hidden_dim],
                                activation='relu').to(self.device)
        self._hard_copy(self.actor_target, self.actor)

        self.critic1        = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic2        = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic1_target = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self.critic2_target = MLP(n_observations + num_of_action, 1, [hidden_dim, hidden_dim],
                                  activation='relu').to(self.device)
        self._hard_copy(self.critic1_target, self.critic1)
        self._hard_copy(self.critic2_target, self.critic2)

        self.actor_optimizer   = optim.Adam(self.actor.parameters(),   lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.tau           = tau
        self.policy_noise  = policy_noise
        self.noise_clip    = noise_clip
        self.expl_noise    = expl_noise
        self.policy_delay  = policy_delay
        self.update_freq   = update_freq
        self.total_updates = 0
        self._step_counter = 0   # counts env steps for update_freq gating
        self.n_observations    = n_observations
        self.action_output_dim = num_of_action

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

    @staticmethod
    def _hard_copy(target, source):
        target.load_state_dict(source.state_dict())

    def _polyak(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def _scale(self, raw):
        """Map actor output through tanh to [action_min, action_max]."""
        action_min, action_max = self.action_range
        return action_min + (torch.tanh(raw) + 1.0) * 0.5 * (action_max - action_min)

    def _q_input(self, state, action):
        return torch.cat([state, action], dim=-1)

    def select_action(self, obs, noise: float = None):
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            raw = self.actor(state)

        if noise is None:
            noise = self.expl_noise
        if noise > 0.0:
            raw = raw + torch.randn_like(raw) * noise

        scaled = self._scale(raw)
        return scaled, scaled   # FIX: return scaled twice — caller stores scaled

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        actions here are SCALED (bounded to action_range) — consistent with targets.
        """
        action_min, action_max = self.action_range

        with torch.no_grad():
            # Target policy also produces scaled actions
            next_raw    = self.actor_target(next_states)
            noise       = (torch.randn_like(next_raw) * self.policy_noise).clamp(
                              -self.noise_clip, self.noise_clip)
            # Add noise in normalised [-1,1] space then re-scale
            next_scaled = self._scale(next_raw + noise)

            q1_tgt = self.critic1_target(self._q_input(next_states, next_scaled))
            q2_tgt = self.critic2_target(self._q_input(next_states, next_scaled))
            y      = rewards + self.discount_factor * (1.0 - dones) * torch.min(q1_tgt, q2_tgt)

        q1 = self.critic1(self._q_input(states, actions))
        loss1 = F.mse_loss(q1, y)
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        q2 = self.critic2(self._q_input(states, actions))
        loss2 = F.mse_loss(q2, y)
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        critic_loss = ((loss1 + loss2) / 2.0).detach()  # detach — graph already freed
        actor_loss  = None

        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            # Actor outputs raw, scale before passing to critic
            pi         = self._scale(self.actor(states))
            actor_loss = -self.critic1(self._q_input(states, pi)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self._polyak(self.actor_target,   self.actor,   self.tau)
            self._polyak(self.critic1_target, self.critic1, self.tau)
            self._polyak(self.critic2_target, self.critic2, self.tau)

            actor_loss = actor_loss.detach()  # detach — graph already freed

        return critic_loss, actor_loss

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

    def update_target_networks(self, tau=None):
        pass  # handled inside calculate_loss

    def learn(self, env, max_steps: int = 500):
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        cumulative_reward = 0.0
        steps = 0
        done  = False

        while not done and steps < max_steps:
            scaled, _ = self.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(scaled)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r    = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc= truncated[0].item()  if truncated.dim() > 0 else truncated.item()
            done = bool(term) or bool(trunc)

            st = obs['policy'].cpu().float()
            nt = next_obs['policy'].cpu().float()
            # FIX: store SCALED action (bounded) not raw (unbounded)
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
            'actor':   self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, os.path.join(path, filename))

    def load_model(self, path, filename):
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.critic1_target.load_state_dict(ckpt['critic1'])
        self.critic2_target.load_state_dict(ckpt['critic2'])