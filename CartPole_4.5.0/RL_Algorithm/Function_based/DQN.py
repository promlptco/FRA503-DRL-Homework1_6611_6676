"""
RL_Algorithm/Function_based/DQN.py

Deep Q-Network (DQN).

Classification:
  - Type:        Value-based
  - Policy:      Deterministic (ε-greedy at train time)
  - On/Off:      Off-policy (experience replay)
  - Action space: Discrete
  - Exploration: ε-greedy, linearly decayed
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os

from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    """
    Q-network: state → Q-values for every discrete action.

    Architecture: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
    """

    def __init__(self, n_observations: int, hidden_size: int, n_actions: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network agent.

    Uses a policy network for action selection and a target network
    (soft-updated via Polyak averaging) for stable Bellman targets.
    Transitions are stored in a ReplayBuffer (inherited from OffPolicyAlgorithm).
    """

    def __init__(
        self,
        device          = None,
        num_of_action:  int   = 11,
        action_range:   list  = [-10.0, 10.0],
        n_observations: int   = 4,
        hidden_dim:     int   = 128,
        dropout:        float = 0.1,
        learning_rate:  float = 1e-3,
        tau:            float = 0.005,
        initial_epsilon:float = 1.0,
        epsilon_decay:  float = 1e-3,
        final_epsilon:  float = 0.01,
        discount_factor:float = 0.99,
        buffer_size:    int   = 10000,
        batch_size:     int   = 64,
        update_freq:         int = 4,
        target_update_freq:  int = 100,
    ) -> None:

        self.device = device if device is not None else torch.device('cpu')

        # Networks
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.tau                = tau
        self.update_freq        = update_freq
        self.target_update_freq = target_update_freq
        self._step_counter      = 0
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        self.episode_rewards   = []
        self.episode_lengths   = []

        super().__init__(
            buffer_size     = buffer_size,
            batch_size      = batch_size,
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            initial_epsilon = initial_epsilon,
            epsilon_decay   = epsilon_decay,
            final_epsilon   = final_epsilon,
            discount_factor = discount_factor,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs):
        """
        ε-greedy action selection.

        Args:
            obs: State observation (dict or tensor).

        Returns:
            Tuple[torch.Tensor, int]: (scaled action tensor, discrete index)
        """
        if isinstance(obs, dict):
            state = obs['policy'].to(self.device).float()
        else:
            state = obs.to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_of_action - 1)
        else:
            with torch.no_grad():
                action_idx = int(self.policy_net(state).argmax(dim=1).item())

        return self.scale_action(action_idx), action_idx

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _prepare_batch(self):
        """
        Sample from replay buffer and build training tensors.

        Returns:
            Tuple or None if buffer not ready.
        """
        batch = self.generate_sample()
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)

        def _to_tensor(obs):
            if isinstance(obs, dict):
                return obs['policy'].to(self.device).float()
            return obs.to(self.device).float()

        state_batch  = torch.cat([_to_tensor(s) for s in states])
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.long,    device=self.device).unsqueeze(1)

        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool, device=self.device)
        non_final_list = [_to_tensor(ns) for ns, d in zip(next_states, dones) if not d]

        if non_final_list:
            non_final_next = torch.cat(non_final_list)
        else:
            non_final_next = torch.zeros((0, state_batch.shape[1]), device=self.device)

        return non_final_mask, non_final_next, state_batch, action_batch, reward_batch

    # ------------------------------------------------------------------
    # Loss and update
    # ------------------------------------------------------------------

    def calculate_loss(self, non_final_mask, non_final_next, state_batch, action_batch, reward_batch):
        """
        Bellman (Huber) loss.

        Target: r + γ * max Q_target(s', ·)    (zero for terminal states)

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Q(s, a) from policy net
        q_sa = self.policy_net(state_batch).gather(1, action_batch)  # (B, 1)

        # V(s') from target net — 0 for terminal states
        next_v = torch.zeros(reward_batch.shape[0], device=self.device)
        if non_final_next.shape[0] > 0:
            with torch.no_grad():
                next_v[non_final_mask] = self.target_net(non_final_next).max(1).values

        expected = reward_batch + self.discount_factor * next_v
        return F.smooth_l1_loss(q_sa.squeeze(), expected)

    def update_policy(self):
        """Sample a batch and perform one gradient step on the policy network."""
        sample = self._prepare_batch()
        if sample is None:
            return

        non_final_mask, non_final_next, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next, state_batch, action_batch, reward_batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_networks(self, tau=None):
        """
        Polyak soft-update:  θ_target = τ*θ_policy + (1-τ)*θ_target
        """
        if tau is None:
            tau = self.tau
        policy_sd = self.policy_net.state_dict()
        target_sd = self.target_net.state_dict()
        for key in policy_sd:
            target_sd[key] = tau * policy_sd[key] + (1.0 - tau) * target_sd[key]
        self.target_net.load_state_dict(target_sd)

    # ------------------------------------------------------------------
    # Training loop (one episode)
    # ------------------------------------------------------------------

    def learn(self, env, max_steps: int = 500):
        """
        Train for one episode.

        Returns:
            Tuple[float, int]: (cumulative reward, episode length)
        """
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        cumulative_reward = 0.0
        steps = 0
        done  = False
        self.policy_net.train()

        while not done and steps < max_steps:
            action_tensor, action_idx = self.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r     = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term  = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc = truncated[0].item()  if truncated.dim() > 0 else truncated.item()
            done  = bool(term) or bool(trunc)

            state_t = obs['policy'].cpu().float()
            next_t  = next_obs['policy'].cpu().float()
            self.store_transition(state_t, action_idx, r, next_t, float(term))

            cumulative_reward += r
            steps += 1
            obs    = next_obs

            self._step_counter += 1
            if self._step_counter % self.update_freq == 0:
                self.update_policy()
            if self._step_counter % self.target_update_freq == 0:
                self.update_target_networks(tau=1.0)  # hard copy every N steps

        self.decay_epsilon()
        self.episode_durations.append(steps)
        self.episode_rewards.append(cumulative_reward)
        self.episode_lengths.append(steps)
        return cumulative_reward, steps

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load_model(self, path: str, filename: str):
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())