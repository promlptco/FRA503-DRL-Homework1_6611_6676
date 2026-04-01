"""
RL_Algorithm/Function_based/MC_REINFORCE.py

REINFORCE (Monte Carlo Policy Gradient).

Classification:
  - Type:        Policy-based
  - Policy:      Stochastic (categorical distribution over discrete actions)
  - On/Off:      On-policy (full episode required before update)
  - Action space: Discrete
  - Exploration: Stochastic policy (entropy-driven natural exploration)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import os

from RL_Algorithm.RL_base_function import BaseAlgorithm


class MC_REINFORCE_network(nn.Module):
    """
    Stochastic policy network: state → action probability distribution.

    Architecture: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Softmax
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
        """Returns softmax action probabilities."""
        return F.softmax(self.net(x), dim=-1)


class MC_REINFORCE(BaseAlgorithm):
    """
    REINFORCE agent — trained end-of-episode using Monte Carlo returns.

    No replay buffer needed: policy is updated directly from the collected
    trajectory using discounted returns G_t weighted by log π(a_t | s_t).
    """

    def __init__(
        self,
        device           = None,
        num_of_action:   int   = 11,
        action_range:    list  = [-10.0, 10.0],
        n_observations:  int   = 4,
        hidden_dim:      int   = 128,
        dropout:         float = 0.1,
        learning_rate:   float = 1e-3,
        discount_factor: float = 0.99,
    ) -> None:

        self.device = device if device is not None else torch.device('cpu')

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.optimizer  = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.episode_rewards = []
        self.episode_lengths = []

        super().__init__(
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            discount_factor = discount_factor,
        )

    # ------------------------------------------------------------------
    # Return computation
    # ------------------------------------------------------------------

    def calculate_stepwise_returns(self, rewards: list) -> torch.Tensor:
        """
        Compute and normalise discounted returns G_t for each timestep.

        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

        Args:
            rewards (list[float]): Per-step rewards from one episode.

        Returns:
            torch.Tensor: Normalised returns, shape (T,).
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    # ------------------------------------------------------------------
    # Trajectory collection
    # ------------------------------------------------------------------

    def generate_trajectory(self, env, max_steps: int = 500):
        """
        Run one full episode and collect (obs, action, reward) trajectory.

        Args:
            env:       Isaac Lab environment.
            max_steps: Maximum episode length.

        Returns:
            Tuple:
              - episode_return  (float):  Total undiscounted reward.
              - stepwise_returns(Tensor): Normalised discounted returns, (T,).
              - log_prob_actions(Tensor): Log-probs of sampled actions, (T,).
              - trajectory      (list):   Raw (obs, action_idx, r) triples.
        """
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        trajectory, log_probs, rewards = [], [], []
        episode_return = 0.0
        done = False
        t = 0

        # IMPORTANT: must be in train() mode (NOT no_grad) so that
        # log_probs retain their grad_fn for loss.backward() later.
        self.policy_net.train()

        while not done and t < max_steps:
            state = obs['policy'].to(self.device).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # No torch.no_grad() here — we need gradients through log_prob
            probs = self.policy_net(state)                 # (1, n_actions)

            dist        = distributions.Categorical(probs)
            action_t    = dist.sample()                    # scalar tensor
            action_idx  = action_t.item()
            log_prob    = dist.log_prob(action_t)          # scalar tensor — has grad_fn

            action = self.scale_action(action_idx)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r     = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term  = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc = truncated[0].item()  if truncated.dim() > 0 else truncated.item()
            done  = bool(term) or bool(trunc)

            log_probs.append(log_prob)
            rewards.append(r)
            trajectory.append((obs, action_idx, r))
            episode_return += r
            obs = next_obs
            t  += 1

        log_prob_actions = torch.stack(log_probs)           # (T,)
        stepwise_returns = self.calculate_stepwise_returns(rewards)

        self.plot_durations(t)
        return episode_return, stepwise_returns, log_prob_actions, trajectory

    # ------------------------------------------------------------------
    # Loss and update
    # ------------------------------------------------------------------

    def calculate_loss(self, stepwise_returns: torch.Tensor, log_prob_actions: torch.Tensor) -> torch.Tensor:
        """
        REINFORCE policy gradient loss.

        Loss = -E[ G_t * log π(a_t | s_t) ]

        Negative because PyTorch minimises; we want to maximise expected return.

        Args:
            stepwise_returns  (Tensor): Discounted normalised returns, (T,).
            log_prob_actions  (Tensor): Log-probabilities of chosen actions, (T,).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return -(log_prob_actions * stepwise_returns).mean()

    def update_policy(self, stepwise_returns: torch.Tensor, log_prob_actions: torch.Tensor) -> float:
        """
        Compute loss and perform one gradient ascent step.

        Returns:
            float: Loss value.
        """
        self.policy_net.train()
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Training loop (one episode)
    # ------------------------------------------------------------------

    def learn(self, env, max_steps: int = 500):
        """
        Collect one trajectory then update the policy.

        Returns:
            Tuple[float, float, int]: (episode_return, loss, episode_length)
        """
        episode_return, stepwise_returns, log_prob_actions, trajectory = \
            self.generate_trajectory(env, max_steps)

        loss = self.update_policy(stepwise_returns, log_prob_actions)

        self.episode_rewards.append(episode_return)
        self.episode_lengths.append(len(trajectory))
        return episode_return, loss, len(trajectory)

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