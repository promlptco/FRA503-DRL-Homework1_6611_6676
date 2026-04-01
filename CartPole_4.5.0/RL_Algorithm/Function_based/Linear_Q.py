"""
RL_Algorithm/Function_based/Linear_Q.py

Linear Q-Learning with function approximation.

Classification:
  - Type:        Value-based
  - Policy:      Deterministic (ε-greedy)
  - On/Off:      Off-policy (Q-learning)
  - Action space: Discrete
  - Exploration: ε-greedy, linearly decayed
"""

from __future__ import annotations
import numpy as np
import json
import os
import torch

from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning agent.

    Approximates Q(s, a) as a linear function of raw state features:
        Q(s, a) = s · w[:, a]

    where w ∈ R^{state_dim × num_of_action} is updated by TD(0) gradient descent.

    self.w lives here (not in BaseAlgorithm).
    """

    def __init__(
        self,
        num_of_action:   int   = 11,
        action_range:    list  = [-10.0, 10.0],
        learning_rate:   float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay:   float = 1e-3,
        final_epsilon:   float = 0.01,
        discount_factor: float = 0.99,
        n_observations:  int   = 4,
    ) -> None:
        super().__init__(
            num_of_action   = num_of_action,
            action_range    = action_range,
            learning_rate   = learning_rate,
            initial_epsilon = initial_epsilon,
            epsilon_decay   = epsilon_decay,
            final_epsilon   = final_epsilon,
            discount_factor = discount_factor,
        )

        self.n_observations  = n_observations
        # Weight matrix: shape (state_dim, num_of_action)
        self.w = np.zeros((n_observations, num_of_action))

        self.episode_rewards = []
        self.episode_lengths = []

    # ------------------------------------------------------------------
    # Q-value helpers
    # ------------------------------------------------------------------

    def _to_state(self, obs) -> np.ndarray:
        """Extract flat numpy state from obs (dict, tensor, or array)."""
        if isinstance(obs, dict):
            return obs['policy'].cpu().numpy().flatten()
        elif isinstance(obs, torch.Tensor):
            return obs.cpu().numpy().flatten()
        return np.array(obs).flatten()

    def q(self, obs, a=None):
        """
        Return linear Q-value(s).

        Args:
            obs: State observation.
            a (int, optional): Action index. If None, return all Q-values.

        Returns:
            np.ndarray shape (num_of_action,) or float.
        """
        state = self._to_state(obs)
        if a is None:
            return state @ self.w         # (state_dim,) @ (state_dim, n) = (n,)
        return float(state @ self.w[:, a])

    # ------------------------------------------------------------------
    # Core update (TD error)
    # ------------------------------------------------------------------

    def update(self, obs, action: int, reward: float, next_obs, terminated: bool):
        """
        TD(0) weight update.

        Rule:
            target  = r + γ * max_a Q(s', a)   (or just r if terminated)
            error   = target - Q(s, a)
            w[:,a] += α * error * s

        Args:
            obs:        Current state.
            action:     Discrete action index taken.
            reward:     Reward received.
            next_obs:   Next state.
            terminated: True if episode ended.
        """
        state      = self._to_state(obs)
        next_state = self._to_state(next_obs)

        if terminated:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(next_state @ self.w)

        current_q = float(state @ self.w[:, action])
        td_error  = td_target - current_q

        self.w[:, action] += self.lr * td_error * state

        self.training_error.append(td_error)
        if len(self.training_error) > 1000:
            self.training_error = self.training_error[-1000:]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs):
        """
        ε-greedy action selection.

        Args:
            obs: Current state observation.

        Returns:
            Tuple[torch.Tensor, int]: (scaled action tensor, discrete index)
        """
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            action_idx = int(np.argmax(self.q(obs)))

        return self.scale_action(action_idx), action_idx

    # ------------------------------------------------------------------
    # Training loop (one episode)
    # ------------------------------------------------------------------

    def learn(self, env, max_steps: int = 500):
        """
        Train for one episode.

        Args:
            env:       Isaac Lab environment.
            max_steps: Maximum steps per episode.

        Returns:
            Tuple[float, int]: (cumulative reward, episode length)
        """
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        cumulative_reward = 0.0
        steps = 0
        done  = False

        while not done and steps < max_steps:
            action_tensor, action_idx = self.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r     = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term  = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc = truncated[0].item()  if truncated.dim() > 0 else truncated.item()

            self.update(obs, action_idx, r, next_obs, bool(term))

            cumulative_reward += r
            steps += 1
            done   = bool(term) or bool(trunc)
            obs    = next_obs

        self.decay_epsilon()
        self.episode_rewards.append(cumulative_reward)
        self.episode_lengths.append(steps)
        return cumulative_reward, steps

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, path: str, filename: str):
        """Save weight matrix to a JSON file."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(self.w.tolist(), f)

    def load_model(self, path: str, filename: str):
        """Load weight matrix from a JSON file."""
        with open(os.path.join(path, filename), 'r') as f:
            self.w = np.array(json.load(f))