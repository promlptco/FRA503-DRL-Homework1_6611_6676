"""
storage/buffers.py

Two buffer classes used by different algorithm families:
  - RolloutBuffer : on-policy (AC, A2C, PPO)  — fixed-length parallel rollout
  - ReplayBuffer  : off-policy (DQN, TD3, SAC) — FIFO experience replay
"""

import torch
import numpy as np
from collections import namedtuple, deque
import random


# ---------------------------------------------------------------------------
# RolloutBuffer  (on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Pre-allocated tensor buffer for on-policy parallel rollouts.

    Stores T transitions across N parallel environments:
        shape = (num_transitions_per_env, num_envs, ...)

    Used by: AC (episodic), A2C, PPO
    """

    class Transition:
        """Container for a single environment step's data."""
        def __init__(self):
            self.observations    = None   # shape (num_envs, obs_dim)
            self.actions         = None   # shape (num_envs, act_dim)
            self.rewards         = None   # shape (num_envs,)
            self.dones           = None   # shape (num_envs,)
            self.values          = None   # shape (num_envs,)
            self.actions_log_prob= None   # shape (num_envs,)
            self.mu              = None   # shape (num_envs, act_dim)
            self.sigma           = None   # shape (num_envs, act_dim)

    def __init__(self, num_transitions_per_env, num_envs, obs_shape, actions_shape, device='cpu'):
        """
        Args:
            num_transitions_per_env (int): T — rollout horizon.
            num_envs (int): N — parallel envs.
            obs_shape (tuple): Observation shape per env, e.g. (4,).
            actions_shape (tuple): Action shape per env, e.g. (1,).
            device (str): torch device string.
        """
        self.T       = num_transitions_per_env
        self.N       = num_envs
        self.device  = device
        self.step    = 0

        # Core tensors — shape (T, N, ...)
        self.observations     = torch.zeros(self.T, self.N, *obs_shape,     device=device)
        self.actions          = torch.zeros(self.T, self.N, *actions_shape, device=device)
        self.rewards          = torch.zeros(self.T, self.N,                 device=device)
        self.dones            = torch.zeros(self.T, self.N,                 device=device)

        # Policy quantities
        self.values           = torch.zeros(self.T, self.N,                 device=device)
        self.actions_log_prob = torch.zeros(self.T, self.N,                 device=device)
        self.mu               = torch.zeros(self.T, self.N, *actions_shape, device=device)
        self.sigma            = torch.zeros(self.T, self.N, *actions_shape, device=device)

        # Computed after rollout completion
        self.returns          = torch.zeros(self.T, self.N,                 device=device)
        self.advantages       = torch.zeros(self.T, self.N,                 device=device)

    def add_transition(self, transition: 'RolloutBuffer.Transition'):
        """
        Copy one Transition into the buffer at self.step.

        Args:
            transition (Transition): Data from one step across all envs.

        Raises:
            OverflowError: If buffer is already full.
        """
        if self.step >= self.T:
            raise OverflowError("RolloutBuffer is full. Call clear() before adding more transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones)

        if transition.values is not None:
            self.values[self.step].copy_(transition.values)
        if transition.actions_log_prob is not None:
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob)
        if transition.mu is not None:
            self.mu[self.step].copy_(transition.mu)
        if transition.sigma is not None:
            self.sigma[self.step].copy_(transition.sigma)

        self.step += 1

    def clear(self):
        """Reset step counter so the buffer can be reused."""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        """
        Yield randomly shuffled mini-batches over the completed rollout.

        Yields 8-tuples:
            (obs, actions, target_values, advantages, returns,
             old_log_prob, old_mu, old_sigma)

        Args:
            num_mini_batches (int): Number of mini-batches per epoch.
            num_epochs (int): How many times to iterate over the buffer.
        """
        batch_size = self.T * self.N
        mini_batch_size = batch_size // num_mini_batches

        # Flatten (T, N, ...) → (T*N, ...)
        obs         = self.observations.view(batch_size, -1)
        actions     = self.actions.view(batch_size, -1)
        target_vals = self.returns.view(batch_size)
        advantages  = self.advantages.view(batch_size)
        returns     = self.returns.view(batch_size)
        old_log_prob= self.actions_log_prob.view(batch_size)
        old_mu      = self.mu.view(batch_size, -1)
        old_sigma   = self.sigma.view(batch_size, -1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start: start + mini_batch_size]
                yield (
                    obs[idx],
                    actions[idx],
                    target_vals[idx],
                    advantages[idx],
                    returns[idx],
                    old_log_prob[idx],
                    old_mu[idx],
                    old_sigma[idx],
                )

    def is_full(self):
        return self.step >= self.T

    def __len__(self):
        return self.step


# ---------------------------------------------------------------------------
# ReplayBuffer  (off-policy)
# ---------------------------------------------------------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    FIFO experience replay buffer backed by a deque.

    Oldest entry is automatically discarded when the buffer is full.

    Used by: DQN, TD3, SAC
    """

    def __init__(self, buffer_size: int, batch_size: int = 64):
        """
        Args:
            buffer_size (int): Maximum number of transitions stored.
            batch_size (int): Number of samples drawn per training step.
        """
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Append one transition. Oldest entry discarded automatically when full.

        Args:
            state:      Current state tensor.
            action:     Action taken.
            reward:     Scalar reward.
            next_state: Next state tensor.
            done:       Terminal flag (bool or float).
        """
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self):
        """
        Draw a random batch of Transition namedtuples.

        Returns:
            list[Transition] or None: Batch, or None if buffer not ready.
        """
        if len(self.memory) < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)

    def is_ready(self):
        """Return True when the buffer holds at least batch_size entries."""
        return len(self.memory) >= self.batch_size

    def __len__(self):
        return len(self.memory)