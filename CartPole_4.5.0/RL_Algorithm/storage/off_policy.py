"""
storage/off_policy.py

OffPolicyAlgorithm(BaseAlgorithm) — shared base for DQN, TD3, SAC.

Creates the ReplayBuffer and provides thin wrapper methods so subclasses
never access self.memory directly.
"""

from RL_Algorithm.RL_base_function import BaseAlgorithm
from RL_Algorithm.storage.buffers import ReplayBuffer


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy algorithms (DQN, TD3, SAC).

    Provides:
      - store_transition()       — add one experience to replay buffer
      - generate_sample()        — draw a batch; returns None if not ready
      - update_target_networks() — no-op placeholder for Polyak update

    Subclasses MUST implement:
      - select_action(obs)
      - calculate_loss(...)
      - update_policy()
      - learn(env, ...)
      - update_target_networks()   (override the no-op)
    """

    def __init__(self, buffer_size: int = 10000, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.memory = ReplayBuffer(buffer_size, batch_size)

    # ------------------------------------------------------------------
    # Buffer wrappers
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        """
        Add one experience tuple to the replay buffer.

        Args:
            state:      Current state tensor.
            action:     Action taken.
            reward:     Scalar reward.
            next_state: Next state tensor.
            done:       Terminal flag.
        """
        self.memory.add(state, action, reward, next_state, done)

    def generate_sample(self):
        """
        Draw a random batch from the replay buffer.

        Returns:
            list[Transition] or None: Sampled batch, or None if not ready.
        """
        return self.memory.sample()

    # ------------------------------------------------------------------
    # Target network update placeholder
    # ------------------------------------------------------------------

    def update_target_networks(self, tau=None):
        """
        Polyak soft-update placeholder.

        Each subclass overrides this with its own network-specific logic:
            θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        pass