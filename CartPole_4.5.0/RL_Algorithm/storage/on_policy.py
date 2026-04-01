"""
storage/on_policy.py

OnPolicyAlgorithm(BaseAlgorithm) — shared base for AC, A2C, PPO.

Manages RolloutBuffer allocation and the transition accumulation loop so
that each subclass only needs to implement the update logic.
"""

from RL_Algorithm.RL_base_function import BaseAlgorithm
from RL_Algorithm.storage.buffers import RolloutBuffer


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms (AC, A2C, PPO).

    Provides:
      - _init_storage()     — allocate a new RolloutBuffer
      - set_storage()       — attach an externally created buffer
      - add_transition()    — flush self.transition into self.storage

    Subclasses MUST implement:
      - act(obs)
      - process_env_step(rewards, dones)
      - compute_returns(last_obs)
      - update()
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage    = None
        self.transition = RolloutBuffer.Transition()

    # ------------------------------------------------------------------
    # Storage management
    # ------------------------------------------------------------------

    def _init_storage(self, num_envs, num_transitions_per_env, obs_shape, actions_shape):
        """
        Allocate a fresh RolloutBuffer.

        Args:
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon T.
            obs_shape (tuple): Shape of a single observation, e.g. (4,).
            actions_shape (tuple): Shape of a single action, e.g. (1,).
        """
        device = getattr(self, 'device', 'cpu')
        self.storage = RolloutBuffer(
            num_transitions_per_env,
            num_envs,
            obs_shape,
            actions_shape,
            device=str(device),
        )

    def set_storage(self, storage: RolloutBuffer):
        """
        Attach an externally created RolloutBuffer.

        Useful when multiple objects share one buffer.

        Args:
            storage (RolloutBuffer): Pre-allocated buffer.
        """
        self.storage = storage

    def add_transition(self):
        """Flush self.transition into self.storage."""
        if self.storage is None:
            raise RuntimeError("Call _init_storage() or set_storage() before add_transition().")
        self.storage.add_transition(self.transition)

    # ------------------------------------------------------------------
    # Abstract interface — must be overridden by subclasses
    # ------------------------------------------------------------------

    def act(self, obs):
        """
        Sample action for obs, populate self.transition fields, return action.

        Args:
            obs: Environment observation.

        Returns:
            Action tensor.
        """
        raise NotImplementedError

    def process_env_step(self, rewards, dones):
        """
        Record reward/done into self.transition then call add_transition().

        Args:
            rewards: Reward tensor from env.
            dones:   Done tensor from env.
        """
        raise NotImplementedError

    def compute_returns(self, last_obs):
        """
        Compute advantages and returns over the completed rollout.

        Args:
            last_obs: Final observation used for bootstrap value.
        """
        raise NotImplementedError

    def update(self):
        """Perform the gradient update using self.storage."""
        raise NotImplementedError