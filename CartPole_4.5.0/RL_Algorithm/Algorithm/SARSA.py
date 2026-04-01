from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class SARSA(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the SARSA algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.SARSA,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool,
    ):
        """
        Update Q-values using SARSA (on-policy TD control).

        Q(s, a) <- Q(s, a) + lr * [r + gamma * Q(s', a') - Q(s, a)]

        Args:
            obs: Current observation (dict).
            action (int): Action taken.
            reward (float): Reward received.
            next_obs: Next observation (dict).
            next_action (int): Next action chosen by current policy (on-policy).
            terminated (bool): Whether the episode ended.
        """
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # On-policy: use the actual next action value
        future_value = 0.0 if terminated else self.q_values[next_state][next_action]

        # TD target and error
        td_target = reward + self.discount_factor * future_value
        td_error = td_target - self.q_values[state][action]

        # Update
        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(td_error)
        if len(self.training_error) > 1000:
            self.training_error = self.training_error[-100:]