from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
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
        terminated: bool,
    ):
        """
        Update Q-values using Double Q-Learning.

        With probability 0.5, update QA using QB to evaluate, or vice versa.
        This decouples action selection from action evaluation to reduce
        maximization bias.

        If updating QA:
            a* = argmax_a QA(s', a)
            QA(s, a) <- QA(s, a) + lr * [r + gamma * QB(s', a*) - QA(s, a)]

        If updating QB:
            a* = argmax_a QB(s', a)
            QB(s, a) <- QB(s, a) + lr * [r + gamma * QA(s', a*) - QB(s, a)]

        The combined Q(s,a) = QA(s,a) + QB(s,a) is used for action selection.

        Args:
            obs: Current observation (dict).
            action (int): Action taken.
            reward (float): Reward received.
            next_obs: Next observation (dict).
            terminated (bool): Whether the episode ended.
        """
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        if np.random.random() < 0.5:
            # Update QA: select with QA, evaluate with QB
            if terminated:
                future_value = 0.0
            else:
                best_action = int(np.argmax(self.qa_values[next_state]))
                future_value = self.qb_values[next_state][best_action]

            td_target = reward + self.discount_factor * future_value
            td_error = td_target - self.qa_values[state][action]
            self.qa_values[state][action] += self.lr * td_error
        else:
            # Update QB: select with QB, evaluate with QA
            if terminated:
                future_value = 0.0
            else:
                best_action = int(np.argmax(self.qb_values[next_state]))
                future_value = self.qa_values[next_state][best_action]

            td_target = reward + self.discount_factor * future_value
            td_error = td_target - self.qb_values[state][action]
            self.qb_values[state][action] += self.lr * td_error

        # Keep the combined q_values in sync for saving/loading
        self.q_values[state][action] = (
            self.qa_values[state][action] + self.qb_values[state][action]
        ) / 2.0
        self.training_error.append(td_error)
        if len(self.training_error) > 1000:
            self.training_error = self.training_error[-100:]