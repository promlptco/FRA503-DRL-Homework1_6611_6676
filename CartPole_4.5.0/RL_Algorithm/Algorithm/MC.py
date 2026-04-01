from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class MC(BaseAlgorithm):
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
        Initialize the Monte Carlo algorithm.

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
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

    def update(self):
        """
        Update Q-values using Every-Visit Monte Carlo with incremental mean.

        Called at the END of each episode. Uses the stored episode history
        (obs_hist, action_hist, reward_hist) to compute returns and update
        Q-values.

        Uses incremental mean update (no fixed learning rate needed, but lr
        is supported as an alternative):
            N(s, a) += 1
            Q(s, a) <- Q(s, a) + (1 / N(s, a)) * (G - Q(s, a))

        The histories are cleared after each update.
        """
        # Compute discounted returns by working backwards through the episode
        T = len(self.reward_hist)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self.reward_hist[t] + self.discount_factor * G
            returns[t] = G

        # Update Q-values for every visit
        for t in range(T):
            state = self.discretize_state(self.obs_hist[t])
            action = self.action_hist[t]
            G_t = returns[t]

            # Increment visit count
            self.n_values[state][action] += 1
            N = self.n_values[state][action]

            # Incremental mean update
            error = G_t - self.q_values[state][action]
            self.q_values[state][action] += (1.0 / N) * error
            self.training_error.append(error)

        if len(self.training_error) > 1000:
            self.training_error = self.training_error[-100:]

        # Clear episode history for next episode
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()