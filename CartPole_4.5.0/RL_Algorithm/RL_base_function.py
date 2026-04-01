import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

# Device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class BaseAlgorithm:
    """
    Minimal shared base for all function approximation RL algorithms.

    Provides:
      - Common hyperparameter storage (lr, epsilon, discount_factor, etc.)
      - scale_action()   — discrete index → continuous action tensor
      - decay_epsilon()  — linear epsilon decay
      - plot_durations() — shared matplotlib episode-length visualisation

    NOT included here (moved to appropriate subclasses / storage):
      - ReplayBuffer  → RL_Algorithm/storage/buffers.py
      - self.w        → Linear_QN only
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
    ):
        self.lr               = learning_rate
        self.discount_factor  = discount_factor
        self.epsilon          = initial_epsilon
        self.epsilon_decay    = epsilon_decay
        self.final_epsilon    = final_epsilon
        self.num_of_action    = num_of_action
        self.action_range     = action_range   # [action_min, action_max]
        self.training_error   = []

        # matplotlib setup (shared across algorithms)
        self.episode_durations = []
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()

    # ------------------------------------------------------------------
    # Action scaling
    # ------------------------------------------------------------------

    def scale_action(self, action):
        """
        Map discrete action index in [0, num_of_action-1] to a continuous
        value in [action_min, action_max] and wrap in a (1,1) tensor.

        Args:
            action (int): Discrete action index.

        Returns:
            torch.Tensor: Scaled continuous action, shape (1, 1).
        """
        action_min, action_max = self.action_range
        continuous = action_min + (action_max - action_min) * action / (self.num_of_action - 1)
        return torch.tensor([[continuous]], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Linearly decay epsilon, clamped at final_epsilon."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # ------------------------------------------------------------------
    # Shared visualisation
    # ------------------------------------------------------------------

    def plot_durations(self, timestep=None, show_result=False):
        """
        Plot episode lengths with a 100-episode rolling average.
        Silently skips the interactive pause when running headless (no display).
        """
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        # Suppress FigureCanvasAgg warning on headless servers — skip interactive pause
        if self.is_ipython:
            if not show_result:
                from IPython import display
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                from IPython import display
                display.display(plt.gcf())