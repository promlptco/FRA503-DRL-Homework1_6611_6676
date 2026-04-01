import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch


class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4
    SARSA = 5


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range
        self.discretize_state_weight = discretize_state_weight

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []
        self._error_flush_interval = 1000  # clear every 1000 steps to prevent memory bloat

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
            obs (dict): Observation dictionary containing policy states.
                        obs['policy'] is a tensor of shape (1, 4):
                        [pose_cart, pose_pole, vel_cart, vel_pole]

        Returns:
            tuple: Discretized state as (pose_cart, pose_pole, vel_cart, vel_pole).
        """
        # Extract the policy observation tensor and convert to numpy
        # Only move to CPU if needed to avoid unnecessary GPU->CPU transfers
        policy = obs['policy']
        if policy.device.type != 'cpu':
            policy = policy.cpu()
        state = policy.numpy().flatten()

        # Multiply each state dimension by its discretization weight and round to int
        discretized = tuple(
            int(round(float(state[i]) * self.discretize_state_weight[i]))
            for i in range(len(self.discretize_state_weight))
        )
        return discretized

    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        # Explore: random action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_of_action)
        
        # Exploit: best known action
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            # Use combined Q values for action selection during exploitation
            combined = self.qa_values[obs_dis] + self.qb_values[obs_dis]
            return int(np.argmax(combined))
        else:
            return int(np.argmax(self.q_values[obs_dis]))

    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, num_of_action-1] to a continuous
        value in [action_min, action_max].

        Args:
            action (int): Discrete action index in range [0, num_of_action-1]

        Returns:
            torch.Tensor: Scaled action tensor of shape (1, 1).
        """
        action_min, action_max = self.action_range
        # Map discrete index to continuous value
        if self.num_of_action == 1:
            continuous = 0.0
        else:
            continuous = action_min + (action_max - action_min) * action / (self.num_of_action - 1)
        return torch.tensor([[continuous]], dtype=torch.float32)

    def get_action(self, obs) -> torch.tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)
        return action_tensor, action_idx

    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        os.makedirs(path, exist_ok=True)

        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}

        if self.control_type == ControlType.MONTE_CARLO:
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys
            }
        else:
            model_params = {
                'q_values': q_values_str_keys,
            }
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)
        with open(full_path, 'r') as file:
            data = json.load(file)
            data_q_values = data['q_values']
            for state, action_values in data_q_values.items():
                state = state.replace('(', '')
                state = state.replace(')', '')
                tuple_state = tuple(map(float, state.split(', ')))
                self.q_values[tuple_state] = action_values.copy()
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    self.qa_values[tuple_state] = action_values.copy()
                    self.qb_values[tuple_state] = action_values.copy()
            if self.control_type == ControlType.MONTE_CARLO:
                data_n_values = data['n_values']
                for state, n_values in data_n_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.n_values[tuple_state] = n_values.copy()
            return self.q_values