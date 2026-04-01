"""
train_all.py — Train all 4 RL algorithms with config variations in one run.

Trains:
  - Q_Learning, SARSA, Double_Q_Learning, MC

Config variations tested:
  - num_of_action:           low=5, normal=11, high=21
  - discretize_state_weight: low=[1,5,1,1], normal=[2,10,1,2], high=[4,20,2,4]
  (when varying one, the other is held at normal)

Outputs per algorithm per config:
  1. learning_curve.png        - reward per episode during training
  2. q_surface.png             - 3D surface of max Q-values (cart pos vs pole pos)
  3. episode_length_curve.png  - episode length per episode (stability indicator)

Outputs per config (all algorithms compared):
  4. comparison_reward.png     - all 4 algos reward on one plot
  5. comparison_ep_length.png  - all 4 algos episode length on one plot

Outputs after deployment evaluation (epsilon=0):
  6. deployment_reward.png         - avg deployment reward bar chart
  7. deployment_ep_length.png      - avg episode length bar chart (POLE STABILITY)
  8. deployment_success_rate.png   - % episodes survived to max steps (POLE STABILITY)
  9. resolution_action_effect.png  - effect of num_of_action on each algorithm
  10. resolution_dsw_effect.png    - effect of dsw on each algorithm

Saved to: plots/{task_name}/...
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(_SCRIPT_DIR)
sys.path.append(os.path.join(_PROJECT_ROOT, "source", "CartPole"))

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train all RL agents.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--deploy_episodes", type=int, default=100,
                    help="Number of episodes for deployment evaluation")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from collections import defaultdict

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import CartPole.tasks  # noqa: F401

from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from RL_Algorithm.Algorithm.SARSA import SARSA
from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
from RL_Algorithm.Algorithm.MC import MC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ===========================================================
# CONFIG VARIATIONS
# ===========================================================
NORMAL_NUM_OF_ACTION = 11
NORMAL_DSW = [2, 10, 1, 2]

ACTION_CONFIGS = {
    "action_low":    5,
    "action_normal": 11,
    "action_high":   21,
}

DSW_CONFIGS = {
    "dsw_low":    [1, 5, 1, 1],
    "dsw_normal": [2, 10, 1, 2],
    "dsw_high":   [4, 20, 2, 4],
}

EXPERIMENTS = []
for name, n in ACTION_CONFIGS.items():
    EXPERIMENTS.append((name, n, NORMAL_DSW))
for name, dsw in DSW_CONFIGS.items():
    if name == "dsw_normal":
        continue
    EXPERIMENTS.append((name, NORMAL_NUM_OF_ACTION, dsw))

ALGORITHM_CLASSES = {
    "Q_Learning":        Q_Learning,
    "SARSA":             SARSA,
    "Double_Q_Learning": Double_Q_Learning,
    "MC":                MC,
}

ALGO_COLORS = {
    "Q_Learning":        "#7ecef4",
    "SARSA":             "#90ee90",
    "Double_Q_Learning": "#f4a742",
    "MC":                "#f472b6",
}

BASE_PARAMS = dict(
    action_range=[-10.0, 10.0],
    learning_rate=0.1,
    n_episodes=10000,
    start_epsilon=1.0,
    epsilon_decay=0.001,
    final_epsilon=0.01,
    discount=0.99,
)

BG   = '#0f0f1a'
GRID = '#333355'


# ===========================================================
# PLOT HELPERS
# ===========================================================

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_q_surface(q_values, title, save_path):
    """
    GRAPH 1 — Q-Surface (3D)
    Objective: Visualizes the learned value landscape.
               A well-trained agent shows a clear peak at (cart_pos=0, pole_pos=0)
               meaning it learned that being centered with upright pole = highest value.
    Pole stability: YES — peak centered at origin = agent learned to stabilize.
    """
    points = {}
    for state, q_vals in q_values.items():
        x, y = state[0], state[1]
        max_q = float(np.max(q_vals))
        if (x, y) not in points or max_q > points[(x, y)]:
            points[(x, y)] = max_q
    if len(points) < 4:
        return
    xs = np.array([k[0] for k in points])
    ys = np.array([k[1] for k in points])
    zs = np.array(list(points.values()))
    xi = np.linspace(xs.min(), xs.max(), 80)
    yi = np.linspace(ys.min(), ys.max(), 80)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((xs, ys), zs, (XI, YI), method='linear')

    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)
    surf = ax.plot_surface(XI, YI, ZI, cmap='plasma', alpha=0.92, linewidth=0, antialiased=True)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.1)
    cbar.set_label('Max Q-Value', color='white', fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    ax.set_title(title, color='white', fontsize=12, pad=15)
    ax.set_xlabel('Cart Position', color='white', fontsize=9, labelpad=8)
    ax.set_ylabel('Pole Position', color='white', fontsize=9, labelpad=8)
    ax.set_zlabel('Max Q-Value', color='white', fontsize=9, labelpad=8)
    ax.tick_params(colors='white', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax.grid(True, color=GRID, alpha=0.5)
    plt.tight_layout()
    _save(fig, save_path)


def plot_learning_curves(reward_history, ep_length_history, title, save_dir, window=100):
    """
    GRAPH 2 — Training Reward Curve
    Objective: Learning efficiency — how quickly reward increases over episodes.
               Steeper rise = faster learner.
    Pole stability: INDIRECT — higher reward = pole stayed up longer per episode.

    GRAPH 3 — Training Episode Length Curve
    Objective: Directly shows how many steps the pole stayed up per episode.
    Pole stability: YES — longer episodes = pole not falling = more stable.
                   If curve reaches max steps = perfect stabilization.
    """
    os.makedirs(save_dir, exist_ok=True)

    for vals, color, ylabel, fname, subtitle in [
        (reward_history,    '#7ecef4', 'Cumulative Reward',     'learning_curve.png',        'Training Reward'),
        (ep_length_history, '#90ee90', 'Steps Before Falling',  'episode_length_curve.png',  'Episode Length (Pole Stability)'),
    ]:
        vals = np.array(vals)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG)
        _style_ax(ax)
        ax.plot(vals, alpha=0.25, color=color, linewidth=0.7)
        if len(vals) >= window:
            smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                    color='#f4c542', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_title(f"{subtitle}\n{title}", color='white', fontsize=11)
        ax.set_xlabel('Episode', color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        plt.tight_layout()
        _save(fig, os.path.join(save_dir, fname))


def plot_comparison(all_rewards, all_lengths, config_name, num_of_action, dsw, save_dir, window=100):
    """
    GRAPH 4 — Algorithm Comparison: Reward
    Objective: Which algorithm learns to get higher reward? = best learning efficiency.

    GRAPH 5 — Algorithm Comparison: Episode Length
    Objective: Which algorithm keeps the pole up the longest during training?
    Pole stability: YES — algorithm with highest episode length = best at stabilizing.
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{config_name} (actions={num_of_action}, dsw={dsw})"

    for data, ylabel, fname, subtitle in [
        (all_rewards, 'Cumulative Reward',    'comparison_reward.png',    'Reward Comparison'),
        (all_lengths, 'Steps Before Falling', 'comparison_ep_length.png', 'Episode Length Comparison (Pole Stability)'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG)
        _style_ax(ax)
        for algo_name, vals in data.items():
            vals = np.array(vals)
            color = ALGO_COLORS.get(algo_name, 'white')
            ax.plot(vals, alpha=0.15, color=color, linewidth=0.6)
            if len(vals) >= window:
                smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
                ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                        color=color, linewidth=2, label=algo_name)
        ax.set_title(f"{subtitle}\n{tag}", color='white', fontsize=11)
        ax.set_xlabel('Episode', color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
        plt.tight_layout()
        _save(fig, os.path.join(save_dir, fname))


def plot_deployment_results(deploy_results, task_name, n_deploy):
    """
    GRAPH 6 — Deployment Avg Reward (bar chart)
    Objective: Final performance after training — pure exploitation, no exploration.

    GRAPH 7 — Deployment Avg Episode Length (bar chart)
    Objective: How many steps the pole stayed up during deployment.
    Pole stability: YES — longer bar = pole stayed up longer = better stabilization.

    GRAPH 8 — Deployment Success Rate (bar chart)
    Objective: % of deployment episodes where pole survived to time limit.
    Pole stability: STRONGEST INDICATOR — 100% = pole never fell = perfect stability.
    """
    configs = list(deploy_results.keys())
    algos   = list(ALGORITHM_CLASSES.keys())
    x       = np.arange(len(configs))
    width   = 0.18
    save_dir = os.path.join("plots", task_name, "deployment")
    os.makedirs(save_dir, exist_ok=True)

    for metric, ylabel, fname, is_pct in [
        ('avg_reward',   'Avg Cumulative Reward',      'deployment_reward.png',       False),
        ('avg_length',   'Avg Episode Length (steps)',  'deployment_ep_length.png',    False),
        ('success_rate', 'Success Rate (%)',            'deployment_success_rate.png', True),
    ]:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(BG)
        _style_ax(ax)
        for i, algo in enumerate(algos):
            vals = [deploy_results[cfg].get(algo, {}).get(metric, 0) for cfg in configs]
            if is_pct:
                vals = [v * 100 for v in vals]
            bars = ax.bar(x + i * width, vals, width, label=algo,
                          color=ALGO_COLORS.get(algo, 'white'), alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{v:.1f}{"%" if is_pct else ""}',
                        ha='center', va='bottom', color='white', fontsize=7)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(configs, color='white', fontsize=9, rotation=15)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.set_title(f"Deployment — {ylabel}\n(epsilon=0, {n_deploy} episodes per config)",
                     color='white', fontsize=11)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        plt.tight_layout()
        _save(fig, os.path.join(save_dir, fname))


def plot_resolution_effect(all_results, task_name, window=100):
    """
    GRAPH 9 — Resolution Effect: num_of_action (per algorithm)
    Objective: Shows how low/normal/high action space resolution affects
               learning speed and pole stability for each algorithm.
               Answers HW Part 3 Question 3.

    GRAPH 10 — Resolution Effect: discretize_state_weight (per algorithm)
    Objective: Shows how low/normal/high observation space resolution
               affects learning for each algorithm.
               Answers HW Part 3 Question 3.
    """
    save_dir = os.path.join("plots", task_name, "resolution_effect")
    os.makedirs(save_dir, exist_ok=True)

    res_colors = {"low": "#f472b6", "normal": "#f4c542", "high": "#7ecef4"}

    vary_groups = {
        "action": {
            "low":    ("action_low",    "low (5 actions)"),
            "normal": ("action_normal", "normal (11 actions)"),
            "high":   ("action_high",   "high (21 actions)"),
        },
        "dsw": {
            "low":    ("dsw_low",     "low dsw"),
            "normal": ("action_normal", "normal dsw"),
            "high":   ("dsw_high",    "high dsw"),
        },
    }

    for vary_name, groups in vary_groups.items():
        for algo_name in ALGORITHM_CLASSES.keys():
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            fig.patch.set_facecolor(BG)
            fig.suptitle(
                f"{algo_name} — Effect of {'Action Space' if vary_name=='action' else 'Observation Space'} Resolution",
                color='white', fontsize=12
            )

            for ax, metric_key, ylabel in zip(
                axes,
                ['rewards', 'lengths'],
                ['Cumulative Reward', 'Episode Length (steps) — Pole Stability']
            ):
                _style_ax(ax)
                for level, (cfg_key, label) in groups.items():
                    if cfg_key not in all_results:
                        continue
                    if algo_name not in all_results[cfg_key]:
                        continue
                    vals = np.array(all_results[cfg_key][algo_name][metric_key])
                    color = res_colors[level]
                    ax.plot(vals, alpha=0.15, color=color, linewidth=0.6)
                    if len(vals) >= window:
                        smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
                        ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                                color=color, linewidth=2, label=label)
                ax.set_xlabel('Episode', color='white', fontsize=9)
                ax.set_ylabel(ylabel, color='white', fontsize=9)
                ax.set_title(ylabel, color='white', fontsize=10)
                ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

            plt.tight_layout()
            fname = f"{algo_name}_{vary_name}_resolution.png"
            _save(fig, os.path.join(save_dir, fname))


# ===========================================================
# HELPER — extract single-env obs slice from batched obs
# ===========================================================

def _slice_obs(obs, i):
    """Return a single-env obs dict from a batched obs dict at index i."""
    return {k: v[i:i+1] for k, v in obs.items()}


def _build_batched_action(action_tensors, num_envs):
    """Stack per-env action tensors into a single (num_envs, 1) tensor."""
    return torch.cat(action_tensors, dim=0)


# ===========================================================
# TRAINING FUNCTIONS  (vectorized: each env tracks its own episode)
# ===========================================================

def train_q_learning(env, agent, n_episodes, task_name, algo_name, config_name,
                     num_of_action, action_range, dsw):
    """
    Vectorized Q-Learning training loop.
    Runs num_envs episodes in parallel. Each env independently tracks its own
    episode. When an env finishes, its result is recorded and it seamlessly
    continues with the next episode via Isaac Lab's auto-reset.
    """
    num_envs = env.unwrapped.num_envs
    reward_history, length_history = [], []
    episodes_done = 0
    sum_reward = 0

    # Per-env tracking
    cum_rewards = [0.0] * num_envs
    steps_count = [0]   * num_envs

    obs, _ = env.reset()

    with tqdm(total=n_episodes, desc=f"  {algo_name} [{config_name}]") as pbar:
        while episodes_done < n_episodes:
            # Get action for each env independently
            action_tensors = []
            action_idxs = []
            for i in range(num_envs):
                obs_i = _slice_obs(obs, i)
                a_tensor, a_idx = agent.get_action(obs_i)
                action_tensors.append(a_tensor)
                action_idxs.append(a_idx)
            action_batch = _build_batched_action(action_tensors, num_envs)

            next_obs, reward, terminated, truncated, _ = env.step(action_batch)

            for i in range(num_envs):
                if episodes_done >= n_episodes:
                    break
                obs_i      = _slice_obs(obs, i)
                next_obs_i = _slice_obs(next_obs, i)
                r_i        = reward[i].item()
                term_i     = terminated[i].item()
                trunc_i    = truncated[i].item()

                cum_rewards[i] += r_i
                steps_count[i] += 1

                agent.update(obs=obs_i, action=action_idxs[i], reward=r_i,
                             next_obs=next_obs_i, terminated=term_i)

                if term_i or trunc_i:
                    reward_history.append(cum_rewards[i])
                    length_history.append(steps_count[i])
                    sum_reward += cum_rewards[i]
                    episodes_done += 1
                    pbar.update(1)
                    cum_rewards[i] = 0.0
                    steps_count[i] = 0
                    if episodes_done % 100 == 0:
                        avg_r   = sum_reward / 100
                        avg_len = float(np.mean(length_history[-100:]))
                        print(f"    ep={episodes_done} avg_r={avg_r:.2f} avg_len={avg_len:.1f} eps={agent.epsilon:.3f}")
                        sum_reward = 0

            agent.decay_epsilon()
            obs = next_obs

    _save_q(agent, task_name, algo_name, config_name, n_episodes, num_of_action, action_range, dsw)
    return reward_history, length_history


def train_sarsa(env, agent, n_episodes, task_name, algo_name, config_name,
                num_of_action, action_range, dsw):
    """
    Vectorized SARSA training loop.
    SARSA needs the next action BEFORE updating, so we pre-select actions
    per env and carry them forward.
    """
    num_envs = env.unwrapped.num_envs
    reward_history, length_history = [], []
    episodes_done = 0
    sum_reward = 0

    cum_rewards = [0.0] * num_envs
    steps_count = [0]   * num_envs

    obs, _ = env.reset()

    # Pre-select first action for each env
    action_tensors = []
    action_idxs = []
    for i in range(num_envs):
        a_tensor, a_idx = agent.get_action(_slice_obs(obs, i))
        action_tensors.append(a_tensor)
        action_idxs.append(a_idx)

    with tqdm(total=n_episodes, desc=f"  {algo_name} [{config_name}]") as pbar:
        while episodes_done < n_episodes:
            action_batch = _build_batched_action(action_tensors, num_envs)
            next_obs, reward, terminated, truncated, _ = env.step(action_batch)

            next_action_tensors = []
            next_action_idxs = []
            for i in range(num_envs):
                na_tensor, na_idx = agent.get_action(_slice_obs(next_obs, i))
                next_action_tensors.append(na_tensor)
                next_action_idxs.append(na_idx)

            for i in range(num_envs):
                if episodes_done >= n_episodes:
                    break
                obs_i      = _slice_obs(obs, i)
                next_obs_i = _slice_obs(next_obs, i)
                r_i        = reward[i].item()
                term_i     = terminated[i].item()
                trunc_i    = truncated[i].item()

                cum_rewards[i] += r_i
                steps_count[i] += 1

                agent.update(obs=obs_i, action=action_idxs[i], reward=r_i,
                             next_obs=next_obs_i, next_action=next_action_idxs[i],
                             terminated=term_i)

                if term_i or trunc_i:
                    reward_history.append(cum_rewards[i])
                    length_history.append(steps_count[i])
                    sum_reward += cum_rewards[i]
                    episodes_done += 1
                    pbar.update(1)
                    cum_rewards[i] = 0.0
                    steps_count[i] = 0
                    if episodes_done % 100 == 0:
                        avg_r   = sum_reward / 100
                        avg_len = float(np.mean(length_history[-100:]))
                        print(f"    ep={episodes_done} avg_r={avg_r:.2f} avg_len={avg_len:.1f} eps={agent.epsilon:.3f}")
                        sum_reward = 0

            agent.decay_epsilon()
            obs = next_obs
            action_tensors = next_action_tensors
            action_idxs    = next_action_idxs

    _save_q(agent, task_name, algo_name, config_name, n_episodes, num_of_action, action_range, dsw)
    return reward_history, length_history


def train_double_q(env, agent, n_episodes, task_name, algo_name, config_name,
                   num_of_action, action_range, dsw):
    return train_q_learning(env, agent, n_episodes, task_name, algo_name, config_name,
                            num_of_action, action_range, dsw)


def train_mc(env, agent, n_episodes, task_name, algo_name, config_name,
             num_of_action, action_range, dsw):
    """
    Vectorized MC training loop.
    Each env keeps its own episode history. When an env finishes,
    we call agent.update() with that env's history then clear it.
    """
    num_envs = env.unwrapped.num_envs
    reward_history, length_history = [], []
    episodes_done = 0
    sum_reward = 0

    cum_rewards  = [0.0] * num_envs
    steps_count  = [0]   * num_envs
    # Per-env episode history (MC needs full episode before updating)
    env_obs_hist    = [[] for _ in range(num_envs)]
    env_action_hist = [[] for _ in range(num_envs)]
    env_reward_hist = [[] for _ in range(num_envs)]

    obs, _ = env.reset()

    with tqdm(total=n_episodes, desc=f"  {algo_name} [{config_name}]") as pbar:
        while episodes_done < n_episodes:
            action_tensors = []
            action_idxs = []
            for i in range(num_envs):
                a_tensor, a_idx = agent.get_action(_slice_obs(obs, i))
                action_tensors.append(a_tensor)
                action_idxs.append(a_idx)
            action_batch = _build_batched_action(action_tensors, num_envs)

            next_obs, reward, terminated, truncated, _ = env.step(action_batch)

            for i in range(num_envs):
                if episodes_done >= n_episodes:
                    break
                obs_i  = _slice_obs(obs, i)
                r_i    = reward[i].item()
                term_i = terminated[i].item()
                trunc_i = truncated[i].item()

                env_obs_hist[i].append(obs_i)
                env_action_hist[i].append(action_idxs[i])
                env_reward_hist[i].append(r_i)
                cum_rewards[i] += r_i
                steps_count[i] += 1

                if term_i or trunc_i:
                    # Run MC update using this env's episode history
                    agent.obs_hist    = env_obs_hist[i]
                    agent.action_hist = env_action_hist[i]
                    agent.reward_hist = env_reward_hist[i]
                    agent.update()

                    reward_history.append(cum_rewards[i])
                    length_history.append(steps_count[i])
                    sum_reward += cum_rewards[i]
                    episodes_done += 1
                    pbar.update(1)

                    # Reset this env's tracking
                    env_obs_hist[i]    = []
                    env_action_hist[i] = []
                    env_reward_hist[i] = []
                    cum_rewards[i] = 0.0
                    steps_count[i] = 0

                    if episodes_done % 100 == 0:
                        avg_r   = sum_reward / 100
                        avg_len = float(np.mean(length_history[-100:]))
                        print(f"    ep={episodes_done} avg_r={avg_r:.2f} avg_len={avg_len:.1f} eps={agent.epsilon:.3f}")
                        sum_reward = 0

            agent.decay_epsilon()
            obs = next_obs

    _save_q(agent, task_name, algo_name, config_name, n_episodes, num_of_action, action_range, dsw)
    return reward_history, length_history


TRAIN_FN = {
    "Q_Learning":        train_q_learning,
    "SARSA":             train_sarsa,
    "Double_Q_Learning": train_double_q,
    "MC":                train_mc,
}


def _save_q(agent, task_name, algo_name, config_name, episode, num_of_action, action_range, dsw):
    filename = f"{algo_name}_{config_name}_{episode}_{num_of_action}_{action_range[1]}_{dsw[0]}_{dsw[1]}.json"
    path = os.path.join(f"q_value/{task_name}/{algo_name}/{config_name}")
    agent.save_q_value(path, filename)


# ===========================================================
# DEPLOYMENT EVALUATION
# ===========================================================

def evaluate_agent(env, agent, n_episodes):
    """
    Run agent with epsilon=0 for n_episodes (vectorized).
    success = episode ended with truncated=True = pole survived to time limit.
    """
    agent.epsilon = 0.0
    num_envs = env.unwrapped.num_envs
    rewards, lengths, successes = [], [], []
    episodes_done = 0

    cum_rewards = [0.0] * num_envs
    steps_count = [0]   * num_envs

    obs, _ = env.reset()

    with tqdm(total=n_episodes, desc="    Deploying", leave=False) as pbar:
        while episodes_done < n_episodes:
            action_tensors = []
            for i in range(num_envs):
                a_tensor, _ = agent.get_action(_slice_obs(obs, i))
                action_tensors.append(a_tensor)
            action_batch = _build_batched_action(action_tensors, num_envs)

            next_obs, reward, terminated, truncated, _ = env.step(action_batch)

            for i in range(num_envs):
                if episodes_done >= n_episodes:
                    break
                cum_rewards[i] += reward[i].item()
                steps_count[i] += 1
                if terminated[i].item() or truncated[i].item():
                    rewards.append(cum_rewards[i])
                    lengths.append(steps_count[i])
                    successes.append(float(truncated[i].item()))
                    cum_rewards[i] = 0.0
                    steps_count[i] = 0
                    episodes_done += 1
                    pbar.update(1)

            obs = next_obs

    return {
        'avg_reward':   float(np.mean(rewards)),
        'avg_length':   float(np.mean(lengths)),
        'success_rate': float(np.mean(successes)),
    }



# ===========================================================
# MAIN
# ===========================================================

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    task_name = str(args_cli.task).split('-')[0]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    p   = BASE_PARAMS

    all_results    = defaultdict(dict)   # [config][algo] = {rewards, lengths}
    deploy_results = defaultdict(dict)   # [config][algo] = {avg_reward, avg_length, success_rate}
    trained_agents = defaultdict(dict)   # [config][algo] = agent

    while simulation_app.is_running():
        with torch.inference_mode():

            # ── PHASE 1: TRAINING ──────────────────────────
            for (config_name, num_of_action, dsw) in EXPERIMENTS:
                print(f"\n{'='*60}")
                print(f"TRAINING | {config_name} | actions={num_of_action} | dsw={dsw}")
                print(f"{'='*60}")

                for algo_name, AlgoClass in ALGORITHM_CLASSES.items():
                    print(f"\n--- {algo_name} ---")
                    agent = AlgoClass(
                        num_of_action=num_of_action,
                        action_range=p['action_range'],
                        discretize_state_weight=dsw,
                        learning_rate=p['learning_rate'],
                        initial_epsilon=p['start_epsilon'],
                        epsilon_decay=p['epsilon_decay'],
                        final_epsilon=p['final_epsilon'],
                        discount_factor=p['discount'],
                    )

                    reward_history, length_history = TRAIN_FN[algo_name](
                        env=env, agent=agent, n_episodes=p['n_episodes'],
                        task_name=task_name, algo_name=algo_name,
                        config_name=config_name, num_of_action=num_of_action,
                        action_range=p['action_range'], dsw=dsw,
                    )

                    all_results[config_name][algo_name] = {
                        'rewards': reward_history,
                        'lengths': length_history,
                    }
                    trained_agents[config_name][algo_name] = agent

                    plot_dir = os.path.join("plots", task_name, algo_name, config_name)

                    # Graphs 2 & 3
                    plot_learning_curves(
                        reward_history, length_history,
                        title=f"{algo_name} — {config_name} (actions={num_of_action}, dsw={dsw})",
                        save_dir=plot_dir,
                    )
                    # Graph 1
                    plot_q_surface(
                        agent.q_values,
                        title=f"Q-Surface: {algo_name} — {config_name}",
                        save_path=os.path.join(plot_dir, "q_surface.png"),
                    )

                # Graphs 4 & 5
                plot_comparison(
                    all_rewards={a: all_results[config_name][a]['rewards'] for a in all_results[config_name]},
                    all_lengths={a: all_results[config_name][a]['lengths'] for a in all_results[config_name]},
                    config_name=config_name, num_of_action=num_of_action, dsw=dsw,
                    save_dir=os.path.join("plots", task_name, "comparisons", config_name),
                )

            # ── PHASE 2: DEPLOYMENT ─────────────────────────
            print(f"\n{'='*60}")
            print(f"DEPLOYMENT EVALUATION (epsilon=0, {args_cli.deploy_episodes} episodes each)")
            print(f"{'='*60}")

            for config_name in trained_agents:
                for algo_name, agent in trained_agents[config_name].items():
                    print(f"  {algo_name} [{config_name}]")
                    result = evaluate_agent(env, agent, args_cli.deploy_episodes)
                    deploy_results[config_name][algo_name] = result
                    print(f"    reward={result['avg_reward']:.2f}  "
                          f"length={result['avg_length']:.1f}  "
                          f"success={result['success_rate']*100:.1f}%")

            # Graphs 6, 7, 8
            plot_deployment_results(deploy_results, task_name, args_cli.deploy_episodes)

            # Graphs 9 & 10
            plot_resolution_effect(all_results, task_name)

        print(f"\n!!! Complete! Plots saved to: plots/{task_name}/")
        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()