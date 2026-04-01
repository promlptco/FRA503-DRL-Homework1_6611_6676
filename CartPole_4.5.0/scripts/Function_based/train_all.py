"""
scripts/Function_based/train_all.py

Train all 8 function approximation RL algorithms and generate a full
suite of analysis graphs.

Algorithms:
  Linear_QN, DQN, MC_REINFORCE, AC, A2C, PPO, TD3, SAC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT GRAPH SUMMARY  (total ≈ 53 graphs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per-algorithm  (8 algos)  →  plots/{task}/Function_based/{algo}/
  [A] learning_curve.png            reward per episode (raw + smoothed)
  [B] episode_length_curve.png      steps survived per episode
  [C] reward_with_std.png           smoothed reward ± 1-std band
  [D] actor_loss_curve.png          actor loss over update steps
  [E] critic_loss_curve.png         critic/value loss over update steps
      (D/E only for algorithms that have a neural actor/critic)
  [F] epsilon_curve.png             epsilon decay  (Linear_QN, DQN only)
  [G] entropy_curve.png             policy entropy (AC, A2C, PPO, SAC only)
  [H] steps_vs_reward.png           reward vs total env steps

Comparison  →  plots/{task}/Function_based/comparisons/
  [I]  comparison_reward.png            all algos smoothed reward
  [J]  comparison_ep_length.png         all algos episode length
  [K]  comparison_steps_reward.png      reward vs env steps (sample efficiency)
  [L]  comparison_actor_loss.png        actor loss (neural algos only)
  [M]  comparison_critic_loss.png       critic loss (neural algos only)

Deployment  →  plots/{task}/Function_based/deployment/
  [N]  deployment_reward.png            bar: avg reward
  [O]  deployment_ep_length.png         bar: avg episode length
  [P]  deployment_success_rate.png      bar: pct episodes survived to max_steps
  [Q]  deployment_length_hist.png       histogram of episode lengths
  [R]  deployment_reward_per_ep.png     per-episode deployment reward (line)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run:
  python scripts/Function_based/train_all.py \\
      --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 1
"""

import argparse
import sys
import os
import types
import math

from isaaclab.app import AppLauncher

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "CartPole"))

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train all function approximation RL agents.")
parser.add_argument("--video",           action="store_true", default=False)
parser.add_argument("--video_length",    type=int,   default=200)
parser.add_argument("--num_envs",        type=int,   default=1)
parser.add_argument("--task",            type=str,   default=None)
parser.add_argument("--seed",            type=int,   default=None)
parser.add_argument("--max_iterations",  type=int,   default=None)
parser.add_argument("--deploy_episodes", type=int,   default=100)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import CartPole.tasks  # noqa: F401

from RL_Algorithm.Function_based.Linear_Q     import Linear_QN
from RL_Algorithm.Function_based.DQN          import DQN
from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
from RL_Algorithm.Function_based.AC           import AC
from RL_Algorithm.Function_based.A2C          import A2C
from RL_Algorithm.Function_based.PPO          import PPO
from RL_Algorithm.Function_based.TD3          import TD3
from RL_Algorithm.Function_based.SAC          import SAC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = False

# ==========================================================================
# CONSTANTS
# ==========================================================================

N_EPISODES   = 10000
MAX_STEPS    = 500
N_OBS        = 4
ACTION_RANGE = [-10.0, 10.0]
ALL_ALGOS    = ["Linear_QN", "DQN", "MC_REINFORCE", "AC", "A2C", "PPO", "TD3", "SAC"]

COLORS = {
    "Linear_QN":    "steelblue",
    "DQN":          "darkorange",
    "MC_REINFORCE": "green",
    "AC":           "red",
    "A2C":          "purple",
    "PPO":          "saddlebrown",
    "TD3":          "deeppink",
    "SAC":          "teal",
}

HAS_ACTOR_LOSS  = {"AC", "A2C", "PPO", "TD3", "SAC"}
HAS_CRITIC_LOSS = {"AC", "A2C", "PPO", "TD3", "SAC"}
HAS_EPSILON     = {"Linear_QN", "DQN"}
HAS_ENTROPY     = {"AC", "A2C", "PPO", "SAC"}

# ==========================================================================
# HYPERPARAMETERS
# ==========================================================================

BASE_PARAMS = dict(
    n_observations  = N_OBS,
    action_range    = ACTION_RANGE,
    learning_rate   = 3e-4,
    discount_factor = 0.99,
)

ALGO_PARAMS = {
    "Linear_QN": dict(
        num_of_action   = 11,
        initial_epsilon = 1.0,
        epsilon_decay   = 5e-4,
        final_epsilon   = 0.01,
        learning_rate   = 1e-3,
    ),
    "DQN": dict(
        num_of_action       = 11,
        hidden_dim          = 128,
        dropout             = 0.1,
        tau                 = 0.005,
        initial_epsilon     = 1.0,
        epsilon_decay       = 5e-4,
        final_epsilon       = 0.01,
        buffer_size         = 50000,
        batch_size          = 128,
        update_freq         = 4,    # gradient update every 4 env steps
        target_update_freq  = 200,  # hard copy target net every 200 steps
    ),
    "MC_REINFORCE": dict(
        num_of_action   = 11,
        hidden_dim      = 128,
        dropout         = 0.1,
    ),
    "AC": dict(
        num_of_action   = 1,
        hidden_dim      = 256,
        entropy_coef    = 0.01,
    ),
    "A2C": dict(
        num_of_action            = 1,
        hidden_dim               = 256,
        gae_lambda               = 0.95,
        value_loss_coef          = 0.5,
        entropy_coef             = 0.01,
        num_transitions_per_env  = 64,
        # num_envs is set dynamically from args_cli.num_envs in _make_agent()
        num_envs                 = 1,
    ),
    "PPO": dict(
        num_of_action            = 1,
        hidden_dim               = 256,
        gae_lambda               = 0.95,
        clip_param               = 0.2,
        value_loss_coef          = 0.5,
        entropy_coef             = 0.01,
        num_learning_epochs      = 4,
        num_mini_batches         = 4,
        num_transitions_per_env  = 64,
        # num_envs is set dynamically from args_cli.num_envs in _make_agent()
        num_envs                 = 1,
    ),
    "TD3": dict(
        num_of_action   = 1,
        hidden_dim      = 256,
        tau             = 0.005,
        buffer_size     = 100_000,
        batch_size      = 256,
        policy_noise    = 0.2,
        noise_clip      = 0.5,
        expl_noise      = 0.1,
        policy_delay    = 2,
        update_freq     = 4,     # update every 4 steps → ~4x faster, same sample count
    ),
    "SAC": dict(
        num_of_action   = 1,
        hidden_dim      = 256,
        tau             = 0.005,
        buffer_size     = 100_000,
        batch_size      = 256,
        alpha           = 0.2,
        auto_entropy    = True,
        update_freq     = 4,     # update every 4 steps → ~4x faster, same sample count
    ),
}

_LINEAR_DROP = {
    'n_observations', 'hidden_dim', 'dropout', 'tau', 'buffer_size', 'batch_size',
    'entropy_coef', 'gae_lambda', 'value_loss_coef', 'num_transitions_per_env',
    'num_envs', 'num_learning_epochs', 'num_mini_batches',
    'policy_noise', 'noise_clip', 'expl_noise', 'policy_delay', 'alpha', 'auto_entropy',
}
_REINFORCE_DROP = {
    'initial_epsilon', 'epsilon_decay', 'final_epsilon', 'buffer_size', 'batch_size',
    'tau', 'gae_lambda', 'value_loss_coef', 'num_transitions_per_env', 'num_envs',
    'num_learning_epochs', 'num_mini_batches', 'policy_noise', 'noise_clip',
    'expl_noise', 'policy_delay', 'alpha', 'auto_entropy', 'entropy_coef', 'clip_param',
}

# ==========================================================================
# UTILITY
# ==========================================================================

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def smooth(values, window=50):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def smooth_with_std(values, window=100):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr, np.zeros_like(arr)
    n = len(arr) - window + 1
    m = np.convolve(arr, np.ones(window) / window, mode='valid')
    s = np.array([arr[i:i + window].std() for i in range(n)])
    return m, s


def _ep_x(n):
    return np.arange(1, n + 1)


# ==========================================================================
# [A][B] LEARNING CURVE
# ==========================================================================

def plot_learning_curve(reward_history, length_history, title, save_dir):
    c   = COLORS.get(title, 'steelblue')
    eps = _ep_x(len(reward_history))

    # [A] reward
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, reward_history, alpha=0.15, color=c, lw=0.7)
    s = smooth(reward_history)
    ax.plot(_ep_x(len(s)), s, color=c, lw=2, label='Smoothed (w=50)')
    ax.set_title(f'{title} — Learning Curve (Reward)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'learning_curve.png'))

    # [B] episode length
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, length_history, alpha=0.15, color=c, lw=0.7)
    s = smooth(length_history)
    ax.plot(_ep_x(len(s)), s, color=c, lw=2, label='Smoothed (w=50)')
    ax.axhline(MAX_STEPS, color='black', lw=1, ls='--', label=f'Max steps ({MAX_STEPS})')
    ax.set_title(f'{title} — Episode Length')
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps Survived')
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'episode_length_curve.png'))


# ==========================================================================
# [C] REWARD ± STD BAND
# ==========================================================================

def plot_reward_std(reward_history, title, save_dir):
    """Shows training stability — wide band = high variance = unstable learning."""
    m, s = smooth_with_std(reward_history, window=100)
    x    = _ep_x(len(m))
    c    = COLORS.get(title, 'steelblue')

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, m - s, m + s, alpha=0.25, color=c, label='±1 std')
    ax.plot(x, m, color=c, lw=2, label='Mean (w=100)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title(f'{title} — Reward Stability (Mean ± Std)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'reward_with_std.png'))


# ==========================================================================
# [D][E] ACTOR / CRITIC LOSS CURVES
# ==========================================================================

def plot_loss_curves(actor_losses, critic_losses, title, save_dir):
    c = COLORS.get(title, 'steelblue')
    w = 50

    if actor_losses:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(_ep_x(len(actor_losses)), actor_losses, alpha=0.18, color=c, lw=0.7)
        s = smooth(actor_losses, window=min(w, max(1, len(actor_losses) // 4)))
        ax.plot(_ep_x(len(s)), s, color=c, lw=2, label='Smoothed')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(f'{title} — Actor Loss')
        ax.set_xlabel('Update step'); ax.set_ylabel('Actor Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(save_dir, 'actor_loss_curve.png'))

    if critic_losses:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(_ep_x(len(critic_losses)), critic_losses, alpha=0.18, color='darkorange', lw=0.7)
        s = smooth(critic_losses, window=min(w, max(1, len(critic_losses) // 4)))
        ax.plot(_ep_x(len(s)), s, color='darkorange', lw=2, label='Smoothed')
        ax.set_title(f'{title} — Critic / Value Loss')
        ax.set_xlabel('Update step'); ax.set_ylabel('Critic Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(save_dir, 'critic_loss_curve.png'))


# ==========================================================================
# [F] EPSILON DECAY
# ==========================================================================

def plot_epsilon(epsilon_history, title, save_dir):
    """Confirms epsilon decays as scheduled — flat tail means final_epsilon reached."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(_ep_x(len(epsilon_history)), epsilon_history,
            color=COLORS.get(title, 'steelblue'), lw=1.8)
    ax.set_title(f'{title} — Epsilon Decay')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε (exploration rate)')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'epsilon_curve.png'))


# ==========================================================================
# [G] ENTROPY CURVE
# ==========================================================================

def plot_entropy(entropy_history, title, save_dir):
    """High entropy = exploring. Collapse to zero = policy became deterministic."""
    c = COLORS.get(title, 'green')
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(_ep_x(len(entropy_history)), entropy_history, alpha=0.2, color=c, lw=0.7)
    s = smooth(entropy_history, window=min(50, max(1, len(entropy_history) // 4)))
    ax.plot(_ep_x(len(s)), s, color=c, lw=2, label='Smoothed')
    ax.set_title(f'{title} — Policy Entropy')
    ax.set_xlabel('Episode'); ax.set_ylabel('Entropy')
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'entropy_curve.png'))


# ==========================================================================
# [H] REWARD vs ENV STEPS  (per algorithm)
# ==========================================================================

def plot_steps_vs_reward(reward_history, steps_history, title, save_dir):
    """
    X-axis = total environment steps (not episodes).
    Fairer measure of sample efficiency — algorithms that use more steps
    per episode are penalised on this plot.
    """
    cumsteps = np.cumsum(steps_history).astype(float)
    c        = COLORS.get(title, 'steelblue')

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cumsteps, reward_history, alpha=0.15, color=c, lw=0.7)
    if len(cumsteps) > 50:
        bins   = np.linspace(0, cumsteps[-1], 300)
        binned = np.interp(bins, cumsteps, reward_history)
        s      = smooth(binned, window=20)
        ax.plot(np.linspace(0, cumsteps[-1], len(s)), s, color=c, lw=2, label='Smoothed')
    ax.set_title(f'{title} — Reward vs Environment Steps')
    ax.set_xlabel('Total Env Steps'); ax.set_ylabel('Cumulative Reward')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k'))
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'steps_vs_reward.png'))


# ==========================================================================
# [I–M] COMPARISON PLOTS
# ==========================================================================

def plot_comparison(all_rewards, all_lengths, all_steps,
                    all_actor, all_critic, save_dir):

    # [I] reward
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, r in all_rewards.items():
        s = smooth(r)
        ax.plot(_ep_x(len(s)), s, color=COLORS[name], lw=2, label=name)
    ax.set_title('All Algorithms — Cumulative Reward (Smoothed)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'comparison_reward.png'))

    # [J] episode length
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, l in all_lengths.items():
        s = smooth(l)
        ax.plot(_ep_x(len(s)), s, color=COLORS[name], lw=2, label=name)
    ax.axhline(MAX_STEPS, color='black', lw=1, ls='--', label=f'Max ({MAX_STEPS})')
    ax.set_title('All Algorithms — Episode Length (Smoothed)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps Survived')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'comparison_ep_length.png'))

    # [K] sample efficiency (reward vs env steps)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rewards in all_rewards.items():
        cumsteps = np.cumsum(all_steps[name]).astype(float)
        if len(cumsteps) > 50:
            bins   = np.linspace(0, cumsteps[-1], 400)
            binned = np.interp(bins, cumsteps, rewards)
            s      = smooth(binned, window=20)
            ax.plot(np.linspace(0, cumsteps[-1], len(s)), s,
                    color=COLORS[name], lw=2, label=name)
        else:
            ax.plot(cumsteps, rewards, color=COLORS[name], lw=2, label=name)
    ax.set_title('All Algorithms — Reward vs Env Steps (Sample Efficiency)')
    ax.set_xlabel('Total Env Steps'); ax.set_ylabel('Cumulative Reward')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k'))
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'comparison_steps_reward.png'))

    # [L] actor loss
    neural_actor = {k: v for k, v in all_actor.items() if v}
    if neural_actor:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, losses in neural_actor.items():
            s = smooth(losses, window=min(50, max(1, len(losses) // 4)))
            ax.plot(_ep_x(len(s)), s, color=COLORS[name], lw=2, label=name)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title('Neural Algorithms — Actor Loss (Smoothed)')
        ax.set_xlabel('Update step'); ax.set_ylabel('Actor Loss')
        ax.legend(ncol=2); ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(save_dir, 'comparison_actor_loss.png'))

    # [M] critic loss
    neural_critic = {k: v for k, v in all_critic.items() if v}
    if neural_critic:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, losses in neural_critic.items():
            s = smooth(losses, window=min(50, max(1, len(losses) // 4)))
            ax.plot(_ep_x(len(s)), s, color=COLORS[name], lw=2, label=name)
        ax.set_title('Neural Algorithms — Critic Loss (Smoothed)')
        ax.set_xlabel('Update step'); ax.set_ylabel('Critic / Value Loss')
        ax.legend(ncol=2); ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(save_dir, 'comparison_critic_loss.png'))


# ==========================================================================
# [N][O][P] DEPLOYMENT BAR CHARTS
# ==========================================================================

def plot_deployment_bars(deploy_results, save_dir):
    names   = list(deploy_results.keys())
    rewards = [deploy_results[n]['avg_reward']       for n in names]
    lengths = [deploy_results[n]['avg_length']       for n in names]
    success = [deploy_results[n]['success_rate']*100 for n in names]
    colors  = [COLORS[n] for n in names]

    for values, ylabel, title, fname in [
        (rewards, 'Avg Reward',         'Deployment — Average Reward',         'deployment_reward.png'),
        (lengths, 'Avg Episode Length', 'Deployment — Average Episode Length', 'deployment_ep_length.png'),
        (success, 'Success Rate (%)',   'Deployment — Success Rate',           'deployment_success_rate.png'),
    ]:
        vmax = max(abs(v) for v in values) if values else 1
        vmax = vmax if vmax > 0 else 1
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(names, values, color=colors, edgecolor='black', width=0.55)
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + 0.015 * vmax
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel); ax.set_xlabel('Algorithm')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, axis='y', alpha=0.3)
        _save(fig, os.path.join(save_dir, fname))


# ==========================================================================
# [Q] DEPLOYMENT LENGTH HISTOGRAM
# ==========================================================================

def plot_deployment_histogram(deploy_detail, save_dir):
    """
    Overlapping histogram of episode lengths during greedy deployment.
    Reveals consistency: an agent that always reaches MAX_STEPS stacks
    all its bars at the right edge.  A failed agent spreads across low values.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.linspace(0, MAX_STEPS + 10, 40)
    for name, detail in deploy_detail.items():
        ax.hist(detail['lengths'], bins=bins, alpha=0.45,
                color=COLORS[name], label=name, edgecolor='none')
    ax.axvline(MAX_STEPS, color='black', lw=1.5, ls='--',
               label=f'Max steps ({MAX_STEPS})')
    ax.set_title('Deployment — Distribution of Episode Lengths')
    ax.set_xlabel('Episode Length (steps)'); ax.set_ylabel('Count')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_length_hist.png'))


# ==========================================================================
# [R] DEPLOYMENT REWARD PER EPISODE
# ==========================================================================

def plot_deployment_per_episode(deploy_detail, save_dir):
    """
    Per-episode reward line during deployment for all algorithms.
    A flat high line = reliable.  Noisy/low line = inconsistent or failed.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, detail in deploy_detail.items():
        rewards = detail['rewards']
        ax.plot(_ep_x(len(rewards)), rewards,
                color=COLORS[name], lw=1.3, alpha=0.85, label=name)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title('Deployment — Per-Episode Reward (Greedy Policy)')
    ax.set_xlabel('Deployment Episode'); ax.set_ylabel('Cumulative Reward')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_reward_per_ep.png'))


# ==========================================================================
# [R2] REWARD VARIANCE ACROSS ALL ALGORITHMS (training stability)
# ==========================================================================

def plot_reward_variance_comparison(all_rewards, save_dir):
    """
    Rolling std of reward for all algorithms on one plot.
    Wide band = unstable learning.  Narrow band near zero = stable.
    Directly answers: which algorithm is the most stable learner?
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rewards in all_rewards.items():
        _, s = smooth_with_std(rewards, window=100)
        ax.plot(_ep_x(len(s)), s, color=COLORS[name], lw=2, label=name)
    ax.set_title('All Algorithms — Reward Variance During Training (Rolling std, w=100)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Std of Reward (100-ep window)')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'comparison_reward_variance.png'))


# ==========================================================================
# [R3] FIRST EPISODE TO REACH SOLVED THRESHOLD
# ==========================================================================

def plot_solved_episode(all_lengths, save_dir):
    """
    Bar chart: at which episode did each algorithm first sustain avg_len >= 450?
    Directly answers 'which algorithm learns fastest' in one number.
    N/A bar = algorithm never solved the task.
    """
    SOLVE_THRESHOLD = 450   # steps — 90% of max_steps=500
    WINDOW          = 100

    names   = list(all_lengths.keys())
    solved_at = []
    for name in names:
        lengths = np.array(all_lengths[name], dtype=float)
        found   = None
        for i in range(WINDOW, len(lengths) + 1):
            if np.mean(lengths[i - WINDOW:i]) >= SOLVE_THRESHOLD:
                found = i
                break
        solved_at.append(found)

    colors  = [COLORS[n] for n in names]
    y_vals  = [v if v is not None else 0 for v in solved_at]
    labels  = [str(v) if v is not None else 'Never' for v in solved_at]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, y_vals, color=colors, edgecolor='black', width=0.55)
    for bar, label, val in zip(bars, labels, solved_at):
        ypos = bar.get_height() + 50
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='red' if val is None else 'black')
    ax.set_title(f'Learning Efficiency — First Episode with 100-ep Avg ≥ {SOLVE_THRESHOLD} steps',
                 fontsize=12)
    ax.set_ylabel('Episode number'); ax.set_xlabel('Algorithm')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, os.path.join(save_dir, 'comparison_solved_episode.png'))


# ==========================================================================
# [R4] DEPLOYMENT — POLE ANGLE STD BAR CHART
# ==========================================================================

def plot_pole_angle_stability_bars(deploy_detail, save_dir):
    """
    Two bar charts from the recorded deployment episode:
      - Mean |pole angle| (lower = better, stays more upright)
      - Std of pole angle  (lower = better, less oscillation)
    These give concrete numbers for the 'which algorithm is most stable' question.
    """
    names    = list(deploy_detail.keys())
    colors   = [COLORS[n] for n in names]
    means    = []
    stds     = []

    for name in names:
        angles = deploy_detail[name].get('pole_angles', [])
        if angles:
            arr = np.abs(np.degrees(np.array(angles)))
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(np.degrees(np.array(angles)))))
        else:
            means.append(0.0)
            stds.append(0.0)

    for values, ylabel, title, fname in [
        (means, 'Mean |Pole Angle| (°)',
         'Deployment Stability — Mean Absolute Pole Angle (lower = better)',
         'deployment_pole_mean_angle.png'),
        (stds,  'Std of Pole Angle (°)',
         'Deployment Stability — Pole Angle Standard Deviation (lower = better)',
         'deployment_pole_std_angle.png'),
    ]:
        vmax = max(values) if max(values) > 0 else 1
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(names, values, color=colors, edgecolor='black', width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * vmax,
                    f'{val:.2f}°', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel); ax.set_xlabel('Algorithm')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, axis='y', alpha=0.3)
        _save(fig, os.path.join(save_dir, fname))



def plot_pole_angle_traces(deploy_detail, save_dir):
    """
    [S-individual] One subplot per algorithm showing pole angle (rad) vs timestep.

    A stable agent keeps the angle near 0 for all 500 steps.
    A failing agent shows the angle growing until the pole falls.
    Also plots ±termination threshold as dashed red lines for reference.
    """
    POLE_LIMIT_RAD = 0.5   # ~28.6° — typical CartPole termination threshold
    dt = 0.01              # env step-size from [INFO] logs (0.01s)

    n_algos = len(deploy_detail)
    ncols   = 2
    nrows   = (n_algos + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)

    for i, (name, detail) in enumerate(deploy_detail.items()):
        ax      = axes[i // ncols][i % ncols]
        angles  = detail.get('pole_angles', [])
        if not angles:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        t = np.arange(len(angles)) * dt
        ax.plot(t, np.degrees(angles), color=COLORS[name], lw=1.5)
        ax.axhline(0,  color='black', lw=0.8, ls='--', alpha=0.5, label='Upright (0°)')
        ax.axhline( np.degrees(POLE_LIMIT_RAD), color='red', lw=1, ls='--',
                    alpha=0.6, label='Term. limit')
        ax.axhline(-np.degrees(POLE_LIMIT_RAD), color='red', lw=1, ls='--', alpha=0.6)
        ax.set_title(f'{name}  ({len(angles)} steps)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Pole angle (°)')
        ax.set_ylim(-45, 45)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_algos, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle('Deployment — Pole Angle Over Time (Best Episode per Algorithm)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'deployment_pole_angle_traces.png'))


def plot_pole_angle_comparison(deploy_detail, save_dir):
    """
    [S-comparison] All algorithms on the same pole angle plot.

    Shows at a glance which algorithms maintain tightest control near 0°
    and which ones oscillate or diverge.
    """
    POLE_LIMIT_RAD = 0.5
    dt = 0.01

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, detail in deploy_detail.items():
        angles = detail.get('pole_angles', [])
        if not angles:
            continue
        t = np.arange(len(angles)) * dt
        ax.plot(t, np.degrees(angles), color=COLORS[name], lw=1.8,
                alpha=0.85, label=f"{name} ({len(angles)} steps)")

    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5, label='Upright (0°)')
    ax.axhline( np.degrees(POLE_LIMIT_RAD), color='red', lw=1.2, ls='--',
                alpha=0.7, label='Termination limit')
    ax.axhline(-np.degrees(POLE_LIMIT_RAD), color='red', lw=1.2, ls='--', alpha=0.7)
    ax.set_title('Deployment — Pole Angle Comparison (Best Episode, All Algorithms)',
                 fontsize=12)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Pole angle (°)')
    ax.set_ylim(-50, 50)
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_pole_angle_comparison.png'))


# ==========================================================================
# [T] CART POSITION OVER TIME
# ==========================================================================

def plot_cart_position_traces(deploy_detail, save_dir):
    """
    [T] Cart position (m) over time for all algorithms on one plot.

    Agents that apply too much force push the cart to the boundary (±2.4m)
    triggering termination. A good agent keeps the cart near centre.
    """
    CART_LIMIT = 2.4   # typical CartPole cart boundary (metres)
    dt = 0.01

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, detail in deploy_detail.items():
        cart = detail.get('cart_pos', [])
        if not cart:
            continue
        t = np.arange(len(cart)) * dt
        ax.plot(t, cart, color=COLORS[name], lw=1.8, alpha=0.85,
                label=f"{name} ({len(cart)} steps)")

    ax.axhline(0,          color='black', lw=1,   ls='--', alpha=0.5, label='Centre')
    ax.axhline( CART_LIMIT, color='red',   lw=1.2, ls='--', alpha=0.7, label='Cart limit')
    ax.axhline(-CART_LIMIT, color='red',   lw=1.2, ls='--', alpha=0.7)
    ax.set_title('Deployment — Cart Position Over Time (Best Episode, All Algorithms)',
                 fontsize=12)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Cart position (m)')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_cart_position.png'))


# ==========================================================================
# [U] PHASE PORTRAIT  (pole angle vs angular velocity)
# ==========================================================================

def plot_phase_portrait(deploy_detail, save_dir):
    """
    [U] Phase portrait: pole angle (x-axis) vs pole angular velocity (y-axis).

    A stable controller drives the system to the origin (0,0) and keeps it
    there — appears as a tight cluster near centre.
    A failed controller shows a spiral or trajectory shooting outward.
    This is the most informative single plot for stability analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, detail in deploy_detail.items():
        angles = detail.get('pole_angles', [])
        cart   = detail.get('cart_pos',   [])
        # pole angular velocity = obs[3], but we only stored obs[1] (angle)
        # and obs[0] (cart pos). Approximate ang_vel by finite difference.
        if len(angles) < 2:
            continue
        ang_vel = np.gradient(angles)   # approximate angular velocity
        ax.plot(np.degrees(angles), np.degrees(ang_vel),
                color=COLORS[name], lw=1.2, alpha=0.7, label=name)
        # Mark start
        ax.scatter(np.degrees(angles[0]), np.degrees(ang_vel[0]),
                   color=COLORS[name], s=60, zorder=5, marker='o')
        # Mark end
        ax.scatter(np.degrees(angles[-1]), np.degrees(ang_vel[-1]),
                   color=COLORS[name], s=60, zorder=5, marker='x')

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.scatter([0], [0], color='black', s=120, zorder=6, marker='*', label='Goal (0,0)')
    ax.set_title('Deployment — Phase Portrait (Pole Angle vs Angular Velocity)\n'
                 'Circle = start,  × = end,  ★ = goal', fontsize=11)
    ax.set_xlabel('Pole angle (°)'); ax.set_ylabel('Pole angular velocity (°/s)')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_phase_portrait.png'))


# ==========================================================================
# [V]  ACTION SIGNAL OVER TIME
# ==========================================================================

def plot_action_traces(deploy_detail, save_dir):
    """
    [V] Applied force (action) over time for all algorithms.

    A smooth, small-amplitude action = efficient control.
    High-frequency chattering = reactive/unstable policy.
    Saturated actions (at ±10) = policy is pushing hard but may not be controlling.
    """
    dt = 0.01
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, detail in deploy_detail.items():
        acts = detail.get('actions', [])
        if not acts:
            continue
        t = np.arange(len(acts)) * dt
        ax.plot(t, acts, color=COLORS[name], lw=1.5, alpha=0.75, label=name)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.set_title('Deployment — Applied Force Over Time (Best Episode, All Algorithms)',
                 fontsize=12)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Action (force, N)')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'deployment_action_traces.png'))




def _make_agent(algo_name, device):
    p = {**BASE_PARAMS, **ALGO_PARAMS[algo_name]}

    if algo_name == "Linear_QN":
        return Linear_QN(**{k: v for k, v in p.items() if k not in _LINEAR_DROP})
    elif algo_name == "DQN":
        return DQN(device=device, **p)
    elif algo_name == "MC_REINFORCE":
        return MC_REINFORCE(device=device,
                            **{k: v for k, v in p.items() if k not in _REINFORCE_DROP})
    elif algo_name == "AC":
        drop = {'initial_epsilon', 'epsilon_decay', 'final_epsilon',
                'buffer_size', 'batch_size', 'hidden_dim', 'dropout', 'tau',
                'gae_lambda', 'value_loss_coef', 'clip_param',
                'num_transitions_per_env', 'num_envs',
                'num_learning_epochs', 'num_mini_batches',
                'policy_noise', 'noise_clip', 'expl_noise', 'policy_delay',
                'alpha', 'auto_entropy'}
        return AC(device=device, **{k: v for k, v in p.items() if k not in drop})
    elif algo_name == "A2C":
        drop = {'initial_epsilon', 'epsilon_decay', 'final_epsilon',
                'buffer_size', 'batch_size', 'hidden_dim', 'dropout', 'tau',
                'clip_param', 'num_learning_epochs', 'num_mini_batches',
                'policy_noise', 'noise_clip', 'expl_noise', 'policy_delay',
                'alpha', 'auto_entropy'}
        params = {k: v for k, v in p.items() if k not in drop}
        params['num_envs'] = args_cli.num_envs if args_cli.num_envs else 1
        return A2C(device=device, **params)
    elif algo_name == "PPO":
        drop = {'initial_epsilon', 'epsilon_decay', 'final_epsilon',
                'buffer_size', 'batch_size', 'hidden_dim', 'dropout', 'tau',
                'policy_noise', 'noise_clip', 'expl_noise', 'policy_delay',
                'alpha', 'auto_entropy'}
        params = {k: v for k, v in p.items() if k not in drop}
        params['num_envs'] = args_cli.num_envs if args_cli.num_envs else 1
        return PPO(device=device, **params)
    elif algo_name == "TD3":
        drop = {'initial_epsilon', 'epsilon_decay', 'final_epsilon',
                'hidden_dim', 'dropout', 'gae_lambda', 'value_loss_coef',
                'entropy_coef', 'clip_param',
                'num_transitions_per_env', 'num_envs',
                'num_learning_epochs', 'num_mini_batches',
                'alpha', 'auto_entropy'}
        return TD3(device=device, **{k: v for k, v in p.items() if k not in drop})
    elif algo_name == "SAC":
        drop = {'initial_epsilon', 'epsilon_decay', 'final_epsilon',
                'hidden_dim', 'dropout', 'gae_lambda', 'value_loss_coef',
                'entropy_coef', 'clip_param',
                'num_transitions_per_env', 'num_envs',
                'num_learning_epochs', 'num_mini_batches',
                'policy_noise', 'noise_clip', 'expl_noise', 'policy_delay'}
        return SAC(device=device, **{k: v for k, v in p.items() if k not in drop})

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ==========================================================================
# LOSS / ENTROPY TRACKING  (monkey-patch helpers)
# ==========================================================================

def _init_tracking(agent):
    agent._actor_losses  = []
    agent._critic_losses = []
    agent._entropy_hist  = []
    agent._epsilon_hist  = []


def _patch_dqn(agent):
    """Capture DQN Bellman loss after every update_policy call."""
    orig = agent.update_policy.__func__

    def patched(self):
        sample = self._prepare_batch()
        if sample is None:
            return
        nfm, nfns, sb, ab, rb = sample
        loss = self.calculate_loss(nfm, nfns, sb, ab, rb)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self._critic_losses.append(loss.item())

    agent.update_policy = types.MethodType(patched, agent)


def _patch_on_policy(agent):
    """Capture actor/critic losses from AC, A2C, PPO update()."""
    orig = agent.update.__func__

    def patched(self):
        result = orig(self)
        if result is not None and len(result) == 2 and result[0] is not None:
            # Store floats — never tensors — to avoid retaining computation graphs
            a = result[0].item() if isinstance(result[0], torch.Tensor) else result[0]
            c = result[1].item() if isinstance(result[1], torch.Tensor) else result[1]
            if a is not None: self._actor_losses.append(a)
            if c is not None: self._critic_losses.append(c)
        return result

    agent.update = types.MethodType(patched, agent)


def _patch_off_policy_ac(agent):
    """Capture actor/critic losses from TD3/SAC calculate_loss()."""
    orig = agent.calculate_loss.__func__

    def patched(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        c_loss, a_loss = result
        self._critic_losses.append(c_loss.item())
        if a_loss is not None:
            self._actor_losses.append(a_loss.item())
        return result

    agent.calculate_loss = types.MethodType(patched, agent)


def _patch_for_tracking(agent, algo_name):
    _init_tracking(agent)
    if algo_name == "DQN":
        _patch_dqn(agent)
    elif algo_name in ("AC", "A2C", "PPO"):
        _patch_on_policy(agent)
    elif algo_name in ("TD3", "SAC"):
        _patch_off_policy_ac(agent)


def _record_episode(agent, algo_name):
    """Called once per episode to record epsilon and entropy."""
    if algo_name in HAS_EPSILON and hasattr(agent, 'epsilon'):
        agent._epsilon_hist.append(agent.epsilon)

    if algo_name in HAS_ENTROPY:
        try:
            if hasattr(agent, 'log_std'):
                log_std  = agent.log_std.detach().cpu().numpy()
                # Gaussian differential entropy per dim
                ent = float(np.mean(0.5 + 0.5 * math.log(2 * math.pi * math.e) + log_std))
                agent._entropy_hist.append(ent)
            elif algo_name == "SAC" and hasattr(agent, 'log_alpha'):
                agent._entropy_hist.append(agent.alpha.item())
        except Exception:
            pass


# ==========================================================================
# TRAINING LOOP
# ==========================================================================

def train_agent(algo_name, env, agent, n_episodes, task_name, save_dir):
    _patch_for_tracking(agent, algo_name)

    reward_history = []
    length_history = []
    sum_reward     = 0.0

    for episode in tqdm(range(n_episodes), desc=f"  {algo_name}"):
        if algo_name == "MC_REINFORCE":
            r, _, steps = agent.learn(env, max_steps=MAX_STEPS)
        else:
            r, steps = agent.learn(env, max_steps=MAX_STEPS)

        reward_history.append(r)
        length_history.append(steps)
        sum_reward += r
        _record_episode(agent, algo_name)

        if (episode + 1) % 500 == 0:
            avg_r = sum_reward / 500
            avg_l = float(np.mean(length_history[-500:]))
            eps_s = f"  eps={agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(f"    ep={episode+1:5d}  avg_r={avg_r:8.2f}  avg_len={avg_l:6.1f}{eps_s}")
            sum_reward = 0.0
            _checkpoint(algo_name, agent, task_name, episode + 1)

    return reward_history, length_history


def _checkpoint(algo_name, agent, task_name, episode):
    path  = os.path.join("w", task_name, algo_name)
    fname = (f"{algo_name}_{episode}.json" if algo_name == "Linear_QN"
             else f"{algo_name}_{episode}.pt")
    agent.save_model(path, fname)


# ==========================================================================
# DEPLOYMENT EVALUATION
# ==========================================================================

def evaluate_agent(env, agent, algo_name, n_episodes, max_steps=MAX_STEPS):
    orig_eps = None
    if hasattr(agent, 'epsilon'):
        orig_eps      = agent.epsilon
        agent.epsilon = 0.0

    rewards, lengths, successes = [], [], []
    # Physics state recording — one representative episode per algorithm
    # obs layout: [cart_pos, pole_angle, cart_vel, pole_ang_vel]
    recorded_pole_angles  = []   # pole angle trace for episode 0
    recorded_cart_pos     = []   # cart position trace for episode 0
    recorded_actions      = []   # action trace for episode 0
    recording_done        = False

    for ep_i in tqdm(range(n_episodes), desc=f"    {algo_name}", leave=False):
        obs, _ = env.reset()
        if obs['policy'].shape[0] > 1:
            obs = {k: v[0:1] for k, v in obs.items()}

        total_r   = 0.0
        steps     = 0
        done      = False
        succeeded = False
        ep_pole   = []
        ep_cart   = []
        ep_act    = []

        while not done and steps < max_steps:
            with torch.no_grad():
                if algo_name in ("AC", "A2C", "PPO"):
                    action = agent.act_inference(obs)
                elif algo_name == "SAC":
                    action, _ = agent.select_action(obs, deterministic=True)
                elif algo_name == "TD3":
                    action, _ = agent.select_action(obs, noise=0.0)
                elif algo_name == "MC_REINFORCE":
                    state = obs['policy'].to(agent.device).float()
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    probs      = agent.policy_net(state)
                    action_idx = int(probs.argmax(dim=1).item())
                    action     = agent.scale_action(action_idx)
                else:
                    action, _ = agent.select_action(obs)

            # Record state before stepping
            state_np = obs['policy'].cpu().numpy().flatten()
            ep_cart.append(float(state_np[0]))          # cart position
            ep_pole.append(float(state_np[1]))          # pole angle (rad)
            ep_act.append(float(action.cpu().flatten()[0]))  # applied force

            next_obs, reward, terminated, truncated, _ = env.step(action)
            if next_obs['policy'].shape[0] > 1:
                next_obs = {k: v[0:1] for k, v in next_obs.items()}

            r     = reward[0].item()     if reward.dim()    > 0 else reward.item()
            term  = terminated[0].item() if terminated.dim() > 0 else terminated.item()
            trunc = truncated[0].item()  if truncated.dim() > 0 else truncated.item()

            total_r  += r
            steps    += 1
            done      = bool(term) or bool(trunc)
            obs       = next_obs

        # SUCCESS = survived to max_steps without falling
        # Using steps >= max_steps is more reliable than bool(trunc) because
        # Isaac Lab with --num_envs > 1 auto-resets envs and the truncated
        # signal for env[0] may already be cleared by the time we read it.
        succeeded = (steps >= max_steps) and not bool(term)
        rewards.append(total_r)
        lengths.append(steps)
        successes.append(1.0 if succeeded else 0.0)

        # Keep the longest episode as the representative trace
        # (most informative — shows what a good run looks like)
        if not recording_done or len(ep_pole) > len(recorded_pole_angles):
            recorded_pole_angles = ep_pole
            recorded_cart_pos    = ep_cart
            recorded_actions     = ep_act
            if len(ep_pole) >= max_steps:
                recording_done = True   # found a perfect episode, stop replacing

    if orig_eps is not None:
        agent.epsilon = orig_eps

    summary = {
        'avg_reward':   float(np.mean(rewards)),
        'avg_length':   float(np.mean(lengths)),
        'success_rate': float(np.mean(successes)),
    }
    detail = {
        'rewards':      rewards,
        'lengths':      lengths,
        'successes':    successes,
        'pole_angles':  recorded_pole_angles,   # radians over time
        'cart_pos':     recorded_cart_pos,       # metres over time
        'actions':      recorded_actions,        # force over time
    }
    return summary, detail


# ==========================================================================
# MAIN
# ==========================================================================

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    if args_cli.seed is None or args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = (args_cli.num_envs if args_cli.num_envs is not None
                              else env_cfg.scene.num_envs)
    env_cfg.seed       = agent_cfg["seed"]
    env_cfg.sim.device = (args_cli.device if args_cli.device is not None
                          else env_cfg.sim.device)

    task_name = str(args_cli.task).split('-')[0]
    env       = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    base_save      = os.path.join("plots", task_name, "Function_based")
    all_rewards    = {}
    all_lengths    = {}
    all_steps      = {}
    all_actor_l    = {}
    all_critic_l   = {}
    trained_agents = {}

    while simulation_app.is_running():

        # ── PHASE 1: TRAINING ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"TRAINING — {task_name}")
        print(f"{'='*60}")

        for algo_name in ALL_ALGOS:
            print(f"\n--- {algo_name} ---")
            agent    = _make_agent(algo_name, device)
            save_dir = os.path.join(base_save, algo_name)

            reward_hist, length_hist = train_agent(
                algo_name, env, agent, N_EPISODES, task_name, save_dir
            )

            all_rewards[algo_name]    = reward_hist
            all_lengths[algo_name]    = length_hist
            all_steps[algo_name]      = length_hist
            all_actor_l[algo_name]    = agent._actor_losses
            all_critic_l[algo_name]   = agent._critic_losses
            trained_agents[algo_name] = agent

            # ── per-algorithm graphs ──────────────────────────────────
            plot_learning_curve(reward_hist, length_hist, algo_name, save_dir)   # [A][B]
            plot_reward_std(reward_hist, algo_name, save_dir)                     # [C]

            if algo_name in HAS_ACTOR_LOSS | HAS_CRITIC_LOSS:
                plot_loss_curves(agent._actor_losses, agent._critic_losses,
                                 algo_name, save_dir)                             # [D][E]

            if algo_name in HAS_EPSILON and agent._epsilon_hist:
                plot_epsilon(agent._epsilon_hist, algo_name, save_dir)            # [F]

            if algo_name in HAS_ENTROPY and agent._entropy_hist:
                plot_entropy(agent._entropy_hist, algo_name, save_dir)            # [G]

            plot_steps_vs_reward(reward_hist, length_hist, algo_name, save_dir)  # [H]

            _checkpoint(algo_name, agent, task_name, N_EPISODES)

        # ── PHASE 2: COMPARISON ───────────────────────────────────────
        print(f"\n{'='*60}")
        print("COMPARISON PLOTS [I–M]")
        print(f"{'='*60}")
        plot_comparison(all_rewards, all_lengths, all_steps,
                        all_actor_l, all_critic_l,
                        save_dir=os.path.join(base_save, "comparisons"))
        # Extra comparison graphs
        plot_reward_variance_comparison(all_rewards,
                        save_dir=os.path.join(base_save, "comparisons"))  # [R2]
        plot_solved_episode(all_lengths,
                        save_dir=os.path.join(base_save, "comparisons"))  # [R3]

        # ── PHASE 3: DEPLOYMENT ───────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"DEPLOYMENT EVALUATION [N–R]  ({args_cli.deploy_episodes} eps each)")
        print(f"{'='*60}")

        deploy_summary = {}
        deploy_detail  = {}

        for algo_name, agent in trained_agents.items():
            print(f"  {algo_name}")
            with torch.no_grad():
                summary, detail = evaluate_agent(
                    env, agent, algo_name,
                    n_episodes=args_cli.deploy_episodes,
                )
            deploy_summary[algo_name] = summary
            deploy_detail[algo_name]  = detail
            print(f"    reward={summary['avg_reward']:.2f}  "
                  f"length={summary['avg_length']:.1f}  "
                  f"success={summary['success_rate']*100:.1f}%")

        dep_dir = os.path.join(base_save, "deployment")
        plot_deployment_bars(deploy_summary, dep_dir)               # [N][O][P]
        plot_deployment_histogram(deploy_detail, dep_dir)           # [Q]
        plot_deployment_per_episode(deploy_detail, dep_dir)         # [R]
        plot_pole_angle_stability_bars(deploy_detail, dep_dir)      # [R4]
        plot_pole_angle_traces(deploy_detail, dep_dir)              # [S-individual]
        plot_pole_angle_comparison(deploy_detail, dep_dir)          # [S-comparison]
        plot_cart_position_traces(deploy_detail, dep_dir)           # [T]
        plot_phase_portrait(deploy_detail, dep_dir)                 # [U]
        plot_action_traces(deploy_detail, dep_dir)                  # [V]

        print(f"\n✓ Done!  All graphs saved under: {base_save}/")
        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()