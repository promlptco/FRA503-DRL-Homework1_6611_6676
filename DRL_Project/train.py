"""
Training loop for Splendor DDQN -- DRL_Project_extra (enhanced, GPU-ready).

Experiments:
  C0 -- Score-only reward    vs  Random opponent   (lower bound)
  C1 -- Score-only reward    vs  Greedy opponent   (baseline)
  C2 -- Event-shaped reward  vs  Greedy opponent   (proposed method)
  C3 -- Event-shaped + Self-play + Dueling + PER   (best agent)

GPU tips:
  Run with --num-envs 4 or 8 to fill GPU memory with parallel environments.
  torch.compile is activated via DDQNAgent(use_compile=True) -- set in config below.
  AMP is active automatically when CUDA is available.

Usage:
  python main.py train --experiment C3 --episodes 300000 --num-envs 4
  python main.py train --experiment all --parallel --num-envs 4
"""

import json
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

from agent import DDQNAgent
from opponents import SelfPlayOpponent, greedy_opponent, random_opponent
from reward_shaping import (event_shaped_reward, score_only_reward,
                            card_rush_reward, noble_hunter_reward,
                            balanced_dense_reward)
from splendor_env import SplendorEnv

SNAPSHOT_INTERVAL = 2_000   # how often to push weights into self-play pool


# --- Experiment Configs ----------------------------------------------
def _make_configs():
    _sp = SelfPlayOpponent(fallback_opponent=greedy_opponent)
    return {
        "C0": {
            "reward_fn":   score_only_reward,
            "opponent":    random_opponent,
            "desc":        "Score-only vs Random (lower bound)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
        "C1": {
            "reward_fn":   score_only_reward,
            "opponent":    greedy_opponent,
            "desc":        "Score-only vs Greedy (baseline)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
        "C2": {
            "reward_fn":   event_shaped_reward,
            "opponent":    greedy_opponent,
            "desc":        "Event-shaped vs Greedy (proposed)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
        "C3": {
            "reward_fn":   event_shaped_reward,
            "opponent":    _sp,
            "desc":        "Event-shaped + Self-play + Dueling + PER",
            "use_dueling": True,
            "use_per":     True,
            "use_compile": True,   # torch.compile active for C3 (heaviest compute)
        },
        "C4": {
            "reward_fn":   card_rush_reward,
            "opponent":    greedy_opponent,
            "desc":        "Card Rush vs Greedy (no gem reward, high card reward)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
        "C5": {
            "reward_fn":   noble_hunter_reward,
            "opponent":    greedy_opponent,
            "desc":        "Noble Hunter vs Greedy (2x noble reward, L3 emphasis)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
        "C6": {
            "reward_fn":   balanced_dense_reward,
            "opponent":    greedy_opponent,
            "desc":        "Balanced Dense vs Greedy (all intermediate rewards amplified)",
            "use_dueling": False,
            "use_per":     True,
            "use_compile": False,
        },
    }


def train(
    experiment="C1",
    num_episodes=300_000,
    max_steps_per_episode=200,
    eval_interval=1_000,
    eval_games=100,
    save_dir="checkpoints",
    log_dir="logs",
    # Agent hyperparameters
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.99997,   # per-episode
    buffer_size=100_000,
    batch_size=64,
    target_update_freq=2_000,
    tau=0.005,
    # Vectorized environments
    num_envs=1,
    # Parallel display offset
    tqdm_position=0,
):
    configs = _make_configs()
    config  = configs[experiment]
    print(f"\n{'='*65}")
    print(f"  Experiment {experiment}: {config['desc']}")
    print(f"  Episodes: {num_episodes}  |  LR: {lr}  |  gamma: {gamma}")
    print(f"  Dueling: {config['use_dueling']}  PER: {config['use_per']}  "
          f"compile: {config['use_compile']}")
    print(f"{'='*65}\n")

    agent = DDQNAgent(
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        tau=tau,
        use_dueling=config["use_dueling"],
        use_per=config["use_per"],
        use_layernorm=True,
        use_compile=config["use_compile"],
        use_amp=True,          # AMP auto-disabled on CPU by DDQNAgent
        min_buffer_size=5_000,
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses":          [],
        "epsilons":        [],
        "eval_win_rates":  [],
        "eval_episodes":   [],
    }

    total_steps         = 0
    best_win_rate       = 0.0
    first_50pct_episode = None
    episode_count       = 0
    opponent            = config["opponent"]
    is_self_play        = isinstance(opponent, SelfPlayOpponent)

    def _make_env():
        return SplendorEnv(opponent_policy=opponent, reward_fn=config["reward_fn"])

    def _maybe_evaluate(ep):
        nonlocal best_win_rate, first_50pct_episode
        if ep % eval_interval != 0:
            return None
        win_rate = evaluate_agent(agent, greedy_opponent, eval_games)
        metrics["eval_win_rates"].append(win_rate)
        metrics["eval_episodes"].append(ep)
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            agent.save(os.path.join(save_dir, f"{experiment}_best.pt"))
        if first_50pct_episode is None and win_rate >= 0.5:
            first_50pct_episode = ep
        return win_rate

    def _maybe_snapshot(ep):
        if is_self_play and ep % SNAPSHOT_INTERVAL == 0:
            sd = getattr(agent.online_net, "_orig_mod", agent.online_net).state_dict()
            opponent.add_snapshot(sd, use_dueling=agent.use_dueling)

    # --- Single-env loop ---------------------------------------------
    if num_envs == 1:
        env  = _make_env()
        pbar = tqdm(range(1, num_episodes + 1),
                    desc=f"{experiment}", position=tqdm_position, leave=True)

        for episode in pbar:
            if is_self_play:
                opponent.select_episode_net()

            obs, info  = env.reset()
            legal_mask = info["legal_mask"]
            ep_reward  = 0.0
            ep_losses  = []

            for step in range(max_steps_per_episode):
                action                          = agent.select_action(obs, legal_mask)
                next_obs, reward, done, _, info = env.step(action)
                next_mask                       = info["legal_mask"]

                agent.store(obs, action, reward, next_obs, done, next_mask)
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

                ep_reward   += reward
                total_steps += 1
                obs, legal_mask = next_obs, next_mask
                if done:
                    break

            metrics["episode_rewards"].append(ep_reward)
            metrics["episode_lengths"].append(step + 1)
            metrics["epsilons"].append(agent.epsilon)
            metrics["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)

            agent.decay_epsilon()

            # LR step decay at ep 200k and 250k
            if episode in (200_000, 250_000):
                for pg in agent.optimizer.param_groups:
                    pg["lr"] /= 10
                print(f"\n  [LR decay] ep={episode}, new lr={pg['lr']:.2e}")

            _maybe_snapshot(episode)
            win_rate = _maybe_evaluate(episode)
            if win_rate is not None:
                pbar.set_postfix({
                    "rew":  f"{np.mean(metrics['episode_rewards'][-100:]):.2f}",
                    "win%": f"{win_rate*100:.1f}",
                    "eps":  f"{agent.epsilon:.3f}",
                    "best": f"{best_win_rate*100:.1f}%",
                })

    # --- Vectorized multi-env loop -----------------------------------
    else:
        envs = [_make_env() for _ in range(num_envs)]

        obs_all    = np.zeros((num_envs, agent.online_net.net[0].in_features
                               if not agent.use_dueling
                               else agent.online_net.feature[0].in_features),
                              dtype=np.float32)
        mask_all   = np.zeros((num_envs, agent.action_dim), dtype=bool)
        ep_rewards = np.zeros(num_envs, dtype=np.float32)
        ep_steps   = np.zeros(num_envs,  dtype=np.int32)
        ep_losses  = [[] for _ in range(num_envs)]

        for i, env in enumerate(envs):
            if is_self_play:
                opponent.select_episode_net()
            obs, info   = env.reset()
            obs_all[i]  = obs
            mask_all[i] = info["legal_mask"]

        pbar = tqdm(total=num_episodes,
                    desc=f"{experiment} (x{num_envs})",
                    position=tqdm_position, leave=True)

        while episode_count < num_episodes:
            actions = agent.select_action_batch(obs_all, mask_all)

            for i, env in enumerate(envs):
                if episode_count >= num_episodes:
                    break

                next_obs, reward, done, _, info = env.step(int(actions[i]))
                next_mask = info["legal_mask"]

                agent.store(obs_all[i], int(actions[i]), reward, next_obs, done, next_mask)
                loss = agent.train_step()
                if loss is not None:
                    ep_losses[i].append(loss)

                ep_rewards[i] += reward
                ep_steps[i]   += 1
                total_steps   += 1

                if done or ep_steps[i] >= max_steps_per_episode:
                    episode_count += 1
                    metrics["episode_rewards"].append(float(ep_rewards[i]))
                    metrics["episode_lengths"].append(int(ep_steps[i]))
                    metrics["epsilons"].append(agent.epsilon)
                    metrics["losses"].append(
                        float(np.mean(ep_losses[i])) if ep_losses[i] else 0.0
                    )
                    agent.decay_epsilon()

                    if episode_count in (200_000, 250_000):
                        for pg in agent.optimizer.param_groups:
                            pg["lr"] /= 10

                    _maybe_snapshot(episode_count)
                    win_rate = _maybe_evaluate(episode_count)
                    if win_rate is not None:
                        pbar.set_postfix({
                            "rew":  f"{np.mean(metrics['episode_rewards'][-100:]):.2f}",
                            "win%": f"{win_rate*100:.1f}",
                            "eps":  f"{agent.epsilon:.3f}",
                            "best": f"{best_win_rate*100:.1f}%",
                        })

                    ep_rewards[i] = 0.0
                    ep_steps[i]   = 0
                    ep_losses[i]  = []
                    pbar.update(1)

                    if is_self_play:
                        opponent.select_episode_net()
                    obs, info   = env.reset()
                    obs_all[i]  = obs
                    mask_all[i] = info["legal_mask"]
                else:
                    obs_all[i]  = next_obs
                    mask_all[i] = next_mask

        pbar.close()

    # --- Save --------------------------------------------------------
    agent.save(os.path.join(save_dir, f"{experiment}_final.pt"))
    metrics["first_50pct_episode"] = first_50pct_episode
    metrics["best_win_rate"]       = best_win_rate
    metrics["total_steps"]         = total_steps

    compact = {
        "experiment":          experiment,
        "eval_win_rates":      metrics["eval_win_rates"],
        "eval_episodes":       metrics["eval_episodes"],
        "best_win_rate":       best_win_rate,
        "first_50pct_episode": first_50pct_episode,
        "total_steps":         total_steps,
        "reward_snapshots": [
            float(np.mean(metrics["episode_rewards"][max(0, ep - 100): ep]))
            for ep in metrics["eval_episodes"]
        ],
        "loss_snapshots": [
            float(np.mean(metrics["losses"][max(0, ep - 100): ep]))
            for ep in metrics["eval_episodes"]
        ],
        "epsilon_snapshots": [
            float(metrics["epsilons"][min(ep - 1, len(metrics["epsilons"]) - 1)])
            for ep in metrics["eval_episodes"]
        ],
        "length_snapshots": [
            float(np.mean(metrics["episode_lengths"][max(0, ep - 100): ep]))
            for ep in metrics["eval_episodes"]
        ],
    }
    with open(os.path.join(log_dir, f"{experiment}_compact.json"), "w") as f:
        json.dump(compact, f, indent=2)
    with open(os.path.join(log_dir, f"{experiment}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {experiment} done -- best: {best_win_rate*100:.1f}%  "
          f"steps: {total_steps:,}\n")
    return metrics


def evaluate_agent(agent, opponent_fn, num_games=100):
    env           = SplendorEnv(opponent_policy=opponent_fn, reward_fn=None)
    wins          = 0
    old_eps       = agent.epsilon
    agent.epsilon = 0.0
    for _ in range(num_games):
        obs, info  = env.reset()
        legal_mask = info["legal_mask"]
        for _ in range(200):
            action = agent.select_action(obs, legal_mask)
            obs, _, terminated, _, info = env.step(action)
            legal_mask = info["legal_mask"]
            if terminated:
                break
        if env.winner == 0:
            wins += 1
    agent.epsilon = old_eps
    return wins / num_games


# --- Parallel Training -----------------------------------------------
def _train_worker(kwargs):
    train(**kwargs)


def train_parallel(experiments=("C0", "C1", "C2", "C3"), **shared_kwargs):
    print(f"\n  Parallel training: {list(experiments)}")
    jobs = []
    for i, exp in enumerate(experiments):
        p = multiprocessing.Process(
            target=_train_worker,
            args=(dict(experiment=exp, tqdm_position=i, **shared_kwargs),),
            name=exp,
        )
        p.start()
        jobs.append(p)
    for p in jobs:
        p.join()
    print(f"\n  All experiments finished.")
