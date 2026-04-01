# Homework 3: Cart Pole Function Approximation Algorithms

**Authors**
- Chantouch Orungrote (66340500011)
- Sasish Kaewsing (66340500076)

## Objectives
Implement and evaluate eight function approximation-based RL algorithms — Linear Q-Network, DQN, MC REINFORCE, AC, A2C, PPO, TD3, and SAC — on the Cart-Pole Stabilization environment using Isaac Lab. Analyze how the choice of policy type (value-based vs. policy gradient vs. actor-critic), action space (discrete vs. continuous), and on/off-policy distinction affect learning performance, convergence stability, and deployment behavior.

---

### Part 1: Setup and Run

1. Activate conda environment and navigate to project folder:
   ```bash
   conda activate env_isaaclab
   cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
   ```

2. Run all experiments:
   ```bash
   python scripts/Function_based/train_all.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 1
   ```

**Optional Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_envs` | 1 | Number of parallel environments |
| `--max_iterations` | None | Override default episode count |
| `--deploy_episodes` | 100 | Episodes for deployment evaluation |
| `--seed` | random | Random seed for reproducibility |
| `--video` | False | Enable video recording |

**Alternative: Run Individual Scripts**
- Train a single algorithm:
  ```bash
  python scripts/Function_based/train.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 1
  ```
- Play/deploy a trained agent:
  ```bash
  python scripts/Function_based/play.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 1 --headless
  ```
- Random action baseline:
  ```bash
  python scripts/Function_based/random_action.py
  ```

---

### Part 2: Parameter Definition

**Shared Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 10,000 | Total training episodes |
| Max Steps | 500 | Maximum steps per episode |
| Observations | 4 | State dimension (x, θ, ẋ, θ̇) |
| Discount Factor (γ) | 0.99 | Future reward discounting |
| Learning Rate (α) | 3e-4 | Base optimizer learning rate |
| Action Range | [−10.0, 10.0] | Continuous force applied to cart (N) |

#### **Note: All algorithms use the same shared parameters above unless overridden per-algorithm below**

**Value-Based Methods**

| Parameter | Linear_QN | DQN |
|-----------|-----------|-----|
| `num_of_action` | 11 | 11 |
| `learning_rate` | 1e-3 | 3e-4 |
| `initial_epsilon` | 1.0 | 1.0 |
| `epsilon_decay` | 5e-4 | 5e-4 |
| `final_epsilon` | 0.01 | 0.01 |
| `hidden_dim` | — | 128 |
| `dropout` | — | 0.1 |
| `tau` | — | 0.005 |
| `buffer_size` | — | 50,000 |
| `batch_size` | — | 128 |
| `update_freq` | — | 4 |
| `target_update_freq` | — | 200 |

**Policy Gradient**

| Parameter | MC_REINFORCE |
|-----------|-------------|
| `num_of_action` | 11 |
| `hidden_dim` | 128 |
| `dropout` | 0.1 |

**On-Policy Actor-Critic**

| Parameter | AC | A2C | PPO |
|-----------|----|-----|-----|
| `num_of_action` | 1 | 1 | 1 |
| `hidden_dim` | 256 | 256 | 256 |
| `entropy_coef` | 0.01 | 0.01 | 0.01 |
| `gae_lambda` | — | 0.95 | 0.95 |
| `value_loss_coef` | — | 0.5 | 0.5 |
| `clip_param` | — | — | 0.2 |
| `num_transitions_per_env` | — | 64 | 64 |
| `num_learning_epochs` | — | — | 4 |
| `num_mini_batches` | — | — | 4 |

**Off-Policy Actor-Critic**

| Parameter | TD3 | SAC |
|-----------|-----|-----|
| `num_of_action` | 1 | 1 |
| `hidden_dim` | 256 | 256 |
| `tau` | 0.005 | 0.005 |
| `buffer_size` | 100,000 | 100,000 |
| `batch_size` | 256 | 256 |
| `update_freq` | 4 | 4 |
| `policy_noise` | 0.2 | — |
| `noise_clip` | 0.5 | — |
| `expl_noise` | 0.1 | — |
| `policy_delay` | 2 | — |
| `alpha` | — | 0.2 |
| `auto_entropy` | — | True |

---

### Part 3: Configuration

Algorithms are organized across three categories to evaluate the effect of approximation type and policy architecture.

**Algorithm Overview**

| Algorithm | Type | Policy | On/Off-Policy | Action Space |
|-----------|------|--------|---------------|--------------|
| Linear_QN | Value-based | Deterministic (ε-greedy) | Off-policy | Discrete (11) |
| DQN | Value-based | Deterministic (ε-greedy) | Off-policy | Discrete (11) |
| MC_REINFORCE | Policy-based | Stochastic | On-policy | Discrete (11) |
| AC | Actor-Critic | Stochastic | On-policy | Continuous |
| A2C | Actor-Critic | Stochastic | On-policy | Continuous |
| PPO | Actor-Critic | Stochastic | On-policy | Continuous |
| TD3 | Actor-Critic | Deterministic | Off-policy | Continuous |
| SAC | Actor-Critic | Stochastic | Off-policy | Continuous |

**Deployment Inference Mode**

| Algorithm | Inference Mode |
|-----------|---------------|
| Linear_QN, DQN | ε = 0 (greedy argmax Q) |
| MC_REINFORCE | Argmax over softmax probabilities |
| AC, A2C, PPO | `act_inference()` (deterministic mean action) |
| TD3 | `select_action(noise=0.0)` |
| SAC | `select_action(deterministic=True)` |

---

### Part 4: Results Location

All outputs are saved in the `plots/` and `w/` folders, organized by algorithm:

```
plots/Function_based/
├── Linear_QN/
│   ├── learning_curve.png
│   ├── episode_length_curve.png
│   ├── reward_with_std.png
│   ├── actor_loss_curve.png        # AC, A2C, PPO, TD3, SAC only
│   ├── critic_loss_curve.png       # AC, A2C, PPO, TD3, SAC only
│   ├── epsilon_curve.png           # Linear_QN, DQN only
│   ├── entropy_curve.png           # AC, A2C, PPO, SAC only
│   └── steps_vs_reward.png
├── DQN/                            # Same structure as Linear_QN
├── MC_REINFORCE/
├── AC/
├── A2C/
├── PPO/
├── TD3/
├── SAC/
├── comparisons/
│   ├── comparison_reward.png
│   ├── comparison_ep_length.png
│   ├── comparison_steps_reward.png
│   ├── comparison_actor_loss.png
│   ├── comparison_critic_loss.png
│   ├── comparison_reward_variance.png
│   └── comparison_solved_episode.png
└── deployment/
    ├── deployment_reward.png
    ├── deployment_ep_length.png
    ├── deployment_success_rate.png
    ├── deployment_length_hist.png
    ├── deployment_reward_per_ep.png
    ├── deployment_pole_mean_angle.png
    ├── deployment_pole_std_angle.png
    ├── deployment_pole_angle_traces.png
    ├── deployment_pole_angle_comparison.png
    ├── deployment_cart_position.png
    ├── deployment_phase_portrait.png
    └── deployment_action_traces.png

w/Stabilize/
├── Linear_QN/
│   └── Linear_QN_{episode}.json
├── DQN/
│   └── DQN_{episode}.pt
├── MC_REINFORCE/
│   └── MC_REINFORCE_{episode}.pt
├── AC/
│   └── AC_{episode}.pt
├── A2C/
│   └── A2C_{episode}.pt
├── PPO/
│   └── PPO_{episode}.pt
├── TD3/
│   └── TD3_{episode}.pt
└── SAC/
    └── SAC_{episode}.pt
```

**Metrics per algorithm:**
- `learning_curve.png` — Cumulative reward smoothed over 100-episode window
- `episode_length_curve.png` — Steps survived per episode
- `reward_with_std.png` — Smoothed reward ± 1 std band
- `steps_vs_reward.png` — Reward vs. total environment steps (sample efficiency)
- `actor_loss_curve.png` / `critic_loss_curve.png` — Loss over update steps (neural algos only)
- `epsilon_curve.png` — Epsilon decay schedule (value-based algos only)
- `entropy_curve.png` — Policy entropy over training (stochastic algos only)

---

### Part 5: Structure

```
.
├── RL_Algorithm/
│   ├── Function_based/
│   │   ├── Linear_Q.py                # Linear function approximation Q-Learning
│   │   ├── DQN.py                     # Deep Q-Network with experience replay
│   │   ├── MC_REINFORCE.py            # REINFORCE policy gradient
│   │   ├── AC.py                      # Actor-Critic (MC episodic)
│   │   ├── A2C.py                     # Advantage Actor-Critic (TD + GAE)
│   │   ├── PPO.py                     # Proximal Policy Optimization
│   │   ├── TD3.py                     # Twin Delayed DDPG
│   │   └── SAC.py                     # Soft Actor-Critic
│   ├── storage/
│   │   ├── buffers.py                 # RolloutBuffer (on-policy) and ReplayBuffer (off-policy)
│   │   ├── on_policy.py               # OnPolicyAlgorithm base for AC, A2C, PPO
│   │   └── off_policy.py              # OffPolicyAlgorithm base for DQN, TD3, SAC
│   ├── networks/
│   │   └── mlp.py                     # Shared MLP backbone (relu / elu / tanh)
│   ├── RL_base_function.py            # Base class for function approximation agents
│   └── Algorithm/                     # Tabular methods from HW2 (MC, SARSA, Q-Learning, etc.)
├── scripts/
│   ├── Function_based/
│   │   ├── train_all.py               # Train all 8 algorithms + generate all plots
│   │   ├── train.py                   # Train a single algorithm
│   │   ├── play.py                    # Deploy trained agent (deterministic inference)
│   │   └── random_action.py           # Random action baseline
│   └── RL_Algorithm/
│       ├── train_all.py               # HW2 tabular training
│       ├── train.py
│       ├── play.py
│       └── random_action.py
├── source/
│   └── CartPole/                      # CartPole environment source (Isaac Lab)
├── plots/                             # Generated learning curves and comparison plots
├── w/                                 # Saved model checkpoints (.pt / .json)
└── docker/                            # Docker environment setup
    ├── Dockerfile
    └── docker-compose.yaml
```
