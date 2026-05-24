# Final Project: Splendor DDQN Agent

**Authors**
- Chantouch Orungrote (66340500011)
- Sasish Kaewsing (66340500076)

## Objectives
Implement and evaluate a Dueling Double DQN (DDQN) agent with Prioritized Experience Replay (PER) and self-play on the Splendor board game environment. Seven experiments (C0–C6) are conducted to analyze how reward shaping, opponent type, and architectural enhancements affect learning performance, win rate convergence, and final agent strength against heuristic opponents.

---

### Part 1: Setup and Run

1. Open terminal in the project folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run all experiments:
   ```bash
   python main.py train --experiment all --episodes 300000 --num-envs 4
   ```

**Optional Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | `C3` | Experiment to run: `C0`–`C6` or `all` |
| `--episodes` | 300,000 | Total training episodes |
| `--eval-interval` | 1,000 | Episodes between evaluations |
| `--eval-games` | 100 | Games per evaluation checkpoint |
| `--lr` | 1e-3 | Optimizer learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--batch-size` | 64 | Mini-batch size for training |
| `--buffer-size` | 100,000 | Replay buffer capacity |
| `--tau` | 0.005 | Soft target update coefficient |
| `--save-dir` | `checkpoints` | Directory for saved model weights |
| `--log-dir` | `logs` | Directory for training logs |
| `--parallel` | False | Run all experiments simultaneously |
| `--num-envs` | 1 | Number of parallel environments per experiment (4–8 recommended on GPU) |

**Alternative: Run Individual Commands**
- Train a single experiment:
  ```bash
  python main.py train --experiment C3 --episodes 300000 --num-envs 4
  ```
- Run all experiments in parallel:
  ```bash
  python main.py train --experiment all --parallel --num-envs 4
  ```
- Generate evaluation plots:
  ```bash
  python main.py eval --log-dir logs --plot-dir plots
  ```
- Generate sequence analysis plots (ethogram, transitions, phase flow):
  ```bash
  python plot_sequences.py --log-dir logs --plot-dir plots
  ```
- Generate extra analysis plots (stability, volatility, ablation):
  ```bash
  python analyze_extra.py --log-dir logs --plot-dir plots --skip-checkpoint
  ```
- Watch the trained agent play:
  ```bash
  python main.py play --mode ai_vs_ai --agent checkpoints/C3_best.pt
  ```
- Play against the trained agent:
  ```bash
  python main.py play --mode ai_vs_human --agent checkpoints/C3_best.pt
  ```
- Train with human demonstrations:
  ```bash
  python human_train.py
  ```

---

### Part 2: Parameter Definition

**Shared Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 300,000 | Total training episodes |
| Max Steps | 200 | Maximum steps per episode |
| State Dimension | 213 | Normalized observation vector size |
| Action Dimension | 50 | 15 gem + 15 buy + 15 reserve + 5 discard |
| Discount Factor (γ) | 0.99 | Future reward discounting |
| Learning Rate (α) | 1e-3 | Adam optimizer learning rate |
| Batch Size | 64 | Transitions sampled per update |
| Buffer Size | 100,000 | Replay buffer capacity |
| ε Start | 1.0 | Initial exploration rate |
| ε End | 0.05 | Minimum exploration rate |
| ε Decay | 0.99997 | Per-episode multiplicative decay |
| Target Update Freq | 2,000 | Steps between soft target updates |
| Soft Update (τ) | 0.005 | Soft target interpolation coefficient |
| Eval Interval | 1,000 | Episodes between win-rate evaluations |
| Eval Games | 100 | Games per evaluation checkpoint |
| Min Buffer Size | 5,000 | Transitions before training starts |
| LR Decay | ÷10 | Applied at episode 200k and 250k |

**PER Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (priority exponent) | 0.6 | Controls priority sharpness |
| β Start | 0.4 | Initial importance-sampling correction |
| β End | 1.0 | Final importance-sampling correction |
| β Anneal Steps | 200,000 | Steps to anneal β from start to end |
| Priority Cap | 100.0 | Maximum allowed priority value |

**Network Architecture**

| Component | Layer | Size |
|-----------|-------|------|
| Shared | Input | 213 |
| Shared | Hidden 1 | 256 (LayerNorm + ReLU) |
| Shared | Hidden 2 | 256 (LayerNorm + ReLU) |
| Shared | Hidden 3 | 128 (LayerNorm + ReLU) |
| Value stream | V(s) | 128 → 1 |
| Advantage stream | A(s,a) | 128 → 50 |
| Output | Q(s,a) | V(s) + A(s,a) − mean(A) |

#### **Note: Dueling architecture is used only in C3. All other experiments use a standard MLP Q-network with the same hidden sizes.**

---

### Part 3: Configuration

**Experiment Overview**

| Experiment | Reward Function | Opponent | Dueling | PER | Purpose |
|------------|----------------|----------|---------|-----|---------|
| C0 | Score-only | Random | ✗ | ✓ | Lower bound — weakest opponent |
| C1 | Score-only | Greedy | ✗ | ✓ | Baseline — prestige reward only |
| C2 | Event-shaped | Greedy | ✗ | ✓ | Proposed — dense intermediate rewards |
| C3 | Event-shaped | Self-play | ✓ | ✓ | Full agent — all enhancements |
| C4 | Card Rush | Greedy | ✗ | ✓ | Ablation — reduced noble reward |
| C5 | Noble Hunter | Greedy | ✗ | ✓ | Ablation — doubled noble reward |
| C6 | Balanced Dense | Greedy | ✗ | ✓ | Ablation — all intermediate rewards amplified |

**Reward Weights per Experiment**

| Event | C2 (Baseline) | C4 (Card Rush) | C5 (Noble Hunter) | C6 (Balanced Dense) |
|-------|:---:|:---:|:---:|:---:|
| Gem taken | 0.002 | 0.002 | 0.002 | 0.010 |
| Card bought L1 | 0.05 | 0.15 | 0.05 | 0.10 |
| Card bought L2 | 0.10 | 0.25 | 0.10 | 0.20 |
| Card bought L3 | 0.20 | 0.40 | 0.20 | 0.40 |
| Prestige gained | 1.00 | 1.00 | 1.00 | 1.00 |
| Noble gained | 3.00 | 0.50 | 6.00 | 4.00 |
| Card reserved | 0.01 | 0.01 | 0.01 | 0.02 |
| Win bonus | 10.00 | 10.00 | 10.00 | 10.00 |
| Loss penalty | −10.00 | −10.00 | −10.00 | −10.00 |

**Opponents**

| Opponent | Strategy | Used In |
|----------|----------|---------|
| Random | Uniformly samples from all legal actions | C0 |
| Greedy | Buy highest-prestige card → take gems toward cheapest card → reserve | C1, C2, C4, C5, C6 |
| Self-play | Pool of 10 past agent snapshots, random selection per episode, greedy fallback | C3 |

---

### Part 4: Results Location

All outputs are saved in the `plots/`, `checkpoints/`, and `logs/` folders:

```
plots/
├── training_rewards.png         # Episode reward over training
├── training_dashboard.png       # Multi-panel training summary
├── win_rates.png                # Win rate at each eval checkpoint (all experiments)
├── win_rate_trend.png           # Smoothed win rate trend across experiments
├── win_rate_stability.png       # Win rate stability analysis (best checkpoint)
├── loss_curves.png              # TD loss over training steps
├── episode_lengths.png          # Steps per episode over training
├── epsilon_schedule.png         # ε decay schedule
├── sample_efficiency.png        # Win rate vs. total environment steps
├── final_performance.png        # Final win rate comparison across C0–C6
├── final_score_distribution.png # Score distribution at end of training
├── score_progression.png        # Score progression over episodes
├── reward_distribution.png      # Distribution of episode rewards
├── action_distribution.png      # Frequency of each action type
├── action_heatmap.png           # Action selection heatmap across training
├── card_level_purchases.png     # L1/L2/L3 card purchase distribution
├── ethogram.png                 # Action-type sequence visualization
├── transitions.png              # State transition analysis
├── transition_heatmaps.png      # Heatmap of action-to-action transitions
├── phase_flow.png               # Phase-space trajectory of agent behavior
├── ablation_ladder.png          # Ranked ablation comparison (C0–C6)
├── training_volatility.png      # Episode-to-episode reward volatility
├── best_vs_final.png            # Best checkpoint vs final checkpoint comparison
└── tournament_matrix.png        # Head-to-head win rates between all agents

checkpoints/
├── C0_best.pt / C0_final.pt    # Best and final checkpoints for C0
├── C1_best.pt / C1_final.pt    # Best and final checkpoints for C1
├── C2_best.pt / C2_final.pt    # Best and final checkpoints for C2
├── C3_best.pt / C3_final.pt    # Best and final checkpoints for C3
├── C4_best.pt / C4_final.pt    # Best and final checkpoints for C4
├── C5_best.pt / C5_final.pt    # Best and final checkpoints for C5
├── C6_best.pt / C6_final.pt    # Best and final checkpoints for C6
└── human_trained.pt             # Checkpoint trained with human demonstrations

logs/
├── C0_compact.json              # Eval win rates, best win rate, episode snapshots
├── C0_metrics.json              # Full episode-by-episode metrics
├── C1_compact.json / C1_metrics.json
├── C2_compact.json / C2_metrics.json
├── C3_compact.json / C3_metrics.json
├── C4_compact.json / C4_metrics.json
├── C5_compact.json / C5_metrics.json
└── C6_compact.json / C6_metrics.json
```

**Metrics per experiment:**
- `training_rewards.png` — Raw and smoothed episode rewards over training
- `win_rates.png` — Win rate against greedy opponent at every 1,000-episode checkpoint
- `loss_curves.png` — TD loss smoothed over training updates
- `episode_lengths.png` — Steps survived per episode (proxy for game quality)
- `sample_efficiency.png` — Win rate relative to total environment steps
- `ethogram.png` / `transitions.png` — Behavioral sequence analysis from `plot_sequences.py`
- `ablation_ladder.png` / `tournament_matrix.png` — Cross-experiment analysis from `analyze_extra.py`

---

### Part 5: Structure

```
DRL_Project/
├── main.py                    # Entry point: train / eval / play subcommands
├── agent.py                   # DDQNAgent: DuelingQNetwork, PER buffer, training loop
├── train.py                   # Experiment configs and training loop (C0–C6)
├── evaluate.py                # Plot generation from saved logs
├── analyze_extra.py           # Extra analysis: ablation ladder, tournament matrix, volatility
├── plot_sequences.py          # Behavioral sequence plots: ethogram, transitions, phase flow
├── human_train.py             # Training loop with human demonstration data
├── splendor_env.py            # Splendor Gymnasium environment (2-player, 213-dim obs)
├── opponents.py               # random_opponent, greedy_opponent, SelfPlayOpponent
├── reward_shaping.py          # score_only, event_shaped, card_rush, noble_hunter, balanced_dense
├── card_data.py               # Card definitions, gem color constants
├── visualize.py               # Pygame visualizer (ai_vs_ai / ai_vs_human)
├── splendor_rules.txt         # Plain-text Splendor rules reference
├── requirements.txt           # Python dependencies
├── Report.tex                 # LaTeX report source
├── checkpoints/               # Saved model weights (.pt)
│   ├── C0_best.pt – C6_best.pt
│   ├── C0_final.pt – C6_final.pt
│   └── human_trained.pt
├── logs/                      # Training logs (JSON)
│   ├── C0_compact.json – C6_compact.json
│   └── C0_metrics.json – C6_metrics.json
└── plots/                     # Generated evaluation and analysis plots (PNG)
```
