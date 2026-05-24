"""
Extra Result Analysis for Splendor DRL Experiments
Generates 5 additional plots:
  1. win_rate_stability.png      -- mean ± std win rate in final 50k episodes
  2. training_volatility.png     -- rolling std of episode rewards (training noise)
  3. ablation_ladder.png         -- marginal contribution of each ingredient
  4. best_vs_final.png           -- best checkpoint vs final checkpoint degradation
  5. tournament_matrix.png       -- round-robin win matrix between all agents

Usage:
  cd DRL_Project_extrav3
  python analyze_extra.py --log-dir logs --plot-dir plots --checkpoint-dir checkpoints
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      180,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
})

EXP_STYLE = {
    "C0": {"color": "#27ae60", "label": "C0: Score-only vs Random"},
    "C1": {"color": "#2980b9", "label": "C1: Score-only vs Greedy"},
    "C2": {"color": "#e74c3c", "label": "C2: Event-shaped vs Greedy"},
    "C3": {"color": "#8e44ad", "label": "C3: Event-shaped + Self-play + Dueling + PER"},
    "C4": {"color": "#f39c12", "label": "C4: Card Rush vs Greedy"},
    "C5": {"color": "#16a085", "label": "C5: Noble Hunter vs Greedy"},
    "C6": {"color": "#2c3e50", "label": "C6: Balanced Dense vs Greedy"},
}

EXPS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


# =============================================================================
# Data Loading
# =============================================================================

def load_all(log_dir):
    data = {}
    for exp in EXPS:
        path = os.path.join(log_dir, f"{exp}_metrics.json")
        if not os.path.exists(path):
            path = os.path.join(log_dir, f"{exp}_compact.json")
        if not os.path.exists(path):
            print(f"  [skip] {exp}: no log found")
            continue
        with open(path) as f:
            d = json.load(f)
        data[exp] = d
        print(f"  [loaded] {exp}")
    return data


def _save(fig, plot_dir, filename):
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 1. Win Rate Stability (mean ± std in final 50k episodes)
# =============================================================================

def plot_win_rate_stability(all_data, plot_dir):
    """
    Bars = mean ± std of eval win rates in the last 50k episodes.
    Star marker = best (peak) win rate achieved at any point during training.
    Shows which agent is most *reliable* vs just best-case.
    """
    exps, means, stds, bests, colors = [], [], [], [], []

    for exp, data in all_data.items():
        wr = np.array(data["eval_win_rates"]) * 100
        if len(wr) == 0:
            continue
        tail = wr[-50:] if len(wr) >= 50 else wr
        best = data.get("best_win_rate", max(data["eval_win_rates"]) if data["eval_win_rates"] else 0) * 100
        exps.append(exp)
        means.append(np.mean(tail))
        stds.append(np.std(tail))
        bests.append(best)
        colors.append(EXP_STYLE[exp]["color"])

    x = np.arange(len(exps))
    fig, ax = plt.subplots(figsize=(max(8, len(exps) * 1.3), 6.0))

    bars = ax.bar(x, means, color=colors, edgecolor="black", lw=0.6,
                  width=0.55, alpha=0.85, zorder=3, label="Final 50k mean")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                elinewidth=2.0, capsize=7, capthick=2.0, zorder=4)

    # Best win rate as star markers
    ax.scatter(x, bests, marker="*", s=220, color="gold", edgecolors="black",
               linewidths=0.8, zorder=5, label="Best (peak) win rate")

    ax.axhline(50, color="gray", ls="--", lw=1.2, alpha=0.6, label="50% baseline")

    for i, (bar, m, s, b) in enumerate(zip(bars, means, stds, bests)):
        # Label for mean ± std (below the error bar top)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 1.2,
                f"{m:.1f}%\n±{s:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        # Label for best (above the star)
        ax.text(x[i], b + 1.5, f"{b:.1f}%",
                ha="center", va="bottom", fontsize=8,
                color="darkorange", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(exps)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Win Rate vs Greedy (%)")
    ax.set_title("Win Rate Stability — Final 50k Mean ± Std  vs  Best (Peak)")
    ax.set_ylim(0, max(bests) + 14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.18, axis="y", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, plot_dir, "win_rate_stability.png")


# =============================================================================
# 2. Training Volatility (rolling std of episode rewards)
# =============================================================================

def plot_training_volatility(all_data, plot_dir):
    """
    Rolling standard deviation of episode rewards (window=2000).
    High volatility = unstable training / difficult credit assignment.
    C3 (self-play) should be most volatile due to non-stationary opponent.
    """
    WINDOW = 2000

    fig, ax = plt.subplots(figsize=(11, 5.2))

    for exp, data in all_data.items():
        rv = np.array(data["episode_rewards"], dtype=float)
        if len(rv) < WINDOW:
            continue
        s = EXP_STYLE[exp]
        # Compute rolling std
        vol = np.array([
            rv[max(0, i - WINDOW):i].std()
            for i in range(WINDOW, len(rv), 100)  # sample every 100 eps for speed
        ])
        x = np.arange(WINDOW, len(rv), 100)
        ax.plot(x, vol, color=s["color"], label=s["label"], lw=1.6, alpha=0.88)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling Std of Reward (window={WINDOW})")
    ax.set_title("Training Volatility — Rolling Reward Standard Deviation")
    ax.grid(True, alpha=0.18)
    ax.margins(x=0.01)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, plot_dir, "training_volatility.png")


# =============================================================================
# 3. Reward Shaping Ablation Ladder
# =============================================================================

def plot_ablation_ladder(all_data, plot_dir):
    """
    Shows the marginal contribution of each ingredient added on top of the baseline.
    Ladder: C1 (baseline) → C2 (+event shaping) → C3 (+self-play+dueling+PER)
            and C4 (card rush shaping, same arch as C2).
    Metrics: best win rate + episodes to 50% win rate.
    """
    # Define the ablation chain with descriptions
    ablation = [
        ("C0", "Score-only\nvs Random\n(weakest baseline)"),
        ("C1", "Score-only\nvs Greedy\n(+stronger opponent)"),
        ("C2", "Event-shaped\nvs Greedy\n(+reward shaping)"),
        ("C3", "Event-shaped\n+Self-play\n+Dueling+PER"),
        ("C4", "Card Rush\nvs Greedy\n(alt. shaping)"),
        ("C5", "Noble Hunter\nvs Greedy\n(narrow shaping)"),
        ("C6", "Balanced Dense\nvs Greedy\n(dense shaping)"),
    ]

    exps_avail = [e for e, _ in ablation if e in all_data]
    labels     = [lbl for e, lbl in ablation if e in all_data]
    colors     = [EXP_STYLE[e]["color"] for e in exps_avail]

    best_wr = []
    f50     = []
    final_mean = []
    final_std  = []

    for exp in exps_avail:
        d  = all_data[exp]
        wr = np.array(d["eval_win_rates"]) * 100
        best_wr.append(d.get("best_win_rate", max(d["eval_win_rates"]) if d["eval_win_rates"] else 0) * 100)
        raw_f50 = d.get("first_50pct_episode")
        f50.append(raw_f50 if raw_f50 else 0)
        tail = wr[-50:] if len(wr) >= 50 else wr
        final_mean.append(np.mean(tail))
        final_std.append(np.std(tail))

    x  = np.arange(len(exps_avail))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Reward Shaping Ablation — Marginal Contribution of Each Ingredient",
                 fontsize=13, fontweight="bold")

    # Panel 1: Best win rate
    ax = axes[0]
    bars = ax.bar(x, best_wr, color=colors, edgecolor="black", lw=0.6, width=0.6, alpha=0.85)
    ax.axhline(50, color="gray", ls="--", lw=1.2, alpha=0.6)
    for bar, v in zip(bars, best_wr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Best Win Rate vs Greedy (%)"); ax.set_ylim(0, 80)
    ax.set_title("Peak Win Rate"); ax.grid(True, alpha=0.18, axis="y")

    # Panel 2: Sample efficiency (episodes to 50%)
    ax = axes[1]
    f50_vals = [v if v > 0 else None for v in f50]
    display  = [v if v else 0 for v in f50_vals]
    bars = ax.bar(x, display, color=colors, edgecolor="black", lw=0.6, width=0.6, alpha=0.85)
    for bar, v, raw in zip(bars, display, f50_vals):
        label = f"{v:,}" if raw else "Never"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
                label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Episodes to First 50% Win Rate")
    ax.set_title("Sample Efficiency"); ax.grid(True, alpha=0.18, axis="y")

    # Panel 3: Final-phase stability (mean ± std)
    ax = axes[2]
    bars = ax.bar(x, final_mean, color=colors, edgecolor="black", lw=0.6, width=0.6, alpha=0.85)
    ax.errorbar(x, final_mean, yerr=final_std, fmt="none", ecolor="black",
                elinewidth=2.0, capsize=6, capthick=1.8)
    ax.axhline(50, color="gray", ls="--", lw=1.2, alpha=0.6)
    for bar, m, s in zip(bars, final_mean, final_std):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1.2,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0, 80)
    ax.set_title("Final-Phase Stability\n(last 50k eps, mean ± std)")
    ax.grid(True, alpha=0.18, axis="y")

    fig.tight_layout()
    _save(fig, plot_dir, "ablation_ladder.png")


# =============================================================================
# 4. Best vs Final Checkpoint Degradation
# =============================================================================

def eval_checkpoint(ckpt_path, n_games=100):
    """Load a checkpoint and evaluate it against greedy. Returns win rate."""
    from agent import DDQNAgent
    from opponents import greedy_opponent
    from splendor_env import SplendorEnv

    agent = DDQNAgent()
    agent.load(ckpt_path)
    agent.epsilon = 0.0

    env  = SplendorEnv(opponent_policy=greedy_opponent, reward_fn=None)
    wins = 0
    for _ in range(n_games):
        obs, info = env.reset()
        mask = info["legal_mask"]
        for _ in range(300):
            action = agent.select_action(obs, mask)
            obs, _, done, _, info = env.step(action)
            mask = info["legal_mask"]
            if done:
                break
        if env.winner == 0:
            wins += 1
    return wins / n_games * 100


def plot_best_vs_final(checkpoint_dir, plot_dir, n_games=200):
    """
    For each experiment, evaluate both _best.pt and _final.pt against greedy.
    Degradation = best - final win rate. Positive = agent regressed late in training.
    """
    exps_found = []
    best_wrs   = []
    final_wrs  = []

    for exp in EXPS:
        best_path  = os.path.join(checkpoint_dir, f"{exp}_best.pt")
        final_path = os.path.join(checkpoint_dir, f"{exp}_final.pt")
        if not (os.path.exists(best_path) and os.path.exists(final_path)):
            print(f"  [skip] {exp}: checkpoint(s) missing")
            continue

        print(f"  Evaluating {exp} best  ({n_games} games)...", end=" ", flush=True)
        bw = eval_checkpoint(best_path, n_games)
        print(f"{bw:.1f}%")

        print(f"  Evaluating {exp} final ({n_games} games)...", end=" ", flush=True)
        fw = eval_checkpoint(final_path, n_games)
        print(f"{fw:.1f}%")

        exps_found.append(exp)
        best_wrs.append(bw)
        final_wrs.append(fw)

    if not exps_found:
        print("  No checkpoints found — skipping best_vs_final.png")
        return

    x      = np.arange(len(exps_found))
    colors = [EXP_STYLE[e]["color"] for e in exps_found]
    w      = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Best Checkpoint vs Final Checkpoint — Did Training Too Long Hurt?",
                 fontsize=13, fontweight="bold")

    # Panel 1: side-by-side bars
    ax = axes[0]
    for i, (exp, bw, fw, col) in enumerate(zip(exps_found, best_wrs, final_wrs, colors)):
        ax.bar(i - w/2, bw, w, color=col,   edgecolor="black", lw=0.6, alpha=0.9,  label="Best"  if i == 0 else "")
        ax.bar(i + w/2, fw, w, color=col,   edgecolor="black", lw=0.6, alpha=0.45, label="Final" if i == 0 else "", hatch="//")
        ax.text(i - w/2, bw + 0.5, f"{bw:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i + w/2, fw + 0.5, f"{fw:.1f}%", ha="center", va="bottom", fontsize=8, color="#555")

    ax.axhline(50, color="gray", ls="--", lw=1.2, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(exps_found)
    ax.set_ylabel("Win Rate vs Greedy (%)"); ax.set_ylim(0, 80)
    ax.set_title("Win Rate: Best vs Final Checkpoint")
    ax.grid(True, alpha=0.18, axis="y")

    best_patch  = mpatches.Patch(facecolor="gray", edgecolor="black", label="Best checkpoint")
    final_patch = mpatches.Patch(facecolor="gray", edgecolor="black", alpha=0.45, hatch="//", label="Final checkpoint")
    ax.legend(handles=[best_patch, final_patch], fontsize=9)

    # Panel 2: degradation bar (best - final)
    ax = axes[1]
    deltas = [b - f for b, f in zip(best_wrs, final_wrs)]
    bar_colors = ["#e74c3c" if d > 3 else "#27ae60" if d < -1 else "#f39c12"
                  for d in deltas]
    bars = ax.bar(x, deltas, color=bar_colors, edgecolor="black", lw=0.6, width=0.55, alpha=0.85)
    ax.axhline(0, color="black", lw=1.2)
    for bar, d in zip(bars, deltas):
        ypos = d + 0.3 if d >= 0 else d - 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{d:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(exps_found)
    ax.set_ylabel("Degradation (Best − Final) %")
    ax.set_title("Late-Training Degradation\n(red = regressed, green = improved)")
    ax.grid(True, alpha=0.18, axis="y")

    red_p   = mpatches.Patch(color="#e74c3c", label="Regressed (>3%)")
    orange_p= mpatches.Patch(color="#f39c12", label="Stable (±3%)")
    green_p = mpatches.Patch(color="#27ae60", label="Improved")
    ax.legend(handles=[red_p, orange_p, green_p], fontsize=8)

    fig.tight_layout()
    _save(fig, plot_dir, "best_vs_final.png")


# =============================================================================
# 5. Cross-Experiment Tournament (round-robin win matrix)
# =============================================================================

def run_tournament(checkpoint_dir, n_games=100):
    """
    Round-robin: each agent plays every other agent as both player 1 and 2.
    Returns a win rate matrix W where W[i][j] = win% of exp_i vs exp_j.
    """
    from agent import DDQNAgent
    from splendor_env import SplendorEnv

    # Load all available agents
    agents = {}
    for exp in EXPS:
        path = os.path.join(checkpoint_dir, f"{exp}_best.pt")
        if not os.path.exists(path):
            continue
        try:
            a = DDQNAgent()
            a.load(path)
            a.epsilon = 0.0
            agents[exp] = a
            print(f"  [loaded] {exp}")
        except Exception as e:
            print(f"  [skip] {exp}: {e}")

    exp_list = list(agents.keys())
    n = len(exp_list)
    if n < 2:
        print("  Need at least 2 agents for tournament")
        return exp_list, None

    matrix = np.full((n, n), np.nan)

    for i, exp_a in enumerate(exp_list):
        for j, exp_b in enumerate(exp_list):
            if i == j:
                matrix[i][j] = 50.0  # vs self = 50%
                continue
            agent_a = agents[exp_a]
            agent_b = agents[exp_b]

            # agent_a is player 0, agent_b is the opponent policy
            def opp_policy(obs, legal_mask, env):
                return agent_b.select_action(obs, legal_mask)

            env  = SplendorEnv(opponent_policy=opp_policy, reward_fn=None)
            wins = 0
            for _ in range(n_games):
                obs, info = env.reset()
                mask = info["legal_mask"]
                for _ in range(300):
                    action = agent_a.select_action(obs, mask)
                    obs, _, done, _, info = env.step(action)
                    mask = info["legal_mask"]
                    if done:
                        break
                if env.winner == 0:
                    wins += 1
            wr = wins / n_games * 100
            matrix[i][j] = wr
            print(f"  {exp_a} vs {exp_b}: {wr:.1f}%")

    return exp_list, matrix


def plot_tournament_matrix(exp_list, matrix, plot_dir):
    if matrix is None:
        return

    n   = len(exp_list)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Cross-Experiment Tournament — Round-Robin Win Matrix",
                 fontsize=13, fontweight="bold")

    # Panel 1: heatmap
    ax = axes[0]
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Win Rate (%)")
    ax.set_xticks(range(n)); ax.set_xticklabels(exp_list, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(exp_list, fontsize=9)
    ax.set_xlabel("Opponent (column agent)"); ax.set_ylabel("Agent (row agent)")
    ax.set_title("Win Rate Matrix\n(row = player, col = opponent)")

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if not np.isnan(val):
                color = "white" if abs(val - 50) > 25 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    # Panel 2: ranking by average win rate vs all opponents (excluding self)
    ax = axes[1]
    avg_wrs = []
    for i in range(n):
        row = [matrix[i][j] for j in range(n) if i != j and not np.isnan(matrix[i][j])]
        avg_wrs.append(np.mean(row) if row else 0)

    ranked = sorted(zip(avg_wrs, exp_list), reverse=True)
    r_vals, r_exps = zip(*ranked)
    r_colors = [EXP_STYLE[e]["color"] for e in r_exps]

    bars = ax.barh(range(len(r_exps)), r_vals, color=r_colors,
                   edgecolor="black", lw=0.6, alpha=0.85)
    ax.axvline(50, color="gray", ls="--", lw=1.2, alpha=0.6)
    for bar, v, exp in zip(bars, r_vals, r_exps):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(r_exps)))
    ax.set_yticklabels([f"#{i+1} {e}" for i, e in enumerate(r_exps)], fontsize=9)
    ax.set_xlabel("Average Win Rate vs All Opponents (%)")
    ax.set_title("Tournament Ranking\n(avg win rate vs all others)")
    ax.set_xlim(0, 85)
    ax.grid(True, alpha=0.18, axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, plot_dir, "tournament_matrix.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",        default="logs")
    parser.add_argument("--plot-dir",       default="plots")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--n-games",        type=int, default=200,
                        help="Games per matchup for checkpoint-based analyses")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Skip analyses that require loading checkpoints (faster)")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print("\n Loading training logs...")
    all_data = load_all(args.log_dir)

    if not all_data:
        print("No log data found. Run training first.")
        sys.exit(1)

    # ── Log-based analyses (fast) ─────────────────────────────────────────────
    print("\n[1/5] Win rate stability...")
    plot_win_rate_stability(all_data, args.plot_dir)

    print("\n[2/5] Training volatility...")
    plot_training_volatility(all_data, args.plot_dir)

    print("\n[3/5] Ablation ladder...")
    plot_ablation_ladder(all_data, args.plot_dir)

    if args.skip_checkpoint:
        print("\n Skipping checkpoint-based analyses (--skip-checkpoint).")
        print(" Done.\n")
        return

    # ── Checkpoint-based analyses (slower) ───────────────────────────────────
    # Need to add project dir to sys.path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print(f"\n[4/5] Best vs final checkpoint ({args.n_games} games each)...")
    plot_best_vs_final(args.checkpoint_dir, args.plot_dir, n_games=args.n_games)

    print(f"\n[5/5] Cross-experiment tournament ({args.n_games} games per matchup)...")
    exp_list, matrix = run_tournament(args.checkpoint_dir, n_games=args.n_games)
    if matrix is not None:
        plot_tournament_matrix(exp_list, matrix, args.plot_dir)

    print(f"\n All extra analysis plots saved to: {args.plot_dir}/\n")


if __name__ == "__main__":
    main()
pt_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print(f"\n[4/5] Best vs final checkpoint ({args.n_games} games each)...")
    plot_best_vs_final(args.checkpoint_dir, args.plot_dir, n_games=args.n_games)

    print(f"\n[5/5] Cross-experiment tournament ({args.n_games} games per matchup)...")
    exp_list, matrix = run_tournament(args.checkpoint_dir, n_games=args.n_games)
    if matrix is not None:
        plot_tournament_matrix(exp_list, matrix, args.plot_dir)

    print(f"\n All extra analysis plots saved to: {args.plot_dir}/\n")


if __name__ == "__main__":
    main()
