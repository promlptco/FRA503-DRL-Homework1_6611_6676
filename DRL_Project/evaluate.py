"""
Comprehensive Evaluation & Plotting for Splendor DRL Experiments (extra).

Generates 12 plot types:
  From training logs:
  1.  training_rewards.png     -- smoothed episodic return per experiment
  2.  win_rates.png            -- win % vs greedy over training
  3.  loss_curves.png          -- smoothed TD loss
  4.  epsilon_schedule.png     -- exploration annealing curve
  5.  episode_lengths.png      -- game duration over training
  6.  sample_efficiency.png    -- bar: episodes to reach 50% win rate
  7.  final_performance.png    -- bar: best win rate per experiment
  8.  reward_distribution.png  -- violin: reward distributions (last 20%)
  9.  win_rate_trend.png       -- win rate with smoothed trend overlay
  10. training_dashboard.png   -- 2x3 dashboard combining key metrics

  From live evaluation games (requires checkpoints/):
  11. action_heatmap.png       -- action type distribution at each game step
  12. action_distribution.png  -- overall action type breakdown per experiment
  13. score_progression.png    -- average prestige progress vs greedy
  14. card_level_purchases.png -- average L1/L2/L3 cards bought per game
  15. final_score_distribution.png -- final prestige spread for agent/greedy

Data sources (tried in order):
  <exp>_metrics.json  -- full per-episode data  (preferred)
  <exp>_compact.json  -- eval-interval snapshots (fallback)

Usage:
  python main.py eval --log-dir logs --plot-dir plots
  python main.py eval --log-dir logs --plot-dir plots --checkpoint-dir checkpoints
  python evaluate.py  --log-dir logs --plot-dir plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
})

# -- Action categories --------------------------------------------------------
ACTION_CATS = [
    ("Take 3 Gems",  list(range(0,  10)), "#3498db"),
    ("Take 2 Gems",  list(range(10, 15)), "#1abc9c"),
    ("Buy Card",     list(range(15, 30)), "#27ae60"),
    ("Reserve Card", list(range(30, 45)), "#e67e22"),
    ("Discard",      list(range(45, 50)), "#e74c3c"),
]
_ACTION_CAT_MAP = {}
for _cname, _cacts, _ccol in ACTION_CATS:
    for _a in _cacts:
        _ACTION_CAT_MAP[_a] = _cname

def _cat_of(action):
    return _ACTION_CAT_MAP.get(action, "Unknown")

# -- Experiment styling -------------------------------------------------------
EXP_STYLE = {
    "C0": {"color": "#27ae60", "label": "C0: Score-only vs Random",              "ls": "-"},
    "C1": {"color": "#2980b9", "label": "C1: Score-only vs Greedy",               "ls": "-"},
    "C2": {"color": "#e74c3c", "label": "C2: Event-shaped vs Greedy",             "ls": "-"},
    "C4": {"color": "#f39c12", "label": "C4: Card Rush vs Greedy",                "ls": "--"},
    "C5": {"color": "#8e44ad", "label": "C5: Noble Hunter vs Greedy",             "ls": "--"},
    "C6": {"color": "#16a085", "label": "C6: Balanced Dense vs Greedy",           "ls": "--"},
    "C3": {"color": "#8e44ad", "label": "C3: Event-shaped + Self-play + Dueling", "ls": "--"},
}


# =============================================================================
# Data Loading
# =============================================================================

def _load_one(log_dir, exp):
    full_path    = os.path.join(log_dir, f"{exp}_metrics.json")
    compact_path = os.path.join(log_dir, f"{exp}_compact.json")

    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            data = json.load(f)
        data["_source"] = "full"
        return data

    if os.path.exists(compact_path):
        with open(compact_path, "r") as f:
            raw = json.load(f)
        return {
            "episode_rewards":     raw.get("reward_snapshots",  []),
            "episode_lengths":     raw.get("length_snapshots",  []),
            "losses":              raw.get("loss_snapshots",     []),
            "epsilons":            raw.get("epsilon_snapshots",  []),
            "eval_win_rates":      raw.get("eval_win_rates",     []),
            "eval_episodes":       raw.get("eval_episodes",      []),
            "best_win_rate":       raw.get("best_win_rate",      0),
            "first_50pct_episode": raw.get("first_50pct_episode", None),
            "total_steps":         raw.get("total_steps",         0),
            "_source":             "compact",
        }
    return None


def load_all(log_dir, experiments=None):
    if experiments is None:
        experiments = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    result = {}
    for exp in experiments:
        d = _load_one(log_dir, exp)
        if d is not None:
            result[exp] = d
            n = len(d["episode_rewards"])
            print(f"  [{d['_source']:>7}] {exp}: {n} data points")
        else:
            print(f"  [missing] {exp}: no log found -- skipping")
    return result


# =============================================================================
# Helpers
# =============================================================================

def smooth(values, window=200):
    if len(values) == 0:
        return []
    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        s      = max(0, i - window + 1)
        out[i] = arr[s: i + 1].mean()
    return out.tolist()


def _x(data, key="episode_rewards"):
    return np.arange(1, len(data[key]) + 1)


def _sw(data):
    return 500 if data["_source"] == "full" else 5


def _save(fig, save_dir, filename, **kw):
    path = os.path.join(save_dir, filename)
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  Saved: {path}")


def _short_label(exp):
    return EXP_STYLE[exp]["label"].replace(": ", "\n", 1)


def _legend_below(fig, ax=None, ncol=4):
    handles, labels = (ax or fig.axes[0]).get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.95,
               borderpad=0.35, handlelength=2.2, columnspacing=1.0)


def _clean_line_axes(ax):
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.margins(x=0.01)


# =============================================================================
# Training Log Plots
# =============================================================================

def plot_training_rewards(all_data, save_dir):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for exp, data in all_data.items():
        rv = data["episode_rewards"]
        if not rv:
            continue
        s = EXP_STYLE[exp]
        ax.plot(_x(data), smooth(rv, _sw(data)),
                color=s["color"], label=s["label"], lw=1.8, ls=s["ls"], alpha=0.9)
    ax.set(xlabel="Episode", ylabel="Smoothed Return",
           title="Training Reward Curves")
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, save_dir, "training_rewards.png")


def plot_win_rates(all_data, save_dir):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.5, label="50% baseline")
    for exp, data in all_data.items():
        eps = data["eval_episodes"]
        wr  = [w * 100 for w in data["eval_win_rates"]]
        if not eps:
            continue
        s = EXP_STYLE[exp]
        ax.plot(eps, wr, color=s["color"], label=s["label"],
                marker="o", ms=2.5, lw=1.5, ls=s["ls"], alpha=0.85)
    ax.set(xlabel="Episode", ylabel="Win Rate vs Greedy (%)",
           title="Win Rate During Training (vs greedy benchmark)", ylim=(-3, 103))
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, save_dir, "win_rates.png")


def plot_loss_curves(all_data, save_dir):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for exp, data in all_data.items():
        losses = data["losses"]
        if not losses:
            continue
        s = EXP_STYLE[exp]
        ax.plot(_x(data, "losses"), smooth(losses, _sw(data)),
                color=s["color"], label=s["label"], lw=1.5, ls=s["ls"], alpha=0.9)
    ax.set(xlabel="Episode", ylabel="Smoothed Loss",
           title="Training Loss Curves")
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, save_dir, "loss_curves.png")


def plot_epsilon_schedule(all_data, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for exp, data in all_data.items():
        epsilons = data["epsilons"]
        if not epsilons:
            continue
        s = EXP_STYLE[exp]
        ax.plot(_x(data, "epsilons"), epsilons,
                color=s["color"], label=s["label"], lw=1.5, ls=s["ls"], alpha=0.9)
    ax.axhline(0.05, color="gray", ls=":", lw=1, label="epsilon_min = 0.05")
    ax.set(xlabel="Episode", ylabel="Epsilon",
           title="Exploration Rate (Epsilon) Schedule", ylim=(-0.02, 1.05))
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, save_dir, "epsilon_schedule.png")


def plot_episode_lengths(all_data, save_dir):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for exp, data in all_data.items():
        lengths = data["episode_lengths"]
        if not lengths:
            continue
        s = EXP_STYLE[exp]
        ax.plot(_x(data, "episode_lengths"), smooth(lengths, _sw(data)),
                color=s["color"], label=s["label"], lw=1.5, ls=s["ls"], alpha=0.9)
    ax.set(xlabel="Episode", ylabel="Smoothed Game Length (steps)",
           title="Episode Length Over Training")
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, save_dir, "episode_lengths.png")


def plot_sample_efficiency(all_data, save_dir):
    exps, vals, colors = [], [], []
    for exp, data in all_data.items():
        f50 = data.get("first_50pct_episode")
        exps.append(exp)
        vals.append(f50 if f50 is not None else 0)
        colors.append(EXP_STYLE[exp]["color"])

    fig, ax = plt.subplots(figsize=(max(7, len(exps) * 1.25), 4.8))
    bars = ax.bar(exps, vals, color=colors, edgecolor="black", lw=0.6, width=0.55)
    top  = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        label = f"{v:,}" if v > 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.01,
                label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set(xlabel="Experiment", ylabel="Episodes to First 50% Win Rate",
           title="Sample Efficiency -- Episodes to Reach 50% Win Rate vs Greedy")
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout()
    _save(fig, save_dir, "sample_efficiency.png")


def plot_final_performance(all_data, save_dir):
    exps, bests, colors = [], [], []
    for exp, data in all_data.items():
        bw = data.get("best_win_rate",
                      max(data["eval_win_rates"]) if data["eval_win_rates"] else 0)
        exps.append(exp)
        bests.append(bw * 100)
        colors.append(EXP_STYLE[exp]["color"])

    fig, ax = plt.subplots(figsize=(max(7, len(exps) * 1.25), 4.8))
    bars = ax.bar(exps, bests, color=colors, edgecolor="black", lw=0.6, width=0.55)
    ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.6, label="50% baseline")
    for bar, v in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set(xlabel="Experiment", ylabel="Best Win Rate vs Greedy (%)",
           title="Peak Performance Comparison", ylim=(0, 110))
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout()
    _save(fig, save_dir, "final_performance.png")


def plot_reward_distribution(all_data, save_dir):
    """Violin of rewards from the last 20% of episodes."""
    groups, labels, colors = [], [], []
    for exp, data in all_data.items():
        rv = data["episode_rewards"]
        if len(rv) < 20:
            continue
        groups.append(rv[int(len(rv) * 0.8):])
        labels.append(exp)
        colors.append(EXP_STYLE[exp]["color"])

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.25), 4.8))
    parts = ax.violinplot(groups, positions=range(len(groups)),
                          showmedians=True, showextrema=True, widths=0.7)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set(xlabel="Experiment", ylabel="Episode Return",
           title="Reward Distribution (last 20% of training)")
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout()
    _save(fig, save_dir, "reward_distribution.png")


def plot_win_rate_trend(all_data, save_dir):
    """Raw win rate (faint) + smoothed trend line."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.5)
    for exp, data in all_data.items():
        eps = data["eval_episodes"]
        wr  = [w * 100 for w in data["eval_win_rates"]]
        if not eps:
            continue
        s  = EXP_STYLE[exp]
        sw = max(5, len(wr) // 15)
        ax.plot(eps, wr, color=s["color"], lw=0.7, alpha=0.25, ls=s["ls"])
        ax.plot(eps, smooth(wr, sw), color=s["color"],
                label=s["label"], lw=2.0, ls=s["ls"], alpha=0.95)
    ax.set(xlabel="Episode", ylabel="Win Rate vs Greedy (%)",
           title="Win Rate -- Raw vs Smoothed Trend", ylim=(-3, 103))
    _clean_line_axes(ax)
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, save_dir, "win_rate_trend.png")


def plot_dashboard(all_data, save_dir):
    """2x3 multi-panel dashboard."""
    fig = plt.figure(figsize=(18, 10.5))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    a_rew, a_wr, a_loss, a_eps, a_len, a_perf = axes

    def _series(ax, key, ylabel, title):
        for exp, data in all_data.items():
            vals = data[key]
            if not vals:
                continue
            s = EXP_STYLE[exp]
            ax.plot(_x(data, key), smooth(vals, _sw(data)),
                    color=s["color"], lw=1.4, ls=s["ls"], label=s["label"])
        ax.set(title=title, xlabel="Episode", ylabel=ylabel)
        ax.grid(True, alpha=0.18)

    _series(a_rew,  "episode_rewards", "Return", "Training Rewards")
    _series(a_loss, "losses",          "Loss",   "TD Loss")
    _series(a_eps,  "epsilons",        "Epsilon","Epsilon Schedule")
    _series(a_len,  "episode_lengths", "Steps",  "Episode Length")
    a_eps.axhline(0.05, color="gray", ls=":", lw=0.8)
    a_eps.set_ylim(-0.02, 1.05)

    a_wr.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
    for exp, data in all_data.items():
        eps = data["eval_episodes"]
        wr  = [w * 100 for w in data["eval_win_rates"]]
        if not eps:
            continue
        s  = EXP_STYLE[exp]
        sw = max(3, len(wr) // 15)
        a_wr.plot(eps, smooth(wr, sw), color=s["color"],
                  lw=1.5, ls=s["ls"], label=s["label"])
    a_wr.set(title="Win Rate vs Greedy", xlabel="Episode",
             ylabel="Win %", ylim=(-3, 103))
    a_wr.grid(True, alpha=0.18)

    exps_p, bests_p, colors_p = [], [], []
    for exp, data in all_data.items():
        bw = data.get("best_win_rate",
                      max(data["eval_win_rates"]) if data["eval_win_rates"] else 0)
        exps_p.append(exp)
        bests_p.append(bw * 100)
        colors_p.append(EXP_STYLE[exp]["color"])
    bars = a_perf.bar(exps_p, bests_p, color=colors_p,
                      edgecolor="black", lw=0.5, width=0.55)
    a_perf.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.6)
    for bar, v in zip(bars, bests_p):
        a_perf.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    a_perf.set(title="Best Win Rate", xlabel="Experiment",
               ylabel="Win %", ylim=(0, 110))
    a_perf.grid(True, alpha=0.18, axis="y")

    _legend_below(fig, a_wr)
    fig.suptitle("Splendor DRL -- Training Dashboard",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(fig, save_dir, "training_dashboard.png", bbox_inches="tight")


# =============================================================================
# Sequential Move Plots  (requires checkpoints + game modules)
# =============================================================================

def run_action_sequences(checkpoint_dir, experiments, n_games=100):
    """
    Load best checkpoint for each experiment, play n_games vs greedy,
    and record every action taken.  Returns dict keyed by experiment.
    """
    try:
        from agent import DDQNAgent
        from opponents import greedy_opponent
        from splendor_env import SplendorEnv
    except ImportError as e:
        print(f"  [seq] Cannot import game modules: {e} -- skipping move plots")
        return {}

    seq_data = {}
    for exp in experiments:
        ckpt = os.path.join(checkpoint_dir, f"{exp}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  [seq] No checkpoint for {exp} ({ckpt}) -- skipping")
            continue
        try:
            agent = DDQNAgent()
            agent.load(ckpt)
            agent.epsilon = 0.0
        except Exception as e:
            print(f"  [seq] Could not load {exp}: {e}")
            continue

        env         = SplendorEnv(opponent_policy=greedy_opponent, reward_fn=None)
        sequences   = []
        all_actions = []

        print(f"  [seq] {exp}: running {n_games} games...", end=" ", flush=True)
        for _ in range(n_games):
            obs, info = env.reset()
            mask      = info["legal_mask"]
            game_seq  = []
            for _ in range(200):
                action = agent.select_action(obs, mask)
                game_seq.append(action)
                all_actions.append(action)
                obs, _, done, _, info = env.step(action)
                mask = info["legal_mask"]
                if done:
                    break
            sequences.append(game_seq)

        avg_len = float(np.mean([len(s) for s in sequences]))
        print(f"done  (avg length {avg_len:.1f} steps)")
        seq_data[exp] = {"sequences": sequences, "all_actions": all_actions}

    return seq_data


def plot_action_heatmap(seq_data, save_dir):
    """
    One sub-panel per experiment.
    X-axis = game step position (capped at 60).
    Y-axis = proportion of that action type at that step.
    Stacked bars show how the agent's strategy evolves through a game.
    """
    if not seq_data:
        return

    n_exp   = len(seq_data)
    cols = min(4, n_exp)
    rows = int(np.ceil(n_exp / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.8),
                             sharey=True, squeeze=False)
    if n_exp == 1:
        axes = [axes]

    MAX_STEP   = 60
    cat_names  = [c[0] for c in ACTION_CATS]
    cat_colors = [c[2] for c in ACTION_CATS]

    for idx, (exp, data) in enumerate(seq_data.items()):
        ax = axes[idx // cols][idx % cols]
        counts = {cat: np.zeros(MAX_STEP) for cat in cat_names}
        totals = np.zeros(MAX_STEP)

        for seq in data["sequences"]:
            for step, action in enumerate(seq):
                if step >= MAX_STEP:
                    break
                cat = _cat_of(action)
                if cat in counts:
                    counts[cat][step] += 1
                totals[step] += 1

        steps  = np.arange(MAX_STEP)
        bottom = np.zeros(MAX_STEP)
        for cat, col in zip(cat_names, cat_colors):
            prop = np.divide(counts[cat], totals, out=np.zeros_like(totals),
                             where=totals > 0)
            ax.bar(steps, prop, bottom=bottom, color=col,
                   label=cat, alpha=0.85, width=1.0, linewidth=0)
            bottom += prop

        n_games = len(data["sequences"])
        s = EXP_STYLE.get(exp, {"label": exp})
        short = s["label"].split(":", 1)[-1].strip()
        ax.set_title(f"{exp}: {short}\n({n_games} games)", fontsize=9)
        ax.set_xlabel("Game Step")
        if idx % cols == 0:
            ax.set_ylabel("Proportion of Actions")
        ax.set_xlim(-0.5, MAX_STEP - 0.5)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(n_exp, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(cat_names),
               bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
    fig.suptitle("Sequential Action Distribution by Game Step",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, save_dir, "action_heatmap.png")


def plot_action_distribution(seq_data, save_dir):
    """
    Grouped bar chart: for each experiment, show what fraction of all
    agent actions belong to each category (across all evaluation games).
    """
    if not seq_data:
        return

    cat_names  = [c[0] for c in ACTION_CATS]
    cat_colors = [c[2] for c in ACTION_CATS]
    cat_sets   = [set(c[1]) for c in ACTION_CATS]
    exps       = list(seq_data.keys())
    n_exp, n_cat = len(exps), len(cat_names)

    x = np.arange(n_exp)
    w = 0.75 / n_cat

    fig, ax = plt.subplots(figsize=(max(9, n_exp * 1.35), 5))

    for ci, (cat, col, cset) in enumerate(zip(cat_names, cat_colors, cat_sets)):
        fracs = []
        for exp in exps:
            actions = seq_data[exp]["all_actions"]
            total   = len(actions)
            count   = sum(1 for a in actions if a in cset)
            fracs.append(count / total * 100 if total else 0)
        bars = ax.bar(x + ci * w, fracs, width=w, color=col, label=cat,
                      edgecolor="black", lw=0.4, alpha=0.85)
        for bar, v in zip(bars, fracs):
            if v >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{v:.1f}%", ha="center", va="bottom",
                            fontsize=7, fontweight="bold", color="#222")

    ax.set_xticks(x + w * (n_cat - 1) / 2)
    ax.set_xticklabels(exps)
    ax.set(xlabel="Experiment", ylabel="% of All Actions",
           title="Overall Action Type Distribution per Experiment")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=n_cat, framealpha=0.95)
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, save_dir, "action_distribution.png")


# =============================================================================
# Splendor-Specific Behavior Plots
# =============================================================================

def run_behavior_rollouts(checkpoint_dir, experiments, n_games=100, max_steps=200):
    """
    Replay best checkpoints vs greedy and collect Splendor-specific behavior:
    score progression, final score distribution, and card-level purchases.
    """
    try:
        from agent import DDQNAgent
        from card_data import get_card_level
        from opponents import greedy_opponent
        from splendor_env import SplendorEnv
    except ImportError as e:
        print(f"  [behavior] Cannot import game modules: {e} -- skipping")
        return {}

    behavior = {}
    for exp in experiments:
        ckpt = os.path.join(checkpoint_dir, f"{exp}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  [behavior] No checkpoint for {exp} ({ckpt}) -- skipping")
            continue

        try:
            agent = DDQNAgent()
            agent.load(ckpt)
            agent.epsilon = 0.0
        except Exception as e:
            print(f"  [behavior] Could not load {exp}: {e}")
            continue

        env = SplendorEnv(opponent_policy=greedy_opponent, reward_fn=None)
        agent_scores, opp_scores = [], []
        final_agent, final_opp, wins = [], [], []
        agent_levels = np.zeros(3, dtype=float)
        opp_levels = np.zeros(3, dtype=float)

        print(f"  [behavior] {exp}: running {n_games} games...", end=" ", flush=True)
        for _ in range(n_games):
            obs, info = env.reset()
            mask = info["legal_mask"]
            game_agent_scores = [env.players[0].prestige]
            game_opp_scores = [env.players[1].prestige]

            for _step in range(max_steps):
                before_a = len(env.players[0].owned_cards)
                before_o = len(env.players[1].owned_cards)

                action = agent.select_action(obs, mask)
                obs, _, done, _, info = env.step(action)
                mask = info["legal_mask"]

                for card in env.players[0].owned_cards[before_a:]:
                    lvl = get_card_level(card)
                    if 1 <= lvl <= 3:
                        agent_levels[lvl - 1] += 1
                for card in env.players[1].owned_cards[before_o:]:
                    lvl = get_card_level(card)
                    if 1 <= lvl <= 3:
                        opp_levels[lvl - 1] += 1

                game_agent_scores.append(env.players[0].prestige)
                game_opp_scores.append(env.players[1].prestige)
                if done:
                    break

            agent_scores.append(game_agent_scores)
            opp_scores.append(game_opp_scores)
            final_agent.append(env.players[0].prestige)
            final_opp.append(env.players[1].prestige)
            wins.append(env.winner == 0)

        behavior[exp] = {
            "agent_scores": agent_scores,
            "opp_scores": opp_scores,
            "final_agent": final_agent,
            "final_opp": final_opp,
            "wins": wins,
            "agent_levels": agent_levels / max(1, n_games),
            "opp_levels": opp_levels / max(1, n_games),
        }
        print(f"done  (win {np.mean(wins) * 100:.0f}%)")

    return behavior


def _pad_mean(sequences):
    max_len = max((len(s) for s in sequences), default=0)
    if max_len == 0:
        return np.array([])
    arr = np.full((len(sequences), max_len), np.nan)
    for i, seq in enumerate(sequences):
        arr[i, :len(seq)] = seq
        if seq:
            arr[i, len(seq):] = seq[-1]
    return np.nanmean(arr, axis=0)


def plot_score_progression(behavior, save_dir):
    if not behavior:
        return

    exps = list(behavior.keys())
    cols = min(4, len(exps))
    rows = int(np.ceil(len(exps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.4),
                             sharex=True, sharey=True, squeeze=False)

    for idx, exp in enumerate(exps):
        ax = axes[idx // cols][idx % cols]
        data = behavior[exp]
        a = _pad_mean(data["agent_scores"])
        o = _pad_mean(data["opp_scores"])
        x = np.arange(len(a))
        s = EXP_STYLE[exp]
        ax.plot(x, a, color=s["color"], lw=2.0, label="Agent")
        ax.plot(x[:len(o)], o, color="#555555", lw=1.7, ls="--", label="Greedy")
        ax.axhline(15, color="#888888", lw=0.8, ls=":", alpha=0.8)
        ax.set_title(f"{exp}  Win {np.mean(data['wins']) * 100:.0f}%",
                     color=s["color"], fontweight="bold", fontsize=10)
        ax.set_xlabel("Agent turn")
        if idx % cols == 0:
            ax.set_ylabel("Prestige")
        ax.set_ylim(0, 16)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.18)

    for idx in range(len(exps), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
    fig.suptitle("Prestige Progression vs Greedy (final scores carried forward)",
                 fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, save_dir, "score_progression.png")


def plot_card_level_purchases(behavior, save_dir):
    if not behavior:
        return

    exps = list(behavior.keys())
    x = np.arange(len(exps))
    level_names = ["L1", "L2", "L3"]
    colors = ["#65b96f", "#f2b84b", "#d95f5f"]

    fig, ax = plt.subplots(figsize=(max(9, len(exps) * 1.35), 5))
    width = 0.34
    agent_bottom = np.zeros(len(exps))
    opp_bottom = np.zeros(len(exps))
    for li, (name, col) in enumerate(zip(level_names, colors)):
        agent_vals = np.array([behavior[exp]["agent_levels"][li] for exp in exps])
        opp_vals = np.array([behavior[exp]["opp_levels"][li] for exp in exps])
        ax.bar(x - width / 2, agent_vals, bottom=agent_bottom, width=width,
               color=col, edgecolor="white", linewidth=0.6,
               label=f"Agent {name}")
        ax.bar(x + width / 2, opp_vals, bottom=opp_bottom, width=width,
               color=col, edgecolor="white", linewidth=0.6, alpha=0.45,
               hatch="//", label=f"Greedy {name}")
        agent_bottom += agent_vals
        opp_bottom += opp_vals

    for xi, total in zip(x - width / 2, agent_bottom):
        ax.text(xi, total + 0.25, f"{total:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    for xi, total in zip(x + width / 2, opp_bottom):
        ax.text(xi, total + 0.25, f"{total:.1f}", ha="center", va="bottom",
                fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(exps)
    ax.set_ylabel("Cards bought per game")
    ax.set_xlabel("Experiment")
    ax.set_title("Card Level Purchases per Game")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.95)
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, save_dir, "card_level_purchases.png")


def plot_final_score_distribution(behavior, save_dir):
    if not behavior:
        return

    exps = list(behavior.keys())
    agent_groups = [behavior[exp]["final_agent"] for exp in exps]
    opp_groups = [behavior[exp]["final_opp"] for exp in exps]

    fig, ax = plt.subplots(figsize=(max(9, len(exps) * 1.35), 5.2))
    pos_agent = np.arange(len(exps)) - 0.18
    pos_opp = np.arange(len(exps)) + 0.18

    bp_a = ax.boxplot(agent_groups, positions=pos_agent, widths=0.28,
                      patch_artist=True, showfliers=False)
    bp_o = ax.boxplot(opp_groups, positions=pos_opp, widths=0.28,
                      patch_artist=True, showfliers=False)

    for patch, exp in zip(bp_a["boxes"], exps):
        patch.set_facecolor(EXP_STYLE[exp]["color"])
        patch.set_alpha(0.75)
    for patch in bp_o["boxes"]:
        patch.set_facecolor("#b8b8b8")
        patch.set_alpha(0.7)
    for group in (bp_a, bp_o):
        for key in ("medians", "whiskers", "caps"):
            for line in group[key]:
                line.set_color("#222222")
                line.set_linewidth(1.1)

    ax.axhline(15, color="#777777", lw=1.0, ls="--", alpha=0.7,
               label="15 point win target")
    ax.set_xticks(np.arange(len(exps)))
    ax.set_xticklabels(exps)
    ax.set_ylabel("Final prestige")
    ax.set_xlabel("Experiment")
    ax.set_title("Final Score Distribution (Agent vs Greedy)")
    top = max(max(max(g) for g in agent_groups), max(max(g) for g in opp_groups))
    for i, exp in enumerate(exps):
        win_pct = np.mean(behavior[exp]["wins"]) * 100
        margin = np.mean(np.array(behavior[exp]["final_agent"]) -
                         np.array(behavior[exp]["final_opp"]))
        ax.text(i, top + 0.7, f"W {win_pct:.0f}%\nΔ {margin:+.1f}",
                ha="center", va="bottom", fontsize=8, color=EXP_STYLE[exp]["color"],
                fontweight="bold")
    ax.set_ylim(0, top + 2.6)
    ax.legend([bp_a["boxes"][0], bp_o["boxes"][0]],
              ["Agent", "Greedy"], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.18, axis="y")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, save_dir, "final_score_distribution.png")


# =============================================================================
# Summary Table
# =============================================================================

def print_summary(all_data):
    print(f"\n{'='*75}")
    print(f"  {'Exp':<6} {'Best Win%':>10} {'50% at Ep':>14} {'Total Steps':>14} {'Source':>8}")
    print(f"{'='*75}")
    for exp, data in all_data.items():
        bw  = data.get("best_win_rate",
                       max(data["eval_win_rates"]) if data["eval_win_rates"] else 0) * 100
        f50 = data.get("first_50pct_episode", "N/A")
        ts  = data.get("total_steps", 0)
        src = data.get("_source", "?")
        print(f"  {exp:<6} {bw:>9.1f}% {str(f50):>14} {ts:>14,} {src:>8}")
    print(f"{'='*75}\n")


# =============================================================================
# Main Entry
# =============================================================================

def generate_all_plots(log_dir="logs", save_dir="plots",
                       experiments=None, checkpoint_dir="checkpoints"):
    print(f"\n Generating evaluation plots from '{log_dir}' ...")
    os.makedirs(save_dir, exist_ok=True)

    all_data = load_all(log_dir, experiments)
    if not all_data:
        print("  No metrics found -- run training first!")
        return

    exps = list(all_data.keys())
    print(f"\n  Plotting {len(all_data)} experiment(s)...\n")
    plot_training_rewards(all_data, save_dir)
    plot_win_rates(all_data, save_dir)
    plot_loss_curves(all_data, save_dir)
    plot_epsilon_schedule(all_data, save_dir)
    plot_episode_lengths(all_data, save_dir)
    plot_sample_efficiency(all_data, save_dir)
    plot_final_performance(all_data, save_dir)
    plot_reward_distribution(all_data, save_dir)
    plot_win_rate_trend(all_data, save_dir)
    plot_dashboard(all_data, save_dir)

    # Sequential move plots (requires checkpoints)
    print(f"\n  Generating sequential move plots (checkpoint_dir='{checkpoint_dir}')...")
    seq_data = run_action_sequences(checkpoint_dir, exps, n_games=100)
    if seq_data:
        plot_action_heatmap(seq_data, save_dir)
        plot_action_distribution(seq_data, save_dir)
        print(f"  Move plots saved.")
    else:
        print("  No checkpoints found -- skipping move plots.")

    print(f"\n  Generating Splendor behavior plots...")
    behavior = run_behavior_rollouts(checkpoint_dir, exps, n_games=100)
    if behavior:
        plot_score_progression(behavior, save_dir)
        plot_card_level_purchases(behavior, save_dir)
        plot_final_score_distribution(behavior, save_dir)
        print("  Behavior plots saved.")
    else:
        print("  No behavior rollouts collected -- skipping behavior plots.")

    print_summary(all_data)
    print(f"\n  All plots saved to: {save_dir}/\n")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",        default="logs")
    parser.add_argument("--plot-dir",       default="plots")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    generate_all_plots(args.log_dir, args.plot_dir,
                       checkpoint_dir=args.checkpoint_dir)
