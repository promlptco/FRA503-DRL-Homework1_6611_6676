"""
Sequential Decision-Making Plots for Splendor DRL agents.

Generates 3 plot types per experiment (and a combined comparison):
  1. ethogram.png         -- Action raster: game × step, color = action type
  2. transitions.png      -- Markov transition diagram: weighted directed graph
  3. phase_flow.png       -- Sankey-style: action distribution shift early→mid→late

Usage:
  python plot_sequences.py                              # all checkpoints, 50 games each
  python plot_sequences.py --exps C4 C5 --n-games 100
  python plot_sequences.py --checkpoint-dir checkpoints --plot-dir plots
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

matplotlib.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})

# ── Action categories ────────────────────────────────────────────────────────
CATS = [
    ("Take 3",   list(range(0,  10)), "#3498db"),
    ("Take 2",   list(range(10, 15)), "#1abc9c"),
    ("Buy",      list(range(15, 30)), "#27ae60"),
    ("Reserve",  list(range(30, 45)), "#e67e22"),
    ("Discard",  list(range(45, 50)), "#e74c3c"),
]
CAT_NAMES  = [c[0] for c in CATS]
CAT_COLORS = [c[2] for c in CATS]
CAT_SETS   = [set(c[1]) for c in CATS]
_ACT_MAP   = {a: i for i, (_, acts, _) in enumerate(CATS) for a in acts}
N_CATS     = len(CATS)

EXP_STYLE = {
    "C0": {"color": "#27ae60", "label": "C0: Score-only vs Random"},
    "C1": {"color": "#2980b9", "label": "C1: Score-only vs Greedy"},
    "C2": {"color": "#e74c3c", "label": "C2: Event-shaped vs Greedy"},
    "C3": {"color": "#8e44ad", "label": "C3: Self-play + Dueling"},
    "C4": {"color": "#f39c12", "label": "C4: Card Rush vs Greedy"},
    "C5": {"color": "#7d3c98", "label": "C5: Noble Hunter vs Greedy"},
    "C6": {"color": "#16a085", "label": "C6: Balanced Dense vs Greedy"},
}

PHASE_CUTS = (10, 20)   # early: 0-9, mid: 10-19, late: 20+
PHASE_NAMES = ["Early\n(turns 1-10)", "Mid\n(turns 11-20)", "Late\n(turns 21+)"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Game replay
# ═══════════════════════════════════════════════════════════════════════════════

def _cat_of(action):
    return _ACT_MAP.get(action, 2)   # default → Buy if unknown


def collect_sequences(checkpoint_dir, exp, n_games=50):
    """
    Load the best checkpoint for *exp*, play n_games vs greedy, and return:
      sequences  : list of lists — each inner list is [cat_idx, ...] per game step
      won        : list of bool — did the agent win that game?
    """
    try:
        from agent import DDQNAgent
        from opponents import greedy_opponent
        from splendor_env import SplendorEnv
    except ImportError as e:
        print(f"  [seq] import error: {e}")
        return [], []

    ckpt = os.path.join(checkpoint_dir, f"{exp}_best.pt")
    if not os.path.exists(ckpt):
        print(f"  [seq] No checkpoint: {ckpt}")
        return [], []

    try:
        agent = DDQNAgent()
        agent.load(ckpt)
        agent.epsilon = 0.0
    except Exception as e:
        print(f"  [seq] Load error for {exp}: {e}")
        return [], []

    env = SplendorEnv(opponent_policy=greedy_opponent, reward_fn=None)
    sequences, won = [], []

    print(f"  [{exp}] Playing {n_games} games…", end=" ", flush=True)
    for _ in range(n_games):
        obs, info = env.reset()
        mask = info["legal_mask"]
        game_seq = []
        for _ in range(300):
            action = agent.select_action(obs, mask)
            game_seq.append(_cat_of(action))
            obs, _, done, _, info = env.step(action)
            mask = info["legal_mask"]
            if done:
                break
        sequences.append(game_seq)
        won.append(getattr(env, "winner", -1) == 0)

    win_pct = sum(won) / len(won) * 100 if won else 0
    avg_len = np.mean([len(s) for s in sequences])
    print(f"done  ({win_pct:.0f}% win, avg {avg_len:.1f} steps)")
    return sequences, won


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — Ethogram (Action Raster)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ethogram(all_seq, save_dir, max_step=50, max_games=60):
    """
    Raster grid: rows = games, columns = turn steps.
    Cell colour = action category. Grey = game already ended.
    One subplot per experiment, arranged in a grid.
    """
    exps = list(all_seq.keys())
    n    = len(exps)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5.6, rows * 4.6),
                             squeeze=False)

    cmap = matplotlib.colors.ListedColormap(CAT_COLORS + ["#1a1a2e"])  # last = no-data
    bounds = list(range(N_CATS + 2))
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for idx, exp in enumerate(exps):
        ax   = axes[idx // cols][idx % cols]
        seqs, won = all_seq[exp]
        disp_games = seqs[:max_games]
        n_g  = len(disp_games)

        grid = np.full((n_g, max_step), N_CATS, dtype=float)  # N_CATS → "ended"
        for gi, seq in enumerate(disp_games):
            for si, cat in enumerate(seq[:max_step]):
                grid[gi, si] = cat

        im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest", origin="upper")

        # Win/loss indicator on y-axis
        w = [won[i] for i in range(n_g)]
        for gi in range(n_g):
            col = "#27ae60" if w[gi] else "#e74c3c"
            ax.add_patch(mpatches.Rectangle((-1.5, gi - 0.5), 1, 1,
                                             color=col, clip_on=False))

        s  = EXP_STYLE.get(exp, {"label": exp, "color": "#aaa"})
        wr = sum(w) / len(w) * 100 if w else 0
        short = s["label"].replace(": ", "\n", 1)
        ax.set_title(f"{short}\nWin {wr:.0f}% ({n_g} games)",
                     fontsize=9.5, color=s["color"], fontweight="bold")
        ax.set_xlabel("Turn step", fontsize=9)
        ax.set_ylabel("Game episode", fontsize=9)
        ax.set_xlim(-0.5, max_step - 0.5)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(len(exps), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=CAT_COLORS[i], label=CAT_NAMES[i])
               for i in range(N_CATS)]
    patches.append(mpatches.Patch(color="#1a1a2e", label="Game ended"))
    patches.append(mpatches.Patch(color="#27ae60", label="Win ◀"))
    patches.append(mpatches.Patch(color="#e74c3c", label="Loss ◀"))
    fig.legend(handles=patches, loc="lower center", ncol=N_CATS + 2,
               fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.01),
               columnspacing=0.9, handlelength=1.2)

    fig.suptitle("Action Ethogram — Sequential Behavior Raster",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = os.path.join(save_dir, "ethogram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — Markov Transition Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def _circular_positions(n, r=0.36, cx=0.5, cy=0.5, start_angle=math.pi / 2):
    """Return (x, y) positions on a circle for n nodes."""
    pos = []
    for i in range(n):
        angle = start_angle - 2 * math.pi * i / n
        pos.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pos


NODE_R = 0.075    # node radius in data coords (shared by draw & edge-offset)


def _edge_points(p1, p2, r=NODE_R, rad=0.28):
    """
    For a curved arc from p1 to p2, approximate the arc's tangent direction
    at each endpoint and offset by r so the arrow starts/ends at the node edge.
    The offset direction is rotated by the curvature angle.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist   = math.hypot(dx, dy)
    if dist < 1e-6:
        return p1, p2
    # Unit vector along straight line
    ux, uy = dx / dist, dy / dist
    # Perpendicular (the arc curves in this direction by rad)
    perp_x, perp_y = -uy * rad, ux * rad
    # Approximate arc tangent at start: blend straight + perp
    tang_len = math.hypot(ux + perp_x, uy + perp_y)
    t1x = (ux + perp_x) / tang_len
    t1y = (uy + perp_y) / tang_len
    # Approximate arc tangent at end: reverse + perp
    t2x = (-ux + perp_x) / math.hypot(-ux + perp_x, -uy + perp_y)
    t2y = (-uy + perp_y) / math.hypot(-ux + perp_x, -uy + perp_y)
    s = (p1[0] + r * t1x,  p1[1] + r * t1y)
    e = (p2[0] - r * t2x,  p2[1] - r * t2y)
    return s, e


def _draw_curved_arrow(ax, p1, p2, weight, color, self_loop=False, lw_max=7):
    x1, y1 = p1
    x2, y2 = p2
    lw    = max(1.0, weight * lw_max)
    alpha = min(0.95, 0.35 + weight * 0.65)

    if self_loop:
        # Self-loop: small arc curving away from the circle
        # Use a helper point offset from node center
        offset = 0.12
        lp1 = (x1 - offset * 0.5, y1 + offset)
        lp2 = (x1 + offset * 0.5, y1 + offset)
        loop = mpatches.FancyArrowPatch(
            lp1, lp2,
            connectionstyle="arc3,rad=-1.4",
            arrowstyle="-|>",
            mutation_scale=20,
            lw=lw, color=color, alpha=alpha,
            shrinkA=0, shrinkB=0,
            zorder=6)
        ax.add_patch(loop)
        # Label above the loop
        ax.text(x1, y1 + offset + 0.07, f"{weight:.0%}",
                fontsize=7.5, ha="center", va="center",
                color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                zorder=7)
        return

    # Offset start/end to circle edges
    s, e = _edge_points(p1, p2, r=NODE_R, rad=0.28)

    arr = mpatches.FancyArrowPatch(
        s, e,
        connectionstyle="arc3,rad=0.28",
        arrowstyle="-|>",
        mutation_scale=22,       # large visible arrowhead
        lw=lw, color=color, alpha=alpha,
        shrinkA=0, shrinkB=0,    # no extra shrink; edge points are already offset
        zorder=6)
    ax.add_patch(arr)

    # Weight label near midpoint of arc (offset perpendicularly)
    mx = (x1 + x2) / 2 + (y2 - y1) * 0.18
    my = (y1 + y2) / 2 + (x1 - x2) * 0.18
    ax.text(mx, my, f"{weight:.0%}", fontsize=7.5, ha="center", va="center",
            color="white", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            zorder=7)


def plot_transitions(all_seq, save_dir, min_prob=0.05):
    """
    For each experiment, compute the 5×5 action-type transition matrix
    (P(next_cat | cur_cat)) and draw it as a circular directed graph.
    """
    exps = list(all_seq.keys())
    n    = len(exps)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5.9, rows * 5.0),
                             squeeze=False)

    top_k = 8

    for idx, exp in enumerate(exps):
        ax   = axes[idx // cols][idx % cols]
        seqs, won = all_seq[exp]
        s    = EXP_STYLE.get(exp, {"label": exp, "color": "#aaa"})

        # Build transition count matrix
        T = np.zeros((N_CATS, N_CATS))
        for seq in seqs:
            for t in range(len(seq) - 1):
                T[seq[t], seq[t + 1]] += 1

        # Normalise rows → probabilities
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = T / row_sums

        transitions = []
        for i in range(N_CATS):
            for j in range(N_CATS):
                w = P[i, j]
                if w >= min_prob:
                    transitions.append((w, T[i, j], i, j))
        transitions.sort(reverse=True)
        transitions = transitions[:top_k]

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.9, top_k + 1.1)
        ax.axis("off")

        ax.text(0.03, top_k + 0.65, "Current", fontsize=8.5,
                fontweight="bold", color="#333", ha="left", va="center")
        ax.text(0.25, top_k + 0.65, "Next", fontsize=8.5,
                fontweight="bold", color="#333", ha="left", va="center")
        ax.text(0.48, top_k + 0.65, "P(next | current)", fontsize=8.5,
                fontweight="bold", color="#333", ha="left", va="center")

        if not transitions:
            ax.text(0.5, top_k / 2, "No transitions above threshold",
                    ha="center", va="center", fontsize=10, color="#666")

        for rank, (prob, count, cur, nxt) in enumerate(transitions):
            y = top_k - rank - 0.05
            if rank % 2 == 0:
                ax.add_patch(mpatches.Rectangle((0.02, y - 0.33), 0.94, 0.58,
                                                facecolor="#f7f7f7",
                                                edgecolor="none", zorder=0))

            ax.text(0.03, y, CAT_NAMES[cur], ha="left", va="center",
                    fontsize=9, fontweight="bold", color=CAT_COLORS[cur])
            ax.text(0.215, y, "->", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#333")
            ax.text(0.25, y, CAT_NAMES[nxt], ha="left", va="center",
                    fontsize=9, fontweight="bold", color=CAT_COLORS[nxt])

            bar_x = 0.48
            bar_w = 0.40 * prob
            ax.add_patch(mpatches.Rectangle((bar_x, y - 0.15), 0.40, 0.30,
                                            facecolor="#e9ecef",
                                            edgecolor="none", zorder=1))
            ax.add_patch(mpatches.Rectangle((bar_x, y - 0.15), bar_w, 0.30,
                                            facecolor=CAT_COLORS[cur],
                                            edgecolor="white", lw=0.6,
                                            alpha=0.88, zorder=2))
            ax.text(0.91, y, f"{prob:.0%}", ha="right", va="center",
                    fontsize=9, fontweight="bold", color="#222")

        total = T.sum()
        if total > 0:
            freqs = T.sum(axis=1) / total
            dominant = sorted(range(N_CATS), key=lambda i: freqs[i],
                              reverse=True)[:3]
            mix = "Start mix: " + ", ".join(
                f"{CAT_NAMES[i]} {freqs[i]:.0%}" for i in dominant)
            ax.text(0.03, -0.45, mix, ha="left", va="center",
                    fontsize=7.8, color="#555")

        wr = sum(won) / len(won) * 100 if won else 0
        ax.set_title(f"{s['label']}\n(Win {wr:.0f}%)",
                     fontsize=10, color=s["color"], fontweight="bold",
                     pad=4)

    for idx in range(len(exps), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Action Transition Diagram — Markov Chain of Decisions",
                 fontsize=14, fontweight="bold")
    fig.suptitle("Ranked Directed Action Transitions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "transitions.png")
    try:
        fig.savefig(path, bbox_inches="tight")
    except PermissionError:
        path = os.path.join(save_dir, "transitions_directed.png")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3 — Phase Flow (Sankey-style stacked bars + flow lines)
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_dist(seqs, phase_cuts):
    """
    Return (n_phases, N_CATS) array: fraction of each action type per phase.
    """
    cuts  = [0] + list(phase_cuts) + [9999]
    dists = np.zeros((len(cuts) - 1, N_CATS))
    for seq in seqs:
        for step, cat in enumerate(seq):
            for pi in range(len(cuts) - 1):
                if cuts[pi] <= step < cuts[pi + 1]:
                    dists[pi, cat] += 1
                    break
    # Normalise each phase row
    row_sums = dists.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return dists / row_sums


def plot_phase_flow(all_seq, save_dir):
    """
    For each experiment: 3 stacked bars (early / mid / late) connected by
    flow ribbons showing how action proportions shift across game phases.
    """
    exps = list(all_seq.keys())
    n    = len(exps)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.8, rows * 4.4),
                             squeeze=False)

    bar_w = 0.55
    phase_x = [0.0, 1.0, 2.0]      # x positions of the 3 bars

    for idx, exp in enumerate(exps):
        ax   = axes[idx // cols][idx % cols]
        seqs, won = all_seq[exp]
        s    = EXP_STYLE.get(exp, {"label": exp, "color": "#aaa"})
        dist = _phase_dist(seqs, PHASE_CUTS)   # (3, N_CATS)

        # Draw stacked bars
        for pi, px in enumerate(phase_x):
            bottom = 0.0
            for ci in range(N_CATS):
                h = dist[pi, ci]
                if h < 0.002:
                    bottom += h
                    continue
                ax.bar(px, h, bottom=bottom, width=bar_w,
                       color=CAT_COLORS[ci], alpha=0.88,
                       edgecolor="white", linewidth=0.5)
                if h > 0.06:
                    ax.text(px, bottom + h / 2,
                            f"{h:.0%}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white",
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground="black")])
                bottom += h

        # Draw flow ribbons between adjacent bars
        for pi in range(len(phase_x) - 1):
            px1 = phase_x[pi]  + bar_w / 2
            px2 = phase_x[pi + 1] - bar_w / 2
            bot1 = bot2 = 0.0
            for ci in range(N_CATS):
                h1 = dist[pi,     ci]
                h2 = dist[pi + 1, ci]
                if h1 < 0.01 and h2 < 0.01:
                    bot1 += h1; bot2 += h2
                    continue
                # Bezier ribbon via polygon
                mid_x = (px1 + px2) / 2
                poly_x = [px1, mid_x, mid_x, px2, px2, mid_x, mid_x, px1]
                poly_y = [bot1, bot1, bot2, bot2,
                          bot2 + h2, bot2 + h2, bot1 + h1, bot1 + h1]
                ax.fill(poly_x, poly_y, color=CAT_COLORS[ci],
                        alpha=0.22, linewidth=0)
                bot1 += h1; bot2 += h2

        ax.set_xticks(phase_x)
        ax.set_xticklabels(PHASE_NAMES, fontsize=9)
        ax.set_ylim(0, 1.02)
        if idx % cols == 0:
            ax.set_ylabel("Action proportion", fontsize=9)
        ax.set_xlim(-bar_w, 2.0 + bar_w)
        ax.grid(axis="y", alpha=0.2)
        wr = sum(won) / len(won) * 100 if won else 0
        short = s["label"].replace(": ", "\n", 1)
        ax.set_title(f"{short}\nWin {wr:.0f}%",
                     fontsize=9.5, color=s["color"], fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=CAT_COLORS[i], label=CAT_NAMES[i])
               for i in range(N_CATS)]
    fig.legend(handles=patches, loc="lower center", ncol=N_CATS,
               fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handlelength=1.2)

    for idx in range(len(exps), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Phase Flow — Strategy Shift Across Game Stages",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = os.path.join(save_dir, "phase_flow.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4 — Comparative Transition Heatmap (all conditions on one figure)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_transition_heatmaps(all_seq, save_dir):
    """
    One 5×5 heatmap per experiment showing P(next | current).
    Arranged side-by-side for easy cross-condition comparison.
    """
    exps = list(all_seq.keys())
    n    = len(exps)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.4, rows * 3.5),
                             squeeze=False)

    for idx, exp in enumerate(exps):
        ax   = axes[idx // cols][idx % cols]
        seqs, won = all_seq[exp]
        s    = EXP_STYLE.get(exp, {"label": exp, "color": "#aaa"})

        T = np.zeros((N_CATS, N_CATS))
        for seq in seqs:
            for t in range(len(seq) - 1):
                T[seq[t], seq[t + 1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = T / row_sums

        im = ax.imshow(P, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(N_CATS)); ax.set_xticklabels(CAT_NAMES, rotation=35,
                                                          ha="right", fontsize=8)
        ax.set_yticks(range(N_CATS)); ax.set_yticklabels(CAT_NAMES, fontsize=8)
        ax.set_xlabel("Next action", fontsize=8)
        if idx % cols == 0:
            ax.set_ylabel("Current action", fontsize=8)

        # Annotate cells
        for i in range(N_CATS):
            for j in range(N_CATS):
                v = P[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=7.5, fontweight="bold",
                            color="white" if v > 0.55 else "black")

        wr = sum(won) / len(won) * 100 if won else 0
        ax.set_title(f"{exp}  Win {wr:.0f}%",
                     fontsize=9, color=s["color"], fontweight="bold")

    for idx in range(len(exps), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.88,
                        bottom=0.08, hspace=0.55, wspace=0.35)
    cax = fig.add_axes([0.90, 0.28, 0.015, 0.44])
    fig.colorbar(im, cax=cax, label="Transition prob.")
    fig.suptitle("Action Transition Matrix — P(next action | current action)",
                 fontsize=13, fontweight="bold")
    path = os.path.join(save_dir, "transition_heatmaps.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sequential decision-making plots for Splendor DRL agents.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--plot-dir",       default="plots")
    parser.add_argument("--n-games",        type=int, default=50)
    parser.add_argument("--max-step",       type=int, default=50,
                        help="Max turns shown in ethogram")
    parser.add_argument("--exps",           nargs="+",
                        default=["C0","C1","C2","C3","C4","C5","C6"])
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"\n Collecting game sequences ({args.n_games} games x {len(args.exps)} experiments)...\n")
    all_seq = {}
    for exp in args.exps:
        seqs, won = collect_sequences(args.checkpoint_dir, exp, args.n_games)
        if seqs:
            all_seq[exp] = (seqs, won)

    if not all_seq:
        print("  No sequences collected — check checkpoint directory.")
        return

    print(f"\n Generating plots for {len(all_seq)} experiments…\n")
    plot_ethogram(all_seq, args.plot_dir, max_step=args.max_step)
    plot_transitions(all_seq, args.plot_dir)
    plot_phase_flow(all_seq, args.plot_dir)
    plot_transition_heatmaps(all_seq, args.plot_dir)

    print(f"\n All sequential plots saved to: {args.plot_dir}/\n")
    print("  Files:")
    for f in ["ethogram.png", "transitions.png", "phase_flow.png",
              "transition_heatmaps.png"]:
        p = os.path.join(args.plot_dir, f)
        exists = "OK" if os.path.exists(p) else "MISSING"
        print(f"    {exists} {f}")


if __name__ == "__main__":
    main()
