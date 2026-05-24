"""
Reward shaping functions for Splendor:
  • score_only_reward  — basic prestige + win/loss bonus
  • event_shaped_reward — event-based reward (Rinascimento-inspired)
"""

import numpy as np

from card_data import NUM_GEM_COLORS, get_card_prestige


# ─── Score-Only Reward ───────────────────────────────────────────────
def score_only_reward(prev_state, curr_state, events, won):
    """
    Reward = prestige gained this turn + win/loss bonus.
    """
    prestige_delta = curr_state.prestige - prev_state.prestige
    reward = float(prestige_delta)
    if won is True:
        reward += 10.0
    elif won is False:
        reward -= 10.0
    return reward


# ─── Event-Based Reward Shaping ──────────────────────────────────────
# Weights for different event types (hand-tuned based on literature)
EVENT_WEIGHTS = {
    "gem_taken":        0.002,  # per gem token (tiny — don't reward hoarding)
    "card_bought_l1":   0.05,   # buying a Level-1 card (engine building)
    "card_bought_l2":   0.10,   # buying a Level-2 card
    "card_bought_l3":   0.20,   # buying a Level-3 card
    "prestige_gained":  1.00,   # per prestige point (matches score_only scale)
    "noble_gained":     3.00,   # noble tile visit (strong — nobles are rare & worth it)
    "card_reserved":    0.01,   # reserving a card (tiny — don't encourage hoarding)
    "win_bonus":       10.00,   # winning the game
    "loss_penalty":   -10.00,   # losing the game
}


def _apply_weights(events, won, weights):
    """Generic helper: apply a weight dict to events."""
    reward = 0.0
    gems = events.get("gems_taken", np.zeros(5))
    reward += weights["gem_taken"] * gems.sum()
    card = events.get("card_bought")
    if card is not None:
        level = card[0]
        if level == 1:
            reward += weights["card_bought_l1"]
        elif level == 2:
            reward += weights["card_bought_l2"]
        else:
            reward += weights["card_bought_l3"]
    pts = events.get("prestige_gained", 0)
    reward += weights["prestige_gained"] * pts
    if events.get("noble_gained", False):
        reward += weights["noble_gained"]
    if events.get("card_reserved") is not None:
        reward += weights["card_reserved"]
    if won is True:
        reward += weights["win_bonus"]
    elif won is False:
        reward += weights["loss_penalty"]
    return reward


def event_shaped_reward(prev_state, curr_state, events, won):
    """
    C2 baseline event reward (Rinascimento-inspired, hand-tuned weights).
    """
    return _apply_weights(events, won, EVENT_WEIGHTS)


# ─── C4: Card Rush ───────────────────────────────────────────────────
# Hypothesis: reducing noble reward discourages chasing nobles and
# keeps the agent focused on card purchases as the primary path to points.
CARD_RUSH_WEIGHTS = {
    "gem_taken":       0.002,
    "card_bought_l1":  0.05,
    "card_bought_l2":  0.10,
    "card_bought_l3":  0.20,
    "prestige_gained": 1.00,
    "noble_gained":    0.50,    # reduced — don't chase nobles, focus on cards
    "card_reserved":   0.01,
    "win_bonus":      10.00,
    "loss_penalty":  -10.00,
}

def card_rush_reward(prev_state, curr_state, events, won):
    """
    C4: High card-buy rewards, zero gem/reserve reward.
    Tests whether pushing the agent toward immediate card purchases helps.
    """
    return _apply_weights(events, won, CARD_RUSH_WEIGHTS)


# ─── C5: Noble Hunter ────────────────────────────────────────────────
# Hypothesis: nobles are worth 3 prestige each and ignored by basic agents.
# Doubling the noble reward should guide the agent toward targeting nobles.
NOBLE_HUNTER_WEIGHTS = {
    "gem_taken":       0.002,
    "card_bought_l1":  0.05,
    "card_bought_l2":  0.10,
    "card_bought_l3":  0.20,
    "prestige_gained": 1.00,
    "noble_gained":    6.00,    # 2x original — strong signal to target nobles
    "card_reserved":   0.01,
    "win_bonus":      10.00,
    "loss_penalty":  -10.00,
}

def noble_hunter_reward(prev_state, curr_state, events, won):
    """
    C5: Double noble reward, emphasis on L3 cards.
    Tests whether targeting nobles leads to faster wins.
    """
    return _apply_weights(events, won, NOBLE_HUNTER_WEIGHTS)


# ─── C6: Balanced Dense ──────────────────────────────────────────────
# Hypothesis: the original event weights may be too sparse (very small
# intermediate rewards). Amplifying all of them provides a denser signal
# that may help the agent learn faster in early episodes.
BALANCED_DENSE_WEIGHTS = {
    "gem_taken":       0.010,   # 5x original
    "card_bought_l1":  0.10,    # 2x original
    "card_bought_l2":  0.20,    # 2x original
    "card_bought_l3":  0.40,    # 2x original
    "prestige_gained": 1.00,
    "noble_gained":    4.00,    # slightly higher
    "card_reserved":   0.02,    # 2x original
    "win_bonus":      10.00,
    "loss_penalty":  -10.00,
}

def balanced_dense_reward(prev_state, curr_state, events, won):
    """
    C6: All intermediate rewards amplified ~2-5x.
    Tests whether denser reward signal accelerates learning.
    """
    return _apply_weights(events, won, BALANCED_DENSE_WEIGHTS)
