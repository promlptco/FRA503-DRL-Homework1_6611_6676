"""
Opponent policies for Splendor:
  random_opponent    -- uniformly random legal action
  greedy_opponent    -- heuristic: buy best > take gems > reserve
  SelfPlayOpponent   -- pool of past agent snapshots (self-play curriculum)
"""

import random
import numpy as np
from card_data import NUM_GEM_COLORS, get_card_cost, get_card_prestige


def random_opponent(obs, legal_mask, env):
    legal = np.where(legal_mask)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


def greedy_opponent(obs, legal_mask, env):
    player_idx = 1
    p = env.players[player_idx]

    # Buy highest-prestige affordable card
    best_buy, best_pts = None, -1
    for slot in range(15):
        action = 15 + slot
        if not legal_mask[action]:
            continue
        card = env._get_card_at_slot(player_idx, slot)
        if card is not None:
            pts = get_card_prestige(card)
            if pts > best_pts:
                best_pts, best_buy = pts, action
    if best_buy is not None:
        return best_buy

    # Take gems toward cheapest card
    target_card, min_def = None, float("inf")
    for lvl in range(3):
        for card in env.face_up[lvl]:
            cost = get_card_cost(card)
            deficit = sum(max(0, cost[c] - p.bonuses[c] - p.tokens[c])
                          for c in range(NUM_GEM_COLORS))
            if deficit < min_def:
                min_def, target_card = deficit, card

    if target_card is not None:
        cost = get_card_cost(target_card)
        needed = [c for c in range(NUM_GEM_COLORS)
                  if max(0, cost[c] - p.bonuses[c] - p.tokens[c]) > 0]
        best_gem, best_ov = None, -1
        for i in range(15):
            if not legal_mask[i]:
                continue
            from splendor_env import GEM_ACTIONS
            gem_type, gem_data = GEM_ACTIONS[i]
            ov = (sum(1 for c in gem_data if c in needed)
                  if gem_type == "3diff" else (2 if gem_data in needed else 0))
            if ov > best_ov:
                best_ov, best_gem = ov, i
        if best_gem is not None:
            return best_gem

    # Reserve high-value card
    for slot in range(12):
        action = 30 + slot
        if legal_mask[action]:
            return action

    legal = np.where(legal_mask)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


# --- Self-Play Opponent -----------------------------------------------
class SelfPlayOpponent:
    """
    Opponent that plays using a randomly-selected past agent snapshot.

    How it works:
      1. Every SNAPSHOT_INTERVAL episodes, add_snapshot() stores the agent's
         current online-net weights in a pool (FIFO, max pool_size snapshots).
      2. At the start of each episode, select_episode_net() picks one snapshot
         at random and loads it into a frozen evaluation network.
      3. When the pool is empty, falls back to greedy_opponent.

    This creates a natural curriculum: early episodes face greedy, later
    episodes face increasingly competent versions of the agent itself.
    """

    def __init__(self, fallback_opponent=None, pool_size=10):
        self._pool        = []
        self._pool_size   = pool_size
        self._fallback    = fallback_opponent or greedy_opponent
        self._episode_net = None
        self._device      = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def add_snapshot(self, state_dict, use_dueling=False):
        import copy
        self._pool.append((copy.deepcopy(state_dict), use_dueling))
        if len(self._pool) > self._pool_size:
            self._pool.pop(0)

    def select_episode_net(self):
        if not self._pool:
            self._episode_net = None
            return
        import torch
        state_dict, use_dueling = random.choice(self._pool)
        device = self._get_device()
        if use_dueling:
            from agent import DuelingQNetwork
            net = DuelingQNetwork().to(device)
        else:
            from agent import QNetwork
            net = QNetwork().to(device)
        net.load_state_dict(state_dict)
        net.eval()
        self._episode_net = net

    def __call__(self, obs, legal_mask, env):
        if self._episode_net is None:
            return self._fallback(obs, legal_mask, env)
        import torch
        device = self._get_device()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q     = self._episode_net(obs_t).squeeze(0).cpu().numpy()
        q[~legal_mask] = -np.inf
        return int(np.argmax(q))
