"""
Splendor 2-player Gymnasium Environment.

State vector (213 dimensions):
    Face-up cards:  12 cards × 11 features = 132 dims
    Noble tiles:     3 nobles × 6 features =  18 dims
    Board supply:    6 dims (5 gem colors + gold)
    Agent state:    45 dims (tokens[6] + bonuses[5] + prestige[1]
                             + reserved_cards[3×11])
    Opponent state: 12 dims (tokens[6] + bonuses[5] + prestige[1])
    Total = 132 + 18 + 6 + 45 + 12 = 213

Action space (50 discrete actions):
    0–14:  Take gems   (C(5,3)=10 three-diff + C(5,1)=5 two-same = 15)
    15–29: Buy a card   (4 face-up × 3 levels + 3 reserved = 15)
    30–44: Reserve card (4 face-up × 3 levels + 3 from deck  = 15)
    45–49: Discard a gem color (when >10 tokens, pick color to return; one per gem color)
"""

import copy
import random
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from card_data import (
    ALL_CARDS,
    CARDS_PER_ROW,
    GEM_COLORS,
    GEM_TO_IDX,
    GOLD_IDX,
    GOLD_TOKENS_2P,
    LEVEL_1_CARDS,
    LEVEL_2_CARDS,
    LEVEL_3_CARDS,
    MAX_TOKENS_PER_PLAYER,
    NOBLE_TILES,
    NUM_GEM_COLORS,
    NUM_NOBLES_2P,
    TOKENS_PER_COLOR_2P,
    WIN_POINTS,
    get_card_bonus_idx,
    get_card_cost,
    get_card_level,
    get_card_prestige,
    get_noble_prestige,
    get_noble_requirements,
)

# ─── Gem-taking action definitions ───────────────────────────────────
# 10 three-different combos + 5 two-same combos = 15
THREE_DIFF_COMBOS = list(combinations(range(NUM_GEM_COLORS), 3))  # 10
TWO_SAME_OPTIONS = list(range(NUM_GEM_COLORS))                    # 5
GEM_ACTIONS = []  # will hold (type, data) — type="3diff"/"2same"
for combo in THREE_DIFF_COMBOS:
    GEM_ACTIONS.append(("3diff", combo))
for color in TWO_SAME_OPTIONS:
    GEM_ACTIONS.append(("2same", color))
assert len(GEM_ACTIONS) == 15

# Card slot indices for buy / reserve
# 0-3: level-1 face-up,  4-7: level-2 face-up,  8-11: level-3 face-up
# 12-14: reserved cards (player's hand)
CARD_SLOTS = 15
# Reserve deck indices: 12=deck1, 13=deck2, 14=deck3
RESERVE_DECK_MAP = {12: 0, 13: 1, 14: 2}  # slot → deck level (0-indexed)

TOTAL_ACTIONS = 50  # 15 gem + 15 buy + 15 reserve + 5 discard

# ─── State dimensions ────────────────────────────────────────────────
CARD_FEATURES = 11     # cost[5] + bonus_color_onehot[5] + prestige[1]
STATE_DIM = (
    CARDS_PER_ROW * 3 * CARD_FEATURES   # 132   face-up cards
    + NUM_NOBLES_2P * (NUM_GEM_COLORS + 1)  # 18    nobles
    + (NUM_GEM_COLORS + 1)               # 6     board supply
    + 6 + 5 + 1 + 3 * CARD_FEATURES     # 45    agent state
    + 6 + 5 + 1                          # 12    opponent state
)  # = 213


class PlayerState:
    """Mutable state for one player."""

    def __init__(self):
        self.tokens = np.zeros(6, dtype=np.int32)      # W U G R K Gold
        self.bonuses = np.zeros(5, dtype=np.int32)      # permanent discounts
        self.prestige = 0
        self.owned_cards = []          # list of card tuples
        self.reserved_cards = []       # up to 3 card tuples
        self.nobles_visited = []       # noble tuples awarded

    def total_tokens(self):
        return int(self.tokens.sum())

    def copy(self):
        p = PlayerState()
        p.tokens = self.tokens.copy()
        p.bonuses = self.bonuses.copy()
        p.prestige = self.prestige
        p.owned_cards = list(self.owned_cards)
        p.reserved_cards = list(self.reserved_cards)
        p.nobles_visited = list(self.nobles_visited)
        return p


class SplendorEnv(gym.Env):
    """
    Two-player Splendor environment.

    The *agent* (player 0) interacts via ``step()``.
    The *opponent* (player 1) is driven externally by an ``opponent_policy``
    callable that receives ``(observation, legal_mask)`` and returns an action.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, opponent_policy=None, reward_fn=None, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.opponent_policy = opponent_policy
        self.reward_fn = reward_fn  # optional event-based reward function
        self.render_mode = render_mode

        # Will be set on reset
        self.decks = [[], [], []]       # remaining draw piles per level
        self.face_up = [[], [], []]     # 4 visible cards per level
        self.nobles = []                # active noble tiles
        self.board_tokens = np.zeros(6, dtype=np.int32)
        self.players = [PlayerState(), PlayerState()]
        self.current_player = 0
        self.done = False
        self.winner = None
        self.turn_count = 0
        self.last_events = {}  # for reward shaping
        # Discard phase state (when agent exceeds 10 tokens)
        self.discard_pending = False
        self._pending_prev_state = None
        self._pending_events = None
        self._pending_nobles_before = 0

    # ─── Reset ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Shuffle decks
        l1, l2, l3 = list(LEVEL_1_CARDS), list(LEVEL_2_CARDS), list(LEVEL_3_CARDS)
        rng.shuffle(l1)
        rng.shuffle(l2)
        rng.shuffle(l3)
        self.decks = [l1, l2, l3]

        # Deal face-up cards
        self.face_up = [[], [], []]
        for lvl in range(3):
            for _ in range(CARDS_PER_ROW):
                if self.decks[lvl]:
                    self.face_up[lvl].append(self.decks[lvl].pop())

        # Nobles
        nobles_copy = list(NOBLE_TILES)
        rng.shuffle(nobles_copy)
        self.nobles = nobles_copy[:NUM_NOBLES_2P]

        # Tokens
        self.board_tokens = np.zeros(6, dtype=np.int32)
        for i in range(NUM_GEM_COLORS):
            self.board_tokens[i] = TOKENS_PER_COLOR_2P
        self.board_tokens[GOLD_IDX] = GOLD_TOKENS_2P

        # Players
        self.players = [PlayerState(), PlayerState()]
        self.current_player = 0
        self.done = False
        self.winner = None
        self.turn_count = 0
        self.last_events = {}
        self.discard_pending = False
        self._pending_prev_state = None
        self._pending_events = None
        self._pending_nobles_before = 0

        obs = self._get_obs(player_idx=0)
        info = {"legal_mask": self.get_legal_mask(0)}
        return obs, info

    # ─── Step (agent action) ─────────────────────────────────────────
    def step(self, action):
        """Execute agent's action, then opponent's action. Return agent obs."""
        if self.done:
            return self._get_obs(0), 0.0, True, False, {"legal_mask": self.get_legal_mask(0)}

        p = self.players[0]
        legal = self.get_legal_mask(0)
        if not legal[action]:
            return self._get_obs(0), -1.0, True, False, {"legal_mask": legal}

        # ── Normal agent turn ──────────────────────────────────────────
        prev_state_agent = p.copy()
        nobles_before    = len(p.nobles_visited)
        events_agent     = self._execute_action(0, action)
        self.turn_count += 1

        # Token overflow → auto-discard immediately
        if p.total_tokens() > MAX_TOKENS_PER_PLAYER:
            self._auto_discard_overflow(0)

        # Complete turn normally
        self._check_nobles(0)
        events_agent["noble_gained"] = len(p.nobles_visited) > nobles_before

        terminated = False
        if p.prestige >= WIN_POINTS:
            self.done = True
            self.winner = 0
            terminated = True
        elif self.players[1].prestige >= WIN_POINTS:
            self.done = True
            self.winner = 1
            terminated = True

        if self.reward_fn is not None:
            reward = self.reward_fn(
                prev_state=prev_state_agent,
                curr_state=p,
                events=events_agent,
                won=(self.winner == 0) if terminated else None,
            )
        else:
            reward = float(p.prestige - prev_state_agent.prestige)
            if terminated:
                reward += 10.0 if self.winner == 0 else -10.0

        if terminated:
            return self._get_obs(0), reward, True, False, {"legal_mask": self.get_legal_mask(0)}

        # ── Opponent turn ──────────────────────────────────────────────
        if self.opponent_policy is not None:
            opp_obs   = self._get_obs(player_idx=1)
            opp_legal = self.get_legal_mask(1)
            if opp_legal.any():
                opp_action = self.opponent_policy(opp_obs, opp_legal, self)
                self._execute_action(1, opp_action)
                self._auto_discard_overflow(1)
                self._check_nobles(1)
            self.turn_count += 1

            if self.players[1].prestige >= WIN_POINTS:
                self.done = True
                self.winner = 1
                terminated = True
                reward += -10.0
            elif self.players[0].prestige >= WIN_POINTS:
                self.done = True
                self.winner = 0
                terminated = True
                reward += 10.0

        obs  = self._get_obs(0)
        info = {"legal_mask": self.get_legal_mask(0)}
        return obs, reward, terminated, False, info

    # ─── Execute Action ──────────────────────────────────────────────
    def _execute_action(self, player_idx, action):
        """Apply *action* for *player_idx*. Return event dict for reward shaping."""
        p = self.players[player_idx]
        events = {
            "gems_taken": np.zeros(5, dtype=np.int32),
            "card_bought": None,
            "card_reserved": None,
            "prestige_gained": 0,
            "noble_gained": False,
        }

        if 0 <= action < 15:
            # ── Take gems ──
            gem_type, gem_data = GEM_ACTIONS[action]
            if gem_type == "3diff":
                for c in gem_data:
                    if self.board_tokens[c] > 0:
                        self.board_tokens[c] -= 1
                        p.tokens[c] += 1
                        events["gems_taken"][c] += 1
            else:  # 2same
                c = gem_data
                take = min(2, self.board_tokens[c])
                self.board_tokens[c] -= take
                p.tokens[c] += take
                events["gems_taken"][c] += take

        elif 15 <= action < 30:
            # ── Buy card ──
            slot = action - 15
            card = self._get_card_at_slot(player_idx, slot)
            if card is not None:
                self._buy_card(player_idx, card, slot)
                events["card_bought"] = card
                events["prestige_gained"] = get_card_prestige(card)

        elif 30 <= action < 45:
            # ── Reserve card ──
            slot = action - 30
            card = self._reserve_card(player_idx, slot)
            if card is not None:
                events["card_reserved"] = card

        elif 45 <= action < 50:
            # ── Discard gem (overflow) ──
            color = action - 45
            # Map: 45→least valuable color etc. — simple: discard color index
            # This is a simplified discard — only used in edge cases
            if p.tokens[color] > 0:
                p.tokens[color] -= 1
                self.board_tokens[color] += 1

        return events

    def _auto_discard_overflow(self, player_idx):
        """If player has >10 tokens, auto-discard excess (least useful first)."""
        p = self.players[player_idx]
        while p.total_tokens() > MAX_TOKENS_PER_PLAYER:
            # Discard the gem color with the most tokens (heuristic)
            # Skip gold (index 5) unless it's the only option
            best = -1
            best_count = -1
            for c in range(NUM_GEM_COLORS):
                if p.tokens[c] > best_count:
                    best_count = p.tokens[c]
                    best = c
            if best >= 0 and p.tokens[best] > 0:
                p.tokens[best] -= 1
                self.board_tokens[best] += 1
            elif p.tokens[GOLD_IDX] > 0:
                p.tokens[GOLD_IDX] -= 1
                self.board_tokens[GOLD_IDX] += 1
            else:
                break

    def _get_card_at_slot(self, player_idx, slot):
        """Map slot index (0-14) to actual card object, or None."""
        if slot < 12:
            lvl = slot // CARDS_PER_ROW
            col = slot % CARDS_PER_ROW
            if col < len(self.face_up[lvl]):
                return self.face_up[lvl][col]
        else:
            # Reserved cards
            res_idx = slot - 12
            p = self.players[player_idx]
            if res_idx < len(p.reserved_cards):
                return p.reserved_cards[res_idx]
        return None

    def _can_afford(self, player_idx, card):
        """Check if player can afford a card (tokens + bonuses + gold)."""
        p = self.players[player_idx]
        cost = get_card_cost(card)
        gold_needed = 0
        for c in range(NUM_GEM_COLORS):
            effective = cost[c] - p.bonuses[c]
            if effective > 0:
                shortfall = effective - p.tokens[c]
                if shortfall > 0:
                    gold_needed += shortfall
        return gold_needed <= p.tokens[GOLD_IDX]

    def _buy_card(self, player_idx, card, slot):
        """Buy a card: pay tokens, gain bonus + prestige, refill slot."""
        p = self.players[player_idx]
        cost = get_card_cost(card)
        gold_used = 0

        for c in range(NUM_GEM_COLORS):
            effective = max(0, cost[c] - p.bonuses[c])
            if effective > 0:
                from_tokens = min(effective, p.tokens[c])
                p.tokens[c] -= from_tokens
                self.board_tokens[c] += from_tokens
                remaining = effective - from_tokens
                if remaining > 0:
                    gold_used += remaining

        p.tokens[GOLD_IDX] -= gold_used
        self.board_tokens[GOLD_IDX] += gold_used

        # Gain bonus and prestige
        p.bonuses[get_card_bonus_idx(card)] += 1
        p.prestige += get_card_prestige(card)
        p.owned_cards.append(card)

        # Remove from face-up or reserved
        if slot < 12:
            lvl = slot // CARDS_PER_ROW
            col = slot % CARDS_PER_ROW
            self.face_up[lvl].pop(col)
            if self.decks[lvl]:
                self.face_up[lvl].insert(col, self.decks[lvl].pop())
        else:
            res_idx = slot - 12
            p.reserved_cards.pop(res_idx)

    def _reserve_card(self, player_idx, slot):
        """Reserve a card, gain gold token if available. Returns card or None."""
        p = self.players[player_idx]
        if len(p.reserved_cards) >= 3:
            return None

        card = None
        if slot < 12:
            lvl = slot // CARDS_PER_ROW
            col = slot % CARDS_PER_ROW
            if col < len(self.face_up[lvl]):
                card = self.face_up[lvl].pop(col)
                if self.decks[lvl]:
                    self.face_up[lvl].insert(col, self.decks[lvl].pop())
        else:
            # Reserve from deck top (blind)
            deck_idx = slot - 12  # 0, 1, or 2
            if deck_idx < 3 and self.decks[deck_idx]:
                card = self.decks[deck_idx].pop()

        if card is not None:
            p.reserved_cards.append(card)
            if self.board_tokens[GOLD_IDX] > 0:
                self.board_tokens[GOLD_IDX] -= 1
                p.tokens[GOLD_IDX] += 1

        return card

    def _check_nobles(self, player_idx):
        """Award nobles if the player meets requirements."""
        p = self.players[player_idx]
        awarded = []
        for noble in self.nobles:
            reqs = get_noble_requirements(noble)
            if all(p.bonuses[c] >= reqs[c] for c in range(NUM_GEM_COLORS)):
                awarded.append(noble)

        for noble in awarded:
            self.nobles.remove(noble)
            p.nobles_visited.append(noble)
            p.prestige += get_noble_prestige(noble)

    # ─── Legal Action Mask ───────────────────────────────────────────
    def get_legal_mask(self, player_idx):
        """Return a boolean array of size TOTAL_ACTIONS."""
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
        p = self.players[player_idx]

        # During discard phase only discard actions are legal
        if player_idx == 0 and self.discard_pending:
            for c in range(NUM_GEM_COLORS):
                if p.tokens[c] > 0:
                    mask[45 + c] = True
            return mask

        # ── Gem-taking actions (0-14) ──
        for i, (gem_type, gem_data) in enumerate(GEM_ACTIONS):
            if gem_type == "3diff":
                # Need at least 1 token of each of the 3 colors
                if all(self.board_tokens[c] > 0 for c in gem_data):
                    mask[i] = True
            else:
                # 2 same: need ≥ 4 tokens of that color on the board
                c = gem_data
                if self.board_tokens[c] >= 4:
                    mask[i] = True

        # ── Buy actions (15-29) ──
        for slot in range(CARD_SLOTS):
            card = self._get_card_at_slot(player_idx, slot)
            if card is not None and self._can_afford(player_idx, card):
                mask[15 + slot] = True

        # ── Reserve actions (30-44) ──
        if len(p.reserved_cards) < 3:
            for slot in range(12):
                card = self._get_card_at_slot(player_idx, slot)
                if card is not None:
                    mask[30 + slot] = True
            # Reserve from deck top
            for deck_idx in range(3):
                if self.decks[deck_idx]:
                    mask[30 + 12 + deck_idx] = True

        # ── Discard actions (45-49) ── only if over limit
        if p.total_tokens() > MAX_TOKENS_PER_PLAYER:
            for c in range(NUM_GEM_COLORS):
                if p.tokens[c] > 0:
                    mask[45 + c] = True

        # If nothing is legal, allow all gem-taking with at least some token
        if not mask.any():
            for i, (gem_type, gem_data) in enumerate(GEM_ACTIONS):
                if gem_type == "3diff":
                    # Relaxed: at least some tokens available
                    if any(self.board_tokens[c] > 0 for c in gem_data):
                        mask[i] = True
                else:
                    c = gem_data
                    if self.board_tokens[c] > 0:
                        mask[i] = True

        return mask

    # ─── Observation / State Encoding ────────────────────────────────
    def _get_obs(self, player_idx):
        """Build a normalized 213-dim observation vector."""
        obs = []
        max_cost = 7.0  # for normalization

        # 1) Face-up cards (12 × 11 = 132 dims)
        for lvl in range(3):
            for col in range(CARDS_PER_ROW):
                if col < len(self.face_up[lvl]):
                    card = self.face_up[lvl][col]
                    obs.extend(self._encode_card(card, max_cost))
                else:
                    obs.extend([0.0] * CARD_FEATURES)

        # 2) Nobles (3 × 6 = 18 dims)
        for i in range(NUM_NOBLES_2P):
            if i < len(self.nobles):
                noble = self.nobles[i]
                reqs = get_noble_requirements(noble)
                obs.extend([r / 4.0 for r in reqs])
                obs.append(get_noble_prestige(noble) / 5.0)
            else:
                obs.extend([0.0] * (NUM_GEM_COLORS + 1))

        # 3) Board supply (6 dims)
        for i in range(6):
            obs.append(self.board_tokens[i] / 7.0)

        # 4) Agent state (45 dims)
        agent = self.players[player_idx]
        # Tokens (6)
        for i in range(6):
            obs.append(agent.tokens[i] / 7.0)
        # Bonuses (5)
        for i in range(5):
            obs.append(agent.bonuses[i] / 7.0)
        # Prestige (1)
        obs.append(agent.prestige / 20.0)
        # Reserved cards (3 × 11 = 33)
        for i in range(3):
            if i < len(agent.reserved_cards):
                obs.extend(self._encode_card(agent.reserved_cards[i], max_cost))
            else:
                obs.extend([0.0] * CARD_FEATURES)

        # 5) Opponent state (12 dims)
        opp_idx = 1 - player_idx
        opp = self.players[opp_idx]
        for i in range(6):
            obs.append(opp.tokens[i] / 7.0)
        for i in range(5):
            obs.append(opp.bonuses[i] / 7.0)
        obs.append(opp.prestige / 20.0)

        return np.array(obs, dtype=np.float32)

    def _encode_card(self, card, max_cost=7.0):
        """Encode a card as 11 floats: cost[5] + bonus_onehot[5] + prestige."""
        cost = get_card_cost(card)
        encoded = [c / max_cost for c in cost]
        bonus_oh = [0.0] * 5
        bonus_oh[get_card_bonus_idx(card)] = 1.0
        encoded.extend(bonus_oh)
        encoded.append(get_card_prestige(card) / 5.0)
        return encoded

    # ─── Render ──────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self):
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Turn {self.turn_count}  |  Board Tokens: {self._tokens_str(self.board_tokens)}")
        lines.append("-" * 60)

        # Nobles
        noble_strs = []
        for n in self.nobles:
            reqs = get_noble_requirements(n)
            req_parts = [f"{GEM_COLORS[i][0].upper()}:{reqs[i]}" for i in range(5) if reqs[i] > 0]
            noble_strs.append(f"[{' '.join(req_parts)} → {get_noble_prestige(n)}pt]")
        lines.append(f"  Nobles: {' '.join(noble_strs)}")

        # Face-up cards
        level_names = ["Lv1", "Lv2", "Lv3"]
        for lvl in range(3):
            card_strs = []
            for card in self.face_up[lvl]:
                cost = get_card_cost(card)
                cost_parts = [f"{GEM_COLORS[i][0].upper()}{cost[i]}" for i in range(5) if cost[i] > 0]
                bonus = GEM_COLORS[get_card_bonus_idx(card)][0].upper()
                pts = get_card_prestige(card)
                card_strs.append(f"({'/'.join(cost_parts)}→{bonus} {pts}pt)")
            lines.append(f"  {level_names[lvl]}: {' '.join(card_strs)}  [deck:{len(self.decks[lvl])}]")

        lines.append("-" * 60)

        # Players
        for pi in range(2):
            p = self.players[pi]
            role = "AGENT" if pi == 0 else "OPP  "
            lines.append(
                f"  {role} | Tokens: {self._tokens_str(p.tokens)} "
                f"| Bonuses: {self._bonuses_str(p.bonuses)} "
                f"| Prestige: {p.prestige} "
                f"| Reserved: {len(p.reserved_cards)}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _tokens_str(tokens):
        colors = "WUGRK$"
        parts = [f"{colors[i]}:{tokens[i]}" for i in range(6)]
        return " ".join(parts)

    @staticmethod
    def _bonuses_str(bonuses):
        colors = "WUGRK"
        parts = [f"{colors[i]}:{bonuses[i]}" for i in range(5)]
        return " ".join(parts)

    # ─── Helpers for external use (visualization, opponents) ─────────
    def get_game_state(self):
        """Return a dict snapshot of the full game state for rendering."""
        return {
            "face_up": [list(row) for row in self.face_up],
            "decks_remaining": [len(d) for d in self.decks],
            "nobles": list(self.nobles),
            "board_tokens": self.board_tokens.copy(),
            "players": [p.copy() for p in self.players],
            "turn_count": self.turn_count,
            "done": self.done,
            "winner": self.winner,
        }

    def get_action_description(self, action):
        """Human-readable description of an action."""
        if 0 <= action < 15:
            gem_type, gem_data = GEM_ACTIONS[action]
            if gem_type == "3diff":
                colors = [GEM_COLORS[c] for c in gem_data]
                return f"Take 3 gems: {', '.join(colors)}"
            else:
                return f"Take 2 {GEM_COLORS[gem_data]} gems"
        elif 15 <= action < 30:
            slot = action - 15
            if slot < 12:
                lvl = slot // 4 + 1
                col = slot % 4 + 1
                return f"Buy Level-{lvl} card #{col}"
            else:
                return f"Buy reserved card #{slot - 11}"
        elif 30 <= action < 45:
            slot = action - 30
            if slot < 12:
                lvl = slot // 4 + 1
                col = slot % 4 + 1
                return f"Reserve Level-{lvl} card #{col}"
            else:
                return f"Reserve from Level-{slot - 11} deck"
        elif 45 <= action < 50:
            color = action - 45
            return f"Discard {GEM_COLORS[color]} gem"
        return f"Unknown action {action}"
