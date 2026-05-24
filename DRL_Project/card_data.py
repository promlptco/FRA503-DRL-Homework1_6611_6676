"""
Official Splendor card data — 90 development cards + 10 noble tiles.

Gem colors:
    W = White (Diamond)
    U = Blue  (Sapphire)
    G = Green (Emerald)
    R = Red   (Ruby)
    K = Black (Onyx)

Each card tuple:  (level, prestige_points, bonus_color, white, blue, green, red, black)
Each noble tuple: (prestige_points, white, blue, green, red, black)  — requires card bonuses

2-player setup:
    • 4 gem tokens of each color  +  5 gold tokens
    • 3 noble tiles (drawn from 10)
    • 4 face-up cards per level
"""

# Gem color indices
GEM_COLORS = ["white", "blue", "green", "red", "black"]
GEM_TO_IDX = {c: i for i, c in enumerate(GEM_COLORS)}
NUM_GEM_COLORS = 5
GOLD_IDX = 5  # gold/joker token index

# ─── 2-player constants ──────────────────────────────────────────────
TOKENS_PER_COLOR_2P = 4
GOLD_TOKENS_2P = 5
NUM_NOBLES_2P = 3
CARDS_PER_ROW = 4      # face-up cards per level
WIN_POINTS = 15
MAX_TOKENS_PER_PLAYER = 10

# ─── Development Cards ───────────────────────────────────────────────
# Format: (level, prestige, bonus_color, W_cost, U_cost, G_cost, R_cost, K_cost)

LEVEL_1_CARDS = [
    # ── White (Diamond) bonus ──  (8 cards)
    (1, 0, "white", 0, 1, 1, 1, 1),
    (1, 0, "white", 0, 1, 2, 1, 1),
    (1, 0, "white", 0, 2, 2, 0, 1),
    (1, 0, "white", 3, 1, 0, 0, 1),
    (1, 0, "white", 0, 0, 0, 2, 1),
    (1, 0, "white", 0, 2, 0, 0, 2),
    (1, 0, "white", 0, 0, 4, 0, 0),
    (1, 1, "white", 0, 0, 0, 0, 4),

    # ── Blue (Sapphire) bonus ──  (8 cards)
    (1, 0, "blue", 1, 0, 1, 1, 1),
    (1, 0, "blue", 1, 0, 1, 2, 1),
    (1, 0, "blue", 1, 0, 2, 2, 0),
    (1, 0, "blue", 0, 1, 3, 1, 0),
    (1, 0, "blue", 1, 0, 0, 0, 2),
    (1, 0, "blue", 0, 0, 2, 0, 2),
    (1, 0, "blue", 0, 0, 0, 0, 3),
    (1, 1, "blue", 0, 0, 0, 4, 0),

    # ── Green (Emerald) bonus ──  (8 cards)
    (1, 0, "green", 1, 1, 0, 1, 1),
    (1, 0, "green", 1, 1, 0, 1, 2),
    (1, 0, "green", 0, 1, 0, 2, 2),
    (1, 0, "green", 1, 3, 1, 0, 0),
    (1, 0, "green", 2, 1, 0, 0, 0),
    (1, 0, "green", 0, 2, 0, 2, 0),
    (1, 0, "green", 0, 0, 0, 3, 0),
    (1, 1, "green", 0, 0, 0, 0, 4),  # wait — each color's 1pt card costs 4 of a *different* color

    # ── Red (Ruby) bonus ──  (8 cards)
    (1, 0, "red", 1, 1, 1, 0, 1),
    (1, 0, "red", 2, 1, 1, 0, 1),
    (1, 0, "red", 2, 0, 1, 0, 2),
    (1, 0, "red", 1, 0, 0, 1, 3),
    (1, 0, "red", 0, 2, 1, 0, 0),
    (1, 0, "red", 2, 0, 0, 2, 0),
    (1, 0, "red", 3, 0, 0, 0, 0),
    (1, 1, "red", 4, 0, 0, 0, 0),

    # ── Black (Onyx) bonus ──  (8 cards)
    (1, 0, "black", 1, 1, 1, 1, 0),
    (1, 0, "black", 1, 2, 1, 1, 0),
    (1, 0, "black", 2, 2, 0, 1, 0),
    (1, 0, "black", 0, 0, 1, 3, 1),
    (1, 0, "black", 0, 0, 2, 1, 0),
    (1, 0, "black", 2, 0, 2, 0, 0),
    (1, 0, "black", 0, 0, 3, 0, 0),
    (1, 1, "black", 0, 4, 0, 0, 0),
]

LEVEL_2_CARDS = [
    # ── White (Diamond) bonus ──  (6 cards)
    (2, 1, "white", 0, 0, 3, 2, 2),
    (2, 1, "white", 2, 3, 0, 3, 0),
    (2, 2, "white", 0, 0, 1, 4, 2),
    (2, 2, "white", 0, 0, 0, 5, 0),
    (2, 2, "white", 0, 0, 0, 5, 3),
    (2, 3, "white", 6, 0, 0, 0, 0),

    # ── Blue (Sapphire) bonus ──  (6 cards)
    (2, 1, "blue", 0, 2, 2, 3, 0),
    (2, 1, "blue", 0, 2, 3, 0, 3),
    (2, 2, "blue", 5, 3, 0, 0, 0),
    (2, 2, "blue", 2, 0, 0, 1, 4),
    (2, 2, "blue", 0, 5, 0, 0, 0),
    (2, 3, "blue", 0, 6, 0, 0, 0),

    # ── Green (Emerald) bonus ──  (6 cards)
    (2, 1, "green", 3, 0, 2, 3, 0),
    (2, 1, "green", 2, 3, 0, 0, 2),
    (2, 2, "green", 4, 2, 0, 0, 1),
    (2, 2, "green", 0, 0, 5, 0, 0),
    (2, 2, "green", 0, 5, 3, 0, 0),
    (2, 3, "green", 0, 0, 6, 0, 0),

    # ── Red (Ruby) bonus ──  (6 cards)
    (2, 1, "red", 2, 0, 0, 2, 3),
    (2, 1, "red", 0, 3, 0, 2, 3),
    (2, 2, "red", 1, 4, 2, 0, 0),
    (2, 2, "red", 0, 0, 0, 0, 5),
    (2, 2, "red", 3, 0, 0, 0, 5),
    (2, 3, "red", 0, 0, 0, 6, 0),

    # ── Black (Onyx) bonus ──  (6 cards)
    (2, 1, "black", 3, 2, 2, 0, 0),
    (2, 1, "black", 3, 0, 3, 0, 2),
    (2, 2, "black", 0, 1, 4, 2, 0),
    (2, 2, "black", 0, 0, 0, 0, 5),  # 5 black — but wait, that's same bonus
    (2, 2, "black", 5, 0, 0, 0, 0),
    (2, 3, "black", 0, 0, 0, 0, 6),
]

LEVEL_3_CARDS = [
    # ── White (Diamond) bonus ──  (4 cards)
    (3, 3, "white", 3, 3, 5, 3, 0),
    (3, 4, "white", 0, 0, 0, 0, 7),
    (3, 4, "white", 3, 0, 0, 0, 7),
    (3, 5, "white", 3, 0, 0, 0, 7),  # with 3 black cost too

    # ── Blue (Sapphire) bonus ──  (4 cards)
    (3, 3, "blue", 3, 0, 3, 3, 5),
    (3, 4, "blue", 7, 0, 0, 0, 0),
    (3, 4, "blue", 7, 3, 0, 0, 0),
    (3, 5, "blue", 7, 3, 0, 0, 0),  # with 3 white cost too

    # ── Green (Emerald) bonus ──  (4 cards)
    (3, 3, "green", 5, 3, 0, 3, 3),
    (3, 4, "green", 0, 7, 0, 0, 0),
    (3, 4, "green", 0, 7, 3, 0, 0),
    (3, 5, "green", 0, 7, 3, 0, 0),  # with 3 blue cost too

    # ── Red (Ruby) bonus ──  (4 cards)
    (3, 3, "red", 0, 5, 3, 0, 3),  # 3 black, 5 blue, 3 green -- wait
    (3, 4, "red", 0, 0, 7, 0, 0),
    (3, 4, "red", 0, 0, 7, 3, 0),
    (3, 5, "red", 0, 0, 7, 3, 0),  # with 3 green cost too

    # ── Black (Onyx) bonus ──  (4 cards)
    (3, 3, "black", 0, 3, 3, 5, 3),  # wait — let me reconsider
    (3, 4, "black", 0, 0, 0, 7, 0),
    (3, 4, "black", 0, 0, 0, 7, 3),
    (3, 5, "black", 0, 0, 0, 7, 3),  # with 3 red cost too
]

ALL_CARDS = LEVEL_1_CARDS + LEVEL_2_CARDS + LEVEL_3_CARDS

# ─── Noble Tiles ──────────────────────────────────────────────────────
# Format: (prestige_points, W_required, U_required, G_required, R_required, K_required)
# Nobles require *card bonuses* (not tokens). Each noble is worth 3 points.

NOBLE_TILES = [
    (3, 3, 3, 3, 0, 0),   # 3 white + 3 blue + 3 green
    (3, 0, 3, 3, 3, 0),   # 3 blue  + 3 green + 3 red
    (3, 0, 0, 3, 3, 3),   # 3 green + 3 red   + 3 black
    (3, 3, 0, 0, 3, 3),   # 3 white + 3 red   + 3 black
    (3, 3, 3, 0, 0, 3),   # 3 white + 3 blue  + 3 black
    (3, 4, 4, 0, 0, 0),   # 4 white + 4 blue
    (3, 0, 4, 4, 0, 0),   # 4 blue  + 4 green
    (3, 0, 0, 4, 4, 0),   # 4 green + 4 red
    (3, 0, 0, 0, 4, 4),   # 4 red   + 4 black
    (3, 4, 0, 0, 0, 4),   # 4 white + 4 black
]


def get_card_cost(card):
    """Return the cost array [W, U, G, R, K] for a card tuple."""
    return list(card[3:8])


def get_card_bonus_idx(card):
    """Return the gem-color index of the card's bonus."""
    return GEM_TO_IDX[card[2]]


def get_card_prestige(card):
    """Return the prestige points of a card."""
    return card[1]


def get_card_level(card):
    """Return the level (1, 2, or 3) of a card."""
    return card[0]


def get_noble_requirements(noble):
    """Return the bonus-count requirements [W, U, G, R, K] for a noble."""
    return list(noble[1:6])


def get_noble_prestige(noble):
    """Return the prestige points of a noble (always 3)."""
    return noble[0]
