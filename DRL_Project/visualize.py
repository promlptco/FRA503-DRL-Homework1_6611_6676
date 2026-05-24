"""
Pygame Visualization for Splendor  (v4 - animations)
  Mode 1: AI vs AI
  Mode 2: AI vs Human  — gem staging with Confirm/Clear

Animations:
  • Gems fly in an arc from the supply to the player's token area
  • Cards flash green (buy) or blue (reserve) + floating text
  • Nobles pulse gold when awarded
  • AI input is blocked until animations finish so you can always see what happened
"""
import math
import time
from itertools import combinations

import numpy as np
import pygame

from agent import DDQNAgent
from card_data import (
    GEM_COLORS, GEM_TO_IDX, GOLD_IDX, NUM_GEM_COLORS,
    get_card_bonus_idx, get_card_cost, get_card_prestige,
    get_noble_prestige, get_noble_requirements,
)
from opponents import greedy_opponent
from splendor_env import CARDS_PER_ROW, GEM_ACTIONS, TOTAL_ACTIONS, SplendorEnv

# ── Gem display names ─────────────────────────────────────────────────
GEM_NAMES = ["Diamond", "Sapphire", "Emerald", "Ruby", "Onyx", "Gold"]
GEM_SHORT  = ["Dia",     "Sap",      "Eme",     "Rub",  "Ony",  "Gld"]

# ── Palette ───────────────────────────────────────────────────────────
C_BG       = (14,  17,  30 )
C_PANEL    = (22,  27,  48 )
C_PANEL2   = (30,  36,  62 )
C_BORDER   = (55,  65,  100)
C_BORDER_B = (90, 110, 170)
C_TEXT     = (215, 220, 235)
C_DIM      = (110, 118, 148)
C_GOLD     = (255, 210,  60)
C_GREEN    = ( 50, 200,  90)
C_RED      = (220,  65,  65)
C_SEL      = (255, 225,  50)
C_HOVER    = (180, 195, 255)
C_CONFIRM  = ( 35, 160,  75)
C_CONF_H   = ( 50, 210, 100)
C_CLEAR    = (155,  45,  45)
C_CLEAR_H  = (200,  70,  70)
C_SHADOW   = (  8,  10,  20)
C_CYAN     = ( 55, 215, 230)   # agent-select gem ring
C_PURPLE   = (180, 100, 255)   # reserve accent

GEM_COL = {
    "white": ((228, 228, 232), ( 25,  25,  30), (255, 255, 255)),
    "blue":  (( 45, 110, 215), (230, 240, 255), ( 80, 150, 255)),
    "green": (( 30, 165,  65), (210, 245, 215), ( 70, 210,  90)),
    "red":   ((200,  40,  40), (255, 210, 210), (255,  90,  90)),
    "black": (( 35,  35,  42), (200, 200, 210), ( 80,  80,  95)),
    "gold":  ((235, 185,  20), ( 30,  25,  10), (255, 230, 100)),
}

WIN_POINTS = 15
SCREEN_W, SCREEN_H = 1400, 880
CARD_W, CARD_H     = 112, 152
CARD_GAP           = 10
NOBLE_W, NOBLE_H   = 86, 86
GEM_R              = 25
SEL_R              = 26

_THREE_DIFF = list(combinations(range(NUM_GEM_COLORS), 3))

# ── Action helpers ────────────────────────────────────────────────────
def _action_type(action):
    if action is None:        return "unknown"
    if action < 15:           return "gem"
    if action < 30:           return "buy"
    if action < 45:           return "reserve"
    return "discard"

def _action_gem_colors(action):
    """Return list of gem color indices involved in a gem-take action."""
    if action < 10:
        return list(_THREE_DIFF[action])
    elif action < 15:
        return [action - 10, action - 10]
    return []

# ── Layout constants (pre-computed for animation targeting) ────────────
_BX, _BY_CARDS = 28, 60
_BY_SUPPLY = _BY_CARDS + 3 * (CARD_H + CARD_GAP + 10) + 6   # 582
_SUPPLY_Y  = _BY_SUPPLY + 44                                   # 626
_PX        = SCREEN_W - 318                                    # 1082

def _supply_center(gem_i):
    return (38 + gem_i * 70, _SUPPLY_Y)

def _token_center(player_idx, gem_i):
    py = 58 if player_idx == 0 else 475
    return (1105 + gem_i * 46, py + 91)

def _card_rect(slot):
    """Board rect for a face-up slot (0-11)."""
    lvl = slot // CARDS_PER_ROW
    col = slot % CARDS_PER_ROW
    row = 2 - lvl
    x = _BX + col * (CARD_W + CARD_GAP)
    y = _BY_CARDS + row * (CARD_H + CARD_GAP + 10)
    return pygame.Rect(x, y, CARD_W, CARD_H)

def _noble_rect(noble_i):
    nx = _BX + CARDS_PER_ROW * (CARD_W + CARD_GAP) + 14
    return pygame.Rect(nx, 60 + noble_i * (NOBLE_H + 8), NOBLE_W, NOBLE_H)

def _player_card_dest(player_idx):
    """Landing position for CardFlyAnim (player bonus area centre)."""
    py = 58 if player_idx == 0 else 475
    return (_PX + 155, py + 120)


# ═══════════════════════════════════════════════════════════════════════
#  Animation classes
# ═══════════════════════════════════════════════════════════════════════
class _AnimBase:
    def __init__(self, duration, delay=0):
        self.t     = -delay
        self.duration = duration
        self.done  = False

    def tick(self):
        self.t += 1
        if self.t >= self.duration:
            self.done = True

    def active(self):
        return self.t >= 0

    def _p(self):
        """Smooth progress 0→1."""
        raw = max(0.0, min(1.0, self.t / self.duration))
        return raw * raw * (3 - 2 * raw)   # smoothstep

    def draw(self, surf, fonts):
        pass


class GemFlyAnim(_AnimBase):
    """Gem circle arcs from supply → player token area."""
    def __init__(self, color_name, sx, sy, ex, ey, delay=0):
        super().__init__(duration=48, delay=delay)
        self.color_name = color_name
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey

    def draw(self, surf, fonts):
        if not self.active():
            return
        p  = self._p()
        x  = self.sx + (self.ex - self.sx) * p
        y  = self.sy + (self.ey - self.sy) * p - math.sin(math.pi * p) * 110
        r  = max(7, int(GEM_R * (1.0 - p * 0.35)))
        alpha = int(255 * (1.0 - max(0.0, (p - 0.72) / 0.28)))

        bc, tc, hc = GEM_COL[self.color_name]
        size = (r * 2 + 8, r * 2 + 8)
        s = pygame.Surface(size, pygame.SRCALPHA)
        cx2, cy2 = r + 4, r + 4
        pygame.draw.circle(s, (*bc, alpha), (cx2, cy2), r)
        pygame.draw.circle(s, (*hc, min(alpha, 200)), (cx2 - r//4, cy2 - r//4), r // 4)
        pygame.draw.circle(s, (*hc, alpha // 2), (cx2, cy2), r, 2)
        surf.blit(s, (int(x) - r - 4, int(y) - r - 4))


class FlashAnim(_AnimBase):
    """Glowing flash on a rect (card bought / reserved)."""
    def __init__(self, rect, color, duration=35):
        super().__init__(duration=duration)
        self.rect  = rect
        self.color = color

    def draw(self, surf, fonts):
        if not self.active():
            return
        p = self._p()
        pulse = math.sin(math.pi * p)
        fill_alpha   = int(140 * pulse)
        border_alpha = int(255 * pulse)

        s = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        s.fill((*self.color, fill_alpha))
        surf.blit(s, (self.rect.topleft))

        # Bright border
        exp = pygame.Rect(self.rect.x - 3, self.rect.y - 3,
                          self.rect.w + 6, self.rect.h + 6)
        pygame.draw.rect(surf, (*self.color, border_alpha),
                         exp, 3, border_radius=11)


class TextFloatAnim(_AnimBase):
    """Text that rises and fades out."""
    def __init__(self, text, x, y, color, duration=60, font_key="LG"):
        super().__init__(duration=duration)
        self.text     = text
        self.x, self.y = x, y
        self.color    = color
        self.font_key = font_key

    def draw(self, surf, fonts):
        if not self.active():
            return
        p     = self._p()
        alpha = int(255 * (1.0 - max(0.0, (p - 0.45) / 0.55)))
        dy    = int(p * 55)
        fnt   = fonts.get(self.font_key, fonts["MD"])
        t     = fnt.render(self.text, True, self.color)
        ts    = pygame.Surface(t.get_size(), pygame.SRCALPHA)
        ts.blit(t, (0, 0))
        ts.set_alpha(alpha)
        surf.blit(ts, (self.x - t.get_width() // 2, self.y - dy - t.get_height() // 2))


class NoblePulseAnim(_AnimBase):
    """Golden pulse on a noble tile."""
    def __init__(self, rect):
        super().__init__(duration=55)
        self.rect = rect

    def draw(self, surf, fonts):
        if not self.active():
            return
        p     = self._p()
        pulse = math.sin(math.pi * p)
        alpha = int(200 * pulse)
        exp   = pygame.Rect(self.rect.x - 5, self.rect.y - 5,
                            self.rect.w + 10, self.rect.h + 10)
        s = pygame.Surface((exp.w, exp.h), pygame.SRCALPHA)
        s.fill((255, 210, 60, alpha // 3))
        surf.blit(s, exp.topleft)
        pygame.draw.rect(surf, (255, 210, 60, alpha), exp, 4, border_radius=12)


class AgentSelectAnim(_AnimBase):
    """Expanding pulsing ring shown on the element the agent just chose."""
    def __init__(self, rect, color=(255, 255, 100), duration=28):
        super().__init__(duration=duration)
        self.rect  = rect
        self.color = color

    def draw(self, surf, fonts):
        if not self.active():
            return
        p     = self._p()
        pulse = math.sin(math.pi * p)
        for r_off in (0, 5, 10):
            alpha = int(230 * pulse * max(0.0, 1.0 - r_off / 12))
            exp   = pygame.Rect(self.rect.x - r_off, self.rect.y - r_off,
                                self.rect.w + r_off * 2, self.rect.h + r_off * 2)
            pygame.draw.rect(surf, (*self.color, alpha), exp, 2, border_radius=12)
        s = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        s.fill((*self.color, int(40 * pulse)))
        surf.blit(s, self.rect.topleft)


class CardFlyAnim(_AnimBase):
    """A shrunken card flies from the board slot to the player's panel."""
    def __init__(self, bonus_color_name, sx, sy, ex, ey, delay=0):
        super().__init__(duration=52, delay=delay)
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey
        self.bc, self.tc, self.hc = GEM_COL[bonus_color_name]

    def draw(self, surf, fonts):
        if not self.active():
            return
        p  = self._p()
        x  = self.sx + (self.ex - self.sx) * p
        y  = self.sy + (self.ey - self.sy) * p - math.sin(math.pi * p) * 90
        fw = max(6, int(CARD_W * (0.62 - p * 0.28)))
        fh = max(8, int(CARD_H * (0.62 - p * 0.28)))
        alpha = int(255 * (1.0 - max(0.0, (p - 0.72) / 0.28)))
        s = pygame.Surface((fw, fh), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.bc, alpha), (0, 0, fw, fh), border_radius=5)
        hdr_h = min(18, fh // 3)
        pygame.draw.rect(s, (*self.hc, min(alpha, 190)), (0, 0, fw, hdr_h), border_radius=5)
        pygame.draw.rect(s, (*self.bc, alpha // 2), (0, 0, fw, fh), 2, border_radius=5)
        pygame.draw.circle(s, (*self.hc, min(alpha, 160)),
                           (fw // 4, hdr_h // 2), max(2, fw // 6))
        surf.blit(s, (int(x) - fw // 2, int(y) - fh // 2))


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════
def gems_to_action(sel):
    if len(sel) == 2 and sel[0] == sel[1]:
        return 10 + sel[0]
    if len(sel) == 3 and len(set(sel)) == 3:
        key = tuple(sorted(sel))
        if key in _THREE_DIFF:
            return _THREE_DIFF.index(key)
    return None

def draw_shadow_rect(surf, rect, radius=8, offset=3):
    sr = pygame.Rect(rect.x + offset, rect.y + offset, rect.w, rect.h)
    pygame.draw.rect(surf, C_SHADOW, sr, border_radius=radius)

def draw_gem_circle(surf, cx, cy, r, name, count_str=None, font=None,
                    selected=False, dim=False):
    bc, tc, hc = GEM_COL[name]
    if dim:
        bc = tuple(max(0, v - 60) for v in bc)
    pygame.draw.circle(surf, bc, (cx, cy), r)
    pygame.draw.circle(surf, hc, (cx - r // 4, cy - r // 4), r // 4)
    if selected:
        pygame.draw.circle(surf, C_SEL, (cx, cy), r + 4, 3)
    else:
        pygame.draw.circle(surf, tuple(min(255, v + 40) for v in bc), (cx, cy), r, 2)
    if count_str and font:
        t = font.render(count_str, True, tc)
        surf.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))

def draw_progress_bar(surf, x, y, w, h, value, maxval, color, bg=(40, 40, 60)):
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=h // 2)
    fill = int(w * min(value, maxval) / maxval)
    if fill > 0:
        pygame.draw.rect(surf, color, (x, y, fill, h), border_radius=h // 2)
    pygame.draw.rect(surf, C_BORDER, (x, y, w, h), 1, border_radius=h // 2)


# ═══════════════════════════════════════════════════════════════════════
class SplendorVisualizer:
    def __init__(self, mode="ai_vs_ai", agent_path=None, ai_speed=1.0):
        pygame.init()
        self.screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Splendor  —  Deep Reinforcement Learning")
        self.clock    = pygame.time.Clock()
        self.mode     = mode
        self.ai_speed = max(0.3, ai_speed)

        self.fXL = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.fLG = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.fMD = pygame.font.SysFont("Segoe UI", 17)
        self.fSM = pygame.font.SysFont("Segoe UI", 14)
        self.fXS = pygame.font.SysFont("Segoe UI", 12)

        self._fonts = {"XL": self.fXL, "LG": self.fLG, "MD": self.fMD,
                       "SM": self.fSM, "XS": self.fXS}

        self.agent = DDQNAgent()
        if agent_path:
            try:
                self.agent.load(agent_path)
                print(f"Agent loaded: {agent_path}")
            except Exception as e:
                print(f"Could not load: {e}")
        self.agent.epsilon = 0.0

        # Animation state
        self.anims          = []
        self.ai_anim_block  = False   # block human input while AI anim plays
        self._coin_sound    = self._make_coin_sound()

        self.last_ai_desc   = ""
        self.last_ai_action = None

        if mode == "ai_vs_ai":
            opp = greedy_opponent
        else:
            _ag = self.agent
            _me = self
            def ai_opp(obs, lm, env):
                a = _ag.select_action(obs, lm)
                _me.last_ai_desc   = env.get_action_description(a)
                _me.last_ai_action = a
                return a
            opp = ai_opp

        self.env       = SplendorEnv(opponent_policy=opp, reward_fn=None)
        self.message   = ""
        self.game_over = False
        self.clickable = []
        self.gem_rects = {}
        self.staged    = []
        self.move_history = []   # list of move dicts for the history panel

    # ── Coin sound synthesizer ────────────────────────────────────────
    @staticmethod
    def _make_coin_sound():
        """Generate a short metallic coin-clink using numpy synthesis."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            sr  = 44100
            dur = 0.16
            t   = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = (np.sin(2 * np.pi * 900  * t) * 0.55
                  + np.sin(2 * np.pi * 1350 * t) * 0.30
                  + np.sin(2 * np.pi * 2200 * t) * 0.12)
            env  = np.exp(-t * 22)
            wave = np.clip(wave * env * 28000, -32767, 32767).astype(np.int16)
            stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
            return pygame.sndarray.make_sound(stereo)
        except Exception:
            return None

    def _play_coin(self):
        if self._coin_sound:
            self._coin_sound.play()

    # ── Move history ──────────────────────────────────────────────────
    def _record_move(self, player_idx, action, desc):
        if self.mode == "ai_vs_human":
            name = "You" if player_idx == 0 else "AI"
        else:
            name = "Agent" if player_idx == 0 else "Greedy"
        self.move_history.append({
            "turn":   self.env.turn_count,
            "player": player_idx,
            "name":   name,
            "action": action,
            "desc":   desc,
            "type":   _action_type(action),
            "gems":   _action_gem_colors(action),
        })

    # ── Animation queuing ─────────────────────────────────────────────
    def _queue_anims(self, action, player_idx):
        """Spawn animations for the given action by player_idx."""
        if action is None:
            return

        if action < 10:                    # take 3 different gems
            for i, c in enumerate(_THREE_DIFF[action]):
                sx, sy = _supply_center(c)
                ex, ey = _token_center(player_idx, c)
                gr = pygame.Rect(sx - GEM_R - 5, sy - GEM_R - 5,
                                 (GEM_R + 5) * 2, (GEM_R + 5) * 2)
                self.anims.append(AgentSelectAnim(gr, C_CYAN, duration=20))
                self.anims.append(GemFlyAnim(GEM_COLORS[c], sx, sy, ex, ey, delay=i * 14))
            self._play_coin()

        elif action < 15:                  # take 2 same gems
            c = action - 10
            sx, sy = _supply_center(c)
            ex, ey = _token_center(player_idx, c)
            gr = pygame.Rect(sx - GEM_R - 5, sy - GEM_R - 5,
                             (GEM_R + 5) * 2, (GEM_R + 5) * 2)
            self.anims.append(AgentSelectAnim(gr, C_CYAN, duration=20))
            self.anims.append(GemFlyAnim(GEM_COLORS[c], sx, sy, ex, ey, delay=0))
            self.anims.append(GemFlyAnim(GEM_COLORS[c], sx, sy, ex, ey, delay=16))
            self._play_coin()

        elif 15 <= action <= 26:           # buy face-up card
            slot = action - 15
            r    = _card_rect(slot)
            lvl  = slot // CARDS_PER_ROW
            col  = slot % CARDS_PER_ROW
            bonus_name = "white"
            try:
                if col < len(self.env.face_up[lvl]):
                    bonus_name = GEM_COLORS[get_card_bonus_idx(self.env.face_up[lvl][col])]
            except Exception:
                pass
            ex, ey = _player_card_dest(player_idx)
            self.anims.append(AgentSelectAnim(r, C_GREEN, duration=22))
            self.anims.append(FlashAnim(r, (50, 255, 100), duration=40))
            self.anims.append(CardFlyAnim(bonus_name, r.centerx, r.centery, ex, ey, delay=8))
            self.anims.append(TextFloatAnim("BOUGHT!", r.centerx, r.centery,
                                            C_GREEN, font_key="LG"))

        elif 27 <= action <= 29:           # buy reserved card
            px2 = _PX
            py2 = 58 if player_idx == 0 else 475
            self.anims.append(TextFloatAnim("BOUGHT RESERVED!", px2 + 150, py2 + 230,
                                            C_GREEN, font_key="LG"))

        elif 30 <= action <= 41:           # reserve face-up card
            slot = action - 30
            r    = _card_rect(slot)
            self.anims.append(AgentSelectAnim(r, C_PURPLE, duration=22))
            self.anims.append(FlashAnim(r, (80, 150, 255), duration=40))
            self.anims.append(TextFloatAnim("RESERVED", r.centerx, r.centery,
                                            C_HOVER, font_key="LG"))

        elif 42 <= action <= 44:           # reserve from deck
            self.anims.append(TextFloatAnim("RESERVED FROM DECK",
                                            SCREEN_W // 2 - 160, 420,
                                            C_HOVER, font_key="LG"))

        elif 45 <= action <= 47:           # discard gem
            c    = action - 45
            tx, ty = _token_center(player_idx, c)
            self.anims.append(TextFloatAnim(f"DISCARDED {GEM_NAMES[c]}",
                                            tx, ty, C_RED, font_key="SM"))
            self._play_coin()

    def _queue_noble_anims(self, player_idx, count):
        """Flash nobles when one is earned."""
        py2 = 58 if player_idx == 0 else 475
        for i in range(count):
            r = pygame.Rect(_PX + 10, py2 + 320, NOBLE_W, NOBLE_H)
            self.anims.append(NoblePulseAnim(r))
            self.anims.append(TextFloatAnim("NOBLE EARNED! +3",
                                            _PX + 60, py2 + 310,
                                            C_GOLD, duration=70, font_key="LG"))

    def _anims_blocking(self):
        """True while any animation is still running."""
        return any(not a.done for a in self.anims)

    # ═══════════════════════════════════════════════════════════════════
    def run(self):
        obs, info  = self.env.reset()
        mask       = info["legal_mask"]
        self.message = ("Your turn — select gems or click a card."
                        if self.mode == "ai_vs_human"
                        else "Watching agent…  ESC quit  R restart  +/- speed")
        running  = True
        last_ai  = time.time()
        prev_nobles = [len(p.nobles_visited) for p in self.env.players]

        while running:
            mp = pygame.mouse.get_pos()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        obs, info  = self.env.reset()
                        mask       = info["legal_mask"]
                        self.game_over    = False
                        self.staged       = []
                        self.last_ai_desc = ""
                        self.last_ai_action = None
                        self.anims        = []
                        self.ai_anim_block = False
                        self.move_history = []
                        prev_nobles = [len(p.nobles_visited) for p in self.env.players]
                        self.message = ("Game restarted! Your turn."
                                        if self.mode == "ai_vs_human"
                                        else "Game restarted!")
                    elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.ai_speed = min(self.ai_speed + 0.2, 5.0)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.ai_speed = max(self.ai_speed - 0.2, 0.3)

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if (self.mode == "ai_vs_human"
                            and not self.game_over
                            and not self.ai_anim_block):
                        act = self._human_click(ev.pos, mask)
                        if act is not None:
                            yd = self.env.get_action_description(act)
                            self.staged = []
                            self.last_ai_action = None
                            prev_nobles_snap = [len(p.nobles_visited)
                                                for p in self.env.players]

                            obs, _, done, _, info = self.env.step(act)
                            mask = info["legal_mask"]

                            # Animate the human's own action
                            self._queue_anims(act, 0)

                            # Record human move
                            self._record_move(0, act, yd)

                            # Record AI move (captured inside ai_opp callback)
                            if self.last_ai_action is not None:
                                ad = self.last_ai_desc or "?"
                                self._record_move(1, self.last_ai_action, ad)

                            # Noble check
                            for pidx in range(2):
                                gained = (len(self.env.players[pidx].nobles_visited)
                                          - prev_nobles_snap[pidx])
                                if gained > 0:
                                    self._queue_noble_anims(pidx, gained)
                            prev_nobles = [len(p.nobles_visited) for p in self.env.players]

                            if done:
                                self.game_over = True
                                self._winner_msg()
                            else:
                                # Queue AI animation
                                if self.last_ai_action is not None:
                                    self._queue_anims(self.last_ai_action, 1)
                                    self.ai_anim_block = True
                                ad = self.last_ai_desc or "thinking..."
                                self.message = f"You: {yd}     AI: {ad}     (animating…)"

            # ── Tick + prune animations ────────────────────────────────
            for a in self.anims:
                a.tick()
            self.anims = [a for a in self.anims if not a.done]

            # Unblock human once AI animation finishes
            if self.ai_anim_block and not self._anims_blocking():
                self.ai_anim_block = False
                self.message = "Your turn."

            # ── AI vs AI auto-step ─────────────────────────────────────
            if (self.mode == "ai_vs_ai"
                    and not self.game_over
                    and not self._anims_blocking()):
                now = time.time()
                if now - last_ai >= self.ai_speed:
                    prev_nobles_snap = [len(p.nobles_visited)
                                        for p in self.env.players]
                    act     = self.agent.select_action(obs, mask)
                    desc    = self.env.get_action_description(act)
                    self._queue_anims(act, 0)
                    self._record_move(0, act, desc)
                    self.message = f"Agent: {desc}"

                    obs, _, done, _, info = self.env.step(act)
                    mask    = info["legal_mask"]
                    last_ai = now

                    for pidx in range(2):
                        gained = (len(self.env.players[pidx].nobles_visited)
                                  - prev_nobles_snap[pidx])
                        if gained > 0:
                            self._queue_noble_anims(pidx, gained)
                    prev_nobles = [len(p.nobles_visited) for p in self.env.players]

                    if done:
                        self.game_over = True
                        self._winner_msg()

            # ── Draw ──────────────────────────────────────────────────
            self._draw(mask, mp)

            # Draw animations on top of everything
            for a in self.anims:
                if a.active():
                    a.draw(self.screen, self._fonts)

            # Speed indicator (ai_vs_ai)
            if self.mode == "ai_vs_ai":
                spd = self.fXS.render(
                    f"Speed: {self.ai_speed:.1f}s/move  (+/- to adjust)",
                    True, C_DIM)
                self.screen.blit(spd, (18, SCREEN_H - 20))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    # ── Human click ───────────────────────────────────────────────────
    def _human_click(self, pos, mask):
        if hasattr(self, '_conf_r') and self._conf_r.collidepoint(pos):
            a = gems_to_action(self.staged)
            return a if (a is not None and mask[a]) else None
        if hasattr(self, '_clr_r') and self._clr_r.collidepoint(pos):
            self.staged = []
            self.message = "Selection cleared."
            return None
        for ci, r in self.gem_rects.items():
            if r.collidepoint(pos):
                self._stage(ci, mask)
                return None
        for r, aid in self.clickable:
            if r.collidepoint(pos) and mask[aid] and 15 <= aid < 45:
                self.staged = []
                return aid
        return None

    def _stage(self, ci, mask):
        cnt = self.staged.count(ci)
        tot = len(self.staged)
        if tot >= 3:
            self.message = "Already 3 gems staged. Confirm or Clear."
            return
        if cnt == 1:
            if tot == 1 and mask[10 + ci]:
                self.staged.append(ci)
                self.message = f"Staged: 2x {GEM_NAMES[ci]} — Confirm to take."
            elif tot == 1:
                self.message = f"Need 4+ {GEM_NAMES[ci]} on board to take 2."
            else:
                self.message = "Cannot mix 2-same with other gems."
            return
        self.staged.append(ci)
        a = gems_to_action(self.staged)
        if a is not None and mask[a]:
            names = " + ".join(GEM_NAMES[c] for c in self.staged)
            self.message = f"Ready: {names} — Confirm or keep selecting."
        else:
            names = " + ".join(GEM_NAMES[c] for c in self.staged)
            self.message = f"Staged: {names} — select more or Confirm."

    def _winner_msg(self):
        w = self.env.winner
        if self.mode == "ai_vs_human":
            self.message = ("You win! Press R to play again."
                            if w == 0 else "AI wins! Press R to play again.")
        else:
            self.message = ("Agent wins! R to restart."
                            if w == 0 else "Greedy wins! R to restart.")

    # ═══════════════════════════════════════════════════════════════════
    #  Drawing (unchanged from v3 except header speed hint)
    # ═══════════════════════════════════════════════════════════════════
    def _draw(self, mask, mp):
        self.screen.fill(C_BG)
        self.clickable = []
        self.gem_rects = {}
        self._header()
        self._board(mask, mp)
        self._nobles()
        self._supply(mask, mp)
        self._history_panel()
        self._player(0, SCREEN_W - 318, 58,
                     "You" if self.mode == "ai_vs_human" else "Agent")
        self._player(1, SCREEN_W - 318, 475,
                     "AI Opponent" if self.mode == "ai_vs_human" else "Greedy")
        if self.mode == "ai_vs_human" and not self.game_over:
            self._staging(mask, mp)
        self._status()

        # Dim overlay while AI animation is blocking human input
        if self.ai_anim_block:
            ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 55))
            self.screen.blit(ov, (0, 0))
            t = self.fLG.render("AI is playing…", True, C_GOLD)
            self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, SCREEN_H // 2 - 14))

    # ── Header ────────────────────────────────────────────────────────
    def _header(self):
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, SCREEN_W, 52))
        pygame.draw.line(self.screen, C_BORDER, (0, 52), (SCREEN_W, 52), 1)
        t = self.fXL.render("SPLENDOR", True, C_GOLD)
        self.screen.blit(t, (18, 11))
        sub = self.fSM.render("Deep Reinforcement Learning", True, C_DIM)
        self.screen.blit(sub, (18 + t.get_width() + 12, 21))
        hint = (f"Turn {self.env.turn_count}   |   "
                f"{'AI vs AI' if self.mode == 'ai_vs_ai' else 'AI vs Human'}   |   "
                f"Speed: {self.ai_speed:.1f}s   +/- adjust   ESC quit   R restart")
        right = self.fSM.render(hint, True, C_DIM)
        self.screen.blit(right, (SCREEN_W - right.get_width() - 16, 18))

    # ── Card board ────────────────────────────────────────────────────
    def _board(self, mask, mp):
        bx, by = 28, 60
        lvl_labels = ["Level 3", "Level 2", "Level 1"]
        for row, lvl in enumerate([2, 1, 0]):
            y   = by + row * (CARD_H + CARD_GAP + 10)
            lbl = self.fSM.render(
                f"{lvl_labels[row]}  ({len(self.env.decks[lvl])} remaining)", True, C_DIM)
            self.screen.blit(lbl, (bx, y - 16))
            for col in range(CARDS_PER_ROW):
                x    = bx + col * (CARD_W + CARD_GAP)
                rect = pygame.Rect(x, y, CARD_W, CARD_H)
                if col < len(self.env.face_up[lvl]):
                    card  = self.env.face_up[lvl][col]
                    slot  = lvl * CARDS_PER_ROW + col
                    ba, ra = 15 + slot, 30 + slot
                    hov   = rect.collidepoint(mp)
                    self._card(rect, card, mask[ba], mask[ra], hov)
                    self.clickable.append((pygame.Rect(x, y, CARD_W, CARD_H // 2), ba))
                    self.clickable.append((pygame.Rect(x, y + CARD_H // 2, CARD_W, CARD_H // 2), ra))
                else:
                    draw_shadow_rect(self.screen, rect)
                    pygame.draw.rect(self.screen, C_PANEL, rect, border_radius=8)
                    pygame.draw.rect(self.screen, C_BORDER, rect, 1, border_radius=8)
                    e = self.fXS.render("empty", True, (55, 60, 80))
                    self.screen.blit(e, (x + CARD_W // 2 - e.get_width() // 2,
                                         y + CARD_H // 2 - 6))

    def _card(self, rect, card, can_buy, can_res, hov):
        cost = get_card_cost(card)
        bi   = get_card_bonus_idx(card)
        bn   = GEM_COLORS[bi]
        bc, tc, hc = GEM_COL[bn]
        pts  = get_card_prestige(card)

        draw_shadow_rect(self.screen, rect, radius=9, offset=3)
        pygame.draw.rect(self.screen, C_PANEL2, rect, border_radius=9)

        hdr = pygame.Rect(rect.x, rect.y, rect.w, 32)
        pygame.draw.rect(self.screen, bc, hdr, border_radius=9)
        pygame.draw.rect(self.screen, bc, pygame.Rect(rect.x, rect.y + 16, rect.w, 16))

        if pts > 0:
            ps = self.fLG.render(str(pts), True, tc)
            self.screen.blit(ps, (rect.x + 7, rect.y + 4))

        badge_c = tuple(max(0, v - 40) for v in bc)
        pygame.draw.rect(self.screen, badge_c,
                         (rect.x + rect.w - 32, rect.y + 4, 28, 18), border_radius=4)
        bs = self.fXS.render(f"+{GEM_SHORT[bi]}", True, tc)
        self.screen.blit(bs, (rect.x + rect.w - 30, rect.y + 7))

        cy = rect.y + 40
        for c in range(NUM_GEM_COLORS):
            if cost[c] > 0:
                cname = GEM_COLORS[c]
                cbc, ctc, chc = GEM_COL[cname]
                pygame.draw.circle(self.screen, cbc, (rect.x + 16, cy), 9)
                pygame.draw.circle(self.screen, chc, (rect.x + 13, cy - 3), 3)
                cs = self.fXS.render(str(cost[c]), True, ctc)
                self.screen.blit(cs, (rect.x + 11, cy - 7))
                ns = self.fXS.render(GEM_SHORT[c], True, C_DIM)
                self.screen.blit(ns, (rect.x + 30, cy - 7))
                cy += 20

        if hov:
            mid = rect.y + CARD_H // 2
            ov  = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 60))
            self.screen.blit(ov, (rect.x, rect.y))
            pygame.draw.line(self.screen, C_BORDER,
                             (rect.x + 6, mid), (rect.x + rect.w - 6, mid), 1)
            bc2 = C_GREEN if can_buy else C_DIM
            bc3 = C_HOVER if can_res else C_DIM
            buy_s = self.fSM.render("BUY",     True, bc2)
            res_s = self.fSM.render("RESERVE", True, bc3)
            self.screen.blit(buy_s, (rect.x + rect.w // 2 - buy_s.get_width() // 2, rect.y + 6))
            self.screen.blit(res_s, (rect.x + rect.w // 2 - res_s.get_width() // 2, mid + 5))

        bord = C_GREEN if can_buy else C_BORDER
        pygame.draw.rect(self.screen, bord, rect, 2, border_radius=9)

    # ── Nobles ────────────────────────────────────────────────────────
    def _nobles(self):
        nx  = 28 + CARDS_PER_ROW * (CARD_W + CARD_GAP) + 14
        lbl = self.fSM.render("Nobles", True, C_DIM)
        self.screen.blit(lbl, (nx, 44))
        for i, noble in enumerate(self.env.nobles):
            y  = 60 + i * (NOBLE_H + 8)
            rc = pygame.Rect(nx, y, NOBLE_W, NOBLE_H)
            draw_shadow_rect(self.screen, rc, radius=9, offset=3)
            pygame.draw.rect(self.screen, (48, 30, 62), rc, border_radius=9)
            pygame.draw.rect(self.screen, C_GOLD, rc, 2, border_radius=9)
            pts = self.fLG.render(f"{get_noble_prestige(noble)}", True, C_GOLD)
            self.screen.blit(pts, (nx + 7, y + 5))
            sub = self.fXS.render("pts", True, C_DIM)
            self.screen.blit(sub, (nx + 7 + pts.get_width() + 2, y + 10))
            reqs = get_noble_requirements(noble)
            rx2, ry2 = nx + 5, y + NOBLE_H - 24
            for c in range(NUM_GEM_COLORS):
                if reqs[c] > 0:
                    draw_gem_circle(self.screen, rx2 + 9, ry2 + 9, 8,
                                    GEM_COLORS[c], str(reqs[c]), self.fXS)
                    rx2 += 20

    # ── Board gem supply ─────────────────────────────────────────────
    def _supply(self, mask, mp):
        bx  = 28
        by  = 60 + 3 * (CARD_H + CARD_GAP + 10) + 6
        pw  = CARDS_PER_ROW * (CARD_W + CARD_GAP) + 60
        ph  = 82

        draw_shadow_rect(self.screen, pygame.Rect(bx - 6, by - 6, pw, ph))
        pygame.draw.rect(self.screen, C_PANEL, (bx - 6, by - 6, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, C_BORDER, (bx - 6, by - 6, pw, ph), 1, border_radius=10)
        lbl = self.fSM.render("Gem Supply  (click to select)", True, C_DIM)
        self.screen.blit(lbl, (bx, by - 2))

        x  = bx + 10
        cy = by + 44
        for i in range(6):
            name  = GEM_COLORS[i] if i < 5 else "gold"
            cnt   = int(self.env.board_tokens[i])
            is_g  = (i == GOLD_IDX)
            sc    = self.staged.count(i)
            hov   = (mp[0] - x) ** 2 + (mp[1] - cy) ** 2 < (GEM_R + 6) ** 2
            draw_gem_circle(self.screen, x, cy, GEM_R, name,
                            str(cnt), self.fMD,
                            selected=(sc > 0),
                            dim=(cnt == 0))
            if sc > 0:
                badge = self.fXS.render(f"x{sc}", True, C_SEL)
                self.screen.blit(badge, (x - badge.get_width() // 2, cy - GEM_R - 16))
            nm = self.fXS.render(GEM_NAMES[i], True, C_DIM if not hov else C_TEXT)
            self.screen.blit(nm, (x - nm.get_width() // 2, cy + GEM_R + 5))
            if not is_g:
                self.gem_rects[i] = pygame.Rect(x - GEM_R - 4, cy - GEM_R - 4,
                                                (GEM_R + 4) * 2, (GEM_R + 4) * 2)
            x += GEM_R * 2 + 20

    # ── Player panel ─────────────────────────────────────────────────
    def _player(self, pidx, x, y, label):
        p     = self.env.players[pidx]
        pw, ph = 310, 390
        is_me = (pidx == 0 and self.mode == "ai_vs_human")

        draw_shadow_rect(self.screen, pygame.Rect(x, y, pw, ph), radius=10, offset=4)
        pygame.draw.rect(self.screen, C_PANEL, (x, y, pw, ph), border_radius=10)
        bord = C_GOLD if is_me else C_BORDER
        pygame.draw.rect(self.screen, bord, (x, y, pw, ph), 2, border_radius=10)

        pygame.draw.rect(self.screen, C_PANEL2, (x, y, pw, 40), border_radius=10)
        pygame.draw.rect(self.screen, C_PANEL2, pygame.Rect(x, y + 20, pw, 20))
        lbl_s = self.fLG.render(("► " if is_me else "") + label,
                                 True, C_GOLD if is_me else C_TEXT)
        self.screen.blit(lbl_s, (x + 10, y + 9))
        pts_s = self.fXL.render(f"{p.prestige}", True, C_GOLD)
        self.screen.blit(pts_s, (x + pw - pts_s.get_width() - 10, y + 5))
        pt_lbl = self.fXS.render("/15", True, C_DIM)
        self.screen.blit(pt_lbl, (x + pw - pt_lbl.get_width() - 10,
                                   y + 5 + pts_s.get_height() - pt_lbl.get_height()))

        draw_progress_bar(self.screen, x + 10, y + 42, pw - 20, 7,
                          p.prestige, WIN_POINTS, C_GOLD)

        ty = y + 60
        self.screen.blit(self.fSM.render("Tokens", True, C_DIM), (x + 10, ty))
        ty += 18
        tx = x + 10
        for i in range(6):
            name = GEM_COLORS[i] if i < 5 else "gold"
            draw_gem_circle(self.screen, tx + 13, ty + 13, 13, name,
                            str(p.tokens[i]), self.fSM)
            nm = self.fXS.render(GEM_SHORT[i], True, C_DIM)
            self.screen.blit(nm, (tx + 13 - nm.get_width() // 2, ty + 28))
            tx += 46
        ty += 50

        self.screen.blit(self.fSM.render("Card Bonuses", True, C_DIM), (x + 10, ty))
        ty += 18
        tx = x + 10
        for i in range(5):
            name = GEM_COLORS[i]
            bc2, tc2, hc2 = GEM_COL[name]
            pygame.draw.rect(self.screen, bc2, (tx, ty, 30, 30), border_radius=6)
            pygame.draw.rect(self.screen, hc2, (tx + 3, ty + 3, 8, 8), border_radius=3)
            cs = self.fMD.render(str(p.bonuses[i]), True, tc2)
            self.screen.blit(cs, (tx + 15 - cs.get_width() // 2, ty + 7))
            nm = self.fXS.render(GEM_SHORT[i], True, C_DIM)
            self.screen.blit(nm, (tx + 15 - nm.get_width() // 2, ty + 32))
            tx += 54
        ty += 52

        self.screen.blit(self.fSM.render(
            f"Reserved  ({len(p.reserved_cards)}/3)", True, C_DIM), (x + 10, ty))
        ty += 20
        for i, card in enumerate(p.reserved_cards):
            bi  = get_card_bonus_idx(card)
            bn  = GEM_COLORS[bi]
            bc2, tc2, hc2 = GEM_COL[bn]
            cr  = pygame.Rect(x + 10 + i * 96, ty, 88, 62)
            draw_shadow_rect(self.screen, cr, radius=6, offset=2)
            pygame.draw.rect(self.screen, C_PANEL2, cr, border_radius=6)
            pygame.draw.rect(self.screen, bc2, cr, 2, border_radius=6)
            pygame.draw.rect(self.screen, bc2,
                             pygame.Rect(cr.x, cr.y, cr.w, 18), border_radius=6)
            pygame.draw.rect(self.screen, bc2,
                             pygame.Rect(cr.x, cr.y + 9, cr.w, 9))
            pts2 = self.fSM.render(f"{get_card_prestige(card)}pt", True, tc2)
            self.screen.blit(pts2, (cr.x + 4, cr.y + 2))
            bn2 = self.fXS.render(f"+{GEM_SHORT[bi]}", True, tc2)
            self.screen.blit(bn2, (cr.x + cr.w - bn2.get_width() - 3, cr.y + 3))
            cost = get_card_cost(card)
            cx2, cy2 = cr.x + 5, cr.y + 22
            for c in range(5):
                if cost[c] > 0:
                    cbc, ctc, chc = GEM_COL[GEM_COLORS[c]]
                    pygame.draw.circle(self.screen, cbc, (cx2 + 6, cy2 + 6), 6)
                    vs = self.fXS.render(str(cost[c]), True, ctc)
                    self.screen.blit(vs, (cx2 + 3, cy2))
                    cx2 += 15
                    if cx2 > cr.x + cr.w - 8:
                        break
            if pidx == 0:
                self.clickable.append((cr, 15 + 12 + i))
                b2 = self.fXS.render("BUY", True, C_GREEN)
                self.screen.blit(b2, (cr.x + cr.w // 2 - b2.get_width() // 2,
                                       cr.y + cr.h - 13))
        ty += 72

        if p.nobles_visited:
            ns = self.fSM.render(
                f"Nobles earned: {len(p.nobles_visited)}  (+{len(p.nobles_visited) * 3} pts)",
                True, C_GOLD)
            self.screen.blit(ns, (x + 10, ty))

    # ── Move history panel ────────────────────────────────────────────
    def _history_panel(self):
        """Scrolling move log between the nobles area and player panels."""
        # Panel sits in the gap between nobles (x≈616) and player panels (x=1082)
        hx, hy = 624, 58
        hw, hh  = 442, 760

        # Background
        draw_shadow_rect(self.screen, pygame.Rect(hx, hy, hw, hh), radius=10, offset=3)
        pygame.draw.rect(self.screen, C_PANEL, (hx, hy, hw, hh), border_radius=10)
        pygame.draw.rect(self.screen, C_BORDER, (hx, hy, hw, hh), 1, border_radius=10)

        # Header
        pygame.draw.rect(self.screen, C_PANEL2,
                         (hx, hy, hw, 30), border_radius=10)
        pygame.draw.rect(self.screen, C_PANEL2,
                         (hx, hy + 15, hw, 15))
        title = self.fSM.render("Move History", True, C_TEXT)
        self.screen.blit(title, (hx + hw // 2 - title.get_width() // 2, hy + 7))

        row_h   = 22
        max_vis = (hh - 38) // row_h        # how many rows fit
        history = self.move_history[-max_vis:]  # show most recent

        # Column header
        cy = hy + 34
        col_labels = [("T",  hx + 14),
                      ("Who",hx + 36),
                      ("Action", hx + 90)]
        for lbl, lx in col_labels:
            t = self.fXS.render(lbl, True, C_DIM)
            self.screen.blit(t, (lx, cy))
        pygame.draw.line(self.screen, C_BORDER,
                         (hx + 6, cy + 14), (hx + hw - 6, cy + 14), 1)
        cy += 18

        # Color coding by action type
        TYPE_BG = {
            "gem":     (25, 45, 55),
            "buy":     (20, 50, 30),
            "reserve": (20, 30, 60),
            "discard": (55, 20, 20),
            "unknown": (30, 30, 40),
        }
        TYPE_AC = {
            "gem":     (100, 200, 220),
            "buy":     C_GREEN,
            "reserve": C_HOVER,
            "discard": C_RED,
            "unknown": C_DIM,
        }

        # Player badge colors: [player0, player1]
        BADGE_COL = [(80, 160, 255), (255, 110, 60)]

        for i, mv in enumerate(history):
            atype  = mv["type"]
            bg     = TYPE_BG.get(atype, (30, 30, 40))
            ac     = TYPE_AC.get(atype, C_DIM)
            pidx   = mv["player"]
            is_last = (i == len(history) - 1)

            # Row background (highlight most recent)
            row_rect = pygame.Rect(hx + 4, cy, hw - 8, row_h - 1)
            pygame.draw.rect(self.screen, bg, row_rect, border_radius=4)
            if is_last:
                pygame.draw.rect(self.screen, (*ac, 80), row_rect, 1, border_radius=4)

            # Turn number
            tn = self.fXS.render(str(mv["turn"]), True, C_DIM)
            self.screen.blit(tn, (hx + 10, cy + row_h // 2 - tn.get_height() // 2))

            # Player badge
            bc = BADGE_COL[pidx]
            pygame.draw.rect(self.screen, bc,
                             (hx + 30, cy + 3, 48, row_h - 6), border_radius=3)
            nm = self.fXS.render(mv["name"], True, (10, 10, 20))
            self.screen.blit(nm, (hx + 30 + 24 - nm.get_width() // 2,
                                   cy + row_h // 2 - nm.get_height() // 2))

            # Gem dots for gem-take actions
            dx = hx + 86
            if atype == "gem" and mv["gems"]:
                unique_gems = list(dict.fromkeys(mv["gems"]))  # order-preserving unique
                for gi in unique_gems:
                    gname = GEM_COLORS[gi]
                    bc2, _, hc2 = GEM_COL[gname]
                    pygame.draw.circle(self.screen, bc2, (dx + 6, cy + row_h // 2), 6)
                    pygame.draw.circle(self.screen, hc2, (dx + 4, cy + row_h // 2 - 2), 2)
                    if mv["gems"].count(gi) == 2:
                        # second dot offset
                        pygame.draw.circle(self.screen, bc2, (dx + 14, cy + row_h // 2), 6)
                        pygame.draw.circle(self.screen, hc2, (dx + 12, cy + row_h // 2 - 2), 2)
                        dx += 22
                    else:
                        dx += 16
                dx += 4
            elif atype == "buy":
                tag = self.fXS.render("BUY", True, C_GREEN)
                self.screen.blit(tag, (dx, cy + row_h // 2 - tag.get_height() // 2))
                dx += tag.get_width() + 6
            elif atype == "reserve":
                tag = self.fXS.render("RES", True, C_HOVER)
                self.screen.blit(tag, (dx, cy + row_h // 2 - tag.get_height() // 2))
                dx += tag.get_width() + 6
            elif atype == "discard":
                tag = self.fXS.render("DIS", True, C_RED)
                self.screen.blit(tag, (dx, cy + row_h // 2 - tag.get_height() // 2))
                dx += tag.get_width() + 6

            # Description text (clipped to remaining width)
            max_w   = hx + hw - 10 - dx
            desc    = mv["desc"]
            dt      = self.fXS.render(desc, True, ac if is_last else C_DIM)
            if dt.get_width() > max_w:
                # Truncate with ellipsis
                while desc and self.fXS.size(desc + "…")[0] > max_w:
                    desc = desc[:-1]
                dt = self.fXS.render(desc + "…", True, ac if is_last else C_DIM)
            self.screen.blit(dt, (dx, cy + row_h // 2 - dt.get_height() // 2))

            cy += row_h

        # "No moves yet" placeholder
        if not history:
            ph = self.fSM.render("No moves yet.", True, C_DIM)
            self.screen.blit(ph, (hx + hw // 2 - ph.get_width() // 2, hy + hh // 2))

        # Move count footer
        total = self.fXS.render(f"{len(self.move_history)} moves total", True, C_DIM)
        self.screen.blit(total, (hx + hw - total.get_width() - 8, hy + hh - 14))

    # ── Gem staging panel ─────────────────────────────────────────────
    def _staging(self, mask, mp):
        ph = 105
        py = SCREEN_H - ph
        draw_shadow_rect(self.screen, pygame.Rect(0, py, SCREEN_W, ph), radius=0, offset=0)
        pygame.draw.rect(self.screen, C_PANEL, (0, py, SCREEN_W, ph))
        pygame.draw.line(self.screen, C_BORDER, (0, py), (SCREEN_W, py), 1)

        hd = self.fSM.render(
            "Gem Selection  —  click board gems above, then Confirm", True, C_DIM)
        self.screen.blit(hd, (18, py + 8))

        sx = 18
        sy = py + 28
        for i in range(3):
            cx2 = sx + SEL_R
            cy2 = sy + SEL_R
            if i < len(self.staged):
                c    = self.staged[i]
                name = GEM_COLORS[c]
                draw_gem_circle(self.screen, cx2, cy2, SEL_R, name,
                                GEM_SHORT[c], self.fSM, selected=True)
            else:
                pygame.draw.circle(self.screen, C_PANEL2, (cx2, cy2), SEL_R)
                pygame.draw.circle(self.screen, C_BORDER, (cx2, cy2), SEL_R, 2)
                q = self.fSM.render("?", True, (60, 65, 90))
                self.screen.blit(q, (cx2 - q.get_width() // 2, cy2 - q.get_height() // 2))
            sx += SEL_R * 2 + 12

        act   = gems_to_action(self.staged)
        valid = act is not None and mask[act]
        if len(self.staged) == 0:
            st, sc = "Select gems from the board above.", C_DIM
        elif valid:
            names = " + ".join(GEM_NAMES[c] for c in self.staged)
            st, sc = f"Valid: {names}", C_GREEN
        else:
            st, sc = "Not a valid combo yet — keep selecting.", C_RED
        sv = self.fSM.render(st, True, sc)
        self.screen.blit(sv, (18, py + 80))

        bx2 = SCREEN_W - 290
        by2 = py + 16
        conf = pygame.Rect(bx2, by2, 126, 56)
        ch   = conf.collidepoint(mp)
        pygame.draw.rect(self.screen, C_SHADOW,
                         pygame.Rect(bx2 + 2, by2 + 3, 126, 56), border_radius=9)
        pygame.draw.rect(self.screen,
                         C_CONF_H if (ch and valid) else (C_CONFIRM if valid else C_PANEL2),
                         conf, border_radius=9)
        pygame.draw.rect(self.screen, C_GREEN if valid else C_BORDER, conf, 2, border_radius=9)
        ct = self.fMD.render("Confirm", True, C_TEXT if valid else C_DIM)
        self.screen.blit(ct, (conf.x + conf.w // 2 - ct.get_width() // 2,
                              conf.y + conf.h // 2 - ct.get_height() // 2))
        self._conf_r = conf

        clr = pygame.Rect(bx2 + 140, by2, 100, 56)
        clh = clr.collidepoint(mp)
        pygame.draw.rect(self.screen, C_SHADOW,
                         pygame.Rect(clr.x + 2, clr.y + 3, clr.w, clr.h), border_radius=9)
        pygame.draw.rect(self.screen, C_CLEAR_H if clh else C_CLEAR, clr, border_radius=9)
        pygame.draw.rect(self.screen, C_RED, clr, 2, border_radius=9)
        cl = self.fMD.render("Clear", True, C_TEXT)
        self.screen.blit(cl, (clr.x + clr.w // 2 - cl.get_width() // 2,
                              clr.y + clr.h // 2 - cl.get_height() // 2))
        self._clr_r = clr

        hint = self.fXS.render(
            "Cards: top-half = BUY   bottom-half = RESERVE", True, C_DIM)
        self.screen.blit(hint, (bx2 - hint.get_width() - 16, py + 80))

    # ── Status bar ────────────────────────────────────────────────────
    def _status(self):
        if self.mode == "ai_vs_human" and not self.game_over:
            return
        pygame.draw.rect(self.screen, C_PANEL, (0, SCREEN_H - 32, SCREEN_W, 32))
        pygame.draw.line(self.screen, C_BORDER,
                         (0, SCREEN_H - 32), (SCREEN_W, SCREEN_H - 32), 1)
        ms = self.fMD.render(self.message, True, C_GOLD)
        self.screen.blit(ms, (18, SCREEN_H - 25))


def run_visualization(mode="ai_vs_ai", agent_path=None, ai_speed=1.0):
    v = SplendorVisualizer(mode=mode, agent_path=agent_path, ai_speed=ai_speed)
    v.run()
