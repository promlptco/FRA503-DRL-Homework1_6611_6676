"""
Enhanced DDQN Agent -- DRL_Project_extra (GPU-optimised).

Improvements over original proposal:
  Architecture
    - Dueling DQN  (V(s) + A(s,a) - mean A)        [use_dueling=True]
    - LayerNorm after every hidden layer             (gradient stability)

  Replay
    - Prioritized Experience Replay (PER, SumTree)  [use_per=True]

  Optimisation
    - Huber loss (smooth_l1) instead of MSE         (robust to large TD errors)
    - Gradient clipping at 1.0                      (tighter than original)
    - Soft target updates (tau=0.005)               (smoother than hard copies)

  GPU acceleration (auto-detected)
    - torch.compile()     [use_compile=True]  -- graph compilation (PyTorch 2+)
    - AMP / autocast      [use_amp=True]      -- mixed-precision on CUDA
    - pin_memory transfers                    -- async CPU->GPU data movement

  Backward compatibility
    - load() auto-detects architecture from state-dict keys
    - Can load checkpoints from original DRL_Project or any prior run
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from splendor_env import STATE_DIM, TOTAL_ACTIONS

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "legal_mask")
)


# --- Q-Network (original + optional LayerNorm) -----------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=TOTAL_ACTIONS,
                 hidden_sizes=(256, 256, 128), use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        layers = []
        prev   = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Dueling Q-Network -----------------------------------------------
class DuelingQNetwork(nn.Module):
    """Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)"""

    def __init__(self, state_dim=STATE_DIM, action_dim=TOTAL_ACTIONS,
                 hidden_sizes=(256, 256, 128), use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        h0, h1, h2 = hidden_sizes

        def _blk(in_d, out_d):
            layers = [nn.Linear(in_d, out_d)]
            if use_layernorm:
                layers.append(nn.LayerNorm(out_d))
            layers.append(nn.ReLU())
            return layers

        self.feature = nn.Sequential(*_blk(state_dim, h0), *_blk(h0, h1))
        self.value_stream = nn.Sequential(*_blk(h1, h2), nn.Linear(h2, 1))
        self.advantage_stream = nn.Sequential(*_blk(h1, h2), nn.Linear(h2, action_dim))
        self.net = self.feature   # compatibility alias

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value_stream(feat)
        a    = self.advantage_stream(feat)
        return v + a - a.mean(dim=-1, keepdim=True)


# --- Uniform Replay Buffer -------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, legal_mask):
        self.buffer.append(Transition(state, action, reward, next_state, done, legal_mask))

    def sample(self, batch_size):
        batch       = random.sample(self.buffer, batch_size)
        states      = np.array([t.state      for t in batch], dtype=np.float32)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        legal_masks = np.array([t.legal_mask for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones, legal_masks

    def __len__(self):
        return len(self.buffer)


# --- Sum-Tree (PER) --------------------------------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = [None] * capacity
        self.n_entries = 0
        self.write_ptr = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, idx, s):
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = left + 1

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, float(self.tree[tree_idx]), self.data[data_idx]


# --- Prioritized Replay Buffer ---------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6,
                 beta_start=0.4, beta_end=1.0, beta_steps=200_000):
        self.alpha          = alpha
        self.beta           = beta_start
        self.beta_end       = beta_end
        self.beta_increment = (beta_end - beta_start) / beta_steps
        self.epsilon        = 1e-6
        self.tree           = SumTree(capacity)
        self.max_priority   = 1.0

    def push(self, state, action, reward, next_state, done, legal_mask):
        t = Transition(state, action, reward, next_state, done, legal_mask)
        self.tree.add(self.max_priority ** self.alpha, t)

    def sample(self, batch_size):
        batch, indices, priorities = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        for i in range(batch_size):
            s = (i + np.random.random()) / batch_size * self.tree.total
            idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        if not batch:
            return None
        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-10, 1.0)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        states      = np.array([t.state      for t in batch], dtype=np.float32)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        legal_masks = np.array([t.legal_mask for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones, legal_masks, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = min((abs(float(err)) + self.epsilon) ** self.alpha, 100.0)
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


# --- DDQN Agent ------------------------------------------------------
class DDQNAgent:
    """
    Double DQN agent with optional Dueling, PER, LayerNorm, AMP, and torch.compile.

    Flags
    -----
    use_dueling   : Dueling DQN architecture
    use_per       : Prioritized Experience Replay
    use_layernorm : LayerNorm in hidden layers (default True)
    use_compile   : torch.compile() (PyTorch 2+, best on CUDA)
    use_amp       : Automatic Mixed Precision (CUDA only)
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=TOTAL_ACTIONS,
        hidden_sizes=(256, 256, 128),
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99997,
        buffer_size=100_000,
        batch_size=64,
        target_update_freq=2_000,
        tau=0.005,
        min_buffer_size=5_000,
        use_dueling=False,
        use_per=False,
        use_layernorm=True,
        use_compile=False,
        use_amp=True,
        device=None,
    ):
        self.action_dim         = action_dim
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.tau                = tau
        self.min_buffer_size    = min_buffer_size
        self.train_step_count   = 0
        self.use_dueling        = use_dueling
        self.use_per            = use_per
        self.use_layernorm      = use_layernorm
        self.lr                 = lr

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        is_cuda = self.device.type == "cuda"
        print(f"  [Agent] Device: {self.device}" + (" (GPU)" if is_cuda else " (CPU)"))

        # AMP scaler (CUDA only)
        self.use_amp = use_amp and is_cuda
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Networks
        NetClass = DuelingQNetwork if use_dueling else QNetwork
        kw = dict(state_dim=state_dim, action_dim=action_dim,
                  hidden_sizes=hidden_sizes, use_layernorm=use_layernorm)
        self.online_net = NetClass(**kw).to(self.device)
        self.target_net = NetClass(**kw).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # torch.compile (PyTorch 2+)
        if use_compile and hasattr(torch, "compile"):
            try:
                self.online_net = torch.compile(self.online_net)
                self.target_net = torch.compile(self.target_net)
                print("  [Agent] torch.compile() active")
            except Exception as exc:
                print(f"  [Agent] torch.compile() skipped: {exc}")

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size, beta_steps=500_000
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

        self._pin = is_cuda   # pin_memory flag for async transfers

    # --- Tensor helper -----------------------------------------------
    def _t(self, arr, dtype=torch.float32):
        t = torch.tensor(arr, dtype=dtype)
        if self._pin:
            t = t.pin_memory()
        return t.to(self.device, non_blocking=True)

    # --- Action Selection --------------------------------------------
    def select_action(self, state, legal_mask):
        if random.random() < self.epsilon:
            legal = np.where(legal_mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0
        with torch.no_grad():
            q = self.online_net(self._t(state).unsqueeze(0)).squeeze(0).cpu().numpy()
        q[~legal_mask] = -np.inf
        return int(np.argmax(q))

    def select_action_batch(self, states, legal_masks):
        n       = len(states)
        actions = np.zeros(n, dtype=np.int64)
        explore    = np.random.random(n) < self.epsilon
        greedy_idx = np.where(~explore)[0]
        random_idx = np.where(explore)[0]
        if len(greedy_idx) > 0:
            with torch.no_grad():
                q_batch = self.online_net(self._t(states[greedy_idx])).cpu().numpy()
            for j, i in enumerate(greedy_idx):
                q = q_batch[j].copy()
                q[~legal_masks[i]] = -np.inf
                actions[i] = int(np.argmax(q))
        for i in random_idx:
            legal = np.where(legal_masks[i])[0]
            actions[i] = int(np.random.choice(legal)) if len(legal) > 0 else 0
        return actions

    # --- Store -------------------------------------------------------
    def store(self, state, action, reward, next_state, done, legal_mask):
        self.replay_buffer.push(state, action, reward, next_state, done, legal_mask)

    # --- Training Step -----------------------------------------------
    def train_step(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        if self.use_per:
            result = self.replay_buffer.sample(self.batch_size)
            if result is None:
                return None
            states, actions, rewards, next_states, dones, legal_masks, per_idx, is_w = result
            weights_t = self._t(is_w)
        else:
            states, actions, rewards, next_states, dones, legal_masks = \
                self.replay_buffer.sample(self.batch_size)
            weights_t = None
            per_idx   = None

        st  = self._t(states)
        at  = self._t(actions,     torch.int64)
        rt  = self._t(rewards)
        nst = self._t(next_states)
        dt  = self._t(dones)
        lmt = self._t(legal_masks)

        def _forward():
            qv = self.online_net(st)
            qs = qv.gather(1, at.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                nqo = self.online_net(nst)
                nqo[lmt == 0] = -1e4  # -1e9 overflows float16 in AMP
                ba  = nqo.argmax(dim=1)
                nqt = self.target_net(nst)
                nqs = nqt.gather(1, ba.unsqueeze(1)).squeeze(1)
                tgt = rt + self.gamma * nqs * (1.0 - dt)
            td_err = (qs - tgt).detach()
            if self.use_per and weights_t is not None:
                loss = (weights_t * nn.functional.smooth_l1_loss(
                    qs, tgt, reduction="none")).mean()
            else:
                loss = nn.functional.smooth_l1_loss(qs, tgt)
            return loss, td_err

        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, td_err = _forward()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, td_err = _forward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.optimizer.step()

        if self.use_per and per_idx is not None:
            self.replay_buffer.update_priorities(per_idx, td_err.cpu().numpy())

        self.train_step_count += 1
        if self.tau > 0:
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)
        elif self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Call ONCE PER EPISODE from training loop."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # --- Save / Load -------------------------------------------------
    def save(self, path):
        def _sd(net):
            return getattr(net, "_orig_mod", net).state_dict()
        torch.save({
            "online_net":       _sd(self.online_net),
            "target_net":       _sd(self.target_net),
            "optimizer":        self.optimizer.state_dict(),
            "epsilon":          self.epsilon,
            "train_step_count": self.train_step_count,
            "use_dueling":      self.use_dueling,
            "use_per":          self.use_per,
            "use_layernorm":    self.use_layernorm,
            "tau":              self.tau,
        }, path)

    def load(self, path):
        """Backward-compatible loader: handles original and enhanced checkpoints."""
        ckpt         = torch.load(path, map_location=self.device, weights_only=False)
        online_state = ckpt["online_net"]

        if "use_layernorm" in ckpt:
            saved_ln = ckpt["use_layernorm"]
        else:
            saved_dueling_peek = ckpt.get("use_dueling", False)
            saved_ln = ("feature.1.weight" in online_state if saved_dueling_peek
                        else "net.1.weight" in online_state)

        saved_dueling = ckpt.get("use_dueling", False)

        if saved_dueling != self.use_dueling or saved_ln != self.use_layernorm:
            self.use_dueling   = saved_dueling
            self.use_layernorm = saved_ln
            NetClass = DuelingQNetwork if saved_dueling else QNetwork
            self.online_net = NetClass(use_layernorm=saved_ln).to(self.device)
            self.target_net = NetClass(use_layernorm=saved_ln).to(self.device)
            self.optimizer  = optim.Adam(self.online_net.parameters(), lr=self.lr)

        self.online_net.load_state_dict(online_state)
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon          = ckpt["epsilon"]
        self.train_step_count = ckpt["train_step_count"]
        self.use_per          = ckpt.get("use_per", False)
        print(f"  Loaded: dueling={self.use_dueling} layernorm={self.use_layernorm} "
              f"eps={self.epsilon:.3f}")
