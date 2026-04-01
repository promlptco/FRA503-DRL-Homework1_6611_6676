"""
debug_all.py  —  Dry-run every algorithm with a fake environment.

Run this BEFORE the real Isaac Lab training to catch shape errors,
device mismatches, NaN losses, and crashes without wasting GPU time.

Usage:
    python debug_all.py

Does NOT require Isaac Lab. Uses a FakeEnv that mimics the real obs/action
interface with random tensors so every code path is exercised.
"""

import sys, os, traceback, math

# ── path setup ────────────────────────────────────────────────────────
# debug_all.py lives at:
#   CartPole_4.5.0/scripts/Function_based/debug_all.py
# We need CartPole_4.5.0/ on sys.path so that:
#   RL_Algorithm.RL_base_function        → CartPole_4.5.0/RL_Algorithm/RL_base_function.py
#   RL_Algorithm.storage.buffers         → CartPole_4.5.0/RL_Algorithm/storage/buffers.py
#   RL_Algorithm.network.mlp             → CartPole_4.5.0/RL_Algorithm/network/mlp.py
#   RL_Algorithm.Function_based.DQN     → CartPole_4.5.0/RL_Algorithm/Function_based/DQN.py

_THIS_FILE   = os.path.abspath(__file__)                    # .../scripts/Function_based/debug_all.py
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)                  # .../scripts/Function_based/
_SCRIPTS_UP  = os.path.dirname(_SCRIPTS_DIR)                # .../scripts/
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_UP)                # .../CartPole_4.5.0/

# Insert project root so all RL_Algorithm.* imports resolve
sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np

# ── colour helpers ────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def err(msg): print(f"  {RED}✗ {msg}{RESET}")
def warn(msg):print(f"  {YELLOW}⚠ {msg}{RESET}")
def section(title):
    print(f"\n{BOLD}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{RESET}")

# ══════════════════════════════════════════════════════════════════════
# FAKE ENVIRONMENT  (mimics Isaac Lab CartPole interface)
# ══════════════════════════════════════════════════════════════════════

class FakeEnv:
    """
    Mimics env.reset() → {'policy': Tensor(num_envs, 4)}
    and env.step(action) → (obs, reward, terminated, truncated, info)

    Randomly terminates after 5-15 steps so we exercise the done=True path.
    """
    def __init__(self, num_envs=1, obs_dim=4, max_steps=20):
        self.num_envs  = num_envs
        self.obs_dim   = obs_dim
        self.max_steps = max_steps
        self._step     = 0
        self.device    = torch.device('cpu')

    def reset(self, **kwargs):
        self._step = 0
        obs = {'policy': torch.randn(self.num_envs, self.obs_dim)}
        return obs, {}

    def step(self, action):
        self._step += 1
        obs        = {'policy': torch.randn(self.num_envs, self.obs_dim)}
        reward     = torch.ones(self.num_envs) * 1.0
        # terminate randomly between step 5-15, or at max_steps
        terminate_now = (self._step >= self.max_steps) or \
                        (self._step >= 5 and torch.rand(1).item() < 0.3)
        terminated = torch.tensor([terminate_now] * self.num_envs)
        truncated  = torch.tensor([self._step >= self.max_steps] * self.num_envs)
        return obs, reward, terminated, truncated, {}


# ══════════════════════════════════════════════════════════════════════
# CHECK HELPERS
# ══════════════════════════════════════════════════════════════════════

def has_nan(tensor_or_float):
    if isinstance(tensor_or_float, torch.Tensor):
        return torch.isnan(tensor_or_float).any().item()
    return math.isnan(float(tensor_or_float)) if tensor_or_float is not None else False

def check_output(name, val, expected_shape=None):
    if val is None:
        warn(f"{name} is None (might be ok if buffer not ready)")
        return
    if isinstance(val, torch.Tensor):
        if has_nan(val):
            err(f"{name} contains NaN!  shape={tuple(val.shape)}")
        elif expected_shape and tuple(val.shape) != expected_shape:
            err(f"{name} shape mismatch: got {tuple(val.shape)}, expected {expected_shape}")
        else:
            ok(f"{name} OK  shape={tuple(val.shape)}  device={val.device}")
    elif isinstance(val, (int, float)):
        if has_nan(val):
            err(f"{name} = NaN!")
        else:
            ok(f"{name} = {val:.4f}")

# ══════════════════════════════════════════════════════════════════════
# AGENT FACTORY  (same params as train_all.py)
# ══════════════════════════════════════════════════════════════════════

DEVICE       = torch.device('cpu')
N_OBS        = 4
ACTION_RANGE = [-10.0, 10.0]
MAX_STEPS    = 20   # short for debug

def make_linear_qn():
    from RL_Algorithm.Function_based.Linear_Q import Linear_QN
    return Linear_QN(
        num_of_action=11, action_range=ACTION_RANGE,
        learning_rate=1e-3, initial_epsilon=1.0,
        epsilon_decay=5e-4, final_epsilon=0.01,
        discount_factor=0.99, n_observations=N_OBS,
    )

def make_dqn():
    from RL_Algorithm.Function_based.DQN import DQN
    return DQN(
        device=DEVICE, num_of_action=11, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64, dropout=0.1,
        learning_rate=1e-3, tau=0.005,
        initial_epsilon=1.0, epsilon_decay=5e-4, final_epsilon=0.01,
        discount_factor=0.99, buffer_size=200, batch_size=32,
        update_freq=4, target_update_freq=50,
    )

def make_reinforce():
    from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
    return MC_REINFORCE(
        device=DEVICE, num_of_action=11, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64, dropout=0.1,
        learning_rate=1e-3, discount_factor=0.99,
    )

def make_ac():
    from RL_Algorithm.Function_based.AC import AC
    return AC(
        device=DEVICE, num_of_action=1, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64,
        learning_rate=3e-4, discount_factor=0.99, entropy_coef=0.01,
    )

def make_a2c():
    from RL_Algorithm.Function_based.A2C import A2C
    return A2C(
        device=DEVICE, num_of_action=1, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64,
        learning_rate=3e-4, discount_factor=0.99,
        gae_lambda=0.95, value_loss_coef=0.5, entropy_coef=0.01,
        num_transitions_per_env=8, num_envs=1,
    )

def make_ppo():
    from RL_Algorithm.Function_based.PPO import PPO
    return PPO(
        device=DEVICE, num_of_action=1, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64,
        learning_rate=3e-4, discount_factor=0.99,
        gae_lambda=0.95, clip_param=0.2,
        value_loss_coef=0.5, entropy_coef=0.01,
        num_learning_epochs=2, num_mini_batches=2,
        num_transitions_per_env=8, num_envs=1,
    )

def make_td3():
    from RL_Algorithm.Function_based.TD3 import TD3
    return TD3(
        device=DEVICE, num_of_action=1, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64,
        learning_rate=3e-4, tau=0.005, discount_factor=0.99,
        buffer_size=200, batch_size=32,
        policy_noise=0.2, noise_clip=0.5, expl_noise=0.1, policy_delay=2,
    )

def make_sac():
    from RL_Algorithm.Function_based.SAC import SAC
    return SAC(
        device=DEVICE, num_of_action=1, action_range=ACTION_RANGE,
        n_observations=N_OBS, hidden_dim=64,
        learning_rate=3e-4, tau=0.005, discount_factor=0.99,
        buffer_size=200, batch_size=32,
        alpha=0.2, auto_entropy=True,
    )

MAKERS = {
    "Linear_QN":    make_linear_qn,
    "DQN":          make_dqn,
    "MC_REINFORCE": make_reinforce,
    "AC":           make_ac,
    "A2C":          make_a2c,
    "PPO":          make_ppo,
    "TD3":          make_td3,
    "SAC":          make_sac,
}

# ══════════════════════════════════════════════════════════════════════
# PER-ALGORITHM CHECKS
# ══════════════════════════════════════════════════════════════════════

def check_linear_qn(agent):
    env   = FakeEnv()
    obs,_ = env.reset()

    # Check q() works
    q_vals = agent.q(obs)
    check_output("q(obs)", torch.tensor(q_vals))
    assert q_vals.shape == (11,), f"q shape wrong: {q_vals.shape}"
    ok("q() shape correct (11,)")

    # Check select_action
    action_t, action_idx = agent.select_action(obs)
    check_output("select_action tensor", action_t, (1,1))
    assert 0 <= action_idx < 11, f"action_idx out of range: {action_idx}"
    ok(f"select_action OK  idx={action_idx}")

    # Run 3 full episodes
    for ep in range(3):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r), f"episode {ep} reward is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}")

    # Check weights not nan/inf
    if np.isnan(agent.w).any() or np.isinf(agent.w).any():
        err("Weight matrix w contains NaN or Inf!")
    else:
        ok("Weight matrix w is finite")


def check_dqn(agent):
    env = FakeEnv()

    # Fill buffer enough to sample
    obs,_ = env.reset()
    for _ in range(50):
        a_t, a_idx = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(a_t)
        agent.store_transition(
            obs['policy'].cpu().float(), a_idx,
            r[0].item(), next_obs['policy'].cpu().float(),
            float(term[0].item())
        )
        obs = next_obs if not (term[0].item() or trunc[0].item()) else env.reset()[0]

    ok(f"Buffer filled: {len(agent.memory)} transitions")

    # Test _prepare_batch
    sample = agent._prepare_batch()
    if sample is None:
        err("_prepare_batch returned None — buffer not ready!")
    else:
        nfm, nfns, sb, ab, rb = sample
        ok(f"Batch shapes: states={tuple(sb.shape)} actions={tuple(ab.shape)} rewards={tuple(rb.shape)}")
        loss = agent.calculate_loss(nfm, nfns, sb, ab, rb)
        check_output("DQN loss", loss)

    # Run 3 full episodes
    for ep in range(3):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r), f"episode {ep} reward is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}")


def check_reinforce(agent):
    env = FakeEnv()

    # Run 3 full episodes
    for ep in range(3):
        r, loss, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r),    f"episode {ep} reward is NaN"
        assert not math.isnan(loss), f"episode {ep} loss is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  loss={loss:.4f}  steps={steps}")

    # Check policy network weights are finite
    for name, param in agent.policy_net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            err(f"Policy net param '{name}' has NaN/Inf!")
        else:
            ok(f"Policy param '{name}' finite")


def _run_on_policy_episodes(agent, algo_name, n_episodes=3):
    """Shared episode runner for AC, A2C, PPO."""
    env = FakeEnv()
    for ep in range(n_episodes):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r), f"{algo_name} episode {ep} reward is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}")

    # Check networks finite
    nets = []
    if hasattr(agent, 'actor_base'):  nets += list(agent.actor_base.parameters())
    if hasattr(agent, 'critic'):      nets += list(agent.critic.parameters())
    if hasattr(agent, 'backbone'):    nets += list(agent.backbone.parameters())
    if hasattr(agent, 'mu_head'):     nets += list(agent.mu_head.parameters())
    if hasattr(agent, 'value_head'):  nets += list(agent.value_head.parameters())
    for p in nets:
        if torch.isnan(p).any() or torch.isinf(p).any():
            err(f"Network parameter has NaN/Inf after training!")
            return
    ok("All network parameters finite after training")


def check_ac(agent):
    # Test that storage is pre-allocated (not None) — key OOM fix check
    if agent.storage is None:
        err("AC storage is None! _init_storage not called in __init__. OOM bug still present!")
    else:
        ok(f"AC storage pre-allocated (T={agent.storage.T}, N={agent.storage.N})")

    # Test that repeated learn() calls don't grow memory
    env = FakeEnv()
    for ep in range(5):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        storage_step_after = agent.storage.step
        assert not math.isnan(r), f"AC episode {ep} reward is NaN"
        # After learn(), storage should be cleared (step=0)
        if storage_step_after != 0:
            err(f"AC storage not cleared after episode! step={storage_step_after} (memory leak risk)")
        else:
            ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}  storage cleared ✓")


def check_a2c(agent):
    # Test zero-step rollout (episode ends on first step)
    env_instant_death = FakeEnv(max_steps=1)
    try:
        r, steps = agent.learn(env_instant_death, max_steps=MAX_STEPS)
        if math.isnan(r):
            err("A2C: NaN reward when episode ends on step 1 (T=0 edge case)")
        else:
            ok(f"A2C: zero-step edge case OK  reward={r:.2f}")
    except Exception as e:
        err(f"A2C crashed on zero-step episode: {e}")

    _run_on_policy_episodes(agent, "A2C")


def check_ppo(agent):
    # Same zero-step test
    env_instant_death = FakeEnv(max_steps=1)
    try:
        r, steps = agent.learn(env_instant_death, max_steps=MAX_STEPS)
        if math.isnan(r):
            err("PPO: NaN reward when episode ends on step 1 (T=0 edge case)")
        else:
            ok(f"PPO: zero-step edge case OK  reward={r:.2f}")
    except Exception as e:
        err(f"PPO crashed on zero-step episode: {e}")

    _run_on_policy_episodes(agent, "PPO")


def _fill_off_policy_buffer(agent, env, n=100):
    obs,_ = env.reset()
    for _ in range(n):
        if hasattr(agent, 'select_action'):
            a_t, raw = agent.select_action(obs)
        else:
            a_t = torch.zeros(1,1)
            raw = a_t
        next_obs, r, term, trunc, _ = env.step(a_t)
        done = bool(term[0].item()) or bool(trunc[0].item())

        st = obs['policy'].cpu().float()
        nt = next_obs['policy'].cpu().float()
        # Store scaled action (not raw) — this is the fixed behavior
        agent.store_transition(st, a_t.cpu(), r[0].item(), nt, float(term[0].item()))
        obs = next_obs if not done else env.reset()[0]


def check_td3(agent):
    env = FakeEnv()
    _fill_off_policy_buffer(agent, env, n=100)
    ok(f"Buffer filled: {len(agent.memory)} transitions")

    # Test action scale consistency:
    # stored actions should be in action_range, not unbounded raw values
    sample = agent.memory.sample()
    stored_actions = torch.cat([t.action.float() for t in sample])
    a_min, a_max = agent.action_range
    if stored_actions.min() < a_min - 1 or stored_actions.max() > a_max + 1:
        err(f"TD3 stored actions out of range! min={stored_actions.min():.2f} max={stored_actions.max():.2f}  "
            f"expected [{a_min}, {a_max}]  — storing raw (unscaled) actions is a bug!")
    else:
        ok(f"TD3 stored action range OK: [{stored_actions.min():.2f}, {stored_actions.max():.2f}]")

    # Test update
    agent.update_policy()
    ok("TD3 update_policy() ran without crash")

    # Run full episodes
    for ep in range(3):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r), f"TD3 episode {ep} reward is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}")


def check_sac(agent):
    env = FakeEnv()
    _fill_off_policy_buffer(agent, env, n=100)
    ok(f"Buffer filled: {len(agent.memory)} transitions")

    # Test action scale consistency (same as TD3)
    sample = agent.memory.sample()
    stored_actions = torch.cat([t.action.float() for t in sample])
    a_min, a_max = agent.action_range
    if stored_actions.min() < a_min - 1 or stored_actions.max() > a_max + 1:
        err(f"SAC stored actions out of range! min={stored_actions.min():.2f} max={stored_actions.max():.2f}  "
            f"expected [{a_min}, {a_max}]  — storing raw (unscaled) actions is a bug!")
    else:
        ok(f"SAC stored action range OK: [{stored_actions.min():.2f}, {stored_actions.max():.2f}]")

    # Test update
    agent.update_policy()
    ok("SAC update_policy() ran without crash")

    # Check alpha is positive
    if agent.alpha.item() <= 0:
        err(f"SAC alpha is non-positive: {agent.alpha.item()}")
    else:
        ok(f"SAC alpha = {agent.alpha.item():.4f}")

    # Run full episodes
    for ep in range(3):
        r, steps = agent.learn(env, max_steps=MAX_STEPS)
        assert not math.isnan(r), f"SAC episode {ep} reward is NaN"
        ok(f"Episode {ep+1}: reward={r:.2f}  steps={steps}")


# ══════════════════════════════════════════════════════════════════════
# SAVE / LOAD CHECK
# ══════════════════════════════════════════════════════════════════════

def check_save_load(name, agent):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ext  = ".json" if name == "Linear_QN" else ".pt"
        fname = f"test{ext}"
        try:
            agent.save_model(tmpdir, fname)
            fpath = os.path.join(tmpdir, fname)
            assert os.path.exists(fpath), "save_model did not create file!"
            ok(f"save_model OK  ({os.path.getsize(fpath)} bytes)")
        except Exception as e:
            err(f"save_model crashed: {e}")
            return

        try:
            agent.load_model(tmpdir, fname)
            ok("load_model OK")
        except Exception as e:
            err(f"load_model crashed: {e}")


# ══════════════════════════════════════════════════════════════════════
# IMPORT CHECK
# ══════════════════════════════════════════════════════════════════════

def check_imports():
    section("Import check")
    modules = {
        "RL_base_function":                  "RL_Algorithm.RL_base_function",
        "buffers":                           "RL_Algorithm.storage.buffers",
        "on_policy":                         "RL_Algorithm.storage.on_policy",
        "off_policy":                        "RL_Algorithm.storage.off_policy",
        "mlp":                               "RL_Algorithm.network.mlp",
        "Linear_Q":                          "RL_Algorithm.Function_based.Linear_Q",
        "DQN":                               "RL_Algorithm.Function_based.DQN",
        "MC_REINFORCE":                      "RL_Algorithm.Function_based.MC_REINFORCE",
        "AC":                                "RL_Algorithm.Function_based.AC",
        "A2C":                               "RL_Algorithm.Function_based.A2C",
        "PPO":                               "RL_Algorithm.Function_based.PPO",
        "TD3":                               "RL_Algorithm.Function_based.TD3",
        "SAC":                               "RL_Algorithm.Function_based.SAC",
    }
    all_ok = True
    for label, mod_path in modules.items():
        try:
            __import__(mod_path)
            ok(f"{label}")
        except Exception as e:
            err(f"{label}  →  {e}")
            all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════

CHECKS = {
    "Linear_QN":    check_linear_qn,
    "DQN":          check_dqn,
    "MC_REINFORCE": check_reinforce,
    "AC":           check_ac,
    "A2C":          check_a2c,
    "PPO":          check_ppo,
    "TD3":          check_td3,
    "SAC":          check_sac,
}

def main():
    print(f"\n{BOLD}{'═'*55}")
    print("  RL Algorithm Debug Suite")
    print(f"{'═'*55}{RESET}")

    if not check_imports():
        print(f"\n{RED}Fix import errors before continuing.{RESET}")
        return

    results = {}

    for name, maker_fn in MAKERS.items():
        section(f"{name}")
        try:
            agent = maker_fn()
            ok(f"Instantiated {name}")
        except Exception as e:
            err(f"Failed to instantiate {name}: {e}")
            traceback.print_exc()
            results[name] = "INIT FAILED"
            continue

        check_fn = CHECKS[name]
        try:
            check_fn(agent)
            check_save_load(name, agent)
            results[name] = "PASS"
        except AssertionError as e:
            err(f"Assertion failed: {e}")
            results[name] = "FAIL"
        except Exception as e:
            err(f"Unexpected crash: {e}")
            traceback.print_exc()
            results[name] = "CRASH"

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*55}")
    print("  SUMMARY")
    print(f"{'═'*55}{RESET}")
    for name, status in results.items():
        color = GREEN if status == "PASS" else RED
        print(f"  {color}{status:12s}{RESET}  {name}")

    n_pass  = sum(1 for s in results.values() if s == "PASS")
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} algorithms passed all checks.\n")

if __name__ == "__main__":
    main()