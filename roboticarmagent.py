"""
roboticarmagent.py
DDPG (Deep Deterministic Policy Gradient) agent for a robotic arm reaching task.

Supports two environment backends (auto-detected):
  1. gymnasium + mujoco  ->  Reacher-v5  (best visuals, needs MuJoCo)
  2. Built-in pure-Python 2-DOF planar arm (zero extra install, always works)

Usage
-----
  # Train (saves checkpoints to runs/ddpg_reacher/)
  python roboticarmagent.py --mode train

  # Resume training from a checkpoint (NEVER lose progress again)
  python roboticarmagent.py --mode train --resume

  # Evaluate the best saved model
  python roboticarmagent.py --mode eval

  # Evaluate with rendering (MuJoCo backend only)
  python roboticarmagent.py --mode eval --render

Pip install
-----------
  pip install torch numpy gymnasium
  # Optional (better visuals):
  pip install mujoco "gymnasium[mujoco]"

IMPORTANT: Always run from a terminal (cmd / PowerShell / bash).
Do NOT double-click the .py file or the window will close instantly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1.  ENVIRONMENT
# ============================================================

def _try_make_mujoco(render: bool):
    """Return a Reacher-v5 Gymnasium env, or None if MuJoCo is unavailable."""
    try:
        import gymnasium as gym
        env = gym.make("Reacher-v5",
                       render_mode="human" if render else None)
        return env
    except Exception:
        return None


class PlanarReacherEnv:
    """
    Minimal 2-DOF planar robotic-arm environment — no external dependencies.

    State  (6,) : [cos(q1), sin(q1), cos(q2), sin(q2), tip_x, tip_y]
    Action (2,) : joint torques in [-1, 1]
    Reward      : -distance(fingertip, target)   (dense, ~0 when solved)
    """

    LINK_LEN  = 0.10
    DT        = 0.05
    MAX_SPEED = 8.0

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.obs_dim   = 6
        self.act_dim   = 2
        self.act_low   = np.full(2, -1.0, dtype=np.float32)
        self.act_high  = np.full(2,  1.0, dtype=np.float32)
        self._q        = np.zeros(2)
        self._dq       = np.zeros(2)
        self._target   = np.zeros(2)
        self._step_cnt = 0

    class _Box:
        def __init__(self, low, high, shape):
            self.low, self.high, self.shape = low, high, shape
        def sample(self):
            return np.random.uniform(self.low, self.high).astype(np.float32)

    @property
    def observation_space(self):
        lo = np.array([-1, -1, -1, -1, -0.25, -0.25], dtype=np.float32)
        return self._Box(lo, -lo, (6,))

    @property
    def action_space(self):
        return self._Box(self.act_low, self.act_high, (2,))

    def _fingertip(self):
        x = self.LINK_LEN * (np.cos(self._q[0]) + np.cos(self._q[0] + self._q[1]))
        y = self.LINK_LEN * (np.sin(self._q[0]) + np.sin(self._q[0] + self._q[1]))
        return np.array([x, y])

    def _obs(self):
        ft = self._fingertip()
        return np.array([np.cos(self._q[0]), np.sin(self._q[0]),
                         np.cos(self._q[1]), np.sin(self._q[1]),
                         ft[0], ft[1]], dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        rng = np.random.default_rng(seed)
        self._q        = rng.uniform(-np.pi, np.pi, 2)
        self._dq       = np.zeros(2)
        angle          = rng.uniform(0, 2 * np.pi)
        r              = rng.uniform(0.05, 0.17)
        self._target   = np.array([r * np.cos(angle), r * np.sin(angle)])
        self._step_cnt = 0
        return self._obs(), {}

    def step(self, action):
        action       = np.clip(action, -1.0, 1.0)
        self._dq     = np.clip(self._dq + self.DT * action, -self.MAX_SPEED, self.MAX_SPEED)
        self._q     += self.DT * self._dq
        self._step_cnt += 1
        dist         = np.linalg.norm(self._fingertip() - self._target)
        reward       = -float(dist)
        terminated   = bool(dist < 0.01)
        truncated    = bool(self._step_cnt >= self.max_steps)
        return self._obs(), reward, terminated, truncated, {}

    def close(self): pass
    def render(self):
        ft = self._fingertip()
        d  = np.linalg.norm(ft - self._target)
        print(f"  tip=({ft[0]:.3f},{ft[1]:.3f})  "
              f"target=({self._target[0]:.3f},{self._target[1]:.3f})  dist={d:.4f}")


def make_env(render: bool = False, use_mujoco=None, max_ep_len: int = 200):
    if use_mujoco is not False:
        env = _try_make_mujoco(render)
        if env is not None:
            return env, "mujoco"
    return PlanarReacherEnv(max_steps=max_ep_len), "builtin"


# ============================================================
# 2.  UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningMeanStd:
    """Online Welford mean/variance for observation normalisation."""

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x       = np.atleast_2d(np.asarray(x, dtype=np.float64))
        b_mean  = x.mean(axis=0)
        b_var   = x.var(axis=0)
        b_count = x.shape[0]
        delta   = b_mean - self.mean
        tot     = self.count + b_count
        self.mean  = self.mean + delta * b_count / tot
        m2         = (self.var * self.count + b_var * b_count
                      + delta**2 * self.count * b_count / tot)
        self.var   = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(),
                "var":  self.var.tolist(),
                "count": self.count}

    def load_state_dict(self, d):
        self.mean  = np.array(d["mean"],  dtype=np.float64)
        self.var   = np.array(d["var"],   dtype=np.float64)
        self.count = float(d["count"])


# ============================================================
# 3.  REPLAY BUFFER
# ============================================================

@dataclass
class Transition:
    obs:      np.ndarray
    act:      np.ndarray
    rew:      float
    next_obs: np.ndarray
    done:     float


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 500_000):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew      = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)
        self.ptr = self.size = 0

    def add(self, t: Transition):
        i = self.ptr
        self.obs[i]      = t.obs
        self.act[i]      = t.act
        self.rew[i]      = t.rew
        self.next_obs[i] = t.next_obs
        self.done[i]     = t.done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: torch.tensor(v[idx], device=device)
                for k, v in [("obs",      self.obs),
                              ("act",      self.act),
                              ("rew",      self.rew),
                              ("next_obs", self.next_obs),
                              ("done",     self.done)]}


# ============================================================
# 4.  NETWORKS
# ============================================================

def _mlp(sizes, activation=nn.ReLU, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act = activation if i < len(sizes) - 2 else out_activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = _mlp([obs_dim, 256, 256, act_dim], out_activation=nn.Tanh)
        self.register_buffer("act_limit",
                             torch.tensor(act_limit, dtype=torch.float32))

    def forward(self, obs):
        return self.net(obs) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = _mlp([obs_dim + act_dim, 256, 256, 1])

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


# ============================================================
# 5.  DDPG AGENT
# ============================================================

class DDPG:
    def __init__(self, obs_dim, act_dim, act_low, act_high,
                 gamma=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3,
                 device="cpu"):
        self.device      = torch.device(device)
        act_limit        = np.maximum(np.abs(act_low),
                                      np.abs(act_high)).astype(np.float32)
        self.actor       = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.actor_targ  = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.critic      = Critic(obs_dim, act_dim).to(self.device)
        self.critic_targ = Critic(obs_dim, act_dim).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.actor_opt   = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt  = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma       = gamma
        self.tau         = tau
        self.act_low_np  = act_low
        self.act_high_np = act_high

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        a = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise_std > 0:
            a = a + noise_std * np.random.randn(*a.shape).astype(np.float32)
        return np.clip(a, self.act_low_np, self.act_high_np).astype(np.float32)

    def update(self, batch):
        obs, act, rew, nobs, done = (batch["obs"], batch["act"], batch["rew"],
                                     batch["next_obs"], batch["done"])
        # Critic
        with torch.no_grad():
            na     = self.actor_targ(nobs)
            q_targ = rew + self.gamma * (1 - done) * self.critic_targ(nobs, na)
        c_loss = nn.functional.mse_loss(self.critic(obs, act), q_targ)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()
        # Actor
        a_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()
        # Polyak
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(),
                             self.actor_targ.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic.parameters(),
                             self.critic_targ.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(a_loss.item()), float(c_loss.item())

    # ── Save / Load (includes RMS so progress is never lost) ───────────────

    def save(self, out_dir: Path, rms: RunningMeanStd,
             episode: int, tag: str = "latest"):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   out_dir / f"{tag}_actor.pt")
        torch.save(self.critic.state_dict(),
                   out_dir / f"{tag}_critic.pt")
        torch.save(self.actor_targ.state_dict(),
                   out_dir / f"{tag}_actor_targ.pt")
        torch.save(self.critic_targ.state_dict(),
                   out_dir / f"{tag}_critic_targ.pt")
        with open(out_dir / f"{tag}_meta.json", "w") as f:
            json.dump({"episode": episode, "rms": rms.state_dict()}, f)

    def load(self, out_dir: Path, rms: RunningMeanStd,
             tag: str = "latest") -> int:
        def _ld(name):
            return torch.load(out_dir / f"{tag}_{name}.pt",
                              map_location=self.device,
                              weights_only=True)
        self.actor.load_state_dict(_ld("actor"))
        self.critic.load_state_dict(_ld("critic"))
        self.actor_targ.load_state_dict(_ld("actor_targ"))
        self.critic_targ.load_state_dict(_ld("critic_targ"))
        with open(out_dir / f"{tag}_meta.json") as f:
            meta = json.load(f)
        rms.load_state_dict(meta["rms"])
        return int(meta["episode"])


# ============================================================
# 6.  TRAINING
# ============================================================

def train(args):
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"Device      : {device}")

    env, backend = make_env(render=False, use_mujoco=args.mujoco,
                            max_ep_len=args.max_ep_len)
    print(f"Backend     : {backend}")

    obs_dim  = int(np.prod(env.observation_space.shape))
    act_dim  = int(np.prod(env.action_space.shape))
    act_low  = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    agent = DDPG(obs_dim, act_dim, act_low, act_high,
                 gamma=args.gamma, tau=args.tau,
                 actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                 device=device)
    buf       = ReplayBuffer(obs_dim, act_dim, capacity=args.replay_size)
    rms       = RunningMeanStd(shape=(obs_dim,))
    out_dir   = Path(args.out_dir)
    start_ep  = 1
    best_ret  = -1e9

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if args.resume and (out_dir / "latest_meta.json").exists():
        start_ep = agent.load(out_dir, rms, tag="latest") + 1
        print(f"Resumed from episode {start_ep - 1}")
    else:
        print("Starting fresh training run.")

    # ── Warmup random exploration ───────────────────────────────────────────
    print(f"Warmup: {args.warmup_steps} random steps …")
    obs, _ = env.reset(seed=args.seed)
    for _ in range(args.warmup_steps):
        act = env.action_space.sample()
        nobs, rew, term, trunc, _ = env.step(act)
        buf.add(Transition(obs, act, float(rew), nobs, float(term or trunc)))
        rms.update(obs.reshape(1, -1))
        obs = nobs
        if term or trunc:
            obs, _ = env.reset()
    print("Warmup done. Starting training …\n")

    total_steps = 0
    t0          = time.time()

    for ep in range(start_ep, args.episodes + 1):
        obs, _    = env.reset(seed=args.seed + ep)
        ep_ret    = 0.0
        ep_len    = 0

        while True:
            rms.update(obs.reshape(1, -1))
            obs_n = rms.normalize(obs)
            act   = agent.act(obs_n, noise_std=args.exploration_noise)
            nobs, rew, term, trunc, _ = env.step(act)
            done  = float(term or trunc)
            buf.add(Transition(obs_n, act, float(rew),
                               rms.normalize(nobs), done))
            obs         = nobs
            ep_ret     += float(rew)
            ep_len     += 1
            total_steps += 1

            if buf.size >= args.batch_size:
                agent.update(buf.sample(args.batch_size, agent.device))

            if term or trunc or ep_len >= args.max_ep_len:
                break

        if ep % args.log_every == 0:
            print(f"[train] ep={ep:5d}  return={ep_ret:8.2f}  "
                  f"len={ep_len:4d}  buf={buf.size:>7d}  "
                  f"elapsed={time.time()-t0:.0f}s")

        if ep % args.save_every == 0:
            agent.save(out_dir, rms, ep, tag="latest")
            print(f"         Checkpoint saved (ep {ep})")

        if ep % args.eval_every == 0:
            avg = _run_eval(agent, args, rms)
            print(f"[eval ] ep={ep:5d}  avg_return={avg:8.2f}")
            if avg > best_ret:
                best_ret = avg
                agent.save(out_dir, rms, ep, tag="best")
                print(f"         ★ New best {best_ret:.2f} — saved")

    env.close()
    # Always save at very end so resume is always possible
    agent.save(out_dir, rms, args.episodes, tag="latest")
    print(f"\nTraining complete.  Best eval return: {best_ret:.2f}")
    print(f"Files saved in: {out_dir.resolve()}")


# ============================================================
# 7.  EVALUATION
# ============================================================

@torch.no_grad()
def _run_eval(agent: DDPG, args, rms: RunningMeanStd,
              render: bool = False, n: int | None = None) -> float:
    env, _ = make_env(render=render, use_mujoco=args.mujoco,
                      max_ep_len=args.max_ep_len)
    n = n or args.eval_episodes
    returns = []
    for ep in range(n):
        obs, _ = env.reset(seed=args.seed + 10_000 + ep)
        ep_ret = 0.0
        ep_len = 0
        while True:
            obs_n = rms.normalize(obs.astype(np.float32))
            act   = agent.act(obs_n, noise_std=0.0)
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += float(rew)
            ep_len += 1
            if render:
                time.sleep(1 / 60)
            if term or trunc or ep_len >= args.max_ep_len:
                break
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def evaluate_mode(args):
    device  = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out_dir)

    env, backend = make_env(render=False, use_mujoco=args.mujoco,
                            max_ep_len=args.max_ep_len)
    obs_dim  = int(np.prod(env.observation_space.shape))
    act_dim  = int(np.prod(env.action_space.shape))
    act_low  = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)
    env.close()

    agent = DDPG(obs_dim, act_dim, act_low, act_high, device=device)
    rms   = RunningMeanStd(shape=(obs_dim,))
    tag   = "best" if (out_dir / "best_meta.json").exists() else "latest"
    ep    = agent.load(out_dir, rms, tag=tag)
    print(f"Loaded '{tag}' checkpoint (episode {ep}), backend={backend}")

    avg = _run_eval(agent, args, rms, render=args.render, n=args.eval_episodes)
    print(f"Average return over {args.eval_episodes} episodes: {avg:.2f}")


# ============================================================
# 8.  ENTRY POINT
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="DDPG Robotic Arm Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",    choices=["train", "eval"], default="train")
    p.add_argument("--out_dir", default="runs/ddpg_reacher")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint (never lose progress)")
    p.add_argument("--render",  action="store_true",
                   help="Render during eval (MuJoCo backend only)")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--mujoco",    dest="mujoco", action="store_true",
                   default=None, help="Force MuJoCo backend")
    g.add_argument("--no_mujoco", dest="mujoco", action="store_false",
                   help="Force built-in backend")

    p.add_argument("--episodes",          type=int,   default=800)
    p.add_argument("--max_ep_len",        type=int,   default=200)
    p.add_argument("--replay_size",       type=int,   default=500_000)
    p.add_argument("--warmup_steps",      type=int,   default=5_000)
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--tau",               type=float, default=0.005)
    p.add_argument("--actor_lr",          type=float, default=1e-3)
    p.add_argument("--critic_lr",         type=float, default=1e-3)
    p.add_argument("--exploration_noise", type=float, default=0.2)
    p.add_argument("--save_every",        type=int,   default=50)
    p.add_argument("--eval_every",        type=int,   default=25)
    p.add_argument("--eval_episodes",     type=int,   default=5)
    p.add_argument("--log_every",         type=int,   default=10)
    p.add_argument("--seed",              type=int,   default=123)
    p.add_argument("--cuda",              action="store_true")

    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate_mode(args)


if __name__ == "__main__":
    main()
