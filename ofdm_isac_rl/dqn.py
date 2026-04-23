"""NumPy-accelerated DQN implementation for the OFDM-ISAC reference model.

Public API is identical to the earlier pure-Python version:
    * ReplayBuffer, Transition, MLP, DQNAgent
    * DQNAgent.select_action / remember / train_step / act_greedily / epsilon / export

Serialization format for `MLP.to_dict()` is preserved (nested Python lists) so
that old artifacts remain readable and new artifacts remain comparable.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SystemConfig, TrainingConfig


# ── Transition & replay buffer ────────────────────────────────────────────────

@dataclass(slots=True)
class Transition:
    state: np.ndarray       # (input_dim,) float32
    action: int
    reward: float
    next_state: np.ndarray  # (input_dim,) float32
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.storage: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.storage), batch_size)

    def __len__(self) -> int:
        return len(self.storage)


# ── Vectorised MLP with Adam ──────────────────────────────────────────────────

class MLP:
    """Feed-forward network with ReLU hidden layers and linear output.

    Stores weights as (out_dim, in_dim) arrays so that the forward pass is
    ``h = relu(x @ W.T + b)``, matching the original row-per-neuron layout
    and keeping the serialized format unchanged.
    """

    DTYPE = np.float64

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        rng: random.Random,
    ):
        seed = rng.randrange(2 ** 31)
        self._np_rng = np.random.default_rng(seed)
        layer_sizes = [input_size, *hidden_sizes, output_size]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            W = self._np_rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(self.DTYPE)
            b = np.zeros(fan_out, dtype=self.DTYPE)
            self.weights.append(W)
            self.biases.append(b)

        # Adam state
        self._m_w = [np.zeros_like(W) for W in self.weights]
        self._v_w = [np.zeros_like(W) for W in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._adam_step = 0

    # ── forward ───────────────────────────────────────────────────────────────
    def predict(self, x) -> np.ndarray:
        """Forward pass. Accepts a 1-D vector or a 2-D batch."""
        a = np.asarray(x, dtype=self.DTYPE)
        single = (a.ndim == 1)
        if single:
            a = a[None, :]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b
            a = np.maximum(z, 0.0) if i < len(self.weights) - 1 else z
        return a[0] if single else a

    # ── training step ────────────────────────────────────────────────────────
    def train_batch(
        self,
        states: np.ndarray,   # (B, in_dim)
        actions: np.ndarray,  # (B,) int
        targets: np.ndarray,  # (B,) float
        learning_rate: float,
        clip_norm: float = 5.0,
    ) -> float:
        states  = np.asarray(states,  dtype=self.DTYPE)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=self.DTYPE)
        B = states.shape[0]
        if B == 0:
            return 0.0

        # Forward, caching activations and pre-activations
        activations: list[np.ndarray] = [states]
        pre_activations: list[np.ndarray] = []
        a = states
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b
            pre_activations.append(z)
            a = np.maximum(z, 0.0) if i < len(self.weights) - 1 else z
            activations.append(a)

        q_values = activations[-1]                          # (B, n_actions)
        selected = q_values[np.arange(B), actions]          # (B,)
        errors = selected - targets                         # (B,)
        loss = float((errors ** 2).sum()) / B               # MSE

        # Build output delta (same scaling as original: 2 * err / B at chosen action)
        output_delta = np.zeros_like(q_values)
        output_delta[np.arange(B), actions] = 2.0 * errors / B

        # Backward
        grad_W: list[np.ndarray] = [None] * len(self.weights)  # type: ignore[assignment]
        grad_b: list[np.ndarray] = [None] * len(self.biases)   # type: ignore[assignment]
        delta = output_delta
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[i]                         # (B, in_dim_i)
            grad_W[i] = delta.T @ a_prev                    # (out_i, in_i)
            grad_b[i] = delta.sum(axis=0)                   # (out_i,)
            if i > 0:
                relu_mask = (pre_activations[i - 1] > 0.0).astype(self.DTYPE)
                delta = (delta @ self.weights[i]) * relu_mask

        # Global-norm gradient clipping
        norm_sq = 0.0
        for g in grad_W: norm_sq += float((g * g).sum())
        for g in grad_b: norm_sq += float((g * g).sum())
        total_norm = math.sqrt(norm_sq)
        if total_norm > clip_norm and total_norm > 0.0:
            scale = clip_norm / total_norm
            for g in grad_W: g *= scale
            for g in grad_b: g *= scale

        # Adam update
        self._adam_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1.0e-8
        bc1 = 1.0 - beta1 ** self._adam_step
        bc2 = 1.0 - beta2 ** self._adam_step
        for i in range(len(self.weights)):
            self._m_w[i] = beta1 * self._m_w[i] + (1.0 - beta1) * grad_W[i]
            self._v_w[i] = beta2 * self._v_w[i] + (1.0 - beta2) * (grad_W[i] ** 2)
            m_hat = self._m_w[i] / bc1
            v_hat = self._v_w[i] / bc2
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            self._m_b[i] = beta1 * self._m_b[i] + (1.0 - beta1) * grad_b[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1.0 - beta2) * (grad_b[i] ** 2)
            m_hat = self._m_b[i] / bc1
            v_hat = self._v_b[i] / bc2
            self.biases[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return loss

    # ── cloning & serialisation ──────────────────────────────────────────────
    def copy(self) -> "MLP":
        clone = object.__new__(MLP)
        clone._np_rng = np.random.default_rng(0)   # target net does not train
        clone.weights = [W.copy() for W in self.weights]
        clone.biases  = [b.copy() for b in self.biases]
        clone._m_w = [np.zeros_like(W) for W in clone.weights]
        clone._v_w = [np.zeros_like(W) for W in clone.weights]
        clone._m_b = [np.zeros_like(b) for b in clone.biases]
        clone._v_b = [np.zeros_like(b) for b in clone.biases]
        clone._adam_step = 0
        return clone

    def load_from(self, other: "MLP") -> None:
        self.weights = [W.copy() for W in other.weights]
        self.biases  = [b.copy() for b in other.biases]
        # Target network is not trained; drop its optimiser state.
        self._m_w = [np.zeros_like(W) for W in self.weights]
        self._v_w = [np.zeros_like(W) for W in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._adam_step = 0

    def to_dict(self) -> dict[str, list[dict[str, list[list[float]] | list[float]]]]:
        """Nested-list format preserved for backward compatibility."""
        layers = []
        for W, b in zip(self.weights, self.biases):
            layers.append({
                "weights": [row.tolist() for row in W],
                "biases":  b.tolist(),
            })
        return {"layers": layers}


# ── DQN agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, system_config: SystemConfig, training_config: TrainingConfig):
        self.system_config = system_config
        self.training_config = training_config
        self.rng = random.Random(training_config.seed)
        self.action_space = list(system_config.action_space)
        self.q_network = MLP(
            input_size=7,
            hidden_sizes=training_config.hidden_sizes,
            output_size=len(self.action_space),
            rng=self.rng,
        )
        self.target_network = self.q_network.copy()
        self.replay = ReplayBuffer(training_config.replay_capacity, self.rng)
        self.total_steps = 0

    def encode_state(self, state: list[float]) -> np.ndarray:
        epsilon_hat, mismatch, sinr, pilot_snr, channel_power, rho, beta = state
        epsilon_scale = max(1.0, self.system_config.subcarrier_spacing_hz * 0.25)
        clip = self.training_config.state_clip
        features = np.array([
            epsilon_hat / epsilon_scale,
            mismatch / max(self.system_config.noise_variance, 1.0e-9),
            math.log1p(max(sinr, 0.0)) / self.training_config.log_feature_scale,
            math.log1p(max(pilot_snr, 0.0)) / self.training_config.log_feature_scale,
            channel_power / self.training_config.channel_power_scale,
            rho,
            beta,
        ], dtype=MLP.DTYPE)
        np.clip(features, -clip, clip, out=features)
        return features

    def epsilon(self) -> float:
        progress = min(self.total_steps, self.training_config.epsilon_decay_steps)
        ratio = progress / max(self.training_config.epsilon_decay_steps, 1)
        return (
            self.training_config.epsilon_start
            + ratio * (self.training_config.epsilon_end - self.training_config.epsilon_start)
        )

    def select_action(self, state: list[float], explore: bool = True) -> int:
        encoded = self.encode_state(state)
        if explore and self.rng.random() < self.epsilon():
            return self.rng.randrange(len(self.action_space))
        q_values = self.q_network.predict(encoded)
        return int(np.argmax(q_values))

    def remember(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.replay.add(
            Transition(
                state=self.encode_state(state),
                action=action,
                reward=reward,
                next_state=self.encode_state(next_state),
                done=done,
            )
        )

    def train_step(self) -> float | None:
        self.total_steps += 1
        if len(self.replay) < max(self.training_config.warmup_steps, self.training_config.batch_size):
            return None
        if self.total_steps % self.training_config.train_interval != 0:
            return None

        batch = self.replay.sample(self.training_config.batch_size)
        B = len(batch)
        states      = np.empty((B, 7),     dtype=MLP.DTYPE)
        next_states = np.empty((B, 7),     dtype=MLP.DTYPE)
        actions     = np.empty(B,           dtype=np.int64)
        rewards     = np.empty(B,           dtype=MLP.DTYPE)
        dones       = np.empty(B,           dtype=bool)
        for i, tr in enumerate(batch):
            states[i]      = tr.state
            next_states[i] = tr.next_state
            actions[i]     = tr.action
            rewards[i]     = tr.reward
            dones[i]       = tr.done

        # Bellman targets: r + gamma * max_a' Q_target(s', a')   if not done
        next_q = self.target_network.predict(next_states)        # (B, n_actions)
        max_next_q = next_q.max(axis=1)                          # (B,)
        targets = rewards + self.training_config.gamma * max_next_q * (~dones)

        loss = self.q_network.train_batch(
            states, actions, targets,
            learning_rate=self.training_config.learning_rate,
        )

        if self.total_steps % self.training_config.target_update_interval == 0:
            self.target_network.load_from(self.q_network)
        return loss

    def act_greedily(self, state: list[float]) -> int:
        return self.select_action(state, explore=False)

    def export(self) -> dict[str, object]:
        return {
            "network": self.q_network.to_dict(),
            "total_steps": self.total_steps,
            "actions": [
                {"rho": rho, "beta": beta} for rho, beta in self.action_space
            ],
        }
