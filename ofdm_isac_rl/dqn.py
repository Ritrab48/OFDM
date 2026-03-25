"""Pure-Python DQN implementation for the reference model."""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass

from .config import SystemConfig, TrainingConfig


@dataclass(slots=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
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


class AdamState:
    def __init__(self, layers: list[dict[str, list[list[float]] | list[float]]]):
        self.step = 0
        self.weight_m = [self._zeros_like(layer["weights"]) for layer in layers]
        self.weight_v = [self._zeros_like(layer["weights"]) for layer in layers]
        self.bias_m = [self._zeros_like_vector(layer["biases"]) for layer in layers]
        self.bias_v = [self._zeros_like_vector(layer["biases"]) for layer in layers]

    def _zeros_like(self, matrix: list[list[float]]) -> list[list[float]]:
        return [[0.0 for _ in row] for row in matrix]

    def _zeros_like_vector(self, vector: list[float]) -> list[float]:
        return [0.0 for _ in vector]


class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        rng: random.Random,
    ):
        layer_sizes = [input_size, *hidden_sizes, output_size]
        self.layers: list[dict[str, list[list[float]] | list[float]]] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            weights = [
                [rng.uniform(-limit, limit) for _ in range(fan_in)]
                for _ in range(fan_out)
            ]
            biases = [0.0 for _ in range(fan_out)]
            self.layers.append({"weights": weights, "biases": biases})
        self.optimizer = AdamState(self.layers)

    def predict(self, inputs: list[float]) -> list[float]:
        activations = list(inputs)
        for index, layer in enumerate(self.layers):
            outputs: list[float] = []
            for weights, bias in zip(layer["weights"], layer["biases"]):
                value = sum(weight * feature for weight, feature in zip(weights, activations)) + bias
                if index < len(self.layers) - 1:
                    value = value if value > 0.0 else 0.0
                outputs.append(value)
            activations = outputs
        return activations

    def train_batch(
        self,
        batch: list[tuple[list[float], int, float]],
        learning_rate: float,
        clip_norm: float = 5.0,
    ) -> float:
        gradients = []
        for layer in self.layers:
            gradients.append(
                {
                    "weights": [[0.0 for _ in row] for row in layer["weights"]],
                    "biases": [0.0 for _ in layer["biases"]],
                }
            )

        total_loss = 0.0
        batch_scale = 1.0 / max(len(batch), 1)
        for inputs, action, target in batch:
            activations: list[list[float]] = [list(inputs)]
            pre_activations: list[list[float]] = []
            current = list(inputs)
            for index, layer in enumerate(self.layers):
                layer_output: list[float] = []
                layer_pre_activation: list[float] = []
                for weights, bias in zip(layer["weights"], layer["biases"]):
                    value = sum(weight * feature for weight, feature in zip(weights, current)) + bias
                    layer_pre_activation.append(value)
                    if index < len(self.layers) - 1:
                        layer_output.append(value if value > 0.0 else 0.0)
                    else:
                        layer_output.append(value)
                pre_activations.append(layer_pre_activation)
                activations.append(layer_output)
                current = layer_output

            q_values = activations[-1]
            error = q_values[action] - target
            total_loss += error * error

            deltas: list[list[float]] = []
            output_delta = [0.0 for _ in q_values]
            output_delta[action] = 2.0 * error * batch_scale
            deltas.append(output_delta)

            for layer_index in range(len(self.layers) - 2, -1, -1):
                downstream = deltas[0]
                next_weights = self.layers[layer_index + 1]["weights"]
                layer_delta: list[float] = []
                for neuron_index in range(len(self.layers[layer_index]["biases"])):
                    propagated = 0.0
                    for next_neuron_index, next_neuron_weights in enumerate(next_weights):
                        propagated += next_neuron_weights[neuron_index] * downstream[next_neuron_index]
                    gradient_gate = 1.0 if pre_activations[layer_index][neuron_index] > 0.0 else 0.0
                    layer_delta.append(propagated * gradient_gate)
                deltas.insert(0, layer_delta)

            for layer_index, delta in enumerate(deltas):
                previous_activations = activations[layer_index]
                for neuron_index, neuron_delta in enumerate(delta):
                    gradients[layer_index]["biases"][neuron_index] += neuron_delta
                    for input_index, activation in enumerate(previous_activations):
                        gradients[layer_index]["weights"][neuron_index][input_index] += (
                            neuron_delta * activation
                        )

        gradient_norm = 0.0
        for layer_grads in gradients:
            for row in layer_grads["weights"]:
                for value in row:
                    gradient_norm += value * value
            for value in layer_grads["biases"]:
                gradient_norm += value * value
        gradient_norm = math.sqrt(gradient_norm)
        if gradient_norm > clip_norm and gradient_norm > 0.0:
            scale = clip_norm / gradient_norm
            for layer_grads in gradients:
                for row in layer_grads["weights"]:
                    for index, value in enumerate(row):
                        row[index] = value * scale
                for index, value in enumerate(layer_grads["biases"]):
                    layer_grads["biases"][index] = value * scale

        self._adam_update(gradients, learning_rate)
        return total_loss * batch_scale

    def _adam_update(
        self,
        gradients: list[dict[str, list[list[float]] | list[float]]],
        learning_rate: float,
    ) -> None:
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1.0e-8
        self.optimizer.step += 1
        step = self.optimizer.step
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        for layer_index, (layer, layer_grads) in enumerate(zip(self.layers, gradients)):
            for neuron_index, row in enumerate(layer["weights"]):
                for weight_index, weight in enumerate(row):
                    grad = layer_grads["weights"][neuron_index][weight_index]
                    self.optimizer.weight_m[layer_index][neuron_index][weight_index] = (
                        beta1 * self.optimizer.weight_m[layer_index][neuron_index][weight_index]
                        + (1.0 - beta1) * grad
                    )
                    self.optimizer.weight_v[layer_index][neuron_index][weight_index] = (
                        beta2 * self.optimizer.weight_v[layer_index][neuron_index][weight_index]
                        + (1.0 - beta2) * grad * grad
                    )
                    m_hat = (
                        self.optimizer.weight_m[layer_index][neuron_index][weight_index]
                        / bias_correction1
                    )
                    v_hat = (
                        self.optimizer.weight_v[layer_index][neuron_index][weight_index]
                        / bias_correction2
                    )
                    row[weight_index] = weight - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)

            for bias_index, bias in enumerate(layer["biases"]):
                grad = layer_grads["biases"][bias_index]
                self.optimizer.bias_m[layer_index][bias_index] = (
                    beta1 * self.optimizer.bias_m[layer_index][bias_index]
                    + (1.0 - beta1) * grad
                )
                self.optimizer.bias_v[layer_index][bias_index] = (
                    beta2 * self.optimizer.bias_v[layer_index][bias_index]
                    + (1.0 - beta2) * grad * grad
                )
                m_hat = self.optimizer.bias_m[layer_index][bias_index] / bias_correction1
                v_hat = self.optimizer.bias_v[layer_index][bias_index] / bias_correction2
                layer["biases"][bias_index] = (
                    bias - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
                )

    def copy(self) -> "MLP":
        clone = object.__new__(MLP)
        clone.layers = copy.deepcopy(self.layers)
        clone.optimizer = AdamState(clone.layers)
        return clone

    def load_from(self, other: "MLP") -> None:
        self.layers = copy.deepcopy(other.layers)
        self.optimizer = AdamState(self.layers)

    def to_dict(self) -> dict[str, list[dict[str, list[list[float]] | list[float]]]]:
        return {"layers": copy.deepcopy(self.layers)}


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

    def encode_state(self, state: list[float]) -> list[float]:
        epsilon_hat, mismatch, sinr, pilot_snr, channel_power, rho, beta = state
        epsilon_scale = max(
            1.0,
            self.system_config.subcarrier_spacing_hz * 0.25,
        )
        features = [
            epsilon_hat / epsilon_scale,
            mismatch / max(self.system_config.noise_variance, 1.0e-9),
            math.log1p(max(sinr, 0.0)) / self.training_config.log_feature_scale,
            math.log1p(max(pilot_snr, 0.0)) / self.training_config.log_feature_scale,
            channel_power / self.training_config.channel_power_scale,
            rho,
            beta,
        ]
        return [self._clip(feature) for feature in features]

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
        return max(range(len(q_values)), key=lambda index: q_values[index])

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
        training_examples: list[tuple[list[float], int, float]] = []
        for transition in batch:
            target = transition.reward
            if not transition.done:
                target += self.training_config.gamma * max(
                    self.target_network.predict(transition.next_state)
                )
            training_examples.append((transition.state, transition.action, target))
        loss = self.q_network.train_batch(
            training_examples,
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

    def _clip(self, value: float) -> float:
        limit = self.training_config.state_clip
        return max(-limit, min(limit, value))
