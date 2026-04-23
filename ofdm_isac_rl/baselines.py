"""Baseline policies for the OFDM-ISAC simulator."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import SystemConfig


@dataclass(slots=True)
class Policy:
    name: str

    def select_action(self, state: list[float]) -> int:
        raise NotImplementedError


@dataclass(slots=True)
class FixedActionPolicy(Policy):
    action_index: int

    def select_action(self, state: list[float]) -> int:
        del state
        return self.action_index


class RandomPolicy(Policy):
    def __init__(self, config: SystemConfig, seed: int):
        super().__init__(name="random")
        self._action_count = len(config.action_space)
        self._rng = random.Random(seed)

    def select_action(self, state: list[float]) -> int:
        del state
        return self._rng.randrange(self._action_count)


class HeuristicPolicy(Policy):
    def __init__(self, config: SystemConfig):
        super().__init__(name="heuristic")
        self.config = config
        stationary_cfo_std = config.cfo_noise_std_hz / math.sqrt(
            1.0 - config.cfo_mean_reversion ** 2
        )
        self.cfo_low = 0.75 * stationary_cfo_std
        self.cfo_high = 1.50 * stationary_cfo_std
        self.sync_low = 1.5 * config.noise_variance
        self.sync_high = 2.5 * config.noise_variance
        self.pilot_snr_low = 0.8
        self.pilot_snr_high = 1.6

    def select_action(self, state: list[float]) -> int:
        epsilon_hat, sync_mismatch, _, pilot_snr, channel_power, _, _, *_ = state
        channel_adjust = 0 if channel_power >= 1.0 else 1
        rho_index = 0
        if abs(epsilon_hat) >= self.cfo_high or sync_mismatch >= self.sync_high:
            rho_index = min(len(self.config.pilot_ratios) - 1, 3 + channel_adjust)
        elif abs(epsilon_hat) >= self.cfo_low or sync_mismatch >= self.sync_low:
            rho_index = min(len(self.config.pilot_ratios) - 1, 1 + channel_adjust)

        beta_index = 0
        if pilot_snr <= self.pilot_snr_low or sync_mismatch >= self.sync_high:
            beta_index = 2
        elif pilot_snr <= self.pilot_snr_high or abs(epsilon_hat) >= self.cfo_low:
            beta_index = 1

        rho = self.config.pilot_ratios[rho_index]
        beta = self.config.pilot_power_fractions[beta_index]
        return self.config.action_space.index((rho, beta))


def build_baseline_policies(config: SystemConfig, seed: int) -> list[Policy]:
    policies: list[Policy] = []
    for action_index, (rho, beta) in enumerate(config.action_space):
        policies.append(
            FixedActionPolicy(
                name=f"fixed_rho={rho:.1f}_beta={beta:.1f}",
                action_index=action_index,
            )
        )
    policies.append(RandomPolicy(config=config, seed=seed))
    policies.append(HeuristicPolicy(config=config))
    return policies
