"""Environment for the REFERENCE_MODEL_FINAL.md formulation."""

from __future__ import annotations

import cmath
import math
import random

from .config import SystemConfig


class OfdmIsacEnv:
    """Frame-level OFDM-ISAC simulator with observable metrics."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.frame_index = 0
        self.current_cfo_hz = 0.0
        self.current_channel = complex(1.0, 0.0)
        self.current_regime: int = 0   # 0=low-drift, 1=high-drift
        self.prev_epsilon_hat: float = 0.0
        self.previous_state = self._initial_state()
        self.delay_grid = self._build_delay_grid()

    def reset(self) -> list[float]:
        self.frame_index = 0
        self.current_cfo_hz = self._sample_stationary_cfo()
        self.current_channel = self._sample_complex_gaussian()
        self.current_regime = 0
        self.prev_epsilon_hat = 0.0
        self.previous_state = self._initial_state()
        return list(self.previous_state)

    def step(self, action_index: int) -> tuple[list[float], float, bool, dict[str, float]]:
        rho, beta = self.config.action_space[action_index]
        num_pilots = self._pilot_count(rho)
        pilot_indices = self._pilot_indices(num_pilots)
        pilot_amplitude = math.sqrt(beta * self.config.average_power / num_pilots)
        observations = self._simulate_pilot_observations(
            pilot_indices=pilot_indices,
            pilot_amplitude=pilot_amplitude,
        )

        epsilon_hat = self._estimate_cfo(observations)
        sync_mismatch = self._compute_sync_mismatch(observations, epsilon_hat)
        channel_power_estimate = self._estimate_channel_power(
            observations=observations,
            pilot_amplitude=pilot_amplitude,
            epsilon_hat=epsilon_hat,
        )
        pilot_snr = (
            beta
            * self.config.average_power
            * channel_power_estimate
            / (num_pilots * self.config.noise_variance)
        )
        empirical_sensing_metric = self._compute_empirical_sensing_metric(
            observations=observations,
            pilot_indices=pilot_indices,
            pilot_amplitude=pilot_amplitude,
            epsilon_hat=epsilon_hat,
        )
        sensing_metric = self._compute_sensing_metric(
            rho=rho,
            beta=beta,
            empirical_metric=empirical_sensing_metric,
        )

        residual_cfo = self.current_cfo_hz - epsilon_hat
        actual_channel_power = abs(self.current_channel) ** 2
        data_power = (1.0 - beta) * self.config.average_power
        residual_cfo_sq_estimate = self._estimate_residual_cfo_power(
            sync_mismatch=sync_mismatch,
            beta=beta,
            num_pilots=num_pilots,
            channel_power_estimate=channel_power_estimate,
        )
        observed_ici_power = (
            self.config.ici_constant
            * residual_cfo_sq_estimate
            * data_power
            * channel_power_estimate
        )
        sinr = (
            data_power
            * channel_power_estimate
            / (self.config.noise_variance + observed_ici_power)
        )
        true_ici_power = (
            self.config.ici_constant
            * (residual_cfo ** 2)
            * data_power
            * actual_channel_power
        )
        true_sinr = (
            data_power * actual_channel_power
            / (self.config.noise_variance + true_ici_power)
        )
        rate = (1.0 - rho) * math.log2(1.0 + max(sinr, 0.0))
        true_rate = (1.0 - rho) * math.log2(1.0 + max(true_sinr, 0.0))
        normalized_rate = self._normalize_rate(rate)
        normalized_sync = self._normalize_sync(sync_mismatch)
        normalized_sensing = self._normalize_sensing(sensing_metric)
        reward = (
            self.config.reward_rate_weight * normalized_rate
            - self.config.reward_sync_weight * normalized_sync
            - self.config.reward_sensing_weight * normalized_sensing
        )

        delta_epsilon_hat = epsilon_hat - self.prev_epsilon_hat
        self.prev_epsilon_hat = epsilon_hat

        next_state = [
            epsilon_hat,
            sync_mismatch,
            sinr,
            pilot_snr,
            channel_power_estimate,
            rho,
            beta,
            delta_epsilon_hat,
        ]
        info = {
            "epsilon_true_hz": self.current_cfo_hz,
            "epsilon_hat_hz": epsilon_hat,
            "regime": float(self.current_regime),
            "residual_cfo_hz": residual_cfo,
            "residual_cfo_power_est": residual_cfo_sq_estimate,
            "data_power": data_power,
            "sync_mismatch": sync_mismatch,
            "sinr": sinr,
            "sinr_true": true_sinr,
            "pilot_snr": pilot_snr,
            "sensing_metric": sensing_metric,
            "empirical_sensing_metric": empirical_sensing_metric,
            "ici_power": observed_ici_power,
            "ici_power_true": true_ici_power,
            "channel_power_true": actual_channel_power,
            "channel_power_est": channel_power_estimate,
            "rate": rate,
            "rate_true": true_rate,
            "normalized_rate": normalized_rate,
            "normalized_sync_mismatch": normalized_sync,
            "normalized_sensing_metric": normalized_sensing,
            "rho": rho,
            "beta": beta,
            "num_pilots": float(num_pilots),
        }

        self.frame_index += 1
        done = self.frame_index >= self.config.episode_length
        self.previous_state = next_state
        self._advance_hidden_state()
        return list(next_state), reward, done, info

    def _initial_state(self) -> list[float]:
        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.config.default_pilot_ratio,
            self.config.default_pilot_power_fraction,
            0.0,  # delta_epsilon_hat
        ]

    def _build_delay_grid(self) -> list[float]:
        if self.config.sensing_delay_bins == 1:
            return [self.config.target_delay_seconds]
        step = (
            2.0 * self.config.max_delay_seconds / (self.config.sensing_delay_bins - 1)
        )
        return [
            -self.config.max_delay_seconds + index * step
            for index in range(self.config.sensing_delay_bins)
        ]

    def _sample_complex_gaussian(self) -> complex:
        scale = 1.0 / math.sqrt(2.0)
        return complex(self.rng.gauss(0.0, scale), self.rng.gauss(0.0, scale))

    def _sample_noise(self) -> complex:
        scale = math.sqrt(self.config.noise_variance / 2.0)
        return complex(self.rng.gauss(0.0, scale), self.rng.gauss(0.0, scale))

    def _sample_stationary_cfo(self) -> float:
        variance = (
            self.config.cfo_noise_std_hz ** 2
            / (1.0 - self.config.cfo_mean_reversion ** 2)
        )
        return self.rng.gauss(0.0, math.sqrt(variance))

    def _pilot_count(self, rho: float) -> int:
        return max(1, min(self.config.num_subcarriers - 1, int(round(rho * self.config.num_subcarriers))))

    def _pilot_indices(self, count: int) -> list[int]:
        step = self.config.num_subcarriers / count
        chosen = sorted(
            {
                min(self.config.num_subcarriers - 1, int(index * step))
                for index in range(count)
            }
        )
        candidate = 0
        while len(chosen) < count:
            if candidate not in chosen:
                chosen.append(candidate)
            candidate += 1
        chosen.sort()
        return chosen

    def _subcarrier_frequency(self, index: int) -> float:
        centered_index = index - (self.config.num_subcarriers - 1) / 2.0
        return centered_index * self.config.subcarrier_spacing_hz

    def _simulate_pilot_observations(
        self,
        pilot_indices: list[int],
        pilot_amplitude: float,
    ) -> list[list[complex]]:
        observations: list[list[complex]] = []
        for index in pilot_indices:
            pilot_trace: list[complex] = []
            subcarrier_phase = cmath.exp(
                complex(
                    0.0,
                    -2.0
                    * math.pi
                    * self._subcarrier_frequency(index)
                    * self.config.target_delay_seconds,
                )
            )
            channel_response = self.current_channel * subcarrier_phase
            for symbol in range(self.config.num_ofdm_symbols):
                time_phase = cmath.exp(
                    complex(
                        0.0,
                        2.0
                        * math.pi
                        * self.current_cfo_hz
                        * symbol
                        * self.config.symbol_duration,
                    )
                )
                received = channel_response * pilot_amplitude * time_phase + self._sample_noise()
                pilot_trace.append(received)
            observations.append(pilot_trace)
        return observations

    def _estimate_cfo(self, observations: list[list[complex]]) -> float:
        total_phase = 0.0
        count = 0
        for pilot_trace in observations:
            for symbol in range(self.config.num_ofdm_symbols - 1):
                product = pilot_trace[symbol + 1] * pilot_trace[symbol].conjugate()
                total_phase += cmath.phase(product)
                count += 1
        if count == 0:
            return 0.0
        mean_phase = total_phase / count
        return mean_phase / (2.0 * math.pi * self.config.symbol_duration)

    def _compute_sync_mismatch(
        self,
        observations: list[list[complex]],
        epsilon_hat: float,
    ) -> float:
        total = 0.0
        count = 0
        correction = cmath.exp(
            complex(0.0, 2.0 * math.pi * epsilon_hat * self.config.symbol_duration)
        )
        for pilot_trace in observations:
            for symbol in range(self.config.num_ofdm_symbols - 1):
                residual = pilot_trace[symbol + 1] - pilot_trace[symbol] * correction
                total += abs(residual) ** 2
                count += 1
        return total / max(count, 1)

    def _estimate_channel_power(
        self,
        observations: list[list[complex]],
        pilot_amplitude: float,
        epsilon_hat: float,
    ) -> float:
        total_power = 0.0
        count = 0
        for pilot_trace in observations:
            for symbol, received in enumerate(pilot_trace):
                compensation = cmath.exp(
                    complex(
                        0.0,
                        -2.0
                        * math.pi
                        * epsilon_hat
                        * symbol
                        * self.config.symbol_duration,
                    )
                )
                estimate = received * compensation / pilot_amplitude
                total_power += abs(estimate) ** 2
                count += 1
        return total_power / max(count, 1)

    def _estimate_residual_cfo_power(
        self,
        sync_mismatch: float,
        beta: float,
        num_pilots: int,
        channel_power_estimate: float,
    ) -> float:
        # Under the small-angle approximation, the pilot mismatch grows with
        # pilot power, channel power, and residual CFO squared.
        pilot_power_per_subcarrier = (
            beta * self.config.average_power / max(num_pilots, 1)
        )
        mismatch_above_noise = max(
            0.0,
            sync_mismatch - 2.0 * self.config.noise_variance,
        )
        denominator = (
            ((2.0 * math.pi * self.config.symbol_duration) ** 2)
            * max(pilot_power_per_subcarrier, 1.0e-12)
            * max(channel_power_estimate, 1.0e-12)
        )
        return mismatch_above_noise / max(denominator, 1.0e-12)

    def _compute_sensing_metric(
        self,
        rho: float,
        beta: float,
        empirical_metric: float,
    ) -> float:
        if self.config.sensing_metric_mode == "crlb":
            return self.config.sensing_crlb(rho=rho, beta=beta)
        return empirical_metric

    def _compute_empirical_sensing_metric(
        self,
        observations: list[list[complex]],
        pilot_indices: list[int],
        pilot_amplitude: float,
        epsilon_hat: float,
    ) -> float:
        channel_estimates: list[complex] = []
        for pilot_trace in observations:
            accumulator = 0j
            for symbol, received in enumerate(pilot_trace):
                compensation = cmath.exp(
                    complex(
                        0.0,
                        -2.0
                        * math.pi
                        * epsilon_hat
                        * symbol
                        * self.config.symbol_duration,
                    )
                )
                accumulator += received * compensation / pilot_amplitude
            channel_estimates.append(accumulator / self.config.num_ofdm_symbols)

        profile: list[float] = []
        for delay in self.delay_grid:
            correlation = 0j
            for index, estimate in zip(pilot_indices, channel_estimates):
                correlation += estimate * cmath.exp(
                    complex(0.0, 2.0 * math.pi * self._subcarrier_frequency(index) * delay)
                )
            profile.append(abs(correlation) ** 2)

        peak = max(profile)
        if peak <= 1.0e-12:
            return 1.0
        sidelobes = (sum(profile) - peak) / max(len(profile) - 1, 1)
        return sidelobes / peak

    def _normalize_rate(self, rate: float) -> float:
        if not self.config.reward_normalize_terms:
            return rate
        return rate / max(self.config.max_rate_reference, 1.0e-12)

    def _normalize_sync(self, sync_mismatch: float) -> float:
        if not self.config.reward_normalize_terms:
            return sync_mismatch
        return sync_mismatch / max(self.config.nominal_sync_reference, 1.0e-12)

    def _normalize_sensing(self, sensing_metric: float) -> float:
        if not self.config.reward_normalize_terms:
            return sensing_metric
        return sensing_metric / max(self.config.nominal_sensing_reference, 1.0e-24)

    def _advance_hidden_state(self) -> None:
        if self.config.cfo_regime_switch_prob > 0.0:
            if self.rng.random() < self.config.cfo_regime_switch_prob:
                self.current_regime = 1 - self.current_regime
        if self.current_regime == 1:
            alpha = self.config.cfo_high_drift_alpha
            std = self.config.cfo_high_drift_std_hz
        else:
            alpha = self.config.cfo_mean_reversion
            std = self.config.cfo_noise_std_hz
        self.current_cfo_hz = alpha * self.current_cfo_hz + self.rng.gauss(0.0, std)
        innovation = self._sample_complex_gaussian()
        self.current_channel = (
            self.config.channel_correlation * self.current_channel
            + math.sqrt(1.0 - self.config.channel_correlation ** 2) * innovation
        )
