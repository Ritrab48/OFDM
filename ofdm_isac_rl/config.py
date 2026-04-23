"""Configuration objects for the OFDM-ISAC reference model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class SystemConfig:
    """Physical-layer and environment parameters."""

    num_subcarriers: int = 40
    num_ofdm_symbols: int = 8
    average_power: float = 1.0
    noise_variance: float = 0.05
    symbol_duration: float = 66.666e-6
    subcarrier_spacing_hz: float = 15_000.0
    cfo_mean_reversion: float = 0.95
    cfo_noise_std_hz: float = 5.0
    channel_correlation: float = 0.98
    reward_rate_weight: float = 1.0
    reward_sync_weight: float = 1.0
    reward_sensing_weight: float = 0.5
    reward_normalize_terms: bool = True
    pilot_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    pilot_power_fractions: tuple[float, ...] = (0.2, 0.4, 0.6)
    default_pilot_ratio: float = 0.2
    default_pilot_power_fraction: float = 0.4
    sensing_metric_mode: str = "empirical"
    signal_bandwidth_hz: float | None = None
    rate_reference: float | None = None
    sync_reference: float | None = None
    sensing_reference: float | None = None
    max_delay_seconds: float = 3.0e-6
    target_delay_seconds: float = 1.2e-6
    sensing_delay_bins: int = 41
    episode_length: int = 64
    seed: int = 7
    # Regime-switching CFO parameters (Phase 2a).
    # Set cfo_regime_switch_prob > 0 to enable a hidden two-state Markov chain.
    # Regime 0 (low-drift) uses cfo_mean_reversion / cfo_noise_std_hz.
    # Regime 1 (high-drift) uses cfo_high_drift_alpha / cfo_high_drift_std_hz.
    cfo_regime_switch_prob: float = 0.0
    cfo_high_drift_alpha: float = 0.7
    cfo_high_drift_std_hz: float = 20.0

    def __post_init__(self) -> None:
        if self.num_subcarriers < 4:
            raise ValueError("num_subcarriers must be at least 4")
        if self.num_ofdm_symbols < 2:
            raise ValueError("num_ofdm_symbols must be at least 2")
        if not 0.0 < self.cfo_mean_reversion < 1.0:
            raise ValueError("cfo_mean_reversion must lie in (0, 1)")
        if not 0.0 < self.channel_correlation < 1.0:
            raise ValueError("channel_correlation must lie in (0, 1)")
        if self.sensing_delay_bins < 3:
            raise ValueError("sensing_delay_bins must be at least 3")
        if self.sensing_metric_mode not in {"crlb", "empirical"}:
            raise ValueError("sensing_metric_mode must be 'crlb' or 'empirical'")
        if not 0.0 <= self.cfo_regime_switch_prob < 1.0:
            raise ValueError("cfo_regime_switch_prob must lie in [0, 1)")
        if not 0.0 < self.cfo_high_drift_alpha < 1.0:
            raise ValueError("cfo_high_drift_alpha must lie in (0, 1)")

    @property
    def action_space(self) -> list[tuple[float, float]]:
        return [
            (rho, beta)
            for rho in self.pilot_ratios
            for beta in self.pilot_power_fractions
        ]

    @property
    def ici_constant(self) -> float:
        return (math.pi ** 2) / (3.0 * (self.subcarrier_spacing_hz ** 2))

    @property
    def effective_bandwidth_hz(self) -> float:
        if self.signal_bandwidth_hz is not None:
            return self.signal_bandwidth_hz
        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def max_rate_reference(self) -> float:
        if self.rate_reference is not None:
            return self.rate_reference
        rho = min(self.pilot_ratios)
        beta = min(self.pilot_power_fractions)
        nominal_sinr = ((1.0 - beta) * self.average_power) / self.noise_variance
        return (1.0 - rho) * math.log2(1.0 + nominal_sinr)

    @property
    def nominal_sync_reference(self) -> float:
        if self.sync_reference is not None:
            return self.sync_reference
        return 2.0 * self.noise_variance

    @property
    def nominal_sensing_reference(self) -> float:
        if self.sensing_reference is not None:
            return self.sensing_reference
        if self.sensing_metric_mode == "empirical":
            return 1.0
        return self.sensing_crlb(
            rho=self.default_pilot_ratio,
            beta=self.default_pilot_power_fraction,
        )

    def sensing_crlb(self, rho: float, beta: float) -> float:
        pilot_snr = beta * self.average_power / self.noise_variance
        pilot_resource = rho * self.num_subcarriers * self.num_ofdm_symbols
        denominator = (
            8.0
            * (math.pi ** 2)
            * (self.effective_bandwidth_hz ** 2)
            * pilot_snr
            * pilot_resource
        )
        return 1.0 / denominator


@dataclass(slots=True)
class TrainingConfig:
    """DQN hyperparameters and runtime options."""

    episodes: int = 80
    evaluation_episodes: int = 5
    replay_capacity: int = 4000
    batch_size: int = 32
    hidden_sizes: tuple[int, ...] = (48, 48)
    learning_rate: float = 1.0e-3
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    warmup_steps: int = 128
    train_interval: int = 1
    target_update_interval: int = 100
    evaluation_interval: int = 10
    output_dir: str = "artifacts/reference_model"
    save_history: bool = True
    seed: int = 11
    state_clip: float = 10.0
    channel_power_scale: float = 2.0
    log_feature_scale: float = 5.0
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
