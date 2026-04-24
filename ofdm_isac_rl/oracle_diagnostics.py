"""Oracle diagnostics for regime-switching experiments.

This module does not train a DQN. It answers a narrower question:
does a hidden-regime-aware policy have enough advantage over the best
constant fixed action to justify expecting an adaptive learner to win?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .config import SystemConfig
from .environment import OfdmIsacEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best fixed vs hidden-regime oracle action maps."
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--lambda-rate", type=float, default=1.0)
    parser.add_argument("--lambda-sync", type=float, default=0.2)
    parser.add_argument("--lambda-sensing", type=float, default=0.4)
    parser.add_argument(
        "--reward-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sensing-mode",
        choices=("empirical", "crlb"),
        default="empirical",
    )
    parser.add_argument("--noise-variance", type=float, default=0.05)
    parser.add_argument("--cfo-low-drift-std", type=float, default=5.0)
    parser.add_argument("--cfo-low-drift-alpha", type=float, default=0.95)
    parser.add_argument("--cfo-high-drift-std", type=float, default=20.0)
    parser.add_argument("--cfo-high-drift-alpha", type=float, default=0.7)
    parser.add_argument("--regime-switch-prob", type=float, default=0.05)
    parser.add_argument("--channel-correlation", type=float, default=0.98)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/phase2b_oracle_diagnostics",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, statistics.stdev(values)


def format_action(config: SystemConfig, action_index: int) -> str:
    rho, beta = config.action_space[action_index]
    return f"rho={rho:.1f}_beta={beta:.1f}"


def build_config(args: argparse.Namespace, seed: int) -> SystemConfig:
    return SystemConfig(
        episode_length=args.frames,
        seed=seed,
        reward_rate_weight=args.lambda_rate,
        reward_sync_weight=args.lambda_sync,
        reward_sensing_weight=args.lambda_sensing,
        reward_normalize_terms=args.reward_normalize,
        sensing_metric_mode=args.sensing_mode,
        noise_variance=args.noise_variance,
        cfo_noise_std_hz=args.cfo_low_drift_std,
        cfo_mean_reversion=args.cfo_low_drift_alpha,
        cfo_high_drift_std_hz=args.cfo_high_drift_std,
        cfo_high_drift_alpha=args.cfo_high_drift_alpha,
        cfo_regime_switch_prob=args.regime_switch_prob,
        channel_correlation=args.channel_correlation,
    )


def evaluate_mapping_once(
    config: SystemConfig,
    regime_to_action: tuple[int, int],
    episodes: int,
) -> dict[str, object]:
    env = OfdmIsacEnv(config)
    total_reward = 0.0
    total_rate = 0.0
    total_sync = 0.0
    total_sensing = 0.0
    action_counts = [0 for _ in config.action_space]
    regime_counts = [0, 0]

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            del state
            regime = int(env.current_regime)
            action = regime_to_action[regime]
            regime_counts[regime] += 1
            action_counts[action] += 1
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_rate += float(info["rate"])
            total_sync += float(info["sync_mismatch"])
            total_sensing += float(info["sensing_metric"])

    normalizer = episodes * config.episode_length
    return {
        "avg_reward": total_reward / episodes,
        "avg_rate": total_rate / normalizer,
        "avg_sync": total_sync / normalizer,
        "avg_sensing": total_sensing / normalizer,
        "action_counts": action_counts,
        "regime_counts": regime_counts,
        "high_regime_fraction": regime_counts[1] / max(sum(regime_counts), 1),
    }


def evaluate_mapping_across_seeds(
    args: argparse.Namespace,
    regime_to_action: tuple[int, int],
    seeds: list[int],
) -> dict[str, object]:
    per_seed = [
        evaluate_mapping_once(
            config=build_config(args, seed=seed),
            regime_to_action=regime_to_action,
            episodes=args.eval_episodes,
        )
        for seed in seeds
    ]

    rewards = [float(row["avg_reward"]) for row in per_seed]
    rates = [float(row["avg_rate"]) for row in per_seed]
    syncs = [float(row["avg_sync"]) for row in per_seed]
    sensings = [float(row["avg_sensing"]) for row in per_seed]
    high_fractions = [float(row["high_regime_fraction"]) for row in per_seed]
    reward_mean, reward_std = mean_std(rewards)
    rate_mean, rate_std = mean_std(rates)
    sync_mean, sync_std = mean_std(syncs)
    sensing_mean, sensing_std = mean_std(sensings)

    action_counts = [
        sum(int(row["action_counts"][index]) for row in per_seed)
        for index in range(len(build_config(args, seeds[0]).action_space))
    ]
    regime_counts = [
        sum(int(row["regime_counts"][index]) for row in per_seed)
        for index in range(2)
    ]

    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "rate_mean": rate_mean,
        "rate_std": rate_std,
        "sync_mean": sync_mean,
        "sync_std": sync_std,
        "sensing_mean": sensing_mean,
        "sensing_std": sensing_std,
        "high_regime_fraction_mean": statistics.mean(high_fractions),
        "action_counts": action_counts,
        "regime_counts": regime_counts,
    }


def build_rows(args: argparse.Namespace, seeds: list[int]) -> list[dict[str, object]]:
    config = build_config(args, seed=seeds[0])
    rows: list[dict[str, object]] = []
    for low_action in range(len(config.action_space)):
        for high_action in range(len(config.action_space)):
            metrics = evaluate_mapping_across_seeds(
                args=args,
                regime_to_action=(low_action, high_action),
                seeds=seeds,
            )
            rows.append(
                {
                    "policy": (
                        f"map_low={format_action(config, low_action)}__"
                        f"high={format_action(config, high_action)}"
                    ),
                    "low_action": low_action,
                    "high_action": high_action,
                    "low_action_label": format_action(config, low_action),
                    "high_action_label": format_action(config, high_action),
                    "is_fixed": low_action == high_action,
                    **metrics,
                }
            )
    return rows


def metric_range(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {"min": min(values), "max": max(values), "span": max(values) - min(values)}


def advantage_signal_to_noise(
    best_oracle: dict[str, object],
    best_fixed: dict[str, object],
    seed_count: int,
) -> float:
    """Rough Welch-style signal-to-noise ratio for the oracle advantage."""
    oracle_var = float(best_oracle["reward_std"]) ** 2
    fixed_var = float(best_fixed["reward_std"]) ** 2
    standard_error = math.sqrt((oracle_var + fixed_var) / max(seed_count, 1))
    if standard_error <= 1.0e-12:
        return float("inf")
    return (
        float(best_oracle["reward_mean"]) - float(best_fixed["reward_mean"])
    ) / standard_error


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "policy",
        "low_action",
        "high_action",
        "low_action_label",
        "high_action_label",
        "is_fixed",
        "reward_mean",
        "reward_std",
        "rate_mean",
        "rate_std",
        "sync_mean",
        "sync_std",
        "sensing_mean",
        "sensing_std",
        "high_regime_fraction_mean",
        "regime_counts",
        "action_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row[header] for header in headers})


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    top_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Oracle Diagnostic Report",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether hidden-regime adaptation has meaningful upside before training a DQN.",
        "The oracle enumerates all two-regime action maps `(low_regime_action, high_regime_action)`.",
        "Every fixed policy is included as the special case where both regime actions are identical.",
        "",
        "## Setup",
        "",
        f"- Frames per episode: `{args.frames}`",
        f"- Evaluation episodes per seed: `{args.eval_episodes}`",
        f"- Seeds: `{args.seeds}`",
        f"- Reward weights: `({args.lambda_rate}, {args.lambda_sync}, {args.lambda_sensing})`",
        f"- Noise variance: `{args.noise_variance}`",
        f"- Low-regime CFO: `alpha={args.cfo_low_drift_alpha}`, `std={args.cfo_low_drift_std} Hz`",
        f"- High-regime CFO: `alpha={args.cfo_high_drift_alpha}`, `std={args.cfo_high_drift_std} Hz`",
        f"- Regime switch probability: `{args.regime_switch_prob}`",
        f"- Channel correlation: `{args.channel_correlation}`",
        "",
        "## Summary",
        "",
        f"- Best fixed policy: `{summary['best_fixed']['policy']}`",
        f"- Best fixed reward: `{summary['best_fixed']['reward_mean']:.4f}`",
        f"- Best regime oracle policy: `{summary['best_oracle']['policy']}`",
        f"- Best regime oracle reward: `{summary['best_oracle']['reward_mean']:.4f}`",
        f"- Oracle advantage over best fixed: `{summary['oracle_advantage']:.4f}`",
        f"- Oracle advantage signal-to-noise: `{summary['oracle_advantage_snr']:.3f}`",
        f"- Oracle uses different actions by regime: `{summary['oracle_uses_regime_difference']}`",
        "",
        "## Component Ranges Across Fixed Policies",
        "",
        f"- Reward span: `{summary['fixed_ranges']['reward_mean']['span']:.4f}`",
        f"- Rate span: `{summary['fixed_ranges']['rate_mean']['span']:.4f}`",
        f"- Sync span: `{summary['fixed_ranges']['sync_mean']['span']:.6f}`",
        f"- Sensing span: `{summary['fixed_ranges']['sensing_mean']['span']:.6f}`",
        "",
        "## Interpretation Rule",
        "",
        "- If oracle advantage is near zero, the environment/reward gives adaptation little value.",
        "- If oracle advantage signal-to-noise is below roughly 2, the measured advantage is weak relative to seed variability.",
        "- If oracle advantage is large but DQN loses, investigate state observability, learning stability, or training budget.",
        "- If sync and sensing spans are tiny compared with reward/rate span, the reward is still effectively rate-dominated.",
        "",
        "## Top Regime Maps",
        "",
        "| Rank | Low action | High action | Reward | Rate | Sync | Sensing | Fixed? |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(top_rows, start=1):
        lines.append(
            "| {rank} | `{low}` | `{high}` | {reward:.4f} | {rate:.4f} | "
            "{sync:.6f} | {sensing:.6f} | `{fixed}` |".format(
                rank=index,
                low=row["low_action_label"],
                high=row["high_action_label"],
                reward=float(row["reward_mean"]),
                rate=float(row["rate_mean"]),
                sync=float(row["sync_mean"]),
                sensing=float(row["sensing_mean"]),
                fixed=row["is_fixed"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args, seeds)
    rows.sort(key=lambda row: float(row["reward_mean"]), reverse=True)
    fixed_rows = [row for row in rows if bool(row["is_fixed"])]
    best_fixed = max(fixed_rows, key=lambda row: float(row["reward_mean"]))
    best_oracle = rows[0]
    summary: dict[str, object] = {
        "best_fixed": best_fixed,
        "best_oracle": best_oracle,
        "oracle_advantage": (
            float(best_oracle["reward_mean"]) - float(best_fixed["reward_mean"])
        ),
        "oracle_advantage_snr": advantage_signal_to_noise(
            best_oracle=best_oracle,
            best_fixed=best_fixed,
            seed_count=len(seeds),
        ),
        "oracle_uses_regime_difference": (
            int(best_oracle["low_action"]) != int(best_oracle["high_action"])
        ),
        "fixed_ranges": {
            "reward_mean": metric_range(fixed_rows, "reward_mean"),
            "rate_mean": metric_range(fixed_rows, "rate_mean"),
            "sync_mean": metric_range(fixed_rows, "sync_mean"),
            "sensing_mean": metric_range(fixed_rows, "sensing_mean"),
        },
        "all_map_ranges": {
            "reward_mean": metric_range(rows, "reward_mean"),
            "rate_mean": metric_range(rows, "rate_mean"),
            "sync_mean": metric_range(rows, "sync_mean"),
            "sensing_mean": metric_range(rows, "sensing_mean"),
        },
        "seeds": seeds,
        "num_maps": len(rows),
    }

    write_csv(output_dir / "oracle_mapping_results.csv", rows)
    with (output_dir / "oracle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_markdown(
        output_dir / "oracle_report.md",
        args=args,
        summary=summary,
        top_rows=rows[:10],
    )

    print(
        "oracle_advantage={advantage:.4f} best_fixed={fixed:.4f} "
        "best_oracle={oracle:.4f} oracle_policy={policy}".format(
            advantage=float(summary["oracle_advantage"]),
            fixed=float(best_fixed["reward_mean"]),
            oracle=float(best_oracle["reward_mean"]),
            policy=best_oracle["policy"],
        )
    )


if __name__ == "__main__":
    main()
