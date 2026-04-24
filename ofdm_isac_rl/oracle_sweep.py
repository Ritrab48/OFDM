"""Sweep environment and reward settings with the regime oracle diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .oracle_diagnostics import (
    advantage_signal_to_noise,
    build_rows,
    metric_range,
    parse_int_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank environment/reward combinations by oracle advantage."
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="7,17,27")
    parser.add_argument(
        "--weights",
        type=str,
        default="1.0:0.2:0.4,1.0:0.25:0.5,1.0:0.4:0.5,1.0:0.5:0.4",
        help="Comma-separated lambda triples: rate:sync:sensing.",
    )
    parser.add_argument("--regime-switch-prob", type=str, default="0.02,0.05")
    parser.add_argument("--noise-variance", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--cfo-low-drift-std", type=float, default=5.0)
    parser.add_argument("--cfo-low-drift-alpha", type=float, default=0.95)
    parser.add_argument("--cfo-high-drift-std", type=str, default="100,300,600,1000")
    parser.add_argument("--cfo-high-drift-alpha", type=str, default="0.7,0.9")
    parser.add_argument("--channel-correlation", type=str, default="0.98")
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/phase2b_oracle_sweep",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_weights(raw: str) -> list[tuple[float, float, float]]:
    weights: list[tuple[float, float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [float(part.strip()) for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid weight triple: {item!r}")
        weights.append((parts[0], parts[1], parts[2]))
    return weights


def make_job_args(
    base_args: argparse.Namespace,
    weight: tuple[float, float, float],
    switch_prob: float,
    noise_variance: float,
    high_std: float,
    high_alpha: float,
    channel_correlation: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        frames=base_args.frames,
        eval_episodes=base_args.eval_episodes,
        seeds=base_args.seeds,
        lambda_rate=weight[0],
        lambda_sync=weight[1],
        lambda_sensing=weight[2],
        reward_normalize=base_args.reward_normalize,
        sensing_mode=base_args.sensing_mode,
        noise_variance=noise_variance,
        cfo_low_drift_std=base_args.cfo_low_drift_std,
        cfo_low_drift_alpha=base_args.cfo_low_drift_alpha,
        cfo_high_drift_std=high_std,
        cfo_high_drift_alpha=high_alpha,
        regime_switch_prob=switch_prob,
        channel_correlation=channel_correlation,
    )


def summarize_combo(job_args: argparse.Namespace) -> dict[str, object]:
    seeds = parse_int_list(job_args.seeds)
    rows = build_rows(job_args, seeds)
    rows.sort(key=lambda row: float(row["reward_mean"]), reverse=True)
    fixed_rows = [row for row in rows if bool(row["is_fixed"])]
    best_fixed = max(fixed_rows, key=lambda row: float(row["reward_mean"]))
    best_oracle = rows[0]
    oracle_advantage = float(best_oracle["reward_mean"]) - float(best_fixed["reward_mean"])
    oracle_snr = advantage_signal_to_noise(
        best_oracle=best_oracle,
        best_fixed=best_fixed,
        seed_count=len(seeds),
    )
    fixed_ranges = {
        "reward_mean": metric_range(fixed_rows, "reward_mean"),
        "rate_mean": metric_range(fixed_rows, "rate_mean"),
        "sync_mean": metric_range(fixed_rows, "sync_mean"),
        "sensing_mean": metric_range(fixed_rows, "sensing_mean"),
    }
    return {
        "lambda_rate": job_args.lambda_rate,
        "lambda_sync": job_args.lambda_sync,
        "lambda_sensing": job_args.lambda_sensing,
        "regime_switch_prob": job_args.regime_switch_prob,
        "noise_variance": job_args.noise_variance,
        "cfo_high_drift_std": job_args.cfo_high_drift_std,
        "cfo_high_drift_alpha": job_args.cfo_high_drift_alpha,
        "channel_correlation": job_args.channel_correlation,
        "oracle_advantage": oracle_advantage,
        "oracle_advantage_snr": oracle_snr,
        "oracle_uses_regime_difference": (
            int(best_oracle["low_action"]) != int(best_oracle["high_action"])
        ),
        "best_fixed_policy": best_fixed["policy"],
        "best_fixed_reward": best_fixed["reward_mean"],
        "best_fixed_rate": best_fixed["rate_mean"],
        "best_fixed_sync": best_fixed["sync_mean"],
        "best_fixed_sensing": best_fixed["sensing_mean"],
        "best_oracle_policy": best_oracle["policy"],
        "best_oracle_reward": best_oracle["reward_mean"],
        "best_oracle_rate": best_oracle["rate_mean"],
        "best_oracle_sync": best_oracle["sync_mean"],
        "best_oracle_sensing": best_oracle["sensing_mean"],
        "fixed_reward_span": fixed_ranges["reward_mean"]["span"],
        "fixed_rate_span": fixed_ranges["rate_mean"]["span"],
        "fixed_sync_span": fixed_ranges["sync_mean"]["span"],
        "fixed_sensing_span": fixed_ranges["sensing_mean"]["span"],
        "seeds": seeds,
        "eval_episodes": job_args.eval_episodes,
        "frames": job_args.frames,
    }


def build_jobs(args: argparse.Namespace) -> list[argparse.Namespace]:
    jobs: list[argparse.Namespace] = []
    for weight in parse_weights(args.weights):
        for switch_prob in parse_float_list(args.regime_switch_prob):
            for noise_variance in parse_float_list(args.noise_variance):
                for high_std in parse_float_list(args.cfo_high_drift_std):
                    for high_alpha in parse_float_list(args.cfo_high_drift_alpha):
                        for channel_correlation in parse_float_list(args.channel_correlation):
                            jobs.append(
                                make_job_args(
                                    base_args=args,
                                    weight=weight,
                                    switch_prob=switch_prob,
                                    noise_variance=noise_variance,
                                    high_std=high_std,
                                    high_alpha=high_alpha,
                                    channel_correlation=channel_correlation,
                                )
                            )
    return jobs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "lambda_rate",
        "lambda_sync",
        "lambda_sensing",
        "regime_switch_prob",
        "noise_variance",
        "cfo_high_drift_std",
        "cfo_high_drift_alpha",
        "channel_correlation",
        "oracle_advantage",
        "oracle_advantage_snr",
        "oracle_uses_regime_difference",
        "best_fixed_policy",
        "best_fixed_reward",
        "best_oracle_policy",
        "best_oracle_reward",
        "best_oracle_rate",
        "best_oracle_sync",
        "best_oracle_sensing",
        "fixed_reward_span",
        "fixed_rate_span",
        "fixed_sync_span",
        "fixed_sensing_span",
        "seeds",
        "eval_episodes",
        "frames",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row[header] for header in headers})


def write_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Oracle Environment Sweep",
        "",
        "This sweep ranks environment and reward combinations by hidden-regime oracle advantage.",
        "It is a screening step before any DQN training.",
        "",
        "## Screening Budget",
        "",
        f"- Frames per episode: `{args.frames}`",
        f"- Evaluation episodes per seed: `{args.eval_episodes}`",
        f"- Seeds: `{args.seeds}`",
        f"- Number of combinations: `{len(rows)}`",
        "",
        "## Top Combinations",
        "",
        "| Rank | Weights | p_switch | noise | high std | high alpha | Oracle Adv | Adv SNR | Oracle differs? | Sync span | Best oracle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for index, row in enumerate(rows[:20], start=1):
        lines.append(
            "| {rank} | `({lr:.2g},{ls:.2g},{lg:.2g})` | {ps:.3g} | {noise:.3g} | "
            "{std:.3g} | {alpha:.3g} | {adv:.4f} | {snr:.3f} | `{diff}` | "
            "{sync_span:.6f} | `{policy}` |".format(
                rank=index,
                lr=float(row["lambda_rate"]),
                ls=float(row["lambda_sync"]),
                lg=float(row["lambda_sensing"]),
                ps=float(row["regime_switch_prob"]),
                noise=float(row["noise_variance"]),
                std=float(row["cfo_high_drift_std"]),
                alpha=float(row["cfo_high_drift_alpha"]),
                adv=float(row["oracle_advantage"]),
                snr=float(row["oracle_advantage_snr"]),
                diff=row["oracle_uses_regime_difference"],
                sync_span=float(row["fixed_sync_span"]),
                policy=row["best_oracle_policy"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    print(f"oracle_sweep jobs={len(jobs)} workers={args.workers}")

    rows: list[dict[str, object]] = []
    if args.workers <= 1:
        for index, job in enumerate(jobs, start=1):
            row = summarize_combo(job)
            rows.append(row)
            print(
                "job {index}/{total} adv={adv:.4f} snr={snr:.3f} diff={diff}".format(
                    index=index,
                    total=len(jobs),
                    adv=float(row["oracle_advantage"]),
                    snr=float(row["oracle_advantage_snr"]),
                    diff=row["oracle_uses_regime_difference"],
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(summarize_combo, job): index
                for index, job in enumerate(jobs, start=1)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                row = future.result()
                rows.append(row)
                print(
                    "job {index}/{total} adv={adv:.4f} snr={snr:.3f} diff={diff}".format(
                        index=index,
                        total=len(jobs),
                        adv=float(row["oracle_advantage"]),
                        snr=float(row["oracle_advantage_snr"]),
                        diff=row["oracle_uses_regime_difference"],
                    )
                )

    rows.sort(
        key=lambda row: (
            float(row["oracle_advantage"]),
            float(row["oracle_advantage_snr"]),
            bool(row["oracle_uses_regime_difference"]),
        ),
        reverse=True,
    )
    write_csv(output_dir / "oracle_sweep_results.csv", rows)
    with (output_dir / "oracle_sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_report(output_dir / "oracle_sweep_report.md", rows, args)
    if rows:
        best = rows[0]
        print(
            "best adv={adv:.4f} snr={snr:.3f} weights=({lr},{ls},{lg}) "
            "p={p} noise={noise} high_std={std} high_alpha={alpha}".format(
                adv=float(best["oracle_advantage"]),
                snr=float(best["oracle_advantage_snr"]),
                lr=best["lambda_rate"],
                ls=best["lambda_sync"],
                lg=best["lambda_sensing"],
                p=best["regime_switch_prob"],
                noise=best["noise_variance"],
                std=best["cfo_high_drift_std"],
                alpha=best["cfo_high_drift_alpha"],
            )
        )


if __name__ == "__main__":
    main()
