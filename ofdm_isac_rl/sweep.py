"""Reward-weight sweep and paper-pack generation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .baselines import build_baseline_policies
from .config import SystemConfig, TrainingConfig
from .dqn import DQNAgent
from .environment import OfdmIsacEnv
from .reporting import write_bar_chart_svg, write_line_chart_svg
from .train import evaluate_selector, make_evaluation_config, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep reward weights and generate paper-ready outputs."
    )
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="7,17")
    parser.add_argument("--lambda-rate", type=float, default=1.0)
    parser.add_argument("--lambda-sync", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--lambda-sensing", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--output-dir", type=str, default="artifacts/weight_sweep")
    parser.add_argument(
        "--sensing-mode",
        choices=("empirical", "crlb"),
        default="empirical",
    )
    parser.add_argument(
        "--reward-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, statistics.stdev(values)


def train_single_run(
    system_config: SystemConfig,
    training_config: TrainingConfig,
) -> dict[str, object]:
    env = OfdmIsacEnv(system_config)
    evaluation_config = make_evaluation_config(system_config, seed=system_config.seed + 997)
    agent = DQNAgent(system_config, training_config)
    history: list[dict[str, float]] = []
    for _ in range(training_config.episodes):
        history.append(run_episode(env, agent, training=True))

    policy_results = [
        evaluate_selector(
            OfdmIsacEnv(evaluation_config),
            agent.act_greedily,
            training_config.evaluation_episodes,
            name="dqn",
        )
    ]
    for policy in build_baseline_policies(system_config, seed=system_config.seed + 123):
        policy_results.append(
            evaluate_selector(
                OfdmIsacEnv(evaluation_config),
                policy.select_action,
                training_config.evaluation_episodes,
                name=policy.name,
            )
        )
    return {
        "history": history,
        "policy_results": policy_results,
        "model": agent.export(),
    }


def aggregate_policy_results(policy_results_per_run: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, list[float] | list[list[int]]]] = {}
    for run_results in policy_results_per_run:
        for result in run_results:
            name = str(result["policy"])
            grouped.setdefault(
                name,
                {
                    "avg_reward": [],
                    "avg_rate": [],
                    "avg_sync_mismatch": [],
                    "avg_sensing_metric": [],
                    "action_counts": [],
                },
            )
            grouped[name]["avg_reward"].append(float(result["avg_reward"]))
            grouped[name]["avg_rate"].append(float(result["avg_rate"]))
            grouped[name]["avg_sync_mismatch"].append(float(result["avg_sync_mismatch"]))
            grouped[name]["avg_sensing_metric"].append(float(result["avg_sensing_metric"]))
            grouped[name]["action_counts"].append([int(count) for count in result["action_counts"]])

    aggregated: list[dict[str, object]] = []
    for name, metrics in grouped.items():
        aggregated_counts = [
            sum(run_counts[index] for run_counts in metrics["action_counts"])
            for index in range(len(metrics["action_counts"][0]))
        ]
        reward_mean, reward_std = mean_std(metrics["avg_reward"])
        rate_mean, rate_std = mean_std(metrics["avg_rate"])
        sync_mean, sync_std = mean_std(metrics["avg_sync_mismatch"])
        sensing_mean, sensing_std = mean_std(metrics["avg_sensing_metric"])
        aggregated.append(
            {
                "policy": name,
                "avg_reward_mean": reward_mean,
                "avg_reward_std": reward_std,
                "avg_rate_mean": rate_mean,
                "avg_rate_std": rate_std,
                "avg_sync_mean": sync_mean,
                "avg_sync_std": sync_std,
                "avg_sensing_mean": sensing_mean,
                "avg_sensing_std": sensing_std,
                "action_counts": aggregated_counts,
            }
        )
    return sorted(aggregated, key=lambda item: float(item["avg_reward_mean"]), reverse=True)


def aggregate_history(histories: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    aggregated: list[dict[str, float]] = []
    for episode_index in range(len(histories[0])):
        rewards = [history[episode_index]["reward"] for history in histories]
        rates = [history[episode_index]["avg_rate"] for history in histories]
        syncs = [history[episode_index]["avg_sync_mismatch"] for history in histories]
        sensings = [history[episode_index]["avg_sensing_metric"] for history in histories]
        losses = [history[episode_index]["avg_loss"] for history in histories]
        aggregated.append(
            {
                "episode": episode_index + 1,
                "reward_mean": statistics.mean(rewards),
                "rate_mean": statistics.mean(rates),
                "sync_mean": statistics.mean(syncs),
                "sensing_mean": statistics.mean(sensings),
                "loss_mean": statistics.mean(losses),
            }
        )
    return aggregated


def best_fixed_reward(policy_results: list[dict[str, object]]) -> float:
    fixed_only = [
        float(result["avg_reward"])
        for result in policy_results
        if str(result["policy"]).startswith("fixed_")
    ]
    return max(fixed_only)


def write_csv_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def build_paper_pack(
    output_dir: Path,
    system_config: SystemConfig,
    training_config: TrainingConfig,
    sweep_rows: list[dict[str, object]],
    best_entry: dict[str, object],
    best_history: list[dict[str, float]],
    best_policy_summary: list[dict[str, object]],
) -> None:
    lines = [
        "# Paper Writing Pack",
        "",
        "## Experimental Setup",
        "",
        f"- Sensing mode: `{system_config.sensing_metric_mode}`",
        f"- Reward normalization: `{system_config.reward_normalize_terms}`",
        f"- Episodes per run: `{training_config.episodes}`",
        f"- Frames per episode: `{system_config.episode_length}`",
        f"- Evaluation episodes per run: `{training_config.evaluation_episodes}`",
        f"- Candidate `lambda_sync` values: `{best_entry['lambda_sync_grid']}`",
        f"- Candidate `lambda_sensing` values: `{best_entry['lambda_sensing_grid']}`",
        "",
        "## Best Weight Setting",
        "",
        f"- `lambda_rate = {best_entry['lambda_rate']}`",
        f"- `lambda_sync = {best_entry['lambda_sync']}`",
        f"- `lambda_sensing = {best_entry['lambda_sensing']}`",
        f"- Mean DQN reward across seeds: `{best_entry['mean_dqn_reward']:.4f}`",
        f"- Reward gap to best fixed baseline: `{best_entry['mean_gap_to_best_fixed']:.4f}`",
        "",
        "## Interpretation Notes",
        "",
        "- Weight selection should be justified by final performance against baselines, not only by how fast the reward curve flattens.",
        "- Empirical `S_t` is the paper-faithful default because it preserves the observable-only philosophy.",
        "- If the DQN does not beat the best fixed baseline under the current sweep, that should be stated honestly as a proof-of-concept limitation.",
        "",
        "## Main Tables",
        "",
        "- `sweep_results.csv`: all weight combinations ranked by mean DQN reward",
        "- `best_policy_summary.csv`: DQN and all baselines for the selected weight setting",
        "- `best_history.json`: mean training history across seeds for the selected weight setting",
        "",
        "## Main Figures",
        "",
        "- `weight_sweep_reward.svg`",
        "- `best_reward_curve.svg`",
        "- `best_policy_reward.svg`",
        "- `best_policy_rate.svg`",
        "- `best_policy_sync.svg`",
        "- `best_policy_sensing.svg`",
        "- `best_action_distribution.svg`",
        "",
        "## Remaining Work",
        "",
        "- Revisit the state SINR term if strict observability is required; the current simulator still uses an analytically computed SINR.",
        "- Run a larger multi-seed sweep once faster numerical tooling is available or more runtime is acceptable.",
        "- Decide whether CRLB sensing should appear only as an ablation or also as a historical comparison.",
        "- If the best fixed baseline still beats DQN, improve training length, state design, or reward tuning before claiming adaptive superiority.",
        "",
        "## Links to Audit Notes",
        "",
        "- See `IMPLEMENTATION_DIVERGENCE_LOG.md` for a full paper-vs-implementation audit.",
        "",
    ]
    (output_dir / "paper_pack.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    lambda_sync_values = parse_float_list(args.lambda_sync)
    lambda_sensing_values = parse_float_list(args.lambda_sensing)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict[str, object]] = []
    best_runs_payload: dict[str, object] | None = None
    best_score: float | None = None

    for lambda_sync in lambda_sync_values:
        for lambda_sensing in lambda_sensing_values:
            run_payloads: list[dict[str, object]] = []
            for seed in seeds:
                system_config = SystemConfig(
                    episode_length=args.frames,
                    seed=seed,
                    reward_rate_weight=args.lambda_rate,
                    reward_sync_weight=lambda_sync,
                    reward_sensing_weight=lambda_sensing,
                    reward_normalize_terms=args.reward_normalize,
                    sensing_metric_mode=args.sensing_mode,
                )
                training_config = TrainingConfig(
                    episodes=args.episodes,
                    evaluation_episodes=args.eval_episodes,
                    output_dir=str(output_dir),
                    seed=seed + 1,
                )
                run_payloads.append(train_single_run(system_config, training_config))

            dqn_rewards = [
                float(next(result for result in payload["policy_results"] if result["policy"] == "dqn")["avg_reward"])
                for payload in run_payloads
            ]
            dqn_rates = [
                float(next(result for result in payload["policy_results"] if result["policy"] == "dqn")["avg_rate"])
                for payload in run_payloads
            ]
            dqn_syncs = [
                float(next(result for result in payload["policy_results"] if result["policy"] == "dqn")["avg_sync_mismatch"])
                for payload in run_payloads
            ]
            dqn_sensings = [
                float(next(result for result in payload["policy_results"] if result["policy"] == "dqn")["avg_sensing_metric"])
                for payload in run_payloads
            ]
            gaps = [
                float(next(result for result in payload["policy_results"] if result["policy"] == "dqn")["avg_reward"])
                - best_fixed_reward(payload["policy_results"])
                for payload in run_payloads
            ]
            reward_mean, reward_std = mean_std(dqn_rewards)
            rate_mean, rate_std = mean_std(dqn_rates)
            sync_mean, sync_std = mean_std(dqn_syncs)
            sensing_mean, sensing_std = mean_std(dqn_sensings)
            gap_mean, gap_std = mean_std(gaps)

            row = {
                "lambda_rate": args.lambda_rate,
                "lambda_sync": lambda_sync,
                "lambda_sensing": lambda_sensing,
                "mean_dqn_reward": reward_mean,
                "std_dqn_reward": reward_std,
                "mean_dqn_rate": rate_mean,
                "std_dqn_rate": rate_std,
                "mean_dqn_sync": sync_mean,
                "std_dqn_sync": sync_std,
                "mean_dqn_sensing": sensing_mean,
                "std_dqn_sensing": sensing_std,
                "mean_gap_to_best_fixed": gap_mean,
                "std_gap_to_best_fixed": gap_std,
                "seeds": seeds,
                "lambda_sync_grid": lambda_sync_values,
                "lambda_sensing_grid": lambda_sensing_values,
            }
            sweep_rows.append(row)
            if best_score is None or reward_mean > best_score:
                best_score = reward_mean
                best_runs_payload = {"row": row, "runs": run_payloads}
            print(
                "sweep lambda_sync={lambda_sync} lambda_sensing={lambda_sensing} reward={reward:.4f} gap_to_best_fixed={gap:.4f}".format(
                    lambda_sync=lambda_sync,
                    lambda_sensing=lambda_sensing,
                    reward=reward_mean,
                    gap=gap_mean,
                )
            )

    sweep_rows.sort(key=lambda item: float(item["mean_dqn_reward"]), reverse=True)
    assert best_runs_payload is not None
    best_entry = best_runs_payload["row"]
    best_histories = [payload["history"] for payload in best_runs_payload["runs"]]
    best_policy_results = [payload["policy_results"] for payload in best_runs_payload["runs"]]
    best_history_summary = aggregate_history(best_histories)
    best_policy_summary = aggregate_policy_results(best_policy_results)

    with (output_dir / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(sweep_rows, handle, indent=2)
    with (output_dir / "best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best_entry, handle, indent=2)
    with (output_dir / "best_history.json").open("w", encoding="utf-8") as handle:
        json.dump(best_history_summary, handle, indent=2)
    with (output_dir / "best_policy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(best_policy_summary, handle, indent=2)

    write_csv_table(
        output_dir / "sweep_results.csv",
        [
            "lambda_rate",
            "lambda_sync",
            "lambda_sensing",
            "mean_dqn_reward",
            "std_dqn_reward",
            "mean_dqn_rate",
            "mean_dqn_sync",
            "mean_dqn_sensing",
            "mean_gap_to_best_fixed",
        ],
        [
            [
                row["lambda_rate"],
                row["lambda_sync"],
                row["lambda_sensing"],
                row["mean_dqn_reward"],
                row["std_dqn_reward"],
                row["mean_dqn_rate"],
                row["mean_dqn_sync"],
                row["mean_dqn_sensing"],
                row["mean_gap_to_best_fixed"],
            ]
            for row in sweep_rows
        ],
    )
    write_csv_table(
        output_dir / "best_policy_summary.csv",
        [
            "policy",
            "avg_reward_mean",
            "avg_reward_std",
            "avg_rate_mean",
            "avg_sync_mean",
            "avg_sensing_mean",
        ],
        [
            [
                row["policy"],
                row["avg_reward_mean"],
                row["avg_reward_std"],
                row["avg_rate_mean"],
                row["avg_sync_mean"],
                row["avg_sensing_mean"],
            ]
            for row in best_policy_summary
        ],
    )

    weight_categories = [
        f"s={row['lambda_sync']},g={row['lambda_sensing']}"
        for row in sweep_rows
    ]
    weight_values = [float(row["mean_dqn_reward"]) for row in sweep_rows]
    write_bar_chart_svg(
        output_dir / "weight_sweep_reward.svg",
        title="Mean DQN Reward by Weight Setting",
        y_label="Mean Reward",
        categories=weight_categories,
        values=weight_values,
    )
    write_line_chart_svg(
        output_dir / "best_reward_curve.svg",
        title="Training Reward Curve for Best Weight Setting",
        x_label="Episode",
        y_label="Reward",
        series={
            "mean_reward": [
                (float(record["episode"]), float(record["reward_mean"]))
                for record in best_history_summary
            ]
        },
    )

    categories = [str(row["policy"]) for row in best_policy_summary]
    write_bar_chart_svg(
        output_dir / "best_policy_reward.svg",
        title="Policy Comparison: Average Reward",
        y_label="Average Reward",
        categories=categories,
        values=[float(row["avg_reward_mean"]) for row in best_policy_summary],
    )
    write_bar_chart_svg(
        output_dir / "best_policy_rate.svg",
        title="Policy Comparison: Average Rate",
        y_label="Average Rate",
        categories=categories,
        values=[float(row["avg_rate_mean"]) for row in best_policy_summary],
    )
    write_bar_chart_svg(
        output_dir / "best_policy_sync.svg",
        title="Policy Comparison: Average Sync Mismatch",
        y_label="Average Sync Mismatch",
        categories=categories,
        values=[float(row["avg_sync_mean"]) for row in best_policy_summary],
    )
    write_bar_chart_svg(
        output_dir / "best_policy_sensing.svg",
        title="Policy Comparison: Average Sensing Metric",
        y_label="Average Sensing Metric",
        categories=categories,
        values=[float(row["avg_sensing_mean"]) for row in best_policy_summary],
    )
    dqn_summary = next(row for row in best_policy_summary if row["policy"] == "dqn")
    action_categories = [
        f"{index}:{SystemConfig().action_space[index][0]:.1f}/{SystemConfig().action_space[index][1]:.1f}"
        for index in range(len(dqn_summary["action_counts"]))
    ]
    write_bar_chart_svg(
        output_dir / "best_action_distribution.svg",
        title="DQN Action Distribution",
        y_label="Count Across Evaluation Runs",
        categories=action_categories,
        values=[float(count) for count in dqn_summary["action_counts"]],
    )

    build_paper_pack(
        output_dir=output_dir,
        system_config=SystemConfig(
            episode_length=args.frames,
            seed=seeds[0],
            reward_rate_weight=args.lambda_rate,
            reward_sync_weight=best_entry["lambda_sync"],
            reward_sensing_weight=best_entry["lambda_sensing"],
            reward_normalize_terms=args.reward_normalize,
            sensing_metric_mode=args.sensing_mode,
        ),
        training_config=TrainingConfig(
            episodes=args.episodes,
            evaluation_episodes=args.eval_episodes,
            output_dir=str(output_dir),
            seed=seeds[0] + 1,
        ),
        sweep_rows=sweep_rows,
        best_entry=best_entry,
        best_history=best_history_summary,
        best_policy_summary=best_policy_summary,
    )
    print("best_config=" + json.dumps(best_entry, indent=2))


if __name__ == "__main__":
    main()
