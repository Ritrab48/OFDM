"""Command-line entrypoint for training and comparing policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .baselines import build_baseline_policies
from .config import SystemConfig, TrainingConfig
from .dqn import DQNAgent
from .environment import OfdmIsacEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DQN-based OFDM-ISAC reference model."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="artifacts/reference_model")
    parser.add_argument("--lambda-rate", type=float, default=1.0)
    parser.add_argument("--lambda-sync", type=float, default=1.0)
    parser.add_argument("--lambda-sensing", type=float, default=0.5)
    parser.add_argument(
        "--reward-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize reward terms by fixed reference scales before weighting.",
    )
    parser.add_argument(
        "--sensing-mode",
        choices=("crlb", "empirical"),
        default="empirical",
        help="Use the CRLB override from the supervisor or the observable empirical proxy.",
    )
    return parser.parse_args()


def evaluate_selector(
    env: OfdmIsacEnv,
    select_action,
    episodes: int,
    name: str,
) -> dict[str, object]:
    total_reward = 0.0
    total_rate = 0.0
    total_sensing = 0.0
    total_sync = 0.0
    action_counts = [0 for _ in env.config.action_space]
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = select_action(state)
            action_counts[action] += 1
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_rate += info["rate"]
            total_sensing += info["sensing_metric"]
            total_sync += info["sync_mismatch"]
    normalizer = episodes * env.config.episode_length
    return {
        "policy": name,
        "avg_reward": total_reward / episodes,
        "avg_rate": total_rate / normalizer,
        "avg_sensing_metric": total_sensing / normalizer,
        "avg_sync_mismatch": total_sync / normalizer,
        "action_counts": action_counts,
    }


def run_episode(
    env: OfdmIsacEnv,
    agent: DQNAgent,
    training: bool,
) -> dict[str, float]:
    state = env.reset()
    episode_reward = 0.0
    total_rate = 0.0
    total_sensing = 0.0
    total_sync = 0.0
    total_loss = 0.0
    loss_updates = 0
    done = False

    while not done:
        action = agent.select_action(state, explore=training)
        next_state, reward, done, info = env.step(action)
        if training:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_updates += 1
        episode_reward += reward
        total_rate += info["rate"]
        total_sensing += info["sensing_metric"]
        total_sync += info["sync_mismatch"]
        state = next_state

    steps = env.config.episode_length
    return {
        "reward": episode_reward,
        "avg_rate": total_rate / steps,
        "avg_sensing_metric": total_sensing / steps,
        "avg_sync_mismatch": total_sync / steps,
        "avg_loss": total_loss / max(loss_updates, 1),
    }


def format_value(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-3 or magnitude >= 1.0e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---" for _ in headers]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_summary_markdown(
    output_dir: Path,
    history: list[dict[str, object]],
    policy_results: list[dict[str, object]],
    system_config: SystemConfig,
) -> None:
    training_rows: list[list[str]] = []
    for record in history:
        training_rows.append(
            [
                str(record["episode"]),
                format_value(float(record["reward"])),
                format_value(float(record["avg_rate"])),
                format_value(float(record["avg_sync_mismatch"])),
                format_value(float(record["avg_sensing_metric"])),
                format_value(float(record["avg_loss"])),
            ]
        )

    policy_rows: list[list[str]] = []
    for result in sorted(policy_results, key=lambda item: float(item["avg_reward"]), reverse=True):
        policy_rows.append(
            [
                str(result["policy"]),
                format_value(float(result["avg_reward"])),
                format_value(float(result["avg_rate"])),
                format_value(float(result["avg_sync_mismatch"])),
                format_value(float(result["avg_sensing_metric"])),
            ]
        )

    dqn_result = next(result for result in policy_results if result["policy"] == "dqn")
    total_actions = sum(int(count) for count in dqn_result["action_counts"])
    action_rows: list[list[str]] = []
    for action_index, count in enumerate(dqn_result["action_counts"]):
        rho, beta = system_config.action_space[action_index]
        percentage = (count / total_actions) * 100.0 if total_actions else 0.0
        action_rows.append(
            [
                str(action_index),
                f"{rho:.1f}",
                f"{beta:.1f}",
                str(count),
                f"{percentage:.2f}%",
            ]
        )

    sections = [
        "# Training Summary",
        "",
        "## Training Progress",
        "",
        render_markdown_table(
            ["Episode", "Reward", "Rate", "Sync", "Sensing", "Loss"],
            training_rows,
        ),
        "",
        "## Policy Comparison",
        "",
        render_markdown_table(
            ["Policy", "Avg Reward", "Avg Rate", "Avg Sync", "Avg Sensing"],
            policy_rows,
        ),
        "",
        "## DQN Action Distribution",
        "",
        render_markdown_table(
            ["Action", "rho", "beta", "Count", "Share"],
            action_rows,
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(sections), encoding="utf-8")


def make_evaluation_config(system_config: SystemConfig, seed: int) -> SystemConfig:
    return replace(system_config, seed=seed)


def main() -> None:
    args = parse_args()
    system_config = SystemConfig(
        episode_length=args.frames,
        seed=args.seed,
        reward_rate_weight=args.lambda_rate,
        reward_sync_weight=args.lambda_sync,
        reward_sensing_weight=args.lambda_sensing,
        reward_normalize_terms=args.reward_normalize,
        sensing_metric_mode=args.sensing_mode,
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        evaluation_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed + 1,
    )

    env = OfdmIsacEnv(system_config)
    evaluation_config = make_evaluation_config(system_config, seed=system_config.seed + 997)
    agent = DQNAgent(system_config, training_config)

    history: list[dict[str, object]] = []
    for episode in range(1, training_config.episodes + 1):
        metrics = run_episode(env, agent, training=True)
        record: dict[str, object] = {"episode": episode, **metrics}
        if episode % training_config.evaluation_interval == 0 or episode == training_config.episodes:
            record["evaluation"] = evaluate_selector(
                OfdmIsacEnv(evaluation_config),
                agent.act_greedily,
                training_config.evaluation_episodes,
                name="dqn_eval",
            )
        history.append(record)
        print(
            "episode={episode} reward={reward} rate={rate} "
            "sync={sync} sensing={sensing} loss={loss} epsilon={epsilon:.3f}".format(
                episode=episode,
                reward=format_value(metrics["reward"]),
                rate=format_value(metrics["avg_rate"]),
                sync=format_value(metrics["avg_sync_mismatch"]),
                sensing=format_value(metrics["avg_sensing_metric"]),
                loss=format_value(metrics["avg_loss"]),
                epsilon=agent.epsilon(),
            )
        )

    policy_results = [
        evaluate_selector(
            OfdmIsacEnv(evaluation_config),
            agent.act_greedily,
            training_config.evaluation_episodes,
            name="dqn",
        )
    ]

    for policy in build_baseline_policies(system_config, seed=args.seed + 123):
        policy_results.append(
            evaluate_selector(
                OfdmIsacEnv(evaluation_config),
                policy.select_action,
                training_config.evaluation_episodes,
                name=policy.name,
            )
        )

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dir / "model.json").open("w", encoding="utf-8") as handle:
        json.dump(agent.export(), handle)
    with (output_dir / "policy_results.json").open("w", encoding="utf-8") as handle:
        json.dump(policy_results, handle, indent=2)

    write_summary_markdown(output_dir, history, policy_results, system_config)
    print("policy_results=" + json.dumps(policy_results, indent=2))


if __name__ == "__main__":
    main()
