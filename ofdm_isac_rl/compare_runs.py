"""Compare a new sweep run against the 100-episode baseline."""

import json
import sys
import pathlib


def load(run_dir: pathlib.Path) -> dict:
    cfg = json.loads((run_dir / "best_config.json").read_text())
    history = json.loads((run_dir / "best_history.json").read_text())
    policies = json.loads((run_dir / "best_policy_summary.json").read_text())
    return {"cfg": cfg, "history": history, "policies": policies}


def main(new_dir: str):
    root = pathlib.Path("/home/dell/dev/FYP-8-SEM/FYP-8-SEM")
    baseline = load(root / "thesis_snapshot/experiment_results/winner_l02_l04")
    new = load(root / new_dir)

    print("=" * 80)
    print(f"100-episode baseline  vs  new run ({new_dir})")
    print("=" * 80)

    print(f"\n{'Metric':<40}{'100-ep':>15}{'new':>15}{'Δ':>15}")
    print("-" * 85)
    for key, label in [
        ("mean_dqn_reward", "Mean DQN reward"),
        ("std_dqn_reward",  "Std DQN reward"),
        ("mean_dqn_rate",   "Mean DQN rate"),
        ("mean_dqn_sync",   "Mean DQN sync"),
        ("mean_dqn_sensing","Mean DQN sensing"),
        ("mean_gap_to_best_fixed", "Gap to best fixed"),
        ("stability_score", "Stability score"),
        ("baseline_gap_factor",  "Baseline gap factor"),
        ("hybrid_score",    "Hybrid score"),
        ("final_window_reward_mean", "Final-window reward mean"),
        ("final_window_reward_std",  "Final-window reward std"),
    ]:
        a = baseline["cfg"][key]
        b = new["cfg"][key]
        delta = b - a
        print(f"{label:<40}{a:>15.4f}{b:>15.4f}{delta:>+15.4f}")

    # Policy ranking
    print("\n" + "=" * 80)
    print("Policy ranking (top 5)")
    print("=" * 80)
    for name, pols in [("100-ep", baseline["policies"]), ("new", new["policies"])]:
        pols_sorted = sorted(pols, key=lambda x: -x["avg_reward_mean"])
        print(f"\n-- {name} --")
        for i, p in enumerate(pols_sorted[:5], 1):
            print(f"  {i}. {p['policy']:<30} reward={p['avg_reward_mean']:7.3f} ± {p['avg_reward_std']:5.3f}")
        dqn_rank = next(i for i, p in enumerate(pols_sorted, 1) if p["policy"] == "dqn")
        dqn = next(p for p in pols_sorted if p["policy"] == "dqn")
        print(f"  DQN rank: #{dqn_rank}  reward={dqn['avg_reward_mean']:.3f}")

    # DQN action distribution
    dqn_new = next(p for p in new["policies"] if p["policy"] == "dqn")
    dqn_old = next(p for p in baseline["policies"] if p["policy"] == "dqn")
    total_new = sum(dqn_new["action_counts"])
    total_old = sum(dqn_old["action_counts"])
    print("\n" + "=" * 80)
    print("DQN action distribution (% share)")
    print("=" * 80)
    actions = [(r, b) for r in (0.1, 0.2, 0.3, 0.4) for b in (0.2, 0.4, 0.6)]
    print(f"{'Action (ρ,β)':<20}{'100-ep %':>15}{'new %':>15}{'Δ':>12}")
    for i, (r, b) in enumerate(actions):
        po = 100 * dqn_old["action_counts"][i] / total_old
        pn = 100 * dqn_new["action_counts"][i] / total_new
        print(f"  ({r},{b}){' '*10}{po:>13.1f}{pn:>14.1f}{pn-po:>+12.1f}")

    # Training curve key points
    hist_new = new["history"]
    print("\n" + "=" * 80)
    print("Training trajectory (new run, per-episode reward means across seeds)")
    print("=" * 80)
    checkpoints = [1, 25, 50, 100, 200, 300, 400, 499]
    for ep in checkpoints:
        if ep <= len(hist_new):
            r = hist_new[ep - 1]["reward_mean"]
            print(f"  Episode {ep:4d}: reward = {r:.3f}")
    peak = max(hist_new, key=lambda h: h["reward_mean"])
    print(f"  Peak: episode {peak['episode']} reward = {peak['reward_mean']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/extended_500ep/winner")
