"""thesis_figures.py

Generates every figure used in Chapter 5 of the thesis from the rerun artifacts:
    artifacts/thesis_rerun/candidate_A/  (lambda_sync=0.4,  lambda_sensing=0.5)
    artifacts/thesis_rerun/candidate_B/  (lambda_sync=0.25, lambda_sensing=0.5)
    artifacts/thesis_rerun/candidate_C/  (lambda_sync=0.2,  lambda_sensing=0.4) [winner]

Outputs (thesis/figures/):
    candidate_reward.png          candidate_gap.png
    candidate_stability.png
    winner_reward_curve.png       winner_policy_sync.png
    winner_policy_reward.png      winner_policy_sensing.png
    winner_policy_rate.png        winner_action_mix.png
    all_policy_bar.png            training_components.png
    action_heatmap.png
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
RERUN = ROOT / "artifacts/thesis_rerun"
OUT = ROOT / "thesis/figures"
OUT.mkdir(parents=True, exist_ok=True)

CAND_DIRS = {
    "A": RERUN / "candidate_A",
    "B": RERUN / "candidate_B",
    "C": RERUN / "candidate_C",
}
WINNER_DIR = CAND_DIRS["C"]

# ── shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BEST     = "#2ca02c"
C_DQN      = "#d62728"
C_FIXED    = "#aec7e8"
C_FIXED_D  = "#1f77b4"    # darker blue for selected fixed policies
C_RAND     = "#9467bd"
C_HEUR     = "#8c564b"
C_CAND_A   = "#bcbd22"
C_CAND_B   = "#17becf"
C_CAND_C   = "#ff7f0e"    # winner
CAND_COLOR = {"A": C_CAND_A, "B": C_CAND_B, "C": C_CAND_C}


# ── helpers ──────────────────────────────────────────────────────────────────
def load_cand_configs() -> dict[str, dict]:
    configs = {}
    for label, path in CAND_DIRS.items():
        with open(path / "best_config.json") as f:
            configs[label] = json.load(f)
    return configs


def load_winner_history() -> list[dict]:
    with open(WINNER_DIR / "best_history.json") as f:
        return json.load(f)


def load_winner_policies() -> list[dict]:
    with open(WINNER_DIR / "best_policy_summary.json") as f:
        return json.load(f)


def pretty_candidate(label: str, cfg: dict) -> str:
    return f"Candidate {label}\n" \
           f"$\\lambda_{{\\mathrm{{sync}}}}={cfg['lambda_sync']},\\;" \
           f"\\lambda_{{\\mathrm{{sensing}}}}={cfg['lambda_sensing']}$"


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# ── 1-3. Candidate comparison bar charts ─────────────────────────────────────
def plot_candidate_bars(configs: dict[str, dict]):
    labels = ["A", "B", "C"]
    colors = [CAND_COLOR[k] for k in labels]
    x = np.arange(len(labels))
    xticklabels = [pretty_candidate(k, configs[k]) for k in labels]

    # 1. reward (mean ± std)
    means = [configs[k]["mean_dqn_reward"] for k in labels]
    stds  = [configs[k]["std_dqn_reward"]  for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6, width=0.55,
                  error_kw=dict(elinewidth=1.4, ecolor="#333"))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.6, f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Mean DQN Reward (± 1 std, 5 seeds)")
    ax.set_title("Candidate Comparison: Mean DQN Reward")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save(fig, "candidate_reward.png")

    # 2. hybrid stability × baseline-gap factor score
    scores = [configs[k]["hybrid_score"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, scores, color=colors, width=0.55)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.005, f"{s:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Hybrid Score\n(stability × baseline-gap factor)")
    ax.set_title("Candidate Comparison: Hybrid Selection Score")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save(fig, "candidate_stability.png")

    # 3. gap to best fixed (negative → DQN below best fixed)
    gaps = [configs[k]["mean_gap_to_best_fixed"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, gaps, color=colors, width=0.55)
    for i, g in enumerate(gaps):
        ax.text(i, g - 0.4, f"{g:.2f}", ha="center", va="top", fontsize=10,
                color="white", fontweight="bold")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Mean Reward Gap to Best Fixed Baseline")
    ax.set_title("Candidate Comparison: Gap to Best Fixed Baseline")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save(fig, "candidate_gap.png")


# ── 4. Winner training reward curve ─────────────────────────────────────────
def plot_training_curve(history: list[dict]):
    episodes = [h["episode"] for h in history]
    reward   = np.array([h["reward_mean"] for h in history])
    # Smoothed trend line (5-episode moving average)
    w = 5
    pad = np.pad(reward, (w // 2, w // 2), mode="edge")
    smooth = np.convolve(pad, np.ones(w) / w, mode="valid")[:len(reward)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, reward, color=C_CAND_C, alpha=0.4, linewidth=1.0,
            label="Per-episode reward")
    ax.plot(episodes, smooth, color=C_CAND_C, linewidth=2.2,
            label="5-episode moving average")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean Reward (across 5 seeds)")
    ax.set_title("Winner Configuration: Training Reward Curve")
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", linewidth=0.5)
    plt.tight_layout()
    save(fig, "winner_reward_curve.png")


# ── 5-8. Winner policy comparison bars (selected policies) ───────────────────
SELECTED = ["fixed_rho=0.1_beta=0.2",
            "fixed_rho=0.2_beta=0.2",
            "fixed_rho=0.3_beta=0.2",
            "dqn",
            "random",
            "heuristic"]
SEL_SHORT = {
    "fixed_rho=0.1_beta=0.2": "F1",
    "fixed_rho=0.2_beta=0.2": "F2",
    "fixed_rho=0.3_beta=0.2": "F3",
    "dqn": "DQN",
    "random": "RAND",
    "heuristic": "HEUR",
}
SEL_COLOR = {
    "fixed_rho=0.1_beta=0.2": C_BEST,
    "fixed_rho=0.2_beta=0.2": C_FIXED_D,
    "fixed_rho=0.3_beta=0.2": C_FIXED_D,
    "dqn": C_DQN,
    "random": C_RAND,
    "heuristic": C_HEUR,
}


def _sel_bar(policies: list[dict], key_mean: str, key_std: str,
             title: str, ylabel: str, filename: str):
    lookup = {p["policy"]: p for p in policies}
    labels = [SEL_SHORT[p] for p in SELECTED]
    means  = [lookup[p][key_mean] for p in SELECTED]
    stds   = [lookup[p][key_std]  for p in SELECTED] if key_std else None
    colors = [SEL_COLOR[p] for p in SELECTED]
    x = np.arange(len(SELECTED))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, means,
           yerr=stds if stds else None,
           color=colors, width=0.6, capsize=5,
           error_kw=dict(elinewidth=1.2, ecolor="#333"))
    for i, m in enumerate(means):
        offset = (stds[i] if stds else 0.0) + max(means) * 0.015
        ax.text(i, m + offset, f"{m:.3f}" if m < 10 else f"{m:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save(fig, filename)


def plot_winner_policy_bars(policies: list[dict]):
    _sel_bar(policies, "avg_reward_mean", "avg_reward_std",
             title="Policy Comparison: Mean Reward",
             ylabel="Mean Reward",
             filename="winner_policy_reward.png")
    _sel_bar(policies, "avg_rate_mean", "avg_rate_std",
             title="Policy Comparison: Mean Communication Rate",
             ylabel="Mean Rate (bits/s/Hz)",
             filename="winner_policy_rate.png")
    _sel_bar(policies, "avg_sync_mean", "avg_sync_std",
             title="Policy Comparison: Mean Sync Mismatch $\\mathcal{E}_t$",
             ylabel="Mean $\\mathcal{E}_t$ (lower is better)",
             filename="winner_policy_sync.png")
    _sel_bar(policies, "avg_sensing_mean", "avg_sensing_std",
             title="Policy Comparison: Mean Sensing Metric $S_t$",
             ylabel="Mean $S_t$ (lower is better)",
             filename="winner_policy_sensing.png")


# ── 9. Action mix bar chart (all 12 actions, DQN only) ──────────────────────
def plot_action_mix(policies: list[dict]):
    dqn = next(p for p in policies if p["policy"] == "dqn")
    counts = np.array(dqn["action_counts"], dtype=float)
    total = counts.sum()
    pct = 100 * counts / total
    actions = [(r, b) for r in (0.1, 0.2, 0.3, 0.4) for b in (0.2, 0.4, 0.6)]
    labels = [f"({r},{b})" for r, b in actions]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, pct, color=C_DQN, width=0.65)
    for i, p in enumerate(pct):
        if p > 0.3:
            ax.text(i, p + 0.3, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Action $(\\rho, \\beta)$")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("DQN Action Mix (Evaluation, 1600 Decisions)")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save(fig, "winner_action_mix.png")


# ── 10. All-policy horizontal bar chart ─────────────────────────────────────
def plot_all_policy_bar(policies: list[dict]):
    label_map = {
        "dqn":       "DQN (*)",
        "random":    "Random",
        "heuristic": "Heuristic",
    }
    def make_label(policy: str) -> str:
        if policy in label_map:
            return label_map[policy]
        parts = policy.replace("fixed_", "").split("_")
        rho = parts[0].replace("rho=", "ρ=")
        beta = parts[1].replace("beta=", "β=")
        return f"Fixed {rho}, {beta}"

    rows = sorted(policies, key=lambda x: x["avg_reward_mean"])
    labels = [make_label(r["policy"]) for r in rows]
    means  = [r["avg_reward_mean"] for r in rows]
    stds   = [r["avg_reward_std"]  for r in rows]

    # best fixed = top-scoring fixed_* policy
    fixed_rows = [r for r in rows if r["policy"].startswith("fixed_")]
    best_fixed_name = max(fixed_rows, key=lambda r: r["avg_reward_mean"])["policy"]

    def bar_color(p):
        if p == "dqn":          return C_DQN
        if p == "random":       return C_RAND
        if p == "heuristic":    return C_HEUR
        if p == best_fixed_name: return C_BEST
        return C_FIXED
    colors = [bar_color(r["policy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=stds, color=colors,
            error_kw=dict(elinewidth=1.2, capsize=3, ecolor="#444"),
            height=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Reward (± 1 std, 5 seeds)")
    ax.set_title("Policy Comparison – All 15 Baselines vs DQN")
    ax.axvline(means[labels.index("DQN (*)")], color=C_DQN,
               linestyle="--", linewidth=1.0, alpha=0.6)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, zorder=0)

    handles = [
        mpatches.Patch(color=C_BEST,  label="Hindsight-best fixed"),
        mpatches.Patch(color=C_DQN,   label="DQN"),
        mpatches.Patch(color=C_FIXED, label="Other fixed policy"),
        mpatches.Patch(color=C_RAND,  label="Random"),
        mpatches.Patch(color=C_HEUR,  label="Heuristic"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    save(fig, "all_policy_bar.png")


# ── 11. Per-component training curves (3 panels) ────────────────────────────
def plot_training_components(history: list[dict]):
    episodes = [h["episode"] for h in history]
    rate     = np.array([h["rate_mean"]    for h in history])
    sync     = np.array([h["sync_mean"]    for h in history])
    sensing  = np.array([h["sensing_mean"] for h in history])

    def smooth(y, w=5):
        p = np.pad(y, (w // 2, w // 2), mode="edge")
        return np.convolve(p, np.ones(w) / w, mode="valid")[:len(y)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(episodes, rate, color="#aec7e8", linewidth=0.8, alpha=0.6)
    axes[0].plot(episodes, smooth(rate), color="#1f77b4", linewidth=2.0,
                 label="Rate (smoothed)")
    axes[0].set_ylabel("Mean Rate\n(bits/s/Hz)")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Reward Component Evolution During Training")
    axes[0].grid(linestyle=":", linewidth=0.5)

    sigma2 = 0.05
    noise_floor = 2 * sigma2
    axes[1].plot(episodes, sync, color="#ffbb78", linewidth=0.8, alpha=0.6)
    axes[1].plot(episodes, smooth(sync), color="#ff7f0e", linewidth=2.0,
                 label="Sync mismatch (smoothed)")
    axes[1].axhline(noise_floor, color="#888", linestyle="--", linewidth=1.0,
                    label=f"Noise floor $2\\sigma^2={noise_floor:.2f}$")
    axes[1].set_ylabel("Mean Sync\nMismatch $\\mathcal{E}_t$")
    axes[1].legend(loc="upper right")
    axes[1].grid(linestyle=":", linewidth=0.5)

    axes[2].plot(episodes, sensing, color="#98df8a", linewidth=0.8, alpha=0.6)
    axes[2].plot(episodes, smooth(sensing), color="#2ca02c", linewidth=2.0,
                 label="Sensing metric (smoothed)")
    axes[2].set_ylabel("Mean Sensing\nMetric $S_t$")
    axes[2].set_xlabel("Training Episode")
    axes[2].legend(loc="upper right")
    axes[2].grid(linestyle=":", linewidth=0.5)

    plt.tight_layout()
    save(fig, "training_components.png")


# ── 12. Action frequency heatmap ─────────────────────────────────────────────
def plot_action_heatmap(policies: list[dict]):
    dqn = next(p for p in policies if p["policy"] == "dqn")
    counts = np.array(dqn["action_counts"], dtype=float)
    pct = (counts / counts.sum() * 100).reshape(4, 3)

    rho_labels  = [r"$\rho = 0.1$", r"$\rho = 0.2$", r"$\rho = 0.3$", r"$\rho = 0.4$"]
    beta_labels = [r"$\beta = 0.2$", r"$\beta = 0.4$", r"$\beta = 0.6$"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax = max(35.0, pct.max() * 1.05)
    im = ax.imshow(pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(3));  ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(4));  ax.set_yticklabels(rho_labels)
    ax.set_xlabel("Pilot Power Fraction $\\beta$")
    ax.set_ylabel("Pilot Ratio $\\rho$")
    ax.set_title("DQN Action Selection Frequency (%)")

    for i in range(4):
        for j in range(3):
            v = pct[i, j]
            c = "white" if v > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=11, color=c, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Selection frequency (%)")
    plt.tight_layout()
    save(fig, "action_heatmap.png")


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating thesis figures…")
    configs  = load_cand_configs()
    history  = load_winner_history()
    policies = load_winner_policies()

    plot_candidate_bars(configs)
    plot_training_curve(history)
    plot_winner_policy_bars(policies)
    plot_action_mix(policies)
    plot_all_policy_bar(policies)
    plot_training_components(history)
    plot_action_heatmap(policies)
    print("Done.")
