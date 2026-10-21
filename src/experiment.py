# experiment.py
# runs every scenario many times with different seeds and compares them
# one run can be lucky or unlucky, so we look at the average + confidence interval

import argparse
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # same tkinter problem as in metrics.py
import matplotlib.pyplot as plt

from src.simulation import load_config, run_simulation
from src.metrics import compute_metrics

# what we compare
# baseline = no intervention + engagement feed, everything else gets compared to this
SCENARIOS = {
    "baseline": {"intervention": "none", "ranking": "engagement"},
    "fact_check": {"intervention": "fact_check", "ranking": "engagement"},
    "downrank": {"intervention": "downrank", "ranking": "engagement"},
    "prebunk": {"intervention": "prebunk", "ranking": "engagement"},
    "chronological": {"intervention": "none", "ranking": "chronological"},
}

# the metrics that go into the summary table
METRICS = ["fake_shares", "true_shares", "fake_reach", "fake_depth", "believers", "polarization"]


def run_experiment(base_cfg, n_seeds):
    rows = []

    for name, setup in SCENARIOS.items():
        for seed in range(n_seeds):
            # copy the config so one run doesn't mess up the next one
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["intervention"]["type"] = setup["intervention"]
            cfg["feed"]["ranking"] = setup["ranking"]

            # rule agents only here, with the llm 50 runs would take hours on a laptop
            cfg["agents"]["n_llm_agents"] = 0

            result = run_simulation(cfg)
            rows.append({"scenario": name, "seed": seed, **compute_metrics(result)})

        print(f"  {name} done ({n_seeds} runs)")

    return pd.DataFrame(rows)


def summarize(df):
    # mean and 95% confidence interval for every metric per scenario
    # ci = 1.96 * std / sqrt(n), if two scenarios' intervals don't overlap
    # the difference is probably real and not just random luck
    out = []
    for name, g in df.groupby("scenario", sort=False):
        row = {"scenario": name}
        for col in METRICS:
            values = g[col]
            row[col] = values.mean()
            row[col + "_ci"] = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        out.append(row)
    return pd.DataFrame(out)


def plot_comparison(summary, path="results/interventions.png"):
    panels = [
        ("fake_shares", "avg shares per fake post"),
        ("fake_reach", "share of people reached per fake post"),
        ("believers", "share of people believing fake claims"),
    ]

    # baseline in gray so it stands out as the reference
    colors = ["gray" if s == "baseline" else "tab:blue" for s in summary.scenario]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, panels):
        ax.bar(summary.scenario, summary[col], yerr=summary[col + "_ci"],
               color=colors, capsize=5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# run with: python -m src.experiment
# or fewer seeds for a quick test: python -m src.experiment --seeds 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="runs per scenario")
    args = parser.parse_args()

    cfg = load_config()
    total = len(SCENARIOS) * args.seeds
    print(f"running {len(SCENARIOS)} scenarios x {args.seeds} seeds = {total} simulations\n")

    df = run_experiment(cfg, args.seeds)
    summary = summarize(df)

    # save everything, raw runs + summary
    df.to_csv("results/experiment_runs.csv", index=False)
    summary.to_csv("results/experiment_summary.csv", index=False)

    print("\n--- results (mean +- 95% ci) ---")
    base = summary[summary.scenario == "baseline"].iloc[0]

    for row in summary.itertuples():
        shares = f"{row.fake_shares:5.1f} +- {row.fake_shares_ci:.1f}"
        believers = f"{row.believers:.0%} +- {row.believers_ci:.0%}"

        # how much better or worse than doing nothing
        change = (row.fake_shares - base.fake_shares) / base.fake_shares
        change_text = "" if row.scenario == "baseline" else f"({change:+.0%} vs baseline)"

        print(f"{row.scenario:14s} fake shares {shares}   believers {believers}   {change_text}")

    plot_comparison(summary)
    print("\nsaved results/experiment_runs.csv, results/experiment_summary.csv, results/interventions.png")