# metrics.py
# takes the event log from the simulation and turns it into numbers we can compare
# also makes the plots

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw straight to a file, no popup window (tkinter is broken on my python install)
import matplotlib.pyplot as plt


def cascade_depths(events):
    # Depth is the longest chain of reposts from a post.
    # Example: post -> person shares -> friend shares again.
    depth = {}
    max_depth = {}

    for e in events.itertuples():
        if e.action == "post":
            depth[(e.post, e.agent)] = 0
            max_depth[e.post] = 0
        elif e.action == "share":
            d = depth.get((e.post, e.parent), 0) + 1
            depth[(e.post, e.agent)] = d
            max_depth[e.post] = max(max_depth[e.post], d)

    return max_depth


def post_stats(events):
    # One row for each post, with reach, depth, and share count.
    depths = cascade_depths(events)
    rows = []

    for post_id, ev in events.groupby("post"):
        shares = ev[ev.action == "share"]

        # reach = everyone who saw it + the author
        reached = set(ev[ev.action == "view"].agent) | set(ev[ev.action == "post"].agent)

        # the round where this post got shared the most
        if len(shares) > 0:
            peak = int(shares.step.value_counts().idxmax())
        else:
            peak = 0

        rows.append({
            "post": post_id,
            "is_false": bool(ev.is_false.iloc[0]),
            "shares": len(shares),
            "reach": len(reached),
            "depth": depths.get(post_id, 0),
            "peak_round": peak,
        })

    return pd.DataFrame(rows)


def compute_metrics(result):
    # Summarize one run into a few numbers we can compare across runs.
    events = result["events"]
    personas = result["personas"]
    n = len(personas)

    stats = post_stats(events)
    fake = stats[stats.is_false]
    true = stats[~stats.is_false]

    beliefs = np.array([p.belief for p in personas])

    # Polarization is the gap between the strongest and weakest belief groups.
    groups = sorted(set(p.community for p in personas))
    group_avg = [np.mean([p.belief for p in personas if p.community == c]) for c in groups]

    return {
        "fake_shares": float(fake.shares.mean()),  # avg shares per fake post
        "true_shares": float(true.shares.mean()),
        "fake_reach": float(fake.reach.mean() / n),  # share of the population that saw an avg fake post
        "true_reach": float(true.reach.mean() / n),
        "fake_depth": float(fake.depth.mean()),
        "true_depth": float(true.depth.mean()),
        "fake_peak_round": float(fake.peak_round.mean()),
        "true_peak_round": float(true.peak_round.mean()),
        "avg_belief": float(beliefs.mean()),
        "believers": float((beliefs > 0.2).mean()),  # % of people who clearly believe the fake claims
        "polarization": float(max(group_avg) - min(group_avg)),
    }


def plot_run(result, path="results/run.png"):
    # Save the main charts for one run.
    events = result["events"]
    beliefs = result["beliefs"]
    steps = int(beliefs.step.max())
    shares = events[events.action == "share"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: total shares over time, fake vs true
    for is_false, label, color in [(False, "true posts", "tab:blue"), (True, "fake posts", "tab:red")]:
        per_round = shares[shares.is_false == is_false].groupby("step").size()
        per_round = per_round.reindex(range(1, steps + 1), fill_value=0)  # rounds with 0 shares too
        ax1.plot(per_round.index, per_round.cumsum(), label=label, color=color, linewidth=2)

    ax1.set_title("total shares over time")
    ax1.set_xlabel("round")
    ax1.set_ylabel("shares so far")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # right: how much each group believes the fake claims
    for c, g in beliefs.groupby("community"):
        ax2.plot(g.step, g.avg_belief, label=f"group {c}", linewidth=2)

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")  # 0 = neutral
    ax2.set_title("belief in fake claims")
    ax2.set_xlabel("round")
    ax2.set_ylabel("avg belief")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# test with: python -m src.metrics
if __name__ == "__main__":
    from src.core.simulation import load_config, run_simulation

    cfg = load_config()
    result = run_simulation(cfg)

    print("--- metrics ---")
    for name, value in compute_metrics(result).items():
        print(f"{name:16s} {value:.3f}")

    # the 5 posts that spread the most
    stats = post_stats(result["events"]).sort_values("shares", ascending=False)
    print("\n--- top 5 posts ---")
    for row in stats.head(5).itertuples():
        tag = "FAKE" if row.is_false else "TRUE"
        text = result["posts"][row.post].text[:60]
        print(f"[{tag}] shares={row.shares} reach={row.reach} depth={row.depth}  {text}")

    plot_run(result)
    print("\nsaved plot to results/run.png")