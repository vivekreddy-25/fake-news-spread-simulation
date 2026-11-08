# llm_demo.py
# runs one simulation where some people are llm agents
# and compares how they behave vs the rule based people
# kept small on purpose, every llm answer takes a few seconds on my laptop

import argparse
import time

import pandas as pd

from src.core.simulation import load_config, run_simulation
from src.analysis.metrics import compute_metrics, plot_run


def share_rate(events, is_llm, is_false):
    # Share rate = posts shared divided by posts seen.
    views = events[(events.action == "view") & (events.is_llm == is_llm) & (events.is_false == is_false)]
    shares = events[(events.action == "share") & (events.is_llm == is_llm) & (events.is_false == is_false)]
    if len(views) == 0:
        return float("nan")
    return len(shares) / len(views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-agents", type=int, default=6)
    parser.add_argument("--steps", type=int, default=15)  # most of the spreading happens early anyway
    parser.add_argument("--intervention", default="none")
    parser.add_argument("--memory", choices=["on", "off"], default="on",
                         help="turn agent memory + reflection on or off, run it both ways to compare")
    args = parser.parse_args()

    cfg = load_config()
    cfg["agents"]["n_llm_agents"] = args.llm_agents
    cfg["steps"] = args.steps
    cfg["intervention"]["type"] = args.intervention
    cfg["llm"]["memory"] = (args.memory == "on")

    print(f"{args.llm_agents} llm agents out of {cfg['agents']['n_agents']}, "
          f"{args.steps} rounds, intervention: {args.intervention}, memory: {args.memory}")
    print("this takes a few minutes...\n")

    start = time.time()
    result = run_simulation(cfg, verbose=True)
    minutes = (time.time() - start) / 60

    events = result["events"]
    print(f"\ndone in {minutes:.1f} min, llm stats: {result['llm_stats']}")

    # compare the two kinds of brains
    print("\n--- share rate (shares / posts seen) ---")
    for is_llm, name in [(False, "rule agents"), (True, "llm agents")]:
        fake = share_rate(events, is_llm, True)
        true = share_rate(events, is_llm, False)
        print(f"{name:12s} fake {fake:.0%}   true {true:.0%}")

    # show some of the reasons the llm people gave
    llm_views = events[(events.action == "view") & (events.is_llm)]
    shared = set(zip(events[events.action == "share"].agent, events[events.action == "share"].post))

    print("\n--- what the llm agents were thinking ---")
    for row in llm_views.head(10).itertuples():
        post = result["posts"][row.post]
        tag = "FAKE" if post.is_false else "TRUE"
        did = "SHARED" if (row.agent, row.post) in shared else "ignored"
        print(f"[{tag}] {did:7s} person {row.agent}: {post.text[:55]}...")
        print(f"         \"{row.reason}\"")

    # save every llm decision so i can read them all later
    llm_views.to_csv("results/llm_decisions.csv", index=False)

    # same thing but with the memory that went into each decision (empty file if memory is off)
    llm_brain = result["llm_brain"]
    pd.DataFrame(llm_brain.decision_log).to_csv("results/llm_decisions_memory.csv", index=False)

    print("\n--- example reflections ---")
    reflections = list(llm_brain.reflections.items())
    if not reflections:
        print("(none yet - needs memory on and at least 3 memories per agent)")
    for persona_id, summary in reflections[:3]:
        print(f"person {persona_id}: \"{summary}\"")

    print("\n--- metrics ---")
    for name, value in compute_metrics(result).items():
        print(f"{name:16s} {value:.3f}")

    plot_run(result, path="results/run_llm.png")
    print("\nsaved results/llm_decisions.csv, results/llm_decisions_memory.csv and results/run_llm.png")