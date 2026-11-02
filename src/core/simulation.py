# This file runs the whole simulation.
# Each round, people check posts, decide, and update their beliefs.

import numpy as np
import pandas as pd
import yaml

from src.models.personas import create_personas
from src.models.network import build_network
from src.models.content import create_posts
from src.models.feed import build_feed
from src.agents.rule import RuleAgent
from src.core.interventions import Intervention


def load_config(path="config.yaml"):
    # Read the setup file.
    with open(path) as f:
        return yaml.safe_load(f)


def run_simulation(cfg, verbose=False):
    # Build the world and run the rounds.
    # one random generator for everything -> same seed gives the exact same run
    rng = np.random.default_rng(cfg["seed"])

    # build the world
    personas = create_personas(cfg, rng)
    G = build_network(cfg, rng)
    posts = create_posts(cfg, personas, rng)

    intervention = Intervention(cfg)
    intervention.at_start(personas, rng)  # only does something for prebunk

    rule_brain = RuleAgent()

    # only load the llm stuff if we actually have llm agents
    # (the import is inside the if, so the rule-only runs don't need langgraph or ollama)
    llm_brain = None
    if any(p.is_llm for p in personas):
        from src.agents.llm import LLMAgent
        llm_brain = LLMAgent(cfg)

    feed_size = cfg["feed"]["size"]
    ranking = cfg["feed"]["ranking"]
    max_llm_posts = cfg["llm"]["max_posts_per_step"]

    seen = {p.id: set() for p in personas}  # what each person already saw
    events = []  # every view and share ends up in here
    beliefs = []  # avg belief per group per round, for plotting later

    # round 0 = the authors post their stuff
    for post in posts:
        events.append({
            "step": 0, "agent": post.author, "post": post.id, "parent": None,
            "action": "post", "is_false": post.is_false, "is_llm": False, "reason": "",
        })

    for step in range(1, cfg["steps"] + 1):
        intervention.each_round(step, posts)

        # shuffle the order every round so the same people don't always go first
        order = rng.permutation(len(personas))

        for i in order:
            person = personas[i]

            # not everyone is online every round
            if rng.random() > person.activity:
                continue

            feed = build_feed(person.id, G, posts, seen[person.id], feed_size,
                    ranking=ranking, downranked=intervention.downranked, rng=rng)

            # llm people use the llm brain, but only look at a few posts so it's not too slow
            if person.is_llm and llm_brain is not None:
                brain = llm_brain
                feed = feed[:max_llm_posts]
            else:
                brain = rule_brain

            for post, from_friend in feed:
                seen[person.id].add(post.id)

                decision = brain.decide(person, post, rng)

                # update belief but keep it between -1 and 1
                person.belief = float(np.clip(person.belief + decision["belief_change"], -1, 1))

                events.append({
                    "step": step, "agent": person.id, "post": post.id, "parent": from_friend,
                    "action": "view", "is_false": post.is_false, "is_llm": person.is_llm,
                    "reason": decision["reason"] if person.is_llm else "",  # keep llm reasons even if they don't share
                })

                if decision["share"]:
                    post.shared_by[person.id] = step
                    events.append({
                        "step": step, "agent": person.id, "post": post.id, "parent": from_friend,
                        "action": "share", "is_false": post.is_false, "is_llm": person.is_llm,
                        "reason": decision["reason"],
                    })

        # save how much each group believes the fake stuff after this round
        for c in sorted(set(p.community for p in personas)):
            group = [p.belief for p in personas if p.community == c]
            beliefs.append({"step": step, "community": c, "avg_belief": float(np.mean(group))})

        if verbose:
            n_shares = sum(1 for e in events if e["step"] == step and e["action"] == "share")
            print(f"round {step:2d}: {n_shares} shares")

    return {
        "events": pd.DataFrame(events),
        "beliefs": pd.DataFrame(beliefs),
        "personas": personas,
        "posts": posts,
        "graph": G,
        "llm_stats": llm_brain.stats if llm_brain else None,  # how many llm calls / fallbacks
    }


# run one full simulation with: python -m src.core.simulation
if __name__ == "__main__":
    cfg = load_config()
    print(f"intervention: {cfg['intervention']['type']}, ranking: {cfg['feed']['ranking']}\n")

    result = run_simulation(cfg, verbose=True)
    events = result["events"]

    print("\n--- summary ---")
    for is_false, name in [(False, "TRUE"), (True, "FAKE")]:
        shares = events[(events.action == "share") & (events.is_false == is_false)]
        views = events[(events.action == "view") & (events.is_false == is_false)]
        print(f"{name} posts: {len(shares)} shares, seen by {views.agent.nunique()} people")

    final = result["beliefs"][result["beliefs"].step == cfg["steps"]]
    for _, row in final.iterrows():
        print(f"group {int(row.community)} final belief in fake claims: {row.avg_belief:+.2f}")

    # save the raw log so we can look at it in excel or pandas
    events.to_csv("results/events.csv", index=False)
    print("\nsaved results/events.csv")