# This file creates the people in the simulation.
# Each person has traits like age, politics, and online habits.

from dataclasses import dataclass
import numpy as np


# one person in the network
@dataclass
class Persona:
    id: int
    community: int  # which group they're in, 0 or 1
    age: int
    leaning: float  # politics, -1 is left and +1 is right
    media_literacy: float  # how good they are at noticing fake news, between 0 and 1
    susceptibility: float  # how easily emotional posts affect them
    activity: float  # probability they are online in a round
    belief: float = 0.0  # how much they believe the fake claims (-1 to 1), everyone starts at 0
    is_llm: bool = False  # if true the llm decides for this person instead of the rules

    def describe(self):
        # Turn numbers into a short human-style description.
        if self.leaning < -0.2:
            side = "left-leaning"
        elif self.leaning > 0.2:
            side = "right-leaning"
        else:
            side = "politically moderate"

        if self.media_literacy > 0.66:
            skill = "very good"
        elif self.media_literacy > 0.33:
            skill = "okay"
        else:
            skill = "not very good"

        emotion = "easily" if self.susceptibility > 0.5 else "rarely"

        return (f"You are a {self.age}-year-old, {side} person. "
                f"You are {skill} at telling real news from fake news, "
                f"and you {emotion} get worked up by emotional posts.")


def community_sizes(n_agents, n_communities):
    # splits people into groups as evenly as possible
    # eg 300 people and 2 groups gives [150, 150]
    base = n_agents // n_communities
    sizes = [base] * n_communities

    # if it doesn't divide evenly, the leftover people go to the first groups
    for i in range(n_agents % n_communities):
        sizes[i] += 1
    return sizes


def create_personas(cfg, rng):
    n = cfg["agents"]["n_agents"]
    k = cfg["network"]["communities"]
    sizes = community_sizes(n, k)

    # every group has its own political "center"
    # with 2 groups -> group 0 around -0.5 (left), group 1 around +0.5 (right)
    centers = np.linspace(-0.5, 0.5, k) if k > 1 else [0.0]

    personas = []
    agent_id = 0
    for c, size in enumerate(sizes):
        for _ in range(size):
            # leaning is random but close to the group's center
            leaning = np.clip(rng.normal(centers[c], 0.25), -1, 1)

            # using beta distributions because they stay between 0 and 1
            # beta(2,2) -> most people near 0.5, beta(2,3) -> slightly lower
            p = Persona(
                id=agent_id,
                community=c,
                age=int(rng.integers(18, 70)),
                leaning=float(leaning),
                media_literacy=float(rng.beta(2, 2)),
                susceptibility=float(rng.beta(2, 3)),
                activity=float(min(rng.beta(2, 4) + 0.1, 1.0)),  # most people are not online that often
            )
            personas.append(p)
            agent_id += 1

    # choose some random people to be llm agents
    # (n_llm_agents is 0 in the config for now)
    n_llm = cfg["agents"]["n_llm_agents"]
    if n_llm > 0:
        for i in rng.choice(n, size=n_llm, replace=False):
            personas[i].is_llm = True

    return personas


# just for testing this file on its own
# run it with: python -m src.personas
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # same seed every time so the results don't change between runs
    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)

    print(f"created {len(people)} agents\n")

    # print the first 3 people to see if they look ok
    for p in people[:3]:
        print(p)
        print("  ->", p.describe(), "\n")

    # check that group 0 leans left and group 1 leans right
    for c in range(cfg["network"]["communities"]):
        group = [p for p in people if p.community == c]
        avg = np.mean([p.leaning for p in group])
        print(f"community {c}: {len(group)} people, avg leaning {avg:.2f}")