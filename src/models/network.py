# network.py
# builds the social network, basically who is friends with who
# people mostly connect with people from their own group -> echo chambers

import networkx as nx
import numpy as np

from src.models.personas import community_sizes


def build_network(cfg, rng):
    # Build a social graph where people are more likely to know others in their own group.
    n = cfg["agents"]["n_agents"]
    k = cfg["network"]["communities"]
    p_in = cfg["network"]["p_in"]
    p_out = cfg["network"]["p_out"]

    # Keep the same group sizes as the personas file.
    sizes = community_sizes(n, k)

    # probability of a link between two groups
    # same group -> p_in (high), different group -> p_out (low)
    # for 2 groups this looks like [[0.05, 0.005], [0.005, 0.05]]
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]

    # stochastic block model = random network with groups in it
    # networkx numbers the nodes 0..n-1 group by group, same order as our personas
    G = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(1_000_000)))

    # sometimes a person ends up with zero friends, then they'd never see any post
    # fix: give them one random friend from their own group
    for node in list(nx.isolates(G)):
        group = G.nodes[node]["block"]
        same_group = [v for v in G.nodes if G.nodes[v]["block"] == group and v != node]
        G.add_edge(node, int(rng.choice(same_group)))

    return G


# test this file alone with: python -m src.network
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    G = build_network(cfg, rng)

    degrees = [d for _, d in G.degree()]

    # count how many friendships are inside the same group
    same = 0
    for u, v in G.edges:
        if G.nodes[u]["block"] == G.nodes[v]["block"]:
            same += 1

    print(f"people: {G.number_of_nodes()}")
    print(f"friendships: {G.number_of_edges()}")
    print(f"avg friends per person: {np.mean(degrees):.1f}")
    print(f"min / max friends: {min(degrees)} / {max(degrees)}")
    print(f"links inside own group: {same / G.number_of_edges():.0%}")
    print(f"people with no friends: {len(list(nx.isolates(G)))}")