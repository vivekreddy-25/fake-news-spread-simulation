# feed.py
# builds the timeline for one person each round
# you only see posts that your friends have shared (like a real social media feed)

def build_feed(agent_id, G, posts, seen, size, ranking="engagement", downranked=None, rng=None):
    # agent_id   -> whose feed we're building
    # G          -> the network from network.py
    # seen       -> post ids this person already saw, so they don't get the same post again
    # size       -> how many posts fit in the feed
    # ranking    -> "engagement" (most shared first) or "chronological" (newest first)
    # downranked -> post ids the platform wants to push down (used by an intervention later)

    friends = set(G.neighbors(agent_id))
    if downranked is None:
        downranked = set()

    candidates = []
    for post in posts:
        # skip stuff they already saw or already shared themselves
        if post.id in seen or agent_id in post.shared_by:
            continue

        # find which friends shared this post and in which round
        friend_shares = []
        for person, step in post.shared_by.items():
            if person in friends:
                friend_shares.append((step, person))

        if not friend_shares:
            continue  # none of my friends shared it, so i can't see it
        # downranked posts get way less distribution, like on real platforms
        # 70% of the time this person just never gets it in their feed
        # (we mark it as seen so it doesn't sneak back in a later round)
        if post.id in downranked and rng is not None:
            if rng.random() < 0.7:
                seen.add(post.id)
                continue

        # remember the friend who shared it most recently
        # this is the "parent" later when we measure how deep a post spread
        last_step, from_friend = max(friend_shares)
        candidates.append((post, from_friend, last_step))

    # sorting: False comes before True in python,
    # so downranked posts always end up at the bottom
    if ranking == "chronological":
        def sort_key(c):
            post, _, last_step = c
            return (post.id in downranked, -last_step)
    else:
        # engagement = most shared posts on top, that's what big platforms do
        def sort_key(c):
            post, _, last_step = c
            return (post.id in downranked, -post.shares, -last_step)

    candidates.sort(key=sort_key)

    # only return the top few, people don't scroll forever
    feed = []
    for post, from_friend, _ in candidates[:size]:
        feed.append((post, from_friend))
    return feed


# test with: python -m src.feed
if __name__ == "__main__":
    import yaml
    import numpy as np
    from src.personas import create_personas
    from src.network import build_network
    from src.content import create_posts

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)
    G = build_network(cfg, rng)
    posts = create_posts(cfg, people, rng)

    # find someone who has at least 2 posts in their feed at the start
    # (at round 0 only the authors have shared, so most feeds are still empty)
    for person in people:
        feed = build_feed(person.id, G, posts, seen=set(), size=5)
        if len(feed) >= 2:
            break

    print(f"feed of person {person.id} (group {person.community}):\n")

    for ranking in ["engagement", "chronological"]:
        print(f"--- {ranking} ---")
        feed = build_feed(person.id, G, posts, seen=set(), size=5, ranking=ranking)
        for post, from_friend in feed:
            tag = "FAKE" if post.is_false else "TRUE"
            print(f"[{tag}] shares={post.shares} from friend {from_friend}: {post.text}")
        print()