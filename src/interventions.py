# interventions.py
# things a platform (or society) can do to slow down fake news
# we run the same simulation with each one and compare the results later
#
#   none       -> do nothing, this is our baseline
#   fact_check -> after a few rounds, fake posts get a warning label people can see
#   downrank   -> after a few rounds, fake posts get pushed to the bottom of feeds (no label)
#   prebunk    -> before anything starts, some people get trained to spot fake news


class Intervention:

    def __init__(self, cfg):
        settings = cfg["intervention"]
        self.type = settings["type"]
        self.delay = settings["fact_check_delay"]
        self.prebunk_fraction = settings["prebunk_fraction"]
        self.prebunk_boost = settings["prebunk_boost"]

        # post ids that feed.py should push to the bottom
        self.downranked = set()

        if self.type not in ["none", "fact_check", "downrank", "prebunk"]:
            raise ValueError(f"unknown intervention type: {self.type}")

    def at_start(self, personas, rng):
        # only prebunking does something before round 0
        if self.type != "prebunk":
            return

        # pick a random group of people and raise their media literacy
        n = int(len(personas) * self.prebunk_fraction)
        chosen = rng.choice(len(personas), size=n, replace=False)
        for i in chosen:
            p = personas[i]
            p.media_literacy = min(p.media_literacy + self.prebunk_boost, 1.0)

    def each_round(self, step, posts):
        # fact checkers need some time to check a post, that's the delay
        # i'm assuming they catch every fake post, real fact checkers aren't that good,
        # so these results are kind of a best case
        if step < self.delay:
            return

        for post in posts:
            if not post.is_false:
                continue
            if self.type == "fact_check":
                post.labeled = True  # people will see the warning
            elif self.type == "downrank":
                self.downranked.add(post.id)  # people won't see a label, just less of it


# test with: python -m src.interventions
if __name__ == "__main__":
    import yaml
    import numpy as np
    from src.personas import create_personas
    from src.content import create_posts

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # try prebunking
    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)
    before = np.mean([p.media_literacy for p in people])

    cfg["intervention"]["type"] = "prebunk"
    Intervention(cfg).at_start(people, rng)
    after = np.mean([p.media_literacy for p in people])
    print(f"prebunk: avg media literacy {before:.2f} -> {after:.2f}")

    # try fact checking, labels should only show up from round 5 onwards
    posts = create_posts(cfg, people, rng)
    cfg["intervention"]["type"] = "fact_check"
    fc = Intervention(cfg)
    for step in [3, 4, 5]:
        fc.each_round(step, posts)
        n_labeled = sum(p.labeled for p in posts)
        print(f"fact_check round {step}: {n_labeled} posts labeled")

    # try downranking
    cfg["intervention"]["type"] = "downrank"
    dr = Intervention(cfg)
    dr.each_round(5, posts)
    print(f"downrank: {len(dr.downranked)} posts pushed down")