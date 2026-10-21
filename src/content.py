# content.py
# the posts that get spread around the network
# half of them are true, half are fake, the agents don't know which is which

from dataclasses import dataclass, field


# made up local news so it doesn't touch any real topic
# each one is (text, stance, emotionality)
# stance: which side it appeals to, negative = left, positive = right
# emotionality: how dramatic it sounds (0 = boring, 1 = super dramatic)
TRUE_POSTS = [
    ("City council approves new bike lanes in the city center", -0.4, 0.2),
    ("Local hospital extends weekend opening hours after hiring more staff", -0.1, 0.2),
    ("New report finds the city's tap water meets all safety standards", 0.0, 0.1),
    ("Small business tax relief program opens applications next month", 0.4, 0.2),
    ("Police report a small drop in burglaries compared to last year", 0.3, 0.2),
    ("Public library adds free coding classes for teenagers", -0.3, 0.1),
    ("Regional farmers report a good harvest despite the dry summer", 0.2, 0.2),
    ("Airport train will run every 10 minutes starting January", -0.2, 0.1),
    ("City budget shows road repair spending went up by 5%", 0.3, 0.2),
    ("University opens a new renewable energy research center", -0.4, 0.3),
]

# fake ones are more dramatic on purpose, that's how real misinformation usually looks
FALSE_POSTS = [
    ("SHOCKING: the new bike lanes were built to secretly track everyone's movements", 0.6, 0.9),
    ("Hospital is closing the emergency room next week and nobody is talking about it!!", 0.4, 0.8),
    ("Tap water in the city has chemicals that make people tired and obedient", 0.5, 0.9),
    ("LEAKED: mayor plans to double property taxes for families next year", 0.7, 0.8),
    ("Big companies paid for the bike lanes to get tax breaks, share before it gets deleted", -0.6, 0.8),
    ("Farmers dump pesticides in the river at night and the city just ignores it", -0.5, 0.9),
    ("Airport train only got approved because councillors own the land next to it", 0.3, 0.7),
    ("Library coding classes are just a way to collect kids' data for advertisers", 0.4, 0.8),
    ("Police were told to stop recording burglaries so the numbers look better", 0.6, 0.8),
    ("Rich landlords are behind the rent increase and the media is covering it up", -0.6, 0.8),
]


@dataclass
class Post:
    id: int
    text: str
    is_false: bool  # the truth, only we know this, agents never see it
    stance: float
    emotionality: float
    author: int  # id of the person who posted it first
    labeled: bool = False  # gets set to True when a fact-check label is added
    shared_by: dict = field(default_factory=dict)  # person id -> round they shared it

    @property
    def shares(self):
        return len(self.shared_by)


def pick_author(stance, personas, rng):
    # a left-leaning post should be posted by someone from the left group and so on
    # with 2 groups: stance < 0 -> group 0, stance >= 0 -> group 1
    groups = sorted(set(p.community for p in personas))
    if len(groups) == 1:
        candidates = personas
    else:
        target = groups[0] if stance < 0 else groups[-1]
        candidates = [p for p in personas if p.community == target]
    return int(rng.choice([p.id for p in candidates]))


def create_posts(cfg, personas, rng):
    n_true = cfg["content"]["n_true"]
    n_false = cfg["content"]["n_false"]

    posts = []
    post_id = 0

    # if the config asks for more posts than we have texts, we just loop over the list again
    for i in range(n_true):
        text, stance, emo = TRUE_POSTS[i % len(TRUE_POSTS)]
        author = pick_author(stance, personas, rng)
        post = Post(post_id, text, False, stance, emo, author)
        post.shared_by[author] = 0  # the author "shares" it at round 0
        posts.append(post)
        post_id += 1

    for i in range(n_false):
        text, stance, emo = FALSE_POSTS[i % len(FALSE_POSTS)]
        author = pick_author(stance, personas, rng)
        post = Post(post_id, text, True, stance, emo, author)
        post.shared_by[author] = 0
        posts.append(post)
        post_id += 1

    return posts


# test with: python -m src.content
if __name__ == "__main__":
    import yaml
    import numpy as np
    from src.personas import create_personas

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)
    posts = create_posts(cfg, people, rng)

    print(f"created {len(posts)} posts\n")
    for post in posts:
        tag = "FAKE" if post.is_false else "TRUE"
        group = people[post.author].community
        print(f"[{tag}] id={post.id} author={post.author} (group {group}) "
              f"stance={post.stance} emo={post.emotionality}")
        print(f"   {post.text}")