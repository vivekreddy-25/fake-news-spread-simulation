# This agent makes choices with a simple set of rules.
# It is fast and works well for many people.

import numpy as np

# numbers i can play with to tune the model
BASE_SHARE = 0.12  # starting chance to share any post
EMOTION_WEIGHT = 2.0  # how much dramatic posts boost sharing
LITERACY_PENALTY = 0.5  # how much media literacy reduces sharing fake posts
LABEL_PENALTY = 0.4  # a fact-check label cuts sharing to 40%
BELIEF_STEP = 0.1  # how much one fake post can move someone's belief


def agreement(persona, post):
    # how much the post fits the person's politics
    # 1 = perfect match, 0 = complete opposite
    # leaning and stance are both between -1 and 1, so max distance is 2
    return 1 - abs(persona.leaning - post.stance) / 2


class RuleAgent:

    def share_probability(self, persona, post):
        # How likely is this person to share this post?
        p = BASE_SHARE

        # emotional people + dramatic posts = more sharing
        p *= 1 + EMOTION_WEIGHT * persona.susceptibility * post.emotionality

        # people share things they agree with way more
        # 0.5 + agreement goes from 0.5 (disagree) to 1.5 (agree)
        p *= 0.5 + agreement(persona, post)

        # media literate people notice some of the fake stuff
        # note: they don't know is_false, this just models "this looks off"
        if post.is_false:
            p *= 1 - LITERACY_PENALTY * persona.media_literacy

        # a warning label makes everyone think twice
        if post.labeled:
            p *= LABEL_PENALTY

        return min(p, 1.0)

    def belief_change(self, persona, post):
        # Fake posts can change belief. True posts do not.
        if not post.is_false:
            return 0.0

        # if there's a fact-check label, seeing it actually makes them doubt a bit
        if post.labeled:
            return -BELIEF_STEP / 2

        # low literacy + post fits their views -> believe it more
        return BELIEF_STEP * (1 - persona.media_literacy) * (0.5 + agreement(persona, post))

    def decide(self, persona, post, rng):
        # the llm agent will have the exact same decide() function later,
        # so the simulation doesn't care which kind of brain it's talking to
        p = self.share_probability(persona, post)
        return {
            "share": bool(rng.random() < p),
            "belief_change": self.belief_change(persona, post),
            "reason": f"rule based (share chance {p:.2f})",
        }


# test with: python -m src.agents_rule
if __name__ == "__main__":
    import yaml
    from src.models.personas import create_personas
    from src.models.content import create_posts

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)
    posts = create_posts(cfg, people, rng)
    brain = RuleAgent()

    # average share chance over every person and every post
    fake_probs = []
    true_probs = []
    for person in people:
        for post in posts:
            p = brain.share_probability(person, post)
            if post.is_false:
                fake_probs.append(p)
            else:
                true_probs.append(p)

    print(f"avg share chance TRUE posts: {np.mean(true_probs):.3f}")
    print(f"avg share chance FAKE posts: {np.mean(fake_probs):.3f}\n")

    # compare the most and least media literate person on the same fake post
    fake_post = [p for p in posts if p.is_false][0]
    smart = max(people, key=lambda p: p.media_literacy)
    naive = min(people, key=lambda p: p.media_literacy)

    print(f"fake post: {fake_post.text}\n")
    for label, person in [("high literacy", smart), ("low literacy", naive)]:
        p = brain.share_probability(person, fake_post)
        b = brain.belief_change(person, fake_post)
        print(f"{label} ({person.media_literacy:.2f}): share chance {p:.2f}, belief change {b:+.3f}")

    # what happens to the same post with a fact-check label on it
    fake_post.labeled = True
    p = brain.share_probability(naive, fake_post)
    print(f"\nlow literacy person, same post WITH label: share chance {p:.2f}")