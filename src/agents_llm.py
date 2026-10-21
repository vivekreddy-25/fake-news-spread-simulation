# agents_llm.py
# the "agentic" part: some people make their decisions with a real language model
# runs locally with ollama, so no api key and no cost
#
# it's a small langgraph workflow:
#   build_prompt -> call_llm -> parse_decision
#   if the llm call fails or it answers garbage -> fallback to the rule based brain

import json
import time
from typing import Any, TypedDict

import requests
from langgraph.graph import StateGraph, START, END

from src.agents_rule import RuleAgent


# everything the graph passes from one step to the next
class DecisionState(TypedDict, total=False):
    persona: Any
    post: Any
    rng: Any
    prompt: str
    raw: str  # the llm's raw text answer
    decision: dict  # final result: share, belief_change, reason
    error: str  # gets filled if something goes wrong


SYSTEM_PROMPT = (
    "You are role-playing a normal person scrolling through social media. "
    "Stay in character and react the way this person realistically would. "
    "Answer with JSON only."
)


class LLMAgent:

    def __init__(self, cfg):
        llm = cfg["llm"]
        self.model = llm["model"]
        self.url = llm["url"]
        self.temperature = llm["temperature"]
        self.seed = cfg["seed"]

        self.fallback_brain = RuleAgent()  # backup if the llm doesn't work
        self.stats = {"llm_calls": 0, "fallbacks": 0}
        self.graph = self.build_graph()

    def build_graph(self):
        g = StateGraph(DecisionState)

        g.add_node("build_prompt", self.build_prompt)
        g.add_node("call_llm", self.call_llm)
        g.add_node("parse_decision", self.parse_decision)
        g.add_node("fallback", self.fallback)

        g.add_edge(START, "build_prompt")
        g.add_edge("build_prompt", "call_llm")

        # after calling the llm and after parsing, check if there was an error
        # error -> fallback, no error -> keep going
        g.add_conditional_edges("call_llm", self.route, {"ok": "parse_decision", "failed": "fallback"})
        g.add_conditional_edges("parse_decision", self.route, {"ok": END, "failed": "fallback"})
        g.add_edge("fallback", END)

        return g.compile()

    def route(self, state):
        return "failed" if state.get("error") else "ok"

    # --- the graph nodes ---
    # each one gets the state and returns only the parts it changed

    def build_prompt(self, state):
        persona = state["persona"]
        post = state["post"]

        # tell the llm how this person feels right now, so they can change over time
        if persona.belief > 0.3:
            mood = "Lately you've started believing a lot of the rumours going around your city."
        elif persona.belief < -0.3:
            mood = "Lately you've become quite sceptical of the rumours going around your city."
        else:
            mood = ""

        label = ""
        if post.labeled:
            label = '\nThe post has a warning label: "Independent fact-checkers say this claim is false."'

        # llms say "yes, share" way too often (first test: 54% fake, 100% true vs ~15-20% for rule agents)
        # real people scroll past most things, so i remind the model how people actually behave
        prompt = (
            f"{persona.describe()} {mood}\n\n"
            f'A friend of yours shared this post:\n"{post.text}"{label}\n\n'
            "Keep in mind: most people scroll past almost everything and only repost the few things "
            "they really care about, because reposting puts your name on it.\n"
            "Would you actually repost this to all your friends, or just scroll past? And do you believe it?\n"
            'Reply with JSON like this: {"share": true or false, '
            '"believe": a number from -1 (clearly fake) to 1 (clearly true), '
            '"reason": "one short sentence in first person"}'
        )
        return {"prompt": prompt}

    def call_llm(self, state):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["prompt"]},
            ],
            "stream": False,
            "format": "json",  # ollama forces the answer to be valid json
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        try:
            r = requests.post(self.url, json=payload, timeout=120)
            if r.status_code != 200:
                # show what ollama actually said, eg "model not found"
                return {"error": f"ollama {r.status_code}: {r.text[:80]}"}
            self.stats["llm_calls"] += 1
            return {"raw": r.json()["message"]["content"]}
        except Exception as e:
            return {"error": f"llm call failed: {e}"}

    def parse_decision(self, state):
        try:
            data = json.loads(state["raw"])

            share = data["share"]
            if isinstance(share, str):  # small models sometimes answer "true" as text
                share = share.strip().lower() == "true"

            believe = float(data.get("believe", 0))
            believe = max(-1.0, min(1.0, believe))  # keep it in range
            reason = str(data.get("reason", "")).strip()
        except Exception as e:
            return {"error": f"couldn't read llm answer: {e}"}

        # same rule as the rule agent: only fake posts move the belief score
        post = state["post"]
        change = 0.1 * believe if post.is_false else 0.0

        return {"decision": {
            "share": bool(share),
            "belief_change": change,
            "reason": reason or "no reason given",
        }}

    def fallback(self, state):
        self.stats["fallbacks"] += 1
        d = self.fallback_brain.decide(state["persona"], state["post"], state["rng"])
        d["reason"] = f"fallback ({state['error'][:120]})"
        return {"decision": d}

    # same function name and output as RuleAgent, so simulation.py works with both
    def decide(self, persona, post, rng):
        out = self.graph.invoke({"persona": persona, "post": post, "rng": rng})
        return out["decision"]


# test with: python -m src.agents_llm
if __name__ == "__main__":
    import numpy as np
    from src.simulation import load_config
    from src.personas import create_personas
    from src.content import create_posts

    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    people = create_personas(cfg, rng)
    posts = create_posts(cfg, people, rng)
    agent = LLMAgent(cfg)

    naive = min(people, key=lambda p: p.media_literacy)
    smart = max(people, key=lambda p: p.media_literacy)
    fake_post = [p for p in posts if p.is_false][0]
    true_post = [p for p in posts if not p.is_false][0]

    print("(first answer can take a while, ollama has to load the model)\n")

    for label, person in [("low literacy", naive), ("high literacy", smart)]:
        print(f"=== {label} ===")
        print(person.describe(), "\n")

        for post in [fake_post, true_post]:
            start = time.time()
            d = agent.decide(person, post, rng)
            tag = "FAKE" if post.is_false else "TRUE"
            print(f"[{tag}] {post.text}")
            print(f"  share={d['share']}  belief_change={d['belief_change']:+.2f}  ({time.time() - start:.1f}s)")
            print(f"  reason: {d['reason']}\n")

    # same fake post again, but now with a fact-check label
    fake_post.labeled = True
    d = agent.decide(naive, fake_post, rng)
    print("=== low literacy, fake post WITH label ===")
    print(f"  share={d['share']}  belief_change={d['belief_change']:+.2f}")
    print(f"  reason: {d['reason']}\n")

    print("stats:", agent.stats)