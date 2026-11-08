# This agent uses a local LLM for some people.
# If the LLM fails, it falls back to the simple rule agent.

import json
import time
from typing import Any, TypedDict

import requests
from langgraph.graph import StateGraph, START, END

from src.agents.rule import RuleAgent


# everything the graph passes from one step to the next
class DecisionState(TypedDict, total=False):
    persona: Any
    post: Any
    rng: Any
    prompt: str
    raw: str  # the llm's raw text answer
    decision: dict  # final result: share, belief_change, reason
    error: str  # gets filled if something goes wrong
    memories: list  # last N things this agent remembers doing, from the recall node
    seen_before: bool  # saw this exact post id before, just from another friend
    reflection: str  # the agent's current summary of what it believes, if any
    believe: float  # the -1..1 belief number the llm gave, used for memory


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

        # memory + reflection, off by default if not in the config at all
        self.memory_on = bool(llm.get("memory", False))
        self.memory_size = llm.get("memory_size", 5)
        self.reflect_every = llm.get("reflect_every", 5)

        self.memory = {}       # persona id -> list of past decisions, newest last
        self.reflections = {}  # persona id -> current 1-2 sentence summary
        self.decision_log = []  # every decision + the memory that went into it, for the demo csv

        self.fallback_brain = RuleAgent()  # backup if the llm doesn't work
        self.stats = {"llm_calls": 0, "fallbacks": 0, "reflections": 0, "reflection_failures": 0}
        self.graph = self.build_graph()

    def build_graph(self):
        g = StateGraph(DecisionState)

        g.add_node("recall", self.recall)
        g.add_node("build_prompt", self.build_prompt)
        g.add_node("call_llm", self.call_llm)
        g.add_node("parse_decision", self.parse_decision)
        g.add_node("fallback", self.fallback)
        g.add_node("save_memory", self.save_memory)

        g.add_edge(START, "recall")
        g.add_edge("recall", "build_prompt")
        g.add_edge("build_prompt", "call_llm")

        # after calling the llm and after parsing, check if there was an error
        # error -> fallback, no error -> keep going
        g.add_conditional_edges("call_llm", self.route, {"ok": "parse_decision", "failed": "fallback"})
        g.add_conditional_edges("parse_decision", self.route, {"ok": "save_memory", "failed": "fallback"})
        g.add_edge("fallback", "save_memory")
        g.add_edge("save_memory", END)

        return g.compile()

    def route(self, state):
        return "failed" if state.get("error") else "ok"

    # --- the graph nodes ---
    # each one gets the state and returns only the parts it changed

    def recall(self, state):
        # grabs what this agent remembers, before build_prompt writes it into the prompt.
        # when memory is off this is a no-op, so the prompt comes out exactly like before.
        if not self.memory_on:
            return {}

        persona = state["persona"]
        post = state["post"]
        mem_list = self.memory.get(persona.id, [])

        # same post id showing up again just means a different friend shared it too
        seen_before = any(m["post_id"] == post.id for m in mem_list)

        return {
            # copy it - save_memory mutates self.memory[persona.id] in place later,
            # and this same list object would change under us otherwise
            "memories": list(mem_list),
            "seen_before": seen_before,
            "reflection": self.reflections.get(persona.id),
        }

    def build_prompt(self, state):
        # Turn the person and post into a short prompt for the model.
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

        # memory + reflection, in plain english so a small local model can actually use it
        # (empty when memory is off, or when this is the agent's first decision)
        memory_text = ""
        memories = state.get("memories")
        if memories:
            lines = []
            for m in memories:
                did = "shared it" if m["shared"] else "scrolled past it"
                lines.append(f'- "{m["post_text"]}" - you {did}. you said: "{m["reason"]}"')
            memory_text += "Recently you saw:\n" + "\n".join(lines) + "\n"

        if state.get("seen_before"):
            memory_text += "You've seen this exact claim before, from a different friend.\n"

        reflection = state.get("reflection")
        if reflection:
            memory_text += f"Your current view: {reflection}\n"

        if memory_text:
            memory_text = "\n" + memory_text

        # llms say "yes, share" way too often (first test: 54% fake, 100% true vs ~15-20% for rule agents)
        # real people scroll past most things, so i remind the model how people actually behave
        prompt = (
            f"{persona.describe()} {mood}\n"
            f"{memory_text}\n"
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
        }, "believe": believe}

    def fallback(self, state):
        self.stats["fallbacks"] += 1
        d = self.fallback_brain.decide(state["persona"], state["post"], state["rng"])
        d["reason"] = f"fallback ({state['error'][:120]})"
        return {"decision": d}

    def save_memory(self, state):
        # stash this decision away so future prompts for this persona can recall it.
        # off by default so rule-only / memory-off runs don't pay for this at all.
        if not self.memory_on:
            return {}

        persona = state["persona"]
        post = state["post"]
        decision = state["decision"]

        entry = {
            "post_id": post.id,
            "post_text": post.text,
            "shared": decision["share"],
            "belief": state.get("believe", 0.0),
            "reason": decision["reason"],
            "fallback": decision["reason"].startswith("fallback"),
        }

        mem_list = self.memory.setdefault(persona.id, [])
        mem_list.append(entry)
        if len(mem_list) > self.memory_size:
            del mem_list[0]

        # keep a flat log of every decision + the memory that went into it, for the demo csv
        self.decision_log.append({
            "persona": persona.id,
            "post": post.id,
            "is_false": post.is_false,
            "share": entry["shared"],
            "belief": entry["belief"],
            "reason": entry["reason"],
            "fallback": entry["fallback"],
            "memories_used": len(state.get("memories") or []),
            "seen_before": bool(state.get("seen_before")),
            "reflection_used": state.get("reflection") or "",
        })
        return {}

    def reflect(self, persona, step):
        # once every K rounds, ask the agent to sum up what it now believes.
        # never allowed to crash the run - a failed call just keeps the old summary.
        if not self.memory_on:
            return

        mem_list = self.memory.get(persona.id, [])
        if len(mem_list) < 3:
            return

        lines = [f'- "{m["post_text"]}" ({"shared it" if m["shared"] else "scrolled past it"})'
                  for m in mem_list]
        prompt = (
            f"{persona.describe()}\n\n"
            "Here is what you've seen and done recently:\n" + "\n".join(lines) + "\n\n"
            "In 1-2 sentences, first person, say what you now believe about the rumours "
            "going around your city.\n"
            'Reply with JSON like this: {"summary": "one or two sentences in first person"}'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "seed": self.seed + step},
        }
        try:
            r = requests.post(self.url, json=payload, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"ollama {r.status_code}: {r.text[:80]}")
            data = json.loads(r.json()["message"]["content"])
            summary = str(data["summary"]).strip()
            if summary:
                self.reflections[persona.id] = summary
            self.stats["reflections"] += 1
        except Exception:
            # keep whatever reflection was there before (even if that's none at all)
            self.stats["reflection_failures"] += 1

    def maybe_reflect(self, personas, step):
        # called once per round by simulation.py. no-op unless memory is on and it's time.
        if not self.memory_on or step % self.reflect_every != 0:
            return
        for persona in personas:
            if persona.is_llm:
                self.reflect(persona, step)

    # same function name and output as RuleAgent, so simulation.py works with both
    def decide(self, persona, post, rng):
        out = self.graph.invoke({"persona": persona, "post": post, "rng": rng})
        return out["decision"]


# test with: python -m src.agents.llm
if __name__ == "__main__":
    import numpy as np
    from src.core.simulation import load_config
    from src.models.personas import create_personas
    from src.models.content import create_posts

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