# test_simulation.py
# quick checks that the main parts of the simulation behave the way i expect
# run with: python -m pytest -v

import copy

import numpy as np
import pandas as pd

from src.simulation import load_config, run_simulation
from src.agents_rule import RuleAgent
from src.agents_llm import LLMAgent
from src.content import Post
from src.personas import Persona
from src.interventions import Intervention
from src.metrics import cascade_depths, post_stats


def small_config():
    # tiny version of the config so the tests run in a couple of seconds
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["agents"]["n_agents"] = 60
    cfg["agents"]["n_llm_agents"] = 0
    cfg["steps"] = 5
    return cfg


def make_person(literacy):
    return Persona(id=0, community=0, age=30, leaning=0.5,
                   media_literacy=literacy, susceptibility=0.5, activity=1.0)


def make_fake_post():
    return Post(id=0, text="some fake claim", is_false=True,
                stance=0.5, emotionality=0.8, author=1)


def test_same_seed_gives_same_result():
    # if this breaks, none of the experiment results can be trusted
    cfg = small_config()
    a = run_simulation(cfg)["events"]
    b = run_simulation(cfg)["events"]
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_result():
    cfg = small_config()
    a = run_simulation(cfg)["events"]
    cfg["seed"] = cfg["seed"] + 1
    b = run_simulation(cfg)["events"]
    assert not a.equals(b)


def test_media_literacy_lowers_fake_sharing():
    brain = RuleAgent()
    post = make_fake_post()
    naive = brain.share_probability(make_person(0.0), post)
    smart = brain.share_probability(make_person(1.0), post)
    assert smart < naive


def test_label_lowers_sharing():
    brain = RuleAgent()
    person = make_person(0.3)
    post = make_fake_post()
    before = brain.share_probability(person, post)
    post.labeled = True
    after = brain.share_probability(person, post)
    assert after < before


def test_fact_check_waits_for_the_delay():
    cfg = small_config()
    cfg["intervention"]["type"] = "fact_check"
    cfg["intervention"]["fact_check_delay"] = 3
    post = make_fake_post()
    fc = Intervention(cfg)

    fc.each_round(2, [post])
    assert not post.labeled  # too early

    fc.each_round(3, [post])
    assert post.labeled


def test_llm_agent_falls_back_when_ollama_is_not_there():
    # point it to a port where nothing is running, it should not crash,
    # the rule brain should take over instead
    cfg = small_config()
    cfg["llm"]["url"] = "http://localhost:1/api/chat"
    agent = LLMAgent(cfg)

    d = agent.decide(make_person(0.5), make_fake_post(), np.random.default_rng(0))

    assert d["reason"].startswith("fallback")
    assert agent.stats["fallbacks"] == 1


def test_llm_agent_handles_broken_json():
    agent = LLMAgent(small_config())
    out = agent.parse_decision({"raw": "this is not json", "post": make_fake_post()})
    assert "error" in out


def test_cascade_depth_on_hand_made_log():
    # person 1 posts -> person 2 shares from 1 -> person 3 shares from 2
    # so the longest chain is 2
    events = pd.DataFrame([
        {"step": 0, "agent": 1, "post": 0, "parent": None, "action": "post", "is_false": True},
        {"step": 1, "agent": 2, "post": 0, "parent": 1, "action": "view", "is_false": True},
        {"step": 1, "agent": 2, "post": 0, "parent": 1, "action": "share", "is_false": True},
        {"step": 2, "agent": 3, "post": 0, "parent": 2, "action": "view", "is_false": True},
        {"step": 2, "agent": 3, "post": 0, "parent": 2, "action": "share", "is_false": True},
    ])
    assert cascade_depths(events)[0] == 2

    stats = post_stats(events).iloc[0]
    assert stats.shares == 2
    assert stats.reach == 3  # author + 2 viewers