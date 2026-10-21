# Human Behavior Modeling and Simulation

A simulation of how fake news spreads through a social network of simulated people, and which interventions actually slow it down.

Most of the people are simple rule-based agents, and a few of them "think" with a local LLM (Llama 3.1 8B through Ollama) built as a LangGraph workflow. Everything runs locally on a laptop: no API keys, no cost.

![one run](images/run.png)

## Why I built this

Platforms try lots of things against misinformation, like fact-check labels, showing posts to fewer people, or teaching users to spot fake news. I wanted to see how these compare in a controlled setting where I can run the exact same scenario again and again and only change one thing.

## How it works

```
config.yaml
    |
personas -> social network -> posts (10 true, 10 fake)
                  |
       every round:
         1. intervention does its thing (labels, downranking...)
         2. people come online (random, based on their activity)
         3. they see a feed of posts their friends shared
         4. they decide: share or not? (rule brain or llm brain)
         5. their belief in the fake claims changes
         6. everything gets logged
                  |
       metrics -> experiments over 10 seeds -> plots + dashboard
```

**People:** 300 agents split into two groups (left- and right-leaning). Each one has political leaning, media literacy, how easily emotional posts get to them, and how often they're online.

**Network:** stochastic block model, so people mostly know people from their own group (5% link chance inside a group vs 0.5% across). This creates echo chambers.

**Posts:** made-up local news about a fictional city (bike lanes, tap water, a library...), so it doesn't spread any real misinformation. The fake ones are more emotional on purpose, since that's how misinformation usually looks. Agents never know which posts are fake.

**Two kinds of brains, same interface:**
- `RuleAgent`: share chance based on emotion, agreement with the post, media literacy, and labels. Fast enough for hundreds of agents.
- `LLMAgent`: a LangGraph workflow `build_prompt -> call_llm -> parse_decision`. If Ollama is down or the answer isn't valid JSON, it falls back to the rule brain so the simulation never crashes. It also gives a reason for every decision.

Both have the same `decide()` function, so the simulation doesn't care which one it's talking to.

## Results

5 scenarios x 10 seeds = 50 runs, rule agents only. ± is the 95% confidence interval.

| Scenario | Shares per fake post | People believing fake claims | vs baseline |
|---|---|---|---|
| baseline (no intervention) | 16.5 ± 2.5 | 39% ± 9% | |
| fact-check labels | 5.7 ± 1.2 | 4% ± 1% | -65% |
| downranking | 5.0 ± 0.8 | 6% ± 2% | -70% |
| prebunking | 13.6 ± 2.9 | 25% ± 8% | -17% |
| chronological feed | 20.6 ± 2.5 | 48% ± 7% | +25% |

![interventions](images/interventions.png)

What I took from this:
- **Fact-checking and downranking both work a lot**, and about equally well, even though they work differently. Labels change people's minds, while downranking just means people never see the post.
- **Prebunking helps a bit**, but the confidence intervals almost overlap, so I can't say for sure with 10 seeds.
- **Fake posts spread faster:** in the baseline they peak earlier than true posts (round 4.9 vs 5.8).
- **Echo chambers vs exposure:** when I raised the links between groups from 0.005 to 0.03, polarization dropped from about 0.10 to 0.03, but the share of people believing fake claims went up to 70%, because everyone got exposed to both groups' fake posts.

## The LLM agents

The first version of the LLM agents shared way too much:

| | Fake share rate | True share rate |
|---|---|---|
| LLM agents, first prompt | 54% | 100% |
| LLM agents, after calibration | 5% | 14% |
| Rule agents | 17% | 18% |

The reasons were things like "Not sure if it's true, but sounds plausible" and then sharing anyway. LLMs tend to say yes. I added one sentence to the prompt reminding the model that most people scroll past almost everything, and the share rate dropped about 10x. That shows how sensitive LLM-based simulations are to prompt wording.

After calibration the LLM agents actually tell true from fake (14% vs 5%), and their reasons sound human:

> "Sounds like paranoid nonsense, bike lanes aren't some kind of surveillance tool"

> "This sounds plausible, but it's a pretty dry topic"

One interesting case from testing: a low-literacy agent shared a fake post even with a fact-check label on it, because "I hate bike lanes anyway". The rule agents can't do this (a label always cuts their sharing), and it looks a lot like motivated reasoning in real people.

## Run it yourself

```bash
git clone https://github.com/vivekreddy-25/fake-news-spread-simulation.git
cd fake-news-spread-simulation
python -m venv .venv
.venv\Scripts\activate          # windows
pip install -r requirements.txt
```

For the LLM agents, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.1:8b
```

Then:

```bash
python -m src.simulation         # one run, prints a summary
python -m src.metrics            # metrics + plot in results/
python -m src.experiment         # all scenarios x 10 seeds
python -m src.llm_demo           # run with 6 llm agents (~5 min)
streamlit run app.py             # dashboard
python -m pytest -v              # tests
```

All the settings are in `config.yaml`.

## Project structure

```
src/
  personas.py       agents and their traits
  network.py        social network with two communities
  content.py        the true and fake posts
  feed.py           what each person sees (engagement or chronological)
  agents_rule.py    rule based decisions
  agents_llm.py     langgraph + ollama decisions with fallback
  interventions.py  fact-check, downrank, prebunk
  simulation.py     main loop
  metrics.py        reach, cascade depth, belief, polarization
  experiment.py     runs all scenarios over many seeds
  llm_demo.py       compares llm agents vs rule agents
app.py              streamlit dashboard
tests/              pytest tests (don't need ollama)
```

## Limitations

- The agents and their traits are made up, not calibrated against real user data.
- Only 20 posts, so the feed ranking barely matters (everyone eventually sees almost everything). That's probably why the chronological result isn't meaningful.
- Fact-checkers catch every fake post after 5 rounds, which is a best case.
- The LLM results come from small samples (around 40 decisions from 6 agents), and they change a lot with prompt wording.
- Belief only tracks the fake claims as one number per person.

## What I'd do next

- Many more posts, so feed ranking actually matters
- Compare against a classic SIR epidemic model as a baseline
- Run more LLM agents with a faster model, and more seeds for prebunking
- Calibrate the network and spread against a real dataset like FakeNewsNet