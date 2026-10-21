# app.py
# streamlit dashboard for the simulation
# run with: streamlit run app.py

import copy
import os

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.simulation import load_config, run_simulation
from src.metrics import compute_metrics, post_stats

st.set_page_config(page_title="Misinformation Simulation", layout="wide")
st.title("Human Behavior Modeling and Simulation")
st.caption("how fake news spreads through a network of simulated people, and what slows it down")

base_cfg = load_config()

# ---------- sidebar: all the settings ----------
with st.sidebar:
    st.header("settings")
    n_agents = st.slider("people", 100, 500, base_cfg["agents"]["n_agents"], step=50)
    steps = st.slider("rounds", 10, 50, base_cfg["steps"], step=5)
    intervention = st.selectbox("intervention", ["none", "fact_check", "downrank", "prebunk"])
    ranking = st.selectbox("feed ranking", ["engagement", "chronological"])
    p_out = st.slider("links between groups", 0.0, 0.05, base_cfg["network"]["p_out"], step=0.005,
                      help="higher = more contact between groups = weaker echo chambers")
    seed = st.number_input("seed", 0, 9999, base_cfg["seed"])
    n_llm = st.number_input("llm agents (slow!)", 0, 10, 0,
                            help="each llm decision takes a few seconds, 6 agents is about 5 min")
    run = st.button("run simulation", type="primary")

# streamlit reruns the whole script on every click,
# so the result is kept in session_state, otherwise it would disappear
if run:
    cfg = copy.deepcopy(base_cfg)
    cfg["agents"]["n_agents"] = n_agents
    cfg["agents"]["n_llm_agents"] = int(n_llm)
    cfg["steps"] = steps
    cfg["intervention"]["type"] = intervention
    cfg["feed"]["ranking"] = ranking
    cfg["network"]["p_out"] = p_out
    cfg["seed"] = int(seed)

    with st.spinner("running the simulation..."):
        st.session_state["result"] = run_simulation(cfg)
        st.session_state["cfg"] = cfg

tab_sim, tab_exp = st.tabs(["simulation", "experiment results"])

# ---------- tab 1: one simulation run ----------
with tab_sim:
    if "result" not in st.session_state:
        st.info("pick your settings on the left and click **run simulation**")
    else:
        result = st.session_state["result"]
        cfg = st.session_state["cfg"]
        events = result["events"]
        m = compute_metrics(result)

        # the main numbers on top
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("shares per fake post", f"{m['fake_shares']:.1f}")
        c2.metric("shares per true post", f"{m['true_shares']:.1f}")
        c3.metric("people believing fake claims", f"{m['believers']:.0%}")
        c4.metric("polarization", f"{m['polarization']:.2f}")

        left, right = st.columns(2)

        # left chart: total shares over time
        shares = events[events.action == "share"]
        fig = go.Figure()
        for is_false, name, color in [(False, "true posts", "#1f77b4"), (True, "fake posts", "#d62728")]:
            per_round = shares[shares.is_false == is_false].groupby("step").size()
            per_round = per_round.reindex(range(1, cfg["steps"] + 1), fill_value=0)
            fig.add_trace(go.Scatter(x=per_round.index, y=per_round.cumsum(), name=name,
                                     line=dict(color=color, width=3)))
        fig.update_layout(title="total shares over time", xaxis_title="round",
                          yaxis_title="shares so far", height=380)
        left.plotly_chart(fig)

        # right chart: belief per group
        beliefs = result["beliefs"]
        fig = go.Figure()
        for c, g in beliefs.groupby("community"):
            fig.add_trace(go.Scatter(x=g.step, y=g.avg_belief, name=f"group {c}", line=dict(width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="belief in fake claims", xaxis_title="round",
                          yaxis_title="avg belief", height=380)
        right.plotly_chart(fig)

        # the network itself, colored by what people believe at the end
        st.subheader("the network")
        st.caption("red = believes the fake claims, blue = rejects them, big dots with a black ring = llm agents")

        G = result["graph"]
        personas = result["personas"]
        pos = nx.spring_layout(G, seed=1)  # fixed seed so the picture doesn't jump around

        # all edges in one trace, None breaks the line between two edges
        edge_x, edge_y = [], []
        for u, v in G.edges:
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=0.3, color="#bbbbbb"), hoverinfo="none")

        node_trace = go.Scatter(
            x=[pos[p.id][0] for p in personas],
            y=[pos[p.id][1] for p in personas],
            mode="markers",
            marker=dict(
                size=[14 if p.is_llm else 7 for p in personas],
                color=[p.belief for p in personas],
                colorscale="RdBu_r", cmin=-1, cmax=1,
                colorbar=dict(title="belief"),
                line=dict(width=[2 if p.is_llm else 0 for p in personas], color="black"),
            ),
            text=[f"person {p.id}, group {p.community}<br>belief {p.belief:+.2f}"
                  f"<br>media literacy {p.media_literacy:.2f}" for p in personas],
            hoverinfo="text",
        )

        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(showlegend=False, height=550, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig)

        # which posts spread the most
        st.subheader("posts")
        stats = post_stats(events).sort_values("shares", ascending=False)
        stats["type"] = stats.is_false.map({True: "FAKE", False: "TRUE"})
        stats["text"] = [result["posts"][i].text for i in stats.post]
        st.dataframe(stats[["type", "text", "shares", "reach", "depth", "peak_round"]], hide_index=True)

        # what the llm agents said (only shows up if there were llm agents)
        llm_views = events[(events.action == "view") & (events.is_llm)]
        if len(llm_views) > 0:
            st.subheader("what the llm agents were thinking")
            st.caption(f"llm stats: {result['llm_stats']}")
            shared = set(zip(shares.agent, shares.post))
            table = pd.DataFrame({
                "person": llm_views.agent.values,
                "type": ["FAKE" if f else "TRUE" for f in llm_views.is_false],
                "post": [result["posts"][i].text for i in llm_views.post],
                "shared": [(a, p) in shared for a, p in zip(llm_views.agent, llm_views.post)],
                "reason": llm_views.reason.values,
            })
            st.dataframe(table, hide_index=True)

# ---------- tab 2: results from experiment.py ----------
with tab_exp:
    path = "results/experiment_summary.csv"
    if not os.path.exists(path):
        st.info("no results yet, run `python -m src.experiment` first")
    else:
        summary = pd.read_csv(path)
        st.write("every scenario was run with 10 different seeds, the error bars are 95% confidence intervals")

        metric = st.selectbox("metric", ["fake_shares", "fake_reach", "believers", "fake_depth", "polarization"])
        colors = ["gray" if s == "baseline" else "#1f77b4" for s in summary.scenario]

        fig = go.Figure(go.Bar(
            x=summary.scenario, y=summary[metric],
            error_y=dict(type="data", array=summary[metric + "_ci"]),
            marker_color=colors,
        ))
        fig.update_layout(title=metric, height=420)
        st.plotly_chart(fig)

        st.dataframe(summary.round(3), hide_index=True)