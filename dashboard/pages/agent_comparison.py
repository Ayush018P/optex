"""Agent performance comparison page."""
from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from dashboard.components import heatmap, line_chart

dash.register_page(__name__, path="/agents")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="is-violin"), md=6),
                dbc.Col(dcc.Graph(id="win-heat"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="schedule"), md=6),
                dbc.Col(dcc.Graph(id="learning"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="tail"), md=12),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    [
        Output("is-violin", "figure"),
        Output("win-heat", "figure"),
        Output("schedule", "figure"),
        Output("learning", "figure"),
        Output("tail", "figure"),
    ],
    Input("is-violin", "id"),
)
def render(_):
    strategies = ["TWAP", "VWAP", "AC", "PPO", "SAC", "TD3"]
    data = pd.DataFrame({"strategy": np.repeat(strategies, 200), "is": np.concatenate([np.random.normal(10, 2, 200), np.random.normal(8, 2, 200), np.random.normal(7, 2, 200), np.random.normal(5, 2, 200), np.random.normal(4, 2, 200), np.random.normal(4, 2, 200)])})
    violin = px.violin(data, x="strategy", y="is", template="plotly_dark", box=True, points="all", title="IS distribution")

    win_matrix = np.random.rand(len(strategies), len(strategies))
    win_fig = heatmap(win_matrix, strategies, strategies, title="Win rate matrix")

    traj = pd.DataFrame({"time": np.arange(30)})
    for s in strategies:
        traj[s] = np.exp(-np.arange(30) / (5 + np.random.rand() * 10))
    traj_melt = traj.melt("time", var_name="strategy", value_name="inventory")
    schedule_fig = line_chart(traj_melt, x="time", y="inventory", color="strategy", title="Execution schedule")

    steps = np.arange(100)
    learning_df = pd.DataFrame({"step": steps, "ppo": np.log1p(steps) + np.random.rand(100), "sac": np.log1p(steps) + np.random.rand(100), "td3": np.log1p(steps) + np.random.rand(100)})
    learning_fig = line_chart(learning_df.melt("step", var_name="agent", value_name="reward"), x="step", y="reward", color="agent", title="Learning curves")

    tail_df = data.groupby("strategy").quantile(0.95).reset_index()
    tail_fig = px.bar(tail_df, x="strategy", y="is", template="plotly_dark", title="Worst 5% scenarios")

    return violin, win_fig, schedule_fig, learning_fig, tail_fig
