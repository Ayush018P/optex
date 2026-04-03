"""Regime analysis page."""
from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from dashboard.components import heatmap, line_chart

dash.register_page(__name__, path="/regimes")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="regime-price"), md=12),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="regime-heat"), md=6),
                dbc.Col(dcc.Graph(id="trans-mat"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="rv-dist"), md=12),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    [Output("regime-price", "figure"), Output("regime-heat", "figure"), Output("trans-mat", "figure"), Output("rv-dist", "figure")],
    Input("regime-price", "id"),
)
def render(_):
    t = np.arange(200)
    price = 100 + np.cumsum(np.random.normal(0, 1, 200))
    regimes = np.random.choice(["Low", "Med", "High"], size=200, p=[0.4, 0.4, 0.2])
    df = pd.DataFrame({"t": t, "price": price, "regime": regimes})
    price_fig = px.line(df, x="t", y="price", color="regime", template="plotly_dark", title="Regime detection")

    strategies = ["TWAP", "PPO", "SAC", "TD3"]
    regime_labels = ["Low", "Med", "High"]
    # Corret dimension: rows = len(regime_labels), cols = len(strategies)
    heat = np.random.rand(len(regime_labels), len(strategies))
    heat_fig = heatmap(heat, strategies, regime_labels, title="IS by regime")

    trans = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    trans_fig = heatmap(trans, ["Low", "Med", "High"], ["Low", "Med", "High"], title="Transition matrix")

    rv = pd.DataFrame({"regime": regimes, "rv": np.abs(np.random.normal(0, 1, 200))})
    rv_fig = px.violin(rv, x="regime", y="rv", template="plotly_dark", title="Realized vol per regime")
    return price_fig, heat_fig, trans_fig, rv_fig
