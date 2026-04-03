"""Market impact curves page."""
from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from dashboard.components import heatmap, line_chart

dash.register_page(__name__, path="/impact")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dcc.Slider(0.01, 0.3, 0.01, value=0.05, id="impact-trade")),
                dbc.Col(html.Div(id="impact-output")),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="impact-curves"), md=6),
                dbc.Col(dcc.Graph(id="lambda-ts"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="amihud-heat"), md=6),
                dbc.Col(dcc.Graph(id="spread-box"), md=6),
            ]
        ),
    ],
    fluid=True,
)


@callback(Output("impact-curves", "figure"), Input("impact-trade", "value"))
def update_curves(trade):
    pr = np.linspace(0.01, 0.3, 50)
    curves = pd.DataFrame(
        {
            "participation": pr * 100,
            "linear": pr * 15,
            "sqrt": np.sqrt(pr) * 10,
            "power": pr ** 0.6 * 12,
        }
    )
    fig = line_chart(curves.melt("participation", var_name="model", value_name="impact"), x="participation", y="impact", color="model", title="Impact curves (bps)")
    return fig


@callback(Output("lambda-ts", "figure"), Input("impact-trade", "value"))
def update_lambda(_):
    t = np.arange(200)
    lam = np.abs(np.sin(t / 20)) * 1e-4
    df = pd.DataFrame({"t": t, "lambda": lam})
    return line_chart(df, x="t", y="lambda", title="Kyle's lambda")


@callback(Output("amihud-heat", "figure"), Input("impact-trade", "value"))
def update_amihud(_):
    data = np.random.rand(10, 10)
    return heatmap(data, x_labels=[f"S{i}" for i in range(10)], y_labels=[f"D{i}" for i in range(10)], title="Amihud illiquidity")


@callback(Output("spread-box", "figure"), Input("impact-trade", "value"))
def update_spread(_):
    regimes = ["Low", "Medium", "High"]
    data = pd.DataFrame({"regime": np.repeat(regimes, 100), "spread": np.concatenate([np.random.normal(5, 1, 100), np.random.normal(8, 1.5, 100), np.random.normal(12, 2, 100)])})
    fig = px.box(data, x="regime", y="spread", template="plotly_dark", title="Spread distribution (bps)")
    return fig
