"""Home page."""
from __future__ import annotations

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.H2("OPTEX — Optimal Execution & Market Impact Intelligence"),
        html.P(
            "Simulate, compare, and deploy optimal execution strategies using reinforcement learning and classical models."
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H4("RL Agents"), html.P("PPO, SAC, TD3 trained on microstructure.")])
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H4("Baselines"), html.P("TWAP, VWAP, Almgren-Chriss trajectories.")])
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Market Impact"),
                                html.P("Compare linear, square-root, power-law impact curves."),
                            ]
                        )
                    ),
                    md=4,
                ),
            ],
            className="mb-4",
        ),
        dcc.Markdown(
            """
            ### What you can do
            - Run interactive execution simulations
            - Inspect impact curves and liquidity regimes
            - Compare agents across scenarios and tail risk
            - Paper trade live using Binance data
            """
        ),
    ],
    fluid=True,
)
