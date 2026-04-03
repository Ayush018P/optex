"""Live paper trading page with real NSE/BSE data."""
from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

from dashboard.components import line_chart
from src.data.yfinance_loader import YFinanceLoader, TOP_STOCKS

dash.register_page(__name__, path="/live")

layout = dbc.Container(
    [
        dcc.Store(id="live-price-history", data={"t": [], "mid": []}),
        dcc.Interval(id="live-interval", interval=5000, n_intervals=0, disabled=True),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Live Connection", className="mb-3"),
                                    dbc.Label("Asset (NSE/BSE)"),
                                    dcc.Dropdown(TOP_STOCKS, value=TOP_STOCKS[0], id="live-asset"),
                                    dbc.Label("Order size", className="mt-2"),
                                    dbc.Input(id="live-size", type="number", value=500, placeholder="Order size"),
                                    dbc.Label("Time horizon (sec)", className="mt-2"),
                                    dbc.Input(id="live-horizon", type="number", value=300, placeholder="Time horizon"),
                                    dbc.Button("Connect Live Feed", id="live-connect", color="success", className="mt-4 w-100"),
                                    html.Div(id="live-status", className="mt-3 text-info fw-bold text-center")
                                ]
                            )
                        )
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="live-mid"), md=12),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="live-depth"), md=6),
                                dbc.Col(dcc.Graph(id="live-is"), md=6),
                            ]
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="mt-4"
)


@callback(
    [Output("live-interval", "disabled"), Output("live-connect", "children"), Output("live-connect", "color")],
    Input("live-connect", "n_clicks"),
    State("live-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_connection(n_clicks, currently_disabled):
    # Toggle the interval on/off
    if currently_disabled:
        return False, "Disconnect Feed", "danger"
    return True, "Connect Live Feed", "success"


@callback(
    [
        Output("live-depth", "figure"), 
        Output("live-mid", "figure"), 
        Output("live-is", "figure"),
        Output("live-price-history", "data"),
        Output("live-status", "children")
    ],
    Input("live-interval", "n_intervals"),
    [
        State("live-asset", "value"),
        State("live-price-history", "data"),
        State("live-interval", "disabled")
    ],
    prevent_initial_call=True,
)
def update_live_feed(n, asset, history, disabled):
    if disabled:
        raise dash.exceptions.PreventUpdate

    # Fetch real latest price
    current_price = YFinanceLoader.get_latest_price(asset)
    
    # Update history
    history["t"].append(n * 5) # 5 seconds per interval
    history["mid"].append(current_price)
    
    # Keep only last 100 points
    if len(history["t"]) > 100:
        history["t"] = history["t"][-100:]
        history["mid"] = history["mid"][-100:]
        
    df_mid = pd.DataFrame(history)
    mid_fig = line_chart(df_mid, x="t", y="mid", title=f"Real-time {asset} Price (yfinance)")
    
    # Estimate depth dynamically based on price
    depths = YFinanceLoader.estimate_depth(current_price)
    depth_fig = px.area(x=np.arange(len(depths)), y=depths, template="plotly_dark", title=f"Estimated L2 Depth")
    depth_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    # Running IS based on slippage (mocked logic applied to real price)
    slippage = (np.random.normal(0.01, 0.05, len(history["t"]))) * (np.array(history["mid"]) * 0.0001)
    is_fig = px.line(y=np.cumsum(slippage), template="plotly_dark", title="Running IS (bps)")
    is_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    return depth_fig, mid_fig, is_fig, history, f"Last update: {current_price:.2f}"
