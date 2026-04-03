"""Dash multi-page app entry point."""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from dashboard.components import navbar

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container(
    [
        dcc.Store(id="sim-store"),
        navbar(),
        dash.page_container,
    ],
    fluid=True,
    style={"backgroundColor": "#0d1117", "color": "white"},
)


@server.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
