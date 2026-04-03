"""KPI card helpers."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def metric_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    """Create a colored metric card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted"),
                html.H4(value, className=f"text-{color}"),
            ]
        ),
        className="mb-2",
    )


__all__ = ["metric_card"]
