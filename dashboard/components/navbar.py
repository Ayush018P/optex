"""Shared navbar component."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

NAV_ITEMS = [
    ("Home", "/"),
    ("Simulator", "/simulator"),
    ("Impact Curves", "/impact"),
    ("Agent Comparison", "/agents"),
    ("Regime Analysis", "/regimes"),
    ("Live Trading", "/live"),
]


def navbar() -> dbc.Navbar:
    """Return the top navbar."""
    links = [dbc.NavItem(dbc.NavLink(text, href=href, active="exact")) for text, href in NAV_ITEMS]
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("OPTEX", className="ms-2", href="/"),
                dbc.Nav(links, pills=True, className="me-auto"),
                dbc.Badge("Live", color="success", className="me-2"),
                dbc.Button("GitHub", color="secondary", href="https://github.com", target="_blank"),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-4",
    )


__all__ = ["navbar"]
