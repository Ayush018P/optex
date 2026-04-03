"""Reusable Plotly chart helpers."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"
ACCENT = "#00d4ff"


def line_chart(df, x, y, color=None, title=""):
    fig = px.line(df, x=x, y=y, color=color, template=DARK_TEMPLATE)
    fig.update_layout(title=title, legend_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def animated_lines(frames, title=""):
    fig = go.Figure()
    for name, data in frames.items():
        fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="lines", name=name))
    fig.update_layout(template=DARK_TEMPLATE, title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_chart(df, x, y, error_y=None, title=""):
    fig = px.bar(df, x=x, y=y, error_y=error_y, template=DARK_TEMPLATE)
    fig.update_layout(title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def heatmap(matrix: np.ndarray, x_labels, y_labels, title=""):
    fig = px.imshow(matrix, x=x_labels, y=y_labels, color_continuous_scale="Viridis", template=DARK_TEMPLATE)
    fig.update_layout(title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def stacked_bar(labels, components, title=""):
    fig = go.Figure()
    for name, vals in components.items():
        fig.add_trace(go.Bar(name=name, x=labels, y=vals))
    fig.update_layout(barmode="stack", template=DARK_TEMPLATE, title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


__all__ = ["line_chart", "animated_lines", "bar_chart", "heatmap", "stacked_bar"]
