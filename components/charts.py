"""
Plotly Charting Components for FIREGUARD AI SOC Dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any
import config

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#151E32",
    plot_bgcolor="#151E32",
    font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
    margin=dict(l=30, r=30, t=40, b=30),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#94A3B8")
    )
)

def create_distribution_pie(records: List[Dict[str, Any]]) -> go.Figure:
    """Creates a donut chart of classified CCTV images."""
    if not records:
        fig = go.Figure()
        fig.add_annotation(
            text="No detection records available yet.<br>Analyze CCTV snapshots to generate analytics.",
            showarrow=False,
            font=dict(size=14, color="#94A3B8")
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        return fig

    df = pd.DataFrame(records)
    counts = df["predicted_class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    color_map = {
        "Fire": "#EF4444",
        "Smoke": "#F59E0B",
        "Normal": "#10B981",
        "Non-Fire": "#10B981"
    }
    colors = [color_map.get(c, "#38BDF8") for c in counts["Class"]]

    fig = go.Figure(data=[
        go.Pie(
            labels=counts["Class"],
            values=counts["Count"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#151E32", width=2)),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )
    ])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Class Distribution Breakdown", font=dict(size=14, color="#F8FAFC")),
        height=280
    )
    return fig


def create_activity_timeline(records: List[Dict[str, Any]]) -> go.Figure:
    """Creates a time-series area/bar chart of detection incidents."""
    if not records or len(records) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Accumulating time-series activity data...",
            showarrow=False,
            font=dict(size=14, color="#94A3B8")
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        return fig

    df = pd.DataFrame(records)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df["minute_bin"] = df["dt"].dt.strftime("%H:%M")
    
    grouped = df.groupby(["minute_bin", "predicted_class"]).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    color_map = {"Fire": "#EF4444", "Smoke": "#F59E0B", "Normal": "#10B981", "Non-Fire": "#10B981"}

    for col in grouped.columns:
        if col == "minute_bin":
            continue
        fig.add_trace(go.Bar(
            x=grouped["minute_bin"],
            y=grouped[col],
            name=col,
            marker_color=color_map.get(col, "#38BDF8")
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="Incident Activity Timeline", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(gridcolor="#1E2A47", showgrid=True, title="Time"),
        yaxis=dict(gridcolor="#1E2A47", showgrid=True, title="Detections"),
        height=280
    )
    return fig


def create_confidence_comparison_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """Creates an average confidence bar chart per class."""
    if not records:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence telemetry recorded yet.",
            showarrow=False,
            font=dict(size=14, color="#94A3B8")
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        return fig

    df = pd.DataFrame(records)
    avg_conf = df.groupby("predicted_class")["confidence"].mean().reset_index()
    avg_conf["confidence_pct"] = avg_conf["confidence"] * 100

    color_map = {"Fire": "#EF4444", "Smoke": "#F59E0B", "Normal": "#10B981", "Non-Fire": "#10B981"}
    colors = [color_map.get(c, "#38BDF8") for c in avg_conf["predicted_class"]]

    fig = go.Figure(data=[
        go.Bar(
            x=avg_conf["predicted_class"],
            y=avg_conf["confidence_pct"],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in avg_conf["confidence_pct"]],
            textposition="auto"
        )
    ])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Mean Model Confidence By Class", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(gridcolor="#1E2A47", title="Class"),
        yaxis=dict(gridcolor="#1E2A47", title="Confidence (%)", range=[0, 105]),
        height=280
    )
    return fig
