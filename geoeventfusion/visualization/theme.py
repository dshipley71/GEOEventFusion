"""Shared dark-theme constants and matplotlib configuration for GEOEventFusion.

All visualization modules import theme constants from here.
Rendering only — no data transformation.
"""

from __future__ import annotations

from typing import Any

# ── Color palette ─────────────────────────────────────────────────────────────
THEME_BACKGROUND: str = "#0A0E17"    # Deep navy — chart and page background
THEME_PANEL: str = "#111827"         # Slightly lighter panel background
THEME_TEXT: str = "#E5E7EB"          # Off-white text
THEME_ACCENT: str = "#60A5FA"        # Sky-blue accent
THEME_SPIKE: str = "#F59E0B"         # Amber spike markers

# Escalation risk colors
_ESCALATION_RED: str = "#EF4444"     # High risk >= 0.70
_ESCALATION_AMBER: str = "#F59E0B"   # Medium risk >= 0.45
_ESCALATION_GREEN: str = "#10B981"   # Low risk < 0.45

# Additional chart colors
THEME_SECONDARY: str = "#8B5CF6"     # Violet — secondary series
THEME_GRID: str = "#1F2937"          # Subtle grid lines
THEME_BORDER: str = "#374151"        # Panel borders
THEME_MUTED: str = "#6B7280"         # Muted/de-emphasized text

# Actor role colors
ROLE_HUB_COLOR: str = "#EF4444"      # Red — Hub actors
ROLE_BROKER_COLOR: str = "#F59E0B"   # Amber — Broker actors
ROLE_PERIPHERAL_COLOR: str = "#6B7280"  # Gray — Peripheral actors

# Community colors (cyclic palette for up to 10 communities)
COMMUNITY_COLORS = [
    "#60A5FA",  # Blue
    "#34D399",  # Green
    "#F59E0B",  # Amber
    "#F472B6",  # Pink
    "#818CF8",  # Indigo
    "#FB923C",  # Orange
    "#22D3EE",  # Cyan
    "#A3E635",  # Lime
    "#E879F9",  # Fuchsia
    "#94A3B8",  # Slate
]


def escalation_color(risk_score: float) -> str:
    """Return the appropriate escalation color for a given risk score.

    Args:
        risk_score: Escalation risk score in [0.0, 1.0].

    Returns:
        Hex color string.
    """
    if risk_score >= 0.70:
        return _ESCALATION_RED
    if risk_score >= 0.45:
        return _ESCALATION_AMBER
    return _ESCALATION_GREEN


# Export as callable alias for backwards compatibility
ESCALATION_COLOR = escalation_color


def apply_dark_theme(ax: Any = None) -> None:
    """Apply the GEOEventFusion dark theme to the current matplotlib figure and axes.

    Args:
        ax: Matplotlib Axes object. If None, applies to current axes via plt.gca().
    """
    try:
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        fig.patch.set_facecolor(THEME_BACKGROUND)

        if ax is None:
            ax = plt.gca()

        ax.set_facecolor(THEME_PANEL)
        ax.tick_params(colors=THEME_TEXT)
        ax.xaxis.label.set_color(THEME_TEXT)
        ax.yaxis.label.set_color(THEME_TEXT)
        ax.title.set_color(THEME_TEXT)
        ax.spines["bottom"].set_color(THEME_BORDER)
        ax.spines["top"].set_color(THEME_BORDER)
        ax.spines["left"].set_color(THEME_BORDER)
        ax.spines["right"].set_color(THEME_BORDER)
        ax.grid(True, color=THEME_GRID, linewidth=0.5, alpha=0.7)
    except ImportError:
        pass  # matplotlib not available; silently skip


def get_dark_rcparams() -> dict:
    """Return a matplotlib rcParams dict for the dark theme.

    Use with:
        import matplotlib
        matplotlib.rcParams.update(get_dark_rcparams())
    """
    return {
        "figure.facecolor": THEME_BACKGROUND,
        "axes.facecolor": THEME_PANEL,
        "axes.edgecolor": THEME_BORDER,
        "axes.labelcolor": THEME_TEXT,
        "axes.grid": True,
        "grid.color": THEME_GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "text.color": THEME_TEXT,
        "xtick.color": THEME_TEXT,
        "ytick.color": THEME_TEXT,
        "legend.facecolor": THEME_PANEL,
        "legend.edgecolor": THEME_BORDER,
        "legend.labelcolor": THEME_TEXT,
        "savefig.facecolor": THEME_BACKGROUND,
        "savefig.edgecolor": THEME_BACKGROUND,
        "lines.color": THEME_ACCENT,
        "patch.facecolor": THEME_ACCENT,
    }
