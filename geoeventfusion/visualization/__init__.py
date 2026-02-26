"""GEOEventFusion visualization package.

Rendering functions only — no data transformation or business logic.
All dark-theme constants are shared from visualization/theme.py.
"""

from geoeventfusion.visualization.theme import (
    THEME_BACKGROUND,
    THEME_PANEL,
    THEME_TEXT,
    THEME_ACCENT,
    THEME_SPIKE,
    ESCALATION_COLOR,
    apply_dark_theme,
)

__all__ = [
    "THEME_BACKGROUND",
    "THEME_PANEL",
    "THEME_TEXT",
    "THEME_ACCENT",
    "THEME_SPIKE",
    "ESCALATION_COLOR",
    "apply_dark_theme",
]
