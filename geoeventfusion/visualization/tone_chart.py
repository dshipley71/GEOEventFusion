"""Tone distribution histogram chart for GEOEventFusion.

Renders the GDELT ToneChart bin histogram with modal and mean overlays.
Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def render_tone_chart(
    tonechart: List[Any],
    tone_stats: Optional[Any] = None,
    title: str = "Tone Distribution",
    output_path: Optional[str | Path] = None,
    figsize: tuple = (10, 5),
) -> Optional[str | Path]:
    """Render a GDELT ToneChart histogram with optional mean/modal overlays.

    Args:
        tonechart: List of ToneChartBin objects.
        tone_stats: Optional ToneStats for mean/modal overlay lines.
        title: Chart title string.
        output_path: If provided, save the chart to this path.
        figsize: Figure size tuple (width, height) in inches.

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt

        from geoeventfusion.visualization.theme import (
            THEME_ACCENT,
            THEME_SPIKE,
            THEME_SECONDARY,
            THEME_TEXT,
            apply_dark_theme,
            get_dark_rcparams,
        )

        plt.rcParams.update(get_dark_rcparams())

        if not tonechart:
            logger.warning("Tone chart: empty tonechart data — skipping")
            return None

        tone_values = [b.tone_value for b in tonechart]
        counts = [b.count for b in tonechart]

        # Color bars by polarity
        colors = [
            "#EF4444" if v < 0 else "#10B981"
            for v in tone_values
        ]

        fig, ax = plt.subplots(figsize=figsize)
        apply_dark_theme(ax)

        bar_width = (max(tone_values) - min(tone_values)) / len(tone_values) * 0.9 if len(tone_values) > 1 else 0.8
        ax.bar(tone_values, counts, width=bar_width, color=colors, alpha=0.75, edgecolor="none")

        # Overlay lines for mean and modal tone
        if tone_stats:
            ax.axvline(
                tone_stats.mean_tone,
                color=THEME_ACCENT,
                linewidth=2.0,
                linestyle="-",
                label=f"Mean: {tone_stats.mean_tone:.2f}",
            )
            ax.axvline(
                tone_stats.modal_tone,
                color=THEME_SPIKE,
                linewidth=2.0,
                linestyle="--",
                label=f"Modal: {tone_stats.modal_tone:.2f}",
            )
            ax.legend(fontsize=9)

        ax.axvline(0, color=THEME_TEXT, linewidth=0.8, alpha=0.4, linestyle="-")
        ax.set_xlabel("Tone Score", color=THEME_TEXT)
        ax.set_ylabel("Article Count", color=THEME_TEXT)
        ax.set_title(title, color=THEME_TEXT, pad=12)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Saved tone chart: %s", output_path)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    except ImportError:
        logger.warning("matplotlib is required for tone chart rendering")
        return None
    except Exception as exc:
        logger.error("Tone chart rendering failed: %s", exc)
        return None
