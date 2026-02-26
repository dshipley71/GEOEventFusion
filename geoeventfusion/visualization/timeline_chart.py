"""Coverage volume timeline chart for GEOEventFusion.

Renders the GDELT-native TimelineVolInfo dual-axis chart with spike markers
and optional phase boundary annotations.
Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def render_timeline_chart(
    timeline_volinfo: List[Any],
    timeline_volraw: Optional[List[Any]] = None,
    spikes: Optional[List[Any]] = None,
    phase_boundaries: Optional[List[str]] = None,
    title: str = "Coverage Volume Timeline",
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> Optional[str | Path]:
    """Render the event coverage volume timeline with spike annotations.

    Creates a dual-axis chart:
    - Primary y-axis: Normalized GDELT coverage volume (TimelineVolInfo)
    - Secondary y-axis: Absolute article counts (TimelineVolRaw), if provided
    - Spike markers shown as vertical amber lines
    - Phase boundaries shown as vertical dashed lines

    Args:
        timeline_volinfo: List of TimelineStep objects from GDELT TimelineVolInfo.
        timeline_volraw: Optional list of TimelineStepRaw for dual-axis display.
        spikes: Optional list of SpikeWindow objects for spike markers.
        phase_boundaries: Optional list of ISO date strings for phase annotations.
        title: Chart title string.
        output_path: If provided, save the chart to this path. Returns path on success.
        figsize: Figure size tuple (width, height) in inches.

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        from geoeventfusion.visualization.theme import (
            THEME_ACCENT,
            THEME_SPIKE,
            THEME_BACKGROUND,
            THEME_TEXT,
            THEME_SECONDARY,
            THEME_BORDER,
            apply_dark_theme,
            get_dark_rcparams,
        )

        plt.rcParams.update(get_dark_rcparams())

        if not timeline_volinfo:
            logger.warning("Timeline chart: empty timeline_volinfo — skipping")
            return None

        # Parse dates and values
        dates = []
        values = []
        for step in timeline_volinfo:
            try:
                d = datetime.strptime(step.date[:10], "%Y-%m-%d")
                dates.append(d)
                values.append(step.value)
            except (ValueError, AttributeError):
                continue

        if not dates:
            return None

        fig, ax1 = plt.subplots(figsize=figsize)
        apply_dark_theme(ax1)

        # Primary axis: normalized volume
        ax1.fill_between(dates, values, alpha=0.3, color=THEME_ACCENT)
        ax1.plot(dates, values, color=THEME_ACCENT, linewidth=1.5, label="Coverage Volume %")
        ax1.set_xlabel("Date", color=THEME_TEXT)
        ax1.set_ylabel("Coverage Volume (%)", color=THEME_ACCENT)
        ax1.tick_params(axis="y", labelcolor=THEME_ACCENT)

        # Secondary axis: absolute counts
        if timeline_volraw:
            ax2 = ax1.twinx()
            raw_dates = []
            raw_values = []
            for step in timeline_volraw:
                try:
                    d = datetime.strptime(step.date[:10], "%Y-%m-%d")
                    raw_dates.append(d)
                    raw_values.append(step.volume)
                except (ValueError, AttributeError):
                    continue
            if raw_dates:
                ax2.plot(
                    raw_dates, raw_values,
                    color=THEME_SECONDARY,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.6,
                    label="Absolute Count",
                )
                ax2.set_ylabel("Article Count (raw)", color=THEME_SECONDARY)
                ax2.tick_params(axis="y", labelcolor=THEME_SECONDARY)
                ax2.spines["right"].set_color(THEME_BORDER)

        # Spike markers
        if spikes:
            for spike in spikes[:10]:
                try:
                    sd = datetime.strptime(spike.date[:10], "%Y-%m-%d")
                    ax1.axvline(
                        sd,
                        color=THEME_SPIKE,
                        linewidth=1.5,
                        alpha=0.7,
                        linestyle=":",
                    )
                    # Label top-3 spikes
                    if spike.rank <= 3:
                        ax1.annotate(
                            f"Z={spike.z_score:.1f}",
                            xy=(sd, max(values) * 0.95),
                            fontsize=7,
                            color=THEME_SPIKE,
                            ha="center",
                        )
                except (ValueError, AttributeError):
                    continue

        # Phase boundary lines
        if phase_boundaries:
            for pb_date in phase_boundaries:
                try:
                    pd_dt = datetime.strptime(pb_date[:10], "%Y-%m-%d")
                    ax1.axvline(pd_dt, color=THEME_TEXT, linewidth=1.0, linestyle="--", alpha=0.4)
                except (ValueError, AttributeError):
                    continue

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, ha="right", color=THEME_TEXT)
        ax1.set_title(title, color=THEME_TEXT, pad=12)

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc="upper left", fontsize=8)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Saved timeline chart: %s", output_path)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    except ImportError:
        logger.warning("matplotlib is required for timeline chart rendering")
        return None
    except Exception as exc:
        logger.error("Timeline chart rendering failed: %s", exc)
        return None
