"""Language coverage stacked area chart for GEOEventFusion.

Renders top-language coverage volume over time from GDELT TimelineLang data.
Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def render_language_chart(
    timeline_lang: List[Any],
    top_n: int = 8,
    title: str = "Coverage by Language",
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> Optional[str | Path]:
    """Render a stacked area chart of coverage volume by language.

    Args:
        timeline_lang: List of TimelineStep objects from GDELT TimelineLang mode.
            Each step's label is a language name and value is the coverage volume.
        top_n: Number of top languages to display.
        title: Chart title string.
        output_path: If provided, save the chart to this path.
        figsize: Figure size tuple (width, height) in inches.

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        from datetime import datetime

        from geoeventfusion.visualization.theme import (
            COMMUNITY_COLORS,
            THEME_TEXT,
            apply_dark_theme,
            get_dark_rcparams,
        )

        plt.rcParams.update(get_dark_rcparams())

        if not timeline_lang:
            logger.warning("Language chart: empty timeline_lang — skipping")
            return None

        # Aggregate by date and language
        date_lang_map: dict = defaultdict(lambda: defaultdict(float))
        all_dates_raw = set()
        for step in timeline_lang:
            if step.label:
                date_lang_map[step.date][step.label] += step.value
                all_dates_raw.add(step.date)

        if not date_lang_map:
            return None

        # Find top-N languages by total volume
        lang_totals: dict = defaultdict(float)
        for lang_map in date_lang_map.values():
            for lang, vol in lang_map.items():
                lang_totals[lang] += vol
        top_langs = sorted(lang_totals.keys(), key=lambda l: lang_totals[l], reverse=True)[:top_n]

        # Sort dates
        sorted_dates_str = sorted(all_dates_raw)
        try:
            sorted_dates = [datetime.strptime(d[:10], "%Y-%m-%d") for d in sorted_dates_str]
        except ValueError:
            sorted_dates = list(range(len(sorted_dates_str)))  # type: ignore[assignment]

        # Build stacked data
        stacked = {lang: [] for lang in top_langs}
        for d_str in sorted_dates_str:
            for lang in top_langs:
                stacked[lang].append(date_lang_map[d_str].get(lang, 0.0))

        fig, ax = plt.subplots(figsize=figsize)
        apply_dark_theme(ax)

        # Stacked area plot
        y_stack = np.zeros(len(sorted_dates))
        for i, lang in enumerate(top_langs):
            y_values = np.array(stacked[lang])
            color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
            ax.fill_between(
                sorted_dates,
                y_stack,
                y_stack + y_values,
                label=lang,
                alpha=0.7,
                color=color,
            )
            y_stack += y_values

        if isinstance(sorted_dates[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45, ha="right", color=THEME_TEXT)

        ax.set_xlabel("Date", color=THEME_TEXT)
        ax.set_ylabel("Coverage Volume", color=THEME_TEXT)
        ax.set_title(title, color=THEME_TEXT, pad=12)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Saved language chart: %s", output_path)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    except ImportError:
        logger.warning("matplotlib/numpy is required for language chart rendering")
        return None
    except Exception as exc:
        logger.error("Language chart rendering failed: %s", exc)
        return None
