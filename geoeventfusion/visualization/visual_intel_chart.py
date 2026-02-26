"""Visual intelligence image grid and novelty chart for GEOEventFusion.

Renders a grid of VGKG-processed images with novelty scores and staleness flags.
Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def render_visual_intel_chart(
    visual_images: List[Any],
    title: str = "Visual Intelligence — Novel Images",
    output_path: Optional[str | Path] = None,
    max_images: int = 12,
    figsize: tuple = (14, 8),
) -> Optional[str | Path]:
    """Render a novelty-score bar chart for visual intelligence images.

    Shows top images by novelty score with staleness flags and web appearance counts.
    Images are sorted by novelty score (highest first).

    Args:
        visual_images: List of VisualImage objects.
        title: Chart title string.
        output_path: If provided, save chart to this path.
        max_images: Maximum number of images to display.
        figsize: Figure size tuple (width, height) in inches.

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt

        from geoeventfusion.visualization.theme import (
            THEME_ACCENT,
            THEME_SPIKE,
            THEME_TEXT,
            THEME_PANEL,
            _ESCALATION_RED,
            apply_dark_theme,
            get_dark_rcparams,
        )

        plt.rcParams.update(get_dark_rcparams())

        if not visual_images:
            logger.warning("Visual intel chart: no images — skipping")
            return None

        # Sort by novelty score descending
        sorted_images = sorted(
            visual_images, key=lambda img: img.novelty_score, reverse=True
        )[:max_images]

        if not sorted_images:
            return None

        labels = []
        novelty_scores = []
        colors = []
        tooltips = []

        for img in sorted_images:
            # Short label: article title or URL domain
            if img.article_title:
                label = img.article_title[:40] + "..." if len(img.article_title) > 40 else img.article_title
            else:
                label = img.url.split("/")[2][:40] if "/" in img.url else img.url[:40]
            labels.append(label)
            novelty_scores.append(img.novelty_score)
            # Color: red for stale images, accent otherwise
            colors.append(_ESCALATION_RED if img.staleness_warning else THEME_ACCENT)
            tooltips.append(
                f"Web count: {img.web_appearance_count} | "
                f"Tags: {', '.join(img.imagetags[:3])}"
            )

        fig, ax = plt.subplots(figsize=figsize)
        apply_dark_theme(ax)

        y_positions = range(len(labels))
        bars = ax.barh(
            y_positions,
            novelty_scores,
            color=colors,
            alpha=0.8,
            edgecolor="none",
        )

        # Add novelty score labels
        for bar, score, stale in zip(bars, novelty_scores, [img.staleness_warning for img in sorted_images]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}" + (" ⚠" if stale else ""),
                va="center",
                ha="left",
                fontsize=7,
                color=THEME_SPIKE if stale else THEME_TEXT,
            )

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Novelty Score (1.0 = never seen before)", color=THEME_TEXT)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, color=THEME_TEXT, pad=12)
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=THEME_ACCENT, label="Original image"),
            Patch(facecolor=_ESCALATION_RED, label="Staleness warning (EXIF > 72h)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Saved visual intel chart: %s", output_path)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    except ImportError:
        logger.warning("matplotlib is required for visual intel chart rendering")
        return None
    except Exception as exc:
        logger.error("Visual intel chart rendering failed: %s", exc)
        return None
