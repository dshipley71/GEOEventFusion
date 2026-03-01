"""Google Colab helper utilities for GEOEventFusion.

Provides convenience functions for Colab-specific operations:
secrets loading, file downloads, and drive mounting.
These helpers gracefully degrade when run outside Colab.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check whether the current runtime is Google Colab.

    Returns:
        True if running inside Google Colab, False otherwise.
    """
    try:
        import google.colab  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


def load_colab_secrets(secret_names: list[str]) -> Dict[str, Optional[str]]:
    """Load secrets from Google Colab's userdata store.

    Falls back to environment variables when running outside Colab.

    Args:
        secret_names: List of secret names to retrieve.

    Returns:
        Dict mapping secret name to value (or None if not found).
    """
    result: Dict[str, Optional[str]] = {}
    if is_colab():
        try:
            from google.colab import userdata  # type: ignore[import]

            for name in secret_names:
                try:
                    result[name] = userdata.get(name)
                except Exception:
                    result[name] = os.getenv(name)
                    logger.debug("Colab secret '%s' not found, falling back to env var", name)
        except ImportError:
            for name in secret_names:
                result[name] = os.getenv(name)
    else:
        for name in secret_names:
            result[name] = os.getenv(name)
    return result


def download_run_outputs(run_dir: str | Path) -> None:
    """Trigger browser download of all run output files in Colab.

    Downloads each JSON, HTML, PNG, and GEXF file from the run directory.
    No-op when running outside Colab.

    Args:
        run_dir: Path to the pipeline run output directory.
    """
    if not is_colab():
        logger.debug("download_run_outputs: not in Colab, skipping")
        return

    try:
        from google.colab import files  # type: ignore[import]

        run_dir = Path(run_dir)
        download_patterns = ["*.json", "*.html", "*.png", "*.gexf"]
        for pattern in download_patterns:
            for file_path in run_dir.rglob(pattern):
                try:
                    files.download(str(file_path))
                    logger.info("Downloaded: %s", file_path.name)
                except Exception as exc:
                    logger.warning("Could not download %s: %s", file_path, exc)
    except ImportError:
        logger.warning("google.colab.files not available")


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab if not already mounted.

    No-op when running outside Colab.

    Args:
        mount_point: Drive mount point path.

    Returns:
        True if mounted successfully (or already mounted), False otherwise.
    """
    if not is_colab():
        return False

    try:
        from google.colab import drive  # type: ignore[import]

        if not Path(mount_point).exists():
            drive.mount(mount_point)
            logger.info("Google Drive mounted at %s", mount_point)
        else:
            logger.debug("Google Drive already mounted at %s", mount_point)
        return True
    except ImportError:
        logger.warning("google.colab.drive not available")
        return False
    except Exception as exc:
        logger.error("Failed to mount Google Drive: %s", exc)
        return False


# Backward-compatible alias used by quickstart notebook
download_run_artifacts = download_run_outputs


def display_html_report(html_path: str | Path) -> None:
    """Display an HTML file inline in a Colab notebook cell.

    No-op when running outside Colab.

    Args:
        html_path: Path to the HTML file to display.
    """
    if not is_colab():
        logger.debug("display_html_report: not in Colab, skipping")
        return

    try:
        from IPython.display import HTML, display  # type: ignore[import]

        html_path = Path(html_path)
        if not html_path.exists():
            logger.warning("HTML report not found: %s", html_path)
            return
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        display(HTML(content))
    except ImportError:
        logger.warning("IPython not available for HTML display")
