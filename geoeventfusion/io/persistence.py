"""JSON persistence utilities for GEOEventFusion.

Provides atomic file writes (write-to-temp-then-rename) and safe JSON
load/save operations. No business logic — file I/O only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _DataclassEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and Path objects."""

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Atomically write data to a JSON file.

    Uses a write-to-temp-then-rename strategy to prevent partial writes.
    Creates parent directories if they do not exist.

    Args:
        data: Data to serialize. Supports dicts, lists, dataclasses, and Path objects.
        path: Output file path.
        indent: JSON indentation level (default: 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serialized = json.dumps(data, indent=indent, ensure_ascii=False, cls=_DataclassEncoder)
    except (TypeError, ValueError) as exc:
        logger.error("JSON serialization failed for %s: %s", path, exc)
        raise

    # Atomic write: write to temp file in same directory, then rename
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dir_path,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(serialized)
        tmp_path = tmp.name

    try:
        os.replace(tmp_path, path)
    except OSError as exc:
        os.unlink(tmp_path)
        logger.error("Atomic rename failed for %s: %s", path, exc)
        raise

    logger.debug("Saved JSON to %s (%d bytes)", path, len(serialized))


def load_json(path: str | Path) -> Optional[Any]:
    """Load and parse a JSON file.

    Returns None if the file does not exist or cannot be parsed.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object, or None on error.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("JSON file not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return None


def file_checksum(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        SHA-256 hex digest string, or empty string if the file is unreadable.
    """
    path = Path(path)
    if not path.exists():
        return ""
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError as exc:
        logger.warning("Checksum failed for %s: %s", path, exc)
        return ""


def ensure_output_dir(base_dir: str | Path, run_id: str) -> Path:
    """Create and return the output directory for a pipeline run.

    Args:
        base_dir: Root output directory (e.g., outputs/runs).
        run_id: Pipeline run identifier (YYYYMMDD_HHMMSS_<query_slug>).

    Returns:
        Path to the created run output directory.
    """
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    return run_dir
