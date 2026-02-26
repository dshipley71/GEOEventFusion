"""Logging utilities for GEOEventFusion.

Provides structured logging with run_id context injection and YAML-based
configuration loading. All loggers are namespaced under 'geoeventfusion'.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional

import yaml


def configure_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging from the YAML configuration file.

    Falls back to basicConfig if the YAML file is not found.

    Args:
        config_path: Path to logging.yaml (defaults to config/logging.yaml).
        log_level: Override log level (e.g., "DEBUG", "INFO", "WARNING").
        log_file: Override the log file path.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "config" / "logging.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Override log file if specified
        if log_file and "handlers" in cfg:
            for handler_cfg in cfg["handlers"].values():
                if handler_cfg.get("class") == "logging.FileHandler":
                    handler_cfg["filename"] = log_file

        # Override log level if specified
        if log_level and "loggers" in cfg:
            for logger_cfg in cfg["loggers"].values():
                logger_cfg["level"] = log_level.upper()
            if "root" in cfg:
                cfg["root"]["level"] = log_level.upper()

        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=getattr(logging, (log_level or "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger under 'geoeventfusion'.

    Args:
        name: Module or component name (e.g., "agents.gdelt_agent").

    Returns:
        Logger instance with full 'geoeventfusion.<name>' namespace.
    """
    if name.startswith("geoeventfusion"):
        return logging.getLogger(name)
    return logging.getLogger(f"geoeventfusion.{name}")


class RunContextAdapter(logging.LoggerAdapter):
    """Logger adapter that injects run_id into all log records.

    Usage:
        logger = get_run_logger("agents.gdelt_agent", run_id="20240115_120000_houthi")
        logger.info("Fetching GDELT data")
        # Output: [INFO] [20240115_120000_houthi] geoeventfusion.agents.gdelt_agent: Fetching GDELT data
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        run_id = self.extra.get("run_id", "unknown")
        return f"[{run_id}] {msg}", kwargs


def get_run_logger(name: str, run_id: str) -> RunContextAdapter:
    """Get a run-context-aware logger adapter.

    Args:
        name: Module or component name.
        run_id: Pipeline run identifier (YYYYMMDD_HHMMSS_<query_slug>).

    Returns:
        LoggerAdapter that prefixes all messages with [run_id].
    """
    base_logger = get_logger(name)
    return RunContextAdapter(base_logger, {"run_id": run_id})
