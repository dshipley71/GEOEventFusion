#!/usr/bin/env python3
"""GEOEventFusion Batch Runner — execute multiple pipeline queries from a YAML config file.

Usage:
    python scripts/batch_run.py --config configs/batch_queries.yaml
    python scripts/batch_run.py --config configs/batch_queries.yaml --dry-run
    python scripts/batch_run.py --config configs/batch_queries.yaml --llm-backend anthropic

Config file format (YAML):
    queries:
      - query: "Houthi Red Sea attacks"
        days_back: 90
        llm_backend: ollama
      - query: "Taiwan Strait tensions"
        days_back: 60
        llm_backend: anthropic
        source_country_filter: "CH"
      - query: "Sudan conflict"
        days_back: 90
        ground_truth_sources:
          - acled
        ground_truth_country_filter:
          - Sudan

    # Global defaults (overridden per-query)
    defaults:
      llm_backend: ollama
      max_records: 250
      log_level: INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for consistent import resolution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required for batch_run.py. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from config.defaults import (  # noqa: E402
    DAYS_BACK,
    DEFAULT_LOG_LEVEL,
    MAX_RECORDS,
    OUTPUT_ROOT,
)
from config.settings import PipelineConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the batch runner."""
    parser = argparse.ArgumentParser(
        prog="batch_run",
        description="GEOEventFusion Batch Runner — run multiple pipeline queries from a YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML batch config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse config and print resolved queries without running the pipeline",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default=None,
        help="Override llm_backend for all queries (anthropic | ollama)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root directory for all runs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="Halt the batch if any individual query fails",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="Delay in seconds between consecutive pipeline runs",
    )
    return parser


# ── Config loading ────────────────────────────────────────────────────────────────

def load_batch_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML batch config file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed config dict with 'queries' list and optional 'defaults' dict.

    Raises:
        SystemExit: If the file is missing, invalid YAML, or has no queries.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error("Batch config file not found: %s", config_path)
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML config: %s", exc)
        sys.exit(1)

    if not isinstance(config, dict):
        logger.error("Batch config must be a YAML mapping, got: %s", type(config).__name__)
        sys.exit(1)

    if "queries" not in config or not config["queries"]:
        logger.error("Batch config must have a non-empty 'queries' list")
        sys.exit(1)

    return config


def _merge_query_config(
    query_entry: Dict[str, Any],
    defaults: Dict[str, Any],
    global_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge global defaults, per-query config, and CLI overrides.

    Priority (highest to lowest): global_overrides > query_entry > defaults

    Args:
        query_entry: Per-query config dict from YAML.
        defaults: Global defaults from the top-level 'defaults' section.
        global_overrides: CLI-level overrides (e.g., --llm-backend).

    Returns:
        Merged config dict ready for PipelineConfig construction.
    """
    merged: Dict[str, Any] = {}
    merged.update(defaults)
    merged.update(query_entry)
    # CLI overrides take precedence over YAML values (None values are not applied)
    for k, v in global_overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def build_pipeline_config(merged: Dict[str, Any]) -> PipelineConfig:
    """Construct a PipelineConfig from a merged config dict.

    Only passes keys that PipelineConfig accepts; unknown keys are silently ignored.

    Args:
        merged: Merged config dict.

    Returns:
        PipelineConfig instance.
    """
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}
    return PipelineConfig(**filtered)


# ── Batch execution ───────────────────────────────────────────────────────────────

def run_batch(
    configs: List[PipelineConfig],
    stop_on_error: bool = False,
    delay_seconds: float = 5.0,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Execute a list of PipelineConfig objects sequentially.

    Args:
        configs: List of resolved PipelineConfig objects.
        stop_on_error: If True, halt on first failure.
        delay_seconds: Seconds to wait between runs.
        dry_run: If True, only log queries without running.

    Returns:
        List of run summary dicts (one per query).
    """
    from geoeventfusion.pipeline import run as run_pipeline

    results = []
    total = len(configs)

    for idx, config in enumerate(configs, start=1):
        query = config.query
        logger.info("Batch run [%d/%d]: %s", idx, total, query)

        if dry_run:
            print(f"  [DRY RUN] Would run: {query!r}  (llm_backend={config.llm_backend})")
            results.append({"query": query, "status": "DRY_RUN", "run_id": None})
            continue

        start_ts = time.time()
        try:
            context = run_pipeline(config)
            elapsed = time.time() - start_ts
            run_id = getattr(context, "run_id", "unknown")
            logger.info("  Completed in %.1fs  run_id=%s", elapsed, run_id)
            results.append({
                "query": query,
                "status": "OK",
                "run_id": run_id,
                "elapsed_seconds": round(elapsed, 2),
            })
        except Exception as exc:
            elapsed = time.time() - start_ts
            logger.error("  FAILED after %.1fs: %s", elapsed, exc)
            results.append({
                "query": query,
                "status": "FAILED",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 2),
            })
            if stop_on_error:
                logger.error("Stopping batch due to --stop-on-error flag")
                break

        if idx < total and not dry_run and delay_seconds > 0:
            logger.info("  Waiting %.1fs before next run...", delay_seconds)
            time.sleep(delay_seconds)

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a tabular run summary to stdout."""
    print("\n" + "=" * 60)
    print("BATCH RUN SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    dry = sum(1 for r in results if r["status"] == "DRY_RUN")
    print(f"Total: {len(results)}  OK: {ok}  FAILED: {failed}  DRY_RUN: {dry}")
    print()
    for r in results:
        status_label = r["status"].ljust(8)
        query_label = r["query"][:50].ljust(52)
        elapsed = f"{r.get('elapsed_seconds', 0.0):.1f}s" if "elapsed_seconds" in r else ""
        run_id = r.get("run_id", "")
        print(f"  {status_label}  {query_label}  {elapsed}  {run_id}")
        if r.get("error"):
            print(f"           ERROR: {r['error']}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the batch runner."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    batch_config = load_batch_config(args.config)
    defaults: Dict[str, Any] = batch_config.get("defaults", {})
    query_entries: List[Dict[str, Any]] = batch_config["queries"]

    # CLI global overrides
    global_overrides: Dict[str, Optional[str]] = {}
    if args.llm_backend:
        global_overrides["llm_backend"] = args.llm_backend
    if args.output_root:
        global_overrides["output_root"] = args.output_root

    # Build PipelineConfig for each query
    pipeline_configs: List[PipelineConfig] = []
    for entry in query_entries:
        if "query" not in entry or not entry["query"]:
            logger.warning("Skipping entry with missing 'query' field: %s", entry)
            continue
        merged = _merge_query_config(entry, defaults, global_overrides)
        try:
            cfg = build_pipeline_config(merged)
            pipeline_configs.append(cfg)
        except (TypeError, ValueError) as exc:
            logger.error("Invalid config for query %r: %s", entry.get("query"), exc)
            if args.stop_on_error:
                sys.exit(1)

    if not pipeline_configs:
        logger.error("No valid queries to run after config parsing")
        sys.exit(1)

    logger.info(
        "Batch runner: %d quer%s queued  (dry_run=%s)",
        len(pipeline_configs),
        "y" if len(pipeline_configs) == 1 else "ies",
        args.dry_run,
    )

    # Execute
    results = run_batch(
        pipeline_configs,
        stop_on_error=args.stop_on_error,
        delay_seconds=args.delay_seconds,
        dry_run=args.dry_run,
    )

    print_summary(results)

    # Exit with error code if any run failed
    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
