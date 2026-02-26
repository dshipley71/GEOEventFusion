#!/usr/bin/env python3
"""GEOEventFusion CLI — run the full intelligence pipeline.

Usage:
    python scripts/run_pipeline.py --query "Houthi Red Sea attacks" --days-back 90
    python scripts/run_pipeline.py --query "Taiwan Strait tensions" --llm-backend anthropic
    python scripts/run_pipeline.py --query "test" --test-mode
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for consistent import resolution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.defaults import (  # noqa: E402
    ANTHROPIC_MODEL,
    DAYS_BACK,
    DEFAULT_LOG_LEVEL,
    DOMAIN_CAP_PCT,
    MAX_RECORDS,
    MAX_SPIKES,
    NEAR_WINDOW,
    OLLAMA_MODEL,
    OUTPUT_ROOT,
    REPEAT_THRESHOLD,
    SPIKE_BACKFILL_HOURS,
    SPIKE_Z_THRESHOLD,
    TIMELINE_SMOOTH,
    TONEABS_THRESHOLD,
    TONE_NEGATIVE_THRESHOLD,
)
from config.settings import PipelineConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse argument parser with all PipelineConfig fields as flags."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="GEOEventFusion — Multi-agent geopolitical intelligence pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core query ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--query", type=str, required=True, help="Base geopolitical query string"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=DAYS_BACK,
        help="Analysis window in days (GDELT max: ~90)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS,
        help="Maximum articles per GDELT ArtList fetch (GDELT hard limit: 250)",
    )

    # ── LLM backend ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="ollama",
        choices=["anthropic", "ollama"],
        help="LLM backend to use for extraction and narrative generation",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default=ANTHROPIC_MODEL,
        help="Anthropic model ID",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=OLLAMA_MODEL,
        help="Ollama model name",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per LLM call",
    )

    # ── Spike detection ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--spike-z-threshold",
        type=float,
        default=SPIKE_Z_THRESHOLD,
        help="Z-score threshold for declaring a coverage spike",
    )
    parser.add_argument(
        "--max-spikes",
        type=int,
        default=MAX_SPIKES,
        help="Maximum number of spikes to retain for downstream enrichment",
    )
    parser.add_argument(
        "--spike-backfill-hours",
        type=int,
        default=SPIKE_BACKFILL_HOURS,
        help="Hours on each side of a spike date for article backfill fetch",
    )

    # ── GDELT fetch configuration ────────────────────────────────────────────────
    parser.add_argument(
        "--domain-cap-pct",
        type=float,
        default=DOMAIN_CAP_PCT,
        help="Maximum fraction of any single domain across all article pools",
    )
    parser.add_argument(
        "--timeline-smooth",
        type=int,
        default=TIMELINE_SMOOTH,
        help="GDELT timeline smoothing window in steps (1–30)",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=REPEAT_THRESHOLD,
        help="Minimum keyword repetitions for repeat<N>: operator",
    )
    parser.add_argument(
        "--near-window",
        type=int,
        default=NEAR_WINDOW,
        help="Word proximity window for near<N>: operator",
    )
    parser.add_argument(
        "--tone-negative-threshold",
        type=float,
        default=TONE_NEGATIVE_THRESHOLD,
        help="Tone ceiling for articles_high_neg pool (tone< operator threshold)",
    )
    parser.add_argument(
        "--toneabs-threshold",
        type=float,
        default=TONEABS_THRESHOLD,
        help="Minimum absolute tone for articles_high_emotion pool (toneabs> operator)",
    )

    # ── Conditional GDELT source pools ──────────────────────────────────────────
    parser.add_argument(
        "--source-country",
        type=str,
        default=None,
        help="FIPS country code for country-scoped fetch (e.g., IR, RS, CH)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=None,
        help="ISO 3-char language code for language-scoped fetch (e.g., ara, rus)",
    )
    parser.add_argument(
        "--authoritative-domains",
        type=str,
        nargs="*",
        default=[],
        metavar="DOMAIN",
        help="Exact domains for authority source pool (e.g., un.org state.gov nato.int)",
    )

    # ── Visual intelligence ──────────────────────────────────────────────────────
    parser.add_argument(
        "--enable-visual-intel",
        action="store_true",
        default=False,
        help="Enable visual intelligence fetch modes (ImageCollageInfo)",
    )
    parser.add_argument(
        "--enable-word-clouds",
        action="store_true",
        default=False,
        help="Enable image word cloud fetch (WordCloudImageTags)",
    )
    parser.add_argument(
        "--visual-imagetags",
        type=str,
        nargs="*",
        default=[],
        metavar="TAG",
        help="VGKG imagetag values for visual intel (e.g., military protest fire)",
    )

    # ── RSS feed enrichment ──────────────────────────────────────────────────────
    parser.add_argument(
        "--rss-feeds",
        type=str,
        nargs="*",
        default=[],
        metavar="URL",
        help="RSS/Atom feed URLs for spike enrichment",
    )
    parser.add_argument(
        "--rss-max-articles",
        type=int,
        default=50,
        help="Maximum articles per spike from RSS feeds",
    )
    parser.add_argument(
        "--rss-time-window-hours",
        type=int,
        default=48,
        help="Hours around spike date for RSS article filtering",
    )

    # ── Ground truth datasets ────────────────────────────────────────────────────
    parser.add_argument(
        "--ground-truth-sources",
        type=str,
        nargs="*",
        default=[],
        choices=["acled", "icews"],
        metavar="SOURCE",
        help="Ground truth sources to activate (acled, icews)",
    )
    parser.add_argument(
        "--ground-truth-countries",
        type=str,
        nargs="*",
        default=[],
        metavar="COUNTRY",
        help="Country filters for ground truth datasets (ISO codes or names)",
    )
    parser.add_argument(
        "--ground-truth-event-types",
        type=str,
        nargs="*",
        default=[],
        metavar="TYPE",
        help="Event type filters for ground truth (e.g., 'Battles' 'Violence against civilians')",
    )

    # ── Custom dataset ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--custom-dataset",
        type=str,
        default=None,
        help="Path to a custom dataset file for cross-referencing",
    )
    parser.add_argument(
        "--custom-dataset-format",
        type=str,
        default="csv",
        choices=["csv", "json", "sql", "api"],
        help="Format of the custom dataset",
    )

    # ── Fusion parameters ────────────────────────────────────────────────────────
    parser.add_argument(
        "--fusion-temporal-window-hours",
        type=int,
        default=72,
        help="Maximum hours between events for temporal proximity matching",
    )
    parser.add_argument(
        "--fusion-geographic-threshold-km",
        type=float,
        default=200.0,
        help="Maximum km between events for geographic proximity matching",
    )

    # ── Confidence and citation floors ───────────────────────────────────────────
    parser.add_argument(
        "--max-confidence",
        type=float,
        default=0.82,
        help="Hard cap for LLM confidence scores (epistemic ceiling, max 0.82)",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=3,
        help="Minimum cited sources per storyboard panel",
    )
    parser.add_argument(
        "--min-panel-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence below which panels are flagged LOW_CONFIDENCE",
    )

    # ── Validation thresholds ────────────────────────────────────────────────────
    parser.add_argument(
        "--validation-title-similarity-threshold",
        type=float,
        default=0.55,
        help="Minimum Levenshtein similarity for title-to-claim grounding",
    )
    parser.add_argument(
        "--validation-date-delta-days",
        type=int,
        default=7,
        help="Maximum days between article date and claimed event date",
    )
    parser.add_argument(
        "--validation-min-corroboration",
        type=int,
        default=2,
        help="Minimum cross-source corroboration count for verification",
    )

    # ── Actor graph ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--actor-hub-top-n",
        type=int,
        default=5,
        help="Top-N actors by degree centrality classified as Hub",
    )
    parser.add_argument(
        "--actor-broker-ratio-threshold",
        type=float,
        default=1.5,
        help="Betweenness-to-degree ratio threshold for Broker classification",
    )

    # ── Visual intelligence thresholds ───────────────────────────────────────────
    parser.add_argument(
        "--visual-staleness-hours",
        type=int,
        default=72,
        help="EXIF capture-date staleness warning threshold (hours before article)",
    )

    # ── Output and logging ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output-root",
        type=str,
        default=OUTPUT_ROOT,
        help="Root directory for pipeline run outputs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )

    # ── Test mode ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Use fixture data — no real API calls (for development and CI)",
    )

    return parser


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed CLI arguments to a PipelineConfig instance.

    Args:
        args: Parsed argparse Namespace.

    Returns:
        PipelineConfig populated from CLI flags.
    """
    return PipelineConfig(
        query=args.query,
        days_back=args.days_back,
        max_records=args.max_records,
        llm_backend=args.llm_backend,
        anthropic_model=args.anthropic_model,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        spike_z_threshold=args.spike_z_threshold,
        max_spikes=args.max_spikes,
        spike_backfill_hours=args.spike_backfill_hours,
        domain_cap_pct=args.domain_cap_pct,
        timeline_smooth=args.timeline_smooth,
        repeat_threshold=args.repeat_threshold,
        near_window=args.near_window,
        tone_negative_threshold=args.tone_negative_threshold,
        toneabs_threshold=args.toneabs_threshold,
        source_country_filter=args.source_country,
        source_lang_filter=args.source_lang,
        authoritative_domains=list(args.authoritative_domains or []),
        enable_visual_intel=args.enable_visual_intel,
        enable_word_clouds=args.enable_word_clouds,
        visual_imagetags=list(args.visual_imagetags or []),
        rss_feed_list=list(args.rss_feeds or []),
        rss_max_articles_per_spike=args.rss_max_articles,
        rss_time_window_hours=args.rss_time_window_hours,
        ground_truth_sources=list(args.ground_truth_sources or []),
        ground_truth_country_filter=list(args.ground_truth_countries or []),
        ground_truth_event_types=list(args.ground_truth_event_types or []),
        custom_dataset_path=args.custom_dataset,
        custom_dataset_format=args.custom_dataset_format,
        fusion_temporal_window_hours=args.fusion_temporal_window_hours,
        fusion_geographic_threshold_km=args.fusion_geographic_threshold_km,
        max_confidence=min(args.max_confidence, 0.82),
        min_citations=args.min_citations,
        min_panel_confidence=args.min_panel_confidence,
        validation_title_similarity_threshold=args.validation_title_similarity_threshold,
        validation_date_delta_days=args.validation_date_delta_days,
        validation_min_corroboration=args.validation_min_corroboration,
        actor_hub_top_n=args.actor_hub_top_n,
        actor_broker_ratio_threshold=args.actor_broker_ratio_threshold,
        visual_staleness_hours=args.visual_staleness_hours,
        output_root=args.output_root,
        log_level=args.log_level,
        test_mode=args.test_mode,
    )


def setup_logging(log_level: str) -> None:
    """Configure the root logger with a structured format.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """CLI entrypoint — parse arguments, build config, run pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("run_pipeline")

    config = args_to_config(args)
    logger.info("GEOEventFusion pipeline starting — query: %r", config.query)
    logger.info(
        "Backend: %s | Days back: %d | Test mode: %s",
        config.llm_backend,
        config.days_back,
        config.test_mode,
    )

    try:
        import geoeventfusion.pipeline as pipeline  # type: ignore[import]

        result = pipeline.run(config)
        if result:
            run_id = getattr(result, "run_id", "N/A")
            logger.info("Pipeline complete. Run ID: %s", run_id)
        else:
            logger.error("Pipeline returned no result")
            sys.exit(1)

    except ImportError:
        logger.error(
            "Pipeline orchestrator (geoeventfusion.pipeline) is not yet implemented. "
            "Individual analysis modules are available in geoeventfusion/analysis/, "
            "geoeventfusion/clients/, and geoeventfusion/io/."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed with unhandled exception: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
