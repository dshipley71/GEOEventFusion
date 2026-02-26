"""GDELTAgent — Full 13-mode parallel GDELT DOC 2.0 fetch and analysis.

Implements the complete fetch architecture from AGENTS.md §2.1:
  Group A — 6 core article pools (always active)
  Group B — 6 timeline/signal modes (always active)
  Group C — 3 conditional source-scoped pools
  Group D — 2 optional visual intelligence modes

All GDELT gotchas from CLAUDE.md §10.1 are applied:
- Defensive _safe_parse_json() on every response
- Max 2 concurrent workers, 0.75 s stagger between submissions
- Date fields normalized through date_utils.normalize_date_str()
- near<N>: terms validated for minimum length
- repeat<N>: single-word only
- HybridRel only for post-2018-09-16 content; falls back to DateDesc
- TimelineVolRaw norm field NEVER smoothed — used raw for vol_ratio
- ImageCollageInfo gated behind enable_visual_intel=True
- imagetag: values always quoted
- domain_cap_pct enforced across all article pools
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.analysis.actor_graph import build_actor_graph
from geoeventfusion.analysis.query_builder import QueryBuilder
from geoeventfusion.analysis.spike_detector import compute_vol_ratio, detect_spikes
from geoeventfusion.analysis.tone_analyzer import (
    analyze_tone_distribution,
    compute_country_stats,
    compute_language_stats,
)
from geoeventfusion.clients.gdelt_client import GDELTClient
from geoeventfusion.models.events import (
    Article,
    GDELTAgentResult,
    ImageTopicTag,
    RunMetadata,
    TimelineStep,
    TimelineStepRaw,
    ToneChartBin,
)
from geoeventfusion.utils.date_utils import normalize_date_str, parse_date_range
from geoeventfusion.utils.text import extract_actors_from_articles

logger = logging.getLogger(__name__)

# Cutoff date for HybridRel availability (content before this returns no results)
_HYBRID_REL_MIN_DATE = "2018-09-16"


class GDELTAgent(BaseAgent):
    """Detect anomalous event activity from GDELT and produce normalized article pools.

    Executes up to 13 parallel GDELT DOC 2.0 API calls across four groups,
    applies domain diversity capping, detects coverage spikes, extracts the
    actor co-occurrence graph, and computes tone/language/country statistics.
    """

    name = "GDELTAgent"
    version = "2.0.0"

    def run(self, context: Any) -> GDELTAgentResult:
        """Execute all GDELT fetch groups and analysis.

        Args:
            context: PipelineContext with config and run metadata.

        Returns:
            GDELTAgentResult populated with all article pools, timelines, and analysis.
        """
        cfg = context.config
        result = GDELTAgentResult()

        start_date, end_date = parse_date_range(cfg.days_back)

        # ── Build query strings ─────────────────────────────────────────────────
        qb = QueryBuilder(
            near_min_term_length=cfg.near_min_term_length,
            near_window=cfg.near_window,
            repeat_threshold=cfg.repeat_threshold,
        )

        # Suggest GKG themes via LLM if backend is configured
        gkg_themes: List[str] = []
        if not cfg.test_mode:
            try:
                from geoeventfusion.clients.llm_client import LLMClient

                llm = LLMClient(
                    backend=cfg.llm_backend,
                    anthropic_model=cfg.anthropic_model,
                    ollama_model=cfg.ollama_model,
                    ollama_host=cfg.ollama_host,
                    anthropic_api_key=cfg.anthropic_api_key,
                )
                gkg_themes = qb.suggest_gkg_themes(cfg.query, llm_client=llm)
                if gkg_themes:
                    logger.info("GKG themes suggested: %s", gkg_themes)
            except Exception as exc:
                logger.warning("GKG theme suggestion failed: %s — continuing without themes", exc)

        base_query = qb.build_base_query(
            cfg.query,
            gkg_themes=gkg_themes if gkg_themes else None,
            add_repeat=True,
            add_near=True,
        )
        high_neg_query = qb.build_high_neg_query(base_query, cfg.tone_negative_threshold)
        high_emotion_query = qb.build_high_emotion_query(base_query, cfg.toneabs_threshold)

        # Store final query on context
        final_query = base_query
        context.config.query = context.config.query  # keep original
        logger.info(
            "GDELTAgent: query=%.80s start=%s end=%s",
            base_query,
            start_date,
            end_date,
        )

        # ── Initialize GDELT client ─────────────────────────────────────────────
        client = GDELTClient(
            max_retries=cfg.gdelt_max_retries,
            backoff_base=cfg.gdelt_backoff_base,
            request_timeout=cfg.gdelt_request_timeout,
            stagger_seconds=cfg.gdelt_stagger_seconds,
        )

        # ── Build fetch task list ───────────────────────────────────────────────
        fetch_tasks: List[Tuple[str, Dict[str, Any]]] = []

        # Group A — Core article pools
        fetch_tasks.extend([
            ("articles_recent", {
                "query": base_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "DateDesc",
                "start_date": start_date, "end_date": end_date,
            }),
            ("articles_negative", {
                "query": base_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "ToneAsc",
                "start_date": start_date, "end_date": end_date,
            }),
            ("articles_positive", {
                "query": base_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "ToneDesc",
                "start_date": start_date, "end_date": end_date,
            }),
            ("articles_relevant", {
                "query": base_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "HybridRel",
                "start_date": start_date, "end_date": end_date,
            }),
            ("articles_high_neg", {
                "query": high_neg_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "DateDesc",
                "start_date": start_date, "end_date": end_date,
            }),
            ("articles_high_emotion", {
                "query": high_emotion_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "DateDesc",
                "start_date": start_date, "end_date": end_date,
            }),
        ])

        # Group B — Timeline and signal modes
        fetch_tasks.extend([
            ("timeline_volinfo", {
                "query": base_query, "mode": "TimelineVolInfo",
                "start_date": start_date, "end_date": end_date,
                "timeline_smooth": cfg.timeline_smooth,
            }),
            ("timeline_volraw", {
                "query": base_query, "mode": "TimelineVolRaw",
                "start_date": start_date, "end_date": end_date,
                "timeline_smooth": 1,  # NEVER smooth norm field — use raw
            }),
            ("timeline_tone", {
                "query": base_query, "mode": "TimelineTone",
                "start_date": start_date, "end_date": end_date,
                "timeline_smooth": cfg.timeline_smooth,
            }),
            ("timeline_lang", {
                "query": base_query, "mode": "TimelineLang",
                "start_date": start_date, "end_date": end_date,
                "timeline_smooth": cfg.timeline_smooth,
            }),
            ("timeline_country", {
                "query": base_query, "mode": "TimelineSourceCountry",
                "start_date": start_date, "end_date": end_date,
                "timeline_smooth": cfg.timeline_smooth,
            }),
            ("tonechart", {
                "query": base_query, "mode": "ToneChart",
                "start_date": start_date, "end_date": end_date,
            }),
        ])

        # Group C — Conditional source-scoped pools
        if cfg.source_country_filter:
            country_query = qb.build_source_country_query(base_query, cfg.source_country_filter)
            fetch_tasks.append(("articles_source_country", {
                "query": country_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "DateDesc",
                "start_date": start_date, "end_date": end_date,
            }))

        if cfg.source_lang_filter:
            lang_query = qb.build_source_lang_query(base_query, cfg.source_lang_filter)
            fetch_tasks.append(("articles_source_lang", {
                "query": lang_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "DateDesc",
                "start_date": start_date, "end_date": end_date,
            }))

        if cfg.authoritative_domains:
            auth_query = qb.build_authoritative_domain_query(
                base_query, cfg.authoritative_domains
            )
            fetch_tasks.append(("articles_authoritative", {
                "query": auth_query, "mode": "ArtList",
                "max_records": cfg.max_records, "sort": "HybridRel",
                "start_date": start_date, "end_date": end_date,
            }))

        # Group D — Visual intelligence (gated behind enable_visual_intel)
        if cfg.enable_visual_intel and cfg.visual_imagetags:
            imagetag_query = qb.build_imagetag_query(cfg.visual_imagetags)
            if imagetag_query:
                fetch_tasks.append(("visual_images", {
                    "query": imagetag_query, "mode": "ImageCollageInfo",
                    "max_records": 100,
                    "start_date": start_date, "end_date": end_date,
                }))

        if cfg.enable_visual_intel and cfg.enable_word_clouds:
            fetch_tasks.append(("visual_image_topics", {
                "query": base_query, "mode": "WordCloudImageTags",
                "start_date": start_date, "end_date": end_date,
            }))

        active_modes = [task[0] for task in fetch_tasks]
        logger.info("GDELTAgent: executing %d fetch tasks", len(fetch_tasks))

        # ── Execute fetches with stagger and max 2 workers ──────────────────────
        raw_responses: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=cfg.gdelt_max_workers) as executor:
            future_map = {}
            for pool_key, kwargs in fetch_tasks:
                # Stagger submissions to respect GDELT rate limit
                time.sleep(cfg.gdelt_stagger_seconds)
                future = executor.submit(self._fetch_one, client, pool_key, **kwargs)
                future_map[future] = pool_key

            for future in as_completed(future_map):
                pool_key = future_map[future]
                try:
                    raw_responses[pool_key] = future.result()
                except Exception as exc:
                    logger.warning("Fetch task %s raised exception: %s", pool_key, exc)
                    raw_responses[pool_key] = None

        client.close()

        # ── Parse article pools ─────────────────────────────────────────────────
        article_pool_keys = [
            "articles_recent", "articles_negative", "articles_positive",
            "articles_relevant", "articles_high_neg", "articles_high_emotion",
            "articles_source_country", "articles_source_lang", "articles_authoritative",
        ]

        all_parsed_pools: Dict[str, List[Article]] = {}
        for key in article_pool_keys:
            raw = raw_responses.get(key)
            if raw is None:
                all_parsed_pools[key] = []
                continue
            # HybridRel fallback: if articles_relevant returns nothing, use DateDesc result
            if key == "articles_relevant" and not _extract_articles_from_response(raw):
                logger.info(
                    "GDELTAgent: HybridRel returned no results — falling back to DateDesc "
                    "for articles_relevant"
                )
                fallback = raw_responses.get("articles_recent")
                raw = fallback if fallback else raw
            articles = _extract_articles_from_response(raw)
            all_parsed_pools[key] = articles

        # ── Domain diversity cap ────────────────────────────────────────────────
        for key in article_pool_keys:
            all_parsed_pools[key] = _apply_domain_cap(
                all_parsed_pools[key], cfg.domain_cap_pct
            )

        # Assign article pools to result
        result.articles_recent = all_parsed_pools.get("articles_recent", [])
        result.articles_negative = all_parsed_pools.get("articles_negative", [])
        result.articles_positive = all_parsed_pools.get("articles_positive", [])
        result.articles_relevant = all_parsed_pools.get("articles_relevant", [])
        result.articles_high_neg = all_parsed_pools.get("articles_high_neg", [])
        result.articles_high_emotion = all_parsed_pools.get("articles_high_emotion", [])
        result.articles_source_country = all_parsed_pools.get("articles_source_country", [])
        result.articles_source_lang = all_parsed_pools.get("articles_source_lang", [])
        result.articles_authoritative = all_parsed_pools.get("articles_authoritative", [])

        # ── Parse timeline/signal modes ─────────────────────────────────────────
        result.timeline_volinfo = _parse_timeline_volinfo(
            raw_responses.get("timeline_volinfo")
        )
        result.timeline_volraw = _parse_timeline_volraw(
            raw_responses.get("timeline_volraw")
        )
        result.timeline_tone = _parse_timeline_generic(
            raw_responses.get("timeline_tone"), "timeline"
        )
        result.timeline_lang = _parse_timeline_multilabel(
            raw_responses.get("timeline_lang")
        )
        result.timeline_country = _parse_timeline_multilabel(
            raw_responses.get("timeline_country")
        )
        result.tonechart = _parse_tonechart(raw_responses.get("tonechart"))

        # ── Visual intelligence ─────────────────────────────────────────────────
        if cfg.enable_visual_intel:
            result.visual_images = _parse_visual_images(
                raw_responses.get("visual_images"), cfg.visual_staleness_hours
            )
            result.image_topics = _parse_image_topics(
                raw_responses.get("visual_image_topics")
            )

        # ── Spike detection ─────────────────────────────────────────────────────
        if result.timeline_volinfo:
            result.spikes = detect_spikes(
                result.timeline_volinfo,
                z_threshold=cfg.spike_z_threshold,
                query=cfg.query,
            )[: cfg.max_spikes]
        else:
            result.spikes = []
            result.warnings.append("No TimelineVolInfo data — spike detection skipped")

        # ── Spike article backfill ──────────────────────────────────────────────
        if result.spikes:
            result.spike_articles = self._backfill_spike_articles(
                result.spikes, cfg, client_kwargs={
                    "max_retries": cfg.gdelt_max_retries,
                    "backoff_base": cfg.gdelt_backoff_base,
                    "request_timeout": cfg.gdelt_request_timeout,
                    "stagger_seconds": cfg.gdelt_stagger_seconds,
                }
            )

        # ── Build title→URL map for citation lookup ─────────────────────────────
        result.title_url_map = _build_title_url_map(result.all_articles())

        # ── Vol ratio from TimelineVolRaw ───────────────────────────────────────
        result.vol_ratio = compute_vol_ratio(result.timeline_volraw)

        # ── Tone, language, country statistics ─────────────────────────────────
        if result.tonechart:
            result.tone_stats = analyze_tone_distribution(result.tonechart)
        if result.timeline_lang:
            result.language_stats = compute_language_stats(result.timeline_lang)
        if result.timeline_country:
            result.country_stats = compute_country_stats(result.timeline_country)

        # ── Actor co-occurrence graph ───────────────────────────────────────────
        all_articles = result.all_articles()
        if all_articles:
            triples = extract_actors_from_articles(all_articles)
            if triples:
                result.actor_graph = build_actor_graph(
                    triples,
                    hub_top_n=cfg.actor_hub_top_n,
                    broker_ratio_threshold=cfg.actor_broker_ratio_threshold,
                    pagerank_max_iter=cfg.actor_pagerank_max_iter,
                )

        # ── Run metadata ────────────────────────────────────────────────────────
        from datetime import datetime as _dt

        record_counts = {key: len(all_parsed_pools.get(key, [])) for key in article_pool_keys}
        result.run_metadata = RunMetadata(
            query=cfg.query,
            final_query=final_query,
            days_back=cfg.days_back,
            start_date=start_date,
            end_date=end_date,
            record_counts=record_counts,
            active_fetch_modes=active_modes,
            run_timestamp=_dt.utcnow().isoformat() + "Z",
        )

        # ── Determine status ────────────────────────────────────────────────────
        total_articles = sum(record_counts.values())
        if total_articles == 0:
            result.status = AgentStatus.CRITICAL
            result.warnings.append(
                "All GDELT article pools are empty. Check query, date range, and API access."
            )
            logger.error("GDELTAgent: CRITICAL — all article pools empty")
        elif sum(
            len(p) for p in [result.articles_recent, result.articles_relevant]
        ) == 0:
            result.status = AgentStatus.PARTIAL
            result.warnings.append(
                "Core article pools (recent, relevant) are empty — partial results only"
            )
        else:
            result.status = AgentStatus.OK

        logger.info(
            "GDELTAgent: %d total articles across %d pools | %d spikes | status=%s",
            total_articles,
            len([k for k in record_counts if record_counts[k] > 0]),
            len(result.spikes),
            result.status,
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _fetch_one(
        self,
        client: GDELTClient,
        pool_key: str,
        query: str,
        mode: str,
        max_records: int = 250,
        sort: str = "DateDesc",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeline_smooth: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single GDELT fetch task.

        Args:
            client: GDELTClient instance.
            pool_key: Identifier for this fetch (used in logs).
            query: GDELT query string.
            mode: GDELT API mode.
            max_records: Maximum records for ArtList mode.
            sort: Sort order for ArtList mode.
            start_date: Start date.
            end_date: End date.
            timeline_smooth: Smoothing for timeline modes.

        Returns:
            Parsed API response dict, or None on failure.
        """
        logger.debug("GDELTAgent: fetching pool=%s mode=%s sort=%s", pool_key, mode, sort)
        return client.fetch(
            query=query,
            mode=mode,
            max_records=max_records,
            sort=sort,
            start_date=start_date,
            end_date=end_date,
            timeline_smooth=timeline_smooth,
        )

    def _backfill_spike_articles(
        self,
        spikes: List[Any],
        cfg: Any,
        client_kwargs: Dict[str, Any],
    ) -> Dict[str, List[Article]]:
        """Backfill articles for each spike window via targeted ±N hour fetches.

        Args:
            spikes: List of SpikeWindow objects.
            cfg: PipelineConfig.
            client_kwargs: GDELTClient initialization kwargs.

        Returns:
            Dict mapping spike date → List[Article].
        """
        from geoeventfusion.utils.date_utils import gdelt_date_format

        spike_articles: Dict[str, List[Article]] = {}
        if not spikes:
            return spike_articles

        client = GDELTClient(**client_kwargs)
        try:
            for spike in spikes[:5]:  # Limit backfill to top 5 spikes
                try:
                    from datetime import datetime, timedelta

                    spike_dt = datetime.strptime(spike.date[:10], "%Y-%m-%d")
                    start_dt = (spike_dt - timedelta(hours=cfg.spike_backfill_hours))
                    end_dt = (spike_dt + timedelta(hours=cfg.spike_backfill_hours))
                    start_str = start_dt.strftime("%Y-%m-%d")
                    end_str = end_dt.strftime("%Y-%m-%d")

                    time.sleep(cfg.gdelt_stagger_seconds)
                    raw = client.fetch(
                        query=cfg.query,
                        mode="ArtList",
                        max_records=cfg.max_records,
                        sort="DateDesc",
                        start_date=start_str,
                        end_date=end_str,
                    )
                    articles = _extract_articles_from_response(raw) if raw else []
                    spike_articles[spike.date] = articles
                    logger.debug(
                        "Spike backfill: date=%s articles=%d", spike.date, len(articles)
                    )
                except Exception as exc:
                    logger.warning("Spike backfill failed for %s: %s", spike.date, exc)
        finally:
            client.close()

        return spike_articles


# ── Module-level parse helpers ──────────────────────────────────────────────────


def _extract_articles_from_response(raw: Optional[Dict[str, Any]]) -> List[Article]:
    """Parse GDELT ArtList API response into Article objects.

    Args:
        raw: Parsed GDELT response dict.

    Returns:
        List of Article objects. Empty list if response is invalid.
    """
    if not raw or not isinstance(raw, dict):
        return []

    articles_data = raw.get("articles", [])
    if not isinstance(articles_data, list):
        return []

    articles: List[Article] = []
    for item in articles_data:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        title = item.get("title", "")
        if not url or not title:
            continue

        raw_date = (
            item.get("seendate")
            or item.get("dateadded")
            or item.get("date")
            or ""
        )
        published_at = normalize_date_str(str(raw_date)) if raw_date else ""

        domain = item.get("domain", "")
        if not domain and url:
            try:
                domain = urlparse(url).netloc.lower().lstrip("www.")
            except Exception:
                domain = ""

        tone_raw = item.get("tone")
        tone: Optional[float] = None
        if tone_raw is not None:
            try:
                tone = float(tone_raw)
            except (ValueError, TypeError):
                tone = None

        articles.append(
            Article(
                url=url,
                title=title,
                published_at=published_at,
                source=domain,
                tone=tone,
                domain=domain,
                language=item.get("language", ""),
                source_country=item.get("sourcecountry", ""),
            )
        )
    return articles


def _apply_domain_cap(
    articles: List[Article], domain_cap_pct: float
) -> List[Article]:
    """Apply domain diversity cap — limit any single domain to domain_cap_pct of pool.

    Args:
        articles: Input article list.
        domain_cap_pct: Maximum fraction (0–1) from any single domain.

    Returns:
        Filtered article list with domain diversity enforced.
    """
    if not articles or domain_cap_pct >= 1.0:
        return articles

    max_per_domain = max(1, int(len(articles) * domain_cap_pct))
    domain_counts: Dict[str, int] = defaultdict(int)
    result: List[Article] = []

    for article in articles:
        domain = article.domain or article.source or "unknown"
        if domain_counts[domain] < max_per_domain:
            domain_counts[domain] += 1
            result.append(article)

    if len(result) < len(articles):
        logger.debug(
            "Domain cap applied: %d → %d articles (cap=%.0f%%)",
            len(articles),
            len(result),
            domain_cap_pct * 100,
        )
    return result


def _parse_timeline_volinfo(raw: Optional[Dict[str, Any]]) -> List[TimelineStep]:
    """Parse GDELT TimelineVolInfo response.

    Args:
        raw: Parsed API response dict.

    Returns:
        List of TimelineStep objects.
    """
    if not raw or not isinstance(raw, dict):
        return []

    timeline_data = raw.get("timeline", [])
    if not isinstance(timeline_data, list) or not timeline_data:
        return []

    # TimelineVolInfo returns a list of series; take the first (primary) series
    series = timeline_data[0] if timeline_data else {}
    data_points = series.get("data", []) if isinstance(series, dict) else []

    steps: List[TimelineStep] = []
    for point in data_points:
        if not isinstance(point, dict):
            continue
        raw_date = point.get("date", "")
        value = point.get("value", 0.0)
        date = normalize_date_str(str(raw_date)) if raw_date else ""
        if not date:
            continue

        # Extract top articles if present
        top_articles_raw = point.get("topartarticles", []) or []
        top_articles: List[Article] = []
        for art in top_articles_raw:
            if isinstance(art, dict) and art.get("url") and art.get("title"):
                top_articles.append(
                    Article(
                        url=art["url"],
                        title=art["title"],
                        published_at=date,
                        source=art.get("domain", ""),
                        domain=art.get("domain", ""),
                    )
                )

        try:
            steps.append(TimelineStep(date=date, value=float(value), articles=top_articles))
        except (ValueError, TypeError):
            continue

    return steps


def _parse_timeline_generic(
    raw: Optional[Dict[str, Any]], key: str = "timeline"
) -> List[TimelineStep]:
    """Parse generic GDELT timeline response (tone, etc.).

    Args:
        raw: Parsed API response dict.
        key: Top-level key for timeline data.

    Returns:
        List of TimelineStep objects.
    """
    if not raw or not isinstance(raw, dict):
        return []

    timeline_data = raw.get(key, [])
    if not isinstance(timeline_data, list) or not timeline_data:
        return []

    series = timeline_data[0] if isinstance(timeline_data[0], dict) else {}
    data_points = series.get("data", [])

    steps: List[TimelineStep] = []
    for point in data_points:
        if not isinstance(point, dict):
            continue
        raw_date = point.get("date", "")
        value = point.get("value", 0.0)
        date = normalize_date_str(str(raw_date)) if raw_date else ""
        if not date:
            continue
        try:
            steps.append(TimelineStep(date=date, value=float(value)))
        except (ValueError, TypeError):
            continue

    return steps


def _parse_timeline_multilabel(raw: Optional[Dict[str, Any]]) -> List[TimelineStep]:
    """Parse multi-series GDELT timeline (TimelineLang, TimelineSourceCountry).

    Each series has a label (language/country code) and its own data points.

    Args:
        raw: Parsed API response dict.

    Returns:
        Flat list of TimelineStep objects with .label set to the series name.
    """
    if not raw or not isinstance(raw, dict):
        return []

    timeline_data = raw.get("timeline", [])
    if not isinstance(timeline_data, list):
        return []

    steps: List[TimelineStep] = []
    for series in timeline_data:
        if not isinstance(series, dict):
            continue
        label = series.get("series", "") or series.get("label", "")
        for point in series.get("data", []):
            if not isinstance(point, dict):
                continue
            raw_date = point.get("date", "")
            value = point.get("value", 0.0)
            date = normalize_date_str(str(raw_date)) if raw_date else ""
            if not date:
                continue
            try:
                steps.append(TimelineStep(date=date, value=float(value), label=str(label)))
            except (ValueError, TypeError):
                continue

    return steps


def _parse_timeline_volraw(raw: Optional[Dict[str, Any]]) -> List[TimelineStepRaw]:
    """Parse GDELT TimelineVolRaw response.

    IMPORTANT: The norm field must NOT be smoothed — always use the raw norm value
    as the denominator for vol_ratio computation (per CLAUDE.md §10.1).

    Args:
        raw: Parsed API response dict.

    Returns:
        List of TimelineStepRaw objects with absolute counts and raw norm values.
    """
    if not raw or not isinstance(raw, dict):
        return []

    timeline_data = raw.get("timeline", [])
    if not isinstance(timeline_data, list) or not timeline_data:
        return []

    series = timeline_data[0] if isinstance(timeline_data[0], dict) else {}
    data_points = series.get("data", [])

    steps: List[TimelineStepRaw] = []
    for point in data_points:
        if not isinstance(point, dict):
            continue
        raw_date = point.get("date", "")
        volume = point.get("value", 0)
        norm = point.get("norm", 0.0)  # Raw norm — NEVER smooth this
        date = normalize_date_str(str(raw_date)) if raw_date else ""
        if not date:
            continue
        try:
            steps.append(TimelineStepRaw(date=date, volume=int(volume), norm=float(norm)))
        except (ValueError, TypeError):
            continue

    return steps


def _parse_tonechart(raw: Optional[Dict[str, Any]]) -> List[ToneChartBin]:
    """Parse GDELT ToneChart histogram response.

    Args:
        raw: Parsed API response dict.

    Returns:
        List of ToneChartBin objects.
    """
    if not raw or not isinstance(raw, dict):
        return []

    # ToneChart response uses "topcats" or "tonechart"
    chart_data = raw.get("tonechart", raw.get("topcats", []))
    if not isinstance(chart_data, list):
        return []

    bins: List[ToneChartBin] = []
    for item in chart_data:
        if not isinstance(item, dict):
            continue
        tone_val = item.get("tone", item.get("label", 0)) or 0
        count = item.get("count", item.get("value", 0)) or 0
        try:
            bins.append(ToneChartBin(tone_value=float(tone_val), count=int(count)))
        except (ValueError, TypeError):
            continue

    return bins


def _parse_visual_images(
    raw: Optional[Dict[str, Any]], staleness_hours: int
) -> List[Any]:
    """Parse GDELT ImageCollageInfo response into VisualImage objects.

    Args:
        raw: Parsed API response dict from ImageCollageInfo mode.
        staleness_hours: Hours threshold for EXIF staleness warning.

    Returns:
        List of VisualImage objects.
    """
    if not raw or not isinstance(raw, dict):
        return []

    from geoeventfusion.analysis.visual_intel import parse_image_collage_response

    results: List[Any] = parse_image_collage_response(
        raw, staleness_threshold_hours=staleness_hours
    )
    return results


def _parse_image_topics(raw: Optional[Dict[str, Any]]) -> List[ImageTopicTag]:
    """Parse GDELT WordCloudImageTags response into ImageTopicTag objects.

    Args:
        raw: Parsed API response dict.

    Returns:
        List of ImageTopicTag objects.
    """
    if not raw or not isinstance(raw, dict):
        return []

    tags_data = raw.get("imagetags", raw.get("wordcloud", []))
    if not isinstance(tags_data, list):
        return []

    tags: List[ImageTopicTag] = []
    total_count = sum(
        int(t.get("count", 0)) for t in tags_data if isinstance(t, dict)
    )

    for item in tags_data:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag", item.get("label", ""))
        count = item.get("count", 0)
        if not tag:
            continue
        try:
            normalized = int(count) / total_count if total_count > 0 else 0.0
            tags.append(
                ImageTopicTag(tag=str(tag), count=int(count), normalized_count=normalized)
            )
        except (ValueError, TypeError):
            continue

    return sorted(tags, key=lambda t: t.count, reverse=True)


def _build_title_url_map(articles: List[Article]) -> Dict[str, List[Article]]:
    """Build a date-keyed article lookup map for citation building.

    Args:
        articles: Deduplicated list of all articles.

    Returns:
        Dict mapping ISO date string → list of articles published on that date.
    """
    title_url_map: Dict[str, List[Article]] = defaultdict(list)
    for article in articles:
        date_key = article.published_at[:10] if article.published_at else "unknown"
        title_url_map[date_key].append(article)
    return dict(title_url_map)
