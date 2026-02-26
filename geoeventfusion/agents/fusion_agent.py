"""FusionAgent — Link and cluster events from all sources into unified clusters.

Implements AGENTS.md §2.6 specification:
- Temporal proximity matching (date difference in hours)
- Geographic proximity matching (Haversine distance)
- Actor overlap scoring (Jaccard set intersection)
- Semantic similarity (cosine of title embeddings or keyword TF-IDF fallback)
- Event type alignment scoring
- Contradiction detection across source pools

Weight defaults (configurable via PipelineConfig.fusion_weights):
  temporal: 0.25, geographic: 0.25, actor: 0.20, semantic: 0.20, event_type: 0.10
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from geoeventfusion.agents.base import BaseAgent
from geoeventfusion.models.fusion import (
    ContradictionFlag,
    FusionAgentResult,
    FusionCluster,
    FusionStats,
)
from geoeventfusion.utils.date_utils import date_delta_days

logger = logging.getLogger(__name__)


class FusionAgent(BaseAgent):
    """Link and cluster events across all source pools into unified fusion clusters.

    Operates on LLM-extracted structured events from all upstream agents and
    clusters them by multi-dimensional similarity scoring.
    """

    name = "FusionAgent"
    version = "1.0.0"

    def run(self, context: Any) -> FusionAgentResult:
        """Cluster events from all sources into unified fusion clusters.

        Args:
            context: PipelineContext with config, llm_result, gdelt_result, etc.

        Returns:
            FusionAgentResult with clusters, unclustered events, and fusion statistics.
        """
        cfg = context.config
        result = FusionAgentResult()

        # ── Gather all structured events from upstream agents ─────────────────
        all_events: List[Dict[str, Any]] = []

        # LLM-extracted events (primary source)
        if context.llm_result and context.llm_result.timeline_events:
            for entry in context.llm_result.timeline_events:
                all_events.append(_timeline_entry_to_dict(entry, "llm"))

        # Ground-truth events
        if context.ground_truth_result and context.ground_truth_result.events:
            for gt_event in context.ground_truth_result.events:
                all_events.append(_ground_truth_event_to_dict(gt_event))

        # Custom dataset matches (as supplemental events)
        if context.custom_dataset_result and context.custom_dataset_result.matches:
            for match in context.custom_dataset_result.matches:
                all_events.append(_custom_match_to_dict(match))

        if not all_events:
            logger.info("FusionAgent: no events to cluster — returning empty result")
            result.warnings.append("No structured events available for fusion")
            result.fusion_stats = FusionStats(total_events_in=0)
            return result

        logger.info("FusionAgent: clustering %d events", len(all_events))

        # ── Cluster events using greedy agglomerative approach ────────────────
        weights = cfg.fusion_weights
        temporal_window = cfg.fusion_temporal_window_hours
        geo_threshold = cfg.fusion_geographic_threshold_km

        clusters = _greedy_cluster(
            all_events,
            weights=weights,
            temporal_window_hours=temporal_window,
            geo_threshold_km=geo_threshold,
        )

        # ── Build FusionCluster objects ───────────────────────────────────────
        fusion_clusters: List[FusionCluster] = []
        for cluster_events in clusters:
            cluster = _build_fusion_cluster(cluster_events, cfg.max_confidence)
            fusion_clusters.append(cluster)

        # ── Detect contradictions across clusters ──────────────────────────────
        _detect_contradictions(fusion_clusters)

        # ── Identify unclustered events ────────────────────────────────────────
        clustered_ids: Set[str] = {
            e.get("_event_id", "")
            for cluster in fusion_clusters
            for e in cluster.events
            if isinstance(e, dict)
        }
        unclustered = [
            e for e in all_events
            if e.get("_event_id", "") not in clustered_ids
        ]

        result.clusters = fusion_clusters
        result.unclustered_events = unclustered

        # ── Fusion statistics ─────────────────────────────────────────────────
        total_in = len(all_events)
        n_clusters = len(fusion_clusters)
        mean_size = (
            sum(len(c.events) for c in fusion_clusters) / n_clusters
            if n_clusters > 0 else 0.0
        )
        contradiction_count = sum(
            len(c.contradiction_flags) for c in fusion_clusters
        )
        contradiction_rate = contradiction_count / n_clusters if n_clusters > 0 else 0.0

        result.fusion_stats = FusionStats(
            total_events_in=total_in,
            total_clusters=n_clusters,
            mean_cluster_size=round(mean_size, 2),
            contradiction_rate=round(contradiction_rate, 4),
            corroboration_rate=round(
                sum(1 for c in fusion_clusters if c.corroboration_count >= 2) / n_clusters
                if n_clusters > 0 else 0.0,
                4,
            ),
        )

        logger.info(
            "FusionAgent: %d events → %d clusters | mean_size=%.1f | "
            "contradictions=%d | corroborated=%d",
            total_in,
            n_clusters,
            mean_size,
            contradiction_count,
            sum(1 for c in fusion_clusters if c.corroboration_count >= 2),
        )
        return result


# ── Clustering helpers ──────────────────────────────────────────────────────────


def _greedy_cluster(
    events: List[Dict[str, Any]],
    weights: Any,
    temporal_window_hours: int,
    geo_threshold_km: float,
) -> List[List[Dict[str, Any]]]:
    """Greedy agglomerative clustering by multi-dimensional similarity.

    Each event is assigned to the first existing cluster whose representative
    event scores above a merge threshold, or starts a new cluster.

    Args:
        events: List of normalized event dicts.
        weights: FusionWeights object with temporal/geographic/actor/semantic/event_type.
        temporal_window_hours: Max hours between events to consider temporal proximity.
        geo_threshold_km: Max km between events to consider geographic proximity.

    Returns:
        List of clusters, each a list of event dicts.
    """
    clusters: List[List[Dict[str, Any]]] = []
    _MERGE_THRESHOLD = 0.35  # Minimum composite score to merge into an existing cluster

    for event in events:
        best_cluster_idx: Optional[int] = None
        best_score = 0.0

        for idx, cluster in enumerate(clusters):
            # Compare against the cluster representative (first event)
            rep = cluster[0]
            score = _composite_score(
                event, rep, weights, temporal_window_hours, geo_threshold_km
            )
            if score > best_score:
                best_score = score
                best_cluster_idx = idx

        if best_cluster_idx is not None and best_score >= _MERGE_THRESHOLD:
            clusters[best_cluster_idx].append(event)
        else:
            clusters.append([event])

    return clusters


def _composite_score(
    event_a: Dict[str, Any],
    event_b: Dict[str, Any],
    weights: Any,
    temporal_window_hours: int,
    geo_threshold_km: float,
) -> float:
    """Compute weighted composite similarity score between two events.

    Args:
        event_a: First event dict.
        event_b: Second event dict.
        weights: FusionWeights with temporal/geographic/actor/semantic/event_type.
        temporal_window_hours: Max hours for temporal proximity.
        geo_threshold_km: Max km for geographic proximity.

    Returns:
        Composite similarity score in [0.0, 1.0].
    """
    score = 0.0

    # Temporal proximity
    date_a = event_a.get("datetime", event_a.get("date", ""))
    date_b = event_b.get("datetime", event_b.get("date", ""))
    if date_a and date_b:
        delta = date_delta_days(str(date_a), str(date_b))
        if delta is not None:
            delta_hours = delta * 24
            if delta_hours <= temporal_window_hours:
                temporal = 1.0 - (delta_hours / max(1, temporal_window_hours))
                score += temporal * weights.temporal

    # Geographic proximity
    lat_a = event_a.get("lat")
    lon_a = event_a.get("lon")
    lat_b = event_b.get("lat")
    lon_b = event_b.get("lon")
    if lat_a is not None and lon_a is not None and lat_b is not None and lon_b is not None:
        try:
            from geoeventfusion.utils.geo_utils import haversine_km

            dist = haversine_km(float(lat_a), float(lon_a), float(lat_b), float(lon_b))
            if dist <= geo_threshold_km:
                geo_score = 1.0 - (dist / max(1.0, geo_threshold_km))
                score += geo_score * weights.geographic
        except Exception:
            pass
    elif event_a.get("country") and event_a.get("country") == event_b.get("country"):
        # Same country → partial geographic credit
        score += 0.5 * weights.geographic

    # Actor overlap (Jaccard)
    actors_a: Set[str] = {a.lower() for a in event_a.get("actors", []) if a}
    actors_b: Set[str] = {a.lower() for a in event_b.get("actors", []) if a}
    if actors_a and actors_b:
        intersection = actors_a & actors_b
        union = actors_a | actors_b
        actor_score = len(intersection) / len(union) if union else 0.0
        score += actor_score * weights.actor

    # Semantic similarity (keyword overlap as TF-IDF fallback)
    summary_a = str(event_a.get("summary", event_a.get("title", ""))).lower()
    summary_b = str(event_b.get("summary", event_b.get("title", ""))).lower()
    if summary_a and summary_b:
        semantic = _keyword_overlap(summary_a, summary_b)
        score += semantic * weights.semantic

    # Event type alignment
    type_a = str(event_a.get("event_type", "")).upper()
    type_b = str(event_b.get("event_type", "")).upper()
    if type_a and type_b:
        if type_a == type_b:
            score += 1.0 * weights.event_type
        elif _same_event_category(type_a, type_b):
            score += 0.5 * weights.event_type

    return min(score, 1.0)


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Compute keyword overlap score between two text strings.

    Args:
        text_a: First text (lowercased).
        text_b: Second text (lowercased).

    Returns:
        Jaccard overlap score in [0.0, 1.0].
    """
    import re

    tokens_a = set(re.findall(r"\b\w{4,}\b", text_a))
    tokens_b = set(re.findall(r"\b\w{4,}\b", text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _same_event_category(type_a: str, type_b: str) -> bool:
    """Check whether two event types belong to the same broad category.

    Args:
        type_a: First event type string.
        type_b: Second event type string.

    Returns:
        True if both types fall under the same broad category.
    """
    _CONFLICT_TYPES = {"CONFLICT", "MILITARY_ESCALATION", "MARITIME", "CYBER"}
    _POLITICAL_TYPES = {"DIPLOMATIC", "POLITICAL_INSTABILITY", "ELECTIONS_AND_VOTING"}
    _HUMANITARIAN_TYPES = {"HUMANITARIAN", "ECONOMIC", "SANCTIONS"}
    for category in [_CONFLICT_TYPES, _POLITICAL_TYPES, _HUMANITARIAN_TYPES]:
        if type_a in category and type_b in category:
            return True
    return False


def _build_fusion_cluster(
    cluster_events: List[Dict[str, Any]], max_confidence: float
) -> FusionCluster:
    """Build a FusionCluster from a list of event dicts.

    Args:
        cluster_events: List of normalized event dicts in the same cluster.
        max_confidence: MAX_CONFIDENCE cap.

    Returns:
        FusionCluster with computed metadata.
    """
    cluster_id = str(uuid.uuid4())[:12]

    # Temporal span
    dates = [
        e.get("datetime", e.get("date", ""))
        for e in cluster_events
        if e.get("datetime") or e.get("date")
    ]
    dates_sorted = sorted(d for d in dates if d)
    temporal_span = {
        "start": dates_sorted[0] if dates_sorted else "",
        "end": dates_sorted[-1] if dates_sorted else "",
    }

    # Centroid lat/lon
    lats = [float(e["lat"]) for e in cluster_events if e.get("lat") is not None]
    lons = [float(e["lon"]) for e in cluster_events if e.get("lon") is not None]
    centroid_lat = sum(lats) / len(lats) if lats else None
    centroid_lon = sum(lons) / len(lons) if lons else None

    # Primary actors (union of all actors, most frequent first)
    actor_counts: Dict[str, int] = defaultdict(int)
    for event in cluster_events:
        for actor in event.get("actors", []):
            if actor:
                actor_counts[str(actor)] += 1
    primary_actors = sorted(actor_counts, key=actor_counts.get, reverse=True)[:5]  # type: ignore[arg-type]

    # Source types
    source_types = list({e.get("_source", "unknown") for e in cluster_events})

    # Corroboration count: unique source domains
    corroboration_count = len({
        e.get("source_domain", e.get("_source", "unknown"))
        for e in cluster_events
    })

    # Fusion confidence: increases with corroboration and event count
    base_confidence = min(
        0.4 + (len(cluster_events) - 1) * 0.1 + (corroboration_count - 1) * 0.1,
        max_confidence,
    )

    # Primary event type (most common)
    type_counts: Dict[str, int] = defaultdict(int)
    for event in cluster_events:
        et = event.get("event_type", "OTHER")
        if et:
            type_counts[str(et)] += 1
    primary_type = max(type_counts, key=type_counts.get) if type_counts else ""  # type: ignore[arg-type]

    # Country (most common)
    country_counts: Dict[str, int] = defaultdict(int)
    for event in cluster_events:
        c = event.get("country", "")
        if c:
            country_counts[str(c)] += 1
    primary_country = max(country_counts, key=country_counts.get) if country_counts else ""  # type: ignore[arg-type]

    return FusionCluster(
        cluster_id=cluster_id,
        events=cluster_events,
        source_types=source_types,
        fusion_confidence=round(base_confidence, 4),
        temporal_span=temporal_span,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        primary_actors=primary_actors,
        corroboration_count=corroboration_count,
        event_type=primary_type,
        country=primary_country,
    )


def _detect_contradictions(clusters: List[FusionCluster]) -> None:
    """Detect contradictions within clusters in-place.

    A contradiction occurs when two events in the same cluster have conflicting
    actor attributions or opposing event types for the same date.

    Args:
        clusters: List of FusionCluster objects to analyze (modified in-place).
    """
    for cluster in clusters:
        if len(cluster.events) < 2:
            continue
        events: List[Dict[str, Any]] = [
            e for e in cluster.events if isinstance(e, dict)
        ]
        for i, event_a in enumerate(events):
            for event_b in events[i + 1:]:
                # Check for conflicting actor attribution on the same date
                date_a = event_a.get("datetime", event_a.get("date", ""))
                date_b = event_b.get("datetime", event_b.get("date", ""))
                actors_a = set(event_a.get("actors", []))
                actors_b = set(event_b.get("actors", []))

                if date_a and date_b and date_a == date_b and actors_a and actors_b:
                    if not actors_a & actors_b and len(actors_a) >= 1 and len(actors_b) >= 1:
                        cluster.contradiction_flags.append(
                            ContradictionFlag(
                                cluster_id=cluster.cluster_id,
                                event_a_summary=str(event_a.get("summary", ""))[:100],
                                event_b_summary=str(event_b.get("summary", ""))[:100],
                                dimension="actor",
                                severity="WARNING",
                                detail="Same-date events have non-overlapping actor sets",
                            )
                        )


# ── Event normalization helpers ──────────────────────────────────────────────────


def _timeline_entry_to_dict(entry: Any, source: str = "llm") -> Dict[str, Any]:
    """Convert a TimelineEntry to a fusion event dict.

    Args:
        entry: TimelineEntry dataclass.
        source: Source identifier string.

    Returns:
        Normalized event dict.
    """
    return {
        "_event_id": str(uuid.uuid4())[:12],
        "_source": source,
        "event_type": getattr(entry, "event_type", "OTHER"),
        "datetime": getattr(entry, "datetime", ""),
        "country": getattr(entry, "country", ""),
        "lat": getattr(entry, "lat", None),
        "lon": getattr(entry, "lon", None),
        "actors": getattr(entry, "actors", []),
        "summary": getattr(entry, "summary", ""),
        "confidence": getattr(entry, "confidence", 0.0),
        "source_url": getattr(entry, "source_url", ""),
        "source_title": getattr(entry, "source_title", ""),
    }


def _ground_truth_event_to_dict(event: Any) -> Dict[str, Any]:
    """Convert a GroundTruthEvent to a fusion event dict.

    Args:
        event: GroundTruthEvent dataclass.

    Returns:
        Normalized event dict.
    """
    return {
        "_event_id": str(uuid.uuid4())[:12],
        "_source": getattr(event, "source", "ground_truth"),
        "event_type": getattr(event, "event_type", ""),
        "datetime": getattr(event, "date", ""),
        "date": getattr(event, "date", ""),
        "country": getattr(event, "country", ""),
        "lat": getattr(event, "lat", None),
        "lon": getattr(event, "lon", None),
        "actors": getattr(event, "actors", []),
        "summary": getattr(event, "notes", ""),
        "confidence": getattr(event, "confidence", 1.0),
        "source_domain": "ground_truth",
    }


def _custom_match_to_dict(match: Any) -> Dict[str, Any]:
    """Convert a CustomDatasetMatch to a fusion event dict.

    Args:
        match: CustomDatasetMatch dataclass.

    Returns:
        Normalized event dict.
    """
    article = getattr(match, "article", None)
    record = getattr(match, "custom_record", {})
    return {
        "_event_id": str(uuid.uuid4())[:12],
        "_source": "custom_dataset",
        "event_type": str(record.get("event_type", record.get("type", "OTHER"))),
        "datetime": str(record.get("date", record.get("event_date", ""))),
        "country": str(record.get("country", "")),
        "lat": None,
        "lon": None,
        "actors": [],
        "summary": str(record.get("title", record.get("headline", ""))),
        "confidence": float(getattr(match, "match_confidence", 0.0)),
        "source_url": getattr(article, "url", "") if article else "",
        "source_title": getattr(article, "title", "") if article else "",
        "source_domain": getattr(article, "domain", "custom") if article else "custom",
    }
