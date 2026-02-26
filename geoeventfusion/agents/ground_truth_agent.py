"""GroundTruthAgent — Validated conflict event datasets from ACLED and ICEWS.

Implements AGENTS.md §2.3 specification:
- ACLED REST API integration with country and date-range filtering
- ICEWS file-based ingestion with TSV/CSV loading
- Schema normalization to unified GroundTruthEvent format
- Temporal alignment to pipeline analysis window
- Geographic filtering by country
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geoeventfusion.agents.base import BaseAgent
from geoeventfusion.clients.ground_truth_client import ACLEDClient, ICEWSClient
from geoeventfusion.models.events import GroundTruthAgentResult, GroundTruthEvent
from geoeventfusion.utils.date_utils import parse_date_range

logger = logging.getLogger(__name__)


class GroundTruthAgent(BaseAgent):
    """Provide validated conflict event datasets for calibration and cross-verification.

    Fetches from ACLED (REST API) and/or ICEWS (file-based) depending on which
    sources are configured in PipelineConfig.ground_truth_sources.
    """

    name = "GroundTruthAgent"
    version = "1.0.0"

    def run(self, context: Any) -> GroundTruthAgentResult:
        """Fetch and normalize conflict events from configured ground-truth sources.

        Args:
            context: PipelineContext with config and run metadata.

        Returns:
            GroundTruthAgentResult with normalized events and source counts.
        """
        cfg = context.config
        result = GroundTruthAgentResult()

        if not cfg.ground_truth_sources:
            logger.info("GroundTruthAgent: no sources configured — skipping")
            result.warnings.append("No ground-truth sources configured")
            return result

        start_date, end_date = parse_date_range(cfg.days_back)
        country_filter = cfg.ground_truth_country_filter or []
        event_types = cfg.ground_truth_event_types or []

        for source in cfg.ground_truth_sources:
            source_lower = source.lower().strip()
            if source_lower == "acled":
                acled_events = self._fetch_acled(
                    cfg, start_date, end_date, country_filter, event_types
                )
                result.events.extend(acled_events)
                result.source_counts["acled"] = len(acled_events)
            elif source_lower == "icews":
                icews_events = self._fetch_icews(cfg)
                result.events.extend(icews_events)
                result.source_counts["icews"] = len(icews_events)
            else:
                logger.warning(
                    "GroundTruthAgent: unknown source '%s' — skipping", source
                )
                result.warnings.append(f"Unknown ground-truth source: {source}")

        total = len(result.events)
        if total == 0:
            result.warnings.append(
                "No ground-truth events found in the analysis window"
            )

        logger.info(
            "GroundTruthAgent: %d total events | sources=%s",
            total,
            list(result.source_counts.keys()),
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _fetch_acled(
        self,
        cfg: Any,
        start_date: str,
        end_date: str,
        country_filter: List[str],
        event_types: List[str],
    ) -> List[GroundTruthEvent]:
        """Fetch and normalize events from the ACLED API.

        Args:
            cfg: PipelineConfig.
            start_date: ISO start date.
            end_date: ISO end date.
            country_filter: List of country names to filter.
            event_types: ACLED event type strings.

        Returns:
            List of normalized GroundTruthEvent objects.
        """
        client = ACLEDClient(
            api_key=cfg.acled_api_key or "",
            email=cfg.acled_email or "",
        )

        if not client.api_key or not client.email:
            logger.warning(
                "GroundTruthAgent: ACLED credentials not configured — skipping ACLED"
            )
            return []

        events: List[GroundTruthEvent] = []

        # Fetch per country if specified, otherwise fetch globally
        fetch_countries: List[Optional[str]] = (
            [c for c in country_filter] if country_filter else [None]
        )

        for country in fetch_countries:
            try:
                raw_events = client.fetch_events(
                    country=country,
                    start_date=start_date,
                    end_date=end_date,
                    event_types=event_types if event_types else None,
                )
                normalized = [_normalize_acled_event(r) for r in raw_events]
                events.extend([e for e in normalized if e is not None])
            except Exception as exc:
                logger.warning(
                    "GroundTruthAgent: ACLED fetch failed for country=%s: %s",
                    country,
                    exc,
                )

        logger.info("GroundTruthAgent: ACLED returned %d events", len(events))
        return events

    def _fetch_icews(self, cfg: Any) -> List[GroundTruthEvent]:
        """Load and normalize events from an ICEWS data file.

        Args:
            cfg: PipelineConfig with custom_dataset_path as ICEWS file path if set.

        Returns:
            List of normalized GroundTruthEvent objects.
        """
        # Use custom_dataset_path as the ICEWS file path when source is icews
        icews_path = getattr(cfg, "icews_data_path", None)
        if not icews_path:
            logger.warning(
                "GroundTruthAgent: ICEWS source configured but no icews_data_path set"
            )
            return []

        client = ICEWSClient(data_path=icews_path)
        try:
            raw_events = client.load_from_file(icews_path)
        except Exception as exc:
            logger.warning("GroundTruthAgent: ICEWS load failed: %s", exc)
            return []

        events: List[GroundTruthEvent] = []
        for raw in raw_events:
            normalized = _normalize_icews_event(raw)
            if normalized is not None:
                events.extend(normalized)

        logger.info("GroundTruthAgent: ICEWS returned %d events", len(events))
        return events


# ── Schema normalization helpers ────────────────────────────────────────────────


def _normalize_acled_event(raw: Dict[str, Any]) -> Optional[GroundTruthEvent]:
    """Normalize a raw ACLED API record to GroundTruthEvent.

    Args:
        raw: Raw ACLED event dict from the API.

    Returns:
        GroundTruthEvent, or None if required fields are missing.
    """
    event_id = str(raw.get("event_id_cnty", raw.get("id", "")))
    date_raw = raw.get("event_date", "")
    event_type = str(raw.get("event_type", ""))
    country = str(raw.get("country", ""))

    if not event_id or not date_raw:
        return None

    from geoeventfusion.utils.date_utils import normalize_date_str

    date = normalize_date_str(str(date_raw))

    lat: Optional[float] = None
    lon: Optional[float] = None
    try:
        lat_raw = raw.get("latitude")
        lon_raw = raw.get("longitude")
        if lat_raw is not None:
            lat = float(lat_raw)
        if lon_raw is not None:
            lon = float(lon_raw)
    except (ValueError, TypeError):
        pass

    actors: List[str] = []
    for key in ("actor1", "actor2", "assoc_actor_1", "assoc_actor_2"):
        val = raw.get(key, "")
        if val and str(val).strip():
            actors.append(str(val).strip())

    fatalities = 0
    try:
        fatalities = int(raw.get("fatalities", 0) or 0)
    except (ValueError, TypeError):
        pass

    return GroundTruthEvent(
        event_id=event_id,
        source="acled",
        event_type=event_type,
        date=date,
        country=country,
        region=str(raw.get("region", "")),
        lat=lat,
        lon=lon,
        actors=actors,
        fatalities=fatalities,
        notes=str(raw.get("notes", "")),
        confidence=1.0,  # ACLED events are ground truth
    )


def _normalize_icews_event(raw: Dict[str, Any]) -> Optional[List[GroundTruthEvent]]:
    """Normalize a raw ICEWS file record to GroundTruthEvent(s).

    ICEWS uses CAMEO coding. Multiple actors may be referenced in a single row.

    Args:
        raw: Raw ICEWS row dict from CSV/TSV file.

    Returns:
        List with one GroundTruthEvent, or None if required fields missing.
    """
    from geoeventfusion.utils.date_utils import normalize_date_str

    event_id = str(raw.get("Event ID", raw.get("id", "")))
    date_raw = str(raw.get("Event Date", raw.get("date", "")))
    event_type = str(raw.get("Event Type", raw.get("cameo_code", "")))
    country = str(raw.get("Country", raw.get("country", "")))

    if not event_id or not date_raw:
        return None

    date = normalize_date_str(date_raw)

    actors: List[str] = []
    for key in ("Source Name", "Target Name", "source", "target"):
        val = raw.get(key, "")
        if val and str(val).strip():
            actors.append(str(val).strip())

    lat: Optional[float] = None
    lon: Optional[float] = None
    try:
        lat_raw = raw.get("Latitude", raw.get("lat"))
        lon_raw = raw.get("Longitude", raw.get("lon"))
        if lat_raw:
            lat = float(lat_raw)
        if lon_raw:
            lon = float(lon_raw)
    except (ValueError, TypeError):
        pass

    return [GroundTruthEvent(
        event_id=event_id,
        source="icews",
        event_type=event_type,
        date=date,
        country=country,
        lat=lat,
        lon=lon,
        actors=actors,
        notes=str(raw.get("Story", raw.get("notes", ""))),
        confidence=1.0,
    )]
