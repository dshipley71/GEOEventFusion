"""CustomDatasetAgent — Cross-reference proprietary or user-provided datasets.

Implements AGENTS.md §2.4 specification:
- CSV, JSON, SQL, and API dataset loading
- Title substring and Levenshtein similarity matching
- Actor name overlap scoring
- Temporal proximity matching
- Configurable confidence scoring
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geoeventfusion.agents.base import BaseAgent
from geoeventfusion.models.events import (
    Article,
    CustomDatasetAgentResult,
    CustomDatasetMatch,
)
from geoeventfusion.utils.levenshtein_utils import similarity

logger = logging.getLogger(__name__)


class CustomDatasetAgent(BaseAgent):
    """Cross-reference proprietary or user-provided datasets against article pools.

    Supports CSV, JSON, SQLite, and generic REST API inputs. Matches records
    against GDELT article pools using title similarity, actor overlap, and
    temporal proximity.
    """

    name = "CustomDatasetAgent"
    version = "1.0.0"

    def run(self, context: Any) -> CustomDatasetAgentResult:
        """Load and match custom dataset records against GDELT article pools.

        Args:
            context: PipelineContext with config and gdelt_result.

        Returns:
            CustomDatasetAgentResult with matched records and match statistics.
        """
        cfg = context.config
        result = CustomDatasetAgentResult()

        if not cfg.custom_dataset_path:
            logger.info("CustomDatasetAgent: no custom dataset path configured — skipping")
            result.warnings.append("No custom dataset configured (custom_dataset_path not set)")
            return result

        # Load custom records
        try:
            records = self._load_dataset(cfg.custom_dataset_path, cfg.custom_dataset_format)
        except Exception as exc:
            logger.warning("CustomDatasetAgent: failed to load dataset: %s", exc)
            result.warnings.append(f"Dataset load failed: {exc}")
            return result

        if not records:
            logger.info("CustomDatasetAgent: dataset empty or unreadable")
            result.warnings.append("Custom dataset is empty")
            return result

        # Gather all articles from GDELT result for matching
        articles: List[Article] = []
        if context.gdelt_result:
            articles = context.gdelt_result.all_articles()

        if not articles:
            logger.info("CustomDatasetAgent: no GDELT articles for matching")
            result.unmatched_records = records
            return result

        matches: List[CustomDatasetMatch] = []
        unmatched: List[Dict[str, Any]] = []

        for record in records:
            match = self._find_best_match(record, articles, cfg)
            if match is not None:
                matches.append(match)
            else:
                unmatched.append(record)

        result.matches = matches
        result.unmatched_records = unmatched
        total = len(records)
        result.match_rate = len(matches) / total if total > 0 else 0.0

        logger.info(
            "CustomDatasetAgent: %d/%d records matched (rate=%.2f%%)",
            len(matches),
            total,
            result.match_rate * 100,
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _load_dataset(
        self, path: str, fmt: str
    ) -> List[Dict[str, Any]]:
        """Load records from the configured dataset.

        Args:
            path: File path or API URL.
            fmt: Format string: "csv", "json", "sql", or "api".

        Returns:
            List of record dicts.
        """
        fmt_lower = fmt.lower().strip()
        if fmt_lower == "csv":
            return _load_csv(path)
        if fmt_lower in ("json", "jsonl", "ndjson"):
            return _load_json_file(path)
        if fmt_lower == "sql":
            return _load_sqlite(path)
        if fmt_lower == "api":
            return _load_api(path)
        logger.warning("CustomDatasetAgent: unknown format '%s' — trying CSV", fmt)
        return _load_csv(path)

    def _find_best_match(
        self,
        record: Dict[str, Any],
        articles: List[Article],
        cfg: Any,
    ) -> Optional[CustomDatasetMatch]:
        """Find the best article match for a custom record.

        Uses title similarity, actor overlap, and temporal proximity.

        Args:
            record: Custom dataset record dict.
            articles: Pool of GDELT articles to match against.
            cfg: PipelineConfig for thresholds.

        Returns:
            CustomDatasetMatch if best confidence exceeds threshold, else None.
        """
        record_title = str(record.get("title", record.get("headline", record.get("name", ""))))
        record_date = str(record.get("date", record.get("event_date", "")))
        record_actors = _extract_record_actors(record)

        best_score = 0.0
        best_article: Optional[Article] = None
        best_dimensions: List[str] = []

        for article in articles:
            dims: List[str] = []
            score = 0.0

            # Title similarity (weight 0.5)
            if record_title and article.title:
                title_sim = similarity(record_title, article.title)
                if title_sim > 0.3:
                    score += title_sim * 0.5
                    dims.append("title_similarity")

            # Actor overlap — Jaccard over actor name sets (weight 0.3)
            article_actors = _extract_article_actors(article)
            if record_actors and article_actors:
                intersection = record_actors & article_actors
                union = record_actors | article_actors
                actor_score = len(intersection) / len(union) if union else 0.0
                if actor_score > 0.0:
                    score += actor_score * 0.3
                    dims.append("actor_overlap")

            # Temporal proximity (weight 0.2)
            if record_date and article.published_at:
                from geoeventfusion.utils.date_utils import date_delta_days

                delta = date_delta_days(record_date, article.published_at)
                if delta is not None and delta <= cfg.validation_date_delta_days:
                    temporal_score = 1.0 - (delta / max(1, cfg.validation_date_delta_days))
                    score += temporal_score * 0.2
                    dims.append("temporal")

            if score > best_score:
                best_score = score
                best_article = article
                best_dimensions = dims

        threshold = getattr(cfg, "validation_custom_match_threshold", 0.50)
        if best_score >= threshold and best_article is not None:
            return CustomDatasetMatch(
                article=best_article,
                custom_record=record,
                match_confidence=round(best_score, 4),
                match_dimensions=best_dimensions,
            )
        return None


# ── Dataset loaders ──────────────────────────────────────────────────────────────


def _load_csv(path: str) -> List[Dict[str, Any]]:
    """Load records from a CSV file.

    Args:
        path: File path to the CSV.

    Returns:
        List of record dicts.
    """
    import csv

    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    except (OSError, csv.Error) as exc:
        logger.warning("CustomDatasetAgent: CSV load failed: %s", exc)
    return records


def _load_json_file(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSON or NDJSON file.

    Args:
        path: File path to the JSON/NDJSON file.

    Returns:
        List of record dicts.
    """
    import json

    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Try array first
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records = [r for r in parsed if isinstance(r, dict)]
        elif isinstance(parsed, dict):
            records = [parsed]
    except json.JSONDecodeError:
        # Try NDJSON (newline-delimited)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            import json as _json
                            obj = _json.loads(line)
                            if isinstance(obj, dict):
                                records.append(obj)
                        except json.JSONDecodeError:
                            pass
        except OSError as exc:
            logger.warning("CustomDatasetAgent: NDJSON load failed: %s", exc)
    except OSError as exc:
        logger.warning("CustomDatasetAgent: JSON file load failed: %s", exc)
    return records


def _load_sqlite(path: str) -> List[Dict[str, Any]]:
    """Load records from a SQLite database (first table found).

    Args:
        path: Path to SQLite file, optionally with query suffix (path|query).

    Returns:
        List of record dicts.
    """
    import sqlite3

    # Support "path|SELECT ..." syntax
    if "|" in path:
        db_path, query = path.split("|", 1)
    else:
        db_path = path
        query = None

    records: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if query:
            cur.execute(query)
        else:
            # Use the first table found
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            row = cur.fetchone()
            if row is None:
                return []
            table_name = row[0]
            cur.execute(f"SELECT * FROM {table_name}")  # noqa: S608 — internal path
        rows = cur.fetchall()
        records = [dict(r) for r in rows]
        conn.close()
    except Exception as exc:
        logger.warning("CustomDatasetAgent: SQLite load failed: %s", exc)
    return records


def _load_api(url: str) -> List[Dict[str, Any]]:
    """Load records from a generic REST API endpoint.

    Args:
        url: API URL (GET request, no auth).

    Returns:
        List of record dicts.
    """
    import json

    import requests  # type: ignore[import-untyped]

    records: List[Dict[str, Any]] = []
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning("CustomDatasetAgent: API returned HTTP %d", resp.status_code)
            return []
        data = resp.json()
        if isinstance(data, list):
            records = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            # Common pagination wrappers
            for key in ("data", "results", "items", "records", "events"):
                if key in data and isinstance(data[key], list):
                    records = [r for r in data[key] if isinstance(r, dict)]
                    break
    except Exception as exc:
        logger.warning("CustomDatasetAgent: API load failed for %s: %s", url, exc)
    return records


def _extract_record_actors(record: Dict[str, Any]) -> set:
    """Extract actor names from a custom record dict.

    Checks common actor-related field names.

    Args:
        record: Custom dataset record.

    Returns:
        Set of lowercased actor name strings.
    """
    actors: set = set()
    for key in ("actors", "actor", "actor1", "actor2", "parties", "source", "target"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            actors.update(a.strip().lower() for a in val.split(",") if a.strip())
        elif isinstance(val, list):
            actors.update(str(a).strip().lower() for a in val if a)
    return actors


def _extract_article_actors(article: Article) -> set:
    """Extract actor keywords from an article title.

    Args:
        article: Article object.

    Returns:
        Set of lowercased capitalized tokens from the title.
    """
    import re

    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", article.title)
    return {t.lower() for t in tokens}
