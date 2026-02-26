"""Shared pytest fixtures for GEOEventFusion tests.

All fixtures follow the conventions in CLAUDE.md section 11:
- Fixture data lives in tests/fixtures/ as static JSON files
- mock_llm_client returns pre-defined JSON without real API calls
- mock_gdelt_client mocks HTTP calls at the requests.Session level
- No real external HTTP calls are made in any test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Raw fixture data loaders ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def artlist_raw() -> Dict[str, Any]:
    """Raw GDELT ArtList API response dict loaded from fixture JSON."""
    with open(_FIXTURES_DIR / "sample_artlist.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def timeline_raw() -> List[Dict[str, Any]]:
    """Raw timeline volinfo steps loaded from fixture JSON (30 steps, 2 clear spikes)."""
    with open(_FIXTURES_DIR / "sample_timeline_volinfo.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tonechart_raw() -> List[Dict[str, Any]]:
    """Raw ToneChart bins loaded from fixture JSON."""
    with open(_FIXTURES_DIR / "sample_tonechart.json", encoding="utf-8") as f:
        return json.load(f)


# ── Model object fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_timeline_steps(timeline_raw):
    """30-step TimelineStep list with 2 clear spikes (at index 10 and 25).

    Spike at 2024-01-31: value 9.0 (Z ≈ 4.04)
    Spike at 2024-03-16: value 8.0 (Z ≈ 3.43)
    All other steps: value 2.0
    """
    from geoeventfusion.models.events import TimelineStep

    return [
        TimelineStep(date=s["date"], value=s["value"], label=s.get("label", ""))
        for s in timeline_raw
    ]


@pytest.fixture(scope="session")
def sample_tonechart_bins(tonechart_raw):
    """ToneChartBin list from fixture — negatively skewed distribution."""
    from geoeventfusion.models.events import ToneChartBin

    return [ToneChartBin(tone_value=b["tone_value"], count=b["count"]) for b in tonechart_raw]


@pytest.fixture(scope="session")
def sample_articles(artlist_raw):
    """List of Article objects parsed from fixture ArtList response."""
    from geoeventfusion.models.events import Article

    articles = []
    for raw in artlist_raw.get("articles", []):
        articles.append(
            Article(
                url=raw["url"],
                title=raw["title"],
                published_at=raw.get("seendate", ""),
                source=raw.get("domain", ""),
                domain=raw.get("domain", ""),
                language=raw.get("language", ""),
                source_country=raw.get("sourcecountry", ""),
                tone=raw.get("tone"),
            )
        )
    return articles


@pytest.fixture(scope="session")
def sample_timeline_volraw():
    """5-step TimelineStepRaw list for vol_ratio computation tests."""
    from geoeventfusion.models.events import TimelineStepRaw

    return [
        TimelineStepRaw(date="2024-01-01", volume=50, norm=10000),
        TimelineStepRaw(date="2024-01-04", volume=60, norm=10500),
        TimelineStepRaw(date="2024-01-07", volume=55, norm=10200),
        TimelineStepRaw(date="2024-01-10", volume=250, norm=11000),  # spike
        TimelineStepRaw(date="2024-01-13", volume=45, norm=9800),
    ]


@pytest.fixture(scope="session")
def sample_co_occurrence_triples():
    """Actor co-occurrence triples for actor graph tests."""
    return [
        ("Houthi", "United States", "2024-01-15"),
        ("Houthi", "Yemen", "2024-01-15"),
        ("United States", "Yemen", "2024-01-15"),
        ("Houthi", "United States", "2024-01-16"),
        ("Houthi", "United Kingdom", "2024-01-17"),
        ("United States", "United Kingdom", "2024-01-18"),
        ("United States", "European Union", "2024-01-20"),
        ("Houthi", "European Union", "2024-01-21"),
        ("United Kingdom", "European Union", "2024-01-22"),
        ("Iran", "Houthi", "2024-01-25"),
        ("Iran", "United States", "2024-01-25"),
        ("Iran", "Yemen", "2024-01-26"),
    ]


# ── Mock LLM client ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock LLMClient that returns pre-defined JSON without real API calls.

    Returns structured responses suitable for extraction and hypothesis tests.
    """
    from geoeventfusion.clients.llm_client import LLMClient

    client = MagicMock(spec=LLMClient)
    client.backend = "mock"
    client.max_confidence = 0.82

    # Default call() response — plain text
    client.call.return_value = '{"status": "ok"}'

    # Default call_json() response — parsed dict
    client.call_json.return_value = {
        "events": [
            {
                "event_type": "CONFLICT",
                "datetime": "2024-01-15",
                "country": "Yemen",
                "lat": 15.5,
                "lon": 48.5,
                "actors": ["Houthi", "United States"],
                "summary": "Houthi forces attacked merchant vessel in Red Sea",
                "confidence": 0.75,
                "source_url": "https://www.reuters.com/world/middle-east/test",
                "source_title": "Houthi forces attack merchant vessel in Red Sea",
            }
        ]
    }

    # enforce_confidence_cap: pass through with cap applied
    def _cap(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "confidence" and isinstance(v, (int, float)):
                    data[k] = min(float(v), 0.82)
                else:
                    data[k] = _cap(v)
        elif isinstance(data, list):
            data = [_cap(item) for item in data]
        return data

    client.enforce_confidence_cap.side_effect = _cap
    return client


# ── Mock GDELT client ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gdelt_client(artlist_raw, timeline_raw, tonechart_raw):
    """Mock GDELTClient.fetch() that returns fixture data without HTTP calls.

    Dispatches based on `mode`:
    - ArtList     → returns artlist_raw
    - Timeline*   → returns timeline data
    - ToneChart   → returns tonechart data
    """
    from geoeventfusion.clients.gdelt_client import GDELTClient

    client = MagicMock(spec=GDELTClient)

    def _fake_fetch(query, mode, **kwargs):
        if mode == "ArtList":
            return artlist_raw
        if mode in ("TimelineVolInfo", "TimelineVolRaw", "TimelineTone",
                    "TimelineLang", "TimelineSourceCountry"):
            return {"timeline": [{"date": s["date"], "value": s["value"]} for s in timeline_raw]}
        if mode == "ToneChart":
            return {"tonechart": tonechart_raw}
        if mode == "ImageCollageInfo":
            return {
                "images": [
                    {
                        "url": "https://example.com/image1.jpg",
                        "pageurl": "https://example.com/article1",
                        "title": "Test article with image",
                        "imagetags": "military,protest",
                        "imagewebcount": 5,
                        "priorwebUrls": [],
                        "exifcapturedate": None,
                        "seendate": "2024-01-15",
                    }
                ]
            }
        return {}

    client.fetch.side_effect = _fake_fetch
    return client


# ── Pipeline config fixture ──────────────────────────────────────────────────────

@pytest.fixture
def test_pipeline_config():
    """Minimal PipelineConfig for testing — test_mode=True, no API calls."""
    from config.settings import PipelineConfig

    return PipelineConfig(
        query="Houthi Red Sea attacks",
        days_back=90,
        max_records=10,
        test_mode=True,
        llm_backend="ollama",
        log_level="WARNING",
    )


@pytest.fixture
def test_pipeline_context(test_pipeline_config, tmp_path):
    """PipelineContext wired to a temp output directory for isolation."""
    from geoeventfusion.models.pipeline import PipelineContext

    return PipelineContext(
        config=test_pipeline_config,
        run_id="20240115_120000_test",
        output_dir=tmp_path / "outputs",
    )


# ── Response mock helper for GDELT HTTP tests ────────────────────────────────────

@pytest.fixture
def gdelt_http_mock():
    """Context manager that patches requests.Session.get with a configurable response.

    Usage:
        def test_something(gdelt_http_mock):
            with gdelt_http_mock(status_code=200, text='{"articles":[]}') as mock_get:
                ...
    """
    import requests

    class _HttpMockContext:
        def __call__(self, status_code: int = 200, text: str = "{}"):
            mock_resp = MagicMock()
            mock_resp.status_code = status_code
            mock_resp.text = text
            return patch.object(
                requests.Session, "get", return_value=mock_resp
            )

    return _HttpMockContext()
