# AGENTS.md
## GEOEventFusion — Agent Architecture Specification
Version: 2.0
Date: 2026-02-25

---

# 1. Overview

GEOEventFusion is a modular, multi-agent geopolitical intelligence engine that transforms raw
global event signals into validated, evidence-grounded intelligence storyboards.

The system is composed of specialized agents responsible for detection, enrichment, extraction,
fusion, validation, and export. Each agent has a clearly defined responsibility, produces
structured outputs, operates independently, and can be replaced or scaled without breaking
the pipeline.

---

## 1.1 Pipeline Flow

```
GDELTAgent ──► RSSAgent ──► GroundTruthAgent ──► CustomDatasetAgent
                                                         │
                                                         ▼
                                               LLMExtractionAgent
                                                         │
                                                         ▼
                                                   FusionAgent
                                                         │
                                                   StoryboardAgent
                                                         │
                                                  ValidationAgent
                                                         │
                                                   ExportAgent
```

Every agent receives the shared `PipelineContext` object and appends its output to it.
The pipeline is resumable: each agent checks for cached intermediate outputs before
executing, enabling phase-level restarts without re-running the full pipeline.

---

## 1.2 Base Agent Contract

All agents inherit from `BaseAgent` and implement the following interface:

```python
class BaseAgent(ABC):
    name: str                          # Unique agent identifier
    version: str                       # Semantic version of this agent
    config: PipelineConfig             # Shared configuration

    @abstractmethod
    def run(self, context: PipelineContext) -> AgentResult:
        """Execute the agent. Returns a typed AgentResult."""

    def validate_output(self, result: AgentResult) -> bool:
        """Post-run validation of structured output. Returns True if valid."""

    def reset(self) -> None:
        """Clear internal state for re-use or retry."""
```

---

# 2. Agent Definitions

---

## 2.1 GDELTAgent

**Module:** `geoeventfusion/agents/gdelt_agent.py`  
**Purpose:** Detect anomalous event activity from GDELT data and produce normalized article pools, timeline signals, visual intelligence, and spike windows — using the full capability surface of the GDELT DOC 2.0 API.

---

### 2.1.1 Inputs

| Field                     | Type              | Description                                                              |
|---------------------------|-------------------|--------------------------------------------------------------------------|
| `query`                   | `str`             | Base keyword/keyphrase search query                                      |
| `final_query`             | `str`             | Operator-augmented query (built via `QueryBuilder`)                      |
| `days_back`               | `int`             | Analysis window in days (default: 90; GDELT max: ~90)                    |
| `max_records`             | `int`             | Per-fetch article cap (default: 250; GDELT max: 250)                     |
| `spike_z_threshold`       | `float`           | Z-score threshold for spike detection (default: 1.5)                     |
| `domain_cap_pct`          | `float`           | Domain diversity cap — max fraction from any one domain (default: 0.20)  |
| `timeline_smooth`         | `int`             | GDELT timeline smoothing window in steps (default: 3; max: 30)           |
| `repeat_threshold`        | `int`             | Minimum keyword repetitions for `repeat<N>` operator (default: 3)        |
| `tone_negative_threshold` | `float`           | Tone ceiling for high-negative pool via `tone<` operator (default: -5.0) |
| `toneabs_threshold`       | `float`           | Minimum absolute tone for high-emotion pool via `toneabs>` (default: 8.0)|
| `source_country_filter`   | `Optional[str]`   | FIPS country code for country-scoped fetch (e.g., `"RS"`, `"CH"`)        |
| `source_lang_filter`      | `Optional[str]`   | 3-char ISO language code for language-scoped fetch (e.g., `"ara"`)       |
| `authoritative_domains`   | `List[str]`       | Domains for `domainis:` authority fetch (e.g., `["un.org","state.gov"]`) |
| `visual_imagetags`        | `List[str]`       | VGKG imagetag values for visual intelligence fetch (e.g., `["military","protest","fire"]`) |
| `enable_visual_intel`     | `bool`            | Enable visual intelligence fetch modes (default: `False`)                |
| `enable_word_clouds`      | `bool`            | Enable image word cloud modes (default: `False`)                         |

---

### 2.1.2 Fetch Architecture — 13-Mode Parallel Execution

The agent executes up to 13 parallel GDELT DOC 2.0 API calls, organized into four groups.
All fetches use staggered submission (0.75s between calls) with `max_workers=2` to respect
GDELT's rate limits. Fetch groups B–D are gated by configuration flags and activate only
when the relevant inputs are provided.

#### Group A — Core Article Pools (always active)

| Pool Key               | Mode      | Sort / Filter                                 | Purpose                                          |
|------------------------|-----------|-----------------------------------------------|--------------------------------------------------|
| `articles_recent`      | ArtList   | `DateDesc`                                    | Chronological coverage — full recency window     |
| `articles_negative`    | ArtList   | `ToneAsc`                                     | Most negatively-toned articles                   |
| `articles_positive`    | ArtList   | `ToneDesc`                                    | Most positively-toned articles                   |
| `articles_relevant`    | ArtList   | `HybridRel`                                   | Prominence-ranked — top outlets, high relevance  |
| `articles_high_neg`    | ArtList   | `DateDesc` + `tone<{threshold}` query filter  | Hard negative-tone filter; higher recall than ToneAsc |
| `articles_high_emotion`| ArtList   | `DateDesc` + `toneabs>{threshold}` query filter | Crisis/high-emotion coverage regardless of polarity |

**Why `HybridRel`?** `ToneAsc`/`ToneDesc` sorts surface emotionally extreme articles from obscure
outlets. `HybridRel` (available for content after 2018-09-16) combines textual relevance with
outlet prominence, returning the same event as covered by major news organizations — the most
intelligence-actionable pool for cross-source corroboration.

**Why inline `tone<` vs. `ToneAsc` sort?** The `tone<` query operator applies a hard threshold
across the full article set before sorting, enabling a larger and more diverse negative-tone pool
than sorting alone. Both are retained because they select different article subsets.

#### Group B — Timeline and Signal Modes (always active)

| Key                  | Mode                    | Description                                             |
|----------------------|-------------------------|---------------------------------------------------------|
| `timeline_volinfo`   | TimelineVolInfo         | Volume % of all GDELT coverage + top-10 articles per step |
| `timeline_volraw`    | TimelineVolRaw          | Absolute article counts + `norm` (total monitored articles per step) |
| `timeline_tone`      | TimelineTone            | Average tone over time                                  |
| `timeline_lang`      | TimelineLang            | Coverage volume broken down by language                 |
| `timeline_country`   | TimelineSourceCountry   | Coverage volume broken down by source country           |
| `tonechart`          | ToneChart               | Full tone distribution histogram (−100 to +100 bins)    |

**Why `TimelineVolRaw`?** The existing `TimelineVolInfo` returns volume as a percentage of all
GDELT-monitored coverage, normalizing out global news volume fluctuations. `TimelineVolRaw`
returns the absolute article count alongside a `norm` field (total articles in that interval).
Together they enable both relative-prominence analysis (is this story crowding out other news?)
and absolute-volume analysis (how many articles actually exist?).

#### Group C — Targeted Source Fetches (conditional — activated by config)

These supplemental fetches add source-scoped article pools. Each is a separate ArtList call
with a `sourcecountry:`, `sourcelang:`, or `domainis:` operator prepended to the query.

| Activation Condition          | Pool Key                   | Operator Used               | Intelligence Value                                      |
|-------------------------------|----------------------------|-----------------------------|---------------------------------------------------------|
| `source_country_filter` set   | `articles_source_country`  | `sourcecountry:<FIPS>`      | Local press narrative from a specific country           |
| `source_lang_filter` set      | `articles_source_lang`     | `sourcelang:<ISO>`          | Regional-language coverage for local perspective        |
| `authoritative_domains` set   | `articles_authoritative`   | `domainis:<domain>` OR'd    | Official/government/IGO source pool                     |

**`sourcecountry:` use case:** When analyzing a bilateral dispute (e.g., India–Pakistan), run
separate fetches for `sourcecountry:IN` and `sourcecountry:PK` to compare domestic narratives
directly rather than relying on English-language coverage of both.

**`domainis:` use case:** Fetch `un.org`, `state.gov`, `mod.gov.uk`, `nato.int` directly to
build an authoritative-source pool with higher credibility for the `ValidationAgent`.

#### Group D — Visual Intelligence Modes (conditional — activated by `enable_visual_intel=True`)

| Pool Key              | Mode              | Query Construction                                   | Output                                                  |
|-----------------------|-------------------|------------------------------------------------------|---------------------------------------------------------|
| `visual_images`       | ImageCollageInfo  | `imagetag:"<tag>" OR imagetag:"<tag>"` from `visual_imagetags` | `VisualImage` objects with reverse-image-search data |
| `visual_image_topics` | WordCloudImageTags | Base query                                          | Deep-learning visual topic frequency histogram          |

**Visual intelligence for geoint:** Conflict analysis specifically benefits from visual evidence.
Configuring `visual_imagetags` with values like `["military", "weapon", "protest", "explosion",
"fire", "rubble", "crowd", "flag"]` returns VGKG-processed images from matching articles,
each annotated with:
- Number of times Google has seen the image on the web (novelty indicator)
- Up to 6 prior web appearances (provenance trail)
- Embedded EXIF capture date with a staleness warning if photo predates article by >72 hours
- Google Cloud Vision deep-learning tags (what the image actually depicts)

This directly supports the `ValidationAgent`'s evidence-grounding checks — a novel image
appearing in a breaking story is stronger evidence than a stock photo seen 500+ times.

---

### 2.1.3 Query Operator Capabilities

`QueryBuilder` (`geoeventfusion/analysis/query_builder.py`) constructs the full enriched query
string by composing applicable operators. Operators are additive — each adds a constraint or
context layer to the base query.

#### Text Search Operators

| Operator         | Syntax Example                        | Use in GEOEventFusion                                        |
|------------------|---------------------------------------|--------------------------------------------------------------|
| Phrase           | `"red sea attacks"`                   | Exact phrase anchor for the core query                       |
| Boolean OR       | `(houthi OR "ansar allah" OR huthis)` | Alias expansion — ensures all spellings of a group are caught |
| Exclusion        | `-sourcelang:spanish`                 | Remove languages causing false positives for a specific query |
| `theme:`         | `theme:MARITIME_SECURITY`             | LLM-suggested GKG codes — broadens recall beyond exact phrases |
| `near<N>:`       | `near15:"houthi shipping"`            | Proximity search — terms must co-occur within N words         |
| `repeat<N>:`     | `repeat3:"houthi"`                    | Filters to articles where keyword appears ≥N times; removes passing mentions |
| `tone<` / `tone>`| `tone<-5`                             | Hard tone threshold filter applied to the query before sorting |
| `toneabs>`       | `toneabs>8`                           | High-emotion filter — crisis, escalation, alarm coverage      |

**`repeat<N>:` is a first-order relevance filter.** An article that mentions "Houthi" three or
more times is almost certainly about Houthi activity; one that mentions it once may be a sidebar.
This operator should be applied to the `articles_relevant` and `articles_high_neg` pools where
topic focus matters most. Use sparingly on `articles_recent` to preserve recall.

#### Source Scoping Operators

| Operator         | Syntax Example              | Description                                              |
|------------------|-----------------------------|----------------------------------------------------------|
| `domain:`        | `domain:reuters.com`        | All coverage from a domain (partial match)               |
| `domainis:`      | `domainis:un.org`           | Exact domain match — prevents `catholicsun.org` matching `un.org` |
| `sourcecountry:` | `sourcecountry:IR`          | Articles from outlets physically located in that country |
| `sourcelang:`    | `sourcelang:ara`            | Articles originally published in the specified language   |

#### Visual Query Operators (used only in Group D fetches)

| Operator         | Syntax Example                          | Description                                               |
|------------------|-----------------------------------------|-----------------------------------------------------------|
| `imagetag:`      | `imagetag:"military"`                   | Deep-learning content tags (10,000+ categories)           |
| `imagewebtag:`   | `imagewebtag:"drone strike"`            | Crowdsourced caption tags from reverse image search       |
| `imageocrmeta:`  | `imageocrmeta:"IRGC"`                   | OCR text found in the image + EXIF metadata + captions    |
| `imagefacetone`  | `imagefacetone<-1.5`                    | Average emotional tone of visible human faces in image    |
| `imagenumfaces`  | `imagenumfaces>5`                       | Number of foreground faces (crowd vs. individual shots)   |
| `imagewebcount<` | `imagewebcount<10`                      | Novel images — fewer prior web appearances = higher novelty |

---

### 2.1.4 Outputs — `GDELTAgentResult`

#### Article Pools

| Field                      | Type                       | Description                                                   |
|----------------------------|----------------------------|---------------------------------------------------------------|
| `articles_recent`          | `List[Article]`            | Date-sorted (newest first)                                    |
| `articles_negative`        | `List[Article]`            | Sorted by ToneAsc (most negative first)                       |
| `articles_positive`        | `List[Article]`            | Sorted by ToneDesc (most positive first)                      |
| `articles_relevant`        | `List[Article]`            | HybridRel sorted — prominent outlets, high relevance          |
| `articles_high_neg`        | `List[Article]`            | `tone<{threshold}` filtered — hard negative floor             |
| `articles_high_emotion`    | `List[Article]`            | `toneabs>{threshold}` filtered — crisis/alarm coverage        |
| `articles_source_country`  | `List[Article]`            | Country-scoped pool (populated if `source_country_filter` set)|
| `articles_source_lang`     | `List[Article]`            | Language-scoped pool (populated if `source_lang_filter` set)  |
| `articles_authoritative`   | `List[Article]`            | Authority-domain pool (populated if `authoritative_domains` set) |

#### Timeline and Signal Fields

| Field                | Type                   | Description                                                       |
|----------------------|------------------------|-------------------------------------------------------------------|
| `timeline_volinfo`   | `List[TimelineStep]`   | Coverage volume as % of all GDELT + top-10 articles per step      |
| `timeline_volraw`    | `List[TimelineStepRaw]`| Absolute article count + `norm` (total monitored) per step        |
| `timeline_tone`      | `List[TimelineStep]`   | Average tone per time step                                        |
| `timeline_lang`      | `List[TimelineStep]`   | Per-language volume breakdown                                     |
| `timeline_country`   | `List[TimelineStep]`   | Per-source-country volume breakdown                               |
| `tonechart`          | `List[ToneChartBin]`   | Full tone distribution histogram (−100 to +100)                   |

#### Derived Analysis

| Field                | Type                   | Description                                                       |
|----------------------|------------------------|-------------------------------------------------------------------|
| `spikes`             | `List[SpikeWindow]`    | Z-score spike windows detected on `timeline_volinfo`              |
| `spike_articles`     | `Dict[str, List[Article]]` | Per-spike-date article backfill (±48h window)               |
| `title_url_map`      | `Dict[str, List[Article]]` | Date-keyed article lookup for citation building             |
| `actor_graph`        | `ActorGraph`           | Co-occurrence graph with centrality metrics and role labels        |
| `tone_stats`         | `ToneStats`            | Modal tone, mean tone, std dev, polarity ratio                    |
| `language_stats`     | `LanguageStats`        | Top languages, diversity index                                    |
| `country_stats`      | `CountryStats`         | Source country distribution                                       |
| `vol_ratio`          | `float`                | Absolute volume / norm average — measures story's share of news space |
| `run_metadata`       | `RunMetadata`          | Query, window, record counts, active fetch modes, run timestamp   |

#### Visual Intelligence Fields (populated only when `enable_visual_intel=True`)

| Field                | Type                    | Description                                                      |
|----------------------|-------------------------|------------------------------------------------------------------|
| `visual_images`      | `List[VisualImage]`     | VGKG-processed images with novelty scores and provenance         |
| `image_topics`       | `List[ImageTopicTag]`   | Deep-learning visual topic frequency histogram                   |

`VisualImage` object:
```json
{
  "url": "",
  "article_url": "",
  "article_title": "",
  "imagetags": [],
  "imagewebtags": [],
  "web_appearance_count": 0,
  "prior_web_urls": [],
  "exif_capture_date": "",
  "staleness_warning": false,
  "novelty_score": 0.0
}
```

`novelty_score` is computed as `1.0 / (1.0 + log(1 + web_appearance_count))` — a value near 1.0
indicates an image that has never been seen before on the web (strong evidence of novel coverage),
while values near 0.0 indicate widely-recycled stock imagery.

---

### 2.1.5 Failure Handling

| Failure Mode                          | Handling                                                                     |
|---------------------------------------|------------------------------------------------------------------------------|
| HTTP timeout                          | Retry up to 5 times with exponential backoff (base: 2.0s)                   |
| HTTP 429 rate limit                   | Exponential back-off; log warning; resume remaining fetches after delay      |
| HTTP 500/502/503/504                  | Linear backoff retry; log server error code                                  |
| GDELT body is HTTP header block       | Detected by `_HTTP_HEADER_PREFIXES` check; returns `None`; logs warning      |
| Unparseable JSON body                 | `_safe_parse_json()` fallback to `ast.literal_eval`; returns `None` on failure |
| `HybridRel` returns no results        | Falls back to `DateDesc` for `articles_relevant` pool; logs info             |
| Visual Intel fetch returns no images  | `visual_images = []`; pipeline continues; logs info                         |
| `sourcecountry`/`sourcelang` fetch empty | Corresponding pool is empty list; no pipeline halt                        |
| All article pools empty               | `GDELTAgentResult` marked `CRITICAL`; pipeline halts with descriptive error  |
| Partial pools non-empty               | Downstream agents operate on whatever pools are non-empty; warnings logged   |

---

## 2.2 RSSAgent

**Module:** `geoeventfusion/agents/rss_agent.py`  
**Purpose:** Enrich spike windows with full-text news coverage from RSS/Atom feeds.

### Inputs
| Field               | Type               | Description                                        |
|---------------------|--------------------|----------------------------------------------------|
| `spike_windows`     | `List[SpikeWindow]`| Spike windows from GDELTAgent                      |
| `feed_list`         | `List[str]`        | RSS/Atom feed URLs                                 |
| `time_window_hours` | `int`              | Time window around spike for article filtering     |
| `max_articles_per_spike` | `int`        | Article cap per spike (default: 50)                |

### Core Responsibilities
- Feed ingestion via `feedparser` with timeout and retry
- Time window filtering relative to spike date
- Keyword and semantic filtering against spike query terms
- Full-text extraction from article URLs (via `newspaper3k` or `trafilatura`)
- Deduplication by URL and near-duplicate title detection (Levenshtein)
- HTML cleanup and text normalization

### Outputs — `RSSAgentResult`
```python
{
  "articles": List[Article]   # url, title, published_at, source, full_text, metadata
}
```

Each `Article` object:
```json
{
  "url": "",
  "title": "",
  "published_at": "",
  "source": "",
  "full_text": "",
  "metadata": {
    "feed_url": "",
    "spike_date": "",
    "extraction_method": ""
  }
}
```

### Failure Handling
- Feed timeout: skip that feed, log warning, continue
- Full-text extraction failure: fall back to RSS description field
- Deduplication failure: log, retain both articles
- Empty feed result: continue to next feed without failing pipeline

---

## 2.3 GroundTruthAgent

**Module:** `geoeventfusion/agents/ground_truth_agent.py`  
**Purpose:** Provide validated conflict event datasets for calibration and cross-verification.

### Supported Sources
| Source  | Access Method      | Description                                                 |
|---------|--------------------|-------------------------------------------------------------|
| ACLED   | REST API           | Armed Conflict Location and Event Data — daily event data   |
| ICEWS   | File download/API  | Integrated Crisis Early Warning System — coded event data   |

### Inputs
| Field               | Type                | Description                                       |
|---------------------|---------------------|---------------------------------------------------|
| `sources`           | `List[str]`         | Which sources to activate, e.g. `["acled"]`       |
| `country_filter`    | `List[str]`         | ISO country codes or names                        |
| `date_range`        | `Tuple[str, str]`   | Start and end dates from run metadata             |
| `event_types`       | `List[str]`         | ACLED/ICEWS event type filters                    |

### Core Responsibilities
- Schema normalization across ACLED and ICEWS into unified `GroundTruthEvent`
- Temporal alignment to pipeline date window
- Geographic filtering by country or bounding box
- Ground-truth tagging for downstream calibration

### Outputs — `GroundTruthAgentResult`
```json
{
  "events": [
    {
      "event_id": "",
      "source": "acled",
      "event_type": "",
      "date": "",
      "country": "",
      "region": "",
      "lat": 0.0,
      "lon": 0.0,
      "actors": [],
      "fatalities": 0,
      "notes": "",
      "confidence": 1.0
    }
  ],
  "source_counts": { "acled": 0, "icews": 0 }
}
```

### Failure Handling
- Source unavailable: skip, log warning, flag in `ValidationAgent`
- Schema mismatch: log field-level error, apply partial mapping
- No events found in window: return empty list, proceed

---

## 2.4 CustomDatasetAgent

**Module:** `geoeventfusion/agents/custom_dataset_agent.py`  
**Purpose:** Cross-reference proprietary or user-provided datasets against GDELT article pools.

### Supported Formats
| Format | Description                                         |
|--------|-----------------------------------------------------|
| CSV    | Flat file with configurable column mapping          |
| JSON   | Array of records or newline-delimited JSON          |
| SQL    | SQLite or PostgreSQL query results                  |
| API    | Generic REST endpoint with configurable auth        |

### Matching Capabilities
- Title substring and Levenshtein similarity matching
- Actor name overlap (set intersection scoring)
- Temporal proximity matching (configurable window in hours)
- Configurable confidence scoring per match dimension

### Outputs — `CustomDatasetAgentResult`
```json
{
  "matches": [
    {
      "article": {},
      "custom_record": {},
      "match_confidence": 0.0,
      "match_dimensions": ["title_similarity", "actor_overlap", "temporal"]
    }
  ],
  "unmatched_records": [],
  "match_rate": 0.0
}
```

### Failure Handling
- File not found or empty: skip, set `CUSTOM_DATASET_PRESENT = False`
- Schema error: log column mapping failure, skip that record
- API timeout: retry once, then skip

---

## 2.5 LLMExtractionAgent

**Module:** `geoeventfusion/agents/llm_extraction_agent.py`  
**Purpose:** Convert unstructured article text into structured event objects using an LLM backend.

### LLM Backends
| Backend     | Client Library  | Model Config               |
|-------------|-----------------|----------------------------|
| Ollama      | `ollama`        | `OLLAMA_MODEL` (e.g. gpt-oss:120b) |
| Anthropic   | `anthropic`     | `ANTHROPIC_MODEL` (e.g. claude-sonnet-4-6) |

The active backend is controlled by `LLM_BACKEND` in `PipelineConfig`.

### Core Responsibilities
- Backend-agnostic `llm_call(system, prompt, max_tokens, temperature)` interface
- JSON-only output enforcement with multi-attempt retries
- Defensive JSON parsing (strip markdown fences, locate array/object boundaries)
- Multi-event extraction from a single article
- Self-confidence scoring per extracted event
- Phase 3: Structured timeline generation
- Phase 4: 4-round adversarial hypothesis debate

### Structured Event Output Schema
```json
{
  "event_type": "",
  "datetime": "YYYY-MM-DD",
  "country": "",
  "lat": 0.0,
  "lon": 0.0,
  "actors": [],
  "summary": "",
  "confidence": 0.0,
  "source_url": "",
  "source_title": ""
}
```

### Timeline Output Schema (Phase 3)
```json
{
  "phases": [
    {
      "label": "",
      "date_range": { "start": "", "end": "" },
      "description": "",
      "key_events": [],
      "tone_shift": "",
      "actor_changes": []
    }
  ],
  "turning_points": [
    {
      "date": "",
      "description": "",
      "evidence_title": "",
      "evidence_url": ""
    }
  ],
  "summary": "",
  "confidence": 0.0
}
```

### Hypothesis Schema (Phase 4)
```json
{
  "hypotheses": [
    {
      "id": 0,
      "dimension": "",
      "claim": "",
      "supporting_evidence": [],
      "counter_evidence": [],
      "confidence": 0.0,
      "stress_test_result": ""
    }
  ]
}
```

### Failure Handling
- Invalid JSON: strip fences, retry with stricter prompt, fall back to partial extraction
- Empty LLM response: retry once with increased `max_tokens`
- Partial schema: return what was extracted with `confidence: 0.0` for missing fields
- Backend failure: attempt failover to secondary backend if configured

---

## 2.6 FusionAgent

**Module:** `geoeventfusion/agents/fusion_agent.py`  
**Purpose:** Link and cluster events across all source pools (GDELT, RSS, Ground Truth, Custom) into unified fusion clusters.

### Matching Dimensions
| Dimension          | Method                                 | Weight (default) |
|--------------------|----------------------------------------|-----------------|
| Temporal proximity | Date difference in hours               | 0.25            |
| Geographic proximity | Haversine distance (km)              | 0.25            |
| Actor overlap      | Jaccard set intersection               | 0.20            |
| Semantic similarity | Cosine similarity of title embeddings | 0.20            |
| Event type         | Exact or category-level match          | 0.10            |

Weights are configurable in `PipelineConfig.fusion_weights`.

### Outputs — `FusionAgentResult`
```json
{
  "clusters": [
    {
      "cluster_id": "",
      "events": [],
      "source_types": [],
      "fusion_confidence": 0.0,
      "temporal_span": { "start": "", "end": "" },
      "centroid_lat": 0.0,
      "centroid_lon": 0.0,
      "primary_actors": [],
      "contradiction_flags": [],
      "corroboration_count": 0
    }
  ],
  "unclustered_events": [],
  "fusion_stats": {
    "total_events_in": 0,
    "total_clusters": 0,
    "mean_cluster_size": 0.0,
    "contradiction_rate": 0.0
  }
}
```

### Failure Handling
- Insufficient events: return single-event clusters with `fusion_confidence: 0.0`
- Embedding failure: fall back to keyword overlap for semantic similarity
- Geographic data missing: skip geographic dimension, renormalize weights

---

## 2.7 StoryboardAgent

**Module:** `geoeventfusion/agents/storyboard_agent.py`  
**Purpose:** Generate structured narrative intelligence panels from fusion clusters and upstream analysis outputs.

### Inputs
- `FusionAgentResult` — clustered events
- `GDELTAgentResult` — spikes, actor graph, tone stats
- `LLMExtractionAgentResult` — timeline, hypotheses
- `ValidationAgentResult` — citation pool, grounding scores

### Panel Output Schema
```json
{
  "panel_id": "",
  "date_range": { "start": "", "end": "" },
  "headline": "",
  "key_events": [
    {
      "date": "",
      "description": "",
      "source_title": "",
      "source_url": "",
      "verified": true
    }
  ],
  "actors": [
    {
      "name": "",
      "role": "Hub | Broker | Peripheral",
      "centrality_score": 0.0
    }
  ],
  "narrative_summary": "",
  "confidence": 0.0,
  "grounded_sources": [],
  "unverified_elements": [],
  "recommended_followup": []
}
```

### Full Storyboard Output
```json
{
  "query": "",
  "date_range": { "start": "", "end": "" },
  "panels": [],
  "overall_confidence": 0.0,
  "escalation_risk": 0.0,
  "max_confidence_cap": 0.82,
  "recommended_followup": [],
  "generation_timestamp": ""
}
```

### Failure Handling
- LLM failure during narrative generation: fall back to structured summary from cluster data
- Insufficient citations: auto-supplement from article pool to reach `MIN_CITATIONS`
- All panels below minimum confidence: flag storyboard as `LOW_CONFIDENCE`

---

## 2.8 ValidationAgent

**Module:** `geoeventfusion/agents/validation_agent.py`  
**Purpose:** Ensure all storyboard claims are evidence-grounded and all citations are verifiable.

### Validation Checks
| Check                        | Method                                                | Pass Criteria             |
|------------------------------|-------------------------------------------------------|---------------------------|
| URL reachability             | HTTP HEAD request with timeout                        | HTTP 2xx or 3xx response  |
| Timestamp consistency        | Cited article date within ±7 days of claimed event    | Date delta < threshold    |
| Cross-source corroboration   | Same event referenced by ≥2 source domains            | `corroboration_count >= 2`|
| Ground-truth alignment       | Fuzzy match against `GroundTruthAgent` events         | Levenshtein sim > 0.65    |
| Custom dataset confirmation  | Match found in `CustomDatasetAgent` results           | `match_confidence > 0.50` |
| Title-to-claim grounding     | Levenshtein similarity between claim and article title| Similarity > 0.55         |

### Outputs — `ValidationAgentResult`
```json
{
  "grounding_score": 0.0,
  "verification_percentage": 0.0,
  "verified_events": [],
  "unverified_events": [],
  "url_check_results": {},
  "cross_source_corroboration": {},
  "ground_truth_matches": [],
  "flags": [
    {
      "flag_type": "UNVERIFIED_CLAIM | TIMESTAMP_MISMATCH | DEAD_URL | LOW_CORROBORATION",
      "severity": "WARNING | ERROR",
      "detail": ""
    }
  ]
}
```

### Failure Handling
- URL check failure (network error): mark as `UNCHECKED`, do not penalize grounding score
- Ground truth unavailable: skip that check, note in validation report
- All events unverified: set `grounding_score: 0.0`, surface as `CRITICAL` flag

---

## 2.9 ExportAgent

**Module:** `geoeventfusion/agents/export_agent.py`  
**Purpose:** Package and export all intelligence artifacts in the configured formats.

### Supported Export Formats
| Format               | Description                                               |
|----------------------|-----------------------------------------------------------|
| JSON                 | All structured data — storyboard, timeline, hypotheses, validation |
| HTML                 | Full dark-theme storyboard report with embedded charts    |
| PNG charts           | 5 visualizations (timeline, tone, language, actor, choropleth) |
| Actor network (GEXF) | NetworkX graph export for Gephi or other graph tools      |
| Run manifest (JSON)  | Complete artifact index with paths, sizes, checksums      |

### Output Directory Structure
```
outputs/runs/<YYYYMMDD_HHMMSS>_<sanitized_query>/
├── run_metadata.json
├── storyboard.json
├── timeline.json
├── hypotheses.json
├── validation_report.json
├── storyboard_report.html
├── actor_network.gexf
└── charts/
    ├── event_timeline_annotated.png
    ├── tone_distribution.png
    ├── timeline_language.png
    ├── actor_network.png
    └── source_country_map.html
```

### Future Capabilities
- API endpoint export (FastAPI)
- Database persistence (PostgreSQL / SQLite)
- S3 / GCS bucket export
- Webhook notification on run completion

### Failure Handling
- Individual chart failure: log error, skip chart, continue export
- HTML render failure: fall back to JSON-only export
- Disk write failure: raise `ExportError` with full artifact list for retry

---

# 3. Project Documentation Reference

The following files are the authoritative references for this project. Read them before
making architectural decisions or adding new agents.

| File                     | Purpose                                                          |
|--------------------------|------------------------------------------------------------------|
| `claude.md`              | AI assistant guide — architecture rules, conventions, gotchas, file ownership map |
| `AGENTS.md`              | This file — agent contracts, I/O schemas, failure handling       |
| `skills.md`              | Full capability inventory — what each module can do              |
| `DIRECTORY_STRUCTURE.md` | Canonical file locations for every module and artifact           |
| `config/defaults.py`     | All default threshold values and configuration constants         |
| `geoeventfusion/models/` | Typed dataclass/Pydantic schemas — source of truth for I/O       |

`claude.md` is the recommended first read for any AI assistant working on this codebase.
It contains the pipeline phase reference, data flow diagram, known API gotchas, test
conventions, and a file ownership map that maps features to their canonical source files.

---

# 4. Agent Design Principles

- **Stateless when possible** — all persistent state flows through `PipelineContext`
- **Deterministic structured outputs** — all agents produce typed dataclass/Pydantic results
- **Defensive JSON parsing** — strip fences, locate boundaries, retry before failing
- **No hard-coded credentials** — all API keys via environment variables or secrets managers
- **Parallelizable API calls** — `ThreadPoolExecutor` with configurable `max_workers`
- **Configurable record caps** — no magic numbers; all caps live in `PipelineConfig`
- **Test mode support** — `config.test_mode = True` uses fixture data, no real API calls
- **Resumable phases** — each agent checks for cached outputs before executing
- **Backend-agnostic LLM** — `llm_call()` hides backend details from all consuming agents

---

# 5. PipelineContext

All agents read from and write to a shared `PipelineContext` object that threads through the pipeline.

```python
@dataclass
class PipelineContext:
    config: PipelineConfig
    run_id: str                             # YYYYMMDD_HHMMSS_<query_slug>
    output_dir: Path

    # Agent results — populated progressively
    gdelt_result:          Optional[GDELTAgentResult]         = None
    rss_result:            Optional[RSSAgentResult]           = None
    ground_truth_result:   Optional[GroundTruthAgentResult]   = None
    custom_dataset_result: Optional[CustomDatasetAgentResult] = None
    llm_result:            Optional[LLMExtractionAgentResult] = None
    fusion_result:         Optional[FusionAgentResult]        = None
    storyboard_result:     Optional[StoryboardAgentResult]    = None
    validation_result:     Optional[ValidationAgentResult]    = None
    export_result:         Optional[ExportAgentResult]        = None

    # Pipeline metadata
    start_time:     Optional[datetime] = None
    end_time:       Optional[datetime] = None
    phase_log:      List[PhaseRecord]  = field(default_factory=list)
    warnings:       List[str]          = field(default_factory=list)
    errors:         List[str]          = field(default_factory=list)
```

---

# 6. Deployment Modes

| Mode            | Entry Point                         | Description                               |
|-----------------|-------------------------------------|-------------------------------------------|
| Google Colab    | `notebooks/quickstart.ipynb`        | Thin notebook; delegates to package       |
| Local Python    | `scripts/run_pipeline.py`           | Full CLI with argparse / typer            |
| Batch           | `scripts/batch_run.py`              | Multiple queries from YAML config         |
| Docker          | `Dockerfile` (future)               | Containerized, all deps bundled           |
| API Server      | `geoeventfusion/server.py` (future) | FastAPI with async pipeline execution     |

---

# 7. Reliability & Security

### Reliability
- Retry logic with exponential backoff on all external calls
- Timeout handling on all HTTP and LLM requests
- Graceful degradation: downstream agents operate on partial upstream data
- Resumable phases via cached intermediate JSON files

### Security
- Environment-based API keys only — never committed to source
- `.env.example` provides the template; `.env` is `.gitignore`'d
- Sanitized external inputs before LLM prompt construction
- No eval() or exec() on external data

### Observability
- Structured logging via `logging_utils.py` with run_id context
- Per-phase timing in `PipelineContext.phase_log`
- Warning and error lists surfaced in `run_metadata.json`

---

# 8. Known Issues in v2.4 Notebook (Resolved in Package)

| Issue | Location | Fix |
|---|---|---|
| `from collections import defaultdict, Count` — `Count` does not exist | Cell 4 | Changed to `Counter` |
| `anthropic` model hardcoded to `claude-3-5-sonnet-20241022` | Cell 6 | Moved to `PipelineConfig.anthropic_model` |
| All logic in one file — untestable | Entire notebook | Separated into package modules |
| No unit tests | N/A | `tests/` directory with full coverage |
| Phase outputs written to flat `data/` directory | Cell 12 | Moved to `outputs/runs/<run_id>/` |
| No BaseAgent contract | N/A | `agents/base.py` defines ABC |
| FusionAgent / ValidationAgent implicit (inline code) | Cells 28-36 | Promoted to first-class agents |
| Only 3 article pools (DateDesc, ToneAsc, ToneDesc) | Cell 11 | Expanded to 6 core + 3 conditional pools |
| No `HybridRel` sort — obscure outlets over prominent ones | Cell 11 | Added `articles_relevant` pool with HybridRel |
| No inline tone/toneabs query operators — sort only | Cell 11 | Added `articles_high_neg` and `articles_high_emotion` pools |
| `TimelineVolRaw` absent — no absolute article counts | Cell 11 | Added `timeline_volraw` with norm field |
| No visual intelligence layer | N/A | Added optional Group D image fetch modes |
| No source-scoped article pools | Cell 11 | Added conditional `sourcecountry`/`sourcelang`/`domainis` pools |
| `near<N>:` operator only used ad-hoc in enrichment | Cells 36, 10 | Formalized in `QueryBuilder` operator composition |
| `repeat<N>:` operator not used | N/A | Added to `QueryBuilder` for relevance filtering |
