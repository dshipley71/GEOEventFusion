# GEOEventFusion — Agent Reference

Quick-reference for each agent's inputs, outputs, and key failure modes.
For the full specification, see `AGENTS.md`.

---

## GDELTAgent

**Module:** `geoeventfusion/agents/gdelt_agent.py`
**Phase:** 1
**Key outputs:** Article pools, spikes, actor graph, tone stats, timeline signals

**Activate conditional pools via config:**
```python
config = PipelineConfig(
    source_country_filter="IR",            # FIPS code
    source_lang_filter="ara",              # ISO 3-char
    authoritative_domains=["un.org", "state.gov"],
    enable_visual_intel=True,
    visual_imagetags=["military", "protest"],
)
```

**Critical failure:** All article pools empty → pipeline halts with descriptive error.

---

## RSSAgent

**Module:** `geoeventfusion/agents/rss_agent.py`
**Phase:** 2
**Key outputs:** `rss_result.articles` — full-text articles from spike windows

**Activate via config:**
```python
config = PipelineConfig(
    rss_feed_list=["https://feeds.reuters.com/reuters/topNews", ...],
    rss_max_articles_per_spike=50,
    rss_time_window_hours=48,
)
```

---

## GroundTruthAgent

**Module:** `geoeventfusion/agents/ground_truth_agent.py`
**Phase:** 2
**Key outputs:** `ground_truth_result.events` — ACLED/ICEWS coded events

**Activate via config:**
```python
config = PipelineConfig(
    ground_truth_sources=["acled"],
    ground_truth_country_filter=["Yemen"],
)
```

**Required env vars:** `ACLED_API_KEY`, `ACLED_EMAIL`

---

## CustomDatasetAgent

**Module:** `geoeventfusion/agents/custom_dataset_agent.py`
**Phase:** 2
**Key outputs:** `custom_dataset_result.matches` — records matched to article pools

**Activate via config:**
```python
config = PipelineConfig(
    custom_dataset_path="data/custom_datasets/my_events.csv",
    custom_dataset_format="csv",
)
```

---

## LLMExtractionAgent

**Module:** `geoeventfusion/agents/llm_extraction_agent.py`
**Phase:** 3–4, 6
**Key outputs:**
- `llm_result.timeline_events` — structured `TimelineEntry` objects
- `llm_result.timeline_phases` — named phases with date ranges
- `llm_result.turning_points` — decisive events with evidence citations
- `llm_result.hypotheses` — 4-round adversarial `Hypothesis` objects
- `llm_result.followup_briefs` — enrichment queries for phase 6

**Confidence cap:** All `confidence` fields capped at `MAX_CONFIDENCE = 0.82`.

---

## FusionAgent

**Module:** `geoeventfusion/agents/fusion_agent.py`
**Phase:** 5
**Key outputs:** `fusion_result.clusters` — `FusionCluster` objects

**Fusion dimensions (configurable weights):**
- Temporal proximity (default: 0.25)
- Geographic proximity (default: 0.25)
- Actor overlap / Jaccard (default: 0.20)
- Semantic similarity (default: 0.20)
- Event type alignment (default: 0.10)

---

## StoryboardAgent

**Module:** `geoeventfusion/agents/storyboard_agent.py`
**Phase:** 5
**Key outputs:** `storyboard_result.panels` — `StoryboardPanel` objects with narrative, citations, actors

**Auto-supplements** citations to reach `min_citations` (default: 3) from the article pool.

---

## ValidationAgent

**Module:** `geoeventfusion/agents/validation_agent.py`
**Phase:** 5
**Key outputs:**
- `validation_result.grounding_score` — float in [0, 1]
- `validation_result.flags` — severity-classified `VerificationFlag` objects
- `validation_result.url_check_results` — per-URL reachability

**Validation checks:**
1. URL reachability (HTTP HEAD)
2. Timestamp consistency (±7 days)
3. Cross-source corroboration (≥2 unique domains)
4. Ground-truth alignment (Levenshtein > 0.65)
5. Title-to-claim grounding (Levenshtein > 0.55)

---

## ExportAgent

**Module:** `geoeventfusion/agents/export_agent.py`
**Phase:** 7
**Output artifacts:**

| File | Format |
|---|---|
| `run_metadata.json` | JSON |
| `storyboard.json` | JSON |
| `timeline.json` | JSON |
| `hypotheses.json` | JSON |
| `validation_report.json` | JSON |
| `storyboard_report.html` | HTML |
| `actor_network.gexf` | GEXF (Gephi) |
| `charts/event_timeline_annotated.png` | PNG |
| `charts/tone_distribution.png` | PNG |
| `charts/timeline_language.png` | PNG |
| `charts/actor_network.png` | PNG |
| `charts/source_country_map.html` | HTML (Folium) |

Individual chart failures are logged and skipped — the agent continues to export whatever it can.
