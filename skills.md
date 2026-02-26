# skills.md
## GEOEventFusion — System Skills & Capabilities Reference
Version: 2.0
Date: 2026-02-25

---

# 1. Data Ingestion Skills

## 1.1 GDELT DOC 2.0 Processing

**Module:** `geoeventfusion/clients/gdelt_client.py`, `geoeventfusion/agents/gdelt_agent.py`

### Fetch Modes

The agent executes up to 13 parallel GDELT DOC 2.0 API calls across four functional groups.

**Group A — Core Article Pools (6 pools, always active)**
- `articles_recent` — `ArtList` with `DateDesc` sort: full chronological window
- `articles_negative` — `ArtList` with `ToneAsc` sort: most negatively-toned articles
- `articles_positive` — `ArtList` with `ToneDesc` sort: most positively-toned articles
- `articles_relevant` — `ArtList` with `HybridRel` sort: prominence-ranked by outlet + textual relevance (post-2018-09-16 content)
- `articles_high_neg` — `ArtList` with `DateDesc` + inline `tone<{threshold}` query filter: hard negative-tone floor, broader recall than sort alone
- `articles_high_emotion` — `ArtList` with `DateDesc` + inline `toneabs>{threshold}` query filter: crisis/alarm coverage regardless of polarity

**Group B — Timeline and Signal Modes (6 modes, always active)**
- `TimelineVolInfo` — coverage volume as % of all GDELT + top-10 articles per time step
- `TimelineVolRaw` — absolute article counts with `norm` field (total monitored per interval)
- `TimelineTone` — average tone over time
- `TimelineLang` — coverage volume broken down by language
- `TimelineSourceCountry` — coverage volume broken down by source country
- `ToneChart` — full tone distribution histogram (−100 to +100 bins)

**Group C — Targeted Source Fetches (conditional)**
- `sourcecountry:<FIPS>` — local press from a specific country (e.g., `sourcecountry:IR` for Iran)
- `sourcelang:<ISO>` — original-language coverage (e.g., `sourcelang:ara` for Arabic)
- `domainis:<domain>` OR-chain — authority source pool (e.g., `un.org`, `state.gov`, `nato.int`)

**Group D — Visual Intelligence Modes (conditional, `enable_visual_intel=True`)**
- `ImageCollageInfo` with `imagetag:` filters — VGKG-processed images with novelty scores, provenance chains, and deep-learning content tags
- `WordCloudImageTags` — visual topic frequency histogram from Google Cloud Vision API

### Query Operators

**Text search operators**
- `"phrase"` — exact phrase matching
- `(a OR b OR c)` — boolean OR for alias expansion (alternate spellings, abbreviations)
- `-operator` — exclusion (e.g., `-sourcelang:spanish`)
- `theme:<GKG_CODE>` — GKG theme codes with LLM-assisted suggestion
- `near<N>:"term1 term2"` — proximity search (terms within N words)
- `repeat<N>:"keyword"` — keyword must appear ≥N times in article; removes passing mentions
- `tone<` / `tone>` — inline tone threshold filter (e.g., `tone<-5` for clearly negative)
- `toneabs>` — high-emotion filter regardless of polarity (e.g., `toneabs>8` for crisis coverage)

**Source scoping operators**
- `domain:<domain>` — all coverage from a domain (partial match)
- `domainis:<domain>` — exact domain match (prevents `catholicsun.org` matching `un.org`)
- `sourcecountry:<FIPS>` — articles from outlets in a specific country
- `sourcelang:<ISO>` — articles originally published in a specific language

**Visual query operators** (Group D fetches only)
- `imagetag:"<label>"` — Google Cloud Vision deep-learning content tag (10,000+ categories; e.g., `"military"`, `"protest"`, `"explosion"`, `"rubble"`)
- `imagewebtag:"<label>"` — crowdsourced caption tags from reverse image search
- `imageocrmeta:"<text>"` — OCR text found in image + EXIF metadata + surrounding captions
- `imagefacetone<` — emotional tone of visible faces (e.g., `imagefacetone<-1.5` for distress)
- `imagenumfaces>` — number of foreground faces (useful for crowd vs. individual shots)
- `imagewebcount<` — novelty filter: images seen fewer than N times on the web are more likely original

### Visual Intelligence — Novelty Scoring

Every `VisualImage` object receives a `novelty_score` computed as:

```
novelty_score = 1.0 / (1.0 + log(1 + web_appearance_count))
```

A score near 1.0 indicates a never-before-seen image — strong evidence for novel/breaking events.
A score near 0.0 indicates widely-recycled stock imagery. Images with a staleness warning
(EXIF capture date > 72 hours before article publication) are flagged for analyst review.

### Infrastructure
- Staggered parallel submission (0.75s between calls) with `max_workers=2` to respect GDELT rate limits
- Domain diversity filter — cap any single domain at `domain_cap_pct` (default: 20%) across all article pools
- Defensive `_safe_parse_json()` — handles GDELT HTTP header bleed-through in response bodies
- Rate limit detection (HTTP 429) with exponential back-off (doubles per attempt)
- Configurable record caps (GDELT maximum: 250 per ArtList call)
- `vol_ratio` derivation from `TimelineVolRaw` norm field — measures story's share of total news space

## 1.2 RSS Feed Processing

**Module:** `geoeventfusion/clients/rss_client.py`, `geoeventfusion/agents/rss_agent.py`

- `feedparser`-based Atom and RSS 2.0 ingestion
- Time-window filtering relative to spike dates
- Keyword and semantic relevance scoring against spike query
- Full-text extraction via `trafilatura` (primary) or `newspaper3k` (fallback)
- HTML tag stripping and Unicode normalization
- Near-duplicate title detection via Levenshtein similarity
- Configurable per-spike article cap

## 1.3 Ground-Truth Dataset Handling

**Module:** `geoeventfusion/clients/ground_truth_client.py`, `geoeventfusion/agents/ground_truth_agent.py`

- ACLED REST API integration — country and date range filtering
- ICEWS file/API ingestion — CAMEO-coded event data
- Schema normalization — ACLED and ICEWS event fields mapped to unified `GroundTruthEvent`
- Temporal alignment to pipeline analysis window
- Geospatial filtering by country ISO code or lat/lon bounding box
- Conflict event type mapping and alignment

## 1.4 Custom Dataset Integration

**Module:** `geoeventfusion/agents/custom_dataset_agent.py`

- CSV ingestion with configurable column mapping
- JSON record array and NDJSON ingestion
- SQLite and PostgreSQL query execution
- Generic REST API integration with configurable auth headers
- Similarity scoring against GDELT article pools via title match and actor overlap
- Confidence scoring per matched record

---

# 2. Event Extraction Skills

## 2.1 LLM Structured Extraction

**Module:** `geoeventfusion/agents/llm_extraction_agent.py`, `geoeventfusion/clients/llm_client.py`

- Backend-agnostic `llm_call(system, prompt, max_tokens, temperature)` interface
- JSON-only output enforcement with prompt-level instruction
- Markdown fence stripping and JSON boundary detection in LLM output
- Retry-on-empty-response with increased `max_tokens`
- Multi-event extraction from a single article body
- Per-event confidence self-scoring
- `MAX_CONFIDENCE` cap enforcement (configurable, default 0.82)
- GKG theme suggestion with 2-attempt retry and 256-token budget

## 2.2 Named Entity and Actor Extraction

**Module:** `geoeventfusion/analysis/actor_graph.py`, `geoeventfusion/utils/text.py`

- Regex-based capitalized entity extraction from article titles
- Multi-token actor recognition (up to 4 tokens)
- Media organization exclusion via `_MEDIA_TOKENS`, `_MEDIA_BIGRAMS`, and `_STOPWORDS_UPPER` lookups
- Suffix-based media outlet detection (e.g., "News", "Post", "Times")
- Co-occurrence triple extraction `(actor_a, actor_b, date)`

## 2.3 Event Typing and Classification

**Module:** `geoeventfusion/agents/llm_extraction_agent.py`

- LLM-assigned event type labels per extracted event
- Supported categories: CONFLICT, PROTEST, DIPLOMATIC, MILITARY_ESCALATION, POLITICAL_INSTABILITY, HUMANITARIAN, ECONOMIC, SANCTIONS, MARITIME, CYBER, OTHER
- Confidence-weighted event type assignment
- Hypothesis dimensionality enforcement — ensures 4-round debate covers distinct event dimensions

---

# 3. Signal Analysis Skills

## 3.1 Spike Detection

**Module:** `geoeventfusion/analysis/spike_detector.py`

- Z-score computation over TimelineVolInfo data
- Configurable Z-score threshold (default: 1.5)
- Graceful handling of zero-variance timelines
- Spike ranking by Z-score (descending)
- Per-spike article backfill via targeted ±48h GDELT window fetches
- Date normalization across GDELT's inconsistent format variants

## 3.2 Tone Analysis

**Module:** `geoeventfusion/analysis/tone_analyzer.py`

- ToneChart histogram analysis — modal tone, mean tone, standard deviation
- Polarity ratio computation (negative vs. positive article counts)
- TimelineTone trend analysis over the analysis window
- Language coverage — top languages, diversity index
- Source country distribution analysis

## 3.3 Hypothesis Debate Engine

**Module:** `geoeventfusion/analysis/hypothesis_engine.py`

- Round 1: Generate hypotheses from negative-toned article corpus
- Round 2: Critique hypotheses against recent article corpus
- Round 3: Enforce dimensional diversity — reject hypotheses in already-covered dimensions
- Round 4: Stress-test surviving hypotheses against positive-toned counterevidence
- `MAX_CONFIDENCE` cap applied after each round
- Hypothesis output includes claim, supporting evidence, counter-evidence, and stress-test result

---

# 4. Graph and Network Skills

## 4.1 Actor Co-occurrence Graph

**Module:** `geoeventfusion/analysis/actor_graph.py`

- NetworkX `Graph` construction from co-occurrence triples
- Edge weight accumulation (repeated co-occurrences)
- Isolated node pruning for cleaner visualization

## 4.2 Centrality Computation

**Module:** `geoeventfusion/analysis/actor_graph.py`

- Degree centrality (normalized)
- Betweenness centrality (weighted, normalized)
- PageRank (weighted, max 200 iterations)
- Actor role classification: Hub (top-N by degree), Broker (high betweenness relative to degree), Peripheral (all others)

## 4.3 Community Detection

**Module:** `geoeventfusion/analysis/actor_graph.py`

- Greedy modularity maximization (`networkx.algorithms.community.greedy_modularity_communities`)
- Temporal community shift — Jaccard reorganization score across early/mid/late thirds of analysis window
- Phase boundary candidate derivation from high-reorganization windows

---

# 5. Fusion and Linking Skills

## 5.1 Temporal Matching

**Module:** `geoeventfusion/agents/fusion_agent.py`

- Event date normalization to `YYYY-MM-DD`
- Configurable proximity window in hours
- Time delta scoring (inverse linear decay)

## 5.2 Geographic Matching

**Module:** `geoeventfusion/utils/geo_utils.py`, `geoeventfusion/agents/fusion_agent.py`

- Haversine distance calculation
- Country-level matching by ISO code
- Country centroid lookup for events with country-only geolocation
- Configurable distance threshold (km)

## 5.3 Semantic Similarity

**Module:** `geoeventfusion/agents/fusion_agent.py`

- Cosine similarity between article title embeddings
- Fallback to keyword overlap (TF-IDF style) when embeddings unavailable
- Event-type category alignment scoring

## 5.4 Actor Overlap Scoring

**Module:** `geoeventfusion/agents/fusion_agent.py`

- Jaccard set intersection over actor name sets
- Configurable weight relative to other fusion dimensions
- Partial name matching (Levenshtein fallback)

---

# 6. Validation and Grounding Skills

**Module:** `geoeventfusion/agents/validation_agent.py`

- URL reachability checks via HTTP HEAD (configurable timeout)
- Timestamp consistency checks (article date vs. claimed event date)
- Cross-source corroboration count per claim
- Ground-truth alignment via fuzzy Levenshtein matching
- Custom dataset confirmation scoring
- Title-to-claim grounding via Levenshtein similarity scoring
- Grounding score aggregation across all checks
- Severity-classified flag generation (WARNING / ERROR / CRITICAL)
- Configurable minimum verification threshold

---

# 7. Narrative and Storyboard Skills

**Module:** `geoeventfusion/agents/storyboard_agent.py`

- Structured narrative panel synthesis from fusion clusters
- LLM-driven headline generation grounded in spike article titles
- Multi-event summarization per panel with citation inclusion
- Phase boundary detection from community reorganization scores and spike dates
- Turning-point identification with exact article title quotation as evidence
- Contradiction detection between fusion clusters
- Uncertainty labeling with `unverified_elements` per panel
- Follow-up recommendation generation with LLM-assisted GDELT near-operator query conversion
- Auto-supplementation of citations to meet `MIN_CITATIONS` floor

---

# 8. Visualization Skills

**Module:** `geoeventfusion/visualization/`

| Chart | Module | Description |
|---|---|---|
| Coverage Volume Timeline | `timeline_chart.py` | GDELT-native TimelineVolInfo with spike markers and phase annotations |
| Tone Distribution | `tone_chart.py` | Histogram of ToneChart bins with modal/mean overlays |
| Language Stacked Area | `language_chart.py` | Top-language coverage over time from TimelineLang |
| Actor Network | `actor_network.py` | NetworkX graph with Hub/Broker/Peripheral roles and community coloring |
| Source Country Choropleth | `choropleth.py` | Folium map with bubble markers sized by article volume |
| HTML Storyboard Report | `html_report.py` | Full dark-theme panel report with embedded evidence and confidence indicators |

All chart modules share dark-theme constants from `visualization/theme.py`:
- Background: `#0A0E17`
- Panel: `#111827`
- Text: `#E5E7EB`
- Accent: `#60A5FA`
- Spike: `#F59E0B`
- Escalation colors: Red `#EF4444` ≥ 0.70, Amber `#F59E0B` ≥ 0.45, Green `#10B981` < 0.45

---

# 9. Engineering Skills

**Modules:** `geoeventfusion/utils/`, `geoeventfusion/io/`, `config/`

## 9.0 AI-Assisted Development Documentation

- `claude.md` at the project root provides Claude with authoritative codebase context:
  architecture rules, module responsibility boundaries, coding conventions, common CLI commands,
  pipeline phase reference, data flow summary, known API gotchas, test conventions, and a
  file ownership map
- `AGENTS.md` provides agent-level contracts: input/output schemas, failure handling, and the
  `BaseAgent` interface contract
- `skills.md` (this file) provides the full capability inventory referenced by both developers
  and AI assistants when assessing what the system can and cannot do
- All three files are kept in sync with code changes — update them alongside any architectural
  modification

- Modular agent architecture with `BaseAgent` ABC
- Shared `PipelineContext` threading intermediate results through the pipeline
- Typed dataclass / Pydantic model outputs for all agents
- Defensive JSON parsing with fence stripping and boundary detection
- Parallel API calls via `ThreadPoolExecutor` with staggered submission
- Exponential backoff retry on all external HTTP calls
- Environment-based credential handling via `python-dotenv`
- Configurable record caps and thresholds — no magic numbers
- Disk-level intermediate result caching for resumable phases
- Test mode support via fixture data bypass
- Structured logging with run_id context

---

# 10. Scalability Skills

**Modules:** `geoeventfusion/pipeline.py`, `scripts/`

- Swappable ingestion modules — add a new agent by implementing `BaseAgent`
- Database-ready output schema — all results serializable to PostgreSQL/SQLite
- API deployment readiness — `pipeline.py` designed for async FastAPI wrapping
- Docker container preparation — self-contained dependency manifest
- Batch query runner — `scripts/batch_run.py` for multi-query execution from YAML config
- Timestamped run output directories — full reproducibility and audit trail
- GEXF network export — actor graphs loadable in Gephi or graph analysis tools

---

# 11. System Intelligence Capabilities

GEOEventFusion demonstrates:

- **Multi-source event fusion** — GDELT, RSS, ACLED/ICEWS, and custom datasets linked into unified clusters
- **AI-driven structured extraction** — dual-backend LLM produces typed, schema-validated event objects
- **Evidence-grounded narrative generation** — every storyboard claim traceable to a cited article URL
- **Confidence-aware intelligence reporting** — per-panel confidence scores with enforced maximum cap
- **Adversarial hypothesis testing** — 4-round debate stress-tests each hypothesis against counterevidence
- **Graph-grounded phase analysis** — temporal community reorganization drives phase boundary detection
- **Research-grade geopolitical event modeling** — GDELT DOC 2.0, CAMEO coding, GKG theme alignment
- **Professional visualization suite** — 5 dark-theme charts plus full HTML storyboard report
- **Follow-up enrichment loop** — storyboard recommendations converted to GDELT queries for recursive analysis
