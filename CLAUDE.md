@AGENTS.md
@skills.md

# CLAUDE.md
## GEOEventFusion — AI Assistant Project Guide
Version: 1.0
Date: 2026-02-25

This file provides Claude with the context, conventions, and constraints needed to work
effectively on the GEOEventFusion codebase. Read this before making any code changes,
writing tests, or generating new modules.

---

# 1. Project Purpose

GEOEventFusion is a modular, multi-agent geopolitical intelligence pipeline that:

1. Fetches global event data from GDELT DOC 2.0 (8-mode parallel queries)
2. Detects coverage spikes and extracts actor graphs
3. Enriches spikes with RSS full-text articles and ground-truth conflict datasets
4. Uses an LLM to extract structured events, generate timelines, and debate hypotheses
5. Fuses events across all sources into grounded intelligence storyboard panels
6. Validates all claims against cited article evidence
7. Exports a full artifact set: JSON, HTML storyboard, visualization charts, actor graph

The pipeline is designed for professional intelligence analysis. Every output must be
evidence-grounded. Confidence scores must be honest — never inflate past `MAX_CONFIDENCE`.

---

# 2. Authoritative Reference Files

Before writing any code, always read the relevant reference documents:

| File                    | Contents                                                   |
|-------------------------|------------------------------------------------------------|
| `AGENTS.md`             | Agent contracts, I/O schemas, failure handling per agent   |
| `skills.md`             | Full capability inventory — what each module can do        |
| `DIRECTORY_STRUCTURE.md`| Canonical file locations for every module and artifact     |
| `config/defaults.py`    | All default threshold values and configuration constants   |
| `geoeventfusion/models/`| Typed dataclass/Pydantic schemas — source of truth for I/O |

When in doubt about where a function belongs or what a schema looks like, check these
files before writing new code.

---

# 3. Architecture Rules

## 3.1 Agent Boundaries
- Every agent lives in `geoeventfusion/agents/` and inherits from `BaseAgent`
- Agents **do not call each other directly** — they read from and write to `PipelineContext`
- Agents **do not import from other agents** — shared logic goes into `analysis/`, `utils/`, or `clients/`
- Agent output types are defined in `geoeventfusion/models/` — never return raw `Dict`

## 3.2 Module Responsibilities

| Layer             | Location                        | Rule                                                |
|-------------------|---------------------------------|-----------------------------------------------------|
| API clients       | `geoeventfusion/clients/`       | HTTP calls only — no business logic                 |
| Data models       | `geoeventfusion/models/`        | Schema definitions only — no methods with side effects |
| Analysis logic    | `geoeventfusion/analysis/`      | Pure functions — no I/O, no API calls               |
| Visualization     | `geoeventfusion/visualization/` | Rendering only — no data transformation             |
| I/O               | `geoeventfusion/io/`            | File read/write only — no business logic            |
| Utilities         | `geoeventfusion/utils/`         | Stateless helpers — no state, no external calls     |

## 3.3 Configuration
- All threshold values and defaults live in `config/defaults.py`
- All runtime configuration flows through `PipelineConfig` — never use module-level globals
- No hard-coded API keys, model names, file paths, or magic numbers anywhere in source
- API keys come from environment variables only — use `os.getenv()` or `python-dotenv`

## 3.4 LLM Usage
- All LLM calls go through `llm_call(system, prompt, max_tokens, temperature)` in `clients/llm_client.py`
- Never call `anthropic` or `ollama` libraries directly from agent code
- Always request JSON-only output and apply defensive parsing (strip fences, find boundaries)
- Always retry once on empty LLM response before failing
- Current default Anthropic model: `claude-sonnet-4-6`
- Current default Ollama model: `gpt-oss:120b`

---

# 4. Coding Conventions

## 4.1 Style
- Python 3.10+
- Type annotations on all public functions and class attributes
- `ruff` for linting and formatting (configured in `pyproject.toml`)
- `mypy` for static type checking
- Maximum line length: 100 characters
- Docstrings on all public functions — one-line summary, then Args/Returns if non-trivial

## 4.2 Naming
- Agents: `SnakeCaseAgent` class, `snake_case_agent.py` file
- Models: `SnakeCaseResult` dataclass per agent output
- Analysis functions: `verb_noun()` pattern (e.g., `detect_spikes`, `build_actor_graph`)
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: `_leading_underscore()`

## 4.3 Error Handling
- Never raise bare `Exception` — use specific typed exceptions or return `None` with a logged warning
- All external HTTP calls must have explicit timeout parameters
- Log warnings for recoverable failures; log errors for unrecoverable ones
- Downstream agents must tolerate `None` or empty upstream results gracefully

## 4.4 JSON Handling
All LLM JSON parsing must use the defensive pattern from `clients/llm_client.py`:
```python
def _safe_parse_llm_json(text: str) -> Optional[Any]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = text.find(start_char)
        e = text.rfind(end_char)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                pass
    return None
```
Never call `json.loads()` directly on LLM output without this wrapping pattern.

---

# 5. Common Commands

```bash
# Install all dependencies
pip install -e ".[dev]"

# Run the full pipeline (CLI)
python scripts/run_pipeline.py --query "Houthi Red Sea attacks" --days-back 90

# Run with Anthropic backend
python scripts/run_pipeline.py --query "Taiwan Strait tensions" --llm-backend anthropic

# Run in test mode (fixture data, no API calls)
python scripts/run_pipeline.py --query "test" --test-mode

# Pre-flight environment validation
python scripts/validate_env.py

# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=geoeventfusion --cov-report=term-missing

# Lint and format
ruff check geoeventfusion/ --fix
ruff format geoeventfusion/

# Type check
mypy geoeventfusion/

# Batch run from config file
python scripts/batch_run.py --config configs/batch_queries.yaml
```

---

# 6. Pipeline Phase Reference

| Phase | Agent(s)                         | Key Output                                      |
|-------|----------------------------------|-------------------------------------------------|
| 1     | `GDELTAgent`                     | Article pools, spikes, actor graph, tone stats  |
| 2     | `RSSAgent`, `GroundTruthAgent`, `CustomDatasetAgent` | Enriched articles, ground-truth events |
| 3     | `LLMExtractionAgent` (timeline)  | Structured timeline JSON with phase boundaries  |
| 4     | `LLMExtractionAgent` (hypotheses)| 4-round adversarial hypothesis set              |
| 5     | `FusionAgent`, `StoryboardAgent`, `ValidationAgent` | Grounded storyboard panels |
| 6     | `LLMExtractionAgent` (followup)  | Follow-up GDELT enrichment briefs               |
| 7     | `ExportAgent`                    | JSON, HTML, PNG charts, actor GEXF              |

---

# 7. Data Flow

```
PipelineConfig
     │
     ▼
PipelineContext (instantiated once, threaded through all agents)
     │
     ├─► GDELTAgent.run(ctx)         → ctx.gdelt_result
     ├─► RSSAgent.run(ctx)           → ctx.rss_result
     ├─► GroundTruthAgent.run(ctx)   → ctx.ground_truth_result
     ├─► CustomDatasetAgent.run(ctx) → ctx.custom_dataset_result
     ├─► LLMExtractionAgent.run(ctx) → ctx.llm_result
     ├─► FusionAgent.run(ctx)        → ctx.fusion_result
     ├─► StoryboardAgent.run(ctx)    → ctx.storyboard_result
     ├─► ValidationAgent.run(ctx)    → ctx.validation_result
     └─► ExportAgent.run(ctx)        → ctx.export_result
```

Each agent reads only from `PipelineContext` fields populated by earlier agents.
The pipeline orchestrator (`pipeline.py`) manages ordering and phase-level caching.

---

# 8. Output Artifacts

All outputs go to `outputs/runs/<run_id>/` — never to a flat `outputs/` or `data/` directory.

```
outputs/runs/<YYYYMMDD_HHMMSS>_<query_slug>/
├── run_metadata.json         # Query, window, counts, timing, warnings
├── storyboard.json           # Storyboard panels with evidence citations
├── timeline.json             # Phase-structured timeline with turning points
├── hypotheses.json           # 4-round debate results
├── validation_report.json    # Grounding scores and verification flags
├── storyboard_report.html    # Full dark-theme HTML storyboard
├── actor_network.gexf        # NetworkX graph for Gephi
└── charts/
    ├── event_timeline_annotated.png
    ├── tone_distribution.png
    ├── timeline_language.png
    ├── actor_network.png
    └── source_country_map.html
```

---

# 9. Adding a New Agent

Follow this checklist when adding a new agent:

1. Define the output model in `geoeventfusion/models/` as a typed dataclass
2. Add the result field to `PipelineContext` in `geoeventfusion/pipeline.py`
3. Implement the agent in `geoeventfusion/agents/<name>_agent.py` inheriting `BaseAgent`
4. Register the agent in the pipeline execution order in `pipeline.py`
5. Add the agent contract (inputs, outputs, failure handling) to `AGENTS.md`
6. Add any new capabilities to `skills.md`
7. Add the new module file to `DIRECTORY_STRUCTURE.md`
8. Write unit tests in `tests/unit/test_<name>_agent.py` using fixture data
9. Update `requirements.txt` if new dependencies are introduced

---

# 10. Known Issues and Gotchas

## 10.1 GDELT API Behavior

**Response handling**
- GDELT occasionally returns an HTTP header block as the JSON response body. Always run
  responses through `_safe_parse_json()` — never call `resp.json()` without a try/except.
- GDELT enforces an unofficial rate limit. Never submit more than 2 concurrent requests.
  Always stagger submissions by ≥ 0.75 seconds.
- GDELT date fields are inconsistent across modes — always normalize through `date_utils.normalize_date_str()`.

**Query operators**
- The `near<N>:"term1 term2"` query operator produces zero results for short or abstract terms.
  `_validate_gdelt_query()` strips it when terms are under 5 characters. Use plain keyword
  queries as the fallback.
- The `repeat<N>:"keyword"` operator only accepts a single word — no phrase searches. It requires
  the keyword appears AT LEAST N times; a document matching 10 times also satisfies `repeat3:`.
- The `tone<` and `toneabs>` operators are query-level filters applied before sorting.
  They are additive with any sort mode — `tone<-5` + `ToneAsc` is valid and returns the most
  negative subset of an already-negative-filtered pool. Do not double-count with `articles_negative`.
- `domainis:` requires exact domain name without any subdomain (e.g., `domainis:un.org` not
  `domainis:www.un.org`). Multiple `domainis:` constraints must be OR'd:
  `(domainis:un.org OR domainis:state.gov)`.
- `sourcecountry:` uses FIPS country codes, not ISO 3166. Check `LOOKUP-COUNTRIES.TXT` for codes.
  Spaces in country names are removed: `sourcecountry:saudiarabia` not `sourcecountry:saudi arabia`.

**HybridRel sort**
- `HybridRel` is only available for content published after 2018-09-16. For older date windows,
  GDELT silently falls back to relevance-only scoring. Always check result count; if it matches
  `DateDesc` exactly, HybridRel may have been silently overridden.
- `HybridRel` is not available for image searches — only textual article searches. Do not use
  it in Group D (visual intelligence) fetches.

**TimelineVolRaw**
- The `norm` field in `TimelineVolRaw` is NOT smoothed even when `TIMELINESMOOTH` is set.
  Raw norm values are the true denominator for computing `vol_ratio`. Never smooth the norm
  field independently — use the raw value with the smoothed volume for ratio calculations.

**Visual intelligence**
- `ImageCollageInfo` is significantly slower than `ArtList` due to reverse image search enrichment.
  It should be gated behind `enable_visual_intel=True` and should NOT be in the critical path
  for spike detection or timeline analysis.
- `imagetag:` values must be enclosed in quote marks even for single words: `imagetag:"military"`
  not `imagetag:military`. Browse the full tag list at `LOOKUP-IMAGETAGS.TXT`.
- `imagewebcount` measures the number of *pages* Google has seen the image on, not unique sites.
  A single outlet republishing one image 50 times scores 50, not 1. High counts do not always
  mean the image is a stock photo — verify with `prior_web_urls` before flagging.
- `imagefacetone` only scores faces large enough for emotion detection. Group shots and crowd
  photos will often return no face tone score. Do not treat a missing score as neutral.
- The EXIF capture date staleness warning (photo > 72 hours older than article) is a signal,
  not a disqualification. Many legitimate news images are file photos. Surface the flag for
  analyst review in the `ValidationAgent`, do not auto-discard the image.

## 10.2 LLM Response Handling
- Ollama Cloud with `num_predict=80` (default) returns empty responses for complex prompts.
  Always set `max_tokens ≥ 256` for structured extraction calls.
- Do not trust LLM confidence scores above `MAX_CONFIDENCE` (default 0.82). The cap must be
  enforced after every LLM call that returns a confidence value.
- Multi-event extraction prompts must explicitly state "return only a JSON array" — the model
  will otherwise return a single object.

## 10.3 Actor Extraction
- The `_is_media_actor()` filter in `utils/text.py` must run on every extracted entity before
  it enters the actor graph. Skipping it floods the graph with news outlet names.
- Single-word capitalized tokens are high-noise. The regex in `extract_actors_from_articles()`
  is intentionally conservative — prefer false negatives over false positives for actor quality.

## 10.4 Confidence Score Discipline
- `MAX_CONFIDENCE` (default 0.82) is a hard cap — a deliberate epistemic constraint.
  Never remove or bypass this cap. It signals honest uncertainty in an open-source intelligence context.
- Storyboard panels with fewer than `MIN_CITATIONS` citations must be auto-supplemented from
  the article pool before the confidence score is finalized.

## 10.5 Imports
- `Counter` is in `collections` — `Count` does not exist. Always import as:
  `from collections import defaultdict, Counter`
- All `nx` (NetworkX) calls must check `G.number_of_nodes() > 0` before computing centrality
  or PageRank — both raise on empty graphs.

---

# 11. Test Conventions

- All tests are in `tests/` — unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Fixture data lives in `tests/fixtures/` as static JSON files
- Use the `mock_llm_client` fixture from `conftest.py` for all LLM-dependent tests
  (it returns pre-defined JSON responses without making real API calls)
- Tests must not make real external HTTP calls — mock `gdelt_client` and `rss_client`
- Name test functions as `test_<function_name>_<scenario>()` — e.g., `test_detect_spikes_empty_timeline()`
- Each agent module should have a corresponding `test_<agent_name>.py` with at minimum:
  - A happy-path test with valid fixture data
  - An empty-input graceful degradation test
  - A malformed-input test (e.g., bad JSON, missing fields)

---

# 12. Notebook vs. Package

The original notebook (`notebooks/gdelt_intelligence_pipeline_v2_4.ipynb`) is retained as
a reference implementation. The canonical codebase is the `geoeventfusion/` Python package.

The `notebooks/quickstart.ipynb` is the recommended Colab entry point — it imports the
package and calls `pipeline.run(config)` directly. Never copy pipeline logic back into
the notebook; keep notebooks thin.

---

# 13. File Ownership Map

When asked to modify a feature, use this map to find the right file:

| Feature                                      | Primary File(s)                                              |
|----------------------------------------------|--------------------------------------------------------------|
| GDELT API calls (all modes)                  | `clients/gdelt_client.py`                                    |
| Fetch mode orchestration (13-mode)           | `agents/gdelt_agent.py`                                      |
| GDELT query operator composition             | `analysis/query_builder.py`                                  |
| GKG theme suggestion (LLM-assisted)          | `analysis/query_builder.py`                                  |
| HybridRel / tone / toneabs fetch logic       | `agents/gdelt_agent.py`                                      |
| Source-scoped fetches (country/lang/domain)  | `agents/gdelt_agent.py`, `analysis/query_builder.py`         |
| Visual intelligence fetch (ImageCollageInfo) | `agents/gdelt_agent.py`, `clients/gdelt_client.py`           |
| Visual image novelty scoring                 | `analysis/visual_intel.py`                                   |
| TimelineVolRaw / vol_ratio computation       | `agents/gdelt_agent.py`, `analysis/tone_analyzer.py`         |
| LLM backend switching                        | `clients/llm_client.py`, `config/settings.py`                |
| Spike detection logic                        | `analysis/spike_detector.py`                                 |
| Actor filtering (media tokens)               | `utils/text.py`                                              |
| Actor graph + centrality                     | `analysis/actor_graph.py`                                    |
| Tone analysis                                | `analysis/tone_analyzer.py`                                  |
| Hypothesis debate (4 rounds)                 | `analysis/hypothesis_engine.py`                              |
| Event fusion + clustering                    | `agents/fusion_agent.py`                                     |
| Storyboard narrative generation              | `agents/storyboard_agent.py`                                 |
| Grounding validation                         | `agents/validation_agent.py`                                 |
| Visualization dark theme                     | `visualization/theme.py`                                     |
| Timeline chart (vol + volraw dual axis)      | `visualization/timeline_chart.py`                            |
| Visual intelligence image display            | `visualization/visual_intel_chart.py`                        |
| HTML storyboard report                       | `visualization/html_report.py`                               |
| JSON/file persistence                        | `io/persistence.py`                                          |
| Colab download helpers                       | `io/colab_helpers.py`                                        |
| CLI entrypoint                               | `scripts/run_pipeline.py`                                    |
| All thresholds and defaults                  | `config/defaults.py`                                         |
| Pipeline orchestration order                 | `geoeventfusion/pipeline.py`                                 |
