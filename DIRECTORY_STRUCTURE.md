# GEOEventFusion — Application Directory Structure
Version: 1.0  
Date: 2026-02-25

---

## Overview

This document specifies the canonical directory layout for the GEOEventFusion application.
The structure separates concerns cleanly across agents, clients, models, analysis, visualization,
I/O, and utilities — enabling independent scaling, testing, and replacement of each component
without disrupting the rest of the pipeline.

---

## Full Directory Tree

```
geoeventfusion/
│
├── README.md                         # Project overview, quickstart, feature summary
├── AGENTS.md                         # Agent architecture specification
├── skills.md                         # System skills & capabilities reference
├── claude.md                         # AI assistant project guide (this codebase's context for Claude)
├── CHANGELOG.md                      # Version history and breaking changes
├── LICENSE                           # License file
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git exclusions (data/, outputs/, __pycache__, .env)
├── pyproject.toml                    # Project metadata, build system, tool config (ruff, mypy, pytest)
├── requirements.txt                  # Core runtime dependencies
├── requirements-dev.txt              # Dev/test dependencies (pytest, ruff, mypy, coverage)
│
├── config/
│   ├── __init__.py
│   ├── settings.py                   # PipelineConfig dataclass / Pydantic BaseSettings
│   ├── defaults.py                   # All default values for thresholds, caps, models
│   └── logging.yaml                  # Logging configuration (handlers, levels, formatters)
│
├── geoeventfusion/                   # Main installable package
│   ├── __init__.py                   # Package version, public API surface
│   ├── pipeline.py                   # Top-level orchestrator — runs all phases in order
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseAgent ABC — run(), validate_output(), reset()
│   │   ├── gdelt_agent.py            # GDELTAgent — multi-mode fetch, spike detection
│   │   ├── rss_agent.py              # RSSAgent — feed ingestion, dedup, text normalization
│   │   ├── ground_truth_agent.py     # GroundTruthAgent — ACLED/ICEWS ingestion, alignment
│   │   ├── custom_dataset_agent.py   # CustomDatasetAgent — CSV/JSON/SQL cross-reference
│   │   ├── llm_extraction_agent.py   # LLMExtractionAgent — structured event extraction
│   │   ├── fusion_agent.py           # FusionAgent — multi-source event linking
│   │   ├── storyboard_agent.py       # StoryboardAgent — narrative panel generation
│   │   ├── validation_agent.py       # ValidationAgent — evidence grounding checks
│   │   └── export_agent.py           # ExportAgent — JSON/HTML/graph artifact export
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── gdelt_client.py           # GDELT DOC 2.0 REST client (retry, backoff, rate limit)
│   │   ├── llm_client.py             # Dual-backend LLM (Ollama Cloud / Anthropic)
│   │   ├── rss_client.py             # RSS/Atom feed fetcher with full-text extraction
│   │   └── ground_truth_client.py    # ACLED API + ICEWS file loader clients
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py                 # PipelineConfig, LLMConfig, GDELTConfig dataclasses
│   │   ├── events.py                 # GDELTEvent, Article, SpikeWindow, GroundTruthEvent
│   │   ├── actors.py                 # ActorNode, ActorEdge, ActorGraph, CentralityResult
│   │   ├── fusion.py                 # FusionCluster, FusionResult, ContradictionFlag
│   │   ├── storyboard.py             # StoryboardPanel, Hypothesis, TimelineEntry
│   │   ├── validation.py             # ValidationReport, GroundingScore, UnverifiedFlag
│   │   ├── export.py                 # ExportManifest, ArtifactRecord
│   │   └── visual.py                  # VisualImage, ImageTopicTag, NoveltyScore
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── spike_detector.py         # Z-score spike detection, rolling baseline
│   │   ├── actor_graph.py            # NetworkX graph build, centrality, community detection
│   │   ├── tone_analyzer.py          # Tone chart analysis, language/country coverage
│   │   ├── query_builder.py          # GDELT query construction, GKG theme suggestion
│   │   ├── hypothesis_engine.py      # 4-round adversarial debate logic
│   │   └── visual_intel.py            # Visual image novelty scoring, VGKG tag parsing
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── theme.py                  # Shared dark-theme constants and color helpers
│   │   ├── timeline_chart.py         # Coverage volume timeline (GDELT-native)
│   │   ├── tone_chart.py             # Tone distribution histogram
│   │   ├── language_chart.py         # Language stacked area chart
│   │   ├── actor_network.py          # Actor network — Hub/Broker/Peripheral roles
│   │   ├── choropleth.py             # Source country folium choropleth map
│   │   ├── html_report.py            # HTML storyboard panel renderer
│   │   └── visual_intel_chart.py      # Visual intelligence image grid and novelty display
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── persistence.py            # save_json, load_json, atomic writes
│   │   ├── exporters.py              # JSON export, HTML export, graph export (networkx)
│   │   └── colab_helpers.py          # Colab-specific download helpers, secrets loading
│   │
│   └── utils/
│       ├── __init__.py
│       ├── text.py                   # Media token filtering, text normalization
│       ├── date_utils.py             # Date normalization, datetime range helpers
│       ├── geo_utils.py              # Lat/lon distance, country centroid lookup
│       ├── levenshtein_utils.py      # Fuzzy title matching for grounding validation
│       └── logging_utils.py          # Logger factory, structured log formatting
│
├── notebooks/
│   ├── gdelt_intelligence_pipeline_v2_4.ipynb  # Reference / archived original
│   ├── quickstart.ipynb              # Clean thin notebook — calls geoeventfusion package
│   └── dev_sandbox.ipynb             # Experimental / dev scratch notebook
│
├── data/
│   ├── .gitkeep
│   ├── spikes/                       # Per-spike article JSON files
│   ├── cache/                        # Optional disk-based response cache
│   └── custom_datasets/              # User-provided CSV/JSON reference datasets
│
├── outputs/
│   ├── .gitkeep
│   └── runs/                         # Timestamped output directories per pipeline run
│       └── YYYYMMDD_HHMMSS_<query>/
│           ├── run_metadata.json
│           ├── storyboard.json
│           ├── timeline.json
│           ├── hypotheses.json
│           ├── validation_report.json
│           ├── storyboard_report.html
│           └── charts/
│               ├── event_timeline_annotated.png
│               ├── tone_distribution.png
│               ├── timeline_language.png
│               ├── actor_network.png
│               └── source_country_map.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures, mock LLM client
│   ├── unit/
│   │   ├── test_gdelt_client.py
│   │   ├── test_spike_detector.py
│   │   ├── test_actor_graph.py
│   │   ├── test_tone_analyzer.py
│   │   ├── test_query_builder.py
│   │   ├── test_llm_extraction.py
│   │   ├── test_fusion_agent.py
│   │   ├── test_validation_agent.py
│   │   └── test_persistence.py
│   ├── integration/
│   │   ├── test_pipeline_phases.py   # End-to-end phase tests with fixture data
│   │   └── test_export_agent.py
│   └── fixtures/
│       ├── sample_artlist.json
│       ├── sample_timeline_volinfo.json
│       ├── sample_tonechart.json
│       ├── sample_storyboard.json
│       └── sample_ground_truth.json
│
├── scripts/
│   ├── run_pipeline.py               # CLI entrypoint (argparse / typer)
│   ├── validate_env.py               # Pre-flight check: API keys, dependencies, disk space
│   └── batch_run.py                  # Batch runner for multiple queries from a config file
│
└── docs/
    ├── architecture.md               # System design narrative, data flow diagram
    ├── configuration.md              # All config parameters, types, defaults, descriptions
    ├── deployment.md                 # Colab, local, Docker, API server setup guides
    ├── agents/
    │   ├── agent_reference.md        # Per-agent input/output/failure reference
    │   └── adding_agents.md          # How to implement and register a new agent
    └── api/
        └── reference.md              # Public Python API reference (auto-gen with mkdocs)
```

---

## Key File Descriptions

### `claude.md`
The AI assistant project guide. Provides Claude with the authoritative context needed to work
on this codebase: architecture rules, module responsibility boundaries, coding conventions,
common commands, pipeline phase reference, data flow diagram, known gotchas, and a file
ownership map. Should be read before making any code changes. Updated whenever significant
architectural decisions are made or new gotchas are discovered.

### `geoeventfusion/pipeline.py`
The top-level orchestrator. Accepts a `PipelineConfig`, instantiates each agent in order,
threads intermediate results between agents, and writes the run manifest to `outputs/runs/`.
Each phase is individually resumable via cached intermediate files. The `GDELTAgent` executes
up to 13 parallel GDELT DOC 2.0 API calls across four functional groups (core article pools,
timeline/signal modes, targeted source fetches, and visual intelligence modes).

### `geoeventfusion/agents/base.py`
Defines the `BaseAgent` abstract class:
```python
class BaseAgent(ABC):
    def __init__(self, config: PipelineConfig): ...
    @abstractmethod
    def run(self, context: PipelineContext) -> AgentResult: ...
    def validate_output(self, result: AgentResult) -> bool: ...
    def reset(self) -> None: ...
```

### `geoeventfusion/models/config.py`
All configuration is expressed as typed dataclasses or Pydantic models.
`PipelineConfig` is the single object passed to every agent.

### `config/settings.py`
Loads configuration from environment variables, `.env` file, or inline overrides.
No hard-coded credentials anywhere in the codebase.

### `outputs/runs/`
Every pipeline execution writes to a unique timestamped subdirectory, preserving all
intermediate and final artifacts for reproducibility and post-run inspection.

---

## Deployment Profiles

| Profile       | Entry Point                          | Notes                                    |
|---------------|--------------------------------------|------------------------------------------|
| Colab         | `notebooks/quickstart.ipynb`         | Thin notebook imports geoeventfusion pkg |
| Local Python  | `scripts/run_pipeline.py`            | Full CLI with argparse                   |
| Batch         | `scripts/batch_run.py`               | Multiple queries from YAML config file   |
| Docker        | `Dockerfile` (future)                | Self-contained container with all deps   |
| API Server    | `geoeventfusion/server.py` (future)  | FastAPI wrapper over pipeline.py         |
