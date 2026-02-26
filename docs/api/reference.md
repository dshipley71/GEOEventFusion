# GEOEventFusion — Public Python API Reference

This document describes the public API surface of the `geoeventfusion` package.

---

## Top-Level Entry Point

### `geoeventfusion.pipeline.run_pipeline`

```python
from geoeventfusion.pipeline import run_pipeline
from config.settings import PipelineConfig

config = PipelineConfig(query="Houthi Red Sea attacks", days_back=90)
context = run_pipeline(config)
```

**Args:**
- `config: PipelineConfig` — fully-configured pipeline settings

**Returns:**
- `PipelineContext` — shared state object with all agent results populated

**Notes:**
- Runs all pipeline phases in order
- Caches intermediate results to disk for resumability
- Phase-level caching enables restarts without re-running the full pipeline

---

## Configuration

### `config.settings.PipelineConfig`

```python
from config.settings import PipelineConfig

config = PipelineConfig(
    query="Taiwan Strait tensions",
    days_back=90,
    llm_backend="anthropic",
    max_records=250,
    enable_visual_intel=False,
)
```

See `docs/configuration.md` for the full field reference.

---

## Data Models

All agent output schemas are exported from `geoeventfusion.models`:

```python
from geoeventfusion.models import (
    # Events and articles
    Article,
    GDELTAgentResult,
    SpikeWindow,
    TimelineStep,
    TimelineStepRaw,
    ToneChartBin,
    ToneStats,
    LanguageStats,
    CountryStats,
    GroundTruthEvent,

    # Actors
    ActorNode,
    ActorEdge,
    ActorGraph,
    CentralityResult,

    # Fusion
    FusionCluster,
    FusionAgentResult,
    ContradictionFlag,
    FusionStats,

    # Storyboard and extraction
    StoryboardPanel,
    StoryboardAgentResult,
    Hypothesis,
    LLMExtractionAgentResult,
    TimelineEntry,
    TimelinePhase,
    TurningPoint,

    # Validation
    ValidationAgentResult,
    GroundingFlag,
    VerificationFlag,

    # Visual intelligence
    VisualImage,

    # Export
    ExportManifest,
    ArtifactRecord,
    ExportAgentResult,

    # Pipeline context
    PipelineContext,
    AgentResult,
    PhaseRecord,
)
```

---

## Running Individual Agents

Agents can be run in isolation without the full pipeline:

```python
from geoeventfusion.agents.gdelt_agent import GDELTAgent
from geoeventfusion.models.pipeline import PipelineContext
from config.settings import PipelineConfig

config = PipelineConfig(query="Houthi Red Sea", test_mode=True)
context = PipelineContext(config=config, run_id="test_001", output_dir="/tmp/test")

agent = GDELTAgent(config=config)
context.gdelt_result = agent.run(context)

print(f"Spikes detected: {len(context.gdelt_result.spikes)}")
print(f"Articles fetched: {len(context.gdelt_result.articles_recent)}")
```

---

## Analysis Functions

### Spike Detection

```python
from geoeventfusion.analysis.spike_detector import detect_spikes, compute_vol_ratio

spikes = detect_spikes(timeline_steps, z_threshold=1.5, query="my query")
vol_ratio = compute_vol_ratio(timeline_volraw_steps)
```

### Actor Graph

```python
from geoeventfusion.analysis.actor_graph import build_actor_graph

graph = build_actor_graph(
    co_occurrence_triples,   # List of (actor_a, actor_b, date) tuples
    hub_top_n=10,
    broker_ratio_threshold=0.5,
    pagerank_max_iter=200,
)
```

### Tone Analysis

```python
from geoeventfusion.analysis.tone_analyzer import analyze_tone_distribution

tone_stats = analyze_tone_distribution(tonechart_bins)
print(f"Modal tone: {tone_stats.modal_tone}")
print(f"Mean tone: {tone_stats.mean_tone:.2f}")
```

### Query Builder

```python
from geoeventfusion.analysis.query_builder import QueryBuilder

qb = QueryBuilder(
    base_query="Houthi Red Sea attacks",
    repeat_threshold=3,
    near_window=15,
)
repeat_query = qb.build_repeat_query()
```

---

## LLM Client

```python
from geoeventfusion.clients.llm_client import LLMClient

llm = LLMClient(
    backend="anthropic",
    anthropic_model="claude-sonnet-4-6",
    anthropic_api_key="sk-ant-...",
    max_confidence=0.82,
)

# Plain text call
response = llm.call(
    system="You are a geopolitical analyst.",
    prompt="Summarize the Houthi situation in one paragraph.",
    max_tokens=256,
)

# JSON-only call with defensive parsing
data = llm.call_json(
    system="Return only valid JSON.",
    prompt='Extract events as: [{"event_type": "", "date": "", ...}]',
    max_tokens=1024,
)
```

---

## Visualization

```python
from geoeventfusion.visualization.timeline_chart import render_timeline_chart
from geoeventfusion.visualization.actor_network import render_actor_network_chart
from geoeventfusion.visualization.choropleth import render_choropleth_map

# Render and save charts
render_timeline_chart(
    timeline_steps=context.gdelt_result.timeline_volinfo,
    spikes=context.gdelt_result.spikes,
    output_path="charts/event_timeline_annotated.png",
)

render_actor_network_chart(
    actor_graph=context.gdelt_result.actor_graph,
    output_path="charts/actor_network.png",
)

render_choropleth_map(
    country_stats=context.gdelt_result.country_stats,
    output_path="charts/source_country_map.html",
)
```

---

## I/O Helpers

```python
from geoeventfusion.io.persistence import save_json, load_json

# Save any dict to JSON with atomic writes
save_json({"key": "value"}, "/path/to/output.json")

# Load JSON safely
data = load_json("/path/to/output.json")
```
