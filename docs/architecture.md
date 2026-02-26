# GEOEventFusion — System Architecture

## Overview

GEOEventFusion is a modular, multi-agent geopolitical intelligence pipeline that transforms
raw global event signals into validated, evidence-grounded intelligence storyboards. It is
designed for professional intelligence analysis where every output must be evidence-grounded
and confidence scores must be honest.

---

## Data Flow

```
PipelineConfig
     │
     ▼
PipelineContext (instantiated once, threaded through all agents)
     │
     ├─► GDELTAgent.run(ctx)             → ctx.gdelt_result
     │     13-mode parallel GDELT fetch
     │     Spike detection (Z-score)
     │     Actor co-occurrence graph
     │
     ├─► RSSAgent.run(ctx)               → ctx.rss_result
     │     Feed ingestion for spike windows
     │     Full-text extraction
     │
     ├─► GroundTruthAgent.run(ctx)       → ctx.ground_truth_result
     │     ACLED / ICEWS event data
     │
     ├─► CustomDatasetAgent.run(ctx)     → ctx.custom_dataset_result
     │     User-provided CSV / JSON / SQL
     │
     ├─► LLMExtractionAgent.run(ctx)     → ctx.llm_result
     │     Phase 3: Timeline + turning points
     │     Phase 4: 4-round adversarial hypotheses
     │     Phase 6: Follow-up enrichment briefs
     │
     ├─► FusionAgent.run(ctx)            → ctx.fusion_result
     │     Multi-source event clustering
     │     Contradiction detection
     │
     ├─► StoryboardAgent.run(ctx)        → ctx.storyboard_result
     │     Narrative panel generation
     │     Evidence citation
     │
     ├─► ValidationAgent.run(ctx)        → ctx.validation_result
     │     URL reachability checks
     │     Ground truth alignment
     │     Grounding score aggregation
     │
     └─► ExportAgent.run(ctx)            → ctx.export_result
           JSON / HTML / PNG / GEXF artifacts
```

---

## Layer Architecture

| Layer | Location | Responsibility |
|---|---|---|
| API clients | `geoeventfusion/clients/` | HTTP calls only — no business logic |
| Data models | `geoeventfusion/models/` | Schema definitions only — no side effects |
| Analysis logic | `geoeventfusion/analysis/` | Pure functions — no I/O, no API calls |
| Agents | `geoeventfusion/agents/` | Orchestrate clients + analysis; write to context |
| Visualization | `geoeventfusion/visualization/` | Rendering only — no data transformation |
| I/O | `geoeventfusion/io/` | File read/write only — no business logic |
| Utilities | `geoeventfusion/utils/` | Stateless helpers — no state, no external calls |
| Configuration | `config/` | Settings loading and defaults |

---

## GDELT Fetch Architecture

The `GDELTAgent` executes up to 13 parallel GDELT DOC 2.0 API calls organized into four groups:

### Group A — Core Article Pools (always active, 6 fetches)
| Pool | Sort/Filter | Intelligence Value |
|---|---|---|
| `articles_recent` | `DateDesc` | Full recency coverage |
| `articles_negative` | `ToneAsc` | Most negatively-toned articles |
| `articles_positive` | `ToneDesc` | Most positively-toned articles |
| `articles_relevant` | `HybridRel` | Prominent outlets + high relevance |
| `articles_high_neg` | `DateDesc` + `tone<-5` | Hard negative threshold |
| `articles_high_emotion` | `DateDesc` + `toneabs>8` | Crisis/alarm coverage |

### Group B — Timeline and Signal Modes (always active, 6 fetches)
`TimelineVolInfo`, `TimelineVolRaw`, `TimelineTone`, `TimelineLang`, `TimelineSourceCountry`, `ToneChart`

### Group C — Targeted Source Fetches (conditional)
Activated by `source_country_filter`, `source_lang_filter`, or `authoritative_domains` config.

### Group D — Visual Intelligence (conditional, `enable_visual_intel=True`)
`ImageCollageInfo` with `imagetag:` filters, `WordCloudImageTags`

---

## Confidence Score Discipline

`MAX_CONFIDENCE = 0.82` is a hard cap enforced after every LLM extraction call.
This is a deliberate epistemic constraint — it signals honest uncertainty in an
open-source intelligence context. Never remove or bypass this cap.

---

## Pipeline Resumability

Each agent checks for cached intermediate outputs before executing. If a cached result
exists in the run output directory, the agent loads it and skips execution. This enables
phase-level restarts without re-running the full pipeline.

---

## Deployment Modes

| Mode | Entry Point | Notes |
|---|---|---|
| Google Colab | `notebooks/quickstart.ipynb` | Thin notebook, delegates to package |
| Local Python | `scripts/run_pipeline.py` | Full CLI with argparse |
| Batch | `scripts/batch_run.py` | Multiple queries from YAML config |
| Docker | `Dockerfile` (future) | Containerized, all deps bundled |
| API Server | `geoeventfusion/server.py` (future) | FastAPI wrapper |
