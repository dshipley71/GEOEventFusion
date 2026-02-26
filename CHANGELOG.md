# GEOEventFusion — Changelog

All notable changes are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.4.0] — 2026-02-25

### Package Launch (v2.4 Notebook → Python Package)

This release marks the full refactor from a monolithic Jupyter notebook (`v2.4`) into the
`geoeventfusion` modular Python package. All issues documented in `AGENTS.md §8` are resolved.

#### New Features
- **13-mode parallel GDELT fetch architecture** replacing the original 3-mode fetch
  - Group A: 6 core article pools including `HybridRel`, inline `tone<`, and `toneabs>` filters
  - Group B: 6 timeline and signal modes including `TimelineVolRaw` for absolute volume metrics
  - Group C: Conditional source-scoped fetches (`sourcecountry:`, `sourcelang:`, `domainis:`)
  - Group D: Optional visual intelligence modes (`ImageCollageInfo`, `WordCloudImageTags`)
- **`LLMExtractionAgent`** — structured event extraction, timeline generation, 4-round hypothesis debate
- **`FusionAgent`** — multi-source event clustering with 5-dimensional similarity scoring
- **`StoryboardAgent`** — narrative intelligence panel generation with evidence citation
- **`ValidationAgent`** — grounding score aggregation, URL reachability, corroboration checks
- **`ExportAgent`** — JSON/HTML/PNG/GEXF artifact export with run manifest
- **`GroundTruthAgent`** — ACLED and ICEWS event data integration
- **`CustomDatasetAgent`** — CSV/JSON/SQL/API cross-reference with similarity scoring
- **`RSSAgent`** — full-text RSS/Atom feed ingestion with deduplication
- **`QueryBuilder`** — GDELT operator composition with LLM-assisted GKG theme suggestion
- **`vol_ratio`** — story prominence metric derived from `TimelineVolRaw` norm field
- **Visual intelligence novelty scoring** — `1.0 / (1.0 + log(1 + web_appearance_count))`
- **Community detection** — temporal graph reorganization for phase boundary detection
- **`BaseAgent` ABC** — standardized agent contract with `run()`, `validate_output()`, `reset()`
- **`PipelineContext`** — typed shared state object threaded through all pipeline phases
- **Phase-level caching** — resumable pipeline via cached intermediate JSON files
- **Test mode** — `config.test_mode=True` uses fixture data, no real API calls
- **`scripts/batch_run.py`** — batch runner for multiple queries from YAML config
- **`notebooks/quickstart.ipynb`** — thin Colab entry point delegating to the package
- **Comprehensive test suite** — unit and integration tests with fixture data

#### Breaking Changes (from v2.4 notebook)
- All logic moved from notebook cells to `geoeventfusion/` Python package
- Output directory changed from flat `data/` to `outputs/runs/<run_id>/`
- `PipelineConfig` replaces all module-level globals and hardcoded values
- Model names moved to `PipelineConfig.anthropic_model` / `PipelineConfig.ollama_model`
- All LLM calls must go through `LLMClient.call()` — no direct `anthropic` library calls

#### Bug Fixes
- Fixed `from collections import Count` (typo) → `Counter`
- Fixed `NetworkX` centrality and PageRank raising on empty graphs (added `number_of_nodes() > 0` guards)
- Fixed GDELT HTTP header bleed-through in response bodies (`_safe_parse_json()` guard)
- Fixed single-word `near<N>:` operator producing zero results for short terms
- Fixed `TimelineVolRaw` norm field being incorrectly smoothed in vol_ratio computation

---

## [2.3.0] — Previous (Notebook Only)

Single-file Jupyter notebook with 3 article pools (DateDesc, ToneAsc, ToneDesc),
inline LLM extraction, and flat `data/` output directory. Archived as
`notebooks/gdelt_intelligence_pipeline_v2_4.ipynb`.

---

*Earlier versions are not tracked in this changelog.*
