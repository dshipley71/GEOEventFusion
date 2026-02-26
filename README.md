# GEOEventFusion

**Multi-agent geopolitical intelligence pipeline** — transforms raw global event signals into
validated, evidence-grounded intelligence storyboards.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

GEOEventFusion is a modular, multi-agent pipeline for professional geopolitical intelligence analysis:

1. **Fetches global event data** from GDELT DOC 2.0 (up to 13 parallel query modes)
2. **Detects coverage spikes** using Z-score analysis and extracts actor co-occurrence graphs
3. **Enriches spikes** with RSS full-text articles and ground-truth conflict datasets (ACLED/ICEWS)
4. **Extracts structured events** via LLM (Anthropic or Ollama) — timelines, turning points, hypotheses
5. **Fuses events** across all sources into evidence-grounded intelligence storyboard panels
6. **Validates all claims** against cited article evidence with grounding scores
7. **Exports artifacts** — JSON, HTML storyboard, PNG charts, GEXF actor graph

Every output is evidence-grounded. Confidence scores are honest — never inflated past `MAX_CONFIDENCE = 0.82`.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Run the pipeline
python scripts/run_pipeline.py --query "Houthi Red Sea attacks" --days-back 90

# Validate environment
python scripts/validate_env.py
```

**Google Colab:** Open `notebooks/quickstart.ipynb` — set API keys in Colab Secrets and run all cells.

---

## Example Output

Running the pipeline for "Houthi Red Sea attacks" (90-day window) produces:

```
outputs/runs/20240116_143022_houthi_red_sea_attacks/
├── storyboard_report.html        # Full dark-theme intelligence storyboard
├── storyboard.json               # Structured panels with evidence citations
├── timeline.json                 # Phase-structured timeline with turning points
├── hypotheses.json               # 4-round adversarial hypothesis debate results
├── validation_report.json        # Grounding scores and verification flags
├── actor_network.gexf            # Actor graph (open in Gephi)
└── charts/
    ├── event_timeline_annotated.png
    ├── tone_distribution.png
    ├── timeline_language.png
    ├── actor_network.png
    └── source_country_map.html
```

---

## Architecture

```
GDELTAgent → RSSAgent → GroundTruthAgent → CustomDatasetAgent
                                                    ↓
                                         LLMExtractionAgent
                                                    ↓
                                              FusionAgent
                                                    ↓
                                           StoryboardAgent
                                                    ↓
                                          ValidationAgent
                                                    ↓
                                            ExportAgent
```

All agents share a `PipelineContext` object — no agent calls another directly.
See `docs/architecture.md` for the full design narrative.

---

## GDELT Fetch Modes

The `GDELTAgent` executes up to **13 parallel GDELT DOC 2.0 queries** across four groups:

| Group | Description |
|---|---|
| **A — Core Pools** | DateDesc, ToneAsc, ToneDesc, HybridRel, tone< filter, toneabs> filter |
| **B — Timeline Signals** | TimelineVolInfo, TimelineVolRaw, TimelineTone, TimelineLang, TimelineSourceCountry, ToneChart |
| **C — Source Scoped** | `sourcecountry:`, `sourcelang:`, `domainis:` (conditional) |
| **D — Visual Intel** | ImageCollageInfo, WordCloudImageTags (optional, `enable_visual_intel=True`) |

---

## CLI Reference

```bash
# Full options
python scripts/run_pipeline.py --help

# Key flags
--query "your query"                  # Required
--days-back 90                        # Analysis window
--llm-backend anthropic               # or ollama
--test-mode                           # No API calls, use fixtures
--ground-truth-sources acled          # Add ACLED ground truth
--source-country-filter RS            # Add Russia-sourced articles
--authoritative-domains un.org nato.int
--enable-visual-intel                 # Enable image analysis

# Batch run
python scripts/batch_run.py --config configs/batch_queries.yaml
```

---

## LLM Backends

| Backend | Default Model | Config |
|---|---|---|
| Anthropic | `claude-sonnet-4-6` | Set `ANTHROPIC_API_KEY` in `.env` |
| Ollama | `gemma3:27b` | Install Ollama, pull model, set `OLLAMA_HOST` |

```bash
# Switch backends
python scripts/run_pipeline.py --query "..." --llm-backend anthropic
python scripts/run_pipeline.py --query "..." --llm-backend ollama
```

---

## Configuration

All defaults live in `config/defaults.py`. Override via `PipelineConfig` or environment variables.

```python
from config.settings import PipelineConfig

config = PipelineConfig(
    query="Taiwan Strait tensions",
    days_back=90,
    llm_backend="anthropic",
    source_country_filter="CH",           # China-sourced articles
    authoritative_domains=["state.gov"],  # Official US government sources
    ground_truth_sources=["acled"],       # ACLED conflict data
)
```

See `docs/configuration.md` for the complete field reference.

---

## Tests

```bash
pytest tests/ -v                              # All tests
pytest tests/unit/ -v                         # Unit tests only
pytest tests/ --cov=geoeventfusion            # With coverage
```

---

## Documentation

| File | Contents |
|---|---|
| `CLAUDE.md` | AI assistant guide — architecture, conventions, gotchas |
| `AGENTS.md` | Agent contracts — I/O schemas, failure handling |
| `skills.md` | Capability inventory |
| `docs/architecture.md` | System design and data flow |
| `docs/configuration.md` | Full configuration reference |
| `docs/deployment.md` | Colab, local, Docker, API setup |
| `docs/agents/agent_reference.md` | Per-agent quick reference |
| `docs/agents/adding_agents.md` | How to implement a new agent |
| `docs/api/reference.md` | Public Python API reference |

---

## License

MIT — see `LICENSE` for details.
