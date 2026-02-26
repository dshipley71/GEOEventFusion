# GEOEventFusion — Configuration Reference

All runtime configuration flows through `PipelineConfig` in `config/settings.py`.
Default values for all thresholds live in `config/defaults.py`.

---

## PipelineConfig Fields

### Core Query Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | `""` | Base geopolitical query string |
| `days_back` | `int` | `90` | Analysis window in days (GDELT max: ~90) |
| `max_records` | `int` | `250` | Articles per GDELT fetch (GDELT max: 250) |

### LLM Backend

| Field | Type | Default | Description |
|---|---|---|---|
| `llm_backend` | `str` | `"ollama"` | Active backend: `"anthropic"` or `"ollama"` |
| `anthropic_model` | `str` | `"claude-sonnet-4-6"` | Anthropic model ID |
| `ollama_model` | `str` | `"gemma3:27b"` | Ollama model name |
| `ollama_host` | `str` | `"http://localhost:11434"` | Ollama server URL |
| `llm_temperature` | `float` | `0.3` | LLM sampling temperature |
| `llm_max_tokens` | `int` | `4096` | Maximum tokens per LLM response |
| `llm_min_max_tokens` | `int` | `256` | Minimum `max_tokens` for structured calls |

### API Credentials (environment variables only)

| Environment Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ACLED_API_KEY` | ACLED REST API key |
| `ACLED_EMAIL` | ACLED account email |
| `LLM_BACKEND` | Override LLM backend |
| `ANTHROPIC_MODEL` | Override Anthropic model |
| `OLLAMA_MODEL` | Override Ollama model |
| `OLLAMA_HOST` | Override Ollama host URL |
| `OUTPUT_ROOT` | Override output root directory |
| `LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) |

### Spike Detection

| Field | Type | Default | Description |
|---|---|---|---|
| `spike_z_threshold` | `float` | `1.5` | Z-score threshold for spike detection |
| `max_spikes` | `int` | `5` | Maximum spikes to process |
| `spike_backfill_hours` | `int` | `48` | Backfill window around each spike (±hours) |

### GDELT Fetch Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `domain_cap_pct` | `float` | `0.20` | Max fraction of articles from any single domain |
| `timeline_smooth` | `int` | `3` | GDELT timeline smoothing window (1–30) |
| `repeat_threshold` | `int` | `3` | Min keyword repetitions for `repeat<N>` operator |
| `near_window` | `int` | `15` | Word proximity window for `near<N>` operator |
| `near_min_term_length` | `int` | `5` | Min term length to use `near<N>` |
| `tone_negative_threshold` | `float` | `-5.0` | Tone ceiling for `articles_high_neg` pool |
| `toneabs_threshold` | `float` | `8.0` | Min absolute tone for `articles_high_emotion` pool |
| `gdelt_stagger_seconds` | `float` | `0.75` | Delay between concurrent GDELT requests |
| `gdelt_max_workers` | `int` | `2` | ThreadPoolExecutor workers for GDELT fetches |
| `gdelt_max_retries` | `int` | `5` | Max retries on GDELT failures |
| `gdelt_backoff_base` | `float` | `2.0` | Exponential backoff base (seconds) |
| `gdelt_request_timeout` | `int` | `30` | HTTP timeout per GDELT request (seconds) |

### Conditional GDELT Fetch Pools

| Field | Type | Default | Description |
|---|---|---|---|
| `source_country_filter` | `Optional[str]` | `None` | FIPS country code for country-scoped fetch |
| `source_lang_filter` | `Optional[str]` | `None` | ISO 3-char language code for language-scoped fetch |
| `authoritative_domains` | `List[str]` | `[]` | Domains for `domainis:` authority fetch |
| `visual_imagetags` | `List[str]` | `[]` | VGKG imagetag values for visual intelligence fetch |
| `enable_visual_intel` | `bool` | `False` | Enable visual intelligence fetch modes |
| `enable_word_clouds` | `bool` | `False` | Enable image word cloud modes |

### RSS Feed Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `rss_feed_list` | `List[str]` | `[]` | RSS/Atom feed URLs to ingest |
| `rss_max_articles_per_spike` | `int` | `50` | Article cap per spike window |
| `rss_time_window_hours` | `int` | `48` | Time window around spike for RSS filtering |
| `rss_request_timeout` | `int` | `15` | HTTP timeout per RSS request (seconds) |
| `rss_dedup_threshold` | `float` | `0.85` | Levenshtein similarity threshold for deduplication |

### Ground Truth Datasets

| Field | Type | Default | Description |
|---|---|---|---|
| `ground_truth_sources` | `List[str]` | `[]` | Active sources: `["acled"]`, `["icews"]`, or both |
| `ground_truth_country_filter` | `List[str]` | `[]` | ISO country codes or names to filter |
| `ground_truth_event_types` | `List[str]` | `[]` | Event type filters (ACLED/ICEWS categories) |

### Fusion Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `fusion_weights` | `FusionWeights` | (see below) | Per-dimension fusion scoring weights |
| `fusion_temporal_window_hours` | `int` | `72` | Temporal proximity window for event matching |
| `fusion_geographic_threshold_km` | `float` | `200.0` | Geographic proximity threshold in km |

**Default FusionWeights:**
```python
temporal=0.25, geographic=0.25, actor=0.20, semantic=0.20, event_type=0.10
# Must sum to 1.0
```

### Confidence and Citation

| Field | Type | Default | Description |
|---|---|---|---|
| `max_confidence` | `float` | `0.82` | Hard confidence cap — never exceed this |
| `min_citations` | `int` | `3` | Minimum citations per storyboard panel |
| `min_panel_confidence` | `float` | `0.40` | Minimum panel confidence threshold |

### Validation Thresholds

| Field | Type | Default | Description |
|---|---|---|---|
| `validation_title_similarity_threshold` | `float` | `0.55` | Title-to-claim Levenshtein similarity |
| `validation_ground_truth_similarity_threshold` | `float` | `0.65` | Ground truth fuzzy match threshold |
| `validation_custom_match_threshold` | `float` | `0.50` | Custom dataset match confidence |
| `validation_date_delta_days` | `int` | `7` | Max days between article date and claimed event |
| `validation_url_timeout` | `int` | `10` | HTTP HEAD timeout for URL checks (seconds) |
| `validation_min_corroboration` | `int` | `2` | Minimum source domains for corroboration |

### Test Mode

| Field | Type | Default | Description |
|---|---|---|---|
| `test_mode` | `bool` | `False` | Use fixture data, no real API calls |

---

## Environment Variable Setup

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
ACLED_API_KEY=your-acled-key
ACLED_EMAIL=your@email.com
LLM_BACKEND=anthropic
OUTPUT_ROOT=outputs
LOG_LEVEL=INFO
```

---

## Batch Config File Format (YAML)

Used with `scripts/batch_run.py`:

```yaml
defaults:
  llm_backend: ollama
  days_back: 90
  max_records: 250

queries:
  - query: "Houthi Red Sea attacks"
    days_back: 90
    ground_truth_sources: [acled]
    ground_truth_country_filter: [Yemen]

  - query: "Taiwan Strait tensions"
    llm_backend: anthropic
    source_country_filter: "CH"

  - query: "Sudan conflict 2024"
    ground_truth_sources: [acled]
    authoritative_domains: [un.org, state.gov]
```
