"""GEOEventFusion — All default threshold values and configuration constants.

All tuneable values live here. Never hard-code magic numbers in source files.
Import constants from this module; override via PipelineConfig at runtime.
"""

# ── Confidence scoring ─────────────────────────────────────────────────────────
# Hard epistemic cap on LLM confidence scores. Never raise this value.
MAX_CONFIDENCE: float = 0.82

# Minimum confidence below which storyboard panels are flagged LOW_CONFIDENCE
MIN_PANEL_CONFIDENCE: float = 0.25

# ── Citation requirements ──────────────────────────────────────────────────────
# Minimum number of cited sources per storyboard panel before confidence is finalized
MIN_CITATIONS: int = 3

# ── GDELT article fetching ─────────────────────────────────────────────────────
# Maximum articles returned per GDELT ArtList call (GDELT hard limit: 250)
MAX_RECORDS: int = 250

# Default analysis window in days
DAYS_BACK: int = 90

# Maximum fraction of any one domain across all article pools (diversity cap)
DOMAIN_CAP_PCT: float = 0.20

# ── GDELT timeline smoothing ───────────────────────────────────────────────────
# Smoothing window for TimelineVolInfo (GDELT TIMELINESMOOTH parameter, max 30)
TIMELINE_SMOOTH: int = 3

# ── Spike detection ────────────────────────────────────────────────────────────
# Z-score threshold for declaring a coverage spike
SPIKE_Z_THRESHOLD: float = 1.5

# Number of top spikes to retain for downstream enrichment
MAX_SPIKES: int = 10

# Hours on each side of a spike date for article backfill fetch
SPIKE_BACKFILL_HOURS: int = 48

# ── Tone filters ───────────────────────────────────────────────────────────────
# Tone ceiling for articles_high_neg pool (tone< operator)
TONE_NEGATIVE_THRESHOLD: float = -5.0

# Absolute tone floor for articles_high_emotion pool (toneabs> operator)
TONEABS_THRESHOLD: float = 8.0

# ── GDELT query operators ──────────────────────────────────────────────────────
# Minimum keyword repetitions for repeat<N>: operator
REPEAT_THRESHOLD: int = 3

# Minimum term length for near<N>: operator (shorter terms produce zero results)
NEAR_MIN_TERM_LENGTH: int = 5

# Proximity window for near<N>: operator (number of words)
NEAR_WINDOW: int = 15

# ── GDELT rate limits ──────────────────────────────────────────────────────────
# Minimum seconds between successive GDELT API submissions
GDELT_STAGGER_SECONDS: float = 0.75

# Maximum concurrent GDELT fetch workers
GDELT_MAX_WORKERS: int = 2

# Maximum retry attempts on GDELT HTTP failures
GDELT_MAX_RETRIES: int = 5

# Base seconds for GDELT exponential backoff
GDELT_BACKOFF_BASE: float = 2.0

# HTTP request timeout for GDELT calls (seconds)
GDELT_REQUEST_TIMEOUT: int = 30

# ── LLM backends ──────────────────────────────────────────────────────────────
# Default active LLM backend: "anthropic" or "ollama"
LLM_BACKEND: str = "ollama"

# Anthropic model identifier
ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

# Ollama model identifier
OLLAMA_MODEL: str = "gemma3:27b"

# Default Ollama server base URL.
# Override via the OLLAMA_HOST environment variable or PipelineConfig(ollama_host=...)
# for cloud-hosted Ollama instances (e.g. 'https://api.ollama.com') or custom deployments.
OLLAMA_HOST: str = "http://localhost:11434"

# Ollama Cloud API key for Bearer token authentication.
# Required when ollama_host points to Ollama Cloud (https://api.ollama.com).
# Set via the OLLAMA_API_KEY environment variable; empty string disables auth headers.
OLLAMA_API_KEY: str = ""

# Minimum max_tokens for structured extraction calls
LLM_MIN_MAX_TOKENS: int = 256

# Default temperature for LLM extraction calls
LLM_TEMPERATURE: float = 0.1

# Default max_tokens for LLM calls unless overridden
LLM_DEFAULT_MAX_TOKENS: int = 2048

# ── RSS feed ingestion ─────────────────────────────────────────────────────────
# Maximum articles per spike window from RSS feeds
RSS_MAX_ARTICLES_PER_SPIKE: int = 50

# Time window (hours) around spike date for RSS article filtering
RSS_TIME_WINDOW_HOURS: int = 48

# RSS feed request timeout (seconds)
RSS_REQUEST_TIMEOUT: int = 15

# Levenshtein similarity threshold for RSS near-duplicate detection
RSS_DEDUP_THRESHOLD: float = 0.85

# ── Fusion parameters ──────────────────────────────────────────────────────────
# Default weights for fusion dimension scoring (must sum to 1.0)
FUSION_WEIGHT_TEMPORAL: float = 0.25
FUSION_WEIGHT_GEOGRAPHIC: float = 0.25
FUSION_WEIGHT_ACTOR: float = 0.20
FUSION_WEIGHT_SEMANTIC: float = 0.20
FUSION_WEIGHT_EVENT_TYPE: float = 0.10

# Maximum hours between events to consider temporal proximity
FUSION_TEMPORAL_WINDOW_HOURS: int = 72

# Maximum km between events to consider geographic proximity
FUSION_GEOGRAPHIC_THRESHOLD_KM: float = 200.0

# ── Validation thresholds ──────────────────────────────────────────────────────
# Minimum Levenshtein similarity for title-to-claim grounding
VALIDATION_TITLE_SIMILARITY_THRESHOLD: float = 0.55

# Minimum Levenshtein similarity for ground-truth alignment
VALIDATION_GROUND_TRUTH_SIMILARITY_THRESHOLD: float = 0.65

# Minimum match confidence for custom dataset confirmation
VALIDATION_CUSTOM_MATCH_THRESHOLD: float = 0.50

# Maximum days between article date and claimed event date
VALIDATION_DATE_DELTA_DAYS: int = 7

# HTTP HEAD check timeout for URL reachability (seconds)
VALIDATION_URL_TIMEOUT: int = 10

# Minimum corroboration count for cross-source verification
VALIDATION_MIN_CORROBORATION: int = 2

# ── Actor graph ────────────────────────────────────────────────────────────────
# Top-N actors by degree centrality classified as "Hub"
ACTOR_HUB_TOP_N: int = 5

# Betweenness-to-degree ratio threshold for "Broker" classification
ACTOR_BROKER_RATIO_THRESHOLD: float = 1.5

# Maximum PageRank iterations
ACTOR_PAGERANK_MAX_ITER: int = 200

# ── Visual intelligence ────────────────────────────────────────────────────────
# EXIF capture-date staleness warning threshold (hours before article publication)
VISUAL_STALENESS_HOURS: int = 72

# Maximum web appearance count below which image is considered novel
VISUAL_NOVELTY_WEB_COUNT_THRESHOLD: int = 10

# ── Output paths ──────────────────────────────────────────────────────────────
# Root directory for all pipeline run outputs
OUTPUT_ROOT: str = "outputs/runs"

# ── Logging ────────────────────────────────────────────────────────────────────
DEFAULT_LOG_LEVEL: str = "INFO"
