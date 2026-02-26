# GEOEventFusion — Deployment Guide

## 1. Google Colab (Recommended for Quick Start)

Open `notebooks/quickstart.ipynb` in Google Colab.

**Steps:**
1. Upload the repository to Google Drive or clone it in Colab
2. Set API keys in Colab Secrets (key icon in the left sidebar):
   - `ANTHROPIC_API_KEY`
   - `ACLED_API_KEY` (optional, for ground truth)
   - `ACLED_EMAIL` (optional)
3. Run all cells in `notebooks/quickstart.ipynb`

**Colab-specific notes:**
- Use `from geoeventfusion.io.colab_helpers import download_run_artifacts` to download outputs
- Colab sessions reset — install dependencies at the start of each session
- Outputs are saved to Google Drive if mounted

---

## 2. Local Python Environment

**Requirements:**
- Python 3.10+
- pip

**Setup:**
```bash
# Clone the repository
git clone https://github.com/your-org/GEOEventFusion.git
cd GEOEventFusion

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install all dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Validate environment
python scripts/validate_env.py
```

**Run a pipeline:**
```bash
# Basic run with Ollama backend
python scripts/run_pipeline.py --query "Houthi Red Sea attacks" --days-back 90

# With Anthropic backend
python scripts/run_pipeline.py --query "Taiwan Strait tensions" --llm-backend anthropic

# With ground truth enrichment
python scripts/run_pipeline.py \
    --query "Sudan conflict" \
    --ground-truth-sources acled \
    --ground-truth-country-filter Sudan \
    --llm-backend anthropic

# Test mode (fixture data, no API calls)
python scripts/run_pipeline.py --query "test" --test-mode

# Batch run from YAML config
python scripts/batch_run.py --config configs/batch_queries.yaml
```

**Output location:**
```
outputs/runs/<YYYYMMDD_HHMMSS>_<query_slug>/
├── run_metadata.json
├── storyboard.json
├── timeline.json
├── hypotheses.json
├── validation_report.json
├── storyboard_report.html
├── actor_network.gexf
└── charts/
    ├── event_timeline_annotated.png
    ├── tone_distribution.png
    ├── timeline_language.png
    ├── actor_network.png
    └── source_country_map.html
```

---

## 3. Ollama Setup (Local LLM)

Install Ollama from https://ollama.ai and pull the recommended model:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull gemma3:27b

# Start the Ollama server (runs in background)
ollama serve
```

Then set in `.env`:
```bash
LLM_BACKEND=ollama
OLLAMA_MODEL=gemma3:27b
OLLAMA_HOST=http://localhost:11434
```

**Note:** `gemma3:27b` requires ~18 GB VRAM. For smaller GPUs, use `gemma3:9b` or `llama3.2`.

---

## 4. Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=geoeventfusion --cov-report=term-missing

# Single test file
pytest tests/unit/test_spike_detector.py -v
```

---

## 5. Linting and Type Checking

```bash
# Lint and auto-fix
ruff check geoeventfusion/ --fix
ruff format geoeventfusion/

# Type checking
mypy geoeventfusion/
```

---

## 6. Docker (Future)

A `Dockerfile` is planned for containerized deployment. It will:
- Bundle all Python dependencies
- Copy the `geoeventfusion` package
- Expose a CLI entrypoint at `/usr/local/bin/run_pipeline`

---

## 7. API Server (Future)

A FastAPI wrapper (`geoeventfusion/server.py`) is planned for async pipeline execution.
It will expose:
- `POST /pipeline/run` — start a pipeline run
- `GET /pipeline/status/{run_id}` — check run status
- `GET /pipeline/results/{run_id}` — retrieve artifacts

---

## 8. Pre-flight Validation

Run this before executing the pipeline to verify all dependencies and API keys:

```bash
python scripts/validate_env.py
```

This checks:
- Python version (3.10+ required)
- All required packages installed
- `ANTHROPIC_API_KEY` set and non-empty (if using Anthropic backend)
- Output root directory is writable
- Ollama server reachable (if using Ollama backend)
