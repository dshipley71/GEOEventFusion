#!/bin/bash
# GEOEventFusion — Cloud environment setup
# Runs automatically at the start of every Claude Code web session via SessionStart hook.
# Also safe to run locally; skips the pip install step outside remote environments.

set -e

echo "=== GEOEventFusion environment setup ==="
echo "CLAUDE_CODE_REMOTE=${CLAUDE_CODE_REMOTE:-not set}"

# Only run full dependency install in Anthropic-managed remote (web) environments.
# Locally, assume the developer has a pre-configured virtualenv.
if [ "$CLAUDE_CODE_REMOTE" = "true" ]; then
    echo "Remote environment detected — installing Python dependencies..."

    # Upgrade pip quietly first
    pip install --quiet --upgrade pip

    # ── Production dependencies ───────────────────────────────────────────────
    pip install --quiet \
        "anthropic>=0.40.0" \
        "ollama>=0.6.1" \
        "requests>=2.31.0" \
        "networkx>=3.2" \
        "matplotlib>=3.8" \
        "folium>=0.15" \
        "scipy>=1.11" \
        "numpy>=1.26" \
        "pydantic>=2.5" \
        "pydantic-settings>=2.1" \
        "python-dotenv>=1.0" \
        "python-Levenshtein>=0.23" \
        "tqdm>=4.66" \
        "feedparser>=6.0" \
        "trafilatura>=1.7" \
        "pandas>=2.1"

    # ── Dev / test dependencies ───────────────────────────────────────────────
    pip install --quiet \
        "pytest>=7.4" \
        "pytest-cov>=4.1" \
        "coverage>=7.3" \
        "ruff>=0.1" \
        "mypy>=1.7"

    echo "All dependencies installed."

    # ── Verify key imports ────────────────────────────────────────────────────
    echo "Verifying critical imports..."
    python3 -c "
import anthropic, ollama, requests, networkx, matplotlib, scipy, numpy
import pydantic, dotenv, Levenshtein, tqdm, feedparser, trafilatura
import pytest, ruff
print('  ✓ All critical imports verified')
" || {
        echo "  ✗ Import verification failed — check pip output above for errors"
        exit 1
    }

else
    echo "Local environment — skipping remote dependency install."
    echo "Run: pip install -e '.[dev]' to install locally."
fi

# ── Project directory check ───────────────────────────────────────────────────
if [ -n "$CLAUDE_PROJECT_DIR" ]; then
    echo "Project directory: $CLAUDE_PROJECT_DIR"
    # Ensure outputs directory exists for pipeline runs
    mkdir -p "$CLAUDE_PROJECT_DIR/outputs/runs"
    mkdir -p "$CLAUDE_PROJECT_DIR/outputs/logs"
    echo "  ✓ Output directories ready"
fi

echo "=== Setup complete ==="
