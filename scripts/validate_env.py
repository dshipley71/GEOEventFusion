#!/usr/bin/env python3
"""GEOEventFusion — pre-flight environment validation.

Checks:
  1. Python version compatibility (3.10+)
  2. Required package imports
  3. Environment variable presence
  4. GDELT API connectivity ping
  5. Optional LLM backend connectivity

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --skip-network
    python scripts/validate_env.py --llm-backend anthropic
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── ANSI colours ────────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✓{_RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}✗{_RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}⚠{_RESET}  {msg}"


def _header(msg: str) -> str:
    return f"\n{_BOLD}{msg}{_RESET}"


# ── Check functions ──────────────────────────────────────────────────────────────

def check_python_version() -> Tuple[bool, str]:
    """Verify Python version is 3.10 or newer."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major < 3 or (major == 3 and minor < 10):
        return False, f"Python {version_str} detected — requires ≥ 3.10"
    return True, f"Python {version_str}"


def check_package_imports() -> List[Tuple[bool, str]]:
    """Verify all required packages can be imported."""
    required = [
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("feedparser", "feedparser"),
        ("trafilatura", "trafilatura"),
        ("networkx", "networkx"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("folium", "folium"),
        ("Levenshtein", "Levenshtein"),
        ("dateutil", "python-dateutil"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]

    results = []
    for import_name, package_name in required:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            results.append((True, f"{package_name} ({version})"))
        except ImportError:
            results.append((False, f"{package_name} — NOT installed (pip install {package_name})"))

    return results


def check_optional_package_imports() -> List[Tuple[bool, str]]:
    """Check optional packages (LLM backends, dev tools)."""
    optional = [
        ("anthropic", "anthropic"),
        ("ollama", "ollama"),
        ("pytest", "pytest"),
        ("responses", "responses"),
    ]
    results = []
    for import_name, package_name in optional:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            results.append((True, f"{package_name} ({version})"))
        except ImportError:
            results.append((None, f"{package_name} — not installed (optional)"))  # type: ignore[misc]
    return results


def check_geoeventfusion_imports() -> List[Tuple[bool, str]]:
    """Verify the geoeventfusion package modules can be imported."""
    modules = [
        "config.defaults",
        "config.settings",
        "geoeventfusion.models.events",
        "geoeventfusion.models.actors",
        "geoeventfusion.models.pipeline",
        "geoeventfusion.analysis.spike_detector",
        "geoeventfusion.analysis.actor_graph",
        "geoeventfusion.analysis.query_builder",
        "geoeventfusion.analysis.tone_analyzer",
        "geoeventfusion.analysis.visual_intel",
        "geoeventfusion.clients.gdelt_client",
        "geoeventfusion.clients.llm_client",
        "geoeventfusion.io.persistence",
        "geoeventfusion.utils.date_utils",
        "geoeventfusion.utils.text",
    ]
    results = []
    for module in modules:
        try:
            importlib.import_module(module)
            results.append((True, module))
        except ImportError as exc:
            results.append((False, f"{module} — {exc}"))
    return results


def check_env_vars() -> List[Tuple[bool, str]]:
    """Check presence of important environment variables."""
    # Try to load .env if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    results = []

    # LLM_BACKEND (optional, defaults to "ollama")
    backend = os.getenv("LLM_BACKEND", "ollama")
    results.append((True, f"LLM_BACKEND = {backend!r} (default: ollama)"))

    # ANTHROPIC_API_KEY (required only when backend=anthropic)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        results.append((True, f"ANTHROPIC_API_KEY = {masked}"))
    else:
        results.append((None, "ANTHROPIC_API_KEY — not set (required for Anthropic backend)"))  # type: ignore[misc]

    # OLLAMA_HOST (optional)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    results.append((True, f"OLLAMA_HOST = {ollama_host!r}"))

    # ACLED_API_KEY (optional)
    acled_key = os.getenv("ACLED_API_KEY")
    if acled_key:
        results.append((True, "ACLED_API_KEY = set"))
    else:
        results.append((None, "ACLED_API_KEY — not set (required for ground truth enrichment)"))  # type: ignore[misc]

    # OUTPUT_ROOT
    output_root = os.getenv("OUTPUT_ROOT", "outputs/runs")
    results.append((True, f"OUTPUT_ROOT = {output_root!r}"))

    return results


def check_gdelt_connectivity(timeout: int = 10) -> Tuple[bool, str]:
    """Ping the GDELT DOC 2.0 API to verify network reachability.

    Args:
        timeout: HTTP request timeout in seconds.

    Returns:
        (success, message) tuple.
    """
    try:
        import requests

        url = "https://api.gdeltproject.org/api/v2/doc/doc?query=test&mode=ArtList&maxrecords=1&format=json"
        resp = requests.head(url, timeout=timeout)
        if resp.status_code in (200, 301, 302, 405):
            return True, f"GDELT API reachable (HTTP {resp.status_code})"
        return (
            False,
            f"GDELT API returned unexpected HTTP {resp.status_code}",
        )
    except Exception as exc:
        return False, f"GDELT API unreachable: {exc}"


def check_ollama_connectivity(host: str = "http://localhost:11434", timeout: int = 5) -> Tuple[bool, str]:
    """Ping the Ollama server to verify it is running.

    Args:
        host: Ollama server URL.
        timeout: HTTP request timeout in seconds.

    Returns:
        (success, message) tuple.
    """
    try:
        import requests

        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            model_names = [m.get("name", "?") for m in data.get("models", [])]
            model_str = ", ".join(model_names[:5]) or "no models found"
            return True, f"Ollama running at {host} — models: {model_str}"
        return False, f"Ollama at {host} returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Ollama not reachable at {host}: {exc}"


def check_output_dir() -> Tuple[bool, str]:
    """Verify the output root directory is writable."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    output_root = os.getenv("OUTPUT_ROOT", "outputs/runs")
    output_path = _ROOT / output_root

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        test_file = output_path / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"Output directory writable: {output_path}"
    except OSError as exc:
        return False, f"Output directory not writable ({output_path}): {exc}"


# ── Report ───────────────────────────────────────────────────────────────────────

def _print_results(results: List[Tuple], indent: int = 2) -> int:
    """Print check results and return count of failures."""
    failures = 0
    pad = " " * indent
    for item in results:
        ok, msg = item[0], item[1]
        if ok is True:
            print(f"{pad}{_ok(msg)}")
        elif ok is False:
            print(f"{pad}{_fail(msg)}")
            failures += 1
        else:
            # None = optional / warning
            print(f"{pad}{_warn(msg)}")
    return failures


def main() -> None:
    """Run all pre-flight checks and report results."""
    parser = argparse.ArgumentParser(
        description="GEOEventFusion — pre-flight environment validation",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        default=False,
        help="Skip network connectivity checks (GDELT ping, Ollama ping)",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default=None,
        choices=["anthropic", "ollama"],
        help="LLM backend to test connectivity for",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL to ping",
    )
    args = parser.parse_args()

    total_failures = 0

    print(f"\n{_BOLD}╔══════════════════════════════════════════════════════╗{_RESET}")
    print(f"{_BOLD}║  GEOEventFusion — Environment Validation              ║{_RESET}")
    print(f"{_BOLD}╚══════════════════════════════════════════════════════╝{_RESET}")

    # 1. Python version
    print(_header("1. Python Version"))
    ok, msg = check_python_version()
    print(f"  {_ok(msg) if ok else _fail(msg)}")
    if not ok:
        total_failures += 1

    # 2. Required packages
    print(_header("2. Required Package Imports"))
    results = check_package_imports()
    total_failures += _print_results(results)

    # 3. Optional packages
    print(_header("3. Optional Packages"))
    opt_results = check_optional_package_imports()
    _print_results(opt_results)  # Optionals don't count as failures

    # 4. GEOEventFusion module imports
    print(_header("4. GEOEventFusion Module Imports"))
    module_results = check_geoeventfusion_imports()
    total_failures += _print_results(module_results)

    # 5. Environment variables
    print(_header("5. Environment Variables"))
    env_results = check_env_vars()
    _print_results(env_results)  # Missing env vars are warnings, not hard failures

    # 6. Output directory
    print(_header("6. Output Directory"))
    ok, msg = check_output_dir()
    print(f"  {_ok(msg) if ok else _fail(msg)}")
    if not ok:
        total_failures += 1

    # 7. Network checks (skippable)
    if not args.skip_network:
        print(_header("7. Network Connectivity"))

        ok, msg = check_gdelt_connectivity()
        print(f"  {_ok(msg) if ok else _warn(msg)}")
        # GDELT connectivity is a warning, not a hard failure

        backend = args.llm_backend or os.getenv("LLM_BACKEND", "ollama")
        if backend == "ollama":
            ok, msg = check_ollama_connectivity(host=args.ollama_host)
            print(f"  {_ok(msg) if ok else _warn(msg)}")
    else:
        print(_header("7. Network Connectivity"))
        print(f"  {_warn('Skipped (--skip-network)')}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 54}")
    if total_failures == 0:
        print(f"{_GREEN}{_BOLD}All required checks passed.{_RESET} Environment is ready.")
        sys.exit(0)
    else:
        print(
            f"{_RED}{_BOLD}{total_failures} check(s) failed.{_RESET} "
            "Resolve the errors above before running the pipeline."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
