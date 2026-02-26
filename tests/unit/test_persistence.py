"""Unit tests for geoeventfusion.io.persistence.

Covers:
- save_json: happy path, creates parent dirs, atomic write, dataclass encoding
- load_json: happy path, missing file, invalid JSON
- file_checksum: existing file, missing file
- ensure_output_dir: creates run dir and charts/ subdirectory
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from geoeventfusion.io.persistence import (
    ensure_output_dir,
    file_checksum,
    load_json,
    save_json,
)


# ── save_json ─────────────────────────────────────────────────────────────────────

class TestSaveJson:
    def test_save_json_happy_path(self, tmp_path):
        """save_json must write valid JSON to the given path."""
        target = tmp_path / "output.json"
        data = {"query": "Houthi", "days_back": 90}

        save_json(data, target)

        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_save_json_creates_parent_directories(self, tmp_path):
        """save_json must create intermediate parent directories if they do not exist."""
        target = tmp_path / "a" / "b" / "c" / "output.json"
        save_json({"key": "value"}, target)

        assert target.exists()

    def test_save_json_unicode_preserved(self, tmp_path):
        """Non-ASCII characters must be preserved (ensure_ascii=False)."""
        target = tmp_path / "unicode.json"
        data = {"text": "Привет мир — مرحبا — 你好"}

        save_json(data, target)

        content = target.read_text(encoding="utf-8")
        assert "Привет мир" in content

    def test_save_json_uses_indent(self, tmp_path):
        """Output must be indented at the default indent level."""
        target = tmp_path / "indented.json"
        save_json({"key": "value"}, target, indent=2)

        content = target.read_text()
        # Indented JSON contains newlines
        assert "\n" in content

    def test_save_json_overwrites_existing_file(self, tmp_path):
        """Saving to an existing path must overwrite the old content."""
        target = tmp_path / "output.json"
        save_json({"version": 1}, target)
        save_json({"version": 2}, target)

        loaded = json.loads(target.read_text())
        assert loaded["version"] == 2

    def test_save_json_list_data(self, tmp_path):
        """Lists must be saved and loaded correctly."""
        target = tmp_path / "list.json"
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        save_json(data, target)

        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_save_json_dataclass_serializable(self, tmp_path):
        """Dataclass instances must be serialized via _DataclassEncoder."""

        @dataclasses.dataclass
        class TestResult:
            name: str
            count: int

        target = tmp_path / "dataclass.json"
        save_json(TestResult(name="test", count=42), target)

        loaded = json.loads(target.read_text())
        assert loaded == {"name": "test", "count": 42}

    def test_save_json_path_object_serialized(self, tmp_path):
        """Path objects embedded in data must be serialized as strings."""
        target = tmp_path / "paths.json"
        data = {"output_dir": tmp_path / "runs"}

        save_json(data, target)

        loaded = json.loads(target.read_text())
        assert isinstance(loaded["output_dir"], str)

    def test_save_json_atomic_no_partial_write(self, tmp_path, monkeypatch):
        """On rename failure, the temp file must be cleaned up (no orphaned temps)."""
        target = tmp_path / "output.json"

        # Simulate os.replace failure
        original_replace = os.replace

        def failing_replace(src, dst):
            raise OSError("Disk full")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError):
            save_json({"key": "value"}, target)

        # Target should not exist (write failed)
        assert not target.exists()
        # No .tmp files should be left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_json_accepts_string_path(self, tmp_path):
        """save_json must accept a string path as well as a Path object."""
        target = str(tmp_path / "string_path.json")
        save_json({"key": "ok"}, target)
        assert Path(target).exists()


# ── load_json ─────────────────────────────────────────────────────────────────────

class TestLoadJson:
    def test_load_json_happy_path(self, tmp_path):
        """load_json must return the parsed content of a valid JSON file."""
        target = tmp_path / "data.json"
        target.write_text('{"result": "ok", "count": 7}', encoding="utf-8")

        loaded = load_json(target)

        assert loaded == {"result": "ok", "count": 7}

    def test_load_json_list_file(self, tmp_path):
        """load_json must handle JSON files containing arrays."""
        target = tmp_path / "list.json"
        target.write_text('[1, 2, 3]', encoding="utf-8")

        loaded = load_json(target)
        assert loaded == [1, 2, 3]

    def test_load_json_missing_file_returns_none(self, tmp_path):
        """Non-existent file path must return None without raising."""
        result = load_json(tmp_path / "does_not_exist.json")
        assert result is None

    def test_load_json_invalid_json_returns_none(self, tmp_path):
        """A file containing invalid JSON must return None without raising."""
        target = tmp_path / "bad.json"
        target.write_text("{this is not valid json}", encoding="utf-8")

        result = load_json(target)
        assert result is None

    def test_load_json_accepts_string_path(self, tmp_path):
        """load_json must accept a string path."""
        target = str(tmp_path / "data.json")
        Path(target).write_text('{"key": "value"}', encoding="utf-8")

        loaded = load_json(target)
        assert loaded == {"key": "value"}

    def test_load_json_roundtrip_with_save(self, tmp_path):
        """Data saved with save_json must be loadable with load_json unchanged."""
        target = tmp_path / "roundtrip.json"
        original = {"query": "Houthi", "spikes": [{"date": "2024-01-31", "z_score": 4.04}]}

        save_json(original, target)
        loaded = load_json(target)

        assert loaded == original

    def test_load_json_empty_file_returns_none(self, tmp_path):
        """Empty file must return None (JSONDecodeError)."""
        target = tmp_path / "empty.json"
        target.write_text("", encoding="utf-8")

        result = load_json(target)
        assert result is None


# ── file_checksum ─────────────────────────────────────────────────────────────────

class TestFileChecksum:
    def test_checksum_returns_hex_digest(self, tmp_path):
        """Checksum of an existing file must return a 64-char hex string (SHA-256)."""
        target = tmp_path / "file.txt"
        target.write_text("GEOEventFusion test content")

        digest = file_checksum(target)

        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_checksum_is_deterministic(self, tmp_path):
        """Same file content must always produce the same checksum."""
        target = tmp_path / "file.txt"
        target.write_text("deterministic content")

        assert file_checksum(target) == file_checksum(target)

    def test_different_content_different_checksum(self, tmp_path):
        """Different file contents must produce different checksums."""
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("content A")
        file_b.write_text("content B")

        assert file_checksum(file_a) != file_checksum(file_b)

    def test_missing_file_returns_empty_string(self, tmp_path):
        """Non-existent file must return an empty string without raising."""
        result = file_checksum(tmp_path / "nonexistent.txt")
        assert result == ""

    def test_checksum_accepts_string_path(self, tmp_path):
        """file_checksum must accept a string path."""
        target = tmp_path / "file.txt"
        target.write_text("hello")

        result = file_checksum(str(target))
        assert len(result) == 64


# ── ensure_output_dir ─────────────────────────────────────────────────────────────

class TestEnsureOutputDir:
    def test_creates_run_directory(self, tmp_path):
        """ensure_output_dir must create the run_id directory."""
        run_id = "20240115_120000_houthi_red_sea"
        run_dir = ensure_output_dir(tmp_path, run_id)

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.name == run_id

    def test_creates_charts_subdirectory(self, tmp_path):
        """ensure_output_dir must create a charts/ subdirectory inside the run dir."""
        run_dir = ensure_output_dir(tmp_path, "test_run")
        charts_dir = run_dir / "charts"

        assert charts_dir.exists()
        assert charts_dir.is_dir()

    def test_returns_run_dir_path(self, tmp_path):
        """ensure_output_dir must return the Path to the created run directory."""
        run_dir = ensure_output_dir(tmp_path, "test_run_001")

        assert isinstance(run_dir, Path)
        assert run_dir.name == "test_run_001"

    def test_idempotent_on_existing_directory(self, tmp_path):
        """Calling ensure_output_dir twice with the same run_id must not raise."""
        ensure_output_dir(tmp_path, "idempotent_run")
        run_dir = ensure_output_dir(tmp_path, "idempotent_run")  # second call

        assert run_dir.exists()

    def test_creates_nested_base_dir(self, tmp_path):
        """base_dir that doesn't exist yet must be created."""
        base = tmp_path / "deep" / "nested" / "outputs"
        run_dir = ensure_output_dir(base, "test_run")

        assert run_dir.exists()
        assert (run_dir / "charts").exists()

    def test_accepts_string_base_dir(self, tmp_path):
        """ensure_output_dir must accept a string base_dir."""
        run_dir = ensure_output_dir(str(tmp_path), "string_base_run")

        assert run_dir.exists()
