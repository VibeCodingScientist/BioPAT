"""Tests for reproducibility utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from biopat.reproducibility import (
    REPRODUCIBILITY_SEED,
    AuditLogger,
    ChecksumEngine,
    create_manifest,
    get_reproducibility_seed,
)


class TestReproducibilitySeed:
    """Tests for reproducibility seed constant."""

    def test_seed_value(self):
        """Test that REPRODUCIBILITY_SEED is 42."""
        assert REPRODUCIBILITY_SEED == 42

    def test_get_reproducibility_seed(self):
        """Test get_reproducibility_seed function."""
        assert get_reproducibility_seed() == 42


class TestChecksumEngine:
    """Tests for ChecksumEngine class."""

    def test_compute_sha256(self):
        """Test SHA256 computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello, BioPAT!")

            engine = ChecksumEngine()
            sha256 = engine.compute_sha256(test_file)

            assert len(sha256) == 64  # SHA256 hex length
            assert sha256.isalnum()

    def test_compute_sha256_reproducible(self):
        """Test that SHA256 is reproducible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Deterministic content")

            engine = ChecksumEngine()
            hash1 = engine.compute_sha256(test_file)
            hash2 = engine.compute_sha256(test_file)

            assert hash1 == hash2

    def test_log_download(self):
        """Test download logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"
            test_file = tmpdir / "test.txt"
            test_file.write_text("Test content")

            engine = ChecksumEngine(manifest_path)
            record = engine.log_download(test_file, "https://example.com/test.txt")

            assert record["source_url"] == "https://example.com/test.txt"
            assert record["filename"] == "test.txt"
            assert "sha256" in record
            assert "downloaded_at" in record

    def test_manifest_persistence(self):
        """Test that manifest is persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"
            test_file = tmpdir / "test.txt"
            test_file.write_text("Test content")

            engine = ChecksumEngine(manifest_path)
            engine.log_download(test_file, "https://example.com/test.txt")

            assert manifest_path.exists()

            with open(manifest_path) as f:
                data = json.load(f)

            assert "downloads" in data
            assert "test.txt" in data["downloads"]

    def test_verify_checksum_valid(self):
        """Test checksum verification with valid hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"
            test_file = tmpdir / "test.txt"
            test_file.write_text("Test content")

            engine = ChecksumEngine(manifest_path)
            record = engine.log_download(test_file, "https://example.com/test.txt")
            expected_hash = record["sha256"]

            assert engine.verify_checksum(test_file, expected_hash)

    def test_verify_checksum_invalid(self):
        """Test checksum verification with invalid hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content")

            engine = ChecksumEngine()
            assert not engine.verify_checksum(test_file, "invalid_hash")


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_api_call(self):
        """Test API call logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            logger = AuditLogger(manifest_path)
            logger.log_api_call(
                service="test_service",
                endpoint="test_endpoint",
                method="GET",
                params={"key": "value"},
                response_status=200,
                response_count=10,
            )

            summary = logger.get_summary()
            assert summary["total_api_calls"] == 1
            assert "test_service" in summary["services_used"]

    def test_call_counts(self):
        """Test API call counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            logger = AuditLogger(manifest_path)
            logger.log_api_call("service1", "endpoint1")
            logger.log_api_call("service1", "endpoint1")
            logger.log_api_call("service1", "endpoint2")
            logger.log_api_call("service2", "endpoint1")

            counts = logger.get_call_counts()
            assert counts["service1:endpoint1"] == 2
            assert counts["service1:endpoint2"] == 1
            assert counts["service2:endpoint1"] == 1

    def test_sanitize_params(self):
        """Test that sensitive params are sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            logger = AuditLogger(manifest_path)
            logger.log_api_call(
                "service",
                "endpoint",
                params={"api_key": "secret123", "query": "test"},
            )

            # Read manifest to check sanitization
            with open(manifest_path) as f:
                data = json.load(f)

            call = data["api_calls"][0]
            assert call["params"]["api_key"] == "[REDACTED]"
            assert call["params"]["query"] == "test"

    def test_manifest_persistence(self):
        """Test that audit log is persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            logger = AuditLogger(manifest_path)
            logger.log_api_call("service", "endpoint")

            assert manifest_path.exists()

            with open(manifest_path) as f:
                data = json.load(f)

            assert "api_calls" in data
            assert len(data["api_calls"]) == 1


class TestCreateManifest:
    """Tests for create_manifest function."""

    def test_create_new_manifest(self):
        """Test creating a new manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = create_manifest(Path(tmpdir))

            assert manifest_path.exists()

            with open(manifest_path) as f:
                data = json.load(f)

            assert data["biopat_version"] == "0.1.0"
            assert data["reproducibility_seed"] == 42
            assert "created_at" in data
            assert "downloads" in data
            assert "api_calls" in data

    def test_idempotent_creation(self):
        """Test that create_manifest is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = create_manifest(Path(tmpdir))
            path2 = create_manifest(Path(tmpdir))

            assert path1 == path2
