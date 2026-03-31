"""Reproducibility utilities for BioPAT.

Provides checksum computation, audit logging, and determinism enforcement
to meet Open Science & Reproducibility standards.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fixed seed for all random operations - ensures deterministic benchmark generation
REPRODUCIBILITY_SEED = 42


class ChecksumEngine:
    """Computes and records SHA256 hashes for downloaded files."""

    def __init__(self, manifest_path: Optional[Path] = None):
        """Initialize checksum engine.

        Args:
            manifest_path: Path to manifest file. If None, checksums are computed
                          but not persisted.
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._checksums: Dict[str, Dict[str, Any]] = {}

        if self.manifest_path and self.manifest_path.exists():
            self._load_manifest()

    def _load_manifest(self) -> None:
        """Load existing manifest from disk."""
        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
                self._checksums = data.get("downloads", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load manifest: {e}")
            self._checksums = {}

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        if not self.manifest_path:
            return

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing manifest to preserve other data
        existing = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        existing["downloads"] = self._checksums

        with open(self.manifest_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def compute_sha256(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file.
            chunk_size: Size of chunks to read.

        Returns:
            Hexadecimal SHA256 hash string.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def log_download(
        self,
        file_path: Path,
        source_url: str,
        compute_hash: bool = True,
    ) -> Dict[str, Any]:
        """Log a downloaded file with metadata and checksum.

        Args:
            file_path: Path to the downloaded file.
            source_url: URL the file was downloaded from.
            compute_hash: If True, compute SHA256 hash.

        Returns:
            Dict with download metadata including checksum.
        """
        file_path = Path(file_path)

        record = {
            "source_url": source_url,
            "local_path": str(file_path.absolute()),
            "filename": file_path.name,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else None,
        }

        if compute_hash and file_path.exists():
            logger.info(f"Computing SHA256 for {file_path.name}...")
            record["sha256"] = self.compute_sha256(file_path)
            logger.info(f"SHA256: {record['sha256']}")

        # Store by filename for easy lookup
        self._checksums[file_path.name] = record
        self._save_manifest()

        return record

    def verify_checksum(self, file_path: Path, expected_hash: Optional[str] = None) -> bool:
        """Verify a file's checksum against expected or recorded hash.

        Args:
            file_path: Path to the file to verify.
            expected_hash: Expected SHA256 hash. If None, uses recorded hash.

        Returns:
            True if checksum matches, False otherwise.
        """
        file_path = Path(file_path)

        if expected_hash is None:
            record = self._checksums.get(file_path.name)
            if not record or "sha256" not in record:
                logger.warning(f"No recorded checksum for {file_path.name}")
                return False
            expected_hash = record["sha256"]

        actual_hash = self.compute_sha256(file_path)
        matches = actual_hash == expected_hash

        if matches:
            logger.info(f"Checksum verified for {file_path.name}")
        else:
            logger.error(
                f"Checksum mismatch for {file_path.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        return matches

    def get_all_checksums(self) -> Dict[str, str]:
        """Get all recorded checksums.

        Returns:
            Dict mapping filenames to SHA256 hashes.
        """
        return {
            name: record.get("sha256", "")
            for name, record in self._checksums.items()
            if "sha256" in record
        }


class AuditLogger:
    """Logs API calls and operations for reproducibility tracking."""

    def __init__(self, manifest_path: Optional[Path] = None):
        """Initialize audit logger.

        Args:
            manifest_path: Path to manifest file for persistence.
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._api_calls: List[Dict[str, Any]] = []
        self._call_counts: Dict[str, int] = {}

        if self.manifest_path and self.manifest_path.exists():
            self._load_manifest()

    def _load_manifest(self) -> None:
        """Load existing manifest from disk."""
        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
                self._api_calls = data.get("api_calls", [])
                self._call_counts = data.get("api_call_counts", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load audit manifest: {e}")

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        if not self.manifest_path:
            return

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing manifest to preserve other data
        existing = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        existing["api_calls"] = self._api_calls
        existing["api_call_counts"] = self._call_counts

        with open(self.manifest_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        response_count: Optional[int] = None,
    ) -> None:
        """Log an API call.

        Args:
            service: Name of the service (e.g., "patentsview", "openalex").
            endpoint: API endpoint called.
            method: HTTP method used.
            params: Request parameters (sanitized, no secrets).
            response_status: HTTP response status code.
            response_count: Number of items returned.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "params": self._sanitize_params(params) if params else None,
            "response_status": response_status,
            "response_count": response_count,
        }

        self._api_calls.append(record)

        # Update call counts
        key = f"{service}:{endpoint}"
        self._call_counts[key] = self._call_counts.get(key, 0) + 1

        self._save_manifest()

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from params.

        Args:
            params: Original parameters dict.

        Returns:
            Sanitized parameters dict.
        """
        sensitive_keys = {"api_key", "apikey", "key", "token", "secret", "password"}
        return {
            k: "[REDACTED]" if k.lower() in sensitive_keys else v
            for k, v in params.items()
        }

    def get_call_counts(self) -> Dict[str, int]:
        """Get API call counts by service:endpoint.

        Returns:
            Dict mapping service:endpoint to call count.
        """
        return dict(self._call_counts)

    def get_summary(self) -> Dict[str, Any]:
        """Get audit summary.

        Returns:
            Dict with audit summary statistics.
        """
        return {
            "total_api_calls": len(self._api_calls),
            "call_counts_by_endpoint": self._call_counts,
            "services_used": list(set(c["service"] for c in self._api_calls)),
            "first_call": self._api_calls[0]["timestamp"] if self._api_calls else None,
            "last_call": self._api_calls[-1]["timestamp"] if self._api_calls else None,
        }


def get_reproducibility_seed() -> int:
    """Get the fixed reproducibility seed.

    Returns:
        The fixed seed value (42).
    """
    return REPRODUCIBILITY_SEED


def create_manifest(output_dir: Path) -> Path:
    """Create or get path to manifest.json for a build.

    Args:
        output_dir: Output directory for the build.

    Returns:
        Path to manifest.json file.
    """
    manifest_path = Path(output_dir) / "manifest.json"

    if not manifest_path.exists():
        initial_manifest = {
            "biopat_version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reproducibility_seed": REPRODUCIBILITY_SEED,
            "downloads": {},
            "api_calls": [],
            "api_call_counts": {},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(initial_manifest, f, indent=2)

    return manifest_path
