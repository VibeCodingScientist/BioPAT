"""Data ingestion modules for BioPAT."""

from .ros import RelianceOnScienceLoader
from .patentsview import PatentsViewClient
from .openalex import OpenAlexClient
from .retry import retry_with_backoff, retry_sync

# Re-export reproducibility utilities for convenience
from biopat.reproducibility import ChecksumEngine, AuditLogger

__all__ = [
    "RelianceOnScienceLoader",
    "PatentsViewClient",
    "OpenAlexClient",
    "ChecksumEngine",
    "AuditLogger",
    "retry_with_backoff",
    "retry_sync",
]
