"""BioPAT: Biomedical Patent-to-Article Retrieval Benchmark."""

__version__ = "0.1.0"

# Export reproducibility utilities at package level
from biopat.reproducibility import (
    REPRODUCIBILITY_SEED,
    ChecksumEngine,
    AuditLogger,
    create_manifest,
    get_reproducibility_seed,
)
