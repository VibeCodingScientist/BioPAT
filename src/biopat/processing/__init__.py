"""Data processing modules for BioPAT."""

from .patents import PatentProcessor
from .papers import PaperProcessor
from .linking import CitationLinker
from .npl_parser import NPLParser, NPLLinker
from .claim_mapper import ClaimMapper, ClaimCitationMapper
from .prior_patents import PriorPatentSelector
from .patent_ids import (
    Jurisdiction,
    ParsedPatentId,
    PatentIdNormalizer,
    normalize_patent_id,
    normalize_patent_ids,
    extract_jurisdiction,
    are_same_patent,
    deduplicate_patent_ids,
    group_by_jurisdiction,
    validate_patent_id,
)
from .sequence_index import (
    BlastHit,
    BlastResult,
    SequenceRecord,
    BlastDatabaseManager,
    BlastSearcher,
    SequenceIndex,
    compute_sequence_hash,
    identity_to_relevance_tier,
)
from .chemical_index import (
    ChemicalRecord,
    ChemicalSearchHit,
    MorganFingerprintCalculator,
    FaissChemicalIndex,
    ChemicalIndex,
    compute_tanimoto,
    tanimoto_to_relevance_tier,
    compute_chemical_id,
    RDKIT_AVAILABLE,
    FAISS_AVAILABLE,
)

__all__ = [
    "PatentProcessor",
    "PaperProcessor",
    "CitationLinker",
    "NPLParser",
    "NPLLinker",
    "ClaimMapper",
    "ClaimCitationMapper",
    "PriorPatentSelector",
    # Patent ID normalization
    "Jurisdiction",
    "ParsedPatentId",
    "PatentIdNormalizer",
    "normalize_patent_id",
    "normalize_patent_ids",
    "extract_jurisdiction",
    "are_same_patent",
    "deduplicate_patent_ids",
    "group_by_jurisdiction",
    "validate_patent_id",
    # Sequence indexing (Phase 4.0)
    "BlastHit",
    "BlastResult",
    "SequenceRecord",
    "BlastDatabaseManager",
    "BlastSearcher",
    "SequenceIndex",
    "compute_sequence_hash",
    "identity_to_relevance_tier",
    # Chemical indexing (Phase 4.0)
    "ChemicalRecord",
    "ChemicalSearchHit",
    "MorganFingerprintCalculator",
    "FaissChemicalIndex",
    "ChemicalIndex",
    "compute_tanimoto",
    "tanimoto_to_relevance_tier",
    "compute_chemical_id",
    "RDKIT_AVAILABLE",
    "FAISS_AVAILABLE",
]
