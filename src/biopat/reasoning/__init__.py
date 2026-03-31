"""LLM-based Reasoning Module for Patent Novelty Assessment.

This module provides SOTA LLM-powered components for:
- Claim parsing and decomposition
- Prior art relevance assessment
- Novelty reasoning and determination
- Explanation generation

These components form the "brain" of the novelty assessment pipeline,
using large language models to perform complex legal/technical reasoning.
"""

from biopat.reasoning.claim_parser import (
    LLMClaimParser,
    ClaimElement,
    ClaimType,
    ParsedClaim,
)
from biopat.reasoning.novelty_reasoner import (
    LLMNoveltyReasoner,
    NoveltyAssessment,
    NoveltyStatus,
    PriorArtMapping,
)
from biopat.reasoning.explanation_generator import (
    ExplanationGenerator,
    NoveltyReport,
)

__all__ = [
    # Claim parsing
    "LLMClaimParser",
    "ClaimElement",
    "ClaimType",
    "ParsedClaim",
    # Novelty reasoning
    "LLMNoveltyReasoner",
    "NoveltyAssessment",
    "NoveltyStatus",
    "PriorArtMapping",
    # Explanation
    "ExplanationGenerator",
    "NoveltyReport",
]
