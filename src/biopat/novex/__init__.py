"""BioPAT-NovEx: 3-tier prior art discovery benchmark.

Tier 1: Retrieval — Tier 2: Relevance — Tier 3: Novelty
100 statements, ~160K dual-corpus docs, multi-LLM annotation.
"""

from biopat.novex.benchmark import NovExBenchmark, NovExStatement
from biopat.novex.evaluator import NovExEvaluator, TierResult
from biopat.novex.annotation import AnnotationProtocol, RelevanceJudgment, ConsensusLabel
from biopat.novex.curate import StatementCurator
from biopat.novex.analysis import NovExAnalyzer

__all__ = [
    "NovExBenchmark", "NovExStatement", "NovExEvaluator", "TierResult",
    "AnnotationProtocol", "RelevanceJudgment", "ConsensusLabel",
    "StatementCurator", "NovExAnalyzer",
]
