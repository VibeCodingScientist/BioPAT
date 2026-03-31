"""Evaluation modules for BioPAT baselines and LLM benchmarking."""

from .bm25 import BM25Evaluator
from .llm_evaluator import LLMBenchmarkRunner, ModelSpec, ExperimentResult
from .metrics import (
    MetricsComputer,
    DOC_TYPE_PAPER,
    DOC_TYPE_PATENT,
    JURISDICTION_US,
    JURISDICTION_EP,
    JURISDICTION_WO,
    ALL_JURISDICTIONS,
)
from .dense import DenseRetriever, DenseRetrieverConfig, DenseEvaluator, MODEL_REGISTRY
from .hybrid import ResultFusion, FusionConfig, HybridRetriever, BM25DenseHybrid
from .reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    BM25CrossEncoderPipeline,
    DenseCrossEncoderPipeline,
    CROSS_ENCODER_REGISTRY,
)
from .ablation import (
    AblationConfig,
    AblationResult,
    QueryRepresentationAblation,
    DocumentRepresentationAblation,
    DomainAblation,
    TemporalAblation,
    IPCAblation,
    AblationRunner,
)
from .error_analysis import (
    ErrorAnalyzer,
    ErrorAnalysisConfig,
    FailureCategory,
    FailureCase,
    VocabularyAnalyzer,
    DomainAnalyzer,
    ErrorReportGenerator,
    run_error_analysis,
)
from .trimodal_retrieval import (
    MatchType,
    ModalityScore,
    TrimodalHit,
    TrimodalConfig,
    ScoreNormalizer,
    TrimodalRetriever,
    TrimodalEvaluator,
    TrimodalEvaluationResult,
    reciprocal_rank_fusion,
    create_trimodal_retriever,
)
from .agent_retrieval import (
    AgentConfig,
    AgentTrace,
    DualCorpusSearchTool,
    RetrievalAgent,
    results_to_qrels_format,
)
from .agent_metrics import compute_agent_metrics, compute_refinement_curve
from .dual_qrels import (
    build_dual_corpus,
    invert_qrels,
    select_type_b_queries,
    load_doc_types,
)

__all__ = [
    # BM25
    "BM25Evaluator",
    # LLM benchmarking
    "LLMBenchmarkRunner",
    "ModelSpec",
    "ExperimentResult",
    # Metrics
    "MetricsComputer",
    "DOC_TYPE_PAPER",
    "DOC_TYPE_PATENT",
    "JURISDICTION_US",
    "JURISDICTION_EP",
    "JURISDICTION_WO",
    "ALL_JURISDICTIONS",
    # Dense retrieval
    "DenseRetriever",
    "DenseRetrieverConfig",
    "DenseEvaluator",
    "MODEL_REGISTRY",
    # Hybrid
    "ResultFusion",
    "FusionConfig",
    "HybridRetriever",
    "BM25DenseHybrid",
    # Reranking
    "CrossEncoderReranker",
    "RerankerConfig",
    "BM25CrossEncoderPipeline",
    "DenseCrossEncoderPipeline",
    "CROSS_ENCODER_REGISTRY",
    # Ablation
    "AblationConfig",
    "AblationResult",
    "QueryRepresentationAblation",
    "DocumentRepresentationAblation",
    "DomainAblation",
    "TemporalAblation",
    "IPCAblation",
    "AblationRunner",
    # Error analysis
    "ErrorAnalyzer",
    "ErrorAnalysisConfig",
    "FailureCategory",
    "FailureCase",
    "VocabularyAnalyzer",
    "DomainAnalyzer",
    "ErrorReportGenerator",
    "run_error_analysis",
    # Trimodal retrieval (Phase 4.0)
    "MatchType",
    "ModalityScore",
    "TrimodalHit",
    "TrimodalConfig",
    "ScoreNormalizer",
    "TrimodalRetriever",
    "TrimodalEvaluator",
    "TrimodalEvaluationResult",
    "reciprocal_rank_fusion",
    "create_trimodal_retriever",
    # Agent retrieval (Experiment 7)
    "AgentConfig",
    "AgentTrace",
    "DualCorpusSearchTool",
    "RetrievalAgent",
    "results_to_qrels_format",
    "compute_agent_metrics",
    "compute_refinement_curve",
    "build_dual_corpus",
    "invert_qrels",
    "select_type_b_queries",
    "load_doc_types",
]
