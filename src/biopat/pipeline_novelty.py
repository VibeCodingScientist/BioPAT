"""End-to-End Novelty Assessment Pipeline.

The complete SOTA pipeline for patent novelty assessment:
1. Parse patent claims into structured elements
2. Execute trimodal search (text + chemical + sequence)
3. Rerank and filter prior art candidates
4. Map prior art to claim elements
5. Reason about novelty (anticipation vs obviousness)
6. Generate comprehensive assessment report

This is the main entry point for using BioPAT as an AI-powered
patent novelty assessment system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the novelty assessment pipeline."""

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None

    # Retrieval method
    # Options: "hybrid" (BM25+dense), "splade", "colbert", "dense", "sparse"
    retrieval_method: str = "hybrid"

    # Retrieval settings
    text_model: str = "BAAI/bge-base-en-v1.5"
    splade_model: str = "naver/splade-cocondenser-ensembledistil"
    colbert_model: str = "colbert-ir/colbertv2.0"
    chemical_method: str = "morgan"  # morgan, chemberta
    sequence_method: str = "esm2-small"  # esm2, protbert, blast

    # Modality weights
    text_weight: float = 0.5
    chemical_weight: float = 0.3
    sequence_weight: float = 0.2

    # Retrieval parameters
    initial_retrieval_k: int = 200
    rerank_k: int = 50
    final_k: int = 20

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    # Query expansion
    use_hyde: bool = False

    # Hardware
    use_gpu: bool = False

    # Output
    output_dir: Optional[str] = None
    save_intermediate: bool = False


@dataclass
class TrimodalSearchResult:
    """Results from trimodal search."""
    doc_id: str
    combined_score: float
    text_score: Optional[float] = None
    chemical_score: Optional[float] = None
    sequence_score: Optional[float] = None
    modalities_matched: List[str] = field(default_factory=list)


@dataclass
class PatentInput:
    """Input patent for novelty assessment."""
    patent_id: str
    title: str
    abstract: str
    claims: List[str]
    smiles: Optional[str] = None
    sequence: Optional[str] = None
    sequence_type: Optional[str] = None  # protein, nucleotide
    metadata: Dict[str, Any] = field(default_factory=dict)


class NoveltyAssessmentPipeline:
    """SOTA End-to-End Novelty Assessment Pipeline.

    Combines all components into a unified pipeline:
    - LLM Claim Parser
    - Trimodal Retrieval (text + chemical + sequence)
    - Cross-encoder Reranking
    - LLM Novelty Reasoner
    - Report Generator

    Example:
        ```python
        from biopat.pipeline_novelty import NoveltyAssessmentPipeline, PatentInput

        # Initialize pipeline
        pipeline = NoveltyAssessmentPipeline(
            llm_provider="openai",
            llm_model="gpt-4"
        )

        # Load corpus
        pipeline.load_corpus(corpus_dict)

        # Assess patent
        patent = PatentInput(
            patent_id="US10500001",
            title="Anti-PD-1 antibody",
            abstract="A humanized monoclonal antibody...",
            claims=["1. A method of treating cancer..."],
            sequence="QVQLVQSGAEVKKPGAS..."
        )

        report = pipeline.assess(patent)
        print(report.full_report)
        ```
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        **kwargs,
    ):
        """Initialize the novelty assessment pipeline.

        Args:
            config: Pipeline configuration
            **kwargs: Override config values
        """
        self.config = config or PipelineConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Components (initialized lazily)
        self._claim_parser = None
        self._text_retriever = None
        self._chemical_retriever = None
        self._sequence_retriever = None
        self._reranker = None
        self._novelty_reasoner = None
        self._explanation_generator = None
        self._hyde_expander = None

        # Corpus storage
        self.corpus: Dict[str, Any] = {}
        self.chemicals: List[Dict[str, str]] = []
        self.sequences: List[Dict[str, str]] = []

        self._initialized = False

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("Initializing pipeline components...")

        # 1. LLM Claim Parser
        try:
            from biopat.reasoning.claim_parser import LLMClaimParser
            self._claim_parser = LLMClaimParser(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
            )
            logger.info("  ✓ LLM Claim Parser")
        except Exception as e:
            logger.warning(f"  ✗ LLM Claim Parser: {e}")

        # 2. Text Retriever (based on config)
        try:
            method = self.config.retrieval_method

            if method == "splade":
                # SPLADE learned sparse retrieval
                from biopat.retrieval.splade import SPLADERetriever
                self._text_retriever = SPLADERetriever(
                    model_name=self.config.splade_model,
                )
                logger.info("  ✓ Text Retriever (SPLADE)")

            elif method == "colbert":
                # ColBERT late interaction
                from biopat.retrieval.colbert import ColBERTRetriever
                self._text_retriever = ColBERTRetriever(
                    model_name=self.config.colbert_model,
                )
                logger.info("  ✓ Text Retriever (ColBERT)")

            elif method == "dense":
                # Pure dense retrieval
                from biopat.retrieval.dense import DenseRetriever
                self._text_retriever = DenseRetriever(
                    model_name=self.config.text_model,
                    device="cuda" if self.config.use_gpu else "cpu",
                )
                logger.info("  ✓ Text Retriever (Dense)")

            elif method == "sparse":
                # Pure BM25
                from biopat.retrieval.hybrid import SparseRetriever
                self._text_retriever = SparseRetriever()
                logger.info("  ✓ Text Retriever (BM25)")

            else:
                # Default: Hybrid (BM25 + Dense)
                from biopat.retrieval.hybrid import HybridRetriever, SparseRetriever
                from biopat.retrieval.dense import DenseRetriever

                dense = DenseRetriever(
                    model_name=self.config.text_model,
                    device="cuda" if self.config.use_gpu else "cpu",
                )
                self._text_retriever = HybridRetriever(
                    dense_retriever=dense,
                    sparse_retriever=SparseRetriever(),
                )
                logger.info("  ✓ Text Retriever (Hybrid)")

        except Exception as e:
            logger.warning(f"  ✗ Text Retriever: {e}")
            # Fallback to sparse only
            try:
                from biopat.retrieval.hybrid import SparseRetriever
                self._text_retriever = SparseRetriever()
                logger.info("  ✓ Text Retriever (BM25 fallback)")
            except Exception:
                pass

        # 3. Chemical Retriever
        try:
            from biopat.retrieval.molecular import MolecularRetriever
            self._chemical_retriever = MolecularRetriever(
                method=self.config.chemical_method,
            )
            logger.info("  ✓ Chemical Retriever")
        except Exception as e:
            logger.warning(f"  ✗ Chemical Retriever: {e}")

        # 4. Sequence Retriever
        try:
            from biopat.retrieval.sequence import SequenceRetriever
            self._sequence_retriever = SequenceRetriever(
                method=self.config.sequence_method,
                sequence_type="protein",
            )
            logger.info("  ✓ Sequence Retriever")
        except Exception as e:
            logger.warning(f"  ✗ Sequence Retriever: {e}")

        # 5. Reranker
        if self.config.use_reranker:
            try:
                from biopat.retrieval.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker(
                    model_name=self.config.reranker_model,
                )
                logger.info("  ✓ Cross-Encoder Reranker")
            except Exception as e:
                logger.warning(f"  ✗ Reranker: {e}")

        # 6. HyDE Query Expander
        if self.config.use_hyde:
            try:
                from biopat.retrieval.hyde import HyDEQueryExpander
                self._hyde_expander = HyDEQueryExpander(
                    config=None,
                    api_key=self.config.llm_api_key,
                )
                logger.info("  ✓ HyDE Query Expander")
            except Exception as e:
                logger.warning(f"  ✗ HyDE: {e}")

        # 7. LLM Novelty Reasoner
        try:
            from biopat.reasoning.novelty_reasoner import LLMNoveltyReasoner
            self._novelty_reasoner = LLMNoveltyReasoner(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
            )
            logger.info("  ✓ LLM Novelty Reasoner")
        except Exception as e:
            logger.warning(f"  ✗ LLM Novelty Reasoner: {e}")

        # 8. Explanation Generator
        try:
            from biopat.reasoning.explanation_generator import ExplanationGenerator
            self._explanation_generator = ExplanationGenerator()
            logger.info("  ✓ Explanation Generator")
        except Exception as e:
            logger.warning(f"  ✗ Explanation Generator: {e}")

        self._initialized = True
        logger.info("Pipeline initialization complete")

    def load_corpus(
        self,
        corpus: Dict[str, Any],
        chemicals: Optional[List[Dict[str, str]]] = None,
        sequences: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Load corpus for retrieval.

        Args:
            corpus: Dictionary mapping doc_id to document (with title, text)
            chemicals: List of {"doc_id": ..., "smiles": ...} for chemical search
            sequences: List of {"doc_id": ..., "sequence": ...} for sequence search
        """
        self._init_components()

        self.corpus = corpus
        self.chemicals = chemicals or []
        self.sequences = sequences or []

        logger.info(f"Loading corpus: {len(corpus)} documents")

        # Index text
        if self._text_retriever is not None:
            self._text_retriever.index_corpus(corpus)

        # Index chemicals
        if self._chemical_retriever is not None and self.chemicals:
            self._chemical_retriever.index_chemicals(self.chemicals)
            logger.info(f"  Indexed {len(self.chemicals)} chemical structures")

        # Index sequences
        if self._sequence_retriever is not None and self.sequences:
            self._sequence_retriever.index_sequences(self.sequences)
            logger.info(f"  Indexed {len(self.sequences)} sequences")

    def _trimodal_search(
        self,
        query_text: str,
        query_smiles: Optional[str] = None,
        query_sequence: Optional[str] = None,
        top_k: int = 200,
    ) -> List[TrimodalSearchResult]:
        """Execute trimodal search across text, chemical, and sequence.

        Args:
            query_text: Text query (claim/abstract)
            query_smiles: Optional SMILES query
            query_sequence: Optional sequence query
            top_k: Number of results per modality

        Returns:
            Combined and ranked search results
        """
        results_by_doc: Dict[str, TrimodalSearchResult] = {}

        # 1. Text search
        if self._text_retriever is not None:
            text_results = self._text_retriever.search(query_text, top_k=top_k)

            for doc_id, score in text_results:
                if doc_id not in results_by_doc:
                    results_by_doc[doc_id] = TrimodalSearchResult(
                        doc_id=doc_id,
                        combined_score=0.0,
                    )
                results_by_doc[doc_id].text_score = score
                results_by_doc[doc_id].modalities_matched.append("text")

        # 2. Chemical search
        if self._chemical_retriever is not None and query_smiles:
            try:
                chem_results = self._chemical_retriever.search(query_smiles, top_k=top_k)

                for doc_id, score in chem_results:
                    if doc_id not in results_by_doc:
                        results_by_doc[doc_id] = TrimodalSearchResult(
                            doc_id=doc_id,
                            combined_score=0.0,
                        )
                    results_by_doc[doc_id].chemical_score = score
                    if "chemical" not in results_by_doc[doc_id].modalities_matched:
                        results_by_doc[doc_id].modalities_matched.append("chemical")
            except Exception as e:
                logger.warning(f"Chemical search failed: {e}")

        # 3. Sequence search
        if self._sequence_retriever is not None and query_sequence:
            try:
                seq_results = self._sequence_retriever.search(query_sequence, top_k=top_k)

                for doc_id, score in seq_results:
                    if doc_id not in results_by_doc:
                        results_by_doc[doc_id] = TrimodalSearchResult(
                            doc_id=doc_id,
                            combined_score=0.0,
                        )
                    results_by_doc[doc_id].sequence_score = score
                    if "sequence" not in results_by_doc[doc_id].modalities_matched:
                        results_by_doc[doc_id].modalities_matched.append("sequence")
            except Exception as e:
                logger.warning(f"Sequence search failed: {e}")

        # 4. Combine scores with weights
        for result in results_by_doc.values():
            score = 0.0
            total_weight = 0.0

            if result.text_score is not None:
                score += self.config.text_weight * result.text_score
                total_weight += self.config.text_weight

            if result.chemical_score is not None:
                score += self.config.chemical_weight * result.chemical_score
                total_weight += self.config.chemical_weight

            if result.sequence_score is not None:
                score += self.config.sequence_weight * result.sequence_score
                total_weight += self.config.sequence_weight

            # Normalize by active weights
            result.combined_score = score / total_weight if total_weight > 0 else 0.0

            # Boost for multi-modal matches
            if len(result.modalities_matched) > 1:
                result.combined_score *= (1 + 0.1 * (len(result.modalities_matched) - 1))

        # 5. Sort by combined score
        sorted_results = sorted(
            results_by_doc.values(),
            key=lambda x: x.combined_score,
            reverse=True,
        )

        return sorted_results[:top_k]

    def assess(
        self,
        patent: PatentInput,
        claims_to_assess: Optional[List[int]] = None,
    ) -> Any:
        """Perform complete novelty assessment on a patent.

        Args:
            patent: Patent to assess
            claims_to_assess: Optional list of claim numbers (1-indexed). If None, assess all.

        Returns:
            NoveltyReport with complete assessment
        """
        self._init_components()

        if not self.corpus:
            raise ValueError("No corpus loaded. Call load_corpus first.")

        logger.info(f"Assessing patent: {patent.patent_id}")

        # Select claims to assess
        claims = patent.claims
        if claims_to_assess:
            claims = [patent.claims[i-1] for i in claims_to_assess if 0 < i <= len(patent.claims)]

        # 1. Parse claims
        logger.info("Step 1: Parsing claims...")
        parsed_claims = []
        if self._claim_parser is not None:
            context = f"{patent.title}\n\n{patent.abstract}"
            for i, claim_text in enumerate(claims, 1):
                try:
                    parsed = self._claim_parser.parse_claim(
                        claim_text,
                        claim_number=i,
                        patent_context=context,
                    )
                    parsed_claims.append(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse claim {i}: {e}")
        else:
            logger.warning("Claim parser not available")

        # 2. Build search query
        logger.info("Step 2: Executing trimodal search...")
        query_text = f"{patent.title} {patent.abstract}"
        if claims:
            query_text += " " + " ".join(claims[:2])  # Add first 2 claims

        # Optional HyDE expansion
        if self._hyde_expander is not None and self.config.use_hyde:
            try:
                query_text = self._hyde_expander.expand_query(query_text)
            except Exception as e:
                logger.warning(f"HyDE expansion failed: {e}")

        # Execute trimodal search
        search_results = self._trimodal_search(
            query_text=query_text,
            query_smiles=patent.smiles,
            query_sequence=patent.sequence,
            top_k=self.config.initial_retrieval_k,
        )

        logger.info(f"  Found {len(search_results)} candidates")

        # 3. Rerank
        if self._reranker is not None and self.config.use_reranker:
            logger.info("Step 3: Reranking candidates...")
            candidates = [(r.doc_id, r.combined_score) for r in search_results[:self.config.rerank_k]]

            reranked = self._reranker.rerank(
                query=query_text,
                candidates=candidates,
                corpus=self.corpus,
                top_k=self.config.final_k,
            )

            # Update scores from reranker
            reranked_dict = dict(reranked)
            for result in search_results:
                if result.doc_id in reranked_dict:
                    result.combined_score = reranked_dict[result.doc_id]

            # Re-sort
            search_results = sorted(search_results, key=lambda x: x.combined_score, reverse=True)

        # 4. Novelty assessment for each claim
        logger.info("Step 4: Assessing novelty...")
        claim_assessments = []

        if self._novelty_reasoner is not None and parsed_claims:
            prior_art_docs = [(r.doc_id, r.combined_score) for r in search_results[:self.config.final_k]]

            for parsed in parsed_claims:
                try:
                    assessment = self._novelty_reasoner.assess_novelty(
                        parsed_claim=parsed,
                        prior_art_docs=prior_art_docs,
                        corpus=self.corpus,
                    )
                    claim_assessments.append(assessment)
                except Exception as e:
                    logger.warning(f"Failed to assess claim {parsed.claim_number}: {e}")
        else:
            logger.warning("Novelty reasoner not available or no parsed claims")

        # 5. Generate report
        logger.info("Step 5: Generating report...")
        if self._explanation_generator is not None and claim_assessments:
            report = self._explanation_generator.generate_report(
                patent_id=patent.patent_id,
                claim_assessments=claim_assessments,
                parsed_claims=parsed_claims,
            )
        else:
            # Fallback: basic report
            from biopat.reasoning.explanation_generator import NoveltyReport, NoveltyStatus
            report = NoveltyReport(
                patent_id=patent.patent_id,
                report_date="",
                overall_status=NoveltyStatus.POTENTIALLY_NOVEL,
                full_report=f"Assessment for {patent.patent_id}: {len(search_results)} prior art candidates found.",
            )

        # Save if configured
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"{patent.patent_id}_report.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report.to_json())
            logger.info(f"Report saved to {output_path}")

        logger.info("Assessment complete")
        return report

    def assess_batch(
        self,
        patents: List[PatentInput],
    ) -> List[Any]:
        """Assess multiple patents.

        Args:
            patents: List of patents to assess

        Returns:
            List of NoveltyReports
        """
        reports = []
        for patent in patents:
            try:
                report = self.assess(patent)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to assess {patent.patent_id}: {e}")
        return reports


def create_pipeline(
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    use_gpu: bool = False,
    **kwargs,
) -> NoveltyAssessmentPipeline:
    """Factory function for novelty assessment pipeline.

    Args:
        llm_provider: "openai" or "anthropic"
        llm_model: Model name (e.g., "gpt-4", "claude-3-opus")
        use_gpu: Use GPU for retrieval models
        **kwargs: Additional config overrides

    Returns:
        Configured NoveltyAssessmentPipeline
    """
    config = PipelineConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_gpu=use_gpu,
        **kwargs,
    )
    return NoveltyAssessmentPipeline(config=config)


# Convenience function for quick assessment
def assess_patent_novelty(
    patent_id: str,
    title: str,
    abstract: str,
    claims: List[str],
    corpus: Dict[str, Any],
    smiles: Optional[str] = None,
    sequence: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
) -> Any:
    """Quick novelty assessment for a single patent.

    Args:
        patent_id: Patent identifier
        title: Patent title
        abstract: Patent abstract
        claims: List of claim texts
        corpus: Prior art corpus
        smiles: Optional chemical structure (SMILES)
        sequence: Optional protein sequence
        llm_provider: LLM provider
        llm_model: LLM model name

    Returns:
        NoveltyReport
    """
    pipeline = create_pipeline(llm_provider=llm_provider, llm_model=llm_model)
    pipeline.load_corpus(corpus)

    patent = PatentInput(
        patent_id=patent_id,
        title=title,
        abstract=abstract,
        claims=claims,
        smiles=smiles,
        sequence=sequence,
    )

    return pipeline.assess(patent)
