"""Cross-Encoder Reranking for SOTA retrieval pipelines.

Implements:
- Cross-encoder reranking with domain-specific models
- MonoT5/RankT5 sequence-to-sequence reranking
- LLM-based reranking (GPT-4, Claude)
- Listwise reranking
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Lazy imports
_transformers = None
_torch = None


def _import_transformers():
    global _transformers
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            raise ImportError(
                "transformers required for reranking. "
                "Install with: pip install transformers"
            )
    return _transformers


def _import_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("torch required for reranking")
    return _torch


# Available reranking models
RERANKER_MODELS = {
    # Cross-encoders (best quality)
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "ms-marco-large": "cross-encoder/ms-marco-electra-base",

    # Domain-specific
    "biomedical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",

    # MonoT5 (sequence-to-sequence)
    "monot5": "castorini/monot5-base-msmarco",
    "monot5-large": "castorini/monot5-large-msmarco",

    # SOTA rerankers
    "bge-reranker": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "jina-reranker": "jinaai/jina-reranker-v1-base-en",
}


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""

    model_name: str = "BAAI/bge-reranker-base"
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = False

    # For MonoT5
    use_monot5: bool = False

    # For LLM reranking
    use_llm: bool = False
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None


class CrossEncoderReranker:
    """SOTA Cross-Encoder Reranker.

    Cross-encoders process query-document pairs jointly, achieving
    much higher quality than bi-encoders at the cost of speed.
    Use after initial retrieval to rerank top candidates.

    Example:
        ```python
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

        # Get initial candidates
        candidates = retriever.search(query, top_k=100)

        # Rerank top 100 to get best 10
        reranked = reranker.rerank(query, candidates, corpus, top_k=10)
        ```
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        config: Optional[RerankerConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or RerankerConfig(model_name=model_name)
        self.model_name = model_name

        # Import dependencies
        self.transformers = _import_transformers()
        self.torch = _import_torch()

        # Determine device
        if device:
            self.device = device
        elif self.config.use_gpu and self.torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load model
        logger.info(f"Loading reranker model: {model_name}")

        if "monot5" in model_name.lower() or self.config.use_monot5:
            self._load_monot5(model_name)
        else:
            self._load_cross_encoder(model_name)

    def _load_cross_encoder(self, model_name: str) -> None:
        """Load cross-encoder model."""
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = self.transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "cross-encoder"

    def _load_monot5(self, model_name: str) -> None:
        """Load MonoT5 model."""
        self.tokenizer = self.transformers.T5Tokenizer.from_pretrained(model_name)
        self.model = self.transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "monot5"

    def _get_doc_text(self, doc: Union[str, Dict]) -> str:
        """Extract text from document."""
        if isinstance(doc, str):
            return doc
        return f"{doc.get('title', '')} {doc.get('text', '')}".strip()

    def _score_cross_encoder(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score documents using cross-encoder."""
        scores = []

        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Score
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                scores.extend(batch_scores)

        return scores

    def _score_monot5(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score documents using MonoT5."""
        scores = []

        # MonoT5 prompt format
        for doc in documents:
            input_text = f"Query: {query} Document: {doc} Relevant:"

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                # Get probability of "true" token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Simple scoring based on output logits
                if outputs.scores:
                    logits = outputs.scores[0][0]
                    # Get softmax probability
                    probs = self.torch.softmax(logits, dim=-1)
                    # Score based on probability (simplified)
                    score = probs.max().item()
                else:
                    score = 0.5

                scores.append(score)

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        corpus: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rerank candidate documents.

        Args:
            query: Query text
            candidates: List of (doc_id, initial_score) from first-stage retrieval
            corpus: Full corpus for document lookup
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of (doc_id, score) tuples
        """
        if not candidates:
            return []

        # Get document texts
        doc_ids = [doc_id for doc_id, _ in candidates]
        doc_texts = [self._get_doc_text(corpus.get(doc_id, "")) for doc_id in doc_ids]

        # Score using appropriate method
        if self.model_type == "monot5":
            scores = self._score_monot5(query, doc_texts)
        else:
            scores = self._score_cross_encoder(query, doc_texts)

        # Combine with doc IDs and sort
        results = list(zip(doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def rerank_batch(
        self,
        queries: List[str],
        candidates_per_query: Dict[str, List[Tuple[str, float]]],
        corpus: Dict[str, Any],
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch reranking for multiple queries."""
        results = {}
        for query in queries:
            candidates = candidates_per_query.get(query, [])
            results[query] = self.rerank(query, candidates, corpus, top_k=top_k)
        return results


class LLMReranker:
    """LLM-based pointwise reranking using any LLM provider.

    Scores each document individually against the query.

    Example:
        ```python
        from biopat.llm import create_provider
        provider = create_provider("openai", model="gpt-4o")
        reranker = LLMReranker(llm_provider=provider)
        reranked = reranker.rerank(query, candidates, corpus, top_k=10)
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        provider: str = "openai",
        llm_provider: Optional[Any] = None,
    ):
        self.model = model
        self.provider_name = provider
        self._llm = llm_provider

        if self._llm is None:
            try:
                from biopat.llm import create_provider as _create
                self._llm = _create(provider, model=model, api_key=api_key)
            except (ImportError, ValueError):
                # Legacy fallback
                if provider == "openai":
                    import openai
                    self.client = openai.OpenAI(api_key=api_key)
                elif provider == "anthropic":
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=api_key)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

    def _score_document(
        self,
        query: str,
        doc_id: str,
        doc_text: str,
    ) -> Tuple[float, str]:
        """Score a single document."""
        prompt = f"""Rate the relevance of this scientific article as prior art for the given patent claim.

Patent Claim:
{query[:1000]}

Scientific Article ({doc_id}):
{doc_text[:1500]}

Rate relevance from 0-10 where:
- 0: Not relevant at all
- 5: Somewhat related topic
- 10: Directly anticipates or discloses the claimed invention

Respond with ONLY a JSON object:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""

        if self._llm is not None:
            response = self._llm.generate(prompt=prompt, max_tokens=200, temperature=0)
            self._last_response = response
            try:
                import json
                result = json.loads(response.text)
                return result.get("score", 5) / 10.0, result.get("reasoning", "")
            except Exception:
                return 0.5, "Failed to parse response"

        # Legacy fallback
        if self.provider_name == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=200,
            )
            text = resp.choices[0].message.content
        else:
            resp = self.client.messages.create(
                model=self.model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text

        try:
            import json
            result = json.loads(text)
            return result.get("score", 5) / 10.0, result.get("reasoning", "")
        except Exception:
            return 0.5, "Failed to parse response"

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        corpus: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rerank using LLM."""
        results = []
        candidates = candidates[:min(len(candidates), 20)]

        for doc_id, _ in candidates:
            doc = corpus.get(doc_id, {})
            doc_text = doc if isinstance(doc, str) else f"{doc.get('title', '')} {doc.get('text', '')}"
            score, _ = self._score_document(query, doc_id, doc_text)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class ListwiseLLMReranker:
    """Listwise LLM reranker — sends all candidates in a single prompt.

    More efficient than pointwise: one API call per query instead of per document.

    Example:
        ```python
        from biopat.llm import create_provider
        provider = create_provider("openai", model="gpt-4o")
        reranker = ListwiseLLMReranker(provider)
        ranked = reranker.rerank(query, candidates, corpus, top_k=10)
        ```
    """

    def __init__(self, llm_provider: Any, window_size: int = 20):
        self._llm = llm_provider
        self.window_size = window_size
        self._last_response = None

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        corpus: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rerank candidates using a single listwise prompt."""
        candidates = candidates[:self.window_size]

        # Build document list for prompt
        doc_summaries = []
        doc_ids = []
        for i, (doc_id, _) in enumerate(candidates):
            doc = corpus.get(doc_id, {})
            if isinstance(doc, str):
                text = doc[:300]
            else:
                text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()[:300]
            doc_summaries.append(f"[{i+1}] {doc_id}: {text}")
            doc_ids.append(doc_id)

        docs_text = "\n\n".join(doc_summaries)

        prompt = f"""You are a patent prior art expert. Rank these scientific articles by relevance to the patent claim.

PATENT CLAIM:
{query[:1000]}

CANDIDATE ARTICLES:
{docs_text}

Rank ALL articles from most to least relevant. Respond with ONLY a JSON object:
{{"ranking": [<list of article numbers 1-{len(candidates)} from most to least relevant>]}}"""

        response = self._llm.generate(
            prompt=prompt,
            system_prompt="You are an expert patent examiner. Respond only with valid JSON.",
            max_tokens=500,
            temperature=0,
        )
        self._last_response = response

        try:
            import json
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            data = json.loads(text)
            ranking = data.get("ranking", list(range(1, len(candidates) + 1)))
        except (json.JSONDecodeError, Exception):
            ranking = list(range(1, len(candidates) + 1))

        # Convert ranking to scored results
        results = []
        for rank, idx in enumerate(ranking):
            if 1 <= idx <= len(doc_ids):
                score = 1.0 - (rank / len(ranking))
                results.append((doc_ids[idx - 1], score))

        return results[:top_k]


def create_reranker(
    model_type: str = "cross-encoder",
    use_gpu: bool = False,
) -> Union[CrossEncoderReranker, LLMReranker]:
    """Factory function for rerankers.

    Args:
        model_type: "cross-encoder", "bge", "monot5", or "llm"
        use_gpu: Use GPU acceleration

    Returns:
        Configured reranker instance
    """
    if model_type == "llm":
        return LLMReranker()

    if model_type == "monot5":
        model_name = RERANKER_MODELS["monot5"]
        config = RerankerConfig(model_name=model_name, use_monot5=True, use_gpu=use_gpu)
    elif model_type == "bge":
        model_name = RERANKER_MODELS["bge-reranker"]
        config = RerankerConfig(model_name=model_name, use_gpu=use_gpu)
    else:  # cross-encoder
        model_name = RERANKER_MODELS["ms-marco"]
        config = RerankerConfig(model_name=model_name, use_gpu=use_gpu)

    return CrossEncoderReranker(model_name=model_name, config=config)
