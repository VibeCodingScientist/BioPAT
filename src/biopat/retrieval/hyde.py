"""HyDE: Hypothetical Document Embeddings for Query Expansion.

Implements the SOTA HyDE technique:
1. Given a query, use LLM to generate a hypothetical document that would answer it
2. Embed the hypothetical document instead of the query
3. Retrieve documents similar to the hypothetical document

This significantly improves retrieval quality by bridging the
vocabulary gap between queries and documents.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE query expansion."""

    # LLM settings
    llm_provider: str = "openai"  # "openai", "anthropic", "local"
    llm_model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None

    # Generation settings
    num_hypothetical_docs: int = 1  # Generate multiple for ensemble
    max_tokens: int = 256
    temperature: float = 0.7

    # Domain-specific prompts
    domain: str = "patent"  # "patent", "scientific", "general"


# Domain-specific prompts for generating hypothetical documents
HYDE_PROMPTS = {
    "patent": """You are a scientific researcher writing about prior art for a patent application.

Given this patent claim, write a short scientific article abstract (100-150 words) that would be relevant prior art. The abstract should:
- Describe research that predates and anticipates the claimed invention
- Use scientific/technical language matching the domain
- Include specific details about methods, compounds, or results

Patent Claim:
{query}

Write ONLY the hypothetical prior art abstract, nothing else:""",

    "scientific": """Given this research query, write a short scientific article abstract (100-150 words) that would directly answer or address the query.

Query: {query}

Write ONLY the hypothetical abstract:""",

    "general": """Given this search query, write a short passage (100-150 words) that would be a perfect search result.

Query: {query}

Write ONLY the hypothetical passage:""",
}


class HyDEQueryExpander:
    """SOTA HyDE Query Expansion using LLM-generated hypothetical documents.

    HyDE works by:
    1. Using an LLM to generate a "hypothetical" document that would answer the query
    2. Embedding this hypothetical document (not the query)
    3. Using this embedding to retrieve similar real documents

    This bridges the vocabulary/semantic gap between short queries and long documents.

    Example:
        ```python
        from biopat.llm import create_provider

        provider = create_provider("openai", model="gpt-4o")
        hyde = HyDEQueryExpander(provider=provider)

        hypothetical = hyde.expand_query(
            "anti-PD-1 antibody for treating melanoma"
        )
        ```
    """

    def __init__(
        self,
        config: Optional[HyDEConfig] = None,
        api_key: Optional[str] = None,
        provider: Optional[Any] = None,
    ):
        self.config = config or HyDEConfig()

        if api_key:
            self.config.api_key = api_key

        # Use unified LLMProvider if supplied, else fall back to legacy init
        self._provider = provider
        if self._provider is None:
            self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize the LLM client based on provider (legacy path)."""
        try:
            from biopat.llm import create_provider
            self._provider = create_provider(
                self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.api_key,
            )
        except (ImportError, ValueError):
            # Fall back to direct SDK init for backwards compat
            if self.config.llm_provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=self.config.api_key)
            elif self.config.llm_provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            else:
                raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate a single hypothetical document."""
        prompt_template = HYDE_PROMPTS.get(
            self.config.domain, HYDE_PROMPTS["general"]
        )
        prompt = prompt_template.format(query=query)

        if self._provider is not None:
            response = self._provider.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that generates scientific text.",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            self._last_response = response
            return response.text

        # Legacy fallback (direct SDK)
        if self.config.llm_provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates scientific text."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content.strip()
        elif self.config.llm_provider == "anthropic":
            resp = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        raise ValueError(f"Cannot generate with provider: {self.config.llm_provider}")

    def expand_query(self, query: str) -> str:
        """Expand query to hypothetical document.

        Args:
            query: Original query text

        Returns:
            Hypothetical document text
        """
        logger.debug(f"Generating hypothetical document for: {query[:50]}...")
        hypothetical = self._generate_hypothetical_doc(query)
        logger.debug(f"Generated: {hypothetical[:100]}...")
        return hypothetical

    def expand_queries_batch(self, queries: List[str]) -> List[str]:
        """Expand multiple queries.

        Args:
            queries: List of query texts

        Returns:
            List of hypothetical documents
        """
        return [self.expand_query(q) for q in queries]

    def expand_with_ensemble(self, query: str) -> List[str]:
        """Generate multiple hypothetical documents for ensemble retrieval.

        Generates N hypothetical documents and returns all of them.
        Caller can retrieve using each and combine results (e.g., via RRF).

        Args:
            query: Original query text

        Returns:
            List of hypothetical documents
        """
        hypotheticals = []
        for _ in range(self.config.num_hypothetical_docs):
            hypotheticals.append(self._generate_hypothetical_doc(query))
        return hypotheticals


class QueryExpansionPipeline:
    """Complete query expansion pipeline combining multiple techniques.

    Implements:
    - HyDE (hypothetical document generation)
    - Pseudo-relevance feedback (PRF)
    - Query reformulation
    """

    def __init__(
        self,
        hyde_expander: Optional[HyDEQueryExpander] = None,
        use_prf: bool = True,
        prf_top_k: int = 5,
        prf_terms: int = 10,
    ):
        self.hyde = hyde_expander
        self.use_prf = use_prf
        self.prf_top_k = prf_top_k
        self.prf_terms = prf_terms

    def expand_with_hyde(self, query: str) -> str:
        """Expand query using HyDE."""
        if self.hyde is None:
            return query
        return self.hyde.expand_query(query)

    def expand_with_prf(
        self,
        query: str,
        initial_results: List[Tuple[str, float]],
        corpus: Dict[str, Any],
    ) -> str:
        """Expand query using pseudo-relevance feedback.

        Takes top-k initial results and extracts key terms to expand query.
        """
        if not initial_results:
            return query

        # Get text from top results
        top_docs_text = []
        for doc_id, _ in initial_results[:self.prf_top_k]:
            doc = corpus.get(doc_id, {})
            if isinstance(doc, str):
                top_docs_text.append(doc)
            else:
                top_docs_text.append(f"{doc.get('title', '')} {doc.get('text', '')}")

        # Extract key terms (simplified TF-based)
        combined_text = " ".join(top_docs_text).lower()
        import re
        words = re.findall(r'\b[a-z][a-z0-9-]+\b', combined_text)

        # Count term frequency
        from collections import Counter
        term_freq = Counter(words)

        # Remove stopwords and query terms
        stopwords = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'with', 'by', 'as', 'at', 'from',
                    'or', 'is', 'are', 'be', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
                    'this', 'that', 'these', 'those', 'which', 'who', 'what', 'where', 'when'}
        query_terms = set(query.lower().split())

        filtered_terms = {
            term: count for term, count in term_freq.items()
            if term not in stopwords and term not in query_terms and len(term) > 2
        }

        # Get top expansion terms
        expansion_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in expansion_terms[:self.prf_terms]]

        # Expand query
        expanded = f"{query} {' '.join(expansion_terms)}"
        return expanded

    def expand(
        self,
        query: str,
        initial_results: Optional[List[Tuple[str, float]]] = None,
        corpus: Optional[Dict[str, Any]] = None,
        method: str = "hyde",  # "hyde", "prf", "both"
    ) -> str:
        """Expand query using specified method.

        Args:
            query: Original query
            initial_results: Results from first-pass retrieval (for PRF)
            corpus: Document corpus (for PRF)
            method: Expansion method ("hyde", "prf", or "both")

        Returns:
            Expanded query
        """
        if method == "hyde":
            return self.expand_with_hyde(query)

        elif method == "prf":
            if initial_results is None or corpus is None:
                logger.warning("PRF requires initial_results and corpus")
                return query
            return self.expand_with_prf(query, initial_results, corpus)

        elif method == "both":
            # First HyDE, then use that for retrieval, then PRF
            hyde_expanded = self.expand_with_hyde(query)
            # Note: caller would need to do retrieval between steps
            return hyde_expanded

        else:
            raise ValueError(f"Unknown expansion method: {method}")


def create_hyde_expander(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    domain: str = "patent",
) -> HyDEQueryExpander:
    """Factory function for HyDE expander.

    Args:
        provider: LLM provider ("openai" or "anthropic")
        model: Model name
        api_key: API key (or set via environment)
        domain: Domain for prompt selection ("patent", "scientific", "general")

    Returns:
        Configured HyDEQueryExpander
    """
    config = HyDEConfig(
        llm_provider=provider,
        llm_model=model,
        api_key=api_key,
        domain=domain,
    )

    return HyDEQueryExpander(config=config)
