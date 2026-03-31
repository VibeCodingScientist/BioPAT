"""Dual-retrieval agent for BioPAT Experiment 7.

Given a scientific claim, an LLM agent iteratively searches a dual corpus
(papers + patents) using BM25, refines its strategy, and produces a final
ranked list. The full reasoning trace is captured for analysis.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from biopat.llm import LLMProvider, LLMResponse, CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for the retrieval agent."""

    max_search_calls: int = 5
    search_top_k: int = 20
    final_list_size: int = 100
    temperature: float = 0.2
    max_tokens_plan: int = 800
    max_tokens_refine: int = 600
    max_tokens_rank: int = 1500


# ---------------------------------------------------------------------------
# Agent trace (session log)
# ---------------------------------------------------------------------------

@dataclass
class SearchCall:
    """A single search invocation by the agent."""

    query: str
    num_results: int
    top_doc_ids: List[str]
    step: int


@dataclass
class AgentStep:
    """A single reasoning step (LLM call)."""

    phase: str  # "plan", "refine", "rank"
    prompt_preview: str  # First 200 chars of prompt
    response_preview: str  # First 500 chars of response
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class AgentTrace:
    """Full session trace for one query."""

    query_id: str
    query_text: str
    search_calls: List[SearchCall] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    final_ranking: List[Tuple[str, float]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text[:200],
            "num_search_calls": len(self.search_calls),
            "num_steps": len(self.steps),
            "final_ranking_size": len(self.final_ranking),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
            "search_calls": [
                {"query": sc.query, "num_results": sc.num_results, "step": sc.step}
                for sc in self.search_calls
            ],
            "steps": [
                {"phase": s.phase, "tokens": s.input_tokens + s.output_tokens, "cost": s.cost_usd}
                for s in self.steps
            ],
            "final_ranking": [
                {"doc_id": did, "score": score}
                for did, score in self.final_ranking[:20]  # Top-20 for logging
            ],
        }


# ---------------------------------------------------------------------------
# BM25 search tool over dual corpus
# ---------------------------------------------------------------------------

class DualCorpusSearchTool:
    """BM25 search over a unified corpus of papers and patents.

    Built once per experiment and shared across all agent invocations.
    """

    def __init__(self, corpus: Dict[str, dict]):
        """Build BM25 index over the dual corpus.

        Args:
            corpus: Dict mapping doc_id to {title, text, doc_type}.
        """
        self.doc_ids: List[str] = list(corpus.keys())
        self.doc_types: Dict[str, str] = {
            did: doc.get("doc_type", "paper") for did, doc in corpus.items()
        }
        self.doc_texts: Dict[str, str] = {}

        tokenized: List[List[str]] = []
        for did in self.doc_ids:
            doc = corpus[did]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            self.doc_texts[did] = text
            tokenized.append(text.lower().split())

        logger.info("Building BM25 index over %d documents...", len(self.doc_ids))
        t0 = time.time()
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built in %.1fs", time.time() - t0)

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the dual corpus.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            doc_type_filter: If set, only return docs of this type.

        Returns:
            List of dicts with doc_id, score, doc_type, title_preview.
        """
        scores = self.bm25.get_scores(query.lower().split())

        # Get more candidates if filtering by type
        fetch_k = top_k * 3 if doc_type_filter else top_k
        top_indices = scores.argsort()[-fetch_k:][::-1]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            did = self.doc_ids[idx]
            dtype = self.doc_types[did]

            if doc_type_filter and dtype != doc_type_filter:
                continue

            text = self.doc_texts[did]
            results.append({
                "doc_id": did,
                "score": float(scores[idx]),
                "doc_type": dtype,
                "text_preview": text[:300],
            })

            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """You are an expert prior-art search agent. Given a scientific claim \
(from a patent or paper), you must find relevant papers AND patents in a dual corpus.

You have access to a BM25 keyword search tool. You can make up to {max_calls} search \
calls. Plan your search strategy carefully:
- Decompose the claim into key technical concepts
- Consider synonyms, alternate terminology, and related fields
- Search for both papers and patents that are relevant

Respond with a JSON object:
{{
  "analysis": "Brief analysis of the claim's key concepts",
  "search_queries": ["query1", "query2", ...],
  "reasoning": "Why these queries cover the claim well"
}}

Return 1-{max_calls} search queries. Each should target different aspects of the claim."""

PLAN_USER = """Scientific Claim:
{claim}

Plan your search queries (up to {max_calls} BM25 searches over a corpus of {corpus_size} \
papers and patents). Return JSON only."""

REFINE_SYSTEM = """You are an expert prior-art search agent reviewing initial search results.

You have used {used_calls}/{max_calls} search calls. You may issue more searches to \
fill gaps, or stop if results are sufficient.

Respond with a JSON object:
{{
  "assessment": "Are the results covering both papers and patents relevant to the claim?",
  "gaps": "What aspects of the claim are not yet covered?",
  "action": "search" or "done",
  "new_queries": ["query1", ...] (only if action is "search", max {remaining} more)
}}"""

REFINE_USER = """Original Claim:
{claim}

Results so far ({total_results} unique documents, {paper_count} papers, {patent_count} patents):
{results_summary}

Decide: search more or finalize? Return JSON only."""

RANK_SYSTEM = """You are an expert prior-art search agent. Given a scientific claim and \
a set of candidate documents, produce a final relevance ranking.

For each document, assign a relevance score from 0.0 to 1.0:
- 1.0: Directly anticipates or is highly relevant prior art
- 0.7-0.9: Strongly relevant, covers most key concepts
- 0.4-0.6: Moderately relevant, covers some concepts
- 0.1-0.3: Marginally relevant
- 0.0: Not relevant

Respond with a JSON object:
{{
  "ranking": [
    {{"doc_id": "...", "score": 0.95, "reason": "..."}},
    ...
  ]
}}

Return up to {final_size} documents, sorted by score descending. Include BOTH papers \
and patents if relevant."""

RANK_USER = """Scientific Claim:
{claim}

Candidate Documents ({num_candidates} total):
{candidates}

Produce your final ranked list (up to {final_size} documents). Return JSON only."""


# ---------------------------------------------------------------------------
# Retrieval agent
# ---------------------------------------------------------------------------

class RetrievalAgent:
    """Multi-turn LLM agent for dual-corpus retrieval.

    Phases:
      1. Plan — LLM analyzes claim, outputs search queries
      2. Refine — LLM reviews results, optionally searches more
      3. Rank — LLM produces final ranked list from accumulated results
    """

    def __init__(
        self,
        provider: LLMProvider,
        search_tool: DualCorpusSearchTool,
        config: Optional[AgentConfig] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.provider = provider
        self.search_tool = search_tool
        self.config = config or AgentConfig()
        self.cost_tracker = cost_tracker

    def run(self, query_id: str, query_text: str) -> AgentTrace:
        """Run the full agent loop for one query.

        Args:
            query_id: Unique query identifier.
            query_text: The scientific claim text.

        Returns:
            AgentTrace with full session log and final ranking.
        """
        trace = AgentTrace(query_id=query_id, query_text=query_text)
        accumulated: Dict[str, Dict[str, Any]] = {}  # doc_id → result dict
        search_count = 0

        try:
            # Phase 1: Plan
            plan_queries = self._plan(query_text, trace)

            # Execute planned searches
            for q in plan_queries[:self.config.max_search_calls]:
                results = self.search_tool.search(q, top_k=self.config.search_top_k)
                search_count += 1
                trace.search_calls.append(SearchCall(
                    query=q,
                    num_results=len(results),
                    top_doc_ids=[r["doc_id"] for r in results[:5]],
                    step=search_count,
                ))
                for r in results:
                    if r["doc_id"] not in accumulated:
                        accumulated[r["doc_id"]] = r

            # Phase 2: Refine (if budget remains)
            remaining = self.config.max_search_calls - search_count
            if remaining > 0 and accumulated:
                refine_queries = self._refine(
                    query_text, accumulated, search_count, trace,
                )
                for q in refine_queries[:remaining]:
                    results = self.search_tool.search(q, top_k=self.config.search_top_k)
                    search_count += 1
                    trace.search_calls.append(SearchCall(
                        query=q,
                        num_results=len(results),
                        top_doc_ids=[r["doc_id"] for r in results[:5]],
                        step=search_count,
                    ))
                    for r in results:
                        if r["doc_id"] not in accumulated:
                            accumulated[r["doc_id"]] = r

            # Phase 3: Rank
            if accumulated:
                trace.final_ranking = self._rank(query_text, accumulated, trace)
            else:
                trace.final_ranking = []

        except Exception as e:
            logger.error("Agent failed for %s: %s", query_id, e)
            trace.error = str(e)
            # Fall back to BM25 scores if we have any results
            if accumulated:
                trace.final_ranking = sorted(
                    [(did, r["score"]) for did, r in accumulated.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:self.config.final_list_size]

        # Totals
        trace.total_tokens = sum(s.input_tokens + s.output_tokens for s in trace.steps)
        trace.total_cost_usd = sum(s.cost_usd for s in trace.steps)
        trace.total_latency_ms = sum(s.latency_ms for s in trace.steps)

        return trace

    # --- Phase implementations ---

    def _plan(self, claim: str, trace: AgentTrace) -> List[str]:
        """Phase 1: Analyze claim and generate search queries."""
        system = PLAN_SYSTEM.format(max_calls=self.config.max_search_calls)
        user = PLAN_USER.format(
            claim=claim[:1500],
            max_calls=self.config.max_search_calls,
            corpus_size=len(self.search_tool.doc_ids),
        )

        response = self.provider.generate(
            prompt=user,
            system_prompt=system,
            max_tokens=self.config.max_tokens_plan,
            temperature=self.config.temperature,
        )
        self._record_step("plan", user, response, trace)

        try:
            data = self._parse_json(response.text)
            queries = data.get("search_queries", [])
            if not queries:
                # Fallback: use the claim itself
                queries = [claim[:200]]
            return queries
        except (json.JSONDecodeError, KeyError):
            logger.warning("Plan parse failed, using claim as query")
            return [claim[:200]]

    def _refine(
        self,
        claim: str,
        accumulated: Dict[str, Dict[str, Any]],
        used_calls: int,
        trace: AgentTrace,
    ) -> List[str]:
        """Phase 2: Review results and optionally refine search."""
        remaining = self.config.max_search_calls - used_calls

        # Build results summary
        papers = [r for r in accumulated.values() if r["doc_type"] == "paper"]
        patents = [r for r in accumulated.values() if r["doc_type"] == "patent"]

        # Show top results by score
        sorted_results = sorted(accumulated.values(), key=lambda x: x["score"], reverse=True)
        summary_lines = []
        for r in sorted_results[:15]:
            summary_lines.append(
                f"  [{r['doc_type']}] {r['doc_id']}: score={r['score']:.2f} — {r['text_preview'][:100]}"
            )

        system = REFINE_SYSTEM.format(
            used_calls=used_calls,
            max_calls=self.config.max_search_calls,
            remaining=remaining,
        )
        user = REFINE_USER.format(
            claim=claim[:1000],
            total_results=len(accumulated),
            paper_count=len(papers),
            patent_count=len(patents),
            results_summary="\n".join(summary_lines),
        )

        response = self.provider.generate(
            prompt=user,
            system_prompt=system,
            max_tokens=self.config.max_tokens_refine,
            temperature=self.config.temperature,
        )
        self._record_step("refine", user, response, trace)

        try:
            data = self._parse_json(response.text)
            if data.get("action") == "search":
                return data.get("new_queries", [])
            return []
        except (json.JSONDecodeError, KeyError):
            return []

    def _rank(
        self,
        claim: str,
        accumulated: Dict[str, Dict[str, Any]],
        trace: AgentTrace,
    ) -> List[Tuple[str, float]]:
        """Phase 3: Produce final relevance ranking."""
        # Sort candidates by BM25 score and take top candidates for LLM ranking
        sorted_candidates = sorted(
            accumulated.values(), key=lambda x: x["score"], reverse=True,
        )
        # Send more than final_list_size to give LLM room to filter
        candidates_for_ranking = sorted_candidates[:min(
            len(sorted_candidates), self.config.final_list_size * 2,
        )]

        # Format candidates
        candidate_lines = []
        for i, r in enumerate(candidates_for_ranking):
            candidate_lines.append(
                f"{i+1}. [{r['doc_type']}] {r['doc_id']}: {r['text_preview'][:200]}"
            )

        system = RANK_SYSTEM.format(final_size=self.config.final_list_size)
        user = RANK_USER.format(
            claim=claim[:1200],
            num_candidates=len(candidates_for_ranking),
            candidates="\n".join(candidate_lines),
            final_size=self.config.final_list_size,
        )

        response = self.provider.generate(
            prompt=user,
            system_prompt=system,
            max_tokens=self.config.max_tokens_rank,
            temperature=self.config.temperature,
        )
        self._record_step("rank", user, response, trace)

        try:
            data = self._parse_json(response.text)
            ranking = data.get("ranking", [])
            result: List[Tuple[str, float]] = []
            for item in ranking:
                doc_id = item.get("doc_id", "")
                score = float(item.get("score", 0.0))
                if doc_id and doc_id in accumulated:
                    result.append((doc_id, score))
            return result[:self.config.final_list_size]
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Rank parse failed, using BM25 scores as fallback")
            return [
                (r["doc_id"], r["score"])
                for r in sorted_candidates[:self.config.final_list_size]
            ]

    # --- Helpers ---

    def _record_step(
        self,
        phase: str,
        prompt: str,
        response: LLMResponse,
        trace: AgentTrace,
    ) -> None:
        """Record an LLM call in the trace and cost tracker."""
        step = AgentStep(
            phase=phase,
            prompt_preview=prompt[:200],
            response_preview=response.text[:500],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        trace.steps.append(step)

        if self.cost_tracker:
            self.cost_tracker.record_response(
                response, task=f"agent_{phase}", query_id=trace.query_id,
            )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)


def results_to_qrels_format(
    traces: List[AgentTrace],
) -> Dict[str, Dict[str, float]]:
    """Convert agent traces to the standard results format for metrics computation.

    Args:
        traces: List of AgentTrace objects from agent.run().

    Returns:
        Dict mapping query_id to {doc_id: score} for use with MetricsComputer.
    """
    results: Dict[str, Dict[str, float]] = {}
    for trace in traces:
        if trace.final_ranking:
            results[trace.query_id] = {
                doc_id: score for doc_id, score in trace.final_ranking
            }
        else:
            results[trace.query_id] = {}
    return results
