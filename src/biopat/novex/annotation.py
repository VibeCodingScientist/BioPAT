"""Multi-LLM annotation protocol for NovEx.

3-model relevance judgment + novelty determination with consensus voting.
Computes Fleiss' kappa, Cohen's kappa, and examiner agreement.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from biopat.llm import CostTracker, create_provider
from biopat.novex._util import CheckpointMixin, parse_llm_json, majority_vote

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM = (
    "You are a patent examiner. Judge if this document is prior art for the claim. "
    "Scale: 0=not relevant, 1=marginally, 2=relevant, 3=highly relevant. "
    'Output JSON: {"relevance_score": <0-3>, "reasoning": "<brief>", "is_prior_art": <bool>}'
)

NOVELTY_SYSTEM = (
    "Determine novelty of the claim given prior art. "
    'Output JSON: {"novelty_label": "NOVEL"|"ANTICIPATED"|"PARTIALLY_ANTICIPATED", '
    '"reasoning": "<brief>", "key_evidence": ["<doc_id>"], "confidence": <0-1>}'
)

DEFAULT_MODELS = {
    "openai": {"provider": "openai", "model_id": "gpt-5.2"},
    "anthropic": {"provider": "anthropic", "model_id": "claude-sonnet-4-6"},
    "google": {"provider": "google", "model_id": "gemini-3-pro-preview"},
}


@dataclass
class RelevanceJudgment:
    statement_id: str
    doc_id: str
    model: str
    provider: str
    relevance_score: int
    reasoning: str
    is_prior_art: bool
    cost_usd: float = 0.0


@dataclass
class ConsensusLabel:
    statement_id: str
    doc_id: str
    consensus_score: int
    individual_scores: Dict[str, int]
    agreement: str  # unanimous / majority / no_consensus
    examiner_agrees: Optional[bool] = None


class AnnotationProtocol(CheckpointMixin):
    """Multi-LLM annotation: 3 models judge relevance, consensus via majority vote.

    Usage:
        protocol = AnnotationProtocol()
        judgments, consensus = await protocol.annotate_tier2(statements, candidates)
    """

    def __init__(
        self,
        output_dir: str = "data/novex/annotation",
        checkpoint_dir: Optional[str] = None,
        models: Optional[Dict[str, Dict]] = None,
        budget_usd: float = 100.0,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.output_dir / "checkpoints")
        self.model_configs = models or DEFAULT_MODELS
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = CostTracker(max_budget_usd=budget_usd)
        self._providers: Dict[str, Any] = {}

    def _get_provider(self, name: str):
        if name not in self._providers:
            cfg = self.model_configs[name]
            self._providers[name] = create_provider(cfg["provider"], model=cfg["model_id"])
        return self._providers[name]

    # --- Tier 2: Relevance ---

    async def annotate_tier2(
        self,
        statements: List[Dict],
        candidates_per_stmt: Dict[str, List[Dict]],
        examiner_gt: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Tuple[List[RelevanceJudgment], List[ConsensusLabel]]:
        cached = self._load_checkpoint("tier2_judgments")
        if cached is not None:
            return (
                [RelevanceJudgment(**j) for j in cached["judgments"]],
                [ConsensusLabel(**c) for c in cached["consensus"]],
            )

        all_j: List[RelevanceJudgment] = []
        all_c: List[ConsensusLabel] = []
        pair_n = 0

        for stmt in statements:
            sid, text = stmt["statement_id"], stmt["text"]
            for doc in candidates_per_stmt.get(sid, []):
                did = doc["_id"]
                prompt = f"CLAIM: {text}\n\nDOCUMENT: {doc.get('title','')}\n{doc.get('text','')[:2000]}"
                scores: Dict[str, int] = {}
                for mname in self.model_configs:
                    try:
                        parsed = parse_llm_json(
                            self._get_provider(mname), prompt, RELEVANCE_SYSTEM,
                            self.cost_tracker, f"{sid}_{did}", "tier2",
                            thinking=True,
                        )
                        score = min(3, max(0, int(parsed.get("relevance_score", 0))))
                        scores[mname] = score
                        all_j.append(RelevanceJudgment(
                            statement_id=sid, doc_id=did, model=self.model_configs[mname]["model_id"],
                            provider=mname, relevance_score=score,
                            reasoning=parsed.get("reasoning", ""),
                            is_prior_art=parsed.get("is_prior_art", False),
                        ))
                    except Exception as e:
                        logger.error("Tier2 fail (%s,%s,%s): %s", mname, sid, did, e)

                if len(scores) >= 2:
                    winner, agree = majority_vote(scores)
                    ex_agrees = None
                    if examiner_gt and sid in examiner_gt:
                        ex = examiner_gt[sid].get(did)
                        if ex is not None:
                            ex_agrees = (winner > 0) == (ex > 0)
                    all_c.append(ConsensusLabel(
                        statement_id=sid, doc_id=did, consensus_score=winner,
                        individual_scores=scores, agreement=agree, examiner_agrees=ex_agrees,
                    ))
                pair_n += 1
                if pair_n % 200 == 0:
                    logger.info("Tier2: %d pairs, $%.2f", pair_n, self.cost_tracker.total_cost)

        self._save_checkpoint("tier2_judgments", {
            "judgments": [j.__dict__ for j in all_j],
            "consensus": [c.__dict__ for c in all_c],
        })
        logger.info("Tier2 done: %d judgments, %d consensus", len(all_j), len(all_c))
        return all_j, all_c

    # --- Tier 3: Novelty ---

    async def annotate_tier3(
        self,
        statements: List[Dict],
        prior_art: Dict[str, List[Dict]],
    ) -> Tuple[List[Dict], List[Dict]]:
        cached = self._load_checkpoint("tier3_novelty")
        if cached is not None:
            return cached["judgments"], cached["consensus"]

        all_j, all_c = [], []
        for stmt in statements:
            sid, text = stmt["statement_id"], stmt["text"]
            docs = prior_art.get(sid, [])
            pa_text = "\n---\n".join(
                f"[{d.get('_id','')}] {d.get('title','')}\n{d.get('text','')[:500]}"
                for d in docs[:10]
            ) or "No prior art found."
            prompt = f"CLAIM: {text}\n\nPRIOR ART:\n{pa_text}"

            labels: Dict[str, str] = {}
            for mname in self.model_configs:
                try:
                    parsed = parse_llm_json(
                        self._get_provider(mname), prompt, NOVELTY_SYSTEM,
                        self.cost_tracker, sid, "tier3",
                        thinking=True,
                    )
                    label = parsed.get("novelty_label", "NOVEL")
                    labels[mname] = label
                    all_j.append({"statement_id": sid, "provider": mname, "label": label,
                                  "reasoning": parsed.get("reasoning", "")})
                except Exception as e:
                    logger.error("Tier3 fail (%s,%s): %s", mname, sid, e)

            if len(labels) >= 2:
                winner, agree = majority_vote(labels)
                all_c.append({"statement_id": sid, "label": winner, "agreement": agree,
                              "individual": labels})

        self._save_checkpoint("tier3_novelty", {"judgments": all_j, "consensus": all_c})
        return all_j, all_c

    # --- Agreement Statistics ---

    def compute_agreement(self, judgments: List[RelevanceJudgment], consensus: List[ConsensusLabel]) -> Dict[str, Any]:
        models = sorted(set(j.provider for j in judgments))
        # Group by pair
        pairs: Dict[Tuple[str, str], Dict[str, int]] = {}
        for j in judgments:
            pairs.setdefault((j.statement_id, j.doc_id), {})[j.provider] = j.relevance_score

        total = len(consensus)
        examiner_compared = [c for c in consensus if c.examiner_agrees is not None]
        return {
            "fleiss_kappa": self._fleiss_kappa(pairs, models),
            "pairwise_kappa": {
                f"{m1}_vs_{m2}": self._cohens_kappa(pairs, m1, m2)
                for i, m1 in enumerate(models) for m2 in models[i+1:]
            },
            "unanimous": sum(1 for c in consensus if c.agreement == "unanimous") / total if total else 0,
            "majority": sum(1 for c in consensus if c.agreement == "majority") / total if total else 0,
            "examiner_agreement": (
                sum(1 for c in examiner_compared if c.examiner_agrees) / len(examiner_compared)
                if examiner_compared else None
            ),
            "total_pairs": total,
        }

    def _fleiss_kappa(self, pairs, models, k=4) -> float:
        n = len(models)
        matrix = []
        for scores in pairs.values():
            if len(scores) < n:
                continue
            row = [0] * k
            for m in models:
                if m in scores:
                    row[min(k-1, max(0, scores[m]))] += 1
            matrix.append(row)
        if not matrix:
            return 0.0
        p_i = [(sum(x*x for x in r) - sum(r)) / (sum(r) * (sum(r)-1)) for r in matrix if sum(r) > 1]
        if not p_i:
            return 0.0
        p_bar = sum(p_i) / len(p_i)
        total = sum(sum(r) for r in matrix)
        p_e = sum((sum(r[j] for r in matrix) / total) ** 2 for j in range(k))
        return (p_bar - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-10 else 1.0

    def _cohens_kappa(self, pairs, m1, m2, k=4) -> float:
        conf = [[0]*k for _ in range(k)]
        for scores in pairs.values():
            if m1 in scores and m2 in scores:
                conf[min(k-1, max(0, scores[m1]))][min(k-1, max(0, scores[m2]))] += 1
        total = sum(sum(r) for r in conf)
        if total == 0:
            return 0.0
        p_o = sum(conf[i][i] for i in range(k)) / total
        p_e = sum((sum(conf[i]) / total) * (sum(conf[j][i] for j in range(k)) / total) for i in range(k))
        return (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-10 else 1.0

    def save_outputs(self, judgments, consensus, stats) -> None:
        with open(self.output_dir / "llm_judgments.jsonl", "w") as f:
            for j in judgments:
                f.write(json.dumps(j.__dict__ if hasattr(j, '__dict__') else j, default=str) + "\n")
        with open(self.output_dir / "agreement.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        self.cost_tracker.save(str(self.output_dir / "annotation_costs.json"))
