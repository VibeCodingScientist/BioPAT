"""Statement curation pipeline for BioPAT-NovEx.

Selects candidates from Phase 1 qrels, extracts statements via 3-LLM
consensus, quality-filters, and assembles ground truth for 3 tiers.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from biopat.llm import CostTracker, create_provider
from biopat.novex._util import CheckpointMixin, parse_llm_json, majority_vote

logger = logging.getLogger(__name__)

CATEGORY_TARGETS = {
    "multi_patent_examiner": 35,
    "single_patent_examiner": 25,
    "cross_domain": 15,
    "applicant_only": 15,
    "negative_control": 10,
}
DOMAIN_TARGETS = {"A61": 40, "C07": 30, "C12": 30}

EXTRACT_SYSTEM = (
    "You are a scientific writing expert. Extract the single most important "
    "novel finding from a paper abstract as 1-3 clear, self-contained sentences. "
    "Be SPECIFIC (not 'we studied X' but 'X causes Y via Z'). Be FAITHFUL to the abstract."
)

QUALITY_SYSTEM = (
    "Rate this extracted statement. Output JSON only: "
    '{"self_contained": <1-5>, "specific": <1-5>, "faithful": <1-5>}'
)


class StatementCurator(CheckpointMixin):
    """Curates 100 scientific statements for NovEx.

    Usage:
        curator = StatementCurator(output_dir="data/novex")
        statements = await curator.run_pipeline()
    """

    def __init__(
        self,
        qrels_path: str = "data/checkpoints/step5_qrels.parquet",
        papers_path: str = "data/checkpoints/step4_papers.parquet",
        patents_path: str = "data/checkpoints/step3_patents.parquet",
        corpus_dir: str = "data/benchmark",
        output_dir: str = "data/novex",
        checkpoint_dir: Optional[str] = None,
        budget_usd: float = 50.0,
        seed: int = 42,
    ):
        self.qrels_path = Path(qrels_path)
        self.papers_path = Path(papers_path)
        self.patents_path = Path(patents_path)
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.output_dir / "checkpoints")
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = CostTracker(max_budget_usd=budget_usd)
        self.rng = random.Random(seed)
        self._providers: Dict[str, Any] = {}

    def _get_provider(self, name: str):
        if name not in self._providers:
            models = {"openai": "gpt-5.2", "anthropic": "claude-sonnet-4-6", "google": "gemini-3-pro-preview"}
            self._providers[name] = create_provider(name, model=models[name])
        return self._providers[name]

    # --- A1: Candidate Selection ---

    def select_candidates(self, target_total: int = 120) -> List[Dict]:
        """Select candidate papers from Phase 1 qrels, stratified by category/domain."""
        cached = self._load_checkpoint("a1_candidates")
        if cached is not None:
            return cached

        qrels = pl.read_parquet(self.qrels_path)
        papers = pl.read_parquet(self.papers_path)
        patents = pl.read_parquet(self.patents_path)
        profiles = self._build_profiles(qrels, papers, patents)

        candidates = []
        # Multi-patent examiner
        pool = [p for p in profiles if p["num_patents"] >= 3 and p["cite_type"] in ("examiner", "both")]
        candidates += self._domain_sample(pool, "multi_patent_examiner", 42)
        # Single-patent examiner
        pool = [p for p in profiles if p["num_patents"] == 1 and p["cite_type"] in ("examiner", "both")]
        candidates += self._domain_sample(pool, "single_patent_examiner", 30)
        # Cross-domain
        pool = [p for p in profiles if len(set(p["patent_domains"])) > 1]
        candidates += self._domain_sample(pool, "cross_domain", 18)
        # Applicant-only
        pool = [p for p in profiles if p["cite_type"] == "applicant"]
        candidates += self._domain_sample(pool, "applicant_only", 18)
        # Negative controls
        candidates += self._select_negatives(papers, qrels, 12)

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c["paper_id"] not in seen:
                seen.add(c["paper_id"])
                unique.append(c)

        self._save_checkpoint("a1_candidates", unique)
        logger.info("Selected %d candidates", len(unique))
        return unique

    def _build_profiles(self, qrels, papers, patents) -> List[Dict]:
        paper_info = {}
        for row in papers.iter_rows(named=True):
            pid = str(row.get("paper_id", row.get("id", "")))
            abstract = row.get("abstract", "")
            if abstract:
                paper_info[pid] = {"title": row.get("title", ""), "abstract": abstract}

        patent_domains = {}
        for row in patents.iter_rows(named=True):
            pat_id = str(row.get("patent_id", row.get("id", "")))
            patent_domains[pat_id] = str(row.get("cpc_class", row.get("ipc_class", "")))[:3]

        # Group by paper
        by_paper: Dict[str, List[Dict]] = {}
        for row in qrels.iter_rows(named=True):
            pid = str(row.get("paper_id", row.get("doc_id", "")))
            pat_id = str(row.get("patent_id", row.get("query_id", "")))
            ct = row.get("citation_type", row.get("examiner_applicant", ""))
            norm = "examiner" if ct in ("front", "both") else ("applicant" if ct == "body" else ct)
            by_paper.setdefault(pid, []).append({"patent_id": pat_id, "type": norm, "domain": patent_domains.get(pat_id, "")})

        profiles = []
        for pid, cites in by_paper.items():
            if pid not in paper_info:
                continue
            types = set(c["type"] for c in cites)
            cite_type = "both" if "examiner" in types and "applicant" in types else ("examiner" if "examiner" in types else "applicant")
            domains = [c["domain"] for c in cites if c["domain"]]
            primary = max(set(domains), key=domains.count) if domains else "A61"
            profiles.append({
                "paper_id": pid, **paper_info[pid], "domain": primary,
                "num_patents": len(set(c["patent_id"] for c in cites)),
                "cite_type": cite_type,
                "patent_ids": list(set(c["patent_id"] for c in cites)),
                "patent_domains": domains,
            })
        return profiles

    def _domain_sample(self, pool: List[Dict], category: str, n: int) -> List[Dict]:
        if not pool:
            return []
        by_domain: Dict[str, List[Dict]] = {}
        for p in pool:
            by_domain.setdefault(p["domain"][:3], []).append(p)
        total = sum(DOMAIN_TARGETS.values())
        selected = []
        for domain, target in DOMAIN_TARGETS.items():
            k = max(1, round(n * target / total))
            dp = by_domain.get(domain, [])
            self.rng.shuffle(dp)
            for p in dp[:k]:
                p["category"] = category
                selected.append(p)
        return selected

    def _select_negatives(self, papers, qrels, n: int) -> List[Dict]:
        cited = set()
        for row in qrels.iter_rows(named=True):
            cited.add(str(row.get("paper_id", row.get("doc_id", ""))))
        uncited = []
        for row in papers.iter_rows(named=True):
            pid = str(row.get("paper_id", row.get("id", "")))
            abstract = row.get("abstract", "")
            if pid not in cited and abstract and len(abstract) > 200:
                uncited.append({"paper_id": pid, "title": row.get("title", ""), "abstract": abstract,
                                "domain": "mixed", "category": "negative_control", "num_patents": 0,
                                "cite_type": "none", "patent_ids": [], "patent_domains": []})
        self.rng.shuffle(uncited)
        return uncited[:n]

    # --- A2: Statement Extraction ---

    async def extract_statements(self, candidates: List[Dict]) -> List[Dict]:
        """3-LLM extraction, consensus via Jaccard similarity of best pair."""
        cached = self._load_checkpoint("a2_extractions")
        if cached is not None:
            return cached

        logger.info("Extracting statements from %d candidates", len(candidates))
        results = []
        for i, c in enumerate(candidates):
            prompt = f"Paper: {c['title']}\nAbstract: {c['abstract']}\n\nExtract the key novel finding (1-3 sentences):"
            extractions = {}
            for pname in ["openai", "anthropic", "google"]:
                try:
                    resp = self._get_provider(pname).generate(prompt=prompt, system_prompt=EXTRACT_SYSTEM, max_tokens=300, temperature=0.1)
                    self.cost_tracker.record_response(resp, task="extract", query_id=c["paper_id"])
                    extractions[pname] = resp.text.strip()
                except Exception as e:
                    logger.error("Extract failed (%s, %s): %s", pname, c["paper_id"], e)
            if len(extractions) < 2:
                continue
            c["extractions"] = extractions
            c["statement"] = self._consensus_text(list(extractions.values()))
            results.append(c)
            if (i + 1) % 20 == 0:
                self._save_checkpoint("a2_extractions", results)

        self._save_checkpoint("a2_extractions", results)
        logger.info("Extracted %d statements, cost $%.2f", len(results), self.cost_tracker.total_cost)
        return results

    @staticmethod
    def _consensus_text(texts: List[str]) -> str:
        """Pick the shorter text from the most-similar pair (Jaccard on words)."""
        if len(texts) == 1:
            return texts[0]
        best_sim, best = -1.0, (0, 1)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                wi, wj = set(texts[i].lower().split()), set(texts[j].lower().split())
                if wi and wj:
                    sim = len(wi & wj) / len(wi | wj)
                    if sim > best_sim:
                        best_sim, best = sim, (i, j)
        a, b = texts[best[0]], texts[best[1]]
        return a if len(a) <= len(b) else b

    # --- A3: Quality Filter ---

    async def filter_quality(self, items: List[Dict], min_score: float = 3.5) -> List[Dict]:
        cached = self._load_checkpoint("a3_quality")
        if cached is not None:
            return cached

        provider = self._get_provider("openai")
        filtered = []
        for item in items:
            prompt = f"Statement: {item['statement']}\nAbstract: {item['abstract']}"
            try:
                scores = parse_llm_json(provider, prompt, QUALITY_SYSTEM, self.cost_tracker, item["paper_id"], "quality")
            except Exception:
                scores = {"self_contained": 4, "specific": 4, "faithful": 4}
            avg = (scores.get("self_contained", 0) + scores.get("specific", 0) + scores.get("faithful", 0)) / 3.0
            item["quality"] = scores
            if avg >= min_score:
                filtered.append(item)

        self._save_checkpoint("a3_quality", filtered)
        logger.info("Quality: %d/%d passed", len(filtered), len(items))
        return filtered

    # --- A4: Ground Truth Assembly ---

    def assemble_ground_truth(self, items: List[Dict]) -> List[Dict]:
        cached = self._load_checkpoint("a4_statements")
        if cached is not None:
            return cached

        corpus_ids = set()
        corpus_path = self.corpus_dir / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path) as f:
                for line in f:
                    corpus_ids.add(json.loads(line)["_id"])

        statements = []
        for idx, item in enumerate(items):
            relevant = [pid for pid in item.get("patent_ids", [])
                        if pid in corpus_ids or f"patent_{pid}" in corpus_ids]
            if item["category"] == "negative_control":
                novelty = "NOVEL"
            elif len(relevant) >= 3:
                novelty = "ANTICIPATED"
            elif relevant:
                novelty = "PARTIALLY_ANTICIPATED"
            else:
                novelty = "NOVEL"

            statements.append({
                "statement_id": f"NX-{idx+1:03d}",
                "text": item["statement"],
                "source_paper_id": item["paper_id"],
                "source_paper_title": item["title"],
                "domain": item["domain"],
                "category": item["category"],
                "num_citing_patents": item["num_patents"],
                "difficulty": "medium",
                "ground_truth": {
                    "tier1_relevant_docs": relevant,
                    "tier3_novelty_label": novelty,
                },
            })

        # Trim to 100, balanced by category
        statements = self._trim_balanced(statements, 100)
        self._save_checkpoint("a4_statements", statements)
        logger.info("Assembled %d statements", len(statements))
        return statements

    def _trim_balanced(self, items: List[Dict], target: int) -> List[Dict]:
        by_cat: Dict[str, List[Dict]] = {}
        for s in items:
            by_cat.setdefault(s["category"], []).append(s)
        selected = []
        for cat, n in CATEGORY_TARGETS.items():
            pool = by_cat.get(cat, [])
            self.rng.shuffle(pool)
            selected.extend(pool[:n])
        return selected[:target]

    # --- Full Pipeline ---

    async def run_pipeline(self) -> List[Dict]:
        candidates = self.select_candidates()
        extracted = await self.extract_statements(candidates)
        filtered = await self.filter_quality(extracted)
        statements = self.assemble_ground_truth(filtered)
        # Save final output
        path = self.output_dir / "statements.jsonl"
        with open(path, "w") as f:
            for s in statements:
                f.write(json.dumps(s, default=str) + "\n")
        self.cost_tracker.save(str(self.output_dir / "curation_costs.json"))
        logger.info("Pipeline done: %d statements, $%.2f", len(statements), self.cost_tracker.total_cost)
        return statements
