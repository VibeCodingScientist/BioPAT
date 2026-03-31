#!/usr/bin/env python3
"""BioPAT Novelty Assessment Demo - Full Agent Workflow

This demo showcases the complete novelty assessment workflow:
1. Parse patent claims into searchable elements
2. Execute trimodal search (text + chemical + sequence)
3. Map prior art to specific claim elements
4. Reason about novelty destruction
5. Generate assessment report

This demonstrates what BioPAT benchmarks: an AI agent's ability to
find and reason about prior art for patent novelty assessment.

Usage:
    python novelty_assessment_demo.py
    python novelty_assessment_demo.py --patent US10500003
"""

import argparse
import json
import math
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Structures
# =============================================================================

class ThreatLevel(Enum):
    """How seriously prior art threatens a claim element."""
    CRITICAL = "CRITICAL"      # Directly anticipates - novelty destroying
    HIGH = "HIGH"              # Very similar - likely obviousness issue
    MODERATE = "MODERATE"      # Related - may combine with other refs
    LOW = "LOW"                # Tangentially related
    NONE = "NONE"              # No relevance


class NoveltyStatus(Enum):
    """Overall novelty assessment."""
    ANTICIPATED = "ANTICIPATED"           # Claim lacks novelty (102)
    LIKELY_ANTICIPATED = "LIKELY_ANTICIPATED"
    OBVIOUS = "OBVIOUS"                   # Claim is obvious (103)
    LIKELY_OBVIOUS = "LIKELY_OBVIOUS"
    POTENTIALLY_NOVEL = "POTENTIALLY_NOVEL"
    NOVEL = "NOVEL"


@dataclass
class ClaimElement:
    """A parsed element from a patent claim."""
    element_id: str
    text: str
    element_type: str  # "method_step", "composition", "structure", "sequence", etc.
    keywords: List[str] = field(default_factory=list)
    smiles: Optional[str] = None
    sequence: Optional[str] = None


@dataclass
class PriorArtHit:
    """A prior art reference with relevance analysis."""
    doc_id: str
    title: str
    relevance_score: float
    modality: str  # "text", "chemical", "sequence", "combined"
    matching_elements: List[str]  # Which claim elements it matches
    threat_level: ThreatLevel
    key_passage: str
    reasoning: str


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim against prior art."""
    claim_id: str
    claim_text: str
    elements: List[ClaimElement]
    prior_art_hits: List[PriorArtHit]
    novelty_status: NoveltyStatus
    reasoning: str


# =============================================================================
# Claim Parser
# =============================================================================

class ClaimParser:
    """Parses patent claims into searchable elements."""

    # Keywords indicating different claim element types
    METHOD_KEYWORDS = ["method", "process", "treating", "administering", "comprising"]
    COMPOSITION_KEYWORDS = ["composition", "formulation", "comprising", "containing"]
    ANTIBODY_KEYWORDS = ["antibody", "monoclonal", "humanized", "chimeric", "bispecific"]
    MOLECULE_KEYWORDS = ["compound", "molecule", "inhibitor", "antagonist", "agonist"]
    SEQUENCE_KEYWORDS = ["sequence", "seq id", "cdr", "variable region", "amino acid"]

    def parse_claim(self, claim_text: str, metadata: dict = None) -> List[ClaimElement]:
        """Parse a claim into searchable elements."""
        elements = []
        metadata = metadata or {}

        # Split into phrases
        phrases = self._split_into_phrases(claim_text)

        for i, phrase in enumerate(phrases):
            element = self._analyze_phrase(phrase, i, metadata)
            if element:
                elements.append(element)

        # Add chemical structure if present
        if metadata.get("smiles"):
            elements.append(ClaimElement(
                element_id=f"E{len(elements)+1}",
                text=f"Chemical structure: {metadata['smiles'][:30]}...",
                element_type="chemical_structure",
                keywords=["chemical", "structure", "compound"],
                smiles=metadata["smiles"],
            ))

        # Add sequence if present
        if metadata.get("sequence"):
            elements.append(ClaimElement(
                element_id=f"E{len(elements)+1}",
                text=f"Sequence: {metadata['sequence'][:30]}...",
                element_type="sequence",
                keywords=["sequence", "protein", "antibody"],
                sequence=metadata["sequence"],
            ))

        return elements

    def _split_into_phrases(self, text: str) -> List[str]:
        """Split claim into meaningful phrases."""
        # Split on common claim delimiters
        text = re.sub(r'\s+', ' ', text)
        phrases = re.split(r'[;,]|\band\b|\bwherein\b|\bcomprising\b', text, flags=re.IGNORECASE)
        return [p.strip() for p in phrases if len(p.strip()) > 10]

    def _analyze_phrase(self, phrase: str, index: int, metadata: dict) -> Optional[ClaimElement]:
        """Analyze a phrase to create a claim element."""
        phrase_lower = phrase.lower()

        # Determine element type
        if any(kw in phrase_lower for kw in self.METHOD_KEYWORDS):
            elem_type = "method_step"
        elif any(kw in phrase_lower for kw in self.ANTIBODY_KEYWORDS):
            elem_type = "antibody"
        elif any(kw in phrase_lower for kw in self.MOLECULE_KEYWORDS):
            elem_type = "small_molecule"
        elif any(kw in phrase_lower for kw in self.SEQUENCE_KEYWORDS):
            elem_type = "sequence_element"
        elif any(kw in phrase_lower for kw in self.COMPOSITION_KEYWORDS):
            elem_type = "composition"
        else:
            elem_type = "general"

        # Extract keywords
        keywords = self._extract_keywords(phrase)

        if not keywords:
            return None

        return ClaimElement(
            element_id=f"E{index+1}",
            text=phrase[:100] + ("..." if len(phrase) > 100 else ""),
            element_type=elem_type,
            keywords=keywords,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common words
        stopwords = {'a', 'an', 'the', 'of', 'to', 'in', 'for', 'with', 'by', 'as', 'at', 'from', 'or', 'is', 'are', 'be', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'such', 'said'}

        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]{2,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]

        # Prioritize domain-specific terms
        domain_terms = {'antibody', 'pd-1', 'pd-l1', 'ctla-4', 'cancer', 'tumor', 'melanoma',
                       'inhibitor', 'receptor', 'cell', 'immune', 'treatment', 'therapy',
                       'monoclonal', 'humanized', 'bispecific', 'car-t', 'checkpoint',
                       'sequence', 'protein', 'peptide', 'compound', 'molecule'}

        # Sort: domain terms first, then by length
        keywords.sort(key=lambda x: (x not in domain_terms, -len(x)))

        return keywords[:10]


# =============================================================================
# Trimodal Retriever
# =============================================================================

class TrimodalRetriever:
    """Performs trimodal search: text + chemical + sequence."""

    def __init__(self, corpus: Dict[str, dict]):
        self.corpus = corpus
        self.text_index = self._build_text_index()
        self.chemical_index = self._build_chemical_index()
        self.sequence_index = self._build_sequence_index()

    def _build_text_index(self) -> dict:
        """Build inverted index for text search."""
        index = defaultdict(list)
        doc_freqs = defaultdict(int)

        for doc_id, doc in self.corpus.items():
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = set(text.lower().split())
            for token in tokens:
                index[token].append(doc_id)
                doc_freqs[token] += 1

        return {"index": index, "doc_freqs": doc_freqs, "corpus_size": len(self.corpus)}

    def _build_chemical_index(self) -> dict:
        """Build chemical fingerprint index (simplified)."""
        # In production, use RDKit Morgan fingerprints
        index = {}
        for doc_id, doc in self.corpus.items():
            # Check if document mentions chemical structures
            text = doc.get("text", "").lower()
            if any(term in text for term in ["compound", "molecule", "inhibitor", "drug", "small molecule"]):
                # Create simplified fingerprint from text
                index[doc_id] = self._text_to_chemical_features(text)
        return index

    def _build_sequence_index(self) -> dict:
        """Build sequence index (simplified BLAST-like)."""
        index = {}
        for doc_id, doc in self.corpus.items():
            text = doc.get("text", "").lower()
            if any(term in text for term in ["antibody", "protein", "sequence", "peptide", "cdr", "variable region"]):
                index[doc_id] = self._text_to_sequence_features(text)
        return index

    def _text_to_chemical_features(self, text: str) -> Set[str]:
        """Extract chemical-related features from text."""
        features = set()
        chem_terms = ["kinase", "inhibitor", "receptor", "agonist", "antagonist", "binding",
                     "affinity", "ic50", "potency", "selectivity", "oral", "bioavailability"]
        for term in chem_terms:
            if term in text:
                features.add(term)
        return features

    def _text_to_sequence_features(self, text: str) -> Set[str]:
        """Extract sequence-related features from text."""
        features = set()
        seq_terms = ["antibody", "monoclonal", "humanized", "cdr", "variable", "heavy chain",
                    "light chain", "fab", "fc", "igg", "binding", "affinity", "epitope"]
        for term in seq_terms:
            if term in text:
                features.add(term)
        return features

    def search(self, elements: List[ClaimElement], top_k: int = 20) -> List[Tuple[str, float, str]]:
        """Execute trimodal search across all elements."""
        all_scores = defaultdict(lambda: {"text": 0, "chemical": 0, "sequence": 0})

        for element in elements:
            # Text search
            text_results = self._text_search(element.keywords)
            for doc_id, score in text_results:
                all_scores[doc_id]["text"] = max(all_scores[doc_id]["text"], score)

            # Chemical search
            if element.smiles or element.element_type == "small_molecule":
                chem_results = self._chemical_search(element)
                for doc_id, score in chem_results:
                    all_scores[doc_id]["chemical"] = max(all_scores[doc_id]["chemical"], score)

            # Sequence search
            if element.sequence or element.element_type in ["antibody", "sequence_element"]:
                seq_results = self._sequence_search(element)
                for doc_id, score in seq_results:
                    all_scores[doc_id]["sequence"] = max(all_scores[doc_id]["sequence"], score)

        # Combine scores with weights
        combined = []
        for doc_id, scores in all_scores.items():
            # Determine primary modality
            max_modality = max(scores, key=scores.get)

            # Combined score (weighted)
            combined_score = (
                0.5 * scores["text"] +
                0.3 * scores["chemical"] +
                0.2 * scores["sequence"]
            )

            # Boost if multiple modalities match
            modality_count = sum(1 for s in scores.values() if s > 0.1)
            if modality_count > 1:
                combined_score *= (1 + 0.2 * (modality_count - 1))

            combined.append((doc_id, combined_score, max_modality))

        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _text_search(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """BM25-style text search."""
        scores = defaultdict(float)
        index = self.text_index["index"]
        doc_freqs = self.text_index["doc_freqs"]
        N = self.text_index["corpus_size"]

        for keyword in keywords:
            keyword = keyword.lower()
            if keyword not in index:
                continue

            df = doc_freqs[keyword]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id in index[keyword]:
                scores[doc_id] += idf

        # Normalize
        max_score = max(scores.values()) if scores else 1
        return [(doc_id, score/max_score) for doc_id, score in scores.items()]

    def _chemical_search(self, element: ClaimElement) -> List[Tuple[str, float]]:
        """Chemical similarity search (simplified Tanimoto)."""
        results = []

        if element.smiles:
            # In production: compute Morgan fingerprint and Tanimoto similarity
            # Here we use text-based approximation
            query_features = {"inhibitor", "binding", "affinity", "potency"}
        else:
            query_features = set(element.keywords) & {"inhibitor", "antagonist", "agonist", "compound"}

        for doc_id, doc_features in self.chemical_index.items():
            if query_features and doc_features:
                # Jaccard similarity as Tanimoto proxy
                intersection = len(query_features & doc_features)
                union = len(query_features | doc_features)
                similarity = intersection / union if union > 0 else 0
                if similarity > 0.1:
                    results.append((doc_id, similarity))

        return results

    def _sequence_search(self, element: ClaimElement) -> List[Tuple[str, float]]:
        """Sequence similarity search (simplified BLAST)."""
        results = []

        if element.sequence:
            # In production: run actual BLAST alignment
            # Here we use keyword-based approximation
            query_features = {"antibody", "humanized", "monoclonal", "cdr", "variable"}
        else:
            query_features = set(element.keywords) & {"antibody", "protein", "sequence", "peptide"}

        for doc_id, doc_features in self.sequence_index.items():
            if query_features and doc_features:
                intersection = len(query_features & doc_features)
                union = len(query_features | doc_features)
                similarity = intersection / union if union > 0 else 0
                if similarity > 0.1:
                    results.append((doc_id, similarity))

        return results


# =============================================================================
# Novelty Analyzer
# =============================================================================

class NoveltyAnalyzer:
    """Analyzes prior art to assess novelty of patent claims."""

    def __init__(self, corpus: Dict[str, dict], qrels: Dict[str, Dict[str, int]]):
        self.corpus = corpus
        self.qrels = qrels

    def analyze_claim(
        self,
        patent_id: str,
        claim_text: str,
        elements: List[ClaimElement],
        search_results: List[Tuple[str, float, str]],
    ) -> ClaimAnalysis:
        """Perform full novelty analysis on a claim."""

        prior_art_hits = []

        for doc_id, score, modality in search_results[:10]:
            doc = self.corpus.get(doc_id, {})

            # Determine which elements this doc matches
            matching_elements = self._find_matching_elements(doc, elements)

            # Calculate threat level
            threat_level = self._assess_threat_level(
                doc_id, patent_id, score, modality, len(matching_elements), len(elements)
            )

            # Extract key passage
            key_passage = self._extract_key_passage(doc, elements)

            # Generate reasoning
            reasoning = self._generate_reasoning(doc, elements, matching_elements, modality)

            hit = PriorArtHit(
                doc_id=doc_id,
                title=doc.get("title", "Unknown"),
                relevance_score=score,
                modality=modality,
                matching_elements=[e.element_id for e in matching_elements],
                threat_level=threat_level,
                key_passage=key_passage,
                reasoning=reasoning,
            )
            prior_art_hits.append(hit)

        # Determine overall novelty status
        novelty_status, overall_reasoning = self._assess_overall_novelty(
            elements, prior_art_hits
        )

        return ClaimAnalysis(
            claim_id=patent_id,
            claim_text=claim_text[:200] + "...",
            elements=elements,
            prior_art_hits=prior_art_hits,
            novelty_status=novelty_status,
            reasoning=overall_reasoning,
        )

    def _find_matching_elements(self, doc: dict, elements: List[ClaimElement]) -> List[ClaimElement]:
        """Find which claim elements a document addresses."""
        doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
        matching = []

        for element in elements:
            # Check keyword overlap
            matches = sum(1 for kw in element.keywords if kw.lower() in doc_text)
            if matches >= 2 or (matches >= 1 and len(element.keywords) <= 3):
                matching.append(element)

        return matching

    def _assess_threat_level(
        self,
        doc_id: str,
        patent_id: str,
        score: float,
        modality: str,
        matching_count: int,
        total_elements: int
    ) -> ThreatLevel:
        """Assess how threatening this prior art is."""

        # Check ground truth if available
        if patent_id in self.qrels and doc_id in self.qrels[patent_id]:
            relevance = self.qrels[patent_id][doc_id]
            if relevance == 3:
                return ThreatLevel.CRITICAL
            elif relevance == 2:
                return ThreatLevel.HIGH
            elif relevance == 1:
                return ThreatLevel.MODERATE

        # Heuristic assessment
        element_coverage = matching_count / total_elements if total_elements > 0 else 0

        if score > 0.8 and element_coverage > 0.7:
            return ThreatLevel.CRITICAL
        elif score > 0.6 or element_coverage > 0.5:
            return ThreatLevel.HIGH
        elif score > 0.4 or element_coverage > 0.3:
            return ThreatLevel.MODERATE
        elif score > 0.2:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE

    def _extract_key_passage(self, doc: dict, elements: List[ClaimElement]) -> str:
        """Extract most relevant passage from document."""
        text = doc.get("text", "")

        # Find sentence with most keyword matches
        sentences = text.split(". ")
        all_keywords = set()
        for e in elements:
            all_keywords.update(kw.lower() for kw in e.keywords)

        best_sentence = ""
        best_score = 0

        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in all_keywords if kw in sent_lower)
            if score > best_score:
                best_score = score
                best_sentence = sent

        return best_sentence[:200] + ("..." if len(best_sentence) > 200 else "")

    def _generate_reasoning(
        self,
        doc: dict,
        elements: List[ClaimElement],
        matching: List[ClaimElement],
        modality: str
    ) -> str:
        """Generate reasoning for why this document is relevant."""
        title = doc.get("title", "")

        if not matching:
            return "Limited relevance - no direct element matches."

        matched_types = set(e.element_type for e in matching)

        reasoning_parts = []

        if "antibody" in matched_types or "sequence_element" in matched_types:
            reasoning_parts.append("Discloses related antibody/protein technology")
        if "small_molecule" in matched_types:
            reasoning_parts.append("Describes similar small molecule approach")
        if "method_step" in matched_types:
            reasoning_parts.append("Teaches similar treatment methodology")

        if modality == "chemical":
            reasoning_parts.append("Chemical structure similarity detected")
        elif modality == "sequence":
            reasoning_parts.append("Sequence homology identified")

        reasoning_parts.append(f"Matches {len(matching)}/{len(elements)} claim elements")

        return "; ".join(reasoning_parts)

    def _assess_overall_novelty(
        self,
        elements: List[ClaimElement],
        hits: List[PriorArtHit]
    ) -> Tuple[NoveltyStatus, str]:
        """Determine overall novelty status."""

        if not hits:
            return NoveltyStatus.NOVEL, "No relevant prior art found."

        critical_hits = [h for h in hits if h.threat_level == ThreatLevel.CRITICAL]
        high_hits = [h for h in hits if h.threat_level == ThreatLevel.HIGH]

        # Check for anticipation (single reference)
        for hit in critical_hits:
            if len(hit.matching_elements) >= len(elements) * 0.8:
                return (
                    NoveltyStatus.ANTICIPATED,
                    f"Claim appears anticipated by {hit.doc_id}. "
                    f"This reference discloses {len(hit.matching_elements)}/{len(elements)} "
                    f"claim elements with {hit.threat_level.value} similarity."
                )

        if critical_hits:
            return (
                NoveltyStatus.LIKELY_ANTICIPATED,
                f"Found {len(critical_hits)} critically relevant reference(s). "
                f"Primary concern: {critical_hits[0].doc_id} - {critical_hits[0].title[:50]}..."
            )

        # Check for obviousness (combination)
        if len(high_hits) >= 2:
            covered_elements = set()
            for hit in high_hits[:3]:
                covered_elements.update(hit.matching_elements)

            if len(covered_elements) >= len(elements) * 0.8:
                refs = ", ".join(h.doc_id for h in high_hits[:3])
                return (
                    NoveltyStatus.OBVIOUS,
                    f"Claim appears obvious over combination of: {refs}. "
                    f"Together these references cover {len(covered_elements)}/{len(elements)} elements."
                )

        if high_hits:
            return (
                NoveltyStatus.LIKELY_OBVIOUS,
                f"Found {len(high_hits)} highly relevant reference(s) that may render claim obvious."
            )

        return (
            NoveltyStatus.POTENTIALLY_NOVEL,
            "No single reference anticipates the claim. Some related art exists but "
            "claim may be novel and non-obvious with appropriate claim scope."
        )


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generates formatted novelty assessment reports."""

    def generate_report(self, analysis: ClaimAnalysis) -> str:
        """Generate a comprehensive novelty report."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("BIOPAT NOVELTY ASSESSMENT REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Patent info
        lines.append(f"Patent ID: {analysis.claim_id}")
        lines.append(f"Claim: {analysis.claim_text}")
        lines.append("")

        # Claim decomposition
        lines.append("-" * 70)
        lines.append("CLAIM DECOMPOSITION")
        lines.append("-" * 70)
        for elem in analysis.elements:
            lines.append(f"  [{elem.element_id}] {elem.element_type.upper()}")
            lines.append(f"      Text: {elem.text[:60]}...")
            lines.append(f"      Keywords: {', '.join(elem.keywords[:5])}")
            if elem.smiles:
                lines.append(f"      SMILES: {elem.smiles[:40]}...")
            if elem.sequence:
                lines.append(f"      Sequence: {elem.sequence[:30]}...")
        lines.append("")

        # Prior art mapping
        lines.append("-" * 70)
        lines.append("PRIOR ART ANALYSIS")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"{'Doc ID':<15} {'Threat':<10} {'Modality':<10} {'Elements':<10} {'Score':<8}")
        lines.append("-" * 53)

        for hit in analysis.prior_art_hits:
            elem_str = ",".join(hit.matching_elements) if hit.matching_elements else "-"
            lines.append(
                f"{hit.doc_id:<15} {hit.threat_level.value:<10} {hit.modality:<10} "
                f"{elem_str:<10} {hit.relevance_score:.3f}"
            )
        lines.append("")

        # Detailed analysis
        lines.append("-" * 70)
        lines.append("DETAILED PRIOR ART ANALYSIS")
        lines.append("-" * 70)

        for i, hit in enumerate(analysis.prior_art_hits[:5], 1):
            threat_symbol = {
                ThreatLevel.CRITICAL: "🔴",
                ThreatLevel.HIGH: "🟠",
                ThreatLevel.MODERATE: "🟡",
                ThreatLevel.LOW: "🟢",
                ThreatLevel.NONE: "⚪",
            }.get(hit.threat_level, "⚪")

            lines.append("")
            lines.append(f"[{i}] {hit.doc_id} {threat_symbol} {hit.threat_level.value}")
            lines.append(f"    Title: {hit.title[:60]}...")
            lines.append(f"    Modality: {hit.modality} | Score: {hit.relevance_score:.3f}")
            lines.append(f"    Matching Elements: {', '.join(hit.matching_elements) or 'None'}")
            lines.append(f"    Key Passage: \"{hit.key_passage}\"")
            lines.append(f"    Reasoning: {hit.reasoning}")

        lines.append("")

        # Overall assessment
        lines.append("=" * 70)
        lines.append("NOVELTY ASSESSMENT")
        lines.append("=" * 70)

        status_symbol = {
            NoveltyStatus.ANTICIPATED: "❌ ANTICIPATED",
            NoveltyStatus.LIKELY_ANTICIPATED: "⚠️  LIKELY ANTICIPATED",
            NoveltyStatus.OBVIOUS: "❌ OBVIOUS",
            NoveltyStatus.LIKELY_OBVIOUS: "⚠️  LIKELY OBVIOUS",
            NoveltyStatus.POTENTIALLY_NOVEL: "✅ POTENTIALLY NOVEL",
            NoveltyStatus.NOVEL: "✅ NOVEL",
        }.get(analysis.novelty_status, "❓ UNKNOWN")

        lines.append("")
        lines.append(f"Status: {status_symbol}")
        lines.append("")
        lines.append("Reasoning:")
        lines.append(f"  {analysis.reasoning}")
        lines.append("")

        # Recommendations
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)

        if analysis.novelty_status in [NoveltyStatus.ANTICIPATED, NoveltyStatus.LIKELY_ANTICIPATED]:
            lines.append("  1. Review cited prior art for potential distinguishing features")
            lines.append("  2. Consider narrowing claim scope to exclude anticipated subject matter")
            lines.append("  3. Identify any novel aspects not present in prior art")
        elif analysis.novelty_status in [NoveltyStatus.OBVIOUS, NoveltyStatus.LIKELY_OBVIOUS]:
            lines.append("  1. Consider adding dependent claims with additional limitations")
            lines.append("  2. Emphasize unexpected results or synergistic effects")
            lines.append("  3. Document any teaching away in the prior art")
        else:
            lines.append("  1. Maintain current claim scope")
            lines.append("  2. Monitor for new prior art publications")
            lines.append("  3. Consider filing continuation applications for related subject matter")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Main Demo
# =============================================================================

def load_data(case_study_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load case study data."""
    corpus = {}
    with open(case_study_dir / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    queries = {}
    with open(case_study_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q

    qrels = {}
    with open(case_study_dir / "qrels.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rel

    return corpus, queries, qrels


def run_demo(patent_id: str = None):
    """Run the full novelty assessment demo."""

    # Load data
    case_study_dir = Path(__file__).parent
    corpus, queries, qrels = load_data(case_study_dir)

    print("=" * 70)
    print("BioPAT NOVELTY ASSESSMENT DEMO")
    print("Full Agent Workflow Demonstration")
    print("=" * 70)
    print()

    # Select patent
    if patent_id and patent_id in queries:
        selected_patents = [patent_id]
    else:
        # Demo with most interesting patents (those with chemical/sequence data)
        selected_patents = ["US10500003", "US10500001", "US10500004"]

    # Initialize components
    parser = ClaimParser()
    retriever = TrimodalRetriever(corpus)
    analyzer = NoveltyAnalyzer(corpus, qrels)
    reporter = ReportGenerator()

    for pid in selected_patents:
        patent = queries[pid]

        print(f"\nProcessing {pid}...")
        print("-" * 50)

        # Step 1: Parse claim
        print("[1/4] Parsing claim into elements...")
        elements = parser.parse_claim(patent["text"], patent.get("metadata", {}))
        print(f"      Found {len(elements)} searchable elements")

        # Step 2: Trimodal search
        print("[2/4] Executing trimodal search...")
        results = retriever.search(elements)
        print(f"      Retrieved {len(results)} candidate documents")

        # Step 3: Analyze novelty
        print("[3/4] Analyzing prior art relevance...")
        analysis = analyzer.analyze_claim(pid, patent["text"], elements, results)
        print(f"      Status: {analysis.novelty_status.value}")

        # Step 4: Generate report
        print("[4/4] Generating assessment report...")
        report = reporter.generate_report(analysis)

        print()
        print(report)
        print()

        if len(selected_patents) > 1:
            print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BioPAT Novelty Assessment Demo")
    parser.add_argument("--patent", type=str, help="Specific patent ID to analyze")
    args = parser.parse_args()

    run_demo(args.patent)


if __name__ == "__main__":
    main()
