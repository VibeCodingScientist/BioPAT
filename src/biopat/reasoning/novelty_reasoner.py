"""LLM-based Novelty Reasoner for Patent Prior Art Analysis.

Performs the core novelty assessment task:
1. Maps prior art references to specific claim elements
2. Determines if each element is disclosed in prior art
3. Assesses anticipation (single reference) vs obviousness (combination)
4. Generates legal reasoning supporting the assessment

This is the "reasoning engine" that mimics expert patent attorney analysis.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from biopat.reasoning.claim_parser import ClaimElement, ParsedClaim

logger = logging.getLogger(__name__)


class NoveltyStatus(Enum):
    """Patent novelty assessment status."""
    NOVEL = "novel"                              # No anticipation or obviousness
    POTENTIALLY_NOVEL = "potentially_novel"      # Minor concerns, likely patentable
    LIKELY_OBVIOUS = "likely_obvious"            # Combination renders obvious
    OBVIOUS = "obvious"                          # Clearly obvious (103)
    LIKELY_ANTICIPATED = "likely_anticipated"    # Strong anticipation evidence
    ANTICIPATED = "anticipated"                  # Clearly anticipated (102)


class ThreatLevel(Enum):
    """How threatening a prior art reference is."""
    CRITICAL = "critical"      # Directly anticipates claim
    HIGH = "high"              # Strong obviousness contribution
    MODERATE = "moderate"      # Partial disclosure
    LOW = "low"                # Tangentially related
    NONE = "none"              # Not relevant


@dataclass
class ElementMapping:
    """Mapping of a claim element to prior art."""
    element_id: str
    element_text: str
    is_disclosed: bool                    # Is this element found in prior art?
    disclosing_references: List[str]      # Doc IDs that disclose this element
    disclosure_type: str                  # "explicit", "implicit", "obvious_variant"
    confidence: float                     # 0-1 confidence in assessment
    reasoning: str                        # Why we believe it's disclosed/not


@dataclass
class PriorArtMapping:
    """Complete mapping of prior art to a claim."""
    doc_id: str
    title: str
    threat_level: ThreatLevel
    disclosed_elements: List[str]         # Element IDs disclosed by this reference
    coverage_ratio: float                 # Fraction of claim elements disclosed
    key_disclosures: List[str]            # Most important disclosures
    missing_elements: List[str]           # Elements NOT found in this reference
    reasoning: str                        # Why this reference is relevant


@dataclass
class NoveltyAssessment:
    """Complete novelty assessment for a claim."""
    claim_number: int
    claim_text: str
    status: NoveltyStatus
    confidence: float                     # 0-1 confidence in assessment

    # Element-level analysis
    element_mappings: List[ElementMapping]
    elements_disclosed: int               # Count of elements found in prior art
    elements_total: int

    # Reference-level analysis
    prior_art_mappings: List[PriorArtMapping]
    anticipating_references: List[str]    # References that anticipate (single ref)
    obviousness_combinations: List[List[str]]  # Reference combinations for obviousness

    # Legal analysis
    primary_rejection_type: str           # "102" (anticipation) or "103" (obviousness)
    rejection_basis: str                  # Summary of rejection basis
    detailed_reasoning: str               # Full legal reasoning

    # Recommendations
    claim_amendments: List[str]           # Suggested amendments to overcome
    arguments: List[str]                  # Potential arguments against rejection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_number": self.claim_number,
            "claim_text": self.claim_text[:500] + "..." if len(self.claim_text) > 500 else self.claim_text,
            "status": self.status.value,
            "confidence": self.confidence,
            "elements_disclosed": self.elements_disclosed,
            "elements_total": self.elements_total,
            "coverage_ratio": self.elements_disclosed / self.elements_total if self.elements_total > 0 else 0,
            "anticipating_references": self.anticipating_references,
            "obviousness_combinations": self.obviousness_combinations,
            "primary_rejection_type": self.primary_rejection_type,
            "rejection_basis": self.rejection_basis,
            "claim_amendments": self.claim_amendments,
            "arguments": self.arguments,
        }


# LLM Prompts
ELEMENT_MAPPING_PROMPT = '''You are an expert patent attorney analyzing whether prior art discloses specific claim elements.

CLAIM ELEMENT:
"""
Element ID: {element_id}
Type: {element_type}
Text: {element_text}
Keywords: {keywords}
"""

PRIOR ART DOCUMENT:
"""
Doc ID: {doc_id}
Title: {doc_title}
Content: {doc_content}
"""

Analyze whether this prior art document discloses the claim element. Respond with JSON:

{{
    "is_disclosed": <true|false>,
    "disclosure_type": "<explicit|implicit|obvious_variant|not_disclosed>",
    "confidence": <0.0-1.0>,
    "matching_passages": ["<relevant passages from the prior art>"],
    "reasoning": "<detailed explanation of why the element is or isn't disclosed>"
}}

Consider:
- Explicit disclosure: Same concept with same/equivalent language
- Implicit disclosure: Not explicitly stated but inherent/necessarily present
- Obvious variant: Minor variation that would be obvious to skilled artisan

Respond with ONLY the JSON.'''


NOVELTY_ASSESSMENT_PROMPT = '''You are an expert patent examiner performing a novelty assessment.

PATENT CLAIM:
"""
Claim {claim_number}: {claim_text}
"""

CLAIM ELEMENTS:
{elements_summary}

ELEMENT DISCLOSURE ANALYSIS:
{element_disclosures}

PRIOR ART REFERENCES:
{prior_art_summary}

Based on this analysis, provide a complete novelty assessment. Respond with JSON:

{{
    "status": "<novel|potentially_novel|likely_obvious|obvious|likely_anticipated|anticipated>",
    "confidence": <0.0-1.0>,
    "primary_rejection_type": "<102|103|none>",
    "anticipating_references": ["<doc_ids that individually anticipate>"],
    "obviousness_combinations": [["<doc_id1>", "<doc_id2>"], ...],
    "rejection_basis": "<concise statement of rejection grounds>",
    "detailed_reasoning": "<thorough legal analysis explaining the assessment>",
    "claim_amendments": ["<suggested claim amendments to overcome prior art>"],
    "arguments": ["<potential arguments applicant could make>"]
}}

Apply proper patent law standards:
- Anticipation (102): Single reference must disclose EVERY element
- Obviousness (103): Combination of references + motivation to combine
- Consider differences that may support novelty/non-obviousness

Respond with ONLY the JSON.'''


class LLMNoveltyReasoner:
    """SOTA LLM-based novelty reasoner for patent analysis.

    Performs expert-level novelty assessment by:
    1. Mapping each claim element to prior art
    2. Determining disclosure status
    3. Assessing anticipation vs obviousness
    4. Generating legal reasoning

    Example:
        ```python
        reasoner = LLMNoveltyReasoner(provider="openai", model="gpt-4")

        # Given parsed claim and retrieved prior art
        assessment = reasoner.assess_novelty(
            parsed_claim=parsed_claim,
            prior_art_docs=retrieved_documents,
            corpus=full_corpus
        )

        print(f"Status: {assessment.status.value}")
        print(f"Basis: {assessment.rejection_basis}")
        ```
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        llm_provider: Optional[Any] = None,
    ):
        """Initialize the novelty reasoner.

        Args:
            provider: LLM provider name ("openai", "anthropic", "google")
            model: Model name
            api_key: API key
            temperature: Generation temperature
            llm_provider: Pre-configured LLMProvider instance (preferred)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._llm = llm_provider
        self._last_response = None

        if self._llm is None:
            self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]) -> None:
        """Initialize LLM client."""
        try:
            from biopat.llm import create_provider
            self._llm = create_provider(self.provider, model=self.model, api_key=api_key)
        except (ImportError, ValueError):
            if self.provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call LLM and return response."""
        try:
            if self._llm is not None:
                response = self._llm.generate(
                    prompt=prompt,
                    system_prompt="You are an expert patent examiner. Respond only with valid JSON.",
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                self._last_response = response
                return response.text
            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert patent examiner. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            else:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:-1]
            text = "\n".join(lines)
        return json.loads(text)

    def _get_doc_content(self, doc: Any) -> str:
        """Extract text content from document."""
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            title = doc.get("title", "")
            text = doc.get("text", doc.get("abstract", doc.get("content", "")))
            return f"{title}\n\n{text}"
        return str(doc)

    def map_element_to_prior_art(
        self,
        element: ClaimElement,
        doc_id: str,
        doc: Any,
    ) -> ElementMapping:
        """Map a single claim element to a prior art document.

        Args:
            element: The claim element to analyze
            doc_id: Document identifier
            doc: Document content

        Returns:
            ElementMapping with disclosure analysis
        """
        doc_content = self._get_doc_content(doc)
        doc_title = doc.get("title", "") if isinstance(doc, dict) else ""

        prompt = ELEMENT_MAPPING_PROMPT.format(
            element_id=element.element_id,
            element_type=element.element_type.value,
            element_text=element.text,
            keywords=", ".join(element.keywords[:10]),
            doc_id=doc_id,
            doc_title=doc_title,
            doc_content=doc_content[:3000],  # Limit content length
        )

        response = self._call_llm(prompt, max_tokens=1000)

        try:
            data = self._parse_json(response)
        except json.JSONDecodeError:
            # Fallback
            data = {
                "is_disclosed": False,
                "disclosure_type": "not_disclosed",
                "confidence": 0.5,
                "reasoning": "Unable to determine disclosure status",
            }

        return ElementMapping(
            element_id=element.element_id,
            element_text=element.text[:200],
            is_disclosed=data.get("is_disclosed", False),
            disclosing_references=[doc_id] if data.get("is_disclosed") else [],
            disclosure_type=data.get("disclosure_type", "not_disclosed"),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
        )

    def analyze_prior_art_reference(
        self,
        parsed_claim: ParsedClaim,
        doc_id: str,
        doc: Any,
        element_mappings: List[ElementMapping],
    ) -> PriorArtMapping:
        """Analyze how a prior art reference relates to the claim.

        Args:
            parsed_claim: The parsed patent claim
            doc_id: Document identifier
            doc: Document content
            element_mappings: Element mappings for this document

        Returns:
            PriorArtMapping with reference-level analysis
        """
        doc_title = doc.get("title", "") if isinstance(doc, dict) else ""

        # Calculate disclosed elements
        disclosed_elements = [
            em.element_id for em in element_mappings
            if em.is_disclosed and doc_id in em.disclosing_references
        ]

        coverage_ratio = len(disclosed_elements) / len(parsed_claim.elements) if parsed_claim.elements else 0

        missing_elements = [
            e.element_id for e in parsed_claim.elements
            if e.element_id not in disclosed_elements
        ]

        # Determine threat level
        if coverage_ratio >= 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif coverage_ratio >= 0.7:
            threat_level = ThreatLevel.HIGH
        elif coverage_ratio >= 0.4:
            threat_level = ThreatLevel.MODERATE
        elif coverage_ratio > 0:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE

        # Get key disclosures
        key_disclosures = []
        for em in element_mappings:
            if em.is_disclosed and doc_id in em.disclosing_references:
                key_disclosures.append(f"{em.element_id}: {em.element_text[:50]}...")

        # Build reasoning
        reasoning = f"This reference discloses {len(disclosed_elements)}/{len(parsed_claim.elements)} claim elements ({coverage_ratio:.0%} coverage)."
        if missing_elements:
            reasoning += f" Missing elements: {', '.join(missing_elements)}."

        return PriorArtMapping(
            doc_id=doc_id,
            title=doc_title[:100],
            threat_level=threat_level,
            disclosed_elements=disclosed_elements,
            coverage_ratio=coverage_ratio,
            key_disclosures=key_disclosures[:5],
            missing_elements=missing_elements,
            reasoning=reasoning,
        )

    def assess_novelty(
        self,
        parsed_claim: ParsedClaim,
        prior_art_docs: List[Tuple[str, float]],  # (doc_id, score) pairs
        corpus: Dict[str, Any],
        max_refs_to_analyze: int = 10,
    ) -> NoveltyAssessment:
        """Perform complete novelty assessment.

        Args:
            parsed_claim: Parsed patent claim
            prior_art_docs: Retrieved prior art (doc_id, score) pairs
            corpus: Full corpus for document lookup
            max_refs_to_analyze: Maximum references to analyze in detail

        Returns:
            Complete NoveltyAssessment
        """
        logger.info(f"Assessing novelty for claim {parsed_claim.claim_number}...")

        # Limit references to analyze
        docs_to_analyze = prior_art_docs[:max_refs_to_analyze]

        # Step 1: Map each element to each prior art reference
        all_element_mappings: Dict[str, List[ElementMapping]] = {}

        for element in parsed_claim.elements:
            element_mappings = []

            for doc_id, _ in docs_to_analyze:
                doc = corpus.get(doc_id, {})
                mapping = self.map_element_to_prior_art(element, doc_id, doc)
                element_mappings.append(mapping)

            all_element_mappings[element.element_id] = element_mappings

        # Step 2: Consolidate element mappings
        consolidated_mappings = []
        for element in parsed_claim.elements:
            mappings = all_element_mappings.get(element.element_id, [])

            # Find all references that disclose this element
            disclosing_refs = [m.disclosing_references[0] for m in mappings if m.is_disclosed and m.disclosing_references]

            is_disclosed = len(disclosing_refs) > 0
            best_mapping = max(mappings, key=lambda m: m.confidence) if mappings else None

            consolidated_mappings.append(ElementMapping(
                element_id=element.element_id,
                element_text=element.text[:200],
                is_disclosed=is_disclosed,
                disclosing_references=disclosing_refs,
                disclosure_type=best_mapping.disclosure_type if best_mapping else "not_disclosed",
                confidence=best_mapping.confidence if best_mapping else 0.5,
                reasoning=best_mapping.reasoning if best_mapping else "",
            ))

        # Step 3: Analyze each prior art reference
        prior_art_mappings = []
        for doc_id, _ in docs_to_analyze:
            doc = corpus.get(doc_id, {})

            # Get element mappings for this doc
            doc_element_mappings = []
            for element in parsed_claim.elements:
                mappings = all_element_mappings.get(element.element_id, [])
                for m in mappings:
                    if doc_id in m.disclosing_references or (m.is_disclosed and doc_id == mappings[0].disclosing_references[0] if mappings[0].disclosing_references else False):
                        doc_element_mappings.append(m)
                        break

            ref_mapping = self.analyze_prior_art_reference(
                parsed_claim, doc_id, doc, doc_element_mappings
            )
            prior_art_mappings.append(ref_mapping)

        # Sort by threat level
        prior_art_mappings.sort(key=lambda x: ["critical", "high", "moderate", "low", "none"].index(x.threat_level.value))

        # Step 4: Perform overall novelty assessment with LLM
        elements_disclosed = sum(1 for m in consolidated_mappings if m.is_disclosed)
        elements_total = len(parsed_claim.elements)

        # Build prompt for overall assessment
        elements_summary = "\n".join([
            f"- {e.element_id} ({e.element_type.value}): {e.text[:100]}..."
            for e in parsed_claim.elements
        ])

        element_disclosures = "\n".join([
            f"- {m.element_id}: {'DISCLOSED' if m.is_disclosed else 'NOT DISCLOSED'} "
            f"(refs: {', '.join(m.disclosing_references[:3]) if m.disclosing_references else 'none'})"
            for m in consolidated_mappings
        ])

        prior_art_summary = "\n".join([
            f"- {pm.doc_id} [{pm.threat_level.value.upper()}]: {pm.coverage_ratio:.0%} coverage, "
            f"discloses {', '.join(pm.disclosed_elements[:3])}"
            for pm in prior_art_mappings[:5]
        ])

        assessment_prompt = NOVELTY_ASSESSMENT_PROMPT.format(
            claim_number=parsed_claim.claim_number,
            claim_text=parsed_claim.claim_text[:1000],
            elements_summary=elements_summary,
            element_disclosures=element_disclosures,
            prior_art_summary=prior_art_summary,
        )

        response = self._call_llm(assessment_prompt, max_tokens=2000)

        try:
            assessment_data = self._parse_json(response)
        except json.JSONDecodeError:
            # Fallback assessment based on coverage
            max_coverage = max((pm.coverage_ratio for pm in prior_art_mappings), default=0)

            if max_coverage >= 0.95:
                status = NoveltyStatus.ANTICIPATED
            elif max_coverage >= 0.8:
                status = NoveltyStatus.LIKELY_ANTICIPATED
            elif max_coverage >= 0.6:
                status = NoveltyStatus.LIKELY_OBVIOUS
            else:
                status = NoveltyStatus.POTENTIALLY_NOVEL

            assessment_data = {
                "status": status.value,
                "confidence": 0.6,
                "primary_rejection_type": "102" if max_coverage >= 0.9 else "103",
                "anticipating_references": [],
                "obviousness_combinations": [],
                "rejection_basis": f"Based on {max_coverage:.0%} maximum element coverage",
                "detailed_reasoning": "Automated assessment based on element coverage analysis",
                "claim_amendments": [],
                "arguments": [],
            }

        # Build final assessment
        try:
            status = NoveltyStatus(assessment_data.get("status", "potentially_novel"))
        except ValueError:
            status = NoveltyStatus.POTENTIALLY_NOVEL

        return NoveltyAssessment(
            claim_number=parsed_claim.claim_number,
            claim_text=parsed_claim.claim_text,
            status=status,
            confidence=assessment_data.get("confidence", 0.5),
            element_mappings=consolidated_mappings,
            elements_disclosed=elements_disclosed,
            elements_total=elements_total,
            prior_art_mappings=prior_art_mappings,
            anticipating_references=assessment_data.get("anticipating_references", []),
            obviousness_combinations=assessment_data.get("obviousness_combinations", []),
            primary_rejection_type=assessment_data.get("primary_rejection_type", "none"),
            rejection_basis=assessment_data.get("rejection_basis", ""),
            detailed_reasoning=assessment_data.get("detailed_reasoning", ""),
            claim_amendments=assessment_data.get("claim_amendments", []),
            arguments=assessment_data.get("arguments", []),
        )


def create_novelty_reasoner(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMNoveltyReasoner:
    """Factory function for novelty reasoner.

    Args:
        provider: "openai" or "anthropic"
        model: Model name (defaults to gpt-4)
        api_key: API key

    Returns:
        Configured LLMNoveltyReasoner
    """
    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-opus-20240229"

    return LLMNoveltyReasoner(
        provider=provider,
        model=model,
        api_key=api_key,
    )
