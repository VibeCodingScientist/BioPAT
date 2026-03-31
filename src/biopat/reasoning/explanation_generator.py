"""Explanation Generator for Patent Novelty Reports.

Generates human-readable, legally-grounded explanations of novelty assessments.
Produces reports suitable for patent attorneys, inventors, and examiners.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from biopat.reasoning.claim_parser import ParsedClaim
from biopat.reasoning.novelty_reasoner import (
    NoveltyAssessment,
    NoveltyStatus,
    PriorArtMapping,
    ThreatLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class NoveltyReport:
    """Complete novelty assessment report."""

    # Metadata
    patent_id: str
    report_date: str
    report_version: str = "1.0"

    # Summary
    executive_summary: str = ""
    overall_status: NoveltyStatus = NoveltyStatus.POTENTIALLY_NOVEL
    confidence_score: float = 0.5

    # Claim analysis
    claims_analyzed: int = 0
    claims_anticipated: int = 0
    claims_obvious: int = 0
    claims_novel: int = 0

    # Detailed assessments
    claim_assessments: List[NoveltyAssessment] = field(default_factory=list)

    # Key findings
    critical_prior_art: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # Full report text
    full_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patent_id": self.patent_id,
            "report_date": self.report_date,
            "report_version": self.report_version,
            "executive_summary": self.executive_summary,
            "overall_status": self.overall_status.value,
            "confidence_score": self.confidence_score,
            "claims_analyzed": self.claims_analyzed,
            "claims_anticipated": self.claims_anticipated,
            "claims_obvious": self.claims_obvious,
            "claims_novel": self.claims_novel,
            "critical_prior_art": self.critical_prior_art,
            "recommended_actions": self.recommended_actions,
            "claim_assessments": [a.to_dict() for a in self.claim_assessments],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Report templates
EXECUTIVE_SUMMARY_TEMPLATE = """
EXECUTIVE SUMMARY
=================

Patent: {patent_id}
Assessment Date: {date}
Status: {status}
Confidence: {confidence:.0%}

Claims Analyzed: {total_claims}
- Anticipated (102): {anticipated}
- Obvious (103): {obvious}
- Novel: {novel}

{key_finding}

Top Prior Art Concerns:
{top_concerns}
"""

CLAIM_ANALYSIS_TEMPLATE = """
CLAIM {claim_number} ANALYSIS
{'=' * 50}

Claim Text:
{claim_text}

Assessment: {status}
Confidence: {confidence:.0%}

Element Coverage:
{element_coverage}

Most Threatening Prior Art:
{threatening_refs}

Legal Basis:
{legal_basis}

Recommendations:
{recommendations}
"""

PRIOR_ART_TEMPLATE = """
Reference: {doc_id}
Title: {title}
Threat Level: {threat_level}
Element Coverage: {coverage:.0%}

Disclosed Elements:
{disclosed}

Missing Elements:
{missing}

Reasoning:
{reasoning}
"""


class ExplanationGenerator:
    """Generates human-readable novelty assessment reports.

    Creates comprehensive reports that include:
    - Executive summary
    - Claim-by-claim analysis
    - Prior art mapping details
    - Legal reasoning
    - Recommendations

    Example:
        ```python
        generator = ExplanationGenerator()

        # Generate report from assessments
        report = generator.generate_report(
            patent_id="US10500001",
            claim_assessments=[assessment1, assessment2],
            parsed_claims=[claim1, claim2]
        )

        print(report.full_report)
        report.save("novelty_report.json")
        ```
    """

    def __init__(
        self,
        include_full_text: bool = True,
        include_legal_citations: bool = True,
        verbose: bool = True,
    ):
        """Initialize explanation generator.

        Args:
            include_full_text: Include full claim text in report
            include_legal_citations: Include 35 USC citations
            verbose: Include detailed reasoning
        """
        self.include_full_text = include_full_text
        self.include_legal_citations = include_legal_citations
        self.verbose = verbose

    def _format_status(self, status: NoveltyStatus) -> str:
        """Format status with emoji/symbol."""
        status_formats = {
            NoveltyStatus.NOVEL: "✅ NOVEL",
            NoveltyStatus.POTENTIALLY_NOVEL: "✅ POTENTIALLY NOVEL",
            NoveltyStatus.LIKELY_OBVIOUS: "⚠️ LIKELY OBVIOUS",
            NoveltyStatus.OBVIOUS: "❌ OBVIOUS",
            NoveltyStatus.LIKELY_ANTICIPATED: "⚠️ LIKELY ANTICIPATED",
            NoveltyStatus.ANTICIPATED: "❌ ANTICIPATED",
        }
        return status_formats.get(status, status.value.upper())

    def _format_threat_level(self, level: ThreatLevel) -> str:
        """Format threat level with emoji."""
        formats = {
            ThreatLevel.CRITICAL: "🔴 CRITICAL",
            ThreatLevel.HIGH: "🟠 HIGH",
            ThreatLevel.MODERATE: "🟡 MODERATE",
            ThreatLevel.LOW: "🟢 LOW",
            ThreatLevel.NONE: "⚪ NONE",
        }
        return formats.get(level, level.value.upper())

    def _generate_executive_summary(
        self,
        patent_id: str,
        assessments: List[NoveltyAssessment],
    ) -> str:
        """Generate executive summary."""

        # Count claim statuses
        anticipated = sum(1 for a in assessments if a.status in [NoveltyStatus.ANTICIPATED, NoveltyStatus.LIKELY_ANTICIPATED])
        obvious = sum(1 for a in assessments if a.status in [NoveltyStatus.OBVIOUS, NoveltyStatus.LIKELY_OBVIOUS])
        novel = sum(1 for a in assessments if a.status in [NoveltyStatus.NOVEL, NoveltyStatus.POTENTIALLY_NOVEL])

        # Determine overall status
        if anticipated > 0:
            overall_status = NoveltyStatus.LIKELY_ANTICIPATED
        elif obvious > len(assessments) / 2:
            overall_status = NoveltyStatus.LIKELY_OBVIOUS
        elif novel == len(assessments):
            overall_status = NoveltyStatus.NOVEL
        else:
            overall_status = NoveltyStatus.POTENTIALLY_NOVEL

        # Calculate average confidence
        avg_confidence = sum(a.confidence for a in assessments) / len(assessments) if assessments else 0

        # Key finding
        if anticipated > 0:
            key_finding = f"⚠️ ALERT: {anticipated} claim(s) may lack novelty under 35 U.S.C. § 102."
        elif obvious > 0:
            key_finding = f"⚠️ CONCERN: {obvious} claim(s) may be obvious under 35 U.S.C. § 103."
        else:
            key_finding = "✅ No critical novelty or obviousness issues identified."

        # Top concerns
        all_mappings = []
        for a in assessments:
            for pm in a.prior_art_mappings:
                all_mappings.append((pm, a.claim_number))

        all_mappings.sort(key=lambda x: ["critical", "high", "moderate", "low", "none"].index(x[0].threat_level.value))

        top_concerns = []
        for pm, claim_num in all_mappings[:3]:
            top_concerns.append(
                f"  • {pm.doc_id} ({self._format_threat_level(pm.threat_level)}): "
                f"Affects Claim {claim_num}, {pm.coverage_ratio:.0%} element coverage"
            )

        return EXECUTIVE_SUMMARY_TEMPLATE.format(
            patent_id=patent_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            status=self._format_status(overall_status),
            confidence=avg_confidence,
            total_claims=len(assessments),
            anticipated=anticipated,
            obvious=obvious,
            novel=novel,
            key_finding=key_finding,
            top_concerns="\n".join(top_concerns) if top_concerns else "  None identified",
        )

    def _generate_claim_analysis(
        self,
        assessment: NoveltyAssessment,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> str:
        """Generate detailed claim analysis."""

        # Element coverage
        element_lines = []
        for em in assessment.element_mappings:
            status = "✅ Disclosed" if em.is_disclosed else "❌ Not found"
            refs = ", ".join(em.disclosing_references[:2]) if em.disclosing_references else "N/A"
            element_lines.append(f"  {em.element_id}: {status} (Refs: {refs})")

        # Threatening references
        threat_lines = []
        for pm in assessment.prior_art_mappings[:3]:
            threat_lines.append(
                f"  • {pm.doc_id} [{self._format_threat_level(pm.threat_level)}]: "
                f"{pm.coverage_ratio:.0%} coverage"
            )

        # Legal basis
        if assessment.primary_rejection_type == "102":
            legal_basis = (
                "Under 35 U.S.C. § 102, a claim is anticipated if a single prior art "
                "reference discloses every element of the claim. "
                f"References {', '.join(assessment.anticipating_references[:2])} "
                "appear to anticipate this claim."
            )
        elif assessment.primary_rejection_type == "103":
            legal_basis = (
                "Under 35 U.S.C. § 103, a claim is obvious if the differences between "
                "the claim and prior art would have been obvious to one skilled in the art. "
                f"Combination of references would render this claim obvious."
            )
        else:
            legal_basis = "No clear rejection basis identified."

        # Recommendations
        recommendations = []
        if assessment.claim_amendments:
            recommendations.extend([f"  • Amendment: {a}" for a in assessment.claim_amendments[:3]])
        if assessment.arguments:
            recommendations.extend([f"  • Argument: {a}" for a in assessment.arguments[:3]])
        if not recommendations:
            recommendations = ["  • No specific recommendations"]

        return CLAIM_ANALYSIS_TEMPLATE.format(
            claim_number=assessment.claim_number,
            claim_text=assessment.claim_text[:500] + "..." if len(assessment.claim_text) > 500 else assessment.claim_text,
            status=self._format_status(assessment.status),
            confidence=assessment.confidence,
            element_coverage="\n".join(element_lines),
            threatening_refs="\n".join(threat_lines) if threat_lines else "  None identified",
            legal_basis=legal_basis,
            recommendations="\n".join(recommendations),
        )

    def _generate_prior_art_section(
        self,
        assessments: List[NoveltyAssessment],
    ) -> str:
        """Generate prior art reference section."""

        # Collect all unique prior art references
        all_refs: Dict[str, PriorArtMapping] = {}
        for a in assessments:
            for pm in a.prior_art_mappings:
                if pm.doc_id not in all_refs or pm.threat_level.value < all_refs[pm.doc_id].threat_level.value:
                    all_refs[pm.doc_id] = pm

        # Sort by threat level
        sorted_refs = sorted(
            all_refs.values(),
            key=lambda x: ["critical", "high", "moderate", "low", "none"].index(x.threat_level.value)
        )

        lines = ["\nPRIOR ART REFERENCE DETAILS", "=" * 50, ""]

        for pm in sorted_refs[:10]:  # Top 10 references
            disclosed = "\n".join([f"    • {e}" for e in pm.disclosed_elements[:5]])
            missing = "\n".join([f"    • {e}" for e in pm.missing_elements[:5]])

            lines.append(PRIOR_ART_TEMPLATE.format(
                doc_id=pm.doc_id,
                title=pm.title,
                threat_level=self._format_threat_level(pm.threat_level),
                coverage=pm.coverage_ratio,
                disclosed=disclosed if disclosed else "    None",
                missing=missing if missing else "    None",
                reasoning=pm.reasoning,
            ))
            lines.append("-" * 50)

        return "\n".join(lines)

    def generate_report(
        self,
        patent_id: str,
        claim_assessments: List[NoveltyAssessment],
        parsed_claims: Optional[List[ParsedClaim]] = None,
    ) -> NoveltyReport:
        """Generate complete novelty report.

        Args:
            patent_id: Patent identifier
            claim_assessments: List of novelty assessments
            parsed_claims: Optional parsed claims for additional detail

        Returns:
            NoveltyReport with full analysis
        """
        logger.info(f"Generating novelty report for {patent_id}...")

        # Create report object
        report = NoveltyReport(
            patent_id=patent_id,
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            claims_analyzed=len(claim_assessments),
            claim_assessments=claim_assessments,
        )

        # Count statuses
        report.claims_anticipated = sum(
            1 for a in claim_assessments
            if a.status in [NoveltyStatus.ANTICIPATED, NoveltyStatus.LIKELY_ANTICIPATED]
        )
        report.claims_obvious = sum(
            1 for a in claim_assessments
            if a.status in [NoveltyStatus.OBVIOUS, NoveltyStatus.LIKELY_OBVIOUS]
        )
        report.claims_novel = sum(
            1 for a in claim_assessments
            if a.status in [NoveltyStatus.NOVEL, NoveltyStatus.POTENTIALLY_NOVEL]
        )

        # Determine overall status
        if report.claims_anticipated > 0:
            report.overall_status = NoveltyStatus.LIKELY_ANTICIPATED
        elif report.claims_obvious > report.claims_analyzed / 2:
            report.overall_status = NoveltyStatus.LIKELY_OBVIOUS
        elif report.claims_novel == report.claims_analyzed:
            report.overall_status = NoveltyStatus.NOVEL
        else:
            report.overall_status = NoveltyStatus.POTENTIALLY_NOVEL

        # Calculate confidence
        report.confidence_score = (
            sum(a.confidence for a in claim_assessments) / len(claim_assessments)
            if claim_assessments else 0.5
        )

        # Collect critical prior art
        for a in claim_assessments:
            for pm in a.prior_art_mappings:
                if pm.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                    report.critical_prior_art.append({
                        "doc_id": pm.doc_id,
                        "title": pm.title,
                        "threat_level": pm.threat_level.value,
                        "affects_claims": [a.claim_number],
                        "coverage": pm.coverage_ratio,
                    })

        # Generate recommendations
        if report.claims_anticipated > 0:
            report.recommended_actions.append(
                "Review anticipating references and consider claim amendments"
            )
        if report.claims_obvious > 0:
            report.recommended_actions.append(
                "Prepare arguments for unexpected results or teaching away"
            )
        if report.claims_novel == report.claims_analyzed:
            report.recommended_actions.append(
                "Claims appear allowable; monitor for new prior art"
            )

        # Build full report text
        report_parts = []

        # Header
        report_parts.append("=" * 70)
        report_parts.append("PATENT NOVELTY ASSESSMENT REPORT")
        report_parts.append("BioPAT Prior Art Analysis System")
        report_parts.append("=" * 70)

        # Executive summary
        report.executive_summary = self._generate_executive_summary(patent_id, claim_assessments)
        report_parts.append(report.executive_summary)

        # Claim-by-claim analysis
        report_parts.append("\nDETAILED CLAIM ANALYSIS")
        report_parts.append("=" * 50)

        for i, assessment in enumerate(claim_assessments):
            parsed = parsed_claims[i] if parsed_claims and i < len(parsed_claims) else None
            report_parts.append(self._generate_claim_analysis(assessment, parsed))

        # Prior art details
        report_parts.append(self._generate_prior_art_section(claim_assessments))

        # Recommendations
        report_parts.append("\nRECOMMENDATIONS")
        report_parts.append("=" * 50)
        for rec in report.recommended_actions:
            report_parts.append(f"  • {rec}")

        # Footer
        report_parts.append("\n" + "=" * 70)
        report_parts.append("END OF REPORT")
        report_parts.append("=" * 70)

        report.full_report = "\n".join(report_parts)

        return report

    def save_report(
        self,
        report: NoveltyReport,
        path: str,
        format: str = "json",  # "json", "txt", "md"
    ) -> None:
        """Save report to file.

        Args:
            report: NoveltyReport to save
            path: Output file path
            format: Output format ("json", "txt", "md")
        """
        if format == "json":
            with open(path, "w") as f:
                f.write(report.to_json())
        elif format == "txt":
            with open(path, "w") as f:
                f.write(report.full_report)
        elif format == "md":
            # Convert to markdown
            md_content = report.full_report.replace("=" * 70, "---")
            md_content = md_content.replace("=" * 50, "###")
            with open(path, "w") as f:
                f.write(md_content)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Report saved to {path}")


def create_explanation_generator(verbose: bool = True) -> ExplanationGenerator:
    """Factory function for explanation generator."""
    return ExplanationGenerator(verbose=verbose)
