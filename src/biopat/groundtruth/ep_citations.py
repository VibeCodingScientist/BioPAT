"""EP Search Report Citation Parser.

Phase 6 (v3.0): Parses European Patent Office search report citations
to build ground truth relevance labels for EP and WO patents.

EPO Search Report Citation Categories:
- X: Particularly relevant when taken alone (anticipates novelty or inventive step)
- Y: Particularly relevant when combined with another document
- A: Background art, general state of the art
- D: Document cited in the application
- E: Earlier patent document but published after filing date
- P: Intermediate document (published between priority and filing dates)
- T: Theory or principle underlying the invention
- L: Document cited for other reasons
- O: Non-written disclosure (oral, use, exhibition)

Category-to-Relevance Mapping (aligned with USPTO 102/103 mapping):
- X (novelty-destroying) -> 3 (single-document 102/103)
- Y (combination relevance) -> 2 (multi-document 103)
- A (background) -> 1 (general relevance)
- D, E, P, T, L, O -> 1 (general relevance)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from lxml import etree
import polars as pl

from biopat.processing.patent_ids import (
    normalize_patent_id,
    ParsedPatentId,
    Jurisdiction,
)

logger = logging.getLogger(__name__)


class EPCitationCategory(str, Enum):
    """EPO Search Report citation categories."""
    X = "X"  # Particularly relevant alone
    Y = "Y"  # Relevant in combination
    A = "A"  # Background art
    D = "D"  # Cited in application
    E = "E"  # Earlier document, later published
    P = "P"  # Intermediate document
    T = "T"  # Theory/principle
    L = "L"  # Other reasons
    O = "O"  # Non-written disclosure
    UNKNOWN = "UNKNOWN"


# Map EP categories to relevance scores (0-3 scale)
EP_CATEGORY_TO_RELEVANCE: Dict[EPCitationCategory, int] = {
    EPCitationCategory.X: 3,  # Single-document anticipation -> highest relevance
    EPCitationCategory.Y: 2,  # Combination relevance -> multi-doc 103
    EPCitationCategory.A: 1,  # Background -> general relevance
    EPCitationCategory.D: 1,
    EPCitationCategory.E: 1,
    EPCitationCategory.P: 1,
    EPCitationCategory.T: 1,
    EPCitationCategory.L: 1,
    EPCitationCategory.O: 1,
    EPCitationCategory.UNKNOWN: 1,
}


@dataclass
class EPCitation:
    """Parsed citation from an EP search report."""

    cited_document: ParsedPatentId
    category: EPCitationCategory
    relevance_score: int
    relevant_claims: List[int] = field(default_factory=list)
    citation_text: str = ""
    is_patent: bool = True
    npl_reference: str = ""  # For non-patent literature

    @classmethod
    def from_patent_number(
        cls,
        patent_number: str,
        category: str,
        claims: Optional[List[int]] = None,
    ) -> "EPCitation":
        """Create citation from patent number and category.

        Args:
            patent_number: Patent document number.
            category: EP citation category (X, Y, A, etc.).
            claims: List of relevant claim numbers.

        Returns:
            EPCitation instance.
        """
        parsed = normalize_patent_id(patent_number)
        cat = EPCitationCategory(category.upper()) if category.upper() in [c.value for c in EPCitationCategory] else EPCitationCategory.UNKNOWN
        rel_score = EP_CATEGORY_TO_RELEVANCE.get(cat, 1)

        return cls(
            cited_document=parsed,
            category=cat,
            relevance_score=rel_score,
            relevant_claims=claims or [],
            is_patent=True,
        )

    @classmethod
    def from_npl(
        cls,
        reference_text: str,
        category: str,
        claims: Optional[List[int]] = None,
    ) -> "EPCitation":
        """Create citation from NPL reference.

        Args:
            reference_text: NPL citation text.
            category: EP citation category.
            claims: List of relevant claim numbers.

        Returns:
            EPCitation instance.
        """
        cat = EPCitationCategory(category.upper()) if category.upper() in [c.value for c in EPCitationCategory] else EPCitationCategory.UNKNOWN
        rel_score = EP_CATEGORY_TO_RELEVANCE.get(cat, 1)

        return cls(
            cited_document=ParsedPatentId(Jurisdiction.UNKNOWN, "", original=reference_text),
            category=cat,
            relevance_score=rel_score,
            relevant_claims=claims or [],
            is_patent=False,
            npl_reference=reference_text,
        )


@dataclass
class EPSearchReport:
    """Parsed EP search report with citations."""

    application_number: str
    publication_number: str
    search_report_date: str
    citations: List[EPCitation] = field(default_factory=list)
    examiner_name: str = ""

    @property
    def x_citations(self) -> List[EPCitation]:
        """Get X-category (novelty) citations."""
        return [c for c in self.citations if c.category == EPCitationCategory.X]

    @property
    def y_citations(self) -> List[EPCitation]:
        """Get Y-category (combination) citations."""
        return [c for c in self.citations if c.category == EPCitationCategory.Y]

    @property
    def a_citations(self) -> List[EPCitation]:
        """Get A-category (background) citations."""
        return [c for c in self.citations if c.category == EPCitationCategory.A]

    @property
    def patent_citations(self) -> List[EPCitation]:
        """Get patent citations only."""
        return [c for c in self.citations if c.is_patent]

    @property
    def npl_citations(self) -> List[EPCitation]:
        """Get NPL citations only."""
        return [c for c in self.citations if not c.is_patent]


class EPSearchReportParser:
    """Parser for EP search report data.

    Supports parsing from EPO OPS API responses (JSON/XML) and
    extracting ground truth relevance labels.
    """

    def __init__(self):
        """Initialize parser."""
        self._namespaces = {
            "ops": "http://ops.epo.org",
            "epo": "http://www.epo.org/exchange",
            "ep": "http://www.epo.org/ep",
        }

    def parse_from_json(self, response: Dict[str, Any]) -> Optional[EPSearchReport]:
        """Parse search report from EPO OPS JSON response.

        Args:
            response: Raw JSON response from EPO OPS API.

        Returns:
            Parsed EPSearchReport or None if parsing fails.
        """
        try:
            # Navigate EPO response structure
            world_data = response.get("ops:world-patent-data", {})
            register_data = world_data.get("ops:register-search", {})
            results = register_data.get("reg:register-documents", {})

            if not results:
                return None

            # Get first document
            docs = results.get("reg:register-document", [])
            if not isinstance(docs, list):
                docs = [docs]

            if not docs:
                return None

            doc = docs[0]

            # Extract basic info
            bib_data = doc.get("reg:bibliographic-data", {})
            app_ref = bib_data.get("reg:application-reference", {})
            doc_id = app_ref.get("document-id", {})

            app_number = doc_id.get("doc-number", {}).get("$", "")
            pub_number = self._extract_publication_number(bib_data)

            # Extract search report info
            search_data = doc.get("reg:search-report-data", {})
            sr_date = search_data.get("reg:search-report-date", {}).get("$", "")

            # Parse citations
            citations = self._parse_citations_json(search_data)

            return EPSearchReport(
                application_number=app_number,
                publication_number=pub_number,
                search_report_date=sr_date,
                citations=citations,
            )

        except Exception as e:
            logger.error(f"Failed to parse EP search report JSON: {e}")
            return None

    def parse_from_xml(self, xml_text: str) -> Optional[EPSearchReport]:
        """Parse search report from EPO XML response.

        Args:
            xml_text: Raw XML response from EPO.

        Returns:
            Parsed EPSearchReport or None if parsing fails.
        """
        try:
            root = etree.fromstring(xml_text.encode())

            # Find search report element
            sr = root.find(".//ep:search-report", self._namespaces)
            if sr is None:
                # Try alternative path
                sr = root.find(".//ops:search-report", self._namespaces)

            if sr is None:
                return None

            # Extract application info
            app_ref = sr.find(".//ep:application-reference", self._namespaces)
            app_number = ""
            pub_number = ""

            if app_ref is not None:
                doc_id = app_ref.find(".//epo:document-id", self._namespaces)
                if doc_id is not None:
                    country = doc_id.findtext("epo:country", "", self._namespaces)
                    doc_num = doc_id.findtext("epo:doc-number", "", self._namespaces)
                    app_number = f"{country}{doc_num}"

            # Extract publication number
            pub_ref = sr.find(".//ep:publication-reference", self._namespaces)
            if pub_ref is not None:
                doc_id = pub_ref.find(".//epo:document-id", self._namespaces)
                if doc_id is not None:
                    country = doc_id.findtext("epo:country", "", self._namespaces)
                    doc_num = doc_id.findtext("epo:doc-number", "", self._namespaces)
                    kind = doc_id.findtext("epo:kind", "", self._namespaces)
                    pub_number = f"{country}{doc_num}{kind}"

            # Extract date
            sr_date = sr.get("date", "") or sr.findtext(".//ep:date", "", self._namespaces)

            # Parse citations
            citations = self._parse_citations_xml(sr)

            return EPSearchReport(
                application_number=app_number,
                publication_number=pub_number,
                search_report_date=sr_date,
                citations=citations,
            )

        except Exception as e:
            logger.error(f"Failed to parse EP search report XML: {e}")
            return None

    def _extract_publication_number(self, bib_data: Dict) -> str:
        """Extract publication number from bibliographic data."""
        pub_ref = bib_data.get("reg:publication-reference", {})
        doc_id = pub_ref.get("document-id", {})

        country = doc_id.get("country", {}).get("$", "")
        doc_num = doc_id.get("doc-number", {}).get("$", "")
        kind = doc_id.get("kind", {}).get("$", "")

        return f"{country}{doc_num}{kind}"

    def _parse_citations_json(self, search_data: Dict) -> List[EPCitation]:
        """Parse citations from search report JSON data."""
        citations = []

        citation_list = search_data.get("reg:citation", [])
        if not isinstance(citation_list, list):
            citation_list = [citation_list]

        for cit in citation_list:
            try:
                category = cit.get("@category", "A")

                # Check for patent citation
                pat_cit = cit.get("reg:patcit", {})
                if pat_cit:
                    doc_id = pat_cit.get("document-id", {})
                    country = doc_id.get("country", {}).get("$", "")
                    doc_num = doc_id.get("doc-number", {}).get("$", "")
                    kind = doc_id.get("kind", {}).get("$", "")

                    patent_num = f"{country}{doc_num}{kind}"

                    # Extract relevant claims
                    claims = self._parse_claims(cit.get("reg:rel-claims", ""))

                    citations.append(
                        EPCitation.from_patent_number(patent_num, category, claims)
                    )

                # Check for NPL citation
                npl_cit = cit.get("reg:nplcit", {})
                if npl_cit:
                    text = npl_cit.get("text", {}).get("$", "")
                    claims = self._parse_claims(cit.get("reg:rel-claims", ""))

                    citations.append(
                        EPCitation.from_npl(text, category, claims)
                    )

            except Exception as e:
                logger.warning(f"Failed to parse citation: {e}")
                continue

        return citations

    def _parse_citations_xml(self, search_report: etree._Element) -> List[EPCitation]:
        """Parse citations from search report XML element."""
        citations = []

        for cit in search_report.findall(".//ep:citation", self._namespaces):
            try:
                category = cit.get("category", "A")

                # Check for patent citation
                pat_cit = cit.find(".//ep:patcit", self._namespaces)
                if pat_cit is not None:
                    doc_id = pat_cit.find(".//epo:document-id", self._namespaces)
                    if doc_id is not None:
                        country = doc_id.findtext("epo:country", "", self._namespaces)
                        doc_num = doc_id.findtext("epo:doc-number", "", self._namespaces)
                        kind = doc_id.findtext("epo:kind", "", self._namespaces)

                        patent_num = f"{country}{doc_num}{kind}"

                        # Extract relevant claims
                        rel_claims = cit.findtext(".//ep:rel-claims", "", self._namespaces)
                        claims = self._parse_claims(rel_claims)

                        citations.append(
                            EPCitation.from_patent_number(patent_num, category, claims)
                        )

                # Check for NPL citation
                npl_cit = cit.find(".//ep:nplcit", self._namespaces)
                if npl_cit is not None:
                    text = npl_cit.findtext(".//ep:text", "", self._namespaces)
                    rel_claims = cit.findtext(".//ep:rel-claims", "", self._namespaces)
                    claims = self._parse_claims(rel_claims)

                    citations.append(
                        EPCitation.from_npl(text, category, claims)
                    )

            except Exception as e:
                logger.warning(f"Failed to parse XML citation: {e}")
                continue

        return citations

    def _parse_claims(self, claims_str: str) -> List[int]:
        """Parse relevant claims string to list of integers.

        Args:
            claims_str: Claims string like "1-5,7,9-11" or "1,2,3".

        Returns:
            List of claim numbers.
        """
        if not claims_str:
            return []

        claims = []
        parts = claims_str.replace(" ", "").split(",")

        for part in parts:
            if "-" in part:
                # Range like "1-5"
                try:
                    start, end = part.split("-")
                    claims.extend(range(int(start), int(end) + 1))
                except ValueError:
                    continue
            else:
                # Single claim
                try:
                    claims.append(int(part))
                except ValueError:
                    continue

        return sorted(set(claims))

    def create_qrels_from_search_report(
        self,
        search_report: EPSearchReport,
        query_id: str,
    ) -> List[Tuple[str, str, int]]:
        """Create qrels entries from parsed search report.

        Args:
            search_report: Parsed EP search report.
            query_id: Query ID for the qrels (typically the EP publication number).

        Returns:
            List of (query_id, doc_id, relevance) tuples.
        """
        qrels = []

        for citation in search_report.patent_citations:
            doc_id = citation.cited_document.canonical
            qrels.append((query_id, doc_id, citation.relevance_score))

        return qrels

    def search_reports_to_dataframe(
        self,
        search_reports: List[EPSearchReport],
    ) -> pl.DataFrame:
        """Convert multiple search reports to a DataFrame.

        Args:
            search_reports: List of parsed search reports.

        Returns:
            DataFrame with columns: query_patent, cited_document, category,
            relevance_score, relevant_claims, is_patent.
        """
        records = []

        for sr in search_reports:
            for cit in sr.citations:
                records.append({
                    "query_patent": sr.publication_number,
                    "cited_document": cit.cited_document.full if cit.is_patent else cit.npl_reference,
                    "cited_document_normalized": cit.cited_document.canonical if cit.is_patent else "",
                    "jurisdiction": cit.cited_document.jurisdiction.value if cit.is_patent else "NPL",
                    "category": cit.category.value,
                    "relevance_score": cit.relevance_score,
                    "relevant_claims": cit.relevant_claims,
                    "is_patent": cit.is_patent,
                })

        return pl.DataFrame(records)


def map_ep_category_to_relevance(category: str) -> int:
    """Map EP citation category to relevance score.

    Args:
        category: EP citation category (X, Y, A, etc.).

    Returns:
        Relevance score (0-3).
    """
    try:
        cat = EPCitationCategory(category.upper())
    except ValueError:
        cat = EPCitationCategory.UNKNOWN

    return EP_CATEGORY_TO_RELEVANCE.get(cat, 1)


def combine_us_ep_relevance(us_score: int, ep_score: int) -> int:
    """Combine US and EP relevance scores.

    When the same document is cited by both USPTO and EPO,
    use the maximum relevance score.

    Args:
        us_score: USPTO-based relevance score.
        ep_score: EPO-based relevance score.

    Returns:
        Combined relevance score.
    """
    return max(us_score, ep_score)
