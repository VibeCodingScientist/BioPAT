"""Non-Patent Literature citation parser.

Parses NPL citations from Office Actions and links them
to papers in OpenAlex/PubMed.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ParsedCitation:
    """Parsed NPL citation structure."""
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    raw_text: str = ""
    parse_method: str = "none"


class NPLParser:
    """Parser for NPL citations from Office Actions."""

    # Regex patterns for extracting identifiers
    PMID_PATTERNS = [
        r"PMID[:\s]*(\d{7,8})",
        r"PubMed[:\s]*(\d{7,8})",
        r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})",
    ]

    DOI_PATTERNS = [
        r"(?:DOI|doi)[:\s]*(10\.\d{4,}/[^\s,;]+)",
        r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s,;]+)",
        r"(10\.\d{4,}/[^\s,;\"']+)",
    ]

    # Pattern for standard bibliographic format
    # e.g., "Smith et al., J Med Chem, 2015, 58, 1234-1240"
    BIBLIO_PATTERN = re.compile(
        r"([A-Z][a-z]+(?:\s+(?:et\s+al\.?|[A-Z][a-z]+))?)"  # Authors
        r"[,.]?\s*"
        r"([^,]+?)"  # Journal
        r"[,.]?\s*"
        r"(\d{4})"  # Year
        r"[,.]?\s*"
        r"(\d+)?"  # Volume
        r"[,.]?\s*"
        r"([\d-]+)?",  # Pages
        re.IGNORECASE
    )

    def __init__(self):
        self._pmid_patterns = [re.compile(p, re.IGNORECASE) for p in self.PMID_PATTERNS]
        self._doi_patterns = [re.compile(p, re.IGNORECASE) for p in self.DOI_PATTERNS]

    def extract_pmid(self, text: str) -> Optional[str]:
        """Extract PubMed ID from citation text.

        Args:
            text: Citation text.

        Returns:
            PMID if found, None otherwise.
        """
        for pattern in self._pmid_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from citation text.

        Args:
            text: Citation text.

        Returns:
            DOI if found, None otherwise.
        """
        for pattern in self._doi_patterns:
            match = pattern.search(text)
            if match:
                doi = match.group(1)
                # Clean up DOI (remove trailing punctuation)
                doi = re.sub(r"[.,;)\]]+$", "", doi)
                return doi
        return None

    def extract_year(self, text: str) -> Optional[int]:
        """Extract publication year from citation text.

        Args:
            text: Citation text.

        Returns:
            Year if found, None otherwise.
        """
        # Look for 4-digit years between 1900 and 2030
        year_pattern = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")
        match = year_pattern.search(text)
        if match:
            return int(match.group(1))
        return None

    def extract_title(self, text: str) -> Optional[str]:
        """Extract title from citation text.

        Attempts to find quoted text or text between author and journal.

        Args:
            text: Citation text.

        Returns:
            Title if found, None otherwise.
        """
        # Try quoted title first
        quoted = re.search(r'"([^"]{20,})"', text)
        if quoted:
            return quoted.group(1).strip()

        # Try title in italics or special formatting (often appears after period)
        # Pattern: Author(s). Title. Journal...
        title_match = re.search(
            r"[A-Z][a-z]+(?:\s+et\s+al\.?)?\.\s+([A-Z][^.]{15,}\.)",
            text
        )
        if title_match:
            return title_match.group(1).strip()

        return None

    def extract_authors(self, text: str) -> Optional[List[str]]:
        """Extract author names from citation text.

        Args:
            text: Citation text.

        Returns:
            List of author names if found, None otherwise.
        """
        # Pattern for "Smith et al." or "Smith, Jones, and Brown"
        et_al = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s+et\s+al\.?", text)
        if et_al:
            return [et_al.group(1)]

        # Pattern for comma-separated authors
        author_section = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z]\.?)?)(?:,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*", text)
        if author_section:
            authors_text = author_section.group(0)
            authors = [a.strip() for a in re.split(r",\s*(?:and\s+)?", authors_text)]
            return [a for a in authors if a]

        return None

    def parse_citation(self, text: str) -> ParsedCitation:
        """Parse a single NPL citation.

        Args:
            text: Raw citation text.

        Returns:
            ParsedCitation with extracted fields.
        """
        citation = ParsedCitation(raw_text=text)

        # Try to extract PMID (highest confidence)
        pmid = self.extract_pmid(text)
        if pmid:
            citation.pmid = pmid
            citation.parse_method = "pmid"

        # Try to extract DOI
        doi = self.extract_doi(text)
        if doi:
            citation.doi = doi
            if citation.parse_method == "none":
                citation.parse_method = "doi"

        # Extract other fields
        citation.year = self.extract_year(text)
        citation.title = self.extract_title(text)
        citation.authors = self.extract_authors(text)

        # If no identifier found, mark as bibliographic parse
        if citation.parse_method == "none" and (citation.title or citation.authors):
            citation.parse_method = "bibliographic"

        return citation

    def parse_citations_df(self, citations_df: pl.DataFrame, text_col: str = "cite_npl_str") -> pl.DataFrame:
        """Parse all citations in a DataFrame.

        Args:
            citations_df: DataFrame with NPL citation text.
            text_col: Name of column containing citation text.

        Returns:
            DataFrame with parsed citation fields added.
        """
        parsed_data = []

        for row in citations_df.iter_rows(named=True):
            text = row.get(text_col, "") or ""
            parsed = self.parse_citation(text)

            parsed_data.append({
                "pmid": parsed.pmid,
                "doi": parsed.doi,
                "title": parsed.title,
                "year": parsed.year,
                "parse_method": parsed.parse_method,
            })

        parsed_df = pl.DataFrame(parsed_data)

        # Concatenate with original data
        result = pl.concat([citations_df, parsed_df], how="horizontal")

        # Log parsing statistics
        total = len(result)
        pmid_count = result.filter(pl.col("pmid").is_not_null()).height
        doi_count = result.filter(pl.col("doi").is_not_null()).height
        title_count = result.filter(pl.col("title").is_not_null()).height

        logger.info(
            f"Parsed {total} citations: "
            f"{pmid_count} with PMID ({pmid_count/total*100:.1f}%), "
            f"{doi_count} with DOI ({doi_count/total*100:.1f}%), "
            f"{title_count} with title ({title_count/total*100:.1f}%)"
        )

        return result


class NPLLinker:
    """Links parsed NPL citations to papers in the corpus."""

    def __init__(self, papers_df: pl.DataFrame):
        """Initialize linker with papers corpus.

        Args:
            papers_df: Papers DataFrame with paper_id, pmid, doi, title.
        """
        self.papers_df = papers_df
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for fast matching."""
        # PMID index
        self.pmid_to_paper = {}
        pmid_df = self.papers_df.filter(pl.col("pmid").is_not_null())
        for row in pmid_df.select(["paper_id", "pmid"]).iter_rows(named=True):
            self.pmid_to_paper[str(row["pmid"])] = row["paper_id"]

        # DOI index (normalized)
        self.doi_to_paper = {}
        doi_df = self.papers_df.filter(pl.col("doi").is_not_null())
        for row in doi_df.select(["paper_id", "doi"]).iter_rows(named=True):
            doi = self._normalize_doi(str(row["doi"]))
            if doi:
                self.doi_to_paper[doi] = row["paper_id"]

        # Title index (normalized for fuzzy matching)
        self.title_to_paper = {}
        title_df = self.papers_df.filter(pl.col("title").is_not_null())
        for row in title_df.select(["paper_id", "title"]).iter_rows(named=True):
            title = self._normalize_title(str(row["title"]))
            if title:
                self.title_to_paper[title] = row["paper_id"]

        logger.info(
            f"Built indices: {len(self.pmid_to_paper)} PMIDs, "
            f"{len(self.doi_to_paper)} DOIs, "
            f"{len(self.title_to_paper)} titles"
        )

    def _normalize_doi(self, doi: str) -> Optional[str]:
        """Normalize DOI for matching."""
        if not doi:
            return None
        # Remove URL prefix, lowercase
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.lower())
        return doi.strip()

    def _normalize_title(self, title: str) -> Optional[str]:
        """Normalize title for matching."""
        if not title:
            return None
        # Lowercase, remove punctuation, collapse whitespace
        title = title.lower()
        title = re.sub(r"[^\w\s]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title if len(title) > 10 else None

    def link_citation(self, parsed: ParsedCitation) -> Optional[str]:
        """Link a parsed citation to a paper.

        Args:
            parsed: ParsedCitation object.

        Returns:
            paper_id if linked, None otherwise.
        """
        # Try PMID first (highest confidence)
        if parsed.pmid and parsed.pmid in self.pmid_to_paper:
            return self.pmid_to_paper[parsed.pmid]

        # Try DOI
        if parsed.doi:
            norm_doi = self._normalize_doi(parsed.doi)
            if norm_doi and norm_doi in self.doi_to_paper:
                return self.doi_to_paper[norm_doi]

        # Try title matching
        if parsed.title:
            norm_title = self._normalize_title(parsed.title)
            if norm_title and norm_title in self.title_to_paper:
                return self.title_to_paper[norm_title]

        return None

    def link_citations_df(
        self,
        parsed_citations_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Link all parsed citations to papers.

        Args:
            parsed_citations_df: DataFrame with parsed citation fields.

        Returns:
            DataFrame with paper_id column added.
        """
        linked_paper_ids = []
        link_methods = []

        for row in parsed_citations_df.iter_rows(named=True):
            paper_id = None
            link_method = "none"

            # Try PMID
            pmid = row.get("pmid")
            if pmid and str(pmid) in self.pmid_to_paper:
                paper_id = self.pmid_to_paper[str(pmid)]
                link_method = "pmid"

            # Try DOI
            if paper_id is None:
                doi = row.get("doi")
                if doi:
                    norm_doi = self._normalize_doi(str(doi))
                    if norm_doi and norm_doi in self.doi_to_paper:
                        paper_id = self.doi_to_paper[norm_doi]
                        link_method = "doi"

            # Try title
            if paper_id is None:
                title = row.get("title")
                if title:
                    norm_title = self._normalize_title(str(title))
                    if norm_title and norm_title in self.title_to_paper:
                        paper_id = self.title_to_paper[norm_title]
                        link_method = "title"

            linked_paper_ids.append(paper_id)
            link_methods.append(link_method)

        # Add columns
        result = parsed_citations_df.with_columns([
            pl.Series("linked_paper_id", linked_paper_ids),
            pl.Series("link_method", link_methods),
        ])

        # Log linking statistics
        total = len(result)
        linked = result.filter(pl.col("linked_paper_id").is_not_null()).height
        by_pmid = result.filter(pl.col("link_method") == "pmid").height
        by_doi = result.filter(pl.col("link_method") == "doi").height
        by_title = result.filter(pl.col("link_method") == "title").height

        logger.info(
            f"Linked {linked}/{total} citations ({linked/total*100:.1f}%): "
            f"{by_pmid} by PMID, {by_doi} by DOI, {by_title} by title"
        )

        return result
