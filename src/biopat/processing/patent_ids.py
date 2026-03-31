"""International Patent ID Normalization.

Phase 6 (v3.0): Provides utilities for normalizing and standardizing patent
identifiers across multiple jurisdictions (US, EP, WO, JP, CN, etc.).

Patent number formats vary significantly between jurisdictions:
- US: US1234567, US 1,234,567, US1234567A1
- EP: EP1234567, EP 1234567 A1, EP1234567B1
- WO: WO2020/123456, WO2020123456A1, WO 2020 123456
- JP: JP2020-123456, JP2020123456A
- CN: CN112345678A, CN 1123456789 A

This module provides canonical normalization for consistent comparison and lookup.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class Jurisdiction(str, Enum):
    """Known patent jurisdictions."""
    US = "US"  # United States
    EP = "EP"  # European Patent Office
    WO = "WO"  # WIPO PCT
    JP = "JP"  # Japan
    CN = "CN"  # China
    KR = "KR"  # South Korea
    GB = "GB"  # United Kingdom
    DE = "DE"  # Germany
    FR = "FR"  # France
    CA = "CA"  # Canada
    AU = "AU"  # Australia
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedPatentId:
    """Parsed and normalized patent identifier."""

    jurisdiction: Jurisdiction
    doc_number: str
    kind_code: Optional[str] = None
    original: str = ""

    @property
    def canonical(self) -> str:
        """Return canonical normalized form (jurisdiction + doc_number)."""
        return f"{self.jurisdiction.value}{self.doc_number}"

    @property
    def full(self) -> str:
        """Return full form with kind code if available."""
        if self.kind_code:
            return f"{self.jurisdiction.value}{self.doc_number}{self.kind_code}"
        return self.canonical

    def __str__(self) -> str:
        return self.full

    def __hash__(self) -> int:
        return hash(self.canonical)

    def __eq__(self, other) -> bool:
        if isinstance(other, ParsedPatentId):
            return self.canonical == other.canonical
        return False


# Jurisdiction detection patterns
JURISDICTION_PATTERNS = [
    (r"^(US)\s*", Jurisdiction.US),
    (r"^(EP)\s*", Jurisdiction.EP),
    (r"^(WO)\s*", Jurisdiction.WO),
    (r"^(JP)\s*", Jurisdiction.JP),
    (r"^(CN)\s*", Jurisdiction.CN),
    (r"^(KR)\s*", Jurisdiction.KR),
    (r"^(GB)\s*", Jurisdiction.GB),
    (r"^(DE)\s*", Jurisdiction.DE),
    (r"^(FR)\s*", Jurisdiction.FR),
    (r"^(CA)\s*", Jurisdiction.CA),
    (r"^(AU)\s*", Jurisdiction.AU),
]

# Kind code patterns (A1, B1, B2, etc.)
KIND_CODE_PATTERN = r"([A-Z][0-9]?)$"


def normalize_patent_id(patent_id: str) -> ParsedPatentId:
    """Normalize a patent identifier to canonical form.

    Handles various formats:
    - "US 1,234,567" -> US1234567
    - "EP 1234567 A1" -> EP1234567A1
    - "WO 2020/123456" -> WO2020123456
    - "US2020/0123456A1" -> US20200123456A1

    Args:
        patent_id: Raw patent identifier string.

    Returns:
        ParsedPatentId with normalized components.
    """
    if not patent_id:
        return ParsedPatentId(
            jurisdiction=Jurisdiction.UNKNOWN,
            doc_number="",
            original=patent_id or "",
        )

    original = patent_id

    # Clean up: remove extra whitespace, convert to uppercase
    cleaned = patent_id.strip().upper()

    # Detect jurisdiction
    jurisdiction = Jurisdiction.UNKNOWN
    for pattern, jur in JURISDICTION_PATTERNS:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            jurisdiction = jur
            cleaned = cleaned[match.end():]
            break

    # Remove common separators: spaces, commas, slashes, hyphens in numbers
    cleaned = re.sub(r"[\s,/\-]", "", cleaned)

    # Extract kind code if present
    kind_code = None
    kind_match = re.search(KIND_CODE_PATTERN, cleaned)
    if kind_match:
        kind_code = kind_match.group(1)
        cleaned = cleaned[:kind_match.start()]

    # The remaining string is the document number
    doc_number = cleaned

    # Additional jurisdiction-specific normalization
    if jurisdiction == Jurisdiction.US:
        # US patents: strip leading zeros from non-application numbers
        # Application numbers are typically 8+ digits with leading year
        if len(doc_number) <= 7:
            doc_number = doc_number.lstrip("0") or "0"

    elif jurisdiction == Jurisdiction.WO:
        # WO numbers: ensure year prefix is preserved
        # Format: WO + year (4 digits) + serial (6 digits)
        if len(doc_number) >= 10:
            # Looks like a proper WO number
            pass
        elif len(doc_number) == 6:
            # Just serial number, missing year - can't normalize fully
            logger.warning(f"WO patent {original} missing year prefix")

    return ParsedPatentId(
        jurisdiction=jurisdiction,
        doc_number=doc_number,
        kind_code=kind_code,
        original=original,
    )


def normalize_patent_ids(patent_ids: List[str]) -> List[ParsedPatentId]:
    """Normalize multiple patent identifiers.

    Args:
        patent_ids: List of raw patent ID strings.

    Returns:
        List of ParsedPatentId objects.
    """
    return [normalize_patent_id(pid) for pid in patent_ids]


def extract_jurisdiction(patent_id: str) -> Jurisdiction:
    """Extract jurisdiction from a patent identifier.

    Args:
        patent_id: Patent identifier string.

    Returns:
        Detected Jurisdiction enum value.
    """
    return normalize_patent_id(patent_id).jurisdiction


def are_same_patent(id1: str, id2: str) -> bool:
    """Check if two patent IDs refer to the same document (ignoring kind code).

    Args:
        id1: First patent identifier.
        id2: Second patent identifier.

    Returns:
        True if both IDs normalize to the same patent.
    """
    parsed1 = normalize_patent_id(id1)
    parsed2 = normalize_patent_id(id2)
    return parsed1.canonical == parsed2.canonical


def get_patent_family_key(patent_id: str) -> str:
    """Generate a family key for grouping related patents.

    Note: This is a simplified key based on normalized ID.
    True family relationships require data from DOCDB or similar.

    Args:
        patent_id: Patent identifier string.

    Returns:
        Normalized canonical key for the patent.
    """
    return normalize_patent_id(patent_id).canonical


def format_patent_id(
    patent_id: str,
    style: str = "compact",
) -> str:
    """Format a patent ID in a specified style.

    Styles:
    - "compact": US1234567A1
    - "spaced": US 1234567 A1
    - "full": US Patent No. 1,234,567 A1

    Args:
        patent_id: Raw patent identifier.
        style: Formatting style.

    Returns:
        Formatted patent ID string.
    """
    parsed = normalize_patent_id(patent_id)

    if style == "compact":
        return parsed.full

    elif style == "spaced":
        parts = [parsed.jurisdiction.value, parsed.doc_number]
        if parsed.kind_code:
            parts.append(parsed.kind_code)
        return " ".join(parts)

    elif style == "full":
        # Format with commas for US-style numbers
        doc_num = parsed.doc_number
        if parsed.jurisdiction == Jurisdiction.US and doc_num.isdigit():
            doc_num = f"{int(doc_num):,}"

        result = f"{parsed.jurisdiction.value} {doc_num}"
        if parsed.kind_code:
            result += f" {parsed.kind_code}"
        return result

    return parsed.full


def classify_document_type(patent_id: str) -> str:
    """Classify a patent document by its kind code.

    Common classifications:
    - A/A1/A2: Patent Application
    - B/B1/B2: Granted Patent
    - C: Correction
    - S: Design Patent (US)

    Args:
        patent_id: Patent identifier.

    Returns:
        Document type string: "application", "grant", "other".
    """
    parsed = normalize_patent_id(patent_id)
    kind = parsed.kind_code or ""

    if kind.startswith("A"):
        return "application"
    elif kind.startswith("B"):
        return "grant"
    elif kind.startswith("C"):
        return "correction"
    elif kind.startswith("S"):
        return "design"
    else:
        return "other"


def deduplicate_patent_ids(patent_ids: List[str]) -> List[str]:
    """Remove duplicate patent IDs based on canonical form.

    Keeps the first occurrence of each unique patent.

    Args:
        patent_ids: List of patent ID strings (may contain duplicates).

    Returns:
        Deduplicated list preserving order.
    """
    seen = set()
    result = []

    for pid in patent_ids:
        parsed = normalize_patent_id(pid)
        if parsed.canonical not in seen:
            seen.add(parsed.canonical)
            result.append(pid)

    return result


def group_by_jurisdiction(
    patent_ids: List[str],
) -> dict[Jurisdiction, List[ParsedPatentId]]:
    """Group patent IDs by jurisdiction.

    Args:
        patent_ids: List of patent ID strings.

    Returns:
        Dict mapping Jurisdiction to list of ParsedPatentIds.
    """
    groups: dict[Jurisdiction, List[ParsedPatentId]] = {}

    for pid in patent_ids:
        parsed = normalize_patent_id(pid)
        if parsed.jurisdiction not in groups:
            groups[parsed.jurisdiction] = []
        groups[parsed.jurisdiction].append(parsed)

    return groups


def validate_patent_id(patent_id: str) -> Tuple[bool, Optional[str]]:
    """Validate a patent identifier format.

    Args:
        patent_id: Patent identifier to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not patent_id or not patent_id.strip():
        return False, "Empty patent ID"

    parsed = normalize_patent_id(patent_id)

    if parsed.jurisdiction == Jurisdiction.UNKNOWN:
        return False, f"Unknown jurisdiction in: {patent_id}"

    if not parsed.doc_number:
        return False, f"Missing document number in: {patent_id}"

    # Check minimum document number length
    if len(parsed.doc_number) < 4:
        return False, f"Document number too short: {patent_id}"

    # Jurisdiction-specific validation
    if parsed.jurisdiction == Jurisdiction.US:
        # US patents should be 7-11 digits
        if not parsed.doc_number.isdigit():
            return False, f"US patent number should be numeric: {patent_id}"
        if not (4 <= len(parsed.doc_number) <= 11):
            return False, f"US patent number length invalid: {patent_id}"

    elif parsed.jurisdiction == Jurisdiction.EP:
        # EP patents: typically 7 digits
        if not parsed.doc_number.isdigit():
            return False, f"EP patent number should be numeric: {patent_id}"

    elif parsed.jurisdiction == Jurisdiction.WO:
        # WO: year (4) + serial (typically 6)
        if not parsed.doc_number.isdigit():
            return False, f"WO patent number should be numeric: {patent_id}"

    return True, None


class PatentIdNormalizer:
    """Stateful normalizer with caching and batch operations."""

    def __init__(self):
        """Initialize normalizer with empty cache."""
        self._cache: dict[str, ParsedPatentId] = {}

    def normalize(self, patent_id: str) -> ParsedPatentId:
        """Normalize with caching.

        Args:
            patent_id: Patent identifier string.

        Returns:
            Cached or newly parsed ParsedPatentId.
        """
        if patent_id not in self._cache:
            self._cache[patent_id] = normalize_patent_id(patent_id)
        return self._cache[patent_id]

    def normalize_batch(self, patent_ids: List[str]) -> List[ParsedPatentId]:
        """Normalize multiple IDs efficiently.

        Args:
            patent_ids: List of patent ID strings.

        Returns:
            List of ParsedPatentId objects.
        """
        return [self.normalize(pid) for pid in patent_ids]

    def get_canonical_mapping(
        self,
        patent_ids: List[str],
    ) -> dict[str, str]:
        """Create mapping from original IDs to canonical forms.

        Args:
            patent_ids: List of patent ID strings.

        Returns:
            Dict mapping original -> canonical.
        """
        return {pid: self.normalize(pid).canonical for pid in patent_ids}

    def clear_cache(self):
        """Clear the normalization cache."""
        self._cache.clear()
