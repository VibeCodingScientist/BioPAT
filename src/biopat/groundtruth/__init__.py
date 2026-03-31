"""Ground truth computation modules for BioPAT."""

from .relevance import RelevanceAssigner, RelevanceLevel, RELEVANCE_LABELS
from .temporal import TemporalValidator
from .stratification import DomainStratifier, create_stratified_evaluation
from .ep_citations import (
    EPCitationCategory,
    EPCitation,
    EPSearchReport,
    EPSearchReportParser,
    EP_CATEGORY_TO_RELEVANCE,
    map_ep_category_to_relevance,
    combine_us_ep_relevance,
)

__all__ = [
    "RelevanceAssigner",
    "RelevanceLevel",
    "RELEVANCE_LABELS",
    "TemporalValidator",
    "DomainStratifier",
    "create_stratified_evaluation",
    # EP search report parsing (Phase 6)
    "EPCitationCategory",
    "EPCitation",
    "EPSearchReport",
    "EPSearchReportParser",
    "EP_CATEGORY_TO_RELEVANCE",
    "map_ep_category_to_relevance",
    "combine_us_ep_relevance",
]
