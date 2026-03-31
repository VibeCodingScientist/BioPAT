"""Tests for processing modules."""

import pytest
import polars as pl
from pathlib import Path
import tempfile

from biopat.processing.patents import PatentProcessor
from biopat.processing.papers import PaperProcessor
from biopat.processing.linking import CitationLinker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPatentProcessor:
    """Tests for PatentProcessor."""

    def test_parse_independent_claim(self):
        """Test identification of independent claims."""
        processor = PatentProcessor(Path("/tmp"))

        # Independent claim
        claim_text = "A method of treating cancer comprising administering to a subject a therapeutically effective amount of a compound."
        claim_type, depends_on = processor.parse_claim_type(claim_text, 1)
        assert claim_type == "independent"
        assert depends_on is None

    def test_parse_dependent_claim(self):
        """Test identification of dependent claims."""
        processor = PatentProcessor(Path("/tmp"))

        # Dependent claim
        claim_text = "The method of claim 1, wherein the compound is administered orally."
        claim_type, depends_on = processor.parse_claim_type(claim_text, 2)
        assert claim_type == "dependent"
        assert depends_on == 1

    def test_parse_dependent_claim_according_to(self):
        """Test dependent claim with 'according to' phrasing."""
        processor = PatentProcessor(Path("/tmp"))

        claim_text = "A composition according to claim 5, further comprising an excipient."
        claim_type, depends_on = processor.parse_claim_type(claim_text, 6)
        assert claim_type == "dependent"
        assert depends_on == 5

    def test_get_ipc3(self):
        """Test IPC3 extraction."""
        processor = PatentProcessor(Path("/tmp"))

        assert processor.get_ipc3(["A61K39/395"]) == "A61K"
        assert processor.get_ipc3(["C07D213/00", "A61K31/00"]) == "C07D"
        assert processor.get_ipc3([]) == ""

    def test_filter_by_ipc(self, temp_dir):
        """Test IPC filtering."""
        processor = PatentProcessor(temp_dir)

        patents_df = pl.DataFrame({
            "patent_id": ["US1", "US2", "US3"],
            "ipc_codes": [["A61K39/00"], ["G06F17/00"], ["C12N15/00"]],
        })

        filtered = processor.filter_by_ipc(patents_df, ["A61", "C12"])

        assert len(filtered) == 2
        assert "US1" in filtered["patent_id"].to_list()
        assert "US3" in filtered["patent_id"].to_list()
        assert "US2" not in filtered["patent_id"].to_list()


class TestPaperProcessor:
    """Tests for PaperProcessor."""

    def test_validate_papers(self, temp_dir):
        """Test paper validation."""
        processor = PaperProcessor(temp_dir)

        papers_df = pl.DataFrame({
            "paper_id": ["W1", "W2", None, "W4"],
            "title": ["Title 1", "", "Title 3", "Title 4"],
            "publication_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        })

        validated = processor.validate_papers(papers_df)

        # Should remove W2 (empty title) and row with None paper_id
        assert len(validated) == 2
        assert "W1" in validated["paper_id"].to_list()
        assert "W4" in validated["paper_id"].to_list()

    def test_deduplicate(self, temp_dir):
        """Test paper deduplication."""
        processor = PaperProcessor(temp_dir)

        papers_df = pl.DataFrame({
            "paper_id": ["W1", "W1", "W2"],
            "title": ["Title 1", "Title 1 Dup", "Title 2"],
        })

        deduped = processor.deduplicate(papers_df)

        assert len(deduped) == 2


class TestCitationLinker:
    """Tests for CitationLinker."""

    def test_create_citation_links(self, temp_dir):
        """Test citation link creation."""
        linker = CitationLinker(temp_dir)

        ros_df = pl.DataFrame({
            "patent_id": ["US1", "US1", "US2", "US3"],
            "openalex_id": ["W1", "W2", "W1", "W3"],
            "confidence": [9, 8, 9, 9],
        })

        patents_df = pl.DataFrame({
            "patent_id": ["US1", "US2"],
        })

        papers_df = pl.DataFrame({
            "paper_id": ["W1", "W2"],
        })

        links = linker.create_citation_links(ros_df, patents_df, papers_df)

        # Should only include links where both patent and paper are valid
        assert len(links) == 3  # US1-W1, US1-W2, US2-W1

    def test_expand_to_claims(self, temp_dir):
        """Test expansion of patent links to claim level."""
        linker = CitationLinker(temp_dir)

        citation_links = pl.DataFrame({
            "patent_id": ["US1", "US1"],
            "paper_id": ["W1", "W2"],
            "confidence": [9, 8],
            "source": ["examiner", "applicant"],
        })

        claims_df = pl.DataFrame({
            "query_id": ["US1-c1", "US1-c3"],
            "patent_id": ["US1", "US1"],
            "claim_number": [1, 3],
            "priority_date": ["2020-01-01", "2020-01-01"],
        })

        expanded = linker.expand_to_claims(citation_links, claims_df)

        # 2 claims x 2 citations = 4 expanded links
        assert len(expanded) == 4
