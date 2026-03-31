"""Tests for sequence indexing and BLAST database management."""

import pytest
from pathlib import Path
import tempfile

from biopat.processing.sequence_index import (
    BlastHit,
    BlastResult,
    SequenceRecord,
    BlastDatabaseManager,
    BlastSearcher,
    SequenceIndex,
    compute_sequence_hash,
    identity_to_relevance_tier,
)


class TestBlastHit:
    """Tests for BlastHit dataclass."""

    def test_blast_hit_creation(self):
        """Test creating a BLAST hit."""
        hit = BlastHit(
            query_id="query1",
            subject_id="subject1",
            identity=95.5,
            alignment_length=100,
            mismatches=5,
            gap_opens=0,
            query_start=1,
            query_end=100,
            subject_start=1,
            subject_end=100,
            evalue=1e-50,
            bit_score=200.0,
            query_coverage=100.0,
        )
        assert hit.query_id == "query1"
        assert hit.subject_id == "subject1"
        assert hit.identity == 95.5
        assert hit.evalue == 1e-50

    def test_normalized_identity(self):
        """Test normalized identity calculation."""
        hit = BlastHit(
            query_id="q",
            subject_id="s",
            identity=75.0,
            alignment_length=50,
            mismatches=0,
            gap_opens=0,
            query_start=1,
            query_end=50,
            subject_start=1,
            subject_end=50,
            evalue=1e-10,
            bit_score=100.0,
        )
        assert hit.normalized_identity == 0.75

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = BlastHit(
            query_id="q",
            subject_id="s",
            identity=90.0,
            alignment_length=100,
            mismatches=10,
            gap_opens=0,
            query_start=1,
            query_end=100,
            subject_start=1,
            subject_end=100,
            evalue=1e-30,
            bit_score=150.0,
            query_coverage=95.0,
        )
        d = hit.to_dict()
        assert d["query_id"] == "q"
        assert d["identity"] == 90.0
        assert d["query_coverage"] == 95.0


class TestBlastResult:
    """Tests for BlastResult dataclass."""

    def test_blast_result_creation(self):
        """Test creating a BLAST result."""
        result = BlastResult(
            query_id="query1",
            query_length=150,
            program="blastp",
            database="test_db",
        )
        assert result.query_id == "query1"
        assert result.query_length == 150
        assert result.hits == []

    def test_top_hit(self):
        """Test getting top hit by bit score."""
        hit1 = BlastHit(
            query_id="q", subject_id="s1", identity=80.0,
            alignment_length=100, mismatches=20, gap_opens=0,
            query_start=1, query_end=100, subject_start=1, subject_end=100,
            evalue=1e-20, bit_score=100.0,
        )
        hit2 = BlastHit(
            query_id="q", subject_id="s2", identity=95.0,
            alignment_length=100, mismatches=5, gap_opens=0,
            query_start=1, query_end=100, subject_start=1, subject_end=100,
            evalue=1e-40, bit_score=200.0,
        )
        result = BlastResult(query_id="q", query_length=100, hits=[hit1, hit2])

        assert result.top_hit == hit2

    def test_top_hit_empty(self):
        """Test top hit when no hits."""
        result = BlastResult(query_id="q", query_length=100)
        assert result.top_hit is None


class TestSequenceRecord:
    """Tests for SequenceRecord dataclass."""

    def test_sequence_record_creation(self):
        """Test creating a sequence record."""
        record = SequenceRecord(
            sequence_id="SEQ1",
            sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            sequence_type="AA",
            source_id="US12345678",
            source_type="patent",
            description="Hemoglobin alpha chain",
        )
        assert record.sequence_id == "SEQ1"
        assert record.sequence_type == "AA"
        assert record.source_type == "patent"

    def test_to_fasta(self):
        """Test conversion to FASTA format."""
        record = SequenceRecord(
            sequence_id="SEQ1",
            sequence="MVLSPADKTNVKAAWGKVGA",
            sequence_type="AA",
            source_id="US12345678",
            source_type="patent",
            description="Test protein",
        )
        fasta = record.to_fasta()
        assert fasta.startswith(">SEQ1|patent:US12345678|Test protein")
        assert "MVLSPADKTNVKAAWGKVGA" in fasta

    def test_from_fasta(self):
        """Test parsing FASTA format."""
        fasta_text = ">SEQ1|patent:US12345678|Test protein\nMVLSPADKTNVKAAWGKVGA"
        record = SequenceRecord.from_fasta(fasta_text)
        assert record.sequence_id == "SEQ1"
        assert record.sequence == "MVLSPADKTNVKAAWGKVGA"
        assert record.source_type == "patent"
        assert record.source_id == "US12345678"

    def test_from_fasta_simple(self):
        """Test parsing simple FASTA format."""
        fasta_text = ">simple_seq\nACGTACGT"
        record = SequenceRecord.from_fasta(fasta_text)
        assert record.sequence_id == "simple_seq"
        assert record.sequence == "ACGTACGT"
        assert record.sequence_type == "NT"

    def test_sequence_type_detection_protein(self):
        """Test protein sequence type detection."""
        fasta_text = ">prot\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        record = SequenceRecord.from_fasta(fasta_text)
        assert record.sequence_type == "AA"

    def test_sequence_type_detection_nucleotide(self):
        """Test nucleotide sequence type detection."""
        fasta_text = ">nucl\nACGTACGTACGTACGTACGT"
        record = SequenceRecord.from_fasta(fasta_text)
        assert record.sequence_type == "NT"


class TestBlastDatabaseManager:
    """Tests for BlastDatabaseManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            assert manager.db_dir.exists()
            assert manager.blast_path is None

    def test_get_db_path(self):
        """Test getting database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            path = manager.get_db_path("test_db")
            assert str(path).endswith("test_db")

    def test_get_fasta_path(self):
        """Test getting FASTA file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            path = manager.get_fasta_path("test_db")
            assert str(path).endswith("test_db.fasta")

    def test_database_exists_false(self):
        """Test database_exists when database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            assert not manager.database_exists("nonexistent_db")

    def test_delete_nonexistent_database(self):
        """Test deleting a database that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            # Should not raise, just return False
            result = manager.delete_database("nonexistent")
            assert result is False


class TestSequenceIndex:
    """Tests for SequenceIndex."""

    def test_index_initialization(self):
        """Test sequence index initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SequenceIndex(index_dir=Path(tmpdir))
            assert index.index_dir.exists()
            assert index.db_manager is not None
            assert index.searcher is not None

    def test_standard_database_names(self):
        """Test standard database name constants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SequenceIndex(index_dir=Path(tmpdir))
            assert index.PATENT_PROTEIN_DB == "patent_proteins"
            assert index.PATENT_NUCLEOTIDE_DB == "patent_nucleotides"
            assert index.PUBLICATION_PROTEIN_DB == "publication_proteins"
            assert index.PUBLICATION_NUCLEOTIDE_DB == "publication_nucleotides"

    def test_get_index_stats(self):
        """Test getting index statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SequenceIndex(index_dir=Path(tmpdir))
            stats = index.get_index_stats()
            assert "patent_proteins" in stats
            assert "publication_proteins" in stats
            assert stats["patent_proteins"]["exists"] is False


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_sequence_hash(self):
        """Test sequence hash computation."""
        seq1 = "MVLSPADKTNVKAAWGKVGA"
        seq2 = "mvlspadktnvkaawgkvga"  # Same but lowercase

        hash1 = compute_sequence_hash(seq1)
        hash2 = compute_sequence_hash(seq2)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_sequence_hash_with_gaps(self):
        """Test hash ignores gap characters."""
        seq1 = "MVLSPADKTNVKAAWGKVGA"
        seq2 = "MVL-SPADKTN-VKAAWGKVGA"

        hash1 = compute_sequence_hash(seq1)
        hash2 = compute_sequence_hash(seq2)

        assert hash1 == hash2

    def test_identity_to_relevance_tier_high(self):
        """Test high identity mapping."""
        assert identity_to_relevance_tier(95.0) == 3
        assert identity_to_relevance_tier(90.0) == 3

    def test_identity_to_relevance_tier_relevant(self):
        """Test relevant identity mapping."""
        assert identity_to_relevance_tier(80.0) == 2
        assert identity_to_relevance_tier(70.0) == 2

    def test_identity_to_relevance_tier_marginal(self):
        """Test marginal identity mapping."""
        assert identity_to_relevance_tier(60.0) == 1
        assert identity_to_relevance_tier(50.0) == 1

    def test_identity_to_relevance_tier_not_relevant(self):
        """Test not relevant identity mapping."""
        assert identity_to_relevance_tier(40.0) == 0
        assert identity_to_relevance_tier(0.0) == 0


class TestFastaRoundTrip:
    """Tests for FASTA serialization round-trip."""

    def test_fasta_round_trip(self):
        """Test that FASTA conversion is reversible."""
        original = SequenceRecord(
            sequence_id="TEST1",
            sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            sequence_type="AA",
            source_id="US12345678",
            source_type="patent",
            description="Test protein sequence",
        )

        fasta = original.to_fasta()
        parsed = SequenceRecord.from_fasta(fasta)

        assert parsed.sequence_id == original.sequence_id
        assert parsed.sequence == original.sequence
        assert parsed.source_type == original.source_type
        assert parsed.source_id == original.source_id

    def test_fasta_multiline_sequence(self):
        """Test FASTA with long sequence that gets wrapped."""
        long_seq = "MVLSPADKTNVKAAWGKVGA" * 10  # 200 characters
        original = SequenceRecord(
            sequence_id="LONG",
            sequence=long_seq,
            sequence_type="AA",
            source_id="US99999999",
            source_type="patent",
        )

        fasta = original.to_fasta()
        # Check it's wrapped
        lines = fasta.split("\n")
        assert len(lines) > 2  # Header + multiple sequence lines

        # Parse back
        parsed = SequenceRecord.from_fasta(fasta)
        assert parsed.sequence == long_seq


class TestBlastSearcher:
    """Tests for BlastSearcher (without actual BLAST execution)."""

    def test_searcher_initialization(self):
        """Test searcher initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            searcher = BlastSearcher(manager)
            assert searcher.default_evalue == 10.0
            assert searcher.default_max_hits == 100

    def test_searcher_custom_defaults(self):
        """Test searcher with custom defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BlastDatabaseManager(db_dir=Path(tmpdir))
            searcher = BlastSearcher(
                manager,
                default_evalue=0.001,
                default_max_hits=50,
            )
            assert searcher.default_evalue == 0.001
            assert searcher.default_max_hits == 50
