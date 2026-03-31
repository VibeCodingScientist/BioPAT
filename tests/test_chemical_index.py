"""Tests for chemical indexing and FAISS-based similarity search."""

import pytest
import tempfile
from pathlib import Path

import numpy as np

from biopat.processing.chemical_index import (
    ChemicalRecord,
    ChemicalSearchHit,
    MorganFingerprintCalculator,
    compute_tanimoto,
    tanimoto_to_relevance_tier,
    compute_chemical_id,
    RDKIT_AVAILABLE,
    FAISS_AVAILABLE,
)


# Skip markers for tests requiring optional dependencies
requires_rdkit = pytest.mark.skipif(
    not RDKIT_AVAILABLE, reason="RDKit not installed"
)
requires_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE, reason="FAISS not installed"
)
requires_both = pytest.mark.skipif(
    not (RDKIT_AVAILABLE and FAISS_AVAILABLE),
    reason="RDKit and FAISS required"
)


class TestChemicalRecord:
    """Tests for ChemicalRecord dataclass."""

    def test_chemical_record_creation(self):
        """Test creating a chemical record."""
        record = ChemicalRecord(
            chemical_id="CHEM001",
            smiles="CCO",
            source_id="US12345678",
            source_type="patent",
            name="Ethanol",
        )
        assert record.chemical_id == "CHEM001"
        assert record.smiles == "CCO"
        assert record.source_type == "patent"
        assert record.name == "Ethanol"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ChemicalRecord(
            chemical_id="CHEM001",
            smiles="CCO",
            source_id="US12345678",
            source_type="patent",
        )
        d = record.to_dict()
        assert d["chemical_id"] == "CHEM001"
        assert d["smiles"] == "CCO"
        assert d["source_type"] == "patent"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "chemical_id": "CHEM002",
            "smiles": "CC(=O)O",
            "source_id": "PMID123456",
            "source_type": "publication",
            "name": "Acetic acid",
        }
        record = ChemicalRecord.from_dict(data)
        assert record.chemical_id == "CHEM002"
        assert record.smiles == "CC(=O)O"
        assert record.name == "Acetic acid"

    def test_round_trip(self):
        """Test dict conversion round-trip."""
        original = ChemicalRecord(
            chemical_id="CHEM003",
            smiles="c1ccccc1",
            source_id="US99999999",
            source_type="patent",
            name="Benzene",
            mol_formula="C6H6",
        )
        d = original.to_dict()
        restored = ChemicalRecord.from_dict(d)
        assert restored.chemical_id == original.chemical_id
        assert restored.smiles == original.smiles
        assert restored.name == original.name


class TestChemicalSearchHit:
    """Tests for ChemicalSearchHit dataclass."""

    def test_search_hit_creation(self):
        """Test creating a search hit."""
        hit = ChemicalSearchHit(
            query_id="query_smiles",
            chemical_id="CHEM001",
            smiles="CCO",
            source_id="US12345678",
            source_type="patent",
            similarity=0.85,
        )
        assert hit.similarity == 0.85
        assert hit.source_type == "patent"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = ChemicalSearchHit(
            query_id="q",
            chemical_id="CHEM001",
            smiles="CCO",
            source_id="US12345678",
            source_type="patent",
            similarity=0.75,
            inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )
        d = hit.to_dict()
        assert d["similarity"] == 0.75
        assert d["inchi_key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


class TestMorganFingerprintCalculator:
    """Tests for MorganFingerprintCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = MorganFingerprintCalculator(radius=2, n_bits=2048)
        assert calc.radius == 2
        assert calc.n_bits == 2048

    @requires_rdkit
    def test_compute_fingerprint(self):
        """Test fingerprint computation."""
        calc = MorganFingerprintCalculator()
        fp = calc.compute_fingerprint("CCO")
        assert fp is not None
        assert len(fp) == 2048
        assert fp.dtype == np.float32

    @requires_rdkit
    def test_compute_fingerprint_invalid_smiles(self):
        """Test fingerprint computation with invalid SMILES."""
        calc = MorganFingerprintCalculator()
        fp = calc.compute_fingerprint("invalid_smiles_xyz")
        assert fp is None

    @requires_rdkit
    def test_compute_fingerprint_normalized(self):
        """Test L2-normalized fingerprint computation."""
        calc = MorganFingerprintCalculator()
        fp = calc.compute_fingerprint_normalized("CCO")
        assert fp is not None
        # Check L2 normalization
        norm = np.linalg.norm(fp)
        assert abs(norm - 1.0) < 1e-6

    @requires_rdkit
    def test_compute_batch(self):
        """Test batch fingerprint computation."""
        calc = MorganFingerprintCalculator()
        smiles_list = ["CCO", "CC(=O)O", "invalid_xyz", "c1ccccc1"]
        fps, valid_indices = calc.compute_batch(smiles_list)

        # Should have 3 valid fingerprints (invalid_xyz should fail)
        assert len(valid_indices) == 3
        assert fps.shape[0] == 3
        assert fps.shape[1] == 2048

    @requires_rdkit
    def test_fingerprint_consistency(self):
        """Test that same molecule yields same fingerprint."""
        calc = MorganFingerprintCalculator()
        fp1 = calc.compute_fingerprint("CCO")
        fp2 = calc.compute_fingerprint("OCC")  # Same molecule, different SMILES

        # RDKit should canonicalize and produce identical fingerprints
        assert np.allclose(fp1, fp2)


class TestTanimotoFunctions:
    """Tests for Tanimoto similarity functions."""

    def test_compute_tanimoto_identical(self):
        """Test Tanimoto for identical fingerprints."""
        fp = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.float32)
        sim = compute_tanimoto(fp, fp)
        assert sim == 1.0

    def test_compute_tanimoto_different(self):
        """Test Tanimoto for different fingerprints."""
        fp1 = np.array([1, 1, 0, 0], dtype=np.float32)
        fp2 = np.array([1, 0, 1, 0], dtype=np.float32)
        sim = compute_tanimoto(fp1, fp2)
        # Intersection = 1, Union = 3
        assert abs(sim - 1/3) < 1e-6

    def test_compute_tanimoto_no_overlap(self):
        """Test Tanimoto for non-overlapping fingerprints."""
        fp1 = np.array([1, 1, 0, 0], dtype=np.float32)
        fp2 = np.array([0, 0, 1, 1], dtype=np.float32)
        sim = compute_tanimoto(fp1, fp2)
        assert sim == 0.0

    def test_compute_tanimoto_empty(self):
        """Test Tanimoto for zero vectors."""
        fp1 = np.array([0, 0, 0, 0], dtype=np.float32)
        fp2 = np.array([0, 0, 0, 0], dtype=np.float32)
        sim = compute_tanimoto(fp1, fp2)
        assert sim == 0.0


class TestRelevanceTiers:
    """Tests for Tanimoto to relevance tier conversion."""

    def test_tier_3_high_similarity(self):
        """Test tier 3 for high similarity."""
        assert tanimoto_to_relevance_tier(0.90) == 3
        assert tanimoto_to_relevance_tier(0.85) == 3
        assert tanimoto_to_relevance_tier(1.0) == 3

    def test_tier_2_relevant(self):
        """Test tier 2 for relevant similarity."""
        assert tanimoto_to_relevance_tier(0.80) == 2
        assert tanimoto_to_relevance_tier(0.70) == 2
        assert tanimoto_to_relevance_tier(0.84) == 2

    def test_tier_1_marginal(self):
        """Test tier 1 for marginal similarity."""
        assert tanimoto_to_relevance_tier(0.60) == 1
        assert tanimoto_to_relevance_tier(0.50) == 1
        assert tanimoto_to_relevance_tier(0.69) == 1

    def test_tier_0_not_relevant(self):
        """Test tier 0 for not relevant."""
        assert tanimoto_to_relevance_tier(0.40) == 0
        assert tanimoto_to_relevance_tier(0.0) == 0
        assert tanimoto_to_relevance_tier(0.49) == 0


class TestComputeChemicalId:
    """Tests for chemical ID computation."""

    def test_compute_chemical_id_basic(self):
        """Test basic chemical ID computation."""
        cid = compute_chemical_id("CCO")
        assert cid is not None
        assert len(cid) > 0

    def test_compute_chemical_id_consistency(self):
        """Test that same SMILES yields same ID."""
        cid1 = compute_chemical_id("CCO")
        cid2 = compute_chemical_id("CCO")
        assert cid1 == cid2

    @requires_rdkit
    def test_compute_chemical_id_canonical(self):
        """Test that equivalent SMILES yield same ID."""
        cid1 = compute_chemical_id("CCO")
        cid2 = compute_chemical_id("OCC")  # Same molecule
        assert cid1 == cid2


@requires_both
class TestFaissChemicalIndex:
    """Tests for FaissChemicalIndex (requires both RDKit and FAISS)."""

    def test_index_creation(self):
        """Test index creation."""
        from biopat.processing.chemical_index import FaissChemicalIndex

        index = FaissChemicalIndex(fingerprint_dim=2048)
        assert index.size == 0
        assert index.fingerprint_dim == 2048

    def test_add_and_search(self):
        """Test adding fingerprints and searching."""
        from biopat.processing.chemical_index import FaissChemicalIndex

        index = FaissChemicalIndex(fingerprint_dim=128)

        # Create dummy records and fingerprints
        records = [
            ChemicalRecord("C1", "CCO", "PAT1", "patent"),
            ChemicalRecord("C2", "CC(=O)O", "PAT2", "patent"),
        ]

        # Create random L2-normalized fingerprints
        fps = np.random.randn(2, 128).astype(np.float32)
        fps = fps / np.linalg.norm(fps, axis=1, keepdims=True)

        index.add(fps, records)
        assert index.size == 2

        # Search with first fingerprint
        results = index.search(fps[0], k=2)
        assert len(results) > 0
        assert results[0][0].chemical_id == "C1"  # Should find itself first

    def test_save_and_load(self):
        """Test saving and loading index."""
        from biopat.processing.chemical_index import FaissChemicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate index
            index = FaissChemicalIndex(fingerprint_dim=64)
            records = [ChemicalRecord("C1", "CCO", "PAT1", "patent")]
            fps = np.random.randn(1, 64).astype(np.float32)
            fps = fps / np.linalg.norm(fps, axis=1, keepdims=True)
            index.add(fps, records)

            # Save
            save_path = Path(tmpdir) / "test_index"
            index.save(save_path)

            # Load
            loaded = FaissChemicalIndex.load(save_path)
            assert loaded.size == 1
            assert loaded._records[0].chemical_id == "C1"


@requires_both
class TestChemicalIndex:
    """Tests for high-level ChemicalIndex."""

    def test_index_initialization(self):
        """Test chemical index initialization."""
        from biopat.processing.chemical_index import ChemicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = ChemicalIndex(index_dir=Path(tmpdir))
            assert index.index_dir.exists()
            assert index.fingerprint_bits == 2048

    def test_get_stats(self):
        """Test getting index statistics."""
        from biopat.processing.chemical_index import ChemicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = ChemicalIndex(index_dir=Path(tmpdir))
            stats = index.get_stats()
            assert "patent_chemicals" in stats
            assert "publication_chemicals" in stats
            assert stats["fingerprint_bits"] == 2048

    def test_index_chemicals(self):
        """Test indexing chemicals."""
        from biopat.processing.chemical_index import ChemicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = ChemicalIndex(index_dir=Path(tmpdir))

            chemicals = [
                ChemicalRecord("C1", "CCO", "PAT1", "patent"),
                ChemicalRecord("C2", "CC(=O)O", "PAT2", "patent"),
                ChemicalRecord("C3", "c1ccccc1", "PAT3", "patent"),
            ]

            count = index.index_chemicals(chemicals, source_type="patent")
            assert count == 3

    def test_search_prior_art(self):
        """Test searching for prior art."""
        from biopat.processing.chemical_index import ChemicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = ChemicalIndex(index_dir=Path(tmpdir))

            # Index some publication chemicals
            chemicals = [
                ChemicalRecord("C1", "CCO", "PMID1", "publication"),
                ChemicalRecord("C2", "CCCO", "PMID2", "publication"),
            ]
            index.index_chemicals(chemicals, source_type="publication")

            # Search for ethanol-like structure
            hits = index.search_prior_art("CCO", k=10, min_similarity=0.0)
            assert len(hits) > 0
            # Should find CCO as highly similar
            assert any(h.smiles == "CCO" for h in hits)
