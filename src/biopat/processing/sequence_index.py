"""BLAST+ Database Management System.

Phase 4.0 (Advanced): Manages local BLAST+ databases for sequence similarity
search in the BioPAT benchmark.

This module handles:
- Creating BLAST databases from FASTA files (makeblastdb)
- Running BLAST searches (blastp for proteins, blastn for nucleotides)
- Parsing BLAST XML results and extracting alignment statistics
- Managing sequence collections for patents and publications
"""

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class BlastHit:
    """A single BLAST alignment hit."""

    query_id: str
    subject_id: str
    identity: float  # Percent identity (0-100)
    alignment_length: int
    mismatches: int
    gap_opens: int
    query_start: int
    query_end: int
    subject_start: int
    subject_end: int
    evalue: float
    bit_score: float
    query_coverage: float = 0.0  # Percentage of query aligned

    @property
    def normalized_identity(self) -> float:
        """Return identity as 0-1 scale."""
        return self.identity / 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "subject_id": self.subject_id,
            "identity": self.identity,
            "alignment_length": self.alignment_length,
            "mismatches": self.mismatches,
            "gap_opens": self.gap_opens,
            "query_start": self.query_start,
            "query_end": self.query_end,
            "subject_start": self.subject_start,
            "subject_end": self.subject_end,
            "evalue": self.evalue,
            "bit_score": self.bit_score,
            "query_coverage": self.query_coverage,
        }


@dataclass
class BlastResult:
    """Results from a BLAST search."""

    query_id: str
    query_length: int
    hits: List[BlastHit] = field(default_factory=list)
    program: str = "blastp"
    database: str = ""
    search_time: float = 0.0

    @property
    def top_hit(self) -> Optional[BlastHit]:
        """Get the best hit by bit score."""
        if not self.hits:
            return None
        return max(self.hits, key=lambda h: h.bit_score)

    @property
    def has_significant_hit(self, evalue_threshold: float = 1e-5) -> bool:
        """Check if any hit meets significance threshold."""
        return any(h.evalue <= evalue_threshold for h in self.hits)


@dataclass
class SequenceRecord:
    """A biological sequence for indexing."""

    sequence_id: str  # Unique identifier
    sequence: str  # Actual sequence (AA or NT)
    sequence_type: str  # "AA" (amino acid) or "NT" (nucleotide)
    source_id: str  # Patent number or paper ID
    source_type: str  # "patent" or "publication"
    description: str = ""
    organism: str = ""

    def to_fasta(self) -> str:
        """Convert to FASTA format string."""
        header = f">{self.sequence_id}|{self.source_type}:{self.source_id}"
        if self.description:
            header += f"|{self.description}"

        # Wrap sequence at 80 characters
        wrapped = "\n".join(
            self.sequence[i : i + 80] for i in range(0, len(self.sequence), 80)
        )
        return f"{header}\n{wrapped}"

    @classmethod
    def from_fasta(cls, fasta_text: str, source_type: str = "unknown") -> "SequenceRecord":
        """Parse a single FASTA record."""
        lines = fasta_text.strip().split("\n")
        if not lines or not lines[0].startswith(">"):
            raise ValueError("Invalid FASTA format")

        header = lines[0][1:]  # Remove '>'
        sequence = "".join(lines[1:])

        # Parse header
        parts = header.split("|")
        seq_id = parts[0]
        source_id = ""
        description = ""

        if len(parts) > 1:
            # Try to extract source info
            if ":" in parts[1]:
                src_type, source_id = parts[1].split(":", 1)
                source_type = src_type
        if len(parts) > 2:
            description = "|".join(parts[2:])

        # Detect sequence type
        # Nucleotides contain only A, C, G, T, U, N and gaps
        # Proteins have many more letter codes
        upper_seq = sequence.upper()
        nucleotide_chars = set("ACGTUNRYWSMKHBVD-")  # IUPAC nucleotide codes
        protein_only_chars = set("EFIKLPQRVWY")  # Letters exclusive to proteins

        if any(c in protein_only_chars for c in upper_seq):
            seq_type = "AA"
        elif all(c in nucleotide_chars for c in upper_seq):
            seq_type = "NT"
        else:
            seq_type = "AA"  # Default to protein if ambiguous

        return cls(
            sequence_id=seq_id,
            sequence=sequence,
            sequence_type=seq_type,
            source_id=source_id,
            source_type=source_type,
            description=description,
        )


class BlastDatabaseManager:
    """Manages local BLAST+ databases for sequence similarity search.

    Handles creation, updating, and searching of BLAST databases.
    Supports both nucleotide (blastn) and protein (blastp) searches.
    """

    def __init__(
        self,
        db_dir: Path,
        blast_path: Optional[Path] = None,
    ):
        """Initialize BLAST database manager.

        Args:
            db_dir: Directory for storing BLAST databases and FASTA files.
            blast_path: Optional path to BLAST+ executables. If None,
                       assumes they are in system PATH.
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.blast_path = Path(blast_path) if blast_path else None

        # Track database metadata
        self._databases: Dict[str, Dict[str, Any]] = {}

    def _get_blast_cmd(self, program: str) -> str:
        """Get full path to BLAST executable."""
        if self.blast_path:
            return str(self.blast_path / program)
        return program

    def _check_blast_available(self) -> bool:
        """Check if BLAST+ is available."""
        try:
            result = subprocess.run(
                [self._get_blast_cmd("blastp"), "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_db_path(self, db_name: str) -> Path:
        """Get path to a BLAST database."""
        return self.db_dir / db_name

    def get_fasta_path(self, db_name: str) -> Path:
        """Get path to the source FASTA file for a database."""
        return self.db_dir / f"{db_name}.fasta"

    def create_database(
        self,
        sequences: List[SequenceRecord],
        db_name: str,
        db_type: str = "prot",
        title: Optional[str] = None,
    ) -> bool:
        """Create a BLAST database from sequences.

        Args:
            sequences: List of SequenceRecord objects to index.
            db_name: Name for the database.
            db_type: "prot" for protein, "nucl" for nucleotide.
            title: Optional title for the database.

        Returns:
            True if successful, False otherwise.
        """
        if not sequences:
            logger.warning(f"No sequences provided for database {db_name}")
            return False

        if not self._check_blast_available():
            logger.error("BLAST+ is not available. Please install NCBI BLAST+.")
            return False

        # Write sequences to FASTA file
        fasta_path = self.get_fasta_path(db_name)
        with open(fasta_path, "w") as f:
            for seq in sequences:
                f.write(seq.to_fasta())
                f.write("\n")

        # Build BLAST database
        db_path = self.get_db_path(db_name)
        cmd = [
            self._get_blast_cmd("makeblastdb"),
            "-in", str(fasta_path),
            "-dbtype", db_type,
            "-out", str(db_path),
        ]
        if title:
            cmd.extend(["-title", title])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes max
            )
            if result.returncode != 0:
                logger.error(f"makeblastdb failed: {result.stderr}")
                return False

            # Store metadata
            self._databases[db_name] = {
                "type": db_type,
                "sequence_count": len(sequences),
                "fasta_path": str(fasta_path),
                "db_path": str(db_path),
            }

            logger.info(f"Created BLAST database {db_name} with {len(sequences)} sequences")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"makeblastdb timed out for {db_name}")
            return False
        except subprocess.SubprocessError as e:
            logger.error(f"makeblastdb error: {e}")
            return False

    def add_sequences(
        self,
        sequences: List[SequenceRecord],
        db_name: str,
    ) -> bool:
        """Add sequences to an existing database (rebuilds database).

        Args:
            sequences: New sequences to add.
            db_name: Database name.

        Returns:
            True if successful.
        """
        # Load existing sequences
        fasta_path = self.get_fasta_path(db_name)
        existing = []
        if fasta_path.exists():
            existing = self.load_fasta_file(fasta_path)

        # Combine and rebuild
        all_sequences = existing + sequences
        db_type = self._databases.get(db_name, {}).get("type", "prot")

        return self.create_database(all_sequences, db_name, db_type)

    def load_fasta_file(self, fasta_path: Path) -> List[SequenceRecord]:
        """Load sequences from a FASTA file."""
        sequences = []
        current_header = None
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_header and current_seq:
                        fasta_text = f"{current_header}\n{''.join(current_seq)}"
                        try:
                            sequences.append(SequenceRecord.from_fasta(fasta_text))
                        except ValueError:
                            pass
                    current_header = line
                    current_seq = []
                elif line:
                    current_seq.append(line)

        # Last sequence
        if current_header and current_seq:
            fasta_text = f"{current_header}\n{''.join(current_seq)}"
            try:
                sequences.append(SequenceRecord.from_fasta(fasta_text))
            except ValueError:
                pass

        return sequences

    def database_exists(self, db_name: str) -> bool:
        """Check if a database exists."""
        db_path = self.get_db_path(db_name)
        # BLAST creates multiple files with extensions
        return any(
            (self.db_dir / f"{db_name}{ext}").exists()
            for ext in [".phr", ".pin", ".psq", ".nhr", ".nin", ".nsq"]
        )

    def get_database_info(self, db_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a database."""
        if not self.database_exists(db_name):
            return None

        # Use blastdbcmd to get info
        try:
            result = subprocess.run(
                [
                    self._get_blast_cmd("blastdbcmd"),
                    "-db", str(self.get_db_path(db_name)),
                    "-info",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return {"info": result.stdout}
        except subprocess.SubprocessError:
            pass

        return self._databases.get(db_name)

    def delete_database(self, db_name: str) -> bool:
        """Delete a database and its files."""
        deleted = False
        for ext in [".phr", ".pin", ".psq", ".nhr", ".nin", ".nsq", ".pdb", ".pot", ".ptf", ".pto", ".fasta"]:
            path = self.db_dir / f"{db_name}{ext}"
            if path.exists():
                path.unlink()
                deleted = True

        if db_name in self._databases:
            del self._databases[db_name]

        return deleted


class BlastSearcher:
    """Execute BLAST searches against local databases."""

    def __init__(
        self,
        db_manager: BlastDatabaseManager,
        default_evalue: float = 10.0,
        default_max_hits: int = 100,
    ):
        """Initialize BLAST searcher.

        Args:
            db_manager: Database manager instance.
            default_evalue: Default E-value threshold.
            default_max_hits: Default maximum number of hits to return.
        """
        self.db_manager = db_manager
        self.default_evalue = default_evalue
        self.default_max_hits = default_max_hits

    async def search(
        self,
        query_sequence: str,
        db_name: str,
        program: str = "blastp",
        evalue: Optional[float] = None,
        max_hits: Optional[int] = None,
        query_id: str = "query",
    ) -> BlastResult:
        """Run a BLAST search.

        Args:
            query_sequence: Query sequence string.
            db_name: Name of database to search.
            program: BLAST program (blastp, blastn, blastx, tblastn).
            evalue: E-value threshold.
            max_hits: Maximum number of hits.
            query_id: Identifier for the query.

        Returns:
            BlastResult with hits.
        """
        if not self.db_manager.database_exists(db_name):
            logger.error(f"Database {db_name} does not exist")
            return BlastResult(query_id=query_id, query_length=len(query_sequence))

        evalue = evalue or self.default_evalue
        max_hits = max_hits or self.default_max_hits

        # Write query to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(f">{query_id}\n{query_sequence}\n")
            query_file = f.name

        try:
            # Run BLAST asynchronously
            result = await self._run_blast(
                query_file=query_file,
                db_name=db_name,
                program=program,
                evalue=evalue,
                max_hits=max_hits,
            )
            result.query_id = query_id
            result.query_length = len(query_sequence)
            return result

        finally:
            # Clean up temp file
            os.unlink(query_file)

    async def _run_blast(
        self,
        query_file: str,
        db_name: str,
        program: str,
        evalue: float,
        max_hits: int,
    ) -> BlastResult:
        """Execute BLAST and parse results."""
        db_path = self.db_manager.get_db_path(db_name)

        cmd = [
            self.db_manager._get_blast_cmd(program),
            "-query", query_file,
            "-db", str(db_path),
            "-evalue", str(evalue),
            "-max_target_seqs", str(max_hits),
            "-outfmt", "5",  # XML output
        ]

        # Run BLAST in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            ),
        )

        if result.returncode != 0:
            logger.error(f"BLAST failed: {result.stderr}")
            return BlastResult(query_id="", query_length=0, program=program, database=db_name)

        # Parse XML output
        return self._parse_blast_xml(result.stdout, program, db_name)

    def _parse_blast_xml(
        self,
        xml_output: str,
        program: str,
        database: str,
    ) -> BlastResult:
        """Parse BLAST XML output."""
        result = BlastResult(
            query_id="",
            query_length=0,
            program=program,
            database=database,
        )

        try:
            root = ET.fromstring(xml_output)

            # Get query info
            iterations = root.findall(".//Iteration")
            if not iterations:
                return result

            iteration = iterations[0]
            result.query_id = iteration.findtext("Iteration_query-def", "")
            result.query_length = int(iteration.findtext("Iteration_query-len", "0"))

            # Parse hits
            for hit in iteration.findall(".//Hit"):
                hit_id = hit.findtext("Hit_id", "")
                hit_def = hit.findtext("Hit_def", "")

                for hsp in hit.findall(".//Hsp"):
                    alignment_length = int(hsp.findtext("Hsp_align-len", "0"))
                    identity = int(hsp.findtext("Hsp_identity", "0"))

                    # Calculate percent identity
                    pct_identity = (identity / alignment_length * 100) if alignment_length > 0 else 0

                    # Calculate query coverage
                    query_from = int(hsp.findtext("Hsp_query-from", "0"))
                    query_to = int(hsp.findtext("Hsp_query-to", "0"))
                    query_coverage = ((query_to - query_from + 1) / result.query_length * 100) if result.query_length > 0 else 0

                    blast_hit = BlastHit(
                        query_id=result.query_id,
                        subject_id=hit_id,
                        identity=pct_identity,
                        alignment_length=alignment_length,
                        mismatches=int(hsp.findtext("Hsp_gaps", "0")),
                        gap_opens=int(hsp.findtext("Hsp_gaps", "0")),
                        query_start=query_from,
                        query_end=query_to,
                        subject_start=int(hsp.findtext("Hsp_hit-from", "0")),
                        subject_end=int(hsp.findtext("Hsp_hit-to", "0")),
                        evalue=float(hsp.findtext("Hsp_evalue", "999")),
                        bit_score=float(hsp.findtext("Hsp_bit-score", "0")),
                        query_coverage=query_coverage,
                    )
                    result.hits.append(blast_hit)

        except ET.ParseError as e:
            logger.error(f"Failed to parse BLAST XML: {e}")

        return result

    async def batch_search(
        self,
        queries: List[Tuple[str, str]],  # List of (query_id, sequence)
        db_name: str,
        program: str = "blastp",
        evalue: Optional[float] = None,
        max_concurrent: int = 4,
    ) -> List[BlastResult]:
        """Run multiple BLAST searches concurrently.

        Args:
            queries: List of (query_id, sequence) tuples.
            db_name: Database name.
            program: BLAST program.
            evalue: E-value threshold.
            max_concurrent: Maximum concurrent searches.

        Returns:
            List of BlastResult objects.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_with_limit(query_id: str, sequence: str) -> BlastResult:
            async with semaphore:
                return await self.search(
                    query_sequence=sequence,
                    db_name=db_name,
                    program=program,
                    evalue=evalue,
                    query_id=query_id,
                )

        tasks = [search_with_limit(qid, seq) for qid, seq in queries]
        return await asyncio.gather(*tasks)


class SequenceIndex:
    """High-level sequence index for BioPAT benchmark.

    Manages separate databases for patents and publications,
    supporting sequence-based prior art search.
    """

    def __init__(
        self,
        index_dir: Path,
        blast_path: Optional[Path] = None,
    ):
        """Initialize sequence index.

        Args:
            index_dir: Directory for storing indices.
            blast_path: Optional path to BLAST+ binaries.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.db_manager = BlastDatabaseManager(
            db_dir=self.index_dir / "blast_dbs",
            blast_path=blast_path,
        )
        self.searcher = BlastSearcher(self.db_manager)

        # Standard database names
        self.PATENT_PROTEIN_DB = "patent_proteins"
        self.PATENT_NUCLEOTIDE_DB = "patent_nucleotides"
        self.PUBLICATION_PROTEIN_DB = "publication_proteins"
        self.PUBLICATION_NUCLEOTIDE_DB = "publication_nucleotides"

    def index_patent_sequences(
        self,
        sequences: List[SequenceRecord],
    ) -> Tuple[int, int]:
        """Index sequences from patents.

        Args:
            sequences: Patent sequences to index.

        Returns:
            Tuple of (protein_count, nucleotide_count) indexed.
        """
        proteins = [s for s in sequences if s.sequence_type == "AA"]
        nucleotides = [s for s in sequences if s.sequence_type == "NT"]

        prot_count = 0
        nucl_count = 0

        if proteins:
            if self.db_manager.create_database(
                proteins, self.PATENT_PROTEIN_DB, "prot", "BioPAT Patent Proteins"
            ):
                prot_count = len(proteins)

        if nucleotides:
            if self.db_manager.create_database(
                nucleotides, self.PATENT_NUCLEOTIDE_DB, "nucl", "BioPAT Patent Nucleotides"
            ):
                nucl_count = len(nucleotides)

        return prot_count, nucl_count

    def index_publication_sequences(
        self,
        sequences: List[SequenceRecord],
    ) -> Tuple[int, int]:
        """Index sequences from publications.

        Args:
            sequences: Publication sequences to index.

        Returns:
            Tuple of (protein_count, nucleotide_count) indexed.
        """
        proteins = [s for s in sequences if s.sequence_type == "AA"]
        nucleotides = [s for s in sequences if s.sequence_type == "NT"]

        prot_count = 0
        nucl_count = 0

        if proteins:
            if self.db_manager.create_database(
                proteins, self.PUBLICATION_PROTEIN_DB, "prot", "BioPAT Publication Proteins"
            ):
                prot_count = len(proteins)

        if nucleotides:
            if self.db_manager.create_database(
                nucleotides, self.PUBLICATION_NUCLEOTIDE_DB, "nucl", "BioPAT Publication Nucleotides"
            ):
                nucl_count = len(nucleotides)

        return prot_count, nucl_count

    async def search_prior_art(
        self,
        query_sequence: str,
        sequence_type: str = "AA",
        min_identity: float = 70.0,
        max_hits: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search for prior art sequences.

        Args:
            query_sequence: Query sequence string.
            sequence_type: "AA" for protein, "NT" for nucleotide.
            min_identity: Minimum percent identity threshold.
            max_hits: Maximum hits to return.

        Returns:
            List of hits with source information.
        """
        program = "blastp" if sequence_type == "AA" else "blastn"
        db_name = (
            self.PUBLICATION_PROTEIN_DB
            if sequence_type == "AA"
            else self.PUBLICATION_NUCLEOTIDE_DB
        )

        if not self.db_manager.database_exists(db_name):
            logger.warning(f"Database {db_name} does not exist")
            return []

        result = await self.searcher.search(
            query_sequence=query_sequence,
            db_name=db_name,
            program=program,
            max_hits=max_hits,
        )

        # Filter by identity and format results
        hits = []
        for hit in result.hits:
            if hit.identity >= min_identity:
                # Parse source info from subject ID
                parts = hit.subject_id.split("|")
                source_type = "unknown"
                source_id = parts[0]

                if len(parts) > 1 and ":" in parts[1]:
                    source_type, source_id = parts[1].split(":", 1)

                hits.append({
                    "sequence_id": hit.subject_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "identity": hit.identity,
                    "evalue": hit.evalue,
                    "bit_score": hit.bit_score,
                    "alignment_length": hit.alignment_length,
                    "query_coverage": hit.query_coverage,
                })

        return sorted(hits, key=lambda x: x["bit_score"], reverse=True)[:max_hits]

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed sequences."""
        stats = {}
        for db_name in [
            self.PATENT_PROTEIN_DB,
            self.PATENT_NUCLEOTIDE_DB,
            self.PUBLICATION_PROTEIN_DB,
            self.PUBLICATION_NUCLEOTIDE_DB,
        ]:
            info = self.db_manager.get_database_info(db_name)
            stats[db_name] = {
                "exists": self.db_manager.database_exists(db_name),
                "info": info,
            }
        return stats


def compute_sequence_hash(sequence: str) -> str:
    """Compute a hash for a sequence (for deduplication)."""
    normalized = sequence.upper().replace("-", "").replace("*", "")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def identity_to_relevance_tier(identity: float) -> int:
    """Convert sequence identity to relevance tier.

    Based on standard sequence similarity interpretation:
    - >= 90%: Highly relevant (tier 3)
    - >= 70%: Relevant (tier 2)
    - >= 50%: Marginally relevant (tier 1)
    - < 50%: Not relevant (tier 0)
    """
    if identity >= 90:
        return 3
    elif identity >= 70:
        return 2
    elif identity >= 50:
        return 1
    return 0
