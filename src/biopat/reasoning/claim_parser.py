"""LLM-based Patent Claim Parser.

Parses patent claims into structured elements using LLMs for:
- Claim type identification (method, composition, system, etc.)
- Element extraction (active steps, components, limitations)
- Entity recognition (compounds, proteins, diseases, etc.)
- Dependency analysis (which elements depend on others)

This enables fine-grained prior art mapping at the claim element level.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Patent claim type classification."""
    METHOD = "method"                    # Method/process claims
    COMPOSITION = "composition"          # Composition of matter
    COMPOUND = "compound"                # Chemical compound
    ANTIBODY = "antibody"                # Antibody/protein therapeutic
    NUCLEIC_ACID = "nucleic_acid"        # DNA/RNA sequences
    DEVICE = "device"                    # Medical device
    SYSTEM = "system"                    # System claims
    USE = "use"                          # Use/application claims
    PRODUCT_BY_PROCESS = "product_by_process"
    UNKNOWN = "unknown"


class ElementType(Enum):
    """Claim element type."""
    PREAMBLE = "preamble"                # Introductory statement
    ACTIVE_STEP = "active_step"          # Method step (e.g., "administering")
    COMPONENT = "component"              # Composition component
    STRUCTURE = "structure"              # Chemical/biological structure
    SEQUENCE = "sequence"                # Amino acid/nucleotide sequence
    PROPERTY = "property"                # Physical/chemical property
    PARAMETER = "parameter"              # Numerical limitation
    RESULT = "result"                    # Functional result/outcome
    CONDITION = "condition"              # Condition or limitation
    REFERENCE = "reference"              # Reference to other claims


@dataclass
class ClaimElement:
    """A parsed element from a patent claim."""

    element_id: str                      # Unique identifier (E1, E2, etc.)
    element_type: ElementType            # Type of element
    text: str                            # Original text
    normalized_text: str                 # Cleaned/normalized text

    # Extracted entities
    compounds: List[str] = field(default_factory=list)      # Chemical names/SMILES
    proteins: List[str] = field(default_factory=list)       # Protein names/sequences
    diseases: List[str] = field(default_factory=list)       # Disease/condition names
    organisms: List[str] = field(default_factory=list)      # Species/organisms
    numerical_limits: List[Dict] = field(default_factory=list)  # Ranges, percentages

    # Relationships
    depends_on: List[str] = field(default_factory=list)     # Element dependencies

    # Search keywords
    keywords: List[str] = field(default_factory=list)

    # Confidence score
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "compounds": self.compounds,
            "proteins": self.proteins,
            "diseases": self.diseases,
            "organisms": self.organisms,
            "numerical_limits": self.numerical_limits,
            "depends_on": self.depends_on,
            "keywords": self.keywords,
            "confidence": self.confidence,
        }


@dataclass
class ParsedClaim:
    """A fully parsed patent claim."""

    claim_number: int
    claim_text: str
    claim_type: ClaimType
    is_independent: bool
    depends_on_claim: Optional[int]      # For dependent claims

    # Parsed elements
    elements: List[ClaimElement]

    # Overall claim analysis
    primary_invention: str               # One-sentence summary
    technical_field: str                 # IPC-style field
    key_limitations: List[str]           # Most important limitations

    # Extracted structures
    smiles: Optional[str] = None         # Chemical structure if present
    sequence: Optional[str] = None       # Sequence if present
    sequence_type: Optional[str] = None  # "protein" or "nucleic_acid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_number": self.claim_number,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type.value,
            "is_independent": self.is_independent,
            "depends_on_claim": self.depends_on_claim,
            "elements": [e.to_dict() for e in self.elements],
            "primary_invention": self.primary_invention,
            "technical_field": self.technical_field,
            "key_limitations": self.key_limitations,
            "smiles": self.smiles,
            "sequence": self.sequence,
            "sequence_type": self.sequence_type,
        }


# LLM Prompts
CLAIM_PARSING_PROMPT = '''You are an expert patent attorney analyzing patent claims. Parse the following patent claim into structured elements.

PATENT CLAIM:
"""
{claim_text}
"""

Analyze this claim and respond with a JSON object containing:

{{
    "claim_type": "<method|composition|compound|antibody|nucleic_acid|device|system|use|unknown>",
    "is_independent": <true|false>,
    "depends_on_claim": <claim_number or null>,
    "primary_invention": "<one sentence describing the core invention>",
    "technical_field": "<technical field, e.g., 'cancer immunotherapy', 'small molecule therapeutics'>",
    "key_limitations": ["<list of 2-4 most important claim limitations>"],
    "elements": [
        {{
            "element_id": "E1",
            "element_type": "<preamble|active_step|component|structure|sequence|property|parameter|result|condition|reference>",
            "text": "<exact text from claim>",
            "normalized_text": "<cleaned version>",
            "compounds": ["<chemical names or SMILES if present>"],
            "proteins": ["<protein names, antibody names, or sequences>"],
            "diseases": ["<disease/condition names>"],
            "keywords": ["<5-10 key search terms>"],
            "depends_on": ["<element_ids this depends on>"]
        }}
    ],
    "smiles": "<SMILES string if chemical structure is claimed, else null>",
    "sequence": "<amino acid or nucleotide sequence if present, else null>",
    "sequence_type": "<protein|nucleic_acid|null>"
}}

Be precise and extract ALL distinct claim elements. Each limitation, step, or component should be a separate element.
Respond with ONLY the JSON object, no other text.'''


ENTITY_EXTRACTION_PROMPT = '''Extract biomedical entities from this patent claim text:

TEXT:
"""
{text}
"""

Respond with a JSON object:
{{
    "compounds": ["<chemical compound names, drug names, SMILES>"],
    "proteins": ["<protein names, antibody names, receptor names>"],
    "diseases": ["<disease names, conditions, indications>"],
    "organisms": ["<species, cell types, organisms>"],
    "mechanisms": ["<biological mechanisms, pathways>"],
    "numerical_values": [
        {{"value": "<number>", "unit": "<unit>", "context": "<what it describes>"}}
    ]
}}

Only include entities that are explicitly mentioned. Respond with ONLY the JSON.'''


class LLMClaimParser:
    """SOTA LLM-based patent claim parser.

    Uses large language models to parse patent claims into structured
    elements, enabling fine-grained prior art analysis.

    Features:
    - Claim type classification
    - Element-level decomposition
    - Biomedical entity extraction
    - Chemical structure detection
    - Sequence identification

    Example:
        ```python
        parser = LLMClaimParser(provider="openai", model="gpt-4")

        claim_text = "A method of treating cancer comprising administering..."
        parsed = parser.parse_claim(claim_text, claim_number=1)

        for element in parsed.elements:
            print(f"{element.element_id}: {element.element_type.value}")
            print(f"  Keywords: {element.keywords}")
        ```
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        llm_provider: Optional[Any] = None,
    ):
        """Initialize the LLM claim parser.

        Args:
            provider: LLM provider name ("openai", "anthropic", "google")
            model: Model name (e.g., "gpt-4", "claude-3-opus-20240229")
            api_key: API key (or uses environment variable)
            temperature: Generation temperature (lower = more deterministic)
            max_retries: Number of retries on failure
            llm_provider: Pre-configured LLMProvider instance (preferred)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._llm = llm_provider
        self._last_response = None

        if self._llm is None:
            self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]) -> None:
        """Initialize the LLM client."""
        try:
            from biopat.llm import create_provider
            self._llm = create_provider(self.provider, model=self.model, api_key=api_key)
        except (ImportError, ValueError):
            # Legacy fallback
            if self.provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response."""
        for attempt in range(self.max_retries):
            try:
                if self._llm is not None:
                    response = self._llm.generate(
                        prompt=prompt,
                        system_prompt="You are an expert patent attorney. Respond only with valid JSON.",
                        max_tokens=4000,
                        temperature=self.temperature,
                    )
                    self._last_response = response
                    return response.text
                elif self.provider == "openai":
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert patent attorney. Respond only with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=4000,
                    )
                    return resp.choices[0].message.content.strip()
                elif self.provider == "anthropic":
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.content[0].text.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise

        return ""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        # Try to extract JSON from response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (code block markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            # Try to fix common issues
            response = response.replace("'", '"')
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)
            return json.loads(response)

    def _determine_claim_type(self, text: str) -> ClaimType:
        """Fallback rule-based claim type detection."""
        text_lower = text.lower()

        if text_lower.startswith(("a method", "a process", "method of", "process of")):
            return ClaimType.METHOD
        elif "antibod" in text_lower or "immunoglobulin" in text_lower:
            return ClaimType.ANTIBODY
        elif "composition" in text_lower or "formulation" in text_lower:
            return ClaimType.COMPOSITION
        elif "compound" in text_lower and ("formula" in text_lower or "structure" in text_lower):
            return ClaimType.COMPOUND
        elif "nucleic acid" in text_lower or "polynucleotide" in text_lower or "dna" in text_lower:
            return ClaimType.NUCLEIC_ACID
        elif "use of" in text_lower or "use in" in text_lower:
            return ClaimType.USE
        elif "system" in text_lower or "apparatus" in text_lower:
            return ClaimType.SYSTEM
        elif "device" in text_lower:
            return ClaimType.DEVICE
        else:
            return ClaimType.UNKNOWN

    def _extract_sequences(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract protein or nucleotide sequences from text."""
        # Look for SEQ ID NO references
        seq_refs = re.findall(r'SEQ ID NO[:\s]*(\d+)', text, re.IGNORECASE)

        # Look for inline sequences (amino acids)
        aa_pattern = r'\b([ACDEFGHIKLMNPQRSTVWY]{10,})\b'
        aa_matches = re.findall(aa_pattern, text)

        # Look for inline sequences (nucleotides)
        nt_pattern = r'\b([ACGTU]{15,})\b'
        nt_matches = re.findall(nt_pattern, text)

        sequence = None
        seq_type = None

        if aa_matches:
            sequence = aa_matches[0]
            seq_type = "protein"
        elif nt_matches:
            sequence = nt_matches[0]
            seq_type = "nucleic_acid"

        return sequence, seq_type

    def _extract_smiles(self, text: str) -> Optional[str]:
        """Extract SMILES strings from text."""
        # Basic SMILES pattern (simplified)
        smiles_pattern = r'\b([A-Za-z0-9@+\-\[\]\(\)=#$\\\/\.]+)\b'

        # Look for explicit SMILES mentions
        smiles_match = re.search(r'SMILES[:\s]*([^\s,;]+)', text, re.IGNORECASE)
        if smiles_match:
            return smiles_match.group(1)

        return None

    def parse_claim(
        self,
        claim_text: str,
        claim_number: int = 1,
        patent_context: Optional[str] = None,
    ) -> ParsedClaim:
        """Parse a patent claim into structured elements.

        Args:
            claim_text: The claim text to parse
            claim_number: Claim number (1 for first claim)
            patent_context: Optional context (title, abstract) for better parsing

        Returns:
            ParsedClaim with structured elements
        """
        logger.info(f"Parsing claim {claim_number}...")

        # Build prompt
        prompt = CLAIM_PARSING_PROMPT.format(claim_text=claim_text)

        if patent_context:
            prompt = f"PATENT CONTEXT:\n{patent_context}\n\n{prompt}"

        # Call LLM
        response = self._call_llm(prompt)

        try:
            data = self._parse_json_response(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response, using fallback")
            data = self._fallback_parse(claim_text)

        # Build ParsedClaim
        claim_type_str = data.get("claim_type", "unknown")
        try:
            claim_type = ClaimType(claim_type_str)
        except ValueError:
            claim_type = self._determine_claim_type(claim_text)

        # Parse elements
        elements = []
        for elem_data in data.get("elements", []):
            try:
                elem_type_str = elem_data.get("element_type", "condition")
                elem_type = ElementType(elem_type_str)
            except ValueError:
                elem_type = ElementType.CONDITION

            element = ClaimElement(
                element_id=elem_data.get("element_id", f"E{len(elements)+1}"),
                element_type=elem_type,
                text=elem_data.get("text", ""),
                normalized_text=elem_data.get("normalized_text", elem_data.get("text", "")),
                compounds=elem_data.get("compounds", []),
                proteins=elem_data.get("proteins", []),
                diseases=elem_data.get("diseases", []),
                organisms=elem_data.get("organisms", []),
                numerical_limits=elem_data.get("numerical_values", []),
                depends_on=elem_data.get("depends_on", []),
                keywords=elem_data.get("keywords", []),
            )
            elements.append(element)

        # Extract sequences if not already found
        sequence = data.get("sequence")
        sequence_type = data.get("sequence_type")
        if not sequence:
            sequence, sequence_type = self._extract_sequences(claim_text)

        # Extract SMILES if not already found
        smiles = data.get("smiles")
        if not smiles:
            smiles = self._extract_smiles(claim_text)

        # Determine if independent
        is_independent = data.get("is_independent", True)
        depends_on_claim = data.get("depends_on_claim")

        # Check for dependency markers
        if re.match(r'^\d+\.\s*(The|A|An)\s+\w+\s+of\s+claim\s+\d+', claim_text, re.IGNORECASE):
            is_independent = False
            dep_match = re.search(r'of\s+claim\s+(\d+)', claim_text, re.IGNORECASE)
            if dep_match:
                depends_on_claim = int(dep_match.group(1))

        return ParsedClaim(
            claim_number=claim_number,
            claim_text=claim_text,
            claim_type=claim_type,
            is_independent=is_independent,
            depends_on_claim=depends_on_claim,
            elements=elements,
            primary_invention=data.get("primary_invention", ""),
            technical_field=data.get("technical_field", ""),
            key_limitations=data.get("key_limitations", []),
            smiles=smiles,
            sequence=sequence,
            sequence_type=sequence_type,
        )

    def _fallback_parse(self, claim_text: str) -> Dict[str, Any]:
        """Fallback rule-based parsing when LLM fails."""
        # Split on common delimiters
        parts = re.split(r'[;,]\s*(?:and\s+)?(?=\w)', claim_text)

        elements = []
        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) < 10:
                continue

            # Determine element type
            part_lower = part.lower()
            if any(word in part_lower for word in ["comprising", "consisting", "including"]):
                elem_type = "preamble"
            elif any(word in part_lower for word in ["administering", "contacting", "treating", "mixing"]):
                elem_type = "active_step"
            elif any(word in part_lower for word in ["wherein", "where", "such that"]):
                elem_type = "condition"
            else:
                elem_type = "component"

            # Extract keywords
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]{2,}\b', part)
            stopwords = {'the', 'and', 'for', 'with', 'from', 'wherein', 'comprising', 'consisting'}
            keywords = [w.lower() for w in words if w.lower() not in stopwords][:10]

            elements.append({
                "element_id": f"E{i+1}",
                "element_type": elem_type,
                "text": part[:200],
                "normalized_text": part[:200],
                "compounds": [],
                "proteins": [],
                "diseases": [],
                "keywords": keywords,
                "depends_on": [],
            })

        return {
            "claim_type": self._determine_claim_type(claim_text).value,
            "is_independent": True,
            "depends_on_claim": None,
            "primary_invention": claim_text[:200] + "...",
            "technical_field": "biomedical",
            "key_limitations": [],
            "elements": elements,
            "smiles": None,
            "sequence": None,
            "sequence_type": None,
        }

    def parse_all_claims(
        self,
        claims: List[str],
        patent_context: Optional[str] = None,
    ) -> List[ParsedClaim]:
        """Parse all claims from a patent.

        Args:
            claims: List of claim texts
            patent_context: Optional context for better parsing

        Returns:
            List of ParsedClaim objects
        """
        parsed_claims = []

        for i, claim_text in enumerate(claims, 1):
            try:
                parsed = self.parse_claim(
                    claim_text,
                    claim_number=i,
                    patent_context=patent_context,
                )
                parsed_claims.append(parsed)
            except Exception as e:
                logger.error(f"Failed to parse claim {i}: {e}")
                continue

        return parsed_claims

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract biomedical entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary of entity types to entity lists
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        response = self._call_llm(prompt)

        try:
            return self._parse_json_response(response)
        except json.JSONDecodeError:
            return {
                "compounds": [],
                "proteins": [],
                "diseases": [],
                "organisms": [],
                "mechanisms": [],
                "numerical_values": [],
            }


def create_claim_parser(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClaimParser:
    """Factory function for claim parser.

    Args:
        provider: "openai" or "anthropic"
        model: Model name (defaults to gpt-4 or claude-3-opus)
        api_key: API key

    Returns:
        Configured LLMClaimParser
    """
    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-opus-20240229"

    return LLMClaimParser(
        provider=provider,
        model=model,
        api_key=api_key,
    )
