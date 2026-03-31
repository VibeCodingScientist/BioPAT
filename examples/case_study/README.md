# BioPAT Case Study: Cancer Immunotherapy Patents

This case study demonstrates the BioPAT benchmark data format using a realistic mock dataset focused on **cancer immunotherapy patents** and their cited scientific literature.

## Overview

The dataset simulates prior art search for 10 patent queries against a corpus of 50 scientific articles. It demonstrates:

- **Trimodal data**: Text descriptions, chemical structures (SMILES), and protein sequences
- **Graded relevance**: 4-level relevance scale (0-3) based on citation type and similarity
- **Temporal ordering**: Patents cite articles published before the priority date

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Queries (Patents) | 10 |
| Corpus (Articles) | 50 |
| Total Judgments | 85 |
| Avg. Relevant/Query | 8.5 |

### Relevance Distribution

| Score | Meaning | Count |
|-------|---------|-------|
| 3 | Novelty-destroying (examiner cited) | 12 |
| 2 | Highly relevant (applicant cited) | 28 |
| 1 | Relevant (semantic similarity) | 45 |
| 0 | Not relevant | (implicit) |

## File Formats

### corpus.jsonl
```json
{"_id": "W2100000001", "title": "...", "text": "...", "metadata": {...}}
```

Each document contains:
- `_id`: OpenAlex Work ID
- `title`: Article title
- `text`: Abstract text
- `metadata`: Additional fields (DOI, journal, publication date, MeSH terms)

### queries.jsonl
```json
{"_id": "US10500001", "text": "...", "metadata": {...}}
```

Each query contains:
- `_id`: Patent ID
- `text`: Combined title + abstract + claims
- `metadata`: Priority date, IPC codes, chemicals (SMILES), sequences

### qrels.tsv
```
query_id    doc_id    relevance
US10500001  W2100000001    3
```

Tab-separated relevance judgments (TREC format).

## Loading the Data

```python
import json

# Load corpus
corpus = {}
with open("corpus.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = doc

# Load queries
queries = {}
with open("queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q

# Load qrels
qrels = {}
with open("qrels.tsv") as f:
    next(f)  # Skip header
    for line in f:
        qid, did, rel = line.strip().split("\t")
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = int(rel)

print(f"Corpus: {len(corpus)} documents")
print(f"Queries: {len(queries)} patents")
print(f"Judgments: {sum(len(v) for v in qrels.values())}")
```

## Example Query

**Patent US10500001**: *Anti-PD-1 antibody for treating melanoma*

```
Title: Humanized anti-PD-1 monoclonal antibody and methods of use

Abstract: The present invention relates to humanized monoclonal antibodies
that specifically bind to human PD-1 (programmed death-1) receptor...

Claims: 1. A humanized monoclonal antibody comprising a heavy chain variable
region with CDR sequences SEQ ID NO:1, SEQ ID NO:2, and SEQ ID NO:3...
```

**Relevant Prior Art (Score=3)**:
- W2100000005: "PD-1 blockade enhances anti-tumor immunity in melanoma models"

**Chemical Structure** (from patent):
```
SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)NC2=CC=C(C=C2)NC(=O)C3=CC=NC=C3
```

**Protein Sequence** (from patent):
```
>SEQ_001 Anti-PD-1 Heavy Chain Variable Region
QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGEINPSNGGTNFNEKFKS
```

## Evaluation

### Basic BM25 Baseline

Run basic retrieval evaluation:

```bash
python evaluate_case_study.py
```

### Full Novelty Assessment Demo

**This is the main demo** - it shows the complete AI agent workflow for patent novelty assessment:

```bash
python novelty_assessment_demo.py
python novelty_assessment_demo.py --patent US10500003  # Small molecule patent
python novelty_assessment_demo.py --patent US10500001  # Antibody patent
```

The novelty assessment demo performs:

1. **Claim Decomposition** - Parses patent claims into searchable elements
2. **Trimodal Search** - Searches via text, chemical structure, AND sequence similarity
3. **Threat Assessment** - Rates each prior art reference (CRITICAL → LOW)
4. **Element Mapping** - Shows which claim elements each reference addresses
5. **Novelty Determination** - Provides overall assessment (ANTICIPATED, OBVIOUS, NOVEL)
6. **Recommendations** - Suggests next steps based on findings

**Example Output:**
```
======================================================================
BIOPAT NOVELTY ASSESSMENT REPORT
======================================================================

Patent ID: US10500003
Claim: Small molecule inhibitor of PD-1/PD-L1 interaction...

----------------------------------------------------------------------
CLAIM DECOMPOSITION
----------------------------------------------------------------------
  [E1] SMALL_MOLECULE: inhibitor of PD-1/PD-L1 interaction...
  [E2] COMPOSITION: compounds restore T cell function...
  [E6] CHEMICAL_STRUCTURE: CC1=NC(=C(C=C1)C2=CN=C(N2)...

----------------------------------------------------------------------
PRIOR ART ANALYSIS
----------------------------------------------------------------------
Doc ID          Threat     Modality   Elements   Score
-----------------------------------------------------
W2100000009     CRITICAL   text       E1,E2,E6   0.780
W2100000024     MODERATE   chemical   E1,E2,E3   0.753

======================================================================
NOVELTY ASSESSMENT
======================================================================

Status: ⚠️  LIKELY ANTICIPATED

Reasoning:
  Found 1 critically relevant reference(s).
  Primary concern: W2100000009 - Small molecule inhibitors of PD-1/PD-L1...

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
  1. Review cited prior art for potential distinguishing features
  2. Consider narrowing claim scope to exclude anticipated subject matter
  3. Identify any novel aspects not present in prior art
```

## What BioPAT Benchmarks

BioPAT measures an AI agent's ability to perform **patent novelty assessment** - the task of determining whether a patent claim is novel over prior art. This is a complex, multi-step reasoning task that requires:

### The Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PATENT NOVELTY ASSESSMENT                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. CLAIM PARSING                                                       │
│     "A method of treating melanoma comprising administering a           │
│      humanized anti-PD-1 antibody with SEQ ID NO:1..."                  │
│                    ↓                                                    │
│     Elements: [anti-PD-1] [melanoma] [humanized] [SEQ ID NO:1]          │
│                                                                         │
│  2. TRIMODAL SEARCH                                                     │
│     ┌──────────┐  ┌──────────────┐  ┌──────────────┐                    │
│     │   TEXT   │  │   CHEMICAL   │  │   SEQUENCE   │                    │
│     │  "PD-1"  │  │ Tanimoto sim │  │ BLAST align  │                    │
│     │"melanoma"│  │ to SMILES    │  │ to SEQ ID    │                    │
│     └────┬─────┘  └──────┬───────┘  └──────┬───────┘                    │
│          └───────────────┼─────────────────┘                            │
│                          ↓                                              │
│                   SCORE FUSION                                          │
│                                                                         │
│  3. PRIOR ART MAPPING                                                   │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ CLAIM ELEMENT        PRIOR ART              THREAT          │     │
│     ├─────────────────────────────────────────────────────────────┤     │
│     │ "anti-PD-1"          W2100000001            🔴 CRITICAL     │     │
│     │ "melanoma"           W2100000005            🔴 CRITICAL     │     │
│     │ "humanized"          W2100000025            🟡 MODERATE     │     │
│     │ "SEQ ID NO:1"        95% identity to W05    🔴 CRITICAL     │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  4. NOVELTY REASONING                                                   │
│     "Claim ANTICIPATED by W2100000005 which discloses anti-PD-1         │
│      antibody for melanoma with 95% sequence identity to claimed        │
│      sequence. The humanization technique is taught by W2100000025."    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

| Capability | Why It's Hard | What BioPAT Measures |
|------------|---------------|----------------------|
| **Claim parsing** | Legal language is complex | Can agent identify key limitations? |
| **Trimodal search** | Must search text, molecules, sequences | Does agent use all modalities? |
| **Relevance assessment** | Must understand technical overlap | Can agent rate prior art threats? |
| **Legal reasoning** | Anticipation vs obviousness | Does agent apply correct standard? |
| **Explanation** | Must justify conclusions | Can agent explain its reasoning? |

### Benchmark Metrics

BioPAT evaluates agents on:
- **Recall@100**: Does the agent find all relevant prior art?
- **NDCG@10**: Does the agent rank the most threatening references highest?
- **Claim coverage**: Does the agent map prior art to specific claim elements?
- **Novelty accuracy**: Does the agent's conclusion match expert assessment?

## Use Cases

This case study demonstrates:

1. **Text-based retrieval**: Finding articles by semantic similarity to patent text
2. **Chemical structure search**: Finding articles describing similar compounds
3. **Sequence similarity**: Finding articles about related proteins/antibodies
4. **Citation verification**: Validating that examiner/applicant citations are retrieved
5. **Novelty reasoning**: Explaining why prior art threatens specific claim elements

## Notes

- All data is synthetic/mock for demonstration purposes
- Real BioPAT datasets contain thousands of patents and millions of articles
- Chemical SMILES and protein sequences are simplified examples
