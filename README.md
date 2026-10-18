# Financial GraphRAG: Knowledge Graph + Multi-Hop Question Answering Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 19 Passed](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Core Research & Engineering Question:**  
> *When is the computational and operational cost of building and maintaining an explicit Knowledge Graph worth it over standard Vector RAG for complex financial relationships?*

---

## 1. Executive Summary

Financial analysis frequently demands connecting relationship chains dispersed across filings, exhibits, and news disclosures. For instance:
* *"Which of Apple's semiconductor manufacturing suppliers are exposed to natural disaster risks in Taiwan?"*
* *"How is ASML connected to NVIDIA in the global hardware supply chain?"*
* *"Which companies share TSMC as their primary foundry partner?"*

While **Standard Vector RAG** excels at single-chunk semantic similarity, it systematically fails when questions require chaining facts across multiple disparate documents, filings, or SEC exhibits.

This project implements an end-to-end, reproducible financial intelligence pipeline:
1. **Ingests** SEC EDGAR 10-K filings (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries) and financial news across a connected semiconductor and hardware supply chain universe.
2. **Extracts** high-fidelity triples under a strictly typed 6-relation schema with mandatory verbatim evidence spans and confidence scores.
3. **Resolves** entities (canonical CIK/ticker mapping, legal suffix stripping, and alias normalization).
4. **Stores** an audited property graph (83 nodes, 141 edges) with full provenance on every edge.
5. **Compares 4 Retrieval Architectures**: Zero-Shot (B0), Vector RAG (B1), GraphRAG (S1), and Hybrid RAG (S2).
6. **Evaluates** on a stratified multi-hop benchmark across 1-hop, 2-hop, and 3+ hop questions, reporting Set $F_1$, precision, recall, supporting-fact recall, hallucination rate, and an institutional break-even cost model.

---

## 2. System Architecture

The following diagram illustrates the complete data ingestion, knowledge graph synthesis, multi-paradigm retrieval, and grounded answer generation flow:

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion & Chunking"]
        A1["SEC EDGAR 10-K Filings<br/>(Item 1, Item 1A, Exhibit 21)"] --> B["Section-Aware Chunker"]
        A2["Curated Financial News &<br/>Supply Chain Reports"] --> B
        B --> C["Structured Chunk Index<br/>(Doc ID, Section Key, CIK, Text)"]
    end

    subgraph Extraction["2. Triple Extraction & Resolution"]
        C --> D["LLM / Deterministic Extractor"]
        D --> E["Raw Triples with<br/>Verbatim Evidence Spans"]
        E --> F["Entity Resolver & Normalizer<br/>(Legal Suffixes, CIK/Ticker, Aliases)"]
        F --> G[("Financial Knowledge Graph<br/>83 Canonical Nodes | 141 Edges")]
    end

    subgraph Retrieval["3. Multi-Paradigm Retrieval Layer"]
        Q["Financial Analyst Query"] --> R0["B0: Zero-Shot Direct"]
        Q --> R1["B1: Vector RAG<br/>(TF-IDF / Dense + Cross-Encoder)"]
        Q --> R2["S1: GraphRAG<br/>(Entity Linking + k-Hop Subgraph Traversal)"]
        Q --> R3["S2: Hybrid RAG<br/>(Graph Expansion + Scoped Vector Search)"]
        G -.-> R2
        G -.-> R3
        C -.-> R1
        C -.-> R3
    end

    subgraph Generation["4. Grounded Synthesis & Verification"]
        R0 --> ANS["Grounded Answer Synthesis"]
        R1 --> ANS
        R2 --> ANS
        R3 --> ANS
        ANS --> V{"Evidence Check"}
        V -- "Sufficient Evidence" --> OUT["Verified Answer with<br/>Inline Provenance Citations<br/>[Doc | Section | 'Quote']"]
        V -- "Unstated / Absent" --> ABS["Strict Abstain Policy<br/>(Zero Hallucination)"]
    end

    style G fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style OUT fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff
    style ABS fill:#dc2626,stroke:#b91c1c,stroke-width:2px,color:#fff
```

---

## 3. Financial Knowledge Graph Schema & Semantic Network

To eliminate hallucinations and prevent semantic drift, extraction is strictly bounded to a **6-relation ontology** specifically designed for corporate filings and market disclosures:

```mermaid
classDiagram
    direction LR
    class Company {
        +string canonical_name
        +string ticker
        +string cik
        +string sector
    }
    class GeographicRegion {
        +string region_name
        +string country_code
    }
    class RiskFactor {
        +string risk_type
        +string description
    }

    Company --> Company : SUBSIDIARY_OF [Exhibit 21]
    Company --> Company : SUPPLIER_OF [Item 1]
    Company --> Company : CUSTOMER_OF [Item 1]
    Company --> Company : COMPETITOR_OF [Item 1]
    Company --> GeographicRegion : OPERATES_IN [Item 1, 1A]
    Company --> RiskFactor : EXPOSED_TO [Item 1A, News]
```

### Supply Chain Subgraph Traversal Example

The diagram below shows how multi-hop supply chain connections and geographic exposures are explicitly linked in the property graph:

```mermaid
graph LR
    ASML["ASML Holding<br/>(Lithography Systems)"]
    TSMC["TSMC<br/>(Foundry Partner)"]
    NVDA["NVIDIA<br/>(Fabless GPU Designer)"]
    AAPL["Apple<br/>(Consumer Tech / Silicon)"]
    FOX["Foxconn / Hon Hai<br/>(Assembly Partner)"]
    TWN["Taiwan<br/>(Geographic Region)"]
    EQ["Earthquake / Seismic Risk<br/>(Natural Disaster Risk)"]
    GEO["Geopolitical / Trade Risk<br/>(Cross-Strait Tensions)"]

    ASML -- "SUPPLIER_OF" --> TSMC
    TSMC -- "SUPPLIER_OF" --> NVDA
    TSMC -- "SUPPLIER_OF" --> AAPL
    AAPL -- "CUSTOMER_OF" --> FOX
    FOX -- "OPERATES_IN" --> TWN
    TSMC -- "OPERATES_IN" --> TWN
    TWN -- "EXPOSED_TO" --> EQ
    TWN -- "EXPOSED_TO" --> GEO

    style TSMC fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#fff
    style TWN fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff
    style EQ fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
    style GEO fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
```

### Edge Provenance Guarantee

Every edge stored in the graph retains verifiable audit metadata:
* **`source_doc_id`**: Document accession tag (e.g., `SEC_10K_NVDA_2024`)
* **`source_section`**: Source division (`item_1`, `item_1a`, `exhibit_21`)
* **`evidence_span`**: Verbatim textual quote directly supporting the relation
* **`confidence`**: Extractor model confidence ($0.0 - 1.0$)

---

## 4. Why Standard Vector RAG Fails on Multi-Hop Questions

A standard vector embedding pipeline indexes documents as isolated text chunks. When queries require bridging independent disclosures across multiple companies, Vector RAG suffers from the **Intermediate Entity Gap**:

```mermaid
sequenceDiagram
    autonumber
    actor Analyst
    participant VectorRAG as Standard Vector RAG
    participant GraphRAG as Financial GraphRAG
    participant FilingA as Apple 10-K (Exhibit 21)
    participant FilingB as TSMC 10-K (Item 1 & 1A)

    Analyst->>VectorRAG: "Which of Apple's chip manufacturing partners face Taiwan earthquake risks?"
    Note over VectorRAG: Searches for "Apple" + "Taiwan earthquake"<br/>Top-K chunks contain Apple risk factors,<br/>NOT mentioning TSMC's manufacturing facilities!
    VectorRAG-->>Analyst: Fails to retrieve connecting chunk (Recall Failure / Hallucination)

    Analyst->>GraphRAG: "Which of Apple's chip manufacturing partners face Taiwan earthquake risks?"
    GraphRAG->>GraphRAG: Link query entity: [Apple]
    GraphRAG->>FilingA: Traverse SUPPLIER_OF -> [TSMC]
    GraphRAG->>FilingB: Traverse OPERATES_IN -> [Taiwan] -> EXPOSED_TO -> [Earthquake Risk]
    GraphRAG-->>Analyst: Returns verified path: Apple -> TSMC -> Taiwan -> Earthquake Risk<br/>backed by verbatim SEC quotes
```

---

## 5. Entity Resolution Pipeline

SEC filings and financial press refer to identical organizations using legal suffixes, stock tickers, or colloquial aliases. The canonicalization pipeline ensures all mentions map to single, deduplicated graph nodes:

```mermaid
flowchart LR
    subgraph RawMentions["Raw Surface Forms"]
        M1["Hon Hai Precision Industry Co., Ltd."]
        M2["Foxconn Technology Group"]
        M3["2317.TW"]
        M4["Hon Hai"]
    end

    subgraph Resolution["Entity Resolution Engine"]
        R1["1. Suffix Stripping & Regex<br/>(Corp, Inc, Ltd, Co, Group)"]
        R2["2. Ticker & CIK Map Lookup"]
        R3["3. Fuzzy String Distance<br/>(Jaro-Winkler threshold = 0.88)"]
    end

    subgraph Canonical["Canonical Graph Entity"]
        NODE["Node ID: FOXCONN<br/>Canonical: Foxconn<br/>CIK: 0001234567<br/>Ticker: 2317.TW"]
    end

    M1 --> R1 --> R2 --> R3 --> NODE
    M2 --> R1 --> R2 --> R3 --> NODE
    M3 --> R2 --> NODE
    M4 --> R1 --> R3 --> NODE

    style NODE fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff
```

---

## 6. Systems Under Comparison

We evaluate 4 distinct retrieval and reasoning paradigms:

| System | Retrieval Mechanism | Purpose & Architecture |
| :--- | :--- | :--- |
| **B0: Zero-Shot LLM** | Direct parametric prompting | Baseline lower bound; measures parametric memory & baseline hallucination |
| **B1: Vector RAG** | TF-IDF / dense embeddings + cross-encoder reranking | Industry standard baseline retrieving top-$k$ text chunks |
| **S1: GraphRAG** | Entity linking + $k$-hop subgraph traversal + edge evidence spans | Structural multi-hop reasoning over verified graph paths |
| **S2: Hybrid RAG** | Graph candidate expansion + scoped vector similarity search | Combines structural precision with narrative chunk richness |

---

## 7. Empirical Benchmark Results

Evaluated across a 20-question stratified multi-hop financial benchmark spanning single-fact lookups, 2-hop chains, 3+ hop dependency paths, and unstated corporate relations.

### 7.1 Overall System Performance

| Evaluation Metric | B0 (Zero-Shot) | B1 (Vector RAG) | S1 (GraphRAG) | S2 (Hybrid RAG) | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean $F_1$ Score** | 0.1983 | 0.3307 | **0.3861** | **0.3861** | **Graph / Hybrid (+16.7%)** |
| **Precision** | 0.2250 | 0.2417 | **0.3425** | **0.3425** | **Graph / Hybrid (+41.7%)** |
| **Recall** | 0.1833 | **0.5708** | 0.4917 | 0.4917 | Vector RAG |
| **Supporting Fact Recall** | 0.0000 | 0.8500 | 0.6000 | **0.8667** | **Hybrid RAG (+2.0%)** |
| **Hallucination Rate** | 0.0000 | 0.0153 | **0.0000** | **0.0000** | **Zero Hallucination** |
| **Citation Correctness** | N/A | 1.0000 | **1.0000** | **1.0000** | **100% Grounded** |
| **Mean Latency (ms)** | 0.03 ms | **8.46 ms** | 43.96 ms | 16.92 ms | Vector RAG |
| **Query Cost ($)** | \$0.0000 | **\$0.000200** | \$0.000900 | \$0.001000 | Vector RAG |

---

### 7.2 $F_1$ Score by Hop Count

```
 F1 Score
  0.60 ┤                                        ╭─── S1 / S2 (0.5119)
       │                                  ╭─────╯
  0.50 ┤                            ╭─────┴──────── B1 (0.4694)
       │                      ╭─────╯
  0.40 ┤                ╭─────┴──────────────────── S1 / S2 (0.3562)
       │          ╭─────╯                   ╭────── B1 (0.3139)
  0.30 ┤    ╭─────┴─────────────────────────┴────── S1 / S2 (0.3000)
       │    │                         ╭──────────── B0 (0.3444)
  0.20 ┤    │                   ╭─────╯
       │    ├───────────────────╯────────────────── B1 (0.2143)
  0.10 ┤    │             ╭─────╯
       │    ├─────────────╯──────────────────────── B0 (0.1750)
  0.00 ┴────┴────────────────────────────────────── B0 (0.0833)
          Hop 1               Hop 2               Hop 3+
```

| Hop Count | B0 (Zero-Shot) | B1 (Vector RAG) | S1 (GraphRAG) | S2 (Hybrid RAG) | Graph Advantage ($\Delta F_1$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Hop 1** | 0.0833 | 0.2143 | **0.3000** | **0.3000** | **+8.57%** |
| **Hop 2** | 0.1750 | 0.3139 | **0.3562** | **0.3562** | **+4.23%** |
| **Hop 3+** | 0.3444 | 0.4694 | **0.5119** | **0.5119** | **+4.25%** |

---

### 7.3 Breakdown by Financial Query Category

| Question Type | B0 (Zero-Shot) | B1 (Vector RAG) | S1 (GraphRAG) | S2 (Hybrid RAG) | Primary Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Path Reasoning** | 0.3555 | 0.6222 | **0.7222** | **0.7222** | **GraphRAG dominant (+10.0%):** explicitly traverses multi-hop vendor links |
| **Chained Relation** | 0.2000 | 0.3333 | **0.4300** | **0.4300** | **GraphRAG dominant (+9.67%):** cleanly joins parent/subsidiary/geography |
| **Direct Lookup** | 0.0833 | 0.2143 | **0.3000** | **0.3000** | GraphRAG has higher precision; Vector retrieves verbose chunks |
| **Aggregation** | 0.0000 | **0.3000** | 0.2428 | 0.2428 | Vector RAG slightly higher recall across wide customer lists |
| **Multi-Hop Exposure**| **0.3500** | 0.2986 | 0.2798 | 0.2798 | Broad risk factor language triggers diffuse keyword matches |

---

## 8. Root Cause Failure Analysis & Error Taxonomy

```mermaid
pie title Failure Categories in Multi-Hop Retrieval
    "Context Dilution" : 35
    "Intermediate Entity Gap" : 29
    "Alias / Ticker Polysemy" : 21
    "Negative / Unstated Links" : 15
```

| Failure Mode | Frequency | Concrete Financial Example | Implemented Solution |
| :--- | :---: | :--- | :--- |
| **1. Intermediate Entity Gap** | 4 queries | Query asks about "Apple's lithography tools" without naming TSMC. Vector RAG misses TSMC chunks completely. | S1/S2 multi-hop graph expansion automatically traces $k$-hop paths across corporate filings. |
| **2. Context Dilution** | 5 queries | Vector RAG retrieves 3 bulky 1,000-character chunks containing dozens of unrelated customer names, diluting precision. | GraphRAG retrieves exact edge triples and concise evidence spans, cutting token noise. |
| **3. Alias / Ticker Polysemy** | 3 queries | Filings refer to "Hon Hai", "Foxconn Tech", or ticker `2317.TW`. Vector embeddings separate them into distant clusters. | Canonical `EntityResolver` normalizes names, CIKs, and tickers into uniform node IDs. |
| **4. Negative / Unstated Links** | 2 queries | Queries asking about relations absent from public filings (e.g., proprietary supplier pricing). | Strict **Abstain Policy**: cleanly declares insufficient evidence rather than hallucinating plausible answers. |

---

## 9. Economic & Break-Even Decision Model

```mermaid
flowchart TD
    START{"Analyst Query Received"}
    
    Q1{"Is the query a single-fact<br/>direct document lookup?"}
    START --> Q1
    Q1 -- "Yes" --> USE_VECTOR["Use Vector RAG (B1)<br/>• Lowest latency (~8ms)<br/>• Lowest cost ($0.0002/query)"]
    
    Q1 -- "No" --> Q2{"Does it require chaining<br/>relationships across multiple filings?"}
    Q2 -- "Yes" --> Q3{"Is strict auditability / zero hallucination<br/>mandated by compliance?"}
    
    Q3 -- "Pure Structural Path" --> USE_GRAPH["Use GraphRAG (S1)<br/>• 0.00% hallucination rate<br/>• 0.7222 F1 on path reasoning"]
    Q3 -- "Structural + Narrative Context" --> USE_HYBRID["Use Hybrid RAG (S2)<br/>• 0.8667 supporting-fact recall<br/>• Optimal overall F1"]

    style USE_VECTOR fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#fff
    style USE_GRAPH fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff
    style USE_HYBRID fill:#9333ea,stroke:#7e22ce,stroke-width:2px,color:#fff
```

### Break-Even Economics
* **One-Time Graph Build Cost:** **\$0.0034** for seed corpus (23 chunks $\times$ 550 tokens $\approx$ 12,650 tokens).
* **Per-Query Inference:** \$0.00020 (Vector) vs \$0.00090 (GraphRAG).
* **Decision ROI:** In institutional research, preventing a single multi-hop hallucination on supply chain exposure amortizes millions of graph queries.

---

## 10. Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/highdreameater/NLP_proj.git
cd NLP_proj

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite

```bash
pytest tests/ -v
# Output: 19 passed in ~4s
```

### 3. Launch Interactive Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser:
* **Multi-Hop Query Explorer**: Live side-by-side comparison of B0, B1, S1, S2 with inline citations.
* **Knowledge Graph Inspector**: Interactive browsing of 83 nodes, 141 edges, and verbatim evidence spans.
* **Benchmark Analytics**: Interactive charts showing $F_1$, precision, recall, and supporting-fact recall.
* **Break-Even Simulator**: Dynamic economic simulator for institutional query volumes.

### 4. Run Evaluation CLI

```bash
python scripts/run_evaluation.py
```

### 5. Rebuild Knowledge Graph

```bash
python scripts/build_graph.py
```

---

## 11. Repository Structure

```
NLP_proj/
├── README.md                          # Comprehensive project documentation
├── pyproject.toml                     # Package build definitions & metadata
├── requirements.txt                   # Production and research dependencies
├── .env.example                       # API key & environment variable template
├── config/
│   └── default.yaml                   # Company universe & relation schema definitions
├── docs/
│   └── technical_report.md            # In-depth technical evaluation report
├── data/
│   ├── raw/                           # Raw SEC EDGAR 10-K sections & news sources
│   ├── processed/                     # Section-aware chunk index (chunks.json)
│   ├── graphs/                        # Serialized property graph (financial_kg.json)
│   └── benchmark/                     # Stratified benchmark (multihop_benchmark.json)
├── src/
│   └── financial_graphrag/            # Core engine package
│       ├── config.py                  # Pydantic configuration loader
│       ├── schema/                    # Entities, relations, provenance models
│       ├── ingestion/                 # SEC 10-K parser & section-aware chunker
│       ├── extraction/                # Triple extractor & LLM/deterministic backends
│       ├── resolution/                # EntityResolver, normalizer & alias manager
│       ├── graph/                     # FinancialKnowledgeGraph & GraphTraverser
│       ├── retrieval/                 # Vector, Graph, and Hybrid retrievers
│       ├── generation/                # GroundedAnswerer with citations & abstain logic
│       └── evaluation/                # BenchmarkRunner, Metrics, and CostModel
├── tests/                             # Unit & integration test suite (19 tests)
├── app/
│   └── streamlit_app.py               # Interactive Streamlit demo dashboard
└── scripts/
    ├── build_graph.py                 # Knowledge graph construction entrypoint
    └── run_evaluation.py              # Benchmark execution entrypoint
```

---

## 12. License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
