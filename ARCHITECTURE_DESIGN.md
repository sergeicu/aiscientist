# AI Scientist - System Architecture & Implementation Design

## Executive Summary

This document outlines the comprehensive architecture for the AI Scientist platform - a multi-agent research intelligence system for analyzing biomedical literature and clinical trials from institutional affiliations.

**Current State**: Clinical trial classifier with structured output (Phase 1 complete)
**Target State**: Full research intelligence platform with scraping, analysis, multi-agent evaluation, and visualization

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Implementation Phases](#implementation-phases)
4. [Module Specifications](#module-specifications)
5. [Multi-Agent System Design](#multi-agent-system-design)
6. [Data Flow & Pipelines](#data-flow--pipelines)
7. [Technology Stack](#technology-stack)
8. [API Integrations](#api-integrations)

---

## System Overview

### Core Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI SCIENTIST PLATFORM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA ACQUISITION                                            │
│     ├─ PubMed Scraper (by affiliation)                         │
│     ├─ ClinicalTrials.gov MCP Integration                      │
│     └─ Author Network Extractor                                │
│                                                                  │
│  2. DATA PROCESSING & STRUCTURING                               │
│     ├─ Supervised Classification (clinical trials, phases)      │
│     ├─ Unsupervised Clustering (embeddings, UMAP)              │
│     ├─ Entity Extraction (authors, institutions, outcomes)      │
│     └─ Knowledge Graph Construction                             │
│                                                                  │
│  3. INTELLIGENT ANALYSIS                                        │
│     ├─ Multi-Agent Investment Evaluation                        │
│     ├─ Prior Art Research Engine                                │
│     ├─ Research Direction Discovery                             │
│     └─ Hypothesis Generation                                    │
│                                                                  │
│  4. VISUALIZATION & INSIGHTS                                    │
│     ├─ Author Collaboration Networks (Neo4j)                    │
│     ├─ Geographic Mapping (Mapbox)                              │
│     ├─ Research Landscape UMAP                                  │
│     └─ Interactive Dashboards                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### Layer 1: Data Acquisition (Scrapers & APIs)

```
┌──────────────────────────────────────────────────────────┐
│                    DATA SOURCES                           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  PubMed/NCBI           ClinicalTrials.gov      ORCID     │
│     │                         │                   │       │
│     v                         v                   v       │
│  ┌─────────┐           ┌──────────┐        ┌─────────┐  │
│  │ PubMed  │           │ CT.gov   │        │  ORCID  │  │
│  │ Scraper │           │   MCP    │        │   API   │  │
│  └─────────┘           └──────────┘        └─────────┘  │
│       │                      │                   │       │
│       └──────────┬───────────┴───────────────────┘       │
│                  v                                        │
│         ┌────────────────┐                               │
│         │  Data Lake     │                               │
│         │  (Raw JSON/XML)│                               │
│         └────────────────┘                               │
└──────────────────────────────────────────────────────────┘
```

**Module: `src/scrapers/`**

- `pubmed_scraper.py` - Query PubMed by affiliation, bulk download
- `clinical_trials_scraper.py` - MCP integration for ClinicalTrials.gov
- `author_network_scraper.py` - Extract co-authorship data
- `rate_limiter.py` - Respect API limits, exponential backoff

### Layer 2: Data Processing & Embedding

```
┌──────────────────────────────────────────────────────────┐
│              PROCESSING PIPELINE                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Raw Data                                                │
│     │                                                     │
│     v                                                     │
│  ┌──────────────────────┐                               │
│  │  Parsers & Validators│                               │
│  │  - XML → Structured  │                               │
│  │  - Field Extraction  │                               │
│  └──────────────────────┘                               │
│            │                                              │
│     ┌──────┴──────┐                                      │
│     v             v                                       │
│  ┌─────────┐  ┌──────────────┐                          │
│  │Supervised│  │ Unsupervised │                          │
│  │Classifier│  │  Embeddings  │                          │
│  │          │  │              │                          │
│  │-Clinical │  │- SBERT/E5    │                          │
│  │ Trial?   │  │- UMAP/t-SNE  │                          │
│  │-Phase    │  │- Clustering  │                          │
│  │-Outcomes │  │  (HDBSCAN)   │                          │
│  └─────────┘  └──────────────┘                          │
│       │              │                                    │
│       v              v                                    │
│  ┌──────────────────────────┐                           │
│  │   Structured Database    │                           │
│  │   - PostgreSQL (meta)    │                           │
│  │   - ChromaDB (vectors)   │                           │
│  │   - Neo4j (graph)        │                           │
│  └──────────────────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

**Module: `src/processing/`**

- `embedder.py` - Generate embeddings (SBERT, PubMedBERT, E5)
- `clustering.py` - UMAP dimension reduction, HDBSCAN clustering
- `entity_extractor.py` - NER for authors, institutions, interventions
- `knowledge_graph.py` - Build Neo4j graph (authors, papers, institutions)

### Layer 3: Multi-Agent Intelligence System

```
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-AGENT ORCHESTRATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────────────┐                      │
│                    │ Coordinator     │                      │
│                    │ Agent           │                      │
│                    └────────┬────────┘                      │
│                             │                                │
│         ┌───────────────────┼───────────────────┐           │
│         │                   │                   │           │
│         v                   v                   v           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ Investment   │   │ Prior Art    │   │ Hypothesis   │   │
│  │ Evaluator    │   │ Researcher   │   │ Generator    │   │
│  │ Agent        │   │ Agent        │   │ Agent        │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                   │                   │           │
│         v                   v                   v           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shared Context & Memory                 │  │
│  │  - Vector Store (semantic search)                    │  │
│  │  - Working Memory (conversation history)             │  │
│  │  - Long-term Memory (findings database)              │  │
│  └──────────────────────────────────────────────────────┘  │
│                             │                                │
│                             v                                │
│                    ┌─────────────────┐                      │
│                    │ Report Generator│                      │
│                    │ (Markdown/PDF)  │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**Module: `src/agents/`**

- `coordinator.py` - Orchestrate multi-agent workflows
- `investment_evaluator.py` - Tech transfer/VC perspective
- `prior_art_researcher.py` - Patent search, competitive analysis
- `hypothesis_generator.py` - Identify research gaps, propose directions
- `memory_manager.py` - Shared context, conversation history

### Layer 4: Visualization & Interaction

```
┌──────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Neo4j        │  │ Mapbox       │  │ Plotly/      │  │
│  │ Browser      │  │ GL JS        │  │ Streamlit    │  │
│  │              │  │              │  │              │  │
│  │ - Author     │  │ - Geographic │  │ - UMAP       │  │
│  │   networks   │  │   collab map │  │   clusters   │  │
│  │ - Citation   │  │ - Institution│  │ - Dashboards │  │
│  │   graphs     │  │   hubs       │  │ - Filtering  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │            REST API / GraphQL                       │ │
│  │            (FastAPI backend)                        │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Module: `src/visualization/`**

- `graph_viz.py` - Neo4j integration for network graphs
- `geo_viz.py` - Mapbox integration for geographic visualization
- `embedding_viz.py` - Interactive UMAP/t-SNE plots
- `api/` - FastAPI endpoints for data access

---

## Implementation Phases

### Phase 1: Foundation ✅ (COMPLETE)

**Status**: Implemented
- Clinical trial classifier
- Structured output (Ollama + Outlines)
- CSV processing
- Checkpoint/resume

### Phase 2: Data Acquisition (Weeks 1-3)

**Priority**: HIGH

#### Week 1-2: PubMed Scraper
```python
# src/scrapers/pubmed_scraper.py

from Bio import Entrez
from typing import List, Dict
import asyncio

class PubMedScraper:
    """Scrape PubMed articles by affiliation."""

    async def search_by_affiliation(
        self,
        affiliation: str,
        max_results: int = 10000,
        batch_size: int = 500
    ) -> List[Dict]:
        """
        Query PubMed for articles from specific affiliation.

        Example affiliation strings:
        - "Boston Children's Hospital[Affiliation]"
        - "Harvard Medical School[Affiliation]"

        Returns list of PMID IDs, then fetches details in batches.
        """

    async def fetch_article_details(
        self,
        pmids: List[str],
        fields: List[str] = None
    ) -> List[Dict]:
        """Fetch full article metadata for PMIDs."""

    def parse_article_xml(self, xml: str) -> Dict:
        """Parse PubMed XML to structured dict."""
```

**Deliverables**:
- [ ] PubMed scraper with affiliation search
- [ ] Rate limiting and retry logic
- [ ] XML parser for article metadata
- [ ] Author extraction with affiliations
- [ ] Data validation and storage

#### Week 3: ClinicalTrials.gov Integration

```python
# src/scrapers/clinical_trials_scraper.py

from mcp import MCPClient  # Model Context Protocol

class ClinicalTrialsScraper:
    """Scrape clinical trials using ClinicalTrials.gov MCP."""

    async def search_by_institution(
        self,
        institution: str,
        status: List[str] = None
    ) -> List[Dict]:
        """
        Search trials by lead organization.

        Args:
            institution: "Boston Children's Hospital"
            status: ["Recruiting", "Active", "Completed"]
        """

    def extract_trial_metadata(self, trial: Dict) -> Dict:
        """
        Extract structured metadata:
        - NCT ID
        - Phase
        - Intervention
        - Primary/secondary outcomes
        - Enrollment
        - Sponsor information
        """
```

**Deliverables**:
- [ ] MCP client integration
- [ ] Institution-based search
- [ ] Trial metadata extraction
- [ ] Link trials to PubMed publications (if available)

### Phase 3: Embedding & Clustering (Weeks 4-5)

**Priority**: HIGH

#### Week 4: Vector Embeddings

```python
# src/processing/embedder.py

from sentence_transformers import SentenceTransformer
import chromadb

class ArticleEmbedder:
    """Generate and store embeddings for articles."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Recommended models:
        - BAAI/bge-large-en-v1.5 (best general)
        - microsoft/BiomedNLP-PubMedBERT-base-uncased (biomedical)
        - intfloat/e5-large-v2 (strong alternative)
        """
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma")

    async def embed_articles(
        self,
        articles: List[Dict],
        fields: List[str] = ["title", "abstract"]
    ) -> None:
        """
        Generate embeddings and store in ChromaDB.

        Combines title + abstract for rich semantic representation.
        """

    async def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        filter: Dict = None
    ) -> List[Dict]:
        """Semantic search across article corpus."""
```

**Deliverables**:
- [ ] Embedding generation pipeline
- [ ] ChromaDB integration
- [ ] Semantic search API
- [ ] Batch processing for 20k articles

#### Week 5: Dimensionality Reduction & Clustering

```python
# src/processing/clustering.py

import umap
import hdbscan
from sklearn.cluster import KMeans
import plotly.express as px

class ArticleClusterer:
    """Cluster articles using UMAP + HDBSCAN."""

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> np.ndarray:
        """
        UMAP for visualization and clustering.

        Parameters tuned for biomedical literature:
        - n_neighbors=15: local structure preservation
        - min_dist=0.1: tight clusters
        """

    def cluster_articles(
        self,
        reduced_embeddings: np.ndarray,
        min_cluster_size: int = 50
    ) -> np.ndarray:
        """
        HDBSCAN for automatic cluster detection.

        Identifies natural groupings without pre-specifying K.
        """

    def generate_cluster_labels(
        self,
        cluster_articles: List[Dict]
    ) -> str:
        """
        Use LLM to generate human-readable cluster label.

        Analyze common themes in cluster and generate name.
        """
```

**Deliverables**:
- [ ] UMAP dimension reduction
- [ ] HDBSCAN clustering
- [ ] Automatic cluster labeling (LLM-based)
- [ ] Interactive visualization (Plotly)

### Phase 4: Multi-Agent Evaluation System (Weeks 6-8)

**Priority**: CRITICAL

This is the core intelligence layer. Design inspired by investment committee workflows.

#### Agent Architecture

```python
# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pydantic import BaseModel

class AgentMessage(BaseModel):
    """Inter-agent communication message."""
    sender: str
    recipient: str
    content: str
    metadata: Dict
    timestamp: datetime

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        llm_client: OllamaClient,
        memory: MemoryManager
    ):
        self.name = name
        self.llm = llm_client
        self.memory = memory

    @abstractmethod
    async def process(self, task: Dict) -> Dict:
        """Process assigned task."""

    async def query_memory(self, query: str) -> List[Dict]:
        """Semantic search through shared memory."""

    async def send_message(self, recipient: str, content: str):
        """Send message to another agent."""
```

#### Investment Evaluator Agent

```python
# src/agents/investment_evaluator.py

class InvestmentEvaluatorAgent(BaseAgent):
    """
    Evaluate clinical trials from tech transfer perspective.

    Evaluation criteria:
    1. Market size & unmet need
    2. Competitive landscape
    3. IP potential
    4. Regulatory pathway
    5. Commercial viability
    6. Exit opportunities
    """

    async def evaluate_trial(self, trial: Dict) -> Dict:
        """
        Multi-step evaluation process:

        Step 1: Market Analysis
        - Extract indication/disease
        - Estimate patient population
        - Identify competitors

        Step 2: Scientific Assessment
        - Novelty of approach
        - Clinical endpoints
        - Preliminary efficacy data

        Step 3: Commercial Assessment
        - Reimbursement landscape
        - Pricing potential
        - Partnership opportunities

        Returns structured evaluation report.
        """

    async def research_market(self, indication: str) -> Dict:
        """Research market size and competition."""

    async def assess_ip_landscape(self, trial: Dict) -> Dict:
        """
        Search for:
        - Related patents
        - Freedom to operate
        - White space opportunities
        """
```

#### Prior Art Researcher Agent

```python
# src/agents/prior_art_researcher.py

class PriorArtResearcherAgent(BaseAgent):
    """
    Comprehensive prior art search.

    Sources:
    1. PubMed/MEDLINE
    2. Google Scholar
    3. USPTO/EPO patents
    4. ClinicalTrials.gov
    5. Regulatory databases (FDA)
    """

    async def research_topic(
        self,
        topic: str,
        depth: str = "comprehensive"
    ) -> Dict:
        """
        Depth levels:
        - quick: Top 20 most relevant papers
        - standard: 50-100 papers + patents
        - comprehensive: Full landscape (200+)
        """

    async def identify_key_authors(self, papers: List[Dict]) -> List[Dict]:
        """Find leading researchers in field."""

    async def trace_research_lineage(self, paper: Dict) -> Dict:
        """
        Build citation tree:
        - What this paper cites
        - Who cites this paper
        - Research evolution over time
        """
```

#### Hypothesis Generator Agent

```python
# src/agents/hypothesis_generator.py

class HypothesisGeneratorAgent(BaseAgent):
    """
    Identify research gaps and generate hypotheses.

    Workflow:
    1. Analyze cluster of related papers
    2. Identify common themes
    3. Find contradictions or open questions
    4. Generate testable hypotheses
    5. Propose experimental designs
    """

    async def analyze_research_cluster(
        self,
        papers: List[Dict]
    ) -> Dict:
        """
        Extract:
        - Common methods
        - Shared outcomes
        - Population characteristics
        - Limitations mentioned
        """

    async def identify_gaps(self, analysis: Dict) -> List[str]:
        """
        Find gaps:
        - Unstudied populations
        - Alternative mechanisms
        - Combination therapies
        - Different endpoints
        """

    async def generate_hypotheses(self, gaps: List[str]) -> List[Dict]:
        """
        For each gap, generate:
        - Hypothesis statement
        - Rationale
        - Proposed methodology
        - Expected outcomes
        - Feasibility assessment
        """
```

#### Coordinator Agent

```python
# src/agents/coordinator.py

class CoordinatorAgent(BaseAgent):
    """
    Orchestrate multi-agent workflows.

    Workflows:
    1. Investment Evaluation
       - Evaluator → Prior Art → Evaluator

    2. Research Discovery
       - Hypothesis Generator → Prior Art → Hypothesis Generator

    3. Comprehensive Analysis
       - All agents collaborate
    """

    async def run_investment_workflow(
        self,
        trial: Dict
    ) -> Dict:
        """
        1. Investment Evaluator: Initial assessment
        2. Prior Art Researcher: Deep dive on competition
        3. Investment Evaluator: Final recommendation
        4. Report Generator: Create memo
        """

    async def run_discovery_workflow(
        self,
        research_area: str
    ) -> Dict:
        """
        1. Collect relevant papers (semantic search)
        2. Hypothesis Generator: Identify gaps
        3. Prior Art Researcher: Validate gaps
        4. Hypothesis Generator: Refine hypotheses
        5. Report Generator: Research proposal
        """
```

**Deliverables**:
- [ ] Base agent framework
- [ ] Investment evaluator agent
- [ ] Prior art researcher agent
- [ ] Hypothesis generator agent
- [ ] Coordinator agent
- [ ] Shared memory system
- [ ] Agent communication protocol

### Phase 5: Knowledge Graph & Network Analysis (Weeks 9-10)

**Priority**: MEDIUM

```python
# src/processing/knowledge_graph.py

from neo4j import GraphDatabase

class KnowledgeGraphBuilder:
    """Build Neo4j knowledge graph of research network."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_graph(self, articles: List[Dict]) -> None:
        """
        Node types:
        - Author
        - Institution
        - Paper
        - Topic (from clustering)
        - Intervention
        - Outcome

        Edge types:
        - AUTHORED
        - AFFILIATED_WITH
        - CITES
        - COLLABORATED_WITH
        - STUDIES (intervention)
        - MEASURES (outcome)
        - BELONGS_TO (topic cluster)
        """

    def find_collaborations(
        self,
        institution: str,
        min_papers: int = 3
    ) -> List[Dict]:
        """
        Find external collaborators.

        Query: Authors from other institutions who
        co-authored ≥3 papers with target institution.
        """

    def identify_research_hubs(self) -> List[Dict]:
        """
        Find institutions with:
        - High centrality (many connections)
        - Bridge positions (connect disparate groups)
        """
```

**Deliverables**:
- [ ] Neo4j graph database
- [ ] Author/institution network
- [ ] Citation network
- [ ] Collaboration analysis queries
- [ ] Network metrics (centrality, clustering)

### Phase 6: Visualization Dashboard (Weeks 11-12)

**Priority**: MEDIUM

```python
# src/visualization/dashboard.py

import streamlit as st
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge

class ResearchDashboard:
    """Interactive dashboard for exploration."""

    def render_umap_clusters(self, articles: List[Dict]):
        """
        Interactive UMAP scatter plot.

        Features:
        - Hover: title, abstract preview
        - Click: full article details
        - Filter: by cluster, year, journal
        - Color: by cluster or classification
        """

    def render_author_network(self, graph_data: Dict):
        """
        Force-directed graph of authors.

        Node size: publication count
        Edge width: collaboration strength
        Color: institution
        """

    def render_geographic_map(self, institutions: List[Dict]):
        """
        Mapbox visualization of collaborations.

        - Points: institution locations
        - Lines: collaboration strength
        - Heat map: publication density
        """
```

**Deliverables**:
- [ ] Streamlit dashboard
- [ ] Interactive UMAP visualization
- [ ] Network graph viewer (Neo4j integration)
- [ ] Geographic collaboration map (Mapbox)
- [ ] Filtering and search interface

---

## Module Specifications

### Database Schema

#### PostgreSQL (Structured Metadata)

```sql
-- Articles
CREATE TABLE articles (
    pmid VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    journal VARCHAR(255),
    publication_date DATE,
    doi VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Authors
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    orcid VARCHAR(50),
    h_index INTEGER
);

-- Affiliations
CREATE TABLE affiliations (
    affiliation_id SERIAL PRIMARY KEY,
    institution VARCHAR(255) NOT NULL,
    department VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8)
);

-- Article-Author relationships
CREATE TABLE article_authors (
    pmid VARCHAR(20) REFERENCES articles(pmid),
    author_id INTEGER REFERENCES authors(author_id),
    affiliation_id INTEGER REFERENCES affiliations(affiliation_id),
    author_position INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (pmid, author_id)
);

-- Classifications
CREATE TABLE classifications (
    pmid VARCHAR(20) PRIMARY KEY REFERENCES articles(pmid),
    is_clinical_trial BOOLEAN,
    confidence DECIMAL(3, 2),
    trial_phase VARCHAR(50),
    study_type VARCHAR(100),
    intervention_type VARCHAR(100),
    sample_size INTEGER,
    randomized BOOLEAN,
    blinded BOOLEAN,
    cluster_id INTEGER
);

-- Clinical Trials (from ClinicalTrials.gov)
CREATE TABLE clinical_trials (
    nct_id VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    status VARCHAR(50),
    phase VARCHAR(50),
    enrollment INTEGER,
    lead_sponsor VARCHAR(255),
    start_date DATE,
    completion_date DATE,
    primary_outcome TEXT,
    intervention TEXT,
    pmid VARCHAR(20) REFERENCES articles(pmid)  -- Link to publication
);
```

#### ChromaDB (Vector Embeddings)

```python
# Collections
collections = {
    "article_embeddings": {
        "documents": ["title + abstract"],
        "metadatas": [{"pmid": "...", "year": 2023, "cluster_id": 5}],
        "embeddings": [[0.1, 0.2, ...]],  # 1024-dim vectors
        "ids": ["pmid_12345"]
    },

    "author_embeddings": {
        "documents": ["author research profile"],
        "metadatas": [{"author_id": 123, "institution": "BCH"}],
        "embeddings": [[...]],
        "ids": ["author_123"]
    }
}
```

#### Neo4j (Knowledge Graph)

```cypher
// Node types
(:Author {name, orcid, h_index})
(:Institution {name, city, country, lat, lon})
(:Paper {pmid, title, year, journal})
(:Topic {cluster_id, label, description})
(:Intervention {name, type})
(:Outcome {name, type})

// Relationships
(:Author)-[:AUTHORED]->(:Paper)
(:Author)-[:AFFILIATED_WITH]->(:Institution)
(:Author)-[:COLLABORATED_WITH {count}]->(:Author)
(:Paper)-[:CITES]->(:Paper)
(:Paper)-[:BELONGS_TO]->(:Topic)
(:Paper)-[:STUDIES]->(:Intervention)
(:Paper)-[:MEASURES]->(:Outcome)
```

---

## Multi-Agent System Design

### Communication Protocol

```python
# src/agents/protocol.py

class AgentProtocol:
    """
    Message passing between agents.

    Patterns:
    1. Request-Response: Agent A asks Agent B for analysis
    2. Broadcast: Coordinator sends task to all agents
    3. Chain: A → B → C (sequential processing)
    4. Consensus: All agents vote on conclusion
    """

    async def send(self, message: AgentMessage):
        """Send message to message queue."""

    async def receive(self, agent_name: str) -> AgentMessage:
        """Receive next message for agent."""

    async def subscribe(self, agent_name: str, topic: str):
        """Subscribe to broadcast topic."""
```

### Evaluation Workflow Example

```
User Query: "Evaluate this clinical trial for investment potential"
│
v
┌─────────────────────────────────────────────────────────┐
│ Coordinator Agent                                        │
│ - Parses query                                          │
│ - Identifies required agents                            │
│ - Creates execution plan                                │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Investment Evaluator Agent                              │
│ Task: Initial assessment                                │
│                                                          │
│ 1. Extract trial details                                │
│ 2. Identify indication/market                           │
│ 3. Flag areas needing deeper research                   │
│ 4. Send findings to Coordinator                         │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Coordinator Agent                                        │
│ - Receives initial assessment                           │
│ - Identifies research questions                         │
│ - Dispatches to Prior Art Researcher                    │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Prior Art Researcher Agent                              │
│ Task: Competitive landscape analysis                    │
│                                                          │
│ 1. Search PubMed for similar interventions              │
│ 2. Query ClinicalTrials.gov for competing trials        │
│ 3. Search patents (USPTO, EPO)                          │
│ 4. Synthesize findings                                  │
│ 5. Return to Coordinator                                │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Coordinator Agent                                        │
│ - Combines all findings                                 │
│ - Sends back to Investment Evaluator                    │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Investment Evaluator Agent                              │
│ Task: Final recommendation                              │
│                                                          │
│ 1. Integrate prior art findings                         │
│ 2. Assess competitive position                          │
│ 3. Evaluate IP landscape                                │
│ 4. Generate recommendation (Go/No-Go/More Data)         │
│ 5. Create investment memo                               │
└─────────┬───────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────┐
│ Report Generator                                         │
│ - Format findings                                       │
│ - Create executive summary                              │
│ - Generate detailed memo                                │
│ - Return to user                                        │
└─────────────────────────────────────────────────────────┘
```

### Prompt Templates for Agents

```yaml
# prompts/investment_evaluator.yaml

system_prompt: |
  You are an expert technology transfer analyst evaluating biomedical
  innovations for commercial potential. You work at a university technology
  transfer office and assess research for:

  1. Market opportunity
  2. Competitive landscape
  3. IP potential
  4. Regulatory pathway
  5. Commercial viability

  You provide data-driven, objective assessments with clear reasoning.

evaluation_template: |
  Evaluate the following clinical trial:

  Title: {title}
  Phase: {phase}
  Intervention: {intervention}
  Primary Outcome: {primary_outcome}
  Sample Size: {sample_size}

  Provide assessment:

  ## Market Analysis
  - Indication and patient population
  - Market size estimation
  - Unmet medical need

  ## Competitive Position
  - Current standard of care
  - Competing approaches (from prior art research)
  - Differentiation

  ## Commercial Viability
  - Reimbursement potential
  - Pricing considerations
  - Partnership opportunities

  ## Recommendation
  - Investment rating: [High/Medium/Low/Pass]
  - Key risks
  - Key opportunities
  - Next steps
```

---

## Data Flow & Pipelines

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│                     PIPELINE                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. ACQUISITION                                         │
│     ├─ Scrape PubMed (affiliation)                      │
│     ├─ Scrape ClinicalTrials.gov                        │
│     └─ Extract author networks                          │
│                                                          │
│  2. STORAGE (Raw)                                       │
│     └─ data/raw/                                        │
│        ├─ pubmed_YYYYMMDD.json                          │
│        ├─ trials_YYYYMMDD.json                          │
│        └─ authors_YYYYMMDD.json                         │
│                                                          │
│  3. PROCESSING                                          │
│     ├─ Parse XML → structured JSON                      │
│     ├─ Generate embeddings                              │
│     ├─ Classify (clinical trial, phase, etc.)           │
│     ├─ Cluster (UMAP + HDBSCAN)                         │
│     └─ Build knowledge graph                            │
│                                                          │
│  4. STORAGE (Processed)                                 │
│     ├─ PostgreSQL (metadata)                            │
│     ├─ ChromaDB (vectors)                               │
│     └─ Neo4j (graph)                                    │
│                                                          │
│  5. ANALYSIS                                            │
│     ├─ Multi-agent evaluation                           │
│     ├─ Research discovery                               │
│     └─ Hypothesis generation                            │
│                                                          │
│  6. VISUALIZATION                                       │
│     ├─ Dashboard (Streamlit)                            │
│     ├─ Network graphs (Neo4j Browser)                   │
│     └─ Geographic maps (Mapbox)                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Daily Update Pipeline

```bash
# scripts/daily_update.sh

#!/bin/bash

# 1. Scrape new articles (last 24 hours)
python -m src.scrapers.pubmed_scraper \
    --affiliation "Boston Children's Hospital" \
    --date-range "last_24h"

# 2. Process new articles
python -m src.processing.embedder \
    --input data/raw/pubmed_$(date +%Y%m%d).json \
    --output data/processed/

# 3. Update classifications
python main.py process \
    --input-csv data/processed/new_articles.csv \
    --resume

# 4. Update clusters (incremental)
python -m src.processing.clustering \
    --mode incremental

# 5. Update knowledge graph
python -m src.processing.knowledge_graph \
    --mode update

# 6. Generate daily report
python -m src.agents.coordinator \
    --workflow daily_summary \
    --output reports/$(date +%Y%m%d)_summary.md
```

---

## Technology Stack

### Core Dependencies

```toml
# pyproject.toml

[project]
name = "aiscientist"
version = "2.0.0"
dependencies = [
    # Existing
    "click>=8.0",
    "loguru>=0.7",
    "rich>=13.0",
    "pydantic>=2.0",
    "outlines>=0.0.34",
    "json-repair>=0.7",

    # Data acquisition
    "biopython>=1.81",  # PubMed/Entrez
    "aiohttp>=3.9",
    "httpx>=0.25",

    # Embeddings & ML
    "sentence-transformers>=2.2",
    "transformers>=4.35",
    "torch>=2.1",
    "umap-learn>=0.5",
    "hdbscan>=0.8",
    "scikit-learn>=1.3",

    # Vector database
    "chromadb>=0.4",

    # Graph database
    "neo4j>=5.14",

    # SQL database
    "psycopg2-binary>=2.9",
    "sqlalchemy>=2.0",

    # Visualization
    "streamlit>=1.28",
    "plotly>=5.17",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "networkx>=3.2",

    # Geographic
    "folium>=0.15",  # Mapbox integration
    "geopy>=2.4",

    # Agent framework
    "langchain>=0.1",
    "langgraph>=0.0.20",

    # MCP
    "mcp>=0.1",  # Model Context Protocol

    # Utilities
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]
```

### Infrastructure

```yaml
# docker-compose.yml

version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: aiscientist
      POSTGRES_USER: researcher
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.14
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  postgres_data:
  neo4j_data:
  chroma_data:
  ollama_data:
```

---

## API Integrations

### PubMed E-utilities

```python
# src/scrapers/pubmed_api.py

class PubMedAPI:
    """
    NCBI E-utilities integration.

    Rate limits:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second

    Best practices:
    - Use ESearch → EFetch pattern
    - Batch fetch (500 articles per request)
    - Use history server for large queries
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    async def esearch(
        self,
        query: str,
        retmax: int = 10000
    ) -> List[str]:
        """Search for PMIDs matching query."""

    async def efetch(
        self,
        pmids: List[str],
        rettype: str = "xml"
    ) -> str:
        """Fetch article details."""
```

### ClinicalTrials.gov API

```python
# src/scrapers/clinicaltrials_api.py

class ClinicalTrialsAPI:
    """
    ClinicalTrials.gov REST API v2.

    Endpoint: https://clinicaltrials.gov/api/v2/studies
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    async def search_studies(
        self,
        query: Dict,
        page_size: int = 100
    ) -> List[Dict]:
        """
        Search clinical trials.

        Query examples:
        - By sponsor: {"query.lead": "Boston Children's Hospital"}
        - By status: {"filter.overallStatus": "RECRUITING"}
        - By phase: {"filter.phase": "PHASE3"}
        """
```

### Mapbox GL JS

```javascript
// For geographic visualization
// API: https://docs.mapbox.com/

mapboxgl.accessToken = 'YOUR_TOKEN';

const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v10',
    center: [-71.1056, 42.3601],  // Boston
    zoom: 2
});

// Add collaboration network
map.on('load', () => {
    // Points for institutions
    map.addLayer({
        'id': 'institutions',
        'type': 'circle',
        'source': {
            'type': 'geojson',
            'data': institutionsGeoJSON
        },
        'paint': {
            'circle-radius': ['get', 'publicationCount'],
            'circle-color': '#007cbf'
        }
    });

    // Lines for collaborations
    map.addLayer({
        'id': 'collaborations',
        'type': 'line',
        'source': {
            'type': 'geojson',
            'data': collaborationsGeoJSON
        },
        'paint': {
            'line-width': ['get', 'strength'],
            'line-color': '#ff7f00'
        }
    });
});
```

---

## Configuration Management

```python
# src/config.py (extended)

@dataclass
class Config:
    # Existing fields...

    # Database connections
    postgres_url: str = "postgresql://researcher:password@localhost/aiscientist"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    chroma_path: str = "./data/chroma"

    # API keys
    ncbi_api_key: str = os.getenv("NCBI_API_KEY")
    mapbox_token: str = os.getenv("MAPBOX_TOKEN")

    # Embedding models
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    # Clustering
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 50

    # Agent configuration
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 300

    # Scraping
    pubmed_affiliation: str = "Boston Children's Hospital[Affiliation]"
    max_articles_per_batch: int = 500
    scrape_delay_seconds: float = 0.1
```

---

## Testing Strategy

```python
# tests/test_agents.py

import pytest
from src.agents import InvestmentEvaluatorAgent, CoordinatorAgent

@pytest.fixture
def sample_trial():
    return {
        "nct_id": "NCT12345678",
        "title": "Phase 2 Trial of Novel CAR-T Therapy",
        "phase": "Phase II",
        "intervention": "CAR-T Cell Therapy",
        "enrollment": 50
    }

@pytest.mark.asyncio
async def test_investment_evaluation(sample_trial):
    """Test investment evaluator agent."""
    agent = InvestmentEvaluatorAgent(...)

    result = await agent.evaluate_trial(sample_trial)

    assert "market_analysis" in result
    assert "recommendation" in result
    assert result["recommendation"] in ["High", "Medium", "Low", "Pass"]

@pytest.mark.asyncio
async def test_multi_agent_workflow(sample_trial):
    """Test coordinator orchestration."""
    coordinator = CoordinatorAgent(...)

    report = await coordinator.run_investment_workflow(sample_trial)

    assert "executive_summary" in report
    assert "prior_art_findings" in report
    assert "final_recommendation" in report
```

---

## Performance Optimization

### Embedding Generation

```python
# Batch processing for efficiency

async def embed_articles_batch(
    articles: List[Dict],
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Optimize embedding generation:
    - Batch size 32 for GPU
    - FP16 precision
    - Sentence pooling

    20k articles: ~10 minutes on A100
    """
    model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device=device
    )
    model.half()  # FP16

    texts = [f"{a['title']} {a['abstract']}" for a in articles]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings
```

### Agent Concurrency

```python
# Parallel agent execution

async def run_agents_parallel(tasks: List[Dict]) -> List[Dict]:
    """Run multiple agents concurrently."""
    results = await asyncio.gather(
        *[agent.process(task) for task in tasks],
        return_exceptions=True
    )
    return results
```

---

## Deployment

### Production Checklist

- [ ] Set up PostgreSQL with replication
- [ ] Configure Neo4j for production (GDS, APOC plugins)
- [ ] Deploy ChromaDB with persistence
- [ ] Set up Ollama with GPU
- [ ] Configure nginx reverse proxy
- [ ] Set up SSL certificates
- [ ] Implement authentication (OAuth)
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure backups (daily snapshots)
- [ ] Set up logging aggregation (ELK stack)

### Monitoring

```python
# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Metrics
articles_processed = Counter(
    'articles_processed_total',
    'Total articles processed'
)

embedding_duration = Histogram(
    'embedding_generation_seconds',
    'Time to generate embeddings'
)

agent_task_duration = Histogram(
    'agent_task_duration_seconds',
    'Agent task completion time',
    ['agent_name']
)

active_agents = Gauge(
    'active_agents',
    'Number of active agents'
)
```

---

## Future Enhancements

### Phase 7+: Advanced Features

1. **Real-time Alerts**
   - Monitor new publications matching criteria
   - Alert when relevant clinical trials are posted
   - Track competitor activity

2. **Multi-Modal Analysis**
   - Extract figures/tables from PDFs
   - Analyze molecular structures
   - Process supplementary data

3. **Automated Report Generation**
   - Weekly research summaries
   - Monthly landscape reports
   - Quarterly trend analysis

4. **Collaboration Features**
   - Multi-user support
   - Shared workspaces
   - Annotation and comments

5. **Integration with Lab Systems**
   - Electronic Lab Notebooks (ELN)
   - LIMS integration
   - Protocol databases

---

## Summary

This architecture provides:

1. **Modularity**: Each component is independent and testable
2. **Scalability**: Designed to handle millions of articles
3. **Extensibility**: Easy to add new agents, data sources, or analyses
4. **Intelligence**: Multi-agent system for sophisticated evaluations
5. **Visualization**: Rich, interactive exploration of research landscape

The phased implementation ensures you can build incrementally, validating each component before moving forward.

Next steps:
1. Review and approve architecture
2. Set up infrastructure (Docker Compose)
3. Begin Phase 2 implementation (scrapers)
4. Iterate based on feedback

Questions or areas needing clarification?
