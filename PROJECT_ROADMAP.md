# AI Scientist - Project Roadmap & Quick Reference

## 🎯 Project Vision

Build an intelligent research platform that:
1. **Scrapes** PubMed & ClinicalTrials.gov by institution
2. **Structures** data using supervised/unsupervised methods
3. **Analyzes** with multi-agent AI system
4. **Visualizes** research landscapes and collaboration networks

---

## 📚 Documentation Guide

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **ARCHITECTURE_DESIGN.md** | Complete system architecture | Planning, big picture |
| **IMPLEMENTATION_GUIDE.md** | Step-by-step setup (Weeks 1-2) | Getting started coding |
| **MULTI_AGENT_DESIGN.md** | Agent system specifications | Building intelligence layer |
| **PROJECT_ROADMAP.md** | This file - quick reference | Daily reference, tracking |

---

## 🚀 Implementation Phases

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Clinical trial classifier
- [x] Structured output (Ollama + Outlines)
- [x] CSV processing
- [x] Checkpoint/resume capability

**Files**: `main.py`, `src/processor.py`, `src/ollama_client.py`, `src/structured_output.py`

---

### 📦 Phase 2: Data Acquisition (Weeks 1-3)

#### Week 1-2: PubMed Scraper
**Goal**: Scrape 20k+ articles from Boston Children's Hospital

**Tasks**:
- [ ] Set up Entrez/Biopython integration
- [ ] Implement rate limiter (3 req/s without key, 10 req/s with key)
- [ ] Create affiliation search (`"Boston Children's Hospital[Affiliation]"`)
- [ ] Build XML parser for article metadata
- [ ] Extract authors + affiliations
- [ ] Store raw JSON

**Files to Create**:
- `src/scrapers/pubmed_scraper.py`
- `src/scrapers/rate_limiter.py`
- `src/scrapers/__init__.py`

**CLI Command**:
```bash
python main.py scrape \
  --affiliation "Boston Children's Hospital[Affiliation]" \
  --max-results 25000 \
  --output data/raw/pubmed_bch.json
```

**Success Metric**: 20,000+ articles scraped with full metadata

---

#### Week 3: ClinicalTrials.gov Integration
**Goal**: Integrate MCP for clinical trials scraping

**Tasks**:
- [ ] Set up MCP client
- [ ] Search trials by institution
- [ ] Extract structured metadata (phase, outcomes, enrollment)
- [ ] Link trials to PubMed publications (via PMIDs)
- [ ] Store in database

**Files to Create**:
- `src/scrapers/clinical_trials_scraper.py`
- `src/scrapers/mcp_client.py`

**CLI Command**:
```bash
python main.py scrape-trials \
  --institution "Boston Children's Hospital" \
  --status Recruiting,Active,Completed
```

**Success Metric**: All active/completed BCH trials retrieved

---

### 🧮 Phase 3: Embedding & Clustering (Weeks 4-5)

#### Week 4: Vector Embeddings
**Goal**: Generate embeddings for 20k articles

**Tasks**:
- [ ] Set up ChromaDB
- [ ] Load embedding model (BAAI/bge-large-en-v1.5)
- [ ] Generate embeddings (title + abstract)
- [ ] Store in ChromaDB with metadata
- [ ] Implement semantic search API

**Files to Create**:
- `src/processing/embedder.py`
- `src/processing/__init__.py`

**CLI Command**:
```bash
python main.py embed \
  --input data/raw/pubmed_bch.json \
  --model BAAI/bge-large-en-v1.5
```

**Success Metric**: 20k articles embedded, semantic search working

---

#### Week 5: Clustering
**Goal**: Discover natural groupings in research

**Tasks**:
- [ ] UMAP dimension reduction (1024D → 2D/3D)
- [ ] HDBSCAN clustering
- [ ] Generate cluster labels (LLM-based)
- [ ] Interactive Plotly visualization
- [ ] Store cluster assignments in DB

**Files to Create**:
- `src/processing/clustering.py`
- `scripts/cluster_pipeline.py`

**Output**:
- `output/clusters.html` - Interactive visualization
- `output/cluster_assignments.json` - PMID → cluster mapping

**Success Metric**: Clear, interpretable clusters (20-50 clusters expected)

---

### 🤖 Phase 4: Multi-Agent System (Weeks 6-8)

#### Week 6: Base Framework + Coordinator
**Goal**: Agent communication infrastructure

**Tasks**:
- [ ] Base agent class
- [ ] Agent message protocol
- [ ] Shared memory manager
- [ ] Coordinator agent
- [ ] Workflow orchestration

**Files to Create**:
- `src/agents/base_agent.py`
- `src/agents/coordinator.py`
- `src/agents/memory_manager.py`
- `src/agents/protocol.py`
- `prompts/coordinator.yaml`

---

#### Week 7: Investment Evaluator Agent
**Goal**: Commercial evaluation of clinical trials

**Tasks**:
- [ ] Evaluation framework (5 dimensions, weighted scoring)
- [ ] Initial assessment workflow
- [ ] Final recommendation with prior art context
- [ ] Structured output (JSON schema)
- [ ] Investment memo generation

**Files to Create**:
- `src/agents/investment_evaluator.py`
- `prompts/investment_evaluator.yaml`

**Test Case**:
```python
trial = {
    "nct_id": "NCT04567890",
    "title": "Phase 2 CAR-T for B-cell Lymphoma",
    "phase": "Phase II",
    ...
}

report = await coordinator.process({
    "type": "investment_evaluation",
    "input": trial
})

# Should return recommendation + score + memo
```

---

#### Week 8: Prior Art Researcher + Hypothesis Generator
**Goal**: Complete agent ecosystem

**Tasks**:
- [ ] Prior art researcher agent
  - Literature search (PubMed semantic search)
  - Clinical trial landscape
  - Patent search integration
  - Competitive analysis
- [ ] Hypothesis generator agent
  - Cluster analysis
  - Gap identification
  - Hypothesis generation
  - Feasibility assessment

**Files to Create**:
- `src/agents/prior_art_researcher.py`
- `src/agents/hypothesis_generator.py`
- `prompts/prior_art_researcher.yaml`
- `prompts/hypothesis_generator.yaml`

---

### 📊 Phase 5: Knowledge Graph (Weeks 9-10)

**Goal**: Build Neo4j graph of research networks

**Tasks**:
- [ ] Set up Neo4j (Docker)
- [ ] Create graph schema (Authors, Institutions, Papers, Topics)
- [ ] Build graph from scraped data
- [ ] Implement collaboration queries
- [ ] Network analysis (centrality, clustering coefficient)

**Files to Create**:
- `src/processing/knowledge_graph.py`
- `schema/neo4j_schema.cypher`

**Example Queries**:
```cypher
// Find top collaborators with BCH
MATCH (bch:Institution {name: "Boston Children's Hospital"})
      -[:AFFILIATED_WITH]-(a1:Author)
      -[:COLLABORATED_WITH]-(a2:Author)
      -[:AFFILIATED_WITH]-(other:Institution)
WHERE other <> bch
RETURN other.name, COUNT(*) as collaborations
ORDER BY collaborations DESC
LIMIT 20
```

---

### 🎨 Phase 6: Visualization Dashboard (Weeks 11-12)

**Goal**: Interactive exploration interface

**Tasks**:
- [ ] Streamlit dashboard
- [ ] UMAP cluster viewer (Plotly)
- [ ] Author network graph (Neo4j → D3.js/Plotly)
- [ ] Geographic collaboration map (Mapbox)
- [ ] Search & filter interface
- [ ] Report export (PDF/Markdown)

**Files to Create**:
- `src/visualization/dashboard.py`
- `src/visualization/graph_viz.py`
- `src/visualization/geo_viz.py`
- `src/api/` - FastAPI backend

**Launch**:
```bash
streamlit run src/visualization/dashboard.py
```

---

## 🛠️ Tech Stack Summary

### Data Acquisition
- **Biopython** (Entrez) - PubMed scraping
- **MCP Client** - ClinicalTrials.gov
- **asyncio/aiohttp** - Async scraping

### Data Storage
- **PostgreSQL** - Structured metadata
- **ChromaDB** - Vector embeddings
- **Neo4j** - Knowledge graph

### ML/AI
- **Sentence Transformers** - Embeddings (bge-large-en-v1.5)
- **UMAP** - Dimension reduction
- **HDBSCAN** - Clustering
- **Ollama** - Local LLM inference
- **Outlines** - Structured generation

### Visualization
- **Streamlit** - Dashboard
- **Plotly** - Interactive charts
- **Mapbox GL JS** - Geographic maps
- **Neo4j Browser** - Graph exploration

### Infrastructure
- **Docker Compose** - Service orchestration
- **FastAPI** - REST API
- **Prometheus/Grafana** - Monitoring

---

## 📋 Weekly Checklist Template

Copy this for each week:

```markdown
### Week X: [Phase Name]

**Goal**: [One sentence goal]

**Monday-Tuesday**:
- [ ] Task 1
- [ ] Task 2

**Wednesday-Thursday**:
- [ ] Task 3
- [ ] Task 4

**Friday**:
- [ ] Testing
- [ ] Documentation
- [ ] Demo/Review

**Success Metric**: [Quantifiable metric]

**Blockers**: [None or list]

**Notes**: [Learnings, decisions]
```

---

## 🎯 Success Metrics by Phase

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 2 | Articles scraped | 20,000+ |
| Phase 2 | Trials scraped | 500+ |
| Phase 3 | Embeddings generated | 20,000 |
| Phase 3 | Clusters identified | 20-50 |
| Phase 3 | Cluster coherence | Eyeball test passes |
| Phase 4 | Investment evals | 10 trials evaluated |
| Phase 4 | Evaluation quality | Expert review: Good+ |
| Phase 5 | Graph nodes | 50,000+ |
| Phase 5 | Graph edges | 200,000+ |
| Phase 6 | Dashboard responsive | < 2s load time |
| Phase 6 | User satisfaction | Positive feedback |

---

## 🔧 Development Setup

### Initial Setup (One-time)

```bash
# 1. Clone repo (already done)
cd /home/user/aiscientist

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with API keys

# 4. Start infrastructure
docker-compose up -d

# 5. Initialize databases
psql -U researcher -d aiscientist -f schema/init.sql

# 6. Pull Ollama model
ollama pull llama3.1:8b

# 7. Verify setup
python main.py check
```

### Daily Development

```bash
# Start services
docker-compose start

# Run tests
pytest tests/

# Check logs
docker-compose logs -f

# Stop services (end of day)
docker-compose stop
```

---

## 📊 Current Status (Track Progress Here)

### Phase Status
- [x] Phase 1: Foundation
- [ ] Phase 2: Data Acquisition
  - [ ] Week 1-2: PubMed Scraper
  - [ ] Week 3: Clinical Trials
- [ ] Phase 3: Embedding & Clustering
  - [ ] Week 4: Embeddings
  - [ ] Week 5: Clustering
- [ ] Phase 4: Multi-Agent System
  - [ ] Week 6: Base + Coordinator
  - [ ] Week 7: Investment Evaluator
  - [ ] Week 8: Prior Art + Hypothesis
- [ ] Phase 5: Knowledge Graph
- [ ] Phase 6: Visualization

### Key Metrics
- Articles in database: **0** / 20,000
- Embeddings generated: **0** / 20,000
- Clusters identified: **0** / 50
- Trials evaluated: **0** / 10
- Dashboard components: **0** / 5

---

## 🚨 Common Issues & Solutions

### Issue: PubMed rate limiting
**Solution**: Get free API key from NCBI (increases from 3 to 10 req/sec)
```bash
# https://www.ncbi.nlm.nih.gov/account/settings/
export NCBI_API_KEY="your_key"
```

### Issue: ChromaDB "Batch size too large"
**Solution**: Reduce batch size to 5000
```python
chroma_batch_size = 5000  # Instead of default
```

### Issue: CUDA out of memory
**Solution**: Use CPU or smaller model
```python
embedder = ArticleEmbedder(device="cpu")
```

### Issue: Ollama generation too slow
**Solution**: Use GPU, smaller model, or reduce batch size
```bash
# GPU
docker-compose up -d  # Ensure GPU enabled

# Or use smaller model
ollama pull gemma2:2b
```

---

## 📖 Learning Resources

### PubMed/NCBI
- E-utilities Guide: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- Affiliation Search: https://pubmed.ncbi.nlm.nih.gov/help/#affiliation-search

### Embeddings
- Sentence Transformers: https://www.sbert.net/
- BGE Models: https://huggingface.co/BAAI

### Clustering
- UMAP: https://umap-learn.readthedocs.io/
- HDBSCAN: https://hdbscan.readthedocs.io/

### Databases
- ChromaDB: https://docs.trychroma.com/
- Neo4j: https://neo4j.com/docs/
- PostgreSQL: https://www.postgresql.org/docs/

### Multi-Agent
- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain: https://python.langchain.com/

---

## 🎓 Key Decisions & Rationale

### Why local LLMs (Ollama) instead of OpenAI?
- **Privacy**: Sensitive research data
- **Cost**: Free after initial setup
- **Control**: Can fine-tune, customize
- **Offline**: Works without internet

### Why UMAP instead of t-SNE?
- **Scalability**: Handles 20k+ points better
- **Preservation**: Better global structure
- **Speed**: Faster than t-SNE
- **Theory**: Solid mathematical foundation

### Why HDBSCAN instead of K-means?
- **Automatic K**: No need to specify number of clusters
- **Density-based**: Finds natural groupings
- **Outliers**: Handles noise (-1 labels)
- **Hierarchical**: Can explore at different scales

### Why Neo4j for knowledge graph?
- **Cypher**: Intuitive query language
- **Performance**: Optimized for graph queries
- **Visualization**: Built-in browser
- **Ecosystem**: Rich plugin ecosystem (APOC, GDS)

---

## 🎯 Next Immediate Actions

1. **Review Documents**
   - [ ] Read ARCHITECTURE_DESIGN.md
   - [ ] Review IMPLEMENTATION_GUIDE.md
   - [ ] Study MULTI_AGENT_DESIGN.md

2. **Set Up Infrastructure**
   - [ ] Create docker-compose.yml
   - [ ] Start services
   - [ ] Initialize databases

3. **Begin Phase 2**
   - [ ] Implement PubMed scraper
   - [ ] Test on 100 articles
   - [ ] Scale to full dataset

4. **Track Progress**
   - [ ] Update this roadmap weekly
   - [ ] Log blockers and solutions
   - [ ] Celebrate milestones! 🎉

---

## 📞 Support & Questions

- **Documentation**: See docs listed at top
- **Code Examples**: `examples/` directory
- **Testing**: `tests/` directory
- **Issues**: GitHub Issues (if applicable)

---

## 🎉 Vision: Where We're Going

**3 months from now**, you'll have:

1. **Complete dataset**: 20,000+ papers, 500+ trials from BCH
2. **Intelligent search**: Semantic search across entire corpus
3. **Research landscape**: Interactive UMAP showing research clusters
4. **Multi-agent analysis**: AI system that evaluates trials and discovers gaps
5. **Network visualization**: See collaborations across institutions
6. **Research proposals**: Auto-generated based on gap analysis

**Use cases enabled**:
- "Show me all CAR-T therapy papers and their commercial potential"
- "Identify gaps in pediatric asthma research"
- "Which external institutions collaborate most with BCH?"
- "Evaluate this clinical trial for investment"
- "Generate research proposal for understudied area X"

**Impact**:
- Accelerate technology transfer decisions
- Identify collaboration opportunities
- Discover new research directions
- Map institutional research strengths

---

Let's build this! 🚀

Start with: **IMPLEMENTATION_GUIDE.md → Week 1**
