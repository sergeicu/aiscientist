# Quality Validation Checklist for Parallel Tasks

This document provides a systematic approach to validate each completed task before integration.

## Overview

Each task must pass the following validation stages:
1. **Automated Testing** - All tests pass with adequate coverage
2. **Functional Validation** - Manual verification of core functionality
3. **Integration Readiness** - Interfaces and contracts are correct
4. **Documentation Check** - Code is documented and understandable

---

## Task 1: PubMed Scraper

### Automated Testing ✓
```bash
cd src/data_acquisition/pubmed_scraper
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Requirements:**
- [ ] All tests pass (100% success rate)
- [ ] Test coverage ≥ 80%
- [ ] No failing edge cases

### Functional Validation ✓
```bash
# Test with real API (small batch)
python -m pytest tests/ -k "test_fetch_institution_papers" -v
```

**Verify:**
- [ ] Successfully fetches papers for at least 3 test institutions
- [ ] Rate limiting works (no API blocks)
- [ ] XML parsing handles malformed data
- [ ] Saves data to correct directory structure

**Manual Test Script:**
```python
from pubmed_scraper import PubMedScraper

scraper = PubMedScraper(email="test@example.com", api_key=None)
results = scraper.fetch_institution_papers(
    institution="Harvard Medical School",
    max_results=10
)
print(f"Fetched {len(results)} papers")
print(results[0] if results else "No results")
```

**Expected Output:**
- Returns list of papers with: title, authors, abstract, pub_date, pmid
- No errors or exceptions
- Output saved to `data/pubmed/harvard_medical_school/`

### Integration Readiness ✓
**Check Output Format:**
```python
# Output should be JSON with this schema
{
    "pmid": "12345678",
    "title": "...",
    "abstract": "...",
    "authors": [
        {"name": "John Doe", "affiliation": "Harvard Medical School"}
    ],
    "publication_date": "2024-01-15",
    "journal": "...",
    "institution": "Harvard Medical School"
}
```

**Verify:**
- [ ] Output matches schema above
- [ ] Files saved to `data/pubmed/{institution_slug}/papers_{timestamp}.json`
- [ ] No PII or sensitive data logged
- [ ] Exports `PubMedScraper` class with documented API

### Documentation Check ✓
- [ ] README.md exists with setup instructions
- [ ] All public methods have docstrings
- [ ] Example usage provided
- [ ] Dependencies listed in requirements.txt

---

## Task 2: ClinicalTrials.gov Scraper

### Automated Testing ✓
```bash
cd src/data_acquisition/clinicaltrials_scraper
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Requirements:**
- [ ] All tests pass
- [ ] Test coverage ≥ 80%
- [ ] Mock API tests work

### Functional Validation ✓
```bash
# Test with real API
python -m pytest tests/ -k "test_fetch_by_sponsor" -v
```

**Manual Test Script:**
```python
from clinicaltrials_scraper import ClinicalTrialsScraper

scraper = ClinicalTrialsScraper()
results = scraper.fetch_by_sponsor(
    sponsor="Massachusetts General Hospital",
    max_results=10
)
print(f"Fetched {len(results)} trials")
print(results[0] if results else "No results")
```

**Expected Output:**
- Returns list of trials with: nct_id, title, status, sponsor, conditions, interventions
- Output saved to `data/clinicaltrials/massachusetts_general_hospital/`

### Integration Readiness ✓
**Check Output Format:**
```python
{
    "nct_id": "NCT12345678",
    "title": "...",
    "status": "Recruiting",
    "sponsor": "Massachusetts General Hospital",
    "conditions": ["Cancer", "Leukemia"],
    "interventions": ["Drug: Pembrolizumab"],
    "start_date": "2024-01-01",
    "locations": [...]
}
```

**Verify:**
- [ ] Output matches schema
- [ ] Files saved to `data/clinicaltrials/{sponsor_slug}/trials_{timestamp}.json`
- [ ] Exports `ClinicalTrialsScraper` class

---

## Task 3: Author Network Extractor

### Automated Testing ✓
```bash
cd src/data_acquisition/author_network
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Functional Validation ✓
**Manual Test Script:**
```python
from author_network import AuthorNetworkExtractor
import json

# Load sample PubMed data
with open('data/pubmed/harvard_medical_school/papers_*.json') as f:
    papers = json.load(f)

extractor = AuthorNetworkExtractor()
network = extractor.build_network(papers)

print(f"Authors: {len(network['authors'])}")
print(f"Collaborations: {len(network['collaborations'])}")
```

**Verify:**
- [ ] Parses author affiliations correctly
- [ ] Disambiguates authors (same name, different affiliations)
- [ ] Builds collaboration edges
- [ ] Outputs graph format compatible with Neo4j

### Integration Readiness ✓
**Check Output Format:**
```python
{
    "authors": [
        {
            "author_id": "uuid",
            "name": "John Doe",
            "affiliations": ["Harvard Medical School"],
            "papers": ["pmid1", "pmid2"]
        }
    ],
    "collaborations": [
        {
            "author_1": "uuid1",
            "author_2": "uuid2",
            "papers": ["pmid1"],
            "weight": 1
        }
    ]
}
```

---

## Task 4: Neo4j Graph Setup

### Automated Testing ✓
```bash
cd src/graph_database/neo4j_setup
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Prerequisites:**
- [ ] Neo4j instance running (docker or local)
- [ ] Environment variables set: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

### Functional Validation ✓
**Manual Test Script:**
```python
from neo4j_setup import Neo4jConnection, Neo4jSchema

# Test connection
conn = Neo4jConnection(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
assert conn.verify_connectivity(), "Connection failed"

# Test schema creation
schema = Neo4jSchema(conn)
schema.create_constraints()
schema.create_indexes()

# Verify
constraints = schema.list_constraints()
print(f"Created {len(constraints)} constraints")
```

**Verify:**
- [ ] Connection pool works
- [ ] Constraints created: Author.author_id, Paper.pmid, Institution.name
- [ ] Indexes created on commonly queried fields
- [ ] CRUD operations work

### Integration Readiness ✓
- [ ] Exports `Neo4jConnection`, `Neo4jSchema`, `Neo4jCRUD` classes
- [ ] Connection can be shared across modules
- [ ] Proper error handling and logging

---

## Task 5: Graph Analytics

### Automated Testing ✓
```bash
cd src/graph_database/graph_analytics
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Prerequisites:**
- [ ] Task 4 completed
- [ ] Sample data loaded in Neo4j

### Functional Validation ✓
**Manual Test Script:**
```python
from graph_analytics import GraphAnalytics

analytics = GraphAnalytics(neo4j_conn)

# Test centrality
top_authors = analytics.compute_author_centrality(metric="degree", limit=10)
print(f"Top 10 authors by degree centrality: {top_authors}")

# Test community detection
communities = analytics.detect_communities(algorithm="louvain")
print(f"Found {len(communities)} communities")
```

**Verify:**
- [ ] Centrality metrics return reasonable results
- [ ] Community detection completes without errors
- [ ] Results match manual inspection

---

## Task 6: UMAP Visualization

### Automated Testing ✓
```bash
cd src/visualization/umap_viz
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Functional Validation ✓
**Manual Test Script:**
```python
from umap_viz import UMAPVisualizer
import numpy as np

# Generate test embeddings
embeddings = np.random.randn(100, 384)
labels = np.random.randint(0, 5, 100)

viz = UMAPVisualizer()
fig = viz.create_cluster_plot(embeddings, labels, titles=["Paper " + str(i) for i in range(100)])
fig.show()  # Opens in browser
```

**Verify:**
- [ ] UMAP reduction works
- [ ] Plot renders correctly
- [ ] Interactive hover shows paper titles
- [ ] Exports HTML file

---

## Task 7: Network & Geographic Visualization

### Automated Testing ✓
```bash
cd src/visualization/network_geo_viz
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Functional Validation ✓
**Manual Test Script:**
```python
from network_geo_viz import NetworkVisualizer, GeoVisualizer

# Test network viz
net_viz = NetworkVisualizer()
graph_data = {
    "nodes": [{"id": "1", "label": "Author 1"}, {"id": "2", "label": "Author 2"}],
    "edges": [{"source": "1", "target": "2", "weight": 5}]
}
fig = net_viz.create_network_plot(graph_data)
fig.show()

# Test geo viz
geo_viz = GeoVisualizer()
locations = [
    {"name": "Harvard", "lat": 42.3770, "lon": -71.1167, "count": 10}
]
geo_map = geo_viz.create_map(locations)
geo_map.save("test_map.html")
```

**Verify:**
- [ ] Network graph renders
- [ ] Geographic map shows markers
- [ ] Interactive features work

---

## Task 8: Embeddings & Vector Store

### Automated Testing ✓
```bash
cd src/embeddings/embeddings_vectorstore
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Functional Validation ✓
**Manual Test Script:**
```python
from embeddings_vectorstore import ArticleEmbedder, VectorStore

# Test embedding generation
embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
articles = [
    {"pmid": "1", "title": "Cancer Research", "abstract": "..."},
    {"pmid": "2", "title": "Heart Disease Study", "abstract": "..."}
]
embeddings = embedder.embed_batch(articles)
print(f"Generated embeddings shape: {embeddings.shape}")

# Test vector store
store = VectorStore(persist_directory="./chroma_db")
store.add_articles(articles, embeddings)

# Test search
results = store.search("cancer treatment", k=5)
print(f"Found {len(results)} similar articles")
```

**Verify:**
- [ ] Embeddings generated correctly (shape: [n_articles, 384])
- [ ] ChromaDB stores and retrieves
- [ ] Semantic search returns relevant results
- [ ] Persistence works (reload from disk)

---

## Task 9: Clustering Pipeline

### Automated Testing ✓
```bash
cd src/embeddings/clustering_pipeline
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Prerequisites:**
- [ ] Task 8 completed (need embeddings)

### Functional Validation ✓
**Manual Test Script:**
```python
from clustering_pipeline import ClusteringPipeline
import numpy as np

# Load embeddings from Task 8
embeddings = np.load('data/embeddings/article_embeddings.npy')

pipeline = ClusteringPipeline(
    n_neighbors=15,
    min_cluster_size=10
)
labels, reduced_embeddings = pipeline.fit_predict(embeddings)

print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
print(f"Noise points: {sum(labels == -1)}")

# Evaluate
metrics = pipeline.evaluate(embeddings, labels)
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
```

**Verify:**
- [ ] Clusters are found (not all noise)
- [ ] Silhouette score > 0.3
- [ ] Reduced embeddings saved
- [ ] Cluster assignments saved

---

## Task 10: Cluster Labeling

### Automated Testing ✓
```bash
cd src/embeddings/cluster_labeling
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Prerequisites:**
- [ ] Task 9 completed (need cluster assignments)

### Functional Validation ✓
**Manual Test Script:**
```python
from cluster_labeling import ClusterLabeler

# Load data
import json
with open('data/pubmed/all_papers.json') as f:
    articles = json.load(f)

labels = np.load('data/embeddings/cluster_labels.npy')

labeler = ClusterLabeler()
cluster_info = labeler.label_all_clusters(articles, labels)

for cluster_id, info in cluster_info.items():
    print(f"\nCluster {cluster_id}: {info['label']}")
    print(f"  Size: {info['size']}")
    print(f"  Keywords: {', '.join(info['keywords'][:5])}")
```

**Verify:**
- [ ] Labels are meaningful and descriptive
- [ ] Keywords represent cluster content
- [ ] Export format is correct

---

## Integration Smoke Test

Once all tasks pass individual validation, run this integration smoke test:

```bash
# This script will be provided in integration tasks
python scripts/integration_smoke_test.py
```

**What it tests:**
1. Data flows from scrapers → embeddings → clustering
2. Graph database ingests author networks
3. Visualizations render with real data
4. All components can communicate

---

## Quality Metrics Summary

Create a summary report for each task:

```markdown
## Task X: [Name]

**Status:** ✅ PASS / ❌ FAIL / ⚠️ NEEDS ATTENTION

**Test Results:**
- Tests Pass: Yes/No
- Coverage: X%
- Functional Tests: Pass/Fail

**Integration Readiness:**
- Output format correct: Yes/No
- Interfaces documented: Yes/No
- Dependencies clear: Yes/No

**Issues Found:**
1. [List any issues]

**Ready for Integration:** Yes/No
```

---

## Quick Validation Commands

```bash
# Run all tests across all tasks
find tasks/ -name "tests" -type d | while read dir; do
    echo "Testing $dir..."
    pytest "$dir" -v --cov || echo "FAILED: $dir"
done

# Check test coverage
find tasks/ -name "tests" -type d | while read dir; do
    cd "$(dirname "$dir")"
    pytest tests/ --cov=. --cov-report=term-missing | grep "TOTAL"
done

# Validate output formats
python scripts/validate_outputs.py
```

---

## Troubleshooting Common Issues

### Low Test Coverage
- Add integration tests
- Test error paths
- Test edge cases

### Failing Functional Tests
- Check API keys/credentials
- Verify external services (Neo4j, internet)
- Check data formats

### Integration Issues
- Verify output schemas match
- Check file paths are correct
- Ensure modules export correct classes

---

## Sign-off Checklist

Before marking a task as "Integration Ready":

- [ ] All automated tests pass
- [ ] Functional validation completed
- [ ] Manual testing successful
- [ ] Output format verified
- [ ] Documentation complete
- [ ] No critical bugs
- [ ] Code reviewed (if possible)
- [ ] Performance acceptable

**Validator:** _____________
**Date:** _____________
**Notes:** _____________
