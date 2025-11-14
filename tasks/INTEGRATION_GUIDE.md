# Integration Guide: Putting It All Together

This guide explains how to integrate all 13 tasks into a working system and deploy it for use.

---

## Overview

You now have 13 completed tasks:

### Phase 1-2: Data Acquisition (Tasks 1-3)
- **Task 1**: PubMed Scraper
- **Task 2**: ClinicalTrials.gov Scraper
- **Task 3**: Author Network Extractor

### Phase 3: Embeddings & Clustering (Tasks 8-10)
- **Task 8**: Embedding Generation & Vector Store
- **Task 9**: UMAP + HDBSCAN Clustering
- **Task 10**: Cluster Labeling

### Phase 5: Knowledge Graph (Tasks 4-5)
- **Task 4**: Neo4j Setup & Core Operations
- **Task 5**: Graph Analytics

### Phase 6: Visualization (Tasks 6-7)
- **Task 6**: UMAP Clustering Visualization
- **Task 7**: Network & Geographic Visualization

### Integration Layer (Tasks 11-13)
- **Task 11**: Data Pipeline Orchestrator
- **Task 12**: Graph Database Integration
- **Task 13**: Streamlit Dashboard (End-User Interface)

---

## Integration Order

### Step 1: Verify Individual Tasks (Parallel)

First, ensure each completed task passes validation (use `QUALITY_VALIDATION_CHECKLIST.md`):

```bash
# Run validation for each task
cd tasks/task_01_pubmed_scraper && pytest tests/
cd tasks/task_02_clinicaltrials_scraper && pytest tests/
cd tasks/task_03_author_network_extractor && pytest tests/
# ... repeat for all tasks
```

**Status Check:**
- [ ] All task tests pass
- [ ] Test coverage ≥ 80% for each task
- [ ] Functional validation completed

---

### Step 2: Set Up Infrastructure

**Neo4j Database:**
```bash
# Option 1: Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Option 2: Neo4j Desktop
# Download from https://neo4j.com/download/
```

**Environment Variables:**
```bash
# Create .env file
cat > .env << EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword

PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=optional

DATA_DIR=./data
EOF
```

---

### Step 3: Run Data Collection Pipeline (Task 11)

**Create Pipeline Configuration:**
```yaml
# data/config/pipeline_config.yaml

pubmed:
  email: your.email@example.com
  institutions:
    - Harvard Medical School
    - Mayo Clinic
    - Johns Hopkins University
  max_results_per_institution: 500

clinicaltrials:
  sponsors:
    - Massachusetts General Hospital
    - Mayo Clinic
    - Johns Hopkins Hospital
  max_results_per_sponsor: 300

output_dir: ./data
log_level: INFO
parallel_workers: 3
```

**Run Pipeline:**
```bash
# This will run Tasks 1, 2, 3 in coordinated fashion
python -m pipeline.orchestrator --config data/config/pipeline_config.yaml
```

**Expected Output:**
```
✓ Harvard Medical School: 487 papers
✓ Mayo Clinic: 512 papers
✓ Johns Hopkins University: 498 papers
✓ Massachusetts General Hospital: 234 trials
✓ Mayo Clinic: 189 trials
✓ Johns Hopkins Hospital: 267 trials
✓ Extracted 1,243 authors, 4,567 collaborations

Pipeline Completed!
Duration: 324.5s
Total Papers: 1,497
Total Trials: 690
```

**Verification:**
```bash
# Check data was collected
ls -lh data/raw/pubmed/
ls -lh data/raw/clinicaltrials/
ls -lh data/raw/author_networks/
ls -lh data/processed/  # unified_dataset_*.json should exist
```

---

### Step 4: Load Data into Neo4j (Task 12)

**Load Author Network:**
```bash
python -m graph_integration.loader \
    --network-file data/raw/author_networks/network_all_institutions_*.json \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password yourpassword
```

**Expected Output:**
```json
{
  "status": "success",
  "authors": {"created": 1243, "updated": 0},
  "papers": {"created": 1497, "updated": 0},
  "institutions": {"created": 45, "updated": 0},
  "authorship": {"created": 3421, "updated": 0},
  "collaborations": {"created": 4567, "updated": 0}
}

Database Statistics:
{
  "nodes": {
    "Author": 1243,
    "Paper": 1497,
    "Institution": 45
  },
  "relationships": {
    "AUTHORED": 3421,
    "COLLABORATED_WITH": 4567,
    "AFFILIATED_WITH": 1876
  }
}
```

**Verification:**
```bash
# Open Neo4j Browser: http://localhost:7474
# Run test query:
MATCH (a:Author)-[:COLLABORATED_WITH]->(b:Author)
RETURN a.name, b.name
LIMIT 10
```

---

### Step 5: Generate Embeddings & Clusters (Tasks 8, 9, 10)

**Run Embedding Pipeline:**
```bash
# Task 8: Generate embeddings
python -m embeddings.embeddings_vectorstore \
    --input data/processed/unified_dataset_*.json \
    --output data/embeddings/

# Expected: article_embeddings.npy created (shape: [1497, 384])
```

**Run Clustering:**
```bash
# Task 9: UMAP + HDBSCAN clustering
python -m embeddings.clustering_pipeline \
    --embeddings data/embeddings/article_embeddings.npy \
    --output data/embeddings/

# Expected: cluster_labels.npy, reduced_embeddings.npy
```

**Generate Labels:**
```bash
# Task 10: Label clusters
python -m embeddings.cluster_labeling \
    --papers data/processed/unified_dataset_*.json \
    --labels data/embeddings/cluster_labels.npy \
    --output data/embeddings/cluster_labels.json
```

**Expected Output:**
```
Found 12 clusters:
  Cluster 0: Cancer Immunotherapy (247 papers)
  Cluster 1: Cardiovascular Disease (189 papers)
  Cluster 2: Neurodegenerative Disorders (156 papers)
  ...
```

---

### Step 6: Launch Dashboard (Task 13)

**Install Dashboard Dependencies:**
```bash
cd src/dashboard
pip install -r requirements.txt
```

**Run Locally:**
```bash
streamlit run app.py
```

**Access Dashboard:**
```
Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

**Verification:**
- [ ] Home page loads and shows statistics
- [ ] Search finds papers by keyword
- [ ] Network page shows collaboration graph
- [ ] Trends page shows clusters
- [ ] Map page shows geographic distribution
- [ ] Can export results to CSV/PDF

---

## Complete Integration Test

Run this end-to-end test to verify everything works:

```bash
#!/bin/bash
# integration_test.sh

set -e

echo "=== Integration Test ==="

# 1. Check Neo4j is running
echo "Checking Neo4j..."
curl -f http://localhost:7474 || (echo "Neo4j not running!" && exit 1)

# 2. Run pipeline
echo "Running data collection pipeline..."
python -m pipeline.orchestrator --config data/config/pipeline_config.yaml

# 3. Load into Neo4j
echo "Loading data into Neo4j..."
python -m graph_integration.loader \
    --network-file data/raw/author_networks/network_all_institutions_*.json \
    --neo4j-password yourpassword

# 4. Generate embeddings
echo "Generating embeddings..."
python -m embeddings.embeddings_vectorstore \
    --input data/processed/unified_dataset_*.json

# 5. Run clustering
echo "Running clustering..."
python -m embeddings.clustering_pipeline \
    --embeddings data/embeddings/article_embeddings.npy

# 6. Label clusters
echo "Labeling clusters..."
python -m embeddings.cluster_labeling \
    --papers data/processed/unified_dataset_*.json \
    --labels data/embeddings/cluster_labels.npy

# 7. Test dashboard
echo "Testing dashboard..."
streamlit run src/dashboard/app.py --server.headless true &
STREAMLIT_PID=$!
sleep 10
curl -f http://localhost:8501 || (echo "Dashboard not running!" && exit 1)
kill $STREAMLIT_PID

echo "=== Integration Test PASSED ==="
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Data Collection (Task 11)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ PubMed   │  │Clinical  │  │ Author   │              │
│  │ Scraper  │  │ Trials   │  │ Network  │              │
│  │ (Task 1) │  │ (Task 2) │  │ (Task 3) │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │                     │
│       └─────────────┴─────────────┘                     │
│                     │                                   │
│         ┌───────────▼───────────┐                       │
│         │   Unified Dataset     │                       │
│         │  data/processed/      │                       │
│         └───────────┬───────────┘                       │
└─────────────────────┼───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  Graph Database │       │   Embeddings    │
│    (Task 12)    │       │   (Task 8-10)   │
│                 │       │                 │
│  ┌───────────┐  │       │ ┌─────────────┐ │
│  │  Neo4j    │  │       │ │  ChromaDB   │ │
│  │           │  │       │ │  UMAP       │ │
│  │  Authors  │  │       │ │  HDBSCAN    │ │
│  │  Papers   │  │       │ │  Labels     │ │
│  │  Collabs  │  │       │ └─────────────┘ │
│  └───────────┘  │       └─────────────────┘
└────────┬────────┘                │
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   Streamlit Dashboard   │
         │      (Task 13)          │
         │                         │
         │  • Search               │
         │  • Network Explorer     │
         │  • Research Trends      │
         │  • Geographic Map       │
         │  • Reports              │
         └─────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │      End User           │
         │   (Hospital CMO)        │
         └─────────────────────────┘
```

---

## Troubleshooting

### Issue: Pipeline fails with API rate limit errors

**Solution:**
```yaml
# Reduce rate in pipeline_config.yaml
pubmed:
  rate_limit_per_second: 1.0  # Slower

clinicaltrials:
  rate_limit_per_second: 1.0
```

### Issue: Neo4j connection refused

**Solution:**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs neo4j

# Restart
docker restart neo4j
```

### Issue: Dashboard shows "No data found"

**Solution:**
```bash
# Verify data exists
ls -lh data/processed/unified_dataset_*.json

# Check file permissions
chmod +r data/processed/unified_dataset_*.json

# Restart dashboard
streamlit run src/dashboard/app.py
```

### Issue: Embeddings generation out of memory

**Solution:**
```python
# In embeddings_vectorstore.py, reduce batch size
embedder.embed_batch(articles, batch_size=16)  # Instead of 32
```

---

## Performance Optimization

### For Large Datasets (>10,000 papers)

**1. Use Batch Processing:**
```python
# In pipeline orchestrator
config.parallel_workers = 5  # More workers
config.pubmed.max_results_per_institution = 10000
```

**2. Optimize Neo4j:**
```bash
# Increase heap size
docker run -e NEO4J_dbms_memory_heap_max__size=4G neo4j
```

**3. Use Smaller Embedding Model:**
```python
# In embeddings config
model_name = 'all-MiniLM-L6-v2'  # Fast, 384 dims
# Instead of 'all-mpnet-base-v2'  # Slower, 768 dims
```

---

## Maintenance & Updates

### Daily: Update Data

```bash
# Run incremental update
python -m pipeline.orchestrator --config data/config/pipeline_config.yaml

# Load new data into Neo4j
python -m graph_integration.loader \
    --network-file data/raw/author_networks/network_all_institutions_*.json \
    --neo4j-password yourpassword
```

### Weekly: Re-cluster

```bash
# Regenerate embeddings for new papers
python -m embeddings.embeddings_vectorstore --input data/processed/unified_dataset_*.json

# Re-run clustering
python -m embeddings.clustering_pipeline --embeddings data/embeddings/article_embeddings.npy
```

### Monthly: Backup

```bash
# Backup Neo4j
docker exec neo4j neo4j-admin dump --to=/backups/neo4j_backup_$(date +%Y%m%d).dump

# Backup data
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

---

## Success Checklist

- [ ] All 13 tasks completed and tested individually
- [ ] Infrastructure set up (Neo4j running)
- [ ] Pipeline successfully collects data
- [ ] Data loaded into Neo4j
- [ ] Embeddings and clusters generated
- [ ] Dashboard runs and displays data correctly
- [ ] End-to-end integration test passes
- [ ] Dashboard deployed and accessible
- [ ] Documentation provided to end users
- [ ] Backup and maintenance procedures in place

---

## Next Steps

Once integration is complete:

1. **User Training**: Train hospital CMO on using the dashboard
2. **Monitoring**: Set up monitoring for data freshness and system health
3. **Scaling**: Consider cloud deployment (AWS, GCP) for larger datasets
4. **Features**: Add additional features based on user feedback

---

**End of Integration Guide**
