# Task Branch Merge Instructions

This guide provides step-by-step instructions to merge all completed task branches into a single integration branch.

---

## Overview

You have completed tasks 1-9, each on separate branches:
- `claude/task-01-pubmed-scraper-<session-id>`
- `claude/task-02-clinicaltrials-scraper-<session-id>`
- `claude/task-03-author-network-<session-id>`
- `claude/task-04-neo4j-setup-<session-id>`
- `claude/task-05-graph-analytics-<session-id>`
- `claude/task-06-umap-viz-<session-id>`
- `claude/task-07-network-geo-viz-<session-id>`
- `claude/task-08-embeddings-<session-id>`
- `claude/task-09-clustering-<session-id>`

**Goal**: Merge all changes into a single integration branch that contains all working code.

---

## Directory Structure Expected After Merge

```
aiscientist/
├── src/
│   ├── data_acquisition/
│   │   ├── pubmed_scraper/          (from Task 1)
│   │   ├── clinicaltrials_scraper/  (from Task 2)
│   │   └── author_network/          (from Task 3)
│   ├── graph_database/
│   │   ├── neo4j_setup/             (from Task 4)
│   │   └── graph_analytics/         (from Task 5)
│   ├── embeddings/
│   │   ├── embeddings_vectorstore/  (from Task 8)
│   │   └── clustering_pipeline/     (from Task 9)
│   └── visualization/
│       ├── umap_viz/                (from Task 6)
│       └── network_geo_viz/         (from Task 7)
├── tests/
│   ├── data_acquisition/
│   ├── graph_database/
│   ├── embeddings/
│   └── visualization/
├── data/
├── requirements.txt
└── README.md
```

---

## Step-by-Step Merge Instructions

### Step 1: Fetch All Remote Branches

First, ensure you have all the latest branches from remote:

```bash
git fetch --all
```

This will download all branch references without modifying your working directory.

---

### Step 2: Create Integration Branch

Create a new integration branch from the current base:

```bash
# Start from the base branch (the one with task specs)
git checkout claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc

# Create new integration branch
git checkout -b claude/integration-tasks-1-9-<session-id>
```

Replace `<session-id>` with your current Claude Code session ID.

---

### Step 3: List All Task Branches

First, identify the exact branch names for all completed tasks:

```bash
# List all task branches
git branch -a | grep -E "claude/task-0[1-9]"
```

Copy the exact branch names for use in the next steps.

---

### Step 4: Merge Each Task Branch (One by One)

Merge each task branch individually to better handle conflicts:

#### Task 1: PubMed Scraper

```bash
# Merge Task 1
git merge origin/claude/task-01-pubmed-scraper-<actual-session-id> \
  --no-ff \
  -m "Merge Task 1: PubMed Scraper"

# If there are conflicts, resolve them (see Step 6)
# Then continue with git merge --continue
```

#### Task 2: ClinicalTrials Scraper

```bash
git merge origin/claude/task-02-clinicaltrials-scraper-<actual-session-id> \
  --no-ff \
  -m "Merge Task 2: ClinicalTrials.gov Scraper"
```

#### Task 3: Author Network Extractor

```bash
git merge origin/claude/task-03-author-network-<actual-session-id> \
  --no-ff \
  -m "Merge Task 3: Author Network Extractor"
```

#### Task 4: Neo4j Setup

```bash
git merge origin/claude/task-04-neo4j-setup-<actual-session-id> \
  --no-ff \
  -m "Merge Task 4: Neo4j Graph Setup"
```

#### Task 5: Graph Analytics

```bash
git merge origin/claude/task-05-graph-analytics-<actual-session-id> \
  --no-ff \
  -m "Merge Task 5: Graph Analytics"
```

#### Task 6: UMAP Visualization

```bash
git merge origin/claude/task-06-umap-viz-<actual-session-id> \
  --no-ff \
  -m "Merge Task 6: UMAP Visualization"
```

#### Task 7: Network & Geographic Visualization

```bash
git merge origin/claude/task-07-network-geo-viz-<actual-session-id> \
  --no-ff \
  -m "Merge Task 7: Network & Geographic Visualization"
```

#### Task 8: Embeddings & Vector Store

```bash
git merge origin/claude/task-08-embeddings-<actual-session-id> \
  --no-ff \
  -m "Merge Task 8: Embeddings & Vector Store"
```

#### Task 9: Clustering Pipeline

```bash
git merge origin/claude/task-09-clustering-<actual-session-id> \
  --no-ff \
  -m "Merge Task 9: Clustering Pipeline"
```

---

### Step 5: Alternative - Merge All at Once (Octopus Merge)

If you're confident there are no conflicts, you can merge all branches at once:

```bash
git merge \
  origin/claude/task-01-pubmed-scraper-<session-id> \
  origin/claude/task-02-clinicaltrials-scraper-<session-id> \
  origin/claude/task-03-author-network-<session-id> \
  origin/claude/task-04-neo4j-setup-<session-id> \
  origin/claude/task-05-graph-analytics-<session-id> \
  origin/claude/task-06-umap-viz-<session-id> \
  origin/claude/task-07-network-geo-viz-<session-id> \
  origin/claude/task-08-embeddings-<session-id> \
  origin/claude/task-09-clustering-<session-id> \
  --no-ff \
  -m "Merge all completed tasks (1-9)"
```

**Note**: This only works if there are no conflicts between branches. If conflicts occur, Git will abort and you'll need to merge one by one.

---

### Step 6: Handling Merge Conflicts

If you encounter conflicts during merge:

#### Check Conflict Files

```bash
git status
```

Look for files marked as "both modified" or "both added".

#### Common Conflict Scenarios

**Scenario 1: Different tasks modified different files**
- ✅ **No conflict** - Git will auto-merge

**Scenario 2: Multiple tasks modified `requirements.txt`**
- ⚠️ **Conflict likely**
- **Resolution**: Combine all dependencies from all tasks

**Scenario 3: Multiple tasks created different test files in same directory**
- ✅ **No conflict** - Different files

**Scenario 4: Multiple tasks modified same README or root-level file**
- ⚠️ **Conflict likely**
- **Resolution**: Manually combine content

#### Resolve Conflicts

```bash
# Open conflicted file
# Look for conflict markers:
<<<<<<< HEAD
(current branch content)
=======
(merging branch content)
>>>>>>> branch-name

# Edit file to combine both changes appropriately
# Remove conflict markers

# After resolving all conflicts:
git add <conflicted-file>
git merge --continue
```

#### Common Files That May Conflict

1. **`requirements.txt`** - Combine all unique dependencies
2. **Root `README.md`** - Combine documentation
3. **`.gitignore`** - Combine ignore patterns
4. **`setup.py` or `pyproject.toml`** - Combine configurations

---

### Step 7: Verify Directory Structure

After all merges, verify the directory structure is correct:

```bash
# Check directory structure
tree -L 3 src/

# Should show:
# src/
# ├── data_acquisition/
# │   ├── pubmed_scraper/
# │   ├── clinicaltrials_scraper/
# │   └── author_network/
# ├── graph_database/
# │   ├── neo4j_setup/
# │   └── graph_analytics/
# ├── embeddings/
# │   ├── embeddings_vectorstore/
# │   └── clustering_pipeline/
# └── visualization/
#     ├── umap_viz/
#     └── network_geo_viz/
```

---

### Step 8: Consolidate `requirements.txt`

Multiple tasks may have added dependencies. Consolidate them:

```bash
# View current requirements.txt
cat requirements.txt
```

Create a unified `requirements.txt`:

```txt
# Data Acquisition (Tasks 1, 2, 3)
requests>=2.31.0
aiohttp>=3.8.0
tenacity>=8.2.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Graph Database (Tasks 4, 5)
neo4j>=5.12.0
networkx>=3.1

# Embeddings & Clustering (Tasks 8, 9)
sentence-transformers>=2.2.0
chromadb>=0.4.0
umap-learn>=0.5.0
hdbscan>=0.8.0
scikit-learn>=1.3.0

# Visualization (Tasks 6, 7)
plotly>=5.17.0
folium>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Common
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
pytest-cov>=4.1.0

# Development
black>=23.0.0
ruff>=0.0.280
mypy>=1.4.0
```

---

### Step 9: Run All Tests

Verify that all tests still pass after merging:

```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run tests by module
pytest tests/data_acquisition/ -v
pytest tests/graph_database/ -v
pytest tests/embeddings/ -v
pytest tests/visualization/ -v
```

**Expected Output:**
- All tests pass
- Coverage ≥ 80%
- No import errors

If tests fail:
1. Check for missing dependencies
2. Verify directory structure
3. Check for import path issues

---

### Step 10: Create Master README

Create or update the root `README.md` to document the integrated project:

```markdown
# AI Scientist - Research Intelligence Platform

Integrated implementation of Tasks 1-9.

## Modules

### Data Acquisition (Tasks 1-3)
- **PubMed Scraper**: Fetch papers by institution
- **ClinicalTrials.gov Scraper**: Fetch trials by sponsor
- **Author Network Extractor**: Build collaboration networks

### Graph Database (Tasks 4-5)
- **Neo4j Setup**: Database schema and operations
- **Graph Analytics**: Centrality, community detection

### Embeddings & Clustering (Tasks 8-9)
- **Embeddings**: Sentence-BERT embeddings + ChromaDB
- **Clustering**: UMAP + HDBSCAN pipeline

### Visualization (Tasks 6-7)
- **UMAP Visualization**: Interactive cluster plots
- **Network & Geographic Viz**: Collaboration networks and maps

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Usage

See individual module READMEs in `src/` directories.
```

---

### Step 11: Commit Integration

```bash
# Add all changes
git add -A

# Commit
git commit -m "Integration of Tasks 1-9: All modules merged and tested

Merged branches:
- Task 1: PubMed Scraper
- Task 2: ClinicalTrials.gov Scraper
- Task 3: Author Network Extractor
- Task 4: Neo4j Graph Setup
- Task 5: Graph Analytics
- Task 6: UMAP Visualization
- Task 7: Network & Geographic Visualization
- Task 8: Embeddings & Vector Store
- Task 9: Clustering Pipeline

All tests passing. Ready for Tasks 10-13 integration."
```

---

### Step 12: Push Integration Branch

```bash
# Push to remote
git push -u origin claude/integration-tasks-1-9-<session-id>
```

---

### Step 13: Verify Integration

Create a quick smoke test to verify all modules can be imported:

```python
# smoke_test.py
"""Quick smoke test to verify all modules are integrated correctly."""

def test_imports():
    """Test that all modules can be imported."""
    # Data Acquisition
    from data_acquisition.pubmed_scraper import PubMedScraper
    from data_acquisition.clinicaltrials_scraper import ClinicalTrialsScraper
    from data_acquisition.author_network import AuthorNetworkExtractor

    # Graph Database
    from graph_database.neo4j_setup import Neo4jConnection
    from graph_database.graph_analytics import GraphAnalytics

    # Embeddings
    from embeddings.embeddings_vectorstore import ArticleEmbedder, VectorStore
    from embeddings.clustering_pipeline import ClusteringPipeline

    # Visualization
    from visualization.umap_viz import UMAPVisualizer
    from visualization.network_geo_viz import NetworkVisualizer

    print("✅ All imports successful!")

if __name__ == "__main__":
    test_imports()
```

Run it:

```bash
python smoke_test.py
# Should output: ✅ All imports successful!
```

---

## Troubleshooting

### Issue: Merge conflict in `requirements.txt`

**Solution:**
```bash
# Accept both versions and manually combine
git checkout --ours requirements.txt  # or --theirs
# Edit file to include all dependencies
# Remove duplicates
# Sort alphabetically
git add requirements.txt
git merge --continue
```

### Issue: Import errors after merge

**Cause**: Incorrect directory structure or missing `__init__.py` files

**Solution:**
```bash
# Ensure all directories have __init__.py
find src/ -type d -exec touch {}/__init__.py \;
find tests/ -type d -exec touch {}/__init__.py \;
```

### Issue: Tests failing after merge

**Cause**: Conflicting test fixtures or incompatible changes

**Solution:**
```bash
# Run tests individually to identify failures
pytest tests/data_acquisition/ -v
pytest tests/graph_database/ -v
# Fix identified issues
```

### Issue: Duplicate code in different modules

**Cause**: Multiple tasks implemented similar utilities independently

**Solution:**
```bash
# Create a shared utilities module
mkdir -p src/common
# Move common code there
# Update imports
```

---

## Expected Results After Integration

✅ **Directory Structure**: All 9 modules in correct locations
✅ **Tests**: All tests pass with ≥ 80% coverage
✅ **Imports**: All modules can be imported without errors
✅ **Dependencies**: Single consolidated `requirements.txt`
✅ **Documentation**: Updated root README
✅ **Git History**: Clean merge history with descriptive commits

---

## Next Steps

After successful integration:

1. **Complete Tasks 10-13**: Cluster Labeling, Pipeline Orchestrator, Graph Integration, Dashboard
2. **Merge Tasks 10-13**: Use similar process for remaining tasks
3. **Final Integration**: Merge all tasks into production branch
4. **Deploy**: Follow `DEPLOYMENT_GUIDE.md`

---

## Quick Reference Commands

```bash
# Fetch all branches
git fetch --all

# Create integration branch
git checkout claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc
git checkout -b claude/integration-tasks-1-9-<session-id>

# Merge all tasks (replace with actual session IDs)
git merge origin/claude/task-01-pubmed-scraper-XXX --no-ff -m "Merge Task 1"
git merge origin/claude/task-02-clinicaltrials-scraper-XXX --no-ff -m "Merge Task 2"
git merge origin/claude/task-03-author-network-XXX --no-ff -m "Merge Task 3"
git merge origin/claude/task-04-neo4j-setup-XXX --no-ff -m "Merge Task 4"
git merge origin/claude/task-05-graph-analytics-XXX --no-ff -m "Merge Task 5"
git merge origin/claude/task-06-umap-viz-XXX --no-ff -m "Merge Task 6"
git merge origin/claude/task-07-network-geo-viz-XXX --no-ff -m "Merge Task 7"
git merge origin/claude/task-08-embeddings-XXX --no-ff -m "Merge Task 8"
git merge origin/claude/task-09-clustering-XXX --no-ff -m "Merge Task 9"

# Run tests
pytest tests/ -v --cov=src

# Commit and push
git add -A
git commit -m "Integration of Tasks 1-9: All modules merged and tested"
git push -u origin claude/integration-tasks-1-9-<session-id>
```

---

**End of Merge Instructions**
