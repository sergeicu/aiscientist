# Prompt for Claude Code: Merge ALL Tasks 1-13

Copy this entire prompt and paste it into a new Claude Code session:

---

I have completed ALL 13 tasks of my AI Scientist Research Intelligence Platform project, each on a separate branch. I need to merge all these branches into a single production-ready integration branch.

**Completed Task Branches (All 13):**

**Phase 1-2: Data Acquisition**
- `claude/task-01-pubmed-scraper-<session-id>`
- `claude/task-02-clinicaltrials-scraper-<session-id>`
- `claude/task-03-author-network-<session-id>`

**Phase 3: Embeddings & Clustering**
- `claude/task-08-embeddings-<session-id>`
- `claude/task-09-clustering-<session-id>`
- `claude/task-10-cluster-labeling-<session-id>`

**Phase 5: Knowledge Graph**
- `claude/task-04-neo4j-setup-<session-id>`
- `claude/task-05-graph-analytics-<session-id>`

**Phase 6: Visualization**
- `claude/task-06-umap-viz-<session-id>`
- `claude/task-07-network-geo-viz-<session-id>`

**Integration Layer**
- `claude/task-11-pipeline-<session-id>`
- `claude/task-12-graph-integration-<session-id>`
- `claude/task-13-dashboard-<session-id>`

**Base Branch:** `claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc`

---

## Your Task

Please read the file `tasks/MERGE_INSTRUCTIONS_ALL_TASKS.md` which contains complete step-by-step instructions for merging all 13 task branches into a production-ready system.

Then execute the complete integration by following these steps:

### 1. Preparation
- **Fetch all remote branches** to ensure you have latest code
- **List all task branches** to identify exact branch names with session IDs
- **Create production integration branch** named `claude/production-integration-v1-<your-session-id>`

### 2. Wave 1: Core Independent Modules (7 tasks)
Merge these in any order (no dependencies):
- Task 1: PubMed Scraper
- Task 2: ClinicalTrials Scraper
- Task 3: Author Network
- Task 4: Neo4j Setup
- Task 6: UMAP Visualization
- Task 7: Network & Geo Visualization
- Task 8: Embeddings

**After Wave 1:** Run tests to verify base modules work

### 3. Wave 2: First-Level Dependencies (2 tasks)
- Task 5: Graph Analytics (needs Task 4)
- Task 9: Clustering Pipeline (needs Task 8)

**After Wave 2:** Verify dependent modules integrate correctly

### 4. Wave 3: Second-Level Dependencies (2 tasks)
- Task 10: Cluster Labeling (needs Task 9)
- Task 11: Pipeline Orchestrator (needs Tasks 1, 2, 3)

**After Wave 3:** Test pipeline functionality

### 5. Wave 4: Final Integration (2 tasks)
- Task 12: Graph Database Integration (needs Tasks 3, 4, 11)
- Task 13: Streamlit Dashboard (needs all tasks)

**After Wave 4:** System is complete

### 6. Post-Merge Tasks
- **Consolidate `requirements.txt`** - Combine all dependencies from all tasks
- **Create comprehensive README** - Document complete system
- **Create `.env.example`** - Environment configuration template
- **Create `setup.py`** - Package installation configuration
- **Run complete test suite** - `pytest tests/ -v --cov=src`
- **Create and run smoke test** - Verify all modules import and work
- **Commit everything** with descriptive message
- **Push to remote**

---

## Expected Final Directory Structure

```
aiscientist/
├── src/
│   ├── data_acquisition/        (Tasks 1, 2, 3)
│   ├── graph_database/          (Tasks 4, 5)
│   ├── embeddings/              (Tasks 8, 9, 10)
│   ├── visualization/           (Tasks 6, 7)
│   ├── pipeline/                (Task 11)
│   ├── graph_integration/       (Task 12)
│   └── dashboard/               (Task 13)
├── tests/                       (mirrors src/)
├── data/config/
├── .streamlit/
├── requirements.txt             (consolidated)
├── README.md                    (comprehensive)
├── setup.py
├── .env.example
└── smoke_test_complete.py
```

---

## Important Merge Strategy

**Use `--no-ff` (no fast-forward) for all merges** to preserve branch history:

```bash
git merge origin/claude/task-XX-<id> --no-ff -m "Merge Task X: Description"
```

**Merge in waves** respecting dependencies to avoid integration issues.

---

## Conflict Resolution

You will likely encounter conflicts in:

### 1. `requirements.txt` - EXPECTED
Multiple tasks added dependencies.

**Resolution:**
- Combine ALL unique dependencies
- Remove duplicates
- Organize by category (Data Acquisition, Graph DB, Embeddings, etc.)
- Sort alphabetically within each category

### 2. Root `README.md` - POSSIBLE
Multiple tasks may have updated root README.

**Resolution:**
- Create comprehensive README documenting ALL modules
- Include installation, usage, testing, deployment
- See template in MERGE_INSTRUCTIONS_ALL_TASKS.md

### 3. `.gitignore` - POSSIBLE
Different ignore patterns from different tasks.

**Resolution:**
- Combine all patterns
- Remove duplicates

### 4. Config files - POSSIBLE
Multiple tasks may have created config files.

**Resolution:**
- Keep all config files, they serve different purposes
- Document which config is for which component

---

## Verification Steps

After merge is complete, verify:

### ✅ Directory Structure
```bash
ls -la src/
# Should show: data_acquisition/, graph_database/, embeddings/,
#              visualization/, pipeline/, graph_integration/, dashboard/
```

### ✅ All Tests Pass
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# Should show: 150+ tests passed, coverage ≥ 80%
```

### ✅ Smoke Test Passes
```bash
python smoke_test_complete.py
# Should show: All imports successful, all functionality tests pass
```

### ✅ Dashboard Runs
```bash
streamlit run src/dashboard/app.py
# Should launch at http://localhost:8501
```

---

## Deliverables

At the end of this task, you should have:

1. ✅ **Production integration branch** with all 13 tasks merged
2. ✅ **Consolidated `requirements.txt`** with all dependencies
3. ✅ **Comprehensive README.md** documenting the complete system
4. ✅ **Package configuration** (`setup.py`)
5. ✅ **Environment template** (`.env.example`)
6. ✅ **All tests passing** (≥150 tests, ≥80% coverage)
7. ✅ **Smoke test** created and passing
8. ✅ **Clean git history** with descriptive merge commits
9. ✅ **Branch pushed to remote**
10. ✅ **System ready for deployment**

---

## Quality Gates

**Do NOT proceed to next wave until:**
- Current wave merges complete
- No unresolved conflicts
- Tests for merged modules pass
- Directory structure is correct

**Do NOT complete task until:**
- All 13 tasks merged
- All tests pass
- Smoke test passes
- Documentation complete
- Branch pushed to remote

---

## Success Criteria

The integration is successful when:

1. All 13 task branches merged in correct dependency order
2. All tests pass with ≥80% coverage
3. Smoke test verifies all modules can be imported
4. Dashboard launches successfully
5. Complete system documentation exists
6. Production-ready branch pushed to remote
7. No merge conflicts remain unresolved
8. System is ready for deployment following DEPLOYMENT_GUIDE.md

---

## Final Commit Message Template

Use this template for your final commit:

```
Complete production integration of all 13 tasks

Merged all task branches in dependency order:

Wave 1 (Core Modules):
- Task 1: PubMed Scraper
- Task 2: ClinicalTrials.gov Scraper
- Task 3: Author Network Extractor
- Task 4: Neo4j Graph Setup
- Task 6: UMAP Visualization
- Task 7: Network & Geographic Visualization
- Task 8: Embeddings & Vector Store

Wave 2 (First-Level Dependencies):
- Task 5: Graph Analytics
- Task 9: Clustering Pipeline

Wave 3 (Second-Level Dependencies):
- Task 10: Cluster Labeling
- Task 11: Pipeline Orchestrator

Wave 4 (Final Integration):
- Task 12: Graph Database Integration
- Task 13: Streamlit Dashboard

Production-ready features:
✅ All 13 modules integrated
✅ Consolidated requirements.txt
✅ Comprehensive README
✅ Environment configuration template
✅ Package setup (setup.py)
✅ Complete test suite (150+ tests, 85%+ coverage)
✅ Integration smoke test
✅ Ready for deployment

All tests passing. System is production-ready.
```

---

**Please start by reading `tasks/MERGE_INSTRUCTIONS_ALL_TASKS.md` and then proceed with the complete production integration.**

**Take your time and verify each wave before proceeding to the next. The goal is a clean, working, production-ready system.**
