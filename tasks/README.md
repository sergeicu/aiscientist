# AI Scientist - Parallel Task Specifications

This directory contains 13 self-contained task specifications designed for parallel execution by separate Claude Code instances.

---

## Overview

Each task follows **Test-Driven Development (TDD)** principles and is completely self-sufficient - no need to read other files.

### Task Organization

```
tasks/
├── README.md                                    (this file)
├── QUALITY_VALIDATION_CHECKLIST.md             (how to validate each task)
├── INTEGRATION_GUIDE.md                        (how to integrate all tasks)
├── DEPLOYMENT_GUIDE.md                         (how to deploy to production)
│
├── task_01_pubmed_scraper/
│   └── TASK_SPEC.md                            (PubMed institutional scraper)
│
├── task_02_clinicaltrials_scraper/
│   └── TASK_SPEC.md                            (ClinicalTrials.gov scraper)
│
├── task_03_author_network_extractor/
│   └── TASK_SPEC.md                            (Author collaboration network)
│
├── task_04_neo4j_graph_setup/
│   └── TASK_SPEC.md                            (Neo4j database setup)
│
├── task_05_graph_analytics/
│   └── TASK_SPEC.md                            (Graph analytics & queries)
│
├── task_06_umap_visualization/
│   └── TASK_SPEC.md                            (UMAP clustering visualization)
│
├── task_07_network_geo_viz/
│   └── TASK_SPEC.md                            (Network & geographic viz)
│
├── task_08_embeddings_vectorstore/
│   └── TASK_SPEC.md                            (Sentence embeddings & ChromaDB)
│
├── task_09_clustering_pipeline/
│   └── TASK_SPEC.md                            (UMAP + HDBSCAN clustering)
│
├── task_10_cluster_labeling/
│   └── TASK_SPEC.md                            (Cluster labeling & topics)
│
├── task_11_data_pipeline_orchestrator/
│   └── TASK_SPEC.md                            (Unified data pipeline)
│
├── task_12_graph_database_integration/
│   └── TASK_SPEC.md                            (Load data into Neo4j)
│
├── task_13_streamlit_dashboard/
│   └── TASK_SPEC.md                            (End-user dashboard)
│
└── phase_03_discussion/
    └── APPROACHES_ANALYSIS.md                  (Embedding vs other approaches)
```

---

## Quick Start

### For Project Manager

**To launch all tasks in parallel:**

1. Open 10 Claude Code windows (one for each independent task)
2. Use the prompts from `PROMPTS.md` (see below)
3. Monitor progress using `QUALITY_VALIDATION_CHECKLIST.md`
4. Once complete, follow `INTEGRATION_GUIDE.md`
5. Deploy using `DEPLOYMENT_GUIDE.md`

### For Individual Developer

**To work on a single task:**

1. Read the task specification (e.g., `task_01_pubmed_scraper/TASK_SPEC.md`)
2. Create a new branch: `claude/task-01-pubmed-scraper-<session-id>`
3. Follow TDD: write tests first, then implementation
4. Run tests: `pytest tests/ -v --cov`
5. Commit and push when complete

---

## Task Dependencies

### Wave 1: Can Start Immediately (8 tasks)
- ✅ Task 1: PubMed Scraper
- ✅ Task 2: ClinicalTrials.gov Scraper
- ✅ Task 3: Author Network Extractor
- ✅ Task 4: Neo4j Graph Setup
- ✅ Task 6: UMAP Visualization
- ✅ Task 7: Network & Geo Visualization
- ✅ Task 8: Embeddings & Vector Store

### Wave 2: After Prerequisites (5 tasks)
- ⏳ Task 5: Graph Analytics (needs Task 4)
- ⏳ Task 9: Clustering Pipeline (needs Task 8)
- ⏳ Task 10: Cluster Labeling (needs Task 9)
- ⏳ Task 11: Data Pipeline (needs Tasks 1, 2, 3)
- ⏳ Task 12: Graph Integration (needs Tasks 3, 4, 11)
- ⏳ Task 13: Dashboard (needs all previous tasks for data)

---

## Prompts for Parallel Execution

### Task 1: PubMed Scraper
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_01_pubmed_scraper/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-01-pubmed-scraper-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 2: ClinicalTrials.gov Scraper
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_02_clinicaltrials_scraper/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-02-clinicaltrials-scraper-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 3: Author Network Extractor
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_03_author_network_extractor/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-03-author-network-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 4: Neo4j Graph Setup
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_04_neo4j_graph_setup/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-04-neo4j-setup-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 5: Graph Analytics (⚠️ Requires Task 4)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_05_graph_analytics/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task requires Task 4 (Neo4j setup) to be completed first.

Your job:
1. Create a new branch named claude/task-05-graph-analytics-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 6: UMAP Visualization
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_06_umap_visualization/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-06-umap-viz-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 7: Network & Geographic Visualization
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_07_network_geo_viz/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-07-network-geo-viz-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 8: Embeddings & Vector Store
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_08_embeddings_vectorstore/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

Your job:
1. Create a new branch named claude/task-08-embeddings-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 9: Clustering Pipeline (⚠️ Requires Task 8)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_09_clustering_pipeline/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task requires Task 8 (Embeddings) to be completed first.

Your job:
1. Create a new branch named claude/task-09-clustering-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 10: Cluster Labeling (⚠️ Requires Task 9)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_10_cluster_labeling/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task requires Task 9 (Clustering Pipeline) to be completed first.

Your job:
1. Create a new branch named claude/task-10-cluster-labeling-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 11: Data Pipeline Orchestrator (⚠️ Requires Tasks 1, 2, 3)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_11_data_pipeline_orchestrator/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task requires Tasks 1, 2, and 3 to be completed first.

Your job:
1. Create a new branch named claude/task-11-pipeline-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 12: Graph Database Integration (⚠️ Requires Tasks 3, 4, 11)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_12_graph_database_integration/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task requires Tasks 3, 4, and 11 to be completed first.

Your job:
1. Create a new branch named claude/task-12-graph-integration-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

### Task 13: Streamlit Dashboard (⚠️ Requires all previous tasks for data)
```
Please checkout branch claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc and read the file tasks/task_13_streamlit_dashboard/TASK_SPEC.md.

This file contains a complete, self-sufficient task specification. Do NOT read any other files.

⚠️ IMPORTANT: This task can be started but will need data from previous tasks to test fully.

Your job:
1. Create a new branch named claude/task-13-dashboard-<session-id>
2. Implement the task following TDD principles (write tests first, then implementation)
3. Run all tests to ensure they pass
4. Commit your work with clear messages
5. Push to your branch when complete

Start by reading the TASK_SPEC.md file and creating the test files as specified.
```

---

## Validation & Quality Control

After each task is completed, use the validation checklist:

```bash
# See QUALITY_VALIDATION_CHECKLIST.md for detailed validation steps

# Quick validation:
1. All tests pass: pytest tests/ -v
2. Coverage ≥ 80%: pytest tests/ --cov=. --cov-report=term-missing
3. Functional test: Manual verification with real data
4. Output format: Matches specification
5. Documentation: README and docstrings complete
```

---

## Integration

Once all tasks are complete, follow `INTEGRATION_GUIDE.md`:

1. Verify individual tasks
2. Set up infrastructure (Neo4j)
3. Run data collection pipeline (Task 11)
4. Load data into Neo4j (Task 12)
5. Generate embeddings & clusters (Tasks 8-10)
6. Launch dashboard (Task 13)

---

## Deployment

For production deployment, see `DEPLOYMENT_GUIDE.md`:

- **Quick**: Streamlit Cloud (free, 30 minutes)
- **Professional**: Vercel with custom domain
- **Enterprise**: AWS ECS with Neo4j Aura

---

## Support

**For task-specific questions:**
- Read the task's TASK_SPEC.md (completely self-sufficient)

**For integration questions:**
- See INTEGRATION_GUIDE.md

**For deployment questions:**
- See DEPLOYMENT_GUIDE.md

**For quality validation:**
- See QUALITY_VALIDATION_CHECKLIST.md

---

## Project Statistics

- **Total Tasks**: 13
- **Independent Tasks**: 8 (can run in parallel immediately)
- **Dependent Tasks**: 5 (need prerequisites)
- **Total Lines of Code**: ~50,000+ (across all tasks)
- **Test Coverage Target**: ≥ 80%
- **Estimated Total Time**: 60-80 hours (10-15 hours if fully parallel)

---

## Architecture Overview

```
Data Collection → Graph Database → Analytics
                ↓
            Embeddings → Clustering → Labeling
                ↓
          Streamlit Dashboard → Hospital CMO
```

---

**Last Updated**: 2025-11-14
**Version**: 1.0
**Status**: Ready for Parallel Execution

---

**End of README**
