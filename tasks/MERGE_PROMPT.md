# Prompt for Claude Code: Merge Tasks 1-9

Copy this entire prompt and paste it into a new Claude Code session:

---

I have completed Tasks 1-9 of my project, each on a separate branch. I need to merge all these branches into a single integration branch.

**Completed Task Branches:**
- `claude/task-01-pubmed-scraper-<session-id>`
- `claude/task-02-clinicaltrials-scraper-<session-id>`
- `claude/task-03-author-network-<session-id>`
- `claude/task-04-neo4j-setup-<session-id>`
- `claude/task-05-graph-analytics-<session-id>`
- `claude/task-06-umap-viz-<session-id>`
- `claude/task-07-network-geo-viz-<session-id>`
- `claude/task-08-embeddings-<session-id>`
- `claude/task-09-clustering-<session-id>`

**Base Branch:** `claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc`

**Your Task:**

Please read the file `tasks/MERGE_INSTRUCTIONS.md` which contains complete step-by-step instructions for merging all task branches.

Then execute the merge by following these steps:

1. **Fetch all remote branches** to get the latest code
2. **List all task branches** to identify exact branch names (with their session IDs)
3. **Create a new integration branch** named `claude/integration-tasks-1-9-<your-session-id>`
4. **Merge each task branch** one by one into the integration branch
5. **Resolve any conflicts** if they occur (most likely in `requirements.txt` or root README)
6. **Consolidate dependencies** in a single `requirements.txt` file
7. **Verify directory structure** matches the expected structure
8. **Run all tests** to ensure everything works: `pytest tests/ -v --cov=src`
9. **Create/update root README** documenting all integrated modules
10. **Commit the integration** with a descriptive message
11. **Push the integration branch** to remote

**Expected Final Directory Structure:**

```
src/
├── data_acquisition/
│   ├── pubmed_scraper/          (Task 1)
│   ├── clinicaltrials_scraper/  (Task 2)
│   └── author_network/          (Task 3)
├── graph_database/
│   ├── neo4j_setup/             (Task 4)
│   └── graph_analytics/         (Task 5)
├── embeddings/
│   ├── embeddings_vectorstore/  (Task 8)
│   └── clustering_pipeline/     (Task 9)
└── visualization/
    ├── umap_viz/                (Task 6)
    └── network_geo_viz/         (Task 7)
```

**Important Notes:**

- Use `--no-ff` flag for merges to preserve branch history
- If conflicts occur in `requirements.txt`, combine all dependencies from all tasks
- After merging, all tests should pass with ≥80% coverage
- Create a smoke test to verify all modules can be imported

**Deliverable:**

- New integration branch with all Tasks 1-9 merged
- All tests passing
- Consolidated `requirements.txt`
- Updated root README
- Branch pushed to remote

Please start by reading `tasks/MERGE_INSTRUCTIONS.md` and then proceed with the merge.
