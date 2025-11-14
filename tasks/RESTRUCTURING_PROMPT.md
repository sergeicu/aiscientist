# Prompt for Claude Code: Repository Restructuring

Copy this entire prompt and paste it into Claude Code:

---

I need to restructure my repository to integrate my existing clinical trial classifier code with 13 completed parallel tasks.

**Current Situation:**

1. **Original Code** (already in repo):
   - Clinical trial classifier using Ollama (local LLM)
   - Files: `main.py`, `src/*.py`, `prompts/clinical_trial_classifier.yaml`
   - Documentation: `QUICKSTART.md`, `CLASSIFIER_README.md`, `PROMPT_TUNING_GUIDE.md`
   - Purpose: Classify PubMed articles as clinical trials

2. **Tasks 1-13** (completed on separate branches):
   - Data acquisition, graph analytics, embeddings, clustering, dashboard
   - Each on branch: `claude/task-XX-<name>-<session-id>`
   - Purpose: Complete research intelligence platform

3. **Challenge**: Need to integrate both without losing functionality

**Your Task:**

Please read the file `tasks/RESTRUCTURING_GUIDE.md` which contains complete step-by-step instructions for restructuring the repository.

Then execute the restructuring by following these phases:

## Phase 1: Preparation

1. Create restructuring branch from current branch
2. Create backup
3. Create new directory structure

## Phase 2: Move Original Code to Classification Module

Move existing files to new locations:
- `src/*.py` → `src/classification/`
- `prompts/clinical_trial_classifier.yaml` → `prompts/classification/`
- Documentation → `docs/`
- `main.py` → `cli/main.py`

Update all imports to reflect new structure.

## Phase 3: Merge Tasks 1-13

Follow the complete merge process from `MERGE_INSTRUCTIONS_ALL_TASKS.md`:
- Merge all 13 task branches in dependency order (4 waves)
- This creates: `src/data_acquisition/`, `src/embeddings/`, `src/graph_database/`, etc.

## Phase 4: Integration

1. Update top-level `src/__init__.py` to export all modules
2. Consolidate `requirements.txt` with dependencies from all modules
3. Fix any import conflicts
4. Create `src/common/` for shared utilities

## Phase 5: Create New Structure

1. Create unified CLI in `cli/main.py` with subcommands:
   - `classifier` - Classification commands
   - `pipeline` - Data collection commands
   - `dashboard` - Dashboard commands

2. Create comprehensive examples in `examples/`:
   - `end_to_end_workflow.py` - Complete workflow
   - `classifier_example.py` - Just classification
   - `pipeline_example.py` - Just pipeline

3. Reorganize documentation in `docs/`:
   - `docs/quickstart/` - Quick start guides
   - `docs/guides/` - Detailed guides
   - `docs/architecture/` - Architecture documentation

## Phase 6: Update Documentation

1. Create `docs/architecture/system_architecture.md` explaining complete system
2. Update root `README.md` to reflect new structure
3. Create `docs/README.md` as documentation index
4. Update `.env.example` with all required environment variables

## Phase 7: Testing & Finalization

1. Run all tests: `pytest tests/ -v --cov=src`
2. Create and run integration smoke test
3. Update `.gitignore`
4. Create `setup.py` for package installation
5. Commit all changes
6. Push to remote

---

## Expected Final Structure

```
aiscientist/
├── src/
│   ├── classification/         # Original code (classifier)
│   ├── data_acquisition/       # Tasks 1-3
│   ├── embeddings/             # Tasks 8-10
│   ├── graph_database/         # Tasks 4-5
│   ├── visualization/          # Tasks 6-7
│   ├── pipeline/               # Task 11
│   ├── graph_integration/      # Task 12
│   ├── dashboard/              # Task 13
│   └── common/                 # Shared utilities
├── cli/                        # Unified CLI
├── docs/                       # All documentation
├── tests/                      # All tests
├── examples/                   # Usage examples
├── prompts/                    # LLM prompts
├── requirements.txt            # Consolidated deps
└── README.md                   # Updated README
```

---

## Key Points

1. **No Code Loss**: All original classifier code preserved in `src/classification/`
2. **Clean Separation**: Classifier is one module among many
3. **Unified Interface**: Single CLI for all operations
4. **Clear Documentation**: Comprehensive docs for complete system
5. **Ready for Phase 4**: Structure supports future multi-agent system

---

## Important Notes

- The original classifier code is **complementary** to Tasks 1-13, not a duplicate
- Task 1 (PubMed Scraper) = Data Collection
- Original Code (Classifier) = Data Processing/Classification
- Both work together in the complete workflow
- Preserve all functionality from both

---

## Quality Checks

After restructuring:

- [ ] All original classifier files in `src/classification/`
- [ ] All task modules in correct locations
- [ ] Unified `requirements.txt` with all dependencies
- [ ] All imports updated correctly
- [ ] All tests pass
- [ ] CLI works with all subcommands
- [ ] Documentation updated and organized
- [ ] Examples demonstrate full workflow

---

## Deliverables

1. Restructured repository with clean organization
2. All modules integrated and working
3. Unified CLI with classifier, pipeline, dashboard commands
4. Comprehensive documentation
5. Integration examples
6. All tests passing
7. Ready for deployment

---

Please start by reading `tasks/RESTRUCTURING_GUIDE.md` and then proceed with the restructuring.

Take your time, verify each phase before proceeding to the next. The goal is a clean, well-organized, production-ready codebase that integrates everything seamlessly.
