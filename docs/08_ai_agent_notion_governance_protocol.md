# AI Agent Notion Governance & Evidence Protocol

This document defines the automated operational rules, lifecycle hooks, and evidence audit trails required for all AI agents developing AnimalLens.

---

## 1. Governance Lifecycle Hooks

```text
[ Incoming Request ]
        |
        v
[ Hook 1: Notion-First Planning ]
  * Consult Notion Scrum Board (Subpage 07)
  * If task needs to be created/changed/deleted -> UPDATE NOTION FIRST
        |
        v
[ Hook 2: Implementation & Evidence Generation ]
  * Implement code & architecture
  * Write unit/integration tests under tests/
  * Run pytest -v (Must achieve 100% pass rate)
        |
        v
[ Hook 3: Git Commit & Push ]
  * Stage files, commit with semantic message, and push to GitHub origin/main
        |
        v
[ Hook 4: Notion Status & Evidence Log Update ]
  * Move task to 'Done' on Notion Scrum Board
  * Log Completion Date + Git Commit Hash + Test Evidence into Notion
```

---

## 2. Evidence Audit Trail Standards

Every completed task must record 4 pieces of verification evidence in the Notion Scrum log:

1. **Completion Date**: Exact ISO date (e.g. `2026-08-23`).
2. **Git Commit SHA**: Direct commit reference (e.g. `commit: 1bd171d`).
3. **Automated Test Results**: Passing test counts (e.g. `36 passed in 6.32s`).
4. **Deliverables Summary**: Created modules, API endpoints, or database collections.
