# AI Agent Operating Guidelines & Notion Governance Protocol

This repository is governed by a strict **Notion-First Lifecycle Hook**. All AI agents (including Antigravity, Gemini, and subagents) must adhere to the following execution protocol:

---

## 1. Notion-First Planning Hook (Pre-Implementation)

Before implementing any new feature, bugfix, refactoring, or deleting/modifying existing tasks:
1. **Check & Update Notion First**: Consult the Notion Scrum Delivery Board:
   👉 [**07. Agile Scrum Board & Sprint Delivery Hub**](https://app.notion.com/p/07-Agile-Scrum-Board-Sprint-Delivery-Hub-3c50d8f7461e81829c2cc368ee95673d)
2. **Task Registration**: If a task is not present, add it to the appropriate Sprint column (e.g. `Backlog` or `To Do`) with:
   - **Task Title & Description**
   - **Priority**: `P0 Must`, `P1 Should`, or `P2 Could`
   - **Story Points**: Fibonacci estimate (`1`, `2`, `3`, `5`, `8`)
   - **Epic**: `Platform`, `Hardware`, `Data Eng`, `ML / DL`, `Temporal`, `MongoDB`, `Ollama`, or `Release`
   - **Acceptance Criteria**: Concrete checklist.

---

## 2. Evidence & Verification Hook (During Implementation)

Every implemented feature must be accompanied by concrete verification evidence:
1. **Automated Unit / Integration Tests**: All new modules must have corresponding `pytest` tests under `tests/`.
2. **Zero Regressions**: Run `pytest -v` and confirm 100% pass rate before committing.
3. **Audit Artifacts**: Capture endpoint status, benchmark latency, or test metrics.

---

## 3. Notion Status & Evidence Update Hook (Post-Implementation)

Immediately upon completing a task and committing to Git:
1. **Update Notion Task Status**:
   - Move task to **`Done (Completed Stories)`**.
   - Record the **Completion Date** (e.g. `Aug 23, 2026`).
   - Record the **Git Commit Hash** and message in the Notion Evidence Log.
2. **Push to Remote**: Ensure all commits are pushed to `origin/main` on GitHub:
   - Repository: `https://github.com/cvrvai/AnimalLens`
