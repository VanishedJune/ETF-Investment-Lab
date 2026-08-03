# Agent-Driven Dual-Market Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the seven-page application with a two-index weekly research workbench whose progressive model iterations are executed and persisted by local agents.

**Architecture:** Keep FastAPI/SQLite as the durable data and read API layer, add an offline deterministic iteration package invoked only by an agent-facing PowerShell command, and rebuild the Vue entry route as one light editorial research desk. The browser reads iteration artifacts but never trains a model.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, SQLite, pytest, Vue 3, TypeScript, ECharts, Playwright.

---

### Task 1: Repair Weekly Volume Provenance

**Files:**
- Modify: `backend/app/services/providers.py`
- Modify: `backend/app/services/market_data.py`
- Test: `backend/tests/test_market_data.py`

- [ ] **Step 1: Write failing tests**

Add tests proving that a valid upstream weekly record is retained, an empty/zero upstream weekly volume falls back to summed daily volume, and the rebuilt row source contains `AGGREGATED_DAILY_VOLUME`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_market_data.py -k "weekly_volume" -v
```

Expected: failures because weekly-provider validation and explicit fallback provenance do not exist.

- [ ] **Step 3: Implement the provider and fallback contract**

Add an optional weekly fetch path to `AkShareIndexProvider`. Validate date, volume, coverage and positive values. Update `_aggregate` so daily-derived weekly rows use `AGGREGATED_DAILY_VOLUME:<daily sources>` and replace stale/null weekly rows.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command and expect all selected tests to pass.

### Task 2: Define Progressive Weekly Model Behavior

**Files:**
- Create: `backend/app/agent_iterations/__init__.py`
- Create: `backend/app/agent_iterations/model.py`
- Create: `backend/app/agent_iterations/runner.py`
- Create: `backend/app/agent_iterations/storage.py`
- Test: `backend/tests/test_agent_iterations.py`

- [ ] **Step 1: Write failing model tests**

Tests must assert:

```python
assert model.fund_etf_ratio in {"7:3", "6:4", "5:5"}
assert model.fund_allocation + model.etf_allocation == model.target_position
assert all(isinstance(value, int) for value in (model.fund_allocation, model.etf_allocation))
assert next_model.parent_version == current_model.version
assert next_model.iteration == current_model.iteration + 1
assert max_weight_change <= 2
```

Also encode the four approved position examples as regression tests.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_iterations.py -v
```

Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement minimal deterministic model**

Implement weekly DIF/DEA/MACD feature interpretation, six bounded scores, confidence shrinkage, integer target allocation, three discrete fund/ETF ratios, batch plans and conservative mode. Map `DIP=DIF` and `EDA=DEA` explicitly in serialized output.

- [ ] **Step 4: Implement chronological online feedback**

The runner must stratify 100 cutoffs with a fixed seed, sort them chronologically, prevent overlap with unrevealed 20-trading-day labels, update each model from the immediately preceding result, and reject any source row after the cutoff.

- [ ] **Step 5: Verify GREEN**

Run the agent iteration tests and expect all to pass.

### Task 3: Persist Agent Protocol and Baseline

**Files:**
- Create: `AGENTS.md`
- Create: `docs/AGENT_MODEL_ITERATION.md`
- Create: `config/agent_iteration_policy.json`
- Create: `scripts/agent_iteration.py`
- Create: `scripts/agent-model-iteration.ps1`
- Create at runtime: `data/model_iterations/399006/manifest.json`
- Create at runtime: `data/model_iterations/NDX/manifest.json`
- Test: `backend/tests/test_agent_iteration_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test `status`, `baseline` and `due` modes against a temporary database/output directory. Assert that Web startup does not invoke the runner.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_iteration_cli.py -v
```

- [ ] **Step 3: Implement the explicit Agent command**

Support:

```powershell
.\scripts\agent-model-iteration.ps1 -Mode Status
.\scripts\agent-model-iteration.ps1 -Mode Baseline
.\scripts\agent-model-iteration.ps1 -Mode Due
```

Write atomic JSON manifests and JSONL audit rows. Store data hashes, model hashes, parent versions, cutoff dates and execution-agent labels.

- [ ] **Step 4: Generate the 2026-07-30 baseline**

Run:

```powershell
.\scripts\agent-model-iteration.ps1 -Mode Baseline
```

Expected: exactly 100 completed iterations for `399006` and 100 for `NDX`, both ending at model `M100`.

- [ ] **Step 5: Audit the baseline**

Run Status and verify counts, chronological cutoffs, parent hashes and no-leakage dates.

### Task 4: Add Read-Only Iteration API

**Files:**
- Create: `backend/app/services/agent_iteration_service.py`
- Modify: `backend/web.py`
- Test: `backend/tests/test_agent_iteration_api.py`

- [ ] **Step 1: Write failing API tests**

Assert the API returns summary metrics, the 100-point deviation curve, latest advice and model version, and exposes no training mutation endpoint.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_iteration_api.py -v
```

- [ ] **Step 3: Implement read endpoints**

Add:

```text
GET /api/agent-iterations/{instrument_code}
GET /api/agent-iterations/{instrument_code}/{iteration_number}
```

Restrict codes to `399006` and `NDX`. Read persisted artifacts only.

- [ ] **Step 4: Verify GREEN**

Run the API tests and the full backend suite.

### Task 5: Add Durable Investment Calendar

**Files:**
- Modify: `backend/app/models/models.py`
- Modify: `backend/app/database/migrations.py`
- Create: `backend/app/schemas/investment_calendar.py`
- Create: `backend/app/services/investment_calendar_service.py`
- Modify: `backend/web.py`
- Test: `backend/tests/test_investment_calendar.py`

- [ ] **Step 1: Write failing persistence tests**

Test create/update/delete/list, positive share validation, allowed instruments, allowed product types and net-share calculation after reopening SQLite.

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_investment_calendar.py -v
```

- [ ] **Step 3: Add schema version 13**

Create `investment_calendar_entries` with instrument, product, side, operation date, quantity, optional price and note. Add CRUD endpoints under `/api/investment-calendar`.

- [ ] **Step 4: Verify GREEN**

Run the calendar tests and database migration tests.

### Task 6: Build the Single Research Workbench

**Files:**
- Modify: `frontend/src/router.ts`
- Modify: `frontend/src/App.vue`
- Create: `frontend/src/views/ResearchWorkbenchView.vue`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/tests/critical-pages.spec.ts`

- [ ] **Step 1: Write failing Playwright assertions**

Assert one route, two index buttons, default weekly timeframe, weekly volume, colored DIP/DIF and EDA/DEA labels, three-ratio advice, 100-point iteration chart and investment-calendar CRUD.

- [ ] **Step 2: Verify RED**

```powershell
npm.cmd run test:e2e --prefix frontend
```

Expected: failures because the new workbench does not exist.

- [ ] **Step 3: Implement the light editorial desk**

Use a warm paper background, ink-blue typography, vermilion/cyan market accents, a fixed research rail, generous 16px+ body type and dense but readable ECharts panels. Remove all old navigation links and redirect every path to `/`.

- [ ] **Step 4: Verify build and GREEN**

```powershell
npm.cmd run build --prefix frontend
npm.cmd run test:e2e --prefix frontend
```

### Task 7: Documentation, Data Repair and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/使用说明.md`
- Runtime update: `data/investment_lab.db`

- [ ] **Step 1: Repair derived periods**

Run the explicit aggregation and indicator recalculation for `399006` and `NDX`, then audit missing weekly/monthly volume counts.

- [ ] **Step 2: Update documentation**

Document the single page, two markets, Agent-only iteration command, monthly two-iteration protocol and three integer ratios.

- [ ] **Step 3: Run the full verification**

```powershell
.\scripts\test.ps1
npm.cmd run test:e2e --prefix frontend
.\scripts\agent-model-iteration.ps1 -Mode Status
```

Expected: backend tests pass, frontend build passes, Playwright passes, and both manifests report 100 completed baseline iterations.

The workspace is not a Git repository, so plan checkpoints are recorded through passing tests and generated audit manifests rather than commits.
