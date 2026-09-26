# Project operating instructions

The active model-core workflow is Investment Research
**V3.7_DEEPSEEK_FUSION_8W**. Read `docs/V37_OPTIMIZATION_PLAN.md` completely
before changing model-core, training, probability, scenario, strategy,
simulation-account, promotion, DeepSeek-AI, fusion, or schema-27 code.
V3.7 runs a local multi-timeframe quant model and an independent DeepSeek
analyst in parallel; DeepSeek only has prediction/proposal rights and can
never bypass the local replay gate or modify frozen records.

V3.7 is the only effective requirements and acceptance version. V3.5
(`docs/V35_MODEL_PROTOCOL.md`), V3.5.1, and V3.6 remain as read-only frozen
records; they are never modified. V3.7 appends `v37_*` tables (schema 29) and
reuses `v35_*`/`v36_*` tables under its own `protocol_version`; it never
overwrites, migrates, or reinterprets earlier records.

## V3.7 model-core boundary

- The active horizon is exactly 8 complete weeks; labels are
  `Close_(T+k)/Close_T - 1` for `k = 1..8`.
- Local models use weekly (52–104W) and daily (60–100D) branches plus
  interaction features (`MULTI_TIMEFRAME_STATE`); an incomplete natural week
  never leaks as a complete weekly bar.
- DeepSeek calls are cached and frozen in `v37_ai_requests`/`v37_ai_forecasts`;
  the same `(market, anchor, input_hash, model, prompt_version)` is never
  re-called or overwritten; historical inputs are anonymized and
  point-in-time audited.
- Local/AI/Fusion accounts start from the same `v36_account_snapshots` row and
  run through the same execution engine (T+1 shares, real costs, SELL,
  pending-batch inheritance). AI never exceeds the 80% cap and never shorts.
- AI raw confidence is stored as `ai_raw_score`; only calibrated
  `ai_calibrated_probability` (matured Forward-OOS) enters fusion.
- Fusion AI weight is capped by Forward-OOS windows: <16 shadow, 16–29 10%,
  30–49 20%, 50–99 30%, ≥100 40%, with quality/account/participation gates.
- Repair proposals only change whitelisted parameters; any
  `LOCAL_REPAIR_CHALLENGER` must pass the local replay/promotion gates.
- The production Bootstrap runs on the production database with idempotent
  weekly transactions; test-database results are never copied into production.

## V3.5 model-core boundary

- The active horizon is exactly 8 complete weeks. Direction probabilities are
  frozen at 4 and 8 weeks; no V3.5 page, API, table, or report may output
  9–13 week or 12/13-week current results.
- Labels are `Close_(T+k)/Close_T - 1` for `k = 1..8`; a forecast becomes
  `FULLY_MATURE_8W` only after the full 8-week outcome is known.
- The initial Champion is trained from scratch (399006 ≥24 matured 8W labels,
  159941 ≥12 in the strongly regularized small-sample mode) and becomes
  effective from the next complete week.
- 399006 may use its own OHLCV, derived indicators, and approved valuation
  fields. 159941 may use only its own OHLCV and derived indicators; NDX, FX,
  NAV, premium/discount, and external valuation inputs are forbidden.
- Strategy decisions use the DIF four-quadrant state and confirmation-state
  position caps; `UNCONFIRMED` only lowers the cap and never forces 0%.
- Champion promotion uses ≥8 sealed out-of-sample 8-week 100,000-CNY accounts
  with real costs/slippage; no-action windows cannot be used as an advantage.
- The production Bootstrap runs on the production database with idempotent
  weekly transactions and resume-from-failure support; test-database results
  are never copied into production.

## V3.4.1 model-core boundary (historical reference)

The V3.4.1-13W workflow (schema 21/22 tables, 13-week labels, 4/8/13 frozen
probabilities) is preserved as read-only legacy. Read
`docs/V341_MODEL_PROTOCOL.md` before changing schema-21 code for
backward-compatibility audits. V3.4.1 is a deterministic
local desktop/backend workflow. It does not call an Agent, LLM, Codex, OpenAI,
Anthropic, or any external AI service at runtime.

Before changing strategy, training, analysis, position-calendar, or market-data
code:

1. Read `docs/V341_MODEL_PROTOCOL.md` and `docs/V33_MODEL_PROTOCOL.md`
   completely. Read
   `docs/V32_MODEL_PROTOCOL.md`, `docs/V31_MODEL_PROTOCOL.md`, and
   `docs/AGENT_MODEL_ITERATION.md` when checking backward compatibility.
2. Preserve `data/model_iterations/399006`, `data/model_iterations/NDX`, and
   `data/weekly_analysis_v2` as read-only legacy audit artifacts.
3. Preserve every V1/V2/V3.1/V3.2 database table. V3.3 model/data/training
   state writes only to independent `v33_*` tables. The existing local
   investment-calendar table is the sole shared user-ledger exception.
4. Run `scripts/audit-v33-readonly-baseline.py` before release. Any mismatch
   against `reports/V33_BASELINE_FREEZE.md` is a release failure.
5. Do not run `scripts/agent-model-iteration.ps1`, V2 iteration writers, or a
   V3.2 bootstrap to generate or maintain V3.3 data.

## V3.4.1 model-core boundary

- The active horizon is exactly 13 complete weeks, with frozen direction
  probabilities and thresholds at 4, 8, and 13 weeks.
- Schema 21 appends only the 17 `v341_*` tables listed in
  `docs/V341_MODEL_PROTOCOL.md`. Never overwrite a V3.2, V3.3, or old V3.4
  model, forecast, evaluation, calibrator, or training record.
- Analysis and training must share the same effective Champion, mature-event
  selection, residual-pool identity, and random-plan construction.
- `ROLLING_520W` means the exact last 520 mature events;
  `EXPANDING_AVAILABLE_HISTORY` means all compatible mature history through
  the anchor. Structure selection occurs only in the low-frequency inner/outer
  audit described by the protocol.
- Production scenarios use only frozen mature OOS residuals with
  `Residual = Actual - Prediction` and
  `Scenario = CurrentPrediction + SampledResidual`.
- Model reliability is a diagnostic score, never a forecast-correctness
  probability.
- Do not begin V3.4.2 position-policy changes until the V3.4.1 replay, full
  backend tests, production migration, frozen-table hash audit, and independent
  final audit all pass.

## Active instruments and roles

- `399006` (创业板指数) is the A-share V3.3 training, analysis, and active
  position target.
- `159941` (广发纳斯达克100ETF) is the QDII V3.3 training, analysis, and
  active position target. Use its own real CNY OHLCV, adjusted price, NAV, and
  QDII features.
- `NDX` is benchmark-only for `159941` and a read-only V3.2 audit market. It
  must never become a V3.3 active position, replace `159941` price/volume, or
  inherit/mutate `159941` model state.
- Keep the two active markets' samples, labels, parameters, optimizer state,
  checkpoints, Champions, forecasts, evaluations, advice, and task locks
  independent.

## V3.3 data and feature contract

- The forecast horizon is exactly 20 complete trading weeks.
- Weekly structure uses only completed weeks before the sampled natural week.
  Aggregate real daily OHLCV when a trustworthy upstream weekly bar is absent.
- A feature snapshot requires 100 real visible daily sessions and at least 20
  completed weekly bars.  Weekly DIF/DEA/MACD remains `NULL` before 35 bars;
  features needing 40/52/60 weeks remain `NULL` with missing masks until their
  real windows exist.  These are not a 60-week hard gate.  Machine-rounding
  tolerance may repair only last-bit adjusted-OHLC excursions, never a genuine
  invalid candle.
- Every training cutoff uses exactly the latest 100 visible daily sessions,
  including the complete OHLCV, DIF, DEA, MACD histogram, derivative,
  curvature, divergence, candlestick, and volume/price sequence features.
- Standard names are `DIF`, `DEA`, and `MACD柱`. Do not reintroduce `DIP` or
  `EDA`. A DIF derivative zero and a DIF axis zero are distinct metrics.
- Valuation, earnings, rates, fund-flow, macro, NAV, FX, shares, AUM, and QDII
  observations are point-in-time data. Require both
  `effective_date <= cutoff_date` and `available_at <= cutoff_at`.
- Missing auxiliary data remains missing and carries an explicit missing mask.
  Never substitute zero, 50%, or another fabricated neutral value.
- For `159941`, an NDX close is usable only after its real availability time;
  the same-labelled US close cannot leak into an earlier China close.

## V3.3 training contract

- “训练模型” and “数据分析” are separate actions. Data analysis must never
  mutate iteration counts, model versions, optimizer rows, or Champion hashes.
- The first delivered database contains one formal progressive iteration per
  unique complete ISO natural week in the actual ten-year window. A month with
  five complete natural weeks has five iterations. Do not target a fixed 100,
  480, or 500 iterations.
- Select one real session in every complete week with the fixed deterministic
  seed `330020`. Earlier history is warm-up only and is not shown as a formal
  iteration.
- `399006` needs at least 24 real matured 20-week labels for its initial fit.
  `159941` may start with exactly 12 real matured ETF labels only in the
  strongly regularised `small_sample_degraded` mode.  Before 24 labels it must
  not run or promote a Challenger.  Standard purged validation still needs
  `24 + 20 + 4 = 48` eligible samples.  Never use NDX, another ETF, or invented
  pre-listing data to satisfy these thresholds.
- Iteration N inherits iteration N-1 parameters, optimizer memory, and
  `parent_state_hash`. Do not train historical rounds independently.
- A round may use a 20-week outcome only when the complete outcome was already
  available at that round's cutoff. Use a 20-week purge/embargo for validation.
- The central 20-week path comes from the parameter model. Historical
  analogues may calibrate residual quantiles only, with at most 25% blend.
- Persist the exact forecast as issued. When it matures, attach the real
  evaluation; never recompute it with a later model.
- The latest 20 formal forecasts must remain `pending`. Their loss, price-turn
  error, DIF-turn error, and all other evaluation metrics remain SQL `NULL`.
  The UI must not fill, interpolate, or connect across those missing values.
- Challenger promotion uses measured purged out-of-sample loss, direction
  stability, and interval coverage. More iterations do not imply monotonic
  improvement, and measured errors must never be edited to look better.
- After bootstrap, a new complete week and at least seven elapsed days may add
  at most one iteration per market. A multiweek offline gap still adds only one
  and records the skipped complete weeks. Repeated same-week actions are
  idempotent.

## V3.3 analysis and position contract

- Load the latest market-specific Champion and produce a continuous 20-week
  P10/P50/P90/expected path plus direction probabilities and turning windows.
- Read the current local investment-calendar position before advice. A missing
  ledger is an explicit cleared `0%` position.
- Every current/target position, total change, and execution batch is on a
  five-percentage-point grid. Advice uses at most four conditional batches.
- Fund/ETF ratios are restricted to `7:3`, `6:4`, `5:5`, `4:6`, or `3:7`.
- Data analysis may persist an analysis run and live forecast, but training
  identity before and after analysis must be byte-for-byte unchanged.
- This project does not connect to a broker and does not execute trades.

## Persistence and desktop delivery contract

- V3.3 uses the 15 `v33_*` tables listed in `docs/V33_MODEL_PROTOCOL.md`; keep
  source URLs, raw hashes, availability timestamps, vintages, missing masks,
  state hashes, checkpoints, exact forecasts, evaluations, and analysis runs.
- `InvestmentLab.exe` opens an embedded WebView2 window without requiring a
  browser URL.
- Program resources and writable `data`/`config` folders remain separate.
- Keep dynamic localhost port selection, cross-version single-instance
  protection, backup/restore, and a clear WebView2 prerequisite message.
- The Chinese one-click batch should prefer the packaged EXE and may fall back
  to the source launcher only when no packaged EXE exists.

After a V3.3 change, run the relevant focused backend tests, the full backend
suite before final delivery, the frontend production build, V3.3 data/model
audits, the V3.2 read-only baseline audit, native-Python Playwright visual
review, the TypeScript Playwright suite, and a packaged-EXE smoke test before
reporting completion. Do not claim final delivery while any required check or
the full ten-year progressive bootstrap is still incomplete.
