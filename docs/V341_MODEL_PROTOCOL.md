# V3.4.1 Model Core Protocol

V3.4.1 is an append-only model-core upgrade. It does not change the position
policy, add display-only ETFs, redesign the frontend, or rewrite any frozen
V3.2, V3.3, or earlier V3.4 record.

## Markets and forecast horizon

- `399006` and `159941` remain independent active markets.
- Each issued forecast contains a continuous 13-complete-week path.
- Direction probabilities are frozen independently at 4, 8, and 13 weeks.
- Labels use the threshold and weekly volatility stored with the forecast;
  later volatility estimates must not relabel historical forecasts.

## Weekly ordering and time boundary

For each complete natural-week anchor, the runtime performs these operations
in one market-specific transaction and in this order:

1. Load the Champion effective at the anchor.
2. Build point-in-time features using observations available at the anchor.
3. Issue and freeze the forecast and its random plan.
4. Mature older forecasts whose 4-, 8-, or 13-week outcomes are now known.
5. Add only frozen out-of-sample residuals that are fully mature at 13 weeks.
6. Decide whether candidate training or a structure audit is due.
7. Persist the iteration result and model-health snapshot.

A newly promoted Champion becomes effective no earlier than the following
complete week. Analysis and training use the same effective model, mature-event
selection, residual-pool identity, and scenario-plan construction. Running
analysis before training or training before analysis must freeze the same
forecast identity for the same database state and anchor.

## Feature and training windows

The normal forecast boundary is up to 520 formal mature weeks plus exactly 52
preceding weeks of feature warm-up for the production Bootstrap. The warm-up
weeks never count as formal iterations. If less than 520 formal weeks remain
after warm-up, the real available count is used, but it must be at least 400.
The persisted audit fields include:

- `training_window_mode`
- `formal_training_weeks`
- `minimum_training_weeks`
- `feature_warmup_weeks`
- `earliest_training_anchor`
- `latest_matured_anchor`
- `raw_sample_count`
- `effective_sample_count`

The only allowed model-window structures are `ROLLING_520W` and
`EXPANDING_AVAILABLE_HISTORY`. A rolling model uses the exact last 520 mature
events. An expanding model uses every compatible mature event through the
anchor. A low-frequency structure audit may compare the two structures using
complete available mature history; ordinary weekly prediction must retain the
bounded feature boundary.

Structure audits require all of the following: at least 26 complete weeks
since the previous audit, an increase of at least three effective independent
samples, and no unfinished structure candidate. Structure selection uses inner
purged walk-forward folds. After the structure is locked, one sealed outer
block compares Champion and challenger; the outer block is not reused to pick
the structure.

## Candidate training and promotion

Ordinary candidates search only the deduplicated set
`champion_alpha * [0.50, 0.75, 1.00, 1.25, 1.50]`, clipped to
`0.1 <= alpha <= 100`, with at most five candidates. Candidate training needs
at least four newly fully mature 13-week events, four elapsed complete weeks,
and a genuine increase in effective sample size.

Champion and candidates share the same purged walk-forward folds,
standardisation, seed, residual sampling indices, scenario count, and frozen
direction thresholds. Promotion uses a time-block bootstrap confidence gate,
worst-fold non-degradation, interval coverage, turning-type quality, and the
norm ratio of coefficients shared in standardised feature space. CORE and
EXTENDED profiles are compared only by the low-frequency structure audit.

The primary normalised validation score is:

- 35% weekly path MAE across weeks 1-13;
- 15% endpoint MAE at weeks 4, 8, and 13 (5% each);
- 20% mean three-class Brier score at weeks 4, 8, and 13;
- 15% WIS or equivalent quantile interval score;
- 10% maximum-drawdown and realised-volatility error;
- 5% recent-fold stability.

Metrics with incompatible units are normalised against fold scale or a
baseline before aggregation. Hard promotion gates are not mixed into the
weighted score.

## Probability calibration

Each horizon stores raw and calibrated `up`, `sideways`, and `down`
probabilities. Calibration uses one versioned temperature applied to smoothed
probability logits. A calibrator becomes effective from the following week.

- fewer than 30 effective samples: shrinkage only, no formal calibration;
- at least 30: preliminary calibration is allowed;
- at least 50: formal calibration is allowed.

The three-class Brier score is the formal probability metric. Tests validate
the formula, version lookup, frozen linkage, and known examples; they do not
require every input vector to change numerically.

## Out-of-sample residual scenarios

The only valid residual sign is:

`Residual = Actual - Prediction`

The only valid production scenario construction is:

`Scenario = CurrentPrediction + SampledResidual`

Production residuals must come from mature forecasts that were frozen before
the realised path was known. Compatibility requires the same market, model
family, label definition, return unit, and scenario schema. Residuals are
standardised by volatility frozen with the source forecast and restored using
current forecast volatility. Small residual pools use explicit prior shrinkage
and reduce reliability; in-sample residuals may not silently enter production.

Every forecast stores a reproducible random plan, residual-pool identity,
scenario count, residual indices, plan hash, and forecast hash. Candidate
comparisons use common random plans.

## Reliability, OOD, and model health

The UI/API value is a `0-100` model reliability score, not the probability that
the forecast is correct. Components are sample sufficiency, calibration
quality, interval coverage, recent out-of-sample performance, OOD degree, and
indicator consistency.

Hard caps are:

- fewer than 30 effective samples: 40;
- no formal calibration: 45;
- material OOD: 50;
- `MODEL_DEGRADED`: 50.

Health states are `MODEL_NORMAL`, `MODEL_DEGRADED`, and
`MODEL_OUT_OF_DISTRIBUTION`. OOD uses robust core-feature z-scores, the share
outside training quantile ranges, and a multivariate distance fitted inside
training folds. Degradation uses only mature out-of-sample results. Entry and
exit use different thresholds and consecutive-confirmation counts to avoid
weekly state flapping.

## Schema 21 persistence

Schema 21 appends exactly these 17 tables:

- `v341_training_profiles`
- `v341_feature_snapshots`
- `v341_training_runs`
- `v341_model_versions`
- `v341_optimizer_states`
- `v341_scenario_adapters`
- `v341_probability_calibrators`
- `v341_random_plans`
- `v341_outer_evaluation_blocks`
- `v341_forecasts`
- `v341_forecast_calibrators`
- `v341_forecast_evaluations`
- `v341_residual_records`
- `v341_candidate_trials`
- `v341_model_health_snapshots`
- `v341_training_iterations`
- `v341_analysis_runs`

Migration is additive, transactional, idempotent, and validated before commit.
Production migration must be performed only by the lead implementer after a
SQLite online backup and a successful migration/rollback exercise on a test
copy. `initialize_database()` must not be used as the production migration
entry point; call `run_migrations()` directly.

## Release gates

V3.4.1 cannot be declared complete until all of these are evidenced:

- no future-data leakage and no analysis/training order fork;
- forecast-time thresholds and calibrator versions are loaded correctly;
- out-of-sample residual sign, provenance, and compatibility are correct;
- candidate count is bounded and comparisons use common randomness;
- Champion promotion passes sealed-outer and bootstrap hard gates;
- reliability is never presented as a correctness probability;
- both markets pass bounded replay with 4/8/13-week maturity;
- Schema 20 to 21 migration is idempotent and rollback-safe;
- SQLite integrity and foreign keys pass;
- all 75 frozen legacy tables have identical row-level SHA-256 hashes;
- focused and complete backend test suites finish with a real final summary;
- the independent final auditor reports no unresolved P0 or P1 issue.

Do not start V3.4.2 position-policy work until these gates pass.
