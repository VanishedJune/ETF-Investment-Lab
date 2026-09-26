/** Data-only chart/calendar instruments shipped with the desktop build. */
export type ActiveInstrumentCode =
  | "399006"
  | "159941"
  | "159915"
  | "518600"
  | "512800"
  | "512690"
  | "512010"
  | "159622"
  | "516150"
  | "517520"
  | "515220"
  | "159611"
  | (string & {});
export type ModelInstrumentCode = ActiveInstrumentCode;
export type ResearchInstrumentCode =
  | ModelInstrumentCode
  | "159915"
  | "518600"
  | "512800"
  | "512690"
  | "512010"
  | "159622"
  | "516150"
  | (string & {});
/** V3.2 is immutable audit history; NDX remains its original benchmark key. */
export type LegacyV32InstrumentCode = "399006" | "NDX";
export type InstrumentCode = ResearchInstrumentCode | LegacyV32InstrumentCode;
export type MarketTimeframe = "daily" | "weekly" | "monthly";
export type RatioTier = "7:3" | "6:4" | "5:5" | "4:6" | "3:7";
export type TaskStatus =
  | "queued"
  | "preparing_data"
  | "building_features"
  | "iterating"
  | "validating"
  | "generating_advice"
  | "completed"
  | "failed"
  | "recoverable";

export interface PriceRow {
  date: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string | null;
  source: string | null;
  is_complete?: boolean;
  period_status?: "COMPLETE" | "INCOMPLETE_CURRENT_PERIOD";
}

export interface PricePayload {
  rows: PriceRow[];
  current_period_status?: "COMPLETE" | "INCOMPLETE_CURRENT_PERIOD";
}

export interface IndicatorRow {
  date: string;
  dif?: string | null;
  dea?: string | null;
  macd_histogram?: string | null;
  dif_first_change?: string | null;
  rsi_6?: string | null;
}

export interface PositionEvent {
  id: number;
  instrument_code: ActiveInstrumentCode;
  direction: "increase" | "decrease";
  operation_date: string;
  change_percent: number;
  position_after: number;
  sequence?: number;
  note: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface AnalysisTask {
  id: string;
  instrument_code: LegacyV32InstrumentCode;
  status: TaskStatus;
  completed_weeks: number;
  total_weeks: number | null;
  last_iteration: number;
  last_work_version: string;
  last_model_version: string;
  last_completed_week: string | null;
  progress: Record<string, unknown>;
  message: string | null;
  reused: boolean;
}

export interface WindowMetric {
  instrument_code: LegacyV32InstrumentCode;
  window: string;
  sample_count: number;
  total_loss: string;
  direction_hit_rate: string;
  calibration_loss: string;
  overtrade_penalty: string;
}

export interface ModelMetrics {
  symbol: LegacyV32InstrumentCode;
  model_version: string;
  iteration_count: number;
  last_iteration: number;
  accepted_iterations: number;
  rejected_iterations: number;
  feedback_count: number;
  mature_count: number;
  partial_count: number;
  pending_count: number;
  candidate_acceptance_rate: string;
  windows: Record<string, WindowMetric>;
  curve: IterationMetricPoint[];
}

export interface IterationMetricPoint {
  iteration_number: number;
  iteration_id: string;
  cutoff_date: string;
  model_version: string;
  accepted: boolean;
  deviation_days: number | null;
  absolute_deviation: string | null;
  rolling_20_abs_deviation: string | null;
  rolling_52_abs_deviation: string | null;
  direction_correct: boolean | null;
}

export interface AdviceBatch {
  confirmation_condition: string;
  expected_date: string;
  operation_side: "buy" | "sell";
  percent: number;
  sequence: number;
  status: "ready" | "waiting_confirmation";
  tolerance_trading_days: 3;
}

export interface LatestAdvice {
  id: number;
  instrument_code: LegacyV32InstrumentCode;
  iteration_id: string;
  work_version: string;
  model_version: string;
  advice_generation: number;
  advice_at: string;
  advice_version: string;
  audit_fields: Record<string, unknown>;
  audit_notes: string[];
  batches: AdviceBatch[];
  confidence: string;
  current_position: number | null;
  data_cutoff_date: string;
  direction: "up" | "down" | "neutral";
  direction_probabilities: Record<string, string>;
  feature_set_version: string;
  forecast_horizon_weeks: 13;
  fund_etf_ratio: RatioTier;
  market_state: string;
  operation_side: "buy" | "sell" | "hold";
  probability: string;
  recommendation:
    | "staged_increase"
    | "staged_decrease"
    | "hold"
    | "wait_confirmation"
    | "analysis_only";
  signed_position_change: number | null;
  source_data_max_date: string | null;
  target_position: number;
  target_position_range: [number, number];
  total_adjustment: number | null;
}

export interface V31PathPoint {
  horizon_week: number;
  p10_cumulative_return: number;
  p50_cumulative_return: number;
  p90_cumulative_return: number;
  expected_cumulative_return: number;
}

export interface V31Stage {
  stage: string;
  status: "completed" | "failed" | "running";
  duration_ms: number;
  error?: string | null;
}

export interface V31Run {
  id: string;
  market: LegacyV32InstrumentCode;
  status: "running" | "completed" | "failed";
  current_stage: string;
  error_code?: string | null;
  error_message?: string | null;
  result?: Record<string, unknown>;
  stages: V31Stage[];
}

export interface V31Analysis {
  run_id: string;
  market: LegacyV32InstrumentCode;
  generated_at: string;
  model: Record<string, any>;
  freshness: Record<string, any>;
  data_gate: Record<string, any>;
  weekly: Record<string, any>;
  daily: Record<string, any>;
  path: Record<string, any> & { points: V31PathPoint[] };
  position: Record<string, any>;
  advice: Record<string, any>;
  evaluation: Record<string, any>;
  stages?: V31Stage[];
}

export interface V32TrainingRunStatus {
  id: string;
  market: LegacyV32InstrumentCode;
  run_type: "bootstrap" | "bootstrap_resume" | "incremental";
  status: "running" | "completed" | "failed" | "interrupted";
  current_stage: string;
  created_iteration_count: number;
  started_at: string;
  completed_at?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  result?: Record<string, any>;
}

export interface V32MarketTrainingStatus {
  is_training: boolean;
  bootstrapped: boolean;
  iteration_count: number;
  last_training_week_key: string | null;
  last_successful_training_at: string | null;
  next_training_eligible_at: string | null;
  champion_weekly_version: string | null;
  champion_daily_version: string | null;
  latest_run: V32TrainingRunStatus | null;
}

export type V32TrainingStatus = Record<LegacyV32InstrumentCode, V32MarketTrainingStatus>;

export interface V32CurvePoint {
  iteration: number;
  parent_iteration: number | null;
  week_key: string;
  cutoff_date: string;
  maturity_status: "full" | "pending";
  composite_loss: number | null;
  path_error: number | null;
  terminal_return_error: number | null;
  direction_score: number | null;
  interval_coverage: number | null;
  high_week_error: number | null;
  low_week_error: number | null;
  high_low_deviation_days: number | null;
  rolling_20_loss: number | null;
  rolling_52_loss: number | null;
  rolling_20_deviation_days: number | null;
  rolling_52_deviation_days: number | null;
  champion_version: string;
  challenger_version: string | null;
  promoted: boolean;
  rejection_reason: string | null;
}

export interface V32CurvePayload {
  market: LegacyV32InstrumentCode;
  points: V32CurvePoint[];
}

export interface V32AnalysisRun {
  id: string;
  market: LegacyV32InstrumentCode;
  status: "running" | "completed" | "failed";
  current_stage: string;
  started_at?: string;
  completed_at?: string | null;
  model_version?: string | null;
  data_gate_status?: string;
  stages: V31Stage[];
  result?: Record<string, any>;
  error_code?: string | null;
  error_message?: string | null;
}

export interface V32Analysis {
  implementation_revision: "V3.2.0";
  run_id: string;
  market: LegacyV32InstrumentCode;
  training_mutated: false;
  model: Record<string, any>;
  freshness: Record<string, any>;
  weekly: Record<string, any>;
  daily: Record<string, any>;
  path: Record<string, any> & { points: V31PathPoint[] };
  position: Record<string, any>;
  advice: Record<string, any>;
  data_snapshot_id: string;
  forecast_id: number;
}

export interface V33MarketModelStatus {
  instrument_code: ActiveInstrumentCode;
  is_training: boolean;
  bootstrapped: boolean;
  iteration_count: number;
  champion_version: string | null;
  last_training_week_key: string | null;
  last_successful_training_at: string | null;
  pending_count: 20 | number;
  message?: string | null;
}

export interface V33ModelStatusPayload {
  implementation_revision: "V3.3-20W" | string;
  markets: Partial<Record<ActiveInstrumentCode, V33MarketModelStatus>>;
}

export interface V33AnalysisPayload {
  implementation_revision: "V3.3-20W";
  run_id: string;
  market: ActiveInstrumentCode;
  benchmark: string;
  training_mutated: false;
  model: {
    version: string;
    iteration_number: number;
    state_hash: string;
    trained_through: string;
    training_sample_count: number;
    horizon_weeks: 20;
    daily_window_sessions: 100;
  };
  path: {
    weeks: number[];
    p10: number[];
    p50: number[];
    p90: number[];
    expected: number[];
    weekly_base?: number[];
    daily_correction?: number[];
    direction: "up" | "down" | "sideways";
    weekly_confidence?: number;
    daily_confidence_adjustment?: number;
    confidence: number;
    up_probability: number;
    sideways_probability: number;
    down_probability: number;
    expected_max_drawdown: number;
    predicted_high_week: number | null;
    predicted_low_week: number | null;
    analogue_role: "residual_interval_calibration_only" | string;
    analogue_calibration_count: number;
  };
  turning_points: {
    stable: boolean;
    kind: string;
    dif_derivative_zero: V33DateWindow | null;
    price_turn: V33DateWindow | null;
    dif_axis_zero_is_distinct: true;
    date_basis: string;
  };
  position: {
    current: number;
    weekly_base_target?: number;
    daily_adjustment?: number;
    target: number;
    change: number;
    source: string;
  };
  advice: {
    action: "buy" | "sell" | "hold";
    summary: string;
    batches: V33AdviceBatch[];
    fund_etf_ratio: RatioTier;
    policy: string[];
    conditional: true;
    position_grid: 5;
  };
  features: {
    weekly: Record<string, unknown>;
    daily: Record<string, unknown>;
    daily_sequence_count: 100;
  };
  data_quality: {
    source_data_max_date: string | null;
    cutoff_date: string | null;
    price_data_as_of?: string | null;
    analysis_as_of?: string | null;
    missing_masks: Record<string, unknown> | string[];
    missing_series: string[];
    degraded: boolean;
    provenance: Record<string, unknown> | unknown[];
  };
  analysis_hash: string;
}

export interface V33DateWindow {
  start: string | null;
  center: string | null;
  end: string | null;
}

export interface V33AdviceBatch {
  batch: number;
  action: "buy" | "sell";
  percentage_points: number;
  window_start: string;
  expected_date: string;
  window_end: string;
  condition: string;
}

export interface V33CurvePoint {
  iteration: number;
  cutoff_date: string;
  status: "full" | "pending";
  loss: number | null;
  price_turn_error_days: number | null;
  dif_turn_error_days: number | null;
}

export interface V33IterationRecord {
  market: ActiveInstrumentCode;
  iteration_number: number;
  cutoff_date: string;
  parent_state_hash: string | null;
  state_hash: string;
  champion_version: string;
  challenger_promoted: boolean;
  promotion_reason: string | null;
  forecast: Record<string, unknown>;
  status: "full" | "pending";
  evaluation: Record<string, unknown> | null;
}

export interface V33ProgressiveResult {
  market: ActiveInstrumentCode;
  state: string;
  iterations: V33IterationRecord[];
  pending_count: number;
  full_count: number;
  curve: V33CurvePoint[];
  expected_iteration_count: number;
  audited_iteration_count: number;
  weekly_sampling_seed: string | number;
  sampled_dates: string[];
}

export interface V34ChampionStatus {
  market: ActiveInstrumentCode;
  version: string;
  parent_version: string | null;
  horizon_weeks: 13;
  trained_through_date: string;
  effective_from_date: string;
  raw_matured_sample_count: number;
  effective_independent_sample_count: number;
  parameter_hash: string;
}

export interface V34MarketStatus {
  market: ActiveInstrumentCode;
  version: "V3.4_13W" | string;
  horizon_weeks: 13;
  bootstrapped: boolean;
  champion: V34ChampionStatus | null;
  last_anchor_date: string | null;
  weekly_iteration_count: number;
  candidate_training_count: number;
  champion_promotion_count: number;
}

export interface V34CurvePoint {
  iteration: number;
  anchor_date: string;
  forecast_anchor_date?: string;
  evaluation_available_date?: string | null;
  maturity_status?: string;
  pending?: boolean;
  normalized_endpoint_error?: number | null;
  price_turn_deviation_trading_days?: number | null;
  dif_turn_deviation_trading_days?: number | null;
  mean_turn_deviation_trading_days?: number | null;
  turn_deviation_component_count?: number;
  raw_matured_sample_count?: number;
  effective_independent_sample_count?: number;
  training_triggered: boolean;
  promoted: boolean;
  champion_before?: string;
  candidate: Record<string, unknown> | string | null;
  champion_after?: string;
  champion_loss?: number[];
  candidate_loss?: number[];
  forecast_model_id?: string;
  champion_after_model_id?: string;
}

export interface V34CurvePayload {
  market: ActiveInstrumentCode;
  horizon_weeks: 13;
  points: V34CurvePoint[];
}

export interface V34ForecastCandle {
  week: number;
  week_start: string;
  week_end: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume_p10: number;
  volume_p50: number;
  volume_p90: number;
  direction_probabilities: Record<"up" | "sideways" | "down", number>;
}

export interface V34ForecastIndicator {
  week?: number;
  week_end: string;
  dif: number | null;
  dea: number | null;
  macd: number | null;
  dif_first_change: number | null;
  dif_second_change: number | null;
  trend_state?: string;
}

export interface V34PriceQuantile {
  week: number;
  week_end: string;
  close_p10: number;
  close_p50: number;
  close_p90: number;
  low_quantile: number;
  high_quantile: number;
}

export interface V34AdviceBatch {
  batch: number;
  action: "increase" | "decrease";
  position_change_percentage_points: number;
  window_start: string;
  window_end: string;
  trigger: string;
  invalidation: string;
}

export interface V34ForecastPayload {
  protocol_version: "V3.4.1_MODEL_CORE" | string;
  display_version: "V3.4.3_MARKET_UI" | string;
  market: ActiveInstrumentCode;
  forecast_anchor_date: string;
  model_version: string;
  scenario_adapter_id: string;
  scenario_seed: number;
  scenario_count: number;
  historical_ohlcv: Array<{
    week_end: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number | null;
  }>;
  historical_indicators: V34ForecastIndicator[];
  representative_ohlcv: V34ForecastCandle[];
  indicators: V34ForecastIndicator[];
  price_quantiles: V34PriceQuantile[];
  horizon_probabilities: Record<string, Record<"up" | "sideways" | "down", number>>;
  path_probabilities: Record<string, number>;
  model_reliability: {
    score: number;
    semantics: string;
    [key: string]: unknown;
  };
  health_status: string;
  policy: {
    turning_assessment: {
      price_turn_status: string;
      dif_turn_status: string;
      consistency_status: string;
      candidates: Array<{
        signal_kind: "PRICE" | "DIF";
        turn_kind: "top" | "bottom";
        classification: string;
        confirmation_status: string;
        window_start_date: string;
        window_end_date: string;
      }>;
    };
    position_source: { position_percent: number; source_kind: string };
    decision: {
      current_position: number;
      target_position: number;
      next_executable_position: number;
      total_change: number;
      action: "BUY" | "SELL" | "HOLD";
      fund_etf_ratio: RatioTier;
      batches: Array<{
        batch_number: number;
        action: "BUY" | "SELL";
        change_pp: number;
        target_after_pp: number;
        window_start_date: string;
        window_end_date: string;
        initial_state: string;
        trigger: Record<string, unknown>;
        invalidation: Record<string, unknown>;
      }>;
    };
  } | null;
  chart_semantics: {
    dif_first_change: string;
    frozen_forecast_unchanged: boolean;
    [key: string]: unknown;
  };
  scenario_audit: Record<string, unknown>;
}

export interface V34TrainingRun {
  run_id: string;
  market: ActiveInstrumentCode;
  status: "queued" | "running" | "completed" | "failed";
  current_stage: string;
  weekly_iteration_count: number;
  candidate_training_count: number;
  champion_promotion_count: number;
  error_message?: string | null;
}

export interface V34AnalysisRun {
  run_id: string;
  protocol_version: string;
  market: ActiveInstrumentCode;
  status: "queued" | "running" | "completed" | "failed";
  forecast_anchor_date: string | null;
  started_at: string;
  completed_at: string | null;
  result: V34ForecastPayload | Record<string, never>;
  error_code: string | null;
  error_message: string | null;
}

export const legendColors = Object.freeze({
  priceUp: "#d64b4b",
  priceDown: "#27845a",
  volume: "#64748b",
  volumeIncrease: "#d64b4b",
  volumeDecrease: "#27845a",
  dif: "#52789c",
  dea: "#a57738",
  macdPositive: "#d64b4b",
  macdNegative: "#27845a",
  difFirstChange: "#807096",
  ma5: "#ae794b",
  ma10: "#52789c",
  ma20: "#97718f",
});

export type VolumeBarDatum = {
  value: number;
  itemStyle: { color: string; opacity: number };
} | null;

/**
 * Color each real volume bar by its change from the preceding real period.
 * Missing sparse-axis slots do not reset the comparison.  The first real bar
 * and unchanged volume remain neutral because neither has a direction.
 */
export function volumeChangeBarData(
  values: readonly (number | null | undefined)[],
  opacity = 0.76,
): VolumeBarDatum[] {
  let previous: number | null = null;
  return values.map((raw) => {
    if (raw === null || raw === undefined || !Number.isFinite(raw)) return null;
    const value = Number(raw);
    const color = previous === null || value === previous
      ? legendColors.volume
      : value > previous
        ? legendColors.volumeIncrease
        : legendColors.volumeDecrease;
    previous = value;
    return { value, itemStyle: { color, opacity } };
  });
}

export interface NormalizedSeries {
  dates: string[];
  candles: [number, number, number, number][];
  closes: number[];
  volumes: (number | null)[];
  dif: number[];
  dea: number[];
  macd: number[];
  difFirstChange: number[];
  ma5: number[];
  ma10: number[];
  ma20: number[];
  volumeAvailable: boolean;
  source: string | null;
}

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function movingAverage(values: readonly number[], window: number): number[] {
  let running = 0;
  return values.map((value, index) => {
    running += value;
    if (index >= window) running -= values[index - window];
    const divisor = Math.min(index + 1, window);
    return Number((running / divisor).toFixed(6));
  });
}

/**
 * Align price and momentum rows on the closed set of dates that has complete
 * OHLC and MACD values. Trimming only unsupported leading/trailing rows keeps
 * every published curve continuous without inventing values inside the range.
 */
export function normalizeSeries(
  prices: readonly PriceRow[],
  indicators: readonly IndicatorRow[],
): NormalizedSeries {
  const indicatorsByDate = new Map(indicators.map((row) => [row.date, row]));
  const aligned = prices.flatMap((price) => {
    const indicator = indicatorsByDate.get(price.date);
    const open = finiteNumber(price.open);
    const high = finiteNumber(price.high);
    const low = finiteNumber(price.low);
    const close = finiteNumber(price.close);
    const dif = finiteNumber(indicator?.dif);
    const dea = finiteNumber(indicator?.dea);
    const macd = finiteNumber(indicator?.macd_histogram);
    if (open === null || high === null || low === null || close === null) {
      return [];
    }
    return [{
      price,
      candle: [open, close, low, high] as [
        number,
        number,
        number,
        number,
      ],
      close,
      volume: finiteNumber(price.volume),
      dif: dif ?? Number.NaN,
      dea: dea ?? Number.NaN,
      macd: macd ?? Number.NaN,
      difFirstChange: finiteNumber(indicator?.dif_first_change),
    }];
  });
  const closes = aligned.map((row) => row.close);
  const volumes = aligned.map((row) => row.volume);
  return {
    dates: aligned.map((row) => row.price.date),
    candles: aligned.map((row) => row.candle),
    closes,
    volumes,
    dif: aligned.map((row) => row.dif),
    dea: aligned.map((row) => row.dea),
    macd: aligned.map((row) => row.macd),
    difFirstChange: aligned.map((row) => row.difFirstChange ?? Number.NaN),
    ma5: movingAverage(closes, 5),
    ma10: movingAverage(closes, 10),
    ma20: movingAverage(closes, 20),
    volumeAvailable: volumes.length > 0 && volumes.every((value) => value !== null),
    source: aligned.at(-1)?.price.source ?? null,
  };
}
