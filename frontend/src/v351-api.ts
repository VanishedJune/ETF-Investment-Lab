import { api } from "./api";

const V351_PROTOCOL_VERSION =
  "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS";

export interface V351SlotModelState {
  model_namespace: string | null;
  model_status: string | null;
  small_sample_champion: boolean;
  prewarming: boolean;
  mature_8w_count: number;
  bootstrap_state: string;
}

export interface V351Slot {
  slot_id: string;
  slot_order: number;
  instrument_code: string;
  exchange: string;
  instrument_name: string;
  instrument_type: string;
  model_enabled: boolean;
  active: boolean;
  binding_version: string;
  bound_at: string;
  replaced_from_code: string | null;
  replacement_status: string;
  data_start_date: string | null;
  data_end_date: string | null;
  history_week_count: number;
  mature_8w_count: number;
  model_status: string;
  champion_package_id: string | null;
  last_market_update: string | null;
  model_state: V351SlotModelState | null;
}

export interface V351ReplacementPreview {
  slot_id: string;
  target_code: string;
  metadata: {
    instrument_code: string;
    official_name: string;
    exchange: string;
    instrument_type: string;
    data_source: string;
  };
  replaced_from_code: string;
  replaced_from_name: string;
  preview: {
    earliest_data_date: string | null;
    daily_count: number | null;
    weekly_count: number | null;
    estimated_mature_8w: number | null;
    can_initial_champion: boolean | null;
    requires_prewarming: boolean;
  };
  confirm_required: boolean;
}

export interface V351ReplacementStatus {
  job_id: string;
  slot_id: string;
  target_code: string;
  state: string;
  current_step: string;
  error_code: string | null;
  error_message: string | null;
  finished_at: string | null;
}

export interface IndicatorCalculationResult {
  status: "success" | "validation_error" | "no_data";
  instrument_code: string;
  timeframe: string;
  price_rows: number;
  records_added: number;
  records_updated: number;
  records_skipped: number;
  error: string | null;
}

export interface V351EquivalenceSummary {
  total_candidates: number;
  effectively_identical: number;
  valid_candidates: number;
  by_market: Record<string, { total: number; identical: number; valid: number }>;
}

export interface V351MarketStatus {
  market: string;
  protocol_version: string;
  state: string;
  first_formal_anchor: string | null;
  last_completed_anchor: string | null;
  weekly_iteration_count: number;
  prediction_challenge_count: number;
  strategy_challenge_count: number;
  promotion_count: number;
  forecast_count: number;
  evaluation_count: number;
  champion_package_id: string | null;
}

export interface V351Batch {
  batch_number: number;
  action: "BUY" | "SELL";
  position_pp: number;
  batch_change_pp: number;
  target_position_pp: number;
  cooldown_trading_days: number;
  condition: string;
  earliest_execution_week: number;
  execution_window_start: string;
  execution_window_end: string;
}

export interface V351TurningAssessment {
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
}

export interface V351StrategyPayload {
  market: string;
  forecast_anchor_date: string;
  dif_trend_state: string;
  confirmation_status: string;
  strategy_score: number;
  base_target_position_pp: number;
  state_position_cap_pp: number;
  final_target_position_pp: number;
  batches: V351Batch[];
  empty_reason: string | null;
  reasons: string[];
  turning_assessment: V351TurningAssessment | null;
  strategy_hash: string;
}

export interface V351ForecastCandle {
  week: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface V351ForecastPayload {
  protocol_version: string;
  display_version: string;
  market: string;
  forecast_anchor_date: string;
  model_version: string;
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
  historical_indicators: Array<Record<string, unknown>>;
  representative_ohlcv: V351ForecastCandle[];
  indicators: Array<{
    week: number;
    dif: number;
    dea: number;
    macd: number;
    dif_first_change: number;
  }>;
  expected_path: number[];
  price_quantiles: { p10: number[]; p50: number[]; p90: number[] };
  horizon_probabilities: Record<string, number[]>;
  model_reliability: { score: number; semantics: string };
  health_status: string;
  strategy: V351StrategyPayload | null;
  policy: {
    turning_assessment: V351TurningAssessment | null;
    decision: {
      current_position: number | null;
      target_position: number | null;
      batches: V351Batch[];
      empty_reason: string | null;
    };
  } | null;
  chart_semantics: Record<string, unknown>;
  forecast_hash: string;
}

export interface V351SimulationPayload {
  market: string;
  evaluation_count: number;
  accounts: Array<{
    scope: string;
    window_start: string | null;
    ending_equity: number;
    net_return: number;
    max_drawdown: number;
    average_position_pp: number;
    trade_count: number;
    no_action_window: boolean;
    status: string;
  }>;
  continuous: {
    ending_equity: number;
    cumulative_return: number;
    annualized_return: number;
    max_drawdown: number;
    average_position_pp: number;
    turnover: number;
    buy_hold_return: number;
    fixed_30_return: number;
    cash_return: number;
  } | null;
}

export interface V351ChampionPayload {
  market: string;
  champion: {
    id: string;
    kind: string;
    effective_from_date: string;
    prediction_model_id: string;
    policy_version: string;
  } | null;
  promotion_count: number;
  prediction_challenge_count: number;
  strategy_challenge_count: number;
}

export const v351Routes = Object.freeze({
  slots: "/api/v351/instrument-slots",
  slot: (slotId: string) => `/api/v351/instrument-slots/${slotId}`,
  validate: (slotId: string) =>
    `/api/v351/instrument-slots/${slotId}/validate-replacement`,
  replace: (slotId: string) => `/api/v351/instrument-slots/${slotId}/replace`,
  replacementStatus: (slotId: string) =>
    `/api/v351/instrument-slots/${slotId}/replacement-status`,
  slotHistory: (slotId: string) =>
    `/api/v351/instrument-slots/${slotId}/history`,
  equivalence: "/api/v351/challenges/equivalence-summary",
  rejection: "/api/v351/challenges/rejection-summary",
  status: "/api/v351/status",
  statusMarket: (market: string) => `/api/v351/status/${market}`,
  forecast: (market: string) => `/api/v351/forecast/${market}`,
  strategy: (market: string) => `/api/v351/strategy/${market}`,
  simulation: (market: string) => `/api/v351/simulation/${market}`,
  champions: (market: string) => `/api/v351/champions/${market}`,
});

export function loadV351Slots(): Promise<V351Slot[]> {
  return api<V351Slot[]>(v351Routes.slots);
}

export function validateV351Replacement(
  slotId: string,
  code: string,
  idempotencyKey?: string,
): Promise<V351ReplacementPreview> {
  return api(v351Routes.validate(slotId), {
    method: "POST",
    body: JSON.stringify({
      instrument_code: code,
      idempotency_key: idempotencyKey,
      protocol_version: V351_PROTOCOL_VERSION,
    }),
  });
}

export function replaceV351Slot(
  slotId: string,
  code: string,
  idempotencyKey?: string,
): Promise<{ job_id: string; state: string; target_code: string }> {
  return api(v351Routes.replace(slotId), {
    method: "POST",
    body: JSON.stringify({
      instrument_code: code,
      idempotency_key: idempotencyKey,
      protocol_version: V351_PROTOCOL_VERSION,
    }),
  });
}

export function loadV351ReplacementStatus(
  slotId: string,
): Promise<V351ReplacementStatus> {
  return api<V351ReplacementStatus>(v351Routes.replacementStatus(slotId));
}

export function recalculateV351Indicators(
  code: string,
  timeframe: "daily" | "weekly" | "monthly",
): Promise<IndicatorCalculationResult> {
  return api<IndicatorCalculationResult>(
    `/api/indicators/${encodeURIComponent(code)}/calculate?timeframe=${timeframe}`,
    { method: "POST" },
  );
}

export function loadV351EquivalenceSummary(): Promise<V351EquivalenceSummary> {
  return api<V351EquivalenceSummary>(v351Routes.equivalence);
}

export function loadV351RejectionSummary(): Promise<{
  reasons: Record<string, number>;
  by_market: Record<string, Record<string, number>>;
}> {
  return api(v351Routes.rejection);
}

export function loadV351Status(): Promise<V351MarketStatus[]> {
  return api<V351MarketStatus[]>(v351Routes.status);
}

export function loadV351Forecast(market: string): Promise<V351ForecastPayload | null> {
  return api<V351ForecastPayload>(v351Routes.forecast(market));
}

export function loadV351Strategy(market: string): Promise<V351StrategyPayload | null> {
  return api<V351StrategyPayload>(v351Routes.strategy(market));
}

export function loadV351Simulation(market: string): Promise<V351SimulationPayload> {
  return api<V351SimulationPayload>(v351Routes.simulation(market));
}

export function loadV351Champion(market: string): Promise<V351ChampionPayload> {
  return api<V351ChampionPayload>(v351Routes.champions(market));
}
