import { api } from "./api";

export interface V37Status {
  market: string;
  protocol_version: string;
  state: string;
  first_formal_anchor?: string | null;
  last_completed_anchor?: string | null;
  weekly_iteration_count: number;
  forecast_count: number;
  evaluation_count: number;
  promotion_count: number;
  champion_package_id?: string | null;
}

export interface V37Champion {
  market: string;
  champion: {
    id: string;
    kind: string;
    effective_from_date: string;
    policy_version: string;
  } | null;
  promotion_count: number;
  prediction_challenge_count: number;
  strategy_challenge_count: number;
}

export interface V37AiForecastPayload {
  market: string;
  protocol_version: string;
  ai_forecast: {
    anchor: string;
    model: string;
    trend_1w: string;
    trend_2w: string;
    trend_4w: string;
    trend_8w: string;
    direction_scores: Record<string, number>;
    daily_trend: string;
    weekly_trend: string;
    multi_timeframe_state: string;
    risk_level: string;
    confidence_raw: number;
    expected_return_4w: number;
    expected_return_8w: number;
    reason_codes: string[];
    status: string;
  } | null;
  health?: {
    status: string;
    calibration_status: string;
    forward_oos_window_count: number;
    weight_cap_pp: number;
    metrics: Record<string, unknown>;
  } | null;
}

export interface V37FusionPayload {
  market: string;
  configs: Array<{
    config_version: string;
    local_weight: number;
    ai_weight: number;
    conflict_policy: string;
  }>;
  evaluation_count: number;
}

export interface V37FunnelPayload {
  market: string;
  local: Array<Record<string, unknown>>;
  ai: Array<Record<string, unknown>>;
  fusion: Array<Record<string, unknown>>;
  fusion_config: {
    config_version: string;
    local_weight: number;
    ai_weight: number;
  } | null;
}

export function v37Status(market: string): Promise<V37Status> {
  return api<V37Status>(`/api/v37/status/${encodeURIComponent(market)}`);
}

export function v37Forecast(market: string): Promise<Record<string, unknown>> {
  return api(`/api/v37/forecast/${encodeURIComponent(market)}`);
}

export function v37Strategy(market: string): Promise<Record<string, unknown>> {
  return api(`/api/v37/strategy/${encodeURIComponent(market)}`);
}

export function v37Ai(market: string): Promise<V37AiForecastPayload> {
  return api(`/api/v37/ai/${encodeURIComponent(market)}`);
}

export function v37Funnel(market: string): Promise<V37FunnelPayload> {
  return api(`/api/v37/funnel/${encodeURIComponent(market)}`);
}

export function v37Fusion(market: string): Promise<V37FusionPayload> {
  return api(`/api/v37/fusion/${encodeURIComponent(market)}`);
}

export function v37Simulation(market: string): Promise<Record<string, unknown>> {
  return api(`/api/v37/simulation/${encodeURIComponent(market)}`);
}

export function v37Champions(market: string): Promise<V37Champion> {
  return api(`/api/v37/champions/${encodeURIComponent(market)}`);
}

export function v37RepairProposals(): Promise<
  Array<Record<string, unknown>>
> {
  return api(`/api/v37/repair/proposals`);
}

export function v37ModelConflicts(market: string): Promise<
  Array<Record<string, unknown>>
> {
  return api(`/api/v37/model-conflicts/${encodeURIComponent(market)}`);
}
