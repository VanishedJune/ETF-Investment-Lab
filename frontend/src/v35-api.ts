import { api, ApiError } from "./api";
import type { ActiveInstrumentCode } from "./types/research";

export interface V35MarketStatus {
  market: ActiveInstrumentCode;
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

export interface V35ForecastPayload {
  forecast_anchor_date: string;
  expected_path: number[];
  price_quantiles: number[][];
  horizon_probabilities: Record<string, number[]>;
  health_status: string;
  maturity_status: string;
  forecast_hash: string;
}

export interface V35Batch {
  batch_number: number;
  action: "BUY" | "SELL" | "HOLD";
  position_pp: number;
  batch_change_pp: number;
  target_position_pp: number;
  condition?: Record<string, unknown> | string;
}

export interface V35StrategyPayload {
  forecast_anchor_date: string;
  dif_trend_state: string;
  confirmation_status: string;
  strategy_score: number;
  base_target_position_pp: number;
  state_position_cap_pp: number;
  final_target_position_pp: number;
  batches: V35Batch[];
  reasons: string[];
  strategy_hash: string;
}

export interface V35AccountSummary {
  scope: string;
  window_start: string | null;
  ending_equity: number;
  net_return: number;
  max_drawdown: number;
  average_position_pp: number;
  trade_count: number;
  no_action_window: boolean;
  status: string;
}

export interface V35ContinuousSummary {
  ending_equity: number;
  cumulative_return: number;
  annualized_return: number;
  max_drawdown: number;
  average_position_pp: number;
  turnover: number;
  buy_hold_return: number;
  fixed_30_return: number;
  cash_return: number;
}

export interface V35SimulationPayload {
  market: ActiveInstrumentCode;
  evaluation_count: number;
  accounts: V35AccountSummary[];
  continuous: V35ContinuousSummary | null;
}

export interface V35ChampionPayload {
  market: ActiveInstrumentCode;
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

export const v35Routes = Object.freeze({
  status: "/api/v35/status",
  statusMarket: (market: ActiveInstrumentCode) => `/api/v35/status/${market}`,
  bootstrap: (market: ActiveInstrumentCode) => `/api/v35/bootstrap/${market}`,
  forecast: (market: ActiveInstrumentCode) => `/api/v35/forecast/${market}`,
  strategy: (market: ActiveInstrumentCode) => `/api/v35/strategy/${market}`,
  simulation: (market: ActiveInstrumentCode) => `/api/v35/simulation/${market}`,
  champions: (market: ActiveInstrumentCode) => `/api/v35/champions/${market}`,
});

export function loadV35Status(): Promise<V35MarketStatus[]> {
  return api<V35MarketStatus[]>(v35Routes.status);
}

export function loadV35Forecast(market: ActiveInstrumentCode): Promise<V35ForecastPayload | null> {
  return api<V35ForecastPayload>(v35Routes.forecast(market));
}

export function loadV35Strategy(market: ActiveInstrumentCode): Promise<V35StrategyPayload | null> {
  return api<V35StrategyPayload>(v35Routes.strategy(market));
}

export function loadV35Simulation(market: ActiveInstrumentCode): Promise<V35SimulationPayload> {
  return api<V35SimulationPayload>(v35Routes.simulation(market));
}

export function loadV35Champion(market: ActiveInstrumentCode): Promise<V35ChampionPayload> {
  return api<V35ChampionPayload>(v35Routes.champions(market));
}

export async function startV35Bootstrap(market: ActiveInstrumentCode): Promise<{
  market: ActiveInstrumentCode;
  run_id: string;
  processed_weeks: number;
  warmup_weeks: number;
}> {
  return api(v35Routes.bootstrap(market), {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function loadV35StrategyOrNull(
  market: ActiveInstrumentCode,
): Promise<V35StrategyPayload | null> {
  try {
    return await loadV35Strategy(market);
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}

export async function loadV35ForecastOrNull(
  market: ActiveInstrumentCode,
): Promise<V35ForecastPayload | null> {
  try {
    return await loadV35Forecast(market);
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}
