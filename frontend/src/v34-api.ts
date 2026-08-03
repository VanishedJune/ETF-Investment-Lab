import { api, ApiError } from "./api";
import type {
  ActiveInstrumentCode,
  V34CurvePayload,
  V34ForecastPayload,
  V34MarketStatus,
  V34TrainingRun,
} from "./types/research";

export const v34Routes = Object.freeze({
  status: "/api/v34/model/status",
  train: "/api/v34/model/train",
  analysis: "/api/v34/model/analysis",
  iterations: (market: ActiveInstrumentCode) => `/api/v34/iterations/${market}`,
  forecast: (market: ActiveInstrumentCode) => `/api/v34/forecast/${market}`,
  run: (runId: string) => `/api/v34/training/runs/${runId}`,
});

export function loadV34Status(): Promise<V34MarketStatus[]> {
  return api<V34MarketStatus[]>(v34Routes.status);
}

export function trainV34Model(market: ActiveInstrumentCode): Promise<{
  run_id: string;
  status: string;
  market: ActiveInstrumentCode;
}> {
  return api(v34Routes.train, {
    method: "POST",
    body: JSON.stringify({ instrument_code: market }),
  });
}

export function loadV34TrainingRun(runId: string): Promise<V34TrainingRun> {
  return api<V34TrainingRun>(v34Routes.run(runId));
}

export function analyzeWithV34(market: ActiveInstrumentCode): Promise<V34ForecastPayload> {
  return api<V34ForecastPayload>(v34Routes.analysis, {
    method: "POST",
    body: JSON.stringify({ instrument_code: market }),
  });
}

export function loadV34Iterations(market: ActiveInstrumentCode): Promise<V34CurvePayload> {
  return api<V34CurvePayload>(v34Routes.iterations(market));
}

export async function loadV34Forecast(
  market: ActiveInstrumentCode,
): Promise<V34ForecastPayload | null> {
  try {
    return await api<V34ForecastPayload>(v34Routes.forecast(market));
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}
