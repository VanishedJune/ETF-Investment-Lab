import { api, ApiError } from "./api";
import type {
  ActiveInstrumentCode,
  V34AnalysisRun,
  V34CurvePayload,
  V34ForecastPayload,
  V34MarketStatus,
  V34TrainingRun,
} from "./types/research";

export const v341Routes = Object.freeze({
  status: "/api/v341/model/status",
  train: "/api/v341/model/train",
  analysis: "/api/v343/model/analysis",
  analysisRun: (runId: string) => `/api/v343/model/analysis/runs/${runId}`,
  iterations: (market: ActiveInstrumentCode) => `/api/v343/iterations/${market}`,
  forecast: (market: ActiveInstrumentCode) => `/api/v343/forecast/${market}`,
  run: (runId: string) => `/api/v341/training/runs/${runId}`,
});

export function loadV341Status(): Promise<V34MarketStatus[]> {
  return api<V34MarketStatus[]>(v341Routes.status);
}

export function trainV341Model(market: ActiveInstrumentCode): Promise<{
  run_id: string;
  status: string;
  market: ActiveInstrumentCode;
}> {
  return api(v341Routes.train, {
    method: "POST",
    body: JSON.stringify({ instrument_code: market }),
  });
}

export function loadV341TrainingRun(runId: string): Promise<V34TrainingRun> {
  return api<V34TrainingRun>(v341Routes.run(runId));
}

export function analyzeWithV341(market: ActiveInstrumentCode): Promise<{
  run_id: string;
  status: string;
  market: ActiveInstrumentCode;
}> {
  return api(v341Routes.analysis, {
    method: "POST",
    body: JSON.stringify({ instrument_code: market }),
  });
}

export function loadV341AnalysisRun(runId: string): Promise<V34AnalysisRun> {
  return api<V34AnalysisRun>(v341Routes.analysisRun(runId));
}

export function loadV341Iterations(market: ActiveInstrumentCode): Promise<V34CurvePayload> {
  return api<V34CurvePayload>(v341Routes.iterations(market));
}

export async function loadV341Forecast(
  market: ActiveInstrumentCode,
): Promise<V34ForecastPayload | null> {
  try {
    return await api<V34ForecastPayload>(v341Routes.forecast(market));
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}
