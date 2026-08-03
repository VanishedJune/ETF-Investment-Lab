import { api, ApiError } from "./api";
import type {
  ActiveInstrumentCode,
  V33AnalysisPayload,
  V33ModelStatusPayload,
  V33ProgressiveResult,
} from "./types/research";

/**
 * Central V3.3 endpoint adapter. Keeping route construction here prevents the
 * active 159941 market from ever falling through to legacy V3.2 NDX routes.
 */
export const v33Routes = Object.freeze({
  status: "/api/v33/model/status",
  train: "/api/v33/model/train",
  analysis: "/api/v33/model/analysis",
  iterations: (instrument: ActiveInstrumentCode) =>
    `/api/v33/iterations/${instrument}`,
  forecast: (instrument: ActiveInstrumentCode) =>
    `/api/v33/forecast/${instrument}`,
});

export function loadV33Status(): Promise<V33ModelStatusPayload> {
  return api<V33ModelStatusPayload>(v33Routes.status);
}

export function trainV33Model(
  instrumentCode: ActiveInstrumentCode,
): Promise<V33ProgressiveResult | V33MarketModelStatusResponse> {
  return api<V33ProgressiveResult | V33MarketModelStatusResponse>(v33Routes.train, {
    method: "POST",
    body: JSON.stringify({ instrument_code: instrumentCode }),
  });
}

export function analyzeWithV33(
  instrumentCode: ActiveInstrumentCode,
): Promise<V33AnalysisPayload> {
  return api<V33AnalysisPayload>(v33Routes.analysis, {
    method: "POST",
    body: JSON.stringify({ instrument_code: instrumentCode }),
  });
}

export function loadV33Iterations(
  instrumentCode: ActiveInstrumentCode,
): Promise<V33ProgressiveResult> {
  return api<V33ProgressiveResult>(v33Routes.iterations(instrumentCode));
}

export async function loadV33Forecast(
  instrumentCode: ActiveInstrumentCode,
): Promise<V33AnalysisPayload | null> {
  try {
    return await api<V33AnalysisPayload>(v33Routes.forecast(instrumentCode));
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}

export type V33MarketModelStatusResponse = {
  implementation_revision: "V3.3-20W" | string;
  market: ActiveInstrumentCode;
  status: string;
  iteration_count?: number;
  champion_version?: string | null;
  message?: string | null;
};
