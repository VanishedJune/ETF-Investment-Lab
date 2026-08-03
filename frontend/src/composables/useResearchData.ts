import { computed, onBeforeUnmount, ref } from "vue";

import { api, ApiError } from "../api";
import type {
  ActiveInstrumentCode,
  AnalysisTask,
  IndicatorRow,
  LegacyV32InstrumentCode,
  LatestAdvice,
  MarketTimeframe,
  ModelMetrics,
  PositionEvent,
  PricePayload,
  PriceRow,
  ResearchInstrumentCode,
  V31Analysis,
  V31Run,
  V32Analysis,
  V32AnalysisRun,
  V32CurvePayload,
  V32TrainingStatus,
  V33AnalysisPayload,
  V33MarketModelStatus,
  V33ProgressiveResult,
  V34CurvePayload,
  V34ForecastPayload,
  V34MarketStatus,
} from "../types/research";
import { normalizeSeries } from "../types/research";
import {
  analyzeWithV33,
  loadV33Forecast,
  loadV33Iterations,
  loadV33Status,
  trainV33Model as requestV33Training,
} from "../v33-api";
import {
  analyzeWithV341 as analyzeWithV34,
  loadV341Forecast as loadV34Forecast,
  loadV341Iterations as loadV34Iterations,
  loadV341AnalysisRun as loadV34AnalysisRun,
  loadV341Status as loadV34Status,
  loadV341TrainingRun as loadV34TrainingRun,
  trainV341Model as requestV34Training,
} from "../v341-api";

type PositionEventWrite = {
  instrument_code: ActiveInstrumentCode;
  direction: "increase" | "decrease";
  operation_date: string;
  change_percent: number;
  note: string | null;
};

const TERMINAL_TASKS = new Set(["completed", "failed"]);
const POLL_INTERVAL_MS = 1_000;
const V34_ANALYSIS_TIMEOUT_MS = 120_000;

export function useResearchData() {
  const prices = ref<PriceRow[]>([]);
  const indicators = ref<IndicatorRow[]>([]);
  const currentPeriodStatus = ref<"COMPLETE" | "INCOMPLETE_CURRENT_PERIOD">("COMPLETE");
  const task = ref<AnalysisTask | null>(null);
  const metrics = ref<ModelMetrics | null>(null);
  const advice = ref<LatestAdvice | null>(null);
  const v31Analysis = ref<V31Analysis | null>(null);
  const v31Run = ref<V31Run | null>(null);
  const v32Analysis = ref<V32Analysis | null>(null);
  const v32Run = ref<V32AnalysisRun | null>(null);
  const v32TrainingStatus = ref<Partial<V32TrainingStatus>>({});
  const v32Curves = ref<Partial<Record<LegacyV32InstrumentCode, V32CurvePayload>>>({});
  const trainingLoading = ref<Record<LegacyV32InstrumentCode, boolean>>({
    "399006": false,
    NDX: false,
  });
  const trainingError = ref<Record<LegacyV32InstrumentCode, string>>({
    "399006": "",
    NDX: "",
  });
  const v33Analysis = ref<V33AnalysisPayload | null>(null);
  const v33Statuses = ref<Partial<Record<ActiveInstrumentCode, V33MarketModelStatus>>>({});
  const v33Curves = ref<Partial<Record<ActiveInstrumentCode, V33ProgressiveResult>>>({});
  const v33TrainingLoading = ref<Record<ActiveInstrumentCode, boolean>>({
    "399006": false,
    "159941": false,
  });
  const v33TrainingError = ref<Record<ActiveInstrumentCode, string>>({
    "399006": "",
    "159941": "",
  });
  const v33AnalysisLoading = ref(false);
  const v33AnalysisError = ref("");
  const v34Analysis = ref<V34ForecastPayload | null>(null);
  const v34Statuses = ref<Partial<Record<ActiveInstrumentCode, V34MarketStatus>>>({});
  const v34Curves = ref<Partial<Record<ActiveInstrumentCode, V34CurvePayload>>>({});
  const v34TrainingLoading = ref<Record<ActiveInstrumentCode, boolean>>({
    "399006": false,
    "159941": false,
  });
  const v34TrainingError = ref<Record<ActiveInstrumentCode, string>>({
    "399006": "",
    "159941": "",
  });
  const v34AnalysisLoading = ref(false);
  const v34AnalysisError = ref("");
  const positionEvents = ref<PositionEvent[]>([]);
  const currentPositions = ref<Record<ActiveInstrumentCode, number | null>>({
    "399006": 0,
    "159941": 0,
  });

  const marketLoading = ref(false);
  const analysisLoading = ref(false);
  const calendarLoading = ref(false);
  const marketError = ref("");
  const analysisError = ref("");
  const calendarError = ref("");

  let marketGeneration = 0;
  let analysisGeneration = 0;
  let calendarGeneration = 0;
  let v33StateGeneration = 0;
  let v33AnalysisGeneration = 0;
  const v33TrainingGeneration: Record<ActiveInstrumentCode, number> = {
    "399006": 0,
    "159941": 0,
  };
  let v34StateGeneration = 0;
  let v34AnalysisGeneration = 0;
  const v34TrainingGeneration: Record<ActiveInstrumentCode, number> = {
    "399006": 0,
    "159941": 0,
  };
  let pollTimer: ReturnType<typeof setTimeout> | undefined;

  const normalized = computed(() => normalizeSeries(prices.value, indicators.value));

  function stopPolling() {
    analysisGeneration += 1;
    if (pollTimer !== undefined) {
      clearTimeout(pollTimer);
      pollTimer = undefined;
    }
    analysisLoading.value = false;
  }

  async function loadMarket(
    symbol: ResearchInstrumentCode,
    timeframe: MarketTimeframe,
  ): Promise<void> {
    const generation = ++marketGeneration;
    marketLoading.value = true;
    marketError.value = "";
    prices.value = [];
    indicators.value = [];
    currentPeriodStatus.value = "COMPLETE";
    try {
      const [pricePayload, indicatorPayload] = await Promise.all([
        api<PricePayload>(`/api/market/${symbol}/prices?timeframe=${timeframe}`),
        api<IndicatorRow[]>(`/api/indicators/${symbol}?timeframe=${timeframe}`),
      ]);
      if (generation !== marketGeneration) return;
      prices.value = pricePayload.rows ?? [];
      indicators.value = indicatorPayload;
      currentPeriodStatus.value = pricePayload.current_period_status ?? "COMPLETE";
    } catch (reason) {
      if (generation !== marketGeneration) return;
      prices.value = [];
      indicators.value = [];
      currentPeriodStatus.value = "COMPLETE";
      marketError.value = messageFrom(reason);
    } finally {
      if (generation === marketGeneration) marketLoading.value = false;
    }
  }

  async function refreshMarket(
    symbol: ResearchInstrumentCode,
    timeframe: MarketTimeframe,
  ): Promise<void> {
    marketError.value = "";
    await api(`/api/market/${symbol}/refresh`, { method: "POST" });
    await loadMarket(symbol, timeframe);
  }

  async function optional<T>(path: string): Promise<T | null> {
    try {
      return await api<T>(path);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 404) return null;
      throw reason;
    }
  }

  async function loadAnalysisState(symbol: LegacyV32InstrumentCode): Promise<void> {
    const generation = ++analysisGeneration;
    analysisError.value = "";
    try {
      const legacyPayload = Promise.all([
        optional<ModelMetrics>(`/api/v2/models/${symbol}/metrics`),
        optional<LatestAdvice>(`/api/v2/advice/${symbol}/latest`),
      ]);
      const v31Payload = await optional<V31Run>(`/api/v3.1/analysis/${symbol}/latest`);
      if (generation !== analysisGeneration) return;
      v31Run.value = v31Payload;
      v31Analysis.value = v31Payload?.status === "completed"
        ? (v31Payload.result as unknown as V31Analysis)
        : null;
      if (v31Analysis.value && v31Payload?.stages) {
        v31Analysis.value.stages = v31Payload.stages;
      }
      const [metricsPayload, advicePayload] = await legacyPayload;
      if (generation !== analysisGeneration) return;
      metrics.value = metricsPayload;
      advice.value = advicePayload;
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisError.value = messageFrom(reason);
    }
  }

  async function loadV32State(symbol: LegacyV32InstrumentCode | null): Promise<void> {
    const generation = analysisGeneration;
    v32Analysis.value = null;
    v32Run.value = null;
    try {
      const [status, cybCurve, ndxCurve, latest] = await Promise.all([
        api<V32TrainingStatus>("/api/v3.2/training/status"),
        api<V32CurvePayload>("/api/v3.2/training/399006/curve"),
        api<V32CurvePayload>("/api/v3.2/training/NDX/curve"),
        symbol
          ? optional<V32AnalysisRun>(`/api/v3.2/analysis/${symbol}/latest`)
          : Promise.resolve(null),
      ]);
      if (generation !== analysisGeneration) return;
      v32TrainingStatus.value = status;
      v32Curves.value = { "399006": cybCurve, NDX: ndxCurve };
      v32Run.value = latest;
      v32Analysis.value = latest?.status === "completed"
        ? (latest.result as unknown as V32Analysis)
        : null;
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisError.value = messageFrom(reason);
    }
  }

  async function loadV33State(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v33StateGeneration;
    v33AnalysisError.value = "";
    v33Analysis.value = null;
    try {
      const [status, cybCurve, etfCurve, latest] = await Promise.all([
        loadV33Status(),
        loadV33Iterations("399006"),
        loadV33Iterations("159941"),
        loadV33Forecast(symbol),
      ]);
      if (generation !== v33StateGeneration) return;
      v33Statuses.value = status.markets;
      v33Curves.value = { "399006": cybCurve, "159941": etfCurve };
      v33Analysis.value = latest;
    } catch (reason) {
      if (generation !== v33StateGeneration) return;
      v33AnalysisError.value = messageFrom(reason);
    }
  }

  async function trainV33Model(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v33TrainingGeneration[symbol];
    v33TrainingLoading.value[symbol] = true;
    v33TrainingError.value[symbol] = "";
    try {
      const created = await requestV33Training(symbol);
      if (generation !== v33TrainingGeneration[symbol]) return;
      if ("curve" in created) {
        v33Curves.value = { ...v33Curves.value, [symbol]: created };
      }
      for (;;) {
        const status = await loadV33Status();
        if (generation !== v33TrainingGeneration[symbol]) return;
        v33Statuses.value = status.markets;
        if (!status.markets[symbol]?.is_training) break;
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }
      const curve = await loadV33Iterations(symbol);
      if (generation !== v33TrainingGeneration[symbol]) return;
      v33Curves.value = { ...v33Curves.value, [symbol]: curve };
    } catch (reason) {
      if (generation === v33TrainingGeneration[symbol]) {
        v33TrainingError.value[symbol] = messageFrom(reason);
      }
    } finally {
      if (generation === v33TrainingGeneration[symbol]) {
        v33TrainingLoading.value[symbol] = false;
      }
    }
  }

  async function startV33Analysis(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v33AnalysisGeneration;
    v33AnalysisLoading.value = true;
    v33AnalysisError.value = "";
    try {
      const result = await analyzeWithV33(symbol);
      if (generation !== v33AnalysisGeneration) return;
      v33Analysis.value = result;
    } catch (reason) {
      if (generation !== v33AnalysisGeneration) return;
      v33AnalysisError.value = messageFrom(reason);
    } finally {
      if (generation === v33AnalysisGeneration) v33AnalysisLoading.value = false;
    }
  }

  async function loadV34State(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v34StateGeneration;
    v34AnalysisError.value = "";
    try {
      const [statuses, cybCurve, etfCurve, latest] = await Promise.all([
        loadV34Status(),
        loadV34Iterations("399006"),
        loadV34Iterations("159941"),
        loadV34Forecast(symbol),
      ]);
      if (generation !== v34StateGeneration) return;
      v34Statuses.value = Object.fromEntries(
        statuses.map((status) => [status.market, status]),
      ) as Partial<Record<ActiveInstrumentCode, V34MarketStatus>>;
      v34Curves.value = { "399006": cybCurve, "159941": etfCurve };
      v34Analysis.value = latest;
    } catch (reason) {
      if (generation === v34StateGeneration) v34AnalysisError.value = messageFrom(reason);
    }
  }

  async function trainV34Model(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v34TrainingGeneration[symbol];
    v34TrainingLoading.value[symbol] = true;
    v34TrainingError.value[symbol] = "";
    try {
      const created = await requestV34Training(symbol);
      for (;;) {
        const run = await loadV34TrainingRun(created.run_id);
        if (generation !== v34TrainingGeneration[symbol]) return;
        if (run.status === "failed") throw new Error(run.error_message || "V3.4训练失败");
        if (run.status === "completed") break;
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }
      await loadV34State(symbol);
    } catch (reason) {
      if (generation === v34TrainingGeneration[symbol]) {
        v34TrainingError.value[symbol] = messageFrom(reason);
      }
    } finally {
      if (generation === v34TrainingGeneration[symbol]) {
        v34TrainingLoading.value[symbol] = false;
      }
    }
  }

  async function startV34Analysis(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++v34AnalysisGeneration;
    v34AnalysisLoading.value = true;
    v34AnalysisError.value = "";
    try {
      const created = await analyzeWithV34(symbol);
      const deadline = Date.now() + V34_ANALYSIS_TIMEOUT_MS;
      for (;;) {
        const run = await loadV34AnalysisRun(created.run_id);
        if (generation !== v34AnalysisGeneration) return;
        if (run.status === "failed") {
          throw new Error(run.error_message || "V3.4 数据分析失败");
        }
        if (run.status === "completed") {
          v34Analysis.value = run.result as V34ForecastPayload;
          break;
        }
        if (Date.now() >= deadline) {
          throw new Error("数据分析超过120秒，请检查本地数据或任务状态后重试");
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }
    } catch (reason) {
      if (generation === v34AnalysisGeneration) v34AnalysisError.value = messageFrom(reason);
    } finally {
      if (generation === v34AnalysisGeneration) v34AnalysisLoading.value = false;
    }
  }

  async function pollAnalysis(taskId: string, generation: number): Promise<void> {
    if (generation !== analysisGeneration) return;
    try {
      const current = await api<AnalysisTask>(`/api/v2/analysis/tasks/${taskId}`);
      if (generation !== analysisGeneration) return;
      task.value = current;
      if (TERMINAL_TASKS.has(current.status)) {
        analysisLoading.value = false;
        if (current.status === "completed") {
          await loadAnalysisState(current.instrument_code);
        } else {
          analysisError.value = current.message ?? "分析任务失败，可检查数据后重试。";
        }
        return;
      }
      pollTimer = setTimeout(
        () => void pollAnalysis(taskId, generation),
        POLL_INTERVAL_MS,
      );
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisLoading.value = false;
      analysisError.value = messageFrom(reason);
    }
  }

  async function startAnalysis(symbol: LegacyV32InstrumentCode): Promise<void> {
    stopPolling();
    const generation = ++analysisGeneration;
    analysisLoading.value = true;
    analysisError.value = "";
    task.value = null;
    v31Run.value = null;
    try {
      const created = await api<V31Analysis | V31Run>(`/api/v3.1/analysis/${symbol}/run`, {
        method: "POST",
        body: JSON.stringify({ refresh: true }),
      });
      if (generation !== analysisGeneration) return;
      if ("status" in created) {
        v31Run.value = created;
        v31Analysis.value = null;
        analysisError.value = [
          created.current_stage,
          created.error_code,
          created.error_message,
        ].filter(Boolean).join(" · ");
      } else {
        v31Analysis.value = created;
        v31Run.value = {
          id: created.run_id,
          market: created.market,
          status: "completed",
          current_stage: "已完成",
          result: created as unknown as Record<string, unknown>,
          stages: created.stages ?? [],
        };
      }
      analysisLoading.value = false;
      const [metricsPayload, advicePayload] = await Promise.all([
        optional<ModelMetrics>(`/api/v2/models/${symbol}/metrics`),
        optional<LatestAdvice>(`/api/v2/advice/${symbol}/latest`),
      ]);
      metrics.value = metricsPayload;
      advice.value = advicePayload;
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisLoading.value = false;
      analysisError.value = messageFrom(reason);
    }
  }

  async function startV32Analysis(symbol: LegacyV32InstrumentCode): Promise<void> {
    stopPolling();
    const generation = ++analysisGeneration;
    analysisLoading.value = true;
    analysisError.value = "";
    v32Run.value = null;
    try {
      const created = await api<V32Analysis>(`/api/v3.2/analysis/${symbol}/run`, {
        method: "POST",
        body: JSON.stringify({ refresh: true }),
      });
      if (generation !== analysisGeneration) return;
      v32Analysis.value = created;
      v32Run.value = {
        id: created.run_id,
        market: created.market,
        status: "completed",
        current_stage: "已完成",
        result: created as unknown as Record<string, any>,
        stages: [],
      };
      await loadV32State(symbol);
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisError.value = messageFrom(reason);
      try {
        v32Run.value = await optional<V32AnalysisRun>(`/api/v3.2/analysis/${symbol}/latest`);
      } catch { /* preserve the original analysis error */ }
    } finally {
      if (generation === analysisGeneration) analysisLoading.value = false;
    }
  }

  async function trainV32Model(symbol: LegacyV32InstrumentCode): Promise<void> {
    trainingLoading.value[symbol] = true;
    trainingError.value[symbol] = "";
    try {
      let status = await api<V32TrainingStatus>("/api/v3.2/training/status");
      const action = status[symbol].bootstrapped ? "incremental" : "bootstrap";
      await api(`/api/v3.2/training/${symbol}/${action}`, { method: "POST" });
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 750));
        status = await api<V32TrainingStatus>("/api/v3.2/training/status");
        v32TrainingStatus.value = status;
        if (!status[symbol].is_training) break;
      }
      const latest = status[symbol].latest_run;
      if (latest?.status === "failed" || latest?.status === "interrupted") {
        throw new Error(latest.error_message || "模型训练失败，请查看训练审计记录。");
      }
      const curve = await api<V32CurvePayload>(`/api/v3.2/training/${symbol}/curve`);
      v32Curves.value = { ...v32Curves.value, [symbol]: curve };
    } catch (reason) {
      trainingError.value[symbol] = messageFrom(reason);
    } finally {
      trainingLoading.value[symbol] = false;
    }
  }

  async function resumeAnalysis(): Promise<void> {
    const current = task.value;
    if (!current || current.status !== "recoverable") return;
    stopPolling();
    const generation = ++analysisGeneration;
    analysisLoading.value = true;
    analysisError.value = "";
    try {
      const resumed = await api<AnalysisTask>(
        `/api/v2/analysis/tasks/${current.id}/resume`,
        { method: "POST" },
      );
      if (generation !== analysisGeneration) return;
      task.value = resumed;
      await pollAnalysis(resumed.id, generation);
    } catch (reason) {
      if (generation !== analysisGeneration) return;
      analysisLoading.value = false;
      analysisError.value = messageFrom(reason);
    }
  }

  async function loadPositions(symbol: ActiveInstrumentCode): Promise<void> {
    const generation = ++calendarGeneration;
    calendarLoading.value = true;
    calendarError.value = "";
    try {
      const [events, positions] = await Promise.all([
        api<PositionEvent[]>(`/api/v2/position-events?instrument_code=${symbol}`),
        api<Record<ActiveInstrumentCode, number | null>>(
          "/api/investment-calendar/current-positions",
        ),
      ]);
      if (generation !== calendarGeneration) return;
      positionEvents.value = events;
      currentPositions.value = positions;
    } catch (reason) {
      if (generation !== calendarGeneration) return;
      calendarError.value = messageFrom(reason);
    } finally {
      if (generation === calendarGeneration) calendarLoading.value = false;
    }
  }

  async function createPosition(values: PositionEventWrite): Promise<void> {
    await api<PositionEvent>("/api/v2/position-events", {
      method: "POST",
      body: JSON.stringify(values),
    });
    await loadPositions(values.instrument_code);
  }

  async function updatePosition(
    id: number,
    values: Partial<PositionEventWrite>,
    symbol: ActiveInstrumentCode,
  ): Promise<void> {
    await api<PositionEvent>(`/api/v2/position-events/${id}`, {
      method: "PATCH",
      body: JSON.stringify(values),
    });
    await loadPositions(symbol);
  }

  async function deletePosition(id: number, symbol: ActiveInstrumentCode): Promise<void> {
    await api<{ deleted: true; id: number }>(`/api/v2/position-events/${id}?confirmed=true`, {
      method: "DELETE",
    });
    await loadPositions(symbol);
  }

  onBeforeUnmount(() => {
    marketGeneration += 1;
    calendarGeneration += 1;
    v33StateGeneration += 1;
    v33AnalysisGeneration += 1;
    v33TrainingGeneration["399006"] += 1;
    v33TrainingGeneration["159941"] += 1;
    v34StateGeneration += 1;
    v34AnalysisGeneration += 1;
    v34TrainingGeneration["399006"] += 1;
    v34TrainingGeneration["159941"] += 1;
    stopPolling();
  });

  return {
    prices,
    indicators,
    currentPeriodStatus,
    normalized,
    task,
    metrics,
    advice,
    v31Analysis,
    v31Run,
    v32Analysis,
    v32Run,
    v32TrainingStatus,
    v32Curves,
    trainingLoading,
    trainingError,
    v33Analysis,
    v33Statuses,
    v33Curves,
    v33TrainingLoading,
    v33TrainingError,
    v33AnalysisLoading,
    v33AnalysisError,
    v34Analysis,
    v34Statuses,
    v34Curves,
    v34TrainingLoading,
    v34TrainingError,
    v34AnalysisLoading,
    v34AnalysisError,
    positionEvents,
    currentPositions,
    marketLoading,
    analysisLoading,
    calendarLoading,
    marketError,
    analysisError,
    calendarError,
    loadMarket,
    refreshMarket,
    loadAnalysisState,
    loadV32State,
    loadV33State,
    startAnalysis,
    startV32Analysis,
    trainV32Model,
    trainV33Model,
    startV33Analysis,
    loadV34State,
    trainV34Model,
    startV34Analysis,
    resumeAnalysis,
    stopPolling,
    loadPositions,
    createPosition,
    updatePosition,
    deletePosition,
  };
}

function messageFrom(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}
