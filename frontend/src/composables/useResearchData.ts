import { computed, onBeforeUnmount, ref } from "vue";

import { api, ApiError } from "../api";
import type {
  ActiveInstrumentCode,
  IndicatorRow,
  MarketTimeframe,
  PositionEvent,
  PricePayload,
  PriceRow,
  ResearchInstrumentCode,
} from "../types/research";
import { normalizeSeries } from "../types/research";

type PositionEventWrite = {
  instrument_code: ActiveInstrumentCode;
  direction: "increase" | "decrease";
  operation_date: string;
  change_percent: number;
  note: string | null;
};

/**
 * V3.7 data-only state boundary.
 *
 * This composable intentionally contains no training, AI, inference or
 * forecast requests.  It only loads chart prices/indicators and the local
 * investment-calendar records.  Keeping this boundary small also prevents a
 * refresh or a route mount from accidentally starting a legacy model task.
 */
export function useResearchData() {
  const prices = ref<PriceRow[]>([]);
  const indicators = ref<IndicatorRow[]>([]);
  // The data-only market view keeps daily and weekly series resident at the
  // same time.  This lets the UI compare the two resolutions without
  // throwing away the previous chart when a tab is clicked.
  const dailyPrices = ref<PriceRow[]>([]);
  const dailyIndicators = ref<IndicatorRow[]>([]);
  const weeklyPrices = ref<PriceRow[]>([]);
  const weeklyIndicators = ref<IndicatorRow[]>([]);
  const indicatorSnapshots = ref<{
    daily: IndicatorRow | null;
    weekly: IndicatorRow | null;
  }>({ daily: null, weekly: null });
  const currentPeriodStatus = ref<"COMPLETE" | "INCOMPLETE_CURRENT_PERIOD">("COMPLETE");

  const positionEvents = ref<PositionEvent[]>([]);
  const currentPositions = ref<Partial<Record<ActiveInstrumentCode, number | null>>>({});

  const marketLoading = ref(false);
  const refreshing = ref(false);
  const calendarLoading = ref(false);
  const marketError = ref("");
  const calendarError = ref("");

  const normalized = computed(() => normalizeSeries(prices.value, indicators.value));
  let marketGeneration = 0;
  let calendarGeneration = 0;

  async function loadMarket(
    symbol: ResearchInstrumentCode,
    timeframe: MarketTimeframe,
  ): Promise<void> {
    const generation = ++marketGeneration;
    marketLoading.value = true;
    marketError.value = "";
    prices.value = [];
    indicators.value = [];
    dailyPrices.value = [];
    dailyIndicators.value = [];
    weeklyPrices.value = [];
    weeklyIndicators.value = [];
    indicatorSnapshots.value = { daily: null, weekly: null };
    currentPeriodStatus.value = "COMPLETE";
    try {
      const monthlyRequest = timeframe === "monthly"
        ? Promise.all([
            api<PricePayload>(`/api/market/${symbol}/prices?timeframe=monthly`),
            api<IndicatorRow[]>(`/api/indicators/${symbol}?timeframe=monthly`),
          ])
        : Promise.resolve(null);
      const [basePayloads, monthlyPayloads] = await Promise.all([
        Promise.all([
          api<PricePayload>(`/api/market/${symbol}/prices?timeframe=daily`),
          api<IndicatorRow[]>(`/api/indicators/${symbol}?timeframe=daily`),
          api<PricePayload>(`/api/market/${symbol}/prices?timeframe=weekly`),
          api<IndicatorRow[]>(`/api/indicators/${symbol}?timeframe=weekly`),
        ]),
        monthlyRequest,
      ]);
      if (generation !== marketGeneration) return;
      const [dailyPricePayload, dailyPayload, weeklyPricePayload, weeklyPayload] = basePayloads;
      const selectedPricePayload = timeframe === "daily"
        ? dailyPricePayload
        : timeframe === "weekly"
          ? weeklyPricePayload
          : monthlyPayloads?.[0] ?? { rows: [] };
      const selectedIndicatorPayload = timeframe === "daily"
        ? dailyPayload
        : timeframe === "weekly"
          ? weeklyPayload
          : monthlyPayloads?.[1] ?? [];
      prices.value = selectedPricePayload.rows ?? [];
      indicators.value = selectedIndicatorPayload ?? [];
      dailyPrices.value = dailyPricePayload.rows ?? [];
      dailyIndicators.value = dailyPayload ?? [];
      weeklyPrices.value = weeklyPricePayload.rows ?? [];
      weeklyIndicators.value = weeklyPayload ?? [];
      indicatorSnapshots.value = {
        daily: dailyPayload.at(-1) ?? null,
        weekly: weeklyPayload.at(-1) ?? null,
      };
      currentPeriodStatus.value = weeklyPricePayload.current_period_status ?? "COMPLETE";
    } catch (reason) {
      if (generation !== marketGeneration) return;
      prices.value = [];
      indicators.value = [];
      dailyPrices.value = [];
      dailyIndicators.value = [];
      weeklyPrices.value = [];
      weeklyIndicators.value = [];
      indicatorSnapshots.value = { daily: null, weekly: null };
      currentPeriodStatus.value = "COMPLETE";
      marketError.value = messageFrom(reason);
    } finally {
      if (generation === marketGeneration) marketLoading.value = false;
    }
  }

  async function refreshMarket(
    symbol: ResearchInstrumentCode,
    timeframe: MarketTimeframe,
  ): Promise<Record<string, unknown>> {
    marketError.value = "";
    refreshing.value = true;
    const generationBeforeRefresh = marketGeneration;
    try {
      const payload = await api<Record<string, unknown>>(
        `/api/market/${symbol}/refresh`,
        { method: "POST" },
      );
      refreshing.value = false;
      // If the user selected another ETF while the provider was downloading,
      // that newer selection owns the chart.  Never let the old refresh
      // overwrite it when the network response finally arrives.
      if (generationBeforeRefresh === marketGeneration) {
        await loadMarket(symbol, timeframe);
      }
      return payload;
    } finally {
      refreshing.value = false;
    }
  }

  async function loadPositions(symbol: ActiveInstrumentCode): Promise<void> {
    // The selected instrument still controls the entry form and the complete
    // per-instrument ledger, but the month calendar needs every instrument's
    // events so one selected date can show the whole portfolio activity.
    void symbol;
    const generation = ++calendarGeneration;
    calendarLoading.value = true;
    calendarError.value = "";
    try {
      const [events, positions] = await Promise.all([
        api<PositionEvent[]>("/api/v2/position-events?scope=all"),
        api<Partial<Record<ActiveInstrumentCode, number | null>>>(
          "/api/investment-calendar/current-positions",
        ),
      ]);
      if (generation !== calendarGeneration) return;
      positionEvents.value = events ?? [];
      currentPositions.value = positions ?? {};
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
  });

  return {
    prices,
    indicators,
    dailyPrices,
    dailyIndicators,
    weeklyPrices,
    weeklyIndicators,
    indicatorSnapshots,
    currentPeriodStatus,
    normalized,
    positionEvents,
    currentPositions,
    marketLoading,
    refreshing,
    calendarLoading,
    marketError,
    calendarError,
    loadMarket,
    refreshMarket,
    loadPositions,
    createPosition,
    updatePosition,
    deletePosition,
  };
}

function messageFrom(reason: unknown): string {
  if (reason instanceof ApiError) return reason.message;
  return reason instanceof Error ? reason.message : String(reason);
}
