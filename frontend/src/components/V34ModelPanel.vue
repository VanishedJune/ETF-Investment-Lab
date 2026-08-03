<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type {
  ActiveInstrumentCode,
  V34CurvePayload,
  V34ForecastPayload,
  V34MarketStatus,
} from "../types/research";
import { calculateDynamicYAxis } from "../utils/chartScale";

const props = defineProps<{
  selected: ActiveInstrumentCode;
  result: V34ForecastPayload | null;
  statuses: Partial<Record<ActiveInstrumentCode, V34MarketStatus>>;
  curves: Partial<Record<ActiveInstrumentCode, V34CurvePayload>>;
  analysisLoading: boolean;
  trainingLoading: Record<ActiveInstrumentCode, boolean>;
  trainingError: Record<ActiveInstrumentCode, string>;
  analysisError?: string;
}>();

const emit = defineEmits<{
  analyze: [];
  train: [market: ActiveInstrumentCode];
}>();

const markets: ActiveInstrumentCode[] = ["399006", "159941"];
const names: Record<ActiveInstrumentCode, string> = {
  "399006": "创业板指数",
  "159941": "广发纳斯达克100ETF",
};
const ranges = [52, 104, 0] as const;
const range = ref<(typeof ranges)[number]>(0);
const curveElements = ref<Partial<Record<ActiveInstrumentCode, HTMLDivElement>>>({});
const curveCharts = new Map<ActiveInstrumentCode, echarts.ECharts>();
const forecastElement = ref<HTMLDivElement>();
let forecastChart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;
let forecastScaleFrame: number | undefined;
let forecastPointerFrame: number | undefined;
let pendingForecastPointerIndex = -1;
let forecastZoomStart = 0;
let forecastZoomEnd = 100;
let forecastLegend: Record<string, boolean> = {};
const forecastHoveredIndex = ref(-1);
const forecastPointerUpdateCount = ref(0);

const selectedStatus = computed(() => props.statuses[props.selected]);
const directionAt13 = computed(() => {
  const horizon = props.result?.horizon_probabilities?.["13"] as unknown as {
    up?: number;
    sideways?: number;
    down?: number;
    calibrated?: { up?: number; sideways?: number; down?: number };
  } | undefined;
  const probabilities = horizon?.calibrated ?? horizon;
  return {
    up: probabilities?.up ?? 0,
    sideways: probabilities?.sideways ?? 0,
    down: probabilities?.down ?? 0,
  };
});
const decision = computed(() => props.result?.policy?.decision ?? null);
const assessment = computed(() => props.result?.policy?.turning_assessment ?? null);

function confirmedWindow(signal: "PRICE" | "DIF"): string {
  const candidates = assessment.value?.candidates.filter((candidate) => (
    candidate.signal_kind === signal
    && candidate.classification === "VALID_TURN"
    && candidate.confirmation_status === "CONFIRMED"
  )) ?? [];
  const candidate = candidates.at(0);
  return candidate
    ? `${candidate.window_start_date}—${candidate.window_end_date} · ${candidate.turn_kind}`
    : "尚无已确认有效拐点";
}

function setCurveElement(market: ActiveInstrumentCode, element: unknown) {
  if (element instanceof HTMLDivElement) curveElements.value[market] = element;
}

function percentage(value: unknown): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  const percent = Math.abs(parsed) <= 1 ? parsed * 100 : parsed;
  return `${percent.toFixed(1)}%`;
}

function number(value: unknown, digits = 2): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
}

function visibleCurvePoints(market: ActiveInstrumentCode) {
  const all = props.curves[market]?.points ?? [];
  return range.value ? all.slice(-range.value) : all;
}

function latestCurveEvent(market: ActiveInstrumentCode, kind: "pending" | "candidate" | "promotion") {
  return [...visibleCurvePoints(market)].reverse().find((point) => (
    kind === "pending" ? point.pending : kind === "candidate" ? point.training_triggered : point.promoted
  ));
}

function renderCurves() {
  for (const market of markets) {
    const element = curveElements.value[market];
    if (!element) continue;
    const chart = curveCharts.get(market) ?? echarts.init(element);
    curveCharts.set(market, chart);
    const points = visibleCurvePoints(market);
    chart.setOption({
      animationDuration: 220,
      tooltip: {
        trigger: "axis",
        formatter(parameters: any[]) {
          const point = points[parameters[0]?.dataIndex ?? 0];
          if (!point) return "";
          return [
            `<b>第 ${point.iteration} 次 · ${point.anchor_date}</b>`,
            `成熟状态：${point.pending ? "PENDING" : point.maturity_status ?? "—"}`,
            `评价可用日期：${point.evaluation_available_date ?? "—"}`,
            `价格拐点偏离：${point.price_turn_deviation_trading_days ?? "—"} 个实际交易日`,
            `DIF拐点偏离：${point.dif_turn_deviation_trading_days ?? "—"} 个实际交易日`,
            `候选训练：${point.training_triggered ? "是" : "否"}`,
            `冠军晋级：${point.promoted ? "是" : "否"}`,
          ].join("<br>");
        },
      },
      legend: { top: 4, data: ["价格拐点偏离", "DIF拐点偏离"] },
      grid: { left: 62, right: 24, top: 48, bottom: 48 },
      xAxis: {
        type: "category",
        name: "迭代次数",
        data: points.map((point) => point.iteration),
        axisLabel: { hideOverlap: true },
      },
      yAxis: {
        type: "value",
        name: "实际交易日偏离",
        scale: true,
        splitLine: { lineStyle: { color: "#e5e8e9", type: "dashed" } },
      },
      series: [
        {
          name: "价格拐点偏离",
          type: "line",
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.pending ? null : point.price_turn_deviation_trading_days),
          lineStyle: { color: "#12687e", width: 2 },
        },
        {
          name: "DIF拐点偏离",
          type: "line",
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.pending ? null : point.dif_turn_deviation_trading_days),
          lineStyle: { color: "#d9553f", width: 2 },
        },
      ],
    }, true);
  }
}

const forecastSeries = computed(() => {
  if (!props.result) return null;
  const history = props.result.historical_ohlcv ?? [];
  const prediction = props.result.representative_ohlcv;
  const historicalIndicators = props.result.historical_indicators ?? [];
  const indicatorByDate = new Map(historicalIndicators.map((row) => [row.week_end, row]));
  const alignedHistoricalIndicators = history.map((row) => indicatorByDate.get(row.week_end));
  const historyLength = history.length;
  const dates = [
    ...history.map((row) => row.week_end),
    ...prediction.map((row) => row.week_end),
  ];
  const historyCandles = history.map((row) => [row.open, row.close, row.low, row.high]);
  const predictedCandles = prediction.map((row) => [row.open, row.close, row.low, row.high]);
  const emptyHistory = Array.from({ length: historyLength }, () => "-");
  return {
    historyLength,
    dates,
    combinedCandles: [...historyCandles, ...predictedCandles] as number[][],
    historyCandles: [...historyCandles, ...prediction.map(() => "-")],
    predictedCandles: [...emptyHistory, ...predictedCandles],
    volumes: [
      ...history.map((row) => row.volume),
      ...prediction.map((row) => row.volume_p50),
    ],
    p10: [...emptyHistory, ...props.result.price_quantiles.map((row) => row.close_p10)],
    p50: [...emptyHistory, ...props.result.price_quantiles.map((row) => row.close_p50)],
    band: [
      ...emptyHistory,
      ...props.result.price_quantiles.map((row) => row.close_p90 - row.close_p10),
    ],
    dif: [
      ...alignedHistoricalIndicators.map((row) => row?.dif ?? null),
      ...props.result.indicators.map((row) => row.dif),
    ],
    dea: [
      ...alignedHistoricalIndicators.map((row) => row?.dea ?? null),
      ...props.result.indicators.map((row) => row.dea),
    ],
    macd: [
      ...alignedHistoricalIndicators.map((row) => row?.macd ?? null),
      ...props.result.indicators.map((row) => row.macd),
    ],
    difFirstChange: [
      ...alignedHistoricalIndicators.map((row) => row?.dif_first_change ?? null),
      ...props.result.indicators.map((row) => row.dif_first_change),
    ],
  };
});

const forecastReadout = computed(() => {
  const values = forecastSeries.value;
  const result = props.result;
  if (!values || !result || values.dates.length === 0) return null;
  const fallbackIndex = Math.min(values.dates.length - 1, values.historyLength);
  const index = Math.max(0, Math.min(
    values.dates.length - 1,
    forecastHoveredIndex.value >= 0 ? forecastHoveredIndex.value : fallbackIndex,
  ));
  if (index < values.historyLength) {
    const candle = result.historical_ohlcv[index];
    const indicator = (result.historical_indicators ?? []).find(
      (row) => row.week_end === candle?.week_end,
    );
    if (!candle) return null;
    return {
      index,
      phase: "历史周K",
      date: candle.week_end,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      p10: null,
      p50: null,
      p90: null,
      up: null,
      sideways: null,
      down: null,
      dif: indicator?.dif,
      dea: indicator?.dea,
      macd: indicator?.macd,
      difFirstChange: indicator?.dif_first_change,
      trendState: indicator?.trend_state ?? "历史实测",
    };
  }
  const predictionIndex = index - values.historyLength;
  const candle = result.representative_ohlcv[predictionIndex];
  const quantile = result.price_quantiles[predictionIndex];
  const indicator = result.indicators[predictionIndex];
  if (!candle || !quantile || !indicator) return null;
  const direction = candle.direction_probabilities;
  return {
    index,
    phase: `预测第${candle.week}周`,
    date: `${candle.week_start} — ${candle.week_end}`,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume_p50,
    p10: quantile.close_p10,
    p50: quantile.close_p50,
    p90: quantile.close_p90,
    up: direction?.up ?? null,
    sideways: direction?.sideways ?? null,
    down: direction?.down ?? null,
    dif: indicator.dif,
    dea: indicator.dea,
    macd: indicator.macd,
    difFirstChange: indicator.dif_first_change,
    trendState: indicator.trend_state ?? "—",
  };
});

function forecastPointerIndex(event: unknown): number | null {
  const values = forecastSeries.value;
  if (!values) return null;
  const payload = event as {
    axesInfo?: { axisDim?: string; axisIndex?: number; value?: number | string }[];
  };
  const axis = payload.axesInfo?.find((item) => item.axisDim === "x" && item.axisIndex === 0)
    ?? payload.axesInfo?.find((item) => item.axisDim === "x");
  if (!axis) return null;
  const numeric = Number(axis.value);
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < values.dates.length) return numeric;
  const byDate = values.dates.indexOf(String(axis.value));
  return byDate >= 0 ? byDate : null;
}

function scheduleForecastPointer(index: number) {
  pendingForecastPointerIndex = index;
  if (forecastPointerFrame !== undefined) return;
  forecastPointerFrame = requestAnimationFrame(() => {
    forecastPointerFrame = undefined;
    if (pendingForecastPointerIndex === forecastHoveredIndex.value) return;
    forecastHoveredIndex.value = pendingForecastPointerIndex;
    forecastPointerUpdateCount.value += 1;
  });
}

function applyForecastScale() {
  forecastScaleFrame = undefined;
  const values = forecastSeries.value;
  if (!forecastChart || !forecastElement.value || !values) return;
  const option = forecastChart.getOption() as { dataZoom?: { start?: number; end?: number }[] };
  const start = Number(option.dataZoom?.[0]?.start ?? forecastZoomStart);
  const end = Number(option.dataZoom?.[0]?.end ?? forecastZoomEnd);
  const numeric = (rows: Array<number | string | null>) => rows.map((value) => (
    typeof value === "number" ? value : Number.NaN
  ));
  const ranges = calculateDynamicYAxis({
    candles: values.combinedCandles,
    ma5: numeric(values.p10),
    ma10: numeric(values.p50),
    ma20: numeric(values.p10.map((value, index) => (
      typeof value === "number" && typeof values.band[index] === "number"
        ? value + values.band[index]
        : Number.NaN
    ))),
    volumes: values.volumes.map((value) => value == null ? null : Number(value)),
    dif: numeric(values.dif),
    dea: numeric(values.dea),
    macd: numeric(values.macd),
    difFirstChange: numeric(values.difFirstChange),
  }, start, end, {
    MA5: forecastLegend.P10,
    MA10: forecastLegend.P50,
    MA20: forecastLegend["P10—P90价格带"],
    成交量: forecastLegend.预测成交量,
    DIF: forecastLegend.DIF,
    DEA: forecastLegend.DEA,
    MACD柱: forecastLegend.MACD柱,
    DIF一阶变化: forecastLegend.DIF一阶变化,
  });
  forecastChart.setOption({
    yAxis: [
      { min: ranges.price.min, max: ranges.price.max },
      { min: ranges.volume.min, max: ranges.volume.max },
      { min: ranges.macd.min, max: ranges.macd.max },
      { min: ranges.difFirstChange.min, max: ranges.difFirstChange.max },
    ],
  }, { lazyUpdate: true });
  forecastElement.value.dataset.priceYMin = String(ranges.price.min);
  forecastElement.value.dataset.priceYMax = String(ranges.price.max);
  forecastElement.value.dataset.volumeYMin = String(ranges.volume.min);
  forecastElement.value.dataset.volumeYMax = String(ranges.volume.max);
  forecastElement.value.dataset.macdYMin = String(ranges.macd.min);
  forecastElement.value.dataset.macdYMax = String(ranges.macd.max);
  forecastElement.value.dataset.difFirstChangeYMin = String(ranges.difFirstChange.min);
  forecastElement.value.dataset.difFirstChangeYMax = String(ranges.difFirstChange.max);
  forecastElement.value.dataset.visibleStartIndex = String(ranges.startIndex);
  forecastElement.value.dataset.visibleEndIndex = String(ranges.endIndex);
}

function scheduleForecastScale() {
  if (forecastScaleFrame !== undefined) cancelAnimationFrame(forecastScaleFrame);
  forecastScaleFrame = requestAnimationFrame(applyForecastScale);
}

function renderForecast() {
  if (!forecastElement.value) return;
  forecastChart ??= echarts.init(forecastElement.value, undefined, {
    renderer: "canvas",
    useDirtyRect: true,
  });
  const values = forecastSeries.value;
  if (!values || props.result?.representative_ohlcv.length !== 13) {
    forecastChart.clear();
    forecastHoveredIndex.value = -1;
    return;
  }
  if (forecastHoveredIndex.value < 0 || forecastHoveredIndex.value >= values.dates.length) {
    forecastHoveredIndex.value = Math.min(values.dates.length - 1, values.historyLength);
  }
  forecastChart.setOption({
    animation: false,
    axisPointer: { link: [{ xAxisIndex: [0, 1, 2, 3] }] },
    tooltip: {
      trigger: "axis",
      triggerOn: "mousemove|click",
      showContent: false,
      transitionDuration: 0,
      axisPointer: { type: "line", snap: true },
    },
    legend: {
      top: 2,
      selected: forecastLegend,
      data: ["历史周K", "P50代表性预测周K", "P10—P90价格带", "预测成交量", "MACD柱", "DIF", "DEA", "DIF一阶变化"],
    },
    grid: [
      { left: 68, right: 58, top: 48, height: "37%" },
      { left: 68, right: 58, top: "45%", height: "10%" },
      { left: 68, right: 58, top: "58%", height: "13%" },
      { left: 68, right: 58, top: "74%", height: "13%" },
    ],
    xAxis: [0, 1, 2, 3].map((gridIndex) => ({
      type: "category",
      gridIndex,
      data: values.dates,
      axisLabel: { show: gridIndex === 3, hideOverlap: true },
      axisTick: { show: false },
      axisPointer: { label: { show: gridIndex === 3, backgroundColor: "#223b4a" } },
    })),
    yAxis: [
      { type: "value", scale: true, position: "right" },
      { type: "value", gridIndex: 1, min: 0, position: "right", axisLabel: { formatter: (value: number) => Intl.NumberFormat("zh-CN", { notation: "compact" }).format(value) } },
      { type: "value", gridIndex: 2, position: "right" },
      { type: "value", gridIndex: 3, position: "right" },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2, 3], start: forecastZoomStart, end: forecastZoomEnd, filterMode: "none" },
      { type: "slider", xAxisIndex: [0, 1, 2, 3], start: forecastZoomStart, end: forecastZoomEnd, bottom: 4, height: 20, filterMode: "none" },
    ],
    series: [
      {
        name: "历史周K",
        type: "candlestick",
        data: values.historyCandles,
        itemStyle: { color: "#d9553f", color0: "#14836d", borderColor: "#d9553f", borderColor0: "#14836d" },
      },
      {
        name: "P50代表性预测周K",
        type: "candlestick",
        data: values.predictedCandles,
        itemStyle: { color: "#f0a54a", color0: "#4f91a6", borderColor: "#b56616", borderColor0: "#12687e" },
        markLine: {
          silent: true,
          symbol: "none",
          label: { formatter: "历史 / 预测", color: "#8b4e12" },
          lineStyle: { color: "#b56616", type: "dashed", width: 2 },
          data: [{ xAxis: values.historyLength - 0.5 }],
        },
        markPoint: {
          symbolSize: 34,
          data: [1, 4, 8, 13].map((week) => ({
            name: `第${week}周`,
            coord: [values.historyLength + week - 1, props.result?.representative_ohlcv[week - 1]?.high],
            value: `W${week}`,
          })),
        },
      },
      { name: "P10", type: "line", stack: "band", symbol: "none", data: values.p10, lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 } },
      { name: "P10—P90价格带", type: "line", stack: "band", symbol: "none", data: values.band, lineStyle: { opacity: 0 }, areaStyle: { color: "rgba(18,104,126,.20)" } },
      { name: "P50", type: "line", symbol: "none", data: values.p50, lineStyle: { color: "#12687e", width: 2 } },
      { name: "预测成交量", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: values.volumes, itemStyle: { color: "#668b96", opacity: .72 } },
      { name: "MACD柱", type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: values.macd, itemStyle: { color: (item: { value: number }) => item.value >= 0 ? "#d9553f" : "#14836d" } },
      { name: "DIF", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, data: values.dif, lineStyle: { color: "#12687e", width: 2 } },
      { name: "DEA", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, data: values.dea, lineStyle: { color: "#d88b2c", width: 2 } },
      { name: "DIF一阶变化", type: "line", xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, data: values.difFirstChange, lineStyle: { color: "#7b4ca0", width: 2 } },
    ],
  }, true);
  forecastChart.off("dataZoom");
  forecastChart.on("dataZoom", (event: any) => {
    const payload = event.batch?.[0] ?? event;
    if (typeof payload.start === "number") forecastZoomStart = payload.start;
    if (typeof payload.end === "number") forecastZoomEnd = payload.end;
    scheduleForecastScale();
  });
  forecastChart.off("legendselectchanged");
  forecastChart.on("legendselectchanged", (event: any) => {
    forecastLegend = { ...(event.selected ?? {}) };
    scheduleForecastScale();
  });
  forecastChart.off("updateAxisPointer");
  forecastChart.on("updateAxisPointer", (event: unknown) => {
    const index = forecastPointerIndex(event);
    if (index !== null) scheduleForecastPointer(index);
  });
  scheduleForecastScale();
}

watch([() => props.curves, range], () => void nextTick(renderCurves), { deep: true });
watch(() => props.result, () => {
  forecastHoveredIndex.value = -1;
  void nextTick(renderForecast);
});
onMounted(() => {
  renderCurves();
  renderForecast();
  resizeObserver = new ResizeObserver(() => {
    curveCharts.forEach((chart) => chart.resize());
    forecastChart?.resize();
  });
  Object.values(curveElements.value).forEach((element) => element && resizeObserver?.observe(element));
  if (forecastElement.value) resizeObserver.observe(forecastElement.value);
});
onBeforeUnmount(() => {
  if (forecastScaleFrame !== undefined) cancelAnimationFrame(forecastScaleFrame);
  if (forecastPointerFrame !== undefined) cancelAnimationFrame(forecastPointerFrame);
  resizeObserver?.disconnect();
  curveCharts.forEach((chart) => chart.dispose());
  forecastChart?.dispose();
});
</script>

<template>
  <section class="v34-lab" aria-labelledby="v34-title">
    <header class="lab-head">
      <div>
        <p class="kicker">V3.4.1 MODEL CORE + V3.4.2 TURNING POLICY</p>
        <h2 id="v34-title">13周概率情景周K · 冻结预测与条件仓位</h2>
        <p>训练模型与数据分析完全分开。周K主模型输出13周分布，近100日日K只修正前1—4周；159941仅使用自身OHLCV及自身派生指标。</p>
      </div>
      <div class="primary-actions">
        <button data-testid="v34-train" type="button" :disabled="trainingLoading[selected]" @click="emit('train', selected)">
          {{ trainingLoading[selected] ? "训练进行中…" : "训练模型" }}
        </button>
        <button data-testid="v34-analyze" class="primary" type="button" :disabled="analysisLoading || trainingLoading[selected] || !selectedStatus?.bootstrapped" @click="emit('analyze')">
          {{ analysisLoading ? "正在分析…" : "数据分析" }}
        </button>
      </div>
    </header>

    <div class="status-strip">
      <span><b>{{ names[selected] }}</b>{{ selected }}</span>
      <span><b>{{ selectedStatus?.weekly_iteration_count ?? 0 }}</b>周度迭代</span>
      <span><b>{{ selectedStatus?.candidate_training_count ?? 0 }}</b>候选训练</span>
      <span><b>{{ selectedStatus?.champion_promotion_count ?? 0 }}</b>冠军晋级</span>
      <span><b>{{ selectedStatus?.champion?.version ?? "尚未训练" }}</b>当前冠军</span>
    </div>
    <p v-if="trainingError[selected]" class="error" role="alert">{{ trainingError[selected] }}</p>
    <p v-if="analysisError" class="error" role="alert">{{ analysisError }}</p>

    <div class="curve-toolbar">
      <div><b>两个模型的递进迭代曲线</b><span>横轴为迭代次数，纵轴为已成熟预测的真实交易日偏离；PENDING保持空值，不补零。</span></div>
      <div class="range-tabs">
        <button v-for="item in ranges" :key="item" type="button" :aria-pressed="range === item" @click="range = item">{{ item || "全部" }}</button>
      </div>
    </div>
    <div class="curve-grid">
      <article v-for="market in markets" :key="market" class="curve-card">
        <header><div><span>{{ market }}</span><h3>{{ names[market] }}</h3></div><b>{{ statuses[market]?.weekly_iteration_count ?? 0 }} 次</b></header>
        <div :ref="(element) => setCurveElement(market, element)" :data-testid="`v34-curve-${market}`" class="curve" role="img"></div>
        <div class="iteration-event-strip" :data-testid="`v34-events-${market}`">
          <span class="pending-event">PENDING {{ visibleCurvePoints(market).filter((point) => point.pending).length }}<small>{{ latestCurveEvent(market, "pending")?.anchor_date ?? "—" }}</small></span>
          <span class="candidate-event">候选训练 {{ visibleCurvePoints(market).filter((point) => point.training_triggered).length }}<small>{{ latestCurveEvent(market, "candidate")?.anchor_date ?? "—" }}</small></span>
          <span class="promotion-event">Champion晋级 {{ visibleCurvePoints(market).filter((point) => point.promoted).length }}<small>{{ latestCurveEvent(market, "promotion")?.anchor_date ?? "—" }}</small></span>
        </div>
      </article>
    </div>

    <template v-if="result">
      <section class="verdict" data-testid="v34-analysis-result">
        <article>
          <p class="kicker">13-WEEK DIRECTION</p>
          <h3>第13周最终方向概率</h3>
          <div class="probabilities">
            <span><b>{{ percentage(directionAt13.up) }}</b>上涨</span>
            <span><b>{{ percentage(directionAt13.sideways) }}</b>横盘</span>
            <span><b>{{ percentage(directionAt13.down) }}</b>下跌</span>
          </div>
          <p>模型可靠性 <strong>{{ number(result.model_reliability.score, 0) }}/100</strong> · {{ result.health_status }}</p>
          <small>P10/P50/P90是每周收盘价分布的分位数，不代表90%置信度或必然上涨。</small>
        </article>
        <article>
          <p class="kicker">POSITION DECISION</p>
          <h3>{{ decision?.action === "BUY" ? "条件分批增仓" : decision?.action === "SELL" ? "条件分批减仓" : "维持仓位" }}</h3>
          <div class="position"><b>{{ decision?.current_position ?? 0 }}%</b><i>→</i><b>{{ decision?.target_position ?? 0 }}%</b></div>
          <p>基金 : ETF = <strong>{{ decision?.fund_etf_ratio ?? "—" }}</strong></p>
          <small v-if="!decision">尚未运行V3.4.2仓位分析；预测与训练结果仍可独立审阅。</small>
          <small v-else>后续批次必须满足已冻结触发条件；系统不会自动执行交易。</small>
        </article>
      </section>

      <section class="forecast-card">
        <header>
          <div><p class="kicker">HISTORY / FORECAST BOUNDARY</p><h3>历史周K + 13根P50代表性预测周K</h3></div>
          <span>{{ result.model_version }} · 情景 {{ result.scenario_count }} 条 · 锚点 {{ result.forecast_anchor_date }}</span>
        </header>
        <section
          class="forecast-point-details"
          data-testid="forecast-point-details"
          :data-hover-index="forecastReadout?.index ?? -1"
          :data-update-count="forecastPointerUpdateCount"
          aria-live="polite"
        >
          <template v-if="forecastReadout">
            <div class="forecast-point-heading">
              <span>{{ forecastReadout.phase }}</span>
              <time>{{ forecastReadout.date }}</time>
              <b>{{ forecastReadout.trendState }}</b>
            </div>
            <dl>
              <div><dt>开 / 高</dt><dd>{{ number(forecastReadout.open) }} / {{ number(forecastReadout.high) }}</dd></div>
              <div><dt>低 / 收</dt><dd>{{ number(forecastReadout.low) }} / {{ number(forecastReadout.close) }}</dd></div>
              <div><dt>成交量</dt><dd>{{ number(forecastReadout.volume, 0) }}</dd></div>
              <div><dt>DIF / DEA</dt><dd>{{ number(forecastReadout.dif, 4) }} / {{ number(forecastReadout.dea, 4) }}</dd></div>
              <div><dt>MACD / DIF一阶变化</dt><dd>{{ number(forecastReadout.macd, 4) }} / {{ number(forecastReadout.difFirstChange, 4) }}</dd></div>
              <div v-if="forecastReadout.p50 !== null"><dt>P10 / P50 / P90</dt><dd>{{ number(forecastReadout.p10) }} / {{ number(forecastReadout.p50) }} / {{ number(forecastReadout.p90) }}</dd></div>
              <div v-if="forecastReadout.up !== null"><dt>上涨 / 横盘 / 下跌</dt><dd>{{ percentage(forecastReadout.up) }} / {{ percentage(forecastReadout.sideways) }} / {{ percentage(forecastReadout.down) }}</dd></div>
            </dl>
          </template>
        </section>
        <div ref="forecastElement" class="forecast-chart" data-testid="v34-forecast-chart" role="img" aria-label="13周P50代表性预测K线、成交量、价格分位带和MACD指标"></div>
        <p class="semantic-note">半透明区域是逐周收盘价P10—P90边际分位带，并不保证上下边界可组成一条单独可实现路径。预测K线是一条最接近逐周中位数的完整连续情景。</p>
      </section>

      <section class="turning-grid">
        <article>
          <span>价格拐点窗口</span>
          <h3>{{ confirmedWindow("PRICE") }}</h3>
          <p>{{ assessment?.price_turn_status ?? "等待V3.4.2分析" }}</p>
        </article>
        <article>
          <span>DIF一阶变化零点窗口</span>
          <h3>{{ confirmedWindow("DIF") }}</h3>
          <p>{{ assessment?.dif_turn_status ?? "等待V3.4.2分析" }} · 独立计算</p>
        </article>
        <article :class="{ warning: assessment?.consistency_status !== 'TEMPORALLY_CONSISTENT' }">
          <span>一致性检查</span>
          <h3>{{ assessment?.consistency_status ?? "等待分析" }}</h3>
          <p>窗口极值不等于有效交易拐点；只有已确认且时间一致的价格/DIF信号才可进入仓位条件。</p>
        </article>
      </section>

      <section class="batch-card">
        <header><p class="kicker">CONDITIONAL EXECUTION WINDOWS</p><h3>分批仓位建议</h3></header>
        <p v-if="!decision?.batches.length" class="empty">当前没有可执行批次，等待下一完整交易周复核。</p>
        <ol v-else>
          <li v-for="batch in decision?.batches" :key="batch.batch_number">
            <b>{{ String(batch.batch_number).padStart(2, "0") }}</b>
            <time>{{ batch.window_start_date }}—{{ batch.window_end_date }}</time>
            <strong>{{ batch.action === "BUY" ? "增加" : "减少" }} {{ batch.change_pp }} 个百分点</strong>
            <small>状态：{{ batch.initial_state }}；目标仓位：{{ batch.target_after_pp }}%</small>
          </li>
        </ol>
      </section>

      <section class="audit-grid">
        <div><span>模型可靠性</span><b>{{ number(result.model_reliability.score, 0) }}/100</b></div>
        <div><span>模型健康</span><b>{{ result.health_status }}</b></div>
        <div><span>情景适配器</span><b>{{ result.scenario_adapter_id }}</b></div>
        <div><span>策略一致性</span><b>{{ assessment?.consistency_status ?? "未分析" }}</b></div>
      </section>
    </template>
  </section>
</template>

<style scoped>
.v34-lab { display: grid; gap: 18px; }
.lab-head, .curve-toolbar, .forecast-card > header, .batch-card > header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }
.lab-head { padding: 26px; border: 1px solid #b9b09e; background: linear-gradient(135deg, #fffdf6, #edf6f4); box-shadow: 6px 6px 0 rgba(23,43,58,.08); }
.lab-head h2, .forecast-card h3, .batch-card h3 { margin: 4px 0 8px; color: #172b3a; }
.lab-head p { max-width: 820px; margin: 0; color: #5c6a72; line-height: 1.7; }
.kicker { margin: 0; color: #12687e; font-size: 12px; font-weight: 900; letter-spacing: .14em; }
.primary-actions { display: flex; gap: 10px; flex-shrink: 0; }
button { padding: 9px 14px; border: 1px solid #172b3a; background: #fffdf6; color: #172b3a; font-weight: 800; cursor: pointer; }
button.primary { background: #172b3a; color: #fffdf6; }
button:disabled { cursor: not-allowed; opacity: .5; }
.status-strip { display: flex; flex-wrap: wrap; gap: 8px; }
.status-strip span { padding: 9px 12px; border: 1px solid #d4cdc0; background: #fffdf6; color: #657279; }
.status-strip b { margin-right: 6px; color: #172b3a; }
.error { padding: 10px 14px; border-left: 4px solid #a23f32; background: #fff0ed; color: #8d3026; }
.curve-toolbar span { display: block; margin-top: 4px; color: #68767c; }
.range-tabs { display: flex; gap: 6px; }
.range-tabs button[aria-pressed="true"] { background: #12687e; color: white; }
.curve-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.curve-card, .forecast-card, .batch-card { border: 1px solid #c9c1b3; background: #fffdf6; }
.curve-card header { display: flex; justify-content: space-between; padding: 14px 16px 0; }
.curve-card header span { color: #12687e; font-weight: 800; }
.curve-card h3 { margin: 2px 0; }
.curve { height: 320px; }
.iteration-event-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; padding: 0 14px 14px; }
.iteration-event-strip span { padding: 7px 9px; border: 1px solid #d7d1c6; background: #f7f4ec; color: #4f626b; font-size: 12px; font-weight: 800; }
.iteration-event-strip small { display: block; margin-top: 3px; color: #79858a; font-weight: 500; }
.iteration-event-strip .pending-event { border-color: #b8a57e; }
.iteration-event-strip .candidate-event { border-color: #6f9aaa; }
.iteration-event-strip .promotion-event { border-color: #b9773d; }
.verdict { display: grid; grid-template-columns: 1.2fr .8fr; gap: 16px; }
.verdict article { padding: 22px; border: 1px solid #c9c1b3; background: #fffdf6; }
.probabilities { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.probabilities span { padding: 12px; background: #edf2f1; color: #617078; }
.probabilities b { display: block; color: #172b3a; font-size: 24px; }
.position { display: flex; align-items: center; gap: 14px; font-size: 28px; }
.position i { color: #b56616; font-style: normal; }
.forecast-card { padding: 20px; }
.forecast-card > header span { max-width: 440px; color: #66747b; text-align: right; }
.forecast-point-details { min-height: 150px; margin-top: 14px; padding: 14px 16px; border: 1px solid #cfc7b8; background: #f6f2e9; }
.forecast-point-heading { display: grid; grid-template-columns: auto minmax(220px, 1fr) auto; gap: 14px; align-items: center; padding-bottom: 10px; border-bottom: 1px solid #ddd5c7; }
.forecast-point-heading span { color: #12687e; font-size: 12px; font-weight: 900; letter-spacing: .1em; }
.forecast-point-heading time { color: #172b3a; font-size: 18px; font-weight: 850; }
.forecast-point-heading b { padding: 5px 8px; background: #172b3a; color: #fffdf6; font-size: 12px; }
.forecast-point-details dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px 16px; margin: 12px 0 0; }
.forecast-point-details dl > div { min-width: 0; }
.forecast-point-details dt { color: #6a777d; font-size: 11px; font-weight: 800; }
.forecast-point-details dd { margin: 3px 0 0; color: #172b3a; font-size: 14px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.forecast-chart { height: 760px; margin-top: 10px; }
.semantic-note { margin: 0; padding: 12px 14px; border-left: 4px solid #12687e; background: #edf6f4; color: #4f626b; line-height: 1.6; }
.turning-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.turning-grid article { padding: 18px; border: 1px solid #c9c1b3; background: #fffdf6; }
.turning-grid span { color: #65737a; font-weight: 800; }
.turning-grid h3 { margin: 8px 0; }
.turning-grid .warning { border-color: #d9553f; background: #fff3ee; }
.batch-card { padding: 20px; }
.batch-card ol { display: grid; gap: 8px; padding: 0; list-style: none; }
.batch-card li { display: grid; grid-template-columns: 42px 190px 1fr; gap: 12px; align-items: center; padding: 12px; border-top: 1px solid #e2ddd3; }
.batch-card li > b { color: #12687e; font-size: 20px; }
.batch-card li small { grid-column: 2 / -1; color: #66747b; }
.empty { color: #66747b; }
.audit-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.audit-grid div { padding: 14px; border: 1px solid #c9c1b3; background: #fffdf6; }
.audit-grid span { display: block; color: #68767c; }
.audit-grid b { display: block; margin-top: 5px; color: #172b3a; overflow-wrap: anywhere; }
@media (max-width: 980px) {
  .curve-grid, .verdict, .turning-grid { grid-template-columns: 1fr; }
  .lab-head, .curve-toolbar, .forecast-card > header { flex-direction: column; }
  .forecast-card > header span { text-align: left; }
  .audit-grid { grid-template-columns: repeat(2, 1fr); }
  .forecast-point-details dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .forecast-point-heading { grid-template-columns: 1fr; gap: 5px; }
}
</style>
