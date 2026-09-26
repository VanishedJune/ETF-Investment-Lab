<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";

import type {
  IndicatorRow,
  InstrumentCode,
  PriceRow,
} from "../types/research";
import { legendColors, normalizeSeries, volumeChangeBarData } from "../types/research";

type Normalized = ReturnType<typeof normalizeSeries>;
type Candle = [number, number, number, number];
type SparseNumber = (number | null)[];
// ECharts' candlestick data parser does not accept a literal `null` item
// (it tries to read `item.value` during series initialization).  The shared
// daily/weekly axis necessarily has dates where one resolution has no bar, so
// use ECharts' documented "-" empty-data marker for those slots instead.
type SparseCandle = (Candle | "-")[];

const props = defineProps<{
  instrumentCode: InstrumentCode;
  dailyPrices: PriceRow[];
  dailyIndicators: IndicatorRow[];
  weeklyPrices: PriceRow[];
  weeklyIndicators: IndicatorRow[];
  loading?: boolean;
  error?: string;
}>();

const chartElement = ref<HTMLDivElement>();
const chartState = ref<"empty" | "loading" | "error" | "rendered">("empty");
const WEEKLY_BAR_WIDTH = 18;
const daily = computed(() => normalizeSeries(props.dailyPrices, props.dailyIndicators));
const weekly = computed(() => normalizeSeries(props.weeklyPrices, props.weeklyIndicators));
const commonDates = computed(() => Array.from(new Set([
  ...daily.value.dates,
  ...weekly.value.dates,
])).sort());
const hoveredIndex = ref(-1);
const pointerUpdateCount = ref(0);
const legendSelection = reactive<Record<string, boolean>>({});

const dailyRowsByDate = computed(() => new Map(props.dailyPrices.map((row) => [row.date, row])));
const weeklyRowsByDate = computed(() => new Map(props.weeklyPrices.map((row) => [row.date, row])));
const dailyIndexByDate = computed(() => new Map(daily.value.dates.map((date, index) => [date, index])));
const weeklyIndexByDate = computed(() => new Map(weekly.value.dates.map((date, index) => [date, index])));

function finite(value: number | null | undefined): number | null {
  return value !== null && value !== undefined && Number.isFinite(value) ? value : null;
}

function sparseNumbers(series: Normalized, values: readonly number[]): SparseNumber {
  const byDate = new Map(series.dates.map((date, index) => [date, finite(values[index])]));
  return commonDates.value.map((date) => byDate.get(date) ?? null);
}

function sparseCandles(series: Normalized): SparseCandle {
  const byDate = new Map(series.dates.map((date, index) => [date, series.candles[index]]));
  return commonDates.value.map((date) => byDate.get(date) ?? "-");
}

const aligned = computed(() => ({
  dailyCandles: sparseCandles(daily.value),
  weeklyCandles: sparseCandles(weekly.value),
  dailyClose: sparseNumbers(daily.value, daily.value.closes),
  weeklyClose: sparseNumbers(weekly.value, weekly.value.closes),
  dailyVolume: sparseNumbers(daily.value, daily.value.volumes.map((value) => value ?? Number.NaN)),
  weeklyVolume: sparseNumbers(weekly.value, weekly.value.volumes.map((value) => value ?? Number.NaN)),
  dailyDif: sparseNumbers(daily.value, daily.value.dif),
  weeklyDif: sparseNumbers(weekly.value, weekly.value.dif),
  dailyDea: sparseNumbers(daily.value, daily.value.dea),
  weeklyDea: sparseNumbers(weekly.value, weekly.value.dea),
  dailyMacd: sparseNumbers(daily.value, daily.value.macd),
  weeklyMacd: sparseNumbers(weekly.value, weekly.value.macd),
  dailyDifFirstChange: sparseNumbers(daily.value, daily.value.difFirstChange),
  weeklyDifFirstChange: sparseNumbers(weekly.value, weekly.value.difFirstChange),
  dailyMa5: sparseNumbers(daily.value, daily.value.ma5),
  dailyMa10: sparseNumbers(daily.value, daily.value.ma10),
  dailyMa20: sparseNumbers(daily.value, daily.value.ma20),
  weeklyMa5: sparseNumbers(weekly.value, weekly.value.ma5),
  weeklyMa10: sparseNumbers(weekly.value, weekly.value.ma10),
  weeklyMa20: sparseNumbers(weekly.value, weekly.value.ma20),
}));
const weeklyCandleData = computed(() => aligned.value.weeklyCandles.flatMap(
  (candle, index) => (Array.isArray(candle) ? [[index, ...candle]] : []),
));

const priceExtents = computed(() => ({
  daily: commonDates.value.map((_, index) => {
    const candle = aligned.value.dailyCandles[index];
    return Array.isArray(candle)
      ? [candle[2], candle[3], aligned.value.dailyMa5[index], aligned.value.dailyMa10[index], aligned.value.dailyMa20[index]]
      : [];
  }),
  weekly: commonDates.value.map((_, index) => {
    const candle = aligned.value.weeklyCandles[index];
    return Array.isArray(candle)
      ? [candle[2], candle[3], aligned.value.weeklyMa5[index], aligned.value.weeklyMa10[index], aligned.value.weeklyMa20[index]]
      : [];
  }),
}));

type DetailPoint = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  ma5: number | null;
  ma10: number | null;
  ma20: number | null;
  volume: number | null;
  dif: number | null;
  dea: number | null;
  macd: number | null;
  difFirstChange: number | null;
  source: string | null;
} | null;

function nearestIndex(dates: readonly string[], target: string): number {
  if (!dates.length) return -1;
  const exact = dates.indexOf(target);
  if (exact >= 0) return exact;
  let candidate = 0;
  for (let index = 0; index < dates.length; index += 1) {
    if (dates[index] > target) break;
    candidate = index;
  }
  return candidate;
}

function detailPoint(series: Normalized, rows: Map<string, PriceRow>, targetDate: string): DetailPoint {
  const index = nearestIndex(series.dates, targetDate);
  if (index < 0) return null;
  const candle = series.candles[index];
  const date = series.dates[index];
  return {
    date,
    open: finite(candle[0]),
    high: finite(candle[3]),
    low: finite(candle[2]),
    close: finite(candle[1]),
    ma5: finite(series.ma5[index]),
    ma10: finite(series.ma10[index]),
    ma20: finite(series.ma20[index]),
    volume: finite(series.volumes[index]),
    dif: finite(series.dif[index]),
    dea: finite(series.dea[index]),
    macd: finite(series.macd[index]),
    difFirstChange: finite(series.difFirstChange[index]),
    source: rows.get(date)?.source ?? series.source,
  };
}

const selectedPoint = computed(() => {
  if (!commonDates.value.length) return null;
  const index = Math.max(0, Math.min(commonDates.value.length - 1, hoveredIndex.value));
  const date = commonDates.value[index];
  return {
    index,
    date,
    daily: detailPoint(daily.value, dailyRowsByDate.value, date),
    weekly: detailPoint(weekly.value, weeklyRowsByDate.value, date),
  };
});

let chart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;
let zoomStart = 0;
let zoomEnd = 100;
let scaleFrame: number | undefined;
let pointerFrame: number | undefined;
let pendingPointerIndex = -1;

function formatted(value: number | null | undefined, digits = 2): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(parsed);
}

function formattedVolume(value: number | null | undefined): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(parsed);
}

function defaultZoomStart(count: number): number {
  const visible = 260;
  return count <= visible ? 0 : Math.max(0, 100 - (visible / count) * 100);
}

function resetZoom() {
  zoomStart = defaultZoomStart(commonDates.value.length);
  zoomEnd = 100;
  chart?.dispatchAction({ type: "dataZoom", start: zoomStart, end: zoomEnd });
  scheduleDynamicAxes();
}

function currentZoom(): { start: number; end: number } {
  const option = chart?.getOption() as { dataZoom?: { start?: number; end?: number }[] } | undefined;
  return {
    start: Number(option?.dataZoom?.[0]?.start ?? zoomStart),
    end: Number(option?.dataZoom?.[0]?.end ?? zoomEnd),
  };
}

type AxisValue = number | null | readonly (number | null)[];

function axisRange(
  values: readonly AxisValue[],
  start: number,
  end: number,
  includeZero: boolean,
): { min: number; max: number } {
  const startIndex = Math.max(0, Math.floor(values.length * start / 100));
  const endIndex = Math.min(values.length - 1, Math.max(startIndex, Math.ceil(values.length * end / 100) - 1));
  const numbers: number[] = [];
  for (const value of values.slice(startIndex, endIndex + 1)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "number" && Number.isFinite(item)) numbers.push(item);
      }
    } else if (typeof value === "number" && Number.isFinite(value)) {
      numbers.push(value);
    }
  }
  if (!numbers.length) return { min: includeZero ? 0 : -1, max: includeZero ? 1 : 1 };
  let min = Math.min(...numbers);
  let max = Math.max(...numbers);
  if (includeZero) {
    min = Math.min(0, min);
    max = Math.max(0, max);
  }
  const span = max - min;
  const padding = span > 0 ? span * 0.08 : Math.max(Math.abs(max) * 0.08, 1e-6);
  return { min: min - padding, max: max + padding };
}

function applyDynamicAxes() {
  scaleFrame = undefined;
  if (!chart || !chartElement.value || !commonDates.value.length) return;
  const zoom = currentZoom();
  // Keep the two resolutions on independent Y scales.  A daily extreme must
  // not flatten the weekly price/MA curves (and vice versa) on the shared X
  // axis.  Each grid therefore receives its own visible-window range.
  const dailyPrice = axisRange(priceExtents.value.daily, zoom.start, zoom.end, false);
  const weeklyPrice = axisRange(priceExtents.value.weekly, zoom.start, zoom.end, false);
  const dailyMomentum = commonDates.value.map((_, index) => [
    aligned.value.dailyMacd[index],
    aligned.value.dailyDif[index],
    aligned.value.dailyDea[index],
  ]);
  const weeklyMomentum = commonDates.value.map((_, index) => [
    aligned.value.weeklyMacd[index],
    aligned.value.weeklyDif[index],
    aligned.value.weeklyDea[index],
  ]);
  const ranges = [
    dailyPrice,
    axisRange(aligned.value.dailyVolume, zoom.start, zoom.end, true),
    axisRange(dailyMomentum, zoom.start, zoom.end, true),
    axisRange(aligned.value.dailyDifFirstChange, zoom.start, zoom.end, true),
    weeklyPrice,
    axisRange(aligned.value.weeklyVolume, zoom.start, zoom.end, true),
    axisRange(weeklyMomentum, zoom.start, zoom.end, true),
    axisRange(aligned.value.weeklyDifFirstChange, zoom.start, zoom.end, true),
  ];
  chart.setOption({ yAxis: ranges.map((range) => ({ min: range.min, max: range.max })) }, { lazyUpdate: true });
  chartElement.value.dataset.priceYMin = String(dailyPrice.min);
  chartElement.value.dataset.priceYMax = String(dailyPrice.max);
  chartElement.value.dataset.dailyPriceYMin = String(dailyPrice.min);
  chartElement.value.dataset.dailyPriceYMax = String(dailyPrice.max);
  chartElement.value.dataset.weeklyPriceYMin = String(weeklyPrice.min);
  chartElement.value.dataset.weeklyPriceYMax = String(weeklyPrice.max);
  chartElement.value.dataset.visibleStartIndex = String(Math.floor(commonDates.value.length * zoom.start / 100));
  chartElement.value.dataset.visibleEndIndex = String(Math.min(commonDates.value.length - 1, Math.ceil(commonDates.value.length * zoom.end / 100) - 1));
}

function scheduleDynamicAxes() {
  if (scaleFrame !== undefined) cancelAnimationFrame(scaleFrame);
  scaleFrame = requestAnimationFrame(applyDynamicAxes);
}

function pointerIndex(event: unknown): number | null {
  const payload = event as { axesInfo?: { axisDim?: string; axisIndex?: number; value?: number | string }[] };
  const axis = payload.axesInfo?.find((item) => item.axisDim === "x" && item.axisIndex === 0)
    ?? payload.axesInfo?.find((item) => item.axisDim === "x");
  if (!axis) return null;
  const numeric = Number(axis.value);
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < commonDates.value.length) return numeric;
  const byDate = commonDates.value.indexOf(String(axis.value));
  return byDate >= 0 ? byDate : null;
}

function schedulePointer(index: number) {
  pendingPointerIndex = index;
  if (pointerFrame !== undefined) return;
  pointerFrame = requestAnimationFrame(() => {
    pointerFrame = undefined;
    if (pendingPointerIndex === hoveredIndex.value) return;
    hoveredIndex.value = pendingPointerIndex;
    pointerUpdateCount.value += 1;
  });
}

function lineSeries(
  name: string,
  data: SparseNumber,
  color: string,
  xAxisIndex: number,
  yAxisIndex: number,
  width = 1.6,
  connectNulls = false,
  zeroAxisLabel?: string,
) {
  return {
    name,
    type: "line",
    xAxisIndex,
    yAxisIndex,
    data,
    showSymbol: false,
    // Weekly points sit on Friday slots of the shared daily axis.  Connecting
    // across the intervening daily nulls is required to render a weekly line
    // as a curve instead of isolated, invisible points.
    connectNulls,
    sampling: "lttb",
    lineStyle: { color, width, type: name.includes('MA10') ? 'dashed' : name.includes('MA20') ? 'dotted' : 'solid' },
    itemStyle: { color },
    emphasis: { focus: "series" },
    ...(zeroAxisLabel ? { markLine: zeroMarkLine(color, zeroAxisLabel) } : {}),
  };
}

function zeroMarkLine(color: string, label: string) {
  return {
    silent: true,
    symbol: ["none", "none"],
    animation: false,
    lineStyle: { color, width: 2.4, type: "solid", opacity: 0.96 },
    label: { show: false, formatter: label },
    data: [{ yAxis: 0 }],
  };
}

function volumeBarSeries(name: string, data: SparseNumber, xAxisIndex: number, yAxisIndex: number, width = 7) {
  return {
    name,
    type: "bar",
    xAxisIndex,
    yAxisIndex,
    data: volumeChangeBarData(data, 0.78),
    // Fixed pixels keep sparse weekly bars as wide as the volume bars on the
    // same shared daily category axis.
    barWidth: width,
    barMaxWidth: 22,
    barMinWidth: width,
  };
}

function macdSeries(name: string, data: SparseNumber, xAxisIndex: number, yAxisIndex: number, width = 6) {
  return {
    name,
    type: "bar",
    xAxisIndex,
    yAxisIndex,
    data,
    barWidth: width,
    barMaxWidth: 18,
    barMinWidth: width,
    itemStyle: {
      color: (item: { value: number | null }) => (
        Number(item.value) >= 0 ? legendColors.macdPositive : legendColors.macdNegative
      ),
      opacity: 0.82,
    },
    markLine: zeroMarkLine("#294f63", "MACD 0轴"),
  };
}

function weeklyCandleSeries() {
  return {
    name: "周K价格",
    type: "custom",
    xAxisIndex: 4,
    yAxisIndex: 4,
    dimensions: ["dateIndex", "open", "close", "low", "high"],
    encode: { x: 0, y: [1, 2, 3, 4] },
    data: weeklyCandleData.value,
    z: 4,
    renderItem: (_params: unknown, api: any) => {
      const index = Number(api.value(0));
      const open = Number(api.value(1));
      const close = Number(api.value(2));
      const low = Number(api.value(3));
      const high = Number(api.value(4));
      const openPoint = api.coord([index, open]);
      const closePoint = api.coord([index, close]);
      const lowPoint = api.coord([index, low]);
      const highPoint = api.coord([index, high]);
      const rising = close >= open;
      const color = rising ? legendColors.priceUp : legendColors.priceDown;
      const bodyHeight = Math.max(4, Math.abs(closePoint[1] - openPoint[1]));
      const bodyTop = ((closePoint[1] + openPoint[1]) / 2) - (bodyHeight / 2);
      return {
        type: "group",
        children: [
          {
            type: "line",
            shape: { x1: openPoint[0], y1: highPoint[1], x2: openPoint[0], y2: lowPoint[1] },
            style: { stroke: color, lineWidth: 2 },
          },
          {
            type: "rect",
            shape: {
              x: openPoint[0] - (WEEKLY_BAR_WIDTH / 2),
              y: bodyTop,
              width: WEEKLY_BAR_WIDTH,
              height: bodyHeight,
            },
            style: { fill: color, stroke: color, lineWidth: 1.5 },
          },
        ],
      };
    },
  };
}

function render() {
  if (!chartElement.value) return;
  chart ??= echarts.init(chartElement.value, undefined, { renderer: "canvas", useDirtyRect: true });
  if (props.loading) {
    chart.clear();
    chart.showLoading("default", {
      text: "正在加载日 K / 周 K 共享时间轴…",
      color: legendColors.dif,
      textColor: "#3e5260",
      maskColor: "rgba(244, 246, 248, 0.86)",
    });
    chartState.value = "loading";
    return;
  }
  chart.hideLoading();
  if (props.error || !commonDates.value.length) {
    chart.clear();
    chartState.value = props.error ? "error" : "empty";
    return;
  }
  if (hoveredIndex.value < 0 || hoveredIndex.value >= commonDates.value.length) {
    hoveredIndex.value = commonDates.value.length - 1;
  }
  if (zoomEnd === 100 && zoomStart === 0) zoomStart = defaultZoomStart(commonDates.value.length);

  const xAxes = Array.from({ length: 8 }, (_, gridIndex) => ({
    type: "category",
    gridIndex,
    data: commonDates.value,
    boundaryGap: true,
    axisLine: { show: gridIndex === 7, lineStyle: { color: "#d5dce5" } },
    axisTick: { show: gridIndex === 7 },
    axisLabel: {
      show: gridIndex === 7,
      color: "#657184",
      hideOverlap: true,
      fontSize: 12,
      margin: 9,
      formatter: (value: string) => value.slice(0, 7),
    },
    axisPointer: { label: { show: gridIndex === 7, backgroundColor: "#223b4a" } },
    splitLine: { show: false },
  }));
  const grid = [
    { left: 72, right: 72, top: "2%", height: "23%" },
    { left: 72, right: 72, top: "27%", height: "6%" },
    { left: 72, right: 72, top: "34%", height: "7%" },
    { left: 72, right: 72, top: "42%", height: "6%" },
    { left: 72, right: 72, top: "50%", height: "23%" },
    { left: 72, right: 72, top: "74%", height: "6%" },
    { left: 72, right: 72, top: "81%", height: "7%" },
    { left: 72, right: 72, top: "88%", height: "6%" },
  ];
  const yAxes = [
    { scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 11 }, splitLine: { lineStyle: { color: "#e3e8ee" } } },
    { gridIndex: 1, scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 10, formatter: compactAxis }, splitLine: { show: false } },
    { gridIndex: 2, scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 10 }, splitLine: { lineStyle: { color: "#e3e8ee" } } },
    { gridIndex: 3, scale: true, position: "right", axisLabel: { color: legendColors.difFirstChange, fontSize: 10 }, splitLine: { lineStyle: { color: "#ece5f2", type: "dashed" } } },
    { gridIndex: 4, scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 11 }, splitLine: { lineStyle: { color: "#e3e8ee" } } },
    { gridIndex: 5, scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 10, formatter: compactAxis }, splitLine: { show: false } },
    { gridIndex: 6, scale: true, position: "right", axisLabel: { color: "#657184", fontSize: 10 }, splitLine: { lineStyle: { color: "#e3e8ee" } } },
    { gridIndex: 7, scale: true, position: "right", axisLabel: { color: legendColors.difFirstChange, fontSize: 10 }, splitLine: { lineStyle: { color: "#ece5f2", type: "dashed" } } },
  ];
  chart.setOption({
    animation: false,
    backgroundColor: "transparent",
    axisPointer: { link: [{ xAxisIndex: [0, 1, 2, 3, 4, 5, 6, 7] }], label: { backgroundColor: "#223b4a" } },
    tooltip: { trigger: "axis", triggerOn: "mousemove|click", showContent: false, transitionDuration: 0, axisPointer: { type: "line", snap: true } },
    legend: { show: false },
    grid,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2, 3, 4, 5, 6, 7], start: zoomStart, end: zoomEnd, filterMode: "filter", throttle: 40 },
      { type: "slider", xAxisIndex: [0, 1, 2, 3, 4, 5, 6, 7], start: zoomStart, end: zoomEnd, bottom: 0, height: 18, borderColor: "#d5dce5", backgroundColor: "#e3e8ee", fillerColor: "rgba(102, 132, 167, .12)", handleStyle: { color: "#52789c", borderColor: "#52789c" }, textStyle: { color: "#657184" }, filterMode: "filter", throttle: 40 },
    ],
    series: [
      { name: "日K价格", type: "candlestick", xAxisIndex: 0, yAxisIndex: 0, data: aligned.value.dailyCandles, barWidth: 7, barMaxWidth: 24, barMinWidth: 7, itemStyle: { color: legendColors.priceUp, color0: legendColors.priceDown, borderColor: legendColors.priceUp, borderColor0: legendColors.priceDown, borderWidth: 1.1 } },
      lineSeries("日K MA5", aligned.value.dailyMa5, legendColors.ma5, 0, 0, 1.7),
      lineSeries("日K MA10", aligned.value.dailyMa10, legendColors.ma10, 0, 0, 1.7),
      lineSeries("日K MA20", aligned.value.dailyMa20, legendColors.ma20, 0, 0, 2),
      volumeBarSeries("日K成交量", aligned.value.dailyVolume, 1, 1),
      macdSeries("日K MACD柱", aligned.value.dailyMacd, 2, 2),
      lineSeries("日K DIF", aligned.value.dailyDif, legendColors.dif, 2, 2, 1.6),
      lineSeries("日K DEA", aligned.value.dailyDea, legendColors.dea, 2, 2, 1.6),
      lineSeries("日K DIF一阶变化", aligned.value.dailyDifFirstChange, legendColors.difFirstChange, 3, 3, 1.45, false, "DIF变化 0轴"),
      weeklyCandleSeries(),
      lineSeries("周K MA5", aligned.value.weeklyMa5, legendColors.ma5, 4, 4, 2, true),
      lineSeries("周K MA10", aligned.value.weeklyMa10, legendColors.ma10, 4, 4, 2, true),
      lineSeries("周K MA20", aligned.value.weeklyMa20, legendColors.ma20, 4, 4, 2.3, true),
      volumeBarSeries("周K成交量", aligned.value.weeklyVolume, 5, 5, WEEKLY_BAR_WIDTH),
      macdSeries("周K MACD柱", aligned.value.weeklyMacd, 6, 6, 14),
      lineSeries("周K DIF", aligned.value.weeklyDif, legendColors.dif, 6, 6, 1.6, true),
      lineSeries("周K DEA", aligned.value.weeklyDea, legendColors.dea, 6, 6, 1.6, true),
      lineSeries("周K DIF一阶变化", aligned.value.weeklyDifFirstChange, legendColors.difFirstChange, 7, 7, 1.45, true, "DIF变化 0轴"),
    ],
  }, true);
  chart.off("dataZoom");
  chart.on("dataZoom", (event: unknown) => {
    const payload = event as { start?: number; end?: number; batch?: { start: number; end: number }[] };
    const value = payload.batch?.[0] ?? payload;
    if (typeof value.start === "number") zoomStart = value.start;
    if (typeof value.end === "number") zoomEnd = value.end;
    scheduleDynamicAxes();
  });
  chart.off("updateAxisPointer");
  chart.on("updateAxisPointer", (event: unknown) => {
    const index = pointerIndex(event);
    if (index !== null) schedulePointer(index);
  });
  scheduleDynamicAxes();
  chartState.value = "rendered";
}

function compactAxis(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1e12) return `${(value / 1e12).toFixed(1)}万亿`;
  if (absolute >= 1e8) return `${(value / 1e8).toFixed(1)}亿`;
  if (absolute >= 1e4) return `${(value / 1e4).toFixed(1)}万`;
  return String(Math.round(value));
}

function toggleSeries(name: string) {
  const next = legendSelection[name] !== false;
  legendSelection[name] = !next;
  chart?.dispatchAction({ type: "legendToggleSelect", name });
}

function movingAverageSelected(period: "MA5" | "MA10" | "MA20"): boolean {
  return legendSelection[`日K ${period}`] !== false
    && legendSelection[`周K ${period}`] !== false;
}

function toggleMovingAverage(period: "MA5" | "MA10" | "MA20") {
  const selected = !movingAverageSelected(period);
  for (const name of [`日K ${period}`, `周K ${period}`]) {
    legendSelection[name] = selected;
    chart?.dispatchAction({ type: selected ? "legendSelect" : "legendUnSelect", name });
  }
}

watch(
  () => [props.instrumentCode, props.dailyPrices, props.dailyIndicators, props.weeklyPrices, props.weeklyIndicators, props.loading, props.error],
  () => void nextTick(render),
);

onMounted(() => {
  render();
  if (chartElement.value) {
    resizeObserver = new ResizeObserver(() => chart?.resize());
    resizeObserver.observe(chartElement.value);
  }
});

onBeforeUnmount(() => {
  if (scaleFrame !== undefined) cancelAnimationFrame(scaleFrame);
  if (pointerFrame !== undefined) cancelAnimationFrame(pointerFrame);
  resizeObserver?.disconnect();
  chart?.dispose();
  chart = undefined;
});
</script>

<template>
  <div
    class="combined-market-chart"
    data-testid="unified-market-chart"
    data-chart-mode="daily-weekly-shared"
    :data-chart-state="chartState"
    :data-weekly-candle-width="WEEKLY_BAR_WIDTH"
    :data-weekly-volume-width="WEEKLY_BAR_WIDTH"
  >
    <div class="chart-workspace">
      <div class="chart-stage">
        <div class="chart-rail" aria-label="日 K 与周 K 共享时间轴图例">
          <span class="rail-caption">共享横轴</span>
          <span data-testid="timeframe-daily" aria-pressed="true" class="resolution-label">日 K</span>
          <span data-testid="timeframe-weekly" aria-pressed="true" class="resolution-label">周 K</span>
          <b class="ma-label ma5-label" data-testid="ma5-label" :class="{ selected: movingAverageSelected('MA5') }" role="button" tabindex="0" @click="toggleMovingAverage('MA5')" @keydown.enter="toggleMovingAverage('MA5')">MA5</b>
          <b class="ma-label ma10-label" data-testid="ma10-label" :class="{ selected: movingAverageSelected('MA10') }" role="button" tabindex="0" @click="toggleMovingAverage('MA10')" @keydown.enter="toggleMovingAverage('MA10')">MA10</b>
          <b class="ma-label ma20-label" data-testid="ma20-label" :class="{ selected: movingAverageSelected('MA20') }" role="button" tabindex="0" @click="toggleMovingAverage('MA20')" @keydown.enter="toggleMovingAverage('MA20')">MA20</b>
          <span class="series-summary">价格 · 成交量（量增红 / 量减绿） · MACD · DIF · DEA · DIF 一阶变化</span>
          <b class="macd-positive">MACD柱（正）</b>
          <b class="macd-negative">MACD柱（负）</b>
          <b class="dif-label" data-testid="dif-label" :class="{ selected: legendSelection['日K DIF'] !== false }" role="button" tabindex="0" @click="toggleSeries('日K DIF')" @keydown.enter="toggleSeries('日K DIF')">DIF</b>
          <b class="dea-label" data-testid="dea-label" :class="{ selected: legendSelection['日K DEA'] !== false }" role="button" tabindex="0" @click="toggleSeries('日K DEA')" @keydown.enter="toggleSeries('日K DEA')">DEA</b>
          <b class="derivative-label" data-testid="dif-first-change-label" :class="{ selected: legendSelection['日K DIF一阶变化'] !== false }" role="button" tabindex="0" @click="toggleSeries('日K DIF一阶变化')" @keydown.enter="toggleSeries('日K DIF一阶变化')">DIF 一阶变化</b>
          <button type="button" class="chart-reset" @click="resetZoom">重置共享缩放</button>
        </div>
        <p class="chart-series-note">日 K 与周 K 共用日期轴；周线按周末日期落点，指标曲线连续连接，底部时间条同步缩放。</p>
        <div class="chart-canvas-wrap">
          <div class="chart-region-label daily-region-label">日 K</div>
          <div class="chart-region-label weekly-region-label">周 K</div>
          <div ref="chartElement" class="unified-chart-canvas" role="img" aria-label="日 K 与周 K 共用横轴的价格、成交量、MACD、DIF、DEA和DIF一阶变化联动图"></div>
        </div>
        <div v-if="error" class="chart-status error">{{ error }}</div>
        <div v-else-if="!loading && !commonDates.length" class="chart-status">当前没有可对齐的日线与周线数据。</div>
      </div>

      <aside class="market-point-details" data-testid="market-point-details" :data-hover-index="selectedPoint?.index ?? -1" :data-update-count="pointerUpdateCount" aria-live="polite">
        <header>
          <div><span class="details-kicker">CURSOR DATA</span><h3>共享时间点</h3></div>
          <b>日 K + 周 K</b>
        </header>
        <template v-if="selectedPoint">
          <time>{{ selectedPoint.date }}</time>
          <section v-for="item in [{ label: '日 K', point: selectedPoint.daily }, { label: '周 K', point: selectedPoint.weekly }]" :key="item.label" class="details-resolution">
            <h4>{{ item.label }} <small>{{ item.point?.date ?? '无对应数据' }}</small></h4>
            <dl v-if="item.point">
              <div><dt>收盘</dt><dd>{{ formatted(item.point.close, 3) }}</dd></div>
              <div><dt class="ma5-detail">MA5</dt><dd>{{ formatted(item.point.ma5, 3) }}</dd></div>
              <div><dt class="ma10-detail">MA10</dt><dd>{{ formatted(item.point.ma10, 3) }}</dd></div>
              <div><dt class="ma20-detail">MA20</dt><dd>{{ formatted(item.point.ma20, 3) }}</dd></div>
              <div><dt>成交量</dt><dd>{{ formattedVolume(item.point.volume) }}</dd></div>
              <div><dt class="dif-label">DIF</dt><dd>{{ formatted(item.point.dif, 4) }}</dd></div>
              <div><dt class="dea-label">DEA</dt><dd>{{ formatted(item.point.dea, 4) }}</dd></div>
              <div><dt>MACD 柱</dt><dd>{{ formatted(item.point.macd, 4) }}</dd></div>
              <div><dt class="derivative-label">DIF 一阶变化</dt><dd>{{ formatted(item.point.difFirstChange, 4) }}</dd></div>
            </dl>
            <p v-else class="details-empty">该共享日期没有对应的{{ item.label }}记录。</p>
          </section>
          <footer><span>数据来源</span><b>{{ selectedPoint.daily?.source ?? selectedPoint.weekly?.source ?? '—' }}</b></footer>
        </template>
        <p v-else class="details-empty">等待行情数据。</p>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.combined-market-chart { min-height: 0; }
.chart-workspace { display: grid; grid-template-columns: minmax(0, 1fr) 318px; gap: 18px; align-items: start; }
.chart-stage { position: relative; min-width: 0; height: clamp(780px, calc(100vh - 96px), 920px); display: flex; flex-direction: column; }
.chart-canvas-wrap { position: relative; width: 100%; flex: 1 1 auto; min-height: 0; }
.unified-chart-canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.chart-region-label { position: absolute; z-index: 2; left: 82px; padding: 4px 9px; border-left: 3px solid #52789c; background: rgba(244, 246, 248, .92); color: #263445; font-size: 13px; font-weight: 900; letter-spacing: .08em; pointer-events: none; }
.daily-region-label { top: 2.2%; }
.weekly-region-label { top: 50.2%; }
.chart-rail { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 12px; max-width: calc(100% - 36px); padding: 7px 10px; border: 1px solid rgba(220,226,232,.9); background: rgba(244, 246, 248, .9); box-shadow: 3px 3px 0 rgba(38, 52, 69, .04); font-size: 12px; margin-bottom: 8px; }
.chart-rail b { cursor: pointer; user-select: none; opacity: .55; }
.chart-rail b.selected { opacity: 1; text-decoration: underline; text-underline-offset: 3px; }
.chart-rail > span:not(.rail-caption) { font-weight: 800; white-space: nowrap; }
.rail-caption { color: #657184; font-weight: 900; letter-spacing: .1em; }
.resolution-label { color: #263445; font-weight: 900; }
.series-summary { color: #657184; font-weight: 700; }
.ma-label { display: inline-flex; align-items: center; gap: 5px; }
.ma-label::before { width: 20px; height: 3px; background: currentColor; content: ""; box-shadow: 0 1px 0 rgba(23,43,58,.16); }
.ma5-label, .ma5-detail { color: #ae794b; }
.ma10-label, .ma10-detail { color: #52789c; }
.ma20-label, .ma20-detail { color: #97718f; }
.shared-resolution-label { font-weight: 900; }
.macd-positive { color: #d64b4b; }
.macd-negative { color: #27845a; }
.dif-label { color: #52789c; }
.dea-label { color: #a57738; }
.derivative-label { color: #807096; }
.chart-reset { margin-left: auto; padding: 5px 9px; border: 1px solid #8f887a; color: #263445; background: transparent; font-size: 12px; font-weight: 800; }
.chart-reset:hover, .chart-reset:focus-visible { color: #ffffff; background: #263445; outline: none; }
.chart-series-note { margin: 0 0 4px; color: #657184; font-size: 12px; }
.chart-status { position: absolute; inset: 70px 24px 48px; display: grid; place-items: center; color: #657184; background: rgba(244, 246, 248, .84); font-size: 15px; }
.chart-status.error { color: #a23f32; }
.market-point-details { position: sticky; top: 18px; height: clamp(780px, calc(100vh - 96px), 920px); min-height: 0; overflow-y: auto; padding: 18px; border: 1px solid #d5dce5; background: #e9edf2; box-shadow: 5px 5px 0 rgba(38, 52, 69, .04); color: #263445; }
.market-point-details > header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding-bottom: 14px; border-bottom: 2px solid #263445; }
.market-point-details h3, .market-point-details h4, .market-point-details p { margin: 0; }
.market-point-details h3 { margin-top: 4px; font-size: 20px; }
.market-point-details > header > b { padding: 5px 8px; background: #263445; color: #ffffff; font-size: 11px; }
.details-kicker { color: #52789c; font-size: 10px; font-weight: 900; letter-spacing: .14em; }
.market-point-details > time { display: block; padding: 18px 0 10px; font-size: 22px; font-weight: 850; }
.details-resolution { margin-top: 12px; padding-top: 12px; border-top: 1px solid #d5dce5; }
.details-resolution h4 { margin-bottom: 6px; color: #263445; font-size: 14px; }
.details-resolution h4 small { color: #657184; font-size: 10px; font-weight: 500; }
.details-resolution dl { margin: 0; }
.details-resolution dl > div { display: flex; justify-content: space-between; gap: 10px; padding: 4px 0; }
.details-resolution dt { color: #657184; font-size: 12px; }
.details-resolution dd { margin: 0; font-variant-numeric: tabular-nums; font-size: 12px; font-weight: 800; text-align: right; }
.details-resolution .dif-label { color: #52789c; }
.details-resolution .dea-label { color: #a57738; }
.details-resolution .derivative-label { color: #807096; }
.details-empty { padding-top: 12px; color: #657184; font-size: 12px; }
.market-point-details footer { display: grid; gap: 4px; margin-top: 16px; padding-top: 12px; border-top: 1px solid #d5dce5; color: #657184; font-size: 11px; }
.market-point-details footer b { color: #263445; overflow-wrap: anywhere; }
@media (max-width: 1180px) {
  .chart-workspace { grid-template-columns: 1fr; }
  .market-point-details { position: static; height: auto; min-height: 0; overflow: visible; }
}
@media (max-width: 760px) {
  .combined-market-chart, .chart-stage { min-height: 1180px; height: 1180px; }
  .chart-rail { gap: 6px 9px; }
}
</style>
