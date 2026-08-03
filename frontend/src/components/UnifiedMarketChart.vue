<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type {
  IndicatorRow,
  InstrumentCode,
  MarketTimeframe,
  PriceRow,
} from "../types/research";
import { legendColors, normalizeSeries } from "../types/research";
import { calculateDynamicYAxis } from "../utils/chartScale";

const props = defineProps<{
  instrumentCode: InstrumentCode;
  timeframe: MarketTimeframe;
  prices: PriceRow[];
  indicators: IndicatorRow[];
  loading?: boolean;
  error?: string;
}>();

const chartElement = ref<HTMLDivElement>();
const chartState = ref<"empty" | "loading" | "error" | "rendered">("empty");
const series = computed(() => normalizeSeries(props.prices, props.indicators));
const hoveredIndex = ref(-1);
const pointerUpdateCount = ref(0);
const priceRowsByDate = computed(() => new Map(props.prices.map((row) => [row.date, row])));
const selectedPoint = computed(() => {
  const values = series.value;
  if (!values.dates.length) return null;
  const index = Math.max(0, Math.min(values.dates.length - 1, hoveredIndex.value));
  const candle = values.candles[index];
  const date = values.dates[index];
  return {
    index,
    date,
    open: candle[0],
    close: candle[1],
    low: candle[2],
    high: candle[3],
    ma5: values.ma5[index],
    ma10: values.ma10[index],
    ma20: values.ma20[index],
    volume: values.volumes[index],
    dif: values.dif[index],
    dea: values.dea[index],
    macd: values.macd[index],
    difFirstChange: values.difFirstChange[index],
    source: priceRowsByDate.value.get(date)?.source ?? values.source,
  };
});
let chart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;
let zoomStart = 0;
let zoomEnd = 100;
let renderedTimeframe: MarketTimeframe | undefined;
let scaleFrame: number | undefined;
let pointerFrame: number | undefined;
let pendingPointerIndex = -1;
let legendSelection: Record<string, boolean> = {};

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

function pointerIndex(event: unknown): number | null {
  const payload = event as {
    axesInfo?: { axisDim?: string; axisIndex?: number; value?: number | string }[];
  };
  const axis = payload.axesInfo?.find((item) => item.axisDim === "x" && item.axisIndex === 0)
    ?? payload.axesInfo?.find((item) => item.axisDim === "x");
  if (!axis) return null;
  const numeric = Number(axis.value);
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < series.value.dates.length) {
    return numeric;
  }
  const byDate = series.value.dates.indexOf(String(axis.value));
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

function defaultZoomStart(timeframe: MarketTimeframe, count: number): number {
  const visible = timeframe === "daily" ? 260 : timeframe === "weekly" ? 260 : 180;
  return count <= visible ? 0 : Math.max(0, 100 - (visible / count) * 100);
}

function resetZoom() {
  zoomStart = defaultZoomStart(props.timeframe, series.value.dates.length);
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

function applyDynamicAxes() {
  scaleFrame = undefined;
  if (!chart || !chartElement.value || !series.value.dates.length) return;
  const zoom = currentZoom();
  const ranges = calculateDynamicYAxis(series.value, zoom.start, zoom.end, legendSelection);
  chart.setOption({
    yAxis: [
      { min: ranges.price.min, max: ranges.price.max },
      { min: ranges.volume.min, max: ranges.volume.max },
      { min: ranges.macd.min, max: ranges.macd.max },
      { min: ranges.difFirstChange.min, max: ranges.difFirstChange.max },
    ],
  }, { lazyUpdate: true });
  chartElement.value.dataset.priceYMin = String(ranges.price.min);
  chartElement.value.dataset.priceYMax = String(ranges.price.max);
  chartElement.value.dataset.volumeYMin = String(ranges.volume.min);
  chartElement.value.dataset.volumeYMax = String(ranges.volume.max);
  chartElement.value.dataset.macdYMin = String(ranges.macd.min);
  chartElement.value.dataset.macdYMax = String(ranges.macd.max);
  chartElement.value.dataset.difFirstChangeYMin = String(ranges.difFirstChange.min);
  chartElement.value.dataset.difFirstChangeYMax = String(ranges.difFirstChange.max);
  chartElement.value.dataset.visibleStartIndex = String(ranges.startIndex);
  chartElement.value.dataset.visibleEndIndex = String(ranges.endIndex);
}

function scheduleDynamicAxes() {
  if (scaleFrame !== undefined) cancelAnimationFrame(scaleFrame);
  scaleFrame = requestAnimationFrame(applyDynamicAxes);
}

function render() {
  if (!chartElement.value) return;
  chart ??= echarts.init(chartElement.value, undefined, {
    renderer: "canvas",
    useDirtyRect: true,
  });
  if (props.loading) {
    chart.clear();
    chart.showLoading("default", {
      text: "正在校准价格、量能与周线动能…",
      color: legendColors.dif,
      textColor: "#3e5260",
      maskColor: "rgba(255, 253, 246, 0.86)",
    });
    chartState.value = "loading";
    return;
  }
  chart.hideLoading();
  if (props.error || series.value.dates.length === 0) {
    chart.clear();
    chartState.value = props.error ? "error" : "empty";
    return;
  }

  if (renderedTimeframe !== props.timeframe) {
    zoomStart = defaultZoomStart(props.timeframe, series.value.dates.length);
    zoomEnd = 100;
    renderedTimeframe = props.timeframe;
    hoveredIndex.value = Math.max(0, series.value.dates.length - 1);
  }

  const volumeSeries = series.value.volumeAvailable
    ? [{
        name: "成交量",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: series.value.volumes,
        barMaxWidth: 8,
        itemStyle: { color: legendColors.volume, opacity: 0.72 },
      }]
    : [];

  chart.setOption({
    animation: false,
    backgroundColor: "transparent",
    axisPointer: {
      link: [{ xAxisIndex: [0, 1, 2, 3] }],
      label: { backgroundColor: "#223b4a" },
    },
    tooltip: {
      trigger: "axis",
      triggerOn: "mousemove|click",
      showContent: false,
      transitionDuration: 0,
      axisPointer: { type: "line", snap: true },
    },
    legend: {
      top: 2,
      right: 72,
      selectedMode: true,
      selected: legendSelection,
      itemWidth: 18,
      itemHeight: 8,
      data: ["MA5", "MA10", "MA20", "成交量", "MACD柱", "DIF", "DEA", "DIF一阶变化"],
      formatter: (name: string) => {
        if (name === "DIF") return "{dif|DIF}";
        if (name === "DEA") return "{dea|DEA}";
        if (name === "MACD柱") return "{macd|MACD柱}";
        if (name === "DIF一阶变化") return "{derivative|DIF一阶变化}";
        return name;
      },
      textStyle: {
        color: "#5f6c72",
        fontSize: 12,
        rich: {
          dif: { color: legendColors.dif, fontWeight: 800 },
          dea: { color: legendColors.dea, fontWeight: 800 },
          macd: { color: legendColors.macdPositive, fontWeight: 800 },
          derivative: { color: legendColors.difFirstChange, fontWeight: 800 },
        },
      },
    },
    grid: [
      { left: 72, right: 72, top: 46, height: "39%" },
      { left: 72, right: 72, top: "47%", height: "11%" },
      { left: 72, right: 72, top: "61%", height: "14%" },
      { left: 72, right: 72, top: "78%", height: "11%" },
    ],
    xAxis: [0, 1, 2, 3].map((gridIndex) => ({
      type: "category",
      gridIndex,
      data: series.value.dates,
      boundaryGap: true,
      axisLine: { lineStyle: { color: "#b9b09e" } },
      axisTick: { show: false },
      axisLabel: {
        show: gridIndex === 3,
        color: "#66737a",
        hideOverlap: true,
        fontSize: 12,
      },
      axisPointer: {
        label: { show: gridIndex === 3, backgroundColor: "#223b4a" },
      },
      splitLine: { show: false },
    })),
    yAxis: [
      {
        scale: true,
        position: "right",
        axisLabel: { color: "#66737a", fontSize: 12 },
        splitLine: { lineStyle: { color: "#e6dfd2" } },
      },
      {
        gridIndex: 1,
        scale: true,
        position: "right",
        splitNumber: 2,
        axisLabel: {
          color: "#66737a",
          fontSize: 11,
          formatter: (value: number) => compactAxis(value),
        },
        splitLine: { show: false },
      },
      {
        gridIndex: 2,
        scale: true,
        position: "right",
        axisLabel: { color: "#66737a", fontSize: 12 },
        splitLine: { lineStyle: { color: "#e6dfd2" } },
      },
      {
        gridIndex: 3,
        scale: true,
        position: "right",
        axisLabel: { color: legendColors.difFirstChange, fontSize: 11 },
        splitLine: { lineStyle: { color: "#ece5f2", type: "dashed" } },
      },
    ],
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: [0, 1, 2, 3],
        start: zoomStart,
        end: zoomEnd,
        filterMode: "none",
      },
      {
        type: "slider",
        xAxisIndex: [0, 1, 2, 3],
        start: zoomStart,
        end: zoomEnd,
        bottom: 6,
        height: 22,
        borderColor: "#b9b09e",
        backgroundColor: "#eee8dc",
        fillerColor: "rgba(18, 104, 126, .15)",
        handleStyle: { color: "#12687e", borderColor: "#12687e" },
        textStyle: { color: "#66737a" },
        filterMode: "none",
      },
    ],
    graphic: series.value.volumeAvailable
      ? []
      : [{
          type: "text",
          left: 82,
          top: "60%",
          silent: true,
          style: {
            text: props.instrumentCode === "NDX"
              ? "纳斯达克100直接指数无可靠成交量；本研究不使用ETF成交量替代"
              : "当前周期成交量不可用",
            fill: "#7a6f61",
            font: "600 13px IBM Plex Sans, Microsoft YaHei, sans-serif",
          },
        }],
    series: [
      {
        name: "标的K线",
        type: "candlestick",
        data: series.value.candles,
        itemStyle: {
          color: legendColors.priceUp,
          color0: legendColors.priceDown,
          borderColor: legendColors.priceUp,
          borderColor0: legendColors.priceDown,
        },
      },
      lineSeries("MA5", series.value.ma5, legendColors.ma5, 0, 0, 1.15),
      lineSeries("MA10", series.value.ma10, legendColors.ma10, 0, 0, 1.15),
      lineSeries("MA20", series.value.ma20, legendColors.ma20, 0, 0, 1.35),
      ...volumeSeries,
      {
        name: "MACD柱",
        type: "bar",
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: series.value.macd,
        barMaxWidth: 7,
        itemStyle: {
          color: (item: { value: number }) => (
            item.value >= 0 ? legendColors.macdPositive : legendColors.macdNegative
          ),
          opacity: 0.82,
        },
      },
      lineSeries("DIF", series.value.dif, legendColors.dif, 2, 2, 1.9),
      lineSeries("DEA", series.value.dea, legendColors.dea, 2, 2, 1.9),
      lineSeries(
        "DIF一阶变化",
        series.value.difFirstChange,
        legendColors.difFirstChange,
        3,
        3,
        1.8,
      ),
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
  chart.off("legendselectchanged");
  chart.on("legendselectchanged", (event: unknown) => {
    const payload = event as { selected?: Record<string, boolean> };
    legendSelection = { ...(payload.selected ?? {}) };
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

function lineSeries(
  name: string,
  data: number[],
  color: string,
  xAxisIndex: number,
  yAxisIndex: number,
  width: number,
) {
  return {
    name,
    type: "line",
    xAxisIndex,
    yAxisIndex,
    data,
    showSymbol: false,
    connectNulls: false,
    sampling: "lttb",
    lineStyle: { color, width },
    itemStyle: { color },
    emphasis: { focus: "series" },
  };
}

function compactAxis(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1e12) return `${(value / 1e12).toFixed(1)}万亿`;
  if (absolute >= 1e8) return `${(value / 1e8).toFixed(1)}亿`;
  if (absolute >= 1e4) return `${(value / 1e4).toFixed(1)}万`;
  return String(Math.round(value));
}

watch(
  () => [
    props.instrumentCode,
    props.timeframe,
    props.prices,
    props.indicators,
    props.loading,
    props.error,
  ],
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
    class="unified-market-chart"
    data-testid="unified-market-chart"
    :data-chart-state="chartState"
  >
    <div class="chart-workspace">
      <div class="chart-stage">
        <div class="chart-rail" aria-label="动能颜色说明">
          <span class="rail-caption">动能</span>
          <b class="macd-positive">MACD柱（正）</b>
          <b class="macd-negative">MACD柱（负）</b>
          <b class="dif-label" data-testid="dif-label">DIF</b>
          <b class="dea-label" data-testid="dea-label">DEA</b>
          <b class="derivative-label" data-testid="dif-first-change-label">DIF一阶变化</b>
          <button type="button" class="chart-reset" @click="resetZoom">重置缩放</button>
        </div>
        <div
          ref="chartElement"
          class="unified-chart-canvas"
          role="img"
          aria-label="价格、成交量、DIF、DEA、MACD和DIF一阶变化联动图"
        ></div>
        <div v-if="error" class="chart-status error">{{ error }}</div>
        <div v-else-if="!loading && series.dates.length === 0" class="chart-status">
          当前周期没有可连续展示的价格与动能数据。
        </div>
      </div>

      <aside
        class="market-point-details"
        data-testid="market-point-details"
        :data-hover-index="selectedPoint?.index ?? -1"
        :data-update-count="pointerUpdateCount"
        aria-live="polite"
      >
        <header>
          <div>
            <span class="details-kicker">CURSOR DATA</span>
            <h3>当前读取位置</h3>
          </div>
          <b>{{ timeframe === "daily" ? "日K" : timeframe === "weekly" ? "周K" : "月K" }}</b>
        </header>
        <template v-if="selectedPoint">
          <time>{{ selectedPoint.date }}</time>
          <section class="details-section price-details">
            <h4>价格</h4>
            <dl>
              <div><dt>开盘</dt><dd>{{ formatted(selectedPoint.open, 3) }}</dd></div>
              <div><dt>最高</dt><dd>{{ formatted(selectedPoint.high, 3) }}</dd></div>
              <div><dt>最低</dt><dd>{{ formatted(selectedPoint.low, 3) }}</dd></div>
              <div class="primary-value"><dt>收盘</dt><dd>{{ formatted(selectedPoint.close, 3) }}</dd></div>
            </dl>
          </section>
          <section class="details-section">
            <h4>均线与成交量</h4>
            <dl>
              <div><dt class="ma5-detail">MA5</dt><dd>{{ formatted(selectedPoint.ma5, 3) }}</dd></div>
              <div><dt class="ma10-detail">MA10</dt><dd>{{ formatted(selectedPoint.ma10, 3) }}</dd></div>
              <div><dt class="ma20-detail">MA20</dt><dd>{{ formatted(selectedPoint.ma20, 3) }}</dd></div>
              <div><dt>成交量</dt><dd>{{ formattedVolume(selectedPoint.volume) }}</dd></div>
            </dl>
          </section>
          <section class="details-section momentum-details">
            <h4>动能</h4>
            <dl>
              <div><dt class="dif-label">DIF</dt><dd>{{ formatted(selectedPoint.dif, 4) }}</dd></div>
              <div><dt class="dea-label">DEA</dt><dd>{{ formatted(selectedPoint.dea, 4) }}</dd></div>
              <div><dt>MACD柱</dt><dd>{{ formatted(selectedPoint.macd, 4) }}</dd></div>
              <div><dt class="derivative-label">DIF一阶变化</dt><dd>{{ formatted(selectedPoint.difFirstChange, 4) }}</dd></div>
            </dl>
          </section>
          <footer><span>数据来源</span><b>{{ selectedPoint.source ?? "—" }}</b></footer>
        </template>
        <p v-else class="details-empty">等待行情数据。</p>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.unified-market-chart {
  min-height: 900px;
}

.chart-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 318px;
  gap: 18px;
  align-items: start;
}

.chart-stage {
  position: relative;
  min-width: 0;
}

.unified-chart-canvas {
  width: 100%;
  height: 900px;
}

.chart-rail {
  position: absolute;
  z-index: 2;
  top: 590px;
  left: 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  max-width: calc(100% - 36px);
  padding: 7px 10px;
  border: 1px solid rgba(185, 176, 158, .8);
  background: rgba(255, 253, 246, .9);
  box-shadow: 3px 3px 0 rgba(23, 43, 58, .08);
  font-size: 13px;
}

.rail-caption {
  color: #69747a;
  font-weight: 800;
  letter-spacing: .12em;
}

.macd-positive { color: #d9553f; }
.macd-negative { color: #14836d; }
.dif-label { color: #12687e; }
.dea-label { color: #d88b2c; }
.derivative-label { color: #7b4ca0; }

.chart-reset {
  margin-left: auto;
  padding: 5px 9px;
  border: 1px solid #8f887a;
  color: #172b3a;
  background: transparent;
  font-size: 12px;
  font-weight: 800;
}

.chart-reset:hover,
.chart-reset:focus-visible {
  color: #fffdf6;
  background: #172b3a;
  outline: none;
}

.chart-status {
  position: absolute;
  inset: 70px 24px 48px;
  display: grid;
  place-items: center;
  color: #69747a;
  background: rgba(255, 253, 246, .84);
  font-size: 15px;
}

.chart-status.error { color: #a23f32; }

.market-point-details {
  position: sticky;
  top: 18px;
  min-height: 846px;
  padding: 18px;
  border: 1px solid #b9b09e;
  background: #f7f4ec;
  box-shadow: 5px 5px 0 rgba(23, 43, 58, .08);
  color: #172b3a;
}

.market-point-details > header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  padding-bottom: 14px;
  border-bottom: 2px solid #172b3a;
}

.market-point-details h3,
.market-point-details h4,
.market-point-details p { margin: 0; }

.market-point-details h3 { margin-top: 4px; font-size: 20px; }
.market-point-details > header > b { padding: 5px 8px; background: #172b3a; color: #fffdf6; }
.details-kicker { color: #12687e; font-size: 10px; font-weight: 900; letter-spacing: .14em; }
.market-point-details > time { display: block; padding: 18px 0 10px; font-size: 24px; font-weight: 850; }

.details-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #d4cdc0;
}

.details-section h4 {
  margin-bottom: 6px;
  color: #69747a;
  font-size: 11px;
  letter-spacing: .12em;
}

.details-section dl { margin: 0; }
.details-section dl > div { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; }
.details-section dt { color: #657279; }
.details-section dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 800; text-align: right; }
.details-section .primary-value { margin: 4px -8px 0; padding: 8px; background: #e6efed; }
.details-section .primary-value dd { color: #12687e; font-size: 18px; }
.ma5-detail { color: #9f5f43 !important; }
.ma10-detail { color: #6c7fa9 !important; }
.ma20-detail { color: #7d6b32 !important; }

.market-point-details footer {
  display: grid;
  gap: 4px;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #d4cdc0;
  color: #69747a;
  font-size: 11px;
}

.market-point-details footer b { color: #172b3a; overflow-wrap: anywhere; }
.details-empty { padding-top: 18px; color: #69747a; }

@media (max-width: 1180px) {
  .chart-workspace { grid-template-columns: 1fr; }
  .market-point-details {
    position: static;
    min-height: 0;
  }
}

@media (max-width: 760px) {
  .unified-market-chart,
  .unified-chart-canvas {
    min-height: 760px;
    height: 760px;
  }

  .chart-rail {
    top: 538px;
    right: 12px;
    left: 12px;
    flex-wrap: wrap;
    gap: 6px 10px;
  }
}
</style>
