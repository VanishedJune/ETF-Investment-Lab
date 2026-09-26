<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";

import type {
  IndicatorRow,
  InstrumentCode,
  MarketTimeframe,
  PriceRow,
} from "../types/research";
import { legendColors, normalizeSeries, volumeChangeBarData } from "../types/research";
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
const legendSelection = reactive<Record<string, boolean>>({});

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
      maskColor: "rgba(244, 246, 248, 0.86)",
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
        data: volumeChangeBarData(series.value.volumes, 0.72),
        // Filtered dataZoom keeps the visible category width instead of
        // compressing thousands of historical bars into hairlines.
        barWidth: "72%",
        barMaxWidth: 26,
        barMinWidth: 5,
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
    legend: { show: false },
    grid: [
      { left: 72, right: 72, top: 12, height: "39%" },
      { left: 72, right: 72, top: "47%", height: "11%" },
      { left: 72, right: 72, top: "61%", height: "14%" },
      { left: 72, right: 72, top: "78%", height: "11%" },
    ],
    xAxis: [0, 1, 2, 3].map((gridIndex) => ({
      type: "category",
      gridIndex,
      data: series.value.dates,
      boundaryGap: true,
      axisLine: { lineStyle: { color: "#d5dce5" } },
      axisTick: { show: false },
      axisLabel: {
        show: gridIndex === 3,
        color: "#657184",
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
        axisLabel: { color: "#657184", fontSize: 12 },
        splitLine: { lineStyle: { color: "#e3e8ee" } },
      },
      {
        gridIndex: 1,
        scale: true,
        position: "right",
        splitNumber: 2,
        axisLabel: {
          color: "#657184",
          fontSize: 11,
          formatter: (value: number) => compactAxis(value),
        },
        splitLine: { show: false },
      },
      {
        gridIndex: 2,
        scale: true,
        position: "right",
        axisLabel: { color: "#657184", fontSize: 12 },
        splitLine: { lineStyle: { color: "#e3e8ee" } },
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
        filterMode: "filter",
        throttle: 40,
      },
      {
        type: "slider",
        xAxisIndex: [0, 1, 2, 3],
        start: zoomStart,
        end: zoomEnd,
        bottom: 6,
        height: 22,
        borderColor: "#d5dce5",
        backgroundColor: "#e3e8ee",
        fillerColor: "rgba(102, 132, 167, .12)",
        handleStyle: { color: "#52789c", borderColor: "#52789c" },
        textStyle: { color: "#657184" },
        filterMode: "filter",
        throttle: 40,
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
        // A real candlestick body must remain visible at the default zoom;
        // the previous 3px minimum rendered as a single vertical line.
        barWidth: "70%",
        barMaxWidth: 30,
        barMinWidth: 6,
        itemStyle: {
          color: legendColors.priceUp,
          color0: legendColors.priceDown,
          borderColor: legendColors.priceUp,
          borderColor0: legendColors.priceDown,
          borderWidth: 1.2,
        },
      },
      lineSeries("MA5", series.value.ma5, legendColors.ma5, 0, 0, 1.8),
      lineSeries("MA10", series.value.ma10, legendColors.ma10, 0, 0, 1.8),
      lineSeries("MA20", series.value.ma20, legendColors.ma20, 0, 0, 2.1),
      ...volumeSeries,
      {
        name: "MACD柱",
        type: "bar",
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: series.value.macd,
        // Keep histogram columns visually comparable to the volume bars.
        barWidth: "68%",
        barMaxWidth: 22,
        barMinWidth: 4,
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
    for (const key of Object.keys(legendSelection)) {
      delete legendSelection[key];
    }
    Object.assign(legendSelection, payload.selected ?? {});
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
    lineStyle: { color, width, type: name.includes('MA10') ? 'dashed' : name.includes('MA20') ? 'dotted' : 'solid' },
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

function toggleSeries(name: string) {
  const next = legendSelection[name] !== false;
  legendSelection[name] = !next;
  chart?.dispatchAction({
    type: "legendToggleSelect",
    name,
  });
  scheduleDynamicAxes();
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
          <span class="rail-caption">完整序列</span>
          <span class="price-label">价格/K线</span>
          <b
            class="ma-label ma5-label"
            data-testid="ma5-label"
            :class="{ selected: legendSelection['MA5'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('MA5')"
            @keydown.enter="toggleSeries('MA5')"
          >MA5</b>
          <b
            class="ma-label ma10-label"
            data-testid="ma10-label"
            :class="{ selected: legendSelection['MA10'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('MA10')"
            @keydown.enter="toggleSeries('MA10')"
          >MA10</b>
          <b
            class="ma-label ma20-label"
            data-testid="ma20-label"
            :class="{ selected: legendSelection['MA20'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('MA20')"
            @keydown.enter="toggleSeries('MA20')"
          >MA20</b>
          <span class="volume-label">成交量（量增红 / 量减绿）</span>
          <b
            class="macd-positive"
            :class="{ selected: legendSelection['MACD柱'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('MACD柱')"
            @keydown.enter="toggleSeries('MACD柱')"
          >MACD柱（正）</b>
          <b
            class="macd-negative"
            :class="{ selected: legendSelection['MACD柱'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('MACD柱')"
            @keydown.enter="toggleSeries('MACD柱')"
          >MACD柱（负）</b>
          <b
            class="dif-label"
            data-testid="dif-label"
            :class="{ selected: legendSelection['DIF'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('DIF')"
            @keydown.enter="toggleSeries('DIF')"
          >DIF</b>
          <b
            class="dea-label"
            data-testid="dea-label"
            :class="{ selected: legendSelection['DEA'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('DEA')"
            @keydown.enter="toggleSeries('DEA')"
          >DEA</b>
          <b
            class="derivative-label"
            data-testid="dif-first-change-label"
            :class="{ selected: legendSelection['DIF一阶变化'] !== false }"
            role="button"
            tabindex="0"
            @click="toggleSeries('DIF一阶变化')"
            @keydown.enter="toggleSeries('DIF一阶变化')"
          >DIF一阶变化</b>
          <button type="button" class="chart-reset" @click="resetZoom">重置缩放</button>
        </div>
        <p class="chart-series-note">
          {{ timeframe === "daily" ? "日K" : timeframe === "weekly" ? "周K" : "月K" }}完整序列：{{ series.dates.length }} 个周期；价格、成交量、MACD柱、DIF、DEA与DIF一阶变化共用时间轴
        </p>
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
  height: 900px;
  display: flex;
  flex-direction: column;
}

.unified-chart-canvas {
  width: 100%;
  flex: 1 1 auto;
  min-height: 0;
  height: auto;
}

.chart-rail {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  max-width: calc(100% - 36px);
  padding: 7px 10px;
  border: 1px solid rgba(220, 226, 232, .9);
  background: rgba(244, 246, 248, .9);
  box-shadow: 3px 3px 0 rgba(38, 52, 69, .04);
  font-size: 13px;
  margin-bottom: 8px;
}

.chart-rail b {
  cursor: pointer;
  user-select: none;
  opacity: .55;
}

.chart-rail b.selected {
  opacity: 1;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.chart-rail > span:not(.rail-caption) {
  font-weight: 800;
  white-space: nowrap;
}

.price-label { color: #8c5d49; }
.volume-label { color: #64748b; }
.ma-label { display: inline-flex; align-items: center; gap: 5px; }
.ma-label::before { width: 20px; height: 3px; background: currentColor; content: ""; box-shadow: 0 1px 0 rgba(23, 43, 58, .16); }
.ma5-label { color: #ae794b; }
.ma10-label { color: #52789c; }
.ma20-label { color: #97718f; }

.chart-series-note {
  margin: 0 0 4px;
  color: #657184;
  font-size: 12px;
  letter-spacing: .01em;
}

.rail-caption {
  color: #657184;
  font-weight: 800;
  letter-spacing: .12em;
}

.macd-positive { color: #d64b4b; }
.macd-negative { color: #27845a; }
.dif-label { color: #52789c; }
.dea-label { color: #a57738; }
.derivative-label { color: #807096; }

.chart-reset {
  margin-left: auto;
  padding: 5px 9px;
  border: 1px solid #8f887a;
  color: #263445;
  background: transparent;
  font-size: 12px;
  font-weight: 800;
}

.chart-reset:hover,
.chart-reset:focus-visible {
  color: #ffffff;
  background: #263445;
  outline: none;
}

.chart-status {
  position: absolute;
  inset: 70px 24px 48px;
  display: grid;
  place-items: center;
  color: #657184;
  background: rgba(244, 246, 248, .84);
  font-size: 15px;
}

.chart-status.error { color: #a23f32; }

.market-point-details {
  position: sticky;
  top: 18px;
  min-height: 846px;
  padding: 18px;
  border: 1px solid #d5dce5;
  background: #e9edf2;
  box-shadow: 5px 5px 0 rgba(38, 52, 69, .04);
  color: #263445;
}

.market-point-details > header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  padding-bottom: 14px;
  border-bottom: 2px solid #263445;
}

.market-point-details h3,
.market-point-details h4,
.market-point-details p { margin: 0; }

.market-point-details h3 { margin-top: 4px; font-size: 20px; }
.market-point-details > header > b { padding: 5px 8px; background: #263445; color: #ffffff; }
.details-kicker { color: #52789c; font-size: 10px; font-weight: 900; letter-spacing: .14em; }
.market-point-details > time { display: block; padding: 18px 0 10px; font-size: 24px; font-weight: 850; }

.details-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #d5dce5;
}

.details-section h4 {
  margin-bottom: 6px;
  color: #657184;
  font-size: 11px;
  letter-spacing: .12em;
}

.details-section dl { margin: 0; }
.details-section dl > div { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; }
.details-section dt { color: #657184; }
.details-section dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 800; text-align: right; }
.details-section .primary-value { margin: 4px -8px 0; padding: 8px; background: #e6efed; }
.details-section .primary-value dd { color: #52789c; font-size: 18px; }
.ma5-detail { color: #ae794b !important; }
.ma10-detail { color: #52789c !important; }
.ma20-detail { color: #97718f !important; }

.market-point-details footer {
  display: grid;
  gap: 4px;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #d5dce5;
  color: #657184;
  font-size: 11px;
}

.market-point-details footer b { color: #263445; overflow-wrap: anywhere; }
.details-empty { padding-top: 18px; color: #657184; }

@media (max-width: 1180px) {
  .chart-workspace { grid-template-columns: 1fr; }
  .market-point-details {
    position: static;
    min-height: 0;
  }
}

@media (max-width: 760px) {
  .unified-market-chart,
  .chart-stage {
    min-height: 760px;
    height: 760px;
  }

  .chart-rail {
    flex-wrap: wrap;
    gap: 6px 10px;
  }
}
</style>
