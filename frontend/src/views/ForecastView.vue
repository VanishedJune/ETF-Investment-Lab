<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { api } from "../api";

type Availability = {
  daily_rows: number;
  required_rows: number;
  can_run: boolean;
  data_cutoff?: string | null;
  sources: string[];
  demo: boolean;
};

type ValuationPoint = {
  week: number;
  date: string;
  pe_ratio: string | number | null;
  pb_ratio: string | number | null;
  valuation_percentile: string | number | null;
  source?: string | null;
};

type MatchedWindow = {
  end_date: string;
  distance: number;
  valuation_path?: ValuationPoint[];
};

const WINDOW_DAYS = 60;
const HORIZON_WEEKS = 13;
const CANDIDATE_COUNT = 5;
const instruments = ref<any[]>([]);
const selected = ref("000688");
const availability = ref<Availability>();
const result = ref<any>();
const error = ref("");
const running = ref(false);
const loadingAvailability = ref(false);
const chartEl = ref<HTMLElement>();
const valuationHistoryEl = ref<HTMLElement>();
const valuationMetric = ref<keyof Pick<ValuationPoint, "pe_ratio" | "pb_ratio" | "valuation_percentile">>("valuation_percentile");
let chart: echarts.ECharts | undefined;
let valuationHistoryChart: echarts.ECharts | undefined;

const selectedInstrument = computed(() => instruments.value.find((item) => item.code === selected.value));
const shortfall = computed(() => Math.max(0, (availability.value?.required_rows ?? 0) - (availability.value?.daily_rows ?? 0)));
const hasValuationHistory = computed(() => {
  const windows = (result.value?.matched_windows ?? []) as MatchedWindow[];
  return windows.some((window) => window.valuation_path?.some((point) => point[valuationMetric.value] !== null && point[valuationMetric.value] !== undefined));
});

async function loadAvailability() {
  if (!selected.value) return;
  loadingAvailability.value = true;
  try {
    availability.value = await api<Availability>(
      `/forecasts/${selected.value}/availability?window_days=${WINDOW_DAYS}&horizon_weeks=${HORIZON_WEEKS}&candidate_count=${CANDIDATE_COUNT}`,
    );
  } catch (reason: any) {
    availability.value = undefined;
    error.value = reason.message;
  } finally {
    loadingAvailability.value = false;
  }
}

async function run() {
  if (!availability.value?.can_run) return;
  running.value = true;
  error.value = "";
  try {
    result.value = await api<any>(`/forecasts/${selected.value}`, {
      method: "POST",
      body: JSON.stringify({ window_days: WINDOW_DAYS, horizon_weeks: HORIZON_WEEKS, candidate_count: CANDIDATE_COUNT }),
    });
    await nextTick();
    render();
  } catch (reason: any) {
    error.value = reason.message;
  } finally {
    running.value = false;
  }
}

function render() {
  if (!chartEl.value || !result.value) return;
  chart?.dispose();
  chart = echarts.init(chartEl.value);
  const points = result.value.points;
  chart.setOption({
    animation: false,
    color: ["#2767cc", "#d84b45"],
    grid: { left: 66, right: 32, top: 46, bottom: 48 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(28, 43, 55, .94)",
      borderWidth: 0,
      textStyle: { color: "#fff", fontSize: 13 },
    },
    legend: { top: 12, data: ["情景中位路径", "历史样本离散区间"], textStyle: { color: "#40515d", fontSize: 12 } },
    xAxis: {
      type: "category",
      data: points.map((point: any) => point.date),
      axisLine: { lineStyle: { color: "#b9c5cd" } },
      axisTick: { show: false },
      axisLabel: { color: "#596874", fontSize: 12 },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLabel: { color: "#596874", fontSize: 12 },
      splitLine: { lineStyle: { color: "#e3e9ec", type: "dashed" } },
    },
    series: [
      {
        name: "区间下界",
        type: "line",
        data: points.map((point: any) => point.lower_bound),
        stack: "range",
        lineStyle: { opacity: 0 },
        symbol: "none",
      },
      {
        name: "历史样本离散区间",
        type: "line",
        data: points.map((point: any) => Number(point.upper_bound) - Number(point.lower_bound)),
        stack: "range",
        lineStyle: { opacity: 0 },
        areaStyle: { color: "rgba(39, 103, 204, .18)" },
        symbol: "none",
      },
      {
        name: "情景中位路径",
        type: "line",
        data: points.map((point: any) => point.predicted_price),
        smooth: true,
        symbol: "circle",
        symbolSize: 7,
        lineStyle: { color: "#d84b45", width: 2.4 },
        itemStyle: { color: "#d84b45" },
      },
    ],
  });
  renderValuationHistory();
}

function metricLabel(metric: typeof valuationMetric.value): string {
  return ({
    pe_ratio: "PE",
    pb_ratio: "PB",
    valuation_percentile: "估值百分位",
  } as const)[metric];
}

function metricValue(point: ValuationPoint | undefined): number | null {
  if (!point) return null;
  const value = Number(point[valuationMetric.value]);
  return Number.isFinite(value) ? value : null;
}

function renderValuationHistory() {
  if (!valuationHistoryEl.value || !result.value) return;
  valuationHistoryChart?.dispose();
  valuationHistoryChart = echarts.init(valuationHistoryEl.value);
  const windows = (result.value.matched_windows ?? []) as MatchedWindow[];
  const label = metricLabel(valuationMetric.value);
  const palette = ["#2767cc", "#d84b45", "#16885f", "#d6911e", "#8b5ca7"];
  const weeks = Array.from({ length: HORIZON_WEEKS + 1 }, (_, week) => `第 ${week} 周`);
  valuationHistoryChart.setOption({
    animation: false,
    color: palette,
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(28, 43, 55, .94)",
      borderWidth: 0,
      textStyle: { color: "#fff", fontSize: 13 },
      formatter: (items: Array<{ dataIndex: number; marker: string; seriesName: string; value: number | null }>) => {
        const week = items[0]?.dataIndex ?? 0;
        const lines = items.map((item) => {
          const window = windows.find((candidate) => candidate.end_date === item.seriesName);
          const point = window?.valuation_path?.[week];
          const value = item.value == null ? "暂无本地数据" : valuationMetric.value === "valuation_percentile" ? `${(item.value * 100).toFixed(2)}%` : item.value.toFixed(2);
          const detail = point ? ` · ${point.date} · PE ${point.pe_ratio ?? "—"} · PB ${point.pb_ratio ?? "—"}` : "";
          return `${item.marker}${item.seriesName}: ${value}${detail}`;
        });
        return `第 ${week} 周<br/>${lines.join("<br/>")}`;
      },
    },
    legend: { top: 8, type: "scroll", textStyle: { color: "#4c5966", fontSize: 12 } },
    grid: { left: 62, right: 28, top: 44, bottom: 42 },
    xAxis: {
      type: "category",
      data: weeks,
      axisLine: { lineStyle: { color: "#bdc6ce" } },
      axisTick: { show: false },
      axisLabel: { color: "#73808c", fontSize: 12 },
    },
    yAxis: {
      type: "value",
      scale: true,
      name: label,
      axisLabel: {
        color: "#73808c",
        formatter: valuationMetric.value === "valuation_percentile" ? (value: number) => `${(value * 100).toFixed(0)}%` : undefined,
      },
      splitLine: { lineStyle: { color: "#e6eaed", type: "dashed" } },
    },
    series: windows.map((window, index) => ({
      name: window.end_date,
      type: "line",
      data: Array.from({ length: HORIZON_WEEKS + 1 }, (_, week) => metricValue(window.valuation_path?.[week])),
      showSymbol: true,
      symbolSize: 5,
      connectNulls: false,
      lineStyle: { width: index === 0 ? 2.4 : 1.6 },
    })),
  });
}

function resize() { chart?.resize(); valuationHistoryChart?.resize(); }

onMounted(async () => {
  try {
    instruments.value = await api<any[]>("/instruments");
    selected.value = instruments.value.find((item) => item.code === "000688")?.code ?? instruments.value[0]?.code ?? "";
    await loadAvailability();
  } catch (reason: any) {
    error.value = reason.message;
  }
  window.addEventListener("resize", resize);
});

onBeforeUnmount(() => {
  chart?.dispose();
  valuationHistoryChart?.dispose();
  window.removeEventListener("resize", resize);
});

watch(selected, async () => {
  result.value = undefined;
  error.value = "";
  await loadAvailability();
});

watch(valuationMetric, async () => {
  await nextTick();
  renderValuationHistory();
});
</script>

<template>
  <section class="panel forecast-panel">
    <div class="panel-head forecast-head">
      <div>
        <span class="eyebrow">HISTORICAL SCENARIO / LOCAL DATA</span>
        <h2>13 周历史相似情景</h2>
        <p class="muted">将最近 60 个交易日与更早、且不重叠并已完成后续路径的历史窗口匹配。结果是历史统计情景，不是价格预测，不使用 AI 或未来数据。</p>
      </div>
      <div class="toolbar">
        <select v-model="selected" class="select" aria-label="选择指数">
          <option v-for="instrument in instruments" :key="instrument.code" :value="instrument.code">{{ instrument.code }} · {{ instrument.name }}</option>
        </select>
        <button class="button" :disabled="running || loadingAvailability || !availability?.can_run" @click="run">
          {{ running ? "正在推演…" : "运行情景推演" }}
        </button>
      </div>
    </div>

    <div class="forecast-readiness" :class="{ ready: availability?.can_run, unavailable: availability && !availability.can_run }">
      <div><span>当前指数</span><b>{{ selectedInstrument?.name ?? "正在读取…" }}</b></div>
      <div><span>本地日线</span><b>{{ availability?.daily_rows ?? "—" }} / {{ availability?.required_rows ?? 189 }} 条</b></div>
      <div><span>数据截止</span><b>{{ availability?.data_cutoff ?? "—" }}</b></div>
      <div><span>推演状态</span><b v-if="loadingAvailability">检查中</b><b v-else-if="availability?.can_run">可运行</b><b v-else>还差 {{ shortfall }} 条</b></div>
    </div>

    <div v-if="availability?.demo" class="status demo-status">当前仅有 DEMO_INDEX 演示日线。该数据不会被伪装为真实指数行情；请在“行情与指标”页刷新直接指数数据后再推演。</div>
    <div v-else-if="availability && !availability.can_run" class="status">13 周、60 日模式需要至少 {{ availability.required_rows }} 个交易日，才能同时保留不重叠的历史窗口和 13 周已实现后续路径。请刷新直接指数数据或导入该指数的历史 CSV。</div>
    <div v-if="error" class="status error">{{ error }}</div>

    <div v-if="!result && !error" class="forecast-empty">
      <strong>{{ availability?.can_run ? "数据已就绪，可运行历史情景推演。" : "等待足够长的本地直接指数历史数据。" }}</strong>
      <span>使用 {{ CANDIDATE_COUNT }} 个历史相似窗口，展示后续 {{ HORIZON_WEEKS }} 周已经发生的路径分布。</span>
    </div>

    <template v-if="result">
      <div class="status">{{ result.label }} 数据截止：{{ result.data_cutoff }}。阴影带仅表示历史样本离散度，不构成未来收益承诺。</div>
      <div ref="chartEl" class="chart forecast-chart"></div>
      <section class="history-valuation-section section-gap" data-testid="valuation-history-chart">
        <div class="history-valuation-head">
          <div>
            <span class="eyebrow">MATCHED WINDOWS / RECORDED VALUATION</span>
            <h3>历史窗口估值变化</h3>
            <p class="muted">每条线对应一个匹配窗口从第 0 周到第 13 周的已记录估值。缺失值保留为空档，不用预测、前填或 ETF 数据替代。</p>
          </div>
          <label class="field metric-picker">显示指标
            <select v-model="valuationMetric">
              <option value="valuation_percentile">估值百分位</option>
              <option value="pe_ratio">PE</option>
              <option value="pb_ratio">PB</option>
            </select>
          </label>
        </div>
        <div v-if="hasValuationHistory" ref="valuationHistoryEl" class="history-valuation-chart"></div>
        <div v-else class="valuation-history-empty" data-testid="valuation-history-unavailable">
          匹配窗口的日期尚无本地估值记录，因此不能绘制真实的估值变化曲线。系统不会用当前估值倒填历史，也不会用 ETF 估值替代指数估值；请在“估值与来源”刷新或导入覆盖该历史区间的直接指数估值数据。
        </div>
      </section>
      <h3>匹配的历史窗口</h3>
      <div class="mini-list section-gap">
        <div v-for="window in result.matched_windows" :key="window.end_date"><span>历史窗口结束日：{{ window.end_date }}</span><b>形态距离：{{ Number(window.distance).toFixed(6) }}</b></div>
      </div>
    </template>
  </section>
</template>
