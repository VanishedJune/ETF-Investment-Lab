<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type {
  AnalysisTask,
  LatestAdvice,
  ModelMetrics,
  TaskStatus,
} from "../types/research";

const props = defineProps<{
  task: AnalysisTask | null;
  metrics: ModelMetrics | null;
  advice: LatestAdvice | null;
  loading: boolean;
  error?: string;
}>();

const emit = defineEmits<{
  analyze: [];
  resume: [];
}>();

const metricChartElement = ref<HTMLDivElement>();
let metricChart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;

const stageLabels: Record<TaskStatus, string> = {
  queued: "等待执行",
  preparing_data: "准备十年行情",
  building_features: "构建周线特征",
  iterating: "递进迭代",
  validating: "滚动验证",
  generating_advice: "生成13周建议",
  completed: "分析完成",
  failed: "分析失败",
  recoverable: "可从检查点恢复",
};

const progressPercent = computed(() => {
  const completed = props.task?.completed_weeks ?? 0;
  const total = props.task?.total_weeks ?? 0;
  return total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;
});

const latestCurvePoint = computed(() => props.metrics?.curve.at(-1) ?? null);
const latestSingleDeviation = computed(() => {
  const points = props.metrics?.curve ?? [];
  return [...points].reverse().find((point) => point.deviation_days !== null)?.deviation_days ?? null;
});
const latestRolling20 = computed(() => latestAvailable("rolling_20_abs_deviation"));
const latestRolling52 = computed(() => latestAvailable("rolling_52_abs_deviation"));
const improvement = computed(() => {
  const points = (props.metrics?.curve ?? [])
    .map((point) => numberOrNull(point.rolling_20_abs_deviation))
    .filter((value): value is number => value !== null);
  const first = points.at(0);
  const current = points.at(-1);
  if (first === undefined || current === undefined || first === 0) return null;
  return ((first - current) / first) * 100;
});

const directionText = computed(() => {
  if (!props.advice) return "等待分析";
  return { up: "未来13周偏上涨", down: "未来13周偏下跌", neutral: "未来13周震荡" }[
    props.advice.direction
  ];
});

const operationText = computed(() => {
  if (!props.advice) return "尚未生成建议";
  if (props.advice.current_position === null) return "请先设置当前仓位";
  const amount = Math.abs(props.advice.signed_position_change ?? 0);
  if (!amount) return "维持当前仓位，等待下一次周线确认";
  return `${props.advice.operation_side === "buy" ? "计划增加" : "计划减少"} ${amount} 个百分点`;
});

function latestAvailable(
  field: "rolling_20_abs_deviation" | "rolling_52_abs_deviation",
): number | null {
  const points = props.metrics?.curve ?? [];
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = numberOrNull(points[index][field]);
    if (value !== null) return value;
  }
  return null;
}

function numberOrNull(value: unknown): number | null {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number) ? null : number;
}

function percentText(value: unknown): string {
  const number = numberOrNull(value);
  if (number === null) return "—";
  const percent = Math.abs(number) <= 1 ? number * 100 : number;
  return `${percent.toFixed(1)}%`;
}

function metricText(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(2)} 日`;
}

function renderMetricChart() {
  if (!metricChartElement.value) return;
  metricChart ??= echarts.init(metricChartElement.value);
  const curve = props.metrics?.curve ?? [];
  if (!curve.length) {
    metricChart.clear();
    return;
  }
  metricChart.setOption({
    animationDuration: 300,
    tooltip: {
      trigger: "axis",
      backgroundColor: "#fffdf6",
      borderColor: "#b9b09e",
      textStyle: { color: "#172b3a", fontSize: 13 },
    },
    legend: {
      top: 0,
      right: 8,
      data: ["单次偏离", "20次滚动", "52次滚动"],
      textStyle: { color: "#5f6c72", fontSize: 12 },
    },
    grid: { left: 52, right: 20, top: 38, bottom: 48 },
    xAxis: {
      type: "category",
      name: "迭代次数",
      data: curve.map((point) => point.iteration_number),
      axisLabel: { color: "#69747a", hideOverlap: true },
      axisLine: { lineStyle: { color: "#b9b09e" } },
    },
    yAxis: {
      type: "value",
      name: "偏离交易日",
      axisLabel: { color: "#69747a" },
      splitLine: { lineStyle: { color: "#e6dfd2" } },
    },
    dataZoom: [
      { type: "inside", start: curve.length > 120 ? 70 : 0, end: 100 },
      {
        type: "slider",
        start: curve.length > 120 ? 70 : 0,
        end: 100,
        bottom: 4,
        height: 18,
        borderColor: "#b9b09e",
        fillerColor: "rgba(18, 104, 126, .15)",
      },
    ],
    series: [
      {
        name: "单次偏离",
        type: "scatter",
        symbolSize: 5,
        data: curve.map((point) => point.deviation_days),
        itemStyle: {
          color: (item: { dataIndex: number }) => (
            curve[item.dataIndex]?.direction_correct === false ? "#d9553f" : "#7c8b91"
          ),
        },
      },
      {
        name: "20次滚动",
        type: "line",
        data: curve.map((point) => numberOrNull(point.rolling_20_abs_deviation)),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { color: "#12687e", width: 2.2 },
        itemStyle: { color: "#12687e" },
      },
      {
        name: "52次滚动",
        type: "line",
        data: curve.map((point) => numberOrNull(point.rolling_52_abs_deviation)),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { color: "#d88b2c", width: 2.2 },
        itemStyle: { color: "#d88b2c" },
      },
    ],
  }, true);
}

watch(
  () => props.metrics,
  () => void nextTick(renderMetricChart),
  { deep: true },
);

onMounted(() => {
  renderMetricChart();
  if (metricChartElement.value) {
    resizeObserver = new ResizeObserver(() => metricChart?.resize());
    resizeObserver.observe(metricChartElement.value);
  }
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  metricChart?.dispose();
});
</script>

<template>
  <section class="analysis-panel" aria-label="周线模型分析">
    <header class="analysis-head">
      <div>
        <p class="kicker">LOCAL WEEKLY ANALYSIS · NO AI</p>
        <h2>未来13周研判</h2>
        <p>网页仅触发本地可审计计算；周线决定方向，日线只细化执行日期。</p>
      </div>
      <div class="analysis-actions">
        <button
          class="analysis-button"
          type="button"
          :disabled="loading"
          @click="emit('analyze')"
        >
          {{ loading ? "分析进行中" : "数据分析" }}
        </button>
        <button
          v-if="task?.status === 'recoverable'"
          class="resume-button"
          type="button"
          @click="emit('resume')"
        >
          从检查点恢复
        </button>
      </div>
    </header>

    <div v-if="task" class="task-strip" data-testid="analysis-progress">
      <div class="stage">
        <span>当前阶段</span>
        <strong>{{ stageLabels[task.status] }}</strong>
      </div>
      <div class="progress-track" aria-label="分析进度">
        <i :style="{ width: `${progressPercent}%` }"></i>
      </div>
      <div class="progress-number">
        <b>{{ task.completed_weeks }}</b>
        <span>/ {{ task.total_weeks ?? "待计算" }} 周</span>
      </div>
      <div class="version-stack">
        <span>{{ task.last_iteration ? `I${String(task.last_iteration).padStart(4, "0")}` : "I0000" }}</span>
        <span>{{ task.last_work_version }}</span>
        <span>{{ task.last_model_version }}</span>
      </div>
      <small>检查点：{{ task.last_completed_week ?? "尚未写入" }}</small>
    </div>

    <div v-if="error" class="analysis-error">{{ error }}</div>

    <div class="decision-grid">
      <article class="decision-card">
        <p class="kicker">CURRENT DECISION</p>
        <h3>{{ directionText }}</h3>
        <p class="operation" data-testid="position-recommendation">{{ operationText }}</p>
        <div class="position-scale">
          <div>
            <span>当前仓位</span>
            <b data-testid="advice-current-position">
              {{ advice?.current_position == null ? "未设置" : `${advice.current_position}%` }}
            </b>
          </div>
          <i>→</i>
          <div>
            <span>目标仓位</span>
            <b>{{ advice ? `${advice.target_position}%` : "—" }}</b>
          </div>
        </div>
        <dl class="decision-facts">
          <div><dt>方向概率</dt><dd>{{ percentText(advice?.probability) }}</dd></div>
          <div><dt>置信度</dt><dd>{{ percentText(advice?.confidence) }}</dd></div>
          <div><dt>市场状态</dt><dd>{{ advice?.market_state ?? "—" }}</dd></div>
          <div>
            <dt>基金 : ETF</dt>
            <dd data-testid="ratio-advice">{{ advice?.fund_etf_ratio?.replace(":", " : ") ?? "—" }}</dd>
          </div>
        </dl>
        <p class="decision-scope">
          数据截止 {{ advice?.data_cutoff_date ?? "—" }} ·
          {{ advice?.iteration_id ?? "I0000" }} /
          {{ advice?.work_version ?? "W0000" }} /
          {{ advice?.model_version ?? "M0001" }}
        </p>
      </article>

      <article class="batch-card">
        <div class="batch-head">
          <div>
            <p class="kicker">EXECUTION WINDOW</p>
            <h3>分批执行日历</h3>
          </div>
          <b>{{ advice?.batches.length ?? 0 }} / 4 批</b>
        </div>
        <p v-if="advice?.current_position == null" class="empty-guidance">
          请先在下方投资日历设置当前仓位。市场方向仍可分析，但不生成具体买卖百分比。
        </p>
        <p v-else-if="!advice?.batches.length" class="empty-guidance">
          当前没有需要执行的批次；等待下一次完整周确认。
        </p>
        <ol v-else class="batch-list" data-testid="batch-plan">
          <li v-for="batch in advice.batches" :key="batch.sequence">
            <time>{{ batch.expected_date }}</time>
            <strong>
              {{ batch.operation_side === "buy" ? "增加" : "减少" }} {{ batch.percent }}%
            </strong>
            <span>允许偏差 ±{{ batch.tolerance_trading_days }} 个交易日</span>
            <small>{{ batch.confirmation_condition }}</small>
            <em :class="{ ready: batch.status === 'ready' }">
              {{ batch.status === "ready" ? "可执行" : "等待确认" }}
            </em>
          </li>
        </ol>
      </article>
    </div>

    <div class="model-audit">
      <div class="audit-metrics">
        <article><span>单次偏离</span><b>{{ metricText(latestSingleDeviation == null ? null : Math.abs(latestSingleDeviation)) }}</b></article>
        <article><span>20次滚动偏离</span><b>{{ metricText(latestRolling20) }}</b></article>
        <article><span>52次滚动偏离</span><b>{{ metricText(latestRolling52) }}</b></article>
        <article><span>方向命中</span><b>{{ percentText(metrics?.windows["52"]?.direction_hit_rate ?? metrics?.windows["20"]?.direction_hit_rate) }}</b></article>
        <article><span>校准误差</span><b>{{ percentText(metrics?.windows["52"]?.calibration_loss ?? metrics?.windows["20"]?.calibration_loss) }}</b></article>
        <article><span>候选接受率</span><b>{{ percentText(metrics?.candidate_acceptance_rate) }}</b></article>
        <article>
          <span>相对首个20次窗口</span>
          <b :class="{ improved: (improvement ?? 0) > 0 }">
            {{ improvement == null ? "样本积累中" : `${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)}%` }}
          </b>
        </article>
        <article><span>当前验证模型</span><b>{{ metrics?.model_version ?? latestCurvePoint?.model_version ?? "M0001" }}</b></article>
      </div>
      <div
        ref="metricChartElement"
        class="metric-chart"
        data-testid="model-improvement-chart"
        :data-chart-state="metrics?.curve.length ? 'rendered' : 'empty'"
      ></div>
    </div>
  </section>
</template>

<style scoped>
.analysis-panel {
  margin-top: 16px;
  border: 1px solid #b9b09e;
  background: #fffdf6;
  box-shadow: 0 14px 38px rgba(23, 43, 58, .08);
}

.analysis-head {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 24px;
  border-bottom: 1px solid #b9b09e;
}

.analysis-head h2,
.decision-card h3,
.batch-card h3 {
  margin: 0;
  color: #172b3a;
  font-family: "Noto Serif SC", Georgia, serif;
}

.analysis-head h2 { font-size: 30px; }
.analysis-head p:not(.kicker) { margin: 8px 0 0; color: #69747a; font-size: 14px; }
.analysis-actions { display: flex; align-items: flex-start; gap: 8px; }

.analysis-button,
.resume-button {
  min-width: 124px;
  padding: 12px 18px;
  border: 1px solid #172b3a;
  font-weight: 900;
}

.analysis-button { color: #fffdf6; background: #172b3a; box-shadow: 4px 4px 0 #d88b2c; }
.analysis-button:disabled { cursor: wait; opacity: .62; }
.resume-button { color: #172b3a; background: transparent; }

.task-strip {
  display: grid;
  grid-template-columns: 180px minmax(140px, 1fr) auto auto auto;
  gap: 18px;
  align-items: center;
  padding: 14px 24px;
  background: #e9f0ef;
  border-bottom: 1px solid #b9b09e;
}

.stage span,
.progress-number span,
.task-strip small { color: #69747a; font-size: 13px; }
.stage strong { display: block; margin-top: 3px; }
.progress-track { height: 8px; overflow: hidden; background: #d2d9d6; }
.progress-track i { display: block; height: 100%; background: #12687e; transition: width .3s ease; }
.progress-number b { font-size: 22px; }
.version-stack { display: flex; gap: 6px; }
.version-stack span { padding: 4px 7px; border: 1px solid #91a3a6; font: 700 12px ui-monospace, monospace; }

.analysis-error { padding: 12px 24px; border-bottom: 1px solid #d7a89e; color: #9b352a; background: #f8e9e4; }
.decision-grid { display: grid; grid-template-columns: minmax(300px, .78fr) minmax(0, 1.22fr); border-bottom: 1px solid #b9b09e; }
.decision-card { padding: 24px; color: #fffdf6; background: #172b3a; }
.decision-card .kicker { color: #89c6d0; }
.decision-card h3 { color: #fffdf6; font-size: 31px; }
.operation { min-height: 28px; color: #f3c56e; font-size: 18px; font-weight: 800; }

.position-scale { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: end; margin: 22px 0; padding: 18px 0; border-block: 1px solid rgba(255, 255, 255, .2); }
.position-scale div:last-child { text-align: right; }
.position-scale span { display: block; color: #b8c2c7; font-size: 13px; }
.position-scale b { font: 700 42px/1 Georgia, serif; }
.position-scale i { color: #f3c56e; font-size: 24px; font-style: normal; }
.decision-facts { margin: 0; }
.decision-facts div { display: flex; justify-content: space-between; gap: 20px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.14); }
.decision-facts dt { color: #b8c2c7; }
.decision-facts dd { margin: 0; font-weight: 800; }
.decision-scope { margin: 16px 0 0; color: #9eacb2; font-size: 12px; }

.batch-card { padding: 24px; }
.batch-head { display: flex; justify-content: space-between; gap: 16px; }
.batch-head h3 { font-size: 26px; }
.batch-head > b { color: #12687e; font-size: 19px; }
.empty-guidance { display: grid; min-height: 230px; place-items: center; margin: 0; color: #69747a; text-align: center; }
.batch-list { margin: 20px 0 0; padding: 0; list-style: none; counter-reset: batch; }
.batch-list li { position: relative; display: grid; grid-template-columns: 112px 128px 1fr auto; gap: 10px 16px; align-items: center; padding: 15px 0 15px 38px; border-top: 1px solid #e2dacd; counter-increment: batch; }
.batch-list li::before { position: absolute; left: 0; content: counter(batch, decimal-leading-zero); color: #9b8f7c; font: 700 16px Georgia, serif; }
.batch-list time { font-weight: 800; }
.batch-list strong { color: #172b3a; font-size: 16px; }
.batch-list span,
.batch-list small { color: #69747a; font-size: 13px; }
.batch-list small { grid-column: 2 / 4; }
.batch-list em { padding: 5px 8px; color: #8c5d49; border: 1px solid #d4b6a9; font-size: 12px; font-style: normal; }
.batch-list em.ready { color: #146e5e; border-color: #8db9ac; }

.model-audit { display: grid; grid-template-columns: 310px minmax(0, 1fr); }
.audit-metrics { display: grid; grid-template-columns: repeat(2, 1fr); border-right: 1px solid #b9b09e; }
.audit-metrics article { min-height: 82px; padding: 14px; border-right: 1px solid #e2dacd; border-bottom: 1px solid #e2dacd; }
.audit-metrics span { display: block; color: #69747a; font-size: 12px; }
.audit-metrics b { display: block; margin-top: 8px; color: #172b3a; font: 700 19px Georgia, serif; }
.audit-metrics b.improved { color: #14836d; }
.metric-chart { min-height: 350px; height: 100%; }

@media (max-width: 980px) {
  .task-strip { grid-template-columns: 1fr 1fr; }
  .progress-track { grid-column: 1 / -1; grid-row: 2; }
  .decision-grid,
  .model-audit { grid-template-columns: 1fr; }
  .audit-metrics { border-right: 0; }
  .metric-chart { height: 360px; }
}

@media (max-width: 620px) {
  .analysis-head { display: block; }
  .analysis-actions { margin-top: 16px; }
  .task-strip { grid-template-columns: 1fr; }
  .progress-track { grid-column: auto; grid-row: auto; }
  .batch-list li { grid-template-columns: 1fr; }
  .batch-list small { grid-column: auto; }
  .position-scale b { font-size: 34px; }
}
</style>
