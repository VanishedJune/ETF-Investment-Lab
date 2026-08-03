<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type {
  ActiveInstrumentCode,
  V33AnalysisPayload,
  V33MarketModelStatus,
  V33ProgressiveResult,
} from "../types/research";

const props = defineProps<{
  selected: ActiveInstrumentCode;
  result: V33AnalysisPayload | null;
  statuses: Partial<Record<ActiveInstrumentCode, V33MarketModelStatus>>;
  curves: Partial<Record<ActiveInstrumentCode, V33ProgressiveResult>>;
  positions: Partial<Record<ActiveInstrumentCode, number | null>>;
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
const pathElement = ref<HTMLDivElement | null>(null);
let pathChart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;

const selectedStatus = computed(() => props.statuses[props.selected]);
const localPosition = computed(() => props.positions[props.selected] ?? 0);
const pathPoints = computed(() => {
  const path = props.result?.path;
  if (!path) return [];
  const length = Math.min(
    20,
    path.weeks.length,
    path.p10.length,
    path.p50.length,
    path.p90.length,
    path.expected.length,
  );
  return Array.from({ length }, (_, index) => ({
    week: path.weeks[index],
    p10: path.p10[index] * 100,
    p50: path.p50[index] * 100,
    p90: path.p90[index] * 100,
    expected: path.expected[index] * 100,
    weeklyBase: path.weekly_base?.[index] === undefined ? null : path.weekly_base[index] * 100,
    dailyCorrection: path.daily_correction?.[index] === undefined ? null : path.daily_correction[index] * 100,
  }));
});
const hasPathDecomposition = computed(() =>
  props.result?.path.weekly_base?.length === 20
  && props.result.path.daily_correction?.length === 20,
);
const provenanceRecord = computed<Record<string, unknown> | null>(() => {
  const provenance = props.result?.data_quality.provenance;
  return provenance && !Array.isArray(provenance) ? provenance : null;
});
const priceDataAsOf = computed(() =>
  props.result?.data_quality.price_data_as_of
  ?? provenanceRecord.value?.price_data_as_of
  ?? provenanceRecord.value?.price_as_of
  ?? null,
);
const analysisAsOf = computed(() =>
  props.result?.data_quality.analysis_as_of
  ?? provenanceRecord.value?.analysis_as_of
  ?? null,
);
const provenanceEntries = computed(() => {
  const provenance = props.result?.data_quality.provenance;
  if (!provenance) return [];
  if (Array.isArray(provenance)) {
    return provenance.map((value, index) => [`来源${index + 1}`, displayValue(value)] as const);
  }
  return Object.entries(provenance)
    .filter(([key]) => !["price_as_of", "price_data_as_of", "analysis_as_of"].includes(key))
    .map(([key, value]) => [key, displayValue(value)] as const);
});
const weeklyFeatureEntries = computed(() =>
  Object.entries(props.result?.features.weekly ?? {}),
);
const dailyFeatureEntries = computed(() =>
  Object.entries(props.result?.features.daily ?? {}),
);

function setCurveElement(market: ActiveInstrumentCode, element: unknown) {
  if (element instanceof HTMLDivElement) curveElements.value[market] = element;
}

function finite(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function format(value: unknown, digits = 2): string {
  const number = finite(value);
  return number === null ? "待成熟" : number.toFixed(digits);
}

function formatProbability(value: unknown): string {
  const number = finite(value);
  return number === null ? "—" : `${number.toFixed(1)}%`;
}

function formatReturn(value: unknown): string {
  const number = finite(value);
  return number === null ? "—" : `${number.toFixed(2)}%`;
}

function formatPositionAdjustment(value: unknown): string {
  const number = finite(value);
  if (number === null) return "—";
  return `${number > 0 ? "+" : ""}${number.toFixed(0)}个百分点`;
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function windowText(window: { start: string | null; center: string | null; end: string | null } | null): string {
  if (!window?.center) return "本轮未形成稳定日期窗口";
  return `${window.start ?? window.center} — ${window.end ?? window.center}`;
}

function actionName(action: "buy" | "sell" | "hold"): string {
  if (action === "buy") return "分批增仓";
  if (action === "sell") return "分批减仓";
  return "维持仓位";
}

function renderCurves() {
  for (const market of markets) {
    const element = curveElements.value[market];
    if (!element) continue;
    const chart = curveCharts.get(market) ?? echarts.init(element);
    curveCharts.set(market, chart);
    const all = props.curves[market]?.curve ?? [];
    const points = range.value ? all.slice(-range.value) : all;
    if (!points.length) {
      chart.clear();
      continue;
    }
    chart.setOption({
      animationDuration: 260,
      color: ["#0b67d7", "#d9553f", "#14836d"],
      tooltip: {
        trigger: "axis",
        appendToBody: true,
        formatter(parameters: any[]) {
          const point = points[parameters[0]?.dataIndex ?? 0];
          return [
            `<b>第 ${point.iteration} 次 · ${point.cutoff_date}</b>`,
            `状态：${point.status === "pending" ? "等待20周结果" : "已成熟"}`,
            `综合损失：${format(point.loss, 4)}`,
            `价格拐点偏离：${format(point.price_turn_error_days, 1)} 交易日`,
            `DIF拐点偏离：${format(point.dif_turn_error_days, 1)} 交易日`,
          ].join("<br>");
        },
      },
      legend: {
        top: 2,
        right: 8,
        data: ["样本外损失", "价格拐点偏离", "DIF拐点偏离"],
        textStyle: { color: "#596873", fontSize: 11 },
      },
      grid: { left: 58, right: 58, top: 50, bottom: 48 },
      xAxis: {
        type: "category",
        name: "迭代次数",
        boundaryGap: false,
        data: points.map((point) => point.iteration),
        axisLabel: { hideOverlap: true, color: "#687782" },
      },
      yAxis: [
        {
          type: "value",
          name: "样本外损失",
          scale: true,
          axisLabel: { color: "#687782" },
          splitLine: { lineStyle: { color: "#e4e9ee", type: "dashed" } },
        },
        {
          type: "value",
          name: "偏离交易日",
          min: 0,
          axisLabel: { color: "#687782" },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "样本外损失",
          type: "line",
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.status === "full" ? point.loss : null),
          lineStyle: { width: 2.2 },
        },
        {
          name: "价格拐点偏离",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.status === "full" ? point.price_turn_error_days : null),
          lineStyle: { width: 1.7 },
        },
        {
          name: "DIF拐点偏离",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.status === "full" ? point.dif_turn_error_days : null),
          lineStyle: { width: 1.7 },
        },
      ],
    }, true);
  }
}

function renderPath() {
  if (!pathElement.value) return;
  pathChart ??= echarts.init(pathElement.value);
  if (pathPoints.value.length !== 20) {
    pathChart.clear();
    return;
  }
  pathChart.setOption({
    animationDuration: 280,
    color: ["#14836d", "#0b67d7", "#d88b2c", "#d9553f", "#6656a8", "#c56a20"],
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: unknown) => `${format(value)}%`,
    },
    legend: {
      top: 4,
      data: hasPathDecomposition.value
        ? ["P10", "P50", "期望路径", "P90", "周K基线路径", "日K有界修正"]
        : ["P10", "P50", "期望路径", "P90"],
    },
    grid: { left: 62, right: 24, top: 48, bottom: 44 },
    xAxis: {
      type: "category",
      name: "未来周数",
      data: pathPoints.value.map((point) => `第${point.week}周`),
      axisLabel: { hideOverlap: true },
    },
    yAxis: {
      type: "value",
      name: "累计收益",
      axisLabel: { formatter: "{value}%" },
      splitLine: { lineStyle: { color: "#e4e9ee", type: "dashed" } },
    },
    series: [
      { name: "P10", type: "line", showSymbol: false, data: pathPoints.value.map((point) => point.p10), lineStyle: { type: "dashed" } },
      { name: "P50", type: "line", showSymbol: false, data: pathPoints.value.map((point) => point.p50), lineStyle: { width: 2.4 } },
      { name: "期望路径", type: "line", showSymbol: false, data: pathPoints.value.map((point) => point.expected), lineStyle: { width: 2 } },
      { name: "P90", type: "line", showSymbol: false, data: pathPoints.value.map((point) => point.p90), lineStyle: { type: "dashed" } },
      ...(hasPathDecomposition.value ? [
        { name: "周K基线路径", type: "line", showSymbol: false, data: pathPoints.value.map((point) => point.weeklyBase), lineStyle: { width: 2, type: "dotted" } },
        { name: "日K有界修正", type: "line", showSymbol: true, symbolSize: 5, data: pathPoints.value.map((point) => point.dailyCorrection), lineStyle: { width: 1.5, type: "dashed" } },
      ] : []),
    ],
  }, true);
}

watch([() => props.curves, range], () => void nextTick(renderCurves), { deep: true });
watch(() => props.result, () => void nextTick(renderPath), { deep: true });
watch(pathElement, (element, previous) => {
  if (previous && previous !== element) resizeObserver?.unobserve(previous);
  if (pathChart && previous !== element) {
    pathChart.dispose();
    pathChart = undefined;
  }
  if (element) {
    resizeObserver?.observe(element);
    void nextTick(renderPath);
  }
}, { flush: "post" });
onMounted(() => {
  renderCurves();
  renderPath();
  resizeObserver = new ResizeObserver(() => {
    curveCharts.forEach((chart) => chart.resize());
    pathChart?.resize();
  });
  Object.values(curveElements.value).forEach((element) => element && resizeObserver?.observe(element));
  if (pathElement.value) resizeObserver.observe(pathElement.value);
});
onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  curveCharts.forEach((chart) => chart.dispose());
  pathChart?.dispose();
});
</script>

<template>
  <section class="v33-lab" aria-labelledby="v33-title">
    <header class="lab-head">
      <div>
        <p class="kicker">V3.3-20W · TRAIN SEPARATELY / ANALYZE DIRECTLY</p>
        <h2 id="v33-title">20周预测 · 周K主模型＋100日日K有界修正</h2>
        <p>周K主模型负责完整20周基础路径，最近100日日K只在前4周进行有界修正；历史相似情景仅校准概率区间。</p>
      </div>
      <div class="primary-actions">
        <button
          class="train-action"
          data-testid="v33-train"
          type="button"
          :disabled="trainingLoading[selected]"
          @click="emit('train', selected)"
        >{{ trainingLoading[selected] ? "训练进行中…" : "训练模型" }}</button>
        <button
          class="analyze-action"
          data-testid="v33-analyze"
          type="button"
          :disabled="analysisLoading || trainingLoading[selected] || !selectedStatus?.bootstrapped"
          @click="emit('analyze')"
        >{{ analysisLoading ? "正在分析…" : "数据分析" }}</button>
      </div>
    </header>

    <div class="status-strip" aria-live="polite">
      <span><b>{{ names[selected] }}</b>{{ selected }}</span>
      <span><b>{{ selectedStatus?.iteration_count ?? 0 }}</b>正式迭代</span>
      <span><b>{{ selectedStatus?.champion_version ?? "尚未训练" }}</b>最新Champion</span>
      <span><b>{{ selectedStatus?.pending_count ?? 20 }}</b>待成熟轮次</span>
      <span :class="['state-pill', { running: selectedStatus?.is_training }]">
        {{ selectedStatus?.is_training ? "正在训练" : selectedStatus?.message ?? "训练空闲" }}
      </span>
    </div>
    <p v-if="trainingError[selected]" class="lab-error" role="alert">{{ trainingError[selected] }}</p>
    <p v-if="analysisError" class="lab-error" role="alert">{{ analysisError }}</p>

    <div class="training-toolbar">
      <div>
        <b>双市场递进训练曲线</b>
        <span>横轴为真实迭代次数；最新20轮未成熟时保持空白，不补零、不插值。</span>
      </div>
      <div class="range-tabs" aria-label="V3.3迭代曲线范围">
        <button
          v-for="item in ranges"
          :key="item"
          type="button"
          :aria-pressed="range === item"
          @click="range = item"
        >{{ item === 0 ? "全部" : `最近${item}次` }}</button>
      </div>
    </div>

    <div class="curve-grid">
      <article v-for="market in markets" :key="market" class="curve-card">
        <header>
          <div><span>{{ market }}</span><h3>{{ names[market] }}</h3></div>
          <dl>
            <div><dt>正式迭代</dt><dd>{{ statuses[market]?.iteration_count ?? curves[market]?.curve.length ?? 0 }}</dd></div>
            <div><dt>完整评价</dt><dd>{{ curves[market]?.full_count ?? 0 }}</dd></div>
            <div><dt>等待20周</dt><dd>{{ curves[market]?.pending_count ?? statuses[market]?.pending_count ?? 20 }}</dd></div>
          </dl>
        </header>
        <div
          :ref="(element) => setCurveElement(market, element)"
          :data-testid="`v33-curve-${market}`"
          :aria-label="`${names[market]}V3.3迭代损失与拐点偏离曲线`"
          class="training-curve"
          role="img"
        ></div>
      </article>
    </div>

    <template v-if="result">
      <section class="analysis-summary" data-testid="v33-analysis-result">
        <article class="market-verdict">
          <p class="kicker">20-WEEK PROBABILITY VERDICT</p>
          <h3>{{ result.advice.summary }}</h3>
          <div class="probability-row">
            <span><b>{{ formatProbability(result.path.up_probability) }}</b>上涨</span>
            <span><b>{{ formatProbability(result.path.sideways_probability) }}</b>震荡</span>
            <span><b>{{ formatProbability(result.path.down_probability) }}</b>下跌</span>
          </div>
          <dl>
            <div><dt>方向</dt><dd>{{ result.path.direction }}</dd></div>
            <div><dt>置信度</dt><dd>{{ formatProbability(result.path.confidence) }}</dd></div>
            <div><dt>预计最大回撤</dt><dd>{{ formatReturn(result.path.expected_max_drawdown) }}</dd></div>
            <div><dt>预计最高周</dt><dd>{{ result.path.predicted_high_week ?? "—" }}</dd></div>
            <div><dt>预计最低周</dt><dd>{{ result.path.predicted_low_week ?? "—" }}</dd></div>
            <div><dt>历史校准样本</dt><dd>{{ result.path.analogue_calibration_count }}</dd></div>
          </dl>
        </article>
        <article class="position-verdict">
          <p class="kicker">POSITION DECISION · LOCAL CALENDAR</p>
          <h3>{{ actionName(result.advice.action) }}</h3>
          <div class="position-arrow">
            <b>{{ localPosition }}%</b><i aria-hidden="true">→</i><b>{{ result.position.target }}%</b>
          </div>
          <p>基金 : ETF = <strong>{{ result.advice.fund_etf_ratio }}</strong></p>
          <small>当前仓位读取本地月历；目标与每批操作均锁定为5个百分点倍数。</small>
          <small class="policy-line">{{ result.advice.policy.join(" · ") }}</small>
          <p v-if="result.position.current !== localPosition" class="position-warning">
            分析快照记录为{{ result.position.current }}%，与当前月历{{ localPosition }}%不同，请重新运行数据分析。
          </p>
        </article>
      </section>

      <section class="forecast-path">
        <header>
          <div><p class="kicker">FUTURE 20 WEEKS</p><h3>P10 / P50 / P90 / 期望概率路径</h3></div>
          <span>模型 {{ result.model.version }} · 第{{ result.model.iteration_number }}次 · 基准 {{ result.benchmark }} · 日K {{ result.model.daily_window_sessions }}日</span>
        </header>
        <div
          ref="pathElement"
          class="path-chart"
          data-testid="v33-path-chart"
          aria-label="未来20周P10、P50、P90与期望收益路径"
          role="img"
        ></div>
        <p v-if="pathPoints.length !== 20" class="path-error">路径数据不是完整20周，图表已停止绘制，避免产生虚假连线。</p>
        <dl class="model-decomposition" data-testid="v33-model-decomposition">
          <div>
            <dt>周K基线路径</dt>
            <dd>{{ hasPathDecomposition ? "20周完整，图中紫色点线" : "接口未返回" }}</dd>
          </div>
          <div>
            <dt>100日日K有界修正</dt>
            <dd>{{ hasPathDecomposition ? "仅前4周生效，5–20周为0" : "接口未返回" }}</dd>
          </div>
          <div>
            <dt>周K基础目标仓位</dt>
            <dd>{{ result.position.weekly_base_target === undefined ? "—" : `${result.position.weekly_base_target}%` }}</dd>
          </div>
          <div>
            <dt>日K仓位修正</dt>
            <dd>{{ formatPositionAdjustment(result.position.daily_adjustment) }}</dd>
          </div>
        </dl>
      </section>

      <section class="turning-grid" data-testid="v33-turning-points">
        <article>
          <span>DIF一阶导数零点窗口</span>
          <h3>{{ windowText(result.turning_points.dif_derivative_zero) }}</h3>
          <p>类型：{{ result.turning_points.kind }} · {{ result.turning_points.stable ? "多窗口一致" : "信号不稳定" }}</p>
        </article>
        <article>
          <span>价格拐点窗口</span>
          <h3>{{ windowText(result.turning_points.price_turn) }}</h3>
          <p>{{ result.turning_points.date_basis }}</p>
        </article>
        <article class="turning-note">
          <b>DIF = 0 与 d(DIF)/dt = 0 分开评价</b>
          <p>零轴穿越不等同于局部高低点；交易日期使用市场真实交易日校准。</p>
        </article>
      </section>

      <section class="batch-plan" data-testid="v33-batches">
        <header><p class="kicker">CONDITIONAL EXECUTION CALENDAR</p><h3>具体日期窗口与仓位百分比</h3></header>
        <p v-if="!result.advice.batches.length" class="empty-plan">当前维持仓位，等待下一完整交易周复核。</p>
        <ol v-else>
          <li v-for="batch in result.advice.batches" :key="batch.batch">
            <span class="sequence">{{ String(batch.batch).padStart(2, "0") }}</span>
            <time>{{ batch.window_start }} — {{ batch.window_end }}</time>
            <strong>{{ batch.action === "buy" ? "买入" : "卖出" }} {{ batch.percentage_points }}%</strong>
            <span>预计日 {{ batch.expected_date }}</span>
            <small>确认条件：{{ batch.condition }}</small>
          </li>
        </ol>
      </section>

      <details class="feature-audit">
        <summary>查看本轮周K与最近100日日K特征快照</summary>
        <div class="feature-columns">
          <section>
            <h3>周K主模型特征</h3>
            <dl><div v-for="[key, value] in weeklyFeatureEntries" :key="key"><dt>{{ key }}</dt><dd>{{ displayValue(value) }}</dd></div></dl>
          </section>
          <section>
            <h3>100日日K修正特征</h3>
            <p>完整序列长度：{{ result.features.daily_sequence_count }}个交易日</p>
            <dl><div v-for="[key, value] in dailyFeatureEntries" :key="key"><dt>{{ key }}</dt><dd>{{ displayValue(value) }}</dd></div></dl>
          </section>
        </div>
      </details>

      <section class="quality-panel" :class="{ degraded: result.data_quality.degraded }">
        <header>
          <div><p class="kicker">POINT-IN-TIME DATA QUALITY</p><h3>数据来源与缺失状态</h3></div>
          <b>{{ result.data_quality.degraded ? "降级分析" : "数据门禁通过" }}</b>
        </header>
        <dl>
          <div data-testid="v33-price-data-as-of"><dt>价格数据时点</dt><dd>{{ priceDataAsOf ?? "—" }}</dd></div>
          <div data-testid="v33-analysis-as-of"><dt>分析运行时点</dt><dd>{{ analysisAsOf ?? "—" }}</dd></div>
          <div><dt>源数据最大日期</dt><dd>{{ result.data_quality.source_data_max_date ?? "—" }}</dd></div>
          <div><dt>训练截止</dt><dd>{{ result.data_quality.cutoff_date ?? "—" }}</dd></div>
          <div><dt>缺失序列</dt><dd>{{ result.data_quality.missing_series.length ? result.data_quality.missing_series.join("、") : "无" }}</dd></div>
          <div><dt>分析哈希</dt><dd>{{ result.analysis_hash.slice(0, 16) }}</dd></div>
        </dl>
        <ul v-if="provenanceEntries.length">
          <li v-for="[key, value] in provenanceEntries" :key="key"><b>{{ key }}</b><span>{{ value }}</span></li>
        </ul>
      </section>
    </template>

    <div v-else class="analysis-empty">
      <b>当前选择：{{ names[selected] }} · 当前仓位 {{ localPosition }}%</b>
      <span v-if="!selectedStatus?.bootstrapped">先点击“训练模型”完成该市场的递进训练。</span>
      <span v-else>点击“数据分析”直接读取最新Champion；此操作不会增加迭代次数。</span>
    </div>
  </section>
</template>

<style scoped>
.v33-lab { margin-top: 16px; border: 1px solid #9eabb5; color: #172b3a; background: #fff; box-shadow: 0 18px 48px rgba(23,43,58,.1); }
.lab-head { display: flex; justify-content: space-between; gap: 30px; padding: 27px; border-bottom: 1px solid #9eabb5; background: #f8fbfe; }
.lab-head h2,.curve-card h3,.analysis-summary h3,.forecast-path h3,.turning-grid h3,.batch-plan h3,.quality-panel h3 { margin: 0; color: #172b3a; font-family: "Noto Serif SC", Georgia, serif; }
.lab-head h2 { font-size: clamp(28px,3vw,38px); }.lab-head p:not(.kicker) { max-width: 780px; margin: 8px 0 0; color: #65747a; font-size: 15px; }
.primary-actions { display: flex; align-items: flex-start; gap: 10px; }.primary-actions button { min-width: 142px; padding: 13px 18px; border: 1px solid #172b3a; border-radius: 5px; font-weight: 900; }.primary-actions button:disabled { cursor: not-allowed; opacity: .45; }
.v33-lab button:focus-visible { outline: 3px solid rgba(11,103,215,.32); outline-offset: 3px; }
.train-action { color: #172b3a; background: #f4bf61; box-shadow: 4px 4px 0 #172b3a; }.analyze-action { color: #fff; background: #0b67d7; box-shadow: 4px 4px 0 #14836d; }
.status-strip { display: grid; grid-template-columns: 1.2fr .7fr 1.4fr .7fr auto; gap: 1px; border-bottom: 1px solid #aeb9c2; background: #d8e0e8; }.status-strip > span { min-width: 0; padding: 11px 14px; color: #69767c; background: #fff; font-size: 11px; }.status-strip b { display: block; overflow: hidden; color: #172b3a; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }.status-strip .state-pill { display: grid; place-items: center; color: #fff; background: #14836d; font-weight: 900; }.status-strip .state-pill.running { background: #d88b2c; }
.lab-error,.path-error { margin: 0; padding: 12px 25px; color: #8b2f26; border-bottom: 1px solid #dfb0a8; background: #f9e7e2; }
.training-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 16px 24px; background: #eef3f7; border-bottom: 1px solid #aeb9c2; }.training-toolbar b,.training-toolbar span { display: block; }.training-toolbar span { margin-top: 3px; color: #69767c; font-size: 12px; }
.range-tabs { display: flex; gap: 3px; padding: 3px; border: 1px solid #b8c3cc; background: #fff; }.range-tabs button { padding: 7px 10px; border: 0; color: #66747a; background: transparent; }.range-tabs button[aria-pressed="true"] { color: #fff; background: #172b3a; }
.curve-grid { display: grid; grid-template-columns: 1fr 1fr; }.curve-card:first-child { border-right: 1px solid #aeb9c2; }.curve-card header { display: flex; justify-content: space-between; gap: 18px; min-height: 94px; padding: 18px 20px 8px; }.curve-card header span { color: #0b67d7; font: 800 11px/1.2 ui-monospace,monospace; letter-spacing: .14em; }.curve-card h3 { font-size: 22px; }.curve-card dl { display: flex; gap: 16px; margin: 0; }.curve-card dl div { display: block; }.curve-card dt { color: #778187; font-size: 10px; }.curve-card dd { margin: 4px 0 0; color: #172b3a; font-size: 15px; font-weight: 800; }.training-curve { height: 350px; }
.analysis-summary { display: grid; grid-template-columns: 1.25fr .75fr; border-top: 1px solid #aeb9c2; border-bottom: 1px solid #aeb9c2; }.analysis-summary article { padding: 25px; }.market-verdict { border-right: 1px solid #aeb9c2; }.analysis-summary h3 { font-size: 29px; }.probability-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 18px 0; }.probability-row span { padding: 10px; border: 1px solid #d4dde5; color: #69767c; text-align: center; font-size: 12px; }.probability-row b { display: block; color: #172b3a; font: 700 24px/1.1 Georgia,serif; }.market-verdict dl { display: grid; grid-template-columns: 1fr 1fr; gap: 0 18px; margin: 0; }.market-verdict dl div { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e5ebef; }.market-verdict dd { margin: 0; font-weight: 800; }
.position-verdict { color: #fff; background: #172b3a; }.position-verdict .kicker { color: #83bfca; }.position-verdict h3 { color: #fff; }.position-arrow { display: flex; justify-content: space-between; align-items: center; margin: 24px 0; padding: 18px 0; border-block: 1px solid rgba(255,255,255,.22); }.position-arrow b { font: 700 44px/1 Georgia,serif; }.position-arrow i { color: #f4bf61; font-size: 24px; }.position-verdict p { color: #c8d2d5; }.position-verdict strong { color: #f4bf61; font-size: 22px; }.position-verdict small { color: #aebbc0; line-height: 1.6; }.position-verdict .policy-line { display: block; margin-top: 8px; color: #83bfca; }.position-warning { padding: 9px; border-left: 3px solid #f4bf61; background: rgba(255,255,255,.08); font-size: 12px; }
.forecast-path { padding: 24px; border-bottom: 1px solid #aeb9c2; }.forecast-path header,.quality-panel header { display: flex; justify-content: space-between; gap: 20px; }.forecast-path h3,.batch-plan h3,.quality-panel h3 { font-size: 27px; }.forecast-path header>span { color: #68767c; font-size: 12px; }.path-chart { height: 390px; }
.model-decomposition { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; margin: 0; border: 1px solid #d8e0e8; background: #d8e0e8; }
.model-decomposition div { min-width: 0; padding: 11px 13px; background: #f8fbfd; }
.model-decomposition dt { color: #68767c; font-size: 11px; }
.model-decomposition dd { margin: 5px 0 0; color: #172b3a; font-size: 13px; font-weight: 800; }
.turning-grid { display: grid; grid-template-columns: 1fr 1fr .8fr; border-bottom: 1px solid #aeb9c2; }.turning-grid article { min-width: 0; padding: 21px; border-right: 1px solid #d5dde4; }.turning-grid article:last-child { border-right: 0; }.turning-grid span { color: #0b67d7; font-size: 11px; font-weight: 900; letter-spacing: .08em; }.turning-grid h3 { margin-top: 8px; font-size: 20px; }.turning-grid p { margin: 8px 0 0; color: #68767c; font-size: 12px; line-height: 1.6; }.turning-note { background: #eef4fb; }.turning-note b { color: #172b3a; }
.batch-plan { padding: 24px; border-bottom: 1px solid #aeb9c2; }.batch-plan ol { margin: 17px 0 0; padding: 0; list-style: none; }.batch-plan li { display: grid; grid-template-columns: 44px 210px 130px 1fr; gap: 7px 16px; align-items: center; padding: 15px 0; border-top: 1px solid #d8e0e7; }.sequence { color: #0b67d7; font: 700 22px Georgia,serif; }.batch-plan strong { color: #172b3a; }.batch-plan small { grid-column: 2/-1; color: #68767c; }.empty-plan,.analysis-empty { color: #68767c; text-align: center; }
.feature-audit { border-bottom: 1px solid #aeb9c2; background: #fff; }.feature-audit > summary { padding: 15px 24px; color: #0b67d7; cursor: pointer; font-weight: 900; }.feature-audit > summary:focus-visible { outline: 3px solid rgba(11,103,215,.3); outline-offset: -3px; }.feature-columns { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid #d8e0e7; }.feature-columns section { min-width: 0; padding: 20px 24px; }.feature-columns section:first-child { border-right: 1px solid #d8e0e7; }.feature-columns h3 { margin: 0 0 10px; font-size: 20px; }.feature-columns p { color: #68767c; font-size: 12px; }.feature-columns dl { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin: 0; background: #e1e7ec; }.feature-columns dl div { min-width: 0; padding: 8px; background: #f8fbfd; }.feature-columns dt { overflow: hidden; color: #68767c; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }.feature-columns dd { margin: 3px 0 0; overflow-wrap: anywhere; font-size: 12px; font-weight: 800; }
.quality-panel { padding: 24px; background: #f8fbfd; }.quality-panel header>b { align-self: start; padding: 7px 10px; color: #fff; background: #14836d; font-size: 12px; }.quality-panel.degraded header>b { background: #d9553f; }.quality-panel dl { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; margin: 18px 0; background: #d8e0e8; }.quality-panel dl div { min-width: 0; padding: 10px; background: #fff; }.quality-panel dt { color: #778187; font-size: 11px; }.quality-panel dd { margin: 4px 0 0; overflow-wrap: anywhere; font-weight: 800; }.quality-panel ul { margin: 0; padding: 0; list-style: none; }.quality-panel li { display: grid; grid-template-columns: minmax(120px,.35fr) 1fr; gap: 10px; padding: 7px 0; border-top: 1px solid #dce3e9; font-size: 12px; }.quality-panel li span { overflow-wrap: anywhere; color: #68767c; }
.analysis-empty { display: grid; min-height: 150px; place-content: center; gap: 6px; padding: 25px; }.analysis-empty b { color: #172b3a; }
@media (max-width: 1050px) { .curve-grid,.analysis-summary,.feature-columns { grid-template-columns: 1fr; }.curve-card:first-child,.market-verdict { border-right: 0; border-bottom: 1px solid #aeb9c2; }.turning-grid { grid-template-columns: 1fr; }.turning-grid article { border-right: 0; border-bottom: 1px solid #d5dde4; }.feature-columns section:first-child { border-right: 0; border-bottom: 1px solid #d8e0e7; }.quality-panel dl,.model-decomposition { grid-template-columns: 1fr 1fr; } }
@media (max-width: 760px) { .lab-head,.training-toolbar,.forecast-path header,.quality-panel header { display: block; }.primary-actions { margin-top: 18px; }.primary-actions button { flex: 1; min-width: 0; }.status-strip { grid-template-columns: 1fr 1fr; }.status-strip .state-pill { min-height: 42px; grid-column: 1/-1; }.range-tabs { margin-top: 12px; }.curve-card header { display: block; }.curve-card dl { margin-top: 12px; }.training-curve { height: 320px; }.market-verdict dl,.quality-panel dl,.model-decomposition { grid-template-columns: 1fr; }.batch-plan li { grid-template-columns: 36px 1fr; }.batch-plan li>* { grid-column: 2; }.batch-plan .sequence { grid-column: 1; grid-row: 1/4; }.batch-plan small { grid-column: 2; }.quality-panel header>b { display: inline-block; margin-top: 12px; } }
</style>
