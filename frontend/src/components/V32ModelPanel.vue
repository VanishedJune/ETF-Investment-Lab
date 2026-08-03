<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type {
  ActiveInstrumentCode,
  LegacyV32InstrumentCode,
  V32Analysis,
  V32AnalysisRun,
  V32CurvePayload,
  V32MarketTrainingStatus,
} from "../types/research";

const props = defineProps<{
  selected: ActiveInstrumentCode;
  result: V32Analysis | null;
  run: V32AnalysisRun | null;
  statuses: Partial<Record<LegacyV32InstrumentCode, V32MarketTrainingStatus>>;
  curves: Partial<Record<LegacyV32InstrumentCode, V32CurvePayload>>;
  analysisLoading: boolean;
  trainingLoading: Record<LegacyV32InstrumentCode, boolean>;
  trainingError: Record<LegacyV32InstrumentCode, string>;
  analysisError?: string;
}>();

const names: Record<LegacyV32InstrumentCode, string> = {
  "399006": "创业板指数（V3.2历史）",
  NDX: "纳斯达克100底层指数（V3.2审计）",
};
const selectedAuditMarket = computed<LegacyV32InstrumentCode>(() =>
  props.selected === "399006" ? "399006" : "NDX",
);
const ranges = [20, 52, 0] as const;
const range = ref<(typeof ranges)[number]>(0);
const curveElements = ref<Partial<Record<LegacyV32InstrumentCode, HTMLDivElement>>>({});
const curveCharts = new Map<LegacyV32InstrumentCode, echarts.ECharts>();
const pathElement = ref<HTMLDivElement>();
let pathChart: echarts.ECharts | undefined;
let observer: ResizeObserver | undefined;

function setCurveElement(market: LegacyV32InstrumentCode, element: unknown) {
  if (element instanceof HTMLDivElement) curveElements.value[market] = element;
}

function format(value: unknown, digits = 2): string {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "待成熟";
}

function renderCurves() {
  (["399006", "NDX"] as LegacyV32InstrumentCode[]).forEach((market) => {
    const element = curveElements.value[market];
    if (!element) return;
    const chart = curveCharts.get(market) ?? echarts.init(element);
    curveCharts.set(market, chart);
    const all = props.curves[market]?.points ?? [];
    const points = range.value ? all.slice(-range.value) : all;
    if (!points.length) {
      chart.clear();
      return;
    }
    chart.setOption({
      animationDuration: 280,
      color: ["#11697b", "#d47b25", "#247a65"],
      tooltip: {
        trigger: "axis",
        appendToBody: true,
        formatter(parameters: any[]) {
          const index = parameters[0]?.dataIndex ?? 0;
          const point = points[index];
          return [
            `<b>第 ${point.iteration} 次 · ${point.week_key}</b>`,
            `父迭代：${point.parent_iteration ?? "初始模型"}`,
            `Champion：${point.champion_version}`,
            `52次滚动损失：${format(point.rolling_52_loss, 4)}`,
            `52次高低点偏离：${format(point.rolling_52_deviation_days, 1)} 交易日`,
            point.maturity_status === "pending" ? "状态：等待13周结果成熟" : `本轮损失：${format(point.composite_loss, 4)}`,
            point.promoted ? "结果：Challenger晋级" : `结果：保留Champion${point.rejection_reason ? ` · ${point.rejection_reason}` : ""}`,
          ].join("<br>");
        },
      },
      legend: { top: 2, right: 8, data: ["52次滚动损失", "52次高低点偏离", "晋级点"] },
      grid: { left: 58, right: 58, top: 48, bottom: 46 },
      xAxis: {
        type: "category",
        name: "正式迭代次数",
        boundaryGap: false,
        data: points.map((point) => point.iteration),
        axisLabel: { hideOverlap: true },
      },
      yAxis: [
        { type: "value", name: "样本外损失", scale: true, splitLine: { lineStyle: { color: "#e4ded1" } } },
        { type: "value", name: "偏离交易日", scale: true, splitLine: { show: false } },
      ],
      series: [
        {
          name: "52次滚动损失",
          type: "line",
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.maturity_status === "full" ? point.rolling_52_loss : null),
          lineStyle: { width: 2.2 },
        },
        {
          name: "52次高低点偏离",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          connectNulls: false,
          data: points.map((point) => point.maturity_status === "full" ? point.rolling_52_deviation_days : null),
          lineStyle: { width: 1.8 },
        },
        {
          name: "晋级点",
          type: "scatter",
          symbolSize: 9,
          data: points.map((point) => point.promoted ? point.rolling_52_loss : null),
        },
      ],
    }, true);
  });
}

function renderPath() {
  if (!pathElement.value) return;
  pathChart ??= echarts.init(pathElement.value);
  const points = props.result?.path.points ?? [];
  if (!points.length) {
    pathChart.clear();
    return;
  }
  pathChart.setOption({
    animationDuration: 300,
    color: ["#247a65", "#11697b", "#d47b25", "#b34235"],
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => `${format(value)}%` },
    legend: { top: 4, data: ["P10", "P50", "预期路径", "P90"] },
    grid: { left: 58, right: 24, top: 48, bottom: 42 },
    xAxis: { type: "category", name: "未来周数", data: points.map((point) => `第${point.horizon_week}周`) },
    yAxis: { type: "value", name: "累计收益", axisLabel: { formatter: "{value}%" }, splitLine: { lineStyle: { color: "#e4ded1" } } },
    series: [
      { name: "P10", type: "line", showSymbol: false, data: points.map((point) => point.p10_cumulative_return), lineStyle: { type: "dashed" } },
      { name: "P50", type: "line", showSymbol: false, data: points.map((point) => point.p50_cumulative_return), lineStyle: { width: 2.4 } },
      { name: "预期路径", type: "line", showSymbol: false, data: points.map((point) => point.expected_cumulative_return), lineStyle: { width: 2 } },
      { name: "P90", type: "line", showSymbol: false, data: points.map((point) => point.p90_cumulative_return), lineStyle: { type: "dashed" } },
    ],
  }, true);
}

watch([() => props.curves, range], () => void nextTick(renderCurves), { deep: true });
watch(() => props.result, () => void nextTick(renderPath), { deep: true });
onMounted(() => {
  renderCurves();
  renderPath();
  observer = new ResizeObserver(() => {
    curveCharts.forEach((chart) => chart.resize());
    pathChart?.resize();
  });
  Object.values(curveElements.value).forEach((element) => element && observer?.observe(element));
  if (pathElement.value) observer.observe(pathElement.value);
});
onBeforeUnmount(() => {
  observer?.disconnect();
  curveCharts.forEach((chart) => chart.dispose());
  pathChart?.dispose();
});
</script>

<template>
  <section class="v32-lab" aria-label="V3.2只读模型审计">
    <header class="lab-head">
      <div>
        <p class="kicker">V3.2 · IMMUTABLE AUDIT BASELINE</p>
        <h2>V3.2只读历史对照</h2>
        <p>保留原始399006与NDX模型、曲线和审计记录。NDX仅作为底层指数历史基准，不代表159941仓位，也不会被静默迁移。</p>
      </div>
      <div class="archive-seal" data-testid="v32-readonly-badge">
        <span>READ ONLY</span>
        <b>{{ selectedAuditMarket }}</b>
        <small>{{ selected === "159941" ? "NDX底层基准" : "原市场审计" }}</small>
      </div>
    </header>

    <p v-if="trainingError[selectedAuditMarket]" class="lab-error">{{ trainingError[selectedAuditMarket] }}</p>
    <p v-if="analysisError" class="lab-error">{{ analysisError }}</p>
    <p v-if="run?.status === 'failed'" class="lab-error">
      {{ run.current_stage }} · {{ run.error_code }} · {{ run.error_message }}
    </p>

    <div class="training-toolbar">
      <div>
        <b>双市场真实训练曲线</b>
        <span>不平滑、不补零；最近13轮尚未成熟时保持空白。</span>
      </div>
      <div class="range-tabs" aria-label="迭代曲线范围">
        <button v-for="item in ranges" :key="item" type="button" :aria-pressed="range === item" @click="range = item">
          {{ item === 0 ? "全部" : `最近${item}次` }}
        </button>
      </div>
    </div>

    <div class="curve-grid">
      <article v-for="market in (['399006', 'NDX'] as LegacyV32InstrumentCode[])" :key="market" class="curve-card">
        <header>
          <div><span>{{ market }}</span><h3>{{ names[market] }}</h3></div>
          <dl>
            <div><dt>正式迭代</dt><dd>{{ statuses[market]?.iteration_count ?? 0 }}</dd></div>
            <div><dt>最新Champion</dt><dd>{{ statuses[market]?.champion_weekly_version ?? "尚未训练" }}</dd></div>
            <div><dt>训练周</dt><dd>{{ statuses[market]?.last_training_week_key ?? "—" }}</dd></div>
          </dl>
        </header>
        <div
          :ref="(element) => setCurveElement(market, element)"
          :data-testid="`v32-curve-${market}`"
          class="training-curve"
        ></div>
        <footer>
          <span>{{ statuses[market]?.is_training ? statuses[market]?.latest_run?.current_stage ?? "正在建立训练任务" : "训练空闲" }}</span>
          <time>{{ statuses[market]?.last_successful_training_at ?? "尚无成功训练" }}</time>
        </footer>
      </article>
    </div>

    <template v-if="selected === '399006' && result">
      <section class="analysis-summary" data-testid="v32-analysis-result">
        <article class="market-verdict">
          <p class="kicker">13-WEEK MARKET VERDICT</p>
          <h3>{{ result.weekly.market_state }}</h3>
          <div class="probability-row">
            <span><b>{{ format(result.path.up_probability, 1) }}%</b>上涨</span>
            <span><b>{{ format(result.path.sideways_probability, 1) }}%</b>震荡</span>
            <span><b>{{ format(result.path.down_probability, 1) }}%</b>下跌</span>
          </div>
          <dl>
            <div><dt>周K判断</dt><dd>{{ result.weekly.state }}</dd></div>
            <div><dt>日K修正</dt><dd>{{ result.daily.state }}</dd></div>
            <div><dt>估值分位</dt><dd>{{ format(result.weekly.features.valuation_percentile, 1) }}%</dd></div>
            <div><dt>周K MACD差</dt><dd>{{ format(result.weekly.features.macd_spread, 5) }}</dd></div>
            <div><dt>100日 RSI</dt><dd>{{ format(result.daily.features.rsi, 1) }}</dd></div>
          </dl>
        </article>
        <article class="position-verdict">
          <p class="kicker">POSITION DECISION</p>
          <h3>{{ result.advice.direction === "increase" ? "分批增加仓位" : result.advice.direction === "decrease" ? "分批减少仓位" : "维持当前仓位" }}</h3>
          <div class="position-arrow">
            <b>{{ result.advice.current_position }}%</b><i>→</i><b>{{ result.advice.final_target_position }}%</b>
          </div>
          <p>基金 : ETF = <strong>{{ result.advice.fund_etf_ratio }}</strong></p>
          <small>仓位建议全部按5个百分点取整，后续批次只有满足触发条件才执行。</small>
        </article>
      </section>

      <section class="forecast-path">
        <header>
          <div><p class="kicker">FUTURE 13 WEEKS</p><h3>概率路径而非单点涨跌</h3></div>
          <span>模型 {{ result.model.weekly.version }} · 第 {{ result.model.weekly.iteration_number }} 次</span>
        </header>
        <div ref="pathElement" class="path-chart" data-testid="v32-path-chart"></div>
      </section>

      <section class="batch-plan" data-testid="v32-batches">
        <header><p class="kicker">CONDITIONAL CALENDAR</p><h3>具体日期窗口与仓位百分比</h3></header>
        <p v-if="!result.advice.batches.length" class="empty-plan">当前无需调整仓位，等待下一个完整交易周复核。</p>
        <ol v-else>
          <li v-for="batch in result.advice.batches" :key="batch.sequence">
            <span class="sequence">{{ String(batch.sequence).padStart(2, "0") }}</span>
            <time>{{ batch.execution_window.join(" 至 ") }}</time>
            <strong>{{ batch.direction === "increase" ? "买入" : "卖出" }} {{ batch.position_points }}%</strong>
            <span>执行后仓位 {{ batch.position_after }}%</span>
            <small>触发：{{ batch.trigger }}</small>
            <small>失效：{{ batch.invalidation }}</small>
          </li>
        </ol>
      </section>
    </template>

    <div v-else class="analysis-empty">
      <b>历史对照：{{ names[selectedAuditMarket] }}</b>
      <span v-if="selected === '159941'">159941使用独立V3.3模型；这里的NDX仅供底层指数历史审计，不参与仓位账本。</span>
      <span v-else>该区域冻结展示 {{ statuses[selectedAuditMarket]?.champion_weekly_version ?? "V3.2历史版本" }}，不会继续训练或写入。</span>
    </div>
  </section>
</template>

<style scoped>
.v32-lab { margin-top: 16px; border: 1px solid #aaa28f; background: #fffdf7; box-shadow: 0 18px 48px rgba(23,43,58,.1); }
.lab-head { display: flex; justify-content: space-between; gap: 30px; padding: 27px; border-bottom: 1px solid #aaa28f; }
.lab-head h2,.curve-card h3,.analysis-summary h3,.forecast-path h3,.batch-plan h3 { margin: 0; font-family: "Noto Serif SC", Georgia, serif; color: #172b3a; }
.lab-head h2 { font-size: clamp(28px,3vw,38px); }.lab-head p:not(.kicker) { max-width: 760px; margin: 8px 0 0; color: #65747a; font-size: 15px; }
.archive-seal { min-width: 170px; padding: 12px 15px; border: 1px solid #172b3a; color: #172b3a; background: #f1ede3; box-shadow: 4px 4px 0 #172b3a; text-align: right; }
.archive-seal span,.archive-seal small { display: block; color: #69767c; font-size: 10px; letter-spacing: .12em; }.archive-seal b { display: block; margin: 7px 0 4px; font: 800 20px/1 ui-monospace,monospace; }
.lab-error { margin: 0; padding: 12px 25px; color: #8b2f26; border-bottom: 1px solid #dfb0a8; background: #f9e7e2; }
.training-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 16px 24px; background: #f1ede3; border-bottom: 1px solid #aaa28f; }.training-toolbar b,.training-toolbar span { display: block; }.training-toolbar span { margin-top: 3px; color: #69767c; font-size: 12px; }
.range-tabs { display: flex; gap: 3px; padding: 3px; border: 1px solid #bbb29f; background: #fffdf7; }.range-tabs button { padding: 7px 10px; border: 0; color: #66747a; background: transparent; }.range-tabs button[aria-pressed="true"] { color: white; background: #172b3a; }
.curve-grid { display: grid; grid-template-columns: 1fr 1fr; }.curve-card:first-child { border-right: 1px solid #aaa28f; }.curve-card header { display: flex; justify-content: space-between; gap: 18px; min-height: 94px; padding: 18px 20px 8px; }.curve-card header span { color: #11697b; font: 800 11px/1.2 ui-monospace,monospace; letter-spacing: .14em; }.curve-card h3 { font-size: 23px; }.curve-card dl { display: flex; gap: 16px; margin: 0; }.curve-card dl div { display: block; padding: 0; border: 0; }.curve-card dt { color: #778187; font-size: 10px; }.curve-card dd { max-width: 150px; margin: 4px 0 0; overflow: hidden; color: #172b3a; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
.training-curve { height: 350px; }.curve-card footer { display: flex; justify-content: space-between; gap: 12px; padding: 9px 18px 14px; color: #69767c; font-size: 11px; }.curve-card footer time { text-align: right; }
.analysis-summary { display: grid; grid-template-columns: 1.25fr .75fr; border-top: 1px solid #aaa28f; border-bottom: 1px solid #aaa28f; }.analysis-summary article { padding: 25px; }.market-verdict { border-right: 1px solid #aaa28f; }.analysis-summary h3 { font-size: 29px; }.probability-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 18px 0; }.probability-row span { padding: 10px; border: 1px solid #d4cbbb; color: #69767c; text-align: center; font-size: 12px; }.probability-row b { display: block; color: #172b3a; font: 700 24px/1.1 Georgia,serif; }.market-verdict dl { display: grid; grid-template-columns: 1fr 1fr; gap: 0 18px; margin: 0; }.market-verdict dl div { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e5ded1; }.market-verdict dd { margin: 0; font-weight: 800; }
.position-verdict { color: #fffdf7; background: #172b3a; }.position-verdict .kicker { color: #83bfca; }.position-verdict h3 { color: white; }.position-arrow { display: flex; justify-content: space-between; align-items: center; margin: 24px 0; padding: 18px 0; border-block: 1px solid rgba(255,255,255,.22); }.position-arrow b { font: 700 44px/1 Georgia,serif; }.position-arrow i { color: #f4bf61; font-size: 24px; }.position-verdict p { color: #c8d2d5; }.position-verdict strong { color: #f4bf61; font-size: 22px; }.position-verdict small { color: #aebbc0; line-height: 1.6; }
.forecast-path { padding: 24px; border-bottom: 1px solid #aaa28f; }.forecast-path header { display: flex; justify-content: space-between; gap: 20px; }.forecast-path h3,.batch-plan h3 { font-size: 27px; }.forecast-path header>span { color: #68767c; font-size: 12px; }.path-chart { height: 380px; }
.batch-plan { padding: 24px; }.batch-plan ol { margin: 17px 0 0; padding: 0; list-style: none; }.batch-plan li { display: grid; grid-template-columns: 44px 190px 130px 1fr; gap: 7px 16px; align-items: center; padding: 15px 0; border-top: 1px solid #ddd5c7; }.sequence { color: #11697b; font: 700 22px Georgia,serif; }.batch-plan strong { color: #172b3a; }.batch-plan small { grid-column: 2/-1; color: #68767c; }.empty-plan,.analysis-empty { color: #68767c; text-align: center; }.analysis-empty { display: grid; min-height: 150px; place-content: center; gap: 6px; }.analysis-empty b { color: #172b3a; }
@media (max-width: 1050px) { .curve-grid,.analysis-summary { grid-template-columns: 1fr; }.curve-card:first-child,.market-verdict { border-right: 0; border-bottom: 1px solid #aaa28f; } }
@media (max-width: 700px) { .lab-head,.training-toolbar,.forecast-path header { display: block; }.archive-seal { margin-top: 18px; text-align: left; }.range-tabs { margin-top: 12px; }.curve-card header { display: block; }.curve-card dl { margin-top: 12px; }.training-curve { height: 320px; }.market-verdict dl { grid-template-columns: 1fr; }.batch-plan li { grid-template-columns: 36px 1fr; }.batch-plan li>* { grid-column: 2; }.batch-plan .sequence { grid-column: 1; grid-row: 1/4; }.batch-plan small { grid-column: 2; } }
</style>
