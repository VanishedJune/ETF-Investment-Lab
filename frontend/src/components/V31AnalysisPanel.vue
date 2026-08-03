<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import type { LatestAdvice, ModelMetrics, V31Analysis, V31Run } from "../types/research";

const props = defineProps<{
  result: V31Analysis | null;
  run: V31Run | null;
  legacyMetrics: ModelMetrics | null;
  legacyAdvice: LatestAdvice | null;
  loading: boolean;
  error?: string;
}>();

const emit = defineEmits<{ analyze: [] }>();
const chartElement = ref<HTMLDivElement>();
let chart: echarts.ECharts | undefined;
let observer: ResizeObserver | undefined;

const stageNames = ["等待执行", "刷新行情", "校验数据", "计算周K信号", "计算日K修正", "生成概率路径", "读取当前仓位", "生成投资建议", "已完成"];
const directionLabels: Record<string, string> = { up: "上涨", down: "下跌", sideways: "震荡" };
const actionLabels: Record<string, string> = { buy: "增加仓位", sell: "减少仓位", hold: "维持仓位" };
const direction = computed(() => directionLabels[props.result?.weekly.direction] ?? "等待分析");
const action = computed(() => actionLabels[props.result?.advice.side] ?? "等待建议");

function text(value: unknown, suffix = ""): string {
  if (value === null || value === undefined || value === "") return "—";
  return `${value}${suffix}`;
}

function metric(window: "recent_20" | "recent_52" | "all", field: string): string {
  const value = props.result?.evaluation?.[window]?.[field];
  return value === null || value === undefined ? "等待成熟样本" : String(value);
}

function renderChart() {
  if (!chartElement.value) return;
  chart ??= echarts.init(chartElement.value);
  const points = props.result?.path?.points ?? [];
  if (!points.length) {
    chart.clear();
    return;
  }
  const p10 = points.map((point) => point.p10_cumulative_return);
  const p90 = points.map((point) => point.p90_cumulative_return);
  chart.setOption({
    animationDuration: 350,
    color: ["#195f74", "#c6782c", "#b84b3e", "#438577"],
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => `${Number(value).toFixed(2)}%` },
    legend: { top: 4, right: 8, data: ["P10", "P50", "预期路径", "P90"] },
    grid: { left: 56, right: 24, top: 46, bottom: 42 },
    xAxis: { type: "category", name: "未来周数", data: points.map((point) => `第${point.horizon_week}周`), axisLine: { lineStyle: { color: "#a9a18f" } } },
    yAxis: { type: "value", name: "累计收益", axisLabel: { formatter: "{value}%" }, splitLine: { lineStyle: { color: "#e6dfd2" } } },
    series: [
      { name: "P10", type: "line", data: p10, showSymbol: false, lineStyle: { width: 1.5, type: "dashed" } },
      { name: "P50", type: "line", data: points.map((point) => point.p50_cumulative_return), showSymbol: true, symbolSize: 5, lineStyle: { width: 2.6 } },
      { name: "预期路径", type: "line", data: points.map((point) => point.expected_cumulative_return), showSymbol: false, lineStyle: { width: 2 } },
      { name: "P90", type: "line", data: p90, showSymbol: false, lineStyle: { width: 1.5, type: "dashed" } },
    ],
  }, true);
}

watch(() => props.result, () => void nextTick(renderChart), { deep: true });
onMounted(() => {
  renderChart();
  if (chartElement.value) {
    observer = new ResizeObserver(() => chart?.resize());
    observer.observe(chartElement.value);
  }
});
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose(); });
</script>

<template>
  <section class="v31-panel" aria-label="V3.1概率路径分析">
    <header class="analysis-head">
      <div>
        <p class="kicker">V3.1 · WEEKLY CORE / DAILY 100 CORRECTOR</p>
        <h2>未来13周概率路径与仓位计划</h2>
        <p>完整周K决定中期方向；最近100个交易日日K只修正节奏、仓位与置信度。</p>
      </div>
      <button data-testid="v31-analyze" type="button" :disabled="loading" @click="emit('analyze')">
        {{ loading ? "刷新并分析中…" : "数据分析" }}
      </button>
    </header>

    <div class="stage-rail" data-testid="v31-stage-rail">
      <span
        v-for="stage in stageNames"
        :key="stage"
        :class="{
          done: run?.stages?.some((item) => item.stage === stage && item.status === 'completed'),
          failed: run?.stages?.some((item) => item.stage === stage && item.status === 'failed'),
        }"
      >{{ stage }}</span>
    </div>

    <article v-if="run?.status === 'failed'" class="failure-card" data-testid="v31-failure">
      <div><span>失败阶段</span><b>{{ run.current_stage }}</b></div>
      <div><span>错误代码</span><b>{{ run.error_code ?? "UNKNOWN" }}</b></div>
      <div><span>本次运行ID</span><b>{{ run.id }}</b></div>
      <p>{{ run.error_message }}</p>
      <small>数据门禁未通过，本次不生成投资建议。备用数据源默认不允许替代直接指数。</small>
    </article>
    <div v-else-if="error" class="failure-card"><p>{{ error }}</p></div>

    <template v-if="result">
      <section class="freshness-grid" data-testid="v31-freshness">
        <div><span>最新可用交易日</span><b>{{ result.freshness.latest_available_trade_date }}</b></div>
        <div><span>行情截止日</span><b>{{ result.freshness.price_data_as_of }}</b></div>
        <div><span>估值截止日</span><b>{{ result.freshness.valuation_data_as_of ?? "不可用" }}</b></div>
        <div><span>完整周K截止日</span><b>{{ result.freshness.latest_complete_week_as_of }}</b></div>
        <div><span>当前周预览截止</span><b>{{ result.freshness.incomplete_week_data_as_of ?? "无" }}</b></div>
        <div><span>成交量来源</span><b>{{ result.freshness.volume_source }}</b></div>
        <div :class="['gate-state', result.data_gate.status]"><span>数据门禁</span><b>{{ result.data_gate.status === "passed" ? "通过" : "降级运行" }}</b></div>
      </section>
      <p v-if="result.data_gate.degraded_reasons?.length" class="degraded">
        降级项：{{ result.data_gate.degraded_reasons.join("；") }} · 置信度扣减 {{ result.data_gate.confidence_deduction }} 个百分点
      </p>

      <div class="signal-grid">
        <article class="weekly-card">
          <p class="kicker">WEEKLY DIRECTION</p>
          <div class="direction-line"><h3>{{ direction }}</h3><b>{{ result.weekly.confidence }}%</b></div>
          <p class="market-state">{{ result.weekly.market_state }} · {{ result.weekly.state }}</p>
          <dl>
            <div><dt>周K综合评分</dt><dd>{{ result.weekly.score }}</dd></div>
            <div><dt>基础目标仓位</dt><dd>{{ result.weekly.base_target_position }}%</dd></div>
            <div><dt>仓位区间</dt><dd>{{ result.weekly.position_range.join("%—") }}%</dd></div>
            <div><dt>DIF / DEA</dt><dd>{{ result.weekly.features.dif }} / {{ result.weekly.features.dea }}</dd></div>
            <div><dt>MACD柱</dt><dd>{{ result.weekly.features.macd_histogram }}</dd></div>
            <div><dt>估值分位</dt><dd>{{ text(result.weekly.features.valuation_percentile, "%") }}</dd></div>
          </dl>
          <div class="factor-columns">
            <ul><li v-for="item in result.weekly.positive_factors" :key="item">{{ item }}</li></ul>
            <ul class="risks"><li v-for="item in result.weekly.risk_factors" :key="item">{{ item }}</li></ul>
          </div>
        </article>

        <article class="daily-card">
          <p class="kicker">LATEST 100 SESSIONS</p>
          <div class="direction-line"><h3>{{ result.daily.state }}</h3><b>{{ result.daily.consistent_with_weekly ? "同向" : "分歧" }}</b></div>
          <p class="market-state">日K方向：{{ result.daily.direction }} · 执行节奏：{{ result.daily.execution_speed }}</p>
          <dl>
            <div><dt>置信度修正</dt><dd>{{ result.daily.confidence_adjustment > 0 ? "+" : "" }}{{ result.daily.confidence_adjustment }}点</dd></div>
            <div><dt>仓位修正</dt><dd>{{ result.daily.position_adjustment > 0 ? "+" : "" }}{{ result.daily.position_adjustment }}点</dd></div>
            <div><dt>RSI</dt><dd>{{ result.daily.features.rsi }}</dd></div>
            <div><dt>距20日线</dt><dd>{{ result.daily.features.ma20_distance }}%</dd></div>
            <div><dt>20日波动率</dt><dd>{{ result.daily.features.volatility_20d }}%</dd></div>
            <div><dt>首批窗口</dt><dd>{{ result.daily.first_execution_window.join(" 至 ") }}</dd></div>
          </dl>
          <p class="condition"><b>触发：</b>{{ result.daily.trigger_conditions.join("；") }}</p>
          <p class="condition"><b>失效：</b>{{ result.daily.invalidation_conditions.join("；") }}</p>
        </article>
      </div>

      <section class="path-section">
        <div class="path-heading">
          <div><p class="kicker">PROBABILITY RIBBON</p><h3>未来13周累计收益路径</h3></div>
          <div class="probabilities">
            <span><b>{{ result.path.up_probability }}%</b>上涨</span>
            <span><b>{{ result.path.sideways_probability }}%</b>震荡</span>
            <span><b>{{ result.path.down_probability }}%</b>下跌</span>
          </div>
        </div>
        <div ref="chartElement" class="path-chart" data-testid="v31-path-chart"></div>
        <div class="path-facts">
          <span>动态方向阈值 <b>{{ result.path.direction_threshold }}%</b></span>
          <span>预期最大回撤 <b>{{ result.path.expected_max_drawdown }}%</b></span>
          <span>阶段高点 <b>{{ result.path.high_week_range }}</b></span>
          <span>阶段低点 <b>{{ result.path.low_week_range }}</b></span>
          <span>历史情景 <b>{{ result.path.scenario_count }}组</b></span>
        </div>
      </section>

      <section class="advice-grid" data-testid="v31-advice">
        <article class="position-card">
          <p class="kicker">POSITION ACTION</p>
          <h3>{{ action }}</h3>
          <div class="position-flow"><b>{{ result.advice.current_position }}%</b><i>→</i><b>{{ result.advice.target_position }}%</b></div>
          <p>{{ result.advice.summary }}</p>
          <dl>
            <div><dt>周K基础仓位</dt><dd>{{ result.advice.weekly_base_target_position }}%</dd></div>
            <div><dt>日K修正</dt><dd>{{ result.advice.daily_position_adjustment }}点</dd></div>
            <div><dt>模型置信度</dt><dd>{{ result.advice.confidence }}%</dd></div>
            <div><dt>基金:ETF目标</dt><dd>{{ result.advice.fund_etf.target_ratio }}</dd></div>
            <div><dt>取整后配置</dt><dd>基金{{ result.advice.fund_etf.fund_position }}% / ETF{{ result.advice.fund_etf.etf_position }}%</dd></div>
          </dl>
        </article>
        <article class="batches-card">
          <p class="kicker">CONDITIONAL EXECUTION</p>
          <h3>条件化分批计划</h3>
          <p v-if="!result.advice.batches.length" class="empty">当前维持0%仓位；只有周K转为看多且日K满足确认条件后才开始建仓。</p>
          <ol v-else>
            <li v-for="batch in result.advice.batches" :key="batch.sequence">
              <time>{{ batch.execution_window.join(" 至 ") }}</time>
              <strong>{{ batch.side === "buy" ? "增加" : "减少" }} {{ batch.position_points }} 个百分点</strong>
              <span>执行后 {{ batch.position_after }}%</span>
              <small>触发：{{ batch.trigger_condition }}</small>
              <small>失效：{{ batch.invalidation_condition }}</small>
            </li>
          </ol>
          <p class="execution-note">{{ result.advice.execution_note }}</p>
        </article>
      </section>

      <section class="audit-section">
        <div class="audit-heading"><p class="kicker">OUT-OF-SAMPLE ONLY</p><h3>真实成熟预测误差</h3></div>
        <div class="audit-grid">
          <article v-for="window in ['recent_20', 'recent_52', 'all']" :key="window">
            <h4>{{ window === "recent_20" ? "最近20次" : window === "recent_52" ? "最近52次" : "全部成熟预测" }}</h4>
            <span>样本数 <b>{{ result.evaluation[window].sample_count }}</b></span>
            <span>标准化路径误差 <b>{{ metric(window as any, "path_error") }}</b></span>
            <span>第13周收益误差 <b>{{ metric(window as any, "terminal_return_error") }}</b></span>
            <span>P10—P90覆盖率 <b>{{ metric(window as any, "interval_coverage") }}</b></span>
            <span>高/低点周偏离 <b>{{ metric(window as any, "high_week_error") }} / {{ metric(window as any, "low_week_error") }}</b></span>
          </article>
        </div>
        <p class="audit-note">{{ result.evaluation.note }}。迭代次数不再作为模型质量指标。</p>
        <div class="version-compare">
          <span>V3.1 Champion <b>{{ result.model.weekly.version }}</b></span>
          <span>V3.1候选 <b>{{ result.model.challenger.candidate_count }} 个</b></span>
          <span>本次晋级 <b>{{ result.model.challenger.promoted ? "是" : "否" }}</b></span>
          <span>V2审计基线 <b>{{ legacyMetrics?.model_version ?? legacyAdvice?.model_version ?? "尚无" }}</b></span>
        </div>
      </section>
    </template>

    <div v-else-if="!loading && run?.status !== 'failed'" class="empty-state">
      点击“数据分析”后，系统先刷新直接指数行情并通过门禁，再生成路径与仓位计划。
    </div>
  </section>
</template>

<style scoped>
.v31-panel { margin-top: 18px; border: 1px solid #b9b09e; background: #fffdf7; box-shadow: 0 16px 44px rgba(23,43,58,.09); }
.analysis-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 28px; padding: 26px; border-bottom: 1px solid #b9b09e; }
.analysis-head h2, h3, h4 { margin: 0; color: #172b3a; font-family: "Noto Serif SC", Georgia, serif; }
.analysis-head h2 { font-size: clamp(25px, 3vw, 34px); }
.analysis-head p:not(.kicker) { max-width: 760px; margin: 8px 0 0; color: #66747a; }
.analysis-head button { min-width: 150px; padding: 13px 20px; border: 1px solid #172b3a; color: #fffdf7; background: #172b3a; box-shadow: 4px 4px 0 #d88b2c; font-weight: 900; }
.analysis-head button:disabled { opacity: .58; }
.kicker { margin: 0 0 6px; color: #2c7180; font: 800 11px/1.3 ui-monospace, monospace; letter-spacing: .12em; }
.stage-rail { display: grid; grid-template-columns: repeat(9, 1fr); border-bottom: 1px solid #b9b09e; background: #f2eee4; }
.stage-rail span { position: relative; padding: 12px 5px; color: #7f817d; border-right: 1px solid #d8d0c1; font-size: 11px; text-align: center; }
.stage-rail span.done { color: #12687e; background: #e5efed; font-weight: 800; }
.stage-rail span.failed { color: #9b352a; background: #f8e7e1; font-weight: 800; }
.failure-card { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 20px 24px; border-bottom: 1px solid #d7a89e; color: #8e3027; background: #f8e9e4; }
.failure-card span { display: block; font-size: 12px; opacity: .72; }.failure-card b { display: block; margin-top: 3px; }.failure-card p,.failure-card small { grid-column: 1/-1; margin: 0; }
.freshness-grid { display: grid; grid-template-columns: repeat(7, 1fr); border-bottom: 1px solid #b9b09e; }
.freshness-grid div { min-width: 0; padding: 14px; border-right: 1px solid #dfd7c8; }.freshness-grid span { display: block; color: #758086; font-size: 11px; }.freshness-grid b { display: block; overflow: hidden; margin-top: 6px; color: #263943; font-size: 13px; text-overflow: ellipsis; }.gate-state.passed b { color: #16705f; }.gate-state.degraded b { color: #a76726; }
.degraded { margin: 0; padding: 10px 24px; color: #845a28; border-bottom: 1px solid #dec69e; background: #fbf2dd; font-size: 13px; }
.signal-grid { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid #b9b09e; }.signal-grid article { padding: 24px; }.weekly-card { border-right: 1px solid #b9b09e; }
.direction-line { display: flex; justify-content: space-between; gap: 18px; align-items: baseline; }.direction-line h3 { font-size: 29px; }.direction-line > b { color: #c6782c; font: 700 25px Georgia, serif; }.market-state { margin: 7px 0 18px; color: #66747a; }
dl { margin: 0; }dl div { display: flex; justify-content: space-between; gap: 16px; padding: 8px 0; border-bottom: 1px solid #e5ded1; }dt { color: #6f7a7f; }dd { margin: 0; color: #172b3a; font-weight: 800; text-align: right; }
.factor-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 16px; }.factor-columns ul { margin: 0; padding-left: 18px; color: #16705f; }.factor-columns .risks { color: #a84c3e; }.condition { margin: 12px 0 0; color: #53656d; font-size: 13px; }
.path-section { padding: 24px; border-bottom: 1px solid #b9b09e; }.path-heading { display: flex; justify-content: space-between; gap: 24px; }.path-heading h3 { font-size: 27px; }.probabilities { display: flex; gap: 10px; }.probabilities span { min-width: 84px; padding: 9px 12px; border: 1px solid #c9c0af; color: #69747a; font-size: 11px; text-align: center; }.probabilities b { display: block; color: #172b3a; font: 700 20px Georgia, serif; }.path-chart { height: 390px; }.path-facts { display: flex; flex-wrap: wrap; gap: 8px; }.path-facts span { padding: 7px 10px; color: #66747a; background: #f1ede4; font-size: 12px; }.path-facts b { color: #172b3a; }
.advice-grid { display: grid; grid-template-columns: .72fr 1.28fr; border-bottom: 1px solid #b9b09e; }.position-card { padding: 24px; color: #fffdf7; background: #172b3a; }.position-card .kicker { color: #8ec4ce; }.position-card h3 { color: #fffdf7; font-size: 30px; }.position-flow { display: flex; align-items: center; justify-content: space-between; margin: 18px 0; padding: 17px 0; border-block: 1px solid rgba(255,255,255,.2); }.position-flow b { font: 700 42px Georgia, serif; }.position-flow i { color: #f0bd6b; font-size: 24px; }.position-card p { color: #f0bd6b; }.position-card dl div { border-color: rgba(255,255,255,.14); }.position-card dt { color: #b9c4c8; }.position-card dd { color: #fffdf7; }
.batches-card { padding: 24px; }.batches-card h3 { font-size: 27px; }.batches-card ol { margin: 16px 0 0; padding: 0; list-style: none; counter-reset: batch; }.batches-card li { display: grid; grid-template-columns: 170px 1fr auto; gap: 7px 14px; padding: 14px 0; border-top: 1px solid #e2dacd; }.batches-card time { font-weight: 800; }.batches-card strong { color: #172b3a; }.batches-card span,.batches-card small { color: #69747a; }.batches-card small { grid-column: 2/-1; }.empty { display: grid; min-height: 190px; place-items: center; color: #69747a; text-align: center; }.execution-note { margin: 14px 0 0; color: #8c5d49; font-size: 12px; }
.audit-section { padding: 24px; }.audit-heading h3 { font-size: 27px; }.audit-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }.audit-grid article { padding: 16px; border: 1px solid #c9c0af; background: #f7f3ea; }.audit-grid h4 { margin-bottom: 10px; }.audit-grid span { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; color: #69747a; font-size: 12px; }.audit-grid b { color: #172b3a; }.audit-note { color: #69747a; font-size: 12px; }.version-compare { display: flex; flex-wrap: wrap; gap: 8px; padding-top: 14px; border-top: 1px solid #d9d1c3; }.version-compare span { padding: 7px 10px; color: #69747a; background: #e8efed; font-size: 12px; }.version-compare b { color: #172b3a; }.empty-state { padding: 60px 24px; color: #69747a; text-align: center; }
@media (max-width: 1050px) { .freshness-grid { grid-template-columns: repeat(4,1fr); }.stage-rail { grid-template-columns: repeat(3,1fr); }.signal-grid,.advice-grid { grid-template-columns: 1fr; }.weekly-card { border-right: 0; border-bottom: 1px solid #b9b09e; }.audit-grid { grid-template-columns: 1fr; } }
@media (max-width: 650px) { .analysis-head,.path-heading { display: block; }.analysis-head button { width: 100%; margin-top: 16px; }.freshness-grid { grid-template-columns: repeat(2,1fr); }.probabilities { margin-top: 14px; }.probabilities span { min-width: 0; flex: 1; }.path-chart { height: 330px; }.failure-card { grid-template-columns: 1fr; }.failure-card p,.failure-card small { grid-column: auto; }.factor-columns { grid-template-columns: 1fr; }.batches-card li { grid-template-columns: 1fr; }.batches-card small { grid-column: auto; } }
</style>
