<script setup lang="ts">
import * as echarts from "echarts";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import {
  loadV351Champion,
  loadV351Forecast,
  loadV351Simulation,
  loadV351Status,
  type V351ForecastPayload,
} from "../v351-api";

const props = defineProps<{
  selected: string | null;
}>();

const statuses = ref<Awaited<ReturnType<typeof loadV351Status>>>([]);
const forecast = ref<V351ForecastPayload | null>(null);
const simulation = ref<Awaited<ReturnType<typeof loadV351Simulation>> | null>(null);
const champion = ref<Awaited<ReturnType<typeof loadV351Champion>> | null>(null);
const loading = ref(false);
const error = ref("");

const marketStatus = computed(
  () => statuses.value.find((item) => item.market === props.selected) ?? null,
);

function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function pctCell(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  if (number < -1) return "-100%*";
  if (number > 5) return "+500%*";
  return `${(number * 100).toFixed(2)}%`;
}

function money(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

const forecastRows = computed(() =>
  (forecast.value?.expected_path ?? []).map((value, index) => ({
    week: index + 1,
    expected: value,
    p10: forecast.value?.price_quantiles?.p10?.[index] ?? null,
    p50: forecast.value?.price_quantiles?.p50?.[index] ?? null,
    p90: forecast.value?.price_quantiles?.p90?.[index] ?? null,
  })),
);

const turning = computed(() => forecast.value?.policy?.turning_assessment ?? null);
const decision = computed(() => forecast.value?.policy?.decision ?? null);
const strategy = computed(() => forecast.value?.strategy ?? null);

function confirmedWindow(signal: "PRICE" | "DIF"): string {
  const candidate =
    turning.value?.candidates.find(
      (item) =>
        item.signal_kind === signal &&
        item.classification === "VALID_TURN" &&
        item.confirmation_status === "CONFIRMED",
    ) ?? null;
  return candidate
    ? `${candidate.window_start_date}–${candidate.window_end_date} · ${candidate.turn_kind}`
    : "尚无可执行拐点窗口";
}

async function loadAll() {
  const market = props.selected;
  if (!market) return;
  loading.value = true;
  error.value = "";
  forecast.value = null;
  try {
    const [statusResult, forecastResult, simulationResult, championResult] =
      await Promise.all([
        loadV351Status(),
        loadV351Forecast(market),
        loadV351Simulation(market),
        loadV351Champion(market),
      ]);
    statuses.value = statusResult;
    forecast.value = forecastResult;
    simulation.value = simulationResult;
    champion.value = championResult;
    await nextTick();
    renderForecast();
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    loading.value = false;
  }
}

const forecastElement = ref<HTMLDivElement>();
let forecastChart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;

function renderForecast() {
  const element = forecastElement.value;
  if (!element) return;
  forecastChart?.dispose();
  forecastChart = echarts.init(element);
  const payload = forecast.value;
  if (!payload) {
    forecastChart.setOption({
      title: { text: "等待 v351 预测数据", left: "center", top: "middle" },
    });
    return;
  }
  const history = payload.historical_ohlcv ?? [];
  const predicted = payload.representative_ohlcv ?? [];
  const baseClose =
    history.length > 0 ? Number(history[history.length - 1].close) : 1;
  const dates = [
    ...history.map((row) => row.week_end),
    ...predicted.map((row) => `W${row.week}`),
  ];
  const candles = [
    ...history.map((row) => [
      row.open,
      row.close,
      row.low,
      row.high,
    ]),
    ...predicted.map((row) => [row.open, row.close, row.low, row.high]),
  ];
  const p10 = [
    ...history.map(() => null),
    ...(payload.price_quantiles?.p10 ?? []).map(
      (value) => baseClose * (1 + value),
    ),
  ];
  const band = [
    ...history.map(() => null),
    ...(payload.price_quantiles?.p90 ?? []).map(
      (value) => baseClose * (1 + value),
    ),
  ];
  const p50 = [
    ...history.map(() => null),
    ...predicted.map((row) => row.close),
  ];
  const volumes = [
    ...history.map((row) => row.volume ?? 0),
    ...predicted.map((row) => row.volume ?? 0),
  ];
  const indicators = [
    ...history.map(() => null),
    ...payload.indicators.map((row) => row),
  ];
  const dif = indicators.map((row) => row?.dif ?? null);
  const dea = indicators.map((row) => row?.dea ?? null);
  const macd = indicators.map((row) => row?.macd ?? null);
  const difFirst = indicators.map((row) => row?.dif_first_change ?? null);
  forecastChart.setOption(
    {
      animationDuration: 220,
      tooltip: { trigger: "axis" },
      legend: {
        top: 0,
        data: ["历史周K", "P50代表性预测周K", "P10—P90价格带", "预测成交量", "MACD柱", "DIF", "DEA", "DIF一阶变化"],
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: [
        { left: 60, right: 24, top: 36, height: "42%" },
        { left: 60, right: 24, top: "52%", height: "12%" },
        { left: 60, right: 24, top: "68%", height: "24%" },
      ],
      xAxis: [
        { type: "category", data: dates, gridIndex: 0 },
        { type: "category", data: dates, gridIndex: 1 },
        { type: "category", data: dates, gridIndex: 2 },
      ],
      yAxis: [
        { scale: true, gridIndex: 0 },
        { scale: true, gridIndex: 1 },
        { scale: true, gridIndex: 2 },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1, 2], start: 40, end: 100 },
        { type: "slider", xAxisIndex: [0, 1, 2], bottom: 0, start: 40, end: 100 },
      ],
      series: [
        {
          name: "历史周K",
          type: "candlestick",
          data: candles.slice(0, history.length),
          itemStyle: { color: "#e05b4d", color0: "#3fa37c", borderColor: "#b03a2e", borderColor0: "#2c7d5c" },
        },
        {
          name: "P50代表性预测周K",
          type: "candlestick",
          data: candles.slice(history.length),
          itemStyle: { color: "#f0a54a", color0: "#4f91a6", borderColor: "#b56616", borderColor0: "#12687e" },
          markLine: {
            silent: true,
            symbol: "none",
            label: { formatter: "历史 / 预测", color: "#8b4e12" },
            lineStyle: { color: "#b56616", type: "dashed", width: 2 },
            data: [{ xAxis: history.length - 0.5 }],
          },
          markPoint: {
            symbolSize: 30,
            data: [1, 4, 8].map((week) => ({
              name: `第${week}周`,
              coord: [history.length + week - 1, predicted[week - 1]?.high],
              value: `W${week}`,
            })),
          },
        },
        { name: "P10", type: "line", stack: "band", symbol: "none", data: p10, lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 } },
        { name: "P10—P90价格带", type: "line", stack: "band", symbol: "none", data: band, lineStyle: { opacity: 0 }, areaStyle: { color: "rgba(18,104,126,.20)" } },
        { name: "P50", type: "line", symbol: "none", data: p50, lineStyle: { color: "#12687e", width: 2 } },
        { name: "预测成交量", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: volumes, itemStyle: { color: "#668b96", opacity: .72 } },
        { name: "MACD柱", type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: macd, itemStyle: { color: (item: { value: number }) => item.value >= 0 ? "#d9553f" : "#14836d" } },
        { name: "DIF", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, data: dif, lineStyle: { color: "#12687e", width: 2 } },
        { name: "DEA", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, data: dea, lineStyle: { color: "#d88b2c", width: 2 } },
        { name: "DIF一阶变化", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, data: difFirst, lineStyle: { color: "#7b4ca0", width: 2 } },
      ],
    },
    true,
  );
}

watch(() => props.selected, () => void loadAll());
onMounted(() => {
  void loadAll();
  resizeObserver = new ResizeObserver(() => forecastChart?.resize());
  if (forecastElement.value) resizeObserver.observe(forecastElement.value);
});
onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  forecastChart?.dispose();
});
</script>

<template>
  <section class="v35-lab" data-testid="v35-overview-panel">
    <div class="v35-heading">
      <div>
        <p class="kicker">V3.5.1 CAPITAL DRIVEN · 8W</p>
        <h2>V3.5.1 资金收益驱动型 8 周分析</h2>
        <p class="panel-note">
          未来 8 周预测、分级交易策略、拐点/一致性检查、分批仓位建议与 10 万元模拟账户
          （本地计算 · 不调用 AI）。
        </p>
      </div>
      <button
        class="analyze-button"
        type="button"
        data-testid="v351-analyze"
        :disabled="loading"
        @click="loadAll"
      >
        {{ loading ? "正在分析…" : "数据分析" }}
      </button>
    </div>

    <div v-if="error" class="v35-error">{{ error }}</div>
    <div v-if="loading" class="v35-note">正在加载 v351 分析结果…</div>

    <div class="v35-grid">
      <section class="v35-card">
        <h3>Bootstrap 状态</h3>
        <dl>
          <div><dt>状态</dt><dd>{{ marketStatus?.state ?? "未启动" }}</dd></div>
          <div><dt>正式周迭代</dt><dd>{{ marketStatus?.weekly_iteration_count ?? 0 }}</dd></div>
          <div><dt>预测挑战</dt><dd>{{ marketStatus?.prediction_challenge_count ?? 0 }}</dd></div>
          <div><dt>策略挑战</dt><dd>{{ marketStatus?.strategy_challenge_count ?? 0 }}</dd></div>
          <div><dt>晋级次数</dt><dd>{{ marketStatus?.promotion_count ?? 0 }}</dd></div>
          <div><dt>当前 Champion</dt><dd>{{ champion?.champion?.id.split(":").slice(-1)[0] ?? "—" }}</dd></div>
          <div><dt>生效日期</dt><dd>{{ champion?.champion?.effective_from_date ?? "—" }}</dd></div>
          <div><dt>最后完成周</dt><dd>{{ marketStatus?.last_completed_anchor ?? "—" }}</dd></div>
        </dl>
      </section>

      <section class="v35-card">
        <h3>未来 8 周预测（累计收益）</h3>
        <div v-if="!forecast" class="v35-note">尚无 v351 预测。</div>
        <template v-else>
          <table class="v35-table" data-testid="v35-forecast-table">
            <thead>
              <tr><th>周</th><th>期望</th><th>P10</th><th>P50</th><th>P90</th></tr>
            </thead>
            <tbody>
              <tr v-for="row in forecastRows" :key="row.week">
                <td>{{ row.week }}W</td>
                <td>{{ pctCell(row.expected) }}</td>
                <td>{{ pctCell(row.p10) }}</td>
                <td>{{ pctCell(row.p50) }}</td>
                <td>{{ pctCell(row.p90) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="v35-note">* 模型场景未经物理边界约束时，越界值截断展示。</p>
          <dl class="v35-prob">
            <div>
              <dt>第4周</dt>
              <dd>
                涨 {{ pct(forecast.horizon_probabilities["4"]?.[0]) }} ·
                横 {{ pct(forecast.horizon_probabilities["4"]?.[1]) }} ·
                跌 {{ pct(forecast.horizon_probabilities["4"]?.[2]) }}
              </dd>
            </div>
            <div>
              <dt>第8周</dt>
              <dd>
                涨 {{ pct(forecast.horizon_probabilities["8"]?.[0]) }} ·
                横 {{ pct(forecast.horizon_probabilities["8"]?.[1]) }} ·
                跌 {{ pct(forecast.horizon_probabilities["8"]?.[2]) }}
              </dd>
            </div>
            <div><dt>健康</dt><dd>{{ forecast.health_status }}</dd></div>
          </dl>
        </template>
      </section>

      <section class="v35-card">
        <h3>分级交易策略</h3>
        <div v-if="!strategy" class="v35-note">尚无 v351 策略快照。</div>
        <template v-else>
          <dl>
            <div><dt>DIF 状态</dt><dd>{{ strategy.dif_trend_state }}</dd></div>
            <div><dt>确认状态</dt><dd>{{ strategy.confirmation_status }}</dd></div>
            <div><dt>StrategyScore</dt><dd>{{ strategy.strategy_score.toFixed(3) }}</dd></div>
            <div><dt>基础目标仓位</dt><dd>{{ strategy.base_target_position_pp }}%</dd></div>
            <div><dt>状态仓位上限</dt><dd>{{ strategy.state_position_cap_pp }}%</dd></div>
            <div class="primary"><dt>最终目标仓位</dt><dd>{{ strategy.final_target_position_pp }}%</dd></div>
          </dl>
          <p v-if="strategy.reasons?.length" class="v35-note">
            原因：{{ strategy.reasons.join("；") }}
          </p>
        </template>
      </section>

      <section class="v35-card">
        <h3>10 万元模拟账户</h3>
        <div v-if="!simulation" class="v35-note">尚无 v351 账户。</div>
        <template v-else>
          <dl v-if="simulation.continuous">
            <div><dt>连续账户期末</dt><dd>{{ money(simulation.continuous.ending_equity) }}</dd></div>
            <div><dt>累计收益</dt><dd>{{ pct(simulation.continuous.cumulative_return) }}</dd></div>
            <div><dt>年化收益</dt><dd>{{ pct(simulation.continuous.annualized_return) }}</dd></div>
            <div><dt>最大回撤</dt><dd>{{ pct(simulation.continuous.max_drawdown) }}</dd></div>
            <div><dt>平均仓位</dt><dd>{{ simulation.continuous.average_position_pp.toFixed(1) }}%</dd></div>
            <div><dt>相对满仓</dt><dd>{{ pct(simulation.continuous.buy_hold_return) }}</dd></div>
            <div><dt>相对固定30%</dt><dd>{{ pct(simulation.continuous.fixed_30_return) }}</dd></div>
          </dl>
          <p class="v35-note">8 周评价窗口：{{ simulation.evaluation_count }} 个已成熟</p>
          <table class="v35-table">
            <thead><tr><th>窗口</th><th>期末资产</th><th>净收益</th><th>平均仓位</th><th>空仓窗口</th></tr></thead>
            <tbody>
              <tr v-for="account in simulation.accounts.slice(0, 8)" :key="account.window_start ?? 'continuous'">
                <td>{{ account.window_start ?? "连续" }}</td>
                <td>{{ money(account.ending_equity) }}</td>
                <td>{{ pct(account.net_return) }}</td>
                <td>{{ account.average_position_pp.toFixed(1) }}%</td>
                <td>{{ account.no_action_window ? "是" : "否" }}</td>
              </tr>
            </tbody>
          </table>
        </template>
      </section>
    </div>

    <section v-if="forecast" class="forecast-card">
      <header>
        <div>
          <p class="kicker">HISTORY / FORECAST BOUNDARY</p>
          <h3>历史周K + 8根P50代表性预测周K</h3>
        </div>
        <span>{{ forecast.model_version }} · 场景 {{ forecast.scenario_count }} 条 · 锚点 {{ forecast.forecast_anchor_date }}</span>
      </header>
      <div ref="forecastElement" class="forecast-chart" data-testid="v351-forecast-chart" role="img"></div>
      <p class="semantic-note">
        半透明区域是逐周收盘价 P10–P90 边界分位带，并不保证上下边界可组成一条单独可实现路径。
      </p>
    </section>

    <section v-if="forecast" class="turning-grid">
      <article>
        <span>价格拐点窗口</span>
        <h3>{{ confirmedWindow("PRICE") }}</h3>
        <p>{{ turning?.price_turn_status ?? "等待 v351 分析" }}</p>
      </article>
      <article>
        <span>DIF一阶变化零点窗口</span>
        <h3>{{ confirmedWindow("DIF") }}</h3>
        <p>{{ turning?.dif_turn_status ?? "等待 v351 分析" }} · 独立计算</p>
      </article>
      <article :class="{ warning: turning?.consistency_status !== 'TEMPORALLY_CONSISTENT' }">
        <span>一致性检查</span>
        <h3>{{ turning?.consistency_status ?? "等待分析" }}</h3>
        <p>窗口极值不等于有效交易拐点；只有已确认且时间一致的价格/DIF信号才可进入仓位条件。</p>
      </article>
    </section>

    <section v-if="forecast" class="batch-card">
      <header><p class="kicker">CONDITIONAL EXECUTION WINDOWS</p><h3>分批仓位建议</h3></header>
      <p v-if="!decision?.batches.length" class="empty">
        {{ decision?.empty_reason ?? "当前没有可执行批次，等待下一完整交易周复查。" }}
      </p>
      <ol v-else>
        <li v-for="batch in decision?.batches" :key="batch.batch_number">
          <b>{{ String(batch.batch_number).padStart(2, "0") }}</b>
          <time>{{ batch.execution_window_start }} 起（最早第 {{ batch.earliest_execution_week }} 周）</time>
          <strong>{{ batch.action === "BUY" ? "增加" : "减少" }} {{ batch.batch_change_pp }} 个百分点</strong>
          <small>条件：{{ batch.condition }}；目标仓位：{{ batch.target_position_pp }}%；冷却 {{ batch.cooldown_trading_days }} 个交易日</small>
        </li>
      </ol>
    </section>
  </section>
</template>

<style scoped>
.v35-lab {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid #b9b09e;
  background: #fffdf7;
  box-shadow: 5px 5px 0 rgba(23, 43, 58, .08);
  color: #172b3a;
  display: grid;
  gap: 18px;
}
.v35-heading, .forecast-card > header, .batch-card > header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}
.v35-heading {
  padding-bottom: 14px;
  border-bottom: 2px solid #172b3a;
}
.v35-heading h2, .v35-heading p, .forecast-card h3, .batch-card h3 { margin: 0; }
.v35-heading h2 { margin-top: 4px; font-size: 20px; }
.kicker { color: #12687e; font-size: 10px; font-weight: 900; letter-spacing: .14em; }
.panel-note { margin-top: 6px; color: #69747a; font-size: 12px; }
.analyze-button {
  padding: 8px 12px;
  border: 1px solid #172b3a;
  background: #172b3a;
  color: #fffdf6;
  font-weight: 800;
  cursor: pointer;
}
.analyze-button:disabled { opacity: .55; cursor: wait; }
.v35-error { padding: 10px; border: 1px solid #a23f32; color: #a23f32; }
.v35-note { color: #69747a; font-size: 12px; line-height: 1.7; }
.v35-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.v35-card, .forecast-card, .batch-card {
  padding: 14px;
  border: 1px solid #d4cdc0;
  background: #fbf8f0;
}
.v35-card h3 { margin: 0 0 10px; }
.v35-card dl, .v35-prob { margin: 0; }
.v35-card dl > div, .v35-prob > div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
}
.v35-card dl dt, .v35-prob dt { color: #657279; }
.v35-card dl dd, .v35-prob dd { margin: 0; font-weight: 700; text-align: right; }
.v35-card dl .primary dd { color: #12687e; font-size: 15px; }
.v35-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
.v35-table th, .v35-table td { border: 1px solid #d4cdc0; padding: 4px 6px; text-align: right; }
.v35-table th { background: #eef4f2; }
.forecast-card, .batch-card { margin-top: 0; }
.forecast-chart { width: 100%; height: 430px; margin-top: 10px; }
.semantic-note { color: #69747a; font-size: 11px; margin: 6px 0 0; }
.turning-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.turning-grid article {
  padding: 12px;
  border: 1px solid #c9c1b3;
  background: #fffdf6;
}
.turning-grid article.warning { border-color: #b7791f; }
.turning-grid span { color: #12687e; font-size: 11px; font-weight: 900; }
.turning-grid h3 { margin: 6px 0; font-size: 14px; }
.turning-grid p { margin: 0; color: #657279; font-size: 11px; }
.batch-card ol { margin: 12px 0 0; padding: 0; list-style: none; display: grid; gap: 8px; }
.batch-card li {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 10px;
  border: 1px dashed #12687e;
  background: #fffdf7;
}
.batch-card li b { color: #12687e; font-size: 18px; }
.batch-card li strong { color: #172b3a; }
.batch-card li small { grid-column: 2 / 4; color: #657279; }
.batch-card .empty { padding: 12px; border: 1px dashed #b9b09e; color: #657279; }
@media (max-width: 620px) {
  .v35-heading { display: block; }
  .analyze-button { margin-top: 10px; width: 100%; }
}
</style>
