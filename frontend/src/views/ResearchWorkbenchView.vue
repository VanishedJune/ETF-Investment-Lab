<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";

import PositionCalendar from "../components/PositionCalendar.vue";
import UnifiedMarketChart from "../components/UnifiedMarketChart.vue";
import V32ModelPanel from "../components/V32ModelPanel.vue";
import V34ModelPanel from "../components/V34ModelPanel.vue";
import { useResearchData } from "../composables/useResearchData";
import type {
  ActiveInstrumentCode,
  MarketTimeframe,
  ResearchInstrumentCode,
} from "../types/research";

const markets = [
  { code: "399006", name: "创业板指数", market: "A股 / CHINEXT" },
  { code: "159941", name: "广发纳斯达克100ETF", market: "QDII / SZSE" },
  { code: "518600", name: "广发黄金ETF", market: "黄金 / SSE" },
  { code: "512800", name: "华宝银行ETF", market: "银行 / SSE" },
  { code: "512690", name: "鹏华酒ETF", market: "消费 / SSE" },
] satisfies { code: ResearchInstrumentCode; name: string; market: string }[];

const selected = ref<ResearchInstrumentCode>("399006");
const timeframe = ref<MarketTimeframe>("weekly");
const message = ref("");

const {
  prices,
  indicators,
  currentPeriodStatus,
  v32Analysis,
  v32Run,
  v32TrainingStatus,
  v32Curves,
  trainingLoading,
  trainingError,
  v34Analysis,
  v34Statuses,
  v34Curves,
  v34TrainingLoading,
  v34TrainingError,
  v34AnalysisLoading,
  v34AnalysisError,
  positionEvents,
  currentPositions,
  marketLoading,
  analysisLoading,
  calendarLoading,
  marketError,
  analysisError,
  calendarError,
  loadMarket,
  refreshMarket,
  loadV32State,
  loadV34State,
  trainV34Model,
  startV34Analysis,
  stopPolling,
  loadPositions,
  createPosition,
  updatePosition,
  deletePosition,
} = useResearchData();

const currentMarket = computed(() => markets.find((item) => item.code === selected.value)!);
const selectedModel = computed<ActiveInstrumentCode | null>(() => (
  selected.value === "399006" || selected.value === "159941" ? selected.value : null
));
const latest = computed(() => prices.value.at(-1));
const previous = computed(() => prices.value.at(-2));
const latestComplete = computed(() => (
  [...prices.value].reverse().find((row) => row.is_complete !== false)
));
const provisionalDate = computed(() => (
  currentPeriodStatus.value === "INCOMPLETE_CURRENT_PERIOD" ? latest.value?.date : undefined
));
const displayedDataCutoff = computed(() => {
  const failedRunCutoff = v32Run.value?.result?.last_successful_data_date;
  const candidates = [
    latestComplete.value?.date,
    v32Analysis.value?.freshness.price_data_as_of,
    typeof failedRunCutoff === "string" ? failedRunCutoff : undefined,
  ].filter((value): value is string => Boolean(value));
  return candidates.sort().at(-1) ?? "—";
});
const periodChange = computed(() => {
  if (!latest.value || !previous.value) return null;
  const current = Number(latest.value.close);
  const prior = Number(previous.value.close);
  return Number.isFinite(current) && Number.isFinite(prior) && prior !== 0
    ? ((current / prior) - 1) * 100
    : null;
});

async function loadSelected() {
  const symbol = selected.value;
  const period = timeframe.value;
  message.value = "";
  const model = selectedModel.value;
  await Promise.all([
    loadMarket(symbol, period),
    model ? loadV34State(model) : Promise.resolve(),
    loadV32State(model === "399006" ? "399006" : null),
    model ? loadPositions(model) : Promise.resolve(),
  ]);
}

async function analyzeSelectedMarket() {
  const symbol = selectedModel.value;
  if (!symbol) return;
  await startV34Analysis(symbol);
  if (selected.value === symbol) await loadMarket(symbol, timeframe.value);
}

async function trainSelectedMarket(market: ActiveInstrumentCode) {
  await trainV34Model(market);
}

async function refreshSelectedMarket() {
  message.value = "正在刷新公开行情并复算日/周/月指标…";
  try {
    await refreshMarket(selected.value, timeframe.value);
    message.value = marketError.value
      ? "刷新完成，但当前周期没有通过连续性检查。"
      : "行情和指标已刷新；模型不会随行情刷新自动运行。";
  } catch (reason) {
    message.value = reason instanceof Error ? reason.message : String(reason);
  }
}

function compactNumber(value: unknown): string {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 }).format(number)
    : "—";
}

watch(selected, () => {
  stopPolling();
  void loadSelected();
});

watch(timeframe, () => {
  stopPolling();
  message.value = "";
  void loadMarket(selected.value, timeframe.value);
});

onMounted(() => void loadSelected());
</script>

<template>
  <main class="research-shell">
    <header class="masthead">
      <div>
        <p class="kicker">INDEX + ETF PATH RESEARCH · LOCAL V3.4</p>
        <h1>双市场周线研究台</h1>
        <p class="lede">
          五只标的统一展示行情与动能；只有创业板指数和159941进入13周模型、仓位建议与投资日历。
        </p>
      </div>
      <div class="agent-seal">
        <span>V3.4-13W CHAMPION</span>
        <b>{{ selectedModel ? (v34Statuses[selectedModel]?.champion?.version ?? "等待训练") : "仅行情展示" }}</b>
        <small>本地计算 · 不调用AI</small>
      </div>
    </header>

    <section class="market-switch" aria-label="选择指数">
      <button
        v-for="item in markets"
        :key="item.code"
        :class="{ active: selected === item.code }"
        type="button"
        @click="selected = item.code"
      >
        <span>{{ item.market }}</span>
        <strong>{{ item.name }}</strong>
        <em>{{ item.code }}</em>
      </button>
    </section>

    <div v-if="message" class="flash">{{ message }}</div>
    <div v-if="marketError" class="flash error">{{ marketError }}</div>

    <section class="quote-strip">
      <div>
        <span>最新收盘</span>
        <strong>{{ compactNumber(latest?.close) }}</strong>
      </div>
      <div>
        <span>周期变化</span>
        <strong :class="{ down: (periodChange ?? 0) < 0 }">
          {{ periodChange == null ? "—" : `${periodChange.toFixed(2)}%` }}
        </strong>
      </div>
      <div>
        <span>数据截止</span>
        <strong data-testid="data-cutoff">{{ displayedDataCutoff }}</strong>
      </div>
      <div>
        <span>成交量来源</span>
        <strong class="source">
          {{
            latest?.source ?? "—"
          }}
        </strong>
      </div>
      <button
        class="refresh"
        type="button"
        :disabled="marketLoading"
        @click="refreshSelectedMarket"
      >
        {{ marketLoading ? "处理中" : "刷新行情" }}
      </button>
    </section>

    <section class="panel chart-panel">
      <div class="panel-heading">
        <div>
          <p class="kicker">{{ currentMarket.market }}</p>
          <h2>{{ currentMarket.name }} · 价格 / 成交量 / 动能</h2>
          <p class="panel-note">四层图共享时间轴；价格、成交量、MACD与DIF一阶变化分别按可视区间动态缩放。</p>
          <p
            v-if="currentPeriodStatus === 'INCOMPLETE_CURRENT_PERIOD'"
            class="panel-note period-warning"
            data-testid="incomplete-period-notice"
          >
            {{ timeframe === "weekly" ? "本周" : "本月" }}尚未完成；图中最后一根为预览数据（{{ provisionalDate ?? "—" }}），正式完整周期截止 {{ displayedDataCutoff }}。
          </p>
        </div>
        <div class="timeframe-tabs" aria-label="行情周期">
          <button
            data-testid="timeframe-daily"
            :aria-pressed="timeframe === 'daily'"
            type="button"
            @click="timeframe = 'daily'"
          >日K</button>
          <button
            data-testid="timeframe-weekly"
            :aria-pressed="timeframe === 'weekly'"
            type="button"
            @click="timeframe = 'weekly'"
          >周K</button>
          <button
            data-testid="timeframe-monthly"
            :aria-pressed="timeframe === 'monthly'"
            type="button"
            @click="timeframe = 'monthly'"
          >月K</button>
        </div>
      </div>
      <UnifiedMarketChart
        :instrument-code="selected"
        :timeframe="timeframe"
        :prices="prices"
        :indicators="indicators"
        :loading="marketLoading"
        :error="marketError"
      />
    </section>

    <V34ModelPanel
      v-if="selectedModel"
      :selected="selectedModel"
      :result="v34Analysis"
      :statuses="v34Statuses"
      :curves="v34Curves"
      :analysis-loading="v34AnalysisLoading"
      :training-loading="v34TrainingLoading"
      :training-error="v34TrainingError"
      :analysis-error="v34AnalysisError"
      @analyze="analyzeSelectedMarket"
      @train="trainSelectedMarket"
    />

    <details v-if="selectedModel" class="legacy-audit">
      <summary>
        <span>V3.2只读历史审计</span>
        <small>399006 / NDX底层基准 · 不参与V3.3仓位与预测</small>
      </summary>
      <V32ModelPanel
        :selected="selectedModel"
        :result="v32Analysis"
        :run="v32Run"
        :statuses="v32TrainingStatus"
        :curves="v32Curves"
        :analysis-loading="analysisLoading"
        :training-loading="trainingLoading"
        :training-error="trainingError"
        :analysis-error="analysisError"
      />
    </details>

    <PositionCalendar
      v-if="selectedModel"
      :instrument-code="selectedModel"
      :events="positionEvents"
      :positions="currentPositions"
      :loading="calendarLoading"
      :error="calendarError"
      :create-event="createPosition"
      :update-event="updatePosition"
      :delete-event="deletePosition"
    />

    <section v-else class="panel display-only-note">
      <p class="kicker">DISPLAY ONLY</p>
      <h2>该ETF仅用于行情与技术指标观察</h2>
      <p>不会进入模型训练、方向概率、仓位建议或投资日历，避免展示标的污染两套独立模型。</p>
    </section>

    <footer>
      本地持久化研究工具 · 指数与QDII公开行情 · 不连接券商 · 不自动交易 · 结果仅供研究
    </footer>
  </main>
</template>

<style scoped>
.legacy-audit { margin-top: 16px; border: 1px solid #b9b09e; background: #fffdf7; }
.legacy-audit > summary { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 15px 18px; color: #172b3a; cursor: pointer; font-weight: 900; }
.legacy-audit > summary small { color: #69747a; font-weight: 500; }
.legacy-audit[open] > summary { border-bottom: 1px solid #b9b09e; }
.legacy-audit :deep(.v32-lab) { margin-top: 0; border: 0; box-shadow: none; }
.legacy-audit > summary:focus-visible { outline: 3px solid rgba(11,103,215,.3); outline-offset: 2px; }
@media (max-width: 620px) { .legacy-audit > summary { display: block; }.legacy-audit > summary small { display: block; margin-top: 4px; } }
</style>
