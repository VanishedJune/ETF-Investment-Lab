<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch } from "vue";

import PositionCalendar from "../components/PositionCalendar.vue";
import CombinedMarketChart from "../components/CombinedMarketChart.vue";
import UnifiedMarketChart from "../components/UnifiedMarketChart.vue";
import { api } from "../api";
import { finiteNumber, periodReturn } from "../period-return";
import { useResearchData } from "../composables/useResearchData";
import type {
  ActiveInstrumentCode,
  IndicatorRow,
  MarketTimeframe,
  ResearchInstrumentCode,
} from "../types/research";

type MarketDescriptor = {
  code: ResearchInstrumentCode;
  name: string;
  market: string;
  slot_order?: number | null;
  is_current_slot?: boolean;
};

type RefreshAudit = {
  instrumentCode: string;
  outcome: string;
  before: string;
  expected: string;
  after: string;
  source: string;
  added: number;
  updated: number;
  skipped: number;
  networkRequested: boolean;
  verifiedFresh: boolean;
};

const markets = ref<MarketDescriptor[]>([]);
const catalog = ref<MarketDescriptor[]>([]);
const universeError = ref("");
const selected = ref<ResearchInstrumentCode>("159915");
const timeframe = ref<MarketTimeframe>("weekly");
const calendarInstrument = ref<ActiveInstrumentCode>("159915");
const message = ref("");
const lastRefresh = ref<RefreshAudit | null>(null);

const {
  prices,
  indicators,
  dailyPrices,
  dailyIndicators,
  weeklyPrices,
  weeklyIndicators,
  indicatorSnapshots,
  currentPeriodStatus,
  positionEvents,
  currentPositions,
  marketLoading,
  refreshing,
  calendarLoading,
  marketError,
  calendarError,
  loadMarket,
  refreshMarket,
  loadPositions,
  createPosition,
  updatePosition,
  deletePosition,
} = useResearchData();

const currentMarket = computed(
  () => markets.value.find((item) => item.code === selected.value)
    ?? { code: selected.value, name: selected.value, market: "ETF" },
);
const latest = computed(() => prices.value.at(-1));
const previous = computed(() => prices.value.at(-2));
const latestComplete = computed(() => (
  [...prices.value].reverse().find((row) => row.is_complete !== false)
));
const completeCutoff = computed(() => latestComplete.value?.date);
const provisionalDate = computed(() => (
  currentPeriodStatus.value === "INCOMPLETE_CURRENT_PERIOD" ? latest.value?.date : undefined
));
const displayedDataCutoff = computed(() => (
  provisionalDate.value ?? completeCutoff.value ?? "—"
));
const periodChange = computed(() => periodReturn(prices.value));
const snapshotCards = computed(() => [
  { key: 'daily', label: '日 K', row: indicatorSnapshots.value.daily,
    price: dailyPrices.value.at(-1), change: periodReturn(dailyPrices.value), returnLabel: '当日涨跌幅' },
  { key: 'weekly', label: '周 K', row: indicatorSnapshots.value.weekly,
    price: weeklyPrices.value.at(-1), change: periodReturn(weeklyPrices.value),
    returnLabel: weeklyPrices.value.at(-1)?.is_complete === false ? '本周截至当前涨跌幅' : '完整周涨跌幅' },
]);
function labelFor(value: unknown): string {
  const number = finiteNumber(value);
  return number != null
    ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 }).format(number)
    : "—";
}

function percentFor(value: unknown): string {
  const number = finiteNumber(value);
  return number != null ? `${number > 0 ? '+' : ''}${number.toFixed(2)}%` : "—";
}

function indicatorValue(row: IndicatorRow | null, key: keyof IndicatorRow): string {
  return labelFor(row?.[key]);
}

function directionClass(value: unknown): string {
  const number = finiteNumber(value);
  if (number == null || number === 0) return 'value-neutral neutral';
  return number > 0 ? 'macd-positive up' : 'macd-negative down';
}

async function loadUniverse() {
  try {
    const rows = await api<Array<MarketDescriptor & { exchange: string }>>("/api/instruments");
    catalog.value = rows.map(row => ({ ...row, market: row.exchange === 'NASDAQ' ? '指数' : 'ETF' }));
    markets.value = catalog.value.filter(row => row.is_current_slot)
      .sort((a, b) => (a.slot_order ?? 999) - (b.slot_order ?? 999));
    if (!markets.value.length) throw new Error('未配置启用的ETF槽位，请检查本地目录。');
    if (!markets.value.some(row => row.code === selected.value)) selected.value = markets.value[0]!.code;
    if (!catalog.value.some(row => row.code === calendarInstrument.value)) calendarInstrument.value = selected.value;
    universeError.value = '';
  } catch (reason) {
    universeError.value = `标的目录读取失败：${reason instanceof Error ? reason.message : String(reason)}`;
  }
}

async function loadSelected() {
  message.value = "";
  await Promise.all([
    loadMarket(selected.value, timeframe.value),
    loadPositions(calendarInstrument.value),
  ]);
}

async function refreshSelectedMarket() {
  message.value = "正在刷新本地行情缓存…";
  const refreshCode = selected.value;
  try {
    const payload = await refreshMarket(refreshCode, timeframe.value);
    const cutoff = typeof payload.cutoff_date === "string"
      ? payload.cutoff_date
      : displayedDataCutoff.value;
    const outcome = String(payload.refresh_outcome ?? "").toUpperCase();
    const status = String(
      payload.market_refresh_status ?? payload.status ?? "",
    ).toLowerCase();
    const error = typeof payload.refresh_error === "string"
      ? payload.refresh_error
      : typeof payload.error === "string"
        ? payload.error
      : typeof payload.last_error === "string"
        ? payload.last_error
        : "";
    const textField = (key: string, fallback = "—") => (
      typeof payload[key] === "string" && payload[key]
        ? String(payload[key])
        : fallback
    );
    const numberField = (key: string) => {
      const value = Number(payload[key]);
      return Number.isFinite(value) ? value : 0;
    };
    lastRefresh.value = {
      instrumentCode: refreshCode,
      outcome: outcome || "UNKNOWN",
      before: textField("local_cutoff_before"),
      expected: textField("expected_cutoff"),
      after: textField("local_cutoff_after", cutoff),
      source: textField("source"),
      added: numberField("records_added"),
      updated: numberField("records_updated"),
      skipped: numberField("records_skipped"),
      networkRequested: payload.network_requested === true,
      verifiedFresh: payload.verified_fresh === true,
    };

    if (outcome === "ALREADY_FRESH") {
      message.value = `${refreshCode} 本地行情已是最新数据（${cutoff}），本次未重复联网下载。`;
      return;
    }
    if (outcome === "UPDATED") {
      message.value = `${refreshCode} 已更新至 ${cutoff}；新增 ${lastRefresh.value.added} 条，修订 ${lastRefresh.value.updated} 条。`;
      return;
    }
    if (outcome === "STILL_STALE" || outcome === "UPDATED_UNVERIFIED") {
      message.value = `${refreshCode} 刷新后仍未通过最新交易日校验：${error || "请稍后重试"}。`;
      return;
    }
    const cached = payload.cache_used === true || status.includes("degraded") || status === "cached";
    if (outcome === "UPSTREAM_FAILED" || cached || error.trim()) {
      // A cached response is intentionally non-fatal for the chart, but must
      // never be presented as a successful refresh.  Keep the old cutoff
      // visible and surface the provider error so the user knows to retry
      // after connectivity is restored.
      const detail = error.length > 280
        ? `${error.slice(0, 277)}…`
        : (error || "上游行情接口未返回新记录");
      message.value = `${refreshCode} 刷新未完成：${detail}；当前本地数据仍截至 ${cutoff}。请联网后重试。`;
      return;
    }
    message.value = `${refreshCode} 行情已更新至 ${cutoff}；本版本仅刷新行情与技术指标。`;
  } catch (reason) {
    message.value = reason instanceof Error ? reason.message : String(reason);
  }
}

function selectMarket(code: ResearchInstrumentCode) {
  selected.value = code;
}

function changeCalendarInstrument(code: ActiveInstrumentCode) {
  calendarInstrument.value = code;
}

watch(selected, () => void loadSelected());
watch(timeframe, () => void loadMarket(selected.value, timeframe.value));
watch(calendarInstrument, (code) => void loadPositions(code));

async function reloadUniverse() {
  await loadUniverse();
  await loadSelected();
}
onMounted(async () => {
  window.addEventListener('etf-universe-changed', reloadUniverse);
  window.addEventListener('focus', reloadUniverse);
  await reloadUniverse();
});
onBeforeUnmount(() => {
  window.removeEventListener('etf-universe-changed', reloadUniverse);
  window.removeEventListener('focus', reloadUniverse);
});
</script>

<template>
  <main class="research-shell data-only-shell">
    <p v-if="universeError" class="form-error" role="alert">{{ universeError }}</p>
    <header class="masthead">
      <div>
        <p class="kicker">MARKET DATA DESK · LOCAL SQLITE · V3.7 DATA ONLY</p>
        <h1>ETF 行情面板</h1>
        <p class="lede">
          八只 ETF 使用统一行情卡片，日 K / 周 K / 月 K 同步查看；下方同时列出日线与周线 DIF、DEA、MACD，投资日历独立保存仓位记录。
        </p>
      </div>
      <div class="agent-seal data-only-seal">
        <span>PURE MARKET VIEW</span>
        <b>无 AI · 无推理迭代</b>
        <small>只读取本地行情、指标与投资日历</small>
      </div>
      <router-link class="slot-nav" to="/etf-slots">ETF 替换</router-link>
    </header>

    <section class="market-switch" aria-label="选择 ETF">
      <button
        v-for="item in markets"
        :key="item.code"
        :class="{ active: selected === item.code }"
        type="button"
        @click="selectMarket(item.code)"
      >
        <span>{{ String(item.slot_order ?? '').padStart(2, '0') }} · {{ item.market }}</span>
        <strong>{{ item.name }}</strong>
        <em>{{ item.code }}</em>
      </button>
    </section>

    <div v-if="message" class="flash">{{ message }}</div>
    <div v-if="marketError" class="flash error">{{ marketError }}</div>

    <section class="quote-strip">
      <div><span>最新收盘</span><strong>{{ labelFor(latest?.close) }}</strong></div>
      <div><span>周期变化</span><strong :class="directionClass(periodChange)">{{ periodChange == null ? "—" : percentFor(periodChange) }}</strong></div>
      <div><span>数据截止</span><strong data-testid="data-cutoff">{{ displayedDataCutoff }}</strong></div>
      <div><span>数据来源</span><strong class="source">{{ latest?.source ?? "—" }}</strong></div>
      <button class="refresh" type="button" :disabled="marketLoading || refreshing" @click="refreshSelectedMarket">
        {{ refreshing ? "正在下载…" : marketLoading ? "同步界面…" : "刷新行情" }}
      </button>
    </section>

    <section v-if="lastRefresh" class="refresh-audit" data-testid="refresh-audit">
      <div><span>刷新标的</span><strong>{{ lastRefresh.instrumentCode }}</strong></div>
      <div><span>本地原截止</span><strong>{{ lastRefresh.before }}</strong></div>
      <div><span>应有交易日</span><strong>{{ lastRefresh.expected }}</strong></div>
      <div><span>刷新后截止</span><strong>{{ lastRefresh.after }}</strong></div>
      <div><span>数据来源</span><strong>{{ lastRefresh.source }}</strong></div>
      <div><span>写入结果</span><strong>新增 {{ lastRefresh.added }} · 修订 {{ lastRefresh.updated }} · 跳过 {{ lastRefresh.skipped }}</strong></div>
      <div><span>联网请求</span><strong>{{ lastRefresh.networkRequested ? "已执行" : "未执行（本地已最新）" }}</strong></div>
      <div><span>交易日校验</span><strong :class="lastRefresh.verifiedFresh ? 'verified' : 'unverified'">{{ lastRefresh.verifiedFresh ? "通过" : "未通过" }}</strong></div>
    </section>

    <section class="panel indicator-snapshot-panel" aria-label="日线与周线技术指标">
      <div class="panel-heading compact-heading">
        <div>
          <p class="kicker">MOMENTUM SNAPSHOT</p>
          <h2>{{ currentMarket.name }} · 日线 / 周线指标</h2>
        </div>
        <small>指标随标的与行情刷新；不生成预测或仓位建议</small>
      </div>
      <div class="indicator-snapshot-grid">
        <article v-for="item in snapshotCards" :key="item.key" class="indicator-snapshot-card">
          <header><strong>{{ item.label }}</strong><time>价格截至 {{ item.price?.date ?? "暂无数据" }}</time></header>
          <div class="period-return" :data-testid="`${item.key}-return`">
            <span>{{ item.returnLabel }}</span>
            <b :class="directionClass(item.change)">{{ percentFor(item.change) }}</b>
          </div>
          <small class="return-basis">未复权收盘价较上一{{ item.key === 'daily' ? '交易日' : '周' }}收盘价；不含分红与持仓收益</small>
          <small v-if="item.row?.date !== item.price?.date" class="period-warning">技术指标截至 {{ item.row?.date ?? '暂无数据' }}，与价格日期不同</small>
          <dl>
            <div><dt>DIF</dt><dd class="dif-value">{{ indicatorValue(item.row, "dif") }}</dd></div>
            <div><dt>DEA</dt><dd class="dea-value">{{ indicatorValue(item.row, "dea") }}</dd></div>
            <div><dt>MACD 柱</dt><dd :class="directionClass(item.row?.macd_histogram)">{{ indicatorValue(item.row, "macd_histogram") }}</dd></div>
            <div><dt>DIF 一阶变化</dt><dd class="dif-change-value">{{ indicatorValue(item.row, "dif_first_change") }}</dd></div>
          </dl>
        </article>
      </div>
    </section>

    <section class="panel chart-panel">
      <div class="panel-heading">
        <div>
          <p class="kicker">{{ currentMarket.market }}</p>
          <h2>{{ currentMarket.name }} · 日 K + 周 K 共享横轴</h2>
          <p class="panel-note">日 K 与周 K 放在同一个画布中，共用一条日期时间轴和底部时间条；价格、成交量、MACD 柱、DIF、DEA 与 DIF 一阶变化按日线/周线分区对齐。</p>
          <p v-if="currentPeriodStatus === 'INCOMPLETE_CURRENT_PERIOD'" class="panel-note period-warning" data-testid="incomplete-period-notice">
            当前周期尚未完成，最后一根为预览数据（{{ provisionalDate ?? "—" }}），完整周期截止 {{ completeCutoff ?? "—" }}。
          </p>
        </div>
        <div class="timeframe-tabs" aria-label="共享日 K 周 K 与月 K">
          <span class="shared-timeframe-label">日 K + 周 K 共用横轴</span>
          <button data-testid="timeframe-monthly" :aria-pressed="timeframe === 'monthly'" type="button" @click="timeframe = timeframe === 'monthly' ? 'weekly' : 'monthly'">{{ timeframe === 'monthly' ? '关闭月 K' : '显示月 K' }}</button>
        </div>
      </div>
      <CombinedMarketChart
        :instrument-code="selected"
        :daily-prices="dailyPrices"
        :daily-indicators="dailyIndicators"
        :weekly-prices="weeklyPrices"
        :weekly-indicators="weeklyIndicators"
        :loading="marketLoading"
        :error="marketError"
      />
      <article v-if="timeframe === 'monthly'" class="comparison-chart-card monthly-chart-card">
        <header class="comparison-chart-heading">
          <div>
            <span class="chart-period-eyebrow">MONTHLY MARKET SERIES</span>
            <h3>月 K · 长周期参照</h3>
          </div>
          <span>附加视图</span>
        </header>
        <UnifiedMarketChart :instrument-code="selected" timeframe="monthly" :prices="prices" :indicators="indicators" :loading="marketLoading" :error="marketError" />
      </article>
    </section>

    <PositionCalendar
      :instruments="catalog"
      :instrument-code="calendarInstrument"
      :events="positionEvents"
      :positions="currentPositions"
      :loading="calendarLoading"
      :error="calendarError"
      :create-event="createPosition"
      :update-event="updatePosition"
      :delete-event="deletePosition"
      @instrument-change="changeCalendarInstrument"
    />

    <footer>本地行情与技术指标工具 · 不连接券商 · 不自动交易 · 不包含 AI 或预测功能</footer>
  </main>
</template>

<style scoped>
.data-only-seal { min-width: 190px; }
.data-only-seal b { color: #52789c; }
.refresh-audit {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  margin-top: 10px;
  overflow: hidden;
  border: 1px solid #d5dce5;
  background: #d5dce5;
}
.refresh-audit > div {
  display: grid;
  min-width: 0;
  gap: 4px;
  padding: 10px 12px;
  background: #f4f6f8;
}
.refresh-audit span { color: #657184; font-size: 11px; }
.refresh-audit strong {
  overflow-wrap: anywhere;
  color: #263445;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.refresh-audit .verified { color: #27845a; }
.refresh-audit .unverified { color: #bd4938; }
.compact-heading { align-items: center; }
.compact-heading small { color: #657184; }
.indicator-snapshot-panel { margin-top: 16px; }
.indicator-snapshot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.indicator-snapshot-card { border: 1px solid #d5dce5; border-radius: 10px; background: #f4f6f8; padding: 16px 20px; }
.period-return { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 14px 0 4px; }
.period-return span { color: #536778; font-size: 14px; }
.period-return b { font-size: 28px; font-variant-numeric: tabular-nums; }
.return-basis { display: block; color: #657184; font-size: 12px; }
.indicator-snapshot-card header { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid #d5dce5; padding-bottom: 8px; }
.indicator-snapshot-card time { color: #657184; font-size: 12px; }
.indicator-snapshot-card dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 12px 0 0; }
.indicator-snapshot-card dl > div { display: grid; gap: 3px; }
.indicator-snapshot-card dt { color: #657184; font-size: 12px; }
.indicator-snapshot-card dd { margin: 0; font-weight: 900; }
.dif-value { color: #52789c; }
.dea-value { color: #a57738; }
.value-neutral { color: var(--neutral); }
.macd-positive { color: #d64b4b; }
.macd-negative { color: #27845a; }
.dif-change-value { color: #807096; }
.dual-chart-stack { display: grid; gap: 18px; }
.shared-timeframe-label { color: #52789c; font-size: 12px; font-weight: 900; letter-spacing: .03em; }
.comparison-chart-card {
  min-width: 0;
  border: 1px solid #d5dce5;
  background: #f4f6f8;
  scroll-margin-top: 18px;
}
.comparison-chart-heading {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid #d5dce5;
  background: #e9edf2;
}
.comparison-chart-heading h3 { margin: 4px 0 0; color: #263445; font-size: 20px; }
.comparison-chart-heading > span { color: #657184; font-size: 12px; font-variant-numeric: tabular-nums; }
.chart-period-eyebrow { color: #52789c; font-size: 10px; font-weight: 900; letter-spacing: .13em; }
.comparison-chart-card :deep(.unified-market-chart) { min-height: 900px; padding: 12px; }
.monthly-chart-card { margin-top: 18px; }
@media (max-width: 760px) {
  .refresh-audit { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .indicator-snapshot-grid { grid-template-columns: 1fr; }
  .indicator-snapshot-card dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .comparison-chart-heading h3 { font-size: 17px; }
  .comparison-chart-card :deep(.unified-market-chart) { min-height: 760px; }
}
@media (max-width: 440px) {
  .refresh-audit { grid-template-columns: 1fr; }
}
</style>
