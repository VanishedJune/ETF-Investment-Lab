<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { api, percent } from "../api";
import { alignIndicatorValues } from "../chart-data";

type MarketRow = { date: string; open: string | number; high: string | number; low: string | number; close: string | number; volume?: string | number | null; source?: string | null };

const instruments = ref<any[]>([]);
const selected = ref("000688");
const timeframe = ref("daily");
const market = ref<any>();
const indicators = ref<any[]>([]);
const comparison = ref<Record<string, MarketRow[]>>({});
const analysis = ref<any>();
const valuation = ref<any>();
const message = ref("");
const error = ref("");
const loading = ref(false);
const chartEl = ref<HTMLElement>();
const comparisonEl = ref<HTMLElement>();
const valuationEl = ref<HTMLElement>();
let chart: echarts.ECharts | undefined;
let comparisonChart: echarts.ECharts | undefined;
let valuationChart: echarts.ECharts | undefined;

const actionNames: Record<string, string> = {
  INCREASE: "考虑加仓",
  NORMAL: "正常执行",
  REDUCE: "降低投入",
  PAUSE: "暂停投入",
  SELL_PARTIAL: "部分卖出",
  HOLD: "暂不动作",
};
const actionClasses: Record<string, string> = {
  INCREASE: "buy", NORMAL: "normal", REDUCE: "reduce", PAUSE: "pause", SELL_PARTIAL: "sell", HOLD: "hold",
};
const factorNames: Record<string, string> = {
  valuation: "估值位置",
  trend: "趋势结构",
  momentum: "动量状态",
  volume: "成交活跃度",
  volatility: "波动约束",
  risk: "风险约束",
  drawdown: "回撤状态",
};
const ruleNames: Record<string, string> = {
  trend_close_above_ma_20: "收盘价高于 MA20",
  trend_ma_20_above_ma_60: "MA20 高于 MA60",
  volume_above_average: "量能高于 20 日均量",
  volatility_within_limit: "波动率处于策略阈值内",
  risk_drawdown_within_limit: "回撤处于策略阈值内",
  momentum_rsi_oversold: "RSI 处于超卖区间",
  momentum_macd_positive: "MACD 柱线为正",
};
const riskNames: Record<string, string> = {
  valuation_percentile_missing: "估值百分位尚未录入本地数据",
  momentum_macd_negative: "MACD 柱线为负，动量仍需确认",
  trend_below_ma_20: "收盘价低于 MA20，趋势尚未转强",
  volatility_above_limit: "波动率高于策略阈值",
  drawdown_below_limit: "回撤超过策略阈值",
};

const latest = computed<MarketRow | undefined>(() => market.value?.rows?.at(-1));
const previous = computed<MarketRow | undefined>(() => market.value?.rows?.at(-2));
const change = computed(() => latest.value && previous.value ? (numberOf(latest.value.close) - numberOf(previous.value.close)) / numberOf(previous.value.close) : null);
const selectedIndex = computed(() => instruments.value.find((item) => item.code === selected.value));
const actionLabel = computed(() => actionNames[analysis.value?.recommendation] ?? "等待评估");
const actionClass = computed(() => actionClasses[analysis.value?.recommendation] ?? "hold");
const factorScores = computed(() => Object.entries(analysis.value?.component_scores ?? {}) as Array<[string, unknown]>);
const reasons = computed<string[]>(() => analysis.value?.reasons ?? []);
const rules = computed<string[]>(() => analysis.value?.triggered_rules ?? []);
const risks = computed<string[]>(() => analysis.value?.reverse_risks ?? []);
const latestValuation = computed(() => valuation.value?.latest);
const hasUnavailableVolume = computed(() => (market.value?.rows ?? []).some((row: MarketRow) => row.volume == null));

function numberOf(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function volumeAxisLabel(value: number): string {
  const absolute = Math.abs(value);
  if (!Number.isFinite(value)) return "—";
  if (absolute >= 100_000_000) return `${(value / 100_000_000).toFixed(absolute >= 10_000_000_000 ? 0 : 1)}亿`;
  if (absolute >= 10_000) return `${(value / 10_000).toFixed(absolute >= 1_000_000 ? 0 : 1)}万`;
  return value.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
}

function display(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "—";
}

function evidenceLabel(value: string): string {
  if (value.startsWith("missing:valuation:valuation_percentile")) return "估值百分位尚未录入本地数据，估值因子不加分。";
  if (value.startsWith("trend:")) return "趋势：收盘价相对 MA20 / MA60 的位置满足当前规则。";
  if (value.startsWith("momentum:")) return "动量：RSI 与 MACD 状态已参与评分。";
  if (value.startsWith("volume:")) return "成交活跃度：当前量能相对 20 日均量已参与评分。";
  if (value.startsWith("volatility:")) return "波动约束：20 日波动率已与策略阈值比较。";
  if (value.startsWith("risk:")) return "回撤约束：当前回撤已与风险阈值比较。";
  if (value.startsWith("strategy_config_sha256=")) return `规则配置 SHA256：${value.slice(-12)}`;
  if (value.startsWith("source_data_sha256=")) return `数据快照 SHA256：${value.slice(-12)}`;
  return value;
}

function ruleLabel(value: string): string { return ruleNames[value] ?? value; }
function riskLabel(value: string): string { return riskNames[value] ?? value; }

function indicatorValues(key: string): Array<number | null> {
  return alignIndicatorValues(market.value?.rows ?? [], indicators.value, key);
}

function renderChart() {
  if (!chartEl.value || !market.value?.rows?.length) return;
  chart?.dispose();
  chart = echarts.init(chartEl.value);
  const rows: MarketRow[] = market.value.rows;
  const dates = rows.map((row) => row.date);
  const candle = rows.map((row) => [numberOf(row.open), numberOf(row.close), numberOf(row.low), numberOf(row.high)]);
  const volumes = rows.map((row) => {
    if (row.volume == null) return null;
    return { value: numberOf(row.volume), itemStyle: { color: numberOf(row.close) >= numberOf(row.open) ? "#d84b45" : "#16885f" } };
  });
  const histogram = indicatorValues("macd_histogram");
  const start = Math.max(0, 100 - Math.min(100, (100 / Math.max(rows.length, 1)) * 100));

  chart.setOption({
    animation: false,
    color: ["#2767cc", "#e39422", "#c63bb5", "#27824f"],
    tooltip: { trigger: "axis", axisPointer: { type: "cross" }, backgroundColor: "rgba(22, 34, 45, .94)", borderWidth: 0, textStyle: { color: "#fff", fontSize: 13 } },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: [
      { left: 62, right: 18, top: 24, height: "43%" },
      { left: 62, right: 18, top: "53%", height: "13%" },
      { left: 62, right: 18, top: "74%", height: "16%" },
    ],
    xAxis: [0, 1, 2].map((gridIndex) => ({ type: "category", gridIndex, data: dates, boundaryGap: true, axisLine: { lineStyle: { color: "#bbc5cd" } }, axisTick: { show: false }, axisLabel: { show: gridIndex === 2, color: "#65717c", fontSize: 12 }, splitLine: { show: false } })),
    yAxis: [
      { type: "value", scale: true, gridIndex: 0, axisLabel: { color: "#65717c", fontSize: 12 }, splitLine: { lineStyle: { color: "#e4e8eb", type: "dashed" } } },
      { type: "value", scale: true, gridIndex: 1, splitNumber: 3, axisLabel: { color: "#65717c", fontSize: 11, hideOverlap: true, formatter: volumeAxisLabel }, splitLine: { show: false } },
      { type: "value", scale: true, gridIndex: 2, axisLabel: { color: "#65717c", fontSize: 12 }, splitLine: { lineStyle: { color: "#e4e8eb", type: "dashed" } } },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2], start, end: 100 },
      { type: "slider", xAxisIndex: [0, 1, 2], bottom: 2, height: 18, borderColor: "#c9d1d7", fillerColor: "rgba(39, 103, 204, .16)", handleStyle: { color: "#2767cc" } },
    ],
    series: [
      { name: "指数 K 线", type: "candlestick", xAxisIndex: 0, yAxisIndex: 0, data: candle, itemStyle: { color: "#d84b45", color0: "#16885f", borderColor: "#d84b45", borderColor0: "#16885f" } },
      { name: "MA5", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: indicatorValues("ma_5"), symbol: "none", lineStyle: { width: 1.2, color: "#2767cc" } },
      { name: "MA10", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: indicatorValues("ma_10"), symbol: "none", lineStyle: { width: 1.2, color: "#e39422" } },
      { name: "MA20", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: indicatorValues("ma_20"), symbol: "none", lineStyle: { width: 1.2, color: "#c63bb5" } },
      { name: "MA60", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: indicatorValues("ma_60"), symbol: "none", lineStyle: { width: 1.2, color: "#27824f" } },
      { name: "成交量", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: volumes, barMaxWidth: 12 },
      { name: "MACD", type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: histogram.map((value) => value === null ? null : ({ value, itemStyle: { color: value >= 0 ? "#d84b45" : "#16885f" } })), barMaxWidth: 8 },
      { name: "DIF", type: "line", xAxisIndex: 2, yAxisIndex: 2, data: indicatorValues("dif"), symbol: "none", lineStyle: { width: 1.5, color: "#2767cc" } },
      { name: "DEA", type: "line", xAxisIndex: 2, yAxisIndex: 2, data: indicatorValues("dea"), symbol: "none", lineStyle: { width: 1.5, color: "#e39422" } },
    ],
  });
}

function renderComparison() {
  if (!comparisonEl.value || !instruments.value.length) return;
  comparisonChart?.dispose();
  comparisonChart = echarts.init(comparisonEl.value);
  const series = instruments.value.map((instrument) => {
    const rows = (comparison.value[instrument.code] ?? []).slice(-160);
    const base = numberOf(rows[0]?.close) || 1;
    return { name: instrument.name, type: "line", smooth: true, showSymbol: false, data: rows.map((row) => [row.date, Number(((numberOf(row.close) / base) * 100).toFixed(2))]) };
  });
  comparisonChart.setOption({
    animation: false,
    color: ["#2767cc", "#d84b45", "#27824f"],
    tooltip: { trigger: "axis", valueFormatter: (value: number) => `${value.toFixed(2)}` },
    legend: { top: 8, textStyle: { color: "#4c5966" } },
    grid: { left: 56, right: 18, top: 42, bottom: 28 },
    xAxis: { type: "time", axisLabel: { color: "#73808c" }, axisLine: { lineStyle: { color: "#bdc6ce" } } },
    yAxis: { type: "value", name: "起点 = 100", axisLabel: { color: "#73808c" }, splitLine: { lineStyle: { color: "#e6eaed" } } },
    series,
  });
}

function renderValuation() {
  if (!valuationEl.value) return;
  valuationChart?.dispose();
  valuationChart = echarts.init(valuationEl.value);
  const rows = valuation.value?.series ?? [];
  valuationChart.setOption({
    animation: false,
    color: ["#d84b45", "#2767cc", "#16885f"],
    tooltip: { trigger: "axis", backgroundColor: "rgba(28, 43, 55, .94)", borderWidth: 0, textStyle: { color: "#fff", fontSize: 13 } },
    legend: { top: 8, data: ["PE", "PB", "股息率"], textStyle: { color: "#4c5966", fontSize: 12 } },
    grid: { left: 58, right: 24, top: 42, bottom: 32 },
    xAxis: { type: "category", data: rows.map((row: any) => row.date), axisLabel: { color: "#73808c", fontSize: 12 }, axisLine: { lineStyle: { color: "#bdc6ce" } } },
    yAxis: { type: "value", scale: true, axisLabel: { color: "#73808c", fontSize: 12 }, splitLine: { lineStyle: { color: "#e6eaed", type: "dashed" } } },
    series: [
      { name: "PE", type: "line", smooth: true, showSymbol: false, data: rows.map((row: any) => row.pe_ratio) },
      { name: "PB", type: "line", smooth: true, showSymbol: false, data: rows.map((row: any) => row.pb_ratio) },
      { name: "股息率", type: "line", smooth: true, showSymbol: false, data: rows.map((row: any) => row.dividend_yield) },
    ],
  });
}

function resizeCharts() { chart?.resize(); comparisonChart?.resize(); valuationChart?.resize(); }

async function load() {
  if (!selected.value) return;
  loading.value = true;
  error.value = "";
  try {
    [market.value, indicators.value, valuation.value] = await Promise.all([
      api<any>(`/market/${selected.value}/prices?timeframe=${timeframe.value}`),
      api<any[]>(`/indicators/${selected.value}?timeframe=${timeframe.value}`),
      api<any>(`/valuations/${selected.value}`),
    ]);
    await nextTick();
    renderChart();
    renderValuation();
  } catch (reason: any) {
    error.value = reason.message;
  } finally {
    loading.value = false;
  }
}

async function loadComparison() {
  try {
    const data = await Promise.all(instruments.value.map(async (instrument) => [instrument.code, (await api<any>(`/market/${instrument.code}/prices?timeframe=daily`)).rows] as const));
    comparison.value = Object.fromEntries(data);
    await nextTick();
    renderComparison();
  } catch (reason: any) {
    error.value = reason.message;
  }
}

async function refresh() {
  message.value = "正在更新直接指数行情，并重新计算 MACD、DIF、DEA…";
  try {
    const result = await api<any>(`/market/${selected.value}/refresh`, { method: "POST" });
    message.value = result.demo
      ? "公开指数源暂不可用，曲线显示的是明确标记的 DEMO_INDEX 演示数据。"
      : "直接指数行情已写入本地缓存，技术指标已重新计算。";
    await Promise.all([load(), loadComparison()]);
    await evaluate();
  } catch (reason: any) {
    message.value = reason.message;
  }
}

async function refreshValuation() {
  message.value = "正在从公开直接指数估值源刷新并写入本地历史…";
  try {
    const result = await api<any>(`/valuations/${selected.value}/refresh`, { method: "POST" });
    message.value = result.status === "success"
      ? `估值已更新：新增 ${result.records_added} 条，更新 ${result.records_updated} 条。来源：${result.source}。`
      : `估值源本次不可用：${result.error ?? "无可用直接指数记录"}。${result.cached_records ? "已保留并显示本地历史缓存。" : "未生成任何演示估值。"}`;
    await load();
    if (!result.strategy_recomputed) await evaluate();
  } catch (reason: any) {
    message.value = reason.message;
  }
}

async function evaluate() {
  try {
    analysis.value = await api<any>(`/strategies/${selected.value}/evaluate`, { method: "POST" });
  } catch (reason: any) {
    analysis.value = undefined;
    message.value = `策略暂不可评估：${reason.message}`;
  }
}

async function importCsv(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const result = await api<any>(`/market/${selected.value}/csv/import`, { method: "POST", body: await file.text(), headers: { "Content-Type": "text/csv;charset=utf-8" } });
    message.value = `指数 CSV 导入完成：新增 ${result.added}，更新 ${result.updated}，失败 ${result.failed}。`;
    await Promise.all([load(), loadComparison()]);
    await evaluate();
  } catch (reason: any) {
    message.value = reason.message;
  }
}

onMounted(async () => {
  try {
    instruments.value = await api<any[]>("/instruments");
    selected.value = instruments.value.find((item) => item.code === "000688")?.code ?? instruments.value[0]?.code ?? "000688";
    await Promise.all([load(), loadComparison()]);
    await evaluate();
  } catch (reason: any) {
    error.value = reason.message;
  }
  window.addEventListener("resize", resizeCharts);
});

onBeforeUnmount(() => {
  chart?.dispose();
  comparisonChart?.dispose();
  valuationChart?.dispose();
  window.removeEventListener("resize", resizeCharts);
});

watch([selected, timeframe], () => { analysis.value = undefined; void load(); });
</script>

<template>
  <section class="market-terminal">
    <div class="quote-strip">
      <div class="quote-title"><span class="eyebrow">DIRECT INDEX / LOCAL CACHE</span><h2>{{ selectedIndex?.name ?? "指数行情" }} <b>{{ selected }}</b></h2><small>数据截止：{{ latest?.date ?? "—" }} · {{ market?.source ?? "等待数据" }}</small></div>
      <div class="quote-price" :class="Number(change) >= 0 ? 'up' : 'down'"><b>{{ display(latest?.close) }}</b><small>{{ percent(change) }}</small></div>
      <div class="quote-facts"><span>开盘 <b>{{ display(latest?.open) }}</b></span><span>最高 <b>{{ display(latest?.high) }}</b></span><span>最低 <b>{{ display(latest?.low) }}</b></span><span>成交量 <b>{{ display(latest?.volume, 0) }}</b></span></div>
    </div>

    <div class="terminal-tabs">
      <button v-for="item in instruments" :key="item.code" :class="{ active: selected === item.code }" @click="selected = item.code">{{ item.name }}</button>
      <span class="tab-divider"></span>
      <button :class="{ active: timeframe === 'daily' }" @click="timeframe = 'daily'">日 K</button>
      <button :class="{ active: timeframe === 'weekly' }" @click="timeframe = 'weekly'">周 K</button>
      <button :class="{ active: timeframe === 'monthly' }" @click="timeframe = 'monthly'">月 K</button>
    </div>

    <div v-if="error" class="status error">{{ error }}</div>
    <div v-if="message" class="status">{{ message }}</div>
    <div v-if="market?.demo" class="status demo-status">当前曲线为 DEMO_INDEX 演示数据，不能视作真实指数行情。</div>

    <div class="market-workspace">
      <div class="chart-workbench">
        <div class="chart-legend"><b>主图</b><span class="candle-key">指数 K 线</span><span class="ma5">MA5</span><span class="ma10">MA10</span><span class="ma20">MA20</span><span class="ma60">MA60</span><em>下方独立区域：成交量 / MACD 柱 / DIF / DEA</em></div>
        <div class="macd-panel-legend" aria-label="MACD 指标颜色说明"><span class="macd-bar-key">MACD柱</span><span class="macd-dif-key">DIF</span><span class="macd-dea-key">DEA</span></div>
        <div v-if="hasUnavailableVolume" class="volume-warning">当前周期部分指数成交量因本地精度限制不可用，未以零值绘制。</div>
        <div v-if="loading && !market" class="terminal-loading">正在调取本地数据…</div>
        <div ref="chartEl" class="terminal-chart"></div>
      </div>

      <aside class="signal-desk">
        <div class="signal-head"><div><span>规则化策略分析</span><small>可审计 · 不自动交易</small></div><button class="button" :disabled="loading" @click="evaluate">运行分析</button></div>
        <template v-if="analysis">
          <div class="decision" :class="actionClass"><small>当前研究建议</small><strong>{{ actionLabel }}</strong><span>{{ analysis.recommendation }}</span></div>
          <div class="signal-metrics"><div><small>综合评分</small><b>{{ display(analysis.score) }}</b></div><div><small>置信度</small><b>{{ percent(analysis.confidence) }}</b></div><div><small>投入倍数</small><b>{{ display(analysis.multiplier) }}×</b></div><div><small>部分卖出上限</small><b>{{ percent(analysis.suggested_sell_ratio) }}</b></div></div>
          <h3>触发依据</h3>
          <ul class="reason-list"><li v-for="reason in reasons" :key="reason">{{ evidenceLabel(reason) }}</li><li v-if="!reasons.length">需要足够的本地指标数据后显示规则依据。</li></ul>
          <h3>因子得分</h3>
          <div class="factor" v-for="[key, value] in factorScores" :key="key"><span>{{ factorNames[key] ?? key }}</span><i :style="{ width: `${Math.max(0, Math.min(100, (Number(value ?? 0) + 1) * 50))}%` }"></i><b>{{ display(value) }}</b></div>
          <template v-if="rules.length"><h3>命中规则</h3><ul class="compact-list"><li v-for="rule in rules" :key="rule">{{ ruleLabel(rule) }}</li></ul></template>
          <template v-if="risks.length"><h3>反向风险</h3><ul class="compact-list risk-list"><li v-for="risk in risks" :key="risk">{{ riskLabel(risk) }}</li></ul></template>
        </template>
        <div v-else class="signal-empty">更新指数或导入至少 60 个交易日数据后，运行分析即可查看确定性买入、降低投入、暂停或部分卖出参考。</div>
        <div class="signal-actions"><button class="button ghost" @click="refresh">更新直接指数</button><label class="file">导入指数 CSV<input type="file" accept=".csv,text/csv" @change="importCsv" /></label></div>
      </aside>
    </div>
  </section>

  <section class="panel section-gap valuation-panel">
    <div class="panel-head">
      <div><span class="eyebrow">VALUATION / DIRECT INDEX SOURCE</span><h2>估值与来源</h2><p class="muted">仅保存和展示直接指数的公开来源记录。PE、PB、股息率可能因指数与来源覆盖范围不同而缺失；缺失时策略的估值因子不加分，也不会以 ETF 或演示数据替代。</p></div>
      <button class="button ghost" :disabled="loading" @click="refreshValuation">刷新估值数据</button>
    </div>
    <div class="valuation-metrics">
      <div><span>最新 PE</span><b>{{ display(latestValuation?.pe_ratio) }}</b></div>
      <div><span>最新 PB</span><b>{{ display(latestValuation?.pb_ratio) }}</b></div>
      <div><span>股息率</span><b>{{ latestValuation?.dividend_yield == null ? "—" : `${display(latestValuation.dividend_yield)}%` }}</b></div>
      <div><span>PE 历史分位</span><b>{{ latestValuation?.valuation_percentile == null ? "—" : percent(latestValuation.valuation_percentile) }}</b></div>
    </div>
    <div class="valuation-provenance">
      <span>状态：<b>{{ valuation?.source_status === "cached" ? "已保存的本地记录" : "尚无直接指数估值记录" }}</b></span>
      <span>来源：<b>{{ latestValuation?.source ?? "—" }}</b></span>
      <span>数据日期：<b>{{ latestValuation?.date ?? "—" }}</b></span>
      <span>记录数：<b>{{ valuation?.record_count ?? 0 }}</b></span>
      <a v-if="latestValuation?.source_url" :href="latestValuation.source_url" target="_blank" rel="noreferrer">查看来源说明</a>
    </div>
    <div v-if="valuation?.series?.length" ref="valuationEl" class="valuation-chart"></div>
    <div v-else class="valuation-empty">当前公开源尚未返回 {{ selectedIndex?.name ?? selected }} 的直接指数估值。刷新会保留已有本地数据；系统不会伪造或用 ETF 代理补齐。</div>
  </section>

  <section class="panel section-gap comparison-panel"><div class="panel-head"><div><span class="eyebrow">RELATIVE PERFORMANCE</span><h2>三大指数相对走势</h2><p class="muted">每条曲线均以其首个可用交易日 = 100 归一化，仅比较变化，不混入 ETF 价格。</p></div><button class="button ghost" @click="loadComparison">刷新曲线</button></div><div ref="comparisonEl" class="comparison-chart"></div></section>
</template>
