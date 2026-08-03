<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import * as echarts from "echarts";
import { api, money, percent, today } from "../api";

const accounts = ref<any[]>([]);
const plans = ref<any[]>([]);
const selectedAccount = ref<number>();
const selectedInstrument = ref("ALL");
const transactions = ref<any[]>([]);
const positions = ref<any[]>([]);
const series = ref<any[]>([]);
const comparison = ref<any | null>(null);
const preview = ref<any | null>(null);
const previewError = ref("");
const message = ref("");
const error = ref("");
const accountName = ref("");
const timeframe = ref("daily");
const startDate = ref("");
const endDate = ref("");
const operationFilter = ref("ALL");
const pnlFilter = ref("ALL");
const editingId = ref<number | null>(null);
const chartEl = ref<HTMLElement>();
let chart: echarts.ECharts | undefined;
let previewTimer: number | undefined;

const trade = reactive({
  instrument_code: "589850",
  side: "BUY",
  transaction_date: today(),
  price: "",
  notes: "",
});

const instrumentOptions = computed(() => {
  const linked = plans.value.map((plan) => ({ code: plan.instrument_code, name: plan.name, amount: plan.weekly_amount }));
  const fallback = [
    { code: "589850", name: "科创50ETF", amount: null },
    { code: "159205", name: "创业板ETF", amount: null },
    { code: "159941", name: "纳指ETF", amount: null },
  ];
  const seen = new Set<string>();
  return [...linked, ...fallback].filter((item) => !seen.has(item.code) && (seen.add(item.code) || true));
});

const latest = computed(() => series.value[series.value.length - 1] ?? null);
const filteredTransactions = computed(() => transactions.value.filter((row) => {
  if (selectedInstrument.value !== "ALL" && row.instrument_code !== selectedInstrument.value) return false;
  if (operationFilter.value !== "ALL" && row.side !== operationFilter.value) return false;
  if (startDate.value && row.transaction_date < startDate.value) return false;
  if (endDate.value && row.transaction_date > endDate.value) return false;
  const pnl = Number(row.realized_pnl ?? 0);
  if (pnlFilter.value === "PROFIT" && pnl <= 0) return false;
  if (pnlFilter.value === "LOSS" && pnl >= 0) return false;
  if (pnlFilter.value === "FLAT" && pnl !== 0) return false;
  return true;
}));

function queryString() {
  const params = new URLSearchParams({ timeframe: timeframe.value });
  if (selectedInstrument.value !== "ALL") params.set("instrument_code", selectedInstrument.value);
  if (startDate.value) params.set("start_date", startDate.value);
  if (endDate.value) params.set("end_date", endDate.value);
  return params.toString();
}

async function load() {
  try {
    [accounts.value, plans.value] = await Promise.all([api<any[]>("/real-accounts"), api<any[]>("/plans")]);
    selectedAccount.value ??= accounts.value[0]?.id;
    if (!instrumentOptions.value.some((item) => item.code === trade.instrument_code)) trade.instrument_code = instrumentOptions.value[0]?.code ?? "589850";
    await loadDetails();
  } catch (caught: any) {
    error.value = caught.message;
  }
}

async function loadDetails() {
  if (!selectedAccount.value) return;
  try {
    const suffix = queryString();
    const [rows, holding, line, benchmark] = await Promise.all([
      api<any[]>(`/real-accounts/${selectedAccount.value}/transactions`),
      api<any[]>(`/real-accounts/${selectedAccount.value}/positions`),
      api<any[]>(`/real-accounts/${selectedAccount.value}/series?${suffix}`),
      api<any>(`/real-accounts/${selectedAccount.value}/comparison`),
    ]);
    transactions.value = rows;
    positions.value = holding;
    series.value = line;
    comparison.value = benchmark;
    await nextTick();
    renderChart();
    schedulePreview();
  } catch (caught: any) {
    if (caught.status === 400 && String(caught.message).includes("No real-account snapshots")) {
      transactions.value = [];
      positions.value = [];
      series.value = [];
      comparison.value = null;
      await nextTick();
      renderChart();
      return;
    }
    error.value = caught.message;
  }
}

async function createAccount() {
  const name = accountName.value.trim();
  if (!name) return;
  try {
    const created = await api<any>("/real-accounts", { method: "POST", body: JSON.stringify({ name }) });
    selectedAccount.value = created.id;
    accountName.value = "";
    message.value = "已创建本地真实账户。账户现金由每笔自动计算的定投资金与卖出到账共同构成。";
    await load();
  } catch (caught: any) {
    error.value = caught.message;
  }
}

function schedulePreview() {
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(refreshPreview, 240);
}

async function refreshPreview() {
  preview.value = null;
  previewError.value = "";
  if (!selectedAccount.value || !trade.instrument_code || !trade.transaction_date || Number(trade.price) <= 0) return;
  try {
    preview.value = await api<any>(`/real-accounts/${selectedAccount.value}/transactions/preview`, {
      method: "POST",
      body: JSON.stringify({ ...trade, price: Number(trade.price) }),
    });
  } catch (caught: any) {
    previewError.value = caught.message;
  }
}

async function saveTrade() {
  if (!selectedAccount.value) return;
  error.value = "";
  if (!preview.value) {
    await refreshPreview();
    if (!preview.value) return;
  }
  try {
    const path = editingId.value
      ? `/real-accounts/${selectedAccount.value}/transactions/${editingId.value}`
      : `/real-accounts/${selectedAccount.value}/transactions`;
    await api(path, {
      method: editingId.value ? "PATCH" : "POST",
      body: JSON.stringify({ ...trade, price: Number(trade.price) }),
    });
    message.value = editingId.value ? "已修改交易，并已按完整历史交易重新计算持仓和收益。" : "已保存真实交易，并已自动生成份额、费用、指标快照和收益曲线。";
    editingId.value = null;
    trade.price = "";
    trade.notes = "";
    preview.value = null;
    await loadDetails();
  } catch (caught: any) {
    error.value = caught.message;
  }
}

function editTrade(row: any) {
  editingId.value = row.id;
  trade.instrument_code = row.instrument_code;
  trade.side = row.side;
  trade.transaction_date = row.transaction_date;
  trade.price = String(row.price);
  trade.notes = row.notes ?? "";
  window.scrollTo({ top: 0, behavior: "smooth" });
  schedulePreview();
}

function cancelEdit() {
  editingId.value = null;
  trade.price = "";
  trade.notes = "";
  preview.value = null;
}

async function deleteTrade(row: any) {
  if (!selectedAccount.value || !window.confirm(`删除 ${row.transaction_date} 的这笔${row.side === "BUY" ? "买入" : "卖出"}记录？系统将从第一笔交易重算。`)) return;
  try {
    await api(`/real-accounts/${selectedAccount.value}/transactions/${row.id}?confirmed=true`, { method: "DELETE" });
    message.value = "交易已删除，所有历史持仓、收益、资产快照和基准比较均已重算。";
    await loadDetails();
  } catch (caught: any) {
    error.value = caught.message;
  }
}

function exportFile(format: "csv" | "xlsx") {
  if (selectedAccount.value) window.location.href = `/api/real-accounts/${selectedAccount.value}/export?format=${format}`;
}

function pnlText(value: unknown) {
  const number = Number(value ?? 0);
  return number > 0 ? "盈利" : number < 0 ? "亏损" : "持平";
}

function pnlClass(value: unknown) {
  const number = Number(value ?? 0);
  return number > 0 ? "up" : number < 0 ? "down" : "muted";
}

function areas() {
  const ranges: any[] = [];
  let start: string | null = null;
  series.value.forEach((row, index) => {
    const profitable = Number(row.total_pnl ?? 0) >= 0;
    if (profitable && start === null) start = row.date;
    if ((!profitable || index === series.value.length - 1) && start !== null) {
      ranges.push([{ xAxis: start }, { xAxis: profitable && index === series.value.length - 1 ? row.date : series.value[index - 1].date }]);
      start = null;
    }
  });
  return ranges;
}

function renderChart() {
  if (!chartEl.value) return;
  chart?.dispose();
  chart = echarts.init(chartEl.value);
  const dates = series.value.map((row) => row.date);
  const buyMarkers = transactions.value.filter((row) => row.side === "BUY").map((row) => ({ coord: [row.transaction_date, Number(row.total_pnl ?? 0)], value: "买" }));
  const sellMarkers = transactions.value.filter((row) => row.side === "SELL").map((row) => ({ coord: [row.transaction_date, Number(row.total_pnl ?? 0)], value: "卖" }));
  chart.setOption({
    animationDuration: 420,
    color: ["#0f766e", "#c76b32", "#294e8d", "#c51b4a", "#7c3f9c", "#6b7280", "#b88a16", "#334155"],
    tooltip: { trigger: "axis", valueFormatter: (value: number) => Number.isFinite(value) ? value.toFixed(2) : "—" },
    legend: { top: 4, type: "scroll" },
    grid: { left: 58, right: 76, top: 46, bottom: 48 },
    xAxis: { type: "category", data: dates, boundaryGap: false, axisLabel: { color: "#64748b" } },
    yAxis: [
      { type: "value", name: "金额", scale: true, splitLine: { lineStyle: { color: "#e5e7eb" } } },
      { type: "value", name: "收益率", scale: true, axisLabel: { formatter: "{value}%" } },
      { type: "value", name: "份额", position: "right", offset: 42, scale: true, splitLine: { show: false } },
      { type: "value", name: "价格", position: "right", scale: true, splitLine: { show: false } },
    ],
    series: [
      { name: "实际账户资产", type: "line", data: series.value.map((row) => row.total_assets), smooth: true, symbol: "none", lineStyle: { width: 3 }, markArea: { silent: true, itemStyle: { color: "rgba(15,118,110,.055)" }, data: areas() }, markPoint: { symbolSize: 30, data: [...buyMarkers, ...sellMarkers] } },
      { name: "一直持有资产", type: "line", data: series.value.map((row) => row.benchmark_total_assets), smooth: true, symbol: "none", lineStyle: { width: 2, type: "dashed" } },
      { name: "累计投入", type: "line", data: series.value.map((row) => row.contribution), symbol: "none", lineStyle: { width: 1.5 } },
      { name: "累计收益", type: "line", data: series.value.map((row) => row.total_pnl), symbol: "none", lineStyle: { width: 2 } },
      { name: "已实现收益", type: "line", data: series.value.map((row) => row.realized_pnl), symbol: "none", lineStyle: { width: 1.5 } },
      { name: "未实现收益", type: "line", data: series.value.map((row) => row.unrealized_pnl), symbol: "none", lineStyle: { width: 1.5 } },
      { name: "累计收益率", type: "line", yAxisIndex: 1, data: series.value.map((row) => Number(row.total_return ?? 0) * 100), symbol: "none", lineStyle: { width: 1.5 } },
      { name: "持仓份额", type: "line", yAxisIndex: 2, data: series.value.map((row) => row.holding_quantity), symbol: "none", lineStyle: { width: 1.5 } },
      { name: "ETF行情价格", type: "line", yAxisIndex: 3, data: series.value.map((row) => row.market_price), symbol: "none", lineStyle: { width: 1.5, type: "dotted" } },
    ],
  });
}

onMounted(() => {
  load();
  window.addEventListener("resize", renderChart);
});
onBeforeUnmount(() => {
  window.clearTimeout(previewTimer);
  chart?.dispose();
  window.removeEventListener("resize", renderChart);
});
watch([selectedAccount, timeframe, selectedInstrument, startDate, endDate], loadDetails);
watch(() => [trade.instrument_code, trade.side, trade.transaction_date, trade.price], schedulePreview);
</script>

<template>
  <section class="panel real-account-hero">
    <div class="panel-head">
      <div>
        <p class="eyebrow">REAL LEDGER · 本地可审计</p>
        <h2>真实账户</h2>
        <p class="muted">只录入真实成交事实；份额、费用、估值、指标与收益均由本地账本按交易时间完整重算。</p>
      </div>
      <div class="toolbar">
        <select class="select" v-model="selectedAccount"><option v-for="account in accounts" :key="account.id" :value="account.id">{{ account.name }}</option></select>
        <button class="button ghost" @click="exportFile('csv')">导出 CSV</button>
        <button class="button ghost" @click="exportFile('xlsx')">导出 Excel</button>
      </div>
    </div>
    <div class="compact-create">
      <input v-model="accountName" placeholder="新建本地账户名称" @keyup.enter="createAccount" />
      <button class="button ghost" @click="createAccount">新建账户</button>
      <span class="muted">无需预存金额；每次买入会自动计入对应定投资金。</span>
    </div>
  </section>

  <section class="section-gap panel trade-panel">
    <div class="panel-head"><div><p class="eyebrow">{{ editingId ? 'EDIT & REPLAY' : 'RECORD ACTUAL EXECUTION' }}</p><h2>{{ editingId ? '修改真实交易' : '录入实际成交' }}</h2></div><button v-if="editingId" class="button ghost" @click="cancelEdit">取消编辑</button></div>
    <div class="form-grid real-trade-grid">
      <label class="field">ETF 或指数基金<select v-model="trade.instrument_code"><option v-for="item in instrumentOptions" :key="item.code" :value="item.code">{{ item.code }} · {{ item.name }}</option></select></label>
      <label class="field">操作类型<select v-model="trade.side"><option value="BUY">买入</option><option value="SELL">卖出</option></select></label>
      <label class="field">实际操作日期<input v-model="trade.transaction_date" type="date" /></label>
      <label class="field">实际成交价格<input v-model="trade.price" type="number" min="0" step="0.0001" placeholder="仅此价格参与真实账务" /></label>
      <label class="field note-field">备注（可选）<input v-model="trade.notes" maxlength="400" placeholder="例如：手动补录 / 分红再投入" /></label>
    </div>
    <div class="auto-note">未提供交易金额、买入份额、卖出份额、卖出比例或 ETF 净值输入项；系统会自动读取关联定投计划、真实账户费率与本地行情。</div>
    <div v-if="preview" class="preview-grid">
      <article><label>{{ preview.side === 'BUY' ? '关联定投金额' : '当前可卖份额' }}</label><strong>{{ preview.side === 'BUY' ? money(preview.plan_amount) : preview.available_quantity }}</strong></article>
      <article><label>{{ preview.side === 'BUY' ? '预计买入份额' : '预计卖出到账' }}</label><strong>{{ preview.side === 'BUY' ? preview.quantity : money(preview.net_proceeds) }}</strong></article>
      <article><label>预计手续费</label><strong>{{ money(preview.fee) }}</strong><span class="muted">未配置费率时默认为 0</span></article>
      <article><label>{{ preview.side === 'BUY' ? '预计剩余现金' : '预计已实现收益' }}</label><strong :class="pnlClass(preview.side === 'BUY' ? preview.remaining_cash : preview.estimated_realized_pnl)">{{ money(preview.side === 'BUY' ? preview.remaining_cash : preview.estimated_realized_pnl) }}</strong></article>
      <article><label>当日行情 / 估值</label><strong>{{ money(preview.market?.etf_close) }}</strong><span class="muted">{{ preview.market?.valuation_status ?? '估值待补全' }} · {{ preview.market?.macd_status ?? 'MACD待补全' }}</span></article>
      <article><label>数据状态</label><strong>{{ preview.market?.status }}</strong><span class="muted">{{ preview.market?.non_trading_day_note || preview.market?.data_source }}</span></article>
    </div>
    <p v-if="preview?.price_deviation_warning" class="status warning">实际成交价与当日收盘价偏差较大，请检查；系统不会替换你填写的实际价格。</p>
    <p v-if="previewError" class="status error">{{ previewError }}</p>
    <div class="section-gap"><button class="button" :disabled="!preview" @click="saveTrade">{{ editingId ? '保存修改并完整重算' : '保存真实交易' }}</button></div>
    <p v-if="message" class="status">{{ message }}</p><p v-if="error" class="status error">{{ error }}</p>
  </section>

  <section class="section-gap metrics-grid">
    <article class="panel metric"><label>当前账户总资产</label><strong>{{ money(latest?.total_assets) }}</strong><span>现金 {{ money(latest?.cash_balance) }} · 持仓市值 {{ money(latest?.market_value) }}</span></article>
    <article class="panel metric"><label>累计收益</label><strong :class="pnlClass(latest?.total_pnl)">{{ money(latest?.total_pnl) }}</strong><span>{{ pnlText(latest?.total_pnl) }} · 收益率 {{ percent(latest?.total_return) }}</span></article>
    <article class="panel metric"><label>实际操作 vs 一直持有</label><strong :class="pnlClass(comparison?.outperformance)">{{ money(comparison?.outperformance) }}</strong><span>多赚/少赚 · 份额差 {{ comparison?.quantity_difference ?? '—' }}</span></article>
    <article class="panel metric"><label>累计手续费</label><strong>{{ money(latest?.fees_paid) }}</strong><span>已实现 {{ money(latest?.realized_pnl) }} · 未实现 {{ money(latest?.unrealized_pnl) }}</span></article>
  </section>

  <section class="section-gap panel">
    <div class="panel-head"><div><p class="eyebrow">DAILY REBUILT SERIES</p><h2>收益与亏损曲线</h2><p class="muted">买入与卖出标记显示在资产曲线上；淡色区间为累计收益非负的区间。</p></div><div class="toolbar chart-controls"><select class="select" v-model="selectedInstrument"><option value="ALL">全部 ETF 汇总</option><option v-for="item in instrumentOptions" :key="item.code" :value="item.code">{{ item.code }}</option></select><select class="select" v-model="timeframe"><option value="daily">日线</option><option value="weekly">周线</option><option value="monthly">月线</option></select><input v-model="startDate" type="date" aria-label="开始日期" /><input v-model="endDate" type="date" aria-label="结束日期" /></div></div>
    <div ref="chartEl" class="chart real-chart"></div>
    <div v-if="comparison" class="comparison-strip"><span>实际资产 <b>{{ money(comparison.actual_total_assets) }}</b></span><span>一直持有 <b>{{ money(comparison.benchmark_total_assets) }}</b></span><span>实际收益率 <b>{{ percent(comparison.actual_total_return) }}</b></span><span>一直持有收益率 <b>{{ percent(comparison.benchmark_total_return) }}</b></span></div>
  </section>

  <section class="section-gap grid cols-2">
    <article class="panel"><h3>当前持仓</h3><div class="table-wrap"><table class="data-table"><thead><tr><th>ETF</th><th>持有份额</th><th>平均成本</th><th>市值</th><th>未实现收益</th><th>已实现收益</th></tr></thead><tbody><tr v-for="position in positions" :key="position.id"><td>{{ position.instrument_code }}</td><td>{{ position.quantity }}</td><td>{{ money(position.average_cost) }}</td><td>{{ money(position.market_value) }}</td><td :class="pnlClass(position.unrealized_pnl)">{{ money(position.unrealized_pnl) }}</td><td :class="pnlClass(position.realized_pnl)">{{ money(position.realized_pnl) }}</td></tr><tr v-if="!positions.length"><td colspan="6" class="empty">暂无持仓</td></tr></tbody></table></div></article>
    <article class="panel"><h3>一直持有策略说明</h3><p class="muted">以首次买入开始，在每次真实账户新增定投资金的日期，按本地行情和同一费率模拟买入；不执行真实账户的卖出操作，卖出现金仍保留在实际账户中。</p><dl class="audit-list"><div><dt>实际操作累计收益</dt><dd :class="pnlClass(comparison?.actual_total_pnl)">{{ money(comparison?.actual_total_pnl) }}</dd></div><div><dt>一直持有累计收益</dt><dd :class="pnlClass(comparison?.benchmark_total_pnl)">{{ money(comparison?.benchmark_total_pnl) }}</dd></div><div><dt>份额变化贡献</dt><dd>{{ comparison?.quantity_difference ?? '—' }}</dd></div></dl></article>
  </section>

  <section class="section-gap panel">
    <div class="panel-head"><div><p class="eyebrow">TRACEABLE PROFIT & LOSS</p><h2>交易收益明细表</h2></div><div class="toolbar"><select class="select" v-model="operationFilter"><option value="ALL">全部操作</option><option value="BUY">买入</option><option value="SELL">卖出</option></select><select class="select" v-model="pnlFilter"><option value="ALL">全部收益状态</option><option value="PROFIT">盈利</option><option value="LOSS">亏损</option><option value="FLAT">持平</option></select></div></div>
    <div class="table-wrap audit-table"><table class="data-table"><thead><tr><th>序号</th><th>ETF</th><th>操作</th><th>日期</th><th>实际成交价</th><th>当日收盘价</th><th>交易金额</th><th>交易份额</th><th>手续费</th><th>交易后份额</th><th>交易后均价</th><th>本次已实现</th><th>累计已实现</th><th>当前未实现</th><th>累计收益</th><th>收益状态</th><th>PE</th><th>PB</th><th>估值百分位</th><th>DIF / DEA</th><th>MACD</th><th>操作</th></tr></thead><tbody><template v-for="(row, index) in filteredTransactions" :key="row.id"><tr><td>{{ index + 1 }}</td><td>{{ row.instrument_code }}</td><td><span class="tag">{{ row.side === 'BUY' ? '买入' : '卖出' }}</span></td><td>{{ row.transaction_date }}</td><td>{{ money(row.price) }}</td><td>{{ money(row.market_snapshot?.etf_close) }}</td><td>{{ money(row.amount) }}</td><td>{{ row.quantity }}</td><td>{{ money(row.fee) }}</td><td>{{ row.holding_quantity_after }}</td><td>{{ money(row.average_cost_after) }}</td><td :class="pnlClass(row.realized_pnl)">{{ money(row.realized_pnl) }}</td><td :class="pnlClass(row.cumulative_realized_pnl)">{{ money(row.cumulative_realized_pnl) }}</td><td :class="pnlClass(row.current_unrealized_pnl)">{{ money(row.current_unrealized_pnl) }}</td><td :class="pnlClass(row.total_pnl)">{{ money(row.total_pnl) }}</td><td :class="pnlClass(row.realized_pnl)">{{ pnlText(row.realized_pnl) }}</td><td>{{ row.market_snapshot?.pe_ratio ?? '—' }}</td><td>{{ row.market_snapshot?.pb_ratio ?? '—' }}</td><td>{{ row.market_snapshot?.valuation_percentile ?? '—' }}</td><td>{{ row.market_snapshot?.dif ?? '—' }} / {{ row.market_snapshot?.dea ?? '—' }}</td><td>{{ row.market_snapshot?.macd_status ?? '待补全' }}</td><td class="row-actions"><button class="text-button" @click="editTrade(row)">修改</button><button class="text-button danger" @click="deleteTrade(row)">删除</button></td></tr><tr class="detail-row"><td colspan="22"><details><summary>查看计算详情与数据来源</summary><div class="detail-grid"><span>计划投入：{{ money(row.planned_amount) }}</span><span>到账金额：{{ money(row.net_proceeds) }}</span><span>卖出成本：{{ money(row.cost_basis) }}</span><span>数据源：{{ row.market_snapshot?.data_source }}</span><span>数据更新时间：{{ row.market_snapshot?.data_updated_at ?? '待补全' }}</span><span>{{ row.market_snapshot?.non_trading_day_note || '交易日数据' }}</span></div></details></td></tr></template><tr v-if="!filteredTransactions.length"><td colspan="22" class="empty">暂无符合筛选条件的交易记录</td></tr></tbody></table></div>
  </section>
</template>

<style scoped>
.real-account-hero { border-top: 4px solid #0f766e; background: linear-gradient(122deg, #f7fffb 0%, #fffdf7 66%, #f8fafc 100%); }
.eyebrow { margin: 0 0 4px; color: #0f766e; font-size: .72rem; font-weight: 800; letter-spacing: .11em; }
.compact-create { display: flex; align-items: center; gap: 10px; margin-top: 16px; padding-top: 14px; border-top: 1px solid #dce7e4; }.compact-create input { max-width: 240px; }
.trade-panel { background: #fff; }.real-trade-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }.note-field { grid-column: span 2; }
.auto-note { margin-top: 14px; padding: 10px 12px; color: #475569; background: #f8fafc; border-left: 3px solid #0f766e; font-size: .9rem; }
.preview-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }.preview-grid article { min-height: 86px; padding: 13px; border: 1px solid #d8e5e1; background: #fbfefd; }.preview-grid label { display: block; color: #64748b; font-size: .8rem; }.preview-grid strong { display: block; margin: 4px 0; color: #133f3a; font-size: 1.12rem; }.preview-grid span { display: block; font-size: .78rem; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }.metric { min-height: 116px; }.metric strong { display: block; margin: 7px 0 4px; font-family: Georgia, 'Noto Serif SC', serif; font-size: 1.52rem; }.metric span { color: #64748b; font-size: .82rem; }.up { color: #b42318 !important; }.down { color: #0b6e4f !important; }.warning { color: #9a6700; background: #fff7d6; }
.real-chart { height: 440px; }.chart-controls input { max-width: 135px; }.comparison-strip { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; padding: 11px 0 0; border-top: 1px solid #e5e7eb; color: #64748b; font-size: .88rem; }.comparison-strip b { color: #1e293b; margin-left: 4px; }
.audit-list { display: grid; gap: 12px; margin: 20px 0 0; }.audit-list div { display: flex; justify-content: space-between; padding-bottom: 10px; border-bottom: 1px solid #edf0f1; }.audit-list dt { color: #64748b; }.audit-list dd { margin: 0; font-weight: 700; }
.audit-table { max-height: 650px; }.row-actions { white-space: nowrap; }.text-button { border: 0; background: transparent; color: #0f766e; cursor: pointer; font-weight: 700; }.text-button.danger { color: #b42318; }.detail-row td { background: #fbfdfc; color: #64748b; }.detail-row summary { cursor: pointer; color: #0f766e; font-weight: 700; }.detail-grid { display: flex; flex-wrap: wrap; gap: 12px 24px; padding: 10px 0 2px; font-size: .82rem; }
@media (max-width: 900px) { .real-trade-grid, .preview-grid, .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }.note-field { grid-column: span 2; }.panel-head, .compact-create { align-items: flex-start; flex-direction: column; }.chart-controls { width: 100%; }.real-chart { height: 360px; } }
@media (max-width: 560px) { .real-trade-grid, .preview-grid, .metrics-grid { grid-template-columns: 1fr; }.note-field { grid-column: span 1; }.toolbar { flex-wrap: wrap; }.compact-create input { max-width: none; width: 100%; } }
</style>
