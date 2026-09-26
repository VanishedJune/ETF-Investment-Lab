<script setup lang="ts">
import { onMounted, ref } from "vue";

import {
  v37Ai,
  v37Champions,
  v37Forecast,
  v37Funnel,
  v37Fusion,
  v37ModelConflicts,
  v37RepairProposals,
  v37Simulation,
  v37Status,
  type V37AiForecastPayload,
  type V37Champion,
  type V37FunnelPayload,
  type V37FusionPayload,
  type V37Status,
} from "../v37-api";

const market = ref("399006");
const markets = [
  "399006",
  "159941",
  "518600",
  "512800",
  "512690",
  "512010",
];
const status = ref<V37Status | null>(null);
const champion = ref<V37Champion | null>(null);
const forecast = ref<Record<string, unknown> | null>(null);
const ai = ref<V37AiForecastPayload | null>(null);
const fusion = ref<V37FusionPayload | null>(null);
const funnel = ref<V37FunnelPayload | null>(null);
const simulation = ref<Record<string, unknown> | null>(null);
const conflicts = ref<Array<Record<string, unknown>>>([]);
const proposals = ref<Array<Record<string, unknown>>>([]);
const loading = ref(false);
const error = ref("");

async function loadAll() {
  loading.value = true;
  error.value = "";
  try {
    const [s, c, f, a, fu, fun, sim, conf, prop] = await Promise.all([
      v37Status(market.value),
      v37Champions(market.value),
      v37Forecast(market.value).catch(() => null),
      v37Ai(market.value),
      v37Fusion(market.value),
      v37Funnel(market.value),
      v37Simulation(market.value).catch(() => null),
      v37ModelConflicts(market.value),
      v37RepairProposals(),
    ]);
    status.value = s;
    champion.value = c;
    forecast.value = f;
    ai.value = a;
    fusion.value = fu;
    funnel.value = fun;
    simulation.value = sim;
    conflicts.value = conf;
    proposals.value = prop;
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    loading.value = false;
  }
}

onMounted(() => void loadAll());
</script>

<template>
  <main class="v37-shell">
    <header class="v37-masthead">
      <p class="kicker">V3.7 MULTI-MODEL</p>
      <h1>多模型分析（Local / DeepSeek / Fusion / Repair）</h1>
      <p>
        V3.7 独立命名空间；数据未回放时显示空状态，回放后自动填充。
        DeepSeek 仅有预测权/提案权，正式权重依赖 Forward OOS。
      </p>
      <div class="toolbar">
        <select v-model="market" @change="loadAll">
          <option v-for="code in markets" :key="code" :value="code">
            {{ code }}
          </option>
        </select>
        <button :disabled="loading" @click="loadAll">
          {{ loading ? "加载中…" : "刷新" }}
        </button>
        <router-link to="/ai-settings">AI 设置 →</router-link>
        <router-link to="/">← 返回研究台</router-link>
      </div>
    </header>

    <div v-if="error" class="error">{{ error }}</div>

    <section class="panel">
      <h2>运行状态</h2>
      <pre>{{ status ? JSON.stringify(status, null, 2) : "无状态" }}</pre>
    </section>

    <section class="panel">
      <h2>LOCAL_QUANT（Champion）</h2>
      <pre>{{ champion ? JSON.stringify(champion, null, 2) : "无 Champion" }}</pre>
      <h3>最近预测</h3>
      <pre>{{ forecast ? JSON.stringify(forecast, null, 2) : "无 V3.7 预测" }}</pre>
    </section>

    <section class="panel">
      <h2>DEEPSEEK_AI（独立分析）</h2>
      <pre>{{ ai ? JSON.stringify(ai, null, 2) : "无 AI 预测" }}</pre>
      <h3 v-if="ai && ai.health">AI 权重/校准状态</h3>
      <pre v-if="ai && ai.health">{{ JSON.stringify(ai.health, null, 2) }}</pre>
    </section>

    <section class="panel">
      <h2>决策漏斗（预测→Score→Target→Trade→Position）</h2>
      <template v-if="funnel">
        <h3>LOCAL（v36_decision_funnel）</h3>
        <pre>{{ JSON.stringify(funnel.local, null, 2) }}</pre>
        <h3>DEEPSEEK_AI</h3>
        <pre>{{ JSON.stringify(funnel.ai, null, 2) }}</pre>
        <h3>QUANT_AI_FUSION（{{ funnel.fusion_config?.config_version ?? "无配置" }}）</h3>
        <pre>{{ JSON.stringify(funnel.fusion, null, 2) }}</pre>
      </template>
      <pre v-else>无漏斗数据</pre>
    </section>

    <section class="panel">
      <h2>QUANT_AI_FUSION</h2>
      <pre>{{ fusion ? JSON.stringify(fusion, null, 2) : "无 Fusion 配置" }}</pre>
    </section>

    <section class="panel">
      <h2>账户与流水（同一执行引擎）</h2>
      <pre>{{ simulation ? JSON.stringify(simulation, null, 2) : "无账户" }}</pre>
    </section>

    <section class="panel">
      <h2>模型冲突</h2>
      <pre>{{ conflicts.length ? JSON.stringify(conflicts, null, 2) : "无冲突记录" }}</pre>
    </section>

    <section class="panel">
      <h2>Repair 提案（LOCAL_REPAIR_CHALLENGER）</h2>
      <pre>{{ proposals.length ? JSON.stringify(proposals, null, 2) : "无提案" }}</pre>
    </section>
  </main>
</template>

<style scoped>
.v37-shell {
  max-width: 1080px;
  margin: 0 auto;
  padding: 22px 18px;
  color: #172b3a;
}
.v37-masthead {
  padding-bottom: 14px;
  border-bottom: 2px solid #172b3a;
}
.v37-masthead h1 {
  margin: 6px 0;
}
.v37-masthead p,
.v37-masthead a {
  color: #69747a;
}
.kicker {
  color: #12687e;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.14em;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
}
.toolbar select,
.toolbar button {
  padding: 7px 10px;
  border: 1px solid #b9b09e;
  background: #fffdf7;
  color: #172b3a;
}
.panel {
  margin-top: 16px;
  padding: 14px;
  border: 1px solid #c9c1b3;
  background: #fffdf6;
}
.panel h2 {
  margin: 0 0 8px;
}
.panel h3 {
  margin: 14px 0 6px;
}
.panel pre {
  max-height: 360px;
  overflow: auto;
  margin: 0;
  padding: 10px;
  background: #f6f2e8;
  border: 1px solid #e3dccb;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.error {
  margin-top: 12px;
  padding: 10px;
  border: 1px dashed #a23f32;
  background: #fff0ed;
  color: #8d3026;
}
</style>
