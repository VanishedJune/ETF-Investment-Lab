<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, money, percent } from "../api";

const data = ref<any>(); const error = ref(""); const loading = ref(true); const refreshes = ref<Record<string, string>>({});
async function load() { loading.value = true; error.value = ""; try { data.value = await api<any>("/dashboard"); } catch (e: any) { error.value = e.message; } finally { loading.value = false; } }
async function refresh(code: string) { refreshes.value[code] = "正在更新公开行情…"; try { const result = await api<any>(`/market/${code}/refresh`, { method:"POST" }); refreshes.value[code] = result.demo ? "公开数据不可用，已使用明确标记的演示数据。" : "本地行情缓存已更新。"; await load(); } catch (e:any) { refreshes.value[code] = e.message; } }
onMounted(load);
</script>
<template>
  <div v-if="error" class="status error">{{ error }}</div><div v-else-if="loading" class="empty">正在读取本地 SQLite 研究数据…</div>
  <template v-else>
    <div class="grid cols-3"><article class="panel metric"><label>跟踪指数</label><strong>{{ data.instruments.length }}</strong></article><article class="panel metric"><label>模拟账户</label><strong>{{ data.simulation_accounts }}</strong></article><article class="panel metric"><label>真实记录账户</label><strong>{{ data.real_accounts }}</strong></article></div>
    <section class="section-gap grid cols-3"><article v-for="item in data.instruments" :key="item.code" class="panel"><div class="split"><span class="tag" :class="{demo:item.latest?.source==='DEMO'}">{{ item.code }}</span><span v-if="item.latest?.source" class="muted">{{ item.latest.source }}</span></div><h3 style="margin-top:14px">{{ item.name }}</h3><template v-if="item.latest"><strong style="display:block;font-size:28px;font-family:Georgia,serif;margin-top:24px">{{ money(item.latest.close) }}</strong><span :class="Number(item.change) >= 0 ? 'up':'down'" style="font-size:12px">{{ percent(item.change) }} / {{ item.latest.date }}</span></template><p v-else class="muted">尚无本地行情。更新失败时会安全回退到缓存或演示数据。</p><div class="section-gap"><button class="button ghost" @click="refresh(item.code)">更新行情</button></div><p v-if="refreshes[item.code]" class="muted">{{ refreshes[item.code] }}</p></article></section>
    <section class="panel section-gap"><div class="panel-head"><div><h2>研究边界</h2><p class="muted">{{ data.disclaimer }} 当前市场层使用直接指数行情，而非 ETF 行情。</p></div><RouterLink class="button ghost" to="/vault">查看数据金库</RouterLink></div></section>
  </template>
</template>
