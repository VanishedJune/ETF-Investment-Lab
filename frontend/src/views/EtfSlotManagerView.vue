<script setup lang="ts">
import EtfSlotManager from "../components/EtfSlotManager.vue";
</script>

<template>
  <main class="etf-slot-shell">
    <header class="slot-masthead">
      <p class="kicker">LOCAL MARKET DATA · ETF SLOTS</p>
      <h1>ETF 替换</h1>
      <p>后台替换只负责获取行情并生成日/周/月线指标，投资日历中的历史记录保持不变。</p>
      <router-link to="/">← 返回行情面板</router-link>
    </header>
    <EtfSlotManager />
  </main>
</template>

<style scoped>
.etf-slot-shell { min-height: 100vh; background: #f4f6f8; color: #263445; }
.slot-masthead { max-width: 1180px; margin: 0 auto; padding: 22px 18px 0; }
.slot-masthead h1 { margin: 6px 0; font-size: 28px; }
.slot-masthead p { margin: 4px 0; color: #657184; }
.slot-masthead a { color: #52789c; font-weight: 800; }
</style>
