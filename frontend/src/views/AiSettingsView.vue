<script setup lang="ts">
import { onMounted, ref } from "vue";

import {
  analyzeDeepSeek,
  chatDeepSeek,
  deepseekErrorMessage,
  loadDeepSeekConfig,
  saveDeepSeekConfig,
  testDeepSeek,
  type DeepSeekAnalysisResult,
  type DeepSeekChatMessage,
  type DeepSeekTestResult,
} from "../ai-api";

const apiKey = ref("");
const baseUrl = ref("https://api.deepseek.com");
const model = ref("deepseek-chat");
const hasKey = ref(false);
const market = ref("399006");
const saving = ref(false);
const testing = ref(false);
const analyzing = ref(false);
const saveMessage = ref("");
const testResult = ref<DeepSeekTestResult | null>(null);
const testError = ref("");
const analysis = ref<DeepSeekAnalysisResult | null>(null);
const analysisError = ref("");
const chatMessages = ref<DeepSeekChatMessage[]>([]);
const chatInput = ref("");
const chatSending = ref(false);
const chatError = ref("");
const includeMarketContext = ref(true);

async function load() {
  try {
    const config = await loadDeepSeekConfig();
    hasKey.value = config.has_key;
    baseUrl.value = config.base_url;
    model.value = config.model;
    apiKey.value = "";
  } catch (reason) {
    saveMessage.value = deepseekErrorMessage(reason);
  }
}

async function save() {
  saving.value = true;
  saveMessage.value = "";
  try {
    const config = await saveDeepSeekConfig({
      api_key: apiKey.value || undefined,
      base_url: baseUrl.value,
      model: model.value,
    });
    hasKey.value = config.has_key;
    apiKey.value = "";
    saveMessage.value = `已保存（${config.masked_key ?? "密钥未变更"}）`;
  } catch (reason) {
    saveMessage.value = deepseekErrorMessage(reason);
  } finally {
    saving.value = false;
  }
}

async function testConnection() {
  testing.value = true;
  testResult.value = null;
  testError.value = "";
  try {
    testResult.value = await testDeepSeek();
  } catch (reason) {
    testError.value = deepseekErrorMessage(reason);
  } finally {
    testing.value = false;
  }
}

async function runAnalysis() {
  analyzing.value = true;
  analysis.value = null;
  analysisError.value = "";
  try {
    analysis.value = await analyzeDeepSeek(market.value);
  } catch (reason) {
    analysisError.value = deepseekErrorMessage(reason);
  } finally {
    analyzing.value = false;
  }
}

async function sendChat() {
  const content = chatInput.value.trim();
  if (!content || chatSending.value) return;
  chatError.value = "";
  chatMessages.value = [
    ...chatMessages.value,
    { role: "user", content },
  ];
  chatInput.value = "";
  chatSending.value = true;
  try {
    const result = await chatDeepSeek(
      market.value,
      chatMessages.value,
      includeMarketContext.value,
    );
    chatMessages.value = [
      ...chatMessages.value,
      { role: "assistant", content: result.reply },
    ];
  } catch (reason) {
    chatError.value = deepseekErrorMessage(reason);
  } finally {
    chatSending.value = false;
  }
}

function clearChat() {
  chatMessages.value = [];
  chatError.value = "";
}

onMounted(() => void load());
</script>

<template>
  <main class="ai-settings-shell">
    <header class="ai-masthead">
      <p class="kicker">AI CONNECTIVITY TEST · DEEPSEEK</p>
      <h1>DeepSeek API 设置与测试</h1>
      <p>
        仅用于连接与行情分析测试；结果不参与 Champion、Challenger、仓位、交易或晋级。
      </p>
      <router-link to="/">← 返回研究台</router-link>
    </header>

    <section class="panel">
      <h2>DeepSeek API</h2>
      <label>
        API Key（已配置：{{ hasKey ? "是" : "否" }}）
        <input v-model="apiKey" type="password" autocomplete="off" placeholder="sk-..." />
      </label>
      <label>
        Base URL
        <input v-model="baseUrl" type="text" />
      </label>
      <label>
        Model
        <input v-model="model" type="text" />
      </label>
      <div class="actions">
        <button :disabled="saving" @click="save">保存配置</button>
        <button class="primary" :disabled="testing" @click="testConnection">
          {{ testing ? "测试中…" : "测试连接" }}
        </button>
      </div>
      <p v-if="saveMessage" class="message">{{ saveMessage }}</p>
      <div v-if="testResult" class="result ok" data-testid="deepseek-test-ok">
        连接成功 · 模型：{{ testResult.model }} · 响应：{{ testResult.response }} ·
        耗时：{{ testResult.elapsed_ms }} ms
      </div>
      <div v-if="testError" class="result error" data-testid="deepseek-test-error">
        {{ testError }}
      </div>
    </section>

    <section class="panel">
      <h2>AI 分析测试</h2>
      <label>
        市场
        <select v-model="market">
          <option value="399006">399006 创业板指数</option>
          <option value="159941">159941 纳指ETF广发</option>
          <option value="512010">512010 医药ETF易方达</option>
        </select>
      </label>
      <button class="primary" :disabled="analyzing" @click="runAnalysis">
        {{ analyzing ? "分析中…" : "发送当前行情进行 AI 分析" }}
      </button>
      <div v-if="analysis" class="result ok" data-testid="deepseek-analysis-ok">
        <pre>{{ JSON.stringify(analysis, null, 2) }}</pre>
      </div>
      <div v-if="analysisError" class="result error" data-testid="deepseek-analysis-error">
        {{ analysisError }}
      </div>
      <p class="note">分析结果仅测试展示，不写入正式模型/预测/账户数据。</p>
    </section>

    <section class="panel">
      <h2>AI 对话（基于当前行情）</h2>
      <p class="note">
        多轮对话仅保存在当前页面内存中，刷新即清空；不写入任何模型/预测/账户数据。
      </p>
      <label>
        市场
        <select v-model="market">
          <option value="399006">399006 创业板指数</option>
          <option value="159941">159941 纳指ETF广发</option>
          <option value="518600">518600 广发黄金ETF</option>
          <option value="512800">512800 华宝银行ETF</option>
          <option value="512690">512690 鹏华酒ETF</option>
          <option value="512010">512010 医药ETF易方达</option>
        </select>
      </label>
      <label class="inline-label">
        <input
          v-model="includeMarketContext"
          type="checkbox"
          data-testid="deepseek-chat-context-toggle"
        />
        附带当前行情上下文（只读，默认开启）
      </label>
      <div
        v-if="chatMessages.length"
        class="chat-log"
        data-testid="deepseek-chat-log"
      >
        <div
          v-for="(message, index) in chatMessages"
          :key="index"
          class="chat-bubble"
          :class="message.role"
        >
          <strong>{{ message.role === "user" ? "我" : "AI" }}</strong>
          <p>{{ message.content }}</p>
        </div>
      </div>
      <textarea
        v-model="chatInput"
        class="chat-input"
        rows="3"
        placeholder="例如：最近 8 周怎么看？"
        data-testid="deepseek-chat-input"
        @keydown.enter.exact.prevent="sendChat"
      />
      <div class="actions">
        <button
          class="primary"
          :disabled="chatSending || !chatInput.trim()"
          data-testid="deepseek-chat-send"
          @click="sendChat"
        >
          {{ chatSending ? "发送中…" : "发送" }}
        </button>
        <button :disabled="!chatMessages.length" @click="clearChat">
          清空对话
        </button>
      </div>
      <div
        v-if="chatError"
        class="result error"
        data-testid="deepseek-chat-error"
      >
        {{ chatError }}
      </div>
    </section>
  </main>
</template>

<style scoped>
.ai-settings-shell {
  max-width: 760px;
  margin: 0 auto;
  padding: 22px 18px;
  color: #172b3a;
}
.ai-masthead {
  padding-bottom: 16px;
  border-bottom: 2px solid #172b3a;
}
.ai-masthead h1 {
  margin: 6px 0;
}
.ai-masthead p,
.ai-masthead a {
  color: #69747a;
}
.kicker {
  color: #12687e;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.14em;
}
.panel {
  margin-top: 18px;
  padding: 16px;
  border: 1px solid #c9c1b3;
  background: #fffdf6;
}
.panel h2 {
  margin: 0 0 12px;
}
label {
  display: block;
  margin-bottom: 10px;
  color: #657279;
  font-size: 13px;
  font-weight: 700;
}
.inline-label {
  display: flex;
  align-items: center;
  gap: 8px;
}
.inline-label input {
  width: auto;
  margin: 0;
}
input,
select {
  display: block;
  width: 100%;
  margin-top: 4px;
  padding: 8px;
  border: 1px solid #b9b09e;
  background: #fffdf7;
  color: #172b3a;
}
.chat-log {
  max-height: 320px;
  overflow-y: auto;
  margin-bottom: 10px;
  padding: 8px;
  border: 1px solid #d6cdbc;
  background: #faf7ef;
}
.chat-bubble {
  margin: 6px 0;
  padding: 8px 10px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #e3dccb;
}
.chat-bubble.user {
  background: #eef6f8;
  border-color: #bfd8de;
}
.chat-bubble p {
  margin: 4px 0 0;
  white-space: pre-wrap;
  word-break: break-word;
}
.chat-input {
  width: 100%;
  padding: 8px;
  border: 1px solid #b9b09e;
  background: #fffdf7;
  color: #172b3a;
  resize: vertical;
}
.actions {
  display: flex;
  gap: 10px;
  margin-top: 4px;
}
button {
  padding: 9px 14px;
  border: 1px solid #172b3a;
  background: #fffdf6;
  color: #172b3a;
  font-weight: 800;
  cursor: pointer;
}
button.primary {
  background: #172b3a;
  color: #fffdf6;
}
button:disabled {
  opacity: 0.55;
  cursor: wait;
}
.message {
  margin: 10px 0 0;
  color: #12687e;
  font-weight: 700;
}
.result {
  margin-top: 12px;
  padding: 12px;
  border: 1px dashed #12687e;
  background: #f4fbfa;
  white-space: pre-wrap;
  word-break: break-word;
}
.result.error {
  border-color: #a23f32;
  background: #fff0ed;
  color: #8d3026;
}
.result pre {
  margin: 0;
  font-size: 12px;
}
.note {
  margin: 10px 0 0;
  color: #69747a;
  font-size: 12px;
}
</style>
