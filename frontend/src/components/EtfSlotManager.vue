<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  loadV351ReplacementStatus,
  loadV351Slots,
  recalculateV351Indicators,
  replaceV351Slot,
  validateV351Replacement,
  type V351ReplacementPreview,
  type V351Slot,
} from "../v351-api";

const slots = ref<V351Slot[]>([]);
const loading = ref(false);
const error = ref("");
const activeSlot = ref<V351Slot | null>(null);
const inputCode = ref("");
const preview = ref<V351ReplacementPreview | null>(null);
const replacing = ref(false);
const resultMessage = ref("");
const resultIsError = computed(() => /失败|错误/.test(resultMessage.value));

async function loadAll() {
  loading.value = true;
  error.value = "";
  try {
    slots.value = (await loadV351Slots()).sort((a, b) => a.slot_order - b.slot_order);
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    loading.value = false;
  }
}

function openReplace(slot: V351Slot) {
  activeSlot.value = slot;
  inputCode.value = "";
  preview.value = null;
  resultMessage.value = "";
}

async function validateCode() {
  if (!activeSlot.value) return;
  preview.value = null;
  resultMessage.value = "";
  try {
    preview.value = await validateV351Replacement(
      activeSlot.value.slot_id,
      inputCode.value.trim(),
    );
  } catch (reason) {
    resultMessage.value = reason instanceof Error ? reason.message : String(reason);
  }
}

async function confirmReplace() {
  if (!activeSlot.value || !preview.value) return;
  if (!window.confirm(`确认将 ${activeSlot.value.slot_id} 替换为 ${preview.value.target_code}？`)) {
    return;
  }
  replacing.value = true;
  resultMessage.value = "";
  try {
    const result = await replaceV351Slot(
      activeSlot.value.slot_id,
      preview.value.target_code,
    );
    resultMessage.value = `替换任务已提交（${result.job_id}）`;
    preview.value = null;
    const slotId = activeSlot.value.slot_id;
    const deadline = Date.now() + 30 * 60 * 1000;
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const status = await loadV351ReplacementStatus(slotId);
      if (status.state === "SWITCHED") {
        window.dispatchEvent(new Event('etf-universe-changed'));
        const dailyIndicators = await recalculateV351Indicators(status.target_code, "daily");
        if (dailyIndicators.status !== "success") {
          throw new Error(
            `ETF 已替换，但日K指标重算失败：${dailyIndicators.error ?? dailyIndicators.status}`,
          );
        }
        resultMessage.value = "替换任务 SWITCHED，日K指标已更新";
        break;
      }
      if (status.state === "FAILED_ROLLED_BACK") {
        resultMessage.value = status.error_message
          ? `${status.state}：${status.error_message}`
          : `替换任务 ${status.state}`;
        break;
      }
      if (Date.now() > deadline) {
        resultMessage.value = "替换任务超时，请刷新查看状态";
        break;
      }
    }
    await loadAll();
  } catch (reason) {
    resultMessage.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    replacing.value = false;
  }
}

onMounted(() => void loadAll());
</script>

<template>
  <section class="v351-slot-manager" data-testid="etf-slot-manager">
    <div class="v351-heading">
      <div>
        <p class="kicker">ETF SLOT MANAGER · DATA ONLY</p>
        <h2>ETF 替换</h2>
        <p class="panel-note">
          替换仅更新本地行情、日/周/月线与技术指标，不训练模型、不执行 AI 推理。
        </p>
      </div>
      <button class="refresh-slots" type="button" :disabled="loading" @click="loadAll">
        {{ loading ? "加载中…" : "刷新槽位" }}
      </button>
    </div>

    <div v-if="error" class="v351-error">{{ error }}</div>

    <div class="slot-grid">
      <article
        v-for="slot in slots"
        :key="slot.slot_id"
        class="slot-card"
        :data-testid="`slot-${slot.slot_id.toLowerCase()}`"
      >
        <header>
          <b>优先级 {{ String(slot.slot_order).padStart(2, '0') }}</b>
          <span class="status">{{ slot.replacement_status }}</span>
        </header>
        <h3>{{ slot.instrument_code }} · {{ slot.instrument_name }}</h3>
        <dl>
          <div><dt>交易所</dt><dd>{{ slot.exchange }}</dd></div>
          <div><dt>数据起始</dt><dd>{{ slot.data_start_date ?? "—" }}</dd></div>
          <div><dt>数据截止</dt><dd>{{ slot.data_end_date ?? "—" }}</dd></div>
          <div><dt>周线数量</dt><dd>{{ slot.history_week_count }}</dd></div>
          <div><dt>最后更新</dt><dd>{{ slot.last_market_update ?? "—" }}</dd></div>
        </dl>
        <button type="button" class="replace-button" @click="openReplace(slot)">替换</button>
      </article>
    </div>

    <section v-if="activeSlot" class="replace-panel" data-testid="replace-panel">
      <h3>替换 {{ activeSlot.slot_id }}（当前 {{ activeSlot.instrument_code }}）</h3>
      <div class="replace-row">
        <input
          v-model="inputCode"
          data-testid="etf-code-input"
          maxlength="6"
          placeholder="输入 6 位 ETF 代码"
          :disabled="replacing"
        />
        <button type="button" :disabled="replacing || inputCode.trim().length !== 6" @click="validateCode">
          校验并预览
        </button>
      </div>
      <div v-if="preview" class="preview" data-testid="replacement-preview">
        <p><b>{{ preview.target_code }}</b> · {{ preview.metadata.official_name }} · {{ preview.metadata.exchange }}</p>
        <p>将替换：{{ preview.replaced_from_code }}（{{ preview.replaced_from_name }}）</p>
        <p>数据源：{{ preview.metadata.data_source }}</p>
        <button type="button" class="confirm" :disabled="replacing" @click="confirmReplace">
          {{ replacing ? "执行中…" : "确认替换" }}
        </button>
      </div>
      <p v-if="resultMessage" class="result" :class="{ error: resultIsError }">
        {{ resultMessage }}
      </p>
    </section>
  </section>
</template>

<style scoped>
.v351-slot-manager { max-width: 1180px; margin: 0 auto; padding: 18px; }
.v351-heading { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; padding-bottom: 14px; border-bottom: 2px solid #263445; }
.v351-heading h2, .v351-heading p { margin: 0; }
.v351-heading h2 { margin-top: 4px; font-size: 22px; }
.kicker { color: #52789c; font-size: 10px; font-weight: 900; letter-spacing: .14em; }
.panel-note { margin-top: 6px; color: #657184; font-size: 12px; }
.refresh-slots, .replace-button, .replace-row button, .confirm { padding: 8px 12px; border: 1px solid var(--action); background: var(--action); color: #ffffff; font-weight: 800; cursor: pointer; }
.refresh-slots:disabled, .replace-button:disabled, .replace-row button:disabled, .confirm:disabled { opacity: .55; cursor: wait; }
.v351-error { margin-top: 12px; padding: 10px; border: 1px solid #a23f32; color: #a23f32; }
.slot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 14px; }
.slot-card { padding: 14px; border: 1px solid #d5dce5; background: #f4f6f8; }
.slot-card header { display: flex; justify-content: space-between; align-items: center; }
.slot-card header b { color: #52789c; }
.slot-card .status { font-size: 11px; font-weight: 800; padding: 3px 7px; background: #e3e8ee; }
.slot-card h3 { margin: 10px 0; font-size: 16px; }
.slot-card dl { margin: 0; }
.slot-card dl > div { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; font-size: 12px; }
.slot-card dl dt { color: #657184; }
.slot-card dl dd { margin: 0; font-weight: 700; text-align: right; }
.replace-button { width: 100%; margin-top: 12px; }
.replace-panel { margin-top: 18px; padding: 16px; border: 1px solid #d5dce5; background: #f4f6f8; }
.replace-row { display: flex; gap: 10px; margin-top: 10px; }
.replace-row input { flex: 1; padding: 8px; border: 1px solid #d5dce5; font-size: 15px; letter-spacing: .2em; }
.preview { margin-top: 12px; padding: 12px; border: 1px dashed #52789c; }
.preview p { margin: 4px 0; }
.confirm { margin-top: 8px; }
.result { margin-top: 10px; font-weight: 800; color: #52789c; }
.result.error { color: #a23f32; }
@media (max-width: 620px) { .v351-heading { display: block; } .refresh-slots { margin-top: 10px; width: 100%; } .replace-row { flex-direction: column; } }
</style>
