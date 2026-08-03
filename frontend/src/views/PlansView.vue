<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, money } from "../api";

type Plan = {
  id: number;
  name: string;
  instrument_code: string;
  weekly_amount: string | number;
  execution_weekday: number;
  mode: string;
  enabled: boolean;
};

const plans = ref<Plan[]>([]);
const instruments = ref<any[]>([]);
const message = ref("");
const error = ref("");
const editingPlanId = ref<number | null>(null);
const savingPlanId = ref<number | null>(null);
const editedWeeklyAmount = ref("");
const form = ref({
  instrument_code: "589850",
  name: "",
  weekly_amount: "150",
  execution_weekday: 1,
  mode: "REAL_LOT",
  lot_size: 100,
  allow_pause: true,
  enabled: true,
});

async function load() {
  try {
    [plans.value, instruments.value] = await Promise.all([api<Plan[]>("/plans"), api<any[]>("/instruments")]);
  } catch (reason: any) {
    error.value = reason.message;
  }
}

async function save() {
  error.value = "";
  message.value = "";
  try {
    const body = { ...form.value, weekly_amount: Number(form.value.weekly_amount) };
    await api("/plans", { method: "POST", body: JSON.stringify(body) });
    message.value = "定投计划已写入 SQLite。";
    form.value.name = "";
    await load();
  } catch (reason: any) {
    error.value = reason.message;
  }
}

function beginAmountEdit(plan: Plan) {
  error.value = "";
  message.value = "";
  editingPlanId.value = plan.id;
  editedWeeklyAmount.value = String(plan.weekly_amount);
}

function cancelAmountEdit() {
  editingPlanId.value = null;
  editedWeeklyAmount.value = "";
}

async function updateWeeklyAmount(plan: Plan) {
  const weeklyAmount = Number(editedWeeklyAmount.value);
  if (!Number.isFinite(weeklyAmount) || weeklyAmount <= 0) {
    error.value = "定投金额必须大于 0。";
    return;
  }
  error.value = "";
  message.value = "";
  savingPlanId.value = plan.id;
  try {
    const updated = await api<Plan>(`/plans/${plan.id}`, {
      method: "PATCH",
      body: JSON.stringify({ weekly_amount: weeklyAmount }),
    });
    plans.value = plans.value.map((item) => item.id === plan.id ? updated : item);
    editingPlanId.value = null;
    editedWeeklyAmount.value = "";
    message.value = `已将“${plan.name}”单次定投金额更新为 ${money(updated.weekly_amount)}。`;
  } catch (reason: any) {
    error.value = reason.message;
  } finally {
    savingPlanId.value = null;
  }
}

async function toggle(plan: Plan) {
  try {
    const updated = await api<Plan>(`/plans/${plan.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !plan.enabled }) });
    plans.value = plans.value.map((item) => item.id === plan.id ? updated : item);
  } catch (reason: any) {
    error.value = reason.message;
  }
}

async function remove(plan: Plan) {
  if (!confirm(`确定删除“${plan.name}”？`)) return;
  try {
    await api(`/plans/${plan.id}?confirmed=true`, { method: "DELETE" });
    plans.value = plans.value.filter((item) => item.id !== plan.id);
    if (editingPlanId.value === plan.id) cancelAmountEdit();
  } catch (reason: any) {
    error.value = reason.message;
  }
}

onMounted(load);
</script>

<template>
  <div class="grid cols-2">
    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>新建每周定投</h2>
          <p class="muted">计划是可编辑的本地规则；实际执行前会再次校验金额、手数和费用。</p>
        </div>
      </div>
      <div class="form-grid">
        <label class="field">指数或基金
          <select v-model="form.instrument_code"><option v-for="instrument in instruments" :key="instrument.code" :value="instrument.code">{{ instrument.code }} · {{ instrument.name }}</option></select>
        </label>
        <label class="field">计划名称<input v-model="form.name" placeholder="例如：科创50每周定投" /></label>
        <label class="field">每周金额（元）<input v-model="form.weekly_amount" type="number" min="0.01" /></label>
        <label class="field">执行日（0=周一）<input v-model.number="form.execution_weekday" type="number" min="0" max="6" /></label>
        <label class="field">成交模式
          <select v-model="form.mode"><option value="REAL_LOT">整手（100份）</option><option value="FRACTIONAL">允许零碎份额</option></select>
        </label>
        <label class="field">手数基数<input v-model.number="form.lot_size" type="number" min="1" /></label>
      </div>
      <div class="section-gap"><button class="button" @click="save">保存计划</button></div>
      <div v-if="error" class="status error section-gap">{{ error }}</div>
      <div v-if="message" class="status section-gap" data-testid="plan-edit-message">{{ message }}</div>
    </section>

    <section class="panel">
      <h3>规则说明</h3>
      <div class="mini-list section-gap">
        <div><span>估值 / 趋势 / 动量</span><b>本地规则评分</b></div>
        <div><span>手续费</span><b>按交易时配置冻结</b></div>
        <div><span>资金不足</span><b>现金结转，不透支</b></div>
        <div><span>执行记录</span><b>SQLite 事务保存</b></div>
      </div>
    </section>
  </div>

  <section class="panel section-gap">
    <div class="panel-head">
      <div>
        <h2>已保存计划</h2>
        <p class="muted">修改金额只更新后续定投；历史交易与已经保存的账务记录不会被改写。</p>
      </div>
    </div>
    <div v-if="!plans.length" class="empty">暂无计划。</div>
    <div v-else class="table-wrap">
      <table class="data-table">
        <thead><tr><th>名称</th><th>指数 / 基金</th><th>每周金额</th><th>执行日</th><th>模式</th><th>状态</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="plan in plans" :key="plan.id">
            <td>{{ plan.name }}</td>
            <td>{{ plan.instrument_code }}</td>
            <td>
              <template v-if="editingPlanId === plan.id">
                <input :data-testid="`plan-weekly-amount-${plan.id}`" v-model="editedWeeklyAmount" type="number" min="0.01" step="0.01" aria-label="修改每周定投金额" />
              </template>
              <template v-else>{{ money(plan.weekly_amount) }}</template>
            </td>
            <td>周{{ plan.execution_weekday + 1 }}</td>
            <td>{{ plan.mode }}</td>
            <td><span class="tag">{{ plan.enabled ? "启用" : "暂停" }}</span></td>
            <td class="plan-actions">
              <template v-if="editingPlanId === plan.id">
                <button class="button" :data-testid="`save-plan-${plan.id}`" :disabled="savingPlanId === plan.id" @click="updateWeeklyAmount(plan)">{{ savingPlanId === plan.id ? "保存中…" : "保存" }}</button>
                <button class="button ghost" :disabled="savingPlanId === plan.id" @click="cancelAmountEdit">取消</button>
              </template>
              <template v-else>
                <button class="button ghost" :data-testid="`edit-plan-${plan.id}`" @click="beginAmountEdit(plan)">修改金额</button>
                <button class="button ghost" @click="toggle(plan)">{{ plan.enabled ? "暂停" : "启用" }}</button>
                <button class="button danger" @click="remove(plan)">删除</button>
              </template>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.plan-actions { min-width: 238px; white-space: nowrap; }
.plan-actions .button + .button { margin-left: 6px; }
td input { width: 130px; color: #263540; border: 1px solid #8aa5bb; border-radius: 2px; padding: 7px 8px; }
</style>
