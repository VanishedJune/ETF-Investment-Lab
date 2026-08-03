<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";

import type { ActiveInstrumentCode, PositionEvent } from "../types/research";

type PositionEventWrite = {
  instrument_code: ActiveInstrumentCode;
  direction: "increase" | "decrease";
  operation_date: string;
  change_percent: number;
  note: string | null;
};

type CalendarDay = {
  key: string;
  day: number;
  inCurrentMonth: boolean;
  isToday: boolean;
  increaseCount: number;
  decreaseCount: number;
};

const props = defineProps<{
  instrumentCode: ActiveInstrumentCode;
  events: PositionEvent[];
  positions: Partial<Record<ActiveInstrumentCode, number | null>>;
  loading?: boolean;
  error?: string;
  createEvent: (values: PositionEventWrite) => Promise<void>;
  updateEvent: (
    id: number,
    values: Partial<PositionEventWrite>,
    symbol: ActiveInstrumentCode,
  ) => Promise<void>;
  deleteEvent: (id: number, symbol: ActiveInstrumentCode) => Promise<void>;
}>();

const instrumentNames: Record<ActiveInstrumentCode, string> = {
  "399006": "创业板指数",
  "159941": "广发纳斯达克100ETF",
};
const weekdayLabels = ["日", "一", "二", "三", "四", "五", "六"];

const editingId = ref<number | null>(null);
const saving = ref(false);
const deletingId = ref<number | null>(null);
const confirmDeleteId = ref<number | null>(null);
const localError = ref("");
const todayKey = localDateKey(new Date());
const selectedDate = ref(todayKey);
const selectedParts = parseDateKey(todayKey)!;
const visibleYear = ref(selectedParts.year);
const visibleMonth = ref(selectedParts.month);
const form = ref({
  direction: "increase" as "increase" | "decrease",
  operation_date: todayKey,
  change_percent: 5,
  note: "",
});

const selectedPosition = computed(() => props.positions[props.instrumentCode] ?? null);
const visibleMonthLabel = computed(() => `${visibleYear.value}年${visibleMonth.value + 1}月`);
const eventGroups = computed(() => {
  const groups = new Map<string, PositionEvent[]>();
  for (const event of props.events) {
    const group = groups.get(event.operation_date) ?? [];
    group.push(event);
    groups.set(event.operation_date, group);
  }
  return groups;
});
const calendarDays = computed<CalendarDay[]>(() => {
  const firstDay = makeLocalDate(visibleYear.value, visibleMonth.value, 1);
  const gridStart = makeLocalDate(
    visibleYear.value,
    visibleMonth.value,
    1 - firstDay.getDay(),
  );
  return Array.from({ length: 42 }, (_, index) => {
    const date = makeLocalDate(
      gridStart.getFullYear(),
      gridStart.getMonth(),
      gridStart.getDate() + index,
    );
    const key = localDateKey(date);
    const events = eventGroups.value.get(key) ?? [];
    return {
      key,
      day: date.getDate(),
      inCurrentMonth:
        date.getFullYear() === visibleYear.value && date.getMonth() === visibleMonth.value,
      isToday: key === todayKey,
      increaseCount: events.filter((event) => event.direction === "increase").length,
      decreaseCount: events.filter((event) => event.direction === "decrease").length,
    };
  });
});
const selectedDateEvents = computed(() => sortEvents(
  eventGroups.value.get(selectedDate.value) ?? [],
  false,
));
const sortedEvents = computed(() => sortEvents(props.events, true));
const editingEvent = computed(() =>
  editingId.value === null
    ? null
    : props.events.find((event) => event.id === editingId.value) ?? null,
);
const previewPosition = computed(() => {
  const current = selectedPosition.value ?? 0;
  const original = editingEvent.value;
  const originalSignedChange = original
    ? (original.direction === "increase" ? original.change_percent : -original.change_percent)
    : 0;
  const change = finiteChange(form.value.change_percent);
  return current - originalSignedChange
    + (form.value.direction === "increase" ? change : -change);
});
const previewInvalid = computed(() => previewPosition.value < 0 || previewPosition.value > 100);

function makeLocalDate(year: number, month: number, day: number): Date {
  // Local noon avoids DST and UTC-midnight rollovers while moving between days.
  return new Date(year, month, day, 12, 0, 0, 0);
}

function localDateKey(date: Date): string {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function parseDateKey(value: string): { year: number; month: number; day: number } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]) - 1;
  const day = Number(match[3]);
  const date = makeLocalDate(year, month, day);
  if (
    date.getFullYear() !== year
    || date.getMonth() !== month
    || date.getDate() !== day
  ) return null;
  return { year, month, day };
}

function finiteChange(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function sortEvents(events: readonly PositionEvent[], descending: boolean): PositionEvent[] {
  return [...events].sort((left, right) => {
    const dateOrder = left.operation_date.localeCompare(right.operation_date);
    if (dateOrder !== 0) return descending ? -dateOrder : dateOrder;
    const sequenceOrder = (left.sequence ?? 1) - (right.sequence ?? 1);
    if (sequenceOrder !== 0) return descending ? -sequenceOrder : sequenceOrder;
    return descending ? right.id - left.id : left.id - right.id;
  });
}

function setVisibleMonthFromKey(key: string) {
  const parts = parseDateKey(key);
  if (!parts) return;
  visibleYear.value = parts.year;
  visibleMonth.value = parts.month;
}

function selectDate(key: string, moveMonth = true) {
  if (!parseDateKey(key)) return;
  selectedDate.value = key;
  form.value.operation_date = key;
  if (moveMonth) setVisibleMonthFromKey(key);
  localError.value = "";
}

function changeMonth(offset: number) {
  const target = makeLocalDate(visibleYear.value, visibleMonth.value + offset, 1);
  visibleYear.value = target.getFullYear();
  visibleMonth.value = target.getMonth();
  const selected = parseDateKey(selectedDate.value);
  const preferredDay = selected?.day ?? 1;
  const lastDay = makeLocalDate(visibleYear.value, visibleMonth.value + 1, 0).getDate();
  selectDate(localDateKey(makeLocalDate(
    visibleYear.value,
    visibleMonth.value,
    Math.min(preferredDay, lastDay),
  )), false);
}

async function focusDate(key: string) {
  await nextTick();
  document.querySelector<HTMLButtonElement>(`[data-calendar-date="${key}"]`)?.focus();
}

function onDayKeydown(event: KeyboardEvent, key: string) {
  const parts = parseDateKey(key);
  if (!parts) return;
  const current = makeLocalDate(parts.year, parts.month, parts.day);
  let target: Date | null = null;
  if (event.key === "ArrowLeft") target = makeLocalDate(parts.year, parts.month, parts.day - 1);
  if (event.key === "ArrowRight") target = makeLocalDate(parts.year, parts.month, parts.day + 1);
  if (event.key === "ArrowUp") target = makeLocalDate(parts.year, parts.month, parts.day - 7);
  if (event.key === "ArrowDown") target = makeLocalDate(parts.year, parts.month, parts.day + 7);
  if (event.key === "Home") target = makeLocalDate(parts.year, parts.month, parts.day - current.getDay());
  if (event.key === "End") target = makeLocalDate(parts.year, parts.month, parts.day + 6 - current.getDay());
  if (event.key === "PageUp") target = makeLocalDate(parts.year, parts.month - 1, parts.day);
  if (event.key === "PageDown") target = makeLocalDate(parts.year, parts.month + 1, parts.day);
  if (!target) return;
  event.preventDefault();
  const targetKey = localDateKey(target);
  selectDate(targetKey);
  void focusDate(targetKey);
}

function resetForm(keepSelectedDate = true) {
  editingId.value = null;
  form.value = {
    direction: "increase",
    operation_date: keepSelectedDate ? selectedDate.value : todayKey,
    change_percent: 5,
    note: "",
  };
  localError.value = "";
}

function edit(event: PositionEvent) {
  editingId.value = event.id;
  selectDate(event.operation_date);
  form.value = {
    direction: event.direction,
    operation_date: event.operation_date,
    change_percent: event.change_percent,
    note: event.note ?? "",
  };
  localError.value = "";
}

async function save() {
  if (saving.value) return;
  const change = Number(form.value.change_percent);
  if (!Number.isInteger(change) || change < 5 || change > 100 || change % 5 !== 0) {
    localError.value = "仓位变动必须是5到100之间的5倍数。";
    return;
  }
  if (!parseDateKey(form.value.operation_date)) {
    localError.value = "请选择有效的操作日期。";
    return;
  }
  if (previewInvalid.value) {
    localError.value = previewPosition.value < 0
      ? "本次减仓会使当前仓位低于0%，请调整变动比例。"
      : "本次增仓会使当前仓位超过100%，请调整变动比例。";
    return;
  }
  saving.value = true;
  localError.value = "";
  const values: PositionEventWrite = {
    instrument_code: props.instrumentCode,
    direction: form.value.direction,
    operation_date: form.value.operation_date,
    change_percent: change,
    note: form.value.note.trim() || null,
  };
  try {
    if (editingId.value === null) {
      await props.createEvent(values);
    } else {
      await props.updateEvent(editingId.value, values, props.instrumentCode);
    }
    selectDate(values.operation_date);
    resetForm();
  } catch (reason) {
    localError.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    saving.value = false;
  }
}

async function remove(event: PositionEvent) {
  if (confirmDeleteId.value !== event.id) {
    confirmDeleteId.value = event.id;
    return;
  }
  if (deletingId.value !== null) return;
  deletingId.value = event.id;
  localError.value = "";
  try {
    await props.deleteEvent(event.id, props.instrumentCode);
    if (editingId.value === event.id) resetForm();
  } catch (reason) {
    localError.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    deletingId.value = null;
    confirmDeleteId.value = null;
  }
}

function dayAriaLabel(day: CalendarDay): string {
  const eventText = [
    day.increaseCount ? `增仓${day.increaseCount}笔` : "",
    day.decreaseCount ? `减仓${day.decreaseCount}笔` : "",
  ].filter(Boolean).join("，");
  return `${day.key}${day.isToday ? "，今天" : ""}${eventText ? `，${eventText}` : "，无仓位记录"}`;
}

watch(
  () => props.instrumentCode,
  () => {
    selectedDate.value = todayKey;
    setVisibleMonthFromKey(todayKey);
    resetForm(false);
    confirmDeleteId.value = null;
  },
);

watch(
  () => form.value.operation_date,
  (value) => {
    if (value !== selectedDate.value && parseDateKey(value)) selectDate(value);
  },
);
</script>

<template>
  <section class="position-calendar" aria-labelledby="position-calendar-title">
    <header class="calendar-head">
      <div>
        <p class="kicker">POSITION CALENDAR · LOCAL SQLITE</p>
        <h2 id="position-calendar-title">仓位投资日历</h2>
        <p>在月历中记录两个标的的增仓或减仓百分比；历史记录修改后由本地账本按时间顺序重新计算。</p>
      </div>
      <div class="position-totals" aria-label="当前仓位">
        <div :class="{ active: instrumentCode === '399006' }">
          <span>创业板指数</span>
          <b>{{ positions["399006"] == null ? "0%" : `${positions["399006"]}%` }}</b>
        </div>
        <div :class="{ active: instrumentCode === '159941' }">
          <span>广发纳指ETF</span>
          <b>{{ positions["159941"] == null ? "0%" : `${positions["159941"]}%` }}</b>
        </div>
      </div>
    </header>

    <div class="calendar-workspace">
      <section class="month-panel" aria-label="仓位月历">
        <div class="month-toolbar">
          <h3 aria-live="polite">{{ visibleMonthLabel }}</h3>
          <div class="month-actions">
            <button type="button" aria-label="上一个月" data-testid="calendar-previous-month" @click="changeMonth(-1)">
              <span aria-hidden="true">‹</span>
            </button>
            <button type="button" aria-label="回到今天" class="today-action" @click="selectDate(todayKey)">今天</button>
            <button type="button" aria-label="下一个月" data-testid="calendar-next-month" @click="changeMonth(1)">
              <span aria-hidden="true">›</span>
            </button>
          </div>
        </div>

        <div class="weekday-row" role="row">
          <span v-for="weekday in weekdayLabels" :key="weekday" role="columnheader">{{ weekday }}</span>
        </div>
        <div class="month-grid" role="grid" :aria-label="visibleMonthLabel">
          <button
            v-for="day in calendarDays"
            :key="day.key"
            :class="[
              'calendar-day',
              {
                adjacent: !day.inCurrentMonth,
                selected: selectedDate === day.key,
                today: day.isToday,
                'has-increase': day.increaseCount > 0,
                'has-decrease': day.decreaseCount > 0,
              },
            ]"
            :data-calendar-date="day.key"
            :data-testid="`calendar-day-${day.key}`"
            :aria-label="dayAriaLabel(day)"
            :aria-selected="selectedDate === day.key"
            :tabindex="selectedDate === day.key ? 0 : -1"
            role="gridcell"
            type="button"
            @click="selectDate(day.key)"
            @keydown="onDayKeydown($event, day.key)"
          >
            <span class="day-number">{{ day.day }}</span>
            <span v-if="day.increaseCount || day.decreaseCount" class="event-markers" aria-hidden="true">
              <i v-if="day.increaseCount" class="increase-marker"></i>
              <i v-if="day.decreaseCount" class="decrease-marker"></i>
              <em v-if="day.increaseCount + day.decreaseCount > 2">
                {{ day.increaseCount + day.decreaseCount }}
              </em>
            </span>
          </button>
        </div>
        <div class="calendar-legend" aria-label="仓位事件图例">
          <span><i class="increase-marker"></i>增仓</span>
          <span><i class="decrease-marker"></i>减仓</span>
          <span class="today-key"><i></i>今天</span>
        </div>

        <section class="selected-events" data-testid="selected-date-events" aria-live="polite">
          <div class="selected-events-heading">
            <div>
              <span>选中日期</span>
              <h4>{{ selectedDate }}</h4>
            </div>
            <b>{{ selectedDateEvents.length }} 笔</b>
          </div>
          <p v-if="!selectedDateEvents.length" class="selected-empty">
            当天没有仓位记录，可在右侧新增。
          </p>
          <ol v-else>
            <li v-for="event in selectedDateEvents" :key="event.id">
              <span :class="['event-direction', event.direction]">
                {{ event.direction === "increase" ? "增仓" : "减仓" }}
              </span>
              <strong>{{ event.change_percent }}%</strong>
              <span>操作后 {{ event.position_after }}%</span>
              <small>顺序 {{ event.sequence ?? 1 }}</small>
              <button type="button" @click="edit(event)">修改</button>
            </li>
          </ol>
        </section>
      </section>

      <form class="position-form" novalidate @submit.prevent="save">
        <p class="kicker">{{ editingId === null ? "NEW POSITION EVENT" : "EDIT POSITION EVENT" }}</p>
        <h3>{{ editingId === null ? "记录仓位变化" : "修改仓位记录" }}</h3>
        <div class="selected-market">
          <span>当前标的</span>
          <strong>{{ instrumentNames[instrumentCode] }}</strong>
          <em>{{ instrumentCode }}</em>
        </div>
        <label>
          增仓或减仓
          <select v-model="form.direction" data-testid="position-direction">
            <option value="increase">增加仓位</option>
            <option value="decrease">减少仓位</option>
          </select>
        </label>
        <label>
          操作日期
          <input v-model="form.operation_date" data-testid="position-date" type="date" required />
        </label>
        <label>
          仓位变化（%）
          <input
            v-model.number="form.change_percent"
            data-testid="position-change"
            type="number"
            min="5"
            max="100"
            step="5"
            required
          />
        </label>
        <label>
          备注（可选）
          <textarea v-model="form.note" rows="3" maxlength="1000"></textarea>
        </label>
        <p class="position-preview" :class="{ invalid: previewInvalid }">
          当前总仓位
          <b data-testid="current-position">{{ selectedPosition == null ? "0%" : `${selectedPosition}%` }}</b>
          <span aria-hidden="true">→</span>
          本次操作后估算
          <b>{{ previewPosition }}%</b>
        </p>
        <p class="replay-note">补录或修改历史记录时，最终仓位以保存后从第一笔开始重算的结果为准。</p>
        <div v-if="localError || error" class="form-error" role="alert">{{ localError || error }}</div>
        <div class="form-actions">
          <button
            class="save-position"
            data-testid="position-submit"
            type="submit"
            :disabled="saving || loading"
          >
            {{ saving ? "保存中…" : editingId === null ? "保存仓位记录" : "保存修改" }}
          </button>
          <button v-if="editingId !== null" type="button" class="cancel-edit" @click="resetForm()">
            取消修改
          </button>
        </div>
      </form>
    </div>

    <section class="event-ledger" aria-labelledby="position-history-title">
      <div class="ledger-heading">
        <div>
          <p class="kicker">COMPLETE AUDIT HISTORY</p>
          <h3 id="position-history-title">{{ instrumentNames[instrumentCode] }}完整仓位历史</h3>
        </div>
        <b>{{ events.length }} 条</b>
      </div>
      <div v-if="loading" class="ledger-empty">正在读取本地仓位记录…</div>
      <div v-else-if="!events.length" class="ledger-empty">
        当前按清仓状态（0%）处理。选择月历日期后即可新增第一条记录。
      </div>
      <ol v-else class="event-list">
        <li v-for="event in sortedEvents" :key="event.id" :data-testid="`position-event-${event.id}`">
          <div class="event-date">
            <time :datetime="event.operation_date">{{ event.operation_date }}</time>
            <small>同日顺序 {{ event.sequence ?? 1 }}</small>
          </div>
          <span :class="['event-direction', event.direction]">
            {{ event.direction === "increase" ? "增仓" : "减仓" }}
          </span>
          <strong>{{ event.change_percent }}%</strong>
          <div class="position-after">
            <span>操作后仓位</span>
            <b>{{ event.position_after }}%</b>
          </div>
          <p>{{ event.note || "—" }}</p>
          <div class="event-actions">
            <button type="button" @click="edit(event)">修改</button>
            <button type="button" :disabled="deletingId !== null" @click="remove(event)">
              {{
                deletingId === event.id
                  ? "删除中…"
                  : confirmDeleteId === event.id
                    ? "确认删除"
                    : "删除"
              }}
            </button>
          </div>
        </li>
      </ol>
    </section>
  </section>
</template>

<style scoped>
.position-calendar {
  --calendar-bg: #f4f7fb;
  --calendar-paper: #ffffff;
  --calendar-ink: #172b3a;
  --calendar-muted: #7a8792;
  --calendar-blue: #0b67d7;
  --calendar-red: #d9553f;
  --calendar-teal: #14836d;
  --calendar-rule: #d8e0e8;
  margin-top: 16px;
  border: 1px solid #b9c4ce;
  color: var(--calendar-ink);
  background: var(--calendar-bg);
  box-shadow: 0 14px 38px rgba(23, 43, 58, .08);
}

.calendar-head {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-end;
  padding: 24px;
  border-bottom: 1px solid #b9c4ce;
  background: var(--calendar-paper);
}

.calendar-head h2,
.month-toolbar h3,
.position-form h3,
.ledger-heading h3,
.selected-events h4 {
  margin: 0;
  color: var(--calendar-ink);
  font-family: "Noto Serif SC", Georgia, serif;
}

.calendar-head h2 { font-size: 30px; }
.calendar-head > div > p:not(.kicker) { margin: 8px 0 0; color: var(--calendar-muted); font-size: 14px; }
.position-totals { display: grid; grid-template-columns: repeat(2, minmax(150px, 1fr)); border: 1px solid #b9c4ce; }
.position-totals div { padding: 10px 14px; border-left: 1px solid #b9c4ce; }
.position-totals div:first-child { border-left: 0; }
.position-totals div.active { color: #fff; background: var(--calendar-ink); }
.position-totals span { display: block; color: inherit; opacity: .72; font-size: 12px; }
.position-totals b { display: block; margin-top: 5px; font: 700 22px Georgia, serif; }

.calendar-workspace {
  display: grid;
  grid-template-columns: minmax(480px, 1.15fr) minmax(330px, .85fr);
  gap: 1px;
  background: #b9c4ce;
}
.month-panel,
.position-form { min-width: 0; background: var(--calendar-paper); }
.month-panel { padding: 22px 24px 24px; }
.position-form { padding: 24px; }

.month-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.month-toolbar h3 { font-size: 22px; letter-spacing: -.01em; }
.month-actions { display: flex; align-items: center; gap: 4px; }
.month-actions button {
  display: grid;
  width: 38px;
  height: 38px;
  place-items: center;
  border: 1px solid transparent;
  border-radius: 50%;
  color: #53606b;
  background: transparent;
  font-size: 24px;
  line-height: 1;
}
.month-actions .today-action { width: auto; padding: 0 10px; border-radius: 7px; font-size: 12px; font-weight: 800; }
.month-actions button:hover { border-color: var(--calendar-rule); background: #f3f6fa; }
.month-actions button:focus-visible,
.calendar-day:focus-visible,
.position-form :is(input, select, textarea, button):focus-visible,
.event-actions button:focus-visible,
.selected-events button:focus-visible {
  outline: 3px solid rgba(11, 103, 215, .3);
  outline-offset: 2px;
}

.weekday-row,
.month-grid { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); }
.weekday-row { margin-top: 17px; }
.weekday-row span { padding: 8px 0; color: #53606b; text-align: center; font-size: 12px; font-weight: 800; }
.month-grid { gap: 4px; }
.calendar-day {
  position: relative;
  display: grid;
  min-height: 52px;
  place-items: center;
  padding: 5px 3px 13px;
  border: 0;
  border-radius: 9px;
  color: var(--calendar-ink);
  background: transparent;
  font-size: 13px;
  transition: color .15s ease, background-color .15s ease, transform .15s ease;
}
.calendar-day:hover { background: #eef4fb; transform: translateY(-1px); }
.calendar-day.adjacent { color: #a5afb8; }
.calendar-day.today:not(.selected) .day-number::after {
  position: absolute;
  inset: -7px;
  border: 1.5px solid var(--calendar-blue);
  border-radius: 50%;
  content: "";
  pointer-events: none;
}
.calendar-day.selected { color: var(--calendar-ink); background: transparent; box-shadow: none; }
.calendar-day.selected .day-number { display: grid; width: 32px; height: 32px; place-items: center; border-radius: 50%; color: #fff; background: var(--calendar-blue); box-shadow: 0 5px 12px rgba(11, 103, 215, .22); }
.day-number { position: relative; z-index: 1; }
.event-markers { position: absolute; bottom: 6px; left: 50%; display: flex; align-items: center; gap: 3px; transform: translateX(-50%); }
.event-markers i,
.calendar-legend i { display: block; width: 6px; height: 6px; border-radius: 50%; }
.increase-marker { background: var(--calendar-red); }
.decrease-marker { background: var(--calendar-teal); }
.calendar-day.selected .event-markers em { color: var(--calendar-blue); }
.event-markers em { color: currentColor; font-size: 8px; font-style: normal; font-weight: 900; }
.calendar-legend { display: flex; justify-content: flex-end; gap: 15px; margin-top: 13px; color: var(--calendar-muted); font-size: 11px; }
.calendar-legend span { display: flex; align-items: center; gap: 5px; }
.calendar-legend .today-key i { border: 1.5px solid var(--calendar-blue); background: transparent; }

.selected-events { margin-top: 18px; border: 1px solid var(--calendar-rule); border-radius: 10px; overflow: hidden; }
.selected-events-heading { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; background: #f4f7fb; }
.selected-events-heading span { color: var(--calendar-muted); font-size: 11px; }
.selected-events h4 { margin-top: 2px; font-size: 17px; }
.selected-events-heading > b { color: var(--calendar-blue); }
.selected-empty { margin: 0; padding: 19px 14px; color: var(--calendar-muted); font-size: 13px; }
.selected-events ol { margin: 0; padding: 0; list-style: none; }
.selected-events li { display: grid; grid-template-columns: 52px 50px minmax(92px, 1fr) 55px auto; gap: 9px; align-items: center; padding: 10px 13px; border-top: 1px solid var(--calendar-rule); font-size: 12px; }
.selected-events li > small { color: var(--calendar-muted); }
.selected-events li > button { padding: 5px 8px; border: 1px solid var(--calendar-rule); border-radius: 5px; color: var(--calendar-blue); background: #fff; font-weight: 800; }

.position-form h3 { margin-bottom: 18px; font-size: 24px; }
.position-form label { display: block; margin-top: 13px; color: #596873; font-size: 13px; font-weight: 700; }
.position-form input,
.position-form select,
.position-form textarea { width: 100%; margin-top: 6px; padding: 10px 11px; border: 1px solid #b9c4ce; border-radius: 6px; color: var(--calendar-ink); background: #fff; font-size: 15px; }
.position-form textarea { resize: vertical; }
.selected-market { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; padding: 12px; border-radius: 7px; color: #fff; background: var(--calendar-ink); }
.selected-market span { grid-column: 1 / -1; color: #b8c2c7; font-size: 12px; }
.selected-market em { font-style: normal; font-weight: 800; }
.position-preview { padding: 12px; border-left: 4px solid var(--calendar-blue); background: #eaf2fc; color: #596873; font-size: 13px; line-height: 1.7; }
.position-preview.invalid { border-color: var(--calendar-red); background: #f9ece8; }
.position-preview b { color: var(--calendar-ink); font-size: 17px; }
.position-preview span { margin: 0 6px; color: var(--calendar-blue); }
.replay-note { margin: -4px 0 0; color: var(--calendar-muted); font-size: 11px; line-height: 1.55; }
.form-error { margin-top: 12px; padding: 10px; color: #9b352a; background: #f8e9e4; font-size: 13px; }
.form-actions { display: flex; gap: 8px; margin-top: 14px; }
.save-position,
.cancel-edit { padding: 11px 15px; border: 1px solid var(--calendar-ink); border-radius: 6px; font-weight: 900; }
.save-position { color: #fff; background: var(--calendar-blue); border-color: var(--calendar-blue); }
.save-position:disabled { cursor: not-allowed; opacity: .55; }
.cancel-edit { color: var(--calendar-ink); background: transparent; }

.event-ledger { border-top: 1px solid #b9c4ce; background: var(--calendar-paper); }
.ledger-heading { display: flex; justify-content: space-between; gap: 16px; padding: 20px 22px 14px; border-bottom: 1px solid var(--calendar-rule); }
.ledger-heading h3 { font-size: 24px; }
.ledger-heading > b { color: var(--calendar-blue); font-size: 18px; }
.ledger-empty { display: grid; min-height: 160px; place-items: center; padding: 24px; color: var(--calendar-muted); text-align: center; }
.event-list { max-height: 460px; margin: 0; padding: 0; overflow: auto; list-style: none; }
.event-list li { display: grid; grid-template-columns: 108px 54px 60px 110px minmax(100px, 1fr) auto; gap: 13px; align-items: center; padding: 14px 20px; border-bottom: 1px solid #e3e9ee; }
.event-date time { display: block; font-weight: 800; }
.event-date small,
.position-after span { color: var(--calendar-muted); font-size: 11px; }
.event-direction { padding: 5px 7px; border-radius: 4px; color: #fff; text-align: center; font-size: 12px; font-weight: 900; }
.event-direction.increase { background: var(--calendar-red); }
.event-direction.decrease { background: var(--calendar-teal); }
.position-after span,
.position-after b { display: block; }
.event-list p { margin: 0; overflow: hidden; color: var(--calendar-muted); text-overflow: ellipsis; white-space: nowrap; }
.event-actions { display: flex; gap: 5px; }
.event-actions button { padding: 6px 8px; border: 1px solid #b9c4ce; border-radius: 5px; color: var(--calendar-ink); background: transparent; }

@media (prefers-reduced-motion: reduce) {
  .calendar-day { transition: none; }
}

@media (max-width: 980px) {
  .calendar-workspace { grid-template-columns: 1fr; }
  .month-panel { border-bottom: 1px solid #b9c4ce; }
}

@media (max-width: 720px) {
  .calendar-head { display: block; }
  .position-totals { margin-top: 16px; grid-template-columns: 1fr 1fr; }
  .position-totals div { min-width: 0; }
  .month-panel { padding: 18px 10px; }
  .calendar-day { min-height: 48px; }
  .selected-events li { grid-template-columns: 48px 44px 1fr auto; }
  .selected-events li > small { display: none; }
  .event-list li { grid-template-columns: 88px 52px 1fr; }
  .position-after,
  .event-list p,
  .event-actions { grid-column: 3; }
}

@media (max-width: 460px) {
  .position-totals { grid-template-columns: 1fr; }
  .position-totals div { border-left: 0; border-top: 1px solid #b9c4ce; }
  .position-totals div:first-child { border-top: 0; }
  .month-toolbar h3 { font-size: 19px; }
  .month-actions button { width: 34px; height: 34px; }
  .month-actions .today-action { display: none; }
  .calendar-day { min-height: 43px; padding-inline: 1px; }
  .calendar-legend { justify-content: flex-start; }
}
</style>
