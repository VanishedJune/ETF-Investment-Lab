import { expect, type Page, test } from "@playwright/test";

const completedTask = {
  id: "weekly-v2-0123456789abcdef0123456789abcdef",
  instrument_code: "399006",
  status: "completed",
  completed_weeks: 521,
  total_weeks: 521,
  last_iteration: 521,
  last_work_version: "W0521",
  last_model_version: "M0018",
  last_completed_week: "2026-W30",
  progress: { stage: "completed" },
  message: null,
  reused: false,
};

const metricsPayload = {
  symbol: "399006",
  model_version: "M0018",
  iteration_count: 521,
  last_iteration: 521,
  accepted_iterations: 17,
  rejected_iterations: 504,
  feedback_count: 508,
  mature_count: 508,
  partial_count: 9,
  pending_count: 4,
  candidate_acceptance_rate: "0.0326",
  windows: {
    "20": {
      instrument_code: "399006",
      window: "20",
      sample_count: 20,
      total_loss: "1.8",
      direction_hit_rate: "0.70",
      calibration_loss: "0.16",
      overtrade_penalty: "0.05",
    },
    "52": {
      instrument_code: "399006",
      window: "52",
      sample_count: 52,
      total_loss: "1.9",
      direction_hit_rate: "0.673",
      calibration_loss: "0.18",
      overtrade_penalty: "0.05",
    },
  },
  curve: Array.from({ length: 60 }, (_, index) => ({
    iteration_number: index + 1,
    iteration_id: `I${String(index + 1).padStart(4, "0")}`,
    cutoff_date: `2025-${String((index % 12) + 1).padStart(2, "0")}-14`,
    model_version: index < 30 ? "M0001" : "M0018",
    accepted: index === 30,
    deviation_days: index < 8 ? null : (index % 7) - 3,
    absolute_deviation: index < 8 ? null : String(Math.abs((index % 7) - 3)),
    rolling_20_abs_deviation: index < 27 ? null : String(4.2 - (index - 27) * 0.04),
    rolling_52_abs_deviation: index < 59 ? null : "2.40",
    direction_correct: index < 8 ? null : index % 3 !== 0,
  })),
};

const advicePayload = {
  id: 9,
  instrument_code: "399006",
  iteration_id: "I0521",
  work_version: "W0521",
  model_version: "M0018",
  advice_generation: 1,
  advice_at: "2026-07-31T02:03:00Z",
  advice_version: "weekly-v2",
  audit_fields: {},
  audit_notes: [],
  batches: [
    {
      confirmation_condition: "DIF保持在DEA上方且价格站稳20日均线",
      expected_date: "2026-08-04",
      operation_side: "buy",
      percent: 20,
      sequence: 1,
      status: "ready",
      tolerance_trading_days: 3,
    },
    {
      confirmation_condition: "周线MACD柱继续扩大",
      expected_date: "2026-08-18",
      operation_side: "buy",
      percent: 15,
      sequence: 2,
      status: "waiting_confirmation",
      tolerance_trading_days: 3,
    },
    {
      confirmation_condition: "黄金点保持确认",
      expected_date: "2026-09-01",
      operation_side: "buy",
      percent: 15,
      sequence: 3,
      status: "waiting_confirmation",
      tolerance_trading_days: 3,
    },
    {
      confirmation_condition: "回踩不跌破20日均线",
      expected_date: "2026-09-15",
      operation_side: "buy",
      percent: 10,
      sequence: 4,
      status: "waiting_confirmation",
      tolerance_trading_days: 3,
    },
  ],
  confidence: "0.80",
  current_position: 20,
  data_cutoff_date: "2026-07-31",
  direction: "up",
  direction_probabilities: { up: "0.78", down: "0.14", neutral: "0.08" },
  feature_set_version: "weekly-v2",
  forecast_horizon_weeks: 13,
  fund_etf_ratio: "7:3",
  market_state: "低估并出现反转共振",
  operation_side: "buy",
  probability: "0.78",
  recommendation: "staged_increase",
  signed_position_change: 60,
  source_data_max_date: "2026-07-31",
  target_position: 80,
  target_position_range: [75, 80],
  total_adjustment: 60,
};

async function mockAnalysis(page: Page) {
  let getCount = 0;
  await page.route("**/api/v2/analysis/tasks", async (route) => {
    await route.fulfill({
      status: 202,
      json: { ...completedTask, status: "queued", completed_weeks: 0, last_iteration: 0 },
    });
  });
  await page.route("**/api/v2/analysis/tasks/*", async (route) => {
    getCount += 1;
    if (getCount === 1) {
      await route.fulfill({
        json: {
          ...completedTask,
          status: "iterating",
          completed_weeks: 260,
          last_iteration: 260,
          last_work_version: "W0260",
          last_model_version: "M0009",
        },
      });
      return;
    }
    await route.fulfill({ json: completedTask });
  });
  await page.route("**/api/v2/models/*/metrics", async (route) => {
    const symbol = route.request().url().includes("/NDX/") ? "NDX" : "399006";
    await route.fulfill({
      json: {
        ...metricsPayload,
        symbol,
        windows: Object.fromEntries(
          Object.entries(metricsPayload.windows).map(([key, value]) => [
            key,
            { ...value, instrument_code: symbol },
          ]),
        ),
      },
    });
  });
  await page.route("**/api/v2/advice/*/latest", async (route) => {
    const symbol = route.request().url().includes("/NDX/") ? "NDX" : "399006";
    await route.fulfill({ json: { ...advicePayload, instrument_code: symbol } });
  });
}

test("single data-only workbench exposes the unified eight-ETF panel", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
  await expect(page.getByRole("button", { name: /创业板指数/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /纳指ETF广发/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /广发黄金ETF/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /华宝银行ETF/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /鹏华酒ETF/ })).toBeVisible();
  await expect(page.getByText("科创50")).toHaveCount(0);
  const slotLink = page.getByRole("link", { name: "ETF 替换" });
  await expect(slotLink).toHaveCount(1);
  await expect(slotLink).toHaveAttribute("href", "/etf-slots");
  await expect(page.getByText("日 K + 周 K 共用横轴", { exact: true })).toBeVisible();
  await expect(page.getByText("无 AI · 无推理迭代")).toBeVisible();
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "rendered",
  );
  await expect(page.getByTestId("unified-market-chart").locator("canvas")).toHaveCount(1);
  await expect(page.getByText("数据分析", { exact: true })).toHaveCount(0);
});

test("weekly and monthly ETF charts expose continuous volume and colored momentum labels", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  for (const timeframe of ["weekly", "monthly"]) {
    const response = await page.request.get(`/api/market/159941/prices?timeframe=${timeframe}`);
    await expect(response).toBeOK();
    const payload = await response.json();
    expect(payload.rows.length).toBeGreaterThan(timeframe === "weekly" ? 500 : 100);
    expect(payload.rows.every((row: { volume: string | null }) => row.volume !== null)).toBeTruthy();
  }

  await expect(page.getByTestId("dif-label")).toHaveCSS("color", "rgb(18, 104, 126)");
  await expect(page.getByTestId("dea-label")).toHaveCSS("color", "rgb(216, 139, 44)");
  await expect(page.getByText("MACD柱（正）")).toHaveCSS("color", "rgb(217, 85, 63)");
  await expect(page.getByText("MACD柱（负）")).toHaveCSS("color", "rgb(20, 131, 109)");
  await expect(page.getByTestId("dif-first-change-label")).toHaveCSS("color", "rgb(123, 76, 160)");
  await page.getByTestId("timeframe-monthly").click();
  await expect(page.locator(".monthly-chart-card").getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "rendered",
  );
});

test("market cursor data stays outside the plot and pointer updates are frame-coalesced", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const chart = page.getByTestId("unified-market-chart");
  const canvas = chart.locator(".unified-chart-canvas canvas");
  const details = page.getByTestId("market-point-details");
  await expect(canvas).toBeVisible();
  await expect(details).toBeVisible();

  const canvasBox = await canvas.boundingBox();
  const detailsBox = await details.boundingBox();
  expect(canvasBox).not.toBeNull();
  expect(detailsBox).not.toBeNull();
  expect(canvasBox!.x + canvasBox!.width).toBeLessThanOrEqual(detailsBox!.x);

  const beforeIndex = Number(await details.getAttribute("data-hover-index"));
  await canvas.hover({ position: { x: 110, y: 220 } });
  await expect.poll(async () => Number(await details.getAttribute("data-hover-index"))).not.toBe(beforeIndex);
  await expect(details).toContainText("收盘");
  await expect(page.locator(".echarts-tooltip")).toHaveCount(0);

  const updatesBefore = Number(await details.getAttribute("data-update-count"));
  await canvas.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    for (let index = 0; index < 120; index += 1) {
      element.dispatchEvent(new MouseEvent("mousemove", {
        bubbles: true,
        clientX: bounds.left + 160,
        clientY: bounds.top + 220,
      }));
    }
  });
  await page.waitForTimeout(100);
  const updatesAfter = Number(await details.getAttribute("data-update-count"));
  expect(updatesAfter - updatesBefore).toBeLessThanOrEqual(1);
});

test("legacy training and analysis controls stay absent from the data-only workbench", async ({ page }) => {
  const legacyRequests: string[] = [];
  page.on("request", (request) => {
    if (/\/api\/(v2\/analysis|v33|v341|v343|v35\/analysis|v37\/(analysis|fusion|ai))/.test(request.url())) {
      legacyRequests.push(request.url());
    }
  });
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  await expect(page.getByRole("button", { name: "数据分析" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "训练模型" })).toHaveCount(0);
  await expect(page.getByTestId("model-improvement-chart")).toHaveCount(0);
  expect(legacyRequests).toEqual([]);
});

test("position calendar renders 42 local-date cells and audits percentage-event CRUD", async ({ page }) => {
  await page.clock.install({ time: new Date("2026-08-01T12:00:00+09:00") });
  const positions: Record<string, number | null> = {
    "399006": null,
    "159941": null,
    "518600": 10,
  };
  let nextId = 1;
  let events: Record<string, any>[] = [{
    id: 100,
    instrument_code: "518600",
    direction: "increase",
    operation_date: "2026-08-01",
    change_percent: 10,
    position_after: 10,
    sequence: 1,
    note: "黄金仓位",
    created_at: "2026-07-31T01:00:00Z",
    updated_at: "2026-07-31T01:00:00Z",
  }];

  await page.route("**/api/investment-calendar/current-positions", async (route) => {
    await route.fulfill({ json: positions });
  });
  await page.route("**/api/v2/position-events?*", async (route) => {
    await route.fulfill({ json: events });
  });
  await page.route("**/api/v2/position-events", async (route) => {
    const body = route.request().postDataJSON();
    positions[body.instrument_code] = Number(positions[body.instrument_code] ?? 0) + body.change_percent;
    const event = {
      id: nextId++,
      ...body,
      position_after: positions[body.instrument_code],
      sequence: 1,
      created_at: "2026-07-31T02:00:00Z",
      updated_at: "2026-07-31T02:00:00Z",
    };
    events = [event, ...events];
    await route.fulfill({ json: event });
  });
  await page.route(/\/api\/v2\/position-events\/\d+(\?confirmed=true)?$/, async (route) => {
    const id = Number(route.request().url().match(/position-events\/(\d+)/)?.[1]);
    if (route.request().method() === "PATCH") {
      const body = route.request().postDataJSON();
      const event = events.find((item) => item.id === id)!;
      const oldChange = Number(event.change_percent);
      const symbol = String(event.instrument_code);
      positions[symbol] = Number(positions[symbol]) - oldChange + body.change_percent;
      Object.assign(event, body, { position_after: positions[symbol] });
      await route.fulfill({ json: event });
      return;
    }
    const removed = events.find((item) => item.id === id)!;
    events = events.filter((item) => item.id !== id);
    positions[String(removed.instrument_code)] = null;
    await route.fulfill({ json: { deleted: true, id } });
  });

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await expect(page.locator("[data-calendar-date]")).toHaveCount(42);
  await expect(page.getByTestId("calendar-day-2026-07-26")).toBeVisible();
  await expect(page.getByTestId("calendar-day-2026-09-05")).toBeVisible();
  await expect(page.getByTestId("calendar-day-2026-08-01")).toHaveAttribute("aria-selected", "true");
  await page.getByTestId("position-change").fill("7");
  await page.getByTestId("position-submit").click();
  await expect(page.getByText("仓位变动必须是5到100之间的5倍数。")).toBeVisible();

  await page.getByTestId("position-change").fill("15");
  await page.getByTestId("position-submit").click();
  const row = page.getByTestId("position-event-1");
  await expect(row).toContainText("15%");
  await expect(page.getByTestId("current-position")).toContainText("15%");
  await expect(page.getByTestId("calendar-day-2026-08-01").locator(".increase-marker")).toHaveCount(1);
  await expect(page.getByTestId("selected-date-events")).toContainText("2 笔");
  await expect(page.getByTestId("selected-date-events")).toContainText("纳指ETF广发");
  await expect(page.getByTestId("selected-date-events")).toContainText("广发黄金ETF");
  await expect(page.getByTestId("position-event-100")).toContainText("广发黄金ETF");

  await row.getByRole("button", { name: "修改" }).click();
  await page.getByTestId("position-change").fill("20");
  await page.getByTestId("position-submit").click();
  await expect(page.getByTestId("position-event-1")).toContainText("20%");

  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.getByTestId("current-position")).toContainText("20%");
  await page.getByTestId("position-event-1").getByRole("button", { name: "删除" }).click();
  await page.getByTestId("position-event-1").getByRole("button", { name: "确认删除" }).click();
  await expect(page.getByTestId("position-event-1")).toHaveCount(0);
  await expect(page.getByTestId("current-position")).toContainText("0%");
});

test("selecting a calendar date updates the top holdings to that date and today restores latest", async ({ page }) => {
  await page.clock.install({ time: new Date("2026-08-10T12:00:00+09:00") });
  const positions = {
    "159941": 15,
    "159915": 0,
    "518600": 10,
    "512800": 0,
    "512690": 30,
    "512010": 0,
    "159622": 0,
    "516150": 0,
  };
  const events = [
    {
      id: 4,
      instrument_code: "512690",
      direction: "increase",
      operation_date: "2026-08-08",
      change_percent: 30,
      position_after: 30,
      sequence: 1,
      note: "酒ETF建仓",
    },
    {
      id: 3,
      instrument_code: "159941",
      direction: "decrease",
      operation_date: "2026-08-05",
      change_percent: 5,
      position_after: 15,
      sequence: 1,
      note: "纳指减仓",
    },
    {
      id: 2,
      instrument_code: "518600",
      direction: "increase",
      operation_date: "2026-07-31",
      change_percent: 10,
      position_after: 10,
      sequence: 1,
      note: "黄金建仓",
    },
    {
      id: 1,
      instrument_code: "159941",
      direction: "increase",
      operation_date: "2026-07-15",
      change_percent: 20,
      position_after: 20,
      sequence: 1,
      note: "纳指建仓",
    },
  ];

  await page.route("**/api/investment-calendar/current-positions", (route) =>
    route.fulfill({ json: positions }));
  await page.route("**/api/v2/position-events?*", (route) =>
    route.fulfill({ json: events }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await expect(page.getByTestId("position-snapshot-label")).toHaveText("当前总仓位");
  await expect(page.getByTestId("total-position")).toHaveText("55%");
  await expect(page.getByTestId("position-card-512690")).toContainText("30%");

  await page.getByTestId("calendar-previous-month").click();
  await page.getByTestId("calendar-day-2026-07-31").click();
  await expect(page.getByTestId("position-snapshot-label")).toHaveText(
    "截至 2026-07-31 的总仓位",
  );
  await expect(page.getByTestId("total-position")).toHaveText("30%");
  await expect(page.getByTestId("position-card-159941")).toContainText("20%");
  await expect(page.getByTestId("position-card-518600")).toContainText("10%");
  await expect(page.getByTestId("position-card-512690")).toContainText("0%");

  // A day without an event carries forward the most recent positions without
  // inventing a transaction on that date.
  await page.getByTestId("calendar-day-2026-08-02").click();
  await expect(page.getByTestId("total-position")).toHaveText("30%");
  await expect(page.getByTestId("selected-date-events")).toContainText("0 笔");

  await page.getByTestId("calendar-day-2026-08-05").click();
  await expect(page.getByTestId("total-position")).toHaveText("25%");
  await expect(page.getByTestId("position-card-159941")).toContainText("15%");

  await page.getByRole("button", { name: "回到今天" }).click();
  await expect(page.getByTestId("position-snapshot-label")).toHaveText("当前总仓位");
  await expect(page.getByTestId("total-position")).toHaveText("55%");
  await expect(page.getByTestId("position-card-512690")).toContainText("30%");
});

test("new market selection clears the unified chart while data is pending", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "rendered",
  );

  await page.route("**/api/market/518600/prices?timeframe=weekly", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1_000));
    await route.continue();
  });
  await page.getByRole("button", { name: /广发黄金ETF/ }).click();
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "loading",
    { timeout: 500 },
  );
  await expect(page.getByTestId("data-cutoff")).toHaveText("—", { timeout: 500 });
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "rendered",
  );
});

test("all original display ETFs use the same shared daily-weekly chart and calendar", async ({ page }) => {
  const forbidden: string[] = [];
  const displayCodes = ["518600", "512800", "512690"];
  page.on("request", (request) => {
    if (
      displayCodes.some((code) => request.url().includes(code))
      && (request.url().includes("/api/v341") || request.url().includes("/api/v343"))
    ) forbidden.push(request.url());
  });
  await page.route("**/api/market/*/prices?timeframe=*", (route) => route.fulfill({
    json: { rows: [
      { date: "2026-07-24", open: "6.10", high: "6.22", low: "6.08", close: "6.20", volume: "102000", source: "AKSHARE_ETF", is_complete: true },
      { date: "2026-07-31", open: "6.20", high: "6.35", low: "6.18", close: "6.31", volume: "118000", source: "AKSHARE_ETF", is_complete: true },
    ], current_period_status: "COMPLETE" },
  }));
  await page.route("**/api/indicators/*?timeframe=*", (route) => route.fulfill({
    json: [
      { date: "2026-07-24", dif: "0.01", dea: "0.008", macd_histogram: "0.004", dif_first_change: null },
      { date: "2026-07-31", dif: "0.014", dea: "0.010", macd_histogram: "0.008", dif_first_change: "0.004" },
    ],
  }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  for (const name of [/广发黄金ETF/, /华宝银行ETF/, /鹏华酒ETF/]) {
    await page.getByRole("button", { name }).click();
    await expect(page.getByRole("heading", { name: /日线 \/ 周线指标/ })).toBeVisible();
    await expect(page.getByTestId("v34-train")).toHaveCount(0);
    await expect(page.locator(".position-calendar")).toHaveCount(1);
    await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
    await expect(page.getByTestId("dif-first-change-label")).toBeVisible();
    await page.getByTestId("timeframe-monthly").click();
    await expect(page.locator(".monthly-chart-card").getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
    await page.getByTestId("timeframe-monthly").click();
  }
  expect(forbidden).toEqual([]);
});

test("unfinished weekly overlay remains visible and is clearly separated from the complete cutoff", async ({ page }) => {
  await page.route("**/api/market/159941/prices?timeframe=weekly", (route) => route.fulfill({
    json: {
      current_period_status: "INCOMPLETE_CURRENT_PERIOD",
      rows: [
        { date: "2026-07-31", open: "2900", high: "3000", low: "2850", close: "2980", volume: "100", source: "TEST", is_complete: true, period_status: "COMPLETE" },
        { date: "2026-08-05", open: "2980", high: "3010", low: "2800", close: "2860", volume: "60", source: "PROVISIONAL_DAILY_AGGREGATION", is_complete: false, period_status: "INCOMPLETE_CURRENT_PERIOD" },
      ],
    },
  }));
  await page.route("**/api/indicators/159941?timeframe=weekly", (route) => route.fulfill({
    json: [{ date: "2026-07-31", dif: "1", dea: "0.8", macd_histogram: "0.4", dif_first_change: "0.1" }],
  }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await expect(page.getByTestId("incomplete-period-notice")).toContainText("当前周期尚未完成");
  await expect(page.getByTestId("incomplete-period-notice")).toContainText("2026-08-05");
  await expect(page.getByTestId("incomplete-period-notice")).toContainText("2026-07-31");
  await expect(page.getByTestId("data-cutoff")).toHaveText("2026-08-05");
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
});

test("refresh button pulls latest data from the public source and updates the cutoff", async ({ page }) => {
  let refreshCalled = false;
  await page.route("**/api/market/159941/refresh", async (route) => {
    refreshCalled = true;
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.fulfill({
      json: {
        status: "success",
        refresh_outcome: "UPDATED",
        cutoff_date: "2026-08-05",
        local_cutoff_before: "2026-07-31",
        expected_cutoff: "2026-08-05",
        local_cutoff_after: "2026-08-05",
        source: "AKSHARE_ETF",
        records_added: 3,
        records_updated: 1,
        records_skipped: 7,
        network_requested: true,
        verified_fresh: true,
        records: [{ date: "2026-08-05" }],
        aggregation_status: "success",
      },
    });
  });
  await page.route("**/api/market/159941/prices?timeframe=weekly", (route) => route.fulfill({
    json: {
      current_period_status: "INCOMPLETE_CURRENT_PERIOD",
      rows: [
        { date: "2026-07-31", open: "2900", high: "3000", low: "2850", close: "2980", volume: "100", source: "TEST", is_complete: true, period_status: "COMPLETE" },
        { date: "2026-08-05", open: "2980", high: "3010", low: "2800", close: "2860", volume: "60", source: "PROVISIONAL_DAILY_AGGREGATION", is_complete: false, period_status: "INCOMPLETE_CURRENT_PERIOD" },
      ],
    },
  }));
  await page.route("**/api/indicators/159941?timeframe=weekly", (route) => route.fulfill({
    json: [{ date: "2026-07-31", dif: "1", dea: "0.8", macd_histogram: "0.4", dif_first_change: "0.1" }],
  }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: "刷新行情" }).click();
  await expect(page.getByRole("button", { name: "正在下载…" })).toBeDisabled();
  await expect(page.locator(".flash")).toContainText("2026-08-05");
  await expect(page.getByTestId("data-cutoff")).toHaveText("2026-08-05");
  const audit = page.getByTestId("refresh-audit");
  await expect(audit).toContainText("本地原截止2026-07-31");
  await expect(audit).toContainText("应有交易日2026-08-05");
  await expect(audit).toContainText("新增 3 · 修订 1 · 跳过 7");
  await expect(audit.locator(".verified")).toHaveText("通过");
  expect(refreshCalled).toBe(true);
});

test("159941 stays an active ETF and never aliases its position or analysis to V3.2 NDX", async ({ page }) => {
  const invalidV32Requests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/v3.2/") && request.url().includes("159941")) {
      invalidV32Requests.push(request.url());
    }
  });
  await page.route("**/api/investment-calendar/current-positions", (route) =>
    route.fulfill({ json: { "399006": 35, "159941": 10, NDX: 85 } }));
  await page.route("**/api/v2/position-events?*", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/market/159941/prices?timeframe=weekly", (route) => route.fulfill({
    json: {
      rows: [
        { date: "2026-07-24", open: "1.20", high: "1.24", low: "1.19", close: "1.23", volume: "102000", source: "SZSE" },
        { date: "2026-07-31", open: "1.23", high: "1.28", low: "1.22", close: "1.27", volume: "118000", source: "SZSE" },
      ],
    },
  }));
  await page.route("**/api/indicators/159941?timeframe=weekly", (route) => route.fulfill({
    json: [
      { date: "2026-07-24", dif: "0.01", dea: "0.008", macd_histogram: "0.004" },
      { date: "2026-07-31", dif: "0.014", dea: "0.010", macd_histogram: "0.008" },
    ],
  }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: /纳指ETF广发/ }).click();
  await expect(page.getByRole("heading", { name: /纳指ETF广发 · 日 K \+ 周 K 共享横轴/ })).toBeVisible();
  await expect(page.getByTestId("current-position")).toContainText("10%");
  await expect(page.getByTestId("v32-readonly-badge")).toHaveCount(0);
  expect(invalidV32Requests).toEqual([]);
});

test("failed daily request for a new ETF leaves the unified chart in an error state", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.route("**/api/market/518600/prices?timeframe=daily", async (route) => {
    await route.fulfill({ status: 500, json: { detail: "controlled chart load failure" } });
  });
  await page.getByRole("button", { name: /广发黄金ETF/ }).click();

  await expect(page.getByText("controlled chart load failure").first()).toBeVisible();
  await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
    "data-chart-state",
    "error",
  );
  await expect(page.getByTestId("unified-market-chart").locator("canvas")).toHaveCount(1);
});

test("legacy paths redirect to the single workbench and favicon remains local", async ({ page }) => {
  await page.goto("/simulation");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "ETF 行情面板" })).toBeVisible();
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
    "href",
    /investment-lab-avatar\.svg$/,
  );
});
