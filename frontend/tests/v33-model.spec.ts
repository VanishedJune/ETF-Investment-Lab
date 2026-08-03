import { expect, test } from "@playwright/test";

const statuses = ["399006", "159941"].map((market, index) => ({
  market,
  version: "V3.4.1_13W",
  horizon_weeks: 13,
  bootstrapped: true,
  champion: {
    market,
    version: index ? "GFNDXETF_HYBRID_13W_V3.4.8" : "CYB_HYBRID_13W_V3.4.11",
    parent_version: null,
    horizon_weeks: 13,
    trained_through_date: "2026-07-24",
    effective_from_date: "2026-07-31",
    raw_matured_sample_count: 488,
    effective_independent_sample_count: 84.2,
    parameter_hash: `${market}-parameter-hash`,
  },
  last_anchor_date: "2026-07-31",
  weekly_iteration_count: index ? 512 : 511,
  candidate_training_count: 122,
  champion_promotion_count: 18,
}));

function curve(market: "399006" | "159941") {
  return {
    market,
    horizon_weeks: 13,
    points: Array.from({ length: 80 }, (_, index) => ({
      iteration: index + 1,
      anchor_date: `2025-${String((index % 12) + 1).padStart(2, "0")}-25`,
      forecast_anchor_date: `2025-${String((index % 12) + 1).padStart(2, "0")}-25`,
      evaluation_available_date: index < 13 ? null : "2026-01-30",
      maturity_status: index < 13 ? "IMMATURE" : "FULLY_MATURE_13W",
      pending: index < 13,
      price_turn_deviation_trading_days: index < 13 ? null : (index % 8),
      dif_turn_deviation_trading_days: index < 13 ? null : (index % 6),
      normalized_endpoint_error: index < 13 ? null : .08 + index / 1000,
      training_triggered: index % 4 === 3,
      promoted: index === 39,
      champion_before: "champion",
      candidate: index % 4 === 3 ? `candidate-${index}` : null,
      champion_after: index === 39 ? "candidate-39" : "champion",
      champion_loss: index % 4 === 3 ? [0.12, 0.11, 0.13] : [],
      candidate_loss: index % 4 === 3 ? [0.11, 0.10, 0.12] : [],
    })),
  };
}

const historical = Array.from({ length: 26 }, (_, index) => {
  const close = index < 2 ? 160 - index * 20 : 100 + index * .4;
  return {
    week_end: `2026-${String(Math.floor(index / 4) + 1).padStart(2, "0")}-${String((index % 4) * 7 + 1).padStart(2, "0")}`,
    open: close - 1,
    high: close + 2,
    low: close - 2,
    close,
    volume: 1_000_000 + index * 20_000,
  };
});

const predicted = Array.from({ length: 13 }, (_, index) => {
  const close = 110 + Math.sin(index / 2) * 4 + index * .5;
  return {
    week: index + 1,
    week_start: `2026-08-${String(index + 1).padStart(2, "0")}`,
    week_end: `2026-08-${String(index + 7).padStart(2, "0")}`,
    open: close - .7,
    high: close + 1.8,
    low: close - 1.6,
    close,
    volume_p10: 900_000 + index * 10_000,
    volume_p50: 1_100_000 + index * 15_000,
    volume_p90: 1_400_000 + index * 20_000,
  };
});

const forecast = {
  protocol_version: "V3.4.1_MODEL_CORE",
  display_version: "V3.4.3_MARKET_UI",
  version: "V3.4_13W",
  forecast_horizon_weeks: 13,
  market: "399006",
  forecast_anchor_date: "2026-07-31",
  model_version: "CYB_HYBRID_13W_V3.4.11",
  scenario_adapter_id: "V3.4_13W_SCENARIO_1",
  scenario_seed: 340013,
  scenario_count: 1200,
  historical_ohlcv: historical,
  historical_indicators: historical.map((row, index) => ({
    week: index - historical.length,
    week_end: row.week_end,
    dif: -.8 + index * .02,
    dea: -.75 + index * .018,
    macd: -.1 + index * .004,
    dif_first_change: index ? .02 : null,
    dif_second_change: index > 1 ? 0 : null,
    trend_state: "HISTORY",
  })),
  representative_ohlcv: predicted,
  indicators: predicted.map((row, index) => ({
    week: row.week,
    week_end: row.week_end,
    dif: -.5 + index * .09,
    dea: -.4 + index * .065,
    macd: -.2 + index * .05,
    dif_first_change: index ? .09 : null,
    dif_second_change: index > 1 ? 0 : null,
    trend_state: index < 2 ? "UPWARD_TURN" : index < 5 ? "UPTREND_EARLY_CONFIRMED" : "UPTREND_CONFIRMED",
  })),
  price_quantiles: predicted.map((row) => ({
    week: row.week,
    week_end: row.week_end,
    close_p10: row.close - 7,
    close_p50: row.close,
    close_p90: row.close + 8,
    low_quantile: row.low - 4,
    high_quantile: row.high + 4,
  })),
  direction_probabilities: { up: 58.2, sideways: 23.5, down: 18.3 },
  horizon_probabilities: {
    "4": { up: .42, sideways: .38, down: .20 },
    "8": { up: .51, sideways: .30, down: .19 },
    "13": {
      raw: { up: .60, sideways: .22, down: .18 },
      calibrated: { up: .582, sideways: .235, down: .183 },
    },
  },
  path_probabilities: { TREND_UP: 38, TREND_DOWN: 12, V_SHAPE: 28, INVERTED_V: 10, RANGE: 12 },
  confidence: 74.5,
  model_reliability: { score: 74.5, semantics: "model_reliability_not_prediction_probability" },
  health_status: "MODEL_NORMAL",
  calibration: {
    status: "FORMALLY_CALIBRATED",
    raw_matured_sample_count: 488,
    effective_independent_sample_count: 84.2,
  },
  turning_windows: {
    price_turning_window: { start_week: 3, end_week: 5, coverage_probability: 64 },
    dif_derivative_zero_window: { start_week: 2, end_week: 4, coverage_probability: 61 },
    price_turn_kind: "bottom",
    dif_turn_kind: "bottom",
  },
  consistency: {
    status: "CONSISTENT",
    separation_weeks: 0,
    reasons: [],
    single_day_turn_suppressed: false,
    extreme_position_change_allowed: true,
  },
  scenario_audit: { ohlc_legal_rate: 1, ema_initial_state_source: "pre_forecast_real_weekly_close_history" },
  policy: {
    turning_assessment: {
      price_turn_status: "CONFIRMED",
      dif_turn_status: "CONFIRMED",
      consistency_status: "TEMPORALLY_CONSISTENT",
      candidates: [
        { signal_kind: "PRICE", turn_kind: "bottom", classification: "VALID_TURN", confirmation_status: "CONFIRMED", window_start_date: "2026-08-14", window_end_date: "2026-08-28" },
        { signal_kind: "DIF", turn_kind: "bottom", classification: "VALID_TURN", confirmation_status: "CONFIRMED", window_start_date: "2026-08-07", window_end_date: "2026-08-21" },
      ],
    },
    position_source: { position_percent: 0, source_kind: "LOCAL_CALENDAR" },
    decision: {
      current_position: 0,
      target_position: 55,
      next_executable_position: 25,
      total_change: 55,
      action: "BUY",
      fund_etf_ratio: "6:4",
      batches: [25, 15, 10, 5].map((amount, index) => ({
        batch_number: index + 1,
        action: "BUY",
        change_pp: amount,
        target_after_pp: [25, 40, 50, 55][index],
        window_start_date: `2026-08-${String(14 + index * 7).padStart(2, "0")}`,
        window_end_date: `2026-08-${String(20 + index * 7).padStart(2, "0")}`,
        initial_state: index ? "WAITING_DEPENDENCY" : "WAITING_CONFIRMATION",
        trigger: {},
        invalidation: {},
      })),
    },
  },
  chart_semantics: { dif_first_change: "DIF_t - DIF_(t-1)", frozen_forecast_unchanged: true },
  advice: {
    current_position: 0,
    target_position: 55,
    total_change_percentage_points: 55,
    action: "increase",
    batches: [25, 15, 10, 5].map((amount, index) => ({
      batch: index + 1,
      action: "increase",
      position_change_percentage_points: amount,
      window_start: `2026-08-${String(14 + index * 7).padStart(2, "0")}`,
      window_end: `2026-08-${String(20 + index * 7).padStart(2, "0")}`,
      trigger: "价格进入预测拐点窗口并获得DIF/DEA确认",
      invalidation: "价格与MACD结构反向冲突",
    })),
    fund_etf_ratio: "6:4",
    confidence: 74.5,
    forecast_consistency_status: "CONSISTENT",
    disclaimer: "概率情景研究结果，不构成保证收益或确定日期承诺。",
  },
  semantics: {
    p10_p50_p90: "每周收盘价边际分位数，不是置信度，也不是累计收益",
    p50_candles: "P50代表性情景周K，不是唯一确定路径",
  },
};

test("V3.4 separates training and analysis and renders 13-week scenario K-lines", async ({ page }) => {
  let analysisCalls = 0;
  let trainingCalls = 0;
  let forecastCreated = false;
  await page.route("**/api/v341/model/status", (route) => route.fulfill({ json: statuses }));
  await page.route("**/api/v343/iterations/399006", (route) => route.fulfill({ json: curve("399006") }));
  await page.route("**/api/v343/iterations/159941", (route) => route.fulfill({ json: curve("159941") }));
  await page.route("**/api/v343/forecast/399006", (route) => forecastCreated
    ? route.fulfill({ json: forecast })
    : route.fulfill({ status: 404, json: { detail: "not found" } }));
  await page.route("**/api/v343/forecast/159941", (route) => route.fulfill({ json: { ...forecast, market: "159941" } }));
  await page.route("**/api/v343/model/analysis", (route) => {
    analysisCalls += 1;
    forecastCreated = true;
    return route.fulfill({ status: 202, json: {
      run_id: "V341-ANALYSIS-TEST",
      status: "queued",
      market: "399006",
    } });
  });
  await page.route("**/api/v343/model/analysis/runs/V341-ANALYSIS-TEST", (route) =>
    route.fulfill({ json: {
      run_id: "V341-ANALYSIS-TEST",
      protocol_version: "V3.4.1_MODEL_CORE",
      market: "399006",
      status: "completed",
      forecast_anchor_date: "2026-07-31",
      started_at: "2026-08-03T00:00:00Z",
      completed_at: "2026-08-03T00:00:01Z",
      result: forecast,
      error_code: null,
      error_message: null,
    } }),
  );
  await page.route("**/api/v341/model/train", (route) => {
    trainingCalls += 1;
    return route.fulfill({ status: 202, json: { run_id: "V34-TEST-RUN", status: "queued", market: "399006" } });
  });
  await page.route("**/api/v341/training/runs/V34-TEST-RUN", (route) => route.fulfill({ json: {
    run_id: "V34-TEST-RUN", market: "399006", status: "completed", current_stage: "completed",
    weekly_iteration_count: 1, candidate_training_count: 0, champion_promotion_count: 0,
  } }));
  await page.route("**/api/investment-calendar/current-positions", (route) => route.fulfill({ json: { "399006": 0, "159941": 0 } }));
  await page.route("**/api/v2/position-events?*", (route) => route.fulfill({ json: [] }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await expect(page.getByTestId("v34-curve-399006").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("v34-curve-159941").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("v34-train")).toHaveText("训练模型");
  await expect(page.getByTestId("v34-analyze")).toHaveText("数据分析");

  await page.getByTestId("v34-analyze").click();
  await expect(page.getByTestId("v34-analysis-result")).toContainText("58.2%");
  await expect(page.getByTestId("v34-analysis-result")).toContainText("75/100");
  await expect(page.getByTestId("v34-analysis-result")).toContainText("0%");
  await expect(page.getByTestId("v34-analysis-result")).toContainText("55%");
  await expect(page.getByTestId("v34-analysis-result")).toContainText("6:4");
  await expect(page.getByText(/P10\/P50\/P90是每周收盘价分布的分位数/)).toBeVisible();
  await expect(page.getByTestId("v34-forecast-chart").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("v34-forecast-chart")).toHaveAttribute("data-price-y-min", /.+/);
  await expect(page.getByTestId("v34-forecast-chart")).toHaveAttribute("data-volume-y-min", "0");
  const macdMinimum = Number(await page.getByTestId("v34-forecast-chart").getAttribute("data-macd-y-min"));
  const macdMaximum = Number(await page.getByTestId("v34-forecast-chart").getAttribute("data-macd-y-max"));
  expect(macdMinimum).toBeLessThan(0);
  expect(macdMaximum).toBeGreaterThan(0);
  await expect(page.getByTestId("v34-forecast-chart")).toHaveAttribute("data-dif-first-change-y-min", /.+/);
  const forecastDetails = page.getByTestId("forecast-point-details");
  await expect(forecastDetails).toBeVisible();
  await expect(forecastDetails).toContainText("预测第1周");
  await expect(page.locator(".batch-card li")).toHaveCount(4);
  await expect(page.locator(".batch-card")).toContainText("增加 25 个百分点");
  expect(analysisCalls).toBe(1);
  expect(trainingCalls).toBe(0);

  const chart = page.getByTestId("v34-forecast-chart");
  const forecastCanvas = chart.locator("canvas");
  const forecastCanvasBox = await forecastCanvas.boundingBox();
  const forecastDetailsBox = await forecastDetails.boundingBox();
  expect(forecastCanvasBox).not.toBeNull();
  expect(forecastDetailsBox).not.toBeNull();
  expect(forecastDetailsBox!.y + forecastDetailsBox!.height).toBeLessThanOrEqual(forecastCanvasBox!.y);
  await forecastCanvas.hover({ position: { x: 100, y: 220 } });
  await expect(forecastDetails).toContainText("历史周K");
  const forecastUpdatesBefore = Number(await forecastDetails.getAttribute("data-update-count"));
  await forecastCanvas.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    for (let index = 0; index < 120; index += 1) {
      element.dispatchEvent(new MouseEvent("mousemove", {
        bubbles: true,
        clientX: bounds.left + 120,
        clientY: bounds.top + 220,
      }));
    }
  });
  await page.waitForTimeout(100);
  const forecastUpdatesAfter = Number(await forecastDetails.getAttribute("data-update-count"));
  expect(forecastUpdatesAfter - forecastUpdatesBefore).toBeLessThanOrEqual(1);
  await expect(page.locator(".echarts-tooltip")).toHaveCount(0);
  const startBefore = Number(await chart.getAttribute("data-visible-start-index"));
  const endBefore = Number(await chart.getAttribute("data-visible-end-index"));
  const yBefore = {
    priceMin: Number(await chart.getAttribute("data-price-y-min")),
    priceMax: Number(await chart.getAttribute("data-price-y-max")),
    volumeMax: Number(await chart.getAttribute("data-volume-y-max")),
    macdMin: Number(await chart.getAttribute("data-macd-y-min")),
    macdMax: Number(await chart.getAttribute("data-macd-y-max")),
    derivativeMin: Number(await chart.getAttribute("data-dif-first-change-y-min")),
    derivativeMax: Number(await chart.getAttribute("data-dif-first-change-y-max")),
  };
  // Target the actual ECharts canvas. Hovering the wrapper can land on the
  // surrounding report content after scroll-into-view and never reach zrender.
  await forecastCanvas.hover({ position: { x: 520, y: 240 } });
  await page.mouse.wheel(0, -900);
  await expect.poll(async () => {
    const startAfter = Number(await chart.getAttribute("data-visible-start-index"));
    const endAfter = Number(await chart.getAttribute("data-visible-end-index"));
    return endAfter - startAfter;
  }).toBeLessThan(endBefore - startBefore);
  await expect.poll(async () => ({
    priceMin: Number(await chart.getAttribute("data-price-y-min")),
    priceMax: Number(await chart.getAttribute("data-price-y-max")),
    volumeMax: Number(await chart.getAttribute("data-volume-y-max")),
    macdMin: Number(await chart.getAttribute("data-macd-y-min")),
    macdMax: Number(await chart.getAttribute("data-macd-y-max")),
    derivativeMin: Number(await chart.getAttribute("data-dif-first-change-y-min")),
    derivativeMax: Number(await chart.getAttribute("data-dif-first-change-y-max")),
  })).not.toEqual(yBefore);

  await page.getByTestId("v34-train").click();
  await expect.poll(() => trainingCalls).toBe(1);
  expect(analysisCalls).toBe(1);
  await page.screenshot({ path: "../reports/v34-13w-workbench.png", fullPage: true });
});

test("main price, volume, MACD and DIF-change axes rescale independently across periods", async ({ page }) => {
  await page.route("**/api/v341/model/status", (route) => route.fulfill({ json: statuses }));
  await page.route("**/api/v343/iterations/*", (route) => route.fulfill({ json: curve(route.request().url().includes("159941") ? "159941" : "399006") }));
  await page.route("**/api/v343/forecast/*", (route) => route.fulfill({ status: 404, json: { detail: "not found" } }));
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const main = page.getByTestId("unified-market-chart").locator(".unified-chart-canvas");
  for (const timeframe of ["daily", "weekly", "monthly"] as const) {
    await page.getByTestId(`timeframe-${timeframe}`).click();
    await expect(page.getByTestId("unified-market-chart")).toHaveAttribute(
      "data-chart-state",
      "rendered",
      { timeout: 20_000 },
    );
    await expect(main).toHaveAttribute("data-price-y-min", /.+/);
    await expect(main).toHaveAttribute("data-volume-y-min", "0");
    const min = Number(await main.getAttribute("data-macd-y-min"));
    const max = Number(await main.getAttribute("data-macd-y-max"));
    expect(min).toBeLessThan(0);
    expect(max).toBeGreaterThan(0);
    await expect(main).toHaveAttribute("data-dif-first-change-y-min", /.+/);
  }
});
