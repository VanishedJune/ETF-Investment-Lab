import { expect, test, type Page } from "@playwright/test";

const SLOT_01 = {
  slot_id: "ETF_SLOT_01",
  slot_order: 1,
  instrument_code: "159941",
  exchange: "SZSE",
  instrument_name: "纳指ETF广发",
  instrument_type: "ETF",
  model_enabled: true,
  active: true,
  binding_version: "B1",
  bound_at: "2026-08-06T00:00:00Z",
  replaced_from_code: null,
  replacement_status: "READY",
  data_start_date: "2015-07-13",
  data_end_date: "2026-08-05",
  history_week_count: 565,
  mature_8w_count: 514,
  model_status: "MODEL_READY",
  champion_package_id: "v351:159941:B1:INITIAL",
  last_market_update: "2026-08-06T00:00:00Z",
  model_state: {
    model_namespace: "v351:159941:B1",
    model_status: "MODEL_READY",
    small_sample_champion: false,
    prewarming: false,
    mature_8w_count: 514,
    bootstrap_state: "COMPLETED",
  },
};

function slot(id: string, order: number, code: string, name: string): typeof SLOT_01 {
  return {
    ...SLOT_01,
    slot_id: id,
    slot_order: order,
    instrument_code: code,
    instrument_name: name,
    model_state: { ...SLOT_01.model_state, model_namespace: `v351:${code}:B1` },
  };
}

async function mockV351Apis(page: Page) {
  await page.route("**/api/v351/instrument-slots", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        slot("ETF_SLOT_01", 1, "159941", "纳指ETF广发"),
        slot("ETF_SLOT_02", 2, "518600", "广发黄金ETF"),
        slot("ETF_SLOT_03", 3, "512800", "华宝银行ETF"),
        slot("ETF_SLOT_04", 4, "512690", "鹏华酒ETF"),
        slot("ETF_SLOT_05", 5, "512010", "医药ETF易方达"),
      ]),
    }),
  );
  await page.route("**/api/v351/challenges/equivalence-summary", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        total_candidates: 120,
        effectively_identical: 8,
        valid_candidates: 112,
        by_market: { "399006": { total: 40, identical: 2, valid: 38 } },
      }),
    }),
  );
  await page.route("**/api/v351/challenges/rejection-summary", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        reasons: { EXCESS_RETURN_GAP: 12, WIN_RATE_GAP: 5 },
        by_market: { "399006": { EXCESS_RETURN_GAP: 12 } },
      }),
    }),
  );
  await page.route(
    "**/api/v351/instrument-slots/ETF_SLOT_01/validate-replacement",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          slot_id: "ETF_SLOT_01",
          target_code: "512100",
          metadata: {
            instrument_code: "512100",
            official_name: "测试医药ETF",
            exchange: "SSE",
            instrument_type: "ETF",
            data_source: "AKSHARE/TENCENT_LIVE",
          },
          replaced_from_code: "159941",
          replaced_from_name: "纳指ETF广发",
          preview: {
            earliest_data_date: "2015-07-13",
            daily_count: 3000,
            weekly_count: 600,
            estimated_mature_8w: 590,
            can_initial_champion: true,
            requires_prewarming: false,
          },
          confirm_required: true,
        }),
      }),
  );
  await page.route("**/api/v351/instrument-slots/ETF_SLOT_01/replace", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "V351REPL:ETF_SLOT_01:test:00000001",
        state: "VALIDATING",
        target_code: "512100",
      }),
    }),
  );
  await page.route(
    "**/api/v351/instrument-slots/ETF_SLOT_01/replacement-status",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: "V351REPL:ETF_SLOT_01:test:00000001",
          slot_id: "ETF_SLOT_01",
          target_code: "512100",
          state: "SWITCHED",
          current_step: "SWITCHED",
          error_code: null,
          error_message: null,
          finished_at: "2026-08-06T00:00:00Z",
        }),
      }),
  );
  await page.route("**/api/indicators/512100/calculate?timeframe=daily", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "success",
        instrument_code: "512100",
        timeframe: "daily",
        price_rows: 3000,
        records_added: 3000,
        records_updated: 0,
        records_skipped: 0,
        error: null,
      }),
    }),
  );
}

test("ETF 替换页按接口动态渲染五个纯行情槽位", async ({ page }) => {
  await mockV351Apis(page);
  await page.goto("/etf-slots");
  await expect(page.locator('[data-testid="etf-slot-manager"]')).toBeVisible();
  for (const id of ["etf_slot_01", "etf_slot_02", "etf_slot_03", "etf_slot_04", "etf_slot_05"]) {
    await expect(page.locator(`[data-testid="slot-${id}"]`)).toBeVisible();
  }
  await expect(page.getByText("替换仅更新本地行情、日/周/月线与技术指标")).toBeVisible();
  await expect(page.locator('[data-testid="challenger-stats"]')).toHaveCount(0);
});

test("五位 ETF 代码无法发起替换验证", async ({ page }) => {
  await mockV351Apis(page);
  await page.goto("/etf-slots");
  await expect(page.locator('[data-testid="etf-slot-manager"]')).toBeVisible();
  await page.locator('[data-testid="slot-etf_slot_01"] .replace-button').click();
  await expect(page.locator('[data-testid="replace-panel"]')).toBeVisible();
  const input = page.locator('[data-testid="etf-code-input"]');
  await input.fill("51200");
  await expect(page.locator('[data-testid="replace-panel"] button')).toBeDisabled();
  await expect(page.locator('[data-testid="replacement-preview"]')).not.toBeVisible();
});

test("替换流程：预览、二次确认后调用替换接口并刷新槽位", async ({ page }) => {
  await mockV351Apis(page);
  let replaceCalled = false;
  let dailyIndicatorsCalculated = false;
  let replaceRequestBody: Record<string, unknown> | null = null;
  await page.route("**/api/v351/instrument-slots/ETF_SLOT_01/replace", async (route) => {
    replaceCalled = true;
    replaceRequestBody = route.request().postDataJSON() as Record<string, unknown>;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "V351REPL:ETF_SLOT_01:test:00000002",
        state: "VALIDATING",
        target_code: "512100",
      }),
    });
  });
  await page.route("**/api/indicators/512100/calculate?timeframe=daily", (route) => {
    dailyIndicatorsCalculated = true;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "success",
        instrument_code: "512100",
        timeframe: "daily",
        price_rows: 3000,
        records_added: 3000,
        records_updated: 0,
        records_skipped: 0,
        error: null,
      }),
    });
  });
  page.on("dialog", (dialog) => dialog.accept());
  await page.goto("/etf-slots");
  await expect(page.locator('[data-testid="etf-slot-manager"]')).toBeVisible();
  await page.locator('[data-testid="slot-etf_slot_01"] .replace-button').click();
  await page.locator('[data-testid="etf-code-input"]').fill("512100");
  await page.locator('[data-testid="replace-panel"] button').click();
  await expect(page.locator('[data-testid="replacement-preview"]')).toBeVisible();
  await expect(page.locator('[data-testid="replacement-preview"]')).toContainText("512100");
  await page.locator('[data-testid="replacement-preview"] .confirm').click();
  await expect(page.locator(".result")).toContainText("日K指标已更新");
  expect(replaceCalled).toBe(true);
  expect(dailyIndicatorsCalculated).toBe(true);
  expect(replaceRequestBody).toEqual({
    instrument_code: "512100",
    protocol_version: "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS",
  });
  await expect(page.locator('[data-testid="slot-etf_slot_01"]')).toBeVisible();
});
