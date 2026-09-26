import { test, expect } from '@playwright/test';

test('ordered catalog, daily/weekly return and replacement-aware calendar', async ({ page }) => {
  let coal = '515220';
  let close: string | null = '11';
  let histogram: string | null = '0.2';
  const codes = ['159915','517520','516150','159622','159611','515220','512800','512690'];
  const names = ['创业板ETF易方达','黄金股ETF永赢','稀土ETF嘉实','创新药ETF东财','电力ETF广发','煤炭ETF国泰','华宝银行ETF','鹏华酒ETF'];
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    let body: unknown = {};
    if (url.pathname === '/api/instruments') body = [
      ...codes.map((code, i) => ({code: code === '515220' ? coal : code, name: names[i], exchange: 'SSE', slot_order:i+1,is_current_slot:true})),
      {code:'512010', name:'医药ETF易方达',exchange:'SSE',is_current_slot:false},
    ];
    else if (url.pathname.includes('/prices')) body = { rows: [
      {date:'2026-09-18',open:'10',high:'10',low:'10',close:'10',volume:'100',source:'TEST'},
      {date:'2026-09-24',open:'10',high:'11',low:'9',close,volume:'110',source:'TEST',is_complete:url.searchParams.get('timeframe') !== 'weekly'},
    ], current_period_status: url.searchParams.get('timeframe') === 'weekly' ? 'INCOMPLETE_CURRENT_PERIOD' : 'COMPLETE' };
    else if (url.pathname.includes('/indicators/')) body = [{date:'2026-09-24',dif:null,dea:'0.1',macd_histogram:histogram,dif_first_change:'0.01'}];
    else if (url.pathname.includes('/current-positions')) body = Object.fromEntries([...codes.map(code => [code === '515220' ? coal : code, code === '159915' ? 50 : 0]), ['512010',20]]);
    else if (url.pathname.includes('/position-events')) body = [{id:1,instrument_code:'512010',direction:'increase',operation_date:'2026-09-01',change_percent:20,position_after:20,note:null}];
    await route.fulfill({json:body});
  });
  await page.goto('/');
  const buttons = page.locator('.market-switch > button');
  await expect(buttons).toHaveCount(8);
  for (let i=0;i<8;i++) await expect(buttons.nth(i)).toContainText(names[i]!);
  await expect(page.getByTestId('daily-return')).toContainText('+10.00%');
  await expect(page.getByTestId('weekly-return')).toContainText('+10.00%');
  await expect(page.getByTestId('daily-return').locator('b')).toHaveCSS('color', 'rgb(214, 75, 75)');
  await expect(page.locator('.market-switch > button.active')).toHaveCSS('border-top-color', 'rgb(102, 132, 167)');
  await expect(page.getByTestId('position-submit')).toHaveCSS('background-color', 'rgb(71, 107, 150)');
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(233, 237, 242)');
  await expect(page.locator('.indicator-snapshot-card').first()).toHaveCSS('background-color', 'rgb(244, 246, 248)');
  await expect(page.getByTestId('calendar-instrument')).toHaveCSS('background-color', 'rgb(250, 251, 252)');
  await expect(page.locator('.calendar-day.selected .day-number')).toHaveCSS('background-color', 'rgb(226, 235, 245)');
  await expect(page.getByTestId('weekly-return')).toContainText('本周截至');
  await expect(page.getByTestId('total-position')).toHaveText('70%');
  await expect(page.locator('.position-totals > button')).toHaveCount(8);
  await expect(page.getByTestId('position-card-512010')).toHaveCount(0);
  await expect(page.getByTestId('calendar-instrument').locator('optgroup[label="历史标的（原记录保留）"] option[value="512010"]')).toHaveCount(1);
  await expect(page.locator('.dif-value').first()).toHaveText('—');
  await page.getByTestId('calendar-instrument').selectOption('515220');
  await expect(page.getByTestId('calendar-instrument')).toHaveValue('515220');
  await page.screenshot({path:'../reports/soft-colors-overview-preview.png'});
  await page.locator('.indicator-snapshot-panel').screenshot({path:'../reports/soft-colors-indicators-preview.png'});
  await page.locator('.position-calendar').screenshot({path:'../reports/soft-colors-calendar-preview.png'});
  coal = '515790';
  await page.evaluate(() => window.dispatchEvent(new Event('etf-universe-changed')));
  await expect(page.getByTestId('position-card-515790')).toBeVisible();
  await expect(page.locator('.position-totals > button')).toHaveCount(8);
  await expect(page.getByTestId('calendar-instrument').locator('option[value="515220"]')).toHaveCount(0);
  await page.getByTestId('calendar-instrument').selectOption('515790');
  await expect(page.getByTestId('total-position')).toHaveText('70%');
  await page.locator('.indicator-snapshot-panel').screenshot({path:'../reports/catalog-indicators-preview.png'});
  await page.locator('.position-calendar').screenshot({path:'../reports/catalog-calendar-preview.png'});
  await page.setViewportSize({width:1050,height:800});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  for (const scenario of [
    { close: '9', histogram: '-0.2', label: '-10.00%', color: 'rgb(39, 132, 90)' },
    { close: '10', histogram: '0', label: '0.00%', color: 'rgb(100, 116, 139)' },
    { close: null, histogram: null, label: '—', color: 'rgb(100, 116, 139)' },
  ]) {
    close = scenario.close;
    histogram = scenario.histogram;
    await page.reload();
    for (const period of ['daily', 'weekly']) {
      await expect(page.getByTestId(`${period}-return`).locator('b')).toHaveText(scenario.label);
      await expect(page.getByTestId(`${period}-return`).locator('b')).toHaveCSS('color', scenario.color);
    }
    await expect(page.locator('.indicator-snapshot-card dd').nth(2)).toHaveCSS('color', scenario.color);
  }
  expect(errors).toEqual([]);
});
