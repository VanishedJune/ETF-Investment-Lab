import assert from "node:assert/strict";
import test from "node:test";

import {
  legendColors,
  normalizeSeries,
  volumeChangeBarData,
} from "../src/types/research.ts";

test("normalizes market and indicator rows onto one continuous date axis", () => {
  const prices = [
    { date: "2026-07-17", open: "9", high: "10", low: "8", close: "9.5", volume: "80", source: "DIRECT" },
    { date: "2026-07-24", open: "10", high: "12", low: "9", close: "11", volume: "100", source: "DIRECT" },
    { date: "2026-07-31", open: "11", high: "13", low: "10", close: "12", volume: "120", source: "DIRECT" },
  ];
  const indicators = [
    { date: "2026-07-24", dif: "0.2", dea: "0.1", macd_histogram: "0.2", dif_first_change: null },
    { date: "2026-07-31", dif: "0.3", dea: "0.15", macd_histogram: "0.3", dif_first_change: "0.1" },
  ];

  const normalized = normalizeSeries(prices, indicators);

  assert.deepEqual(normalized.dates, ["2026-07-17", "2026-07-24", "2026-07-31"]);
  assert.deepEqual(normalized.candles, [[9, 9.5, 8, 10], [10, 11, 9, 12], [11, 12, 10, 13]]);
  assert.deepEqual(normalized.volumes, [80, 100, 120]);
  assert.equal(Number.isNaN(normalized.dif[0]), true);
  assert.equal(Number.isNaN(normalized.difFirstChange[0]), true);
  assert.equal(Number.isNaN(normalized.difFirstChange[1]), true);
  assert.equal(normalized.difFirstChange[2], 0.1);
  assert.equal(normalized.volumeAvailable, true);
  assert.ok(normalized.ma20.every(Number.isFinite));
});

test("uses the specified curve and histogram legend colors", () => {
  assert.equal(legendColors.priceUp, "#d64b4b");
  assert.equal(legendColors.priceDown, "#27845a");
  assert.equal(legendColors.dif, "#52789c");
  assert.equal(legendColors.dea, "#a57738");
  assert.equal(legendColors.macdPositive, "#d64b4b");
  assert.equal(legendColors.macdNegative, "#27845a");
  assert.equal(legendColors.difFirstChange, "#807096");
  assert.equal(legendColors.volumeIncrease, "#d64b4b");
  assert.equal(legendColors.volumeDecrease, "#27845a");
  assert.equal(legendColors.ma5, "#ae794b");
  assert.equal(legendColors.ma10, "#52789c");
  assert.equal(legendColors.ma20, "#97718f");
});

test("colors volume increases red and decreases green across sparse gaps", () => {
  const bars = volumeChangeBarData([100, 120, 90, 90, null, 110], 0.8);

  assert.equal(bars[0].itemStyle.color, legendColors.volume);
  assert.equal(bars[1].itemStyle.color, legendColors.volumeIncrease);
  assert.equal(bars[2].itemStyle.color, legendColors.volumeDecrease);
  assert.equal(bars[3].itemStyle.color, legendColors.volume);
  assert.equal(bars[4], null);
  assert.equal(bars[5].itemStyle.color, legendColors.volumeIncrease);
  assert.equal(bars[5].itemStyle.opacity, 0.8);
});
