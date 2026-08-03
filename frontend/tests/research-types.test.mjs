import assert from "node:assert/strict";
import test from "node:test";

import {
  legendColors,
  normalizeSeries,
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
  assert.equal(legendColors.dif, "#12687e");
  assert.equal(legendColors.dea, "#d88b2c");
  assert.equal(legendColors.macdPositive, "#d9553f");
  assert.equal(legendColors.macdNegative, "#14836d");
  assert.equal(legendColors.difFirstChange, "#7b4ca0");
});
