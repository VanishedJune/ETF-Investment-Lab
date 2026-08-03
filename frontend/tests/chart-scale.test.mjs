import assert from "node:assert/strict";
import test from "node:test";

import {
  calculateDynamicYAxis,
  visibleIndexRange,
} from "../src/utils/chartScale.ts";

const input = {
  candles: [
    [100, 101, 90, 110],
    [101, 102, 91, 111],
    [102, 103, 92, 112],
    [200, 205, 190, 220],
    [205, 210, 195, 225],
    [210, 215, 200, 230],
  ],
  ma5: [100, 101, 102, 200, 205, 210],
  ma10: [99, 100, 101, 198, 203, 208],
  ma20: [98, 99, 100, 196, 201, 206],
  volumes: [10, 12, 14, 100, 120, 140],
  dif: [-3, -2, -1, 4, 5, 6],
  dea: [-2.5, -2, -1.5, 3, 4, 5],
  macd: [-1, 0, 1, 2, 3, 4],
  difFirstChange: [-0.5, -0.25, 0, 0.75, 1, 1.25],
};

test("maps the zoom percentage to an inclusive visible index range", () => {
  assert.deepEqual(visibleIndexRange(6, 0, 40), { startIndex: 0, endIndex: 2 });
  assert.deepEqual(visibleIndexRange(6, 60, 100), { startIndex: 3, endIndex: 5 });
});

test("recalculates all four Y axes from only the visible range", () => {
  const left = calculateDynamicYAxis(input, 0, 40);
  const right = calculateDynamicYAxis(input, 60, 100);

  assert.ok(left.price.max < right.price.min);
  assert.ok(left.volume.max < right.volume.max);
  assert.ok(left.macd.min < 0);
  assert.ok(right.macd.max > left.macd.max);
  assert.ok(left.difFirstChange.min < 0);
  assert.ok(right.difFirstChange.max > left.difFirstChange.max);
  assert.equal(left.volume.min, 0);
  assert.equal(right.volume.min, 0);
});

test("hidden series do not affect their corresponding visible-range axis", () => {
  const ranges = calculateDynamicYAxis(input, 60, 100, {
    MA5: false,
    MA10: false,
    MA20: false,
    成交量: false,
    DIF: false,
    DEA: false,
    MACD柱: false,
    DIF一阶变化: false,
  });

  assert.deepEqual(ranges.volume, { min: 0, max: 1.1 });
  assert.ok(ranges.macd.min < 0 && ranges.macd.max > 0);
  assert.ok(ranges.difFirstChange.min < 0 && ranges.difFirstChange.max > 0);
});
