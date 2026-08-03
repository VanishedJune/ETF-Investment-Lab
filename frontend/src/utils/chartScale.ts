export interface ScaleInput {
  candles: readonly (readonly number[])[];
  ma5: readonly number[];
  ma10: readonly number[];
  ma20: readonly number[];
  volumes: readonly (number | null)[];
  dif: readonly number[];
  dea: readonly number[];
  macd: readonly number[];
  difFirstChange: readonly number[];
}

export interface SeriesVisibility {
  MA5?: boolean;
  MA10?: boolean;
  MA20?: boolean;
  成交量?: boolean;
  DIF?: boolean;
  DEA?: boolean;
  MACD柱?: boolean;
  DIF一阶变化?: boolean;
}

export interface AxisRange { min: number; max: number }

export interface DynamicYAxisRanges {
  price: AxisRange;
  volume: AxisRange;
  macd: AxisRange;
  difFirstChange: AxisRange;
  startIndex: number;
  endIndex: number;
}

function finite(values: readonly unknown[]): number[] {
  return values
    .map(Number)
    .filter((value) => Number.isFinite(value));
}

function slice(values: readonly number[], start: number, end: number): number[] {
  return finite(values.slice(start, end + 1));
}

function visible(value: boolean | undefined): boolean {
  return value !== false;
}

function paddedRange(values: readonly number[], ratio: number, minimumRatio: number): AxisRange {
  if (!values.length) return { min: -1, max: 1 };
  const ordered = [...values].sort((left, right) => left - right);
  const rawMin = ordered[0];
  const rawMax = ordered.at(-1) ?? rawMin;
  const median = ordered[Math.floor(ordered.length / 2)];
  const epsilon = Math.max(Math.abs(median) * minimumRatio, 1e-8);
  const padding = Math.max((rawMax - rawMin) * ratio, epsilon);
  return { min: rawMin - padding, max: rawMax + padding };
}

export function visibleIndexRange(count: number, startPercent: number, endPercent: number) {
  if (count <= 0) return { startIndex: 0, endIndex: 0 };
  const start = Math.max(0, Math.min(100, startPercent));
  const end = Math.max(start, Math.min(100, endPercent));
  return {
    startIndex: Math.min(count - 1, Math.floor((start / 100) * (count - 1))),
    endIndex: Math.min(count - 1, Math.ceil((end / 100) * (count - 1))),
  };
}

export function calculateDynamicYAxis(
  input: ScaleInput,
  startPercent: number,
  endPercent: number,
  selected: SeriesVisibility = {},
): DynamicYAxisRanges {
  const { startIndex, endIndex } = visibleIndexRange(
    input.candles.length,
    startPercent,
    endPercent,
  );
  const candles = input.candles.slice(startIndex, endIndex + 1);
  const prices = finite(candles.flatMap((candle) => [candle[2], candle[3]]));
  if (visible(selected.MA5)) prices.push(...slice(input.ma5, startIndex, endIndex));
  if (visible(selected.MA10)) prices.push(...slice(input.ma10, startIndex, endIndex));
  if (visible(selected.MA20)) prices.push(...slice(input.ma20, startIndex, endIndex));
  const price = paddedRange(prices, 0.05, 0.002);

  const volumeValues = visible(selected.成交量)
    ? finite(input.volumes.slice(startIndex, endIndex + 1))
    : [];
  const volumeMaximum = volumeValues.length ? Math.max(...volumeValues) : 1;
  const volume = { min: 0, max: Math.max(volumeMaximum * 1.10, 1) };

  const indicators: number[] = [];
  if (visible(selected.DIF)) indicators.push(...slice(input.dif, startIndex, endIndex));
  if (visible(selected.DEA)) indicators.push(...slice(input.dea, startIndex, endIndex));
  if (visible(selected.MACD柱)) indicators.push(...slice(input.macd, startIndex, endIndex));
  indicators.push(0);
  const rawMin = Math.min(...indicators);
  const rawMax = Math.max(...indicators);
  const macdPadding = Math.max((rawMax - rawMin) * 0.08, 1e-8);
  const macd = { min: rawMin - macdPadding, max: rawMax + macdPadding };
  const derivativeValues = visible(selected.DIF一阶变化)
    ? slice(input.difFirstChange, startIndex, endIndex)
    : [];
  derivativeValues.push(0);
  const difFirstChange = paddedRange(derivativeValues, 0.10, 0.01);
  return { price, volume, macd, difFirstChange, startIndex, endIndex };
}
