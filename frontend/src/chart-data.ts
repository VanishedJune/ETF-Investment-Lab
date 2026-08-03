export type DatedChartRow = { date: string };
export type DatedIndicatorRow = DatedChartRow & Record<string, unknown>;

/** Align a named indicator to price dates without turning unavailable values into zero. */
export function alignIndicatorValues(
  rows: readonly DatedChartRow[],
  indicators: readonly DatedIndicatorRow[],
  key: string,
): Array<number | null> {
  const byDate = new Map(indicators.map((item) => [item.date, item]));
  return rows.map((row) => {
    const value = byDate.get(row.date)?.[key];
    if (value === null || value === undefined || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  });
}
