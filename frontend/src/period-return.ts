import type { PriceRow } from "./types/research";

export function finiteNumber(value: unknown): number | null {
  if (value == null || (typeof value === "string" && !value.trim())) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/** Price return, not portfolio P&L. Never skip a missing intervening bar. */
export function periodReturn(rows: readonly PriceRow[]): number | null {
  if (rows.length < 2) return null;
  const current = finiteNumber(rows.at(-1)?.close);
  const prior = finiteNumber(rows.at(-2)?.close);
  if (current == null || prior == null || current <= 0 || prior <= 0) return null;
  if (rows.at(-1)!.date <= rows.at(-2)!.date) return null;
  return (current / prior - 1) * 100;
}
