export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const endpoint = path.startsWith("/api/") ? path : `/api${path}`;
  const response = await fetch(endpoint, {
    ...init,
    headers: { ...(init.body && !(init.body instanceof Blob) ? { "Content-Type": "application/json" } : {}), ...(init.headers ?? {}) },
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try { message = (await response.json()).detail ?? message; } catch { /* no JSON response */ }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function money(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 2 }).format(number) : String(value);
}

export function percent(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : String(value);
}

export function today(): string { return new Date().toISOString().slice(0, 10); }
