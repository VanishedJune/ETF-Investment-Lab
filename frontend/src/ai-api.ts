import { api, ApiError } from "./api";

export interface DeepSeekConfigPayload {
  has_key: boolean;
  base_url: string;
  model: string;
  masked_key?: string;
}

export interface DeepSeekTestResult {
  ok: boolean;
  model?: string;
  response?: string;
  elapsed_ms?: number;
}

export interface DeepSeekAnalysisResult {
  ok: boolean;
  analysis?: {
    trend_1_2w: "BULLISH" | "NEUTRAL" | "BEARISH";
    trend_4w: "BULLISH" | "NEUTRAL" | "BEARISH";
    trend_8w: "BULLISH" | "NEUTRAL" | "BEARISH";
    confidence: number;
    risk_level: "LOW" | "MEDIUM" | "HIGH";
    summary: string;
    decision: "BUY" | "HOLD" | "SELL" | "WAIT";
    decision_reason: string;
  };
  forecast_source_protocol?: string | null;
  context_hash?: string;
  elapsed_ms?: number;
}

export interface DeepSeekChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface DeepSeekChatResult {
  ok: boolean;
  reply: string;
  elapsed_ms?: number;
  market?: string | null;
  include_market_context?: boolean;
}

export function deepseekErrorMessage(reason: unknown): string {
  if (reason instanceof ApiError) {
    const raw = (reason as unknown as { message?: unknown }).message;
    let detail: unknown = raw;
    if (typeof raw === "string" && raw.startsWith("{")) {
      try {
        detail = JSON.parse(raw);
      } catch { /* keep raw string */ }
    }
    if (detail && typeof detail === "object") {
      const record = detail as Record<string, unknown>;
      const code = record.code;
      const text = record.detail;
      if (code) return `${String(code)}：${String(text ?? "")}`;
    }
    return String(raw ?? reason);
  }
  return reason instanceof Error ? reason.message : String(reason);
}

export function loadDeepSeekConfig(): Promise<DeepSeekConfigPayload> {
  return api<DeepSeekConfigPayload>("/api/ai/deepseek/config");
}

export function saveDeepSeekConfig(body: {
  api_key?: string;
  base_url?: string;
  model?: string;
}): Promise<DeepSeekConfigPayload> {
  return api<DeepSeekConfigPayload>("/api/ai/deepseek/config", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function testDeepSeek(): Promise<DeepSeekTestResult> {
  return api<DeepSeekTestResult>("/api/ai/deepseek/test", { method: "POST" });
}

export function analyzeDeepSeek(
  market: string,
): Promise<DeepSeekAnalysisResult> {
  return api<DeepSeekAnalysisResult>(
    `/api/ai/deepseek/analyze-market/${encodeURIComponent(market)}`,
    { method: "POST" },
  );
}

export function chatDeepSeek(
  market: string,
  messages: DeepSeekChatMessage[],
  includeMarketContext: boolean,
): Promise<DeepSeekChatResult> {
  return api<DeepSeekChatResult>("/api/ai/deepseek/chat", {
    method: "POST",
    body: JSON.stringify({
      market,
      include_market_context: includeMarketContext,
      messages,
    }),
  });
}
