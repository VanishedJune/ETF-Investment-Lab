"""Fixed Chinese Markdown template for deterministic strategy research reports."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from .engine import StrategyEvaluation, StrategySnapshot


DISCLAIMER = "本报告仅用于研究与数据复核，不构成投资建议、收益承诺或自动交易指令。"


def _value(value: Decimal | None) -> str:
    if value is None:
        return "缺失"
    return format(value, "f")


def _lines(items: Iterable[str]) -> str:
    materialized = tuple(items)
    return "\n".join(f"- {item}" for item in materialized) if materialized else "- 无"


def render_strategy_report(
    *,
    instrument_name: str,
    snapshot: StrategySnapshot,
    evaluation: StrategyEvaluation,
    strategy_version: str,
    strategy_config_hash: str,
    source_data_hash: str,
    suggested_buy_amount: Decimal | None,
) -> str:
    """Render only the fixed template and explicit stored/formula values."""
    price_date = snapshot.observations.get("price_date", "缺失")
    valuation_date = snapshot.observations.get("valuation_date", "缺失")
    indicator_date = snapshot.observations.get("indicator_date", "缺失")
    price_source = snapshot.observations.get("price_source", "本地存储")
    cutoff = snapshot.data_cutoff.isoformat() if snapshot.data_cutoff else "缺失"
    metrics = (
        ("收盘价", snapshot.close_price),
        ("估值分位", snapshot.valuation.get("valuation_percentile")),
        ("MA20", snapshot.indicators.get("ma_20")),
        ("MA60", snapshot.indicators.get("ma_60")),
        ("RSI6", snapshot.indicators.get("rsi_6")),
        ("MACD柱", snapshot.indicators.get("macd_histogram")),
        ("成交量", snapshot.volume),
        ("成交量MA20", snapshot.indicators.get("volume_ma_20")),
        ("波动率20", snapshot.indicators.get("volatility_20")),
        ("当前回撤", snapshot.indicators.get("current_drawdown")),
        ("运行期最大回撤", snapshot.indicators.get("running_drawdown")),
    )
    metric_lines = "\n".join(f"- {name}：{_value(value)}" for name, value in metrics)
    buy_amount = _value(suggested_buy_amount) if suggested_buy_amount is not None else "不适用"
    return f"""# ETF规则策略研究报告

## 数据来源与截止日期

- 标的：{instrument_name}（{snapshot.instrument_code}）
- 研究日期：{snapshot.as_of_date.isoformat()}
- 数据截止：{cutoff}
- 本地价格来源：{price_source}；价格日期：{price_date}；估值日期：{valuation_date}；指标日期：{indicator_date}

## 价格、估值与指标

{metric_lines}

## 策略结论

- 策略版本：{strategy_version}
- 策略配置哈希：{strategy_config_hash}
- 数据快照哈希：{source_data_hash}
- 综合得分：{_value(evaluation.score)}
- 建议：{evaluation.recommendation}
- 计划倍率：{_value(evaluation.multiplier)}
- 置信度：{_value(evaluation.confidence)}
- 建议买入金额：{buy_amount}
- 建议卖出比例：{_value(evaluation.suggested_sell_ratio)}

## 规则触发与反向风险

### 已触发规则

{_lines(evaluation.triggered_rules)}

### 数值依据与缺失项

{_lines(evaluation.reasons)}

### 反向风险

{_lines(evaluation.reverse_risks)}

## 研究用途免责声明

{DISCLAIMER}
"""
