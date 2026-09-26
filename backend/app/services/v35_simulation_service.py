"""V3.5 standardized 8-week and continuous simulated accounts.

Every 8-week evaluation window starts from 100,000 CNY with a 0% position and
carries cash/position forward week by week.  Decisions formed at an anchor
close execute at the following complete week's open.  Commissions, minimum
commission and slippage are charged on traded notional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V35ContinuousAccount,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    utc_now,
)
from backend.app.services.v35_config import EXECUTION_DEFAULTS
from backend.app.services.v35_config import PROTOCOL_VERSION
from backend.app.services.v35_strategy_service import V35StrategyDecision


INITIAL_CAPITAL = Decimal("100000.00")
WINDOW_WEEKS = 8


class V35SimulationError(RuntimeError):
    pass


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _pct(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def _money(value: float) -> Decimal:
    return Decimal(str(round(float(value), 2)))


@dataclass(frozen=True, slots=True)
class V35LedgerRow:
    sequence: int
    anchor_date: date
    position_pp: int
    market_return: float
    account_return: float
    equity: Decimal
    cash: Decimal
    trade_action: str
    traded_pp: int
    transaction_cost: Decimal
    event_json: dict[str, Any]
    ledger_hash: str
    held_shares: Decimal | None = None
    average_cost: Decimal | None = None
    sellable_shares: Decimal | None = None


@dataclass(frozen=True, slots=True)
class V35AccountResult:
    account_id: str
    market: str
    scope: str
    window_start: date | None
    window_end: date | None
    package_id: str
    initial_capital: Decimal
    ending_equity: Decimal
    current_cash: Decimal
    current_position_pp: int
    net_profit: Decimal
    net_return: float
    max_drawdown: float
    average_position_pp: float
    trade_count: int
    turnover: float
    transaction_cost: Decimal
    no_action_window: bool
    up_market_participation: float
    down_market_defense: float
    status: str
    ledger: tuple[V35LedgerRow, ...]
    result_json: dict[str, Any]
    account_hash: str


def load_weekly_bars(
    session: Session,
    market: str,
    anchors: Sequence[date],
) -> dict[date, dict[str, float | None]]:
    """Return the weekly open/close for each complete-week anchor."""

    instrument = session.scalar(select(Instrument).where(Instrument.code == market))
    if instrument is None:
        raise V35SimulationError(f"missing instrument {market}")
    if not anchors:
        return {}
    rows = session.scalars(
        select(MarketPrice)
        .where(
            MarketPrice.instrument_id == instrument.id,
            MarketPrice.timeframe == "daily",
            MarketPrice.trade_date >= min(anchors),
            MarketPrice.trade_date <= max(anchors),
        )
        .order_by(MarketPrice.trade_date)
    ).all()
    by_week: dict[tuple[int, int], list[Any]] = {}
    for row in rows:
        year, week, _ = row.trade_date.isocalendar()
        by_week.setdefault((year, week), []).append(row)
    anchor_set = set(anchors)
    result: dict[date, dict[str, float | None]] = {}
    for (year, week), week_rows in by_week.items():
        week_rows.sort(key=lambda item: item.trade_date)
        anchor = week_rows[-1].trade_date
        if anchor not in anchor_set:
            continue
        result[anchor] = {
            "open": float(week_rows[0].open_price),
            "close": float(week_rows[-1].close_price),
            "volume": (
                None
                if any(row.volume is None for row in week_rows)
                else float(sum(float(row.volume) for row in week_rows))
            ),
        }
    return result


def _cost_factors(config: Mapping[str, Any]) -> tuple[float, float]:
    execution = dict(config)
    buy = float(execution.get("buy_commission_rate", EXECUTION_DEFAULTS["buy_commission_rate"]))
    sell = float(execution.get("sell_commission_rate", EXECUTION_DEFAULTS["sell_commission_rate"]))
    slippage = float(execution.get("slippage_bps", EXECUTION_DEFAULTS["slippage_bps"])) / 10000.0
    minimum = float(execution.get("minimum_commission", EXECUTION_DEFAULTS["minimum_commission"]))
    return buy + slippage, sell + slippage


def _transaction_cost(notional: float, rate: float, minimum: float) -> Decimal:
    raw = max(notional * rate, minimum)
    return _money(raw)


def _run_window(
    *,
    market: str,
    package_id: str,
    scope: str,
    anchors: Sequence[date],
    decisions: Mapping[date, V35StrategyDecision],
    bars: Mapping[date, dict[str, float | None]],
    config: Mapping[str, Any],
    initial_position_pp: int = 0,
    initial_equity: Decimal = INITIAL_CAPITAL,
    window_weeks: int | None = WINDOW_WEEKS,
    accounting_mode: str = "PERCENT",
    initial_snapshot: Mapping[str, Any] | None = None,
) -> V35AccountResult:
    if len(anchors) < 1:
        raise V35SimulationError("window requires at least one anchor")
    limit = len(anchors) if window_weeks is None else min(WINDOW_WEEKS, len(anchors))
    window_anchors = tuple(anchors[:limit])
    window_start = window_anchors[0]
    window_end = window_anchors[-1]
    buy_rate, sell_rate = _cost_factors(config)
    minimum = float(config.get("minimum_commission", EXECUTION_DEFAULTS["minimum_commission"]))

    if accounting_mode == "SHARES" and initial_snapshot is not None:
        equity = Decimal(str(initial_snapshot["equity"]))
        cash = Decimal(str(initial_snapshot["cash"]))
        position_pp = float(initial_snapshot["position_pp"])
        held_shares = float(initial_snapshot["held_shares"])
        average_cost = float(initial_snapshot.get("average_cost") or 0.0)
        sellable_shares = float(initial_snapshot["sellable_shares"])
        pending_sellable: list[tuple[date, float]] = list(
            initial_snapshot.get("pending_sellable", [])
        )
    else:
        equity = Decimal(initial_equity)
        cash = Decimal(initial_equity)
        position_pp = initial_position_pp
        held_shares = 0.0
        average_cost = 0.0
        sellable_shares = 0.0
        pending_sellable = []
    equity_series: list[Decimal] = [equity]
    positions: list[int] = []
    ledger: list[V35LedgerRow] = []
    trade_count = 0
    traded_notional = 0.0
    total_cost = Decimal("0.00")
    pending_batches: list[Mapping[str, Any]] = (
        [dict(batch) for batch in initial_snapshot.get("pending_batches", [])]
        if accounting_mode == "SHARES" and initial_snapshot is not None
        else []
    )
    weekly_market_returns: list[float] = []
    weekly_account_returns: list[float] = []
    last_trade_sequence = -10

    for sequence, anchor in enumerate(window_anchors):
        if sequence > 0:
            bar = bars.get(anchor)
            if bar is None or bar["open"] in (None, 0.0) or bar["close"] in (None, 0.0):
                raise V35SimulationError(f"missing weekly bar at {anchor}")
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            market_return = close_price / open_price - 1.0
            weekly_market_returns.append(market_return)
            week_traded_pp = 0
            week_action = "HOLD"

            if pending_batches:
                batch = pending_batches[0]
                action = str(batch["action"])
                target = int(batch["target_position_pp"])
                cooldown_days = int(batch.get("cooldown_trading_days", 5) or 5)
                min_gap_weeks = max(1, math.ceil(cooldown_days / 5))
                if accounting_mode == "SHARES":
                    remaining_sellable: list[tuple[date, float]] = []
                    for acquired, shares in pending_sellable:
                        if acquired < anchor:
                            sellable_shares += shares
                        else:
                            remaining_sellable.append((acquired, shares))
                    pending_sellable = remaining_sellable
                if (
                    sequence - last_trade_sequence >= min_gap_weeks
                    and (
                        (action == "BUY" and target > position_pp)
                        or (action == "SELL" and target < position_pp)
                    )
                ):
                    change = int(batch["batch_change_pp"])
                    if action == "BUY":
                        change = min(change, max(0, target - position_pp))
                    else:
                        change = min(change, max(0, position_pp - target))
                    if change >= 5:
                        notional = float(equity) * change / 100.0
                        rate = buy_rate if action == "BUY" else sell_rate
                        cost = _transaction_cost(notional, rate, minimum)
                        if accounting_mode == "SHARES":
                            if action == "BUY":
                                shares = round(notional / open_price, 2)
                                if shares > 0:
                                    new_cost = average_cost * held_shares + notional
                                    held_shares += shares
                                    average_cost = (
                                        new_cost / held_shares if held_shares else 0.0
                                    )
                                    pending_sellable.append((anchor, shares))
                            else:
                                shares_to_sell = min(
                                    sellable_shares, round(notional / open_price, 2)
                                )
                                if shares_to_sell >= 0.01:
                                    held_shares = round(held_shares - shares_to_sell, 2)
                                    sellable_shares = round(
                                        sellable_shares - shares_to_sell, 2
                                    )
                            cash = cash - _money(notional) - cost
                        else:
                            if action == "BUY":
                                cash = cash - _money(notional) - cost
                                position_pp += change
                            else:
                                cash = cash + _money(notional) - cost
                                position_pp -= change
                        traded_notional += notional
                        total_cost += cost
                        trade_count += 1
                        last_trade_sequence = sequence
                        pending_batches = pending_batches[1:]
                        week_traded_pp = change
                        week_action = action

            if accounting_mode == "SHARES" and held_shares > 0:
                position_frac_at_open = (held_shares * open_price) / max(
                    float(equity), 1e-9
                )
                account_return = position_frac_at_open * market_return
                equity = _money(float(equity) * (1.0 + position_frac_at_open * market_return))
                position_pp = (
                    held_shares * close_price
                ) / max(float(equity), 1e-9) * 100.0
            else:
                position_frac = position_pp / 100.0
                account_return = position_frac * market_return
                equity = _money(float(equity) * (1.0 + account_return))
            weekly_account_returns.append(account_return)
            equity_series.append(equity)
            positions.append(position_pp)

            ledger.append(
                V35LedgerRow(
                    sequence=sequence,
                    anchor_date=anchor,
                    position_pp=position_pp,
                    market_return=market_return,
                    account_return=account_return,
                    equity=equity,
                    cash=cash,
                    trade_action=week_action,
                    traded_pp=week_traded_pp,
                    transaction_cost=cost if week_traded_pp else Decimal("0.00"),
                    event_json={"pending_batches": len(pending_batches)},
                    ledger_hash="",
                    held_shares=(
                        _money(round(held_shares, 2))
                        if accounting_mode == "SHARES"
                        else None
                    ),
                    average_cost=(
                        _money(round(average_cost, 6))
                        if accounting_mode == "SHARES" and average_cost
                        else None
                    ),
                    sellable_shares=(
                        _money(round(sellable_shares, 2))
                        if accounting_mode == "SHARES"
                        else None
                    ),
                )
            )

        decision = decisions.get(anchor)
        if decision is not None:
            if accounting_mode == "SHARES":
                hard_clear = decision.confirmation_status in (
                    "TOP_CONFIRMED",
                    "BEARISH_CONFIRMED",
                ) or any(
                    "紧急" in str(reason) or "OOD" in str(reason)
                    for reason in decision.reasons
                )
                lowered_below_position = (
                    decision.final_target_position_pp < position_pp
                )
                if hard_clear or lowered_below_position:
                    pending_batches = list(decision.batches)
                elif pending_batches:
                    pending_target = int(pending_batches[-1]["target_position_pp"])
                    if decision.final_target_position_pp > pending_target:
                        pending_batches = list(decision.batches)
                else:
                    pending_batches = list(decision.batches)
            else:
                pending_batches = list(decision.batches)

    if window_anchors and not any(row.sequence == 0 for row in ledger):
        ledger.insert(
            0,
            V35LedgerRow(
                sequence=0,
                anchor_date=window_anchors[0],
                position_pp=initial_position_pp,
                market_return=0.0,
                account_return=0.0,
                equity=equity,
                cash=cash,
                trade_action="HOLD",
                traded_pp=0,
                transaction_cost=Decimal("0.00"),
                event_json={"pending_batches": 0},
                ledger_hash="",
                held_shares=None,
                average_cost=None,
                sellable_shares=None,
            ),
        )

    equity_values = [float(value) for value in equity_series]
    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    average_position = float(sum(positions)) / len(positions) if positions else 0.0
    turnover = traded_notional / float(initial_equity) if float(initial_equity) > 0 else 0.0
    up_positions = [
        pos / 100.0
        for pos, ret in zip(positions, weekly_market_returns)
        if ret > 0.0
    ]
    down_positions = [
        pos / 100.0
        for pos, ret in zip(positions, weekly_market_returns)
        if ret < 0.0
    ]
    up_participation = float(sum(up_positions)) / len(up_positions) if up_positions else 0.0
    down_defense = float(sum(1.0 - value for value in down_positions)) / len(down_positions) if down_positions else 0.0
    no_action = average_position < 5.0 and trade_count == 0

    net_profit = equity - Decimal(initial_equity)
    net_return = float(net_profit) / float(initial_equity) if float(initial_equity) > 0 else 0.0
    account_id = _hash(
        {
            "market": market,
            "scope": scope,
            "window_start": window_start.isoformat() if window_start else None,
            "package_id": package_id,
        }
    )
    result_json = {
        "window_weeks": len(window_anchors),
        "weekly_market_returns": weekly_market_returns,
        "weekly_account_returns": weekly_account_returns,
        "equity_series": equity_values,
        "positions": positions,
        "buy_rate_bps": round(buy_rate * 10000.0, 2),
        "sell_rate_bps": round(sell_rate * 10000.0, 2),
        "accounting_mode": accounting_mode,
        "final_snapshot": (
            {
                "equity": round(float(equity), 2),
                "cash": round(float(cash), 2),
                "position_pp": round(float(position_pp), 2),
                "held_shares": round(held_shares, 2),
                "average_cost": round(average_cost, 6),
                "sellable_shares": round(sellable_shares, 2),
                "pending_sellable": [
                    (acquired.isoformat(), round(shares, 2))
                    for acquired, shares in pending_sellable
                ],
            }
            if accounting_mode == "SHARES"
            else None
        ),
    }
    account_hash = _hash(result_json)
    final_ledger: list[V35LedgerRow] = []
    for row in ledger:
        payload = {
            "account_id": account_id,
            "sequence": row.sequence,
            "anchor": row.anchor_date.isoformat(),
            "position_pp": row.position_pp,
            "market_return": row.market_return,
            "account_return": row.account_return,
            "equity": str(row.equity),
            "cash": str(row.cash),
            "event": row.event_json,
        }
        final_ledger.append(
            V35LedgerRow(
                sequence=row.sequence,
                anchor_date=row.anchor_date,
                position_pp=row.position_pp,
                market_return=row.market_return,
                account_return=row.account_return,
                equity=row.equity,
                cash=row.cash,
                trade_action=row.trade_action,
                traded_pp=row.traded_pp,
                transaction_cost=row.transaction_cost,
                event_json=row.event_json,
                ledger_hash=_hash(payload),
                held_shares=row.held_shares,
                average_cost=row.average_cost,
                sellable_shares=row.sellable_shares,
            )
        )
    return V35AccountResult(
        account_id=account_id,
        market=market,
        scope=scope,
        window_start=window_start,
        window_end=window_end,
        package_id=package_id,
        initial_capital=Decimal(initial_equity),
        ending_equity=equity,
        current_cash=cash,
        current_position_pp=position_pp,
        net_profit=net_profit,
        net_return=net_return,
        max_drawdown=max_drawdown,
        average_position_pp=average_position,
        trade_count=trade_count,
        turnover=turnover,
        transaction_cost=total_cost,
        no_action_window=no_action,
        up_market_participation=up_participation,
        down_market_defense=down_defense,
        status=(
            "COMPLETED"
            if window_weeks is None or len(window_anchors) == WINDOW_WEEKS
            else "PENDING"
        ),
        ledger=tuple(final_ledger),
        result_json=result_json,
        account_hash=account_hash,
    )


def _upsert_account(
    session: Session,
    result: V35AccountResult,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    account = session.scalar(
        select(V35SimAccount).where(
            V35SimAccount.model_market == result.market,
            V35SimAccount.scope == result.scope,
            V35SimAccount.window_start_date == result.window_start,
            V35SimAccount.model_package_id == result.package_id,
        )
    )
    snapshot = (result.result_json or {}).get("final_snapshot")
    now = utc_now()
    if account is None:
        account = V35SimAccount(
            id=result.account_id,
            protocol_version=protocol_version,
            model_market=result.market,
            scope=result.scope,
            window_start_date=result.window_start,
            window_end_date=result.window_end,
            model_package_id=result.package_id,
            initial_capital=result.initial_capital,
            ending_equity=result.ending_equity,
            current_cash=result.current_cash,
            current_position_pp=result.current_position_pp,
            net_profit=result.net_profit,
            net_return=_pct(result.net_return),
            max_drawdown=_pct(result.max_drawdown),
            average_position_pp=_pct(result.average_position_pp),
            trade_count=result.trade_count,
            turnover=_pct(result.turnover),
            transaction_cost=result.transaction_cost,
            no_action_window=result.no_action_window,
            up_market_participation=_pct(result.up_market_participation),
            down_market_defense=_pct(result.down_market_defense),
            status=result.status,
            result_json=result.result_json,
            held_shares=_money(snapshot["held_shares"]) if snapshot else None,
            average_cost=_money(snapshot["average_cost"]) if snapshot and snapshot.get("average_cost") else None,
            sellable_shares=_money(snapshot["sellable_shares"]) if snapshot else None,
            account_hash=result.account_hash,
            created_at=now,
            updated_at=now,
        )
        session.add(account)
    else:
        account.ending_equity = result.ending_equity
        account.current_cash = result.current_cash
        account.current_position_pp = result.current_position_pp
        account.net_profit = result.net_profit
        account.net_return = _pct(result.net_return)
        account.max_drawdown = _pct(result.max_drawdown)
        account.average_position_pp = _pct(result.average_position_pp)
        account.trade_count = result.trade_count
        account.turnover = _pct(result.turnover)
        account.transaction_cost = result.transaction_cost
        account.no_action_window = result.no_action_window
        account.up_market_participation = _pct(result.up_market_participation)
        account.down_market_defense = _pct(result.down_market_defense)
        account.status = result.status
        account.result_json = result.result_json
        account.held_shares = _money(snapshot["held_shares"]) if snapshot else None
        account.average_cost = _money(snapshot["average_cost"]) if snapshot and snapshot.get("average_cost") else None
        account.sellable_shares = _money(snapshot["sellable_shares"]) if snapshot else None
        account.account_hash = result.account_hash
        account.updated_at = now
    session.flush()

    existing_ledger = session.scalars(
        select(V35SimLedger).where(V35SimLedger.account_id == result.account_id)
    ).all()
    for row in existing_ledger:
        session.delete(row)
    session.flush()
    for row in result.ledger:
        session.add(
            V35SimLedger(
                account_id=result.account_id,
                sequence=row.sequence,
                anchor_date=row.anchor_date,
                position_pp=row.position_pp,
                market_return=_pct(row.market_return),
                account_return=_pct(row.account_return),
                equity=row.equity,
                cash=row.cash,
                trade_action=row.trade_action,
                traded_pp=row.traded_pp,
                transaction_cost=row.transaction_cost,
                event_json=row.event_json,
                held_shares=row.held_shares,
                average_cost=row.average_cost,
                sellable_shares=row.sellable_shares,
                ledger_hash=row.ledger_hash,
                created_at=now,
            )
        )


def _upsert_evaluation(
    session: Session,
    result: V35AccountResult,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    if result.status != "COMPLETED" or result.window_end is None or result.scope != "STANDARD_8W":
        return
    evaluation = session.scalar(
        select(V35SimEvaluation).where(
            V35SimEvaluation.account_id == result.account_id,
            V35SimEvaluation.window_end_date == result.window_end,
        )
    )
    payload = {
        "account_id": result.account_id,
        "window_start": result.window_start.isoformat(),
        "window_end": result.window_end.isoformat(),
        "net_return": result.net_return,
        "net_profit": str(result.net_profit),
        "max_drawdown": result.max_drawdown,
        "average_position_pp": result.average_position_pp,
        "trade_count": result.trade_count,
        "turnover": result.turnover,
        "transaction_cost": str(result.transaction_cost),
        "no_action_window": result.no_action_window,
        "up_market_participation": result.up_market_participation,
        "down_market_defense": result.down_market_defense,
    }
    evaluation_hash = _hash(payload)
    if evaluation is not None:
        return
    session.add(
        V35SimEvaluation(
            account_id=result.account_id,
            model_market=result.market,
            model_package_id=result.package_id,
            window_start_date=result.window_start,
            window_end_date=result.window_end,
            ending_equity=result.ending_equity,
            net_profit=result.net_profit,
            net_return=_pct(result.net_return),
            max_drawdown=_pct(result.max_drawdown),
            average_position_pp=_pct(result.average_position_pp),
            trade_count=result.trade_count,
            turnover=_pct(result.turnover),
            transaction_cost=result.transaction_cost,
            no_action_window=result.no_action_window,
            up_market_participation=_pct(result.up_market_participation),
            down_market_defense=_pct(result.down_market_defense),
            evaluation_hash=evaluation_hash,
            created_at=utc_now(),
        )
    )


def persist_window_result(
    session: Session,
    result: V35AccountResult,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    _upsert_account(session, result, protocol_version=protocol_version)
    _upsert_evaluation(session, result, protocol_version=protocol_version)


def persist_continuous_result(
    session: Session,
    result: V35AccountResult,
    *,
    market_returns: Sequence[float],
    elapsed_years: float,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    if result.scope != "CONTINUOUS":
        raise V35SimulationError("continuous persistence requires a continuous account")
    cumulative = float(result.net_return)
    annualized = (1.0 + cumulative) ** (1.0 / max(elapsed_years, 0.01)) - 1.0 if cumulative > -1.0 else -1.0
    buy_hold = 1.0
    for market_return in market_returns:
        buy_hold *= 1.0 + market_return
    buy_hold_return = buy_hold - 1.0
    fixed_30_return = sum(0.30 * value for value in market_returns)
    cash_return = 0.0
    state_json = {
        "elapsed_years": round(elapsed_years, 6),
        "weekly_market_return_count": len(market_returns),
        "equity_series": result.result_json.get("equity_series", []),
        "positions": result.result_json.get("positions", []),
    }
    account_hash = _hash(
        {
            "market": result.market,
            "scope": "CONTINUOUS",
            "package_id": result.package_id,
            "ending_equity": str(result.ending_equity),
            "cumulative_return": cumulative,
            "annualized_return": annualized,
            "max_drawdown": result.max_drawdown,
            "average_position_pp": result.average_position_pp,
            "turnover": result.turnover,
            "buy_hold_return": buy_hold_return,
            "fixed_30_return": fixed_30_return,
            "cash_return": cash_return,
        }
    )
    snapshot = (result.result_json or {}).get("final_snapshot")
    account = session.scalar(
        select(V35ContinuousAccount).where(
            V35ContinuousAccount.model_market == result.market,
            V35ContinuousAccount.protocol_version == protocol_version,
        )
    )
    now = utc_now()
    if account is None:
        session.add(
            V35ContinuousAccount(
                protocol_version=protocol_version,
                model_market=result.market,
                model_package_id=result.package_id,
                initial_capital=result.initial_capital,
                ending_equity=result.ending_equity,
                cumulative_return=_pct(cumulative),
                annualized_return=_pct(annualized),
                max_drawdown=_pct(result.max_drawdown),
                average_position_pp=_pct(result.average_position_pp),
                turnover=_pct(result.turnover),
                buy_hold_return=_pct(buy_hold_return),
                fixed_30_return=_pct(fixed_30_return),
                cash_return=Decimal("0.00"),
                state_json=state_json,
                held_shares=_money(snapshot["held_shares"]) if snapshot else None,
                average_cost=_money(snapshot["average_cost"]) if snapshot and snapshot.get("average_cost") else None,
                sellable_shares=_money(snapshot["sellable_shares"]) if snapshot else None,
                account_hash=account_hash,
                updated_at=now,
            )
        )
    else:
        account.model_package_id = result.package_id
        account.ending_equity = result.ending_equity
        account.cumulative_return = _pct(cumulative)
        account.annualized_return = _pct(annualized)
        account.max_drawdown = _pct(result.max_drawdown)
        account.average_position_pp = _pct(result.average_position_pp)
        account.turnover = _pct(result.turnover)
        account.buy_hold_return = _pct(buy_hold_return)
        account.fixed_30_return = _pct(fixed_30_return)
        account.cash_return = Decimal("0.00")
        account.state_json = state_json
        account.held_shares = _money(snapshot["held_shares"]) if snapshot else None
        account.average_cost = _money(snapshot["average_cost"]) if snapshot and snapshot.get("average_cost") else None
        account.sellable_shares = _money(snapshot["sellable_shares"]) if snapshot else None
        account.account_hash = account_hash
        account.updated_at = now


def run_standard_window(
    session: Session,
    *,
    market: str,
    package_id: str,
    anchors: Sequence[date],
    decisions: Mapping[date, V35StrategyDecision],
    bars: Mapping[date, dict[str, float | None]],
    config: Mapping[str, Any],
    accounting_mode: str = "PERCENT",
    initial_snapshot: Mapping[str, Any] | None = None,
) -> V35AccountResult:
    return _run_window(
        market=market,
        package_id=package_id,
        scope="STANDARD_8W",
        anchors=anchors,
        decisions=decisions,
        bars=bars,
        config=config,
        accounting_mode=accounting_mode,
        initial_snapshot=initial_snapshot,
    )


def run_continuous_window(
    session: Session,
    *,
    market: str,
    package_id: str,
    anchors: Sequence[date],
    decisions: Mapping[date, V35StrategyDecision],
    bars: Mapping[date, dict[str, float | None]],
    config: Mapping[str, Any],
    accounting_mode: str = "PERCENT",
    initial_snapshot: Mapping[str, Any] | None = None,
) -> V35AccountResult:
    """Run the continuous account across all anchors without resetting."""

    return _run_window(
        market=market,
        package_id=package_id,
        scope="CONTINUOUS",
        anchors=anchors,
        decisions=decisions,
        bars=bars,
        config=config,
        initial_position_pp=0,
        initial_equity=INITIAL_CAPITAL,
        window_weeks=None,
        accounting_mode=accounting_mode,
        initial_snapshot=initial_snapshot,
    )
