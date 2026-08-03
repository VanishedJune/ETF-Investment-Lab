"""Adapters for permitted public/local market-data sources only."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import time as time_module
from typing import Any, Callable, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..schemas.market import MarketDataRecord, ProviderResult, ProviderStatus
from ..schemas.valuation import ValuationDataRecord, ValuationProviderResult
from .csv_import import parse_csv_market_data
from .instrument_universe import etf_exchange_prefix


class MarketDataProvider(Protocol):
    """All providers return the same non-throwing, typed result."""

    source: str

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult: ...


def _to_records(
    rows: Any, source: str, *, amount_multiplier: Decimal = Decimal("1")
) -> list[MarketDataRecord]:
    """Accept dataframe records or mappings from public provider clients."""
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    records: list[MarketDataRecord] = []
    for row in rows or []:
        normalized = {str(key).strip().lower().replace("_", ""): value for key, value in row.items()}
        def value(*names: str):
            for name in names:
                if name in row:
                    return row[name]
                if name.lower().replace("_", "") in normalized:
                    return normalized[name.lower().replace("_", "")]
            return None

        raw_date = value("日期", "trade_date", "date")
        if raw_date is None:
            continue
        def csv_cell(*names: str) -> str:
            raw_value = value(*names)
            return "" if raw_value is None else str(raw_value)

        parsed = parse_csv_market_data(
            "date,open,high,low,close,volume,amount\n"
            f"{raw_date},{csv_cell('开盘', 'open')},{csv_cell('最高', 'high')},"
            f"{csv_cell('最低', 'low')},{csv_cell('收盘', 'close')},"
            f"{csv_cell('成交量', 'vol', 'volume')},{csv_cell('成交额', 'amount')}\n"
        )
        for record in parsed.records:
            # Canonical provider contract: amount is CNY. Tushare fund_daily
            # reports amount in thousand CNY, so its adapter supplies 1000 here.
            amount = record.amount * amount_multiplier if record.amount is not None else None
            records.append(record.model_copy(update={"source": source, "amount": amount}))
    return records


class AkShareProvider:
    """Uses only the installed, public AkShare package API; network errors are data."""

    source = "AKSHARE"

    def __init__(self, history_loader: Callable[..., Any] | None = None) -> None:
        self.history_loader = history_loader

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        try:
            loader = self.history_loader
            if loader is None:
                import akshare as ak  # permitted public-package adapter

                loader = ak.fund_etf_hist_em
            frame = loader(
                symbol=instrument_code,
                period="daily",
                start_date=start_date.strftime("%Y%m%d") if start_date else "",
                end_date=end_date.strftime("%Y%m%d") if end_date else "",
                adjust="",
            )
            records = _to_records(frame, self.source)
            if not records:
                return ProviderResult.failed(self.source, "AkShare returned no usable daily records")
            return ProviderResult.success(self.source, records)
        except Exception as error:  # external/public source failures must not erase local cache
            return ProviderResult.failed(self.source, f"AkShare fetch failed: {error}")

    def fetch_weekly(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        """Read an ETF's native weekly bars before falling back to daily sums.

        ``fund_etf_hist_em`` exposes the same OHLCV fields for ``weekly`` as
        it does for ``daily``.  The caller still validates coverage against
        the complete locally stored daily series, so a partial upstream
        response can never create gaps in the chart.
        """

        source = "AKSHARE_ETF_WEEKLY"
        try:
            loader = self.history_loader
            if loader is None:
                import akshare as ak  # permitted public-package adapter

                loader = ak.fund_etf_hist_em
            frame = loader(
                symbol=instrument_code,
                period="weekly",
                start_date=start_date.strftime("%Y%m%d") if start_date else "",
                end_date=end_date.strftime("%Y%m%d") if end_date else "",
                adjust="",
            )
            records = _to_records(frame, source)
            if not records:
                return ProviderResult.failed(
                    source,
                    "AkShare returned no usable native weekly ETF records",
                )
            return ProviderResult.success(source, records)
        except Exception as error:
            return ProviderResult.failed(
                source,
                f"AkShare native weekly ETF fetch failed: {error}",
            )


class AkShareEtfResearchProvider(AkShareProvider):
    """ETF adapter that preserves trade prices and adds CNY adjusted returns.

    The unadjusted close remains the chart/trade comparison price.  AkShare's
    forward-adjusted (``qfq``) close is stored separately and is the V3.3
    return-model target, avoiding dividend/distribution discontinuities while
    keeping all values in the ETF's CNY trading currency.
    """

    source = "AKSHARE_ETF"

    def __init__(
        self,
        history_loader: Callable[..., Any] | None = None,
        sina_loader: Callable[..., Any] | None = None,
        tencent_loader: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(history_loader=history_loader)
        self.sina_loader = sina_loader
        self.tencent_loader = tencent_loader

    @staticmethod
    def _download_tencent_qfq(
        instrument_code: str,
        start_date: date | None,
        end_date: date | None,
    ) -> list[list[object]]:
        """Download the public Tencent qfq series in bounded 640-row pages."""

        import requests

        rows_by_date: dict[str, list[object]] = {}
        exchange_prefix = etf_exchange_prefix(instrument_code)
        cursor_end = end_date or datetime.now(timezone.utc).date()
        for _page in range(20):
            start_text = start_date.isoformat() if start_date is not None else ""
            parameter = (
                f"{exchange_prefix}{instrument_code},day,{start_text},{cursor_end.isoformat()},640,qfq"
            )
            response = requests.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": parameter},
                headers={
                    "User-Agent": "ETF-Investment-Lab/3.3",
                    "Referer": f"https://gu.qq.com/{exchange_prefix}{instrument_code}",
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("code", -1)) != 0:
                raise ValueError(f"Tencent qfq error: {payload.get('msg')}")
            instrument = (payload.get("data") or {}).get(
                f"{exchange_prefix}{instrument_code}"
            ) or {}
            page_rows = instrument.get("qfqday") or instrument.get("day") or []
            if not page_rows:
                break
            for row in page_rows:
                if row and len(row) >= 6:
                    rows_by_date[str(row[0])] = list(row)
            earliest = date.fromisoformat(str(page_rows[0][0]))
            if len(page_rows) < 640 or (start_date is not None and earliest <= start_date):
                break
            cursor_end = earliest - timedelta(days=1)
        return [rows_by_date[key] for key in sorted(rows_by_date)]

    def _tencent_qfq(
        self,
        instrument_code: str,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[date, tuple[Decimal, Decimal]]:
        loader = self.tencent_loader or self._download_tencent_qfq
        rows = loader(instrument_code, start_date, end_date)
        output: dict[date, tuple[Decimal, Decimal]] = {}
        for row in rows or []:
            try:
                trade_date = date.fromisoformat(str(row[0])[:10])
                adjusted_close = Decimal(str(row[2]).replace(",", ""))
                volume_lots = Decimal(str(row[5]).replace(",", ""))
            except (ArithmeticError, IndexError, TypeError, ValueError):
                continue
            if adjusted_close > 0 and volume_lots >= 0:
                output[trade_date] = (adjusted_close, volume_lots)
        return output

    @staticmethod
    def _load_with_retries(
        loader: Callable[..., Any],
        arguments: dict[str, object],
        *,
        adjust: str,
        attempts: int = 3,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return loader(**arguments, adjust=adjust)
            except Exception as error:
                last_error = error
                if attempt + 1 < attempts:
                    time_module.sleep(0.6 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        try:
            loader = self.history_loader
            if loader is None:
                import akshare as ak  # permitted public-package adapter

                loader = ak.fund_etf_hist_em
            common = {
                "symbol": instrument_code,
                "period": "daily",
                "start_date": start_date.strftime("%Y%m%d") if start_date else "",
                "end_date": end_date.strftime("%Y%m%d") if end_date else "",
            }
            warnings: list[str] = []
            result_source = self.source
            eastmoney_volume_is_lots = True
            try:
                raw_rows = self._load_with_retries(loader, common, adjust="")
            except Exception as primary_error:
                fallback_loader = self.sina_loader
                if fallback_loader is None and self.history_loader is None:
                    import akshare as ak

                    fallback_loader = ak.fund_etf_hist_sina
                if fallback_loader is None:
                    raise
                raw_rows = fallback_loader(
                    symbol=f"{etf_exchange_prefix(instrument_code)}{instrument_code}"
                )
                result_source = "AKSHARE_ETF_SINA"
                eastmoney_volume_is_lots = False
                warnings.append(
                    f"EastMoney ETF history unavailable; Sina public history used: {primary_error}"
                )
            raw_records = _to_records(raw_rows, result_source)
            if start_date is not None:
                raw_records = [record for record in raw_records if record.trade_date >= start_date]
            if end_date is not None:
                raw_records = [record for record in raw_records if record.trade_date <= end_date]
            if not raw_records:
                return ProviderResult.failed(
                    result_source,
                    "AkShare returned no usable unadjusted ETF records",
                )
            try:
                adjusted_records = _to_records(
                    self._load_with_retries(loader, common, adjust="qfq"),
                    f"{self.source}_QFQ",
                )
                adjusted_by_date = {
                    record.trade_date: record.close_price
                    for record in adjusted_records
                }
            except Exception as error:
                adjusted_by_date = {}
                warnings.append(
                    f"CNY qfq adjusted series unavailable; raw ETF prices retained: {error}"
                )
            need_tencent = (
                len(adjusted_by_date) < len(raw_records)
                or any(record.volume is not None and record.volume <= 0 for record in raw_records)
            )
            tencent_by_date: dict[date, tuple[Decimal, Decimal]] = {}
            if need_tencent:
                try:
                    tencent_by_date = self._tencent_qfq(
                        instrument_code,
                        start_date or raw_records[0].trade_date,
                        end_date or raw_records[-1].trade_date,
                    )
                    if tencent_by_date:
                        warnings.append(
                            "Tencent public qfq/volume series cross-checked missing AkShare fields"
                        )
                except Exception as error:
                    warnings.append(f"Tencent ETF cross-check unavailable: {error}")
            records = [
                record.model_copy(
                    update={
                        "adjusted_close_price": (
                            adjusted_by_date.get(record.trade_date)
                            or (
                                tencent_by_date[record.trade_date][0]
                                if record.trade_date in tencent_by_date
                                else None
                            )
                        ),
                        "volume": (
                            tencent_by_date[record.trade_date][1]
                            if record.volume is not None
                            and record.volume <= 0
                            and record.trade_date in tencent_by_date
                            else record.volume
                        ),
                        "volume_multiplier": (
                            100
                            if record.volume is not None
                            and (
                                eastmoney_volume_is_lots
                                or (
                                    record.volume <= 0
                                    and record.trade_date in tencent_by_date
                                )
                            )
                            else 1
                        ),
                        "source": result_source,
                        "volume_source": (
                            f"{result_source}:LOTS_X100"
                            if eastmoney_volume_is_lots and record.volume is not None
                            else "TENCENT_QFQ:LOTS_X100"
                            if record.volume is not None
                            and record.volume <= 0
                            and record.trade_date in tencent_by_date
                            else result_source if record.volume is not None else None
                        ),
                    }
                )
                for record in raw_records
            ]
            missing_adjusted = sum(
                record.adjusted_close_price is None for record in records
            )
            if missing_adjusted and adjusted_by_date:
                warnings.append(
                    f"CNY qfq adjusted close missing for {missing_adjusted} ETF sessions"
                )
            return ProviderResult.success(
                result_source,
                records,
                warnings=warnings,
            )
        except Exception as error:
            return ProviderResult.failed(
                self.source,
                f"AkShare ETF research fetch failed: {error}",
            )

    def fetch_weekly(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        result = super().fetch_weekly(instrument_code, start_date, end_date)
        if result.status is not ProviderStatus.SUCCESS:
            return result
        return result.model_copy(
            update={
                "records": [
                    record.model_copy(
                        update={
                            "volume_multiplier": 100 if record.volume is not None else 1,
                            "volume_source": (
                                "AKSHARE_ETF_WEEKLY:LOTS_X100"
                                if record.volume is not None
                                else None
                            ),
                        }
                    )
                    for record in result.records
                ]
            }
        )


INDEX_UNIVERSE: dict[str, dict[str, str]] = {
    "000688": {"market": "CN", "symbol": "000688", "name": "科创50指数"},
    "399006": {"market": "CN", "symbol": "399006", "name": "创业板指数"},
    "NDX": {"market": "US", "symbol": ".NDX", "name": "纳斯达克100指数"},
}


def _index_records(rows: Any, source: str) -> list[MarketDataRecord]:
    """Normalize the two public AkShare index tables without ETF adapters.

    Chinese index history and the US-index endpoint expose different column
    spellings; accepting both at this boundary keeps every downstream formula
    identical and guarantees that no ETF quote is requested.
    """
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    records: list[MarketDataRecord] = []
    aliases = {
        "date": ("日期", "date", "datetime", "时间"),
        "open": ("开盘", "open"),
        "high": ("最高", "high"),
        "low": ("最低", "low"),
        "close": ("收盘", "close", "latest"),
        "volume": ("成交量", "volume", "vol"),
        "amount": ("成交额", "amount", "turnover"),
    }

    def normalized_key(key: object) -> str:
        return str(key).strip().lower().replace("_", "").replace(" ", "")

    for row in rows or []:
        normalized = {normalized_key(key): value for key, value in row.items()}

        def value(name: str) -> object | None:
            for alias in aliases[name]:
                candidate = normalized.get(normalized_key(alias))
                if candidate is not None and str(candidate).strip() not in {"", "nan", "None"}:
                    return candidate
            return None

        raw_date, raw_open, raw_high, raw_low, raw_close = (
            value("date"), value("open"), value("high"), value("low"), value("close")
        )
        if None in (raw_date, raw_open, raw_high, raw_low, raw_close):
            continue
        try:
            parsed_date = date.fromisoformat(str(raw_date)[:10].replace("/", "-"))
            as_decimal = lambda item: Decimal(str(item).replace(",", ""))
            volume = value("volume")
            records.append(
                MarketDataRecord(
                    trade_date=parsed_date,
                    open_price=as_decimal(raw_open),
                    high_price=as_decimal(raw_high),
                    low_price=as_decimal(raw_low),
                    close_price=as_decimal(raw_close),
                    volume=as_decimal(volume) if volume is not None else None,
                    # Index providers use incompatible turnover units and a
                    # monthly sum can exceed this SQLite fixed-point column's
                    # safe range.  Volume remains available for the chart;
                    # turnover is intentionally absent rather than distorted.
                    amount=None,
                    source=source,
                )
            )
        except (ArithmeticError, ValueError):
            continue
    return records


class AkShareIndexProvider:
    """Direct public-index adapter for 科创50、创业板 and Nasdaq-100.

    The application uses this provider exclusively for market refreshes.  The
    legacy ETF adapter remains isolated for backwards-compatible local tests
    and never receives a request from the user-facing research workflow.
    """

    source = "AKSHARE_INDEX"

    def __init__(
        self,
        china_loader: Callable[..., Any] | None = None,
        china_fallback_loader: Callable[..., Any] | None = None,
        us_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.china_loader = china_loader
        self.china_fallback_loader = china_fallback_loader
        self.us_loader = us_loader

    def _fallback_china_history(
        self,
        spec: dict[str, str],
        start_date: date | None,
        end_date: date | None,
        primary_error: Exception | str,
    ) -> ProviderResult:
        """Use AkShare's documented Sina history endpoint when EastMoney fails.

        This remains a direct index request (`sh000688` / `sz399006`), never
        an ETF quote.  The source is changed so stored rows and UI attribution
        disclose which public endpoint supplied the curve.
        """
        try:
            loader = self.china_fallback_loader
            if loader is None:
                import akshare as ak

                loader = ak.stock_zh_index_daily
            prefix = "sh" if spec["symbol"].startswith("0") else "sz"
            rows = loader(symbol=f"{prefix}{spec['symbol']}")
            records = [
                record
                for record in _index_records(rows, "AKSHARE_INDEX_SINA")
                if (start_date is None or record.trade_date >= start_date)
                and (end_date is None or record.trade_date <= end_date)
            ]
            if records:
                return ProviderResult.success(
                    "AKSHARE_INDEX_SINA",
                    records,
                    warnings=[f"Primary AkShare index endpoint failed: {primary_error}"],
                )
            return ProviderResult.failed(
                self.source,
                f"Primary and Sina AkShare direct-index endpoints returned no usable {spec['name']} records; primary error: {primary_error}",
            )
        except Exception as fallback_error:
            return ProviderResult.failed(
                self.source,
                f"Primary AkShare direct-index fetch failed: {primary_error}; Sina history fallback failed: {fallback_error}",
            )

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        spec = INDEX_UNIVERSE.get(instrument_code)
        if spec is None:
            return ProviderResult.failed(self.source, f"Unsupported direct index: {instrument_code}")
        try:
            if spec["market"] == "CN":
                loader = self.china_loader
                if loader is None:
                    import akshare as ak

                    loader = ak.index_zh_a_hist
                rows = loader(
                    symbol=spec["symbol"],
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d") if start_date else "19700101",
                    end_date=end_date.strftime("%Y%m%d") if end_date else "22220101",
                )
            else:
                loader = self.us_loader
                if loader is None:
                    import akshare as ak

                    loader = ak.index_us_stock_sina
                rows = loader(symbol=spec["symbol"])
            records = [
                record
                for record in _index_records(rows, self.source)
                if (start_date is None or record.trade_date >= start_date)
                and (end_date is None or record.trade_date <= end_date)
            ]
            if instrument_code == "NDX":
                records = [
                    record.model_copy(update={"volume": None})
                    for record in records
                ]
            if not records:
                if spec["market"] == "CN":
                    return self._fallback_china_history(
                        spec, start_date, end_date, "primary endpoint returned no usable rows"
                    )
                return ProviderResult.failed(
                    self.source,
                    f"AkShare returned no usable {spec['name']} records",
                    volume_availability=(
                        "not_available_for_direct_index"
                        if instrument_code == "NDX"
                        else "available"
                    ),
                )
            return ProviderResult.success(
                self.source,
                records,
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            )
        except Exception as error:
            if spec["market"] == "CN":
                return self._fallback_china_history(spec, start_date, end_date, error)
            return ProviderResult.failed(
                self.source,
                f"AkShare direct-index fetch failed: {error}",
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            )

    def fetch_weekly(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        """Try the direct upstream weekly endpoint before local daily aggregation.

        AkShare's China index history endpoint accepts ``period="weekly"``.
        The Sina Nasdaq index endpoint does not expose an equivalent native
        weekly-volume contract, so NDX deliberately falls back to locally
        aggregated daily index rows.
        """
        spec = INDEX_UNIVERSE.get(instrument_code)
        source = "AKSHARE_INDEX_WEEKLY"
        if spec is None:
            return ProviderResult.failed(source, f"Unsupported direct index: {instrument_code}")
        if spec["market"] != "CN":
            return ProviderResult.failed(
                source,
                f"No native upstream weekly-volume endpoint is configured for {instrument_code}",
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            )
        try:
            loader = self.china_loader
            if loader is None:
                import akshare as ak

                loader = ak.index_zh_a_hist
            rows = loader(
                symbol=spec["symbol"],
                period="weekly",
                start_date=start_date.strftime("%Y%m%d") if start_date else "19700101",
                end_date=end_date.strftime("%Y%m%d") if end_date else "22220101",
            )
            records = [
                record
                for record in _index_records(rows, source)
                if (start_date is None or record.trade_date >= start_date)
                and (end_date is None or record.trade_date <= end_date)
            ]
            if not records:
                return ProviderResult.failed(source, "Upstream weekly endpoint returned no usable records")
            return ProviderResult.success(source, records)
        except Exception as error:
            return ProviderResult.failed(source, f"Upstream weekly fetch failed: {error}")


class YahooDirectIndexProvider:
    """Fetch Nasdaq-100 history from Yahoo's direct ``^NDX`` chart series.

    This provider is intentionally limited to NDX.  It is a direct-index
    fallback for gaps in AkShare/Sina history and never substitutes QQQ or any
    other ETF.  Yahoo does not publish a meaningful direct-index volume series,
    so every row carries an explicit unavailable marker.
    """

    source = "YAHOO_DIRECT_INDEX"
    _symbols = {"NDX": "^NDX"}

    def __init__(
        self,
        loader: Callable[[str, date | None, date | None], Any] | None = None,
    ) -> None:
        self.loader = loader or self._download

    @staticmethod
    def _download(
        symbol: str,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[str, Any]:
        first_day = start_date or date(1970, 1, 1)
        last_day = end_date or datetime.now(timezone.utc).date()
        params = urlencode(
            {
                "period1": int(
                    datetime.combine(first_day, datetime.min.time(), timezone.utc).timestamp()
                ),
                "period2": int(
                    datetime.combine(
                        last_day + timedelta(days=1),
                        datetime.min.time(),
                        timezone.utc,
                    ).timestamp()
                ),
                "interval": "1d",
                "events": "history",
            }
        )
        request = Request(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}",
            headers={"User-Agent": "ETF-Investment-Lab/2.0"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _decimal(value: object) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.00000001")).normalize()

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        symbol = self._symbols.get(instrument_code)
        if symbol is None:
            return ProviderResult.failed(
                self.source,
                f"Unsupported Yahoo direct index: {instrument_code}",
                volume_availability="not_available_for_direct_index",
            )
        try:
            payload = self.loader(symbol, start_date, end_date)
            if isinstance(payload, bytes):
                payload = json.loads(payload.decode("utf-8"))
            elif isinstance(payload, str):
                payload = json.loads(payload)
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise ValueError(str(chart["error"]))
            results = chart.get("result") or []
            if not results:
                raise ValueError("Yahoo chart returned no result")
            result = results[0]
            timestamps = result.get("timestamp") or []
            quotes = ((result.get("indicators") or {}).get("quote") or [])
            quote = quotes[0] if quotes else {}
            records: list[MarketDataRecord] = []
            for index, raw_timestamp in enumerate(timestamps):
                values = {
                    name: (quote.get(name) or [])[index]
                    if index < len(quote.get(name) or [])
                    else None
                    for name in ("open", "high", "low", "close")
                }
                if raw_timestamp is None or any(value is None for value in values.values()):
                    continue
                trade_date = datetime.fromtimestamp(
                    int(raw_timestamp), timezone.utc
                ).date()
                if start_date is not None and trade_date < start_date:
                    continue
                if end_date is not None and trade_date > end_date:
                    continue
                records.append(
                    MarketDataRecord(
                        trade_date=trade_date,
                        open_price=self._decimal(values["open"]),
                        high_price=self._decimal(values["high"]),
                        low_price=self._decimal(values["low"]),
                        close_price=self._decimal(values["close"]),
                        volume=None,
                        amount=None,
                        source=self.source,
                        volume_source="VOLUME_UNAVAILABLE:DIRECT_INDEX",
                    )
                )
            if not records:
                return ProviderResult.failed(
                    self.source,
                    "Yahoo returned no usable Nasdaq-100 direct-index records",
                    volume_availability="not_available_for_direct_index",
                )
            return ProviderResult.success(
                self.source,
                records,
                volume_availability="not_available_for_direct_index",
            )
        except Exception as error:
            return ProviderResult.failed(
                self.source,
                f"Yahoo direct-index fetch failed: {error}",
                volume_availability="not_available_for_direct_index",
            )


class AkShareValuationProvider:
    """Direct-index valuation adapter using documented AkShare functions only.

    No ETF substitute is ever requested: if a direct source cannot supply a
    metric, the UI receives an explicit unavailable/cached response instead of
    an invented estimate.  Alpaca's Market Data API is intentionally not used
    here because it provides market prices, not index PE/PB valuation series.
    """

    source = "AKSHARE_VALUATION"

    def __init__(
        self,
        china_index_loader: Callable[..., Any] | None = None,
        china_spot_loader: Callable[..., Any] | None = None,
        us_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.china_index_loader = china_index_loader
        self.china_spot_loader = china_spot_loader
        self.us_loader = us_loader

    @staticmethod
    def _rows(frame: Any) -> list[dict[str, Any]]:
        if hasattr(frame, "to_dict"):
            return [dict(row) for row in frame.to_dict("records")]
        return [dict(row) for row in frame or []]

    @staticmethod
    def _value(row: dict[str, Any], *names: str) -> Any | None:
        normalized = {str(key).strip().lower().replace("_", "").replace("-", ""): value for key, value in row.items()}
        for name in names:
            value = normalized.get(name.strip().lower().replace("_", "").replace("-", ""))
            if value is not None and str(value).strip() not in {"", "-", "--", "nan", "None"}:
                return value
        return None

    @staticmethod
    def _decimal(value: Any | None) -> Decimal | None:
        if value is None:
            return None
        text = str(value).strip().replace(",", "").replace("%", "")
        if text in {"", "-", "--", "nan", "None"}:
            return None
        try:
            return Decimal(text)
        except ArithmeticError:
            return None

    @staticmethod
    def _date(value: Any | None) -> date | None:
        if isinstance(value, date):
            return value
        if value is None:
            return None
        try:
            return date.fromisoformat(str(value)[:10].replace("/", "-"))
        except ValueError:
            return None

    def _china_index(self) -> ValuationProviderResult:
        source = "AKSHARE_CSINDEX"
        try:
            loader = self.china_index_loader
            if loader is None:
                import akshare as ak

                loader = ak.stock_zh_index_value_csindex
            records: list[ValuationDataRecord] = []
            for row in self._rows(loader(symbol="000688")):
                observed = self._date(self._value(row, "日期", "date"))
                pe = self._decimal(self._value(row, "市盈率1", "市盈率2", "pe"))
                dividend_yield = self._decimal(self._value(row, "股息率1", "股息率2", "dividend_yield"))
                if observed is None or (pe is None and dividend_yield is None):
                    continue
                records.append(
                    ValuationDataRecord(
                        valuation_date=observed,
                        pe_ratio=pe,
                        dividend_yield=dividend_yield,
                        source=source,
                        source_url="https://www.csindex.com.cn/",
                        notes="中证指数估值数据；数据日期以源返回日期为准。",
                        raw_values={str(key): str(value) for key, value in row.items()},
                    )
                )
            if not records:
                return ValuationProviderResult.failed(source, "AkShare returned no usable 科创50 valuation records")
            return ValuationProviderResult.success(source, records)
        except Exception as error:
            return ValuationProviderResult.failed(source, f"AkShare 中证指数估值获取失败: {error}")

    def _china_spot(self) -> ValuationProviderResult:
        source = "AKSHARE_EASTMONEY_INDEX"
        try:
            loader = self.china_spot_loader
            if loader is None:
                import akshare as ak

                loader = ak.stock_zh_index_spot_em
            rows = self._rows(loader(symbol="深证系列指数"))
            row = next((item for item in rows if str(self._value(item, "代码", "code")) == "399006"), None)
            if row is None:
                return ValuationProviderResult.failed(source, "AkShare did not return 创业板指 spot valuation", status="unsupported")
            pe = self._decimal(self._value(row, "市盈率动态", "市盈率(动态)", "pe"))
            pb = self._decimal(self._value(row, "市净率", "pb"))
            if pe is None and pb is None:
                return ValuationProviderResult.failed(source, "创业板指 spot source has no PE/PB fields", status="unsupported")
            return ValuationProviderResult.success(
                source,
                [
                    ValuationDataRecord(
                        valuation_date=date.today(),
                        pe_ratio=pe,
                        pb_ratio=pb,
                        source=source,
                        source_url="https://quote.eastmoney.com/center/gridlist.html#index_sz",
                        notes="东方财富指数实时快照；只有快照日期，不应视作历史完整序列。",
                        raw_values={str(key): str(value) for key, value in row.items()},
                    )
                ],
            )
        except Exception as error:
            return ValuationProviderResult.failed(source, f"AkShare 创业板指估值获取失败: {error}")

    def _us_index(self) -> ValuationProviderResult:
        source = "AKSHARE_BAIDU_US"
        try:
            loader = self.us_loader
            if loader is None:
                import akshare as ak

                loader = ak.stock_us_valuation_baidu
            metric_rows: dict[date, dict[str, Any]] = {}
            for metric, target in (("市盈率(TTM)", "pe_ratio"), ("市净率", "pb_ratio")):
                for row in self._rows(loader(symbol="NDX", indicator=metric, period="近一年")):
                    observed = self._date(self._value(row, "日期", "date"))
                    value = self._decimal(self._value(row, "数值", "value", metric))
                    if observed is not None and value is not None:
                        metric_rows.setdefault(observed, {})[target] = value
                        metric_rows[observed].setdefault("raw", {}).update({str(key): str(item) for key, item in row.items()})
            records = [
                ValuationDataRecord(
                    valuation_date=observed,
                    pe_ratio=values.get("pe_ratio"),
                    pb_ratio=values.get("pb_ratio"),
                    source=source,
                    source_url="https://gushitong.baidu.com/",
                    notes="AkShare 百度股市通的 NDX 直接代码；不使用 QQQ 等 ETF 代理。",
                    raw_values=values.get("raw", {}),
                )
                for observed, values in sorted(metric_rows.items())
                if values.get("pe_ratio") is not None or values.get("pb_ratio") is not None
            ]
            if not records:
                return ValuationProviderResult.failed(source, "No direct NDX PE/PB data is available from the configured AkShare source", status="unsupported")
            return ValuationProviderResult.success(source, records)
        except Exception as error:
            return ValuationProviderResult.failed(source, f"AkShare NDX valuation 获取失败: {error}")

    def fetch(self, instrument_code: str) -> ValuationProviderResult:
        if instrument_code == "000688":
            return self._china_index()
        if instrument_code == "399006":
            return self._china_spot()
        if instrument_code == "NDX":
            return self._us_index()
        return ValuationProviderResult.failed(self.source, f"Unsupported direct index: {instrument_code}", status="unsupported")


class TushareProvider:
    """Optional adapter that stays explicitly inactive until a token is configured."""

    source = "TUSHARE"

    def __init__(self, token: str | None = None, client: Any | None = None) -> None:
        self.token = token if token is not None else os.getenv("TUSHARE_TOKEN")
        self.client = client

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        if not self.token:
            return ProviderResult.failed(
                self.source,
                "Tushare is inactive: TUSHARE_TOKEN is not configured.",
                status=ProviderStatus.INACTIVE,
            )
        try:
            client = self.client
            if client is None:
                import tushare as ts

                client = ts.pro_api(self.token)
            suffix = ".SH" if instrument_code.startswith(("5", "6")) else ".SZ"
            frame = client.fund_daily(
                ts_code=f"{instrument_code}{suffix}",
                start_date=start_date.strftime("%Y%m%d") if start_date else None,
                end_date=end_date.strftime("%Y%m%d") if end_date else None,
            )
            records = _to_records(frame, self.source, amount_multiplier=Decimal("1000"))
            if not records:
                return ProviderResult.failed(self.source, "Tushare returned no usable daily records")
            return ProviderResult.success(self.source, records)
        except Exception as error:
            return ProviderResult.failed(self.source, f"Tushare fetch failed: {error}")


class CsvProvider:
    """A local CSV provider for intentional user imports, preserving source=CSV."""

    source = "CSV"

    def __init__(self, contents: str | bytes | Path) -> None:
        self.contents = contents

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        try:
            parsed = parse_csv_market_data(self.contents)
        except (OSError, UnicodeError, ValueError) as error:
            return ProviderResult.failed(self.source, f"CSV cannot be read: {error}")
        warnings = [f"row {error.row_number}: {error.message}" for error in parsed.errors]
        if parsed.errors and not parsed.records:
            return ProviderResult.failed(
                self.source,
                "; ".join(error.message for error in parsed.errors),
            )
        records = [
            record
            for record in parsed.records
            if (start_date is None or record.trade_date >= start_date)
            and (end_date is None or record.trade_date <= end_date)
        ]
        if not records:
            return ProviderResult.failed(self.source, "CSV contains no records in the requested period")
        return ProviderResult.success(self.source, records, warnings=warnings)
