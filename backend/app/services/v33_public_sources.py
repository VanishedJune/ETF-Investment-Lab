"""Leakage-aware public auxiliary-data connectors for V3.3.

The connectors in this module deliberately sit outside the price-bar refresh
path.  They turn public, source-specific payloads into the append-only
``v33_point_in_time_observations`` contract without inventing neutral values.

No source in this file is described as official unless the upstream itself is
an official publisher.  EastMoney, HaoETF, Yahoo, LeGuLeGu and Jin10 are public
aggregators; AkShare is the adapter used to call several of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from html.parser import HTMLParser
import inspect
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .v33_data_service import PointInTimeObservationInput, V33DataService


SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

EASTMONEY_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
EASTMONEY_NAV_PAGE = "https://fundf10.eastmoney.com/jjjz_159941.html"
HAOETF_URL = "https://www.haoetf.com/qdii/159941"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/CNY=X"

# EastMoney silently returns an empty payload for oversized page sizes, hence
# the public endpoint's stable 20-row page.  The hard caps are anomaly guards,
# not historical-window limits: 20,000 daily observations cover far more than
# any plausible lifetime of this ETF, while 2,681 current rows require about
# 135 pages and therefore remain fully supported.
EASTMONEY_NAV_PAGE_SIZE = 20
EASTMONEY_NAV_MAX_RECORDS = 20_000
EASTMONEY_NAV_MAX_PAGES = 1_000

# These are storage codes.  The feature layer canonicalises the suffix after
# selecting the requested market; bare ``pe``/``nav`` codes must not be stored
# because they would collide across the two independent V3.3 targets.
MODEL_SERIES = (
    "pe",
    "pb",
    "earnings_growth",
    "short_rate",
    "long_rate",
    "real_rate",
    "term_spread",
    "fund_flow",
    "market_breadth",
    "liquidity",
    "inflation",
    "pmi",
    "employment",
    "volatility_index",
    "nav",
    "estimated_nav",
    "fx_usdcny",
    "fund_shares",
    "aum",
    "subscriptions",
)

EXPECTED_SERIES_BY_MARKET: Mapping[str, tuple[str, ...]] = {
    "399006": tuple(
        f"399006.{name}"
        for name in MODEL_SERIES
        if name not in {"nav", "estimated_nav", "fx_usdcny", "fund_shares", "aum", "subscriptions"}
    ),
    "159941": tuple(f"159941.{name}" for name in MODEL_SERIES),
}


JsonLoader = Callable[[str, Mapping[str, str], Mapping[str, str]], Any]
TextLoader = Callable[[str, Mapping[str, str]], Any]
RangeLoader = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class PublicSourceResult:
    """Outcome for one independently refreshable upstream family."""

    source: str
    status: str
    observations_received: int
    observations_persisted: int
    series_codes: tuple[str, ...]
    missing_series: tuple[str, ...]
    data_as_of: date | None
    error: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "observations_received": self.observations_received,
            "observations_persisted": self.observations_persisted,
            "series_codes": list(self.series_codes),
            "missing_series": list(self.missing_series),
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of else None,
            "error": self.error,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PublicSourcesRefreshResult:
    """Aggregate result that never hides partial upstream failure."""

    status: str
    started_at: datetime
    completed_at: datetime
    sources: tuple[PublicSourceResult, ...]
    available_series: tuple[str, ...]
    missing_by_market: Mapping[str, tuple[str, ...]]

    @property
    def observations_persisted(self) -> int:
        return sum(item.observations_persisted for item in self.sources)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "observations_persisted": self.observations_persisted,
            "available_series": list(self.available_series),
            "missing_by_market": {
                market: list(series) for market, series in self.missing_by_market.items()
            },
            "sources": [item.as_dict() for item in self.sources],
        }


class _TableParser(HTMLParser):
    """Small dependency-free HTML table reader used by the HaoETF adapter."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "table":
            self._table = []
        elif lowered == "tr" and self._table is not None:
            self._row = []
        elif lowered in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif lowered == "tr" and self._row is not None and self._table is not None:
            if any(cell.strip() for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif lowered == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalise_key(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).strip().lower())


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict("records")
        except TypeError:
            value = value.to_dict()
    if isinstance(value, Mapping):
        value = [value]
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _pick(row: Mapping[str, Any], *aliases: str) -> Any:
    normalised = {_normalise_key(key): value for key, value in row.items()}
    for alias in aliases:
        key = _normalise_key(alias)
        if key in normalised:
            value = normalised[key]
            if value is not None and str(value).strip().lower() not in {"", "nan", "nat", "none"}:
                return value
    return None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text.lower() in {"", "-", "--", "nan", "nat", "none", "null", "n/a"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        number = Decimal(text)
        if not number.is_finite():
            return None
        number = number.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)
        return -number if negative else number
    except (InvalidOperation, ValueError):
        return None


def _date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().date()
        except (AttributeError, TypeError, ValueError):
            pass
    text = str(value).strip()
    match = re.search(r"(\d{4})[-/年](\d{1,2})(?:[-/月](\d{1,2}))?", text)
    if match:
        year, month, raw_day = match.groups()
        try:
            return date(int(year), int(month), int(raw_day or 1))
        except ValueError:
            return None
    if re.fullmatch(r"\d{8}", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _month_end(value: Any) -> date | None:
    parsed = _date_value(value)
    if parsed is None:
        return None
    if parsed.month == 12:
        next_month = date(parsed.year + 1, 1, 1)
    else:
        next_month = date(parsed.year, parsed.month + 1, 1)
    return next_month - timedelta(days=1)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    text = str(value)
    return None if text.lower() in {"nan", "nat"} else text


def _in_range(value: date, start_date: date | None, end_date: date | None) -> bool:
    return (start_date is None or value >= start_date) and (
        end_date is None or value <= end_date
    )


def _at_local(value: date, clock: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(value, clock, tzinfo=zone).astimezone(UTC)


def _advance_weekdays(value: date, count: int) -> date:
    current = value
    remaining = count
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _next_month_first(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _scale_from_header(header: str, *, kind: str) -> Decimal:
    key = _normalise_key(header)
    if kind == "shares":
        if "亿份" in key:
            return Decimal("100000000")
        if "万份" in key:
            return Decimal("10000")
    if kind == "aum":
        if "亿元" in key:
            return Decimal("100000000")
        if "万元" in key:
            return Decimal("10000")
    return Decimal("1")


def _default_json_loader(
    url: str, params: Mapping[str, str], headers: Mapping[str, str]
) -> Any:
    request_url = f"{url}?{urlencode(params)}" if params else url
    request = Request(request_url, headers=dict(headers))
    with urlopen(request, timeout=25) as response:  # noqa: S310 - fixed public URLs
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def _default_text_loader(url: str, headers: Mapping[str, str]) -> str:
    request = Request(url, headers=dict(headers))
    with urlopen(request, timeout=25) as response:  # noqa: S310 - fixed public URLs
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _call_range_loader(
    loader: RangeLoader, start_date: date | None, end_date: date | None
) -> Any:
    """Support both simple zero-argument fixtures and range-aware loaders."""

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(start_date, end_date)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ) or len(positional) >= 2:
        return loader(start_date, end_date)
    if len(positional) == 1:
        return loader(start_date)
    return loader()


class V33PublicSourcesService:
    """Refresh public V3.3 auxiliary series without cross-source coupling."""

    def __init__(
        self,
        data_service: V33DataService,
        *,
        eastmoney_nav_loader: JsonLoader | None = None,
        haoetf_loader: TextLoader | None = None,
        yahoo_loader: JsonLoader | None = None,
        valuation_loaders: Mapping[str, RangeLoader] | None = None,
        macro_loaders: Mapping[str, RangeLoader] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.data_service = data_service
        self.eastmoney_nav_loader = eastmoney_nav_loader or _default_json_loader
        self.haoetf_loader = haoetf_loader or _default_text_loader
        self.yahoo_loader = yahoo_loader or _default_json_loader
        self.valuation_loaders = dict(valuation_loaders or self._default_valuation_loaders())
        self.macro_loaders = dict(macro_loaders or self._default_macro_loaders())
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    @staticmethod
    def _default_valuation_loaders() -> Mapping[str, RangeLoader]:
        def pe_loader(*_args: object) -> Any:
            import akshare as ak

            return ak.stock_market_pe_lg(symbol="创业板")

        def pb_loader(*_args: object) -> Any:
            import akshare as ak

            return ak.stock_market_pb_lg(symbol="创业板")

        return {"cyb_pe": pe_loader, "cyb_pb": pb_loader}

    @staticmethod
    def _default_macro_loaders() -> Mapping[str, RangeLoader]:
        def ak_call(name: str) -> RangeLoader:
            def load(*_args: object) -> Any:
                import akshare as ak

                return getattr(ak, name)()

            return load

        def rates(start_date: date | None = None, _end_date: date | None = None) -> Any:
            import akshare as ak

            return ak.bond_zh_us_rate(
                start_date=(start_date or date(2010, 1, 1)).strftime("%Y%m%d")
            )

        return {
            "china_lpr": ak_call("macro_china_lpr"),
            "china_pmi": ak_call("macro_china_pmi"),
            "china_m2": ak_call("macro_china_money_supply"),
            "china_cpi": ak_call("macro_china_cpi"),
            "us_cpi": ak_call("macro_usa_cpi_monthly"),
            "us_unemployment": ak_call("macro_usa_unemployment_rate"),
            "us_pmi": ak_call("macro_usa_pmi"),
            "us_rate": ak_call("macro_bank_usa_interest_rate"),
            "bond_rates": rates,
        }

    def _now(self) -> datetime:
        return _utc(self.now_provider())

    @staticmethod
    def _observation(
        *,
        market: str,
        series: str,
        dataset: str,
        observation_date: date,
        value: Decimal,
        published_at: datetime,
        available_at: datetime,
        vintage: str,
        source: str,
        unit: str,
        source_url: str,
        observation_period: str,
        payload: Mapping[str, Any],
    ) -> PointInTimeObservationInput:
        if series not in MODEL_SERIES:
            raise ValueError(f"Unsupported V3.3 series suffix: {series}")
        return PointInTimeObservationInput(
            dataset=dataset,
            series_code=f"{market}.{series}",
            observation_date=observation_date,
            effective_date=observation_date,
            published_at=_utc(published_at),
            available_at=_utc(available_at),
            cutoff_at=_utc(available_at),
            vintage=vintage[:64],
            source=source[:64],
            numeric_value=value,
            unit=unit,
            observation_period=observation_period,
            source_url=source_url,
            instrument_code=market,
            payload=_json_safe(dict(payload)),
        )

    def _failed(
        self,
        source: str,
        expected: Iterable[str],
        error: BaseException | str,
        *,
        warnings: Iterable[str] = (),
    ) -> PublicSourceResult:
        return PublicSourceResult(
            source=source,
            status="failed",
            observations_received=0,
            observations_persisted=0,
            series_codes=(),
            missing_series=tuple(sorted(set(expected))),
            data_as_of=None,
            error=str(error),
            warnings=tuple(warnings),
        )

    def _persist(
        self,
        source: str,
        expected: Iterable[str],
        observations: Sequence[PointInTimeObservationInput],
        *,
        warnings: Iterable[str] = (),
        errors: Iterable[str] = (),
    ) -> PublicSourceResult:
        expected_set = set(expected)
        warning_items = tuple(dict.fromkeys(str(item) for item in warnings if item))
        error_items = tuple(dict.fromkeys(str(item) for item in errors if item))
        if not observations:
            return self._failed(
                source,
                expected_set,
                "; ".join(error_items) or "Source returned no usable real observations",
                warnings=warning_items,
            )
        try:
            row_ids = self.data_service.ingest_point_in_time_observations(observations)
        except Exception as error:
            return self._failed(
                source,
                expected_set,
                f"Persistence failed: {error}",
                warnings=warning_items,
            )
        series = tuple(sorted({item.series_code for item in observations}))
        missing = tuple(sorted(expected_set - set(series)))
        status = "partial" if missing or error_items else "success"
        return PublicSourceResult(
            source=source,
            status=status,
            observations_received=len(observations),
            observations_persisted=len(row_ids),
            series_codes=series,
            missing_series=missing,
            data_as_of=max(item.observation_date for item in observations),
            error="; ".join(error_items) or None,
            warnings=warning_items,
        )

    def refresh_eastmoney_nav(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourceResult:
        source = "EASTMONEY_159941_NAV"
        expected = ("159941.nav",)
        rows: list[dict[str, Any]] = []
        try:
            page = 1
            # The endpoint silently returns ``Data: null`` for oversized page
            # requests.  Twenty is its stable public page size; pagination is
            # therefore driven by ``TotalCount`` instead of requesting 500.
            page_size = EASTMONEY_NAV_PAGE_SIZE
            total_count: int | None = None
            expected_pages: int | None = None
            while True:
                payload = self.eastmoney_nav_loader(
                    EASTMONEY_NAV_URL,
                    {
                        "fundCode": "159941",
                        "pageIndex": str(page),
                        "pageSize": str(page_size),
                        "startDate": start_date.isoformat() if start_date else "",
                        "endDate": end_date.isoformat() if end_date else "",
                    },
                    {
                        "Accept": "application/json,text/plain,*/*",
                        "Referer": EASTMONEY_NAV_PAGE,
                        "User-Agent": "Mozilla/5.0 InvestmentLab/3.3",
                    },
                )
                if isinstance(payload, (bytes, bytearray)):
                    payload = payload.decode("utf-8")
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if not isinstance(payload, Mapping):
                    raise ValueError("EastMoney NAV response is not a JSON object")
                if payload.get("ErrCode") not in {None, 0, "0"}:
                    raise ValueError(
                        f"EastMoney NAV error {payload.get('ErrCode')}: {payload.get('ErrMsg')}"
                    )
                data = payload.get("Data")
                batch = data.get("LSJZList") if isinstance(data, Mapping) else None
                if batch is None:
                    raise ValueError("EastMoney NAV response is missing Data.LSJZList")
                parsed_batch = _records(batch)
                raw_total = payload.get("TotalCount")
                if raw_total is not None:
                    response_total = int(raw_total)
                    if response_total < 0:
                        raise ValueError("EastMoney NAV TotalCount cannot be negative")
                    if response_total > EASTMONEY_NAV_MAX_RECORDS:
                        raise ValueError(
                            "EastMoney NAV TotalCount exceeds the 20000-record safety cap"
                        )
                    if total_count is None:
                        total_count = response_total
                        expected_pages = max(
                            1, (total_count + page_size - 1) // page_size
                        )
                        if expected_pages > EASTMONEY_NAV_MAX_PAGES:
                            raise ValueError(
                                "EastMoney NAV pagination exceeds the 1000-page safety cap"
                            )
                    elif response_total != total_count:
                        raise ValueError(
                            "EastMoney NAV TotalCount changed during pagination; retry required"
                        )

                if not parsed_batch:
                    if total_count is not None and len(rows) < total_count:
                        raise ValueError(
                            "EastMoney NAV pagination ended before TotalCount was collected"
                        )
                    break
                rows.extend(parsed_batch)
                if total_count is not None and len(rows) >= total_count:
                    break
                if expected_pages is not None and page >= expected_pages:
                    raise ValueError(
                        "EastMoney NAV pagination did not collect the declared TotalCount"
                    )
                if len(parsed_batch) < page_size:
                    if total_count is not None and len(rows) < total_count:
                        raise ValueError(
                            "EastMoney NAV returned a short page before declared TotalCount"
                        )
                    break
                page += 1
                if page > EASTMONEY_NAV_MAX_PAGES:
                    raise ValueError(
                        "EastMoney NAV pagination exceeds the 1000-page safety cap"
                    )
        except Exception as error:
            return self._failed(source, expected, f"EastMoney NAV fetch failed: {error}")

        observations: list[PointInTimeObservationInput] = []
        warnings: list[str] = []
        for row in rows:
            nav_date = _date_value(row.get("FSRQ"))
            nav = _decimal(row.get("DWJZ"))
            if nav_date is None or nav is None or nav <= 0:
                warnings.append("Skipped EastMoney NAV row with invalid FSRQ or DWJZ")
                continue
            if not _in_range(nav_date, start_date, end_date):
                continue
            # QDII NAV publication timing is not included in this endpoint.
            # T+2 exchange weekdays at 23:59 Shanghai is a deliberately late,
            # documented estimate; it is not represented as an exact release.
            release_day = _advance_weekdays(nav_date, 2)
            available_at = _at_local(release_day, time(23, 59), SHANGHAI)
            observations.append(
                self._observation(
                    market="159941",
                    series="nav",
                    dataset="QDII_NAV",
                    observation_date=nav_date,
                    value=nav,
                    published_at=available_at,
                    available_at=available_at,
                    vintage="historical-t-plus-2-policy-v1",
                    source=source,
                    unit="CNY_per_share",
                    source_url=EASTMONEY_NAV_URL,
                    observation_period="daily",
                    payload={
                        "upstream_record": row,
                        "upstream_field_count": len(row),
                        "cumulative_nav": row.get("LJJZ"),
                        "daily_growth_pct": row.get("JZZZL"),
                        "availability_policy": "CONSERVATIVE_T_PLUS_2_WEEKDAYS_2359_ASIA_SHANGHAI",
                        "publication_time_exact": False,
                        "not_imputed": True,
                    },
                )
            )
        return self._persist(source, expected, observations, warnings=warnings)

    @staticmethod
    def _haoetf_history_rows(html: str) -> tuple[list[dict[str, str]], list[str]]:
        parser = _TableParser()
        parser.feed(html)
        warnings: list[str] = []
        best: tuple[int, list[list[str]]] | None = None
        for table in parser.tables:
            if len(table) < 2:
                continue
            headers = [_normalise_key(item) for item in table[0]]
            score = sum(
                any(marker in header for header in headers)
                for marker in ("日期", "估值", "溢价", "份额", "指数涨跌")
            )
            if any("t1日估值" in header for header in headers):
                score += 5
            if best is None or score > best[0]:
                best = (score, table)
        if best is None or best[0] < 4:
            return [], ["HaoETF HTML contains no recognised history table"]
        table = best[1]
        headers = table[0]
        output: list[dict[str, str]] = []
        for cells in table[1:]:
            if len(cells) != len(headers):
                warnings.append("Skipped HaoETF history row with mismatched column count")
                continue
            output.append(dict(zip(headers, cells)))
        return output, warnings

    def refresh_haoetf(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourceResult:
        source = "HAOETF_159941_HISTORY"
        expected = (
            "159941.estimated_nav",
            "159941.fund_shares",
            "159941.aum",
            "159941.subscriptions",
        )
        retrieved_at = self._now()
        try:
            raw = self.haoetf_loader(
                HAOETF_URL,
                {
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": "Mozilla/5.0 InvestmentLab/3.3",
                },
            )
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            if not isinstance(raw, str):
                raise ValueError("HaoETF loader did not return HTML text")
            rows, warnings = self._haoetf_history_rows(raw)
        except Exception as error:
            return self._failed(source, expected, f"HaoETF fetch failed: {error}")

        parsed_rows: list[dict[str, Any]] = []
        for row in rows:
            row_date = _date_value(_pick(row, "日期", "交易日期", "date"))
            if row_date is None:
                warnings.append("Skipped HaoETF row without a full history date")
                continue
            estimate = _decimal(
                _pick(row, "T-1日估值", "T-1估值", "最新估值", "估算净值", "estimated_nav")
            )
            premium = _decimal(
                _pick(row, "T-1日溢价率", "T-1溢价率", "溢价率", "溢价")
            )
            index_change = _decimal(
                _pick(row, "T-1日指数涨跌", "指数涨跌", "标的指数涨跌")
            )
            share_header = next(
                (header for header in row if "份额" in _normalise_key(header) and "涨幅" not in _normalise_key(header) and "新增" not in _normalise_key(header) and "变化" not in _normalise_key(header)),
                "",
            )
            shares = _decimal(row.get(share_header)) if share_header else None
            if shares is not None:
                shares = shares * _scale_from_header(share_header, kind="shares")
            change_header = next(
                (header for header in row if any(marker in _normalise_key(header) for marker in ("新增份额", "份额变化", "份额变动"))),
                "",
            )
            share_change = _decimal(row.get(change_header)) if change_header else None
            if share_change is not None:
                share_change = share_change * _scale_from_header(change_header, kind="shares")
            aum_header = next(
                (header for header in row if any(marker in _normalise_key(header) for marker in ("资产规模", "基金规模", "资产净值", "aum"))),
                "",
            )
            aum = _decimal(row.get(aum_header)) if aum_header else None
            if aum is not None:
                aum = aum * _scale_from_header(aum_header, kind="aum")
            parsed_rows.append(
                {
                    "date": row_date,
                    "estimate": estimate,
                    "premium": premium,
                    "index_change": index_change,
                    "shares": shares,
                    "share_change": share_change,
                    "aum": aum,
                    "raw": row,
                }
            )

        # The historical table generally supplies a percentage share change,
        # not an absolute subscription count.  Exact adjacent published share
        # totals let us derive the latter without using a default value.
        previous_shares: Decimal | None = None
        for row in sorted(parsed_rows, key=lambda item: item["date"]):
            if row["share_change"] is None and row["shares"] is not None and previous_shares is not None:
                row["share_change"] = row["shares"] - previous_shares
                row["share_change_derived"] = True
            if row["shares"] is not None:
                previous_shares = row["shares"]

        observations: list[PointInTimeObservationInput] = []
        vintage = f"retrieved-{retrieved_at.strftime('%Y%m%dT%H%M%SZ')}"
        for row in parsed_rows:
            row_date = row["date"]
            if not _in_range(row_date, start_date, end_date):
                continue
            common_payload = {
                "upstream_record": row["raw"],
                "premium_pct": row["premium"],
                "underlying_index_change_pct": row["index_change"],
                "availability_policy": "HISTORICAL_BACKFILL_AVAILABLE_AT_RETRIEVAL",
                "publication_time_exact": False,
                "not_imputed": True,
            }
            values = (
                ("estimated_nav", row["estimate"], "CNY_per_share"),
                ("fund_shares", row["shares"], "shares"),
                ("aum", row["aum"], "CNY"),
                ("subscriptions", row["share_change"], "shares"),
            )
            for series, value, unit in values:
                if value is None or (series in {"estimated_nav", "fund_shares", "aum"} and value <= 0):
                    continue
                observations.append(
                    self._observation(
                        market="159941",
                        series=series,
                        dataset="QDII_ESTIMATE",
                        observation_date=row_date,
                        value=value,
                        published_at=retrieved_at,
                        available_at=retrieved_at,
                        vintage=vintage,
                        source=source,
                        unit=unit,
                        source_url=HAOETF_URL,
                        observation_period="daily",
                        payload={
                            **common_payload,
                            "share_change_derived_from_adjacent_totals": bool(
                                row.get("share_change_derived", False)
                            ),
                        },
                    )
                )
        return self._persist(source, expected, observations, warnings=warnings)

    def refresh_yahoo_fx(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourceResult:
        source = "YAHOO_CHART_USDCNY"
        expected = ("159941.fx_usdcny",)
        request_start = start_date or date(2010, 1, 1)
        request_end = end_date or self._now().date()
        try:
            payload = self.yahoo_loader(
                YAHOO_CHART_URL,
                {
                    "period1": str(
                        int(datetime.combine(request_start, time.min, tzinfo=UTC).timestamp())
                    ),
                    "period2": str(
                        int(
                            datetime.combine(
                                request_end + timedelta(days=1), time.min, tzinfo=UTC
                            ).timestamp()
                        )
                    ),
                    "interval": "1d",
                    "events": "history",
                },
                {"Accept": "application/json", "User-Agent": "Mozilla/5.0 InvestmentLab/3.3"},
            )
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)
            chart = payload.get("chart") if isinstance(payload, Mapping) else None
            if not isinstance(chart, Mapping):
                raise ValueError("Yahoo response is missing chart")
            if chart.get("error"):
                raise ValueError(f"Yahoo chart error: {chart.get('error')}")
            results = chart.get("result") or []
            result = results[0] if results else None
            if not isinstance(result, Mapping):
                raise ValueError("Yahoo response contains no result")
            metadata = result.get("meta") if isinstance(result.get("meta"), Mapping) else {}
            zone_name = str(metadata.get("exchangeTimezoneName") or "UTC")
            try:
                source_zone = ZoneInfo(zone_name)
            except Exception:
                source_zone = UTC
            timestamps = list(result.get("timestamp") or [])
            indicators = result.get("indicators") or {}
            quotes = indicators.get("quote") or [] if isinstance(indicators, Mapping) else []
            quote = quotes[0] if quotes and isinstance(quotes[0], Mapping) else {}
            closes = list(quote.get("close") or [])
        except Exception as error:
            return self._failed(source, expected, f"Yahoo CNY=X fetch failed: {error}")

        observations: list[PointInTimeObservationInput] = []
        warnings: list[str] = []
        for index, timestamp in enumerate(timestamps):
            if index >= len(closes):
                warnings.append("Yahoo timestamp has no matching close")
                continue
            try:
                source_timestamp = datetime.fromtimestamp(int(timestamp), UTC).astimezone(
                    source_zone
                )
                observed = source_timestamp.date()
            except (OSError, OverflowError, TypeError, ValueError):
                warnings.append("Skipped Yahoo row with invalid timestamp")
                continue
            # Completed CNY=X daily bars are stamped at source-local midnight.
            # Yahoo may append a live ``regularMarketTime`` point with the
            # current clock time (and sometimes a Saturday date); that value
            # is not a completed historical close and must not enter training.
            if source_timestamp.time() != time.min or observed.weekday() >= 5:
                warnings.append(
                    f"Skipped Yahoo incomplete/non-session point: {source_timestamp.isoformat()}"
                )
                continue
            value = _decimal(closes[index])
            if value is None or value <= 0 or not _in_range(observed, start_date, end_date):
                continue
            next_day = observed + timedelta(days=1)
            available_at = _at_local(next_day, time(0, 5), source_zone)
            observations.append(
                self._observation(
                    market="159941",
                    series="fx_usdcny",
                    dataset="FX",
                    observation_date=observed,
                    value=value,
                    published_at=available_at,
                    available_at=available_at,
                    vintage="yahoo-daily-close-policy-v1",
                    source=source,
                    unit="CNY_per_USD",
                    source_url=YAHOO_CHART_URL,
                    observation_period="daily",
                    payload={
                        "symbol_requested": "CNY=X",
                        "symbol_returned": metadata.get("symbol"),
                        "exchange_timezone": zone_name,
                        "timestamp": timestamp,
                        "availability_policy": "NEXT_SOURCE_LOCAL_DAY_0005",
                        "publication_time_exact": False,
                        "not_imputed": True,
                    },
                )
            )
        return self._persist(source, expected, observations, warnings=warnings)

    def refresh_akshare_valuation(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourceResult:
        source = "AKSHARE_LEGULEGU_CYB_VALUATION"
        expected = ("399006.pe", "399006.pb", "399006.earnings_growth")
        observations: list[PointInTimeObservationInput] = []
        errors: list[str] = []
        warnings = [
            "399006.earnings_growth: no verified reported earnings/EPS source "
            "is configured; left missing by design"
        ]
        specifications = {
            "cyb_pe": ("pe", ("平均市盈率", "市盈率", "pe", "pe_ttm"), "https://legulegu.com/stockdata/cybPE"),
            "cyb_pb": ("pb", ("市净率", "等权市净率", "pb"), "https://legulegu.com/stockdata/cybPB"),
        }
        for loader_name, (series, aliases, source_url) in specifications.items():
            loader = self.valuation_loaders.get(loader_name)
            if loader is None:
                errors.append(f"{loader_name}: loader is not configured")
                continue
            try:
                rows = _records(_call_range_loader(loader, start_date, end_date))
            except Exception as error:
                errors.append(f"{loader_name}: {error}")
                continue
            for row in rows:
                observed = _date_value(_pick(row, "日期", "date", "trade_date"))
                value = _decimal(_pick(row, *aliases))
                if observed is None or value is None or value <= 0:
                    continue
                if not _in_range(observed, start_date, end_date):
                    continue
                available_at = _at_local(observed, time(18, 0), SHANGHAI)
                observations.append(
                    self._observation(
                        market="399006",
                        series=series,
                        dataset="VALUATION",
                        observation_date=observed,
                        value=value,
                        published_at=available_at,
                        available_at=available_at,
                        vintage="akshare-legulegu-close-policy-v1",
                        source=f"{source}_{series.upper()}",
                        unit="ratio",
                        source_url=source_url,
                        observation_period="daily",
                        payload={
                            "upstream_record": row,
                            "upstream_provider": "LeGuLeGu via AkShare",
                            "availability_policy": "SAME_DAY_1800_ASIA_SHANGHAI",
                            "publication_time_exact": False,
                            "not_imputed": True,
                        },
                    )
                )

        return self._persist(
            source,
            expected,
            observations,
            warnings=warnings,
            errors=errors,
        )

    @staticmethod
    def _macro_available_at(policy: str, observed: date) -> datetime:
        if policy == "china_lpr":
            return _at_local(observed, time(12, 0), SHANGHAI)
        if policy == "china_pmi":
            return _at_local(_next_month_first(observed) + timedelta(days=2), time(12, 0), SHANGHAI)
        if policy == "china_cpi":
            return _at_local(_next_month_first(observed) + timedelta(days=14), time(12, 0), SHANGHAI)
        if policy == "china_m2":
            return _at_local(_next_month_first(observed) + timedelta(days=19), time(12, 0), SHANGHAI)
        if policy == "daily_aggregator":
            return datetime.combine(observed + timedelta(days=1), time(0, 0), tzinfo=UTC)
        if policy == "us_release_date":
            return _at_local(observed + timedelta(days=1), time(0, 5), NEW_YORK)
        raise ValueError(f"Unknown macro availability policy: {policy}")

    def refresh_akshare_macro(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourceResult:
        source = "AKSHARE_PUBLIC_MACRO"
        expected = (
            "399006.short_rate",
            "399006.long_rate",
            "399006.term_spread",
            "399006.liquidity",
            "399006.inflation",
            "399006.pmi",
            "159941.short_rate",
            "159941.long_rate",
            "159941.term_spread",
            "159941.inflation",
            "159941.pmi",
            "159941.employment",
        )
        # loader, market, series, aliases, date aliases, policy, unit, URL,
        # and whether a month label represents the observation period.
        specs: Mapping[str, tuple[tuple[Any, ...], ...]] = {
            "china_lpr": (("399006", "short_rate", ("LPR1Y", "1年LPR", "lpr1y"), ("TRADE_DATE", "日期"), "china_lpr", "percent", "https://data.eastmoney.com/cjsj/globalRateLPR.html", False),),
            "china_pmi": (("399006", "pmi", ("制造业-指数", "制造业指数", "今值"), ("月份", "日期"), "china_pmi", "index", "https://data.eastmoney.com/cjsj/pmi.html", True),),
            "china_m2": (("399006", "liquidity", ("货币和准货币(M2)-同比增长", "M2同比增长", "今值"), ("月份", "日期"), "china_m2", "percent_yoy", "https://data.eastmoney.com/cjsj/hbgyl.html", True),),
            "china_cpi": (("399006", "inflation", ("全国-同比增长", "CPI同比增长", "今值"), ("月份", "日期"), "china_cpi", "percent_yoy", "https://data.eastmoney.com/cjsj/cpi.html", True),),
            "us_cpi": (("159941", "inflation", ("今值", "actual", "value"), ("日期", "date"), "us_release_date", "percent_mom", "https://datacenter.jin10.com/reportType/dc_usa_cpi", False),),
            "us_unemployment": (("159941", "employment", ("今值", "actual", "value"), ("日期", "date"), "us_release_date", "unemployment_percent", "https://datacenter.jin10.com/reportType/dc_usa_unemployment_rate", False),),
            "us_pmi": (("159941", "pmi", ("今值", "actual", "value"), ("日期", "date"), "us_release_date", "index", "https://datacenter.jin10.com/reportType/dc_usa_pmi", False),),
            "us_rate": (("159941", "short_rate", ("今值", "actual", "value"), ("日期", "date"), "us_release_date", "percent", "https://datacenter.jin10.com/reportType/dc_usa_interest_rate_decision", False),),
            "bond_rates": (
                ("399006", "long_rate", ("中国国债收益率10年",), ("日期", "date"), "daily_aggregator", "percent", "https://data.eastmoney.com/cjsj/zmgzsyl.html", False),
                ("399006", "term_spread", ("中国国债收益率10年-2年",), ("日期", "date"), "daily_aggregator", "percentage_points", "https://data.eastmoney.com/cjsj/zmgzsyl.html", False),
                ("159941", "long_rate", ("美国国债收益率10年",), ("日期", "date"), "daily_aggregator", "percent", "https://data.eastmoney.com/cjsj/zmgzsyl.html", False),
                ("159941", "term_spread", ("美国国债收益率10年-2年",), ("日期", "date"), "daily_aggregator", "percentage_points", "https://data.eastmoney.com/cjsj/zmgzsyl.html", False),
            ),
        }
        observations: list[PointInTimeObservationInput] = []
        errors: list[str] = []
        warnings: list[str] = []
        for loader_name, configurations in specs.items():
            loader = self.macro_loaders.get(loader_name)
            if loader is None:
                errors.append(f"{loader_name}: loader is not configured")
                continue
            try:
                rows = _records(_call_range_loader(loader, start_date, end_date))
            except Exception as error:
                errors.append(f"{loader_name}: {error}")
                continue
            for configuration in configurations:
                market, series, aliases, date_aliases, policy, unit, source_url, is_monthly = configuration
                produced = 0
                for row in rows:
                    raw_date = _pick(row, *date_aliases)
                    observed = _month_end(raw_date) if is_monthly else _date_value(raw_date)
                    value = _decimal(_pick(row, *aliases))
                    if observed is None or value is None:
                        continue
                    if not _in_range(observed, start_date, end_date):
                        continue
                    available_at = self._macro_available_at(policy, observed)
                    observations.append(
                        self._observation(
                            market=market,
                            series=series,
                            dataset="MACRO",
                            observation_date=observed,
                            value=value,
                            published_at=available_at,
                            available_at=available_at,
                            vintage=f"{loader_name}-availability-policy-v1",
                            source=f"AKSHARE_{loader_name.upper()}",
                            unit=unit,
                            source_url=source_url,
                            observation_period="monthly" if is_monthly else "daily",
                            payload={
                                "upstream_record": row,
                                "availability_policy": policy.upper(),
                                "publication_time_exact": False,
                                "not_imputed": True,
                            },
                        )
                    )
                    produced += 1
                if produced == 0:
                    warnings.append(f"{loader_name}:{market}.{series} returned no in-range usable values")
        return self._persist(
            source,
            expected,
            observations,
            warnings=warnings,
            errors=errors,
        )

    def refresh_all(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> PublicSourcesRefreshResult:
        """Refresh every source independently and report every remaining gap."""

        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        started_at = self._now()
        methods = (
            ("EASTMONEY_159941_NAV", self.refresh_eastmoney_nav, ("159941.nav",)),
            (
                "HAOETF_159941_HISTORY",
                self.refresh_haoetf,
                ("159941.estimated_nav", "159941.fund_shares", "159941.aum", "159941.subscriptions"),
            ),
            ("YAHOO_CHART_USDCNY", self.refresh_yahoo_fx, ("159941.fx_usdcny",)),
            (
                "AKSHARE_LEGULEGU_CYB_VALUATION",
                self.refresh_akshare_valuation,
                ("399006.pe", "399006.pb", "399006.earnings_growth"),
            ),
            ("AKSHARE_PUBLIC_MACRO", self.refresh_akshare_macro, ()),
        )
        results: list[PublicSourceResult] = []
        for name, method, expected in methods:
            try:
                results.append(method(start_date, end_date))
            except Exception as error:  # keep later independent sources alive
                results.append(self._failed(name, expected, f"Unexpected refresh failure: {error}"))

        available = tuple(
            sorted({series for result in results for series in result.series_codes})
        )
        available_set = set(available)
        missing_by_market = {
            market: tuple(sorted(set(expected) - available_set))
            for market, expected in EXPECTED_SERIES_BY_MARKET.items()
        }
        failed = any(result.status == "failed" for result in results)
        partial = any(result.status == "partial" for result in results)
        has_missing = any(missing_by_market.values())
        status = "failed" if all(result.status == "failed" for result in results) else (
            "partial" if failed or partial or has_missing else "success"
        )
        return PublicSourcesRefreshResult(
            status=status,
            started_at=started_at,
            completed_at=self._now(),
            sources=tuple(results),
            available_series=available,
            missing_by_market=missing_by_market,
        )


__all__ = [
    "EXPECTED_SERIES_BY_MARKET",
    "MODEL_SERIES",
    "PublicSourceResult",
    "PublicSourcesRefreshResult",
    "V33PublicSourcesService",
]
