"""Idempotent initialization of the local SQLite schema and seed records."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ..config import DEFAULT_FEE_CONFIG, DEFAULT_STRATEGY_CONFIG, ensure_default_configs
from ..core.paths import database_path as default_database_path
from ..models.models import (
    AppSetting,
    Base,
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccount,
    SimulationAccount,
    StrategyDefinition,
    V33InstrumentRole,
    utc_now,
)
from ..services.instrument_universe import DISPLAY_ONLY_ETFS
from .migrations import run_migrations
from .session import create_database_engine


DEFAULT_INSTRUMENTS = (
    {"code": "589850", "name": "科创50ETF东财", "exchange": "SSE"},
    {"code": "159205", "name": "创业板ETF东财", "exchange": "SZSE"},
    {"code": "159941", "name": "纳指ETF广发", "exchange": "SZSE"},
)

DEFAULT_INDEX_INSTRUMENTS = (
    {
        "code": "000688",
        "name": "科创50指数",
        "exchange": "SSE",
        "description": "直接指数行情；不使用 ETF 行情",
    },
    {
        "code": "399006",
        "name": "创业板指数",
        "exchange": "SZSE",
        "description": "直接指数行情；不使用 ETF 行情",
    },
    {
        "code": "NDX",
        "name": "纳斯达克100指数",
        "exchange": "NASDAQ",
        "description": "直接指数行情；不使用 ETF 行情",
    },
)

DEFAULT_SIMULATION_ACCOUNT_NAMES = (
    "固定定投",
    "估值定投",
    "趋势定投",
    "估值加趋势综合",
)

STALE_LOCK_AGE_SECONDS = 300.0


class DatabaseInitializationLock:
    """Cross-process lock protecting schema migration and initial seed writes."""

    def __init__(self, database: Path, timeout_seconds: float) -> None:
        self.path = database.parent / ".investment_lab.initialize.lock"
        self.recovery_path = database.parent / ".investment_lab.initialize.lock.recovery"
        self.timeout_seconds = timeout_seconds
        self._descriptor: int | None = None

    def __enter__(self) -> DatabaseInitializationLock:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                metadata = json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8")
                os.write(self._descriptor, metadata)
                return self
            except FileExistsError:
                self._reclaim_stale_lock_if_safe()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for database initialization lock: {self.path}"
                    )
                time.sleep(0.05)
            except PermissionError:
                # On Windows another process can transiently surface an
                # existing exclusive lock file as access denied rather than
                # FileExistsError.  Treat that exact case as normal
                # contention; a permission error without an existing lock is
                # still a real filesystem failure and must propagate.
                if os.name != "nt" or not self.path.exists():
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for database initialization lock: {self.path}"
                    )
                time.sleep(0.05)

    @staticmethod
    def _process_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                # Access denied still proves that a process owns the PID. Invalid
                # parameter is the normal Windows response for a missing PID.
                return ctypes.get_last_error() != 87
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as error:
            return getattr(error, "winerror", None) != 87
        return True

    def _unknown_lock_is_old_enough(self) -> bool:
        try:
            return time.time() - self.path.stat().st_mtime >= STALE_LOCK_AGE_SECONDS
        except FileNotFoundError:
            return False

    def _is_reclaimable(self) -> bool:
        try:
            metadata = json.loads(self.path.read_text(encoding="utf-8"))
            pid = metadata.get("pid")
            if isinstance(pid, int):
                return not self._process_is_running(pid)
        except FileNotFoundError:
            return False
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._unknown_lock_is_old_enough()
        return self._unknown_lock_is_old_enough()

    def _acquire_recovery_guard(self) -> int | None:
        try:
            return os.open(self.recovery_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - self.recovery_path.stat().st_mtime >= STALE_LOCK_AGE_SECONDS:
                    self.recovery_path.unlink()
            except FileNotFoundError:
                pass
            return None
        except PermissionError:
            if os.name == "nt" and self.recovery_path.exists():
                return None
            raise

    def _reclaim_stale_lock_if_safe(self) -> None:
        if not self._is_reclaimable():
            return
        recovery_descriptor = self._acquire_recovery_guard()
        if recovery_descriptor is None:
            return
        try:
            try:
                snapshot = self.path.read_bytes()
            except FileNotFoundError:
                return
            if not self._is_reclaimable():
                return
            try:
                if self.path.read_bytes() != snapshot:
                    return
                self.path.unlink()
            except FileNotFoundError:
                return
        finally:
            os.close(recovery_descriptor)
            try:
                self.recovery_path.unlink()
            except FileNotFoundError:
                pass

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        if self._descriptor is None:
            return
        try:
            os.close(self._descriptor)
        finally:
            self._descriptor = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _seed_if_missing(session: Session) -> None:
    instruments: dict[str, Instrument] = {}
    for defaults in DEFAULT_INSTRUMENTS:
        instrument = session.scalar(select(Instrument).where(Instrument.code == defaults["code"]))
        if instrument is None:
            instrument = Instrument(
                **defaults,
                category="etf",
                currency="CNY",
                is_active=defaults["code"] == "159941",
            )
            session.add(instrument)
            session.flush()
        if defaults["code"] == "159941":
            instrument.is_active = True
            instrument.category = "etf"
            instrument.currency = "CNY"
            instrument.exchange = "SZSE"
            instrument.extra_data = {
                **dict(instrument.extra_data or {}),
                "research_kind": "tradable_qdii_etf",
                "model_market": "159941",
                "benchmark": "NDX",
                "preferred_market_data_sources": ["AKSHARE", "TUSHARE"],
            }
        instruments[instrument.code] = instrument

    for defaults in DEFAULT_INDEX_INSTRUMENTS:
        instrument = session.scalar(select(Instrument).where(Instrument.code == defaults["code"]))
        if instrument is None:
            instrument = Instrument(
                **defaults,
                category="index",
                currency="USD" if defaults["code"] == "NDX" else "CNY",
                is_active=True,
                extra_data={"research_kind": "index", "data_source": "AKSHARE_INDEX"},
            )
            session.add(instrument)
            session.flush()
        if defaults["code"] == "NDX":
            instrument.extra_data = {
                **dict(instrument.extra_data or {}),
                "research_kind": "benchmark_index",
                "model_market": "159941",
                "role": "benchmark",
                "tradable": False,
            }

    # Chart-only ETFs are seeded outside DEFAULT_INSTRUMENTS so they never
    # inherit the automatic investment-plan loop below.
    for code, defaults in DISPLAY_ONLY_ETFS.items():
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is None:
            instrument = Instrument(
                code=code,
                name=defaults["name"],
                exchange=defaults["exchange"],
                category="etf",
                currency="CNY",
                is_active=True,
                description="行情图表展示专用；不参与训练、概率、仓位或建议",
                extra_data={},
            )
            session.add(instrument)
            session.flush()
        instrument.name = defaults["name"]
        instrument.exchange = defaults["exchange"]
        instrument.category = "etf"
        instrument.currency = "CNY"
        instrument.is_active = True
        instrument.extra_data = {
            **dict(instrument.extra_data or {}),
            "research_kind": "display_only_etf",
            "role": "display_only",
            "model_eligible": False,
            "advice_eligible": False,
            "position_eligible": False,
            "adjustment_mode": "qfq",
            "exchange_prefix": defaults["exchange_prefix"],
        }

    for key, value, description in (
        ("app.default_currency", "CNY", "Default currency for local records"),
        ("app.timezone", "Asia/Shanghai", "Local display timezone"),
        ("fee.configuration", DEFAULT_FEE_CONFIG, "Default ETF fee configuration"),
    ):
        if session.scalar(select(AppSetting).where(AppSetting.key == key)) is None:
            session.add(AppSetting(key=key, value=value, category="default", description=description))

    strategy = session.scalar(
        select(StrategyDefinition).where(StrategyDefinition.name == "默认规则策略")
    )
    if strategy is None:
        strategy = StrategyDefinition(
            name="默认规则策略",
            strategy_type="rule",
            description="基于估值、趋势与技术指标的本地规则策略",
            enabled=True,
            parameters=DEFAULT_STRATEGY_CONFIG,
        )
        session.add(strategy)
        session.flush()

    for instrument in instruments.values():
        plan_name = f"{instrument.name} 每周定投"
        if session.scalar(select(InvestmentPlan).where(InvestmentPlan.name == plan_name)) is None:
            session.add(
                InvestmentPlan(
                    instrument_id=instrument.id,
                    name=plan_name,
                    frequency="weekly",
                    amount=Decimal("150.00"),
                    weekday=1,
                    enabled=True,
                )
            )

    for account_name in DEFAULT_SIMULATION_ACCOUNT_NAMES:
        if session.scalar(select(SimulationAccount).where(SimulationAccount.name == account_name)) is None:
            session.add(
                SimulationAccount(
                    name=account_name,
                    strategy_id=strategy.id,
                    initial_cash=Decimal("0"),
                    cash_balance=Decimal("0"),
                    portfolio_value=Decimal("0"),
                    enabled=True,
                )
            )

    if session.scalar(select(RealAccount).where(RealAccount.name == "实盘账户")) is None:
        session.add(
            RealAccount(
                name="实盘账户",
                initial_cash=Decimal("0"),
                cash_balance=Decimal("0"),
                enabled=True,
            )
        )


def _seed_v33_instrument_roles(session: Session) -> None:
    """Seed the explicit V3.3 tradable/benchmark split idempotently."""

    instruments = {
        instrument.code: instrument
        for instrument in session.scalars(
            select(Instrument).where(Instrument.code.in_(("159941", "NDX")))
        )
    }
    if set(instruments) != {"159941", "NDX"}:
        return
    available_at = utc_now()
    role_specs = (
        {
            "instrument": instruments["159941"],
            "role": "tradable",
            "benchmark_instrument_id": instruments["NDX"].id,
            "currency": "CNY",
            "exchange_timezone": "Asia/Shanghai",
        },
        {
            "instrument": instruments["NDX"],
            "role": "benchmark",
            "benchmark_instrument_id": None,
            "currency": "USD",
            "exchange_timezone": "America/New_York",
        },
    )
    for spec in role_specs:
        instrument = spec["instrument"]
        role = str(spec["role"])
        existing = session.scalar(
            select(V33InstrumentRole).where(
                V33InstrumentRole.instrument_id == instrument.id,
                V33InstrumentRole.model_market == "159941",
                V33InstrumentRole.role == role,
            )
        )
        if existing is not None:
            continue
        canonical = {
            "instrument_code": instrument.code,
            "model_market": "159941",
            "role": role,
            "benchmark_code": "NDX" if role == "tradable" else None,
            "currency": spec["currency"],
            "exchange_timezone": spec["exchange_timezone"],
            "protocol": "V3.3-20W",
        }
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        session.add(
            V33InstrumentRole(
                instrument_id=instrument.id,
                model_market="159941",
                role=role,
                benchmark_instrument_id=spec["benchmark_instrument_id"],
                currency=str(spec["currency"]),
                exchange_timezone=str(spec["exchange_timezone"]),
                effective_from=None,
                effective_to=None,
                is_active=True,
                available_at=available_at,
                vintage="V3.3-20W",
                source="LOCAL_PROTOCOL",
                source_url=None,
                raw_payload_hash=hashlib.sha256(payload).hexdigest(),
                quality_status="verified_config",
                payload_json=canonical,
                created_at=available_at,
                updated_at=available_at,
            )
        )


def _clear_ndx_legacy_volume(session: Session) -> None:
    """Idempotently remove unverified NDX volume and derived volume MAs."""
    ndx_id = session.scalar(select(Instrument.id).where(Instrument.code == "NDX"))
    if ndx_id is None:
        return
    rows = session.scalars(
        select(MarketPrice).where(
            MarketPrice.instrument_id == ndx_id,
        )
    ).all()
    for row in rows:
        row.volume = None
        row.volume_multiplier = 1
        row.volume_source = "VOLUME_UNAVAILABLE:DIRECT_INDEX"
    indicator_rows = session.scalars(
        select(IndicatorRecord).where(
            IndicatorRecord.instrument_id == ndx_id,
        )
    ).all()
    for row in indicator_rows:
        payload = dict(row.indicator_values or {})
        values = dict(payload.get("values") or {})
        volume_keys = tuple(
            key
            for key in values
            if key.startswith("volume_ma_")
        )
        if not any(values[key] is not None for key in volume_keys):
            continue
        for key in volume_keys:
            values[key] = None
        payload["values"] = values
        row.indicator_values = payload


def _rebuild_real_account_projections(engine: object) -> None:
    """Replay every manual ledger after schema setup.

    Version 9 added persisted cost/P&L projections to databases that may
    already contain a transaction ledger.  Rebuilding from that immutable
    ledger is idempotent and also makes any interrupted historical projection
    write self-healing on the next local start.
    """
    from ..services.real_account_service import RealAccountService

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        account_ids = list(session.scalars(select(RealAccount.id)))
    service = RealAccountService(factory)
    for account_id in account_ids:
        service.recalculate_account(account_id)


def initialize_database(
    path: Path | str | None = None,
    configuration_directory: Path | str | None = None,
    lock_timeout_seconds: float = 60.0,
    legacy_weekly_analysis_audit_root: Path | str | None = None,
):
    """Create schema/defaults when absent without deleting or overwriting local data."""
    target_path = (Path(path) if path is not None else default_database_path()).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with DatabaseInitializationLock(target_path, lock_timeout_seconds):
        ensure_default_configs(configuration_directory)
        engine = create_database_engine(target_path)
        try:
            run_migrations(
                engine,
                legacy_weekly_analysis_audit_root=(
                    legacy_weekly_analysis_audit_root
                ),
            )
            with Session(engine) as session, session.begin():
                _seed_if_missing(session)
                _seed_v33_instrument_roles(session)
                _clear_ndx_legacy_volume(session)
            _rebuild_real_account_projections(engine)
        finally:
            engine.dispose()
    return engine
