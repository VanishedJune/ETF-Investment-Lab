"""Idempotent initialization of the local SQLite schema and seed records."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from sqlalchemy import delete, select, text
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from sqlalchemy.orm import Session, sessionmaker

from ..config import DEFAULT_FEE_CONFIG, DEFAULT_STRATEGY_CONFIG, ensure_default_configs
from ..core.paths import database_path as default_database_path
from ..models.models import (
    AppSetting,
    Base,
    DataUpdateLog,
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccount,
    SimulationAccount,
    SimulationDailySnapshot,
    SimulationPosition,
    SimulationTransaction,
    StrategyDefinition,
    V33InstrumentRole,
    ValuationRecord,
    utc_now,
)
from ..services.instrument_universe import DISPLAY_ONLY_ETFS
from .migrations import run_migrations
from .session import create_database_engine


DEFAULT_INSTRUMENTS = (
    {"code": "589850", "name": "科创50ETF东财", "exchange": "SSE"},
    {"code": "159915", "name": "创业板ETF易方达", "exchange": "SZSE"},
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

LEGACY_REPLACED_ETF_CODE = "159205"
REPLACEMENT_ETF_CODE = "159915"
REPLACEMENT_ETF_NAME = "创业板ETF易方达"

STALE_LOCK_AGE_SECONDS = 300.0
DB_OPEN_RETRY_SECONDS = 45.0


def _migrate_replaced_etf(session: Session) -> None:
    """Replace the retired 159205 catalog entry with the real 159915 ETF.

    The old symbol's quote/indicator cache is intentionally discarded: keeping
    it under the new code would present Eastmoney's 159205 history as
    易方达创业板 ETF history.  User-owned plan/calendar/real-ledger rows keep
    their instrument foreign key (the replacement is performed in place),
    while generated simulation rows are removed and their account projections
    reset.  The operation is idempotent and runs inside the initialization
    transaction before catalog seeding.
    """

    legacy = session.scalar(
        select(Instrument).where(Instrument.code == LEGACY_REPLACED_ETF_CODE)
    )
    replacement = session.scalar(
        select(Instrument).where(Instrument.code == REPLACEMENT_ETF_CODE)
    )
    if legacy is None:
        if replacement is not None:
            replacement.name = REPLACEMENT_ETF_NAME
            replacement.exchange = "SZSE"
            replacement.category = "etf"
            replacement.currency = "CNY"
            replacement.is_active = True
            replacement.description = "创业板 ETF 易方达（159915）；使用 ETF 自身真实行情。"
            cleaned_extra = dict(replacement.extra_data or {})
            cleaned_extra.pop("replaced_legacy_code", None)
            replacement.extra_data = cleaned_extra
        return
    if replacement is not None and replacement.id != legacy.id:
        raise RuntimeError(
            "Cannot replace 159205: both legacy 159205 and replacement 159915 "
            "instruments exist; resolve the duplicate catalog entries first."
        )

    instrument_id = legacy.id
    affected_simulation_accounts = set(
        session.scalars(
            select(SimulationTransaction.account_id).where(
                SimulationTransaction.instrument_id == instrument_id
            )
        )
    )

    # Quote, indicator, valuation, and source logs are symbol-specific and
    # must not survive under 159915.  The plan/calendar/real-ledger tables are
    # user-owned and deliberately retained.
    for model in (MarketPrice, IndicatorRecord, ValuationRecord):
        session.execute(delete(model).where(model.instrument_id == instrument_id))
    session.execute(
        delete(DataUpdateLog).where(DataUpdateLog.instrument_id == instrument_id)
    )
    session.execute(
        delete(SimulationTransaction).where(
            SimulationTransaction.instrument_id == instrument_id
        )
    )
    session.execute(
        delete(SimulationPosition).where(
            SimulationPosition.instrument_id == instrument_id
        )
    )
    for account_id in affected_simulation_accounts:
        session.execute(
            delete(SimulationDailySnapshot).where(
                SimulationDailySnapshot.account_id == account_id
            )
        )
        account = session.get(SimulationAccount, account_id)
        if account is not None:
            account.cash_balance = account.initial_cash
            account.portfolio_value = Decimal("0")

    # Preserve the existing plan amount and user ledger, but make their
    # visible name unambiguously refer to the replacement ETF.
    for plan in session.scalars(
        select(InvestmentPlan).where(InvestmentPlan.instrument_id == instrument_id)
    ):
        plan.name = plan.name.replace("创业板ETF东财", REPLACEMENT_ETF_NAME)
        plan.name = plan.name.replace(LEGACY_REPLACED_ETF_CODE, REPLACEMENT_ETF_CODE)

    legacy.code = REPLACEMENT_ETF_CODE
    legacy.name = REPLACEMENT_ETF_NAME
    legacy.exchange = "SZSE"
    legacy.category = "etf"
    legacy.currency = "CNY"
    legacy.is_active = True
    legacy.description = "创业板 ETF 易方达（159915）；使用 ETF 自身真实行情。"
    legacy.extra_data = {
        **dict(legacy.extra_data or {}),
        "research_kind": "display_only_etf",
        "role": "display_only",
        "model_eligible": False,
        "advice_eligible": False,
        "position_eligible": False,
        "adjustment_mode": "qfq",
        "exchange_prefix": "sz",
    }
    session.flush()


class DatabaseInitializationLock:
    """Cross-process lock protecting schema migration and initial seed writes."""

    def __init__(self, database: Path, timeout_seconds: float) -> None:
        self._database_path = Path(database).resolve()
        self.path = self._database_path.parent / ".investment_lab.initialize.lock"
        self.recovery_path = self._database_path.parent / ".investment_lab.initialize.lock.recovery"
        self.timeout_seconds = timeout_seconds
        self._descriptor: int | None = None

    def __enter__(self) -> DatabaseInitializationLock:
        if os.name == "nt":
            return self._enter_windows()
        return self._enter_posix()

    def _enter_posix(self) -> DatabaseInitializationLock:
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

    def _enter_windows(self) -> DatabaseInitializationLock:
        """Windows kernel byte-range lock without the unlink race.

        ``msvcrt.locking`` is enforced by the OS and released automatically
        when the owning process exits, so concurrent initializers can never
        enter the critical section together.  A pre-existing metadata file
        whose owner is still alive is treated as a live lock (kept for the
        existing reclaim tests); a stale file is simply overwritten.
        """

        import msvcrt

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                existed_before = self.path.exists()
                descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR)
            except PermissionError:
                if not self.path.exists():
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for database initialization lock: {self.path}"
                    )
                time.sleep(0.05)
                continue
            try:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                except OSError:
                    os.close(descriptor)
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for database initialization lock: {self.path}"
                        )
                    time.sleep(0.05)
                    continue
                if existed_before:
                    os.lseek(descriptor, 0, 0)
                    raw = os.read(descriptor, 4096)
                    owner: int | None = None
                    try:
                        parsed = json.loads(raw.decode("utf-8"))
                        if isinstance(parsed.get("pid"), int):
                            owner = int(parsed["pid"])
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        owner = None
                    if owner is not None and self._process_is_running(owner):
                        os.lseek(descriptor, 0, 0)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                        os.close(descriptor)
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out waiting for database initialization lock: {self.path}"
                            )
                        time.sleep(0.05)
                        continue
                os.lseek(descriptor, 0, 0)
                os.ftruncate(descriptor, 0)
                metadata = json.dumps(
                    {"pid": os.getpid(), "created_at": time.time()}
                ).encode("utf-8")
                os.write(descriptor, metadata)
                self._descriptor = descriptor
                return self
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
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
            if os.name == "nt":
                import msvcrt

                try:
                    os.lseek(self._descriptor, 0, 0)
                    msvcrt.locking(self._descriptor, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            os.close(self._descriptor)
        finally:
            self._descriptor = None
            for _attempt in range(20):
                try:
                    self.path.unlink()
                    return
                except FileNotFoundError:
                    return
                except PermissionError:
                    # A concurrent waiter may hold a transient read handle
                    # (Windows does not allow deleting an open file).  Retry
                    # briefly; if it still fails, leave a stale lock that the
                    # next waiter reclaims after this PID exits.
                    time.sleep(0.05)
            # Final best-effort unlink; a stale lock is recoverable.
            try:
                self.path.unlink()
            except OSError:
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
        # The V3.7 data-only catalog is available for charts and the
        # investment calendar, not for automatic investment-plan creation.
        # Keep legacy plans untouched in existing databases; this prevents
        # fresh databases from reintroducing plan rows for display-only ETFs.
        if instrument.code in DISPLAY_ONLY_ETFS:
            continue
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
    from .session import _plain_windows_path

    target_path = _plain_windows_path(
        (Path(path) if path is not None else default_database_path()).resolve()
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with DatabaseInitializationLock(target_path, lock_timeout_seconds):
        ensure_default_configs(configuration_directory)
        engine = create_database_engine(target_path)
        last_error: BaseException | None = None
        for attempt in range(2):
            try:
                run_migrations(
                    engine,
                    legacy_weekly_analysis_audit_root=(
                        legacy_weekly_analysis_audit_root
                    ),
                )
                with Session(engine) as session, session.begin():
                    _migrate_replaced_etf(session)
                    _seed_if_missing(session)
                    _seed_v33_instrument_roles(session)
                    _clear_ndx_legacy_volume(session)
                _rebuild_real_account_projections(engine)
                last_error = None
                break
            except (sqlite3.OperationalError, SQLAlchemyOperationalError) as exc:
                # On Windows another process or the filesystem scanner can
                # transiently hold the freshly created database file.  Dispose
                # every connection and retry from a clean engine.
                last_error = exc
                engine.dispose()
                if attempt < 1 and "unable to open database file" in str(exc):
                    _wait_until_engine_openable(target_path, DB_OPEN_RETRY_SECONDS)
                    engine = create_database_engine(target_path)
                    continue
                raise
            finally:
                engine.dispose()
        if last_error is not None:
            raise last_error
    return engine




def _wait_until_engine_openable(path: Path, timeout_seconds: float) -> None:
    """Wait until a real SQLAlchemy engine can connect to the database.

    On Windows a just-finished initialization can leave the database briefly
    unopenable for a fresh engine (filesystem scanner or sharing-state race).
    Polling with the exact engine configuration used by
    ``initialize_database`` waits out that transient state.
    """

    deadline = time.monotonic() + timeout_seconds
    while True:
        probe_engine = create_database_engine(path)
        try:
            with probe_engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA user_version").scalar_one()
            return
        except (sqlite3.OperationalError, SQLAlchemyOperationalError):
            if time.monotonic() >= deadline:
                return
            time.sleep(0.1)
        finally:
            probe_engine.dispose()
