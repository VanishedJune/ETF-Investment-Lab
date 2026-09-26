"""V3.5.1 ETF slot seeding, validation and transactional replacement."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import re
from typing import Any, Callable, Mapping
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V351InstrumentMetadata,
    V351InstrumentSlot,
    V351InstrumentSlotHistory,
    V35Forecast,
    V351SlotModelState,
    V351SlotReplacementJob,
    V35BootstrapState,
    utc_now,
)
from backend.app.services.v351_config import PROTOCOL_VERSION_351


DEFAULT_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "slot_id": "ETF_SLOT_01",
        "slot_order": 5,
        "instrument_code": "159611",
        "exchange": "SZSE",
        "instrument_name": "电力ETF广发",
        "instrument_type": "ETF",
        "model_enabled": False,
        "model_status": "DATA_ONLY",
    },
    {
        "slot_id": "ETF_SLOT_02",
        "slot_order": 2,
        "instrument_code": "517520",
        "exchange": "SSE",
        "instrument_name": "黄金股ETF永赢",
        "instrument_type": "ETF",
        "model_enabled": False,
        "model_status": "DATA_ONLY",
    },
    {
        "slot_id": "ETF_SLOT_03",
        "slot_order": 7,
        "instrument_code": "512800",
        "exchange": "SSE",
        "instrument_name": "华宝银行ETF",
        "instrument_type": "ETF",
        "model_enabled": False,
        "model_status": "DATA_ONLY",
    },
    {
        "slot_id": "ETF_SLOT_04",
        "slot_order": 8,
        "instrument_code": "512690",
        "exchange": "SSE",
        "instrument_name": "鹏华酒ETF",
        "instrument_type": "ETF",
        "model_enabled": False,
        "model_status": "DATA_ONLY",
    },
    {
        "slot_id": "ETF_SLOT_05",
        "slot_order": 6,
        "instrument_code": "515220",
        "exchange": "SSE",
        "instrument_name": "煤炭ETF国泰",
        "instrument_type": "ETF",
        "model_enabled": False,
        "model_status": "DATA_ONLY",
    },
    *tuple({
        "slot_id": slot_id, "slot_order": order, "instrument_code": code,
        "exchange": exchange, "instrument_name": name,
        "instrument_type": "ETF", "model_enabled": False, "model_status": "DATA_ONLY",
    } for slot_id, order, code, exchange, name in (
        ("ETF_SLOT_06", 1, "159915", "SZSE", "创业板ETF易方达"),
        ("ETF_SLOT_07", 3, "516150", "SSE", "稀土ETF嘉实"),
        ("ETF_SLOT_08", 4, "159622", "SZSE", "创新药ETF东财"),
    )),
)

VALID_STATES = (
    "VALIDATING",
    "DOWNLOADING_DATA",
    "BUILDING_WEEKLY_DATA",
    "BUILDING_FEATURES",
    "PREWARMING",
    "TRAINING_INITIAL_CHAMPION",
    "VALIDATING_MODEL",
    "READY_TO_SWITCH",
    "SWITCHED",
    "FAILED_ROLLED_BACK",
)
PENDING_STATES = {
    "VALIDATING",
    "DOWNLOADING_DATA",
    "BUILDING_WEEKLY_DATA",
    "BUILDING_FEATURES",
    "PREWARMING",
    "TRAINING_INITIAL_CHAMPION",
    "VALIDATING_MODEL",
    "READY_TO_SWITCH",
}


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


def lookup_etf_metadata(code: str) -> dict[str, Any]:
    """Look up ETF name/exchange via the live AkShare ETF spot list."""

    import akshare as ak

    spot = ak.fund_etf_spot_em()
    code_column = "代码" if "代码" in spot.columns else spot.columns[0]
    name_column = "名称" if "名称" in spot.columns else spot.columns[1]
    row = spot[spot[code_column].astype(str) == code]
    if row.empty:
        raise ValueError(f"证券代码 {code} 不存在于公开ETF列表")
    name = str(row.iloc[0][name_column])
    exchange = "SZSE" if code.startswith(("1", "15", "16")) else "SSE"
    return {
        "instrument_code": code,
        "official_name": name,
        "exchange": exchange,
        "instrument_type": "ETF",
        "data_source": "AKSHARE_FUND_ETF_SPOT_EM",
    }


def seed_default_slots(session: Session) -> None:
    """Create missing data-only slots; existing user bindings are never reset."""

    existing = {
        row.slot_id: row
        for row in session.scalars(select(V351InstrumentSlot)).all()
    }
    now = utc_now()
    for spec in DEFAULT_SLOTS:
        row = existing.get(spec["slot_id"])
        if row is not None:
            continue
        instrument = session.scalar(
            select(Instrument).where(Instrument.code == spec["instrument_code"])
        )
        if instrument is None:
            instrument = Instrument(
                code=spec["instrument_code"],
                name=spec["instrument_name"],
                exchange="SSE" if spec["exchange"] == "SSE" else "SZSE",
                category="etf",
                currency="CNY",
                is_active=True,
                extra_data={
                    "research_kind": "etf_slot",
                    "role": "etf_slot",
                    "model_eligible": False,
                    "advice_eligible": False,
                    "position_eligible": True,
                    "adjustment_mode": "qfq",
                },
            )
            session.add(instrument)
            session.flush()
        slot = V351InstrumentSlot(
            slot_id=spec["slot_id"],
            slot_order=spec["slot_order"],
            instrument_code=spec["instrument_code"],
            exchange=spec["exchange"],
            instrument_name=spec["instrument_name"],
            instrument_type=spec["instrument_type"],
            model_enabled=spec["model_enabled"],
            active=True,
            binding_version="B1",
            bound_at=now,
            replacement_status="READY",
            history_week_count=0,
            mature_8w_count=0,
            model_status=spec["model_status"],
            champion_package_id=None,
            created_at=now,
            updated_at=now,
        )
        session.add(slot)
        session.flush()
        session.add(
            V351InstrumentSlotHistory(
                slot_id=spec["slot_id"],
                binding_version="B1",
                instrument_code=spec["instrument_code"],
                action="BIND",
                effective_from=now,
                reason="V3.5.1 初始槽位绑定",
                created_at=now,
            )
        )
        session.add(
            V351SlotModelState(
                slot_id=spec["slot_id"],
                binding_version="B1",
                model_namespace=f"v351:{spec['instrument_code']}:B1",
                account_namespace=f"v351:{spec['instrument_code']}:B1",
                forecast_namespace=f"v351:{spec['instrument_code']}:B1",
                model_status=spec["model_status"],
                champion_package_id=None,
                small_sample_champion=False,
                prewarming=False,
                mature_8w_count=0,
                bootstrap_state="NOT_STARTED",
                state_json={},
                updated_at=now,
            )
        )


def slot_dict(slot: V351InstrumentSlot, model_state: V351SlotModelState | None) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "slot_order": slot.slot_order,
        "instrument_code": slot.instrument_code,
        "exchange": slot.exchange,
        "instrument_name": slot.instrument_name,
        "instrument_type": slot.instrument_type,
        "model_enabled": slot.model_enabled,
        "active": slot.active,
        "binding_version": slot.binding_version,
        "bound_at": slot.bound_at.isoformat(),
        "replaced_from_code": slot.replaced_from_code,
        "replacement_status": slot.replacement_status,
        "data_start_date": slot.data_start_date.isoformat() if slot.data_start_date else None,
        "data_end_date": slot.data_end_date.isoformat() if slot.data_end_date else None,
        "history_week_count": slot.history_week_count,
        "mature_8w_count": slot.mature_8w_count,
        "model_status": slot.model_status,
        "champion_package_id": slot.champion_package_id,
        "last_market_update": slot.last_market_update.isoformat() if slot.last_market_update else None,
        "model_state": {
            "model_namespace": model_state.model_namespace if model_state else None,
            "model_status": model_state.model_status if model_state else None,
            "small_sample_champion": model_state.small_sample_champion if model_state else False,
            "prewarming": model_state.prewarming if model_state else False,
            "mature_8w_count": model_state.mature_8w_count if model_state else 0,
            "bootstrap_state": model_state.bootstrap_state if model_state else "NOT_STARTED",
        },
    }


class V351SlotService:
    """Slot listing, validation and replacement execution."""

    def __init__(
        self,
        factory: sessionmaker,
        *,
        metadata_provider: Callable[[str], dict[str, Any]] = lookup_etf_metadata,
    ) -> None:
        self._factory = factory
        self._metadata_provider = metadata_provider

    def list_slots(self) -> list[dict[str, Any]]:
        with self._factory() as session:
            seed_default_slots(session)
            session.commit()
        with self._factory() as session:
            slots = session.scalars(
                select(V351InstrumentSlot)
                .where(V351InstrumentSlot.active.is_(True))
                .order_by(V351InstrumentSlot.slot_order)
            ).all()
            return [
                slot_dict(
                    slot,
                    session.scalar(
                        select(V351SlotModelState).where(
                            V351SlotModelState.slot_id == slot.slot_id,
                            V351SlotModelState.binding_version == slot.binding_version,
                        )
                    ),
                )
                for slot in slots
            ]

    def _slot(self, session: Session, slot_id: str) -> V351InstrumentSlot:
        slot = session.scalar(
            select(V351InstrumentSlot).where(
                V351InstrumentSlot.slot_id == slot_id,
                V351InstrumentSlot.active.is_(True),
            )
        )
        if slot is None:
            raise ValueError(f"未知或已归档槽位 {slot_id}")
        return slot

    def _validate_code(self, code: str) -> str:
        if not re.fullmatch(r"\d{6}", code):
            raise ValueError("ETF代码必须是6位数字")
        return code

    def _metadata(self, session: Session, code: str) -> dict[str, Any]:
        existing = session.scalar(
            select(V351InstrumentMetadata).where(
                V351InstrumentMetadata.instrument_code == code
            )
        )
        if existing is not None:
            return dict(existing.payload_json)
        metadata = self._metadata_provider(code)
        session.add(
            V351InstrumentMetadata(
                instrument_code=code,
                official_name=str(metadata["official_name"]),
                exchange=str(metadata["exchange"]),
                instrument_type=str(metadata["instrument_type"]),
                data_source=str(metadata["data_source"]),
                verified_at=utc_now(),
                payload_json=metadata,
                metadata_hash=_hash(metadata),
                created_at=utc_now(),
            )
        )
        return metadata

    def validate_replacement(
        self,
        slot_id: str,
        code: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        code = self._validate_code(code)
        with self._factory() as session:
            slot = self._slot(session, slot_id)
            if slot.instrument_code == code:
                raise ValueError("该ETF已经绑定到当前槽位，无需替换")
            duplicates = session.scalars(
                select(V351InstrumentSlot).where(
                    V351InstrumentSlot.active.is_(True),
                    V351InstrumentSlot.instrument_code == code,
                )
            ).all()
            if duplicates:
                raise ValueError(f"代码 {code} 已存在于其他活动槽位")
            metadata = self._metadata(session, code)
            session.commit()
            return {
                "slot_id": slot_id,
                "target_code": code,
                "metadata": metadata,
                "replaced_from_code": slot.instrument_code,
                "replaced_from_name": slot.instrument_name,
                "preview": {
                    "earliest_data_date": None,
                    "daily_count": None,
                    "weekly_count": None,
                    "estimated_mature_8w": None,
                    "can_initial_champion": None,
                    "requires_prewarming": True,
                },
                "confirm_required": True,
            }

    def replacement_status(self, slot_id: str) -> dict[str, Any] | None:
        with self._factory() as session:
            job = session.scalar(
                select(V351SlotReplacementJob)
                .where(V351SlotReplacementJob.slot_id == slot_id)
                .order_by(V351SlotReplacementJob.created_at.desc())
            )
            if job is None:
                return None
            return {
                "job_id": job.id,
                "slot_id": job.slot_id,
                "target_code": job.target_code,
                "state": job.state,
                "current_step": job.current_step,
                "error_code": job.error_code,
                "error_message": job.error_message,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            }

    def slot_history(self, slot_id: str) -> list[dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(V351InstrumentSlotHistory)
                .where(V351InstrumentSlotHistory.slot_id == slot_id)
                .order_by(V351InstrumentSlotHistory.effective_from)
            ).all()
            return [
                {
                    "binding_version": row.binding_version,
                    "instrument_code": row.instrument_code,
                    "action": row.action,
                    "effective_from": row.effective_from.isoformat(),
                    "effective_to": row.effective_to.isoformat() if row.effective_to else None,
                    "reason": row.reason,
                }
                for row in rows
            ]

    def sync_model_state(self, slot_id: str) -> dict[str, Any]:
        """Refresh slot and per-binding model-state metadata from real records.

        Used after a direct bootstrap replay (or an external data backfill) so
        slot cards show the same champion/counts as the frozen records.
        """

        with self._factory() as session:
            slot = self._slot(session, slot_id)
            code = slot.instrument_code
            version = slot.binding_version
            instrument = session.scalar(
                select(Instrument).where(Instrument.code == code)
            )
            bootstrap = session.scalar(
                select(V35BootstrapState).where(
                    V35BootstrapState.model_market == code,
                    V35BootstrapState.protocol_version == PROTOCOL_VERSION_351,
                )
            )
            champion_id = bootstrap.champion_package_id if bootstrap else None
            history_weeks = 0
            mature_count = 0
            data_start = None
            data_end = None
            if instrument is not None:
                history_weeks = int(
                    session.scalar(
                        select(func.count())
                        .select_from(MarketPrice)
                        .where(
                            MarketPrice.instrument_id == instrument.id,
                            MarketPrice.timeframe == "weekly",
                        )
                    )
                    or 0
                )
                mature_count = int(
                    session.scalar(
                        select(func.count())
                        .select_from(V35Forecast)
                        .where(
                            V35Forecast.model_market == code,
                            V35Forecast.protocol_version == PROTOCOL_VERSION_351,
                            V35Forecast.maturity_status == "FULLY_MATURE_8W",
                        )
                    )
                    or 0
                )
                data_start = session.scalar(
                    select(func.min(MarketPrice.trade_date)).where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "daily",
                    )
                )
                data_end = session.scalar(
                    select(func.max(MarketPrice.trade_date)).where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "daily",
                    )
                )
            now = utc_now()
            slot.instrument_name = self._slot_name(session, code, slot_id)
            slot.exchange = self._slot_exchange(session, code, slot_id)
            slot.model_status = (
                "MODEL_READY" if champion_id else "PREWARMING_NOT_ENOUGH_HISTORY"
            )
            slot.champion_package_id = champion_id
            slot.history_week_count = history_weeks
            slot.mature_8w_count = mature_count
            slot.data_start_date = data_start
            slot.data_end_date = data_end
            slot.last_market_update = now
            slot.updated_at = now
            model_state = session.scalar(
                select(V351SlotModelState).where(
                    V351SlotModelState.slot_id == slot_id,
                    V351SlotModelState.binding_version == version,
                )
            )
            if model_state is None:
                model_state = V351SlotModelState(
                    slot_id=slot_id,
                    binding_version=version,
                    model_namespace=f"v351:{code}:{version}",
                    account_namespace=f"v351:{code}:{version}",
                    forecast_namespace=f"v351:{code}:{version}",
                    state_json={},
                )
                session.add(model_state)
            model_state.model_status = slot.model_status
            model_state.champion_package_id = champion_id
            model_state.small_sample_champion = False
            model_state.prewarming = champion_id is None
            model_state.mature_8w_count = mature_count
            model_state.last_bootstrap_anchor = (
                bootstrap.last_completed_anchor if bootstrap else None
            )
            model_state.bootstrap_state = bootstrap.state if bootstrap else "NOT_STARTED"
            model_state.updated_at = now
            session.commit()
            return slot_dict(slot, model_state)

    def start_replace(
        self,
        slot_id: str,
        code: str,
        *,
        idempotency_key: str | None = None,
        maximum_weeks: int | None = None,
    ) -> dict[str, Any]:
        code = self._validate_code(code)
        key = idempotency_key or f"{slot_id}:{code}:{uuid4().hex[:12]}"
        with self._factory() as session:
            existing_job = session.scalar(
                select(V351SlotReplacementJob).where(
                    V351SlotReplacementJob.idempotency_key == key
                )
            )
            if existing_job is not None:
                return {"job_id": existing_job.id, "state": existing_job.state, "reused": True}
            slot = self._slot(session, slot_id)
            if slot.replacement_status in PENDING_STATES:
                raise ValueError(f"槽位 {slot_id} 正在执行替换，请等待完成")
            if slot.instrument_code == code:
                raise ValueError("该ETF已经绑定到当前槽位，无需替换")
            duplicates = session.scalars(
                select(V351InstrumentSlot).where(
                    V351InstrumentSlot.active.is_(True),
                    V351InstrumentSlot.instrument_code == code,
                )
            ).all()
            if duplicates:
                raise ValueError(f"代码 {code} 已存在于其他活动槽位")
            metadata = self._metadata(session, code)
            job_id = f"V351REPL:{slot_id}:{key[:20]}:{uuid4().hex[:8]}"
            session.add(
                V351SlotReplacementJob(
                    id=job_id,
                    slot_id=slot_id,
                    target_code=code,
                    idempotency_key=key,
                    state="VALIDATING",
                    current_step="VALIDATING",
                    created_at=utc_now(),
                    updated_at=utc_now(),
                    payload_json={
                        "slot_id": slot_id,
                        "target_code": code,
                        "idempotency_key": key,
                        "metadata": metadata,
                        "original_slot": {
                            "instrument_code": slot.instrument_code,
                            "instrument_name": slot.instrument_name,
                            "exchange": slot.exchange,
                            "binding_version": slot.binding_version,
                        },
                    },
                    result_json={},
                )
            )
            slot.replacement_status = "VALIDATING"
            slot.updated_at = utc_now()
            session.commit()

        return {
            "job_id": job_id,
            "state": "VALIDATING",
            "target_code": code,
            "idempotency_key": key,
        }

    def replace(
        self,
        slot_id: str,
        code: str,
        *,
        idempotency_key: str | None = None,
        maximum_weeks: int | None = None,
    ) -> dict[str, Any]:
        started = self.start_replace(
            slot_id,
            code,
            idempotency_key=idempotency_key,
            maximum_weeks=maximum_weeks,
        )
        if started.get("reused"):
            return started
        job_id = str(started["job_id"])
        key = str(started["idempotency_key"])
        try:
            self._run_replacement(slot_id, code, job_id, key, maximum_weeks)
        except Exception as exc:  # noqa: BLE001 - job-level failure handling
            self._mark_failed(slot_id, job_id, key, exc)
            raise
        return {"job_id": job_id, "state": "SWITCHED", "target_code": code}

    def cancel_replacement(self, slot_id: str) -> dict[str, Any]:
        with self._factory() as session:
            job = session.scalar(
                select(V351SlotReplacementJob)
                .where(V351SlotReplacementJob.slot_id == slot_id)
                .order_by(V351SlotReplacementJob.created_at.desc())
            )
            if job is None or job.state not in PENDING_STATES:
                raise ValueError("no pending replacement to cancel")
            payload = dict(job.payload_json or {})
            payload["cancel_requested"] = True
            job.payload_json = payload
            job.updated_at = utc_now()
            session.commit()
            return {
                "job_id": job.id,
                "state": job.state,
                "cancel_requested": True,
            }

    @staticmethod
    def _raise_if_cancelled(session: Session, job_id: str) -> None:
        job = session.get(V351SlotReplacementJob, job_id)
        if job is not None and (job.payload_json or {}).get("cancel_requested"):
            raise RuntimeError("replacement cancelled by user")

    def _run_replacement(
        self,
        slot_id: str,
        code: str,
        job_id: str,
        key: str,
        maximum_weeks: int | None,
    ) -> None:
        with self._factory() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == code))
            if instrument is None:
                slot = self._slot(session, slot_id)
                metadata = session.scalar(
                    select(V351InstrumentMetadata).where(
                        V351InstrumentMetadata.instrument_code == code
                    )
                )
                instrument = Instrument(
                    code=code,
                    name=metadata.official_name if metadata else code,
                    exchange=metadata.exchange if metadata else "SSE",
                    category="etf",
                    currency="CNY",
                    is_active=True,
                    extra_data={
                        "research_kind": "etf_slot",
                        "role": "etf_slot",
                        "model_eligible": True,
                        "adjustment_mode": "qfq",
                    },
                )
                session.add(instrument)
                session.flush()
            job = session.get(V351SlotReplacementJob, job_id)
            job.state = "DOWNLOADING_DATA"
            job.current_step = "DOWNLOADING_DATA"
            job.updated_at = utc_now()
            session.commit()
            self._raise_if_cancelled(session, job_id)

        # Download the new ETF's own OHLCV through the existing provider stack.
        from backend.app.services.market_data import MarketDataService
        from backend.app.services.providers import AkShareEtfResearchProvider
        from backend.app.services.market_calendar import ExchangeCalendarProvider

        market = MarketDataService(self._factory, calendar_provider=ExchangeCalendarProvider())
        provider = AkShareEtfResearchProvider()
        result = market.update_from_providers(code, [provider], None, None)
        if not result.records:
            raise RuntimeError(f"下载 {code} 行情失败：未获取到可用记录")

        with self._factory() as session:
            job = session.get(V351SlotReplacementJob, job_id)
            job.state = "BUILDING_WEEKLY_DATA"
            job.current_step = "BUILDING_WEEKLY_DATA"
            job.updated_at = utc_now()
            session.commit()
            self._raise_if_cancelled(session, job_id)
        self._aggregate_instrument(code, result.cutoff_date)

        # Data-only mode: replacement never bootstraps, trains, or invokes an
        # inference service.  It only builds chart timeframes and indicators.
        with self._factory() as session:
            job = session.get(V351SlotReplacementJob, job_id)
            job.state = "READY_TO_SWITCH"
            job.current_step = "READY_TO_SWITCH"
            job.updated_at = utc_now()
            session.commit()
            self._raise_if_cancelled(session, job_id)

        # Finalize the slot switch and archive the previous binding.
        with self._factory() as session:
            slot = self._slot(session, slot_id)
            old_code = slot.instrument_code
            old_version = slot.binding_version
            old_history = session.scalar(
                select(V351InstrumentSlotHistory).where(
                    V351InstrumentSlotHistory.slot_id == slot_id,
                    V351InstrumentSlotHistory.binding_version == old_version,
                )
            )
            now = utc_now()
            version_number = int(re.sub(r"\D", "", old_version) or 0) + 1
            new_version = f"B{version_number}"
            bootstrap = None
            champion_id = None
            history_weeks = 0
            mature_count = 0
            data_start = None
            data_end = None
            instrument_row = session.scalar(
                select(Instrument).where(Instrument.code == code)
            )
            if instrument_row is not None:
                history_weeks = int(
                    session.scalar(
                        select(func.count())
                        .select_from(MarketPrice)
                        .where(
                            MarketPrice.instrument_id == instrument_row.id,
                            MarketPrice.timeframe == "weekly",
                        )
                    )
                    or 0
                )
                data_start = session.scalar(
                    select(func.min(MarketPrice.trade_date)).where(
                        MarketPrice.instrument_id == instrument_row.id,
                        MarketPrice.timeframe == "daily",
                    )
                )
                data_end = session.scalar(
                    select(func.max(MarketPrice.trade_date)).where(
                        MarketPrice.instrument_id == instrument_row.id,
                        MarketPrice.timeframe == "daily",
                    )
                )
            slot.instrument_code = code
            slot.instrument_name = self._slot_name(session, code, slot_id)
            slot.exchange = self._slot_exchange(session, code, slot_id)
            slot.binding_version = new_version
            slot.bound_at = now
            slot.replaced_from_code = old_code
            slot.replacement_status = "READY"
            slot.model_status = "DATA_ONLY"
            slot.champion_package_id = champion_id
            slot.history_week_count = history_weeks
            slot.mature_8w_count = mature_count
            slot.data_start_date = data_start
            slot.data_end_date = data_end
            slot.last_market_update = now
            slot.updated_at = now
            if old_history is not None:
                old_history.effective_to = now
            session.add(
                V351InstrumentSlotHistory(
                    slot_id=slot_id,
                    binding_version=new_version,
                    instrument_code=code,
                    action="REPLACE",
                    effective_from=now,
                    reason=f"用户替换 {old_code} → {code}",
                    created_at=now,
                )
            )
            session.add(
                V351SlotModelState(
                    slot_id=slot_id,
                    binding_version=new_version,
                    model_namespace=f"v351:{code}:{new_version}",
                    account_namespace=f"v351:{code}:{new_version}",
                    forecast_namespace=f"v351:{code}:{new_version}",
                    model_status=slot.model_status,
                    champion_package_id=champion_id,
                    small_sample_champion=False,
                    prewarming=champion_id is None,
                    mature_8w_count=mature_count,
                    last_bootstrap_anchor=None,
                    bootstrap_state="DISABLED_DATA_ONLY",
                    state_json={},
                    updated_at=now,
                )
            )
            job = session.get(V351SlotReplacementJob, job_id)
            job.state = "SWITCHED"
            job.current_step = "SWITCHED"
            job.finished_at = now
            job.result_json = {
                "old_code": old_code,
                "new_code": code,
                "binding_version": new_version,
                "champion_package_id": champion_id,
                "mature_8w_count": mature_count,
            }
            job.updated_at = now
            session.commit()

    def _aggregate_instrument(self, code: str, as_of: date) -> None:
        from backend.app.services.indicator_service import IndicatorService
        from backend.app.services.market_data import MarketDataService
        from backend.app.services.market_calendar import ExchangeCalendarProvider

        market = MarketDataService(self._factory, calendar_provider=ExchangeCalendarProvider())
        indicators = IndicatorService(self._factory)
        market.aggregate_periods(code, weekly_result=None, as_of=as_of)
        # ``update_from_providers`` persists daily prices but does not calculate
        # their indicators.  Rebuild all chart timeframes before exposing the
        # newly bound slot; otherwise the UI receives an empty daily indicator
        # series and renders DIF/DEA/MACD as zeros until a manual repair.
        for timeframe in ("daily", "weekly", "monthly"):
            indicators.recalculate(code, timeframe)

    def _slot_name(self, session: Session, code: str, slot_id: str) -> str:
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is not None:
            return str(instrument.name)
        metadata = session.scalar(
            select(V351InstrumentMetadata).where(
                V351InstrumentMetadata.instrument_code == code
            )
        )
        return str(metadata.official_name) if metadata else code

    def _slot_exchange(self, session: Session, code: str, slot_id: str) -> str:
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is not None and instrument.exchange:
            return str(instrument.exchange)
        metadata = session.scalar(
            select(V351InstrumentMetadata).where(
                V351InstrumentMetadata.instrument_code == code
            )
        )
        return str(metadata.exchange) if metadata else "SSE"

    def _mark_failed(
        self,
        slot_id: str,
        job_id: str,
        key: str,
        error: BaseException,
    ) -> None:
        with self._factory() as session, session.begin():
            job = session.get(V351SlotReplacementJob, job_id)
            if job is not None:
                job.state = "FAILED_ROLLED_BACK"
                job.current_step = "FAILED_ROLLED_BACK"
                job.finished_at = utc_now()
                job.error_code = type(error).__name__
                job.error_message = str(error)[:2000]
                job.updated_at = utc_now()
            slot = session.scalar(
                select(V351InstrumentSlot).where(
                    V351InstrumentSlot.slot_id == slot_id,
                    V351InstrumentSlot.active.is_(True),
                )
            )
            if slot is not None:
                slot.replacement_status = "FAILED_ROLLED_BACK"
                slot.updated_at = utc_now()
