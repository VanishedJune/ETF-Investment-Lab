from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from decimal import Decimal, getcontext
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import Base, IndicatorRecord, Instrument, MarketPrice
from backend.app.indicators.calculator import IndicatorCalculator, IndicatorParameters, PricePoint
from backend.app.services.indicator_service import IndicatorCalculationStatus, IndicatorService


def _points(closes: list[str], volumes: list[str | None] | None = None) -> list[PricePoint]:
    start = date(2026, 1, 1)
    return [
        PricePoint(
            trade_date=start + timedelta(days=index),
            close=Decimal(close),
            volume=None if volumes is None or volumes[index] is None else Decimal(volumes[index]),
        )
        for index, close in enumerate(closes)
    ]


def _reference_ema(values: list[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = []
    consecutive: list[Decimal] = []
    current: Decimal | None = None
    multiplier = Decimal("2") / Decimal(period + 1)
    for value in values:
        if current is None:
            consecutive.append(value)
            if len(consecutive) < period:
                result.append(sum(consecutive, Decimal("0")) / Decimal(len(consecutive)))
                continue
            current = sum(consecutive, Decimal("0")) / Decimal(period)
        else:
            current = (value - current) * multiplier + current
        result.append(current)
    return result


def _reference_macd(closes: list[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    ema_12 = _reference_ema(closes, 12)
    ema_26 = _reference_ema(closes, 26)
    dif = [
        short - long if short is not None and long is not None else None
        for short, long in zip(ema_12, ema_26, strict=True)
    ]
    valid_dif = [value for value in dif if value is not None]
    dea_values = _reference_ema(valid_dif, 9)
    dea = dea_values[-1]
    assert dif[-1] is not None and dea is not None
    return dif[-1], dea, Decimal("2") * (dif[-1] - dea)


def test_calculator_matches_seeded_ema_macd_reference_and_reports_formula_parameters() -> None:
    closes = [Decimal(str(value)) for value in range(10, 45)]

    result = IndicatorCalculator().calculate(_points([str(value) for value in closes]), "daily")
    latest = result.snapshots[-1]
    dif, dea, histogram = _reference_macd(closes)

    assert latest.values["ema_12"] == pytest.approx(_reference_ema(closes, 12)[-1])
    assert latest.values["ema_26"] == pytest.approx(_reference_ema(closes, 26)[-1])
    assert latest.values["dif"] == pytest.approx(dif)
    assert latest.values["dea"] == pytest.approx(dea)
    assert latest.values["macd_histogram"] == pytest.approx(histogram)
    assert result.formula_label == "technical_indicators/v2_partial_window"
    assert result.parameters["macd_histogram_multiplier"] == 2


def test_calculator_honors_explicit_macd_histogram_multiplier() -> None:
    points = _points([str(value) for value in range(10, 45)])

    multiplier_one = IndicatorCalculator(IndicatorParameters(macd_histogram_multiplier=1)).calculate(
        points, "daily"
    )
    multiplier_two = IndicatorCalculator(IndicatorParameters(macd_histogram_multiplier=2)).calculate(
        points, "daily"
    )

    assert multiplier_one.snapshots[-1].values["macd_histogram"] * Decimal("2") == pytest.approx(
        multiplier_two.snapshots[-1].values["macd_histogram"]
    )


def test_calculator_seeds_all_chart_curves_from_the_first_available_price() -> None:
    result = IndicatorCalculator().calculate(
        _points(["10", "11", "12"], ["100", "200", "300"]),
        "monthly",
    )

    chart_keys = ("ma_5", "ma_10", "ma_20", "ma_60", "dif", "dea", "macd_histogram")
    assert all(snapshot.values[key] is not None for snapshot in result.snapshots for key in chart_keys)
    assert result.snapshots[0].values["ma_5"] == Decimal("10")
    assert result.snapshots[0].values["dif"] == Decimal("0")
    assert result.snapshots[0].values["dea"] == Decimal("0")
    assert result.snapshots[0].values["macd_histogram"] == Decimal("0")


def test_calculator_publishes_dif_first_change_without_filling_the_first_point() -> None:
    result = IndicatorCalculator().calculate(
        _points([str(value) for value in range(10, 45)]),
        "weekly",
    )

    assert result.snapshots[0].values["dif_first_change"] is None
    for previous, current in zip(result.snapshots, result.snapshots[1:]):
        assert current.values["dif_first_change"] == pytest.approx(
            current.values["dif"] - previous.values["dif"]
        )


def test_indicator_service_detects_and_repairs_legacy_payload_without_dif_change(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    sessions = create_session_factory(database)
    service = IndicatorService(sessions)
    with sessions() as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "159941"))
        assert instrument is not None
        for offset, close in enumerate(("10", "11", "12"), start=1):
            when = date(2026, 1, offset)
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    trade_date=when,
                    timeframe="daily",
                    open_price=Decimal(close),
                    high_price=Decimal(close),
                    low_price=Decimal(close),
                    close_price=Decimal(close),
                    volume=Decimal("100"),
                    source="TEST",
                )
            )
            session.add(
                IndicatorRecord(
                    instrument_id=instrument.id,
                    indicator_date=when,
                    timeframe="daily",
                    indicator_name="technical_indicators",
                    indicator_values={"values": {"dif": "0", "dea": "0", "macd_histogram": "0"}},
                )
            )

    assert service.stale_timeframes("159941") == ("daily",)
    result = service.recalculate_stale_timeframes("159941")
    assert len(result) == 1
    assert result[0].status is IndicatorCalculationStatus.SUCCESS
    assert service.stale_timeframes("159941") == ()
    with sessions() as session:
        rows = list(
            session.scalars(
                select(IndicatorRecord)
                .join(Instrument)
                .where(Instrument.code == "159941", IndicatorRecord.timeframe == "daily")
                .order_by(IndicatorRecord.indicator_date)
            )
        )
    assert rows[0].indicator_values["values"]["dif_first_change"] is None
    assert rows[1].indicator_values["values"]["dif_first_change"] is not None
    assert all(row.indicator_values["schema_version"] == 3 for row in rows)


def test_calculator_uses_available_early_windows_and_calculates_ma_rsi_volatility_and_drawdown() -> None:
    rising_then_falling = [str(value) for value in range(1, 22)] + ["10"]
    volumes = [str(value * 100) for value in range(1, 23)]

    result = IndicatorCalculator().calculate(_points(rising_then_falling, volumes), "daily")
    early = result.snapshots[4]
    latest = result.snapshots[-1]

    assert early.values["ma_5"] == Decimal("3")
    assert early.values["ma_10"] == Decimal("3")
    assert early.values["rsi_6"] == Decimal("100")
    assert latest.values["ma_20"] == Decimal("11.9")
    assert Decimal("0") < latest.values["rsi_6"] < Decimal("50")
    assert latest.values["volume_ma_20"] == Decimal("1250")
    assert latest.values["volatility_20"] is not None
    assert latest.values["current_drawdown"] == Decimal("10") / Decimal("21") - Decimal("1")
    assert latest.values["running_drawdown"] == latest.values["current_drawdown"]
    assert latest.metadata["cumulative_return"] == Decimal("9")
    assert latest.metadata["annualized_return"] is not None
    assert result.data_cutoff == date(2026, 1, 22)
    assert result.completeness["ma_20"] is True
    assert result.completeness["ma_60"] is True


def test_calculator_uses_only_current_and_prior_rows_even_when_input_is_out_of_order() -> None:
    initial = _points(["10", "11", "12", "13", "14", "15"])
    with_future = list(
        reversed(initial + [PricePoint(date(2026, 1, 7), Decimal("1000"), Decimal("1000"))])
    )

    expected = IndicatorCalculator().calculate(initial, "daily").snapshots[-1]
    actual = IndicatorCalculator().calculate(with_future, "daily").snapshots[5]

    assert actual.trade_date == expected.trade_date
    assert actual.values == expected.values
    assert actual.metadata == expected.metadata


def _seed_prices(database: Path, code: str, closes: list[str], timeframe: str = "daily") -> None:
    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(select(Instrument.id).where(Instrument.code == code)).scalar_one()
        for point in _points(closes, ["1000"] * len(closes)):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=point.trade_date,
                    timeframe=timeframe,
                    close_price=point.close,
                    volume=point.volume,
                )
            )


def test_indicator_service_persists_one_aggregate_record_per_price_and_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", [str(value) for value in range(1, 36)])
    service = IndicatorService(create_session_factory(database))

    first = service.recalculate("589850", "daily")
    second = service.recalculate("589850", "daily")
    engine = create_database_engine(database)
    with Session(engine) as session:
        records = session.scalars(
            select(IndicatorRecord)
            .where(IndicatorRecord.timeframe == "daily")
            .order_by(IndicatorRecord.indicator_date)
        ).all()

    assert first.status is IndicatorCalculationStatus.SUCCESS
    assert first.records_added == 35
    assert first.records_updated == 0
    assert first.records_skipped == 0
    assert second.records_added == 0
    assert second.records_updated == 0
    assert second.records_skipped == 35
    assert len(records) == 35
    assert records[-1].indicator_name == "technical_indicators"
    assert records[-1].indicator_values["values"]["ma_20"] == "25.5"
    assert records[-1].indicator_values["metadata"]["cumulative_return"] == "34"


def test_indicator_service_updates_only_changed_rows_without_duplicates(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", [str(value) for value in range(1, 36)])
    service = IndicatorService(create_session_factory(database))
    service.recalculate("589850", "daily")
    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(select(Instrument.id).where(Instrument.code == "589850")).scalar_one()
        connection.execute(
            MarketPrice.__table__.update()
            .where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.trade_date == date(2026, 2, 4),
                MarketPrice.timeframe == "daily",
            )
            .values(close_price=Decimal("70"))
        )

    result = service.recalculate("589850", "daily")
    with engine.connect() as connection:
        count = connection.execute(select(func.count()).select_from(IndicatorRecord)).scalar_one()

    assert result.records_added == 0
    assert result.records_updated == 1
    assert result.records_skipped == 34
    assert count == 35


def test_indicator_service_removes_stale_periods_after_the_price_endpoint_changes(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", ["10", "11"], timeframe="weekly")
    service = IndicatorService(create_session_factory(database))
    service.recalculate("589850", "weekly")

    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        connection.execute(
            MarketPrice.__table__.delete().where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.trade_date == date(2026, 1, 1),
                MarketPrice.timeframe == "weekly",
            )
        )

    service.recalculate("589850", "weekly")
    with Session(engine) as session:
        records = session.scalars(
            select(IndicatorRecord)
            .where(IndicatorRecord.timeframe == "weekly")
            .order_by(IndicatorRecord.indicator_date)
        ).all()

    assert [record.indicator_date for record in records] == [date(2026, 1, 2)]


def test_indicator_service_recalculates_every_chart_timeframe(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    for timeframe in ("daily", "weekly", "monthly"):
        _seed_prices(database, "589850", ["10", "11", "12"], timeframe=timeframe)
    service = IndicatorService(create_session_factory(database))

    results = service.recalculate_all_timeframes("589850")

    assert [result.timeframe for result in results] == ["daily", "weekly", "monthly"]
    assert all(result.status is IndicatorCalculationStatus.SUCCESS for result in results)
    assert [result.price_rows for result in results] == [3, 3, 3]


def test_ndx_indicator_service_ignores_legacy_volume_for_every_timeframe(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    for timeframe in ("daily", "weekly", "monthly"):
        _seed_prices(
            database,
            "NDX",
            [str(value) for value in range(20_000, 20_020)],
            timeframe=timeframe,
        )
    service = IndicatorService(create_session_factory(database))

    results = service.recalculate_all_timeframes("NDX")

    assert all(result.status is IndicatorCalculationStatus.SUCCESS for result in results)
    assert all(result.completeness["volume_ma_20"] is False for result in results)
    engine = create_database_engine(database)
    with Session(engine) as session:
        latest = {
            timeframe: session.scalar(
                select(IndicatorRecord)
                .join(Instrument)
                .where(
                    Instrument.code == "NDX",
                    IndicatorRecord.timeframe == timeframe,
                )
                .order_by(IndicatorRecord.indicator_date.desc())
            )
            for timeframe in ("daily", "weekly", "monthly")
        }
    assert all(
        record is not None
        and record.indicator_values["values"]["volume_ma_20"] is None
        for record in latest.values()
    )


def test_indicator_service_returns_typed_validation_and_no_data_results(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    service = IndicatorService(create_session_factory(database))

    unknown = service.recalculate("UNKNOWN", "daily")
    invalid_timeframe = service.recalculate("589850", "hourly")
    no_data = service.recalculate("589850", "daily")

    assert unknown.status is IndicatorCalculationStatus.VALIDATION_ERROR
    assert "Unknown instrument" in (unknown.error or "")
    assert invalid_timeframe.status is IndicatorCalculationStatus.VALIDATION_ERROR
    assert no_data.status is IndicatorCalculationStatus.NO_DATA
    assert no_data.data_cutoff is None


def test_indicator_service_no_data_preserves_last_good_indicator_rows(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", ["10", "11"])
    service = IndicatorService(create_session_factory(database))
    service.recalculate("589850", "daily")
    engine = create_database_engine(database)
    with Session(engine) as session:
        before = session.scalars(
            select(IndicatorRecord)
            .where(IndicatorRecord.timeframe == "daily")
            .order_by(IndicatorRecord.indicator_date)
        ).all()
        before_values = [
            (record.indicator_date, record.indicator_values)
            for record in before
        ]
    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        connection.execute(
            MarketPrice.__table__.delete().where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.timeframe == "daily",
            )
        )

    result = service.recalculate("589850", "daily")

    with Session(engine) as session:
        after = session.scalars(
            select(IndicatorRecord)
            .where(IndicatorRecord.timeframe == "daily")
            .order_by(IndicatorRecord.indicator_date)
        ).all()
    assert result.status is IndicatorCalculationStatus.NO_DATA
    assert [(record.indicator_date, record.indicator_values) for record in after] == before_values


def test_indicator_service_returns_typed_validation_for_invalid_stored_price_without_writes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", ["10"])
    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(select(Instrument.id).where(Instrument.code == "589850")).scalar_one()
        connection.execute(
            MarketPrice.__table__.insert().values(
                instrument_id=instrument_id,
                trade_date=date(2026, 1, 2),
                timeframe="daily",
                close_price=Decimal("0"),
                volume=Decimal("1000"),
            )
        )

    result = IndicatorService(create_session_factory(database)).recalculate("589850", "daily")
    with engine.connect() as connection:
        writes = connection.execute(select(func.count()).select_from(IndicatorRecord)).scalar_one()

    assert result.status is IndicatorCalculationStatus.VALIDATION_ERROR
    assert "close_price" in (result.error or "")
    assert "2026-01-02" in (result.error or "")
    assert writes == 0


def test_indicator_service_concurrent_recalculation_is_atomic_per_period(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", ["10"])
    barrier = Barrier(2)

    class BarrierIndicatorService(IndicatorService):
        def __init__(self) -> None:
            super().__init__(create_session_factory(database))
            self.wait_before_first_upsert = True

        def _payload(self, values, metadata):  # type: ignore[no-untyped-def]
            if self.wait_before_first_upsert:
                self.wait_before_first_upsert = False
                barrier.wait(timeout=10)
            return super()._payload(values, metadata)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(BarrierIndicatorService().recalculate, "589850", "daily") for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]

    engine = create_database_engine(database)
    with engine.connect() as connection:
        rows = connection.execute(select(func.count()).select_from(IndicatorRecord)).scalar_one()

    assert [result.status for result in results] == [
        IndicatorCalculationStatus.SUCCESS,
        IndicatorCalculationStatus.SUCCESS,
    ]
    assert sum(result.records_added for result in results) == 1
    assert sum(result.records_skipped for result in results) == 1
    assert rows == 1


def test_indicator_service_returns_typed_validation_for_volatility_period_below_two(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    _seed_prices(database, "589850", ["10", "11"])

    result = IndicatorService(
        create_session_factory(database), IndicatorParameters(volatility_period=1)
    ).recalculate("589850", "daily")
    engine = create_database_engine(database)
    with engine.connect() as connection:
        rows = connection.execute(select(func.count()).select_from(IndicatorRecord)).scalar_one()

    assert result.status is IndicatorCalculationStatus.VALIDATION_ERROR
    assert "volatility_period" in (result.error or "")
    assert rows == 0


def test_indicator_service_validates_volatility_period_before_no_data_handling(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")

    result = IndicatorService(
        create_session_factory(database), IndicatorParameters(volatility_period=1)
    ).recalculate("589850", "daily")

    assert result.status is IndicatorCalculationStatus.VALIDATION_ERROR
    assert "volatility_period" in (result.error or "")
    assert result.data_cutoff is None


def test_v3_migration_keeps_stable_earliest_duplicate_and_retains_all_legacy_values(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            Instrument.__table__.insert().values(
                code="LEGACY",
                name="Legacy ETF",
                exchange="SSE",
                category="etf",
                currency="CNY",
                is_active=True,
                extra_data={},
            )
        )
        instrument_id = connection.execute(select(Instrument.id).where(Instrument.code == "LEGACY")).scalar_one()
        connection.exec_driver_sql("DROP TABLE indicator_records")
        connection.exec_driver_sql(
            """
            CREATE TABLE indicator_records (
                id INTEGER NOT NULL PRIMARY KEY,
                instrument_id INTEGER NOT NULL REFERENCES instruments(id),
                indicator_date DATE NOT NULL,
                timeframe VARCHAR(16) NOT NULL,
                indicator_name VARCHAR(64) NOT NULL,
                indicator_value INTEGER,
                indicator_values JSON NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                CONSTRAINT uq_indicator_record UNIQUE
                    (instrument_id, indicator_date, timeframe, indicator_name)
            )
            """
        )
        values = (
            (10, instrument_id, "2026-01-02", "daily", "ma_5", 125_000_000, '{"source":"earliest"}'),
            (11, instrument_id, "2026-01-02", "daily", "rsi_6", 250_000_000, '{"source":"newer"}'),
            (12, instrument_id, "2026-01-03", "daily", "ma_5", 300_000_000, '{"source":"single"}'),
        )
        for value in values:
            connection.exec_driver_sql(
                """
                INSERT INTO indicator_records
                    (id, instrument_id, indicator_date, timeframe, indicator_name,
                     indicator_value, indicator_values, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, '2026-01-01 00:00:00', '2026-01-01 00:00:00')
                """,
                value,
            )
        connection.exec_driver_sql("PRAGMA user_version = 2")

    assert run_migrations(engine) == SCHEMA_VERSION
    with Session(engine) as session:
        migrated = session.scalars(
            select(IndicatorRecord).order_by(IndicatorRecord.indicator_date, IndicatorRecord.id)
        ).all()

    duplicate_period, single_period = migrated
    assert duplicate_period.id == 10  # Stable retention rule: lowest legacy id wins the physical row.
    assert duplicate_period.indicator_name == "technical_indicators"
    assert duplicate_period.indicator_value is None
    assert duplicate_period.indicator_values["legacy_records"] == [
        {
            "id": 10,
            "indicator_name": "ma_5",
            "indicator_value": "1.25",
            "indicator_values": {"source": "earliest"},
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        },
        {
            "id": 11,
            "indicator_name": "rsi_6",
            "indicator_value": "2.5",
            "indicator_values": {"source": "newer"},
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        },
    ]
    assert single_period.id == 12
    assert single_period.indicator_name == "ma_5"
    assert single_period.indicator_value == Decimal("3")

    with engine.connect() as connection:
        unique_indexes = [
            tuple(
                row[2]
                for row in connection.exec_driver_sql(f'PRAGMA index_info("{index[1]}")')
            )
            for index in connection.exec_driver_sql("PRAGMA index_list(indicator_records)")
            if index[2]
        ]
    assert ("instrument_id", "indicator_date", "timeframe") in unique_indexes
    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(
            IndicatorRecord.__table__.insert().values(
                instrument_id=instrument_id,
                indicator_date=date(2026, 1, 2),
                timeframe="daily",
                indicator_name="another",
                indicator_values={},
            )
        )
