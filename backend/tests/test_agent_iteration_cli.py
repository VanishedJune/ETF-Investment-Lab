from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import agent_iteration as agent_iteration_cli
from backend.app.agent_iterations import runner
from backend.app.agent_iterations.model import initial_model
from backend.app.agent_iterations.runner import (
    PriceBar,
    _current_advice,
    append_due_iterations,
    run_progressive_baseline,
)
from backend.app.agent_iterations.storage import AgentIterationStorage


def _series(size: int = 1000) -> tuple[list[PriceBar], list[PriceBar]]:
    start = date(2018, 1, 1)
    daily: list[PriceBar] = []
    cursor = start
    index = 0
    while len(daily) < size:
        if cursor.weekday() < 5:
            cycle = Decimal((index % 40) - 20)
            close = Decimal("1000") + Decimal(index) * Decimal("0.7") + cycle * Decimal("1.5")
            daily.append(
                PriceBar(
                    trade_date=cursor,
                    open=close - Decimal("2"),
                    high=close + Decimal("5"),
                    low=close - Decimal("5"),
                    close=close,
                    volume=Decimal("1000000") + Decimal(index * 100),
                    source="TEST_DAILY",
                )
            )
            index += 1
        cursor += timedelta(days=1)
    weekly: list[PriceBar] = []
    groups: dict[tuple[int, int], list[PriceBar]] = {}
    for bar in daily:
        year, week, _ = bar.trade_date.isocalendar()
        groups.setdefault((year, week), []).append(bar)
    for bars in groups.values():
        weekly.append(
            PriceBar(
                trade_date=bars[-1].trade_date,
                open=bars[0].open,
                high=max(item.high for item in bars),
                low=min(item.low for item in bars),
                close=bars[-1].close,
                volume=sum((item.volume or Decimal("0")) for item in bars),
                source="AGGREGATED_DAILY_VOLUME:TEST_DAILY",
            )
        )
    return weekly, daily


def test_agent_baseline_is_progressive_auditable_and_persisted_outside_web_backend(tmp_path) -> None:
    weekly, daily = _series()
    storage = AgentIterationStorage(tmp_path / "model_iterations")

    manifest = run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="Codex-test",
    )

    assert manifest["completed_iterations"] == 5
    assert manifest["latest_model_version"] == "M005"
    assert manifest["execution_mode"] == "agent_offline"
    assert manifest["current_advice"]["status"] == "pending"
    records = storage.read_iterations("399006")
    assert len(records) == manifest["completed_iterations"]
    assert [item["model"]["version"] for item in records] == ["M001", "M002", "M003", "M004", "M005"]
    assert records[1]["model"]["parent_version"] == "M001"
    assert records[1]["model"]["feedback_count"] == 1
    assert all(item["source_data_max_date"] <= item["cutoff_date"] for item in records)
    assert all(item["future_label_max_date"] > item["cutoff_date"] for item in records)
    assert all(item["fund_etf_ratio"] in {"7:3", "6:4", "5:5"} for item in records)
    assert manifest["current_advice"]["cutoff_date"] >= records[-1]["cutoff_date"]

    stored_manifest = json.loads((tmp_path / "model_iterations" / "399006" / "manifest.json").read_text("utf-8"))
    assert stored_manifest["completed_iterations"] == 5


@pytest.mark.parametrize(
    ("as_of", "current_bar_date", "expected_cutoff"),
    [
        (date(2026, 7, 30), date(2026, 7, 30), "2026-07-24"),
        (date(2026, 7, 31), date(2026, 7, 31), "2026-07-24"),
        (date(2026, 7, 31), date(2026, 7, 30), "2026-07-24"),
        (date(2026, 8, 1), date(2026, 7, 30), "2026-07-30"),
        (date(2026, 1, 1), date(2025, 12, 31), "2025-12-26"),
        (date(2026, 1, 3), date(2025, 12, 31), "2025-12-31"),
    ],
)
def test_current_advice_uses_m100_and_only_completed_weeks(
    as_of: date,
    current_bar_date: date,
    expected_cutoff: str,
) -> None:
    weekly, _ = _series()
    template = weekly[-1]
    previous_bar_date = date(2025, 12, 26) if as_of.year == 2026 and as_of.month == 1 else date(2026, 7, 24)
    weekly.extend(
        [
            replace(template, trade_date=previous_bar_date, close=Decimal("1700")),
            replace(template, trade_date=current_bar_date, close=Decimal("1800")),
        ]
    )
    model = replace(
        initial_model("399006"),
        iteration=100,
        version="M100",
        parent_version="M099",
        feedback_count=99,
    )

    advice = _current_advice(
        model=model,
        current_position=50,
        weekly=weekly,
        valuation_percentiles={},
        as_of=as_of,
    )

    assert advice["status"] == "pending"
    assert advice["model_version"] == "M100"
    assert advice["cutoff_date"] == expected_cutoff


def test_agent_storage_status_reports_due_without_mutating_model_files(tmp_path) -> None:
    storage = AgentIterationStorage(tmp_path / "model_iterations")

    status = storage.status("NDX", required_baseline=100, monthly_increment=2)

    assert status == {
        "instrument_code": "NDX",
        "completed_iterations": 0,
        "latest_model_version": None,
        "required_baseline": 100,
        "baseline_remaining": 100,
        "monthly_increment": 2,
        "execution_mode": "agent_offline",
        "baseline_status": "incomplete",
    }
    assert not (tmp_path / "model_iterations" / "NDX").exists()


def test_second_baseline_is_unchanged_and_does_not_rewrite_artifacts(tmp_path) -> None:
    weekly, daily = _series()
    root = tmp_path / "model_iterations"
    storage = AgentIterationStorage(root)
    run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="first-run",
    )
    manifest_path = root / "399006" / "manifest.json"
    iterations_path = root / "399006" / "iterations.jsonl"
    before = (manifest_path.read_bytes(), iterations_path.read_bytes())

    result = run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="second-run-must-not-be-persisted",
    )

    assert result["operation_status"] == "unchanged"
    assert (manifest_path.read_bytes(), iterations_path.read_bytes()) == before
    assert storage.status("399006", required_baseline=5, monthly_increment=2)["baseline_status"] == "unchanged"


@pytest.mark.parametrize(
    ("corruption", "expected_message"),
    [
        ("parent_chain", "parent_version"),
        ("cutoff_order", "cutoff_date"),
        ("source_leakage", "source_data_max_date"),
        ("ratio", "fund_etf_ratio"),
        ("fractional_allocation", "fund_allocation"),
        ("allocation_sum", "allocation"),
        ("latest_model_mismatch", "latest_model"),
        ("label_overlap", "future_label_max_date"),
        ("instrument_identity", "instrument_code"),
    ],
)
def test_baseline_rejects_internally_corrupt_completed_artifacts(
    tmp_path,
    corruption: str,
    expected_message: str,
) -> None:
    weekly, daily = _series()
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    manifest = run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="first-run",
    )
    records = storage.read_iterations("399006")
    if corruption == "parent_chain":
        records[1]["model"]["parent_version"] = "M000"
    elif corruption == "cutoff_order":
        records[1]["cutoff_date"] = records[0]["cutoff_date"]
    elif corruption == "source_leakage":
        records[1]["source_data_max_date"] = "9999-12-31"
    elif corruption == "ratio":
        records[1]["fund_etf_ratio"] = "8:2"
    elif corruption == "fractional_allocation":
        records[1]["fund_allocation"] = 12.5
    elif corruption == "allocation_sum":
        records[1]["etf_allocation"] += 1
    elif corruption == "latest_model_mismatch":
        manifest["latest_model"]["weights"]["trend"] = "999"
    elif corruption == "label_overlap":
        records[0]["future_label_max_date"] = records[1]["cutoff_date"]
    elif corruption == "instrument_identity":
        manifest["instrument_code"] = "NDX"
        manifest["latest_model"]["instrument_code"] = "NDX"
        for record in records:
            record["instrument_code"] = "NDX"
            record["model"]["instrument_code"] = "NDX"
    storage.write("399006", manifest=manifest, iterations=records)

    with pytest.raises(runner.ArtifactConsistencyError, match=expected_message):
        run_progressive_baseline(
            instrument_code="399006",
            weekly_bars=weekly,
            daily_bars=daily,
            valuation_percentiles={},
            storage=storage,
            count=5,
            seed=20260730,
            executed_by="must-fail",
        )


def test_baseline_refuses_to_overwrite_inconsistent_artifacts(tmp_path) -> None:
    weekly, daily = _series()
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    manifest = run_progressive_baseline(
        instrument_code="NDX",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="first-run",
    )
    storage.write("NDX", manifest={**manifest, "completed_iterations": 6}, iterations=storage.read_iterations("NDX"))

    with pytest.raises(runner.ArtifactConsistencyError, match="completed_iterations"):
        run_progressive_baseline(
            instrument_code="NDX",
            weekly_bars=weekly,
            daily_bars=daily,
            valuation_percentiles={},
            storage=storage,
            count=5,
            seed=20260730,
            executed_by="must-fail",
        )


def test_baseline_refuses_to_overwrite_consistent_but_incomplete_artifacts(tmp_path) -> None:
    weekly, daily = _series()
    root = tmp_path / "model_iterations"
    storage = AgentIterationStorage(root)
    run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="partial-run",
    )
    before = (
        (root / "399006" / "manifest.json").read_bytes(),
        (root / "399006" / "iterations.jsonl").read_bytes(),
    )

    with pytest.raises(runner.ArtifactConsistencyError, match="incomplete"):
        run_progressive_baseline(
            instrument_code="399006",
            weekly_bars=weekly,
            daily_bars=daily,
            valuation_percentiles={},
            storage=storage,
            count=6,
            seed=20260730,
            executed_by="must-not-overwrite",
        )

    assert (
        (root / "399006" / "manifest.json").read_bytes(),
        (root / "399006" / "iterations.jsonl").read_bytes(),
    ) == before


def test_monthly_due_appends_two_feedback_driven_versions_without_rewriting_baseline(tmp_path) -> None:
    weekly, daily = _series()
    early_daily = daily[:800]
    early_dates = {item.trade_date for item in early_daily}
    early_weekly = [item for item in weekly if item.trade_date in early_dates]
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    run_progressive_baseline(
        instrument_code="NDX",
        weekly_bars=early_weekly,
        daily_bars=early_daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="Codex-baseline",
    )
    baseline_records = storage.read_iterations("NDX")

    manifest = append_due_iterations(
        instrument_code="NDX",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=2,
        executed_by="Codex-monthly",
        as_of=daily[-1].trade_date,
    )

    records = storage.read_iterations("NDX")
    assert records[:5] == baseline_records
    assert manifest["completed_iterations"] == 7
    assert manifest["latest_model_version"] == "M007"
    assert records[5]["model"]["version"] == "M006"
    assert records[5]["model"]["parent_version"] == "M005"
    assert records[6]["model"]["version"] == "M007"
    assert records[6]["model"]["parent_version"] == "M006"
    period = daily[-1].trade_date.strftime("%Y-%m")
    assert manifest["maintenance_ledger"][period]["count"] == 2


def test_monthly_due_is_idempotent_within_month_and_adds_again_next_month(tmp_path) -> None:
    weekly, daily = _series(1200)
    early_daily = daily[:800]
    early_dates = {item.trade_date for item in early_daily}
    early_weekly = [item for item in weekly if item.trade_date in early_dates]
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    run_progressive_baseline(
        instrument_code="399006",
        weekly_bars=early_weekly,
        daily_bars=early_daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="Codex-baseline",
    )
    june = date(2021, 6, 30)
    july = date(2021, 7, 31)

    june_manifest = append_due_iterations(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=2,
        executed_by="June-run",
        as_of=june,
    )
    june_bytes = (
        (tmp_path / "model_iterations" / "399006" / "manifest.json").read_bytes(),
        (tmp_path / "model_iterations" / "399006" / "iterations.jsonl").read_bytes(),
    )
    repeated = append_due_iterations(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=2,
        executed_by="June-repeat",
        as_of=june,
    )

    assert june_manifest["completed_iterations"] == 7
    assert repeated["operation_status"] == "unchanged"
    assert (
        (tmp_path / "model_iterations" / "399006" / "manifest.json").read_bytes(),
        (tmp_path / "model_iterations" / "399006" / "iterations.jsonl").read_bytes(),
    ) == june_bytes

    july_manifest = append_due_iterations(
        instrument_code="399006",
        weekly_bars=weekly,
        daily_bars=daily,
        valuation_percentiles={},
        storage=storage,
        count=2,
        executed_by="July-run",
        as_of=july,
    )

    assert july_manifest["completed_iterations"] == 9
    assert july_manifest["latest_model_version"] == "M009"
    assert july_manifest["maintenance_ledger"]["2021-06"]["count"] == 2
    assert july_manifest["maintenance_ledger"]["2021-07"]["count"] == 2


def test_due_raises_only_dedicated_error_when_mature_data_is_unavailable(tmp_path) -> None:
    weekly, daily = _series()
    early_daily = daily[:800]
    early_dates = {item.trade_date for item in early_daily}
    early_weekly = [item for item in weekly if item.trade_date in early_dates]
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    run_progressive_baseline(
        instrument_code="NDX",
        weekly_bars=early_weekly,
        daily_bars=early_daily,
        valuation_percentiles={},
        storage=storage,
        count=5,
        seed=20260730,
        executed_by="Codex-baseline",
    )

    with pytest.raises(runner.InsufficientMatureData):
        append_due_iterations(
            instrument_code="NDX",
            weekly_bars=weekly,
            daily_bars=daily,
            valuation_percentiles={},
            storage=storage,
            count=2,
            executed_by="too-early",
            as_of=early_daily[-1].trade_date,
        )


def test_cli_accepts_as_of_for_deterministic_due_checks() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "agent_iteration.py"),
            "--mode",
            "status",
            "--as-of",
            "2026-07-30",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_due_cli_only_converts_insufficient_mature_data_to_unchanged(monkeypatch) -> None:
    current = [{
        "instrument_code": "399006",
        "completed_iterations": 100,
        "latest_model_version": "M100",
        "required_baseline": 100,
        "baseline_remaining": 0,
        "monthly_increment": 2,
        "execution_mode": "agent_offline",
        "baseline_status": "unchanged",
    }]
    monkeypatch.setattr(
        agent_iteration_cli,
        "_policy",
        lambda: {
            "instruments": ["399006"],
            "baseline_iterations_per_market": 100,
            "monthly_iterations_per_market": 2,
        },
    )
    monkeypatch.setattr(agent_iteration_cli, "status", lambda: current)
    monkeypatch.setattr(agent_iteration_cli, "_repair_periods", lambda _code: None)
    monkeypatch.setattr(agent_iteration_cli, "_load_series", lambda _code: ([], [], {}))

    def corrupt_artifacts(**_kwargs):
        raise ValueError("corrupt artifacts")

    monkeypatch.setattr(agent_iteration_cli, "append_due_iterations", corrupt_artifacts)
    with pytest.raises(ValueError, match="corrupt artifacts"):
        agent_iteration_cli.due("test", date(2026, 7, 30))

    def insufficient_data(**_kwargs):
        raise runner.InsufficientMatureData("no mature data")

    monkeypatch.setattr(agent_iteration_cli, "append_due_iterations", insufficient_data)
    result = agent_iteration_cli.due("test", date(2026, 7, 30))

    assert result[0]["operation_status"] == "unchanged"
    assert result[0]["latest_model_version"] == "M100"


def test_due_cli_checks_monthly_quota_before_repairing_market_data(monkeypatch) -> None:
    current = [{
        "instrument_code": "399006",
        "completed_iterations": 102,
        "latest_model_version": "M102",
        "required_baseline": 100,
        "baseline_remaining": 0,
        "monthly_increment": 2,
        "execution_mode": "agent_offline",
        "baseline_status": "unchanged",
    }]

    class FullQuotaStorage:
        def __init__(self, _root) -> None:
            pass

        def read_consistent_snapshot(self, _code):
            return (
                {
                    "maintenance_ledger": {
                        "2026-07": {"count": 2, "runs": []},
                    },
                },
                [],
            )

    repair_calls: list[str] = []
    monkeypatch.setattr(
        agent_iteration_cli,
        "_policy",
        lambda: {
            "instruments": ["399006"],
            "baseline_iterations_per_market": 100,
            "monthly_iterations_per_market": 2,
        },
    )
    monkeypatch.setattr(agent_iteration_cli, "status", lambda: current)
    monkeypatch.setattr(agent_iteration_cli, "AgentIterationStorage", FullQuotaStorage)
    monkeypatch.setattr(agent_iteration_cli, "_repair_periods", lambda code: repair_calls.append(code))
    monkeypatch.setattr(
        agent_iteration_cli,
        "_load_series",
        lambda _code: (_ for _ in ()).throw(AssertionError("must not load series")),
    )

    result = agent_iteration_cli.due("test", date(2026, 7, 30))

    assert repair_calls == []
    assert result[0]["operation_status"] == "unchanged"
    assert result[0]["message"] == "Monthly maintenance quota for 2026-07 is already exhausted."
