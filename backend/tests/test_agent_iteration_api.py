from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.agent_iterations.storage import AgentIterationStorage
from backend.app.agent_iterations import storage as storage_module
from backend.app.services.agent_iteration_service import AgentIterationService


def test_read_only_iteration_service_returns_summary_curve_and_detail(tmp_path) -> None:
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    records = [
        {
            "iteration": index,
            "instrument_code": "399006",
            "cutoff_date": f"2025-{index:02d}-01",
            "future_label_max_date": f"2025-{index:02d}-15",
            "deviation_days": index - 2,
            "direction_correct": index % 2 == 0,
            "recommendation": "分批加仓",
            "target_position": 70,
            "fund_etf_ratio": "7:3",
            "fund_allocation": 49,
            "etf_allocation": 21,
            "source_data_max_date": f"2025-{index:02d}-01",
            "model": {
                "instrument_code": "399006",
                "iteration": index,
                "version": f"M{index:03d}",
                "parent_version": None if index == 1 else f"M{index - 1:03d}",
            },
        }
        for index in range(1, 4)
    ]
    storage.write(
        "399006",
        manifest={
            "instrument_code": "399006",
            "completed_iterations": 3,
            "latest_model_version": "M003",
            "latest_model": records[-1]["model"],
            "direction_accuracy": 0.33,
            "mean_absolute_deviation": 0.67,
            "median_absolute_deviation": 1,
            "execution_mode": "agent_offline",
        },
        iterations=records,
    )
    service = AgentIterationService(storage)

    payload = service.summary("399006")
    detail = service.detail("399006", 2)

    assert payload["manifest"]["latest_model_version"] == "M003"
    assert payload["curve"] == [
        {"iteration": 1, "cutoff_date": "2025-01-01", "deviation_days": -1, "direction_correct": False},
        {"iteration": 2, "cutoff_date": "2025-02-01", "deviation_days": 0, "direction_correct": True},
        {"iteration": 3, "cutoff_date": "2025-03-01", "deviation_days": 1, "direction_correct": False},
    ]
    assert payload["latest_advice"] is None
    assert detail["model"]["version"] == "M002"


def test_iteration_service_rejects_unknown_market_and_iteration(tmp_path) -> None:
    service = AgentIterationService(AgentIterationStorage(tmp_path / "model_iterations"))

    try:
        service.summary("000688")
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("unknown market must be rejected")

    try:
        service.detail("NDX", 1)
    except ValueError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("missing iteration must be rejected")


def test_iteration_summary_prefers_pending_current_advice_over_last_completed_record(tmp_path) -> None:
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    current_advice = {
        "status": "pending",
        "cutoff_date": "2026-07-24",
        "model_version": "M100",
        "recommendation": "保持观察",
    }
    storage.write(
        "399006",
        manifest={
            "instrument_code": "399006",
                "completed_iterations": 100,
                "latest_model_version": "M100",
                "latest_model": {
                    "instrument_code": "399006",
                    "iteration": 100,
                    "version": "M100",
                    "parent_version": "M099",
                },
                "current_advice": current_advice,
        },
        iterations=[
            {
                "iteration": index,
                    "instrument_code": "399006",
                    "cutoff_date": (date(2020, 1, 1) + timedelta(days=index * 7)).isoformat(),
                    "future_label_max_date": (
                        date(2020, 1, 2) + timedelta(days=index * 7)
                    ).isoformat(),
                    "source_data_max_date": (
                        date(2020, 1, 1) + timedelta(days=index * 7)
                    ).isoformat(),
                    "deviation_days": 0,
                    "direction_correct": True,
                "recommendation": "分批加仓",
                    "target_position": 70,
                    "fund_etf_ratio": "7:3",
                    "fund_allocation": 49,
                    "etf_allocation": 21,
                    "model": {
                        "instrument_code": "399006",
                        "iteration": index,
                        "version": f"M{index:03d}",
                        "parent_version": None if index == 1 else f"M{index - 1:03d}",
                    },
            }
            for index in range(1, 101)
        ],
    )

    payload = AgentIterationService(storage).summary("399006")

    assert payload["latest_advice"] == current_advice
    assert payload["manifest"]["completed_iterations"] == 100


def test_iteration_summary_retries_once_then_rejects_inconsistent_artifacts(tmp_path, monkeypatch) -> None:
    storage = AgentIterationStorage(tmp_path / "model_iterations")
    storage.write(
        "NDX",
        manifest={
            "instrument_code": "NDX",
            "completed_iterations": 2,
            "latest_model_version": "M002",
        },
        iterations=[
            {
                "iteration": 1,
                "cutoff_date": "2026-01-01",
                "deviation_days": 0,
                "direction_correct": True,
                "model": {"version": "M001"},
            }
        ],
    )
    calls = {"manifest": 0, "iterations": 0}
    original_manifest = storage.read_manifest
    original_iterations = storage.read_iterations

    def counted_manifest(instrument_code: str):
        calls["manifest"] += 1
        return original_manifest(instrument_code)

    def counted_iterations(instrument_code: str):
        calls["iterations"] += 1
        return original_iterations(instrument_code)

    monkeypatch.setattr(storage, "read_manifest", counted_manifest)
    monkeypatch.setattr(storage, "read_iterations", counted_iterations)

    with pytest.raises(storage_module.ArtifactConsistencyError, match="completed_iterations"):
        AgentIterationService(storage).summary("NDX")

    assert calls == {"manifest": 2, "iterations": 2}
