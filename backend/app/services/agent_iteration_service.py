from __future__ import annotations

from typing import Any

from backend.app.agent_iterations.storage import AgentIterationStorage


class AgentIterationService:
    """Read-only projection of artifacts produced by an external Agent run."""

    def __init__(self, storage: AgentIterationStorage) -> None:
        self.storage = storage

    @staticmethod
    def _validate(instrument_code: str) -> None:
        if instrument_code not in {"399006", "NDX"}:
            raise ValueError(f"Unsupported iteration instrument: {instrument_code}")

    def summary(self, instrument_code: str) -> dict[str, Any]:
        self._validate(instrument_code)
        manifest, records = self.storage.read_consistent_snapshot(instrument_code)
        if manifest is None:
            return {
                "manifest": {
                    "instrument_code": instrument_code,
                    "completed_iterations": 0,
                    "latest_model_version": None,
                    "execution_mode": "agent_offline",
                },
                "curve": [],
                "latest_advice": None,
            }
        return {
            "manifest": manifest,
            "curve": [
                {
                    "iteration": item["iteration"],
                    "cutoff_date": item["cutoff_date"],
                    "deviation_days": item["deviation_days"],
                    "direction_correct": item["direction_correct"],
                }
                for item in records
            ],
            "latest_advice": manifest.get("current_advice"),
        }

    def detail(self, instrument_code: str, iteration_number: int) -> dict[str, Any]:
        self._validate(instrument_code)
        _manifest, records = self.storage.read_consistent_snapshot(instrument_code)
        match = next(
            (
                item
                for item in records
                if int(item.get("iteration", -1)) == iteration_number
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Iteration {iteration_number} not found for {instrument_code}")
        return match
