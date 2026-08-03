from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.errors import PublicValidationError
from backend.web import (
    V33ModelActionRequest,
    analyze_v33_model,
    latest_v33_forecast,
    train_v33_model,
    v33_iterations,
    v33_model_status,
)


class _Runtime:
    def __init__(self, *, bootstrapped: bool = False) -> None:
        self.bootstrapped = bootstrapped
        self.calls: list[tuple[str, str | None]] = []

    def model_statuses(self):
        return {
            "implementation_revision": "V3.3-20W",
            "markets": {
                "399006": {"instrument_code": "399006", "bootstrapped": self.bootstrapped},
                "159941": {"instrument_code": "159941", "bootstrapped": self.bootstrapped},
            },
        }

    def model_status(self, market: str):
        self.calls.append(("status", market))
        return {"bootstrapped": self.bootstrapped}

    def start_bootstrap(self, market: str):
        self.calls.append(("bootstrap", market))
        return {"status": "running", "run_id": "bootstrap-run", "market": market}

    def start_incremental(self, market: str):
        self.calls.append(("incremental", market))
        return {"status": "running", "run_id": "incremental-run", "market": market}

    def run_analysis(self, market: str):
        self.calls.append(("analysis", market))
        return {
            "implementation_revision": "V3.3-20W",
            "market": market,
            "training_mutated": False,
        }

    def curve(self, market: str):
        self.calls.append(("curve", market))
        return {"market": market, "curve": [], "iterations": []}

    def latest_analysis(self, market: str):
        self.calls.append(("latest", market))
        return {"implementation_revision": "V3.3-20W", "market": market}


def _request(runtime: _Runtime):
    state = SimpleNamespace(v33_runtime=runtime)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_v33_frontend_contract_selects_bootstrap_then_incremental() -> None:
    bootstrap = _Runtime(bootstrapped=False)
    payload = train_v33_model(
        _request(bootstrap), V33ModelActionRequest(instrument_code="159941")
    )
    assert payload["implementation_revision"] == "V3.3-20W"
    assert payload["run_id"] == "bootstrap-run"
    assert bootstrap.calls == [("status", "159941"), ("bootstrap", "159941")]

    incremental = _Runtime(bootstrapped=True)
    payload = train_v33_model(
        _request(incremental), V33ModelActionRequest(instrument_code="399006")
    )
    assert payload["run_id"] == "incremental-run"
    assert incremental.calls == [("status", "399006"), ("incremental", "399006")]


def test_v33_analysis_is_inference_only_and_159941_never_maps_to_ndx() -> None:
    runtime = _Runtime(bootstrapped=True)
    payload = analyze_v33_model(
        _request(runtime), V33ModelActionRequest(instrument_code="159941")
    )
    assert payload["market"] == "159941"
    assert payload["training_mutated"] is False
    assert runtime.calls == [("analysis", "159941")]

    with pytest.raises(PublicValidationError):
        analyze_v33_model(
            _request(runtime), V33ModelActionRequest(instrument_code="NDX")
        )


def test_v33_status_curve_and_latest_forecast_match_frontend_routes() -> None:
    runtime = _Runtime(bootstrapped=True)
    request = _request(runtime)
    assert set(v33_model_status(request)["markets"]) == {"399006", "159941"}
    assert v33_iterations(request, "159941")["market"] == "159941"
    assert latest_v33_forecast(request, "399006")["market"] == "399006"
    assert runtime.calls == [("curve", "159941"), ("latest", "399006")]
