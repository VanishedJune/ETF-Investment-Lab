"""Creation of local JSON configuration defaults.

The files are deliberately created only when absent so local adjustments remain
owned by the user.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .core.paths import config_directory as default_config_directory


DEFAULT_APP_CONFIG: dict[str, Any] = {
    "app_name": "ETF Investment Research Lab",
    "locale": "zh-CN",
    "timezone": "Asia/Shanghai",
    "database_path": "data/investment_lab.db",
    "data_mode": "local_only",
}

DEFAULT_FEE_CONFIG: dict[str, Any] = {
    "buy_commission_rate": 0.00025,
    "sell_commission_rate": 0.00025,
    "minimum_commission": 5,
    "minimum_commission_enabled": True,
    "etf_stamp_duty_rate": 0,
    "transfer_fee_rate": 0,
    "other_fee_rate": 0,
}

DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "version": "1.0",
    "weights": {
        "valuation": 0.25,
        "trend": 0.25,
        "momentum": 0.15,
        "volume": 0.10,
        "volatility": 0.10,
        "risk": 0.15,
    },
    "thresholds": {
        "valuation_low_percentile": 0.3,
        "valuation_high_percentile": 0.7,
        "momentum_rsi_low": 35,
        "momentum_rsi_high": 70,
        "volume_ratio_low": 0.8,
        "volume_ratio_high": 1.2,
        "volatility_high": 0.35,
        "drawdown_pause": -0.2,
        "increase_score": 0.35,
        "reduce_score": -0.2,
        "sell_partial_score": -0.45,
        "pause_score": -0.75,
    },
    "multipliers": {
        "increase": 1.3,
        "normal": 1.0,
        "reduce": 0.75,
        "pause": 0.5,
        "hold": 1.0,
        "sell_partial": 0.7,
    },
    "maximum_sell_ratio": 0.3,
    "minimum_holding_ratio": 0.2,
}

DEFAULT_WATCHLIST: dict[str, Any] = {
    "instruments": [
        {"code": "589850", "name": "科创50ETF东财", "exchange": "SSE"},
        {"code": "159205", "name": "创业板ETF东财", "exchange": "SZSE"},
        {"code": "159941", "name": "纳指ETF广发", "exchange": "SZSE"},
    ]
}

DEFAULT_CONFIG_FILES: dict[str, dict[str, Any]] = {
    "app_config.json": DEFAULT_APP_CONFIG,
    "fee_config.json": DEFAULT_FEE_CONFIG,
    "strategy_config.json": DEFAULT_STRATEGY_CONFIG,
    "watchlist.json": DEFAULT_WATCHLIST,
}


class ConfigurationError(RuntimeError):
    """Raised when a user-owned configuration file is invalid."""


def _validate_existing_config(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid JSON configuration: {path}") from error


def _write_json_atomically(path: Path, contents: dict[str, Any]) -> None:
    """Fsync a same-directory temporary file before replacing the destination."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(json.dumps(contents, ensure_ascii=False, indent=2) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def ensure_default_configs(directory: Path | str | None = None) -> Path:
    """Create absent default JSON files and return their directory.

    Existing configuration files are intentionally never read-modify-written.
    """
    target_directory = Path(directory) if directory is not None else default_config_directory()
    target_directory.mkdir(parents=True, exist_ok=True)
    for filename, contents in DEFAULT_CONFIG_FILES.items():
        target = target_directory / filename
        if target.exists():
            _validate_existing_config(target)
        else:
            _write_json_atomically(target, contents)
    return target_directory
