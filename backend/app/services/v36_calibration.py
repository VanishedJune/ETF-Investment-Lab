"""V3.6 probability calibration: TEMPERATURE / PLATT / ISOTONIC + selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _brier(probabilities: Sequence[Sequence[float]], actuals: Sequence[int]) -> float:
    total = 0.0
    for row, actual in zip(probabilities, actuals):
        target = [0.0, 0.0, 0.0]
        target[int(actual)] = 1.0
        total += 0.5 * float(np.sum((np.asarray(row) - np.asarray(target)) ** 2))
    return total / max(len(probabilities), 1)


def _fit_platt(
    probabilities: Sequence[Sequence[float]], actuals: Sequence[int]
) -> tuple[float, float]:
    logits = np.log(
        np.clip(
            np.asarray([row[0] for row in probabilities], dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
        / np.clip(
            1.0 - np.asarray([row[0] for row in probabilities], dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
    )
    targets = np.asarray([1.0 if actual == 0 else 0.0 for actual in actuals])
    a = 0.0
    b = 0.0
    for _ in range(40):
        scores = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        error = targets - scores
        gradient = np.array(
            [float(np.sum(error * logits)), float(np.sum(error))]
        )
        weight = np.clip(scores * (1.0 - scores), 1e-9, 1.0)
        hessian = np.array(
            [
                [float(np.sum(weight * logits * logits)), float(np.sum(weight * logits))],
                [float(np.sum(weight * logits)), float(np.sum(weight))],
            ]
        )
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            break
        a += step[0]
        b += step[1]
    return float(a), float(b)


def _fit_isotonic(
    probabilities: Sequence[Sequence[float]], actuals: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row[0] for row in probabilities], dtype=float)
    y = np.asarray([1.0 if actual == 0 else 0.0 for actual in actuals])
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Pool adjacent violators.
    blocks_x: list[float] = []
    blocks_y: list[float] = []
    blocks_n: list[int] = []
    for xv, yv in zip(x_sorted, y_sorted):
        blocks_x.append(xv)
        blocks_y.append(yv)
        blocks_n.append(1)
        while len(blocks_y) >= 2 and blocks_y[-1] < blocks_y[-2]:
            merged_n = blocks_n[-2] + blocks_n[-1]
            merged_y = (
                blocks_y[-2] * blocks_n[-2] + blocks_y[-1] * blocks_n[-1]
            ) / merged_n
            blocks_y[-2] = merged_y
            blocks_n[-2] = merged_n
            blocks_x[-2] = blocks_x[-1]
            blocks_y.pop()
            blocks_n.pop()
            blocks_x.pop()
    return np.asarray(blocks_x), np.asarray(blocks_y)


def _isotonic_apply(
    xs: np.ndarray, ys: np.ndarray, value: float
) -> float:
    if value <= xs[0]:
        return float(ys[0])
    if value >= xs[-1]:
        return float(ys[-1])
    index = int(np.searchsorted(xs, value, side="right") - 1)
    index = max(0, min(len(xs) - 2, index))
    span = max(xs[index + 1] - xs[index], 1e-12)
    fraction = (value - xs[index]) / span
    return float(ys[index] + fraction * (ys[index + 1] - ys[index]))


def fit_method(
    method: str,
    probabilities: Sequence[Sequence[float]],
    actuals: Sequence[int],
) -> dict[str, Any]:
    if method == "PLATT":
        a, b = _fit_platt(probabilities, actuals)
        return {"method": "PLATT", "a": a, "b": b}
    if method == "ISOTONIC":
        xs, ys = _fit_isotonic(probabilities, actuals)
        return {
            "method": "ISOTONIC",
            "xs": xs.tolist(),
            "ys": ys.tolist(),
        }
    return {"method": "TEMPERATURE"}


def apply_method(
    params: Mapping[str, Any],
    probabilities: Sequence[float],
) -> list[float]:
    method = str(params.get("method", "TEMPERATURE"))
    values = [float(value) for value in probabilities]
    if len(values) != 3:
        return values
    if method == "PLATT":
        a = float(params.get("a", 0.0))
        b = float(params.get("b", 0.0))
        logit = np.log(max(min(values[0], 1 - 1e-6), 1e-6) / max(1 - values[0], 1e-6))
        up = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    elif method == "ISOTONIC":
        xs = np.asarray(params.get("xs", []), dtype=float)
        ys = np.asarray(params.get("ys", []), dtype=float)
        if len(xs) == 0:
            up = values[0]
        else:
            up = _isotonic_apply(xs, ys, values[0])
    else:
        temperature = float(params.get("temperature", 1.0))
        logits = np.log(
            np.clip(np.asarray(values), 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - np.asarray(values), 1e-6, 1.0 - 1e-6)
        )
        scaled = 1.0 / (1.0 + np.exp(-logits / max(temperature, 1e-6)))
        scaled = scaled / float(np.sum(scaled))
        return [float(value) for value in scaled]
    up = float(np.clip(up, 1e-6, 1.0 - 1e-6))
    side = values[1] / max(values[1] + values[2], 1e-9) * (1.0 - up)
    down = max(0.0, 1.0 - up - side)
    return [up, side, down]


def select_method(
    probabilities: Sequence[Sequence[float]],
    actuals: Sequence[int],
    *,
    selection_size: int = 30,
    train_size: int = 25,
) -> tuple[str, dict[str, Any]]:
    selection_probs = list(probabilities)[-selection_size:]
    selection_actuals = list(actuals)[-selection_size:]
    train_probs = selection_probs[:train_size]
    train_actuals = selection_actuals[:train_size]
    val_probs = selection_probs[train_size:]
    val_actuals = selection_actuals[train_size:]
    best_method = "TEMPERATURE"
    best_brier = float("inf")
    best_params: dict[str, Any] = {"method": "TEMPERATURE"}
    for method in ("TEMPERATURE", "PLATT", "ISOTONIC"):
        params = fit_method(method, train_probs, train_actuals)
        if method == "TEMPERATURE":
            params["temperature"] = 1.0
        calibrated = [apply_method(params, row) for row in val_probs]
        score = _brier(calibrated, val_actuals) if val_probs else float("inf")
        if score < best_brier:
            best_brier = score
            best_method = method
            best_params = params
    return best_method, {
        **best_params,
        "method_selection": {
            "selection_size": selection_size,
            "train_size": train_size,
            "validation_size": len(val_probs),
            "selected_brier": best_brier,
        },
    }
