"""Pure, deterministic technical-indicator calculations for stored price history."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, localcontext
from typing import Mapping, Sequence


TIMEFRAME_PERIODS_PER_YEAR: Mapping[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


@dataclass(frozen=True, slots=True)
class PricePoint:
    """Minimal price observation consumed by the indicator engine."""

    trade_date: date
    close: Decimal
    volume: Decimal | None = None


class InvalidPriceDataError(ValueError):
    """A stored price cannot be used safely for an indicator calculation."""

    def __init__(self, field: str, trade_date: date, reason: str) -> None:
        self.field = field
        self.trade_date = trade_date
        self.reason = reason
        super().__init__(f"{field} at {trade_date.isoformat()}: {reason}")


class InvalidIndicatorParametersError(ValueError):
    """Indicator settings are internally inconsistent or unsafe to calculate."""


@dataclass(frozen=True, slots=True)
class IndicatorParameters:
    """Explicit calculation settings persisted with every computed snapshot."""

    ma_periods: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
    ema_short_period: int = 12
    ema_long_period: int = 26
    macd_signal_period: int = 9
    macd_histogram_multiplier: int = 2
    rsi_periods: tuple[int, ...] = (6, 12, 24)
    volume_ma_period: int = 20
    volatility_period: int = 20

    def __post_init__(self) -> None:
        periods = (
            *self.ma_periods,
            self.ema_short_period,
            self.ema_long_period,
            self.macd_signal_period,
            *self.rsi_periods,
            self.volume_ma_period,
            self.volatility_period,
        )
        if any(period <= 0 for period in periods):
            raise ValueError("Indicator periods must be positive integers")
        if self.macd_histogram_multiplier <= 0:
            raise ValueError("MACD histogram multiplier must be positive")

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["ma_periods"] = list(self.ma_periods)
        values["rsi_periods"] = list(self.rsi_periods)
        return values

    def validate(self) -> None:
        """Validate constraints that must hold before running a calculation."""
        if self.volatility_period < 2:
            raise InvalidIndicatorParametersError("volatility_period must be at least 2")


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Calculated values available at one price date, using no later price rows."""

    trade_date: date
    values: Mapping[str, Decimal | None]
    metadata: Mapping[str, Decimal | None]


@dataclass(frozen=True, slots=True)
class IndicatorComputation:
    """A complete calculation result suitable for persistence by a service."""

    formula_label: str
    parameters: Mapping[str, object]
    data_cutoff: date | None
    completeness: Mapping[str, bool]
    snapshots: tuple[IndicatorSnapshot, ...]


class IndicatorCalculator:
    """Calculate technical indicators from price rows only, never client-side state."""

    formula_label = "technical_indicators/v2_partial_window"

    def __init__(self, parameters: IndicatorParameters | None = None) -> None:
        self.parameters = parameters or IndicatorParameters()

    def calculate(
        self,
        prices: Sequence[PricePoint],
        timeframe: str,
    ) -> IndicatorComputation:
        """Compute sequential snapshots using only rows at or before each snapshot date."""
        if timeframe not in TIMEFRAME_PERIODS_PER_YEAR:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.parameters.validate()
        ordered = tuple(sorted(prices, key=lambda point: point.trade_date))
        self._validate_prices(ordered)
        if not ordered:
            return IndicatorComputation(
                formula_label=self.formula_label,
                parameters=self.parameters.as_dict(),
                data_cutoff=None,
                completeness=self._empty_completeness(),
                snapshots=(),
            )

        closes = [point.close for point in ordered]
        volumes = [point.volume for point in ordered]
        moving_averages = {
            period: self._moving_average(closes, period) for period in self.parameters.ma_periods
        }
        ema_short = self._ema(closes, self.parameters.ema_short_period)
        ema_long = self._ema(closes, self.parameters.ema_long_period)
        dif = [
            short - long if short is not None and long is not None else None
            for short, long in zip(ema_short, ema_long, strict=True)
        ]
        dif_first_change = [
            (
                None
                if index == 0 or current is None or dif[index - 1] is None
                else current - dif[index - 1]
            )
            for index, current in enumerate(dif)
        ]
        dea = self._ema_optional(dif, self.parameters.macd_signal_period)
        histogram = [
            (current_dif - current_dea) * Decimal(self.parameters.macd_histogram_multiplier)
            if current_dif is not None and current_dea is not None
            else None
            for current_dif, current_dea in zip(dif, dea, strict=True)
        ]
        rsi = {period: self._rsi(closes, period) for period in self.parameters.rsi_periods}
        volume_ma = self._optional_moving_average(volumes, self.parameters.volume_ma_period)
        volatility = self._annualized_volatility(
            closes,
            self.parameters.volatility_period,
            TIMEFRAME_PERIODS_PER_YEAR[timeframe],
        )
        current_drawdown, running_drawdown = self._drawdowns(closes)
        cumulative_return, annualized_return = self._returns(closes, TIMEFRAME_PERIODS_PER_YEAR[timeframe])

        snapshots = tuple(
            IndicatorSnapshot(
                trade_date=point.trade_date,
                values={
                    **{f"ma_{period}": moving_averages[period][index] for period in self.parameters.ma_periods},
                    f"ema_{self.parameters.ema_short_period}": ema_short[index],
                    f"ema_{self.parameters.ema_long_period}": ema_long[index],
                    "dif": dif[index],
                    "dif_first_change": dif_first_change[index],
                    "dea": dea[index],
                    "macd_histogram": histogram[index],
                    **{f"rsi_{period}": rsi[period][index] for period in self.parameters.rsi_periods},
                    f"volume_ma_{self.parameters.volume_ma_period}": volume_ma[index],
                    f"volatility_{self.parameters.volatility_period}": volatility[index],
                    "current_drawdown": current_drawdown[index],
                    "running_drawdown": running_drawdown[index],
                },
                metadata={
                    "cumulative_return": cumulative_return[index],
                    "annualized_return": annualized_return[index],
                },
            )
            for index, point in enumerate(ordered)
        )
        return IndicatorComputation(
            formula_label=self.formula_label,
            parameters=self.parameters.as_dict(),
            data_cutoff=ordered[-1].trade_date,
            completeness={name: value is not None for name, value in snapshots[-1].values.items()},
            snapshots=snapshots,
        )

    def _empty_completeness(self) -> dict[str, bool]:
        names = {
            *(f"ma_{period}" for period in self.parameters.ma_periods),
            f"ema_{self.parameters.ema_short_period}",
            f"ema_{self.parameters.ema_long_period}",
            "dif",
            "dif_first_change",
            "dea",
            "macd_histogram",
            *(f"rsi_{period}" for period in self.parameters.rsi_periods),
            f"volume_ma_{self.parameters.volume_ma_period}",
            f"volatility_{self.parameters.volatility_period}",
            "current_drawdown",
            "running_drawdown",
        }
        return {name: False for name in names}

    @staticmethod
    def _validate_prices(prices: Sequence[PricePoint]) -> None:
        dates: set[date] = set()
        for point in prices:
            if point.close <= 0:
                raise InvalidPriceDataError("close_price", point.trade_date, "must be positive")
            if point.volume is not None and point.volume < 0:
                raise InvalidPriceDataError("volume", point.trade_date, "must be nonnegative")
            if point.trade_date in dates:
                raise InvalidPriceDataError("trade_date", point.trade_date, "is duplicated")
            dates.add(point.trade_date)

    @staticmethod
    def _moving_average(values: Sequence[Decimal], period: int) -> list[Decimal | None]:
        results: list[Decimal | None] = []
        rolling_sum = Decimal("0")
        for index, value in enumerate(values):
            rolling_sum += value
            if index >= period:
                rolling_sum -= values[index - period]
            results.append(rolling_sum / Decimal(min(index + 1, period)))
        return results

    @staticmethod
    def _optional_moving_average(
        values: Sequence[Decimal | None], period: int
    ) -> list[Decimal | None]:
        results: list[Decimal | None] = []
        window: list[Decimal | None] = []
        for value in values:
            window.append(value)
            if len(window) > period:
                window.pop(0)
            present = [item for item in window if item is not None]
            results.append(sum(present, Decimal("0")) / Decimal(len(present)) if present else None)
        return results

    @staticmethod
    def _ema(values: Sequence[Decimal], period: int) -> list[Decimal | None]:
        return IndicatorCalculator._ema_optional(list(values), period)

    @staticmethod
    def _ema_optional(values: Sequence[Decimal | None], period: int) -> list[Decimal | None]:
        results: list[Decimal | None] = [None] * len(values)
        consecutive: list[Decimal] = []
        current: Decimal | None = None
        multiplier = Decimal("2") / Decimal(period + 1)
        for index, value in enumerate(values):
            if value is None:
                consecutive.clear()
                current = None
                continue
            if current is None:
                consecutive.append(value)
                if len(consecutive) < period:
                    results[index] = sum(consecutive, Decimal("0")) / Decimal(len(consecutive))
                    continue
                current = sum(consecutive, Decimal("0")) / Decimal(period)
            else:
                current = (value - current) * multiplier + current
            results[index] = current
        return results

    @staticmethod
    def _rsi(closes: Sequence[Decimal], period: int) -> list[Decimal | None]:
        if not closes:
            return []
        results: list[Decimal | None] = [Decimal("50")]
        average_gain = Decimal("0")
        average_loss = Decimal("0")
        for index in range(1, len(closes)):
            gain = max(closes[index] - closes[index - 1], Decimal("0"))
            loss = max(closes[index - 1] - closes[index], Decimal("0"))
            if index <= period:
                average_gain = (average_gain * Decimal(index - 1) + gain) / Decimal(index)
                average_loss = (average_loss * Decimal(index - 1) + loss) / Decimal(index)
            else:
                average_gain = (average_gain * Decimal(period - 1) + gain) / Decimal(period)
                average_loss = (average_loss * Decimal(period - 1) + loss) / Decimal(period)
            results.append(IndicatorCalculator._rsi_from_averages(average_gain, average_loss))
        return results

    @staticmethod
    def _rsi_from_averages(average_gain: Decimal, average_loss: Decimal) -> Decimal:
        if average_loss == 0:
            return Decimal("100") if average_gain > 0 else Decimal("50")
        if average_gain == 0:
            return Decimal("0")
        relative_strength = average_gain / average_loss
        return Decimal("100") - Decimal("100") / (Decimal("1") + relative_strength)

    @staticmethod
    def _annualized_volatility(
        closes: Sequence[Decimal], period: int, periods_per_year: int
    ) -> list[Decimal | None]:
        returns: list[Decimal | None] = [None]
        returns.extend((closes[index] / closes[index - 1]) - Decimal("1") for index in range(1, len(closes)))
        results: list[Decimal | None] = []
        annualization = Decimal(periods_per_year).sqrt()
        for index in range(len(closes)):
            window = [value for value in returns[max(1, index - period + 1) : index + 1] if value is not None]
            if len(window) < 2:
                results.append(Decimal("0"))
                continue
            mean = sum(window, Decimal("0")) / Decimal(len(window))
            variance = sum((value - mean) ** 2 for value in window) / Decimal(len(window) - 1)
            results.append(variance.sqrt() * annualization)
        return results

    @staticmethod
    def _drawdowns(closes: Sequence[Decimal]) -> tuple[list[Decimal], list[Decimal]]:
        current: list[Decimal] = []
        running: list[Decimal] = []
        high_water_mark = closes[0]
        largest_drawdown = Decimal("0")
        for close in closes:
            high_water_mark = max(high_water_mark, close)
            drawdown = close / high_water_mark - Decimal("1")
            largest_drawdown = min(largest_drawdown, drawdown)
            current.append(drawdown)
            running.append(largest_drawdown)
        return current, running

    @staticmethod
    def _returns(
        closes: Sequence[Decimal], periods_per_year: int
    ) -> tuple[list[Decimal | None], list[Decimal | None]]:
        cumulative: list[Decimal | None] = []
        annualized: list[Decimal | None] = []
        initial_close = closes[0]
        for index, close in enumerate(closes):
            current_return = close / initial_close - Decimal("1")
            cumulative.append(current_return)
            if index == 0:
                annualized.append(Decimal("0"))
                continue
            with localcontext() as context:
                context.prec = 48
                annualized.append(
                    ((Decimal("1") + current_return).ln() * Decimal(periods_per_year) / Decimal(index)).exp()
                    - Decimal("1")
                )
        return cumulative, annualized
