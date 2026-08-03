"""Scale-aware SQL expressions for :class:`FixedPointDecimal` columns.

Do not use raw SQL ``+``, ``-``, ``*``, ``/`` or ``func.avg`` with fixed-point
columns: values are stored as scaled INTEGERs, so generic SQL arithmetic does
not preserve their Decimal meaning.  Use these helpers for product, quotient,
and average queries instead.

Normal Decimal-compatible scalar operands are converted to scaled INTEGER SQL
literals. Scalar-only calls must provide ``scale`` explicitly because there is
no fixed-point column from which to infer it.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

from sqlalchemy import func, literal

from ..models.models import FixedPointDecimal


def _rounded_integer_quotient(numerator: int, denominator: int) -> int:
    """Return an integer quotient using Decimal's ROUND_HALF_EVEN rule."""
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    numerator, denominator = abs(numerator), abs(denominator)
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (twice_remainder == denominator and quotient % 2):
        quotient += 1
    return sign * quotient


def _scaled_multiply(left: int | None, right: int | None, scale: int) -> int | None:
    if left is None or right is None:
        return None
    return _rounded_integer_quotient(left * right, 10**scale)


def _scaled_divide(left: int | None, right: int | None, scale: int) -> int | None:
    if left is None or right is None or right == 0:
        return None
    return _rounded_integer_quotient(left * (10**scale), right)


class _ScaledAverage:
    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.scale: int | None = None

    def step(self, value: int | None, scale: int) -> None:
        if value is None:
            return
        if self.scale is None:
            self.scale = int(scale)
        elif self.scale != int(scale):
            raise ValueError("fixed-point average requires a consistent scale")
        self.total += value
        self.count += 1

    def finalize(self) -> int | None:
        if self.count == 0:
            return None
        return _rounded_integer_quotient(self.total, self.count)


def register_fixed_point_sql_functions(connection: Any) -> None:
    """Register exact scaled-integer SQL functions on one SQLite connection."""
    connection.create_function("fixed_point_multiply", 3, _scaled_multiply, deterministic=True)
    connection.create_function("fixed_point_divide", 3, _scaled_divide, deterministic=True)
    connection.create_aggregate("fixed_point_average", 2, _ScaledAverage)


def _scale_for(*expressions: Any, scale: int | None = None) -> int:
    expression_scales = {
        expression.type.scale
        for expression in expressions
        if isinstance(getattr(expression, "type", None), FixedPointDecimal)
    }
    if scale is not None:
        if expression_scales and expression_scales != {scale}:
            raise ValueError("fixed-point expressions must use the requested scale")
        return scale
    if len(expression_scales) != 1:
        raise ValueError("fixed-point query helpers require FixedPointDecimal expressions at one scale")
    return expression_scales.pop()


def _coerce_fixed_point_operand(operand: Any, scale: int) -> Any:
    if isinstance(getattr(operand, "type", None), FixedPointDecimal):
        return operand
    if isinstance(operand, (Decimal, int, float, str)) and not isinstance(operand, bool):
        return literal(FixedPointDecimal(scale).to_storage(operand))
    raise ValueError(
        "fixed-point query operands must be FixedPointDecimal expressions or Decimal-compatible scalars"
    )


def fixed_point_multiply(left: Any, right: Any, *, scale: int | None = None) -> Any:
    """Return an exact Decimal-typed SQL product at the fixed-point scale."""
    resolved_scale = _scale_for(left, right, scale=scale)
    return func.fixed_point_multiply(
        _coerce_fixed_point_operand(left, resolved_scale),
        _coerce_fixed_point_operand(right, resolved_scale),
        literal(resolved_scale),
        type_=FixedPointDecimal(resolved_scale),
    )


def fixed_point_divide(left: Any, right: Any, *, scale: int | None = None) -> Any:
    """Return an exact Decimal-typed SQL quotient at the fixed-point scale."""
    resolved_scale = _scale_for(left, right, scale=scale)
    return func.fixed_point_divide(
        _coerce_fixed_point_operand(left, resolved_scale),
        _coerce_fixed_point_operand(right, resolved_scale),
        literal(resolved_scale),
        type_=FixedPointDecimal(resolved_scale),
    )


def fixed_point_average(value: Any, *, scale: int | None = None) -> Any:
    """Return an exact Decimal-typed SQL average at the fixed-point scale."""
    resolved_scale = _scale_for(value, scale=scale)
    return func.fixed_point_average(
        value,
        literal(resolved_scale),
        type_=FixedPointDecimal(resolved_scale),
    )
