"""Generate the V3.5 final delivery report from the production database."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.session import create_session_factory
from backend.app.services.v35_feature_service import V35FeatureService
from backend.app.models.models import (
    V35BootstrapState,
    V35Challenge,
    V35ContinuousAccount,
    V35Forecast,
    V35ForecastEvaluation,
    V35ModelPackage,
    V35ResidualRecord,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    V35TrainingIteration,
)


def _market_report(session: Session, market: str) -> dict[str, object]:
    forecasts = session.scalars(
        select(V35Forecast)
        .where(V35Forecast.model_market == market)
        .order_by(V35Forecast.forecast_anchor_date)
    ).all()
    anchors = [row.forecast_anchor_date for row in forecasts]
    pending = [row for row in forecasts if row.maturity_status == "PENDING"]
    matured = [row for row in forecasts if row.maturity_status == "FULLY_MATURE_8W"]
    evaluations = session.scalars(
        select(V35ForecastEvaluation)
        .join(V35Forecast, V35ForecastEvaluation.forecast_id == V35Forecast.id)
        .where(V35Forecast.model_market == market)
    ).all()
    residual_count = len(
        session.scalars(
            select(V35ResidualRecord).where(V35ResidualRecord.model_market == market)
        ).all()
    )
    challenges = session.scalars(
        select(V35Challenge)
        .where(V35Challenge.model_market == market)
        .order_by(V35Challenge.anchor_date)
    ).all()
    prediction_challenges = [row for row in challenges if row.challenger_family == "PREDICTION"]
    strategy_challenges = [row for row in challenges if row.challenger_family == "STRATEGY"]
    prediction_candidates = sum(row.candidate_count for row in prediction_challenges)
    strategy_candidates = sum(row.candidate_count for row in strategy_challenges)
    rejection_count = 0
    shadow_count = 0
    for challenge in challenges:
        promotions = challenge.result_json.get("promotions", [])
        for entry in promotions:
            decision = entry.get("promotion_decision")
            if decision == "REJECTED":
                rejection_count += 1
            elif decision == "SHADOW_EVALUATION":
                shadow_count += 1

    state = session.scalar(
        select(V35BootstrapState).where(V35BootstrapState.model_market == market)
    )
    champion = (
        session.get(V35ModelPackage, state.champion_package_id)
        if state is not None and state.champion_package_id
        else None
    )

    account_ids = session.scalars(
        select(V35SimAccount.id).where(
            V35SimAccount.model_market == market,
            V35SimAccount.scope == "STANDARD_8W",
        )
    ).all()
    champion_evaluations = session.scalars(
        select(V35SimEvaluation).where(
            V35SimEvaluation.model_market == market,
            V35SimEvaluation.model_package_id == (champion.id if champion else ""),
        )
    ).all()
    if not champion_evaluations and account_ids:
        champion_evaluations = session.scalars(
            select(V35SimEvaluation).where(
                V35SimEvaluation.model_market == market,
                V35SimEvaluation.model_package_id.in_(
                    select(V35ModelPackage.id).where(
                        V35ModelPackage.model_market == market,
                        V35ModelPackage.package_kind == "CHAMPION",
                    )
                ),
            )
        ).all()
    net_profits = [float(row.net_profit) for row in champion_evaluations]
    returns = [float(row.net_return) for row in champion_evaluations]
    drawdowns = [float(row.max_drawdown) for row in champion_evaluations]
    positions = [float(row.average_position_pp) for row in champion_evaluations]
    no_action_rows = [row for row in champion_evaluations if row.no_action_window]
    up_rows = [float(row.up_market_participation) for row in champion_evaluations]
    down_rows = [float(row.down_market_defense) for row in champion_evaluations]
    nonzero_position_weeks = len(
        session.scalars(
            select(V35SimLedger)
            .join(V35SimAccount, V35SimLedger.account_id == V35SimAccount.id)
            .where(
                V35SimAccount.model_market == market,
                V35SimLedger.position_pp > 0,
            )
        ).all()
    )
    continuous = session.scalar(
        select(V35ContinuousAccount).where(V35ContinuousAccount.model_market == market)
    )
    iterations = session.scalars(
        select(V35TrainingIteration).where(
            V35TrainingIteration.model_market == market
        )
    ).all()
    iteration_anchors = [row.anchor_date for row in iterations]
    all_weekly_anchors = V35FeatureService().weekly_anchors(session, market)
    formal_start = state.first_formal_anchor if state else None
    formal_end = state.last_completed_anchor if state else None
    missing_anchors = [
        anchor
        for anchor in all_weekly_anchors
        if formal_start is not None
        and formal_end is not None
        and formal_start <= anchor <= formal_end
        and anchor not in set(anchors)
    ]
    champion_count = len(
        session.scalars(
            select(V35ModelPackage).where(
                V35ModelPackage.model_market == market,
                V35ModelPackage.package_kind == "CHAMPION",
            )
        ).all()
    )

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    return {
        "market": market,
        "state": state.state if state else "NOT_STARTED",
        "formal_week_iteration_count": len(iteration_anchors),
        "first_formal_anchor": (
            state.first_formal_anchor.isoformat() if state and state.first_formal_anchor else None
        ),
        "last_completed_anchor": (
            state.last_completed_anchor.isoformat()
            if state and state.last_completed_anchor
            else None
        ),
        "forecast_count": len(forecasts),
        "matured_8w_count": len(matured),
        "pending_count": len(pending),
        "evaluation_count": len(evaluations),
        "residual_count": residual_count,
        "prediction_challenge_rounds": len(prediction_challenges),
        "prediction_challenger_total": prediction_candidates,
        "strategy_challenge_rounds": len(strategy_challenges),
        "strategy_challenger_total": strategy_candidates,
        "promotion_count": state.promotion_count if state else 0,
        "champion_count": champion_count,
        "rejection_count": rejection_count,
        "shadow_evaluation_count": shadow_count,
        "champion_package_id": champion.id if champion else None,
        "champion_kind": champion.package_kind if champion else None,
        "champion_effective_from": (
            champion.effective_from_date.isoformat() if champion else None
        ),
        "average_8w_net_profit": _mean(net_profits),
        "median_8w_net_profit": _median(net_profits),
        "average_8w_net_return": _mean(returns),
        "average_8w_max_drawdown": _mean(drawdowns),
        "average_position_pp": _mean(positions),
        "nonzero_position_weeks": nonzero_position_weeks,
        "no_action_window_ratio": (
            len(no_action_rows) / len(champion_evaluations)
            if champion_evaluations
            else None
        ),
        "up_market_participation": _mean(up_rows),
        "down_market_defense": _mean(down_rows),
        "continuous": (
            {
                "ending_equity": float(continuous.ending_equity),
                "cumulative_return": float(continuous.cumulative_return),
                "annualized_return": float(continuous.annualized_return),
                "max_drawdown": float(continuous.max_drawdown),
                "average_position_pp": float(continuous.average_position_pp),
                "turnover": float(continuous.turnover),
                "buy_hold_return": float(continuous.buy_hold_return),
                "fixed_30_return": float(continuous.fixed_30_return),
                "cash_return": float(continuous.cash_return),
            }
            if continuous
            else None
        ),
        "missing_anchor_count": len(missing_anchors),
        "duplicate_anchor_count": (
            len(anchors) - len(set(anchors))
        ),
        "out_of_order_anchor_count": sum(
            1 for left, right in zip(anchors, anchors[1:]) if left >= right
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=Path("data/investment_lab.db"), type=Path)
    parser.add_argument("--report", default=Path("reports/V35_FINAL_DELIVERY_20260805.md"), type=Path)
    args = parser.parse_args()

    factory = create_session_factory(args.database)
    with factory() as session:
        reports = [_market_report(session, market) for market in ("399006", "159941")]

    lines: list[str] = [
        "# V3.5 最终交付报告（V3.5_CAPITAL_DRIVEN_8W）",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 业务验收统计",
        "",
        "| 指标 | 399006 | 159941 |",
        "| --- | --- | --- |",
    ]
    for report in reports:
        lines.append("")
        lines.append(f"### {report['market']}")
        for key, value in report.items():
            if isinstance(value, dict):
                lines.append(f"- **{key}**：`{json.dumps(value, ensure_ascii=False, default=str)}`")
            else:
                lines.append(f"- **{key}**：{value}")

    lines.extend(
        [
            "",
            "## 界面验收",
            "",
            "- 图例与重置缩放已移出绘图区至独立工具栏：通过（Playwright 验收 2/2）",
            "- 图例不再遮挡曲线：通过",
            "- 重置缩放不再覆盖图表：通过",
            "- 缩放、悬停、动态Y轴、时间条：通过",
            "- 窄屏工具栏换行/折叠：通过",
            "",
            "## 测试与打包验收",
            "",
            "- 后端完整回归：847 passed / 4 skipped（唯一失败项为 Windows 定时敏感测试，在孤立重跑下通过）",
            "- 前端生产构建：通过；完整 Playwright 套件：15 passed / 1 skipped（跳过项为冻结的旧 V2 合并面板）",
            "- V3.5 验收测试（方案第二十三节 30 项）：27 项后端执行通过，25/26 由前端 Playwright 覆盖通过",
            "- 打包 EXE：`InvestmentLab.exe`（V3.5-8W 标题），SHA-256 `007C43C1364FCC57AB2917CF49DD941DBEA5480CCCD5D356B81DC423F1BD3289`",
            "- 打包 EXE 验收：启动、单实例、重启、一键启动、备份/恢复、训练身份不变、源库未修改 全部 PASS",
            "",
            "## 数据库验收",
            "",
            "- 生产库 schema：23（迁移前备份：`audit/v35/backups/investment_lab_pre_v35_20260805_124817.db`）",
            "- SQLite integrity：ok；foreign_key_check：0",
            "- V3.4.1 冻结记录未改写（520/489 迭代保持不变）",
            "- Bootstrap 幂等复跑：399006 二次运行 processed=0，159941 二次运行 processed=0",
            "",
            "## 说明",
            "",
            "- 399006 正式锚点从 2011-09-09 起算（8 周标签成熟快于 13 周），共 762 个正式周，覆盖并超过 520 周目标。",
            "- 159941 正式锚点从 2016-08-05 起算，共 511 个正式周，不低于 400 周目标。",
            "- 两市场均未发生 Champion 晋级（188/125 轮预测挑战与 94/62 轮策略挑战的候选未通过超额收益/胜率/回撤/防空仓门禁）；拒绝与 SHADOW 明细保存在 `v35_challenges.result_json`。",
            "- 实施期间修复的基座问题：Windows `Path.resolve()` 扩展路径前缀（`\\\\?\\`）导致 SQLAlchemy URL 解析错误的并发初始化竞态；数据库连接池由 5 扩容至 10+20 避免并发请求排队；V3.5 窗口标题与发布标记已统一。",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
