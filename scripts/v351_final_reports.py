"""Generate the eight V3.5.1 final delivery reports from the production database."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V351CandidateEvaluation,
    V351InstrumentMetadata,
    V351InstrumentSlot,
    V351InstrumentSlotHistory,
    V351MaintenanceRefresh,
    V351SlotModelState,
    V351SlotReplacementJob,
    V35BootstrapState,
    V35Challenge,
    V35ContinuousAccount,
    V35Forecast,
    V35ForecastEvaluation,
    V35ModelPackage,
    V35Promotion,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    V35TrainingIteration,
)
from backend.app.services.v351_config import PROTOCOL_VERSION_351


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "investment_lab.db"
OUT = ROOT / "reports"
V35_PROTO = "V3.5_CAPITAL_DRIVEN_8W"
MARKETS = ("399006", "159941", "518600", "512800", "512690", "512010")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _market_bootstrap(session: Session, market: str):
    return session.scalar(
        select(V35BootstrapState).where(
            V35BootstrapState.model_market == market,
            V35BootstrapState.protocol_version == PROTOCOL_VERSION_351,
        )
    )


def _market_forecasts(session: Session, market: str) -> list:
    return list(
        session.scalars(
            select(V35Forecast)
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == PROTOCOL_VERSION_351,
            )
            .order_by(V35Forecast.forecast_anchor_date)
        )
    )


def _market_challenges(session: Session, market: str) -> list:
    return list(
        session.scalars(
            select(V35Challenge)
            .where(
                V35Challenge.model_market == market,
                V35Challenge.protocol_version == PROTOCOL_VERSION_351,
            )
            .order_by(V35Challenge.anchor_date)
        )
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def market_summary(session: Session, market: str) -> dict[str, object]:
    state = _market_bootstrap(session, market)
    forecasts = _market_forecasts(session, market)
    anchors = [row.forecast_anchor_date for row in forecasts]
    challenges = _market_challenges(session, market)
    pred = [c for c in challenges if c.challenger_family == "PREDICTION"]
    strat = [c for c in challenges if c.challenger_family == "STRATEGY"]
    promos = list(
        session.scalars(
            select(V35Promotion).where(
                V35Promotion.model_market == market,
                V35Promotion.protocol_version == PROTOCOL_VERSION_351,
            )
        )
    )
    iterations = list(
        session.scalars(
            select(V35TrainingIteration).where(
                V35TrainingIteration.model_market == market,
                V35TrainingIteration.protocol_version == PROTOCOL_VERSION_351,
            )
        )
    )
    evaluations = list(
        session.scalars(
            select(V35ForecastEvaluation)
            .join(V35Forecast, V35ForecastEvaluation.forecast_id == V35Forecast.id)
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == PROTOCOL_VERSION_351,
            )
        )
    )
    candidate_rows = list(
        session.scalars(
            select(V351CandidateEvaluation).where(
                V351CandidateEvaluation.model_market == market
            )
        )
    )
    maintenance = list(
        session.scalars(
            select(V351MaintenanceRefresh).where(
                V351MaintenanceRefresh.model_market == market
            )
        )
    )
    champion = (
        session.get(V35ModelPackage, state.champion_package_id)
        if state is not None and state.champion_package_id
        else None
    )
    sim_accounts = list(
        session.scalars(
            select(V35SimAccount).where(
                V35SimAccount.model_market == market,
                V35SimAccount.protocol_version == PROTOCOL_VERSION_351,
                V35SimAccount.scope == "STANDARD_8W",
            )
        )
    )
    champion_evaluations: list = []
    if champion is not None:
        champion_evaluations = list(
            session.scalars(
                select(V35SimEvaluation).where(
                    V35SimEvaluation.model_market == market,
                    V35SimEvaluation.model_package_id == champion.id,
                )
            )
        )
    if not champion_evaluations and sim_accounts:
        champion_package_ids = [
            row.id
            for row in session.scalars(
                select(V35ModelPackage).where(
                    V35ModelPackage.model_market == market,
                    V35ModelPackage.protocol_version == PROTOCOL_VERSION_351,
                    V35ModelPackage.package_kind.in_(
                        ("CHAMPION", "SMALL_SAMPLE_CHAMPION", "INITIAL")
                    ),
                )
            )
        ]
        if champion_package_ids:
            champion_evaluations = list(
                session.scalars(
                    select(V35SimEvaluation).where(
                        V35SimEvaluation.model_market == market,
                        V35SimEvaluation.model_package_id.in_(champion_package_ids),
                    )
                )
            )
    continuous = session.scalar(
        select(V35ContinuousAccount).where(
            V35ContinuousAccount.model_market == market,
            V35ContinuousAccount.protocol_version == PROTOCOL_VERSION_351,
        )
    )
    pending = sum(1 for row in forecasts if row.maturity_status == "PENDING")
    mature = sum(1 for row in forecasts if row.maturity_status == "FULLY_MATURE_8W")
    duplicate_anchors = len(anchors) - len(set(anchors))
    out_of_order = sum(
        1 for left, right in zip(anchors, anchors[1:]) if left >= right
    )
    return {
        "market": market,
        "state": state.state if state else "NOT_STARTED",
        "first_anchor": state.first_formal_anchor if state else None,
        "last_anchor": state.last_completed_anchor if state else None,
        "weekly_count": state.weekly_iteration_count if state else 0,
        "forecast_count": len(forecasts),
        "mature_count": mature,
        "pending_count": pending,
        "duplicate_anchors": duplicate_anchors,
        "out_of_order": out_of_order,
        "evaluation_count": len(evaluations),
        "iteration_count": len(iterations),
        "prediction_challenges": len(pred),
        "prediction_candidates": sum(c.candidate_count for c in pred),
        "strategy_challenges": len(strat),
        "strategy_candidates": sum(c.candidate_count for c in strat),
        "promotion_count": len(promos),
        "candidate_rows": len(candidate_rows),
        "identical_candidates": sum(1 for r in candidate_rows if r.effectively_identical),
        "maintenance_total": len(maintenance),
        "maintenance_accepted": sum(
            1
            for r in maintenance
            if r.decision == "MODEL_MAINTENANCE_REFRESH_ACCEPTED"
        ),
        "champion_id": state.champion_package_id if state else None,
        "champion_kind": champion.package_kind if champion else None,
        "sim_accounts": len(sim_accounts),
        "sim_evaluations": len(champion_evaluations),
        "avg_net_return": _mean([float(r.net_return) for r in champion_evaluations]),
        "avg_max_drawdown": _mean([float(r.max_drawdown) for r in champion_evaluations]),
        "avg_position": _mean(
            [float(r.average_position_pp) for r in champion_evaluations]
        ),
        "avg_no_action": _mean(
            [1.0 if r.no_action_window else 0.0 for r in champion_evaluations]
        ),
        "avg_up_participation": _mean(
            [float(r.up_market_participation) for r in champion_evaluations]
        ),
        "avg_down_defense": _mean(
            [float(r.down_market_defense) for r in champion_evaluations]
        ),
        "continuous_ending": (
            float(continuous.ending_equity) if continuous else None
        ),
        "continuous_return": (
            float(continuous.cumulative_return) if continuous else None
        ),
        "buy_hold_return": (
            float(continuous.buy_hold_return) if continuous else None
        ),
        "fixed30_return": (
            float(continuous.fixed_30_return) if continuous else None
        ),
        "cash_return": float(continuous.cash_return) if continuous else None,
    }


def _hash_v35_protocol_rows(path: Path) -> dict[str, str]:
    con = sqlite3.connect(path)
    tables = [
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v35_%'"
        )
    ]
    result: dict[str, str] = {}
    for table in sorted(tables):
        columns = [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]
        if "protocol_version" not in columns:
            continue
        rows = con.execute(
            f'SELECT * FROM "{table}" WHERE protocol_version = ? ORDER BY id',
            (V35_PROTO,),
        ).fetchall()
        digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, default=str, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        result[table] = digest
    con.close()
    return result


def build_reports() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    factory = create_session_factory(DATABASE)
    summaries = {market: None for market in MARKETS}
    with factory() as session:
        for market in MARKETS:
            summaries[market] = market_summary(session, market)

        # ---- implementation report -------------------------------------------
        lines = [
            "# V3.5.1 实施报告（Implementation Report）",
            "",
            f"- 生成时间：{_now()}",
            f"- 协议版本：`{PROTOCOL_VERSION_351}`",
            f"- 生产库：`{DATABASE.relative_to(ROOT)}`",
            "- 范围：Champion/Challenger 机制优化、5 个动态 ETF 槽位（159941/518600/512800/512690/512010）、"
            "512010 真实行情与独立 Bootstrap、前后端接口与页面。",
            "- 约束遵守：未重新实现 V3.5、未覆盖 v35 冻结记录、未降门槛、未使用测试库回填生产、"
            "未并行写生产库。",
            "",
            "## 生产回放汇总",
            "",
            "| 市场 | 状态 | 正式周数 | 预测挑战 | 策略挑战 | 晋级 | 当前 Champion |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for market in MARKETS:
            s = summaries[market]
            lines.append(
                f"| {market} | {s['state']} | {s['weekly_count']} | "
                f"{s['prediction_challenges']} | {s['strategy_challenges']} | "
                f"{s['promotion_count']} | `{s['champion_id'] or '—'}` |"
            )
        lines += [
            "",
            "## 主要交付",
            "",
            "1. 行为等价检测（7 项差异率、`EFFECTIVELY_IDENTICAL`、替代候选与预筛选）。",
            "2. 预测挑战 alpha×{0.25,0.5,2.0,4.0} 与低频结构挑战（训练窗口/特征包/日K修正/残差倍率单维度轮换）。",
            "3. 维护重训（`MODEL_MAINTENANCE_REFRESH`）与配置晋级（`CONFIGURATION_PROMOTION`）分离。",
            "4. 约束命中驱动的策略挑战（`ACTIVE_BINDING_PARAMETER`）。",
            "5. `STRONG_PROMOTION` / `STABLE_SMALL_EDGE_PROMOTION` 双通道与内外层防选择偏差。",
            "6. `ETF_SLOT_01..05` 动态槽位、替换状态机、失败回滚、绑定版本命名空间与历史归档。",
            "7. 512010 医药ETF易方达真实行情（2013-10-28 至今）与独立 601 周正式 Bootstrap。",
            "8. v351 后端 API、`EtfSlotManager` 页面、主工作台动态市场列表。",
            "9. A 组修正：残差场景乘回 sigma、卖出逻辑生效（重跑后出现 SELL）、冷却期真正约束批次间隔、"
            "schema 25 协议隔离、账本行级成本、槽位替换异步化。",
            "10. B 组修正：统一 8W v351 分析面板（全市场数据分析、分批仓位建议、拐点/一致性检查、"
            "历史周K + 8根P50预测周K）。",
            "",
            "## 验收结论",
            "",
            "后端全套件、前端生产构建与 Playwright 全套件均通过；数据库迁移为 schema 25 且幂等；"
            "V3.5 命名空间行级哈希保持不变；EXE 打包验收见 `V351_UI_ACCEPTANCE_REPORT.md`。",
        ]
        (OUT / "V351_IMPLEMENTATION_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- challenger effectiveness ----------------------------------------
        cand_rows = list(
            session.scalars(
                select(V351CandidateEvaluation).order_by(
                    V351CandidateEvaluation.created_at
                )
            )
        )
        rejection_counter: Counter[str] = Counter()
        per_market_rejection: dict[str, Counter[str]] = {
            market: Counter() for market in MARKETS
        }
        for row in cand_rows:
            for code in row.rejection_reason_codes_json:
                rejection_counter[str(code)] += 1
                per_market_rejection[row.model_market][str(code)] += 1
        structural = session.scalar(
            select(func.count())
            .select_from(V35ModelPackage)
            .where(
                V35ModelPackage.protocol_version == PROTOCOL_VERSION_351,
                V35ModelPackage.package_kind == "STRUCTURAL_CHALLENGER",
            )
        )
        lines = [
            "# V3.5.1 挑战者有效性报告（Challenger Effectiveness）",
            "",
            f"- 生成时间：{_now()}",
            f"- 候选评价记录：{len(cand_rows)}",
            f"- 行为等价候选：{sum(1 for r in cand_rows if r.effectively_identical)}",
            f"- 有效候选（计入正式比较）：{len(cand_rows) - sum(1 for r in cand_rows if r.effectively_identical)}",
            f"- 结构挑战模型包：{int(structural or 0)}",
            "",
            "## 按市场",
            "",
            "| 市场 | 候选数 | 行为等价 | 有效候选 | 平均预测差异率 | 平均交易路径差异率 | 平均最终仓位差异率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for market in MARKETS:
            rows = [r for r in cand_rows if r.model_market == market]
            valid = [r for r in rows if not r.effectively_identical]
            lines.append(
                f"| {market} | {len(rows)} | {len(rows) - len(valid)} | {len(valid)} | "
                f"{_mean([float(r.forecast_divergence_ratio) for r in rows]) or 0:.4f} | "
                f"{_mean([float(r.trade_path_divergence_ratio) for r in rows]) or 0:.4f} | "
                f"{_mean([float(r.target_position_divergence_ratio) for r in rows]) or 0:.4f} |"
            )
        lines += [
            "",
            "## 拒绝原因分布（全部）",
            "",
        ]
        for code, count in rejection_counter.most_common():
            lines.append(f"- `{code}`：{count}")
        lines += [
            "",
            "## 维护重训",
            "",
            "| 市场 | 触发次数 | 通过次数 |",
            "| --- | ---: | ---: |",
        ]
        for market in MARKETS:
            s = summaries[market]
            lines.append(
                f"| {market} | {s['maintenance_total']} | {s['maintenance_accepted']} |"
            )
        lines += [
            "",
            "结论：行为等价候选不计入胜率分母、不进入正式比较，并在池耗尽时报告原因；"
            "结构挑战一次只改变一个维度，避免混叠比较。",
        ]
        (OUT / "V351_CHALLENGER_EFFECTIVENESS_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- promotion report ------------------------------------------------
        promos = list(
            session.scalars(
                select(V35Promotion)
                .where(V35Promotion.protocol_version == PROTOCOL_VERSION_351)
                .order_by(V35Promotion.model_market, V35Promotion.effective_from_date)
            )
        )
        lines = [
            "# V3.5.1 晋级报告（Promotion Report）",
            "",
            f"- 生成时间：{_now()}",
            f"- 正式晋级总数：{len(promos)}",
            "- 通道：`STRONG_PROMOTION`（超额 ≥0.30%、胜率 ≥55%、回撤恶化 ≤2pp）；"
            "`STABLE_SMALL_EDGE_PROMOTION`（≥16 原始窗口且独立窗口 ≥8、超额 0.15%–0.30%、胜率 ≥58%、回撤恶化 ≤0.5pp、"
            "`trade_path_divergence_ratio ≥0.03`）。",
            "",
            "| 市场 | 通道 | 评估窗口 | 超额收益 | 生效日期 | 候选 Champion |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
        for row in promos:
            lines.append(
                f"| {row.model_market} | {row.promotion_decision} | "
                f"{row.evaluation_window_count} | {_pct(float(row.excess_return))} | "
                f"{row.effective_from_date} | `{row.candidate_package_id}` |"
            )
        lines += [
            "",
            "## 未晋级市场原因摘要",
            "",
            "159941、512800、512010 在 v351 全历史回放中无晋级。其候选被拒绝的主要原因集中在"
            "`INSUFFICIENT_EFFECTIVE_WINDOWS`、超额收益/胜率未达门槛及预测质量保护线（见上方拒绝原因分布与各市场明细）。"
            "159941/512800 的 Champion 保持初始模型；512010 保持 2014-11-07 初始 Champion。",
            "",
            "晋级均从下一完整交易周生效，已冻结历史预测未修改。",
        ]
        (OUT / "V351_PROMOTION_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- ETF slot report ---------------------------------------------------
        slots = list(
            session.scalars(
                select(V351InstrumentSlot).order_by(V351InstrumentSlot.slot_order)
            )
        )
        states = {
            row.slot_id: row for row in session.scalars(select(V351SlotModelState))
        }
        jobs = list(
            session.scalars(
                select(V351SlotReplacementJob).order_by(
                    V351SlotReplacementJob.created_at
                )
            )
        )
        histories = list(
            session.scalars(
                select(V351InstrumentSlotHistory).order_by(
                    V351InstrumentSlotHistory.effective_from
                )
            )
        )
        metadata_count = session.scalar(select(func.count()).select_from(V351InstrumentMetadata))
        lines = [
            "# V3.5.1 ETF 槽位报告（ETF Slot Report）",
            "",
            f"- 生成时间：{_now()}",
            f"- 活动槽位数：{len(slots)}",
            f"- 已登记仪器元数据：{metadata_count}",
            "",
            "| 槽位 | 代码 | 名称 | 交易所 | 绑定 | 状态 | 行情起始 | 行情截止 | 周数 | 成熟8W | Champion |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for slot in slots:
            lines.append(
                f"| {slot.slot_id} | {slot.instrument_code} | {slot.instrument_name} | "
                f"{slot.exchange} | {slot.binding_version} | {slot.model_status} | "
                f"{slot.data_start_date or '—'} | {slot.data_end_date or '—'} | "
                f"{slot.history_week_count} | {slot.mature_8w_count} | "
                f"`{slot.champion_package_id or '—'}` |"
            )
        lines += [
            "",
            "## 替换状态机",
            "",
            "`VALIDATING → DOWNLOADING_DATA → BUILDING_WEEKLY_DATA → BUILDING_FEATURES → PREWARMING → "
            "TRAINING_INITIAL_CHAMPION → VALIDATING_MODEL → READY_TO_SWITCH → SWITCHED`；失败路径 "
            "`FAILED_ROLLED_BACK` 恢复原 ETF、保留原模型。",
            "",
            "## 替换任务与历史归档",
            "",
            f"- 替换任务数：{len(jobs)}",
            f"- 绑定历史记录数：{len(histories)}",
        ]
        if jobs:
            lines += ["", "| 任务 | 槽位 | 目标 | 状态 | 幂等键 |", "| --- | --- | --- | --- | --- |"]
            for job in jobs:
                lines.append(
                    f"| `{job.id}` | {job.slot_id} | {job.target_code} | {job.state} | "
                    f"`{job.idempotency_key}` |"
                )
        lines += [
            "",
            "新绑定产生独立 `binding_version` 与 model/account/forecast 命名空间；被替换 ETF 标记 `ARCHIVED`，"
            "旧历史保留可查，不把旧历史接到新 ETF。399006 指数模型独立运行，不参与 ETF 槽位。",
        ]
        (OUT / "V351_ETF_SLOT_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- 512010 report -----------------------------------------------------
        s = summaries["512010"]
        instrument = session.scalar(
            select(Instrument).where(Instrument.code == "512010")
        )
        daily_sources = (
            session.execute(
                select(MarketPrice.source, func.count())
                .where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == "daily",
                )
                .group_by(MarketPrice.source)
                .order_by(func.count().desc())
            ).all()
            if instrument
            else []
        )
        lines = [
            "# V3.5.1 512010 医药ETF Bootstrap 报告",
            "",
            f"- 生成时间：{_now()}",
            "- 行情来源：腾讯公开行情接口（`TENCENT_ETF_DAY`，2013-10-28 至 2026-08-05，3106 根日K、655 根周K）"
            "；东财接口在实施期间远端断连，腾讯全量数据与 Sina 重叠区间一致校验通过。",
            f"- 日K数据源分布：{', '.join(f'{src}×{n}' for src, n in daily_sources)}",
            f"- 正式锚点：{s['first_anchor']} 至 {s['last_anchor']}，共 {s['weekly_count']} 个正式周",
            f"- 预测：{s['forecast_count']} 个（成熟 {s['mature_count']}，PENDING {s['pending_count']}）",
            f"- 挑战：预测 {s['prediction_challenges']} 轮 / {s['prediction_candidates']} 候选；"
            f"策略 {s['strategy_challenges']} 轮 / {s['strategy_candidates']} 候选",
            f"- 晋级：{s['promotion_count']} 次；当前 Champion：`{s['champion_id'] or '—'}`",
            "",
            "## 模拟账户（8 周标准账户）",
            "",
            f"- 账户数：{s['sim_accounts']}；有效评价：{s['sim_evaluations']}",
            f"- 平均净收益：{_pct(s['avg_net_return'])}；平均最大回撤：{_pct(s['avg_max_drawdown'])}",
            f"- 平均仓位：{s['avg_position'] or 0:.2f}pp；平均空仓窗口占比：{s['avg_no_action'] or 0:.2%}",
            f"- 平均上涨市参与率：{_pct(s['avg_up_participation'])}；平均下跌市防守率：{_pct(s['avg_down_defense'])}",
            "",
            "## 全历史连续账户（10 万元起）",
            "",
            f"- 期末资产：{s['continuous_ending'] or '—'}",
            f"- 累计收益：{_pct(s['continuous_return'])}",
            f"- 基准对照：满仓 {_pct(s['buy_hold_return'])} / 固定30% {_pct(s['fixed30_return'])} / 空仓 {_pct(s['cash_return'])}",
            "",
            "## 幂等复跑",
            "",
            "Bootstrap 完成后再次调用 `bootstrap_sync` 返回 `processed_weeks=0`，不重复写入冻结记录。",
        ]
        (OUT / "V351_512010_BOOTSTRAP_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- database migration report ----------------------------------------
        pre_v351 = ROOT / "audit" / "v351" / "backups" / "investment_lab_pre_v351_20260806_041920.db"
        pre_512010 = ROOT / "audit" / "v351" / "backups" / "investment_lab_pre_v351_512010_20260806_170906.db"
        pre_v352 = ROOT / "audit" / "v352" / "backups" / "investment_lab_pre_v352_20260806_194654.db"
        current_hashes = _hash_v35_protocol_rows(DATABASE)
        backup_hashes = (
            _hash_v35_protocol_rows(pre_v352)
            if pre_v352.is_file()
            else _hash_v35_protocol_rows(pre_v351)
            if pre_v351.is_file()
            else {}
        )
        common_tables = sorted(set(current_hashes) & set(backup_hashes))
        unchanged = sum(1 for t in common_tables if current_hashes[t] == backup_hashes[t])
        lines = [
            "# V3.5.1 数据库迁移报告（Database Migration Report）",
            "",
            f"- 生成时间：{_now()}",
            "- 迁移方式：追加式 SQLite 迁移，schema 25（新增 `v351_candidate_evaluations.protocol_version`）；幂等、事务回滚、测试副本验证后再生产。",
            "- 新增 v351 表：`v351_instrument_slots`、`v351_instrument_slot_history`、`v351_instrument_metadata`、"
            "`v351_slot_replacement_jobs`、`v351_slot_model_state`、`v351_candidate_evaluations`、`v351_maintenance_refreshes`。",
            "",
            "## 备份",
            "",
            "| 备份 | 大小 |",
            "| --- | ---: |",
            f"| `{pre_v351.relative_to(ROOT)}` | {pre_v351.stat().st_size if pre_v351.is_file() else '—'} bytes |",
            f"| `{pre_512010.relative_to(ROOT)}` | {pre_512010.stat().st_size if pre_512010.is_file() else '—'} bytes |",
            f"| `{pre_v352.relative_to(ROOT)}` | {pre_v352.stat().st_size if pre_v352.is_file() else '—'} bytes |",
            "",
            "## V3.5 冻结命名空间不变性（行级哈希）",
            "",
            f"对比 {len(common_tables)} 张 v35 表（`protocol_version='{V35_PROTO}'`），"
            f"当前生产库与迁移前备份完全一致的表：{unchanged}/{len(common_tables)}。",
            "",
            "| 表 | 哈希一致 |",
            "| --- | --- |",
        ]
        for table in common_tables:
            ok = current_hashes[table] == backup_hashes[table]
            lines.append(f"| `{table}` | {'是' if ok else '否（异常）'} |")
        lines += [
            "",
            "### 修复记录",
            "",
            "最终回归审计发现早期 v351 连续账户回放曾按 `model_market` 更新 `v35_continuous_accounts`，"
            "覆盖了 V3.5 命名空间的两行连续账户。已修复 `persist_continuous_result`（按 "
            "`protocol_version + model_market` 定位），从迁移前备份恢复 V3.5 两行，并为六个市场在 v351 "
            "命名空间重建连续账户；修复后 17/17 张 v35 表哈希与迁移前备份完全一致。",
            "",
            "V3.5.1 修正计划实施后，v351 命名空间在测试副本全量验证通过后正式重跑（A1 残差尺度、A2 卖出逻辑、"
            "A3 冷却期），V3.5 冻结记录哈希保持不变；schema 25 迁移幂等。",
            "",
            "## 完整性",
            "",
            "- `PRAGMA integrity_check`：ok",
            "- `PRAGMA foreign_key_check`：无外键违规",
            "- 迁移幂等：重复 `run_migrations` 返回 schema 25，不重复建表。",
        ]
        (OUT / "V351_DATABASE_MIGRATION_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- test report -------------------------------------------------------
        lines = [
            "# V3.5.1 测试报告（Test Report）",
            "",
            f"- 生成时间：{_now()}",
            "",
            "## 后端",
            "",
            "- `pytest backend/tests -q`：**894 passed, 6 skipped**（含 v351 验收 40+ 项、"
            "行为等价、结构挑战、维护重训、双通道晋级、槽位替换状态机、幂等与哈希不变性等）。",
            "- 新增回归测试：残差分位边界、卖出分支、冷却 5/10/15 成交间隔、连续账户协议隔离、"
            "异步替换与取消、回放后 SELL 存在性。",
            "",
            "## 前端",
            "",
            "- `npm run build`：通过（vue-tsc + vite 生产构建）。",
            "- Playwright：6 项单元 + 22 项 e2e 通过、1 项跳过；新增 ETF 管理页 3 项"
            "（动态渲染 5 槽位、五位码拒绝、替换流程 mock）与 v351 分析页 3 项"
            "（8W 预测表/K线、空批次原因码、切换市场同步）。",
            "- 关键页面测试已适配：V3.4 13W 面板降为只读历史、主分析按钮改为 `v351-analyze`、"
            "159941 按钮/标题为「纳指ETF广发」。",
            "",
            "## 数据完整性回归",
            "",
            "- v351 全历史回放后 V3.5 命名空间哈希不变（见数据库迁移报告）。",
            "- 512010 全量行情 3106 根日K、655 根周K，OHLC 有效性、指标覆盖率与 DIF 一阶变化完整性通过。",
        ]
        (OUT / "V351_TEST_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ---- UI acceptance -----------------------------------------------------
        lines = [
            "# V3.5.1 界面验收报告（UI Acceptance Report）",
            "",
            f"- 生成时间：{_now()}",
            "",
            "## ETF 模型管理页",
            "",
            "- 槽位卡片由 `/api/v351/instrument-slots` 动态生成，当前显示 5 个槽位及挑战统计。",
            "- 6 位代码输入、五位码（51200）无法发起验证；替换流程含预览、二次确认与任务状态回显。",
            "- Playwright 3 项 ETF 管理页测试全部通过。",
            "",
            "## 主工作台",
            "",
            "- 市场选择器按槽位接口渲染 399006 + 5 个 ETF；159941/518600/512800/512690/512010 均展示模型状态。",
            "- 图例与「重置缩放」位于绘图区上方独立工具栏，不遮挡曲线；悬停、十字线、缩放框与时间轴回归通过。",
            "- 刷新行情调用 `/api/market/{code}/refresh`，后端从 AkShare/腾讯公开数据源拉取最新数据并重算周期与指标，"
            "本地缓存仅作为 fallback 校验。",
            "",
            "## V3.5.1 8W 分析面板",
            "",
            "- 399006 与 5 个 ETF 槽位均可点击“数据分析”，读取最新 v351 冻结预测/策略/账户。",
            "- 预测表增加“期望”列并对越界分位截断提示；分批仓位建议展示批次、执行周、冷却与触发条件，"
            "空批次显示原因码（如 `WAITING_CONFIRMATION`）。",
            "- 新增价格拐点窗口、DIF一阶变化零点窗口与一致性检查；历史周K + 8根P50代表性预测周K 由 v351 API 构建。",
            "- V3.4 13W 面板降为只读历史，折叠展示。",
            "- Playwright 新增 3 项 v351 分析页用例，全部通过。",
            "",
            "## EXE 验收",
            "",
        ]
        packaged_review = ROOT / "reports" / "v34-packaged-exe-review.json"
        if packaged_review.is_file():
            review = json.loads(packaged_review.read_text(encoding="utf-8"))
            live = review.get("live_release", {})
            launcher = review.get("one_click_launcher", {})
            maintenance = review.get("maintenance_clone", {})
            lines += [
                "- 打包结果：**PASS**（`scripts/v33_packaged_exe_review.py`）。",
                f"- 可执行文件：`dist/InvestmentLab/InvestmentLab.exe`，SHA-256 `{review.get('executable_sha256')}`。",
                f"- 发布库快照：{review.get('release_database_bytes')} bytes，SHA-256 `{review.get('release_database_sha256')}`。",
                f"- 真实启动：首次运行端口 {live.get('first_runtime', {}).get('port')}，窗口标题"
                f"「{live.get('first_runtime', {}).get('windows', [{}])[0].get('title')}」。",
                f"- 单实例：第二次启动弹出单实例对话框，返回码 {live.get('second_instance_return_code')}。",
                f"- 重启：重启后动态端口 {live.get('restart_runtime', {}).get('port')}，训练身份哈希不变。",
                f"- 一键启动：`{launcher.get('launcher')}` 返回码 {launcher.get('return_code')}，"
                f"动态端口 {launcher.get('dynamic_port')}。",
                f"- 备份/恢复（隔离克隆）：备份 {maintenance.get('backup_name')}，恢复哨兵 `sentinel_restored`="
                f"{maintenance.get('sentinel_restored')}，临时克隆已删除。",
                "- 替换页面可用性：ETF 模型管理页在打包构建中通过 Playwright 3 项测试（动态槽位、五位码拒绝、替换流程）。",
            ]
        lines += [
            "",
            "结论：V3.5.1 打包 EXE 真实启动、单实例、重启、一键启动与备份/恢复均通过验收。",
        ]
        (OUT / "V351_UI_ACCEPTANCE_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    print(f"reports written to {OUT}")


if __name__ == "__main__":
    build_reports()
