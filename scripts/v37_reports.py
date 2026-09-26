"""Generate the 12 V3.7 report files from the production database."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


REPORT_NAMES = (
    "V37_IMPLEMENTATION_REPORT.md",
    "V37_MULTI_TIMEFRAME_REPORT.md",
    "V37_DEEPSEEK_API_REPORT.md",
    "V37_AI_FORECAST_REPORT.md",
    "V37_AI_CALIBRATION_REPORT.md",
    "V37_LOCAL_VS_AI_REPORT.md",
    "V37_FUSION_REPORT.md",
    "V37_MODEL_REPAIR_REPORT.md",
    "V37_FORWARD_OOS_REPORT.md",
    "V37_DATABASE_REPORT.md",
    "V37_TEST_REPORT.md",
    "V37_EXE_ACCEPTANCE_REPORT.md",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tables(con: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _counts(con: sqlite3.Connection, table: str, column: str = "protocol_version") -> dict[str, int]:
    tables = _tables(con)
    if table not in tables:
        return {}
    try:
        rows = con.execute(
            f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(key): int(value) for key, value in rows}


def _write(target: Path, name: str, body: str) -> None:
    (target / name).write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=Path("data/investment_lab.db"), type=Path)
    parser.add_argument("--report-dir", default=Path("reports"), type=Path)
    parser.add_argument("--test-summary-json", type=Path, default=None)
    args = parser.parse_args()
    con = sqlite3.connect(args.database)
    tables = _tables(con)
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    now = _now()

    user_version = con.execute("PRAGMA user_version").fetchone()[0]
    v37_tables = sorted(name for name in tables if name.startswith("v37_"))
    row_counts = {
        name: int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        for name in v37_tables
    }
    protocol_counts = _counts(con, "v35_forecasts")
    ai_requests = _counts(con, "v37_ai_requests", "response_status")
    ai_forecasts = _counts(con, "v37_ai_forecasts", "screening_scope")
    ai_evaluations = _counts(con, "v37_ai_evaluations")
    calibrators = _counts(con, "v37_ai_calibrators", "status")
    fusion_configs = _counts(con, "v37_fusion_configs")
    fusion_evaluations = _counts(con, "v37_fusion_evaluations")
    repair_proposals = _counts(con, "v37_model_repair_proposals", "status")
    repair_runs = _counts(con, "v37_model_repair_runs", "status")
    conflicts = _counts(con, "v37_model_conflicts")

    _write(
        out,
        "V37_IMPLEMENTATION_REPORT.md",
        f"""# V3.7 实施报告

> 生成时间：{now} ｜ 协议：V3.7_DEEPSEEK_FUSION_8W

## 状态

- 数据库 schema：{user_version}（V3.7 命名空间 11 张表已建）。
- V3.7 本地回放：尚未执行（`v35_bootstrap_state` 无 V3.7 行）；AI 筛选/Forward OOS：尚未调用。
- 阶段一（P0–P5）代码与测试已完成；阶段二（P6–P12）服务已实现，待生产回放与 EXE 重打包验收。

## 已交付

- schema 27→28→29 迁移（11 张 v37 表、attempt_number、screening_scope）。
- `V37RuntimeService`：多周期本地模型（WEEKLY_ONLY / EARLY_FUSION / LATE_FUSION）。
- `v37_ai_service`：匿名筛选、缓存冻结、有限重试、Forward OOS 隔离、AI 校准与权重档位。
- `v37_fusion_service`、`v37_repair_service`：融合账户与 Repair 泄漏约束。
- `/api/v37/*` 只读接口；前端 AI 设置页新增行情对话。
""",
    )
    _write(
        out,
        "V37_MULTI_TIMEFRAME_REPORT.md",
        f"""# V3.7 多周期特征报告

> 生成时间：{now}

`v37_multitimeframe_features` 行数：{row_counts.get('v37_multitimeframe_features', 0)}

未执行 V3.7 回放前无逐市场统计；执行 `scripts/v37_bootstrap.py` 后本报告将补全各市场周K/日K分支数量与 MULTI_TIMEFRAME_STATE 分布。
""",
    )
    _write(
        out,
        "V37_DEEPSEEK_API_REPORT.md",
        f"""# V3.7 DeepSeek API 报告

> 生成时间：{now}

`v37_ai_requests` 按状态：{ai_requests}

历史筛选（HISTORICAL_SCREENING）与正式 Forward OOS 分离；失败请求保留且可重试（最多 3 次尝试）。
""",
    )
    _write(
        out,
        "V37_AI_FORECAST_REPORT.md",
        f"""# V3.7 AI 预测报告

> 生成时间：{now}

`v37_ai_forecasts` 按 screening_scope：{ai_forecasts}
`v37_ai_evaluations` 按 protocol：{ai_evaluations}
""",
    )
    _write(
        out,
        "V37_AI_CALIBRATION_REPORT.md",
        f"""# V3.7 AI 校准报告

> 生成时间：{now}

`v37_ai_calibrators` 按状态：{calibrators}

档位：<30 AI_SHADOW（0%）；30–49 最高 10%；50–99 最高 30%（Challenger）；≥100 验证通过后最高 40%。
""",
    )
    _write(
        out,
        "V37_LOCAL_VS_AI_REPORT.md",
        f"""# V3.7 Local vs AI 报告（核心）

> 生成时间：{now}

预测协议行数：{protocol_counts}

未执行 V3.7 回放/AI 筛选前无有效对比统计；回放后按市场输出 Local/AI/Fusion 方向准确率、8W 收益、最大回撤、上涨参与率、强上涨遗漏率与互胜比例。
""",
    )
    _write(
        out,
        "V37_FUSION_REPORT.md",
        f"""# V3.7 Fusion 报告

> 生成时间：{now}

`v37_fusion_configs`：{fusion_configs}；`v37_fusion_evaluations`：{fusion_evaluations}

初始候选：LOCAL 90/10、80/20、70/30、60/40；融合只使用 `ai_calibrated_probability`。
""",
    )
    _write(
        out,
        "V37_MODEL_REPAIR_REPORT.md",
        f"""# V3.7 Repair 报告

> 生成时间：{now}

`v37_model_repair_proposals` 按状态：{repair_proposals}
`v37_model_repair_runs` 按状态：{repair_runs}

正式晋级证据仅来自 proposal_anchor + 8 周后的成熟 OOS 窗口（泄漏约束）。
""",
    )
    _write(
        out,
        "V37_FORWARD_OOS_REPORT.md",
        f"""# V3.7 Forward OOS 报告

> 生成时间：{now}

`v37_ai_evaluations`（仅 FORWARD_OOS 计入正式统计）：{ai_evaluations}
`v37_ai_model_health` 行数：{row_counts.get('v37_ai_model_health', 0)}
""",
    )
    _write(
        out,
        "V37_DATABASE_REPORT.md",
        f"""# V3.7 数据库报告

> 生成时间：{now}

`PRAGMA user_version`：{user_version}

| 表 | 行数 |
| --- | --- |
""" + "\n".join(f"| {name} | {row_counts.get(name, 0)} |" for name in v37_tables) + "\n",
    )
    test_body = f"""# V3.7 测试报告

> 生成时间：{now}

"""
    if args.test_summary_json is not None and args.test_summary_json.is_file():
        test_body += "```json\n" + args.test_summary_json.read_text(
            encoding="utf-8"
        ) + "\n```\n"
    else:
        test_body += "未提供 pytest 汇总 JSON；运行 `pytest` 后以 `--test-summary-json` 传入。\n"
    _write(out, "V37_TEST_REPORT.md", test_body)
    _write(
        out,
        "V37_EXE_ACCEPTANCE_REPORT.md",
        f"""# V3.7 EXE 验收报告

> 生成时间：{now}

尚未重新打包 EXE；待 V3.7 生产回放与前端多模型页完成后执行打包与启动/单实例/重启/备份恢复验收。
""",
    )
    con.close()
    print("reports written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
