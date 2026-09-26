"""Render the three V3.5.1 baseline audit markdown reports from baseline.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=Path("audit/v351/baseline.json"), type=Path)
    parser.add_argument("--out", default=Path("audit/v351"), type=Path)
    args = parser.parse_args()
    data = json.loads(args.baseline.read_text(encoding="utf-8"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# V3.5.1 基线审计（Baseline）",
        "",
        f"- 审计时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 数据库：`{data['database']}`",
        f"- Schema：{data['schema_version']}；integrity：{data['integrity_check']}；FK：{data['foreign_key_check']}",
        f"- v35 表数量：{data['v35_table_count']}",
        "",
        "## v35 关键表哈希",
        "",
        "| 表 | SHA-256 |",
        "| --- | --- |",
    ]
    for table, digest in data["v35_table_hashes"].items():
        lines.append(f"| `{table}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## 哈希列汇总",
            "",
            "| 表 | 行数 | 组合 SHA-256 |",
            "| --- | ---: | --- |",
        ]
    )
    for table, info in data["hash_columns"].items():
        lines.append(f"| `{table}` | {info['row_count']} | `{info['combined_sha256']}` |")
    lines.extend(
        [
            "",
            "## 迁移前备份",
            "",
            f"- 备份：`{data['backup']['backup']}`",
            f"- 大小：{data['backup']['bytes']} bytes",
            f"- integrity：{data['backup']['integrity_check']}",
            f"- SHA-256：`{data['backup']['sha256']}`",
        ]
    )
    (out / "V351_BASELINE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    equivalence: list[str] = [
        "# V3.5.1 挑战者行为等价审计（Challenger Equivalence）",
        "",
        "V3.5 已完成的挑战记录作为基线：",
        "",
        "| 市场 | 预测挑战 | 策略挑战 | 候选总数 | 晋级 | 当前 Champion |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for market, summary in data["v35_challenge_summary"].items():
        families = summary["challenges_by_family"]
        equivalence.append(
            f"| {market} | {families.get('PREDICTION', 0)} | {families.get('STRATEGY', 0)} | "
            f"{summary['challenger_total']} | {summary['promotion_count']} | `{summary['champion_package_id'] or '—'}` |"
        )
    equivalence.extend(
        [
            "",
            "V3.5 未实现行为等价检测；V3.5.1 将以上述记录为基线，在 v351 命名空间重新生成并统计",
            "`EFFECTIVELY_IDENTICAL`、候选替换、有效挑战者与拒绝原因分布。",
        ]
    )
    (out / "V351_CHALLENGER_EQUIVALENCE_AUDIT.md").write_text("\n".join(equivalence), encoding="utf-8")

    slots: list[str] = [
        "# V3.5.1 ETF 槽位审计（ETF Slot）",
        "",
        "从当前代码与正式数据库读取的实际配置（不依赖历史说明）：",
        "",
        "| 代码 | 名称 | 类别 | 当前角色 |",
        "| --- | --- | --- | --- |",
    ]
    chart_etfs = set(data["current_chart_etf_slots"])
    for instrument in data["instruments"]:
        if instrument["code"] in chart_etfs:
            role = "模型" if instrument["code"] == "159941" else "仅展示"
        elif instrument["code"] == "399006":
            role = "独立指数模型（不入ETF槽位）"
        elif instrument["code"] == "NDX":
            role = "基准只读"
        else:
            role = "种子化未启用"
        slots.append(
            f"| {instrument['code']} | {instrument['name']} | {instrument['category']} | {role} |"
        )
    slots.extend(
        [
            "",
            "结论：当前实际图表 ETF 为 159941、518600、512800、512690 共 4 个；",
            "V3.5.1 以这 4 个作为 `ETF_SLOT_01..04`，新增 512010 医药ETF易方达为 `ETF_SLOT_05`；",
            "399006 保持独立指数模型，不参与替换。589850/159915 为种子化但无有效行情的 ETF，不入槽。",
        ]
    )
    (out / "V351_ETF_SLOT_AUDIT.md").write_text("\n".join(slots), encoding="utf-8")
    print("audit markdown reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
