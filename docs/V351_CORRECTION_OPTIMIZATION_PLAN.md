# V3.5.1 修正与优化方案（桌面 EXE 全量）

> 整理时间：2026-08-06
> 状态：已实施（2026-08-06 · v352 生产基线）；A4 性能目标未完全达标，详见
> `V351_KNOWN_ISSUES.md`
> 配套记录：`docs/V351_KNOWN_ISSUES.md`（问题明细）

## 0. 总原则与约束

- 不重做项目、不更换技术栈、不删除旧记录。
- V3.5_CAPITAL_DRIVEN_8W 冻结记录永不修改；所有涉及预测/策略/账户的修复只作用于
  v351 命名空间。
- 每次正式迁移/重跑前做 SQLite 在线备份；先测试副本验证，再写生产库。
- 影响 v351 预测、策略、账户的修复（A1、A2、A3）需要重跑六个市场
  （399006/159941/518600/512800/512690/512010）的 v351 全历史回放。
- 完成后重新打包 EXE 并通过启动/单实例/重启/备份恢复验收。

---

## 1. 模型与架构问题（A 组）

### A1 场景残差尺度错误（P10/P50/P90 越界）

**现状与证据**

- `v35_residual_records.standardized_residual_json` 存的是 `残差 / sigma`
  （`v35_runtime_service.py` 约 1022 行）。
- 场景构造直接使用 `expected + residual_scale × standardized_residual`
  （`v35_runtime_service.py` 约 777 行），没有乘回 `sigma`。
- 结果：最新预测出现 1W P10 = -138.31% 这类低于 -100% 的累计收益；P50/P90 也与
  expected 严重偏离。策略打分风险项 `risk = clip(-p10_8 / 0.15 - …)`
  （`v35_strategy_service.py` 约 205 行）被放大后的 P10 长期推到最高风险。

**修正方案**

1. `_generate_forecast` 场景构造改为：
   `scenarios = expected + residual_scale × sigma × standardized_residual`
   （sigma 取自对应残差记录的 `source_sigma`；若希望直接使用原始残差，则改用
   `residual_path_json`，两者取其一并固定写入协议文档）。
2. `V35RandomPlan.payload` 增加 `residual_unit` 与 `sigma_version`，防止新旧计划混用。
3. 前端（并入 A5）：预测表增加“期望”列；P10/P50/P90 展示时对越界值给出
   “模型场景未约束，仅供参考”提示，或按 [-100%, +∞) 物理边界截断显示。

**验收标准**

- 重跑后所有市场最新预测的 P10/P50/P90 与 expected 同量级，不再出现 <-100% 的累计收益。
- risk 组件不再恒为 1.0；相同输入下 `forecast_hash` 稳定可复算。
- Playwright 断言预测表显示期望列与边界提示。

**影响**：v351 全部预测/策略/账户；需重跑回放。

---

### A2 卖出逻辑在生产回放中未生效

**现状与证据**

- `_OnlineAccount.execute_and_mark()` 定义了但从未被调用。
- `bootstrap_sync` 中 `online.position_pp` 恒为 0；`_process_week` 以
  `current_position_pp=0` 调用 `decide()`，`final_target < current_position` 的
  卖出分支永不触发。
- 生产库证据：V3.5 的 2632 条与 v351 的 4503 条 `v35_position_decisions` 全部为
  `BUY`，没有任何 `SELL`。
- 挑战评估 `_recompute_decisions` 同样传入 0，等价放大该问题。

**修正方案**

1. 在 `bootstrap_sync` 每周循环中，先执行
   `online.execute_and_mark(anchor, bars[anchor])`，再调用 `_process_week`；
   `_process_week` 内 `decide(current_position_pp=online.position_pp)`。
2. `_run_window_accounts` 的标准 8W 账户仍从 0 开始模拟（冻结决策不变），但决策
   本身现在能产生 SELL/减仓批次。
3. `_recompute_decisions`（挑战评估）改为传入候选窗口内逐周累计后的真实仓位：
   可先按冻结策略跑一遍窗口得到 `position_pp` 序列，再以该序列作为 `decide` 输入，
   消除“决策假设 0 仓位”的系统性偏差。
4. 新增回归测试：构造 current_position>0 且 final_target 更低的输入，断言输出
   SELL 批次；断言 v351 重跑后 `v35_position_decisions` 出现非零 SELL。

**验收标准**

- 重跑后六个市场均出现 SELL/减仓批次与对应账户流水。
- 紧急风险退出（严重 OOD）、确认减仓、轻度减仓分支有真实样例可查。
- V3.5 命名空间哈希不变。

**影响**：v351 全部决策与账户；需重跑回放。

---

### A3 冷却期参数未真正约束执行节奏

**现状与证据**

- `_run_window` 只检查 `sequence - last_trade_sequence >= 1`（每周最多一批），
  不读取批次里的 `cooldown_trading_days`。
- 默认 5 天恰好约等于一周，因此默认参数无感；但策略挑战者 10/15 天冷却变体
  不会产生额外间隔，挑战维度失效。
- `_decisions_for_anchor` 重建决策时把 `cooldown_trading_days` 硬编码为 5，
  重放路径丢失挑战者参数。

**修正方案**

1. `_run_window` 中 `min_gap_weeks = ceil(cooldown_trading_days / 5)`，
   成交条件改为 `sequence - last_trade_sequence >= min_gap_weeks`。
2. `_decisions_for_anchor` 从 `condition_json` 读取原始
   `cooldown_trading_days`，删除硬编码 5。
3. 回归测试：同一决策在 cooldown=5/10/15 下，模拟成交序列分别为每 1/2/3 周一批。

**验收标准**

- 冷却参数进入实际交易路径；策略挑战者 10/15 天变体与 5 天变体产生可观测差异。
- 行为等价检测能区分不同冷却参数的候选。

**影响**：v351 挑战评估与账户回放；需重跑回放。

---

### A4 v351 回放性能

**现状**：`_run_window_accounts` 每个锚点都重新调用 `self._anchors(session, market)`，
存在重复查询；六市场全量回放约 2 小时。

**修正方案**

1. 缓存 anchors 与全部 8W 窗口列表，循环外一次性构造。
2. 账户/流水改为批量 upsert（按窗口批量 `session.add_all` + 分批 commit）。
3. 校准器增量拟合：记录每个市场已参与校准的最新 evaluation 时间，只对新评价重算
   temperature/Brier，避免每锚点全表扫描。
4. 复用已存在的 `V35RandomPlan`（已有 plan_hash 去重），避免重复压缩/入库。

**验收标准**

- 六个市场回放总耗时目标 < 60 分钟；结果哈希与逻辑修复后的基准一致。
- 幂等续跑仍为 `processed_weeks=0`。

---

### A5 前端预测表可读性

**现状**：`V35OverviewPanel.vue` 只显示 P10/P50/P90，不显示期望路径；越界值无提示。

**修正方案**：增加“期望”列；越界值提示或按物理边界截断展示；文案说明 P10/P50/P90
为 1000 个场景的累计收益分位数。具体并入 A1 一并实施。

---

### A6 其他架构小项

1. **槽位替换异步化**：`V351SlotService.replace` 当前在 HTTP 请求内同步执行完整
   Bootstrap；改为后台任务（FastAPI BackgroundTasks 或本地 worker）+ 状态轮询；
   `cancel-replacement` 真正支持取消，替换期间可查看历史绑定。
2. **账本行级成本**：`V35SimLedger.transaction_cost` 恒为 0；改为记录每笔成交的
   费用与滑点明细。
3. **候选评价协议隔离**：`v351_candidate_evaluations` 增加 `protocol_version` 列
   （schema 25，幂等迁移）。
4. **512010 数据源交叉验证**：网络恢复后重试东财全量日K，与腾讯/Sina 三方比对并
   输出验证报告。

---

## 2. 桌面 EXE 前端问题（B 组）

### B1 其他 ETF 未接入数据分析

**现状与根因**

- 工作台 `selectedModel` 只允许 399006/159941（`ResearchWorkbenchView.vue`），
  其他 ETF 只显示行情与“仅展示”提示。
- 后端 `v35` API 的 `_market()`、`v343/v341/v342` 服务的 `validate_market()` 都只
  允许 MODEL_MARKETS（399006/159941）。
- `V35OverviewPanel` 使用 `/api/v35/*`（V3.5 协议），而 ETF 槽位的正式数据在
  v351 命名空间，前端没有对应读取接口。

**修正方案**

1. **后端 v351 数据路由**：新增
   `/api/v351/status|status/{market}|forecast/{market}|strategy/{market}|simulation/{market}|champions/{market}`
   （或给 `/api/v35/*` 增加 `protocol_version=v351` 参数，推荐前者，保持旧接口只读）。
   市场校验改为“399006/159941 或活动 ETF 槽位中的代码”。
2. **前端市场模型泛化**：`ActiveInstrumentCode` 扩展为模型标的选择类型；`selectedModel`
   由“两市场白名单”改为“399006 + 活动槽位 ETF”；ETF 槽位加载 v351 状态/预测/策略/账户。
3. **数据分析动作**：对 ETF 槽位，点击“数据分析”读取最新 v351 冻结预测/策略/账户
   （Bootstrap 已完成，无需训练）；对 399006/159941 保留 V3.4 13W 分析作为历史协议，
   但主工作台统一展示 8W v351 面板，V3.4 面板折叠为只读历史审计。
4. **界面策略（推荐）**：新建/扩展 `V351AnalysisPanel`，五个 ETF 与两个原模型标的
   使用同一 8W 面板；13W 面板保留入口但标注“历史协议，只读”。

**验收标准**

- 512010/512800/512690/518600/159941 均能点击数据分析并展示 8W 结果。
- Playwright：新增“512010 数据分析展示预测/策略/账户”用例。
- 399006/159941 的 V3.5 旧接口与记录不受影响。

---

### B2 分批仓位建议为空；拐点窗口/一致性检查不同步

**现状与根因**

- V3.4 面板的“分批仓位建议”来自 V3.4.2 policy 的 `decision.batches`；只有当
  已确认拐点且满足触发条件时才有批次，空批次时只显示“当前没有可执行批次”，
  没有说明“预测需要建仓但批次为空”的具体原因（等待确认/硬条件/无目标差）。
- “价格拐点窗口 / DIF一阶变化零点窗口 / 一致性检查”是 V3.4.2 13W 专属；v351
  没有等价 8W 计算与 UI。
- 同步链路：`/api/v343/forecast/{market}` 已附带最新 policy，但前端切换市场后
  `v34Analysis` 未按 `result.market === selected` 校验，存在旧结果残留风险；
  重新分析完成前的轮询窗口内也会显示旧数据。

**修正方案**

1. **v351 8W 拐点与一致性评估（后端新增）**：
   - 价格拐点窗口：基于 P50 预测路径的一阶符号变化 + MA20/MA60 状态 + 预测概率边，
     输出 `window_start/end`、`turn_kind`、`confirmation_status`；
   - DIF 一阶变化零点窗口：基于预测 DIF 路径（由特征服务外推）过零位置；
   - 一致性检查：价格拐点与 DIF 零点窗口的时间一致性
     （沿用 V3.4.2 `TEMPORALLY_CONSISTENT` 语义，改为 8W 口径）；
   - 结果持久化到 v351 策略快照 `strategy_json`（或新增
     `v351_turning_assessments` 表，schema 25），并随策略 API 返回。
2. **分批仓位建议补全**：
   - 从 `V35StrategySnapshot + V35PositionDecision` 生成批次展示数据，补充
     `earliest_execution_week`（下一完整交易周）、窗口起止日期、冷却天数、触发条件；
   - `final_target > current_position` 但批次为空时，返回原因枚举
     （`WAITING_CONFIRMATION` / `HARD_CONDITION_BLOCK` / `BELOW_MIN_BATCH` 等）。
3. **前端同步链路**：
   - `loadV34State`/`loadV351State` 加载最新预测时同时加载最新分析结果，并校验
     `result.market === selected`，不匹配则不渲染；
   - 切换市场时清空对应分析 ref；分析按钮完成后强制刷新同一 ref；
   - V35OverviewPanel 增加四个区块：分批仓位建议、价格拐点窗口、DIF一阶变化零点
     窗口、一致性检查；空批次显示原因而不是空白。

**验收标准**

- 当 final_target > 当前仓位时，“分批仓位建议”非空且每个批次带执行窗口与条件；
- 拐点窗口/一致性检查与最新一次数据分析同 run、同 hash；
- 连续点击两次“数据分析”，结果 hash 不变（幂等）；
- 切换 512010 → 512800 → 512010 不出现旧市场结果残留。

---

### B3 历史周K + P50 代表性预测周K 不同步

**现状与根因**

- 当前 13W 图表由 `V34ModelPanel` 使用 `props.result` 渲染；不同步根因与 B2 的
  同步链路问题相同（旧结果/旧市场残留），且口径仍是 13W，与 8W 新体系不一致。
- v351 的 `V35Forecast.representative_ohlcv_json` 当前为空数组，
  `indicator_path_json` 为空，无法直接渲染预测K线。

**修正方案**

1. **v351 预测K线构建（后端）**：新增
   `v351_forecast_payload_builder`：
   - 历史部分：真实周K（最新 52 根）+ 对应 DIF/DEA/MACD/DIF一阶变化；
   - 预测部分：以锚点收盘价为基准，按 `expected_path` 构造 8 根 P50 代表性预测K线
     （open/high/low/close），P10/P90 构造价格带；
   - 预测 DIF/DEA/MACD/DIF一阶变化路径由特征服务外推并持久化到
     `indicator_path_json`；
   - API 返回与 `V34ForecastPayload` 对齐的字段（`historical_ohlcv`、
     `representative_ohlcv`、`indicators`、`price_quantiles`），并附带
     `forecast_hash`。
2. **前端图表**：`V351AnalysisPanel` 图例/标题改为“历史周K + 8根P50代表性预测周K”，
   markPoint 标注 1/4/8 周；图表数据与 result 同源；切换市场强制重渲染并校验锚点。
3. **V3.4 13W 面板**：保留为只读历史，不再参与“当前分析”主流程。

**验收标准**

- 六个市场均渲染 52 根历史周K + 8 根预测K线 + P10/P90 价格带 + 预测指标；
- 图表锚点与最新 forecast 一致；悬停/缩放回归通过；
- Playwright 断言 512010 图表标题含“8根P50”。

---

## 3. 实施阶段

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 备份、测试副本、schema 25 迁移（若需协议列/评估表） | `audit/v352/backups/`、迁移报告 |
| P1 | A1、A2、A3 修复 + 单测 + 测试副本回放 | v351 新回放、差异报告 |
| P2 | A4 性能、A6 小项 | 性能报告、schema 25 |
| P3 | B1、B2、B3 后端 API + 前端面板 | v351 分析 API、8W 分析面板 |
| P4 | 全量回归 + EXE 打包验收 | 测试报告、EXE 验收 |

## 4. 测试与验收矩阵

| 问题 | 关键测试 | 预期 |
| --- | --- | --- |
| A1 | 场景分位边界 | P10/P50/P90 不再 <-100%，risk 不饱和 |
| A2 | 卖出分支单测 + 回放统计 | `v35_position_decisions` 出现 SELL |
| A3 | 冷却 5/10/15 成交序列 | 每 1/2/3 周一批 |
| A4 | 六市场回放计时 | <60 分钟，幂等 0 周 |
| B1 | ETF 数据分析 Playwright | 5 个 ETF 均可分析并展示 8W 结果 |
| B2 | 批次/拐点/一致性同步 | 非空批次带窗口；无旧结果残留 |
| B3 | 预测K线渲染 | 52+8 根，锚点一致 |
| 回归 | 后端全套 + 前端 build + Playwright | 全绿；V3.5 哈希不变 |

## 5. 风险与回滚

- A1/A2/A3 重跑 v351 会改变当前 v351 正式记录：先备份
  `audit/v352/backups/investment_lab_pre_v352_*.db`，测试副本全量验证后再生产；
  如需回滚，恢复备份并复用 `v351_bootstrap.py` 续跑。
- 冷却期修复会改变挑战者行为基线与行为等价统计，重跑后需重新生成
  `V351_CHALLENGER_EFFECTIVENESS_REPORT.md`。
- B 组改动只新增接口/页面，不修改冻结记录，风险集中在前端回归与 EXE 打包。
