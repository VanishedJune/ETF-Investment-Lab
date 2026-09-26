# V3.5.1 已知问题与优化清单

> 记录时间：2026-08-06。状态：OPEN = 尚未修复；FIXED = 本次实施中已修复。
> 其中标注「需重跑 v351」的修复会改写当前 v351 命名空间正式记录，需用户确认后执行；
> V3.5_CAPITAL_DRIVEN_8W 旧记录一律不动。

## 实施状态（2026-08-06 · v352 基线）

- A1 残差场景尺度：**FIXED**（已重跑 v351 六个市场，分位不再 <-100%）。
- A2 卖出逻辑未生效：**FIXED**（重跑后 `v35_position_decisions` 出现 393 条 SELL）。
- A3 冷却期参数：**FIXED**（模拟器按 `ceil(days/5)` 周约束批次间隔）。
- A4 回放性能：**PARTIAL**（窗口查询与校准器已优化，但六市场重跑仍约 2.9 小时，未达 <60 分钟目标）。
- A5 前端预测表：**FIXED**（增加“期望”列与越界截断提示）。
- A6 槽位异步替换 / 账本行级成本 / schema 25 协议列：**FIXED**；
  512010 东财三方交叉验证：**OPEN**（待网络恢复后执行）。
- B1 全市场数据分析、B2 分批/拐点/一致性、B3 8W 预测K线：**FIXED**
  （统一 8W v351 面板，V3.4 13W 降为只读历史）。

## OPEN

### 1. 场景残差尺度错误（P10/P50/P90 越界）

- 现象：未来 8 周预测表中出现 1W P10 = -138.31% 等低于 -100% 的累计收益，P50/P90 也与
  预测均值严重偏离。
- 根因：残差落库时除以 `sigma` 标准化（
  `backend/app/services/v35_runtime_service.py` 的 `standardized_residual_json`），
  场景构造却直接使用 `expected + residual_scale × standardized_residual`，没有乘回 `sigma`。
- 影响：展示失真；策略打分的风险项 `risk = clip(-p10_8 / 0.15 - …)` 会被放大后的 P10
  长期推到最高风险，影响 StrategyScore 与仓位决策。
- 建议修复：`expected + residual_scale × sigma × standardized_residual`（或直接使用未标准化残差）。
- 修复范围：需重跑 v351 六个市场回放。

### 2. 冻结决策全部为 BUY，卖出逻辑在生产回放中未生效

- 现象：`v35_position_decisions` 中 V3.5 与 V3.5.1 命名空间分别只有 2632 / 4503 条，
  全部是 `BUY`，没有任何 `SELL`。
- 根因：`_OnlineAccount.execute_and_mark()` 从未被调用；`bootstrap_sync` 中
  `online.position_pp` 恒为 0，`decide(current_position_pp=0)` 使
  `final_target < current_position_pp` 的卖出分支永远不触发；挑战评估
  `_recompute_decisions` 同样传入 0。
- 影响：方案要求的“卖出与买入基本对称”、紧急风险退出、确认减仓等在当前正式回放中没有
  实际发生，账户只累积买入并持有。
- 建议修复：让在线账户按周真实执行批次并更新 `position_pp`，或让窗口模拟把“当前仓位”
  回传给决策（例如在 `_process_week` 中按上一窗口结果结算后传入真实仓位）。
- 修复范围：需重跑 v351 六个市场回放。

### 3. 冷却期参数未真正约束执行节奏

- 现象：`_run_window` 只检查 `sequence - last_trade_sequence >= 1`（每周最多一批），
  不读取批次里的 `cooldown_trading_days`；默认 5 天恰好约等于一周，因此默认参数无感，
  但策略挑战者的 10/15 天冷却变体不会产生额外间隔。
- 连带问题：`_decisions_for_anchor` 重建决策时把 `cooldown_trading_days` 硬编码为 5，
  即使未来执行器支持冷却，重放路径也会丢失挑战者参数。
- 建议修复：模拟器按 `ceil(cooldown_days / 5)` 周数约束批次间隔；持久化原始参数。
- 修复范围：需重跑 v351 六个市场回放。

### 4. v351 回放性能

- `_run_window_accounts` 在每个锚点都重新调用 `self._anchors(session, market)`，
  存在重复查询；六市场全量回放合计约 2 小时。
- 建议：缓存 anchors/窗口列表、批量持久化账户与流水、减少每次校准器的重复全表扫描。

### 5. 前端预测表可读性

- `V35OverviewPanel.vue` 只渲染 P10/P50/P90，不显示期望路径；异常分位（<-100% 或 >+100%）
  没有下限或提示。
- 建议：增加“期望”列；对越界值显示说明或按物理边界截断展示（模型内部仍保留原始值）。

### 6. ETF 槽位替换同步阻塞

- `replace` 在 HTTP 请求内同步执行完整下载与 Bootstrap，长任务会阻塞请求；
  `cancel-replacement` 当前只是占位返回。
- 建议：改为后台任务 + 轮询状态；替换期间允许查看历史绑定。

### 7. 账本行级成本明细缺失

- `V35SimLedger.transaction_cost` 恒为 0，成交费用与滑点只体现在账户总成本里。
- 建议：在每行记录当笔费用/滑点，便于回放审计。

### 8. v351 候选评价表缺少协议版本列

- `v351_candidate_evaluations` 目前只有 v351 数据，未来若出现新协议版本需要
  `protocol_version` 隔离，否则会混用。

### 9. 512010 数据源交叉验证

- 实施期间东财接口远端断连，512010 使用腾讯全量日K + Sina 重叠区间校验。
- 建议：后续网络恢复后重试东财全量数据做三方交叉验证，并补充月度数据源校验。

## FIXED（本次实施中已修复）

- `persist_continuous_result` 按 `protocol_version + model_market` 隔离，恢复被覆盖的
  V3.5 连续账户行并重建 v351 连续账户（17/17 表哈希一致）。
- ETF 白名单泛化：任意 6 位 ETF 代码可按前缀识别交易所并下载。
- `EtfSlotManager.vue` 缺少 `onMounted` 导致页面打开不加载槽位。
- 159941 槽位交易所被 `_slot_exchange` 误写为 SSE，改为优先取 Instrument.exchange（SZSE）。
- 打包审计脚本与测试中 schema 23 → 24 的陈旧断言。
