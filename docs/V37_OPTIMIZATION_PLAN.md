# V3.7 DeepSeek 融合实施计划（两阶段交付）

> 状态：已批准，实施中（2026-08-07，含当日补充修正）  
> 协议版本：`V3.7_DEEPSEEK_FUSION_8W`  
> 基线：代码层 V3.6（schema 26），schema 28 追加 V3.7 命名空间；V3.6 生产回放按决策跳过

## 1. 定位

建立四条同时运行、相互比较的模型路径：

```text
LOCAL_QUANT             本地日周联合模型（MULTI_TIMEFRAME_LOCAL_MODEL）
DEEPSEEK_AI             DeepSeek 独立市场判断（INDEPENDENT_ANALYST）
QUANT_AI_FUSION         Local + AI 确定性融合
LOCAL_REPAIR_CHALLENGER DeepSeek 诊断后生成的本地修复候选
```

DeepSeek 只有“预测权”和“提案权”：不能绕过本地回放门禁、不能直接修改正式 Champion、不能自动修改源码。正式 AI 权重与晋级只由 V3.7 上线后的 Forward OOS 决定；历史匿名筛选只用于 Prompt/Fusion/Repair 研究。

## 2. 约束（继承 V3.6 已确定项）

- Point-in-Time、Purged Walk-Forward、8 周预测/成熟评价、真实账户状态、T+1 可卖份额、SELL 逻辑、未完成批次跨周继承、Champion/Challenger 同起点、活动度密封窗口、概率校准防泄漏、EFFECTIVELY_IDENTICAL、完整周锚点、动态 ETF 槽位、增量周更、行情刷新失败保护。
- V3.5 / V3.5.1 / V3.6 冻结记录永不修改；V3.7 使用独立 `protocol_version`；所有写接口必须显式携带协议版本。
- 所有 AI/融合账户走同一执行引擎（费用/滑点/日历/快照一致），AI 不突破 80% 仓位上限、不做空。

## 3. P1 本地日周联合模型

### 周K分支（52–104 周）

OHLCV、1/2/4/8/13/26/52 周收益、MA/EMA、DIF/DEA/MACD、DIF 一阶/二阶变化、MACD 柱变化、量比、20/52 周波动率、当前回撤、距高低点距离、趋势持续周数、均线排列、价格相对均线位置。

### 日K分支（60–100 交易日）

OHLCV、1/3/5/10/20/60 日收益、MA/EMA、DIF/DEA/MACD、DIF 一阶/二阶变化、成交量变化、量价背离、20/60 日波动率、近期高低点与位置、连续涨跌天数、短周期趋势强度。

未完成自然周：日K 可参与“当前状态分析”，但不得伪装成完整周K；正式回放严格 point-in-time。

### 日周交互特征（11 项）与状态

`daily_weekly_direction_agreement / macd_agreement / dif_agreement / momentum_gap / volatility_ratio / trend_strength_gap / daily_reversal_weekly_weak / weekly_bullish_daily_pullback / weekly_bearish_daily_recovery / daily_weekly_double_bullish / daily_weekly_double_bearish`。

`MULTI_TIMEFRAME_STATE` 七状态：`BOTH_BULLISH / BOTH_BEARISH / DAILY_BULLISH_WEEKLY_BEARISH / DAILY_BEARISH_WEEKLY_BULLISH / DAILY_RECOVERY_WEEKLY_UNCONFIRMED / WEEKLY_UPTREND_DAILY_PULLBACK / NEUTRAL_MIXED`。

### 三种本地架构

- `WEEKLY_ONLY`：V3.6 周模型基准，不删除。
- `EARLY_FUSION`：周+日+交互特征一并进入 Multi-output Ridge（V3.7 初始 Champion 默认）。
- `LATE_FUSION`：日/周分别训练，按 1–2W=40/60、3–4W=30/70、5–8W=20/80 融合；权重仅为初始 Challenger 配置，禁止按历史区间人工调参。

## 4. P2–P5 DeepSeek AI

### MULTI_TIMEFRAME_AI_PACKET_V1

- 最近约 60 个交易日紧凑结构化数据（date_relative、OHLC、volume_ratio、return、MA、DIF/DEA/MACD、DIF 一阶变化、volatility）。
- 最近约 52 周K（week_relative、OHLC、volume_ratio、return、MA、DIF/DEA/MACD、DIF 一阶/二阶变化、volatility、drawdown）。
- 本地确定性摘要（由代码计算：日/周趋势、一致性、波动、量能、高低点、日周冲突类型），**不得包含本地模型预测结果**。

### 角色隔离

- `DEEPSEEK_INDEPENDENT_ANALYST`：只看日K、周K、交互特征与账户基本状态；禁止看到 Local 预测/Score/仓位/漏斗。
- `DEEPSEEK_MODEL_REVIEWER`：允许读取 Local 预测、P10/P50/P90、StrategyScore、仓位、决策漏斗、最近 OOS、活动度、Champion 配置、拒绝原因；输出 `LOCAL_MODEL_REPAIR_PROPOSAL`。

### 输出与校验

严格 JSON：`trend_1w/2w/4w/8w`（枚举）、`direction_score_1w..8w ∈ [0,1]`、`daily_trend / weekly_trend`、`multi_timeframe_state`、`risk_level`、`confidence_raw`、`expected_return_4w/8w`、`support/resistance_distance_pct`、`reason_codes`。非法→`AI_INVALID_RESPONSE` 且不进策略；API 故障→`AI_API_UNAVAILABLE/TIMEOUT`，本地模型照常运行并记录 `AI_FALLBACK_LOCAL_ONLY`；禁止用上周 AI 结果替代本周。

### 缓存与审计（2026-08-07 修正）

- `v37_ai_requests` / `v37_ai_forecasts` 冻结每次调用；“同一输入不得重复调用”只约束**成功且冻结**的结果；TIMEOUT/RATE_LIMITED/NETWORK_ERROR/MODEL_NOT_AVAILABLE 等瞬时失败允许有限重试（默认最多 3 次尝试，每次尝试独立 `attempt_number` 请求行，失败行永不覆盖/伪装成功）。
- prompt/模型/输入结构变化时 `ai_generation_version++`，旧结果保留。
- 每次调用写入 `AI_INPUT_POINT_IN_TIME_PASS`；历史回放输入匿名化（`MARKET_A`、`T-n`、`anchor_close=1.0`、量比归一化，禁证券名称/代码/绝对日期/事件/未来标签）。

### 历史筛选与 Forward OOS 隔离（2026-08-07 补充）

- `v37_ai_forecasts` / `v37_ai_requests` 带 `screening_scope`（`HISTORICAL_SCREENING` / `FORWARD_OOS`）。
- 历史每市场 120 个匿名锚点仅用于 Prompt/Fusion/Repair 研究，**不得进入**正式 AI 校准、权重解锁或 Champion 晋级；正式统计只读取 `FORWARD_OOS`。

### AI Challenger 与校准（2026-08-07 修正）

- 每个完整周锚点、每市场 1 次独立分析；Local/AI/Fusion 从同一 `v36_account_snapshots` 快照出发，走同一执行引擎。
- AI 原始 confidence 记 `ai_raw_score`；校准后才成为 `ai_calibrated_probability`。
- **校准与权重档位**：成熟 Forward OOS <30→`AI_CALIBRATION_UNAVAILABLE` 且正式 Fusion AI 权重=0（AI_SHADOW）；30–49→`AI_PRELIMINARY_CALIBRATION`，允许最高 10%；50–99→允许 10%/20%/30% Challenger；≥100→验证通过后允许最高 40%。权重提升必须经过 OOS、收益、回撤、活动度等门禁，不能仅按样本数量自动提升。
- 校准仅使用成熟 Forward OOS，沿用 V3.6 滚动选择协议（TEMPERATURE/PLATT/ISOTONIC）。

## 5. P6–P7 Fusion 与权重递进（2026-08-07 修正）

- 初始融合候选：LOCAL 90/10、80/20、70/30、60/40（AI 上限 40%）；融合只使用 `ai_calibrated_probability`，不可用时该候选退化为 LOCAL_ONLY。
- 一致性/冲突状态写入 `v37_model_conflicts`；冲突候选（不调整/−5pp/−10pp/等待确认）由 OOS 决定，禁止预设“冲突必听 AI”。
- AI 权重按 Forward OOS 成熟窗口开放：<30 窗口 AI_SHADOW（正式权重=0）、30–49 最大 10%、50–99 最大 30%（Challenger）、≥100 验证通过后最大 40%；叠加方向质量、校准有效、账户不差于 Local、防空仓、回撤门禁；退化置 `AI_WEIGHT_DEGRADED` 自动降权。

## 6. P8–P9 Reviewer 与 Repair

- 每新增 8 个成熟样本或检测到 drift 触发 `DEEPSEEK_MODEL_REVIEW`。
- 白名单参数（仅允许修改，**共 14 项**）：ridge_alpha、training_window、feature_subset、daily_feature_set、weekly_feature_set、daily_weekly_fusion_weight、residual_half_life、calibration_mode、model_family、strong_signal_threshold、entry_floor、vol_target、conflict_penalty、risk_gate_parameter；取值只能在系统已验证范围。
- 自动生成 `LOCAL_REPAIR_CHALLENGER`，经本地完整回放与 STRONG/STABLE 晋级后才能成为 Champion；架构级变更只落 `ARCHITECTURE_CHANGE_PROPOSAL` 报告，绝不自动修改源码/打包/自评。
- **Repair 未来泄漏约束（2026-08-07 补充）**：Reviewer 在 T 依据 T 及以前数据提出提案后，可用历史数据训练与 sanity check，但正式晋级证据只能来自 `proposal_anchor=T` 之后新产生并成熟的 OOS 窗口（与训练样本间隔 ≥8 周）；禁止“看完一段历史→用同一段历史证明 Repair 优于 Champion→直接晋级”。

## 6a. 账户隔离核查（2026-08-07 补充）

- `v35_sim_accounts` 唯一键已包含 `model_package_id`；`v35_continuous_accounts` 唯一键已包含 `protocol_version + model_package_id`。
- 四条路径使用不同包 ID：LOCAL=Champion 包、DEEPSEEK_AI=AI 伪包、QUANT_AI_FUSION=Fusion 配置包、LOCAL_REPAIR_CHALLENGER=修复候选包；**已确认隔离，不改 schema、不重构账户体系**。

## 7. 数据库（schema 29）

11 张 `v37_*` 表：`v37_multitimeframe_features / v37_ai_requests / v37_ai_forecasts / v37_ai_evaluations / v37_ai_calibrators / v37_ai_model_health / v37_fusion_configs / v37_fusion_evaluations / v37_model_repair_proposals / v37_model_repair_runs / v37_model_conflicts`。

AI/融合账户写入现有 v35/v36 账户与快照表（按 protocol_version 隔离）+ 新增 v37 表；禁止跨进程并发写库，只允许单 Writer。

## 8. 回放与交付

- 测试副本先每市场约 40 周小规模验证，再做本地 V3.7 全历史回放与有界 AI 匿名筛选（每市场 120 锚点、总计约 720 次真实调用；同输入缓存去重只减不增）。
- 校验 V3.5/V3.5.1 哈希不变、V3.6 表行数不变、幂等续跑 `processed_weeks=0`。
- 生产备份 `audit/v37/backups/` 后执行 V3.7 全历史回放 + AI 筛选落库 + 开启 Forward OOS；重新打包 EXE（AkShare calendar 硬门禁、AI 设置页、多模型页面）。
- 输出 12 份 V37 报告，核心为 `V37_LOCAL_VS_AI_REPORT.md`。

## 9. 完成标准

本地模型真正同时使用日K与周K；AI 每周同时使用日K+周K；AI 具备独立 Challenger 并与 Local 同账户条件比较；Fusion 只使用经校准的 AI 信号；Repair 经本地回放后才能晋级；API 故障不影响本地；历史 AI 输入 point-in-time 且匿名；V3.6 冻结记录不变；所有模型、账户、交易、AI 调用和晋级可复算、可审计。
