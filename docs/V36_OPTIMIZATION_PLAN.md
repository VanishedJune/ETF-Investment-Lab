# V3.6 优化详细方案（活动度驱动 · 8 周体系）

> 整理时间：2026-08-07
> 状态：方案待评审
> 基线：V3.5.1（v352 生产重跑后）

## 0. 目标、范围与约束

### 目标

1. 解决“空仓/不作为”悖论：当前部分市场 Champion 高度空仓（159941 空仓率 95.7%、上涨参与率 0.2%），
   但防空仓门槛却在淘汰交易更积极的候选。
2. 提升挑战与维护重训的有效性：维护重训全项目仅 4 次 ACCEPTED（全在 399006），其余市场 0 次；
   候选大量因 `EXCESS_RETURN_BELOW_0_15`（930）、`WIN_RATE_BELOW_THRESHOLD`（906）、
   `NO_ACTION_DEPENDENCY`（847）被拒。
3. 探索预测层非线性、分状态建模、残差加权与更精细概率校准。
4. 补齐增量周更、性能、数据交叉验证与活动度监控。

### 范围与非目标

- V3.6 使用**新协议命名空间** `V3.6_CAPITAL_DRIVEN_8W`，复用 `v35_*`/`v351_*` 表
  （`protocol_version` 天然隔离）；V3.5、V3.5.1 冻结记录一律不修改。
- 默认 Champion 配置与 V3.5.1 一致；优化项先以“挑战者维度”验证，验证通过后走正式晋级通道。
- 不引入外部宏观/估值数据；159941 与 ETF 仍只使用自身 OHLCV 及派生指标。
- 不改变周线交易节拍、不做日内执行、不做跨 ETF 组合层。

## 1. 现状基线（v352 生产）

| 市场 | 平均仓位 | 空仓窗口 | 上涨参与率 | 平均净收益 |
|---|---:|---:|---:|---:|
| 159941 | 0.2% | 95.7% | 0.2% | +0.01% |
| 512800 | 1.5% | 59.8% | 1.6% | +0.04% |
| 512690 | 4.9% | 27.0% | 4.8% | +0.07% |
| 518600 | 2.9% | 48.8% | 3.1% | +0.22% |
| 512010 | 7.7% | 24.0% | 6.9% | +0.26% |
| 399006 | 6.4% | 45.0% | 6.9% | +0.92% |

- 维护重训：全项目 4 次 ACCEPTED（全部 399006），其余市场 0 次。
- 候选拒绝 Top：`NOT_SELECTED_INNER` 2592、`EXCESS_RETURN_BELOW_0_15` 930、
  `WIN_RATE_BELOW_THRESHOLD` 906、`MEDIAN_PROFIT_GATE_FAILED` 878、`NO_ACTION_DEPENDENCY` 847。
- 性能：六市场全量回放约 2.9 小时（V3.5.1 实测）。

### 新增：批次连续性修复（P0-B 补充 · 2026-08-07）

**问题证据（生产库实测）**

- 强上涨窗口（未来 8W 涨幅 ≥5%（399006）/ ≥8%（ETF））中：
  - 399006：275 个窗口，平均 final_target 18.5pp，实际仓位 0.4pp，实际 <10% 占 97%；
  - 512010：138 个窗口，平均 final_target 38.0pp，实际 1.3pp，实际 <10% 占 96%；
  - 159941：134 个窗口，平均 final_target 22.1pp，实际 0.3pp，实际 <10% 占 100%；
  - 512800：46 个窗口，平均 final_target 9.8pp，实际 0.9pp，实际 <10% 占 100%。
- 结论：低仓位的主因不是“没有看涨信号”，而是执行层把目标仓位变成成交时被反复打断。

**根因**

- `_run_window` 每周用当周新决策的批次整体替换上一周未执行批次；
- 新决策回到 UNCONFIRMED 或空批次时，剩余批次被清空，导致只完成首批 5–10%。

**修复（并入 P0-B，作为执行层硬规则）**

1. 新决策无批次或 final_target 不变时，保留上一周未完成批次继续执行；
2. 仅以下情况清空未完成批次：`TOP_CONFIRMED` / `BEARISH_CONFIRMED` / 紧急风险退出 /
   final_target 明确下调至低于当前已执行仓位；
3. `strong_signal_entry_floor_pp {5,10,15}` 与批次连续性联动：强信号下首笔不得低于下限，
   且后续批次按计划继续；
4. 在 `v36_account_snapshots.pending_batches_json` 持久化未完成批次与冷却状态，
   保证 Champion/Challenger 同起点且中断续跑不丢批次。

**验收（新增）**

- V3.6 回放后统计“强上涨窗口内实际仓位中位数”与“首批后批次完成率”；
- 若强上涨窗口仍低仓位，必须能证明来自预测信号弱，而非批次被打断；
- 单元测试：决策 A（3 批）→ 决策 B（空批次）→ 仍完成 A 的剩余批次；
  决策 B 为 `TOP_CONFIRMED` → 清空未完成批次。

### 补充：剩余硬问题与修复规则（2026-08-07）

**1. 账户口径统一（份额/T+1 记账）**

- schema 26 扩展：`v35_sim_accounts`、`v35_sim_ledger`、`v35_continuous_accounts`
  增加可空列 `held_shares / average_cost / sellable_shares`；
- V3.6 份额模式下写入实际值并以此计算平均仓位、换手与账户 hash；
- V3.5.1 保持 PERCENT 口径，新列写入 NULL，冻结记录不变；
- 验收：同一交易在两种模式下现金/期末资产一致，份额字段仅 SHARES 模式非空。

**2. 校准方法选择防泄漏**

- 固定滚动选择协议：以当前 as_of 为界，取最近 30 个成熟 OOS 样本作为“选择集”
  （前 25 拟合、后 5 验证），在 TEMPERATURE / PLATT / ISOTONIC 中选 Brier 最低者；
- 正式校准器只用“选择集之前”的样本拟合，选择过程写入
  `v35_probability_calibrators.metrics_json.method_selection`；
- 验收：ISOTONIC 不得因同批样本自我验证而恒胜出，选择记录可审计。

**3. 活动度晋级窗口密封**

- 活动度依据只使用“候选训练截止日期之后、且与训练样本无 purged 重叠（间隔 ≥8 周）”的窗口；
- Champion 与 Challenger 使用同一组密封窗口比较；
- 验收：报告列出每个候选活动度窗口数与最早日期，确保全部晚于训练截止。

**4. 维护重训活动度数值阈值**

- `up_opportunity_participation` 或 `strong_up_exposure_pp` 改善 ≥5pp 视为“明显改善”；
  恶化 ≥5pp 视为“明显恶化”；±5pp 内视为“不明显变化”；
- 接受条件：`activity_delta ≥ -5pp`；改善条件之一：`activity_delta ≥ +5pp`；
- 验收：单测覆盖 +5 / -5 / 0 三种情况。

**5. 增量周更完整周判定**

- 新锚点必须是完整自然周：该周最后一个实际交易日 = 该自然周最后一个交易日，
  且收盘行情存在；否则 `processed_weeks=0` 并标记 `PENDING_COMPLETE_WEEK`；
- 全量回放与增量回放必须对同一组完整周锚点一致；
- 验收：以 2026-08-05（周三）构造测试，增量返回 0；周五锚点返回 1。

**6. 决策漏斗口径**

- 漏斗按 `(model_market, forecast_anchor_date, model_package_id)` 粒度统计，
  新增 `v36_decision_funnel` 表（schema 26）；
- 报告分别输出 Champion 与各候选漏斗，159941 归因以 Champion 漏斗为准；
- 验收：同一锚点不同包有各自漏斗记录，报告可区分。

**7. 强上涨基准对照**

- 参考基准：每个强上涨窗口固定 30% 仓位、无成本；
- 报告输出模型 `up_opportunity_participation / strong_up_exposure_pp` 与基准 30pp 的差距；
- “信号弱”判定：仅当强上涨窗口平均 final_target <15pp 且强信号周占比 <30% 时，
  才归因于预测信号弱；否则归因于执行/机制；
- 验收：报告包含基准列与归因结论。

**8. 性能目标降级与并行安全**

- “六市场 <90 分钟”改为尽力目标，不作硬验收；
- 并行仅限同进程多线程计算 + 单 Writer 队列；跨进程禁止共享 SQLite 写连接；
- 校准器缓存与随机种子必须随 run_id 序列化，保证复现；
- 先做单市场基准计时，再决定是否并行；
- 验收：相同 run_id 连续两次回放产出相同 hash。

## 2. 优化项

### P1 活动度门槛与“不作为”惩罚（核心）

**现状问题**

- `no_action = 平均仓位 <5% 且交易次数=0`；STRONG/STABLE 的 `no_action_window_ratio_cap` 均为 0.60，
  `bull_market_min_position_pp` 为 10。
- 结果：Champion 可以在 60% 窗口空仓、上涨市参与率趋近 0 仍不被淘汰；交易积极的候选反而更容易
  触碰“不靠空仓降回撤”等门槛。

**修改**

1. `STRONG_PROMOTION_GATES`：
   - `no_action_window_ratio_cap`：0.60 → **0.40**；
   - 新增 `minimum_up_market_participation_pp`：**10.0**（399006）/ **8.0**（ETF）；
   - 新增 `minimum_average_position_pp`：**5.0**；
   - 新增 `minimum_trade_window_ratio`：**0.30**（至少 30% 窗口有真实交易）。
2. `STABLE_SMALL_EDGE_GATES`：
   - `no_action_window_ratio_cap`：0.60 → **0.40**；
   - 新增 `minimum_up_market_participation_pp`：**8.0**；
   - 新增 `minimum_trade_window_ratio`：**0.35**。
3. 拒绝原因新增 `ACTIVITY_GATE_FAILED`；`gaps_json` 增加
   `activity_no_action_gap`、`activity_participation_gap`、`activity_trade_window_gap`。
4. 实现位置：V3.6 常量组（新增 `v36_config.py` 或在 `v351_config.py` 增加 V3.6 分组）；
   `promotion_channel_for` 增加活动度检查；候选评价与晋级记录持久化新增字段。

**验收**

- V3.6 重跑后：159941 空仓率 <50%、上涨参与率 ≥8%；512800 空仓率 <50%；
- 活动度缺口出现在拒绝原因与报告中，而非静默淘汰；
- V3.5.1 冻结记录不受影响。

---

### P2 维护重训重设计

**现状问题**

- 同配置重拟合必须“OOS MAE 改善”才继续，实践中几乎不满足（其他市场 0 次 ACCEPTED）；
- `NO_OOS_IMPROVEMENT` 等早期返回不落库，无法审计失败原因。

**修改**

1. 接受条件改为：
   - `mean_oos_path_mae ≤ champion_mae × 1.05`；
   - 且满足以下之一：最近 8 个成熟窗口 `mean_excess ≥ -0.0005`，或 MAE 改善 ≥1%；
   - 保留：`brier_degradation ≤ 0.02`、`max_drawdown_degradation ≤ 0.5pp`、
     参与率/空仓率不明显变差（沿用 P1 活动度门槛）。
2. 所有决策（`SKIPPED` / `REJECTED` / `ACCEPTED`）都写入维护重训表，记录
   `reason_code`（如 `INSUFFICIENT_SAMPLES`、`NO_IMPROVEMENT`、`GATE_FAILED`、`ACCEPTED`）。
3. `metrics_before_json` / `metrics_after_json` 增加最近 8 窗口账户指标。

**验收**

- V3.6 每个市场至少出现可审计的维护尝试；拒绝/接受均有 reason_code；
- 若某市场始终无法满足，报告给出分布，而不是无限搜索。

---

### P3 挑战候选池扩展

**现状问题**

- 预测候选 alpha×{0.25,0.5,2.0,4.0}；结构挑战单维度轮换；
- 候选预筛选只看单周差异，不利用近期表现。

**修改**

1. 预测挑战 alpha 集合增加 **{0.10, 1.50}**（裁剪 0.01–200，去重后最多 7 个候选）。
2. 结构挑战维度新增：
   - `model_family`：`RIDGE` / `KERNEL_RIDGE`；
   - `ridge_alpha_multiplier`：{0.5, 2.0}；
   - 仍保持“一次只改一个维度”。
3. 候选生成顺序加入近期表现加权：每第 4 轮结构候选优先从“最近 40 周窗口超额最高”的未尝试维度生成
   （种子固定，可复现）。
4. 行为等价检测阈值不变；候选池耗尽仍按原规则报告原因。

**验收**

- V3.6 候选池的 `forecast_divergence_ratio` / `trade_path_divergence_ratio` 分布相对 V3.5.1 提升；
- `EFFECTIVELY_IDENTICAL` 比例不上升；
- 结构候选仍严格单维度变更。

---

### P4 预测层增强

**4.1 模型家族**

- 新增 `KERNEL_RIDGE`（RBF 核，gamma 由特征方差初始化，`alpha` 沿用候选参数）作为挑战者模型家族；
- 默认 Champion 仍为 `RIDGE`；`model_family` 进入结构挑战维度。

**4.2 残差池加权**

- 场景采样按残差成熟时间指数衰减（半衰期 50 周）；
- `V35RandomPlan.payload` 增加 `residual_weighting="EXP_DECAY_50W"` 与权重向量；
- 场景仍为 `expected + residual_scale × sigma × standardized_residual`（V3.5.1 A1 修复继续生效）；
- 分位展示保持物理边界约束（不出现 <-100%）。

**4.3 概率校准模式**

- `calibration_mode` 增加 `ISOTONIC`（样本 ≥50 时用分段线性单调校准，温度模式保留为默认）；
- `ISOTONIC` 作为结构挑战维度；预测质量保护线（MAE/Brier）继续适用。

**4.4 状态分桶（实验，不默认）**

- `regime_split=UP/DOWN`：分别用上涨/非上涨样本训练子模型，推理时按当期市场状态选择；
- 仅在 399006 测试副本小规模验证（≥40 周）通过后再开放为挑战维度。

**验收**

- `KERNEL_RIDGE` / `ISOTONIC` 挑战者在质量保护线内产生非平凡行为差异；
- OOS MAE/Brier 不劣化；分位边界与残差哈希可复算。

---

### P5 策略执行增强（挑战者维度）

**5.1 波动率目标**

- `vol_target_enabled=true` 时：`final_target × clip(0.10 / σ20w, 0.5, 1.5)`，
  σ20w 为最近 20 周收益年化波动；结果仍按 5% 网格取整、≤80%、≥20% 现金。

**5.2 最小交易频率参数**

- 新增策略参数组 `min_trade_window_ratio`：{0.20, 0.30, 0.40}；
- 该参数参与候选评估与 P1 活动度门槛，不改变 `_run_window` 的成交机制。

**5.3 参与率反馈**

- `decide()` 新增 `recent_up_participation`（最近 40 周连续账户上涨参与率）；
- 低于市场阈值（399006 10% / ETF 8%）时，`UNCONFIRMED` 首笔买入下限从 5% 提高到 10%
  （防“永远试探 5%”），其余分支不变。

**验收**

- 上述参数只出现在挑战者 config，默认 Champion 配置不变；
- 行为差异率可观测；波动率目标不突破全局 80% 上限。

---

### P6 工程与性能

1. **增量周更**：新增 `scripts/v36_bootstrap.py --incremental`，只处理
   `last_completed_anchor` 之后锚点（复用幂等续跑逻辑）。
2. **性能**：挑战评估窗口结果缓存（进程内 LRU + 落库缓存表）、账户/流水批量写、
   校准器增量拟合（V3.5.1 已有）；目标六市场全量重跑 <90 分钟。
3. **数据交叉验证**：新增 `scripts/verify-etf-sources.py`，网络恢复后对 512010 执行
   东财/腾讯/Sina 三方比对并输出报告。
4. **监控**：健康快照增加 `drift_flags`（连续 OOD 周数、参与率骤降、空仓率骤升），
   UI 标记；行情刷新后自动执行 `integrity_check` 与 `foreign_key_check`。

**验收**

- 增量周更：新增一周仅处理新锚点，`processed_weeks=1`；
- 全量重跑 <90 分钟；512010 三方报告生成（网络可用时）。

---

### P7 前端与报告

1. `V35OverviewPanel` 增加“活动度”卡片：平均仓位、空仓率、上涨参与率、防守率、
   最近 8 周交易次数；`ACTIVITY_GATE_FAILED` 时展示缺口值。
2. 工作台顶部增加 `V3.5.1 / V3.6` 版本切换（V3.6 验证通过后默认 V3.6）；
   `/api/v351/*` 增加 `protocol_version` 查询参数（默认当前生效协议）。
3. 新增报告：`V36_IMPLEMENTATION_REPORT.md`、`V36_ACTIVITY_REPORT.md`
   （每市场活动度、门槛缺口、维护重训统计、挑战多样性）。

**验收**

- Playwright 覆盖活动度卡片与版本切换；报告数字与生产库一致。

## 3. 实施阶段

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 备份、测试副本、V3.6 命名空间与常量组（无 schema 变更） | `audit/v36/backups/` |
| P1 | P1–P3（活动度门槛、维护重训、候选池）+ 单测 + 小规模回放 | 门槛/重训/多样性验证 |
| P2 | P4–P5（模型家族、残差加权、校准、波动率目标等）+ 测试副本全量回放 | 预测/策略对照 |
| P3 | P6–P7（增量周更、性能、监控、前端）+ 全量回归 | 报告、UI 验收 |
| P4 | 生产重跑 V3.6 命名空间 + EXE 打包验收 | V3.6 正式基线 |

## 4. 测试与验收矩阵

| 优化项 | 关键测试 | 预期 |
| --- | --- | --- |
| P1 | 活动度门槛单测 + 回放统计 | 159941/512800 空仓率 <50%，参与率 ≥8% |
| P2 | 维护重训 reason_code 落库 | 每市场可审计尝试；拒绝原因可统计 |
| P3 | 候选池多样性单测 | 差异率分布提升、等价比例不升 |
| P4 | KERNEL_RIDGE/ISOTONIC 质量线 | MAE/Brier 不劣化，分位边界有效 |
| P5 | 波动率目标/参与率反馈单测 | 默认 Champion 不变，挑战者差异可观测 |
| P6 | 增量周更 + 计时 | 增量 1 周；全量 <90 分钟 |
| P7 | Playwright 活动度/版本切换 | 卡片与切换可用，报告一致 |

## 5. 假设与默认

- V3.6 使用新协议命名空间；V3.5 / V3.5.1 冻结记录不修改，可随时对比。
- 默认 Champion 配置不变，优化项通过挑战者维度验证后由正式晋级通道提升；
  “不降门槛制造晋级”原则不变。
- P1/P2 的具体数值（0.40、8%/10%、0.30/0.35、MAE×1.05 等）为初版默认；若测试副本显示
  过严/过松，按验证结果调整并在报告中说明，不允许为凑晋级而放松。
- 不引入外部数据、不改变周线节拍、不做日内执行、不做跨 ETF 组合层。
- 生产重跑前备份 `audit/v36/backups/`，先测试副本全量验证再写生产。
