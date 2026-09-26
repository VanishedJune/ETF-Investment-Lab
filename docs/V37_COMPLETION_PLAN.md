# V3.7 补完与验收实施方案（2026-08-07 · 补充版）

> 前置：V3.7 阶段一（P0–P5）与阶段二服务代码已完成；程序启动失败已修复并通过官方验收；本方案补齐未开展部分并定义验收门禁与统计口径。

## 1. 程序启动失败：已修复并验收

**根因（已诊断）**：生产库迁移至 schema 29 后，旧 EXE（2026-08-07 02:50 构建，schema 25/26 时代）无法识别新增表/列，启动即异常退出（code=1）。

**已执行**：
1. 用当前代码执行 `scripts/build-desktop.ps1` 全量打包（前端构建、PyInstaller、AkShare calendar.json 硬门禁、V3.2/V3.3/V3.4 审计）。
2. 修复打包门禁 `scripts/audit-v34-state.py`（支持的 schema 版本补到 29）；完成诊断遗留的 V3.4 队列任务；修正 `scripts/v33_packaged_exe_review.py` 一键启动端口文件路径。
3. 官方验收 PASS：真实启动窗口、动态端口 HTTP 200、单实例、重启身份不变、一键启动、备份/恢复（隔离副本）、源库未修改。

**后续优化（3.1）**：增加启动兼容性自检，避免未来 EXE 与 DB 版本不匹配时再次静默失败。

## 2. 尚未开展的部分（按顺序补完）

| 序号 | 事项 | 入口/产出 |
| --- | --- | --- |
| 2.1 | 测试副本全量回放 → 生产六市场回放 | 见第 4 节前置门禁；`scripts/v37_bootstrap.py <市场>` |
| 2.2 | AI 历史匿名筛选（每市场 120 锚点、约 720 次调用） | `scripts/v37_ai_screening.py`，规则见第 6 节 |
| 2.3 | Fusion 账户与 V37FusionEvaluation 生产运行 | `v37_fusion_service.run_fusion_window_accounts` |
| 2.4 | Repair Reviewer/Proposal/Challenger 生产运行 | `v37_repair_service.ensure_repair_proposal` + `create_repair_challenger` + 密封评价（见 3.4） |
| 2.5 | 多模型页补全：决策漏斗、AI 权重与校准状态卡片 | `/v37-analysis` + `/api/v37/*`，数据来源见第 8 节 |
| 2.6 | 12 份最终报告补齐 | `scripts/v37_reports.py` 回放后重跑；内容对照表见第 4 节 |
| 2.7 | 完整前端 Playwright 套件回归 | `npm run test:e2e` 全量 |
| 2.8 | 手工验收：真实 Key 对话“最近 8 周怎么看”；重启 EXE 后配置/对话正常 | AI 设置页 + 新 EXE |
| 2.9 | 每周 Forward OOS 增量联动（实现缺口见第 7 节） | `v37_runtime.incremental_sync` + `ensure_ai_forecast` 联动 |

## 3. 需要优化的内容

### 3.1 启动兼容性自检

- EXE 启动时读取 `PRAGMA user_version` 与内置 `SCHEMA_VERSION` 比较：
  - 一致：正常初始化；
  - DB 高于 EXE：弹出明确提示“数据库已由更新版本升级（schema N），请使用配套新版程序”，不再含糊异常退出；
  - DB 低于 EXE：走既有迁移流程。
- 自检状态写入启动日志与 `data/desktop-port.json` 旁的状态文件，便于验收脚本读取。

### 3.2 /chat 与 AI 客户端

- `/chat` 非法 messages 统一返回 422（`validate_chat_messages` 已实现）；`chat_messages` 已补 401/429 与消息校验单测。
- 后续补 FastAPI 接口级测试：无 Key、未知市场、非法 messages、mock 成功。

### 3.3 AI 权重门禁、退化检测与模型口径

- 档位：<30→0%（AI_SHADOW）、30–49→10%、50–99→30%（Challenger）、≥100→40%（验证通过后）。
- 实现 `ai_weight_gates`：方向质量（MAE/Brier 不劣于 Local）、校准有效、账户不差于 Local、防空仓、回撤恶化 ≤2pp，全部通过才允许提升。
- 实现 `AI_WEIGHT_DEGRADED` 检测（连续窗口 AI 超额转负或参与率显著下降）并自动降权。
- **模型口径（必补）**：切换模型时 `ai_generation_version++`；正式校准、权重解锁、晋级与报告统计一律按 `(screening_scope=FORWARD_OOS, ai_generation_version, model_name)` 过滤；报告中输出模型分布，禁止混用两个模型的分数。

### 3.4 Repair 泄漏约束硬接入

- `repair_eligible_windows`（proposal_anchor + 8 周后）已实现，需硬接入晋级评价路径：
  - 新增 `evaluate_repair_challenger`：只允许使用提案后 8 周之外的成熟窗口计算 excess/胜率/回撤/参与率；
  - **独立窗口计数（必补）**：沿用 V3.6 conservative independent window count，≥8 个独立窗口才可进入正式比较；不足标记 `SHADOW_EVALUATION` 并禁止晋级；
  - 增加“使用提案前窗口不能晋级”的回归测试。

### 3.5 回放性能与可观测性

- 特征/模型输入/挑战窗口缓存、账户批量写入；每市场目标 90 分钟以内（尽力目标）。
- 先做单市场基准计时，再决定是否并行；并行仅限同进程多线程 + 单 Writer。
- 每周处理把 Local/AI/Fusion 三路漏斗、冲突状态、权重状态写入同一事务。

### 3.6 模型健康提示

- AI 设置页增加模型健康状态（最近 N 次调用空响应/失败率）、一键切换并重试。
- `deepseek-v4-flash` 空响应的处理见第 9 节外部依赖风险。

## 4. 前置门禁与验收基线（必补）

### 4.1 回放顺序门禁

1. 测试副本先做每市场约 40 周小规模冒烟（已完成 399006×60 周）。
2. 测试副本再做六市场**全量回放**，并核对：
   - 锚点唯一、无重复/乱序；
   - V3.5/V3.5.1 行级哈希不变、V3.6 表行数不变；
   - 幂等复跑 `processed_weeks=0`；
   - 每市场正式周数 = V3.5.1 对应正式周数 − 预热期（399006/159941 分别核对）。
3. 全部通过后：在线备份 → 生产六市场回放（可分段续跑，失败周停止并回滚该周事务）→ 重跑同一套核对。

### 4.2 数据完整性核对表（回放与筛选后）

| 检查项 | 要求 |
| --- | --- |
| 各市场 v35_forecasts/strategy/账户行数 | 与 V3.7 正式周数一致 |
| AI 请求与预测数量 | `v37_ai_forecasts` 行数 = 成功请求数；失败请求全部保留在 `v37_ai_requests` |
| screening 与 forward 行数 | 分别统计；正式统计只读 FORWARD_OOS |
| 幂等复跑 | `processed_weeks=0`、行数不增长 |
| Local vs AI 报告样本量 | 报告注明窗口数与样本来源 |
| DB integrity/FK | `integrity_check=ok`、`foreign_key_check=0` |

### 4.3 报告 ↔ 验收项对照表

| 完成标准 | 对应报告 |
| --- | --- |
| 本地日周联合模型 | V37_MULTI_TIMEFRAME_REPORT |
| AI 每周日K+周K独立分析 | V37_AI_FORECAST_REPORT / V37_DEEPSEEK_API_REPORT |
| AI 独立 Challenger 与同账户比较 | V37_LOCAL_VS_AI_REPORT |
| Fusion 只用校准概率 | V37_FUSION_REPORT |
| Repair 经本地回放晋级 | V37_MODEL_REPAIR_REPORT |
| API 故障不影响本地 | V37_DEEPSEEK_API_REPORT |
| 历史输入匿名/PIT | V37_DATABASE_REPORT / V37_AI_FORECAST_REPORT |
| 冻结记录不变 | V37_DATABASE_REPORT |
| 权重档位与门禁 | V37_AI_CALIBRATION_REPORT |
| 增量周更与 Forward OOS | V37_FORWARD_OOS_REPORT |
| 测试与验收 | V37_TEST_REPORT / V37_EXE_ACCEPTANCE_REPORT |

## 5. AI 独立晋级评估（必补）

- 新增 `evaluate_ai_challenger`：AI 独立账户 vs Local Champion 账户，同一 `v36_account_snapshots` 起点、同一执行引擎。
- 门槛沿用：≥8 个有效窗口、平均超额 ≥0.30%、胜率 ≥55%、回撤恶化 ≤2pp、参与率不显著低于 Champion、不得靠长期空仓、预测质量保护线（AI MAE ≤ Local×1.15、Brier ≤ Local+0.02）、真实交易路径差异。
- 不足窗口或未校准：标记 `AI_SHADOW`，不参与晋级。
- 通过后 `AI_FUSION_PROMOTION` / `DEEPSEEK_AI_PROMOTION` 落 `v35_promotions`，并写入报告。

## 6. AI 筛选成本/限流/断点续跑（必补）

- 每市场 120 锚点、总计约 720 次调用；**预算上限**：以每次约 2–4K token 估算，设定总 token 预算，超出即停止并保留已完成缓存。
- **失败率阈值**：单市场失败（非重试类）>20% 时停止该市场并报告，不静默继续。
- **重试退避**：瞬时失败（TIMEOUT/RATE_LIMITED/NETWORK_ERROR/MODEL_NOT_AVAILABLE）最多 3 次，间隔 1s/2s/4s；`INVALID_RESPONSE` 不重试。
- **断点续跑**：`v37_ai_screening.py` 增加 `--only-missing`（跳过已有成功 forecast 的锚点）与 `--skip-cached`；中断后重跑只补缺失。
- **token 用量**：`token_usage_json` 必须写入真实 `prompt_tokens/completion_tokens/total_tokens`，报告输出成本估算。
- 报告口径：`实际调用数 = 成功 + 失败 − 缓存`；每市场输出 selected/new/cached/failed 与失败码分布。

## 7. 每周 Forward OOS 增量联动（实现缺口，必补）

当前 `incremental_sync` 只是 `bootstrap_sync` 的别名，AI 调用未接入。需实现：

1. 刷新行情 → 新完整周判定（该自然周最后交易日存在才创建锚点）；
2. 执行上期待执行交易 → 更新真实账户（份额/T+1/可卖/待执行批次）；
3. 本地预测 → `ensure_ai_forecast`（FORWARD_OOS，成功才冻结；失败记 `AI_FALLBACK_LOCAL_ONLY`，**禁止用上周 AI 结果替代**）；
4. Fusion/冲突/挑战/晋级在同一事务；
5. 无新行情 `processed_weeks=0`，新增一周 `=1`；
6. 测试：注入账户/快照写入间失败 → 整周回滚、无孤儿快照、重试后结果与未注入一致。

## 8. 多模型页数据来源与最终 EXE 打包细节

- **决策漏斗数据来源（必补）**：LOCAL 漏斗读 `v36_decision_funnel`；AI/Fusion 漏斗在 API 层按同一口径现算（预测→Score→Target→Trade→Position），或新增 `v37_decision_funnel` 聚合表（二选一，禁止 UI 无数据）。
- 页面卡片：AI 权重/校准状态读 `v37_ai_model_health`；Repair 状态读 `v37_model_repair_proposals/runs`。
- **最终 EXE 打包细节**：打包前再在线备份；若 schema 再变更需重跑迁移验收；打包后记录根 EXE 与发布库 SHA-256 进 `V37_EXE_ACCEPTANCE_REPORT.md`；完整 Playwright 在打包前跑源码版、打包后跑发布版页面冒烟。

## 9. 外部依赖风险与备份轮换（必补）

- **deepseek-v4-flash 空响应**：以 48 小时为决策期限；超期未恢复则默认切换 `deepseek-chat`，`ai_generation_version++` 并记录协议变更；恢复后如需回切同样递增版本。
- **模型切换不影响冻结记录**：旧 generation 的 forecast 永久保留，正式统计按第 3.3 节口径过滤。
- **备份保留策略**：`audit/v37/backups/` 保留最近 5 份全量备份；回放/筛选/EXE 打包前各产生一次新备份；超过保留策略的旧备份先归档到离线目录再删除。

## 10. 本轮验收（程序打开失败修复）— 已完成

1. `scripts/build-desktop.ps1` 全流程通过。
2. 新 EXE 启动：进程存活、端口文件生成、HTTP 200。
3. 单实例：第二次启动提示已在运行并退出 0。
4. 重启：训练身份 SHA-256 不变。
5. 备份/恢复：隔离副本 backup/restore 均退出 0，哨兵数据恢复。
6. 页面：主工作台、ETF 模型管理、AI 设置（含行情对话）、多模型分析页可打开。
7. 回归：后端全套件、前端 build、ai-settings/v37-analysis Playwright 全绿。

## 11. 完成顺序建议（依赖正确版）

1. ✅ 重建 EXE + 启动/单实例/重启/备份恢复验收（已完成）。
2. 补 AI 权重门禁与退化检测、AI 独立晋级评估、Repair 评价硬约束（含独立窗口计数）。
3. 测试副本六市场全量回放 → 4.1 门禁核对 → 生产备份 → 生产六市场回放。
4. Fusion/Repair 生产运行 + AI 匿名筛选（按第 6 节预算/限流/断点规则）+ 每周 Forward OOS 联动实现。
5. 多模型页漏斗/权重卡片 + 12 份报告（按 4.3 对照表）+ 完整 Playwright + 手工对话验收。
6. 最终 EXE 重打包（按第 8 节细节）与全量验收，宣布 V3.7 完成。
