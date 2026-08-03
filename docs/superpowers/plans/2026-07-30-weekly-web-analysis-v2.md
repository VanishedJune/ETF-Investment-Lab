# Weekly Web Analysis V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有Agent离线100次模型改造成由网页按钮触发、本地FastAPI执行、按十年每个交易周递进、可恢复和可审计的双市场V2研究工作台。

**Architecture:** 保留现有行情和指标表作为已验证原始输入，新增隔离的 `weekly_analysis` 领域模块和V2 SQLite表。后台任务按指数加锁并在每周检查点提交，前端用一个统一ECharts实例呈现价格、成交量和MACD，并通过V2 API显示任务、模型指标、分批建议和仓位日历。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy 2、Pydantic 2、pandas/numpy、SQLite、Vue 3、TypeScript、ECharts 5、pytest、Playwright。

---

## 文件结构

新增或拆分后的职责如下：

```text
backend/app/weekly_analysis/domain.py       # V2不可变领域类型、枚举和编号
backend/app/weekly_analysis/aggregation.py  # 日线到周/月K与质量检查
backend/app/weekly_analysis/features.py     # 周线MACD/估值/趋势特征
backend/app/weekly_analysis/sampling.py     # 十年自然周和固定随机交易日
backend/app/weekly_analysis/optimizer.py    # I/W/M递进、标签和候选验证
backend/app/weekly_analysis/advice.py       # 13周、五档比例和5%分批建议
backend/app/weekly_analysis/repository.py   # V2 SQLite读写和检查点
backend/app/weekly_analysis/engine.py       # 单市场基线/到期周执行编排
backend/app/weekly_analysis/tasks.py        # 线程任务、互斥锁、恢复和进度
backend/app/schemas/weekly_analysis.py      # V2 API模型
backend/app/services/weekly_analysis_service.py # API应用服务
frontend/src/types/research.ts              # 前端V2数据契约
frontend/src/composables/useResearchData.ts # 加载、轮询和竞态保护
frontend/src/components/UnifiedMarketChart.vue # 统一价格/量/MACD图
frontend/src/components/AnalysisPanel.vue  # 任务、建议和模型指标
frontend/src/components/PositionCalendar.vue # 仓位事件CRUD
```

现有 `backend/web.py`、`models.py` 和 `migrations.py` 延续项目既有集中式入口模式，只做路由、ORM表和迁移注册，不承载模型算法。

项目不是Git仓库，所以每个任务的“提交”步骤改为：运行聚焦测试、记录修改文件，并保留开始前的数据备份；不得伪造Git提交。

### Task 1: 建立可恢复备份和V2数据库基础

**Files:**
- Modify: `backend/app/models/models.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/database/migrations.py`
- Create: `backend/tests/test_weekly_analysis_migrations.py`
- Preserve: `data/model_iterations/399006/**`
- Preserve: `data/model_iterations/NDX/**`

- [ ] **Step 1: 创建现有本地数据的可恢复备份**

Run:

```powershell
.\backup.bat
```

Expected: `data/backups` 或项目既有备份目录新增带时间戳归档；旧 `model_iterations` 文件未变化。

- [ ] **Step 2: 写迁移失败测试**

```python
def test_v14_creates_isolated_v2_tables(tmp_path):
    database = tmp_path / "investment_lab.db"
    engine = initialize_database(database, tmp_path / "config")
    with engine.connect() as connection:
        version = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
        names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert version == 14
    assert {
        "v2_market_data_cache", "v2_period_bars", "v2_feature_snapshots",
        "v2_week_samples", "v2_analysis_iterations", "v2_iteration_labels",
        "v2_model_versions", "v2_analysis_tasks", "v2_position_events",
        "v2_position_snapshots", "v2_advice_history",
    } <= names
```

- [ ] **Step 3: 验证测试先失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_analysis_migrations.py -q
```

Expected: FAIL，当前 `PRAGMA user_version` 为13或V2表不存在。

- [ ] **Step 4: 增加V2 ORM模型和版本14迁移**

模型使用现有 `TimestampMixin`、`FixedPointDecimal`、JSON和外键规范。核心唯一约束：

```python
class V2WeekSample(TimestampMixin, Base):
    __tablename__ = "v2_week_samples"
    __table_args__ = (
        UniqueConstraint("instrument_id", "week_key", name="uq_v2_sample_market_week"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), index=True)
    week_key: Mapped[str] = mapped_column(String(10))
    assigned_month: Mapped[str] = mapped_column(String(7))
    candidate_dates: Mapped[list[str]] = mapped_column(JSON)
    sample_date: Mapped[date] = mapped_column(Date)
    seed: Mapped[int] = mapped_column(Integer)
    algorithm_version: Mapped[str] = mapped_column(String(32))
    source_hash: Mapped[str] = mapped_column(String(64))


class V2AnalysisIteration(TimestampMixin, Base):
    __tablename__ = "v2_analysis_iterations"
    __table_args__ = (
        UniqueConstraint("instrument_id", "iteration_number", name="uq_v2_iteration_number"),
        UniqueConstraint("instrument_id", "week_key", name="uq_v2_iteration_week"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), index=True)
    iteration_number: Mapped[int] = mapped_column(Integer)
    week_key: Mapped[str] = mapped_column(String(10))
    cutoff_date: Mapped[date] = mapped_column(Date)
    source_data_max_date: Mapped[date] = mapped_column(Date)
    work_version: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24))
    prediction: Mapped[dict[str, object]] = mapped_column(JSON)
    metrics: Mapped[dict[str, object]] = mapped_column(JSON)
    source_hash: Mapped[str] = mapped_column(String(64))
```

其余V2表采用同样的显式唯一约束和JSON审计字段。`SCHEMA_VERSION = 14`，`_upgrade_to_version_fourteen()` 使用 `Base.metadata.tables[name].create(checkfirst=True)` 创建11张表，不修改V1数据。

- [ ] **Step 5: 运行迁移和旧数据保护测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_analysis_migrations.py backend/tests/test_database_foundation.py -q
```

Expected: PASS；现有数据库可从13升级到14，重复初始化幂等。

### Task 2: 将投资日历迁移为仓位百分比事件

**Files:**
- Modify: `backend/app/schemas/investment_calendar.py`
- Modify: `backend/app/services/investment_calendar_service.py`
- Modify: `backend/web.py`
- Rewrite: `backend/tests/test_investment_calendar.py`
- Create: `backend/tests/test_position_event_api.py`

- [ ] **Step 1: 写5%倍数、重算和回滚测试**

```python
def test_position_events_recalculate_in_date_order(tmp_path):
    service, _ = _service(tmp_path)
    later = service.create(PositionEventCreate(
        instrument_code="399006", direction="increase",
        operation_date=date(2026, 7, 20), change_percent=30,
    ))
    earlier = service.create(PositionEventCreate(
        instrument_code="399006", direction="increase",
        operation_date=date(2026, 7, 10), change_percent=20,
    ))
    assert service.current_positions()["399006"] == 50
    service.update(later["id"], PositionEventUpdate(change_percent=40))
    assert service.current_positions()["399006"] == 60
    service.delete(earlier["id"])
    assert service.current_positions()["399006"] == 40


@pytest.mark.parametrize("value", [1, 4, 6, 101])
def test_position_change_rejects_non_five_multiples(value):
    with pytest.raises(ValueError):
        PositionEventCreate(
            instrument_code="NDX", direction="increase",
            operation_date=date(2026, 7, 10), change_percent=value,
        )
```

另写测试证明：

- 无记录返回 `current_position: null`；
- 第一条只能是 `increase`；
- 任一重放步骤小于0或大于100时整个事务回滚；
- 同日按 `operation_date, created_at, id` 排序；
- 两个指数互不影响；
-重启服务后记录和当前仓位仍存在。

- [ ] **Step 2: 验证旧服务测试失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_investment_calendar.py backend/tests/test_position_event_api.py -q
```

Expected: FAIL，旧schema仍要求产品、份额和成交价。

- [ ] **Step 3: 实现新的Pydantic契约**

```python
class PositionEventCreate(BaseModel):
    instrument_code: Literal["399006", "NDX"]
    direction: Literal["increase", "decrease"]
    operation_date: date
    change_percent: int = Field(ge=5, le=100, multiple_of=5)
    note: str | None = Field(default=None, max_length=1000)


class PositionEventUpdate(BaseModel):
    direction: Literal["increase", "decrease"] | None = None
    operation_date: date | None = None
    change_percent: int | None = Field(default=None, ge=5, le=100, multiple_of=5)
    note: str | None = Field(default=None, max_length=1000)
```

- [ ] **Step 4: 实现事务内全量重放**

`InvestmentCalendarService` 改为读写 `V2PositionEvent` 和 `V2PositionSnapshot`：

```python
def _replay(self, session: Session, instrument_id: int) -> int | None:
    events = list(session.scalars(
        select(V2PositionEvent)
        .where(V2PositionEvent.instrument_id == instrument_id)
        .order_by(V2PositionEvent.operation_date, V2PositionEvent.created_at, V2PositionEvent.id)
    ))
    if not events:
        session.execute(delete(V2PositionSnapshot).where(
            V2PositionSnapshot.instrument_id == instrument_id
        ))
        return None
    position = 0
    for index, event in enumerate(events):
        if index == 0 and event.direction != "increase":
            raise ValueError("第一条仓位记录必须为增加")
        position += event.change_percent if event.direction == "increase" else -event.change_percent
        if not 0 <= position <= 100:
            raise ValueError(f"{event.operation_date} 操作后仓位将变为 {position}%")
        session.add(V2PositionSnapshot(
            instrument_id=instrument_id, event_id=event.id,
            operation_date=event.operation_date, position_percent=position,
        ))
    return position
```

重放前删除该指数旧快照；若任何验证失败，SQLAlchemy事务整体回滚。

- [ ] **Step 5: 更新API**

保留旧 `/api/investment-calendar` 路径以降低前端改动风险，但响应改为仓位事件；增加：

```text
GET /api/investment-calendar/current-positions
```

返回：

```json
{"399006": 60, "NDX": null}
```

- [ ] **Step 6: 运行服务和API测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_investment_calendar.py backend/tests/test_position_event_api.py -q
```

Expected: PASS。

### Task 3: 实现周期聚合和“无未处理空白”质量门

**Files:**
- Create: `backend/app/weekly_analysis/__init__.py`
- Create: `backend/app/weekly_analysis/domain.py`
- Create: `backend/app/weekly_analysis/aggregation.py`
- Create: `backend/tests/test_weekly_aggregation_v2.py`
- Modify: `backend/app/services/market_data.py`
- Modify: `backend/app/services/indicator_service.py`

- [ ] **Step 1: 写聚合规则测试**

```python
def test_weekly_bar_uses_first_high_low_last_and_summed_volume():
    bars = [
        DailyBar(date(2026, 7, 27), D("10"), D("12"), D("9"), D("11"), D("100")),
        DailyBar(date(2026, 7, 28), D("11"), D("14"), D("10"), D("13"), D("150")),
        DailyBar(date(2026, 7, 31), D("13"), D("15"), D("8"), D("12"), D("250")),
    ]
    weekly = aggregate_bars(bars, "weekly", instrument_code="399006")
    assert weekly[0].open == D("10")
    assert weekly[0].high == D("15")
    assert weekly[0].low == D("8")
    assert weekly[0].close == D("12")
    assert weekly[0].volume == D("500")
    assert weekly[0].source == "AGGREGATED_DAILY_VOLUME"
```

另测月线、跨月自然周、重复日期、内部日线缺口、NDX无直接成交量状态，以及所有发布行的OHLC非空。

- [ ] **Step 2: 验证测试先失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_aggregation_v2.py -q
```

Expected: FAIL，`weekly_analysis.aggregation` 尚不存在。

- [ ] **Step 3: 实现纯聚合器和质量报告**

```python
@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Literal["warning", "blocking"]
    start_date: date | None
    end_date: date | None
    message: str


def aggregate_bars(
    daily: Sequence[DailyBar], timeframe: Literal["weekly", "monthly"],
    *, instrument_code: str,
) -> list[PeriodBar]: ...


def validate_series(
    bars: Sequence[PeriodBar], *, require_volume: bool,
) -> QualityReport: ...
```

`399006` 周/月线若上游量无效，强制用日量求和；`NDX` 仅在直接指数量可靠时绘制，否则返回 `volume_availability="not_available_for_direct_index"`，不生成0值。

- [ ] **Step 4: 将现有刷新流程接入统一聚合器**

`MarketDataService.aggregate_timeframes()` 和 `IndicatorService.recalculate_all_timeframes()` 使用同一V2周期边界，保证价格、量和指标日期一一对应。刷新失败时不删除最后一份完整周期数据。

- [ ] **Step 5: 运行行情与指标回归**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_aggregation_v2.py backend/tests/test_market_data.py backend/tests/test_indicators.py -q
```

Expected: PASS。

### Task 4: 实现自然周抽样、特征和标签成熟

**Files:**
- Create: `backend/app/weekly_analysis/sampling.py`
- Create: `backend/app/weekly_analysis/features.py`
- Create: `backend/tests/test_weekly_sampling_v2.py`
- Create: `backend/tests/test_weekly_features_v2.py`

- [ ] **Step 1: 写自然周数量与可复现抽样测试**

```python
def test_one_sample_per_natural_week_and_last_trade_day_month_assignment():
    samples = build_week_samples(
        trading_dates=[
            date(2026, 1, 29), date(2026, 1, 30),
            date(2026, 2, 2), date(2026, 2, 3),
        ],
        seed=20260730,
        algorithm_version="weekly-v2",
    )
    assert [row.week_key for row in samples] == ["2026-W05", "2026-W06"]
    assert samples[0].assigned_month == "2026-01"


def test_sampling_is_reproducible():
    first = build_week_samples(dates, seed=20260730, algorithm_version="weekly-v2")
    second = build_week_samples(dates, seed=20260730, algorithm_version="weekly-v2")
    assert first == second
    assert all(row.sample_date in row.candidate_dates for row in first)
```

另测十年窗口实际周数约520而不是固定100/480、每周只有一个样本、抽样日只能读取上一个已完成周。

- [ ] **Step 2: 写特征和未来泄漏测试**

```python
def test_feature_source_never_passes_cutoff():
    snapshot = build_feature_snapshot(
        completed_weekly_bars=weekly,
        valuations=valuations,
        cutoff_date=date(2026, 7, 29),
    )
    assert snapshot.source_data_max_date <= snapshot.cutoff_date


def test_macd_aliases_are_identical():
    feature = build_feature_snapshot(weekly, valuations, cutoff)
    assert feature.dip == feature.dif
    assert feature.eda == feature.dea
    assert feature.macd_histogram == 2 * (feature.dif - feature.dea)
```

- [ ] **Step 3: 实现抽样器**

按ISO自然周分组，候选日只来自实际交易日；随机键使用 `sha256(f"{seed}:{symbol}:{week_key}")` 派生，避免Python进程哈希随机化。

- [ ] **Step 4: 实现60周预热和周线特征**

`build_feature_snapshot()` 只接收截止日前已完成周，计算：

```text
EMA12、EMA26、DIF、DEA、MACD柱、RSI6、
MA20、MA60、20周量比、波动率、回撤、
估值百分位、黄金点/黑点及强度
```

估值只允许读取 `valuation_date <= cutoff_date`。超过2个交易日的长缺口产生阻断质量问题。

- [ ] **Step 5: 运行聚焦测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_sampling_v2.py backend/tests/test_weekly_features_v2.py -q
```

Expected: PASS。

### Task 5: 实现I/W/M递进优化器

**Files:**
- Create: `backend/app/weekly_analysis/optimizer.py`
- Create: `backend/tests/test_weekly_optimizer_v2.py`

- [ ] **Step 1: 写递进和成熟状态测试**

```python
def test_iteration_n_inherits_work_state_from_n_minus_one():
    first = run_optimizer_step(seed_model("399006"), feedback=partial_feedback(1))
    second = run_optimizer_step(first.work_state, feedback=partial_feedback(2))
    assert second.parent_work_version == first.work_version
    assert second.optimizer_memory["feedback_count"] == 2


def test_pending_labels_never_use_future_rows():
    labels = update_labels(iterations, daily, as_of=date(2026, 7, 30))
    assert labels["I0001"].horizon_4w in {"pending", "mature"}
    assert all(label.observed_through <= date(2026, 7, 30) for label in labels.values())
```

另测：

- `M0001`初始权重分别为创业板 `30/20/20/15/5/10`、NDX `25/22/22/12/9/10`；
- 单次权重变化不超过2个百分点且总和100；
- 每个新周增加I/W，同周重复不增加；
- 20/52滚动损失阈值；
- 方向命中下降最多2个百分点；
- 被拒候选不改变M，但其反馈进入下一W；
- 两市场状态不可交叉读取。

- [ ] **Step 2: 验证测试先失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_optimizer_v2.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现标签和损失**

```python
LOSS_WEIGHTS = {
    "turning_deviation": Decimal("0.45"),
    "direction_calibration": Decimal("0.30"),
    "adverse_excursion": Decimal("0.15"),
    "overtrading": Decimal("0.10"),
}


def actual_turn_date(side: str, rows: Sequence[DailyBar]) -> date:
    chooser = max if side == "sell" else min
    target = chooser(row.close for row in rows)
    return next(row.trade_date for row in rows if row.close == target)
```

4周和13周未成熟时保持 `pending`；部分反馈带 `observed_through` 和低权重，不伪装为完整标签。

- [ ] **Step 4: 实现工作状态和候选验证**

每一步：

1. 继承上一个W；
2. 合并当前可见部分反馈和成熟反馈；
3. 生成单维不超过2个百分点的候选；
4. 在扩展/20/52窗口计算损失；
5. 按规格门槛接受或拒绝；
6. 保存新的W；接受时生成下一个M。

- [ ] **Step 5: 运行优化器测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_optimizer_v2.py -q
```

Expected: PASS。

### Task 6: 实现13周建议、五档比例和5%分批

**Files:**
- Create: `backend/app/weekly_analysis/advice.py`
- Create: `backend/tests/test_weekly_advice_v2.py`

- [ ] **Step 1: 写五档和四种典型场景测试**

```python
@pytest.mark.parametrize(
    ("direction", "probability", "confidence", "expected"),
    [
        ("up", 78, 80, "7:3"), ("up", 61, 65, "6:4"),
        ("neutral", 50, 62, "5:5"), ("down", 62, 66, "4:6"),
        ("down", 81, 84, "3:7"),
    ],
)
def test_ratio_is_one_of_five_integer_tiers(direction, probability, confidence, expected):
    assert fund_etf_ratio(direction, probability, confidence) == expected
```

为用户给出的四个场景分别断言：

- 高估趋势强：90%到70%，总减20%，不一次减90%；
- 高估多项见顶：95%到15%，4批且合计80%；
- 低估趋势弱：10%到20%，只试探增加10%；
- 低估反转共振：20%到80%，4批且合计60%。

- [ ] **Step 2: 写日期和仓位守恒测试**

```python
def test_batches_are_five_multiples_and_sum_to_adjustment():
    advice = build_advice(current_position=95, target_position=15, ...)
    assert len(advice.batches) <= 4
    assert all(batch.percent % 5 == 0 for batch in advice.batches)
    assert sum(batch.percent for batch in advice.batches) == 80
    assert all(batch.tolerance_trading_days == 3 for batch in advice.batches)
```

另测无仓位时不生成百分比、低置信度返回等待确认、日期来自未来交易日历、条件未满足的后续批次状态为 `waiting_confirmation`。

- [ ] **Step 3: 实现建议生成器**

`build_advice()` 输入验证模型、最新特征、未来交易日历和 `current_position: int | None`，输出13周方向、目标仓位、最多4批和审计说明。所有仓位先量化到最近5%，再按风险由前大后小分批。

- [ ] **Step 4: 运行建议测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_advice_v2.py -q
```

Expected: PASS。

### Task 7: 实现V2仓储、检查点和审计快照

**Files:**
- Create: `backend/app/weekly_analysis/repository.py`
- Create: `backend/tests/test_weekly_repository_v2.py`
- Create directory: `data/weekly_analysis_v2/399006`
- Create directory: `data/weekly_analysis_v2/NDX`

- [ ] **Step 1: 写事务、幂等和恢复测试**

```python
def test_checkpoint_survives_repository_reopen(tmp_path):
    repo = WeeklyAnalysisRepository(factory, tmp_path / "audit")
    repo.commit_iteration("399006", iteration, work_state, labels)
    reopened = WeeklyAnalysisRepository(factory, tmp_path / "audit")
    assert reopened.last_iteration("399006").iteration_number == 1


def test_same_market_week_is_idempotent(repo):
    repo.commit_iteration("NDX", iteration, work_state, labels)
    with pytest.raises(DuplicateWeek):
        repo.commit_iteration("NDX", iteration, work_state, labels)
```

另测单周事务失败不留下半条记录、V1目录哈希不变、JSON快照字段齐全且可复算。

- [ ] **Step 2: 实现仓储接口**

```python
class WeeklyAnalysisRepository:
    def list_completed_weeks(self, symbol: str) -> set[str]: ...
    def load_work_state(self, symbol: str) -> WorkState: ...
    def current_model(self, symbol: str) -> ModelVersion: ...
    def commit_iteration(self, symbol, iteration, work_state, labels) -> None: ...
    def save_task_checkpoint(self, task_id: int, progress: TaskProgress) -> None: ...
    def save_advice(self, symbol: str, advice: Advice) -> int: ...
    def metrics(self, symbol: str) -> ModelMetrics: ...
```

每周迭代、标签、W、M和任务进度在同一事务中提交。JSON审计文件先写临时文件再原子替换，路径包含指数、I和生成时间。

- [ ] **Step 3: 运行仓储测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_repository_v2.py -q
```

Expected: PASS。

### Task 8: 实现十年基线、到期周和后台任务

**Files:**
- Create: `backend/app/weekly_analysis/engine.py`
- Create: `backend/app/weekly_analysis/tasks.py`
- Create: `backend/app/services/weekly_analysis_service.py`
- Create: `backend/tests/test_weekly_engine_v2.py`
- Create: `backend/tests/test_weekly_tasks_v2.py`

- [ ] **Step 1: 写基线和幂等测试**

```python
def test_baseline_runs_every_unique_trade_week(progress_fixture):
    result = engine.run("399006", as_of=date(2026, 7, 30))
    expected = len(build_week_samples(progress_fixture.trading_dates[-TEN_YEARS:], ...))
    assert result.total_iterations == expected
    assert 515 <= expected <= 525


def test_second_click_same_complete_week_only_regenerates_advice():
    first = engine.run("399006", as_of=date(2026, 7, 30))
    second = engine.run("399006", as_of=date(2026, 7, 30))
    assert second.total_iterations == first.total_iterations
    assert second.advice_generation > first.advice_generation
```

另测最新未完成周不进入模型、新完整周只追加一次、中断后从最后成功周继续、两个市场任务可并行但同一市场互斥。

- [ ] **Step 2: 实现引擎编排**

```python
class WeeklyAnalysisEngine:
    def run(self, symbol: str, *, as_of: date | None = None,
            progress: Callable[[TaskProgress], None]) -> AnalysisResult:
        dataset = self.data.load_verified_window(symbol, years=10, warmup_weeks=60)
        samples = self.sampler.samples(dataset.daily_dates)
        for sample in samples:
            if self.repository.has_week(symbol, sample.week_key):
                continue
            self._run_one_week(symbol, sample, dataset)
            progress(...)
        return self._generate_current_advice(symbol, dataset)
```

生产API不接受客户端自定义 `as_of`；该参数只供测试。

- [ ] **Step 3: 实现任务管理器**

使用 `ThreadPoolExecutor(max_workers=2)` 和每指数一把 `threading.Lock`。启动时把遗留 `queued/running` 状态改为 `recoverable`。任务状态严格使用规格中的八个状态，异常保存消息和最后检查点。

- [ ] **Step 4: 运行引擎和任务测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_engine_v2.py backend/tests/test_weekly_tasks_v2.py -q
```

Expected: PASS。

### Task 9: 暴露V2 API并接入FastAPI生命周期

**Files:**
- Create: `backend/app/schemas/weekly_analysis.py`
- Modify: `backend/web.py`
- Create: `backend/tests/test_weekly_analysis_api_v2.py`

- [ ] **Step 1: 写API契约测试**

```python
def test_create_task_and_poll_status(client):
    created = client.post("/api/v2/analysis/tasks", json={"instrument_code": "399006"})
    assert created.status_code == 202
    task = created.json()
    status = client.get(f"/api/v2/analysis/tasks/{task['id']}")
    assert status.json()["status"] in {
        "queued", "preparing_data", "building_features", "iterating",
        "validating", "generating_advice", "completed",
    }
```

另测：

- 同一幂等键返回同一任务；
- 未知指数400；
- `GET /models/{symbol}/metrics`；
- `GET /advice/{symbol}/latest`；
- `POST /tasks/{id}/resume` 只恢复 `recoverable`；
- 响应中Decimal和日期正确JSON化。

- [ ] **Step 2: 注册生命周期服务**

在 `lifespan()` 中初始化一个 `WeeklyAnalysisService` 和一个 `AnalysisTaskManager`；在 `yield` 后执行 `task_manager.shutdown(wait=False)`，避免测试进程悬挂。

- [ ] **Step 3: 添加V2路由**

实现规格中的五组V2端点，创建任务返回HTTP 202；任务不存在返回404；同周完成结果可直接返回HTTP 200并带 `reused: true`。

- [ ] **Step 4: 运行API与全后端快速回归**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_weekly_analysis_api_v2.py backend/tests/test_agent_iteration_api.py backend/tests/test_bootstrap.py -q
```

Expected: PASS；V1只读端点仍可审计旧产物。

### Task 10: 拆分前端数据契约和统一图表

**Files:**
- Create: `frontend/src/types/research.ts`
- Create: `frontend/src/composables/useResearchData.ts`
- Create: `frontend/src/components/UnifiedMarketChart.vue`
- Modify: `frontend/src/views/ResearchWorkbenchView.vue`
- Modify: `frontend/src/styles.css`
- Create: `frontend/tests/research-types.test.mjs`

- [ ] **Step 1: 写纯函数和构建失败测试**

在 `research-types.test.mjs` 验证：

```javascript
assert.deepEqual(normalizeSeries(prices, indicators).dates, ["2026-07-24", "2026-07-31"]);
assert.equal(legendColors.dif, "#12687e");
assert.equal(legendColors.dea, "#d88b2c");
assert.equal(legendColors.macdPositive, "#d9553f");
```

Run:

```powershell
Set-Location frontend
npm run build
```

Expected: 在组件尚未创建时FAIL。

- [ ] **Step 2: 定义V2 TypeScript契约**

`research.ts` 定义：

```typescript
export type RatioTier = "7:3" | "6:4" | "5:5" | "4:6" | "3:7";
export type TaskStatus =
  | "queued" | "preparing_data" | "building_features" | "iterating"
  | "validating" | "generating_advice" | "completed" | "failed" | "recoverable";
export interface PositionEvent {
  id: number; instrument_code: "399006" | "NDX";
  direction: "increase" | "decrease"; operation_date: string;
  change_percent: number; position_after: number; note: string | null;
}
```

- [ ] **Step 3: 实现统一ECharts组件**

单个ECharts实例包含三个grid和三个共享category xAxis：

```typescript
axisPointer: { link: [{ xAxisIndex: [0, 1, 2] }] },
dataZoom: [
  { type: "inside", xAxisIndex: [0, 1, 2] },
  { type: "slider", xAxisIndex: [0, 1, 2] },
],
legend: {
  data: ["MACD柱", "DIP / DIF", "EDA / DEA"],
},
```

Grid 0为K线/均线，Grid 1为成交量，Grid 2为MACD/DIF/DEA。日/周/月切换只重载一次数据，缩放范围和十字光标天然同步。NDX无直接量时显示有文字的不可用状态，不渲染空柱。

- [ ] **Step 4: 实现组合式加载器**

`useResearchData` 用递增generation避免指数/周期快速切换时旧请求覆盖新请求，分别维护行情加载、任务轮询和错误状态。行情刷新只调用refresh端点，不触发V2分析。

- [ ] **Step 5: 构建检查**

Run:

```powershell
Set-Location frontend
node --test tests/research-types.test.mjs
npm run build
```

Expected: PASS。

### Task 11: 实现分析结果和仓位日历界面

**Files:**
- Create: `frontend/src/components/AnalysisPanel.vue`
- Create: `frontend/src/components/PositionCalendar.vue`
- Modify: `frontend/src/views/ResearchWorkbenchView.vue`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/tests/critical-pages.spec.ts`

- [ ] **Step 1: 先更新Playwright断言**

新增断言：

```typescript
await expect(page.getByRole("button", { name: "数据分析" })).toBeVisible();
await expect(page.getByTestId("unified-market-chart")).toHaveAttribute("data-chart-state", "rendered");
await expect(page.getByTestId("ratio-advice")).toContainText(/7 : 3|6 : 4|5 : 5|4 : 6|3 : 7/);
await page.getByTestId("position-change").fill("15");
await page.getByRole("button", { name: "保存仓位记录" }).click();
await expect(page.getByTestId("current-position")).toContainText("15%");
```

另覆盖编辑、删除、刷新恢复、非5%错误、任务进度、I/W/M指标、最多4批及±3交易日。

- [ ] **Step 2: 实现“数据分析”任务交互**

按钮调用 `POST /api/v2/analysis/tasks`，每1秒轮询状态；组件卸载或切换指数时停止旧轮询。进度显示阶段、完成周/总周、I/W/M、检查点和耗时。完成后重新拉取metrics和latest advice。

- [ ] **Step 3: 实现建议和模型指标**

展示：

- 当前/目标/建议变化仓位；
- 13周方向、概率、置信度；
- 最多4批日期、百分比、±3交易日和确认条件；
- 五档比例；
- 单次、20次、52次偏离；
- 方向命中、校准误差、候选接受率；
- 第一版至当前版改善曲线。

未设置仓位时显示“请先设置当前仓位”，保留市场分析但隐藏具体百分比。

- [ ] **Step 4: 实现仓位日历**

表单只保留指数、增加/减少、日期、变化百分比和备注。百分比input使用 `min=5 max=100 step=5`，但仍显示后端校验错误。列表显示每笔操作后的仓位，支持编辑和删除。

- [ ] **Step 5: 保持亮色审查界面并扩大可读性**

沿用当前纸张亮主题，将正文最小字号提升到14px、标签至少13px、图表标题至少22px；桌面端统一图表高度不低于780px，窄屏三层grid仍保持可读。

- [ ] **Step 6: 构建前端**

Run:

```powershell
Set-Location frontend
npm run build
```

Expected: PASS，`frontend/dist` 更新。

### Task 12: 更新运行规则、执行真实基线并完成端到端验收

**Files:**
- Modify: `AGENTS.md`
- Rewrite: `docs/AGENT_MODEL_ITERATION.md`
- Modify: `config/agent_iteration_policy.json`
- Modify: `README.md`
- Modify: `docs/使用说明.md`
- Modify: `frontend/tests/critical-pages.spec.ts`

- [ ] **Step 1: 更新项目内运行规则**

明确：

```text
V2模型由网页“数据分析”按钮触发本地后端执行；
不调用Agent或AI；
每个唯一完整交易周一次；
同周幂等；
每市场独立；
五档比例；
V1目录只读保留；
禁止ETF替代指数。
```

删除文档和配置中的“固定100次”“每月两次”“三档比例”和“Web后台不得训练”等冲突文本。保留V1命令仅作为旧产物审计说明。

- [ ] **Step 2: 运行完整后端测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Expected: 全部PASS，无线程泄漏或未关闭SQLite连接警告。

- [ ] **Step 3: 构建前端**

Run:

```powershell
Set-Location frontend
npm run build
```

Expected: PASS。

- [ ] **Step 4: 启动服务并刷新两市场行情**

Run:

```powershell
.\start.bat
```

在本地页面分别刷新 `399006` 和 `NDX`。Expected：

- 创业板日/周/月价格和周/月成交量通过质量检查；
- NDX若无可靠直接指数量，显示明确不可用状态；
- 2021年前支持区间不出现整段空白；
- 服务地址 `http://127.0.0.1:8765` 可访问。

- [ ] **Step 5: 通过网页按钮建立两市场V2基线**

分别选择两个指数并点击“数据分析”。Expected：

- 每市场约520个唯一周迭代；
- 进度可见；
- I连续、W父链连续；
- M只在验证通过时增加；
- 最近4周/13周标签可保持pending；
- 审计JSON和SQLite记录均生成；
- V1目录哈希不变。

- [ ] **Step 6: 运行Playwright**

Run:

```powershell
Set-Location frontend
npx playwright test
```

Expected: 全部PASS。

- [ ] **Step 7: 最终数据审计**

运行只读审计脚本或pytest断言：

```text
每指数迭代数 = 十年窗口内唯一有效交易周数
每指数每周最多一条
source_data_max_date <= cutoff_date
权重和 = 100
批次总和 = abs(target-current)
每批 % 5 = 0
所有已声明可用曲线内部空值数 = 0
399006周/月成交量空值数 = 0
```

Expected: 全部为真；若任何条件失败，不报告完成。

