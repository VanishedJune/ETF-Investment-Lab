# ETF Investment Lab

本项目是一个本地运行的桌面端投资研究助手，用于查看指数/ETF行情、技术指标、模型迭代结果和仓位日历。程序不连接券商、不自动交易，所有分析结果仅用于个人研究和复盘。

当前源码包含 V3.7 工作流及 2026-09-26 的数据界面更新。下面的 V3.4 便携包说明属于历史发布版；源码更新不会自动更新 GitHub Releases 中的可执行程序。

## 普通用户如何直接使用

普通用户不需要安装 Python、Node.js 或开发依赖。GitHub Releases 中的以下压缩包是历史 V3.4 版本，并非当前 V3.7 源码构建：

```text
InvestmentLab-V3.4-13W-portable.zip
```

下载后：

1. 解压整个压缩包；
2. 保持文件夹结构不变；
3. 双击 `InvestmentLab.exe`；
4. 如果 Windows 提示缺少 WebView2 Runtime，请安装 Microsoft Edge WebView2 Evergreen Runtime 后再启动。

发布包内已经包含：

- `InvestmentLab.exe`
- `_internal/` 运行依赖
- `data/investment_lab.db` 本地行情和模型数据库
- `config/` 配置
- 备份和恢复脚本

注意：不要只复制单独的 EXE。桌面版需要和 `_internal`、`data`、`config` 放在同一个目录中运行。

## 开发者如何从源码运行

```powershell
.\setup.bat
.\start.bat
```

源码模式会启动本地 Web 服务并打开页面。开发数据保存在本地 `data/` 目录，该目录不会提交到 Git。

## 重新构建桌面版

```powershell
.\build-desktop.bat
```

构建完成后，便携版目录位于：

```text
dist\InvestmentLab
```

项目根目录下的 `InvestmentLab.exe` 只是构建脚本同步出来的快捷启动入口。正式发布时请以 `dist\InvestmentLab` 整个目录为准。

## V3.4 历史发布包的主要功能

- 创业板指数 `399006` 行情、成交量、DIF、DEA、MACD 和 DIF 一阶变化；
- 广发纳斯达克100 ETF `159941` 行情和指标；
- 广发黄金 ETF `518600`、华宝银行 ETF `512800`、鹏华酒 ETF `512690` 展示型行情曲线；
- 日K、周K、月K切换和可视区间动态缩放；
- V3.4 13周概率情景K线；
- 模型迭代曲线、Champion 模型状态和历史偏离评估；
- 仓位日历，本地记录两个核心市场的仓位百分比；
- 数据分析按钮，使用当前最新 Champion 模型生成仓位建议。

## 数据与隐私

本项目默认将运行数据保存在本地 SQLite 数据库中：

```text
data\investment_lab.db
```

该数据库可能包含本地行情缓存、模型状态、仓位日历和后续个人记录，因此源码仓库默认不提交 `data/`。如果需要共享可直接使用的版本，请通过 GitHub Release 上传便携版压缩包。

### 可选：从决策月报项目同步8只ETF行情

同步是独立的显式操作，不会随软件启动、联网刷新或月报生成自动运行。先关闭InvestmentLab，然后执行：

```powershell
python -m pip install -r requirements-sync.txt
python scripts/import-ai-assistant-market-data.py --check
python scripts/import-ai-assistant-market-data.py
```

默认读取同一`assets`目录下的`AI Investment Assistant`正式清单，只同步8只ETF的日线原始价格和复权收盘价。数据库冲突、清单不完整或任一派生重建失败时会恢复同步前备份。同一`run_id`重复执行为零写入。

自定义路径时使用：

```powershell
python scripts/import-ai-assistant-market-data.py --source-root "<月报项目路径>" --db "<investment_lab.db路径>"
```

WebView2缓存由桌面启动器自动治理；软件完全退出后也可只读检查或显式清理：

```powershell
python scripts/cleanup-webview-cache.py --check
python scripts/cleanup-webview-cache.py
```

## 免责声明

本项目仅用于个人研究、数据可视化和策略复盘，不构成投资建议、收益承诺或交易指令。使用者需要自行承担投资决策风险。
# 2026-09-26 数据界面更新

当前8只ETF和日历统一读取动态槽位，按交易策略优先级排列；日K、周K面板同时展示价格涨跌幅。标的替换不迁移历史账本，旧持仓仍计入汇总。
详见 [当前目录与升级说明](docs/DATA_UI_CATALOG_20260926.md)。以下旧版本模型说明只作历史参考，不代表数据界面的启用功能。
