# ETF Investment Lab

本项目是一个本地运行的桌面端投资研究助手，用于查看指数/ETF行情、技术指标、模型迭代结果和仓位日历。程序不连接券商、不自动交易，所有分析结果仅用于个人研究和复盘。

## 普通用户如何直接使用

普通用户不需要安装 Python、Node.js 或开发依赖。请到 GitHub Releases 下载便携版压缩包：

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

## 当前主要功能

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

## 免责声明

本项目仅用于个人研究、数据可视化和策略复盘，不构成投资建议、收益承诺或交易指令。使用者需要自行承担投资决策风险。
