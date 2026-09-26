# ETF Investment Lab

ETF Investment Lab V3.7 是一款在 Windows 本地运行的 ETF 行情研究工具，提供行情图表、技术指标和持仓日历。软件不连接券商，也不自动交易。

## 主要功能

- 查看 ETF 日K、周K行情、成交量、价格涨跌幅及 MA、DIF、DEA、MACD 等技术指标。
- 按当前展示顺序浏览八只 ETF，并在“ETF 替换”页面调整展示标的。
- 在投资日历中记录和查询持仓变动；更换展示标的不改变已有记录。
- 按需刷新行情。行情可用性取决于网络和数据来源。

当前桌面发行版为 **V3.7 Data Only**，主要提供行情和投资日历功能。

## 下载与使用

在 [Releases](https://github.com/VanishedJune/ETF-Investment-Lab/releases) 下载 `InvestmentLab-V3.7-DataOnly-20260926-portable.zip`，完整解压后双击 `InvestmentLab/InvestmentLab.exe`。请保留解压后的文件夹结构，不要单独移动 EXE。

首次使用时，可在页面中对需要的 ETF 点击“刷新行情”。Windows 如提示缺少 WebView2，请安装 Microsoft Edge WebView2 Evergreen Runtime。

## 从源码运行

```powershell
.\setup.bat
.\start.bat
```

重新构建桌面程序：

```powershell
.\build-desktop.bat
```

## 声明

本项目仅用于研究与复盘，不构成投资建议或收益承诺。
