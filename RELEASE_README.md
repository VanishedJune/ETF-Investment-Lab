# ETF Investment Lab 便携版使用说明

## 快速启动

1. 解压整个 `InvestmentLab-V3.4-13W-portable.zip`；
2. 进入解压后的 `InvestmentLab` 文件夹；
3. 双击 `InvestmentLab.exe`。

请保持以下目录和文件在同一个文件夹内：

```text
InvestmentLab.exe
_internal\
data\
config\
```

不要只复制单独的 `InvestmentLab.exe`，否则程序会缺少运行依赖或数据库。

## 系统要求

- Windows 桌面系统；
- Microsoft Edge WebView2 Runtime。

大部分 Windows 10/11 电脑已经自带 WebView2。如果启动时提示缺少 WebView2，请安装 Microsoft Edge WebView2 Evergreen Runtime。

## 数据说明

便携版内置 `data\investment_lab.db`，包含当前已获取的历史行情、模型迭代状态和分析所需数据。后续你在软件里新增的仓位日历、刷新行情缓存和模型状态都会继续写入本地数据库。

## 备份

发布包内提供：

```text
Create-Data-Backup.bat
Restore-Data-Backup.bat
```

建议在长期使用前定期备份数据库。

## 免责声明

本软件仅用于个人研究和复盘，不构成投资建议，也不会自动交易。
