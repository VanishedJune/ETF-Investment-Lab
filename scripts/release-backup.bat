@echo off
start "" /wait "%~dp0InvestmentLab.exe" --backup
exit /b %ERRORLEVEL%
