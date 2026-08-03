@echo off
setlocal
title ETF Investment Lab V3.4-13W
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
set "exit_code=%ERRORLEVEL%"
if not "%exit_code%"=="0" (
  echo.
  echo Startup failed. Keep this window open and send the error above to Codex.
  pause
)
exit /b %exit_code%
