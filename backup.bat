@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\backup.ps1"
exit /b %ERRORLEVEL%
