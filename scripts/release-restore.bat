@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0restore-data.ps1"
exit /b %ERRORLEVEL%
